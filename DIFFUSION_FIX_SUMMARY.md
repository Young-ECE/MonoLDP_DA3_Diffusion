# DepthDecoderDiffusion 修复总结

## 问题描述

在训练时遇到以下错误：
```
RuntimeError: Given groups=1, weight of size [32, 32, 3, 3], expected input[6, 512, 10, 12] to have 32 channels, but got 512 channels instead
```

错误发生在 `trainer.py:372` 调用 `outputs = self.models["depth"](features, norm_pix_coords, gt_for_diffusion)` 时。

## 根本原因

该项目的扩散模块借鉴了 MonoDiffusion 的实现，但在移植过程中存在以下问题：

### 1. **Encoder 架构差异**
- **MonoDiffusion**: 使用 LiteMono encoder，输出 3 个特征层 `[48/64, 80/128, 128/224]`
- **本项目**: 使用 ResnetEncoder，输出 5 个特征层 `[64, 64, 128, 256, 512]`
- **影响**: Skip connection 的索引映射不正确

### 2. **输出通道数不匹配**
- **MonoDiffusion**: `num_output_channels=1` (单通道深度/视差)
- **原实现**: `num_output_channels=3` (错误)
- **影响**: 扩散模型的噪声预测维度不正确

### 3. **Forward 方法接口不一致**
- **MonoDiffusion**: `forward(input_features, gt, mask=None)`
- **原实现**: `forward(input_features, norm_pix_coords, gt=None, mask=None)`
- **影响**: 传递了不必要的 `norm_pix_coords` 参数，且未正确传递 `gt`

### 4. **Decoder 构造逻辑错误**
- **问题**: Decoder 构造时假设 skip connection 来自 `encoder[i-1]`，但实际需要 `encoder[i+1]`
- **影响**: 卷积层期望的输入通道数与实际不匹配

### 5. **Condition Feature 空间维度不匹配**
- **问题**: Condition feature 的空间维度与目标 GT 不匹配
- **影响**: 扩散模型无法正确融合特征和噪声

## 修复方案

### 1. 修改 `DepthDecoderDiffusion.__init__`

**文件**: `DepthEstimation/networks/depth_decoder_diffusion.py`

**修改点**:
- 移除未使用的 `PixelCoorModu` 参数
- 修改 `num_output_channels` 默认值为 1
- 更新 skip connection 的通道数计算，适配 ResnetEncoder (5 层特征)

```python
# 修改前
def __init__(self, num_ch_enc, scales=range(4), num_output_channels=3, 
             use_skips=True, PixelCoorModu=True):
    ...
    if self.use_skips and i > 0:
        num_ch_in += self.num_ch_enc[i - 1]  # 错误：使用 i-1

# 修改后
def __init__(self, num_ch_enc, scales=range(3), num_output_channels=1, 
             use_skips=True):
    ...
    if self.use_skips and i > 0:
        skip_idx = i + 1  # 正确：decoder i=2 使用 encoder[3], i=1 使用 encoder[2]
        if skip_idx < len(self.num_ch_enc):
            num_ch_in += self.num_ch_enc[skip_idx]
```

### 2. 修改 `DepthDecoderDiffusion.forward`

**修改点**:
- 移除 `norm_pix_coords` 参数
- 修复 decoder 层的顺序处理逻辑 (始终按 2→1→0 处理)
- 修复 skip connection 的索引映射
- 确保 condition feature 与 GT 空间维度一致

```python
# 修改前
def forward(self, input_features, norm_pix_coords, gt=None, mask=None):
    ...
    for scale in sorted_scales:  # 错误：只遍历 self.scales
        x = self.convs[("upconv", scale, 0)](x)
        ...
        if self.use_skips and i > 0:
            x += [input_features[i - 1]]  # 错误索引

# 修改后
def forward(self, input_features, gt=None, mask=None):
    ...
    for i in range(2, -1, -1):  # 正确：始终按 2→1→0 处理
        x = self.convs[("upconv", i, 0)](x)
        ...
        if self.use_skips and i > 0:
            skip_idx = i + 1  # 正确索引
            if skip_idx < len(input_features):
                skip = input_features[skip_idx]
                if skip.shape[-2:] != x[0].shape[-2:]:
                    skip = F.interpolate(skip, size=x[0].shape[-2:], mode="nearest")
                x += [skip]
    ...
    # 确保 condition feature 与 GT 空间维度匹配
    condition = self._resize_to(condition, target_shape[-2:])
```

### 3. 修改 `trainer.py`

**文件**: `DepthEstimation/trainer.py`

**修改点**:
- 更新 decoder 初始化参数
- 移除 `norm_pix_coords` 参数传递

```python
# 修改前
self.models["depth"] = networks.DepthDecoderDiffusion(
    self.models["encoder"].num_ch_enc, 
    self.opt.scales,
    num_output_channels=3,
    PixelCoorModu = not self.opt.disable_pixel_coordinate_modulation)
...
outputs = self.models["depth"](features, norm_pix_coords, gt_for_diffusion)

# 修改后
self.models["depth"] = networks.DepthDecoderDiffusion(
    self.models["encoder"].num_ch_enc, 
    self.opt.scales,
    num_output_channels=1,
    use_skips=True)
...
outputs = self.models["depth"](features, gt_for_diffusion)
```

### 4. 修改 `options.py`

**文件**: `DepthEstimation/options.py`

**修改点**:
- 更新默认 scales 为 `[0, 1, 2]`，与 MonoDiffusion 保持一致

```python
# 修改前
default=[0]

# 修改后
default=[0, 1, 2]
```

## 关键技术要点

### 1. **Encoder-Decoder 特征映射**

对于 ResnetEncoder (5 层特征) 的正确映射：

| Decoder Scale | 空间分辨率 | 使用的 Encoder 特征 | 通道数 |
|--------------|-----------|-------------------|-------|
| 2 | H/16 × W/16 | encoder[3] | 256 |
| 1 | H/8 × W/8 | encoder[2] | 128 |
| 0 | H/4 × W/4 | 无 skip | - |

### 2. **Diffusion Condition Feature 处理**

MonoDiffusion 的处理流程：
1. Decoder 特征 → `conconv` → 16 通道特征
2. 上采样 (bilinear, 2x) → 匹配更高分辨率
3. Resize 到目标 GT 的空间维度
4. 与噪声图像融合进行去噪

### 3. **多尺度渐进式细化**

- Scale 2 (粗) → Scale 1 (中) → Scale 0 (细)
- 每个 scale 使用前一 scale 的输出作为额外条件
- 共享初始噪声，通过上采样传递到更细的 scale

## 验证结果

修复后的测试结果：

```
✓ Decoder初始化成功
✓ Forward成功!
✓ 输出形状正确:
  - scale 0: [2, 1, 256, 320]
  - scale 1: [2, 1, 128, 160]
  - scale 2: [2, 1, 64, 80]
✓ DDIM损失正常:
  - scale 0: 1.114403
  - scale 1: 1.235398
  - scale 2: 1.228084
```

## MonoDiffusion 与本项目的关键差异

| 特性 | MonoDiffusion | 本项目 |
|-----|--------------|-------|
| Encoder | LiteMono (3 层) | ResnetEncoder (5 层) |
| Encoder 输出 | `[48/64, 80/128, 128/224]` | `[64, 64, 128, 256, 512]` |
| Decoder Scales | `[0, 1, 2]` | 现在也是 `[0, 1, 2]` |
| 深度表示 | 单通道视差 | 现在也是单通道 |
| Pixel Coord Modulation | 否 | Teacher 模型使用，Diffusion 模型不使用 |

## 后续建议

1. **训练监控**: 密切关注 DDIM 损失和重建损失的平衡
2. **超参数调整**: 根据实际数据集调整 diffusion steps 和 timesteps
3. **教师模型质量**: 确保教师模型提供高质量的伪 GT
4. **多尺度权重**: 考虑为不同 scale 的损失设置不同权重

## 参考

- MonoDiffusion 论文和代码实现
- MonoDepth2 的 encoder-decoder 架构
- DDIM (Denoising Diffusion Implicit Models)

