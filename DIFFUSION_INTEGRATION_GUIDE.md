# MonoDiffusion 集成到 MonoLDP 指南

## 概述
本指南详细说明如何将 MonoDiffusion 的扩散模块集成到 MonoLDP 项目中，实现基于扩散模型的单目深度估计。

## 核心思想
MonoDiffusion 使用 DDIM (Denoising Diffusion Implicit Models) 来精炼深度预测：
1. 传统的深度解码器生成初始深度特征
2. 扩散模块将深度预测视为去噪过程，通过多步迭代精炼深度图
3. 训练时使用 DDIM loss，推理时使用 DDIM 采样过程

## 第一步：复制必要文件

### 1.1 复制 diffusers 模块
```bash
cp -r MonoDiffusion/diffusers/ DepthEstimation/
```
这个目录包含：
- `scheduling_ddim.py`: DDIM 调度器，控制扩散过程的时间步

### 1.2 参考 layers.py
MonoDiffusion/layers.py 包含基础层定义，但 DepthEstimation 已有 layers.py
需要确保包含必要的辅助函数（upsample, ConvBlock 等）

## 第二步：创建支持扩散的深度解码器

需要创建两个新的解码器（二选一）：

### 选项A：基于 MonoDiffusion 的 HRDepthDecoder
- 文件：`DepthEstimation/networks/depth_decoder_diffusion.py`
- 特点：
  - 多尺度扩散（3个尺度）
  - 每个尺度有独立的扩散模型和调度器
  - 级联结构：从粗到细逐步精炼

### 选项B：基于 MonoDiffusion 的 HRDFDepthDecoder  
- 文件：`DepthEstimation/networks/hr_decoder_diffusion.py`
- 特点：
  - 单尺度扩散（最高分辨率）
  - 包含 HR-Depth 的特征融合和注意力机制
  - 需要额外的 hr_layers.py

**推荐使用选项A（HRDepthDecoder）**，因为它更简洁且与 MonoLDP 的架构更兼容。

## 第三步：修改网络结构

### 3.1 在 `DepthEstimation/networks/__init__.py` 中添加：
```python
from .depth_decoder_diffusion import HRDepthDecoder as HRDepthDecoderDiffusion
```

### 3.2 在 trainer.py 中修改深度解码器初始化：

**原来的代码（第74-77行）：**
```python
self.models["depth"] = networks.DepthDecoder(
    self.models["encoder"].num_ch_enc, 
    self.opt.scales,
    PixelCoorModu = not self.opt.disable_pixel_coordinate_modulation
)
```

**修改为：**
```python
# 添加预训练的教师模型（用于生成伪GT）
self.models["pre_depth_encoder"] = networks.ResnetEncoder(
    self.opt.num_layers, self.opt.weights_init == "pretrained"
)
self.models["pre_depth_decoder"] = networks.DepthDecoder(
    self.models["pre_depth_encoder"].num_ch_enc, 
    self.opt.scales
)

# 加载预训练权重（如果有）
if os.path.exists("path/to/pretrained/encoder.pth"):
    encoder_dict = torch.load("path/to/pretrained/encoder.pth")
    decoder_dict = torch.load("path/to/pretrained/depth.pth")
    self.models["pre_depth_encoder"].load_state_dict(encoder_dict)
    self.models["pre_depth_decoder"].load_state_dict(decoder_dict)
    
# 冻结预训练模型
for param in self.models["pre_depth_encoder"].parameters():
    param.requires_grad = False
for param in self.models["pre_depth_decoder"].parameters():
    param.requires_grad = False
    
self.models["pre_depth_encoder"].to(self.device)
self.models["pre_depth_decoder"].to(self.device)
self.models["pre_depth_encoder"].eval()
self.models["pre_depth_decoder"].eval()

# 使用带扩散的深度解码器
from networks import HRDepthDecoderDiffusion
self.models["depth"] = HRDepthDecoderDiffusion(
    self.models["encoder"].num_ch_enc, 
    self.opt.scales
)
```

## 第四步：修改训练流程

### 4.1 修改 `process_batch` 方法

在 `trainer.py` 的 `process_batch` 方法中（约第303行开始）：

**原来的前向传播：**
```python
features = self.models["encoder"](inputs["color_aug", 0, 0])
outputs = self.models["depth"](features, norm_pix_coords)
```

**修改为：**
```python
# 1. 使用教师模型生成伪GT
with torch.no_grad():
    pre_features = self.models["pre_depth_encoder"](inputs["color_aug", 0, 0])
    pre_disp = self.models["pre_depth_decoder"](pre_features, norm_pix_coords)

# 2. 学生模型前向传播
features = self.models["encoder"](inputs["color_aug", 0, 0])

# 3. 准备扩散输入（需要 disp_diffusion）
gt_for_diffusion = {}
for scale in self.opt.scales:
    # 从 sigmoid 空间转换到 logit 空间用于扩散
    gt_for_diffusion[("disp_diffusion", scale)] = pre_disp[("disp", scale)].detach()

# 4. 使用扩散解码器
outputs = self.models["depth"](features, gt_for_diffusion)

# 5. 保存预测的 disparity 用于后续处理
for scale in self.opt.scales:
    outputs[('predisp', scale)] = pre_disp[('disp', scale)]
```

### 4.2 修改损失计算

在 `compute_losses` 返回后添加扩散损失（约第610行）：

```python
# 原有的光度损失
losses = self.compute_losses(inputs, outputs)

# 添加 L1 损失（学生和教师的一致性）
l1_loss = 0
for i in self.opt.scales:
    l1_loss += F.l1_loss(pre_disp['disp', i], outputs["disp", i])
losses['l1'] = l1_loss / self.num_scales

# 添加 DDIM 损失
losses['ddim'] = (outputs["ddim_loss", 0] + 
                  outputs["ddim_loss", 1] + 
                  outputs["ddim_loss", 2]) / self.num_scales

# 保存原始光度损失
losses['photometric'] = losses["loss"]

# 总损失 = 光度损失 + L1损失 + DDIM损失
losses["loss"] = 1.0 * losses['photometric'] + \
                 1.0 * losses['l1'] + \
                 1.0 * losses['ddim']
```

### 4.3 添加 mask 训练（可选，用于不确定性估计）

在 `run_epoch` 中添加第二阶段训练：

```python
def process_batch_mask(self, inputs, outputs):
    """使用 mask 机制进一步精炼深度"""
    features = self.models["encoder"](inputs["color_aug", 0, 0])
    
    # 使用 identity selection 作为置信度 mask
    identity_mask = outputs["identity_selection/{}".format(0)].unsqueeze(1)
    
    outputs_mask = self.models["depth"](
        features, 
        outputs,  # 使用第一阶段的输出作为GT
        identity_mask
    )
    
    mask_losses = 0
    for scale in self.opt.scales:
        mask_losses += 0.1 * torch.abs(
            outputs_mask[('disp', scale)] - outputs[('disp', scale)].detach()
        ).mean()
    
    return outputs_mask, mask_losses / self.num_scales
```

在 `run_epoch` 中调用：

```python
# 第一阶段：正常训练
outputs, losses = self.process_batch(inputs)
self.model_optimizer.zero_grad()
if self.use_pose_net:
    self.model_pose_optimizer.zero_grad()
losses["loss"].backward()
self.model_optimizer.step()
if self.use_pose_net:
    self.model_pose_optimizer.step()

# 第二阶段：mask 训练
outputs_mask, losses_mask = self.process_batch_mask(inputs, outputs)
self.model_optimizer.zero_grad()
losses_mask.backward()
self.model_optimizer.step()
```

## 第五步：更新日志记录

在 `log` 方法中添加扩散相关的可视化：

```python
# 可视化预测的深度和教师模型的深度
writer.add_image(
    "predisp_{}/{}".format(s, j),
    colormap_magma(outputs[('predisp', s)][j]), 
    self.step
)

# 可视化残差（教师 - 学生）
writer.add_image(
    "residual_{}/{}".format(s, j),
    colormap_jet(outputs[('predisp', s)][j] - outputs[("disp", s)][j]), 
    self.step
)

# 可视化扩散过程中的中间结果
if s == 0:
    for i in range(3):
        writer.add_image(
            "re-diffusion_{}/{}".format(i, j),
            colormap_magma(outputs["re-diffusion"][i][j]), 
            self.step
        )
    
    # 可视化不确定性
    depth_uncertainty = torch.cat([
        outputs["re-diffusion"][-1], 
        outputs["re-diffusion"][-2], 
        outputs["re-diffusion"][-3]
    ], 1)
    depth_uncertainty = torch.std(depth_uncertainty, dim=1, keepdim=True)
    writer.add_image(
        "uncertainty_{}/{}".format(s, j),
        colormap_jet(depth_uncertainty[j]), 
        self.step
    )
```

## 第六步：命令行参数（可选）

在 `options.py` 中添加扩散相关参数：

```python
self.parser.add_argument("--use_diffusion",
                        action="store_true",
                        help="if set, use diffusion decoder")
self.parser.add_argument("--diffusion_steps",
                        type=int,
                        default=20,
                        help="number of diffusion inference steps")
self.parser.add_argument("--diffusion_timesteps",
                        nargs="+",
                        type=int,
                        default=[250, 200, 150],
                        help="number of training timesteps for each scale")
self.parser.add_argument("--teacher_weights_folder",
                        type=str,
                        help="path to pretrained teacher model")
```

## 关键模块说明

### ScheduledCNNRefine
这是扩散模型的核心网络：
- **输入**: 噪声图像 + 时间步 + 特征图
- **输出**: 预测的噪声
- **组件**:
  - `noise_embedding`: 将噪声图像嵌入到特征空间
  - `time_embedding`: 时间步的嵌入
  - `pred`: 预测噪声的网络

### DDIMScheduler
控制扩散过程：
- `add_noise()`: 训练时向图像添加噪声
- `step()`: 推理时从噪声恢复图像
- `set_timesteps()`: 设置推理步数

### CNNDDIMPipeline
推理管道：
- 从纯高斯噪声开始
- 迭代调用模型预测噪声
- 使用调度器更新图像
- 返回最终的深度图

## 训练建议

1. **两阶段训练**：
   - 第一阶段：不使用扩散，训练基础深度网络
   - 第二阶段：冻结编码器，只训练扩散模块

2. **损失权重**：
   - 光度损失权重：1.0
   - L1损失权重：0.5-1.0
   - DDIM损失权重：1.0

3. **学习率**：
   - 扩散模块可以使用较小的学习率（1e-4）

4. **推理步数**：
   - 训练：不影响（随机采样时间步）
   - 推理：5-20步（步数越多越慢但质量越好）

## 可能遇到的问题

1. **内存不足**：扩散模块增加了显存占用
   - 解决：减小 batch size 或减少扩散步数

2. **训练不稳定**：扩散损失过大
   - 解决：降低 DDIM 损失权重

3. **速度慢**：推理时扩散步骤多
   - 解决：减少 `diffusion_inference_steps`

## 总结

集成步骤：
1. ✅ 复制 `diffusers/` 文件夹
2. ✅ 创建 `depth_decoder_diffusion.py`
3. ✅ 修改 trainer.py 的网络初始化
4. ✅ 修改 trainer.py 的前向传播和损失计算
5. ✅ 更新日志和可视化

集成后的优势：
- 更精确的深度估计
- 不确定性估计能力
- 更好的边缘保持

