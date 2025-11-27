# 三个评估脚本的评估逻辑对比

本文档详细对比了三个NYUv2深度估计评估脚本的评估逻辑差异：

1. **`evaluate_nyu_depth.py`** - MonoLDP原始评估脚本
2. **`evaluate_da3mono_nyu_depth.py`** - DA3Mono-Large评估脚本（适配MonoLDP流程）
3. **`evaluate_da3mono_nyu_depth_2.py`** - DA3Mono-Large评估脚本（DA3范式）

---

## 📊 核心差异总览

| 特性 | `evaluate_nyu_depth.py` | `evaluate_da3mono_nyu_depth.py` | `evaluate_da3mono_nyu_depth_2.py` |
|------|-------------------------|----------------------------------|-----------------------------------|
| **模型架构** | encoder+decoder+scalenet+regression_heads | DA3Mono-Large（端到端） | DA3Mono-Large（端到端） |
| **输入分辨率** | 从权重文件读取（通常256x320） | opt.height×opt.width（默认256x320） | 固定518×518（DA3标准） |
| **输入预处理** | 数据集默认预处理 | ImageNet归一化 + 14倍数调整 | ImageNet归一化 |
| **输出格式** | 视差（disparity） | 深度→视差转换 | 直接深度 |
| **Global Depth** | scalenet+regression_heads预测 | 估计（95th percentile）或固定 | 不使用（固定范围） |
| **对齐方法** | Median Scaling | Median Scaling | Least Squares（默认）或Median |
| **Crop Mask** | `dataset.default_crop` | `dataset.default_crop` | 硬编码Eigen Crop |
| **评估指标** | 8个（含log10） | 8个（含log10） | 7个（不含log10） |
| **设计目标** | MonoLDP模型评估 | 适配MonoLDP评估流程 | DA3标准评估范式 |

---

## 🔍 详细差异分析

### 1. 模型加载与推理

#### `evaluate_nyu_depth.py`
```python
# 加载4个组件
encoder = networks.ResnetEncoder(opt.num_layers, False)
depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, [0])
scalenet = networks.ScaleNetwork(encoder.num_ch_enc)
regression_heads = nn.ModuleList([...])

# 推理流程
features = encoder(input_color)
depth_factors = scalenet(features)
scale_predictions = [head(factor) for head, factor in zip(regression_heads, depth_factors)]
max_depth = torch.mean(torch.stack(scale_predictions), dim=0)  # 预测global_depth
output = depth_decoder(features, norm_pix_coords)
pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, max_depth)
```

**特点：**
- 多组件架构
- 输出视差（disparity）
- 通过scalenet和regression_heads预测每张图像的global_depth

#### `evaluate_da3mono_nyu_depth.py`
```python
# 加载DA3Mono-Large模型
da3_model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")

# 推理流程
output = da3_model.forward(input_da3, extrinsics=None, intrinsics=None, ...)
pred_depth_raw = output.get('depth')  # 直接输出深度
# 转换为视差以适配评估流程
pred_disp = depth_to_disparity(pred_depth_i, opt.min_depth, max_depth_i)
```

**特点：**
- 端到端单模型
- 直接输出深度，然后转换为视差
- 从预测深度估计global_depth（95th percentile）或使用固定值

#### `evaluate_da3mono_nyu_depth_2.py`
```python
# 加载DA3Mono-Large模型
model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")

# 推理流程
output = model.forward(input_da3, ...)
pred_depth = output.get('depth')  # 直接使用深度，不转换
```

**特点：**
- 端到端单模型
- 直接使用深度，不转换为视差
- 不使用global_depth，使用固定深度范围

---

### 2. 输入分辨率处理

#### `evaluate_nyu_depth.py`
- 从权重文件（`encoder.pth`）中读取训练时的输入尺寸
- 使用该尺寸创建数据集和进行推理
- 通常为256×320

#### `evaluate_da3mono_nyu_depth.py`
- 使用`opt.height`和`opt.width`（默认256×320）
- 确保输入尺寸是14的倍数（DA3的patch size）
- 如果不符合，自动调整到最近的14的倍数

```python
eval_height = opt.height if hasattr(opt, 'height') else 256
eval_width = opt.width if hasattr(opt, 'width') else 320
# 确保是14的倍数
if H % patch_size != 0 or W % patch_size != 0:
    new_H = math.ceil(H / patch_size) * patch_size
    new_W = math.ceil(W / patch_size) * patch_size
```

#### `evaluate_da3mono_nyu_depth_2.py`
- **强制使用518×518**（DA3标准分辨率）
- 推理时resize到518×518，然后resize回GT尺寸

```python
INPUT_SIZE = 518  # 固定DA3标准分辨率
input_da3_resized = F.interpolate(input_norm, size=(INPUT_SIZE, INPUT_SIZE), ...)
# 推理后resize回GT尺寸
pred_depth = F.interpolate(pred_depth_raw, size=(H_gt, W_gt), ...)
```

---

### 3. 输入预处理

#### `evaluate_nyu_depth.py`
- 使用数据集默认预处理
- 通常已经包含归一化等操作

#### `evaluate_da3mono_nyu_depth.py`
- 添加ImageNet归一化
- 确保输入尺寸是14的倍数

```python
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
input_normalized = (input_color - mean) / std
```

#### `evaluate_da3mono_nyu_depth_2.py`
- 添加ImageNet归一化
- Resize到518×518

```python
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
input_norm = (input_color - mean) / std
input_da3_resized = F.interpolate(input_norm, size=(518, 518), ...)
```

---

### 4. Global Depth处理

#### `evaluate_nyu_depth.py`
- 通过scalenet和regression_heads预测每张图像的global_depth
- 用于mask生成和深度裁剪

```python
depth_factors = scalenet(features)
scale_predictions = [head(factor) for head, factor in zip(regression_heads, depth_factors)]
max_depth = torch.mean(torch.stack(scale_predictions), dim=0)  # 每张图像的global_depth
global_depth.append(max_depth.cpu().numpy())

# 在评估中使用
mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < global_depth[i])
mask_pred_depth[mask_pred_depth > global_depth[i]] = global_depth[i]
```

#### `evaluate_da3mono_nyu_depth.py`
- 从预测深度中估计global_depth（95th percentile）或使用固定值
- 用于mask生成和深度裁剪（与MonoLDP一致）

```python
def estimate_global_depth_from_prediction(pred_depth, percentile=95.0):
    valid_depth = pred_depth[pred_depth > 0.01]
    estimated_max_depth = np.percentile(valid_depth, percentile)
    return estimated_max_depth

# 在评估中使用（与MonoLDP一致）
mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < global_depth[i])
mask_pred_depth[mask_pred_depth > global_depth[i]] = global_depth[i]
```

#### `evaluate_da3mono_nyu_depth_2.py`
- **不使用global_depth**
- 使用固定的深度范围（1e-3到10）

```python
# 固定深度范围，不使用global_depth
mask = np.logical_and(gt_depth > 1e-3, gt_depth < 10)
pred_final = np.clip(pred_final, 1e-3, 10)
```

---

### 5. 深度对齐策略

#### `evaluate_nyu_depth.py`
- **Median Scaling**（中位数缩放）
- 公式：`pred_depth *= median(gt) / median(pred)`

```python
if not opt.disable_median_scaling:
    ratio = np.median(mask_gt_depth) / np.median(mask_pred_depth)
    mask_pred_depth *= ratio
```

#### `evaluate_da3mono_nyu_depth.py`
- **Median Scaling**（与MonoLDP一致）

```python
if not opt.disable_median_scaling:
    ratio = np.median(mask_gt_depth) / np.median(mask_pred_depth)
    mask_pred_depth *= ratio
```

#### `evaluate_da3mono_nyu_depth_2.py`
- **Least Squares Alignment**（最小二乘法对齐，默认）
- 可选Median Scaling
- 公式：`pred = scale * pred + shift`（通过最小二乘求解）

```python
if getattr(opt, 'use_least_squares', True):
    # Least Squares: pred = scale * pred + shift
    coeffs = np.linalg.lstsq(
        np.stack([pred_valid, np.ones_like(pred_valid)], axis=1), 
        gt_valid, rcond=None
    )[0]
    scale, shift = coeffs
    pred_final = pred_valid * scale + shift
else:
    # Median Scaling
    ratio = np.median(gt_valid) / np.median(pred_valid)
    pred_final = pred_valid * ratio
```

**差异说明：**
- Least Squares对齐更精确，适合相对深度模型
- Median Scaling更简单，适合绝对深度模型

---

### 6. Crop Mask生成

#### `evaluate_nyu_depth.py`
- 使用`dataset.default_crop`（动态获取）

```python
crop_mask = np.zeros(mask.shape)
crop_mask[dataset.default_crop[2]:dataset.default_crop[3], 
          dataset.default_crop[0]:dataset.default_crop[1]] = 1
```

#### `evaluate_da3mono_nyu_depth.py`
- 使用`dataset.default_crop`（与MonoLDP一致）

```python
crop_mask = np.zeros(mask.shape)
crop_mask[dataset.default_crop[2]:dataset.default_crop[3], 
          dataset.default_crop[0]:dataset.default_crop[1]] = 1
```

#### `evaluate_da3mono_nyu_depth_2.py`
- 使用硬编码的Eigen Crop

```python
# Eigen Crop (NYU标准裁剪)
crop = np.array([45, 471, 41, 601]).astype(np.int32)
crop_mask = np.zeros(mask.shape)
crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
```

---

### 7. 深度到视差转换

#### `evaluate_nyu_depth.py`
- 直接输出视差，无需转换
- 使用`disp_to_depth`函数将视差转换为深度进行评估

```python
pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, max_depth)
# 评估时转回深度
pred_depth = 1 / pred_disp
```

#### `evaluate_da3mono_nyu_depth.py`
- DA3输出深度，需要转换为视差以适配评估流程
- 评估时再转回深度

```python
def depth_to_disparity(depth, min_depth=0.1, max_depth=10.0):
    depth = np.clip(depth, min_depth, max_depth)
    disp = 1.0 / depth  # 转换为视差
    return disp

# 推理后转换
pred_disp_i = depth_to_disparity(pred_depth_i, opt.min_depth, max_depth_i)

# 评估时转回深度
pred_depth = 1 / pred_disp
```

#### `evaluate_da3mono_nyu_depth_2.py`
- **直接使用深度，不转换为视差**
- 简化流程

```python
# 直接使用深度，不转换
pred_depth = pred_depth_raw[0].cpu().numpy()
# 直接评估深度
errors.append(compute_errors(gt_valid, pred_final))
```

---

### 8. 评估指标

#### `evaluate_nyu_depth.py` 和 `evaluate_da3mono_nyu_depth.py`
- **8个指标**：`abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3`

```python
def compute_errors(gt, pred):
    # ... 计算各种误差 ...
    log10 = np.mean(np.abs(np.log10(pred / gt)))  # 包含log10
    return abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3
```

#### `evaluate_da3mono_nyu_depth_2.py`
- **7个指标**：`abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3`（**缺少log10**）

```python
def compute_errors(gt, pred):
    # ... 计算各种误差 ...
    # 注意：没有计算log10
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
```

---

### 9. 后处理

#### `evaluate_nyu_depth.py`
- 支持左右翻转后处理
- 在batch维度处理

```python
if opt.post_process:
    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
    # ... 推理 ...
    N = pred_disp.shape[0] // 2
    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
```

#### `evaluate_da3mono_nyu_depth.py`
- 支持左右翻转后处理
- 需要两次前向传播（原始+翻转）

```python
if opt.post_process:
    # 翻转输入并重新推理
    input_flipped = torch.flip(input_normalized, [3])
    output_flipped = da3_model.forward(input_flipped_da3, ...)
    # 应用后处理
    processed_disp = batch_post_process_disparity(l_disp, r_disp_flipped)
```

#### `evaluate_da3mono_nyu_depth_2.py`
- **不支持后处理**
- 直接评估

---

## 📈 评估流程对比

### `evaluate_nyu_depth.py` 流程
```
输入图像 → encoder → features
                ↓
         scalenet → depth_factors
                ↓
    regression_heads → global_depth (预测)
                ↓
    depth_decoder → 视差 (disparity)
                ↓
    disp_to_depth → 深度 (depth)
                ↓
    评估：Median Scaling + compute_errors
```

### `evaluate_da3mono_nyu_depth.py` 流程
```
输入图像 → ImageNet归一化 → 14倍数调整
                ↓
        DA3Mono-Large → 深度 (depth)
                ↓
    depth_to_disparity → 视差 (disparity)
                ↓
    估计global_depth (95th percentile)
                ↓
    评估：Median Scaling + compute_errors
    (与MonoLDP完全一致)
```

### `evaluate_da3mono_nyu_depth_2.py` 流程
```
输入图像 → ImageNet归一化 → Resize到518×518
                ↓
        DA3Mono-Large → 深度 (depth)
                ↓
    Resize回GT尺寸
                ↓
    Least Squares对齐 (或Median Scaling)
                ↓
    评估：compute_errors
    (DA3标准范式)
```

---

## ⚠️ 重要注意事项

### 1. 评估结果可能不同

由于对齐策略、global_depth处理、crop mask等差异，三个脚本的评估结果**可能不同**：

- **`evaluate_nyu_depth.py`** vs **`evaluate_da3mono_nyu_depth.py`**：
  - 评估逻辑基本一致，但global_depth来源不同（预测 vs 估计）
  - 结果应该比较接近，但可能有细微差异

- **`evaluate_da3mono_nyu_depth_2.py`** vs 其他两个：
  - 对齐策略不同（Least Squares vs Median Scaling）
  - 不使用global_depth
  - 结果可能差异较大

### 2. 使用建议

| 场景 | 推荐脚本 |
|------|---------|
| 评估MonoLDP模型 | `evaluate_nyu_depth.py` |
| 评估DA3Mono-Large并与MonoLDP对比 | `evaluate_da3mono_nyu_depth.py` |
| 评估DA3Mono-Large的标准性能 | `evaluate_da3mono_nyu_depth_2.py` |
| 论文中报告DA3性能 | `evaluate_da3mono_nyu_depth_2.py`（DA3标准范式） |
| 公平对比不同模型 | `evaluate_da3mono_nyu_depth.py`（使用相同评估流程） |

### 3. 关键差异总结

1. **对齐方法**：Median Scaling（前两个）vs Least Squares（第三个）
2. **Global Depth**：预测/估计（前两个）vs 不使用（第三个）
3. **输入分辨率**：256×320（前两个）vs 518×518（第三个）
4. **深度转换**：深度↔视差（前两个）vs 直接深度（第三个）
5. **评估指标**：8个（前两个）vs 7个（第三个，缺少log10）

---

## 📝 代码位置参考

### `evaluate_nyu_depth.py`
- 模型加载：第85-109行
- 推理流程：第121-180行
- 评估逻辑：第234-263行

### `evaluate_da3mono_nyu_depth.py`
- 模型加载：第201-263行
- 推理流程：第283-455行
- 评估逻辑：第492-534行

### `evaluate_da3mono_nyu_depth_2.py`
- 模型加载：第82-122行
- 推理流程：第127-193行
- 评估逻辑：第195-236行

---

## 🔗 相关文档

- `doc/evaluate_da3mono_README.md` - DA3Mono评估脚本总览
- `doc/evaluate_da3mono_usage.md` - DA3Mono评估脚本使用指南
- `doc/evaluate_da3mono_summary.md` - DA3Mono评估脚本快速参考
- `doc/evaluate_da3mono_changes.md` - DA3Mono评估脚本代码修改点
- `doc/evaluate_da3mono_modifications.md` - DA3Mono评估脚本修改说明

---

**最后更新：** 2024年

