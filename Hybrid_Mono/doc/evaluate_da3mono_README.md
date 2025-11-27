# DA3Mono-Large NYUv2 评估脚本

## 📋 概述

本评估脚本用于评估 DA3Mono-Large 模型在 NYUv2 数据集上的性能，**严格遵循 MonoLDP/DepthEstimation/evaluate_nyu_depth.py 的评估逻辑**，确保评估结果的公平性和可比性。

## 📁 文件清单

### 核心文件
- `evaluate_da3mono_nyu_depth.py` - 主评估脚本

### 文档文件
- `doc/evaluate_da3mono_README.md` - 本文档（总览）
- `doc/evaluate_da3mono_usage.md` - 使用指南（详细使用说明）
- `doc/evaluate_da3mono_summary.md` - 快速参考（修改点总结）
- `doc/evaluate_da3mono_changes.md` - 代码修改点详细说明
- `doc/evaluate_da3mono_modifications.md` - 修改说明和警告信息

## 🚀 快速开始

### 1. 基本使用

```bash
cd Hybrid_Mono

python evaluate_da3mono_nyu_depth.py \
  --data_path /oldisk/home/jingyang/monoldp/datasets/nyu_data \
  --eval_split nyu \
  --height 256 \
  --width 320 \
  --batch_size 6 \
  --num_workers 8 \
  --scales 0
```

### 2. 使用本地模型

```bash
python evaluate_da3mono_nyu_depth.py \
  --data_path /oldisk/home/jingyang/monoldp/datasets/nyu_data \
  --eval_split nyu \
  --da3_model_path /path/to/da3mono-large \
  --height 256 \
  --width 320
```

## 📊 评估指标

脚本计算以下 8 个标准深度估计指标：

| 指标 | 说明 | 越小越好 |
|------|------|----------|
| `abs_rel` | 平均绝对相对误差 | ✅ |
| `sq_rel` | 平均平方相对误差 | ✅ |
| `rmse` | 均方根误差 | ✅ |
| `rmse_log` | 对数空间均方根误差 | ✅ |
| `log10` | 平均对数10误差 | ✅ |
| `a1` | 准确率（阈值 1.25） | ❌（越大越好） |
| `a2` | 准确率（阈值 1.25²） | ❌（越大越好） |
| `a3` | 准确率（阈值 1.25³） | ❌（越大越好） |

## ✅ 评估逻辑一致性

### 完全一致的部分

1. ✅ `compute_errors` 函数：完全相同
2. ✅ mask 生成逻辑：完全相同（MIN_DEPTH, global_depth, crop_mask）
3. ✅ median scaling：完全相同
4. ✅ 深度裁剪：完全相同
5. ✅ 所有评估指标计算：完全相同

### 关键修改点

1. **模型加载**：从 4 组件（encoder+decoder+scalenet+regression_heads）变为 1 模型（DA3Mono-Large）
2. **输入预处理**：添加 ImageNet normalization 和尺寸调整（14 的倍数）
3. **推理流程**：从多步骤变为单步，输出从视差变为深度
4. **Global Depth**：从神经网络预测变为估计（95th percentile）或固定值
5. **深度转换**：新增深度到视差的转换函数（适配评估流程）
6. **后处理**：从 batch 处理变为两次前向传播

## ⚠️ 重要警告

### 警告 1：Global Depth 处理方式 ⚠️⚠️

**问题：**
- MonoLDP：使用 scalenet + regression_heads 预测每张图像的 global_depth
- DA3Mono-Large：使用固定值（MAX_DEPTH=10.0）或从预测中估计（95th percentile）

**影响：**
- 影响 mask 生成：`mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < global_depth[i])`
- 影响深度裁剪：`mask_pred_depth[mask_pred_depth > global_depth[i]] = global_depth[i]`

**建议：**
- 默认使用估计模式（不设置 `--use_fixed_max_depth`）
- 在结果中报告 global_depth 统计信息
- 可以尝试两种模式，对比结果差异

### 警告 2：模型架构差异 ⚠️

**问题：**
- MonoLDP：多组件架构（encoder + decoder + scalenet + regression_heads）
- DA3Mono-Large：端到端深度估计模型

**说明：**
- 这不属于评估不公平，而是模型本身的差异
- 在论文中应明确说明这是不同架构的模型对比

### 警告 3：输入预处理差异 ⚠️

**问题：**
- DA3Mono-Large 使用 ImageNet normalization
- DA3Mono-Large 要求输入尺寸是 14 的倍数

**说明：**
- 这是模型本身的特性，不属于评估不公平
- 确保输入尺寸设置正确（256x320 是 14 的倍数）

### 警告 4：深度到视差转换 ⚠️

**问题：**
- DA3Mono-Large 输出深度，需要转换为视差
- 转换：`disp = 1 / depth`，评估时：`pred_depth = 1 / pred_disp`

**说明：**
- 理论上应该一致，但可能有数值精度差异
- 建议检查转换后的视差范围是否合理

## 📝 代码修改点详细说明

### 修改点 1：模型加载（第 160-200 行）

**原代码：**
```python
encoder = networks.ResnetEncoder(...)
depth_decoder = networks.DepthDecoder(...)
scalenet = networks.ScaleNetwork(...)
regression_heads = nn.ModuleList([...])
```

**修改后：**
```python
da3_model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")
da3_model = da3_model.cuda()
da3_model.eval()
```

### 修改点 2：输入预处理（第 240-270 行）

**新增：**
```python
# ImageNet normalization
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
input_normalized = (input_color - mean) / std

# 确保尺寸是 14 的倍数
if H % patch_size != 0 or W % patch_size != 0:
    new_H = math.ceil(H / patch_size) * patch_size
    new_W = math.ceil(W / patch_size) * patch_size
    input_normalized = F.interpolate(input_normalized, size=(new_H, new_W), ...)
```

### 修改点 3：推理流程（第 272-320 行）

**原代码：**
```python
features = encoder(input_color)
depth_factors = scalenet(features)
scale_predictions = [head(factor) for head, factor in zip(regression_heads, depth_factors)]
max_depth = torch.mean(torch.stack(scale_predictions), dim=0)
output = depth_decoder(features, norm_pix_coords)
disp_i = output[("disp", 0)][i:i+1]
```

**修改后：**
```python
output = da3_model.forward(input_da3, ...)
pred_depth_raw = output.get('depth', ...)
pred_depth_i = pred_depth_raw[i].cpu().numpy()
max_depth_i = estimate_global_depth_from_prediction(pred_depth_i)
pred_disp_i = depth_to_disparity(pred_depth_i, opt.min_depth, max_depth_i)
```

### 修改点 4：Global Depth 处理（第 222-235 行）

**原代码：**
```python
max_depth = torch.mean(torch.stack(scale_predictions), dim=0)
global_depth.append(max_depth.cpu().numpy())
```

**修改后：**
```python
if use_fixed_max_depth:
    max_depth_i = MAX_DEPTH
else:
    max_depth_i = estimate_global_depth_from_prediction(pred_depth_i)
batch_global_depths.append(max_depth_i)
```

### 修改点 5：评估部分（第 400-480 行）

**完全一致，无修改** ✅

## 📚 文档索引

| 文档 | 内容 | 适用场景 |
|------|------|----------|
| `evaluate_da3mono_README.md` | 总览和快速开始 | 初次使用 |
| `evaluate_da3mono_usage.md` | 详细使用指南 | 日常使用 |
| `evaluate_da3mono_summary.md` | 快速参考 | 快速查找 |
| `evaluate_da3mono_changes.md` | 代码修改点详细说明 | 代码审查 |
| `evaluate_da3mono_modifications.md` | 修改说明和警告信息 | 深入理解 |

## 🔍 评估公平性检查清单

运行评估前，请确认：

- [ ] **输入尺寸**：256x320（14 的倍数）
- [ ] **Global Depth 模式**：估计模式（默认）或固定模式
- [ ] **后处理**：根据需求启用/禁用
- [ ] **Median Scaling**：与 MonoLDP 使用相同的设置
- [ ] **评估指标**：确认所有 8 个指标都正确计算
- [ ] **Mask 生成**：确认使用相同的 MIN_DEPTH、global_depth、crop_mask
- [ ] **深度裁剪**：确认使用相同的裁剪逻辑

## 💡 使用建议

1. **默认使用估计模式**（不设置 `--use_fixed_max_depth`）
2. **在结果中明确说明** Global Depth 处理方式
3. **可以尝试两种模式**，对比结果差异
4. **在论文中明确说明**这是不同架构的模型对比

## 📞 问题排查

如果遇到问题，请参考：
- 使用指南：`doc/evaluate_da3mono_usage.md`
- 代码修改说明：`doc/evaluate_da3mono_changes.md`
- 原始评估脚本：`MonoLDP/DepthEstimation/evaluate_nyu_depth.py`

## 📄 许可证

与 MonoLDP 项目保持一致。

