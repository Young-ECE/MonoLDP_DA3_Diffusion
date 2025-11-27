# DA3Mono-Large 评估脚本总结

## 快速开始

```bash
cd Hybrid_Mono
python evaluate_da3mono_nyu_depth.py \
  --data_path /path/to/nyu_data \
  --eval_split nyu \
  --height 256 \
  --width 320 \
  --batch_size 6 \
  --num_workers 8 \
  --scales 0
```

## 代码修改点总结

### 1. 模型加载（第 78-120 行）

**修改前（MonoLDP）：**
- 加载 encoder, decoder, scalenet, regression_heads 四个组件
- 从权重文件加载状态字典

**修改后（DA3Mono-Large）：**
- 直接加载 DA3Mono-Large 模型（端到端）
- 从 HuggingFace 或本地路径加载

**代码位置：**
```python
# 原代码：85-100 行
encoder = networks.ResnetEncoder(...)
depth_decoder = networks.DepthDecoder(...)
scalenet = networks.ScaleNetwork(...)
regression_heads = nn.ModuleList([...])

# 新代码：120-140 行
da3_model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")
da3_model = da3_model.cuda()
da3_model.eval()
```

---

### 2. 输入预处理（第 152-175 行）

**修改前（MonoLDP）：**
- 直接使用数据加载器输出的图像（可能已经预处理）

**修改后（DA3Mono-Large）：**
- 添加 ImageNet normalization
- 确保输入尺寸是 14 的倍数（DA3Mono-Large 的 patch size）

**代码位置：**
```python
# 新代码：152-175 行
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

---

### 3. 模型推理（第 177-220 行）

**修改前（MonoLDP）：**
- encoder → features
- scalenet → depth_factors
- regression_heads → scale_predictions → max_depth
- depth_decoder → output[("disp", 0)]（视差）

**修改后（DA3Mono-Large）：**
- 直接调用 da3_model.forward() → depth（深度）
- 需要转换为视差以适配评估流程

**代码位置：**
```python
# 原代码：139-155 行
features = encoder(input_color)
depth_factors = scalenet(features)
scale_predictions = [head(factor) for head, factor in zip(regression_heads, depth_factors)]
max_depth = torch.mean(torch.stack(scale_predictions), dim=0)
output = depth_decoder(features, norm_pix_coords)
disp_i = output[("disp", 0)][i:i+1]

# 新代码：177-220 行
input_da3 = input_normalized.unsqueeze(1)  # (B, 1, 3, H, W)
output = da3_model.forward(input_da3, ...)
pred_depth_raw = output.get('depth', ...)
pred_depth_i = pred_depth_raw[i].cpu().numpy()
pred_disp_i = depth_to_disparity(pred_depth_i, opt.min_depth, max_depth_i)
```

---

### 4. Global Depth 处理（第 222-235 行）

**修改前（MonoLDP）：**
- 使用 scalenet + regression_heads 预测每张图像的 global_depth

**修改后（DA3Mono-Large）：**
- 选项1：使用固定 MAX_DEPTH=10.0（`--use_fixed_max_depth`）
- 选项2：从预测深度中估计（95th percentile，默认）

**代码位置：**
```python
# 原代码：141-143 行
scale_predictions = [head(factor) for head, factor in zip(regression_heads, depth_factors)]
max_depth = torch.mean(torch.stack(scale_predictions), dim=0)
global_depth.append(max_depth.cpu().numpy())

# 新代码：222-235 行
if use_fixed_max_depth:
    max_depth_i = MAX_DEPTH  # 固定值 10.0
else:
    max_depth_i = estimate_global_depth_from_prediction(pred_depth_i)  # 从预测中估计
batch_global_depths.append(max_depth_i)
```

**新增函数：**
```python
def estimate_global_depth_from_prediction(pred_depth, percentile=95.0):
    """从预测深度中估计 global_depth"""
    valid_depth = pred_depth[pred_depth > 0.01]
    if len(valid_depth) == 0:
        return 10.0
    estimated_max_depth = np.percentile(valid_depth, percentile)
    return min(estimated_max_depth, 10.0)
```

---

### 5. 深度到视差转换（第 70-90 行）

**新增函数：**
```python
def depth_to_disparity(depth, min_depth=0.1, max_depth=10.0):
    """将深度转换为视差，用于适配评估流程"""
    depth = np.clip(depth, min_depth, max_depth)
    disp = 1.0 / depth
    return disp
```

**说明：**
- DA3Mono-Large 输出深度，但评估流程期望视差
- 转换：`disp = 1 / depth`
- 评估时：`pred_depth = 1 / pred_disp`（与 MonoLDP 一致）

---

### 6. 后处理（第 237-280 行）

**修改前（MonoLDP）：**
- 在 batch 维度上处理，一次前向传播

**修改后（DA3Mono-Large）：**
- 需要两次前向传播（原始 + 翻转）

**代码位置：**
```python
# 原代码：133-178 行
if opt.post_process:
    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
    # 一次前向传播处理所有图像
    # ...
    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

# 新代码：237-280 行
if opt.post_process:
    # 需要两次前向传播
    input_flipped = torch.flip(input_normalized, [3])
    output_flipped = da3_model.forward(input_flipped_da3, ...)
    # ...
    processed_disp = batch_post_process_disparity(l_disp[i:i+1], r_disp_flipped[i:i+1])
```

---

### 7. 评估部分（第 282-350 行）

**完全一致，无修改**
- `compute_errors` 函数完全相同
- mask 生成逻辑完全相同
- median scaling 逻辑完全相同
- 深度裁剪逻辑完全相同

---

## 关键警告信息

### ⚠️ 警告 1：Global Depth 处理方式

**问题：**
- MonoLDP 使用神经网络预测每张图像的 global_depth
- DA3Mono-Large 使用固定值或从预测中估计

**影响：**
- 影响 mask 生成：`mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < global_depth[i])`
- 影响深度裁剪：`mask_pred_depth[mask_pred_depth > global_depth[i]] = global_depth[i]`

**建议：**
- 默认使用估计模式（不设置 `--use_fixed_max_depth`）
- 在结果中报告 global_depth 统计信息
- 可以尝试两种模式，对比结果差异

---

### ⚠️ 警告 2：模型架构差异

**问题：**
- MonoLDP：多组件架构（encoder + decoder + scalenet + regression_heads）
- DA3Mono-Large：端到端深度估计模型

**影响：**
- 模型容量和表达能力不同
- 训练数据和训练方式不同

**说明：**
- 这不属于评估不公平，而是模型本身的差异
- 在论文中应明确说明这是不同架构的模型对比

---

### ⚠️ 警告 3：输入预处理差异

**问题：**
- DA3Mono-Large 使用 ImageNet normalization
- DA3Mono-Large 要求输入尺寸是 14 的倍数

**影响：**
- 归一化差异可能影响模型性能
- 尺寸调整可能引入插值误差

**说明：**
- 这是模型本身的特性，不属于评估不公平
- 确保输入尺寸设置正确（256x320 是 14 的倍数）

---

### ⚠️ 警告 4：深度到视差转换

**问题：**
- DA3Mono-Large 输出深度，需要转换为视差
- 转换：`disp = 1 / depth`，评估时：`pred_depth = 1 / pred_disp`

**影响：**
- 理论上应该一致，但可能有数值精度差异

**说明：**
- 转换公式理论上一致
- 建议检查转换后的视差范围是否合理

---

### ⚠️ 警告 5：后处理实现差异

**问题：**
- DA3Mono-Large 需要两次前向传播
- MonoLDP 在 batch 维度处理

**影响：**
- 理论上结果应该一致
- 但可能有微小的数值差异

**说明：**
- 使用相同的 `batch_post_process_disparity` 函数
- 结果应该一致

---

## 评估公平性检查清单

运行评估前，请确认：

- [ ] **输入尺寸**：256x320（14 的倍数）
- [ ] **Global Depth 模式**：估计模式（默认）或固定模式
- [ ] **后处理**：根据需求启用/禁用
- [ ] **Median Scaling**：与 MonoLDP 使用相同的设置
- [ ] **评估指标**：确认所有 8 个指标都正确计算
- [ ] **Mask 生成**：确认使用相同的 MIN_DEPTH、global_depth、crop_mask
- [ ] **深度裁剪**：确认使用相同的裁剪逻辑

---

## 使用示例

### 基本评估

```bash
python evaluate_da3mono_nyu_depth.py \
  --data_path /oldisk/home/jingyang/monoldp/datasets/nyu_data \
  --eval_split nyu \
  --height 256 \
  --width 320 \
  --batch_size 6 \
  --num_workers 8 \
  --scales 0
```

### 使用本地模型

```bash
python evaluate_da3mono_nyu_depth.py \
  --data_path /oldisk/home/jingyang/monoldp/datasets/nyu_data \
  --eval_split nyu \
  --da3_model_path /path/to/da3mono-large \
  --height 256 \
  --width 320
```

### 使用固定 MAX_DEPTH

```bash
python evaluate_da3mono_nyu_depth.py \
  --data_path /oldisk/home/jingyang/monoldp/datasets/nyu_data \
  --eval_split nyu \
  --use_fixed_max_depth \
  --height 256 \
  --width 320
```

### 启用后处理

```bash
python evaluate_da3mono_nyu_depth.py \
  --data_path /oldisk/home/jingyang/monoldp/datasets/nyu_data \
  --eval_split nyu \
  --post_process \
  --height 256 \
  --width 320
```

---

## 结果解读

评估完成后，脚本会输出：

1. **评估指标**：8 个标准深度估计指标
   - abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3

2. **Scaling ratios**：中位数缩放比例的统计信息
   - med: 中位数
   - std: 标准差

3. **Global depth 统计**（如果使用估计模式）
   - min, max, mean

4. **警告信息**：详细的公平性警告和说明

---

## 文件清单

- `evaluate_da3mono_nyu_depth.py`：主评估脚本
- `doc/evaluate_da3mono_modifications.md`：详细修改说明
- `doc/evaluate_da3mono_summary.md`：本文档（快速参考）

---

## 总结

### ✅ 保持一致的评估逻辑

1. `compute_errors` 函数完全相同
2. mask 生成逻辑完全相同
3. median scaling 逻辑完全相同
4. 深度裁剪逻辑完全相同
5. 所有评估指标计算完全相同

### ⚠️ 可能的差异点

1. **Global Depth 处理**：最可能影响公平性
2. **模型架构差异**：模型本身的特性
3. **输入预处理**：模型本身的特性
4. **深度到视差转换**：理论上一致，可能有精度差异

### 💡 建议

1. 默认使用估计模式（`--use_fixed_max_depth` 不设置）
2. 在结果中明确说明 Global Depth 处理方式
3. 可以尝试两种模式，对比结果差异
4. 在论文中明确说明这是不同架构的模型对比

