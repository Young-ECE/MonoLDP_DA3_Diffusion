# DA3Mono-Large 评估脚本修改说明

本文档详细说明 `evaluate_da3mono_nyu_depth.py` 相对于 `MonoLDP/DepthEstimation/evaluate_nyu_depth.py` 的修改点，以及可能影响评估公平性的因素。

## 一、代码修改点

### 修改点 1：模型加载部分（第 78-120 行）

**原 MonoLDP 代码：**
```python
encoder = networks.ResnetEncoder(opt.num_layers, False)
depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, [0])
scalenet = networks.ScaleNetwork(encoder.num_ch_enc)
regression_heads = nn.ModuleList([...])

encoder.load_state_dict(...)
depth_decoder.load_state_dict(...)
scalenet.load_state_dict(...)
regression_heads.load_state_dict(...)
```

**修改后代码：**
```python
da3_model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")
da3_model = da3_model.cuda()
da3_model.eval()
```

**影响分析：**
- ✅ **评估逻辑不变**：只是替换了模型，评估流程保持一致
- ⚠️ **模型架构差异**：DA3Mono-Large 是端到端模型，没有 scalenet 和 regression_heads
- ⚠️ **可能影响**：模型容量和表达能力不同，可能影响性能上限

---

### 修改点 2：推理部分 - 输入预处理（第 152-175 行）

**原 MonoLDP 代码：**
```python
input_color = data[("color", 0, 0)].cuda()  # 已经是 [0, 1] 范围
# 直接使用，可能已经经过 MonoLDP 的预处理
```

**修改后代码：**
```python
input_color = data[("color", 0, 0)].cuda()  # [0, 1] 范围

# DA3Mono-Large 需要 ImageNet normalization
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
input_normalized = (input_color - mean) / std

# DA3Mono-Large 要求输入尺寸是 14 的倍数
if H % patch_size != 0 or W % patch_size != 0:
    new_H = math.ceil(H / patch_size) * patch_size
    new_W = math.ceil(W / patch_size) * patch_size
    input_normalized = F.interpolate(input_normalized, size=(new_H, new_W), ...)
```

**影响分析：**
- ✅ **评估逻辑不变**：只是输入预处理方式不同，这是模型本身的特性
- ⚠️ **归一化差异**：DA3Mono-Large 使用 ImageNet normalization，MonoLDP 可能不同
- ⚠️ **尺寸调整**：如果输入尺寸不是 14 的倍数，会进行 resize，可能引入插值误差
- ⚠️ **可能影响**：预处理差异可能影响模型性能，但这是模型本身的特性，不属于评估不公平

---

### 修改点 3：推理部分 - 模型前向传播（第 177-220 行）

**原 MonoLDP 代码：**
```python
features = encoder(input_color)
depth_factors = scalenet(features)
scale_predictions = [head(factor) for head, factor in zip(regression_heads, depth_factors)]
max_depth = torch.mean(torch.stack(scale_predictions), dim=0)
output = depth_decoder(features, norm_pix_coords)

# 输出是视差 (disp)
disp_i = output[("disp", 0)][i:i+1]
pred_disp_i, _ = disp_to_depth(disp_i, opt.min_depth, max_depth_i)
```

**修改后代码：**
```python
input_da3 = input_normalized.unsqueeze(1)  # (B, 1, 3, H, W)

output = da3_model.forward(
    input_da3,
    extrinsics=None,
    intrinsics=None,
    export_feat_layers=[],
    infer_gs=False
)

# 输出是深度 (depth)
pred_depth_raw = output.get('depth', ...)
pred_depth_i = pred_depth_raw[i].cpu().numpy()

# 转换为视差以适配评估流程
pred_disp_i = depth_to_disparity(pred_depth_i, opt.min_depth, max_depth_i)
```

**影响分析：**
- ✅ **评估逻辑不变**：最终都转换为视差格式，评估流程一致
- ⚠️ **输出格式差异**：DA3Mono-Large 输出深度，需要转换为视差
- ⚠️ **转换精度**：深度→视差→深度的转换理论上应该一致，但可能有数值精度差异
- ✅ **转换公式**：`disp = 1 / depth`，评估时 `pred_depth = 1 / pred_disp`，理论上一致

---

### 修改点 4：Global Depth 处理（第 222-235 行）

**原 MonoLDP 代码：**
```python
depth_factors = scalenet(features)
scale_predictions = [head(factor) for head, factor in zip(regression_heads, depth_factors)]
max_depth = torch.mean(torch.stack(scale_predictions), dim=0)
global_depth.append(max_depth.cpu().numpy())
```

**修改后代码：**
```python
if use_fixed_max_depth:
    max_depth_i = MAX_DEPTH  # 固定值 10.0
else:
    max_depth_i = estimate_global_depth_from_prediction(pred_depth_i)  # 从预测中估计
batch_global_depths.append(max_depth_i)
```

**影响分析：**
- ⚠️ **关键差异**：这是**最可能影响评估公平性的点**
- ⚠️ **固定 MAX_DEPTH 模式**：所有图像使用相同的 MAX_DEPTH=10.0
  - MonoLDP 使用每张图像预测的 global_depth
  - 可能对某些图像不公平（过小或过大的场景）
- ⚠️ **估计模式**：从预测深度中估计（95th percentile）
  - 估计方法不同（MonoLDP 使用神经网络预测）
  - 可能影响 mask 生成和深度裁剪
- ✅ **建议**：使用估计模式（默认），更接近 MonoLDP 的行为

---

### 修改点 5：后处理部分（第 237-280 行）

**原 MonoLDP 代码：**
```python
if opt.post_process:
    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
    # 在 batch 维度上处理，一次前向传播
    # ...
    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
```

**修改后代码：**
```python
if opt.post_process:
    # 需要两次前向传播（原始 + 翻转）
    input_flipped = torch.flip(input_normalized, [3])
    output_flipped = da3_model.forward(input_flipped_da3, ...)
    # ...
    processed_disp = batch_post_process_disparity(l_disp[i:i+1], r_disp_flipped[i:i+1])
```

**影响分析：**
- ✅ **后处理逻辑不变**：使用相同的 `batch_post_process_disparity` 函数
- ⚠️ **实现方式差异**：DA3Mono-Large 需要两次前向传播，MonoLDP 在 batch 维度处理
- ✅ **结果一致性**：理论上应该产生相同的结果
- ⚠️ **性能差异**：DA3Mono-Large 可能更慢，但不影响评估公平性

---

### 修改点 6：评估部分（第 282-350 行）

**评估部分代码完全一致**，包括：
- ✅ `compute_errors` 函数完全相同
- ✅ mask 生成逻辑完全相同（MIN_DEPTH, global_depth, crop_mask）
- ✅ median scaling 逻辑完全相同
- ✅ 深度裁剪逻辑完全相同
- ✅ 所有评估指标计算完全相同

---

## 二、可能影响评估公平性的隐藏因素

### 1. 模型架构差异 ⚠️

**问题：**
- MonoLDP：encoder + decoder + scalenet + regression_heads（多组件）
- DA3Mono-Large：端到端深度估计模型（单模型）

**影响：**
- 模型容量和表达能力不同
- 训练数据和训练方式不同
- **这不属于评估不公平，而是模型本身的差异**

**建议：**
- 这是模型特性，无法避免
- 在论文中应明确说明这是不同架构的模型对比

---

### 2. Global Depth 处理方式 ⚠️⚠️

**问题：**
- MonoLDP：使用 scalenet + regression_heads 预测每张图像的 global_depth
- DA3Mono-Large（固定模式）：所有图像使用 MAX_DEPTH=10.0
- DA3Mono-Large（估计模式）：从预测深度中估计（95th percentile）

**影响：**
- 影响 mask 生成：`mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < global_depth[i])`
- 影响深度裁剪：`mask_pred_depth[mask_pred_depth > global_depth[i]] = global_depth[i]`
- **这是最可能影响评估公平性的因素**

**建议：**
- 默认使用估计模式（`--use_fixed_max_depth` 不设置）
- 在结果中报告 global_depth 的统计信息
- 可以尝试两种模式，对比结果差异

---

### 3. 输入预处理差异 ⚠️

**问题：**
- DA3Mono-Large 使用 ImageNet normalization
- MonoLDP 可能使用不同的归一化方式
- DA3Mono-Large 要求输入尺寸是 14 的倍数

**影响：**
- 归一化差异可能影响模型性能
- 尺寸调整可能引入插值误差

**建议：**
- 这是模型本身的特性，不属于评估不公平
- 确保输入尺寸设置正确（256x320 是 14 的倍数）

---

### 4. 深度到视差的转换 ⚠️

**问题：**
- DA3Mono-Large 输出深度，需要转换为视差
- 转换：`disp = 1 / depth`
- 评估时：`pred_depth = 1 / pred_disp`

**影响：**
- 理论上应该一致，但可能有数值精度差异
- 深度范围可能不同

**建议：**
- 检查转换后的视差范围是否合理
- 验证 `1 / (1 / depth) ≈ depth` 的精度

---

### 5. 后处理实现差异 ⚠️

**问题：**
- DA3Mono-Large 需要两次前向传播
- MonoLDP 在 batch 维度处理

**影响：**
- 理论上结果应该一致
- 但可能有微小的数值差异

**建议：**
- 检查后处理后的结果是否合理
- 可以对比启用/禁用后处理的结果

---

### 6. 输入尺寸和插值 ⚠️

**问题：**
- DA3Mono-Large 要求输入尺寸是 14 的倍数
- 如果输入尺寸不满足，会进行 resize

**影响：**
- 可能引入插值误差
- 可能影响模型性能

**建议：**
- 使用 14 的倍数的输入尺寸（如 256x320）
- 检查 resize 前后的差异

---

## 三、评估公平性检查清单

在运行评估前，请检查以下项目：

- [ ] **输入尺寸**：确保是 14 的倍数（推荐 256x320）
- [ ] **Global Depth 模式**：选择估计模式（默认）或固定模式
- [ ] **后处理**：根据需求启用/禁用 `--post_process`
- [ ] **Median Scaling**：确保与 MonoLDP 使用相同的设置
- [ ] **评估指标**：确认所有 8 个指标都正确计算
- [ ] **Mask 生成**：确认使用相同的 MIN_DEPTH、global_depth、crop_mask
- [ ] **深度裁剪**：确认使用相同的裁剪逻辑

---

## 四、使用建议

### 1. 基本使用

```bash
python evaluate_da3mono_nyu_depth.py \
  --data_path /path/to/nyu_data \
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
  --data_path /path/to/nyu_data \
  --eval_split nyu \
  --da3_model_path /path/to/da3mono-large \
  --height 256 \
  --width 320
```

### 3. 使用固定 MAX_DEPTH

```bash
python evaluate_da3mono_nyu_depth.py \
  --data_path /path/to/nyu_data \
  --eval_split nyu \
  --use_fixed_max_depth \
  --height 256 \
  --width 320
```

### 4. 启用后处理

```bash
python evaluate_da3mono_nyu_depth.py \
  --data_path /path/to/nyu_data \
  --eval_split nyu \
  --post_process \
  --height 256 \
  --width 320
```

---

## 五、结果解读

评估完成后，脚本会输出：

1. **评估指标**：8 个标准深度估计指标
2. **Scaling ratios**：中位数缩放比例的统计信息
3. **Global depth 统计**：每张图像的 global_depth 统计（如果使用估计模式）
4. **警告信息**：详细的公平性警告和说明

**重要提示：**
- 对比结果时，需要考虑模型架构差异
- Global Depth 处理方式可能影响结果
- 建议同时报告两种 Global Depth 模式的结果

---

## 六、总结

### 保持一致的评估逻辑 ✅

1. `compute_errors` 函数完全相同
2. mask 生成逻辑完全相同
3. median scaling 逻辑完全相同
4. 深度裁剪逻辑完全相同
5. 所有评估指标计算完全相同

### 可能的差异点 ⚠️

1. **Global Depth 处理**：最可能影响公平性
2. **模型架构差异**：模型本身的特性
3. **输入预处理**：模型本身的特性
4. **深度到视差转换**：理论上一致，可能有精度差异

### 建议

1. 默认使用估计模式（`--use_fixed_max_depth` 不设置）
2. 在结果中明确说明 Global Depth 处理方式
3. 可以尝试两种模式，对比结果差异
4. 在论文中明确说明这是不同架构的模型对比

