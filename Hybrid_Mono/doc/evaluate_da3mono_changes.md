# DA3Mono-Large 评估脚本代码修改点详细说明

本文档详细列出 `evaluate_da3mono_nyu_depth.py` 相对于 `MonoLDP/DepthEstimation/evaluate_nyu_depth.py` 的所有代码修改点。

---

## 修改点 1：导入部分（第 31-37 行）

### 原代码（MonoLDP）
```python
from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
```

### 修改后代码
```python
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

# 新增：Depth Anything 3 导入
try:
    from depth_anything_3.api import DepthAnything3
    DA3_AVAILABLE = True
except ImportError:
    DA3_AVAILABLE = False
    warnings.warn("depth_anything_3 package not found...")
```

**说明：**
- 移除了 `from layers import disp_to_depth`（不再需要，因为 DA3Mono-Large 输出深度）
- 添加了 Depth Anything 3 的导入和错误处理

---

## 修改点 2：新增函数 - depth_to_disparity（第 82-100 行）

### 新增代码
```python
def depth_to_disparity(depth, min_depth=0.1, max_depth=10.0):
    """
    将深度转换为视差，用于适配 MonoLDP 的评估流程
    
    Args:
        depth: 深度图 (B, H, W) 或 (H, W)，单位：米
        min_depth: 最小深度值
        max_depth: 最大深度值
    
    Returns:
        disparity: 视差图，形状与 depth 相同
    """
    depth = np.clip(depth, min_depth, max_depth)
    disp = 1.0 / depth
    return disp
```

**说明：**
- DA3Mono-Large 输出深度，但评估流程期望视差
- 转换公式：`disp = 1 / depth`
- 评估时会通过 `pred_depth = 1 / pred_disp` 转换回来

---

## 修改点 3：新增函数 - estimate_global_depth_from_prediction（第 111-135 行）

### 新增代码
```python
def estimate_global_depth_from_prediction(pred_depth, percentile=95.0):
    """
    从预测深度中估计每张图像的 global_depth（最大深度）
    
    由于 DA3Mono-Large 没有 scalenet 和 regression_heads 来预测 global_depth，
    我们需要从预测深度中估计。使用百分位数方法，避免异常值影响。
    """
    valid_depth = pred_depth[pred_depth > 0.01]
    if len(valid_depth) == 0:
        return 10.0
    
    estimated_max_depth = np.percentile(valid_depth, percentile)
    estimated_max_depth = min(estimated_max_depth, 10.0)
    estimated_max_depth = max(estimated_max_depth, 0.5)
    
    return estimated_max_depth
```

**说明：**
- MonoLDP 使用 scalenet + regression_heads 预测 global_depth
- DA3Mono-Large 没有这些组件，需要从预测中估计
- 使用 95th percentile 避免异常值影响

---

## 修改点 4：模型加载部分（第 160-200 行）

### 原代码（MonoLDP，第 85-109 行）
```python
encoder = networks.ResnetEncoder(opt.num_layers, False)
depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, [0])
scalenet = networks.ScaleNetwork(encoder.num_ch_enc)
regression_heads = nn.ModuleList([
    networks.ProbabilisticScaleRegressionHead(in_channels=ch) 
    for ch in encoder.num_ch_enc
])

encoder_dict = torch.load(encoder_path)
encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

decoder_dict = torch.load(decoder_path)
depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in model_dict})

scalenet.load_state_dict(torch.load(scalenet_path))
regression_heads.load_state_dict(torch.load(regression_path))

encoder.cuda()
encoder.eval()
depth_decoder.cuda()
depth_decoder.eval()
scalenet.cuda()
scalenet.eval()
regression_heads.cuda()
regression_heads.eval()
```

### 修改后代码（第 160-200 行）
```python
# 加载 DA3Mono-Large 模型
print("-> Loading DA3Mono-Large model...")

if hasattr(opt, 'da3_model_path') and opt.da3_model_path:
    model_path = opt.da3_model_path
    print(f"-> Loading from local path: {model_path}")
else:
    model_path = None
    print("-> Loading from HuggingFace (will use cache if available)")

try:
    hf_endpoint = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
    os.environ['HF_ENDPOINT'] = hf_endpoint
    
    if model_path and os.path.exists(model_path):
        da3_model = DepthAnything3.from_pretrained(model_path)
    else:
        da3_model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")
    
    da3_model = da3_model.cuda()
    da3_model.eval()
    print("-> DA3Mono-Large model loaded successfully")
except Exception as e:
    raise RuntimeError(f"Failed to load DA3Mono-Large model: {e}")
```

**关键差异：**
1. 从 4 个组件（encoder, decoder, scalenet, regression_heads）变为 1 个模型（da3_model）
2. 从本地权重文件加载变为从 HuggingFace 或本地路径加载
3. 不再需要加载 scalenet 和 regression_heads

---

## 修改点 5：推理部分 - 输入预处理（第 240-270 行）

### 原代码（MonoLDP，第 122-123 行）
```python
input_color = data[("color", 0, 0)].cuda()
norm_pix_coords = [data[("norm_pix_coords", s)].cuda() for s in opt.scales]
```

### 修改后代码（第 240-270 行）
```python
input_color = data[("color", 0, 0)].cuda()  # (B, 3, H, W)，范围 [0, 1]

# DA3Mono-Large 需要 ImageNet normalization
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
input_normalized = (input_color - mean) / std

# DA3Mono-Large 需要输入尺寸是 14 的倍数（patch size）
B, C, H, W = input_normalized.shape
patch_size = 14
if H % patch_size != 0 or W % patch_size != 0:
    import math
    new_H = math.ceil(H / patch_size) * patch_size
    new_W = math.ceil(W / patch_size) * patch_size
    input_normalized = F.interpolate(
        input_normalized, 
        size=(new_H, new_W), 
        mode='bilinear', 
        align_corners=False
    )

# DA3Mono-Large forward 需要 (B, N, 3, H, W) 格式，N=1 表示单视图
input_da3 = input_normalized.unsqueeze(1)  # (B, 1, 3, H, W)
```

**关键差异：**
1. 添加了 ImageNet normalization
2. 确保输入尺寸是 14 的倍数
3. 调整输入格式为 (B, 1, 3, H, W)
4. 不再需要 norm_pix_coords（DA3Mono-Large 不需要）

---

## 修改点 6：推理部分 - 模型前向传播（第 272-320 行）

### 原代码（MonoLDP，第 139-170 行）
```python
features = encoder(input_color)
depth_factors = scalenet(features)
scale_predictions = [head(factor) for head, factor in zip(regression_heads, depth_factors)]
max_depth = torch.mean(torch.stack(scale_predictions), dim=0)
global_depth.append(max_depth.cpu().numpy())
output = depth_decoder(features, norm_pix_coords)

all_pred_disps = []

for i in range(max_depth.size(0)):
    disp_i = output[("disp", 0)][i:i+1]
    max_depth_i = max_depth[i].item()
    pred_disp_i, _ = disp_to_depth(disp_i, opt.min_depth, max_depth_i)
    
    if not isinstance(pred_disp_i, torch.Tensor):
        pred_disp_i = torch.tensor(pred_disp_i)
    pred_disp_i = pred_disp_i.view(1, *pred_disp_i.shape[1:])
    all_pred_disps.append(pred_disp_i)

pred_disp = torch.cat(all_pred_disps, dim=0)
pred_disp = pred_disp.cpu()[:, 0].numpy()
```

### 修改后代码（第 272-320 行）
```python
# 运行推理
try:
    output = da3_model.forward(
        input_da3,
        extrinsics=None,
        intrinsics=None,
        export_feat_layers=[],
        infer_gs=False
    )
    
    # 提取深度输出
    if isinstance(output, dict):
        pred_depth_raw = output.get('depth', None)
        if pred_depth_raw is None:
            for key in ['predicted_depth', 'pred', 'depth_map']:
                if key in output:
                    pred_depth_raw = output[key]
                    break
    else:
        pred_depth_raw = output
    
    if pred_depth_raw is None:
        raise RuntimeError("Could not extract depth from DA3 output")
    
    # 确保深度形状正确 (B, H, W) 或 (B, 1, H, W)
    if pred_depth_raw.ndim == 4:
        if pred_depth_raw.shape[1] == 1:
            pred_depth_raw = pred_depth_raw.squeeze(1)
        else:
            pred_depth_raw = pred_depth_raw[:, 0]
    elif pred_depth_raw.ndim == 5:
        pred_depth_raw = pred_depth_raw[:, 0, 0]
    
    # 如果尺寸被调整过，需要 resize 回原始尺寸
    if pred_depth_raw.shape[1:] != (H, W):
        pred_depth_raw = F.interpolate(
            pred_depth_raw.unsqueeze(1),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

except Exception as e:
    print(f"Error during DA3 inference: {e}")
    raise

# 处理后处理和 global_depth
batch_pred_disps = []
batch_global_depths = []

for i in range(B):
    pred_depth_i = pred_depth_raw[i].cpu().numpy()
    
    # 估计或使用固定的 global_depth
    if use_fixed_max_depth:
        max_depth_i = MAX_DEPTH
    else:
        max_depth_i = estimate_global_depth_from_prediction(pred_depth_i)
    batch_global_depths.append(max_depth_i)
    
    # 将深度转换为视差（用于适配评估流程）
    pred_disp_i = depth_to_disparity(pred_depth_i, opt.min_depth, max_depth_i)
    batch_pred_disps.append(pred_disp_i)
```

**关键差异：**
1. 从多步骤推理（encoder → scalenet → regression_heads → decoder）变为单步推理（da3_model.forward）
2. 输出从视差变为深度，需要转换为视差
3. global_depth 从神经网络预测变为从预测深度中估计或使用固定值
4. 需要处理不同的输出格式（dict 或 tensor）

---

## 修改点 7：后处理部分（第 322-380 行）

### 原代码（MonoLDP，第 133-178 行）
```python
if opt.post_process:
    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
    norm_pix_coords = [torch.cat((pc, torch.flip(pc, [3])), 0) for pc in norm_pix_coords]
    norm_pix_coords[0][norm_pix_coords[0].shape[0] // 2:, 0] *= -1

# ... 推理 ...

if opt.post_process:
    N = pred_disp.shape[0] // 2
    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
```

### 修改后代码（第 322-380 行）
```python
if opt.post_process:
    # 后处理需要左右翻转的预测
    # 对于 DA3Mono-Large，我们需要对输入进行翻转并重新推理
    input_flipped = torch.flip(input_normalized, [3])
    input_flipped_da3 = input_flipped.unsqueeze(1)
    
    # 重新推理翻转后的图像
    output_flipped = da3_model.forward(
        input_flipped_da3,
        extrinsics=None,
        intrinsics=None,
        export_feat_layers=[],
        infer_gs=False
    )
    
    # 提取翻转后的深度（类似上面的处理）
    # ...
    
    # 将翻转后的深度转换为视差
    batch_pred_disps_flipped = []
    for i in range(B):
        pred_depth_flipped_i = pred_depth_flipped[i].cpu().numpy()
        if use_fixed_max_depth:
            max_depth_i = MAX_DEPTH
        else:
            max_depth_i = estimate_global_depth_from_prediction(pred_depth_flipped_i)
        pred_disp_flipped_i = depth_to_disparity(pred_depth_flipped_i, opt.min_depth, max_depth_i)
        batch_pred_disps_flipped.append(pred_disp_flipped_i)
    
    # 应用后处理
    l_disp = np.stack([batch_pred_disps[i] for i in range(B)], axis=0)
    r_disp = np.stack([batch_pred_disps_flipped[i] for i in range(B)], axis=0)
    r_disp_flipped = np.flip(r_disp, axis=2)
    
    for i in range(B):
        processed_disp = batch_post_process_disparity(
            l_disp[i:i+1], 
            r_disp_flipped[i:i+1]
        )
        batch_pred_disps[i] = processed_disp[0]
```

**关键差异：**
1. MonoLDP 在 batch 维度上处理，一次前向传播
2. DA3Mono-Large 需要两次前向传播（原始 + 翻转）
3. 后处理函数 `batch_post_process_disparity` 完全相同

---

## 修改点 8：评估部分（第 400-480 行）

### 评估部分代码完全一致 ✅

**完全相同的部分：**
1. `compute_errors` 函数调用
2. mask 生成逻辑：`mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < global_depth[i])`
3. crop_mask 生成：`crop_mask[dataset.default_crop[2]:dataset.default_crop[3], ...]`
4. median scaling：`ratio = np.median(mask_gt_depth) / np.median(mask_pred_depth)`
5. 深度裁剪：`mask_pred_depth[mask_pred_depth > global_depth[i]] = global_depth[i]`
6. 所有评估指标计算

**唯一差异：**
- 使用 `global_depth[i]` 而不是固定的 `MAX_DEPTH`（与 MonoLDP 一致）

---

## 修改点 9：结果保存（第 390-400 行）

### 原代码（MonoLDP，第 216-220 行）
```python
if opt.save_pred_disps:
    output_path = os.path.join(
        opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
    np.save(output_path, pred_disps)
```

### 修改后代码（第 390-400 行）
```python
if opt.save_pred_disps:
    output_path = os.path.join(
        opt.load_weights_folder if hasattr(opt, 'load_weights_folder') and opt.load_weights_folder else ".",
        "disps_da3mono_{}_split.npy".format(opt.eval_split))
    print("-> Saving predicted disparities to ", output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, pred_disps)
```

**差异：**
- 文件名从 `disps_{}_split.npy` 改为 `disps_da3mono_{}_split.npy`（区分不同模型）
- 添加了目录创建逻辑

---

## 修改点 10：结果文件路径（第 450-460 行）

### 原代码（MonoLDP，第 266-267 行）
```python
result_path = os.path.join(opt.load_weights_folder, "result_{}_split.txt".format(opt.eval_split))
```

### 修改后代码（第 450-460 行）
```python
result_dir = opt.load_weights_folder if hasattr(opt, 'load_weights_folder') and opt.load_weights_folder else "."
os.makedirs(result_dir, exist_ok=True)
result_path = os.path.join(result_dir, "result_da3mono_{}_split.txt".format(opt.eval_split))
```

**差异：**
- 文件名从 `result_{}_split.txt` 改为 `result_da3mono_{}_split.txt`（区分不同模型）
- 添加了目录创建逻辑

---

## 修改点 11：警告信息输出（第 480-560 行）

### 新增代码
```python
def print_warnings(use_fixed_max_depth, global_depth):
    """打印评估过程中的警告信息"""
    # 详细的警告信息，包括：
    # 1. 模型架构差异
    # 2. Global Depth 处理
    # 3. 输入预处理
    # 4. 深度到视差转换
    # 5. 输入尺寸处理
    # 6. 后处理
    # 7. 保持一致的评估逻辑
```

**说明：**
- 新增函数，用于输出详细的公平性警告信息
- 帮助用户理解可能的评估差异

---

## 总结：代码修改统计

| 修改类型 | 数量 | 说明 |
|---------|------|------|
| 新增函数 | 3 | `depth_to_disparity`, `estimate_global_depth_from_prediction`, `print_warnings` |
| 修改函数 | 1 | `evaluate` 函数（核心逻辑修改） |
| 保持不变 | 2 | `compute_errors`, `batch_post_process_disparity` |
| 评估逻辑 | 0 | 完全一致 ✅ |

---

## 关键修改点总结

1. **模型加载**：从 4 组件变为 1 模型
2. **输入预处理**：添加 ImageNet normalization 和尺寸调整
3. **推理流程**：从多步骤变为单步，输出从视差变为深度
4. **Global Depth**：从神经网络预测变为估计或固定值
5. **深度转换**：新增深度到视差的转换函数
6. **后处理**：从 batch 处理变为两次前向传播
7. **评估逻辑**：完全一致 ✅

---

## 评估公平性保证

### ✅ 完全一致的部分

1. `compute_errors` 函数：完全相同
2. mask 生成逻辑：完全相同
3. median scaling：完全相同
4. 深度裁剪：完全相同
5. 所有评估指标计算：完全相同

### ⚠️ 可能影响公平性的因素

1. **Global Depth 处理**：最可能影响公平性
2. **模型架构差异**：模型本身的特性
3. **输入预处理**：模型本身的特性
4. **深度到视差转换**：理论上一致，可能有精度差异

---

## 使用建议

1. **默认使用估计模式**（不设置 `--use_fixed_max_depth`）
2. **在结果中明确说明** Global Depth 处理方式
3. **可以尝试两种模式**，对比结果差异
4. **在论文中明确说明**这是不同架构的模型对比

