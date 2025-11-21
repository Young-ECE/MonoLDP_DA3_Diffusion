# 数据集加载逻辑说明文档

## 📋 概述

本文档说明重构后的数据集加载逻辑。重构保持了原有的训练逻辑不变，只是使代码更清晰易懂。

## 🏗️ 代码结构

### 1. `mono_dataset.py` - 基础数据集类

**职责**：提供通用的数据加载和预处理功能

**主要方法**：
- `__init__()`: 初始化数据集配置
- `__getitem__()`: 主入口，加载单个训练样本
- `preprocess()`: 预处理数据（resize、augment、tensor转换）
- 抽象方法：`get_color()`, `get_depth()`, `get_plane()`, `get_line()` 等

### 2. `nyu_dataset.py` - NYU数据集实现

**职责**：实现NYU数据集特定的文件路径和数据格式处理

**主要方法**：
- `get_color()`: 加载RGB图像
- `get_depth()`: 加载深度图
- `get_plane()`: 加载平面分割图
- `get_line()`: 加载线段分割图

## 🔄 数据加载流程

### 完整流程（`__getitem__`方法）

```
1. 解析文件名
   └─> _parse_filename()
       └─> 提取: (folder, frame_index, side)

2. 加载帧图像
   └─> _load_frame_images()
       └─> 对每个frame_id调用 get_color()
           └─> NYUDataset.get_color()
               └─> 加载图像 -> 边缘裁剪 -> 翻转（如需要）

3. 计算相机内参
   └─> _compute_camera_intrinsics()
       └─> 为每个scale计算K矩阵和归一化像素坐标

4. 创建颜色增强
   └─> _create_color_augmentation()
       └─> 50%概率创建ColorJitter变换

5. 加载辅助数据
   └─> _load_auxiliary_data()
       ├─> get_depth() (如果可用)
       ├─> get_plane() (如果可用)
       └─> get_line() (如果可用)

6. 预处理所有数据
   └─> preprocess()
       ├─> Step 1: 多尺度resize
       ├─> Step 2: 颜色增强和tensor转换
       └─> Step 3: 几何正则化数据预处理

7. 清理原生分辨率数据
   └─> _cleanup_native_resolution()
       └─> 删除不需要的原生分辨率(-1)数据

8. 添加立体相机外参
   └─> _add_stereo_extrinsics() (如果使用立体对)
```

### 预处理流程（`preprocess`方法）

#### Step 1: 多尺度Resize

```python
# 对每个颜色图像（scale=-1）
for frame_id in frame_idxs:
    image_native = inputs[("color", frame_id, -1)]
    
    # 生成所有尺度的resized版本
    for scale in range(num_scales):
        inputs[("color", frame_id, scale)] = resize[scale](image_native)

# 对几何数据（plane/line）做同样的处理
for struct_type in ["plane", "line"]:
    struct_native = inputs[(struct_type, 0, -1)]
    for scale in range(num_scales):
        inputs[(struct_type, 0, scale)] = pl_resize[scale](struct_native)
```

**输出**：每个scale都有了resized的数据

#### Step 2: 颜色增强和Tensor转换

```python
# 对每个scale的颜色图像
for scale in range(num_scales):
    resized_image = inputs[("color", frame_id, scale)]
    
    # 转换为tensor
    inputs[("color", frame_id, scale)] = to_tensor(resized_image)
    
    # 应用颜色增强
    augmented = color_aug(resized_image)
    inputs[("color_aug", frame_id, scale)] = to_tensor(augmented)
```

**输出**：
- `("color", frame_id, scale)`: 原始颜色tensor
- `("color_aug", frame_id, scale)`: 增强后的颜色tensor

#### Step 3: 几何正则化数据预处理

对每个scale和每个结构类型（plane/line）：

```python
for scale in range(num_scales):
    for struct_type in ["plane", "line"]:
        # 1. 获取resized的分割图
        resized_map = inputs[(struct_type, 0, scale)]
        
        # 2. 转换为numpy数组
        struct_array = np.expand_dims(np.array(resized_map), 0)
        
        # 3. 存储float和long版本
        inputs[(struct_type + "_float", 0, scale)] = torch.from_numpy(struct_array).float()
        inputs[(struct_type, 0, scale)] = torch.from_numpy(struct_array).long()
        
        # 4. 生成keysets（用于几何正则化损失）
        if num_keysets > 0:
            keysets = _generate_keysets(struct_array, struct_type, scale)
            inputs[(struct_type + "_keysets", 0, scale)] = torch.from_numpy(keysets).long()
```

### Keyset生成逻辑

**目的**：从分割图中采样像素组，用于几何正则化损失（平面正则化、线段正则化）

**流程**：

```
1. 统计每个结构（平面/线段）的像素数
   └─> num_struct_pixels = 总像素数
   └─> num_struct = 结构数量

2. 按比例采样keysets
   for each structure j:
       pixels_j = 结构j的像素数
       num_keysets_j = ceil(total_keysets * pixels_j / total_pixels)
       
       # 从结构j中随机采样
       sample indices from structure j
       
       # 重组为keysets
       keyset_j = reshape(sampled_indices, (samples_per_keyset, num_keysets_j))

3. 合并所有结构的keysets
   └─> concatenate all keyset_j

4. 随机采样到目标数量
   └─> randomly sample to get exact num_keysets
```

**输出格式**：
- Plane: `(4, num_keysets)` - 4个点定义一个平面
- Line: `(3, num_keysets)` - 3个点定义一条线

## 📦 输出数据格式

`__getitem__`返回的字典包含：

### 颜色图像
- `("color", frame_id, scale)`: 原始颜色图像 (tensor, shape: [3, H, W])
- `("color_aug", frame_id, scale)`: 增强后的颜色图像 (tensor, shape: [3, H, W])
- `frame_id`: 例如 0, -1, 1 (时间步) 或 "s" (立体对)
- `scale`: 0, 1, 2, ... (尺度，0是最大尺度)

### 相机参数
- `("K", scale)`: 相机内参矩阵 (tensor, shape: [4, 4])
- `("norm_pix_coords", scale)`: 归一化像素坐标 (tensor, shape: [3, H, W])
- `"stereo_T"`: 立体相机外参矩阵 (tensor, shape: [4, 4]) - 如果使用立体对

### 深度真值
- `"depth_gt"`: 深度真值图 (tensor, shape: [1, H_full, W_full])

### 几何正则化数据（如果启用）
- `("plane", 0, scale)`: 平面分割图 (tensor, shape: [1, H, W], long)
- `("plane_float", 0, scale)`: 平面分割图 (tensor, shape: [1, H, W], float)
- `("plane_keysets", 0, scale)`: 平面keysets (tensor, shape: [4, num_keysets], long)
- `("line", 0, scale)`: 线段分割图 (tensor, shape: [1, H, W], long)
- `("line_float", 0, scale)`: 线段分割图 (tensor, shape: [1, H, W], float)
- `("line_keysets", 0, scale)`: 线段keysets (tensor, shape: [3, num_keysets], long)

## 🔧 NYU数据集特定逻辑

### 文件路径格式

**训练模式**：
- 图像: `scene_name/frame_index.jpg`
- 深度: `scene_name/frame_index.png`
- 平面: `scene_name/frame_index_seg.png`
- 线段: `scene_name/frame_index_line.png`

**测试模式**：
- 图像: `scene_name/XXXXX_colors.png` (5位零填充)
- 深度: `scene_name/XXXXX_depth.png`
- 平面: `scene_name/XXXXX_seg.png`
- 线段: `scene_name/XXXXX_line.png`

### 深度值缩放

- **训练**: 深度值 / 25.6 → 米
- **测试**: 深度值 / 1000.0 → 米

### 边缘裁剪

所有图像都裁剪掉16像素的边缘：
- 原始: 640×480
- 裁剪后: 608×448 (640-32, 480-32)

### 相机内参

内参矩阵归一化到裁剪后的图像尺寸：
- `fx = 518.86 / 608`
- `fy = 519.47 / 448`
- `cx = (325.58 - 16) / 608`
- `cy = (253.74 - 16) / 448`

## ✅ 重构后的改进

1. **代码组织**：
   - 将长方法拆分为小的辅助方法
   - 每个方法职责单一，易于理解

2. **逻辑清晰**：
   - 明确的数据加载流程
   - 详细的文档字符串和注释

3. **易于维护**：
   - 路径生成逻辑统一
   - 裁剪和翻转逻辑提取为方法
   - keyset生成逻辑独立

4. **保持兼容**：
   - 所有接口保持不变
   - 输出格式完全一致
   - 训练逻辑不受影响

## 🔍 关键方法说明

### `_generate_keysets()`
生成用于几何正则化损失的keysets。从分割图中按结构大小比例采样像素组。

### `_preprocess_geometric_data()`
预处理几何数据：resize分割图、转换为tensor、生成keysets。

### `_compute_camera_intrinsics()`
为每个尺度计算相机内参矩阵和归一化像素坐标。

### `_load_frame_images()`
加载所有请求的帧图像（时间帧或立体对）。

## 📝 注意事项

1. **原生分辨率数据**：在预处理后会删除（scale=-1），只在测试模式下保留

2. **Keyset采样**：每个尺度的keyset数量会除以2^scale（粗尺度采样更多）

3. **数据增强**：只在训练模式下应用，且颜色增强和翻转的概率都是50%

4. **立体对**：如果frame_idxs包含"s"，会加载对侧图像并计算立体外参

