# DA3Mono-Large 评估脚本使用指南

## 一、环境准备

### 1. 安装依赖

确保已安装以下依赖：

```bash
# Depth Anything 3
pip install depth-anything-3

# 或使用镜像
export HF_ENDPOINT=https://hf-mirror.com
pip install depth-anything-3
```

### 2. 检查数据集路径

确保 NYUv2 数据集路径正确：

```bash
# 数据集结构应该是：
# /path/to/nyu_data/
#   ├── nyu2_train/
#   │   ├── scene_001/
#   │   │   ├── 00001_colors.png
#   │   │   ├── 00001_depth.png
#   │   │   └── ...
#   └── ...
```

### 3. 检查 splits 文件

确保 `Hybrid_Mono/splits/nyu/test_files.txt` 存在：

```bash
ls Hybrid_Mono/splits/nyu/test_files.txt
```

---

## 二、基本使用

### 1. 从 HuggingFace 加载模型（推荐）

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

**说明：**
- `--data_path`：NYUv2 数据集路径
- `--eval_split`：评估数据集分割（nyu）
- `--height`、`--width`：输入图像尺寸（必须是 14 的倍数）
- `--batch_size`：批次大小
- `--num_workers`：数据加载器工作线程数
- `--scales`：评估尺度（通常使用 [0]）

---

### 2. 使用本地模型

如果模型已下载到本地：

```bash
python evaluate_da3mono_nyu_depth.py \
  --data_path /oldisk/home/jingyang/monoldp/datasets/nyu_data \
  --eval_split nyu \
  --da3_model_path /path/to/da3mono-large \
  --height 256 \
  --width 320 \
  --batch_size 6 \
  --num_workers 8 \
  --scales 0
```

**模型路径格式：**
- 可以是 HuggingFace 缓存目录：`~/.cache/huggingface/hub/models--depth-anything--DA3MONO-LARGE/snapshots/xxx`
- 可以是包含 `model.safetensors` 的目录

---

### 3. 使用固定 MAX_DEPTH

如果希望所有图像使用固定的 MAX_DEPTH=10.0：

```bash
python evaluate_da3mono_nyu_depth.py \
  --data_path /oldisk/home/jingyang/monoldp/datasets/nyu_data \
  --eval_split nyu \
  --use_fixed_max_depth \
  --height 256 \
  --width 320 \
  --batch_size 6 \
  --num_workers 8 \
  --scales 0
```

**说明：**
- 默认使用估计模式（从预测深度中估计 global_depth）
- 使用 `--use_fixed_max_depth` 启用固定模式

---

### 4. 启用后处理

启用左右翻转后处理（可能提升性能）：

```bash
python evaluate_da3mono_nyu_depth.py \
  --data_path /oldisk/home/jingyang/monoldp/datasets/nyu_data \
  --eval_split nyu \
  --post_process \
  --height 256 \
  --width 320 \
  --batch_size 6 \
  --num_workers 8 \
  --scales 0
```

**注意：**
- 后处理需要两次前向传播，评估时间会增加约一倍

---

### 5. 禁用中位数缩放

如果希望禁用中位数缩放：

```bash
python evaluate_da3mono_nyu_depth.py \
  --data_path /oldisk/home/jingyang/monoldp/datasets/nyu_data \
  --eval_split nyu \
  --disable_median_scaling \
  --height 256 \
  --width 320 \
  --batch_size 6 \
  --num_workers 8 \
  --scales 0
```

---

### 6. 保存预测结果

保存预测的视差图：

```bash
python evaluate_da3mono_nyu_depth.py \
  --data_path /oldisk/home/jingyang/monoldp/datasets/nyu_data \
  --eval_split nyu \
  --save_pred_disps \
  --load_weights_folder ./results \
  --height 256 \
  --width 320 \
  --batch_size 6 \
  --num_workers 8 \
  --scales 0
```

**输出：**
- 预测视差：`./results/disps_da3mono_nyu_split.npy`
- 评估结果：`./results/result_da3mono_nyu_split.txt`

---

## 三、完整参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--data_path` | NYUv2 数据集路径 | `/path/to/nyu_data` |
| `--eval_split` | 评估数据集分割 | `nyu` |

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--height` | 输入图像高度 | `256` |
| `--width` | 输入图像宽度 | `320` |
| `--batch_size` | 批次大小 | `6` |
| `--num_workers` | 数据加载器工作线程数 | `8` |
| `--scales` | 评估尺度 | `[0]` |
| `--da3_model_path` | DA3Mono-Large 本地模型路径 | `None`（从 HuggingFace 加载） |
| `--use_fixed_max_depth` | 使用固定 MAX_DEPTH | `False`（使用估计模式） |
| `--post_process` | 启用后处理 | `False` |
| `--disable_median_scaling` | 禁用中位数缩放 | `False` |
| `--pred_depth_scale_factor` | 预测深度缩放因子 | `1.0` |
| `--save_pred_disps` | 保存预测视差 | `False` |
| `--load_weights_folder` | 结果保存目录 | `"."` |
| `--no_eval` | 跳过评估（仅推理） | `False` |

---

## 四、输出说明

### 1. 控制台输出

评估过程中会输出：

```
-> Loading DA3Mono-Large model...
-> DA3Mono-Large model loaded successfully
-> Computing predictions with size 320x256
  Processed 10/100 batches
  ...
-> Total images: 654
-> Global depth stats: min=2.345, max=9.876, mean=6.123
-> Evaluating
   Mono evaluation - using median scaling
 Scaling ratios | med: 1.234 | std: 0.123

  abs_rel |  sq_rel |    rmse | rmse_log |   log10 |      a1 |      a2 |      a3
&  0.123  &  0.045  &  0.567  &  0.234  &  0.056  &  0.789  &  0.912  &  0.956  \\

-> Done!
```

### 2. 结果文件

评估结果保存在：`{load_weights_folder}/result_da3mono_{eval_split}_split.txt`

**文件内容：**
```
 Scaling ratios | med: 1.234 | std: 0.123

  abs_rel |  sq_rel |    rmse | rmse_log |   log10 |      a1 |      a2 |      a3
&  0.123  &  0.045  &  0.567  &  0.234  &  0.056  &  0.789  &  0.912  &  0.956  \\

-> Done!
```

### 3. 预测视差文件（如果启用 --save_pred_disps）

预测视差保存在：`{load_weights_folder}/disps_da3mono_{eval_split}_split.npy`

**格式：**
- NumPy 数组，形状：`(N, H, W)`
- N：图像数量
- H, W：图像高度和宽度

---

## 五、常见问题

### 1. 模型加载失败

**错误：**
```
depth_anything_3 package not found
```

**解决：**
```bash
pip install depth-anything-3
# 或使用镜像
export HF_ENDPOINT=https://hf-mirror.com
pip install depth-anything-3
```

---

### 2. HuggingFace 连接问题

**错误：**
```
Failed to load DA3Mono-Large model: Connection error
```

**解决：**
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型
huggingface-cli download depth-anything/DA3MONO-LARGE --local-dir ./models/DA3MONO-LARGE

# 然后使用本地路径
python evaluate_da3mono_nyu_depth.py \
  --da3_model_path ./models/DA3MONO-LARGE \
  ...
```

---

### 3. 输入尺寸错误

**错误：**
```
Input size must be multiple of 14
```

**解决：**
- 确保 `--height` 和 `--width` 是 14 的倍数
- 推荐尺寸：256x320, 224x320, 256x336 等

---

### 4. CUDA 内存不足

**错误：**
```
CUDA out of memory
```

**解决：**
- 减小 `--batch_size`（如从 6 改为 2 或 1）
- 减小输入尺寸（如从 256x320 改为 224x320）

---

### 5. 数据集路径错误

**错误：**
```
Cannot find dataset at /path/to/nyu_data
```

**解决：**
- 检查数据集路径是否正确
- 确保数据集结构符合要求
- 检查 `splits/nyu/test_files.txt` 是否存在

---

## 六、性能优化建议

### 1. 批次大小

- GPU 内存充足：`--batch_size 6` 或更大
- GPU 内存有限：`--batch_size 2` 或 `1`

### 2. 工作线程数

- CPU 核心数多：`--num_workers 8` 或更多
- CPU 核心数少：`--num_workers 4` 或更少

### 3. 输入尺寸

- 更大尺寸可能提升精度，但会增加计算时间
- 推荐：256x320（平衡精度和速度）

### 4. 后处理

- 后处理可能提升性能，但会增加约一倍的计算时间
- 建议：先不使用后处理评估，如果结果不理想再启用

---

## 七、与 MonoLDP 评估脚本对比

### 相同点

1. ✅ 评估指标计算完全相同
2. ✅ mask 生成逻辑完全相同
3. ✅ median scaling 逻辑完全相同
4. ✅ 深度裁剪逻辑完全相同

### 不同点

1. ⚠️ 模型架构不同（DA3Mono-Large vs MonoLDP）
2. ⚠️ Global Depth 处理方式不同
3. ⚠️ 输入预处理不同（ImageNet normalization）
4. ⚠️ 输出格式不同（深度 vs 视差，需要转换）

### 评估公平性

- ✅ 评估逻辑完全一致
- ⚠️ 模型特性差异（这是模型本身的特性，不属于评估不公平）
- ⚠️ Global Depth 处理可能影响结果（建议使用估计模式）

---

## 八、结果解读

### 评估指标说明

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

### Scaling Ratios 说明

- `med`：中位数缩放比例的中位数
- `std`：缩放比例的标准差
- 如果 `std` 很大，说明不同图像的缩放比例差异很大

### Global Depth 统计说明

- `min`：所有图像中 global_depth 的最小值
- `max`：所有图像中 global_depth 的最大值
- `mean`：所有图像中 global_depth 的平均值

---

## 九、示例脚本

### 完整评估脚本

创建 `evaluate_da3mono.sh`：

```bash
#!/bin/bash

# DA3Mono-Large NYUv2 评估脚本

DATA_PATH="/oldisk/home/jingyang/monoldp/datasets/nyu_data"
RESULT_DIR="./results/da3mono_eval"

mkdir -p ${RESULT_DIR}

python evaluate_da3mono_nyu_depth.py \
  --data_path ${DATA_PATH} \
  --eval_split nyu \
  --height 256 \
  --width 320 \
  --batch_size 6 \
  --num_workers 8 \
  --scales 0 \
  --load_weights_folder ${RESULT_DIR} \
  --save_pred_disps

echo "Evaluation completed! Results saved to ${RESULT_DIR}"
```

运行：
```bash
chmod +x evaluate_da3mono.sh
./evaluate_da3mono.sh
```

---

## 十、注意事项

1. **输入尺寸**：必须是 14 的倍数（DA3Mono-Large 的 patch size）
2. **Global Depth**：默认使用估计模式，更接近 MonoLDP 的行为
3. **后处理**：会增加计算时间，但可能提升性能
4. **模型路径**：如果使用本地模型，确保路径正确
5. **数据集路径**：确保数据集结构正确

---

## 十一、故障排除

如果遇到问题，请检查：

1. ✅ Depth Anything 3 是否已安装
2. ✅ 数据集路径是否正确
3. ✅ splits 文件是否存在
4. ✅ 输入尺寸是否是 14 的倍数
5. ✅ GPU 内存是否充足
6. ✅ CUDA 是否可用

---

## 十二、联系与支持

如有问题，请参考：
- 详细修改说明：`doc/evaluate_da3mono_modifications.md`
- 快速参考：`doc/evaluate_da3mono_summary.md`
- 原始评估脚本：`MonoLDP/DepthEstimation/evaluate_nyu_depth.py`

