# 深度图生成脚本使用说明

## 概述

本脚本用于从NYUv2数据集中选择图片，使用DA3Mono-Large模型和训练好的学生模型生成深度图，并进行对比可视化。

## 文件说明

- `generate_depth_maps.py`: 主要的Python脚本，包含所有深度图生成逻辑
- `run_generate_depth.sh`: 便捷的shell脚本，自动从 `doc/notes.txt` 读取路径配置

## 使用方法

### 方法1: 使用便捷脚本（推荐）

```bash
# 确保 doc/notes.txt 中包含正确的路径配置
./run_generate_depth.sh
```

脚本会自动从 `doc/notes.txt` 读取：
- 学生模型路径
- DA3模型路径
- 数据集路径

默认处理索引为 0, 100, 200 的三张图片。

### 方法2: 直接使用Python脚本

```bash
python generate_depth_maps.py \
    --data_path /path/to/nyu_data \
    --da3_model_path /path/to/da3/model \
    --student_model_path /path/to/student/model \
    --output_dir ./depth_results \
    --image_indices 0 100 200 \
    --eval_split nyu
```

## 参数说明

- `--data_path`: NYU数据集路径（必需）
- `--da3_model_path`: DA3模型路径（可选，如果未指定则尝试从HuggingFace加载）
- `--student_model_path`: 学生模型权重文件夹路径（必需，如 `weights_19`）
- `--output_dir`: 输出目录（默认: `./depth_results`）
- `--image_indices`: 要处理的图片索引列表（默认: `0 100 200`）
- `--eval_split`: 数据集分割（默认: `nyu`）

## 输出结果

脚本会在 `output_dir` 目录下创建以下结构：

```
depth_results/
├── summary.json                    # 结果摘要（包含所有图片路径）
├── image_00000/                    # 第一张图片的结果
│   ├── original_image.png         # 原始RGB图片
│   ├── depth_da3.png              # DA3生成的深度图可视化
│   ├── depth_da3.npy             # DA3深度图（numpy格式）
│   ├── depth_student.png         # 学生模型生成的深度图可视化
│   └── depth_student.npy          # 学生模型深度图（numpy格式）
├── image_00100/                   # 第二张图片的结果
│   └── ...
└── image_00200/                   # 第三张图片的结果
    └── ...
```

### summary.json 格式

```json
[
  {
    "index": 0,
    "filename": "nyu2_test 00000",
    "image_path": "/path/to/nyu2_test/00000_colors.png",
    "output_dir": "./depth_results/image_00000",
    "original_image": "./depth_results/image_00000/original_image.png",
    "da3_depth": "./depth_results/image_00000/depth_da3.png",
    "da3_depth_npy": "./depth_results/image_00000/depth_da3.npy",
    "student_depth": "./depth_results/image_00000/depth_student.png",
    "student_depth_npy": "./depth_results/image_00000/depth_student.npy"
  },
  ...
]
```

## 功能特性

1. **自动配置加载**: 从学生模型的 `opt.json` 自动加载训练时的配置参数
2. **双模型对比**: 同时使用DA3和学生模型生成深度图，便于对比
3. **可视化输出**: 生成包含原始图片和深度图的对比可视化
4. **原始数据保存**: 同时保存numpy格式的深度图，便于后续分析
5. **路径记录**: 在 `summary.json` 中记录所有图片路径，方便后续使用

## 注意事项

1. **DA3模型**: 如果 `depth_anything_3` 包未安装，DA3深度图生成将被跳过
2. **GPU要求**: 脚本需要CUDA支持，模型会在GPU上运行
3. **内存要求**: 确保有足够的GPU内存加载两个模型
4. **路径格式**: 确保 `doc/notes.txt` 中的路径格式正确（每行一个路径）

## 示例输出

```
============================================================
Loading Models...
============================================================
-> Loading DA3 model from: /path/to/da3/model
-> DA3 model loaded successfully
-> Loading student model from: /path/to/student/model
-> Loading configuration from /path/to/models/opt.json
-> Student model loaded successfully

============================================================
Generating Depth Maps...
============================================================

[1/3] Processing image 0: nyu2_test 00000
  -> Image path: /path/to/nyu2_test/00000_colors.png
  -> Generating depth with DA3...
  -> Saved: ./depth_results/image_00000/depth_da3.png
  -> Saved: ./depth_results/image_00000/depth_da3.npy
  -> Generating depth with student model...
  -> Saved: ./depth_results/image_00000/depth_student.png
  -> Saved: ./depth_results/image_00000/depth_student.npy

...

============================================================
Summary
============================================================
-> Processed 3 images
-> Results saved to: ./depth_results
-> Summary saved to: ./depth_results/summary.json

Image paths:
  [00000] /path/to/nyu2_test/00000_colors.png
         Output: ./depth_results/image_00000
  [00100] /path/to/nyu2_test/00100_colors.png
         Output: ./depth_results/image_00100
  [00200] /path/to/nyu2_test/00200_colors.png
         Output: ./depth_results/image_00200
```

