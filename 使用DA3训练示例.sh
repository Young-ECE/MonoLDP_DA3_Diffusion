#!/bin/bash
# 使用 Depth Anything V3 (DA3Mono-Large) 作为教师模型的训练示例

# 安装 DA3 依赖（首次使用）
# pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git

# 训练命令
cd DepthEstimation
python train.py \
    --model_name da3_teacher_training \
    --use_diffusion \
    --use_depth_anything_v3 \
    --depth_anything_v3_model DA3Mono-Large \
    --data_path /oldisk/home/jingyang/monoldp/datasets/nyu_data \
    --batch_size 6 \
    --num_epochs 15 \
    --learning_rate 1e-4 \
    --diffusion_l1_weight 1.0 \
    --diffusion_ddim_weight 1.0

# 如果模型已下载到本地，使用 --depth_anything_v3_weights 指定路径：
# --depth_anything_v3_weights /path/to/DA3Mono-Large

