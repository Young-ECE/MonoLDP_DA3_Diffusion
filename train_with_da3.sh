#!/bin/bash
# 使用 Depth Anything V3 (DA3Mono-Large) 作为教师模型的训练脚本
# 配置了镜像和教师模型输出保存

# 激活 conda 环境
echo "-> 激活 conda 环境 monoldp_depth..."
source ~/miniconda3/bin/activate monoldp_depth

# 设置 HuggingFace 镜像（确保使用镜像下载模型）
export HF_ENDPOINT=https://hf-mirror.com
echo "-> 设置 HuggingFace 镜像: $HF_ENDPOINT"

# 检查 DA3 包是否安装
echo "-> 检查 depth_anything_3 包..."
python -c "from depth_anything_3 import api; print('✓ depth_anything_3 包已安装')" || {
    echo "✗ depth_anything_3 包未安装。请运行以下命令安装:"
    echo "pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
    exit 1
}

# 定义数据路径（请根据你的实际情况修改）
DATA_PATH="/oldisk/home/jingyang/monoldp/datasets/nyu_data"

# 定义 DA3 模型权重路径（可选，如果未提供则会自动从镜像下载）
# 如果你手动下载了模型，请修改为你的本地路径，例如:
# DA3_WEIGHTS="/path/to/your/DA3Mono-Large"
DA3_WEIGHTS=""  # 留空则自动从镜像下载

# 定义日志目录
LOG_DIR="/oldisk/home/jingyang/monoldp/temp/da3_logs"
mkdir -p "$LOG_DIR"

# 训练命令
echo "-> 开始训练 (使用 Depth Anything V3 作为教师模型)..."
echo "   教师模型输出将保存到日志目录的 teacher_outputs/ 子目录"
echo ""

cd DepthEstimation

python train.py \
    --model_name da3_teacher_model \
    --use_diffusion \
    --use_depth_anything_v3 \
    --depth_anything_v3_model DA3Mono-Large \
    --data_path "$DATA_PATH" \
    --batch_size 6 \
    --num_epochs 15 \
    --learning_rate 5e-5 \
    --diffusion_l1_weight 1.0 \
    --diffusion_ddim_weight 1.0 \
    --debug_save_teacher \
    ${DA3_WEIGHTS:+--depth_anything_v3_weights "$DA3_WEIGHTS"} \
    --log_dir "$LOG_DIR"

echo ""
echo "训练完成！"
echo "教师模型输出保存在: $LOG_DIR/*/teacher_outputs/"

