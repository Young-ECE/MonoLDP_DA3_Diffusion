#!/bin/bash
# 验证修复后的 diffusion decoder 能否正常训练

echo "=========================================="
echo "验证 Diffusion Decoder 修复"
echo "=========================================="
echo ""

# 激活环境
source ~/miniconda3/bin/activate monoldp_depth

# 进入工作目录
cd /home/jingyang/MonoLDP/DepthEstimation

echo "1. 检查修改的文件..."
echo "   ✓ depth_decoder_diffusion.py"
echo "   ✓ trainer.py"
echo "   ✓ options.py"
echo ""

echo "2. 运行简短的训练测试 (1个batch)..."
echo ""

# 运行一个很短的训练来验证
python train.py \
    --model_name diffusion_test \
    --use_diffusion \
    --scales 0 1 2 \
    --batch_size 2 \
    --num_epochs 1 \
    --log_frequency 1 \
    --no_eval \
    --debug_no_save 2>&1 | head -200

echo ""
echo "=========================================="
echo "如果没有看到通道不匹配的错误，说明修复成功！"
echo "=========================================="

