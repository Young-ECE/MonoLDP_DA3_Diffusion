# 使用 Depth Anything V3 训练说明

## 快速开始

### 1. 设置镜像（已自动配置）

训练脚本 `train_with_da3.sh` 已自动设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 2. 运行训练

```bash
cd /home/jingyang/MonoLDP
bash train_with_da3.sh
```

或者手动运行：

```bash
source ~/miniconda3/bin/activate monoldp_depth
export HF_ENDPOINT=https://hf-mirror.com

cd DepthEstimation
python train.py \
    --model_name da3_teacher_model \
    --use_diffusion \
    --use_depth_anything_v3 \
    --depth_anything_v3_model DA3Mono-Large \
    --data_path /oldisk/home/jingyang/monoldp/datasets/nyu_data \
    --batch_size 2 \
    --num_epochs 15 \
    --learning_rate 5e-5 \
    --diffusion_l1_weight 1.0 \
    --diffusion_ddim_weight 1.0 \
    --debug_save_teacher \
    --log_dir /oldisk/home/jingyang/monoldp/temp/da3_logs
```

## 教师模型输出保存

启用 `--debug_save_teacher` 后，教师模型的输出会保存到：

```
{log_dir}/{model_name}_{timestamp}/teacher_outputs/
├── disparity/          # 视差图（灰度图）
│   └── step{step:06d}_batch{idx}_scale{scale}_disp.png
├── depth/              # 深度图（彩色可视化 + 原始数据）
│   ├── step{step:06d}_batch{idx}_scale{scale}_depth.png  # 彩色深度图（jet colormap）
│   └── step{step:06d}_batch{idx}_scale{scale}_depth.npy  # 原始深度值（numpy格式）
└── step{step:06d}_batch{idx}_scale{scale}_stats.txt      # 统计信息（每100步保存一次）
```

### 输出文件说明

1. **视差图 (disparity)**：
   - 格式：PNG 灰度图
   - 内容：教师模型预测的视差值
   - 用途：观察视差分布

2. **深度图 (depth)**：
   - PNG 格式：彩色深度图（使用 jet colormap，红色=近，蓝色=远）
   - NPY 格式：原始深度数值（可用于数值分析）
   - 用途：观察深度预测的合理性

3. **统计信息 (stats)**：
   - 格式：文本文件
   - 内容：视差和深度的最小值、最大值、平均值
   - 保存频率：每100步保存一次

## 模型保存位置

- **DA3 教师模型缓存**：`~/.cache/depth_anything_3/models--depth-anything--DA3Mono-Large/`
- **训练日志和模型权重**：`{log_dir}/{model_name}_{timestamp}/`
- **教师模型输出**：`{log_dir}/{model_name}_{timestamp}/teacher_outputs/`

## 观察教师模型输出

训练开始后，可以实时查看教师模型的输出：

```bash
# 查看最新的深度图
ls -lt {log_dir}/*/teacher_outputs/depth/*.png | head -5

# 查看统计信息
cat {log_dir}/*/teacher_outputs/*stats.txt | tail -20
```

## 注意事项

1. **镜像配置**：确保 `HF_ENDPOINT` 环境变量已设置，否则会尝试从原始 HuggingFace 下载（可能较慢）

2. **模型大小**：DA3Mono-Large 模型约 1.3GB，首次下载需要一些时间

3. **输出频率**：教师模型输出会在每个训练步骤保存（前2个batch样本），注意磁盘空间

4. **性能影响**：保存教师输出会增加少量训练时间，但有助于验证教师模型质量

## 验证教师模型输出合理性

检查要点：
- ✅ 深度图是否与输入图像对应（前景物体深度小，背景深度大）
- ✅ 深度值范围是否合理（不应有异常值）
- ✅ 深度图边缘是否清晰（物体边界）
- ✅ 统计信息中的 min/max/mean 是否在合理范围内

如果发现异常，可以：
1. 检查输入图像预处理是否正确
2. 检查模型加载是否成功（查看训练日志）
3. 尝试不同的输入图像验证模型稳定性

