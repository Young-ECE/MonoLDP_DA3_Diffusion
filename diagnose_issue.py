#!/usr/bin/env python3
"""诊断脚本：分析DepthDecoderDiffusion的问题"""

import torch
import numpy as np
import sys
sys.path.append('/home/jingyang/MonoLDP/DepthEstimation')

from networks.depth_decoder_diffusion import DepthDecoderDiffusion
from networks.resnet_encoder import ResnetEncoder

print("=" * 80)
print("诊断 DepthDecoderDiffusion 问题")
print("=" * 80)

# 1. 创建encoder
encoder = ResnetEncoder(18, False)
print(f"\n1. Encoder输出通道: {encoder.num_ch_enc}")

# 2. 创建一个测试输入
batch_size = 2
height, width = 256, 320
test_input = torch.randn(batch_size, 3, height, width)

# 3. 获取encoder特征
with torch.no_grad():
    features = encoder(test_input)

print(f"\n2. Encoder特征形状:")
for i, f in enumerate(features):
    print(f"   特征层{i}: {f.shape}")

# 4. 测试不同scales配置的decoder
print(f"\n3. 测试不同scales配置:")

# 配置1：当前默认 scales=[0]
print(f"\n   配置1: scales=[0]")
try:
    decoder1 = DepthDecoderDiffusion(
        encoder.num_ch_enc, 
        scales=[0],
        num_output_channels=3,
        PixelCoorModu=True
    )
    print(f"   ✓ Decoder初始化成功")
    print(f"   - num_ch_enc: {decoder1.num_ch_enc}")
    print(f"   - num_ch_dec: {decoder1.num_ch_dec}")
    print(f"   - scales: {decoder1.scales}")
    
    # 查看构建的卷积层
    print(f"\n   构建的upconv层:")
    for key in decoder1.convs.keys():
        if 'upconv' in key:
            print(f"   - {key}: {decoder1.convs[key]}")
    
    # 准备测试forward
    norm_pix_coords = [torch.randn(batch_size, 3, height // (2**s), width // (2**s)) 
                       for s in [0]]
    gt_for_diffusion = {
        ("disp_diffusion", 0): torch.randn(batch_size, 3, height, width)
    }
    
    print(f"\n   测试forward...")
    try:
        with torch.no_grad():
            outputs = decoder1(features, norm_pix_coords, gt_for_diffusion)
        print(f"   ✓ Forward成功!")
    except Exception as e:
        print(f"   ✗ Forward失败: {e}")
        print(f"   错误类型: {type(e).__name__}")
        
except Exception as e:
    print(f"   ✗ Decoder初始化失败: {e}")

# 配置2：MonoDiffusion样式 scales=[0, 1, 2]
print(f"\n   配置2: scales=[0, 1, 2] (MonoDiffusion样式)")
try:
    decoder2 = DepthDecoderDiffusion(
        encoder.num_ch_enc, 
        scales=[0, 1, 2],
        num_output_channels=1,  # MonoDiffusion使用1通道
        PixelCoorModu=False
    )
    print(f"   ✓ Decoder初始化成功")
    print(f"   - num_ch_enc: {decoder2.num_ch_enc}")
    print(f"   - num_ch_dec: {decoder2.num_ch_dec}")
    print(f"   - scales: {decoder2.scales}")
    
    # 准备测试forward（不使用norm_pix_coords）
    gt_for_diffusion = {
        ("disp_diffusion", 0): torch.randn(batch_size, 1, height, width),
        ("disp_diffusion", 1): torch.randn(batch_size, 1, height // 2, width // 2),
        ("disp_diffusion", 2): torch.randn(batch_size, 1, height // 4, width // 4),
    }
    
    print(f"\n   测试forward...")
    try:
        with torch.no_grad():
            # 注意：MonoDiffusion不使用norm_pix_coords
            outputs = decoder2(features, None, gt_for_diffusion)
        print(f"   ✓ Forward成功!")
        print(f"   输出keys: {list(outputs.keys())}")
    except Exception as e:
        print(f"   ✗ Forward失败: {e}")
        import traceback
        print(f"   错误详情:")
        traceback.print_exc()

except Exception as e:
    print(f"   ✗ Decoder初始化失败: {e}")

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)

