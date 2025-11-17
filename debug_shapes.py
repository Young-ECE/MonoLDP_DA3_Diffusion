#!/usr/bin/env python3
"""调试形状不匹配问题"""

import torch
import numpy as np
import sys
sys.path.append('/home/jingyang/MonoLDP/DepthEstimation')

from networks.depth_decoder_diffusion import DepthDecoderDiffusion
from networks.resnet_encoder import ResnetEncoder
from layers import upsample

print("=" * 80)
print("调试形状不匹配问题")
print("=" * 80)

# 1. 创建encoder
encoder = ResnetEncoder(18, False)

# 2. 创建测试输入
batch_size = 2
height, width = 256, 320
test_input = torch.randn(batch_size, 3, height, width)

# 3. 获取encoder特征
with torch.no_grad():
    features = encoder(test_input)

print(f"\nEncoder特征形状:")
for i, f in enumerate(features):
    print(f"  特征层{i}: {f.shape}")

# 4. 模拟decoder forward
print(f"\n模拟Decoder forward pass:")

x = features[-1]
print(f"  初始x (features[-1]): {x.shape}")

num_ch_enc = np.array(encoder.num_ch_enc)
num_ch_dec = (num_ch_enc / 2).astype('int')
print(f"  num_ch_dec: {num_ch_dec}")

for i in range(2, -1, -1):
    print(f"\n  === Scale {i} ===")
    print(f"    输入x: {x.shape}")
    
    # Simulate upconv 0
    num_ch_in = num_ch_enc[-1] if i == 2 else num_ch_dec[i + 1]
    num_ch_out = num_ch_dec[i]
    print(f"    upconv({i}, 0): {num_ch_in} -> {num_ch_out}")
    
    # 模拟卷积后的形状（通道数改变，空间维度不变）
    x_after_conv = torch.randn(x.shape[0], num_ch_out, x.shape[2], x.shape[3])
    print(f"    卷积后: {x_after_conv.shape}")
    
    # Upsample
    x_upsampled = upsample(x_after_conv)
    print(f"    上采样后: {x_upsampled.shape}")
    
    x_list = [x_upsampled]
    
    # Skip connection
    if i > 0:
        skip = features[i - 1]
        print(f"    Skip特征 (features[{i-1}]): {skip.shape}")
        x_list.append(skip)
    
    # Cat
    if len(x_list) > 1:
        print(f"    准备concatenate:")
        for idx, t in enumerate(x_list):
            print(f"      - tensor {idx}: {t.shape}")
        try:
            x = torch.cat(x_list, 1)
            print(f"    ✓ Concatenate成功: {x.shape}")
        except Exception as e:
            print(f"    ✗ Concatenate失败: {e}")
            break
    else:
        x = x_list[0]
        print(f"    无需concatenate: {x.shape}")
    
    # Simulate upconv 1
    num_ch_in = num_ch_dec[i]
    if i > 0:
        num_ch_in += num_ch_enc[i - 1]
    num_ch_out = num_ch_dec[i]
    print(f"    upconv({i}, 1): {num_ch_in} -> {num_ch_out}")
    
    # 模拟第二次卷积
    x = torch.randn(x.shape[0], num_ch_out, x.shape[2], x.shape[3])
    print(f"    第二次卷积后: {x.shape}")

print("\n" + "=" * 80)

