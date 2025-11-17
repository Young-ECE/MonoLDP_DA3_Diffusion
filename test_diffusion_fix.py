#!/usr/bin/env python3
"""测试修复后的DepthDecoderDiffusion"""

import torch
import numpy as np
import sys
sys.path.append('/home/jingyang/MonoLDP/DepthEstimation')

from networks.depth_decoder_diffusion import DepthDecoderDiffusion
from networks.resnet_encoder import ResnetEncoder

print("=" * 80)
print("测试修复后的 DepthDecoderDiffusion")
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

# 4. 测试修复后的decoder (scales=[0, 1, 2])
print(f"\n3. 测试修复后的配置: scales=[0, 1, 2], num_output_channels=1")
try:
    decoder = DepthDecoderDiffusion(
        encoder.num_ch_enc, 
        scales=[0, 1, 2],
        num_output_channels=1,
        use_skips=True
    )
    print(f"   ✓ Decoder初始化成功")
    print(f"   - num_ch_enc: {decoder.num_ch_enc}")
    print(f"   - num_ch_dec: {decoder.num_ch_dec}")
    print(f"   - scales: {decoder.scales}")
    
    # 准备测试forward
    gt_for_diffusion = {
        ("disp_diffusion", 0): torch.randn(batch_size, 1, height, width),
        ("disp_diffusion", 1): torch.randn(batch_size, 1, height // 2, width // 2),
        ("disp_diffusion", 2): torch.randn(batch_size, 1, height // 4, width // 4),
    }
    
    print(f"\n4. 测试forward...")
    try:
        with torch.no_grad():
            outputs = decoder(features, gt_for_diffusion)
        print(f"   ✓ Forward成功!")
        print(f"\n5. 输出结果:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"   - {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"   - {key}: dict with {len(value)} items")
        
        # 验证输出形状
        print(f"\n6. 验证输出形状:")
        for scale in [0, 1, 2]:
            if ("disp", scale) in outputs:
                expected_h = height // (2 ** scale)
                expected_w = width // (2 ** scale)
                actual_shape = outputs[("disp", scale)].shape
                expected_shape = (batch_size, 1, expected_h, expected_w)
                match = actual_shape == torch.Size(expected_shape)
                status = "✓" if match else "✗"
                print(f"   {status} scale {scale}: {actual_shape} (期望 {expected_shape})")
        
        # 验证DDIM损失
        print(f"\n7. 验证DDIM损失:")
        for scale in [0, 1, 2]:
            if ("ddim_loss", scale) in outputs:
                loss_value = outputs[("ddim_loss", scale)].item()
                print(f"   ✓ scale {scale}: {loss_value:.6f}")
        
        print(f"\n" + "=" * 80)
        print("✓ 所有测试通过!")
        print("=" * 80)
        
    except Exception as e:
        print(f"   ✗ Forward失败: {e}")
        import traceback
        print(f"\n   错误详情:")
        traceback.print_exc()
        print(f"\n" + "=" * 80)
        print("✗ 测试失败")
        print("=" * 80)

except Exception as e:
    print(f"   ✗ Decoder初始化失败: {e}")
    import traceback
    traceback.print_exc()
    print(f"\n" + "=" * 80)
    print("✗ 测试失败")
    print("=" * 80)

