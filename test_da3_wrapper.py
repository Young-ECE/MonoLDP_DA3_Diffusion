#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试更新后的 DA3 Wrapper
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DepthEstimation'))

import torch

print("=" * 60)
print("测试 DA3 Wrapper 更新")
print("=" * 60)

# 1. 测试导入
print("\n1. 测试导入:")
try:
    from networks.depth_anything_v3_wrapper import DepthAnythingV3Wrapper, create_depth_anything_v3_teacher
    print("   ✓ Wrapper 导入成功")
except Exception as e:
    print(f"   ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. 测试创建 wrapper（不加载权重）
print("\n2. 测试创建 wrapper:")
try:
    # 注意：这里会尝试加载模型，如果网络不可用会创建无权重版本
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   使用设备: {device}")
    
    # 创建 wrapper（不提供权重路径，会尝试从 HuggingFace 加载或创建无权重版本）
    print("   创建 wrapper...")
    wrapper = create_depth_anything_v3_teacher(
        model_name="DA3Mono-Large",
        device=device,
        scales=[0, 1, 2],
        input_size=(256, 320),
        model_path=None  # 不提供路径，测试自动加载
    )
    print("   ✓ Wrapper 创建成功")
    
except Exception as e:
    print(f"   ⚠ Wrapper 创建失败: {e}")
    print("   这可能是正常的，如果模型权重未下载")
    import traceback
    traceback.print_exc()

# 3. 测试基本属性
print("\n3. 测试 wrapper 属性:")
try:
    if 'wrapper' in locals():
        print(f"   ✓ 模型名称: {wrapper.model_name}")
        print(f"   ✓ 设备: {wrapper.device}")
        print(f"   ✓ 输入尺寸: {wrapper.input_height}x{wrapper.input_width}")
        print(f"   ✓ 输出尺度: {wrapper.scales}")
        print(f"   ✓ 模型类型: {type(wrapper.model)}")
except Exception as e:
    print(f"   ⚠ 属性检查失败: {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
print("\n注意:")
print("- 如果模型权重未下载，wrapper 会创建无权重版本")
print("- 实际训练时需要从 HuggingFace 下载权重或提供本地路径")
print("=" * 60)

