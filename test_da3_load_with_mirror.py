#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 DA3 模型加载（使用镜像）
"""

import sys
import os

# 设置镜像端点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DepthEstimation'))

import torch

print("=" * 60)
print("测试 DA3 模型加载（使用镜像）")
print("=" * 60)
print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")
print()

# 1. 测试导入
print("1. 测试导入:")
try:
    from networks.depth_anything_v3_wrapper import create_depth_anything_v3_teacher
    print("   ✓ Wrapper 导入成功")
except Exception as e:
    print(f"   ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. 测试创建 wrapper 并加载模型
print("\n2. 测试模型加载:")
print("   (这可能需要几分钟，取决于网络速度和模型大小)")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   使用设备: {device}")
    
    # 创建 wrapper（会尝试从镜像下载模型）
    wrapper = create_depth_anything_v3_teacher(
        model_name="DA3Mono-Large",
        device=device,
        scales=[0, 1, 2],
        input_size=(256, 320),
        model_path=None  # 不提供路径，测试从镜像下载
    )
    print("\n   ✓ Wrapper 创建成功")
    
    # 检查模型是否已加载
    if hasattr(wrapper, 'model') and wrapper.model is not None:
        print("   ✓ 模型已加载")
        print(f"   模型类型: {type(wrapper.model)}")
        
        # 检查模型参数数量
        total_params = sum(p.numel() for p in wrapper.model.parameters())
        print(f"   模型参数量: {total_params / 1e6:.2f}M")
    
except Exception as e:
    print(f"\n   ✗ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    print("\n   建议:")
    print("   1. 检查网络连接")
    print("   2. 确认镜像配置: export HF_ENDPOINT=https://hf-mirror.com")
    print("   3. 手动下载模型后使用 --depth_anything_v3_weights 指定路径")
    sys.exit(1)

# 3. 显示模型保存位置
print("\n3. 模型保存位置:")
try:
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache_path = os.path.join(cache_dir, "models--depth-anything--DA3Mono-Large")
    
    if os.path.exists(model_cache_path):
        print(f"   ✓ 模型已缓存到: {model_cache_path}")
        # 计算缓存大小
        import subprocess
        result = subprocess.run(['du', '-sh', model_cache_path], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            size = result.stdout.split()[0]
            print(f"   缓存大小: {size}")
    else:
        print(f"   缓存目录: {cache_dir}")
        print(f"   模型路径: {model_cache_path}")
        print("   (如果模型正在下载，路径会在下载完成后创建)")
except Exception as e:
    print(f"   ⚠ 无法确定缓存位置: {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)

