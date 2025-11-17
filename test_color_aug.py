#!/usr/bin/env python3
"""测试 ColorJitter 修复"""

from torchvision import transforms
from PIL import Image
import numpy as np

print("=" * 80)
print("测试 ColorJitter 修复")
print("=" * 80)

# 测试旧方法（会返回 tuple）
print("\n1. 测试 ColorJitter.get_params (旧方法):")
try:
    brightness = (0.8, 1.2)
    contrast = (0.8, 1.2)
    saturation = (0.8, 1.2)
    hue = (-0.1, 0.1)
    
    color_aug_old = transforms.ColorJitter.get_params(
        brightness, contrast, saturation, hue)
    print(f"   返回类型: {type(color_aug_old)}")
    print(f"   返回值: {color_aug_old}")
    print(f"   是否可调用: {callable(color_aug_old)}")
except Exception as e:
    print(f"   错误: {e}")

# 测试新方法（返回可调用对象）
print("\n2. 测试 ColorJitter() (新方法):")
try:
    color_aug_new = transforms.ColorJitter(
        brightness=brightness, 
        contrast=contrast, 
        saturation=saturation, 
        hue=hue)
    print(f"   返回类型: {type(color_aug_new)}")
    print(f"   是否可调用: {callable(color_aug_new)}")
    
    # 测试应用到图像
    dummy_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    result = color_aug_new(dummy_img)
    print(f"   ✓ 成功应用到图像")
    print(f"   输入类型: {type(dummy_img)}, 输出类型: {type(result)}")
    
except Exception as e:
    print(f"   错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✓ ColorJitter 修复验证完成")
print("=" * 80)

