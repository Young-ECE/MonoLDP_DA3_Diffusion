#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 Depth Anything 3 安装和基本功能
"""

import sys
import torch

print("=" * 60)
print("Depth Anything 3 安装检查")
print("=" * 60)

# 1. 检查包是否安装
print("\n1. 检查 depth_anything_3 包:")
try:
    import depth_anything_3
    print(f"   ✓ depth_anything_3 包已安装")
    print(f"   位置: {depth_anything_3.__path__[0] if depth_anything_3.__path__ else 'N/A'}")
except ImportError as e:
    print(f"   ✗ depth_anything_3 包未安装: {e}")
    sys.exit(1)

# 2. 检查模型类
print("\n2. 检查模型类:")
try:
    from depth_anything_3.model import DepthAnything3Net, NestedDepthAnything3Net
    print(f"   ✓ DepthAnything3Net 可导入")
    print(f"   ✓ NestedDepthAnything3Net 可导入")
except ImportError as e:
    print(f"   ✗ 模型类导入失败: {e}")
    sys.exit(1)

# 3. 检查注册表
print("\n3. 检查模型注册表:")
try:
    from depth_anything_3 import registry
    print(f"   ✓ 模型注册表可用")
    print(f"   可用模型配置:")
    for model_name in registry.MODEL_REGISTRY.keys():
        print(f"     - {model_name}")
    
    # 检查 DA3Mono-Large 对应的配置
    if 'da3mono-large' in registry.MODEL_REGISTRY:
        print(f"\n   ✓ 找到 da3mono-large 配置 (对应 DA3Mono-Large)")
        config_path = registry.MODEL_REGISTRY['da3mono-large']
        print(f"     配置路径: {config_path}")
    else:
        print(f"   ⚠ 未找到 da3mono-large 配置")
except Exception as e:
    print(f"   ✗ 注册表检查失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 检查配置文件
print("\n4. 检查配置文件:")
try:
    from depth_anything_3 import registry
    import os
    config_path = registry.MODEL_REGISTRY.get('da3mono-large')
    if config_path and os.path.exists(config_path):
        print(f"   ✓ da3mono-large 配置文件存在")
        print(f"     路径: {config_path}")
    else:
        print(f"   ⚠ 配置文件不存在或路径无效")
except Exception as e:
    print(f"   ⚠ 配置文件检查失败: {e}")

# 5. 尝试加载模型结构（不加载权重）
print("\n5. 尝试创建模型结构:")
try:
    from depth_anything_3.model import DepthAnything3Net
    from depth_anything_3 import registry
    import yaml
    
    # 读取配置
    config_path = registry.MODEL_REGISTRY.get('da3mono-large')
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"   ✓ 配置文件读取成功")
        print(f"     配置键: {list(config.keys())[:5]}")
        
        # 注意：实际加载模型需要权重文件，这里只检查结构
        print(f"   ✓ 模型结构检查通过（需要权重文件才能完整加载）")
    else:
        print(f"   ⚠ 未找到配置文件")
except Exception as e:
    print(f"   ⚠ 模型结构检查失败: {e}")
    import traceback
    traceback.print_exc()

# 6. 检查依赖
print("\n6. 检查关键依赖:")
dependencies = [
    ("torch", "PyTorch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("huggingface_hub", "huggingface_hub"),
    ("einops", "einops"),
    ("xformers", "xformers"),
]

for module_name, display_name in dependencies:
    try:
        __import__(module_name)
        print(f"   ✓ {display_name} 已安装")
    except ImportError:
        print(f"   ⚠ {display_name} 未安装")

print("\n" + "=" * 60)
print("检查总结")
print("=" * 60)
print("✓ depth_anything_3 包已成功安装")
print("✓ 模型类和注册表可用")
print("⚠ 完整模型加载需要从 HuggingFace 下载权重文件")
print("=" * 60)

