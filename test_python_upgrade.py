#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 Python 升级和 PyTorch 安装是否成功
"""

import sys

print("=" * 60)
print("Python 升级测试")
print("=" * 60)

# 1. 检查 Python 版本
print("\n1. Python 版本检查:")
python_version = sys.version_info
print(f"   Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
if python_version.major == 3 and python_version.minor == 10:
    print("   ✓ Python 3.10 安装成功")
else:
    print(f"   ✗ Python 版本不符合预期 (期望 3.10, 实际 {python_version.major}.{python_version.minor})")
    sys.exit(1)

# 2. 检查 PyTorch
print("\n2. PyTorch 检查:")
try:
    import torch
    print(f"   ✓ PyTorch 导入成功")
    print(f"   PyTorch 版本: {torch.__version__}")
except ImportError as e:
    print(f"   ✗ PyTorch 导入失败: {e}")
    sys.exit(1)

# 3. 检查 torchvision
print("\n3. torchvision 检查:")
try:
    import torchvision
    print(f"   ✓ torchvision 导入成功")
    print(f"   torchvision 版本: {torchvision.__version__}")
except ImportError as e:
    print(f"   ✗ torchvision 导入失败: {e}")
    sys.exit(1)

# 4. 检查 torchaudio
print("\n4. torchaudio 检查:")
try:
    import torchaudio
    print(f"   ✓ torchaudio 导入成功")
    print(f"   torchaudio 版本: {torchaudio.__version__}")
except ImportError as e:
    print(f"   ⚠ torchaudio 导入失败 (可选): {e}")

# 5. 检查 CUDA
print("\n5. CUDA 检查:")
try:
    if torch.cuda.is_available():
        print(f"   ✓ CUDA 可用")
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("   ⚠ CUDA 不可用 (CPU 模式)")
except Exception as e:
    print(f"   ⚠ CUDA 检查失败: {e}")

# 6. 测试基本张量操作
print("\n6. 基本张量操作测试:")
try:
    # CPU 测试
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    z = x + y
    print(f"   ✓ CPU 张量操作正常")
    print(f"   示例: {x.shape} + {y.shape} = {z.shape}")
    
    # GPU 测试 (如果可用)
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        y_gpu = y.cuda()
        z_gpu = x_gpu + y_gpu
        print(f"   ✓ GPU 张量操作正常")
        print(f"   示例: {x_gpu.shape} + {y_gpu.shape} = {z_gpu.shape}")
except Exception as e:
    print(f"   ✗ 张量操作失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. 检查其他关键依赖
print("\n7. 其他关键依赖检查:")
dependencies = [
    ("numpy", "numpy"),
    ("PIL", "Pillow"),
    ("cv2", "opencv-python"),
]
for module_name, package_name in dependencies:
    try:
        __import__(module_name)
        print(f"   ✓ {package_name} 可用")
    except ImportError:
        print(f"   ⚠ {package_name} 未安装 (可选)")

print("\n" + "=" * 60)
print("✓ 所有关键测试通过！Python 升级成功")
print("=" * 60)

