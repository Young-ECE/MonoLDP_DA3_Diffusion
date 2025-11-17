#!/usr/bin/env python3
"""测试 DataLoader 是否正常工作"""

import sys
sys.path.append('/home/jingyang/MonoLDP/DepthEstimation')

from options import MonodepthOptions
import datasets

print("=" * 80)
print("测试 DataLoader")
print("=" * 80)

# 创建选项
options = MonodepthOptions()
opt = options.parse()
opt.batch_size = 2
opt.num_workers = 2

print(f"\n配置:")
print(f"  - batch_size: {opt.batch_size}")
print(f"  - num_workers: {opt.num_workers}")

# 创建数据集
print(f"\n创建训练数据集...")
train_dataset = datasets.NYUDataset(
    opt.data_path, 
    'train',
    opt.height, opt.width,
    opt.frame_ids, 4, is_train=True,
    return_plane=not opt.disable_plane_regularization,
    num_plane_keysets=opt.num_plane_keysets,
    return_line=not opt.disable_line_regularization,
    num_line_keysets=opt.num_line_keysets,
    img_ext='.jpg')

print(f"✓ 数据集创建成功，共 {len(train_dataset)} 个样本")

# 创建 DataLoader
print(f"\n创建 DataLoader...")
from torch.utils.data import DataLoader
train_loader = DataLoader(
    train_dataset, opt.batch_size, True,
    num_workers=opt.num_workers, pin_memory=True, drop_last=True)

print(f"✓ DataLoader 创建成功")

# 测试加载一个 batch
print(f"\n测试加载一个 batch...")
try:
    train_iter = iter(train_loader)
    inputs = next(train_iter)
    print(f"✓ 成功加载第一个 batch!")
    print(f"\n  样本数量: {opt.batch_size}")
    print(f"  Keys: {list(inputs.keys())[:10]}... (showing first 10)")
    
    # 测试再加载一个
    inputs = next(train_iter)
    print(f"✓ 成功加载第二个 batch!")
    
    print(f"\n" + "=" * 80)
    print("✓ DataLoader 测试通过!")
    print("=" * 80)
    
except Exception as e:
    print(f"\n✗ 加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

