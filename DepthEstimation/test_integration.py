#!/usr/bin/env python3
"""
测试扩散模块集成是否成功
运行: python test_integration.py
"""

import sys
import os

# 当前脚本位于 DepthEstimation 目录下
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

# 将项目根目录加入 sys.path，以便以包名形式导入
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def test_imports():
    """测试所有必要的导入"""
    print("=" * 60)
    print("测试1: 检查导入")
    print("=" * 60)
    
    try:
        from DepthEstimation.networks import DepthDecoderDiffusion
        print("✅ DepthDecoderDiffusion 导入成功")
    except Exception as e:
        print(f"❌ DepthDecoderDiffusion 导入失败: {e}")
        return False
    
    try:
        from DepthEstimation.diffusers.schedulers.scheduling_ddim import DDIMScheduler
        print("✅ DDIMScheduler 导入成功")
    except Exception as e:
        print(f"❌ DDIMScheduler 导入失败: {e}")
        return False
    
    try:
        from DepthEstimation.trainer import Trainer
        print("✅ Trainer 导入成功")
    except Exception as e:
        print(f"❌ Trainer 导入失败: {e}")
        return False
    
    print()
    return True

def test_model_creation():
    """测试模型创建"""
    print("=" * 60)
    print("测试2: 创建扩散解码器")
    print("=" * 60)
    
    try:
        import torch
        from DepthEstimation.networks import DepthDecoderDiffusion
        
        # 创建模型
        num_ch_enc = [64, 128, 256]
        model = DepthDecoderDiffusion(
            num_ch_enc=num_ch_enc,
            scales=[0, 1, 2],
            num_output_channels=3,
            PixelCoorModu=True
        )
        
        print(f"✅ 模型创建成功")
        print(f"   模型参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        
        # 测试前向传播
        batch_size = 2
        # 构造与解码器期望一致的特征尺寸（逐级下采样）
        # 解码器会从最深层开始逐级上采样
        features = [
            torch.randn(batch_size, 64, 32, 40),   # level 0
            torch.randn(batch_size, 128, 16, 20),  # level 1
            torch.randn(batch_size, 256, 8, 10),   # level 2 (bottleneck)
        ]
        
        norm_pix_coords = [
            torch.randn(batch_size, 2, 64, 80),    # scale 0 (finest)
            torch.randn(batch_size, 2, 32, 40),    # scale 1 (middle)
            torch.randn(batch_size, 2, 16, 20),    # scale 2 (coarsest)
        ]
        
        # GT 尺寸需要匹配解码器上采样后的特征尺寸
        # scale 2 (i=2): 8×10 -> upsample -> 16×20
        # scale 1 (i=1): 16×20 -> upsample -> 32×40
        # scale 0 (i=0): 32×40 -> upsample -> 64×80
        gt = {
            ("disp_diffusion", 0): torch.randn(batch_size, 3, 64, 80),
            ("disp_diffusion", 1): torch.randn(batch_size, 3, 32, 40),
            ("disp_diffusion", 2): torch.randn(batch_size, 3, 16, 20),
        }
        
        print("   测试前向传播...")
        model.eval()
        with torch.no_grad():
            outputs = model(features, norm_pix_coords, gt)
        
        print(f"✅ 前向传播成功")
        print(f"   输出键: {list(outputs.keys())}")
        print(f"   输出形状:")
        for key in [("disp", 0), ("disp", 1), ("disp", 2)]:
            if key in outputs:
                print(f"     {key}: {outputs[key].shape}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ 模型创建/测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """测试文件结构"""
    print("=" * 60)
    print("测试3: 检查文件结构")
    print("=" * 60)
    
    required_files = [
        'DepthEstimation/diffusers/schedulers/scheduling_ddim.py',
        'DepthEstimation/networks/depth_decoder_diffusion.py',
        'DepthEstimation/networks/__init__.py',
        'DepthEstimation/options.py',
        'DepthEstimation/trainer.py',
    ]
    
    all_exist = True
    for filepath in required_files:
        full_path = os.path.join(PROJECT_ROOT, filepath)
        if os.path.exists(full_path):
            print(f"✅ {filepath}")
        else:
            print(f"❌ {filepath} 不存在")
            all_exist = False
    
    print()
    return all_exist

def test_options():
    """测试命令行选项"""
    print("=" * 60)
    print("测试4: 检查扩散相关参数")
    print("=" * 60)
    
    try:
        from DepthEstimation.options import MonodepthOptions
        
        options = MonodepthOptions()
        # 模拟命令行参数
        args = options.parser.parse_args([
            '--use_diffusion',
            '--teacher_weights_folder', './dummy_path',
            '--diffusion_l1_weight', '1.0',
            '--diffusion_ddim_weight', '1.0'
        ])
        
        print(f"✅ 扩散参数解析成功")
        print(f"   use_diffusion: {args.use_diffusion}")
        print(f"   teacher_weights_folder: {args.teacher_weights_folder}")
        print(f"   diffusion_l1_weight: {args.diffusion_l1_weight}")
        print(f"   diffusion_ddim_weight: {args.diffusion_ddim_weight}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ 选项解析失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("MonoDiffusion 集成测试")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行测试
    results.append(("导入测试", test_imports()))
    results.append(("文件结构测试", test_file_structure()))
    results.append(("选项解析测试", test_options()))
    results.append(("模型创建测试", test_model_creation()))
    
    # 总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！集成成功！")
        print("\n下一步:")
        print("  1. 训练基础模型:")
        print("     python train.py --model_name base --num_epochs 15")
        print("\n  2. 使用扩散训练:")
        print("     python train.py --model_name diffusion --use_diffusion \\")
        print("         --teacher_weights_folder ./logs/base/models/weights_14")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
        print("\n请参考以下文档:")
        print("  - QUICK_START.md")
        print("  - 修改完成总结.md")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())

