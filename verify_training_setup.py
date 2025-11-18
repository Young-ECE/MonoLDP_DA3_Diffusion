#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证训练环境是否就绪
检查依赖、模型加载、数据路径等
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DepthEstimation'))

def check_imports():
    """检查必要的导入"""
    print("=" * 60)
    print("检查 1: Python 包导入")
    print("=" * 60)
    
    results = {}
    
    # 基础包
    try:
        import torch
        print(f"✓ torch: {torch.__version__}")
        results['torch'] = True
    except ImportError as e:
        print(f"✗ torch: {e}")
        results['torch'] = False
    
    try:
        import numpy
        print(f"✓ numpy: {numpy.__version__}")
        results['numpy'] = True
    except ImportError as e:
        print(f"✗ numpy: {e}")
        results['numpy'] = False
    
    # MonoLDP 模块
    try:
        import networks
        print("✓ networks 模块")
        results['networks'] = True
    except ImportError as e:
        print(f"✗ networks: {e}")
        results['networks'] = False
    
    try:
        from options import MonodepthOptions
        print("✓ options 模块")
        results['options'] = True
    except ImportError as e:
        print(f"✗ options: {e}")
        results['options'] = False
    
    try:
        from trainer import Trainer
        print("✓ trainer 模块")
        results['trainer'] = True
    except ImportError as e:
        print(f"✗ trainer: {e}")
        results['trainer'] = False
    
    return results

def check_da3_package():
    """检查 DA3 包是否安装"""
    print("\n" + "=" * 60)
    print("检查 2: Depth Anything V3 包")
    print("=" * 60)
    
    try:
        from depth_anything_3 import DA3Model
        print("✓ depth_anything_3 包已安装")
        return True
    except ImportError:
        print("✗ depth_anything_3 包未安装")
        print("  请运行: pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git")
        return False

def check_da3_wrapper():
    """检查 DA3 wrapper 是否可用"""
    print("\n" + "=" * 60)
    print("检查 3: DA3 Wrapper")
    print("=" * 60)
    
    try:
        from networks import create_depth_anything_v3_teacher
        print("✓ DA3 wrapper 导入成功")
        return True
    except ImportError as e:
        print(f"✗ DA3 wrapper 导入失败: {e}")
        return False

def check_options():
    """检查命令行参数"""
    print("\n" + "=" * 60)
    print("检查 4: 命令行参数")
    print("=" * 60)
    
    try:
        from options import MonodepthOptions
        
        # 创建测试参数
        test_args = [
            "--model_name", "test_verify",
            "--use_diffusion",
            "--use_depth_anything_v3",
            "--depth_anything_v3_model", "DA3Mono-Large",
            "--data_path", "/tmp/test",
            "--batch_size", "1",
            "--num_epochs", "1",
            "--debug_no_save"
        ]
        
        import sys
        old_argv = sys.argv
        sys.argv = ["test"] + test_args
        
        options = MonodepthOptions()
        opts = options.parse()
        
        sys.argv = old_argv
        
        print(f"✓ 参数解析成功")
        print(f"  - use_depth_anything_v3: {opts.use_depth_anything_v3}")
        print(f"  - depth_anything_v3_model: {opts.depth_anything_v3_model}")
        return True
    except Exception as e:
        print(f"✗ 参数解析失败: {e}")
        return False

def check_data_path(data_path=None):
    """检查数据路径"""
    print("\n" + "=" * 60)
    print("检查 5: 数据路径")
    print("=" * 60)
    
    if data_path is None:
        # 使用默认路径
        data_path = "/oldisk/home/jingyang/monoldp/datasets/nyu_data"
    
    if os.path.exists(data_path):
        print(f"✓ 数据路径存在: {data_path}")
        
        # 检查是否有必要的文件
        splits_dir = os.path.join(os.path.dirname(__file__), "DepthEstimation", "splits", "nyu")
        train_file = os.path.join(splits_dir, "train_files.txt")
        if os.path.exists(train_file):
            print(f"✓ 训练文件列表存在: {train_file}")
        else:
            print(f"⚠ 训练文件列表不存在: {train_file}")
        
        return True
    else:
        print(f"✗ 数据路径不存在: {data_path}")
        print("  请检查数据路径是否正确")
        return False

def check_model_loading():
    """检查模型加载（不实际加载，只检查代码）"""
    print("\n" + "=" * 60)
    print("检查 6: 模型加载代码")
    print("=" * 60)
    
    try:
        # 检查 wrapper 代码语法
        wrapper_path = os.path.join(
            os.path.dirname(__file__), 
            "DepthEstimation", 
            "networks", 
            "depth_anything_v3_wrapper.py"
        )
        
        if os.path.exists(wrapper_path):
            print(f"✓ DA3 wrapper 文件存在: {wrapper_path}")
            
            # 尝试编译检查语法
            with open(wrapper_path, 'r') as f:
                code = f.read()
            compile(code, wrapper_path, 'exec')
            print("✓ DA3 wrapper 代码语法正确")
            return True
        else:
            print(f"✗ DA3 wrapper 文件不存在")
            return False
    except SyntaxError as e:
        print(f"✗ DA3 wrapper 代码语法错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 检查失败: {e}")
        return False

def check_trainer_integration():
    """检查 trainer 集成"""
    print("\n" + "=" * 60)
    print("检查 7: Trainer 集成")
    print("=" * 60)
    
    try:
        trainer_path = os.path.join(
            os.path.dirname(__file__), 
            "DepthEstimation", 
            "trainer.py"
        )
        
        with open(trainer_path, 'r') as f:
            content = f.read()
        
        # 检查关键代码是否存在
        checks = {
            "use_depth_anything_v3": "use_depth_anything_v3" in content,
            "create_depth_anything_v3_teacher": "create_depth_anything_v3_teacher" in content,
            "use_depth_anything_v3_teacher": "use_depth_anything_v3_teacher" in content,
        }
        
        all_ok = True
        for key, found in checks.items():
            if found:
                print(f"✓ {key} 已集成")
            else:
                print(f"✗ {key} 未找到")
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"✗ 检查失败: {e}")
        return False

def check_gpu():
    """检查 GPU"""
    print("\n" + "=" * 60)
    print("检查 8: GPU 可用性")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用")
            print(f"  - GPU 数量: {torch.cuda.device_count()}")
            print(f"  - 当前 GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA 版本: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA 不可用，将使用 CPU（训练会很慢）")
            return False
    except Exception as e:
        print(f"✗ GPU 检查失败: {e}")
        return False

def main():
    """主验证函数"""
    print("\n" + "🔍 " * 20)
    print("训练环境验证")
    print("🔍 " * 20 + "\n")
    
    results = {}
    
    # 执行检查
    results['imports'] = check_imports()
    results['da3_package'] = check_da3_package()
    results['da3_wrapper'] = check_da3_wrapper()
    results['options'] = check_options()
    results['data_path'] = check_data_path()
    results['model_loading'] = check_model_loading()
    results['trainer_integration'] = check_trainer_integration()
    results['gpu'] = check_gpu()
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    critical_checks = ['imports', 'da3_wrapper', 'options', 'model_loading', 'trainer_integration']
    optional_checks = ['da3_package', 'data_path', 'gpu']
    
    critical_passed = all(
        isinstance(results.get(k), dict) and all(results[k].values()) if isinstance(results.get(k), dict) 
        else results.get(k, False) 
        for k in critical_checks
    )
    
    print("\n关键检查:")
    for key in critical_checks:
        result = results.get(key)
        if isinstance(result, dict):
            status = "✓" if all(result.values()) else "✗"
        else:
            status = "✓" if result else "✗"
        print(f"  {status} {key}")
    
    print("\n可选检查:")
    for key in optional_checks:
        result = results.get(key)
        status = "✓" if result else "⚠"
        print(f"  {status} {key}")
    
    if critical_passed:
        print("\n" + "🎉 " * 20)
        print("✓ 关键检查全部通过！代码可以用于训练")
        print("🎉 " * 20)
        
        if not results.get('da3_package'):
            print("\n⚠️  注意: DA3 包未安装，训练时会失败")
            print("   请先安装: pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git")
        
        if not results.get('data_path'):
            print("\n⚠️  注意: 数据路径不存在，请检查数据路径")
        
        return True
    else:
        print("\n" + "⚠️  " * 20)
        print("✗ 部分关键检查失败，请修复后再训练")
        print("⚠️  " * 20)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

