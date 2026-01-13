"""
深度图生成脚本：使用DA3Mono-Large模型和学生模型从NYUv2数据集中生成深度图

使用方法:
    python generate_depth_maps.py \
        --data_path /path/to/nyu_data \
        --da3_model_path /path/to/da3/model \
        --student_model_path /path/to/student/model \
        --output_dir ./depth_results \
        --image_indices 0 100 200

功能:
    1. 从NYUv2测试集中加载指定索引的RGB图片
    2. 使用DA3Mono-Large模型生成深度图
    3. 使用训练好的学生模型生成深度图
    4. 保存结果并记录图片路径
"""
import os
import argparse
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms

# 导入项目模块
from options import MonodepthOptions
from utils import readlines
from layers import disp_to_depth
import datasets
import networks

# 尝试导入 DA3
try:
    from depth_anything_3.api import DepthAnything3
    DA3_AVAILABLE = True
except ImportError:
    DA3_AVAILABLE = False
    print("Warning: depth_anything_3 package not found. DA3 inference will be disabled.")


def load_da3_model(da3_model_path):
    """加载DA3Mono-Large模型"""
    if not DA3_AVAILABLE:
        raise RuntimeError("depth_anything_3 package is required for DA3 inference")
    
    if da3_model_path:
        print(f"-> Loading DA3 model from: {da3_model_path}")
        model = DepthAnything3.from_pretrained(da3_model_path).cuda().eval()
    else:
        print("-> Loading DA3MONO-LARGE from HuggingFace")
        model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE").cuda().eval()
    
    return model


def load_student_model(student_model_path, opt):
    """加载学生模型"""
    print(f"-> Loading student model from: {student_model_path}")
    
    encoder_path = os.path.join(student_model_path, "encoder.pth")
    decoder_path = os.path.join(student_model_path, "depth.pth")
    scalenet_path = os.path.join(student_model_path, "scalenet.pth")
    regression_path = os.path.join(student_model_path, "regression.pth")
    
    # 加载配置（opt.json 在 models 目录下，不在 weights_X 目录下）
    weights_dir = os.path.dirname(student_model_path)  # 获取 models 目录
    opt_json_path = os.path.join(weights_dir, "opt.json")
    if os.path.exists(opt_json_path):
        print(f"-> Loading configuration from {opt_json_path}")
        with open(opt_json_path, 'r') as f:
            saved_opts = json.load(f)
            # 更新opt（优先使用保存的配置）
            for key, value in saved_opts.items():
                # 对于模型架构相关的关键参数，使用保存的配置
                if key in ['num_layers', 'scales', 'height', 'width', 'min_depth', 'max_depth',
                          'diffusion_steps', 'diffusion_timesteps', 'use_diffusion_decoder',
                          'disable_pixel_coordinate_modulation']:
                    setattr(opt, key, value)
                elif not hasattr(opt, key):
                    setattr(opt, key, value)
    else:
        print(f"-> Warning: opt.json not found at {opt_json_path}, using default/command line arguments")
    
    # 检查权重文件是否存在
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder weights not found: {encoder_path}")
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Decoder weights not found: {decoder_path}")
    if not os.path.exists(scalenet_path):
        raise FileNotFoundError(f"Scalenet weights not found: {scalenet_path}")
    if not os.path.exists(regression_path):
        raise FileNotFoundError(f"Regression weights not found: {regression_path}")
    
    encoder_dict = torch.load(encoder_path, map_location='cpu', weights_only=False)
    
    # 构建模型
    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoderDiffusion(
        encoder.num_ch_enc,
        opt.scales,
        num_output_channels=1,
        use_skips=True,
        diffusion_steps=getattr(opt, 'diffusion_steps', [20, 15, 10]),
        diffusion_timesteps=getattr(opt, 'diffusion_timesteps', [500, 400, 300])
    )
    scalenet = networks.ScaleNetwork(encoder.num_ch_enc)
    regression_heads = nn.ModuleList([
        networks.ProbabilisticScaleRegressionHead(in_channels=ch) 
        for ch in encoder.num_ch_enc
    ])
    
    # 加载权重
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    decoder_dict = torch.load(decoder_path, map_location='cpu', weights_only=False)
    depth_decoder.load_state_dict(decoder_dict, strict=False)
    scalenet.load_state_dict(torch.load(scalenet_path, map_location='cpu', weights_only=False))
    regression_heads.load_state_dict(torch.load(regression_path, map_location='cpu', weights_only=False))
    
    # 移动到GPU
    encoder.cuda().eval()
    depth_decoder.cuda().eval()
    scalenet.cuda().eval()
    regression_heads.cuda().eval()
    
    return encoder, depth_decoder, scalenet, regression_heads


def generate_da3_depth(model, image_path, eval_height=256, eval_width=320):
    """使用DA3模型生成深度图"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    original_width, original_height = image.size
    
    # 预处理：边缘裁剪（NYU数据集标准）
    edge_crop = 16
    image = image.crop((edge_crop, edge_crop, original_width - edge_crop, original_height - edge_crop))
    
    # 调整尺寸（确保是14的倍数，DA3的要求）
    patch_size = 14
    eval_height = (eval_height // patch_size) * patch_size
    eval_width = (eval_width // patch_size) * patch_size
    image_resized = image.resize((eval_width, eval_height), Image.LANCZOS)
    
    # 转换为tensor并归一化
    to_tensor = transforms.ToTensor()
    input_tensor = to_tensor(image_resized).unsqueeze(0)  # (1, 3, H, W)
    
    # ImageNet归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    input_norm = (input_tensor.cuda() - mean) / std
    
    # DA3 forward需要 (B, N, 3, H, W) 格式
    input_da3 = input_norm.unsqueeze(1)  # (1, 1, 3, H, W)
    
    with torch.no_grad():
        output = model.forward(
            input_da3,
            extrinsics=None,
            intrinsics=None,
            export_feat_layers=[],
            infer_gs=False
        )
    
    # 处理输出
    if isinstance(output, dict):
        pred_depth = output.get('depth', None)
        if pred_depth is None:
            for key in ['predicted_depth', 'pred', 'depth_map']:
                if key in output:
                    pred_depth = output[key]
                    break
    else:
        pred_depth = output
    
    if pred_depth is None:
        raise RuntimeError("Could not extract depth from DA3 output")
    
    # 确保深度形状正确 (B, H, W)
    if pred_depth.ndim == 4:
        if pred_depth.shape[1] == 1:
            pred_depth = pred_depth.squeeze(1)
        else:
            pred_depth = pred_depth[:, 0]
    elif pred_depth.ndim == 5:
        pred_depth = pred_depth[:, 0, 0]
    
    # 转换为numpy
    depth_np = pred_depth[0].cpu().numpy()  # (H, W)
    
    # 调整回原始裁剪后的尺寸
    crop_height, crop_width = image.size[1], image.size[0]
    if depth_np.shape[0] != crop_height or depth_np.shape[1] != crop_width:
        depth_np = cv2.resize(depth_np, (crop_width, crop_height), interpolation=cv2.INTER_LINEAR)
    
    return depth_np, image


def generate_student_depth(encoder, decoder, scalenet, regression_heads, image_path, opt):
    """使用学生模型生成深度图"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    original_width, original_height = image.size
    
    # 预处理：边缘裁剪
    edge_crop = 16
    image = image.crop((edge_crop, edge_crop, original_width - edge_crop, original_height - edge_crop))
    crop_width, crop_height = image.size
    
    # 调整尺寸（确保是32的倍数，ResNet的要求）
    eval_height = (opt.height // 32) * 32
    eval_width = (opt.width // 32) * 32
    image_resized = image.resize((eval_width, eval_height), Image.LANCZOS)
    
    # 转换为tensor
    to_tensor = transforms.ToTensor()
    input_tensor = to_tensor(image_resized).unsqueeze(0).cuda()  # (1, 3, H, W)
    
    with torch.no_grad():
        # 编码器
        features = encoder(input_tensor)
        
        # ScaleNet和Regression Heads
        depth_factors = scalenet(features)
        scale_predictions = [head(factor) for head, factor in zip(regression_heads, depth_factors)]
        max_depth = torch.mean(torch.stack(scale_predictions), dim=0)  # (1,)
        
        # 解码器（Diffusion）
        output = decoder(features, gt=None, mask=None)
        
        # 获取disparity并转换为深度
        disp = output[("disp", 0)]  # (1, 1, H, W) 或 (1, H, W)
        if disp.ndim == 3:
            disp = disp.unsqueeze(1)
        
        # 转换为深度
        _, pred_depth = disp_to_depth(disp, opt.min_depth, max_depth[0].item())
        
        # 确保深度形状正确 (1, H, W)
        if pred_depth.ndim == 4:
            pred_depth = pred_depth[:, 0]  # (1, H, W)
        elif pred_depth.ndim == 2:
            pred_depth = pred_depth.unsqueeze(0)
        
        # 调整回原始裁剪后的尺寸
        pred_depth_resized = F.interpolate(
            pred_depth.unsqueeze(1),
            size=(crop_height, crop_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        depth_np = pred_depth_resized[0].cpu().numpy()  # (H, W)
    
    return depth_np, image


def save_depth_visualization(depth_map, image, output_path, title=""):
    """保存深度图可视化"""
    # 创建colormap
    vmax = np.percentile(depth_map, 95)
    normalizer = plt.Normalize(vmin=depth_map.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped = mapper.to_rgba(depth_map)[:, :, :3]
    
    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # 深度图
    axes[1].imshow(colormapped)
    axes[1].set_title(f'Depth Map {title}', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 同时保存纯深度图（numpy格式）
    npy_path = output_path.replace('.png', '.npy')
    np.save(npy_path, depth_map)
    print(f"  -> Saved: {output_path}")
    print(f"  -> Saved: {npy_path}")


def main():
    # 先解析自定义参数（使用 parse_known_args 来避免与 MonodepthOptions 冲突）
    custom_parser = argparse.ArgumentParser(description='Generate depth maps using DA3 and student models', 
                                           add_help=False)  # add_help=False 避免与 MonodepthOptions 的 help 冲突
    custom_parser.add_argument('--data_path', type=str, required=True,
                        help='Path to NYU dataset')
    custom_parser.add_argument('--da3_model_path', type=str, default=None,
                        help='Path to DA3 model (if None, will use HuggingFace cache)')
    custom_parser.add_argument('--student_model_path', type=str, required=True,
                        help='Path to trained student model weights folder')
    custom_parser.add_argument('--output_dir', type=str, default='./depth_results',
                        help='Output directory for depth maps')
    custom_parser.add_argument('--image_indices', type=int, nargs='+', default=[0, 100, 200],
                        help='Indices of images to process (default: 0 100 200)')
    custom_parser.add_argument('--eval_split', type=str, default='nyu',
                        help='Dataset split to use (default: nyu)')
    
    # 解析自定义参数和已知参数（MonodepthOptions 的参数）
    custom_args, remaining_args = custom_parser.parse_known_args()
    
    # 展开用户路径
    custom_args.data_path = os.path.expanduser(custom_args.data_path)
    if custom_args.da3_model_path:
        custom_args.da3_model_path = os.path.expanduser(custom_args.da3_model_path)
    custom_args.student_model_path = os.path.expanduser(custom_args.student_model_path)
    
    # 验证路径
    print("="*60)
    print("Configuration")
    print("="*60)
    print(f"Data path: {custom_args.data_path}")
    print(f"DA3 model path: {custom_args.da3_model_path or 'Will load from HuggingFace'}")
    print(f"Student model path: {custom_args.student_model_path}")
    print(f"Output directory: {custom_args.output_dir}")
    print(f"Image indices: {custom_args.image_indices}")
    print(f"Eval split: {custom_args.eval_split}")
    print()
    
    # 检查数据路径是否存在
    if not os.path.exists(custom_args.data_path):
        print(f"Error: Data path does not exist: {custom_args.data_path}")
        return
    
    # 创建输出目录
    os.makedirs(custom_args.output_dir, exist_ok=True)
    print(f"-> Output directory: {custom_args.output_dir}")
    
    # 加载 MonodepthOptions（使用剩余的参数）
    # 需要临时修改 sys.argv 来只传递剩余的参数
    import sys
    original_argv = sys.argv[:]
    try:
        # 只保留脚本名和剩余的参数
        sys.argv = [sys.argv[0]] + remaining_args
        opt = MonodepthOptions().parse()
    finally:
        sys.argv = original_argv
    
    # 覆盖 opt 中的关键参数
    opt.data_path = custom_args.data_path
    opt.eval_split = custom_args.eval_split
    
    # 确保必要的默认值存在
    if not hasattr(opt, 'num_layers') or opt.num_layers is None:
        opt.num_layers = 18
    if not hasattr(opt, 'scales') or opt.scales is None:
        opt.scales = [0, 1, 2]
    if not hasattr(opt, 'height') or opt.height is None:
        opt.height = 256
    if not hasattr(opt, 'width') or opt.width is None:
        opt.width = 320
    if not hasattr(opt, 'min_depth') or opt.min_depth is None:
        opt.min_depth = 0.1
    if not hasattr(opt, 'max_depth') or opt.max_depth is None:
        opt.max_depth = 10.0
    
    # 加载测试文件列表
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")
    test_files = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    
    # 选择指定索引的图片
    selected_files = []
    for idx in custom_args.image_indices:
        if idx < len(test_files):
            filename = test_files[idx]
            parts = filename.split()
            folder = parts[0]
            frame_index = parts[1]
            # 构建图片路径（测试集格式：folder/XXXXX_colors.png）
            image_path = os.path.join(custom_args.data_path, folder, f"{frame_index}_colors.png")
            
            # 预先检查文件是否存在
            if os.path.exists(image_path):
                selected_files.append((idx, filename))
                print(f"-> Found image at index {idx}: {filename} -> {image_path}")
            else:
                print(f"-> Warning: Image not found at index {idx}: {image_path}")
                print(f"   (Filename: {filename})")
        else:
            print(f"-> Warning: Index {idx} is out of range (total: {len(test_files)}), skipping")
    
    if len(selected_files) == 0:
        print("\nError: No valid images found!")
        print(f"   Data path: {custom_args.data_path}")
        print(f"   Total test files: {len(test_files)}")
        print(f"   Requested indices: {custom_args.image_indices}")
        print("\nPlease check:")
        print("   1. Data path is correct")
        print("   2. Image files exist in the dataset")
        print("   3. Image indices are valid (0 to {})".format(len(test_files) - 1))
        return
    
    print(f"\n-> Selected {len(selected_files)} images for processing")
    
    # 加载模型
    print("\n" + "="*60)
    print("Loading Models...")
    print("="*60)
    
    da3_model = None
    if DA3_AVAILABLE:
        try:
            da3_model = load_da3_model(custom_args.da3_model_path)
            print("-> DA3 model loaded successfully")
        except Exception as e:
            print(f"-> Failed to load DA3 model: {e}")
            print("-> Will skip DA3 depth generation")
    else:
        print("-> DA3 package not available, skipping DA3 depth generation")
    
    try:
        encoder, decoder, scalenet, regression_heads = load_student_model(custom_args.student_model_path, opt)
        print("-> Student model loaded successfully")
    except Exception as e:
        print(f"-> Failed to load student model: {e}")
        raise
    
    # 处理每张图片
    print("\n" + "="*60)
    print("Generating Depth Maps...")
    print("="*60)
    
    results = []
    
    for img_idx, (file_idx, filename) in enumerate(selected_files):
        print(f"\n[{img_idx+1}/{len(selected_files)}] Processing image {file_idx}: {filename}")
        
        # 解析文件名
        parts = filename.split()
        folder = parts[0]
        frame_index = parts[1]
        
        # 构建图片路径（测试集格式：folder/XXXXX_colors.png）
        image_path = os.path.join(custom_args.data_path, folder, f"{frame_index}_colors.png")
        
        # 再次检查文件是否存在（虽然之前已经检查过，但以防万一）
        if not os.path.exists(image_path):
            print(f"  -> Error: Image not found: {image_path}")
            print(f"  -> Data path: {custom_args.data_path}")
            print(f"  -> Folder: {folder}")
            print(f"  -> Frame index: {frame_index}")
            print(f"  -> Expected path: {image_path}")
            # 尝试列出文件夹内容以帮助调试
            folder_path = os.path.join(custom_args.data_path, folder)
            if os.path.exists(folder_path):
                files_in_folder = os.listdir(folder_path)[:5]
                print(f"  -> Files in {folder_path}: {files_in_folder}...")
            continue
        
        print(f"  -> Image path: {image_path}")
        
        # 创建该图片的输出目录
        img_output_dir = os.path.join(custom_args.output_dir, f"image_{file_idx:05d}")
        os.makedirs(img_output_dir, exist_ok=True)
        
        # 保存原始图片
        original_image = Image.open(image_path).convert('RGB')
        original_save_path = os.path.join(img_output_dir, "original_image.png")
        original_image.save(original_save_path)
        
        result = {
            "index": file_idx,
            "filename": filename,
            "image_path": image_path,
            "output_dir": img_output_dir,
            "original_image": original_save_path
        }
        
        # 使用DA3生成深度图
        if da3_model is not None:
            try:
                print("  -> Generating depth with DA3...")
                da3_depth, processed_image = generate_da3_depth(da3_model, image_path)
                da3_output_path = os.path.join(img_output_dir, "depth_da3.png")
                save_depth_visualization(da3_depth, processed_image, da3_output_path, "(DA3)")
                result["da3_depth"] = da3_output_path
                result["da3_depth_npy"] = da3_output_path.replace('.png', '.npy')
                print(f"  -> DA3 depth generated successfully")
            except Exception as e:
                import traceback
                print(f"  -> Error generating DA3 depth: {e}")
                print(f"  -> Traceback: {traceback.format_exc()}")
                result["da3_depth"] = None
        
        # 使用学生模型生成深度图
        try:
            print("  -> Generating depth with student model...")
            student_depth, processed_image = generate_student_depth(
                encoder, decoder, scalenet, regression_heads, image_path, opt
            )
            student_output_path = os.path.join(img_output_dir, "depth_student.png")
            save_depth_visualization(student_depth, processed_image, student_output_path, "(Student)")
            result["student_depth"] = student_output_path
            result["student_depth_npy"] = student_output_path.replace('.png', '.npy')
            print(f"  -> Student depth generated successfully")
        except Exception as e:
            import traceback
            print(f"  -> Error generating student depth: {e}")
            print(f"  -> Traceback: {traceback.format_exc()}")
            result["student_depth"] = None
        
        results.append(result)
    
    # 保存结果摘要
    summary_path = os.path.join(custom_args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"-> Processed {len(results)} images")
    print(f"-> Results saved to: {custom_args.output_dir}")
    print(f"-> Summary saved to: {summary_path}")
    print("\nImage paths:")
    for r in results:
        print(f"  [{r['index']:05d}] {r['image_path']}")
        print(f"         Output: {r['output_dir']}")


if __name__ == '__main__':
    main()

