"""
学生模型评估脚本：用于评估训练好的学生模型（ResNet + Diffusion/Regular Decoder）
评估逻辑与 evaluate_da3mono_nyu_depth.py 保持一致

使用方法:
    python evaluate_student_nyu_depth.py \
        --student_model_path /path/to/models/weights_19 \
        --data_path /path/to/nyu_data \
        --eval_split nyu \
        --use_least_squares

关键特性:
    1. 自动从 opt.json 加载模型训练时的配置（优先于当前项目配置）
    2. 支持 Diffusion Decoder 和 Regular Decoder
    3. 使用模型预测的 global_depth（而非从预测深度估计）
    4. 支持 Least Squares 对齐和 Median Scaling 对齐
    5. 评估指标与 evaluate_da3mono_nyu_depth.py 完全一致

配置加载优先级:
    命令行参数 > opt.json 中的配置 > 默认值
    
    注意：评估相关参数（如 data_path, eval_split 等）始终使用命令行指定的值
"""
import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from options import MonodepthOptions
from utils import readlines
from layers import disp_to_depth
import datasets
import networks

cv2.setNumThreads(0)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def detect_decoder_type(decoder_path):
    """
    通过检查权重文件的键来自动检测解码器类型
    
    Args:
        decoder_path: decoder权重文件路径
        
    Returns:
        str: 'diffusion' 或 'regular'
    """
    try:
        decoder_dict = torch.load(decoder_path, map_location='cpu', weights_only=False)
        keys = list(decoder_dict.keys())
        
        # 检查是否有扩散模型的特征键
        # DepthDecoderDiffusion 的特征键：
        # - diffusion_models.*
        # - convs.("conconv", ...)
        # - schedulers.*
        # - diffusion_pipelines.*
        has_diffusion_keys = any(
            'diffusion_models' in k or 
            'conconv' in k or 
            'schedulers' in k or
            'diffusion_pipelines' in k
            for k in keys
        )
        
        # 检查是否有常规解码器的特征键
        # DepthDecoder 的特征键：
        # - decoder.0.conv.*
        # - decoder.1.conv.*
        has_regular_keys = any('decoder.0.conv' in k or 'decoder.1.conv' in k for k in keys)
        
        if has_diffusion_keys:
            return 'diffusion'
        elif has_regular_keys:
            return 'regular'
        else:
            # 如果无法确定，默认使用 diffusion（因为训练时总是用 diffusion）
            print("-> Warning: Cannot determine decoder type from keys, defaulting to 'diffusion'")
            return 'diffusion'
        
    except Exception as e:
        print(f"-> Warning: Error detecting decoder type: {e}, defaulting to 'diffusion'")
        return 'diffusion'


def load_opts_from_checkpoint(model_path):
    """
    从模型检查点文件夹加载 opt.json 配置
    
    Args:
        model_path: 模型权重文件夹路径（如 weights_19）
        
    Returns:
        dict: 加载的配置字典，如果文件不存在则返回 None
    """
    # opt.json 在 models 文件夹下，不在 weights_X 文件夹下
    # 所以需要向上查找
    weights_dir = os.path.dirname(model_path)  # 获取 models 目录
    opt_json_path = os.path.join(weights_dir, "opt.json")
    
    if os.path.exists(opt_json_path):
        print(f"-> Loading configuration from {opt_json_path}")
        with open(opt_json_path, 'r') as f:
            saved_opts = json.load(f)
        return saved_opts
    else:
        print(f"-> Warning: opt.json not found at {opt_json_path}, using command line arguments")
        return None


def merge_opts(saved_opts, cmd_opts):
    """
    合并保存的配置和命令行参数
    
    优先级：命令行参数 > 保存的配置 > 默认值
    
    Args:
        saved_opts: 从 opt.json 加载的配置字典
        cmd_opts: 命令行解析的配置对象
        
    Returns:
        argparse.Namespace: 合并后的配置对象
    """
    if saved_opts is None:
        return cmd_opts
    
    # 将命令行参数转换为字典
    cmd_dict = vars(cmd_opts).copy()
    
    # 评估相关的参数应该使用命令行指定的值（不覆盖）
    evaluation_specific_keys = [
        'data_path', 'eval_split', 'ext_disp_to_eval', 
        'save_pred_disps', 'no_eval', 'eval_out_dir',
        'post_process', 'disable_median_scaling', 
        'pred_depth_scale_factor', 'use_fixed_max_depth',
        'use_least_squares', 'batch_size', 'num_workers',
        'student_model_path', 'load_weights_folder'  # 评估专用参数
    ]
    
    # 从保存的配置中更新，但命令行参数优先
    for key, value in saved_opts.items():
        # 如果是评估特定参数，跳过（使用命令行值）
        if key in evaluation_specific_keys:
            continue
        
        # 如果命令行中没有这个键，或者值是 None，使用保存的配置
        if key not in cmd_dict:
            cmd_dict[key] = value
        elif cmd_dict[key] is None:
            cmd_dict[key] = value
        # 对于模型架构相关的关键参数，优先使用保存的配置
        elif key in ['num_layers', 'use_diffusion_decoder', 'disable_pixel_coordinate_modulation',
                     'diffusion_steps', 'diffusion_timesteps', 'scales', 'min_depth', 'max_depth',
                     'height', 'width', 'weights_init']:
            # 如果命令行使用的是默认值，使用保存的配置
            # 这里假设如果值等于常见默认值，则使用保存的配置
            cmd_dict[key] = value
    
    # 如果 opt.json 中没有 use_diffusion_decoder，默认设置为 True（因为训练时总是用 diffusion）
    if 'use_diffusion_decoder' not in cmd_dict or cmd_dict.get('use_diffusion_decoder') is None:
        # 但这里不设置，让检测函数来决定
        pass
    
    # 将字典转换回 Namespace
    import argparse
    merged_opts = argparse.Namespace(**cmd_dict)
    return merged_opts


def compute_errors(gt, pred):
    """标准的深度误差计算（与 evaluate_da3mono_nyu_depth.py 一致）"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate_student(opt):
    """评估学生模型"""
    # 1. 加载保存的配置
    # 使用 student_model_path（评估专用参数），如果没有则回退到 load_weights_folder（向后兼容）
    student_model_path = getattr(opt, 'student_model_path', None)
    if student_model_path is None:
        # 向后兼容：如果没有 student_model_path，使用 load_weights_folder
        student_model_path = getattr(opt, 'load_weights_folder', None)
    
    if student_model_path is None:
        raise ValueError("--student_model_path is required for evaluation. "
                        "Please specify the path to trained model weights folder.")
    
    student_model_path = os.path.expanduser(student_model_path)
    assert os.path.isdir(student_model_path), \
        "Cannot find folder at {}".format(student_model_path)
    
    saved_opts = load_opts_from_checkpoint(student_model_path)
    opt = merge_opts(saved_opts, opt)
    
    # 2. 使用配置中的输入分辨率（确保是32的倍数，ResNet的要求）
    eval_height = opt.height if hasattr(opt, 'height') else 256
    eval_width = opt.width if hasattr(opt, 'width') else 320
    
    # 确保是32的倍数
    eval_height = (eval_height // 32) * 32
    eval_width = (eval_width // 32) * 32
    
    print(f"-> 使用输入分辨率: {eval_width}x{eval_height} (已调整为32的倍数)")
    print(f"-> Model configuration:")
    print(f"   num_layers: {opt.num_layers}")
    print(f"   scales: {opt.scales}")
    print(f"   disable_pixel_coordinate_modulation: {getattr(opt, 'disable_pixel_coordinate_modulation', False)}")
    
    # 3. 加载数据
    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    dataset = datasets.NYUDataset(
        opt.data_path, filenames, eval_height, eval_width, 
        [0], 1, is_test=True, 
        return_plane=False, num_plane_keysets=0,
        return_line=False, num_line_keysets=0
    )
    dataloader = DataLoader(
        dataset, opt.batch_size, shuffle=False, 
        num_workers=opt.num_workers, pin_memory=True, drop_last=False
    )
    
    # 4. 加载模型权重
    print("-> Loading model weights from {}".format(student_model_path))
    encoder_path = os.path.join(student_model_path, "encoder.pth")
    decoder_path = os.path.join(student_model_path, "depth.pth")
    scalenet_path = os.path.join(student_model_path, "scalenet.pth")
    regression_path = os.path.join(student_model_path, "regression.pth")
    
    encoder_dict = torch.load(encoder_path, weights_only=False)
    
    # 5. 强制使用 diffusion 解码器（因为训练时总是使用 diffusion）
    # 如果检测到不是 diffusion，给出警告但仍然使用 diffusion
    use_diffusion_decoder = getattr(opt, 'use_diffusion_decoder', None)
    
    if use_diffusion_decoder is None or use_diffusion_decoder is False:
        # 检测权重文件类型，但强制使用 diffusion
        print("-> Auto-detecting decoder type from weights file...")
        decoder_type = detect_decoder_type(decoder_path)
        print(f"-> Detected decoder type: {decoder_type}")
        
        if decoder_type != 'diffusion':
            print(f"-> Warning: Detected {decoder_type} decoder, but forcing use of DepthDecoderDiffusion")
            print(f"-> (Training always uses diffusion decoder, so evaluation should match)")
        
        use_diffusion_decoder = True  # 强制使用 diffusion
        print(f"-> Will use DepthDecoderDiffusion")
    else:
        print(f"-> Using decoder type from config: diffusion")
        use_diffusion_decoder = True
    
    # 6. 构建模型架构（使用保存的配置）
    encoder = networks.ResnetEncoder(opt.num_layers, False)
    
    # 根据检测结果选择解码器类型（优先使用 diffusion）
    if use_diffusion_decoder:
        print("-> Building DepthDecoderDiffusion")
        depth_decoder = networks.DepthDecoderDiffusion(
            encoder.num_ch_enc,
            opt.scales,
            num_output_channels=1,
            use_skips=True,
            diffusion_steps=getattr(opt, 'diffusion_steps', [20, 15, 10]),
            diffusion_timesteps=getattr(opt, 'diffusion_timesteps', [500, 400, 300])
        )
    else:
        print("-> Building DepthDecoder (with Pixel Coordinate Modulation)")
        depth_decoder = networks.DepthDecoder(
            encoder.num_ch_enc,
            opt.scales,
            num_output_channels=3,
            use_skips=True,
            PixelCoorModu=not getattr(opt, 'disable_pixel_coordinate_modulation', False)
        )
    
    scalenet = networks.ScaleNetwork(encoder.num_ch_enc)
    regression_heads = nn.ModuleList([
        networks.ProbabilisticScaleRegressionHead(in_channels=ch) 
        for ch in encoder.num_ch_enc
    ])
    
    # 7. 加载模型权重
    print("-> Loading encoder weights...")
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    
    print("-> Loading decoder weights...")
    model_dict = depth_decoder.state_dict()
    decoder_dict = torch.load(decoder_path, weights_only=False)
    
    # 过滤并加载匹配的键
    pretrained_dict = {k: v for k, v in decoder_dict.items() if k in model_dict}
    missing_keys = set(model_dict.keys()) - set(pretrained_dict.keys())
    unexpected_keys = set(decoder_dict.keys()) - set(model_dict.keys())
    
    if missing_keys:
        print(f"-> Warning: Missing keys in decoder weights: {list(missing_keys)[:5]}... (showing first 5)")
    if unexpected_keys:
        print(f"-> Warning: Unexpected keys in decoder weights: {list(unexpected_keys)[:5]}... (showing first 5)")
    
    depth_decoder.load_state_dict(pretrained_dict, strict=False)
    
    print("-> Loading scalenet weights...")
    scalenet.load_state_dict(torch.load(scalenet_path, weights_only=False))
    
    print("-> Loading regression heads weights...")
    regression_heads.load_state_dict(torch.load(regression_path, weights_only=False))
    
    # 8. 移动到GPU并设置为评估模式
    encoder.cuda().eval()
    depth_decoder.cuda().eval()
    scalenet.cuda().eval()
    regression_heads.cuda().eval()
    
    print("-> Model loaded successfully")
    
    # 9. 评估参数设置
    use_least_squares = getattr(opt, 'use_least_squares', True)
    if use_least_squares:
        print("-> Using Least Squares alignment (Scale + Shift)")
    else:
        print("-> Using Median scaling alignment")
    
    errors = []
    global_depth = []  # 用于存储每张图像的 global_depth
    
    print("-> Starting Evaluation...")
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # 获取输入数据
            input_color = data[("color", 0, 0)].cuda()
            gt_depth = data["depth_gt"][:, 0].numpy()  # (B, H, W)
            
            # 模型推理
            features = encoder(input_color)
            depth_factors = scalenet(features)
            scale_predictions = [head(factor) for head, factor in zip(regression_heads, depth_factors)]
            max_depth = torch.mean(torch.stack(scale_predictions), dim=0)  # (B,)
            
            # 解码器推理
            if use_diffusion_decoder:
                # Diffusion 解码器不需要 norm_pix_coords
                output = depth_decoder(features, gt=None, mask=None)
            else:
                # 常规解码器需要 norm_pix_coords
                norm_pix_coords = [data[("norm_pix_coords", s)].cuda() for s in opt.scales]
                output = depth_decoder(features, norm_pix_coords)
            
            # 获取 disparity 并转换为深度
            disp = output[("disp", 0)]  # (B, 1, H, W) 或 (B, H, W)
            
            # 确保 disp 是 4D tensor (B, 1, H, W)
            if disp.ndim == 3:
                disp = disp.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
            
            # 处理每个batch中的图像
            all_pred_depths = []
            max_depth_batch = []
            for j in range(max_depth.size(0)):
                # 获取当前图像的 disp 和 global_depth
                disp_j = disp[j:j+1]  # (1, 1, H, W)
                max_depth_j = max_depth[j].item()
                max_depth_batch.append(max_depth_j)
                
                # 转换为深度
                _, pred_depth_j = disp_to_depth(disp_j, opt.min_depth, max_depth_j)
                
                # disp_to_depth 返回的 depth 形状是 (1, 1, H, W) 或 (1, H, W)
                # 确保是 (1, H, W)
                if pred_depth_j.ndim == 4:
                    pred_depth_j = pred_depth_j[:, 0]  # (1, H, W)
                elif pred_depth_j.ndim == 2:
                    pred_depth_j = pred_depth_j.unsqueeze(0)  # (1, H, W)
                
                all_pred_depths.append(pred_depth_j)
            
            # 合并所有预测深度
            pred_depth_batch = torch.cat(all_pred_depths, dim=0)  # (B, H, W)
            
            # 转换为numpy并处理
            pred_depth_batch = pred_depth_batch.cpu().numpy()
            max_depth_batch = np.array(max_depth_batch)
            
            # 处理每张图像
            for j in range(pred_depth_batch.shape[0]):
                pred_depth = pred_depth_batch[j]  # (H, W)
                gt_depth_j = gt_depth[j]  # (H, W)
                max_depth_j = max_depth_batch[j]
                global_depth.append(max_depth_j)
                
                # Resize 回 GT 尺寸（如果需要）
                H_gt, W_gt = gt_depth_j.shape
                H_pred, W_pred = pred_depth.shape
                
                if H_pred != H_gt or W_pred != W_gt:
                    pred_depth = cv2.resize(pred_depth, (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)
                
                # 生成有效区域 Mask（与 evaluate_da3mono_nyu_depth.py 一致）
                MIN_DEPTH = 1e-3
                mask = np.logical_and(gt_depth_j > MIN_DEPTH, gt_depth_j < max_depth_j)
                
                # Eigen Crop (NYU 标准裁剪)
                crop = np.array([45, 471, 41, 601]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)
                
                pred_valid = pred_depth[mask]
                gt_valid = gt_depth_j[mask]
                
                if len(gt_valid) == 0:
                    continue
                
                # 对齐策略（与 evaluate_da3mono_nyu_depth.py 一致）
                if use_least_squares:
                    # Least Squares 对齐 (Scale + Shift)
                    coeffs = np.linalg.lstsq(
                        np.stack([pred_valid, np.ones_like(pred_valid)], axis=1), 
                        gt_valid, rcond=None
                    )[0]
                    scale, shift = coeffs
                    pred_final = pred_valid * scale + shift
                else:
                    # Median Scaling
                    ratio = np.median(gt_valid) / np.median(pred_valid)
                    pred_final = pred_valid * ratio
                
                # 截断预测值防止溢出
                pred_final = np.clip(pred_final, MIN_DEPTH, max_depth_j)
                
                # 计算误差
                errors.append(compute_errors(gt_valid, pred_final))
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i+1} batches")
    
    # 合并所有结果
    global_depth = np.array(global_depth)
    
    print(f"\n-> Total images: {len(errors)}")
    print(f"-> Global depth stats: min={global_depth.min():.3f}, max={global_depth.max():.3f}, mean={global_depth.mean():.3f}")
    
    # 汇总结果
    mean_errors = np.array(errors).mean(0)
    print("\n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()))


if __name__ == "__main__":
    options = MonodepthOptions()
    opt = options.parse()
    evaluate_student(opt)

