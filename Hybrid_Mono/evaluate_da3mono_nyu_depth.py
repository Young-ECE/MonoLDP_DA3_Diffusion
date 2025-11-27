"""
评估脚本：使用 DA3Mono-Large 模型在 NYUv2 数据集上进行评估

本脚本严格遵循 MonoLDP/DepthEstimation/evaluate_nyu_depth.py 的评估逻辑，
确保评估结果的公平性和可比性。

关键修改点：
1. 使用 DA3Mono-Large 模型替代 MonoLDP 的 encoder+decoder+scalenet+regression_heads
2. DA3Mono-Large 直接输出深度，需要转换为视差以适配评估流程
3. 模拟 global_depth（使用固定 MAX_DEPTH 或从预测深度中估计）
4. 保持所有评估指标计算、mask 生成、缩放等逻辑完全一致
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import cv2
import numpy as np
import warnings

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import readlines
from options import MonodepthOptions
import datasets
import networks

# 添加 Depth Anything 3 的导入
try:
    from depth_anything_3.api import DepthAnything3
    DA3_AVAILABLE = True
except ImportError:
    DA3_AVAILABLE = False
    warnings.warn("depth_anything_3 package not found. Please install it to use DA3Mono-Large model.")

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# 全局模型缓存（避免重复加载）
_model_cache = {}


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    
    与 MonoLDP 完全一致的评估指标计算函数
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    log10 = np.mean(np.abs(np.log10(pred / gt)))

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    
    与 MonoLDP 完全一致的后处理函数
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def depth_to_disparity(depth, min_depth=0.1, max_depth=10.0):
    """
    将深度转换为视差，用于适配 MonoLDP 的评估流程
    
    注意：这是为了适配评估流程的转换，DA3Mono-Large 直接输出深度，
    但评估流程期望视差输入，然后通过 pred_depth = 1 / pred_disp 转换回深度。
    
    为了保持一致性，我们使用与 MonoLDP 相同的转换公式（反向）：
    - MonoLDP: disp -> depth: depth = 1 / (min_disp + (max_disp - min_disp) * disp)
    - 反向: depth -> disp: 需要确保转换后的 disp 在评估时能正确转换回原始深度范围
    
    Args:
        depth: 深度图 (B, H, W) 或 (H, W)，单位：米
        min_depth: 最小深度值
        max_depth: 最大深度值
    
    Returns:
        disparity: 视差图，形状与 depth 相同
    """
    # 确保深度在有效范围内
    depth = np.clip(depth, min_depth, max_depth)
    
    # 转换为视差：disp = 1 / depth
    # 注意：这里直接使用 1/depth，因为评估时会通过 pred_depth = 1 / pred_disp 转换回来
    disp = 1.0 / depth
    
    return disp


def estimate_global_depth_from_prediction(pred_depth, percentile=95.0):
    """
    从预测深度中估计每张图像的 global_depth（最大深度）
    
    由于 DA3Mono-Large 没有 scalenet 和 regression_heads 来预测 global_depth，
    我们需要从预测深度中估计。使用百分位数方法，避免异常值影响。
    
    Args:
        pred_depth: 预测深度图 (H, W)
        percentile: 用于估计最大深度的百分位数（默认 95%）
    
    Returns:
        estimated_max_depth: 估计的最大深度值
    """
    valid_depth = pred_depth[pred_depth > 0.01]  # 过滤掉过小的深度值
    if len(valid_depth) == 0:
        return 10.0  # 默认最大深度
    
    estimated_max_depth = np.percentile(valid_depth, percentile)
    # 确保不超过合理的最大深度
    estimated_max_depth = min(estimated_max_depth, 10.0)
    estimated_max_depth = max(estimated_max_depth, 0.5)  # 至少 0.5 米
    
    return estimated_max_depth


def evaluate(opt):
    """Evaluates DA3Mono-Large model using NYUv2 test set
    
    严格遵循 MonoLDP/DepthEstimation/evaluate_nyu_depth.py 的评估逻辑
    """
    MIN_DEPTH = 1e-2
    MAX_DEPTH = 10
    
    if not DA3_AVAILABLE:
        raise RuntimeError("depth_anything_3 package is required. Please install it first.")
    
    # ========================================================================
    # 模型加载部分（修改点 1：使用 DA3Mono-Large 替代 MonoLDP 模型）
    # ========================================================================
    print("=" * 80)
    print("WARNING: 使用 DA3Mono-Large 模型进行评估")
    print("=" * 80)
    print("⚠️  关键差异点：")
    print("   1. DA3Mono-Large 直接输出深度（depth），而非视差（disparity）")
    print("   2. DA3Mono-Large 没有 scalenet 和 regression_heads，无法预测 global_depth")
    print("   3. 将使用固定 MAX_DEPTH 或从预测中估计 global_depth")
    print("   4. 输入预处理可能略有不同（ImageNet normalization）")
    print("=" * 80)
    
    # 加载测试文件列表
    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    
    # 确定输入尺寸（使用 opt 中的尺寸，如果没有则使用默认值）
    eval_height = opt.height if hasattr(opt, 'height') else 256
    eval_width = opt.width if hasattr(opt, 'width') else 320
    
    print(f"-> Using input size: {eval_width}x{eval_height}")
    
    # 创建数据集（与 MonoLDP 完全一致）
    dataset = datasets.NYUDataset(
        opt.data_path, 
        filenames, 
        eval_height, 
        eval_width,
        [0], 
        1, 
        is_test=True, 
        return_plane=True, 
        num_plane_keysets=0,
        return_line=True, 
        num_line_keysets=0
    )
    
    # 注意：在测试模式下，数据集会保留 ("color", i, -1) 作为 PIL Image
    # 但在 Hybrid_Mono 的 mono_dataset.py 中，_cleanup_native_resolution 会将其转换为 tensor
    # 已修复：在 _cleanup_native_resolution 中直接使用 transforms.ToTensor() 转换，避免序列化问题
    # 现在可以使用正常的 num_workers 设置
    dataloader = DataLoader(
        dataset, 
        opt.batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers,
        pin_memory=True, 
        drop_last=False
    )
    
    # 加载 DA3Mono-Large 模型（带缓存机制）
    print("-> Loading DA3Mono-Large model...")
    
    # 生成缓存键
    if hasattr(opt, 'da3_model_path') and opt.da3_model_path:
        cache_key = f"local_{opt.da3_model_path}"
        model_path = opt.da3_model_path
        print(f"-> Loading from local path: {model_path}")
    else:
        cache_key = "hf_depth-anything/DA3MONO-LARGE"
        model_path = None
        print("-> Loading from HuggingFace (will use cache if available)")
    
    # 检查全局缓存
    if cache_key in _model_cache:
        print("-> Using cached model (from previous run in this session)")
        da3_model = _model_cache[cache_key]
    else:
        # 尝试从 HuggingFace cache 中查找模型
        cached_model_path = None
        if model_path is None:
            # 自动查找 HuggingFace cache
            hf_cache_dir = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
            hub_cache_dir = os.path.join(hf_cache_dir, 'hub')
            model_cache_name = "models--depth-anything--DA3MONO-LARGE"
            model_cache_path = os.path.join(hub_cache_dir, model_cache_name)
            
            if os.path.exists(model_cache_path):
                snapshots_path = os.path.join(model_cache_path, "snapshots")
                if os.path.exists(snapshots_path):
                    snapshots = [d for d in os.listdir(snapshots_path) 
                               if os.path.isdir(os.path.join(snapshots_path, d))]
                    if snapshots:
                        snapshot_dir = os.path.join(snapshots_path, snapshots[0])
                        if os.path.exists(os.path.join(snapshot_dir, "model.safetensors")):
                            cached_model_path = snapshot_dir
                            print(f"-> Found cached model at: {cached_model_path}")
        
        try:
            # 设置 HuggingFace 镜像（如果需要）
            hf_endpoint = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
            os.environ['HF_ENDPOINT'] = hf_endpoint
            
            # 加载模型（优先使用 cache 中的模型）
            if model_path and os.path.exists(model_path):
                da3_model = DepthAnything3.from_pretrained(model_path)
                print(f"-> Model loaded from local path: {model_path}")
            elif cached_model_path:
                da3_model = DepthAnything3.from_pretrained(cached_model_path)
                print(f"-> Model loaded from HuggingFace cache: {cached_model_path}")
            else:
                # 从 HuggingFace 加载（会自动使用 cache）
                da3_model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")
                print("-> Model loaded from HuggingFace (using cache if available)")
            
            da3_model = da3_model.cuda()
            da3_model.eval()
            
            # 缓存模型对象（避免下次重新加载）
            _model_cache[cache_key] = da3_model
            print("-> DA3Mono-Large model loaded successfully (cached for future runs in this session)")
        except Exception as e:
            raise RuntimeError(f"Failed to load DA3Mono-Large model: {e}")
    
    # ========================================================================
    # 推理部分（修改点 2：使用 DA3Mono-Large 进行推理）
    # ========================================================================
    gt_depths = []
    planes = []
    lines = []
    pred_disps = []
    global_depth = []  # 用于存储每张图像的 global_depth
    
    # 用于估计 global_depth 的方法
    use_fixed_max_depth = getattr(opt, 'use_fixed_max_depth', False)
    if use_fixed_max_depth:
        print(f"-> Using fixed MAX_DEPTH={MAX_DEPTH} for all images (no per-image global_depth)")
    else:
        print("-> Estimating global_depth from predictions (using 95th percentile)")
    
    print("-> Computing predictions with size {}x{}".format(eval_width, eval_height))
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            input_color = data[("color", 0, 0)].cuda()  # (B, 3, H, W)，范围 [0, 1]
            
            gt_depth = data["depth_gt"][:, 0].numpy()  # (B, H, W)
            gt_depths.append(gt_depth)
            
            plane = data[("plane", 0, -1)][:, 0].numpy()
            planes.append(plane)
            line = data[("line", 0, -1)][:, 0].numpy()
            lines.append(line)
            
            # ====================================================================
            # 修改点 3：DA3Mono-Large 的输入预处理
            # ====================================================================
            # DA3Mono-Large 需要 ImageNet normalization
            # input_color 已经是 [0, 1] 范围，需要转换为 ImageNet normalized
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
            input_normalized = (input_color - mean) / std
            
            # DA3Mono-Large 需要输入尺寸是 14 的倍数（patch size）
            B, C, H, W = input_normalized.shape
            patch_size = 14
            if H % patch_size != 0 or W % patch_size != 0:
                import math
                new_H = math.ceil(H / patch_size) * patch_size
                new_W = math.ceil(W / patch_size) * patch_size
                input_normalized = F.interpolate(
                    input_normalized, 
                    size=(new_H, new_W), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # DA3Mono-Large forward 需要 (B, N, 3, H, W) 格式，N=1 表示单视图
            input_da3 = input_normalized.unsqueeze(1)  # (B, 1, 3, H, W)
            
            # 运行推理
            try:
                output = da3_model.forward(
                    input_da3,
                    extrinsics=None,
                    intrinsics=None,
                    export_feat_layers=[],
                    infer_gs=False
                )
                
                # 提取深度输出
                if isinstance(output, dict):
                    pred_depth_raw = output.get('depth', None)
                    if pred_depth_raw is None:
                        # 尝试其他可能的键
                        for key in ['predicted_depth', 'pred', 'depth_map']:
                            if key in output:
                                pred_depth_raw = output[key]
                                break
                else:
                    pred_depth_raw = output
                
                if pred_depth_raw is None:
                    raise RuntimeError("Could not extract depth from DA3 output")
                
                # 确保深度形状正确 (B, H, W) 或 (B, 1, H, W)
                if pred_depth_raw.ndim == 4:
                    if pred_depth_raw.shape[1] == 1:
                        pred_depth_raw = pred_depth_raw.squeeze(1)  # (B, H, W)
                    else:
                        pred_depth_raw = pred_depth_raw[:, 0]  # 取第一个视图
                elif pred_depth_raw.ndim == 5:
                    pred_depth_raw = pred_depth_raw[:, 0, 0]  # (B, N, C, H, W) -> (B, H, W)
                
                # 如果尺寸被调整过，需要 resize 回原始尺寸
                if pred_depth_raw.shape[1:] != (H, W):
                    pred_depth_raw = F.interpolate(
                        pred_depth_raw.unsqueeze(1),
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                
            except Exception as e:
                print(f"Error during DA3 inference: {e}")
                raise
            
            # ====================================================================
            # 修改点 4：处理后处理和 global_depth
            # ====================================================================
            batch_pred_disps = []
            batch_global_depths = []
            
            for i in range(B):
                pred_depth_i = pred_depth_raw[i].cpu().numpy()  # (H, W)
                
                # 估计或使用固定的 global_depth
                if use_fixed_max_depth:
                    max_depth_i = MAX_DEPTH
                else:
                    max_depth_i = estimate_global_depth_from_prediction(pred_depth_i)
                batch_global_depths.append(max_depth_i)
                
                # 将深度转换为视差（用于适配评估流程）
                pred_disp_i = depth_to_disparity(pred_depth_i, opt.min_depth, max_depth_i)
                
                batch_pred_disps.append(pred_disp_i)
            
            # 处理后处理（如果需要）
            if opt.post_process:
                # 后处理需要左右翻转的预测
                # 对于 DA3Mono-Large，我们需要对输入进行翻转并重新推理
                input_flipped = torch.flip(input_normalized, [3])  # 水平翻转
                input_flipped_da3 = input_flipped.unsqueeze(1)
                
                # 重新推理翻转后的图像
                output_flipped = da3_model.forward(
                    input_flipped_da3,
                    extrinsics=None,
                    intrinsics=None,
                    export_feat_layers=[],
                    infer_gs=False
                )
                
                # 提取翻转后的深度
                if isinstance(output_flipped, dict):
                    pred_depth_flipped = output_flipped.get('depth', None)
                else:
                    pred_depth_flipped = output_flipped
                
                if pred_depth_flipped.ndim == 4:
                    if pred_depth_flipped.shape[1] == 1:
                        pred_depth_flipped = pred_depth_flipped.squeeze(1)
                    else:
                        pred_depth_flipped = pred_depth_flipped[:, 0]
                elif pred_depth_flipped.ndim == 5:
                    pred_depth_flipped = pred_depth_flipped[:, 0, 0]
                
                if pred_depth_flipped.shape[1:] != (H, W):
                    pred_depth_flipped = F.interpolate(
                        pred_depth_flipped.unsqueeze(1),
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                
                # 将翻转后的深度转换为视差
                batch_pred_disps_flipped = []
                for i in range(B):
                    pred_depth_flipped_i = pred_depth_flipped[i].cpu().numpy()
                    if use_fixed_max_depth:
                        max_depth_i = MAX_DEPTH
                    else:
                        max_depth_i = estimate_global_depth_from_prediction(pred_depth_flipped_i)
                    pred_disp_flipped_i = depth_to_disparity(pred_depth_flipped_i, opt.min_depth, max_depth_i)
                    batch_pred_disps_flipped.append(pred_disp_flipped_i)
                
                # 应用后处理
                l_disp = np.stack([batch_pred_disps[i] for i in range(B)], axis=0)
                r_disp = np.stack([batch_pred_disps_flipped[i] for i in range(B)], axis=0)
                r_disp_flipped = np.flip(r_disp, axis=2)  # 翻转回来
                
                for i in range(B):
                    processed_disp = batch_post_process_disparity(
                        l_disp[i:i+1], 
                        r_disp_flipped[i:i+1]
                    )
                    batch_pred_disps[i] = processed_disp[0]
            
            # 收集结果
            pred_disps.extend(batch_pred_disps)
            global_depth.extend(batch_global_depths)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # 合并所有结果
    gt_depths = np.concatenate(gt_depths)
    planes = np.concatenate(planes)
    lines = np.concatenate(lines)
    global_depth = np.array(global_depth)
    pred_disps = np.array(pred_disps)
    
    print(f"-> Total images: {len(pred_disps)}")
    print(f"-> Global depth stats: min={global_depth.min():.3f}, max={global_depth.max():.3f}, mean={global_depth.mean():.3f}")
    print(f"-> Pred disp stats: min={pred_disps.min():.6f}, max={pred_disps.max():.6f}, mean={pred_disps.mean():.6f}")
    print(f"-> Pred disp has inf: {np.any(np.isinf(pred_disps))}, has nan: {np.any(np.isnan(pred_disps))}")
    
    # ========================================================================
    # 评估部分（与 MonoLDP 完全一致）
    # ========================================================================
    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder if hasattr(opt, 'load_weights_folder') and opt.load_weights_folder else ".",
            "disps_da3mono_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, pred_disps)
    
    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        return
    
    print("-> Evaluating")
    print("   Mono evaluation - using median scaling")
    
    errors = []
    ratios = []
    
    norm_pix_coords = dataset.get_norm_pix_coords()
    
    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        
        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        # 避免除以 0 或非常小的值
        pred_disp = np.clip(pred_disp, 1e-6, np.inf)
        pred_depth = 1 / pred_disp  # 视差转回深度（与 MonoLDP 一致）
        
        # 使用 global_depth[i] 作为最大深度阈值（与 MonoLDP 一致）
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < global_depth[i])
        crop_mask = np.zeros(mask.shape)
        crop_mask[dataset.default_crop[2]:dataset.default_crop[3], 
                  dataset.default_crop[0]:dataset.default_crop[1]] = 1
        mask = np.logical_and(mask, crop_mask)
        mask_pred_depth = pred_depth[mask]
        mask_gt_depth = gt_depth[mask]
        
        # 检查 mask 是否为空
        if len(mask_pred_depth) == 0 or len(mask_gt_depth) == 0:
            print(f"Warning: Empty mask for image {i}, skipping...")
            continue
        
        mask_pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            median_gt = np.median(mask_gt_depth)
            median_pred = np.median(mask_pred_depth)
            if median_pred > 0:
                ratio = median_gt / median_pred
            else:
                print(f"Warning: Zero median for image {i}, using ratio=1")
                ratio = 1.0
            ratios.append(ratio)
            mask_pred_depth *= ratio
        else:
            ratio = 1
            ratios.append(ratio)
        
        mask_pred_depth[mask_pred_depth < MIN_DEPTH] = MIN_DEPTH
        mask_pred_depth[mask_pred_depth > global_depth[i]] = global_depth[i]
        
        errors.append(compute_errors(mask_gt_depth, mask_pred_depth))
    
    mean_errors = np.array(errors).mean(0)
    
    # 保存结果
    result_dir = opt.load_weights_folder if hasattr(opt, 'load_weights_folder') and opt.load_weights_folder else "."
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, "result_da3mono_{}_split.txt".format(opt.eval_split))
    f = open(result_path, 'w+')
    
    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)), file=f)
    
    print("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
    
    print("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "log10",  "a1", "a2", "a3"), file=f)
    print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\", file=f)
    
    print("\n-> Done!")
    print("\n-> Done!", file=f)
    f.close()
    
    # 打印警告信息
    print("\n" + "=" * 80)
    print("评估完成 - 重要警告信息")
    print("=" * 80)
    print_warnings(use_fixed_max_depth, global_depth)


def print_warnings(use_fixed_max_depth, global_depth):
    """打印评估过程中的警告信息"""
    print("\n⚠️  评估公平性警告：")
    print("-" * 80)
    
    print("\n1. 【模型架构差异】")
    print("   - MonoLDP: encoder + decoder + scalenet + regression_heads")
    print("   - DA3Mono-Large: 端到端深度估计模型")
    print("   → 影响：模型容量和表达能力不同，可能影响性能上限")
    
    print("\n2. 【Global Depth 处理】")
    if use_fixed_max_depth:
        print("   - 使用固定 MAX_DEPTH=10.0 作为所有图像的 global_depth")
        print("   - MonoLDP 使用每张图像预测的 global_depth")
        print("   → 影响：可能对某些图像不公平（过小或过大的场景）")
    else:
        print("   - 从预测深度中估计 global_depth（95th percentile）")
        print("   - MonoLDP 使用 scalenet + regression_heads 预测")
        print("   → 影响：估计方法不同，可能影响 mask 生成和深度裁剪")
        print(f"   - 统计：min={global_depth.min():.3f}, max={global_depth.max():.3f}, mean={global_depth.mean():.3f}")
    
    print("\n3. 【输入预处理】")
    print("   - DA3Mono-Large 使用 ImageNet normalization")
    print("   - MonoLDP 可能使用不同的归一化方式")
    print("   → 影响：可能影响模型性能，但这是模型本身的特性")
    
    print("\n4. 【深度到视差的转换】")
    print("   - DA3Mono-Large 输出深度，转换为视差后评估")
    print("   - 转换：disp = 1 / depth，评估时：pred_depth = 1 / pred_disp")
    print("   → 影响：理论上应该一致，但数值精度可能略有差异")
    
    print("\n5. 【输入尺寸处理】")
    print("   - DA3Mono-Large 要求输入尺寸是 14 的倍数（patch size）")
    print("   - 如果输入尺寸不满足，会自动 resize")
    print("   → 影响：可能引入轻微的插值误差")
    
    print("\n6. 【后处理】")
    print("   - 如果启用 post_process，DA3Mono-Large 需要两次前向传播")
    print("   - MonoLDP 在 batch 维度上处理，可能更高效")
    print("   → 影响：性能差异，但不影响评估公平性")
    
    print("\n✅ 保持一致的评估逻辑：")
    print("   - compute_errors 函数完全相同")
    print("   - mask 生成逻辑完全相同（MIN_DEPTH, global_depth, crop_mask）")
    print("   - median scaling 逻辑完全相同")
    print("   - 深度裁剪逻辑完全相同")
    print("   - 所有评估指标计算完全相同")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    options = MonodepthOptions()
    opt = options.parse()
    
    # 添加 DA3Mono-Large 特定参数
    if not hasattr(opt, 'da3_model_path'):
        opt.da3_model_path = None
    if not hasattr(opt, 'use_fixed_max_depth'):
        opt.use_fixed_max_depth = False
    
    # 如果没有指定 load_weights_folder，使用当前目录
    if not hasattr(opt, 'load_weights_folder') or not opt.load_weights_folder:
        opt.load_weights_folder = "."
    
    # 确保 eval_split 存在
    if not hasattr(opt, 'eval_split'):
        opt.eval_split = "nyu"
    
    evaluate(opt)

