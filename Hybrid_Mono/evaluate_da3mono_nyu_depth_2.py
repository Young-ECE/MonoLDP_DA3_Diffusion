"""
修正版评估脚本：适配 Depth Anything V3 (DA3) 范式
"""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from options import MonodepthOptions
from utils import readlines
import datasets

# 尝试导入 DA3
try:
    from depth_anything_3.api import DepthAnything3
    DA3_AVAILABLE = True
except ImportError:
    DA3_AVAILABLE = False
    import warnings
    warnings.warn("depth_anything_3 package not found. Please install it to use DA3Mono-Large model.")

def compute_errors(gt, pred):
    """标准的深度误差计算"""
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

def least_squares_alignment(pred, gt, valid_mask):
    """
    最小二乘法对齐 (Scale + Shift)，用于评估 Relative Depth 模型
    pred = s * gt + t
    """
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    
    # 构建线性方程组 Ax = b
    # A = [[pred_val, 1], ...]
    # x = [scale, shift]
    # b = [gt_val, ...]
    
    if len(pred_valid) < 10:
        return pred, 1.0, 0.0

    X = np.stack([pred_valid, np.ones_like(pred_valid)], axis=1)
    Y = gt_valid
    
    # 使用 numpy 的最小二乘解
    res = np.linalg.lstsq(X, Y, rcond=None)
    scale, shift = res[0]
    
    # 应用对齐
    pred_aligned = pred * scale + shift
    
    # 确保深度为正
    pred_aligned = np.clip(pred_aligned, 1e-3, 10)
    
    return pred_aligned, scale, shift

def evaluate_da3(opt):
    # 1. 强制设置高分辨率 (DA3 标准)
    INPUT_SIZE = 518 
    print(f"-> 强制使用 DA3 标准输入分辨率: {INPUT_SIZE}x{INPUT_SIZE}")

    # 加载数据 (保持原图尺寸读取，我们在推理时再 Resize)
    filenames = readlines(os.path.join(os.path.dirname(__file__), "splits", opt.eval_split, "test_files.txt"))
    dataset = datasets.NYUDataset(opt.data_path, filenames, opt.height, opt.width, [0], 1, is_test=True)
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

    # 加载模型
    print("-> Loading DA3 model...")
    # 自动判断是 Metric 还是 Mono
    is_metric_model = "metric" in (opt.da3_model_path or "").lower()
    
    if opt.da3_model_path:
        model = DepthAnything3.from_pretrained(opt.da3_model_path).cuda().eval()
    else:
        # 首先尝试从本地缓存加载模型
        cached_model_path = None
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
        
        # 设置 HuggingFace 镜像（如果需要）
        hf_endpoint = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
        os.environ['HF_ENDPOINT'] = hf_endpoint
        
        # 优先使用本地缓存，否则从 HuggingFace 加载
        if cached_model_path:
            print("-> Loading from HuggingFace cache")
            model = DepthAnything3.from_pretrained(cached_model_path).cuda().eval()
        else:
            # 默认加载 DA3MONO-LARGE（本地已下载的模型）
            print("-> Loading DA3MONO-LARGE from HuggingFace (will use cache if available)")
            model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE").cuda().eval()
        is_metric_model = False  # DA3MONO-LARGE 是 Mono 模型

    print(f"-> Model Type: {'Metric (Abs Depth)' if is_metric_model else 'Mono (Relative Depth)'}")

    errors = []
    
    print("-> Starting Evaluation...")
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # 获取原始 RGB (未归一化，范围通常是 [0, 1] 或 [0, 255]，需确认 Dataset 输出)
            input_color = data[("color", 0, 0)].cuda() 
            gt_depth = data["depth_gt"][:, 0].numpy() # (B, H, W)

            # --- 关键修改 1: 预处理 ---
            # DA3 需要 ImageNet mean/std 归一化
            # 假设 input_color 是 [0, 1]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
            input_norm = (input_color - mean) / std

            # --- 关键修改 2: Resize 到 518x518 进行推理 ---
            input_da3_resized = F.interpolate(input_norm, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False)
            
            # DA3 forward 需要 (B, N, 3, H, W) 格式，N=1 表示单视图
            input_da3 = input_da3_resized.unsqueeze(1)  # (B, 1, 3, H, W)
            
            # 推理 - 使用 forward 方法并传入必要参数
            output = model.forward(
                input_da3,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=[],
                infer_gs=False
            )
            
            # 处理输出格式 (可能是 dict 或 tensor)
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
            
            # 确保深度形状正确 (B, H, W)
            if pred_depth_raw.ndim == 4:
                if pred_depth_raw.shape[1] == 1:
                    pred_depth_raw = pred_depth_raw.squeeze(1)  # (B, H, W)
                else:
                    pred_depth_raw = pred_depth_raw[:, 0]  # 取第一个视图
            elif pred_depth_raw.ndim == 5:
                pred_depth_raw = pred_depth_raw[:, 0, 0]  # (B, N, C, H, W) -> (B, H, W)
            
            # --- 关键修改 3: Resize 回 GT 尺寸 ---
            # DA3 输出通常与输入一致 (518)，需要插值回原始 GT 尺寸
            H_gt, W_gt = gt_depth.shape[1], gt_depth.shape[2]
            if pred_depth_raw.shape[1:] != (H_gt, W_gt):
                pred_depth = F.interpolate(
                    pred_depth_raw.unsqueeze(1),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            else:
                pred_depth = pred_depth_raw
            
            pred_depth = pred_depth[0].cpu().numpy()  # 取第一个batch，转为numpy
            gt_depth = gt_depth[0]

            # --- 关键修改 4: 标准 Mask 生成 (NYU 标准) ---
            # 有效深度范围
            mask = np.logical_and(gt_depth > 1e-3, gt_depth < 10)
            
            # Eigen Crop (NYU 标准裁剪)
            crop = np.array([45, 471, 41, 601]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            pred_valid = pred_depth[mask]
            gt_valid = gt_depth[mask]

            if len(gt_valid) == 0:
                continue

            # --- 关键修改 5: 对齐策略 ---
            if is_metric_model:
                # Metric 模型不需要对齐，直接评估
                pred_final = pred_valid
            else:
                # Mono (Relative) 模型：推荐使用 Least Squares 
                # 如果必须用 Median，请将下方改为 pred_valid *= np.median(gt_valid) / np.median(pred_valid)
                if getattr(opt, 'use_least_squares', True):
                    # 论文中 relative model 刷榜通常用这个
                    # np.linalg.lstsq 返回元组，res[0] 是解向量（包含 scale 和 shift）
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
            pred_final = np.clip(pred_final, 1e-3, 10)

            # 计算误差
            errors.append(compute_errors(gt_valid, pred_final))

            if (i + 1) % 50 == 0:
                print(f"Processed {i+1} images")

    # 汇总结果
    mean_errors = np.array(errors).mean(0)
    print("\n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()))

if __name__ == "__main__":
    options = MonodepthOptions()
    opt = options.parse()
    evaluate_da3(opt)