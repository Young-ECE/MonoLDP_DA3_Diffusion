from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils import readlines
from options import MonodepthOptions
import datasets
import networks


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""
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


def evaluate_da3(opt):
    """Evaluates Depth Anything V3 model on NYU v2 dataset"""
    MIN_DEPTH = 1e-2
    MAX_DEPTH = 10.0

    print("=" * 60)
    print("Evaluating Depth Anything V3 on NYU v2 Dataset")
    print("=" * 60)

    # Load test file list
    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    print(f"-> Found {len(filenames)} test images")

    # Create dataset
    dataset = datasets.NYUDataset(
        opt.data_path, 
        filenames, 
        opt.height, 
        opt.width,
        [0], 
        1, 
        is_test=True, 
        return_plane=False, 
        num_plane_keysets=0,
        return_line=False, 
        num_line_keysets=0
    )

    dataloader = DataLoader(
        dataset, 
        opt.batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers,
        pin_memory=True, 
        drop_last=False
    )

    # Load Depth Anything V3 model
    print(f"\n-> Loading Depth Anything V3 model: {opt.depth_anything_v3_model}")
    da3_model = networks.create_depth_anything_v3_teacher(
        model_name=opt.depth_anything_v3_model,
        device="cuda" if torch.cuda.is_available() and not opt.no_cuda else "cpu",
        scales=[0],  # Only need scale 0 for evaluation
        input_size=(opt.height, opt.width),
        model_path=opt.depth_anything_v3_weights,
        cache_dir=getattr(opt, 'depth_anything_v3_cache_dir', None)
    )
    da3_model.eval()

    print(f"-> Computing predictions with size {opt.width}x{opt.height}")

    gt_depths = []
    pred_depths = []

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            input_color = data[("color", 0, 0)].cuda() if torch.cuda.is_available() and not opt.no_cuda else data[("color", 0, 0)]
            
            # Get ground truth depth
            gt_depth = data["depth_gt"][:, 0].numpy()
            gt_depths.append(gt_depth)

            # Run DA3 inference to get raw depth
            # Use the wrapper's _infer_depth method which returns actual depth values
            # Normalize input for DA3 internal model (ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(input_color.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(input_color.device)
            input_normalized = (input_color - mean) / std
            
            # Get raw depth from DA3 model
            # The _infer_depth method handles all the preprocessing and returns depth in (B, 1, H, W) format
            depth = da3_model._infer_depth(input_normalized)
            
            # Convert to numpy: (B, 1, H, W) -> (B, H, W)
            depth_np = depth.cpu().numpy()[:, 0]
            
            # DA3 outputs relative depth (not absolute), so we need to scale it
            # The depth values are typically in a normalized range
            # We'll let median scaling handle the absolute scale during evaluation
            # But we should ensure depth values are positive and in reasonable range
            
            # Clip negative values and very small values
            depth_np = np.maximum(depth_np, 1e-6)
            
            # Optional: Print depth statistics for first batch (for debugging)
            if batch_idx == 0:
                print(f"  Sample depth range: [{depth_np.min():.4f}, {depth_np.max():.4f}]")
                print(f"  Sample depth mean: {depth_np.mean():.4f}, median: {np.median(depth_np):.4f}")

            pred_depths.append(depth_np)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")

    # Concatenate all predictions
    gt_depths = np.concatenate(gt_depths)
    pred_depths = np.concatenate(pred_depths)

    print(f"\n-> Evaluating {pred_depths.shape[0]} images")
    print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_depths.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_depth = pred_depths[i]
        
        # Resize prediction to match ground truth size
        if pred_depth.shape != (gt_height, gt_width):
            pred_depth = cv2.resize(pred_depth, (gt_width, gt_height), interpolation=cv2.INTER_LINEAR)

        # Create mask for valid depth values
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        
        # Apply crop mask (NYU standard crop)
        crop_mask = np.zeros(mask.shape, dtype=np.uint8)
        crop_mask[dataset.default_crop[2]:dataset.default_crop[3], 
                  dataset.default_crop[0]:dataset.default_crop[1]] = 1
        mask = np.logical_and(mask, crop_mask.astype(bool))

        # Extract masked depths
        mask_pred_depth = pred_depth[mask]
        mask_gt_depth = gt_depth[mask]

        if len(mask_pred_depth) == 0:
            print(f"Warning: No valid pixels for image {i}")
            continue

        # Apply scale factor if specified
        mask_pred_depth *= opt.pred_depth_scale_factor

        # Apply median scaling if enabled
        if not opt.disable_median_scaling:
            # Avoid division by zero
            pred_median = np.median(mask_pred_depth)
            if pred_median > 1e-6:
                ratio = np.median(mask_gt_depth) / pred_median
            else:
                ratio = 1.0
                print(f"Warning: Very small predicted depth median for image {i}, using ratio=1.0")
            ratios.append(ratio)
            mask_pred_depth *= ratio
        else:
            ratio = 1.0
            ratios.append(ratio)

        # Clip depth values
        mask_pred_depth[mask_pred_depth < MIN_DEPTH] = MIN_DEPTH
        mask_pred_depth[mask_pred_depth > MAX_DEPTH] = MAX_DEPTH

        # Compute errors
        errors.append(compute_errors(mask_gt_depth, mask_pred_depth))

    if len(errors) == 0:
        print("Error: No valid predictions to evaluate!")
        return

    # Compute mean errors
    mean_errors = np.array(errors).mean(0)

    # Save results
    result_dir = getattr(opt, 'eval_out_dir', None) or os.path.dirname(__file__)
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"da3_result_{opt.eval_split}_split.txt")
    
    with open(result_path, 'w') as f:
        if not opt.disable_median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)
            std_ratio = np.std(ratios / med)
            print(f"\n Scaling ratios | med: {med:.3f} | std: {std_ratio:.3f}")
            print(f" Scaling ratios | med: {med:.3f} | std: {std_ratio:.3f}", file=f)

        print("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")

        print("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "a1", "a2", "a3"), file=f)
        print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\", file=f)

        print(f"\n-> Results saved to {result_path}")
        print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    opt = options.parse()
    
    # Set default DA3 model if not specified
    if not hasattr(opt, 'depth_anything_v3_model') or opt.depth_anything_v3_model is None:
        opt.depth_anything_v3_model = "DA3Mono-Large"
    
    # Set default eval_split if not specified
    if not hasattr(opt, 'eval_split') or opt.eval_split is None:
        opt.eval_split = "nyu"
    
    # Validate required parameters
    if not hasattr(opt, 'data_path') or opt.data_path is None:
        raise ValueError("--data_path is required. Please specify the path to NYU v2 dataset.")
    
    print("\n" + "=" * 60)
    print("Evaluation Configuration:")
    print("=" * 60)
    print(f"  Dataset path: {opt.data_path}")
    print(f"  Eval split: {opt.eval_split}")
    print(f"  DA3 Model: {opt.depth_anything_v3_model}")
    if hasattr(opt, 'depth_anything_v3_weights') and opt.depth_anything_v3_weights:
        print(f"  DA3 Weights: {opt.depth_anything_v3_weights}")
    print(f"  Input size: {opt.width}x{opt.height}")
    print(f"  Batch size: {opt.batch_size}")
    print(f"  Num workers: {opt.num_workers}")
    print(f"  Median scaling: {not opt.disable_median_scaling}")
    print(f"  Depth scale factor: {opt.pred_depth_scale_factor}")
    print("=" * 60 + "\n")
    
    evaluate_da3(opt)

