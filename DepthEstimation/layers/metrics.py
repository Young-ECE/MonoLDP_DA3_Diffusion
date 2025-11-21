"""Evaluation metrics for depth estimation.

This module provides functions for computing depth estimation metrics,
including absolute relative error, RMSE, and accuracy metrics.
"""

from __future__ import absolute_import, division, print_function

import torch


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths.
    
    Computes multiple depth estimation metrics:
    - Absolute relative error (abs_rel)
    - Squared relative error (sq_rel)
    - Root mean squared error (rmse)
    - Root mean squared error in log space (rmse_log)
    - Mean log10 error (log10)
    - Accuracy metrics (a1, a2, a3) - percentage of pixels with error < threshold
    
    Args:
        gt: Ground truth depth tensor
        pred: Predicted depth tensor
    
    Returns:
        tuple: (abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3)
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    log10 = torch.mean(torch.abs(torch.log10(pred / gt)))

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, log10, a1, a2, a3

