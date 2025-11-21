"""Loss functions for depth estimation training.

This module provides various loss functions used in monocular depth estimation,
including smoothness loss and geometric consistency losses.
"""

from __future__ import absolute_import, division, print_function

import torch


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image.
    
    The color image is used for edge-aware smoothness.
    Default: delta 1, first order gradient.
    
    Args:
        disp: Disparity tensor (batch, channels, height, width)
        img: Color image tensor (batch, 3, height, width)
    
    Returns:
        smooth_loss: Scalar smoothness loss value
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def get_plane_loss(plane_keysets, points_3d):
    """Compute plane consistency loss.
    
    This loss enforces that 4 points selected by keysets lie on the same plane.
    The loss is computed as the absolute value of the scalar triple product,
    which should be zero if the 4 points are coplanar.
    
    Args:
        plane_keysets: Tensor of shape (batch_size, 4, num_keysets)
                      Invalid keysets are marked with -1
        points_3d: Tensor of shape (batch_size, 3, H*W)
    
    Returns:
        plane_loss: Scalar loss value
    """
    bs, ch, _ = points_3d.shape

    # Extract points for each keyset
    # plane_keysets shape: (batch_size, 4, num_keysets)
    # We need to handle invalid keysets (marked with -1)
    
    # Clamp negative indices to 0 for gather (will filter out invalid later)
    # Note: torch.gather with negative indices wraps around, so we need to be careful
    plane_keysets_clamped = torch.clamp(plane_keysets, min=0)
    
    start_points = torch.gather(points_3d, 2, torch.stack(ch * [plane_keysets_clamped[:, 0]], 1))
    end_points_A = torch.gather(points_3d, 2, torch.stack(ch * [plane_keysets_clamped[:, 1]], 1))
    end_points_B = torch.gather(points_3d, 2, torch.stack(ch * [plane_keysets_clamped[:, 2]], 1))
    end_points_C = torch.gather(points_3d, 2, torch.stack(ch * [plane_keysets_clamped[:, 3]], 1))

    vector_A = end_points_A - start_points
    vector_B = end_points_B - start_points
    vector_C = end_points_C - start_points

    AxB = torch.cross(vector_A, vector_B, dim=1)

    AxB_dot_C = torch.sum(AxB*vector_C, dim=1)

    # Filter out invalid keysets (where any index is -1)
    # Check if all 4 indices in each keyset are valid (>= 0)
    valid_mask = torch.all(plane_keysets >= 0, dim=1)  # (batch_size, num_keysets)
    
    if torch.any(valid_mask):
        # Only compute loss for valid keysets
        valid_losses = torch.abs(AxB_dot_C) * valid_mask.float()
        # Average over valid keysets only
        plane_loss = valid_losses.sum() / (valid_mask.sum().float() + 1e-7)
    else:
        # No valid keysets, return zero loss
        plane_loss = torch.tensor(0.0, device=points_3d.device, dtype=points_3d.dtype)

    return plane_loss


def get_line_loss(line_keysets, points_3d):
    """Compute line consistency loss.
    
    This loss enforces that 3 points selected by keysets lie on the same line.
    The loss is computed as the norm of the cross product of two vectors,
    which should be zero if the 3 points are collinear.
    
    Args:
        line_keysets: Tensor of shape (batch_size, 3, num_keysets)
                     Invalid keysets are marked with -1
        points_3d: Tensor of shape (batch_size, 3, H*W)
    
    Returns:
        line_loss: Scalar loss value
    """
    bs, ch, _ = points_3d.shape

    # Extract points for each keyset
    # line_keysets shape: (batch_size, 3, num_keysets)
    # We need to handle invalid keysets (marked with -1)
    
    # Clamp negative indices to 0 for gather (will filter out invalid later)
    line_keysets_clamped = torch.clamp(line_keysets, min=0)
    
    start_points = torch.gather(points_3d, 2, torch.stack(ch * [line_keysets_clamped[:, 0]], 1))
    end_points_A = torch.gather(points_3d, 2, torch.stack(ch * [line_keysets_clamped[:, 1]], 1))
    end_points_B = torch.gather(points_3d, 2, torch.stack(ch * [line_keysets_clamped[:, 2]], 1))

    vector_A = end_points_A - start_points
    vector_B = end_points_B - start_points

    AxB = torch.cross(vector_A, vector_B, dim=1)

    line_norms = torch.norm(AxB, p=2, dim=1)  # (batch_size, num_keysets)

    # Filter out invalid keysets (where any index is -1)
    # Check if all 3 indices in each keyset are valid (>= 0)
    valid_mask = torch.all(line_keysets >= 0, dim=1)  # (batch_size, num_keysets)
    
    if torch.any(valid_mask):
        # Only compute loss for valid keysets
        valid_losses = line_norms * valid_mask.float()
        # Average over valid keysets only
        line_loss = valid_losses.sum() / (valid_mask.sum().float() + 1e-7)
    else:
        # No valid keysets, return zero loss
        line_loss = torch.tensor(0.0, device=points_3d.device, dtype=points_3d.dtype)

    return line_loss

