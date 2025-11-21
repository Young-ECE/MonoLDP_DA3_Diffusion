"""Depth and disparity transformation functions.

This module provides functions for converting between different depth representations,
including disparity to depth conversion and coefficient to normal vector conversion.
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's disparity output into depth prediction.
    
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    
    Args:
        disp: Disparity tensor (normalized, typically in [0, 1])
        min_depth: Minimum depth value (in meters)
        max_depth: Maximum depth value (in meters)
    
    Returns:
        tuple: (scaled_disp, depth)
            - scaled_disp: Scaled disparity
            - depth: Depth prediction (in meters)
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def coeff_to_normal(coeff):
    """Convert network's coefficient output into surface normal.
    
    Args:
        coeff: Coefficient tensor (typically from depth decoder)
    
    Returns:
        normal: Surface normal tensor (normalized)
    """
    normal = -F.normalize(coeff, p=2, dim=1)
    return normal

