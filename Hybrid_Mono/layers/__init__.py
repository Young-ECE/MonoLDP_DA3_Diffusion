"""Layers package for depth estimation.

This package provides all layers, transformations, losses, and metrics used
in monocular depth estimation. It is organized into submodules for clarity:
- network_layers: PyTorch layer classes
- geometry: 3D geometric transformations
- transforms: Depth/disparity conversions
- losses: Loss functions
- metrics: Evaluation metrics
"""

from __future__ import absolute_import, division, print_function

# Network layers
from .network_layers import (
    ConvBlock,
    Conv3x3,
    BackprojectDepth,
    Project3D,
    SSIM,
    ConvBlock2,
    upsample,
)

# Geometric transformations
from .geometry import (
    transformation_from_parameters,
    get_translation_matrix,
    rot_from_axisangle,
)

# Depth/disparity transforms
from .transforms import (
    disp_to_depth,
    coeff_to_normal,
)

# Loss functions
from .losses import (
    get_smooth_loss,
    get_plane_loss,
    get_line_loss,
)

# Metrics
from .metrics import (
    compute_depth_errors,
)

__all__ = [
    # Network layers
    'ConvBlock',
    'Conv3x3',
    'BackprojectDepth',
    'Project3D',
    'SSIM',
    'ConvBlock2',
    'upsample',
    # Geometry
    'transformation_from_parameters',
    'get_translation_matrix',
    'rot_from_axisangle',
    # Transforms
    'disp_to_depth',
    'coeff_to_normal',
    # Losses
    'get_smooth_loss',
    'get_plane_loss',
    'get_line_loss',
    # Metrics
    'compute_depth_errors',
]

