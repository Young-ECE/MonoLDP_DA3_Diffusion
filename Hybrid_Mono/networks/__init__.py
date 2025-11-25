"""Network modules for monocular depth estimation.

This package provides all network architectures used in the depth estimation pipeline:
- Encoders: Feature extraction networks
- Decoders: Depth prediction networks
- Pose: Camera pose estimation networks
- Scale: Depth scale prediction networks
- Teacher: Teacher model wrappers
"""

from __future__ import absolute_import, division, print_function

# Encoders
from .encoders.resnet_encoder import ResnetEncoder

# Decoders
from .decoders.depth_decoder import DepthDecoder
from .decoders.depth_decoder_diffusion import DepthDecoderDiffusion

# Pose decoders
from .pose.pose_decoder import PoseDecoder, PoseDecoderRec, PoseDecoderThird

# Scale networks
from .scale.scale_net import ScaleNetwork, ProbabilisticScaleRegressionHead

# Teacher models
from .teacher.depth_anything_v3_wrapper import DepthAnythingV3Wrapper, create_depth_anything_v3_teacher

__all__ = [
    # Encoders
    'ResnetEncoder',
    # Decoders
    'DepthDecoder',
    'DepthDecoderDiffusion',
    # Pose
    'PoseDecoder',
    'PoseDecoderRec',
    'PoseDecoderThird',
    # Scale
    'ScaleNetwork',
    'ProbabilisticScaleRegressionHead',
    # Teacher
    'DepthAnythingV3Wrapper',
    'create_depth_anything_v3_teacher',
]
