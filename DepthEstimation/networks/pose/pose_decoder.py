"""Pose decoder modules for camera pose estimation.

This module provides pose decoders for estimating relative camera poses between frames.
All variants share the same architecture but are kept separate for backward compatibility.
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class PoseDecoder(nn.Module):
    """Standard pose decoder for estimating relative camera poses.
    
    Predicts 6-DOF pose (3 rotation + 3 translation) between camera frames.
    
    Args:
        num_ch_enc: List of encoder channel numbers
        num_input_features: Number of input frames
        num_frames_to_predict_for: Number of poses to predict (default: num_input_features - 1)
        stride: Convolution stride (default: 1)
    """
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        """Forward pass.
        
        Args:
            input_features: List of feature maps from encoder
            
        Returns:
            axisangle: Rotation in axis-angle representation (batch, num_frames, 1, 3)
            translation: Translation vector (batch, num_frames, 1, 3)
        """
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation


class PoseDecoderRec(PoseDecoder):
    """Pose decoder for recursive pose estimation.
    
    This is identical to PoseDecoder but kept as a separate class for backward compatibility.
    """
    pass


class PoseDecoderThird(PoseDecoder):
    """Pose decoder for third-frame pose estimation.
    
    This is identical to PoseDecoder but kept as a separate class for backward compatibility.
    """
    pass

