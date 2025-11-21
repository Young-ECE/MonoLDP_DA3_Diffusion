"""Common utilities and base classes for network modules.

This module provides shared components used across different network architectures,
including basic convolution blocks and utility functions.
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Basic convolution block with batch normalization and ReLU activation.
    
    Used in depth decoders for upsampling operations.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv3x3(nn.Module):
    """Simple 3x3 convolution layer without activation.
    
    Used for final output layers in decoders.
    """
    def __init__(self, in_channels, out_channels):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)

