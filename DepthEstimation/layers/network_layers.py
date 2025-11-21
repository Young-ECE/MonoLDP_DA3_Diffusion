"""Neural network layer modules.

This module provides custom PyTorch layers used in depth estimation networks,
including convolution blocks, 3D projection layers, and SSIM loss layer.
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import conv1x1


class Conv3x3(nn.Module):
    """Layer to pad and convolve input with 3x3 kernel.
    
    Uses reflection padding by default to preserve image boundaries.
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU activation.
    
    Used in depth decoders for upsampling operations.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud.
    
    Backprojects depth values using normalized pixel coordinates to generate
    3D points in camera coordinate system.
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

    def forward(self, depth, norm_pix_coords):
        """Backproject depth to 3D points.
        
        Args:
            depth: Depth tensor (batch, 1, height, width)
            norm_pix_coords: Normalized pixel coordinates (batch, 3, height, width)
        
        Returns:
            cam_points: 3D points in camera coordinates (batch, 4, height*width)
        """
        cam_points = depth * norm_pix_coords
        cam_points = torch.cat([cam_points.view(self.batch_size, cam_points.shape[1], -1), self.ones], 1)
        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T.
    
    Projects 3D points from one camera frame to another using camera intrinsics
    and relative pose transformation.
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        """Project 3D points to pixel coordinates.
        
        Args:
            points: 3D points (batch, 4, num_points)
            K: Camera intrinsics matrix (batch, 4, 4)
            T: Transformation matrix (batch, 4, 4)
        
        Returns:
            pix_coords: Pixel coordinates normalized to [-1, 1] (batch, height, width, 2)
        """
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords[:, 0, :, :] /= self.width - 1
        pix_coords[:, 1, :, :] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2

        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2 using nearest neighbor interpolation.
    
    Args:
        x: Input tensor (batch, channels, height, width)
    
    Returns:
        Upsampled tensor (batch, channels, 2*height, 2*width)
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class SSIM(nn.Module):
    """Layer to compute the SSIM (Structural Similarity Index) loss between a pair of images.
    
    SSIM measures the structural similarity between two images, considering
    luminance, contrast, and structure.
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        """Compute SSIM loss.
        
        Args:
            x: First image tensor
            y: Second image tensor
        
        Returns:
            SSIM loss (1 - SSIM) clamped to [0, 1]
        """
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class ConvBlock2(nn.Module):
    """Layer to perform 1x1 convolution with stride 2 followed by ReLU activation.
    
    Used for downsampling operations.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock2, self).__init__()

        self.conv = conv1x1(in_channels, out_channels, stride=2)
        self.nonlin = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

