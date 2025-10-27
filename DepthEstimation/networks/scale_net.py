import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.lib import NONLocalBlock2D

class ProbabilisticScaleRegressionHead(nn.Module):
    def __init__(self, in_channels, max_scale=10, min_scale=1e-2):
        super(ProbabilisticScaleRegressionHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_scale = nn.Conv2d(128, max_scale + 1, kernel_size=1, stride=1, padding=0)
        self.max_scale = max_scale
        self.min_scale = min_scale

    def forward(self, x):
        x = self.relu(self.conv1(x))
        scale_logits = self.conv_scale(x)  # shape: (batch_size, max_scale+1, H, W)
        
        # Global average pooling to reduce (H, W) to (1, 1)
        scale_logits = F.adaptive_avg_pool2d(scale_logits, (1, 1)).squeeze(-1).squeeze(-1)
        
        # Softmax to get probabilities
        scale_probs = F.softmax(scale_logits, dim=1)  # shape: (batch_size, max_scale+1)
        
        # Compute the expected scale
        scales = torch.arange(0, self.max_scale + 1, device=x.device, dtype=x.dtype)
        expected_scale = torch.sum(scales * scale_probs, dim=1)  # shape: (batch_size,)
        
        # Ensure the scale is at least min_scale
        expected_scale = torch.clamp(expected_scale, min=self.min_scale)
        
        return expected_scale
    
class ScaleNetwork(nn.Module):
    def __init__(self, in_channels_list):
        super(ScaleNetwork, self).__init__()
        self.non_local_blocks = nn.ModuleList([NONLocalBlock2D(in_channels) for in_channels in in_channels_list])
        
    def forward(self, features):
        depth_factors = [non_local_block(feature) for non_local_block, feature in zip(self.non_local_blocks, features)]
        return depth_factors

