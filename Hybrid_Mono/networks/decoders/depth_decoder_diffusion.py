from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from layers.network_layers import upsample
from timm.models.layers import trunc_normal_
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from typing import Union, Dict, Tuple, Optional
from ..common import Conv3x3


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class DepthDecoderDiffusion(nn.Module):
    """Depth decoder with diffusion-based refinement
    
    This follows MonoDiffusion's architecture:
    - Always decodes through scale 2 -> 1 -> 0
    - num_output_channels=1 (single-channel disparity/depth)
    - Does not use pixel coordinate modulation
    """
    def __init__(self, num_ch_enc, scales=range(3), num_output_channels=1, 
                 use_skips=True, diffusion_steps=None, diffusion_timesteps=None):
        """
        Args:
            num_ch_enc: encoder channel numbers
            scales: list of scales to output
            num_output_channels: number of output channels (1 for disparity)
            use_skips: whether to use skip connections
            diffusion_steps: list of inference steps for each scale [coarse to fine]
                           e.g., [5, 4, 3] for scales [2, 1, 0]
            diffusion_timesteps: list of training timesteps for each scale [coarse to fine]
                               e.g., [250, 200, 150] for scales [2, 1, 0]
        """
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = sorted(list(scales))

        self.num_ch_enc = np.array(num_ch_enc)
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int')

        # Decoder
        # Note: For ResnetEncoder with 5 features, we use skip connections:
        # - decoder i=2 uses encoder[3] (256 ch)
        # - decoder i=1 uses encoder[2] (128 ch)
        # - decoder i=0 has no skip
        self.convs = OrderedDict()
        for i in range(2, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                # decoder i uses encoder skip from index i+1
                skip_idx = i + 1
                if skip_idx < len(self.num_ch_enc):
                    num_ch_in += self.num_ch_enc[skip_idx]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        # Feature conditioning for diffusion
        sorted_scales = sorted(self.scales, reverse=True)
        coarsest_scale = sorted_scales[0]

        for s in self.scales:
            self.convs[("conconv", s)] = ConvBlock(self.num_ch_dec[s], 16)
            if s != coarsest_scale:
                self.convs[("disptocon", s)] = ConvBlock(self.num_output_channels, 16)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)
        
        ########## Diffusion modules ##########
        self._sorted_scales = sorted(self.scales, reverse=True)
        self.diffusion_models = nn.ModuleDict()
        self.schedulers = {}
        self.diffusion_pipelines = {}
        self.diffusion_inference_steps = {}
        self.scheduler_train_steps = {}
        
        # 存储配置的扩散步数
        self.diffusion_steps_config = diffusion_steps
        self.diffusion_timesteps_config = diffusion_timesteps

        for idx, scale in enumerate(self._sorted_scales):
            scale_key = str(scale)
            self.diffusion_models[scale_key] = ScheduledCNNRefine(
                channels_in=16,
                channels_noise=self.num_output_channels
            )
            self.scheduler_train_steps[scale_key] = self._get_scheduler_train_steps(scale)
            self.schedulers[scale_key] = DDIMScheduler(
                num_train_timesteps=self.scheduler_train_steps[scale_key],
                clip_sample=False
            )
            self.diffusion_pipelines[scale_key] = CNNDDIMPipeline(
                self.diffusion_models[scale_key],
                self.schedulers[scale_key]
            )
            self.diffusion_inference_steps[scale_key] = self._get_inference_steps(scale, idx)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _get_scheduler_train_steps(self, scale: int) -> int:
        """Get training timesteps for a given scale"""
        if self.diffusion_timesteps_config is not None:
            # 从配置中读取：scale从大到小对应配置列表的顺序
            scale_idx = self._sorted_scales.index(scale)
            if scale_idx < len(self.diffusion_timesteps_config):
                return self.diffusion_timesteps_config[scale_idx]
        # 默认值
        default_steps = {0: 150, 1: 200, 2: 250}
        return default_steps.get(scale, 150)

    def _get_inference_steps(self, scale: int, index: int) -> int:
        """Get inference steps for a given scale"""
        if self.diffusion_steps_config is not None:
            # 从配置中读取：scale从大到小对应配置列表的顺序
            scale_idx = self._sorted_scales.index(scale)
            if scale_idx < len(self.diffusion_steps_config):
                return self.diffusion_steps_config[scale_idx]
        # 默认值
        default_steps = {0: 3, 1: 4, 2: 5}
        return default_steps.get(scale, max(3, 5 - index))

    def forward(self, input_features, gt=None, mask=None):
        """
        Args:
            input_features: encoder features (list of 5 tensors from ResNet)
            gt: ground truth disparity for diffusion (from teacher model), dict with keys ("disp_diffusion", scale)
            mask: optional mask for masked training
        
        Note: This follows MonoDiffusion's architecture where decoder always processes
        scale 2 -> 1 -> 0 in sequence, regardless of which scales are in self.scales.
        """
        self.outputs = {}

        # Apply mask if provided
        if mask is not None:
            b, c, h, w = input_features[0].shape
            mask_initial = (torch.rand(b, 1, h, w).to(input_features[0].device) > 0.2).float()
            input_features[0] = input_features[0] * mask_initial
            self.outputs[("mask", 0)] = mask_initial

            for i in range(len(input_features)):
                if i > 0:
                    b, c, h, w = input_features[i].shape
                    mask_resized = F.interpolate(mask_initial, [h, w], mode="nearest")
                    input_features[i] = input_features[i] * mask_resized
                    self.outputs[("mask", i)] = mask_resized
    
        # Decoder forward pass - ALWAYS decode from scale 2 to 0
        # This matches MonoDiffusion's implementation
        # 
        # Note: MonoDiffusion uses LiteMono encoder with 3 feature layers.
        # We use ResnetEncoder with 5 feature layers.
        # We need to select the last 3 features for skip connections:
        # - features[2] (128 ch, H/8, W/8)  for decoder scale 1
        # - features[3] (256 ch, H/16, W/16) for decoder scale 2
        # - features[4] (512 ch, H/32, W/32) is the starting point
        
        x = input_features[-1]  # Start from deepest feature (512 channels, H/32, W/32)
        conditions = {}
        
        # Process decoder layers in order: 2 -> 1 -> 0
        for i in range(2, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]

            # Add skip connections from appropriate encoder features
            # decoder i=2 -> encoder features[2] (128 ch, H/8, W/8) after upsampling to H/16, W/16
            # decoder i=1 -> encoder features[1] (64 ch, H/4, W/4) after upsampling to H/8, W/8  
            # decoder i=0 -> no skip
            if self.use_skips and i > 0:
                # Map decoder scale to encoder feature index
                # After upsampling x is at resolution H/(2^(4-i)), W/(2^(4-i))
                # We need encoder feature at the same resolution
                skip_idx = i + 1  # decoder i=2 uses encoder[3], i=1 uses encoder[2]
                if skip_idx < len(input_features):
                    skip = input_features[skip_idx]
                    # Ensure spatial dimensions match after upsampling
                    if skip.shape[-2:] != x[0].shape[-2:]:
                        skip = F.interpolate(skip, size=x[0].shape[-2:], mode="nearest")
                    x += [skip]
            
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            # Store condition features for scales we care about
            # Note: Upsample condition features to match target resolution like MonoDiffusion
            if i in self.scales:
                f = self.convs[("conconv", i)](x)
                # Upsample condition feature to target resolution
                f_upsampled = F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False)
                conditions[str(i)] = f_upsampled
        
        # Check if we're in inference mode (no GT provided)
        is_inference = (gt is None)
        
        # Diffusion refinement (training mode with GT) or generation (inference mode without GT)
        # Process scales from coarsest to finest (2 -> 1 -> 0)
        refined_depths = {}
        condition_inputs = {}
        diffusion_traces = {}
        base_noise = None
        
        # Process only the scales that are in self.scales, in reverse order
        sorted_scales = sorted(self.scales, reverse=True)

        for idx, scale in enumerate(sorted_scales):
            scale_key = str(scale)
            condition = conditions[scale_key]
            
            # Get target shape: from GT if available, otherwise from condition
            if is_inference:
                # Inference mode: use condition feature shape to determine output shape
                # condition shape is (B, C, H, W), output should be (B, num_output_channels, H, W)
                B, C, H, W = condition.shape
                target_shape = (self.num_output_channels, H, W)
            else:
                # Training mode: use GT shape
                target_shape = gt[("disp_diffusion", scale)].shape[-3:]
            
            condition = self._resize_to(condition, target_shape[-2:])
            cond_input = condition

            # Add previous scale's refined depth as additional conditioning
            if idx > 0:
                prev_scale = sorted_scales[idx - 1]
                prev_key = str(prev_scale)
                prev_refined = refined_depths[prev_key]

                if ("disptocon", scale) in self.convs:
                    prev_feat = self.convs[("disptocon", scale)](prev_refined)
                    prev_feat = self._resize_to(prev_feat, condition.shape[-2:])
                    cond_input = cond_input + prev_feat

            condition_inputs[scale_key] = cond_input

            pipeline = self.diffusion_pipelines[scale_key]
            num_steps = self.diffusion_inference_steps[scale_key]

            if idx == 0:
                # First scale: generate new noise
                refined_depth, pred_seq, base_noise = pipeline(
                    batch_size=x.shape[0],
                    device=x.device,
                    dtype=x.dtype,
                    shape=target_shape,
                    input_args=(cond_input, None, None, None),
                    num_inference_steps=num_steps
                )
            else:
                # Subsequent scales: reuse upsampled noise
                upsampled_noise = self._resize_to(base_noise, target_shape[-2:])
                refined_depth, pred_seq, _ = pipeline(
                    batch_size=x.shape[0],
                    device=x.device,
                    dtype=x.dtype,
                    shape=target_shape,
                    input_args=(cond_input, None, None, None),
                    num_inference_steps=num_steps,
                    ini_noise=upsampled_noise
                )

            refined_depths[scale_key] = refined_depth
            diffusion_traces[scale_key] = [self.sigmoid(pred) for pred in pred_seq]
            self.outputs[("disp", scale)] = self.sigmoid(refined_depth)

        self.outputs["re-diffusion"] = diffusion_traces

        # Compute DDIM losses (only in training mode when GT is available)
        if not is_inference:
            for scale in self.scales:
                scale_key = str(scale)
                ddim_loss = self._compute_ddim_loss(
                    scale_key=scale_key,
                    gt_depth=gt[("disp_diffusion", scale)],
                    condition_input=condition_inputs[scale_key]
                )
                self.outputs[("ddim_loss", scale)] = ddim_loss
        else:
            # In inference mode, set ddim_loss to zero for all scales
            for scale in self.scales:
                self.outputs[("ddim_loss", scale)] = torch.tensor(0.0, device=condition_inputs[str(scale)].device)

        return self.outputs
    
    def _resize_to(self, tensor: torch.Tensor, spatial_size) -> torch.Tensor:
        if tensor.shape[-2:] == spatial_size:
            return tensor
        return F.interpolate(tensor, size=spatial_size, mode="nearest")
    
    def _compute_ddim_loss(self, scale_key: str, gt_depth: torch.Tensor, condition_input: torch.Tensor):
        scheduler = self.schedulers[scale_key]
        model = self.diffusion_models[scale_key]
        noise = torch.randn_like(gt_depth)
        bs = gt_depth.shape[0]
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (bs,), device=gt_depth.device).long()
        noisy_images = scheduler.add_noise(gt_depth, noise, timesteps)
        noise_pred = model(noisy_images, timesteps, condition_input, None, None, None)
        loss = F.mse_loss(noise_pred, noise)
        return loss


class CNNDDIMPipeline:
    """Generic DDIM pipeline supporting optional initial noise."""
    def __init__(self, model, scheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

    def __call__(
        self,
        batch_size,
        device,
        dtype,
        shape,
        input_args,
        ini_noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        **kwargs,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...], torch.Tensor]:
        image_shape = (batch_size, *shape)
        if ini_noise is None:
            image = torch.randn(image_shape, generator=generator, device=device, dtype=dtype)
            initial_noise = image
        else:
            image = ini_noise
            initial_noise = ini_noise

        self.scheduler.set_timesteps(num_inference_steps)

        pred = []
        for t in self.scheduler.timesteps:
            model_output = self.model(image, t.to(device), *input_args)
            image = self.scheduler.step(
                model_output,
                t,
                image,
                eta=eta,
                use_clipped_model_output=True,
                generator=generator
            )['prev_sample']
            pred.append(image)

        return image, tuple(pred), initial_noise


class ScheduledCNNRefine(nn.Module):
    """CNN-based noise predictor for diffusion"""
    def __init__(self, channels_in, channels_noise, **kwargs):
        super().__init__(**kwargs)
        
        # Noise embedding
        self.noise_embedding = nn.Sequential(
            nn.Conv2d(channels_noise, 16, kernel_size=3, stride=1, padding=1),
        )
        
        # Time embedding
        self.time_embedding = nn.Embedding(1280, channels_in)
        
        # Prediction network
        self.pred = nn.Sequential(
            nn.Conv2d(channels_in, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(True),
            nn.Conv2d(16, channels_noise, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, noisy_image, t, *args):
        feat, blur_depth, sparse_depth, sparse_mask = args
        
        # Combine features and time embedding
        if t.numel() == 1:
            feat = feat + self.time_embedding(t)[..., None, None]
        else:
            feat = feat + self.time_embedding(t)[..., None, None]
        
        # Add noise embedding
        feat = feat + self.noise_embedding(noisy_image)
        
        # Predict noise
        ret = self.pred(feat)
        return ret

