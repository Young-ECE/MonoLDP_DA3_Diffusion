from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from layers import *
from timm.models.layers import trunc_normal_
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from typing import Union, Dict, Tuple, Optional


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
    """Depth decoder with diffusion-based refinement"""
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=3, 
                 use_skips=True, PixelCoorModu=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = sorted(list(scales))
        self.PixelCoorModu = PixelCoorModu

        self.num_ch_enc = np.array(num_ch_enc)
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int')

        # Decoder
        self.convs = OrderedDict()
        for i in range(2, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
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
        default_steps = {0: 150, 1: 200, 2: 250}
        return default_steps.get(scale, 150)

    def _get_inference_steps(self, scale: int, index: int) -> int:
        default_steps = {0: 3, 1: 4, 2: 5}
        return default_steps.get(scale, max(3, 5 - index))

    def forward(self, input_features, norm_pix_coords, gt=None, mask=None):
        """
        Args:
            input_features: encoder features
            norm_pix_coords: normalized pixel coordinates for depth decoding
            gt: ground truth disparity for diffusion (from teacher model)
            mask: optional mask for masked training
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
    
        # Decoder forward pass
        if len(input_features) >= len(self.scales) + 1:
            encoder_features = input_features[-(len(self.scales) + 1):]
        else:
            encoder_features = input_features

        x = encoder_features[-1]
        skip_features = encoder_features[:-1][::-1]
        skip_idx = 0
        sorted_scales = sorted(self.scales, reverse=True)
        conditions = {}
        
        for scale in sorted_scales:
            x = self.convs[("upconv", scale, 0)](x)
            x = [upsample(x)]

            if self.use_skips and scale != sorted_scales[-1] and skip_idx < len(skip_features):
                skip = skip_features[skip_idx]
                if skip.shape[-2:] != x[0].shape[-2:]:
                    skip = F.interpolate(skip, size=x[0].shape[-2:], mode="nearest")
                x += [skip]
                skip_idx += 1
            x = torch.cat(x, 1)
            x = self.convs[("upconv", scale, 1)](x)

            if scale in self.scales:
                f = self.convs[("conconv", scale)](x)
                conditions[str(scale)] = f
        
        # If no GT provided (inference mode), generate initial prediction
        if gt is None:
            raise RuntimeError(
                "DepthDecoderDiffusion requires teacher-provided targets during forward. "
                "Please supply `gt` when using the diffusion decoder."
            )
        
        # Diffusion refinement (training mode with GT)
        refined_depths = {}
        condition_inputs = {}
        diffusion_traces = {}
        base_noise = None

        for idx, scale in enumerate(sorted_scales):
            scale_key = str(scale)
            condition = conditions[scale_key]
            cond_input = condition

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
            target_shape = gt[("disp_diffusion", scale)].shape[-3:]

            if idx == 0:
                refined_depth, pred_seq, base_noise = pipeline(
                    batch_size=x.shape[0],
                    device=x.device,
                    dtype=x.dtype,
                    shape=target_shape,
                    input_args=(cond_input, None, None, None),
                    num_inference_steps=num_steps
                )
            else:
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

        # Compute DDIM losses
        for scale in self.scales:
            scale_key = str(scale)
            ddim_loss = self._compute_ddim_loss(
                scale_key=scale_key,
                gt_depth=gt[("disp_diffusion", scale)],
                condition_input=condition_inputs[scale_key]
            )
            self.outputs[("ddim_loss", scale)] = ddim_loss

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

