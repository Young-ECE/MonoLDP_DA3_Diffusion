"""
Depth Anything V3 Model Wrapper for MonoLDP Training
This module provides a unified interface to use Depth Anything V3 models as teacher models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, Tuple, Optional


class DepthAnythingV3Wrapper(nn.Module):
    """
    Wrapper for Depth Anything V3 model to use as a teacher model in MonoLDP training
    
    Supports:
    - DA3Mono-Large: High-quality relative monocular depth estimation
    
    Features:
    - Automatic model loading from HuggingFace or local path
    - Input normalization and preprocessing
    - Output format adaptation to match MonoLDP expected format
    - Multi-scale output generation
    """
    
    def __init__(
        self,
        model_name: str = "DA3Mono-Large",  # Options: "DA3Mono-Large"
        device: str = "cuda",
        scales: list = [0, 1, 2],
        input_size: Tuple[int, int] = (256, 320),  # (height, width)
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        super(DepthAnythingV3Wrapper, self).__init__()
        
        self.model_name = model_name
        self.device = torch.device(device) if isinstance(device, str) else device
        self.scales = scales
        self.input_height, self.input_width = input_size
        self.model_path = model_path
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/depth_anything_3")
        
        # ImageNet normalization - register buffers BEFORE loading model
        # so they will be on the correct device when the module is moved
        self.register_buffer(
            'mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', 
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        
        # Load the Depth Anything V3 model
        self.model = self._load_da3_model()
        
        # Freeze all parameters - this is a teacher model
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        
        print(f"✓ Depth Anything V3 ({self.model_name}) loaded successfully")
        print(f"  - Model frozen for teacher mode")
        print(f"  - Input size: {self.input_height}x{self.input_width}")
        print(f"  - Output scales: {self.scales}")
    
    def _load_da3_model(self):
        """Load Depth Anything V3 model"""
        try:
            from depth_anything_3.api import DepthAnything3
            from huggingface_hub import HfApi
            import os
            
            print(f"→ Loading Depth Anything V3 model: {self.model_name}")
            
            # 设置 HuggingFace 镜像端点（如果未设置）
            hf_endpoint = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
            if hf_endpoint:
                os.environ['HF_ENDPOINT'] = hf_endpoint
                print(f"  Using HuggingFace endpoint: {hf_endpoint}")
            
            # Map model name to DA3 internal model name
            model_name_map = {
                "DA3Mono-Large": "da3mono-large"
            }
            da3_model_name = model_name_map.get(self.model_name, self.model_name.lower())
            
            # 显示模型缓存目录
            cache_dir = self.cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
            print(f"  Model cache directory: {cache_dir}")
            
            # If model_path is provided, load from local path
            if self.model_path and os.path.exists(self.model_path):
                print(f"  Loading from local path: {self.model_path}")
                model = DepthAnything3.from_pretrained(
                    self.model_path,
                    cache_dir=self.cache_dir
                )
                print(f"  ✓ Model loaded from local path")
            else:
                # Try to load from HuggingFace (with mirror support)
                hf_model_name = f"depth-anything/{self.model_name}"
                print(f"  Loading from HuggingFace: {hf_model_name}")
                print(f"  (This may take a while if downloading for the first time...)")
                
                try:
                    # 启用进度条
                    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
                    enable_progress_bars()
                    
                    model = DepthAnything3.from_pretrained(
                        hf_model_name,
                        cache_dir=self.cache_dir
                    )
                    
                    # 显示模型保存位置
                    model_cache_path = os.path.join(cache_dir, f"models--{hf_model_name.replace('/', '--')}")
                    if os.path.exists(model_cache_path):
                        print(f"  ✓ Model loaded successfully")
                        print(f"  Model saved at: {model_cache_path}")
                    else:
                        # 尝试查找实际缓存位置
                        import glob
                        possible_paths = glob.glob(os.path.join(cache_dir, "*DA3Mono*"))
                        if possible_paths:
                            print(f"  ✓ Model loaded successfully")
                            print(f"  Model saved at: {possible_paths[0]}")
                        else:
                            print(f"  ✓ Model loaded successfully")
                            print(f"  Model cached in: {cache_dir}")
                    
                except Exception as e:
                    print(f"  ⚠ HuggingFace load failed: {e}")
                    # Fallback: create model from preset (without weights)
                    print(f"  → Creating model from preset: {da3_model_name}")
                    model = DepthAnything3(model_name=da3_model_name)
                    print(f"  ⚠ Model created without pretrained weights. Please download weights manually:")
                    print(f"     Option 1: Use mirror (recommended)")
                    print(f"       export HF_ENDPOINT=https://hf-mirror.com")
                    print(f"       huggingface-cli download depth-anything/{self.model_name} --local-dir {cache_dir}")
                    print(f"     Option 2: Direct download")
                    print(f"       git clone https://huggingface.co/{hf_model_name}")
                    print(f"     Then use --depth_anything_v3_weights to specify the path")
            
            # Move model to device
            print(f"  Moving model to device: {self.device}")
            model = model.to(self.device)
            model.device = self.device
            
            return model
            
        except ImportError:
            raise RuntimeError(
                "depth_anything_3 package not found. Please install:\n"
                "  pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git\n"
                "  or\n"
                "  pip install depth-anything-3"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Depth Anything V3 model: {e}")
    
    def _normalize_input(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize input image using ImageNet statistics"""
        return (image - self.mean) / self.std
    
    def _denormalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Convert raw depth output to disparity format compatible with MonoLDP
        
        Args:
            depth: Raw depth prediction from Depth Anything V3
        
        Returns:
            Disparity tensor (inverse depth)
        """
        eps = 1e-6
        
        # Normalize depth to [0, 1]
        depth_min = depth.min()
        depth_max = depth.max()
        depth_normalized = (depth - depth_min) / (depth_max - depth_min + eps)
        
        # Convert to disparity (inverse depth)
        disp = 1.0 / (depth_normalized + 0.01)
        
        # Normalize disparity
        disp = (disp - disp.min()) / (disp.max() - disp.min() + eps)
        
        return disp
    
    def _infer_depth(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference on the Depth Anything V3 model
        
        Args:
            x: Normalized input tensor (B, 3, H, W)
            
        Returns:
            Raw depth prediction (B, 1, H, W)
        """
        try:
            # Store original size for resizing back
            B, C, orig_H, orig_W = x.shape
            
            # DA3 requires input dimensions to be multiples of patch size (14)
            # Resize if necessary
            patch_size = 14
            if orig_H % patch_size != 0 or orig_W % patch_size != 0:
                import math
                new_H = math.ceil(orig_H / patch_size) * patch_size
                new_W = math.ceil(orig_W / patch_size) * patch_size
                x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
            
            # DA3 forward expects (B, N, 3, H, W) where N is number of views
            # For monocular depth, N=1
            if x.ndim == 4:
                x = x.unsqueeze(1)  # (B, 1, 3, H, W)
            
            # Run forward pass (no extrinsics/intrinsics needed for monocular)
            # export_feat_layers should be an empty list, not None
            with torch.no_grad():
                output = self.model.forward(
                    x, 
                    extrinsics=None, 
                    intrinsics=None,
                    export_feat_layers=[],
                    infer_gs=False
                )
            
            # Extract depth from output dictionary
            # DA3 output typically contains 'depth' key
            if isinstance(output, dict):
                depth = output.get('depth', output.get('predicted_depth', output.get('pred', None)))
                if depth is None:
                    # Try to find any tensor that looks like depth
                    for key, value in output.items():
                        if isinstance(value, torch.Tensor) and value.ndim >= 2:
                            depth = value
                            break
            else:
                depth = output
            
            if depth is None:
                raise RuntimeError("Could not extract depth from DA3 model output")
            
            # Ensure depth has correct shape (B, 1, H, W)
            if depth.ndim == 5:  # (B, N, 1, H, W)
                depth = depth[:, 0, :, :, :]  # Take first view
            elif depth.ndim == 4:  # (B, N, H, W) or (B, 1, H, W)
                if depth.shape[1] == 1:
                    pass  # Already (B, 1, H, W)
                else:
                    depth = depth[:, 0:1, :, :]  # Take first channel as depth
            elif depth.ndim == 3:  # (B, H, W)
                depth = depth.unsqueeze(1)  # (B, 1, H, W)
            elif depth.ndim == 2:  # (H, W)
                depth = depth.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            
            # Resize back to original size if we resized earlier
            if depth.shape[2] != orig_H or depth.shape[3] != orig_W:
                depth = F.interpolate(depth, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
            
            return depth
            
        except Exception as e:
            raise RuntimeError(f"Failed to run DA3 inference: {e}")
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[Tuple[str, int], torch.Tensor]:
        """
        Forward pass through Depth Anything V3 model
        
        Args:
            x: Input tensor (B, 3, H, W) in range [0, 1]
        
        Returns:
            Dictionary with keys ("disp", scale) containing disparity predictions
            at multiple scales, matching MonoLDP's DepthDecoder output format
        """
        # Ensure input is in correct format
        if x.shape[2:] != (self.input_height, self.input_width):
            x = F.interpolate(
                x, 
                size=(self.input_height, self.input_width), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Normalize input using ImageNet statistics
        x_normalized = self._normalize_input(x)
        
        # Run inference
        depth = self._infer_depth(x_normalized)
        
        # Resize to original input size if needed
        if depth.shape[2:] != (self.input_height, self.input_width):
            depth = F.interpolate(
                depth,
                size=(self.input_height, self.input_width),
                mode='bilinear',
                align_corners=False
            )
        
        # Convert depth to disparity
        disp_full = self._denormalize_depth(depth)
        
        # Generate multi-scale outputs
        outputs = {}
        for scale in self.scales:
            scale_h = self.input_height // (2 ** scale)
            scale_w = self.input_width // (2 ** scale)
            
            if scale == 0:
                outputs[("disp", scale)] = disp_full
            else:
                disp_scaled = F.interpolate(
                    disp_full,
                    size=(scale_h, scale_w),
                    mode='bilinear',
                    align_corners=False
                )
                outputs[("disp", scale)] = disp_scaled
        
        return outputs


def create_depth_anything_v3_teacher(
    model_name: str = "DA3Mono-Large",
    device: str = "cuda",
    scales: list = [0, 1, 2],
    input_size: Tuple[int, int] = (256, 320),
    model_path: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> DepthAnythingV3Wrapper:
    """
    Factory function to create a Depth Anything V3 teacher model
    
    Args:
        model_name: Model name ("DA3Mono-Large")
        device: Device to load model on
        scales: Output scales for multi-scale training
        input_size: Input image size (height, width)
        model_path: Optional path to local model directory
        cache_dir: Optional cache directory for HuggingFace models
    
    Returns:
        DepthAnythingV3Wrapper instance ready for use as teacher model
    """
    wrapper = DepthAnythingV3Wrapper(
        model_name=model_name,
        device=device,
        scales=scales,
        input_size=input_size,
        model_path=model_path,
        cache_dir=cache_dir
    )
    # Explicitly move the wrapper module to device to ensure all buffers are on the correct device
    if isinstance(device, str):
        device = torch.device(device)
    wrapper = wrapper.to(device)
    return wrapper


if __name__ == "__main__":
    # Test the wrapper
    print("Testing Depth Anything V3 Wrapper...")
    
    # Create teacher model
    teacher = create_depth_anything_v3_teacher(
        model_name="DA3Mono-Large",
        device="cuda" if torch.cuda.is_available() else "cpu",
        scales=[0, 1, 2],
        input_size=(256, 320)
    )
    
    # Test with random input
    batch_size = 2
    x = torch.rand(batch_size, 3, 256, 320)
    if torch.cuda.is_available():
        x = x.cuda()
    
    # Forward pass
    outputs = teacher(x)
    
    # Print output shapes
    print("\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    print("\n✓ Test successful!")

