"""
Depth Anything Model Wrapper for MonoLDP Training
This module provides a unified interface to use Depth Anything models as teacher models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, Tuple, Optional


class DepthAnythingWrapper(nn.Module):
    """
    Wrapper for Depth Anything model to use as a teacher model in MonoLDP training
    
    Supports multiple Depth Anything variants:
    - Depth-Anything-V2 (recommended)
    - Depth-Anything-V1
    
    Features:
    - Automatic model downloading
    - Input normalization and preprocessing
    - Output format adaptation to match MonoLDP expected format
    - Multi-scale output generation
    """
    
    def __init__(
        self,
        model_type: str = "vits",  # Options: "vits", "vitb", "vitl"
        version: str = "v2",  # Options: "v1", "v2"
        device: str = "cuda",
        scales: list = [0, 1, 2],
        input_size: Tuple[int, int] = (256, 320),  # (height, width)
        pretrained: bool = True,
        model_path: Optional[str] = None
    ):
        super(DepthAnythingWrapper, self).__init__()
        
        self.model_type = model_type
        self.version = version
        self.device = device
        self.scales = scales
        self.input_height, self.input_width = input_size
        self.pretrained = pretrained
        self.model_path = model_path
        
        # Load the Depth Anything model
        self.model = self._load_depth_anything_model()
        
        # Freeze all parameters - this is a teacher model
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        
        # ImageNet normalization (used by Depth Anything)
        self.register_buffer(
            'mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', 
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        
        print(f"✓ Depth Anything {self.version.upper()} ({self.model_type}) loaded successfully")
        print(f"  - Model frozen for teacher mode")
        print(f"  - Input size: {self.input_height}x{self.input_width}")
        print(f"  - Output scales: {self.scales}")
    
    def _load_depth_anything_model(self):
        """Load Depth Anything model with error handling and automatic downloading"""
        try:
            if self.version == "v2":
                return self._load_depth_anything_v2()
            else:
                return self._load_depth_anything_v1()
        except Exception as e:
            print(f"⚠ Error loading Depth Anything: {e}")
            print("→ Attempting to install required packages...")
            self._install_dependencies()
            # Retry loading
            if self.version == "v2":
                return self._load_depth_anything_v2()
            else:
                return self._load_depth_anything_v1()
    
    def _load_depth_anything_v2(self):
        """Load Depth Anything V2 model"""
        try:
            # Try to import from depth_anything_v2 package
            from depth_anything_v2.dpt import DepthAnythingV2
            
            # Model configuration mapping
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            }
            
            if self.model_type not in model_configs:
                raise ValueError(f"Invalid model_type: {self.model_type}. Choose from {list(model_configs.keys())}")
            
            config = model_configs[self.model_type]
            
            # Create model
            model = DepthAnythingV2(**config)
            
            # Load pretrained weights if specified
            if self.pretrained and self.model_path is not None:
                if os.path.exists(self.model_path):
                    print(f"→ Loading weights from {self.model_path}")
                    state_dict = torch.load(self.model_path, map_location='cpu')
                    model.load_state_dict(state_dict)
                else:
                    print(f"⚠ Model path {self.model_path} not found")
                    print("→ Attempting to download pretrained weights...")
                    model = self._download_depth_anything_v2(model)
            elif self.pretrained:
                print("→ Attempting to download pretrained weights...")
                model = self._download_depth_anything_v2(model)
            
            model = model.to(self.device)
            return model
            
        except ImportError:
            print("⚠ depth_anything_v2 package not found")
            print("→ Please install: pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git")
            # Fallback: try to use transformers
            return self._load_from_transformers()
    
    def _load_depth_anything_v1(self):
        """Load Depth Anything V1 model"""
        try:
            from depth_anything.dpt import DepthAnything
            from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
            
            # Model configuration
            model = DepthAnything.from_pretrained(
                f'LiheYoung/depth_anything_{self.model_type}14'
            ).to(self.device)
            
            return model
            
        except ImportError:
            print("⚠ depth_anything package not found")
            return self._load_from_transformers()
    
    def _load_from_transformers(self):
        """Fallback: Load Depth Anything from HuggingFace transformers"""
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            
            # Use Depth Anything V2 from HuggingFace
            model_name = f"depth-anything/Depth-Anything-V2-{self.model_type.capitalize()}"
            
            print(f"→ Loading from HuggingFace: {model_name}")
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForDepthEstimation.from_pretrained(model_name)
            model = model.to(self.device)
            
            return model
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Depth Anything model: {e}\n"
                "Please install required packages:\n"
                "  pip install transformers timm\n"
                "  or\n"
                "  pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git"
            )
    
    def _download_depth_anything_v2(self, model):
        """Download pretrained Depth Anything V2 weights"""
        import urllib.request
        
        # Download URLs for pretrained models
        model_urls = {
            'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
            'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
            'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
        }
        
        if self.model_type not in model_urls:
            print(f"⚠ No download URL for model type: {self.model_type}")
            return model
        
        # Create checkpoint directory
        checkpoint_dir = os.path.expanduser("~/.cache/depth_anything_v2")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"depth_anything_v2_{self.model_type}.pth")
        
        # Download if not exists
        if not os.path.exists(checkpoint_path):
            print(f"→ Downloading {self.model_type} model to {checkpoint_path}")
            try:
                urllib.request.urlretrieve(model_urls[self.model_type], checkpoint_path)
                print("✓ Download complete")
            except Exception as e:
                print(f"⚠ Download failed: {e}")
                print(f"  Please manually download from: {model_urls[self.model_type]}")
                print(f"  And save to: {checkpoint_path}")
                return model
        
        # Load the downloaded weights
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print("✓ Pretrained weights loaded")
        
        return model
    
    def _install_dependencies(self):
        """Attempt to install required dependencies"""
        import subprocess
        import sys
        
        print("→ Installing depth_anything dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "transformers", "timm", "huggingface_hub"
            ])
            print("✓ Dependencies installed")
        except Exception as e:
            print(f"⚠ Failed to install dependencies: {e}")
            print("  Please manually install: pip install transformers timm huggingface_hub")
    
    def _normalize_input(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalize input image using ImageNet statistics
        
        Args:
            image: Input tensor in range [0, 1], shape (B, 3, H, W)
        
        Returns:
            Normalized tensor
        """
        return (image - self.mean) / self.std
    
    def _denormalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Convert raw depth output to disparity format compatible with MonoLDP
        
        Args:
            depth: Raw depth prediction from Depth Anything
        
        Returns:
            Disparity tensor (inverse depth)
        """
        # Depth Anything outputs relative depth, convert to disparity
        # Add small epsilon to avoid division by zero
        eps = 1e-6
        
        # Normalize depth to [0, 1]
        depth_min = depth.min()
        depth_max = depth.max()
        depth_normalized = (depth - depth_min) / (depth_max - depth_min + eps)
        
        # Convert to disparity (inverse depth)
        # Use a small offset to avoid infinity
        disp = 1.0 / (depth_normalized + 0.01)
        
        # Normalize disparity
        disp = (disp - disp.min()) / (disp.max() - disp.min() + eps)
        
        return disp
    
    def _infer_depth(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference on the Depth Anything model
        
        Args:
            x: Normalized input tensor (B, 3, H, W)
        
        Returns:
            Raw depth prediction
        """
        try:
            # Try direct inference (for native Depth Anything models)
            depth = self.model(x)
            if isinstance(depth, dict):
                depth = depth['depth'] if 'depth' in depth else depth['predicted_depth']
            return depth
        except:
            # Fallback for transformers-based models
            try:
                with torch.no_grad():
                    outputs = self.model(x)
                    depth = outputs.predicted_depth
                return depth
            except Exception as e:
                raise RuntimeError(f"Failed to run inference: {e}")
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[Tuple[str, int], torch.Tensor]:
        """
        Forward pass through Depth Anything model
        
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
        
        # Ensure depth has correct shape (B, 1, H, W)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        
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


def create_depth_anything_teacher(
    model_type: str = "vits",
    version: str = "v2",
    device: str = "cuda",
    scales: list = [0, 1, 2],
    input_size: Tuple[int, int] = (256, 320),
    model_path: Optional[str] = None
) -> DepthAnythingWrapper:
    """
    Factory function to create a Depth Anything teacher model
    
    Args:
        model_type: Model size ("vits", "vitb", "vitl")
        version: Depth Anything version ("v1", "v2")
        device: Device to load model on
        scales: Output scales for multi-scale training
        input_size: Input image size (height, width)
        model_path: Optional path to pretrained weights
    
    Returns:
        DepthAnythingWrapper instance ready for use as teacher model
    """
    return DepthAnythingWrapper(
        model_type=model_type,
        version=version,
        device=device,
        scales=scales,
        input_size=input_size,
        pretrained=True,
        model_path=model_path
    )


if __name__ == "__main__":
    # Test the wrapper
    print("Testing Depth Anything Wrapper...")
    
    # Create teacher model
    teacher = create_depth_anything_teacher(
        model_type="vits",
        version="v2",
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

