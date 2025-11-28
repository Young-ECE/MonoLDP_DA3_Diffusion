# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    """Load image from path using PIL.
    
    Args:
        path: Image file path
        
    Returns:
        PIL Image in RGB format
    """
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Base class for monocular depth estimation datasets.
    
    This class provides the common functionality for loading and preprocessing
    monocular depth estimation datasets, including:
    - Multi-frame image loading (for pose estimation)
    - Multi-scale image preprocessing
    - Color augmentation
    - Geometric regularization data (plane/line keysets)
    
    Args:
        data_path: Path to the dataset directory
        filenames: List of filenames (format: "folder frame_index [side]")
        height: Target image height
        width: Target image width
        frame_idxs: List of frame indices to load (e.g., [0, -1, 1])
        num_scales: Number of scales for multi-scale training
        is_train: Whether this is training dataset
        is_test: Whether this is test dataset
        return_plane: Whether to load plane segmentation data
        num_plane_keysets: Number of plane keysets to sample
        return_line: Whether to load line segmentation data
        num_line_keysets: Number of line keysets to sample
        img_ext: Image file extension (default: '.jpg')
    """

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 is_test=False,
                 return_plane=False,
                 num_plane_keysets=512,
                 return_line=False,
                 num_line_keysets=128,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        # Basic dataset configuration
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.is_test = is_test
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # Color augmentation parameters
        # Try newer tuple version first, fall back to scalars for older torchvision
        self._setup_color_augmentation()

        # Multi-scale resize transforms
        self._setup_resize_transforms()

        # Check which data to load
        self.load_depth = self.check_depth()
        self.load_plane = return_plane and self.check_plane()
        self.load_line = return_line and self.check_line()

        # Setup geometric regularization (plane/line) if needed
        if self.load_plane or self.load_line:
            self._setup_geometric_regularization(num_plane_keysets, num_line_keysets)

    def _setup_color_augmentation(self):
        """Setup color augmentation parameters with backward compatibility."""
        try:
            # Newer torchvision uses tuple ranges
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            # Test if tuple version works
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            # Older torchvision uses scalar ranges
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

    def _setup_resize_transforms(self):
        """Setup multi-scale resize transforms."""
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize(
                (self.height // s, self.width // s),
                interpolation=self.interp
            )

    def _setup_geometric_regularization(self, num_plane_keysets, num_line_keysets):
        """Setup geometric regularization (plane/line) transforms and parameters."""
        self.pl_resize = {}
        self.num_plane_keysets = num_plane_keysets
        self.num_line_keysets = num_line_keysets
        
        # Use NEAREST interpolation for segmentation maps to preserve labels
        for i in range(self.num_scales):
            s = 2 ** i
            self.pl_resize[i] = transforms.Resize(
                (self.height // s, self.width // s),
                interpolation=Image.NEAREST
            )

    def _parse_filename(self, index):
        """Parse filename line to extract folder, frame_index, and side.
        
        Filename format: "folder frame_index [side]"
        Example: "scene_001 123" or "scene_001 123 l"
        
        Args:
            index: Dataset index
            
        Returns:
            tuple: (folder, frame_index, side)
                folder: Scene/folder name
                frame_index: Frame index (int)
                side: Stereo side ('l' or 'r'), None if not specified
        """
        line = self.filenames[index].split()
        folder = line[0]
        
        if len(line) >= 2:
            frame_index = int(line[1])
        else:
            frame_index = 0
            
        if len(line) == 3:
            side = line[2]
        else:
            side = None
            
        return folder, frame_index, side

    def _create_color_augmentation(self, do_augment):
        """Create color augmentation transform.
        
        Args:
            do_augment: Whether to apply augmentation
            
        Returns:
            ColorJitter transform or identity function
        """
        if do_augment:
            return transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue
            )
        else:
            return lambda x: x

    def _load_frame_images(self, folder, frame_index, side, do_flip):
        """Load images for all requested frames.
        
        Args:
            folder: Scene/folder name
            frame_index: Base frame index
            side: Stereo side ('l' or 'r'), None if monocular
            do_flip: Whether to horizontally flip images
            
        Returns:
            dict: Dictionary with keys ("color", frame_id, -1) containing PIL Images
        """
        inputs = {}
        
        for frame_id in self.frame_idxs:
            if frame_id == "s":
                # Stereo pair: use opposite side
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", frame_id, -1)] = self.get_color(
                    folder, frame_index, other_side, do_flip
                )
            else:
                # Temporal frames: use same side, adjust frame index
                inputs[("color", frame_id, -1)] = self.get_color(
                    folder, frame_index + frame_id, side, do_flip
                )
        
        return inputs

    def _compute_camera_intrinsics(self, do_flip):
        """Compute camera intrinsics for all scales.
        
        Args:
            do_flip: Whether images are horizontally flipped
            
        Returns:
            dict: Dictionary with keys ("K", scale) and ("norm_pix_coords", scale)
        """
        intrinsics = {}
        
        for scale in range(self.num_scales):
            K = self.K.copy()
            
            # Adjust principal point if flipped
            if do_flip:
                K[0, 2] = 1 - K[0, 2]

            # Scale intrinsics to match image resolution at this scale
            width = self.width // (2 ** scale)
            height = self.height // (2 ** scale)
            
            K[0, :] *= width   # fx, cx scaled by width
            K[1, :] *= height  # fy, cy scaled by height

            intrinsics[("K", scale)] = torch.from_numpy(K)

            # Compute normalized pixel coordinates
            # Used for 3D point backprojection
            Us, Vs = np.meshgrid(
                np.linspace(0, width - 1, width, dtype=np.float32),
                np.linspace(0, height - 1, height, dtype=np.float32),
                indexing='xy'
            )
            Ones = np.ones([height, width], dtype=np.float32)
            norm_pix_coords = np.stack((
                (Us - K[0, 2]) / K[0, 0],  # (u - cx) / fx
                (Vs - K[1, 2]) / K[1, 1],  # (v - cy) / fy
                Ones
            ), axis=0)
            intrinsics[("norm_pix_coords", scale)] = torch.from_numpy(norm_pix_coords)
        
        return intrinsics

    def _load_auxiliary_data(self, folder, frame_index, side, do_flip, inputs):
        """Load auxiliary data (depth, plane, line) if available.
        
        Args:
            folder: Scene/folder name
            frame_index: Frame index
            side: Stereo side
            do_flip: Whether images are flipped
            inputs: Dictionary to add loaded data to
        """
        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if self.load_plane:
            inputs[("plane", 0, -1)] = self.get_plane(folder, frame_index, side, do_flip)

        if self.load_line:
            inputs[("line", 0, -1)] = self.get_line(folder, frame_index, side, do_flip)

    def _generate_keysets(self, struct_map, struct_type, scale):
        """Generate keysets for geometric regularization (plane/line).
        
        Keysets are groups of pixels that should lie on the same plane or line.
        Used for geometric regularization losses.
        
        Args:
            struct_map: Segmentation map (numpy array, shape: (1, H, W))
            struct_type: Type of structure ('plane' or 'line')
            scale: Current scale index
            
        Returns:
            numpy array: Keysets array (shape: (num_samples, num_keysets))
                - For planes: num_samples = 4 (4 points define a plane)
                - For lines: num_samples = 3 (3 points define a line)
        """
        # Get parameters based on structure type
        if struct_type == "plane":
            num_keysets = self.num_plane_keysets
            keyset_samples = 4  # 4 points to define a plane
        else:  # line
            num_keysets = self.num_line_keysets
            keyset_samples = 3  # 3 points to define a line

        # Flatten the segmentation map
        # Note: Using C-order (row-major) flatten to match points_3d flatten order
        struct_map_flat = struct_map.flatten()
        
        # Get image dimensions for index validation
        H, W = struct_map.shape[1], struct_map.shape[2]
        max_valid_index = H * W - 1
        
        # Count pixels for each structure
        num_struct_pixels = np.sum(struct_map_flat > 0)
        num_struct = struct_map_flat.max()

        if num_struct == 0 or num_struct_pixels == 0:
            # No structures found, return empty keysets (marked with -1 to indicate invalid)
            target_keysets = num_keysets // (2 ** scale)
            return np.full((keyset_samples, target_keysets), -1, dtype=np.int32)

        # Sample keysets proportionally to structure size
        keysets = []
        acc_num = 0
        target_keysets_per_scale = num_keysets // (2 ** scale)
        
        for struct_id in range(1, int(num_struct) + 1):
            # Count pixels for this structure
            pixels_in_struct = np.sum(struct_map_flat == struct_id)
            
            # Calculate how many keysets to sample for this structure
            # Proportionally to structure size
            num_keysets_for_struct = int(
                np.ceil(target_keysets_per_scale * pixels_in_struct / num_struct_pixels)
            )
            acc_num += num_keysets_for_struct

            # Sample random pixels from this structure
            struct_pixel_indices = np.argwhere(struct_map_flat == struct_id).flatten()
            if len(struct_pixel_indices) == 0:
                continue
                
            # Sample (num_keysets_for_struct * keyset_samples) pixels
            # Ensure we sample enough pixels for at least one complete keyset
            num_samples_needed = num_keysets_for_struct * keyset_samples
            
            # Adjust if structure has fewer pixels than needed
            if len(struct_pixel_indices) < keyset_samples:
                # Not enough pixels for even one keyset, skip this structure
                continue
            
            # Calculate actual number of keysets we can generate
            max_keysets_from_struct = len(struct_pixel_indices) // keyset_samples
            actual_keysets_for_struct = min(num_keysets_for_struct, max_keysets_from_struct)
            
            # Sample exactly (actual_keysets_for_struct * keyset_samples) pixels
            num_samples_actual = actual_keysets_for_struct * keyset_samples
            
            if len(struct_pixel_indices) >= num_samples_actual:
                # Enough pixels, sample without replacement
                sampled_indices = np.random.choice(
                    struct_pixel_indices,
                    size=num_samples_actual,
                    replace=False
                )
                
                # Validate index range
                if np.any(sampled_indices < 0) or np.any(sampled_indices > max_valid_index):
                    raise ValueError(
                        f"Invalid keyset indices for {struct_type} at scale {scale}: "
                        f"range=[0, {max_valid_index}], "
                        f"found=[{sampled_indices.min()}, {sampled_indices.max()}]"
                    )
                
                # Reshape to (keyset_samples, actual_keysets_for_struct)
                keyset_for_struct = sampled_indices.reshape(keyset_samples, actual_keysets_for_struct)
            else:
                # Not enough pixels, sample with replacement
                # But we need to ensure at least keyset_samples unique points per keyset
                # Strategy: sample more than needed, then filter to ensure uniqueness
                # Sample enough to potentially get unique keysets (sample 3x to have buffer)
                num_samples_with_buffer = num_samples_actual * 3
                sampled_indices = np.random.choice(
                    struct_pixel_indices,
                    size=num_samples_with_buffer,
                    replace=True
                )
                
                # Validate index range
                if np.any(sampled_indices < 0) or np.any(sampled_indices > max_valid_index):
                    raise ValueError(
                        f"Invalid keyset indices for {struct_type} at scale {scale}: "
                        f"range=[0, {max_valid_index}], "
                        f"found=[{sampled_indices.min()}, {sampled_indices.max()}]"
                    )
                
                # Reshape to (keyset_samples, num_samples_with_buffer // keyset_samples)
                num_keysets_from_buffer = num_samples_with_buffer // keyset_samples
                keyset_for_struct = sampled_indices[:num_keysets_from_buffer * keyset_samples].reshape(
                    keyset_samples, num_keysets_from_buffer
                )
                
                # Filter out keysets with duplicate points
                valid_keysets = []
                for k in range(keyset_for_struct.shape[1]):
                    keyset_points = keyset_for_struct[:, k]
                    if len(np.unique(keyset_points)) == keyset_samples:
                        # All points are unique, valid keyset
                        valid_keysets.append(keyset_points)
                    # Otherwise, skip this keyset (it has duplicate points)
                
                if len(valid_keysets) >= actual_keysets_for_struct:
                    # Enough valid keysets, take the first actual_keysets_for_struct
                    keyset_for_struct = np.stack(valid_keysets[:actual_keysets_for_struct], axis=1)
                elif len(valid_keysets) > 0:
                    # Some valid keysets, use what we have
                    keyset_for_struct = np.stack(valid_keysets, axis=1)
                else:
                    # No valid keysets from this structure, skip it
                    continue
            
            keysets.append(keyset_for_struct)

        if len(keysets) > 0:
            # Concatenate all structures' keysets
            keysets = np.concatenate(keysets, axis=1)
            
            # Randomly sample to get exact number of keysets
            if keysets.shape[1] > target_keysets_per_scale:
                selected_indices = np.random.choice(
                    keysets.shape[1],
                    size=target_keysets_per_scale,
                    replace=False
                )
                keysets = keysets[:, selected_indices]
            elif keysets.shape[1] < target_keysets_per_scale:
                # Pad if not enough
                num_to_pad = target_keysets_per_scale - keysets.shape[1]
                padding = np.random.choice(
                    keysets.shape[1],
                    size=num_to_pad,
                    replace=True
                )
                keysets = np.concatenate([keysets, keysets[:, padding]], axis=1)
        else:
            # No keysets found, return empty keysets (marked with -1 to indicate invalid)
            keysets = np.full((keyset_samples, target_keysets_per_scale), -1, dtype=np.int32)

        # Final validation: ensure all indices are within valid range
        if keysets.size > 0:
            valid_mask = keysets >= 0  # Only check non-negative indices (negative means invalid/empty)
            if np.any(valid_mask):
                valid_indices = keysets[valid_mask]
                if np.any(valid_indices > max_valid_index):
                    raise ValueError(
                        f"Invalid keyset indices after processing for {struct_type} at scale {scale}: "
                        f"max_valid={max_valid_index}, found_max={valid_indices.max()}"
                    )

        return keysets

    def _preprocess_geometric_data(self, inputs, scale):
        """Preprocess geometric regularization data (plane/line) for a scale.
        
        Args:
            inputs: Dictionary containing data
            scale: Current scale index
        """
        for struct_type in ["plane", "line"]:
            struct_key = (struct_type, 0, -1)
            if struct_key not in inputs:
                continue

            # Get resized segmentation map (created in step 1)
            resized_map = inputs.get((struct_type, 0, scale), None)
            if resized_map is None:
                # If not already resized, resize now
                resized_map = self.pl_resize[scale](inputs[struct_key])

            # Convert to numpy array
            struct_map_array = np.expand_dims(np.array(resized_map), 0)

            # Store both float and long versions
            inputs[(struct_type + "_float", 0, scale)] = torch.from_numpy(struct_map_array).float()
            inputs[(struct_type, 0, scale)] = torch.from_numpy(struct_map_array).long()

            # Generate keysets for geometric regularization
            num_keysets = self.num_plane_keysets if struct_type == "plane" else self.num_line_keysets
            if num_keysets > 0:
                keysets = self._generate_keysets(struct_map_array, struct_type, scale)
                inputs[(struct_type + "_keysets", 0, scale)] = torch.from_numpy(keysets).long()

    def preprocess(self, inputs, color_aug):
        """Preprocess all loaded data (resize, augment, convert to tensors).
        
        This method:
        1. Resizes images to all scales
        2. Applies color augmentation
        3. Converts images to tensors
        4. Preprocesses geometric data (plane/line) and generates keysets
        
        Args:
            inputs: Dictionary containing loaded data
            color_aug: Color augmentation transform
        """
        # Step 1: Resize images to all scales
        for key in list(inputs.keys()):
            if "color" in key:
                name, frame_id, current_scale = key
                if current_scale == -1:
                    # Resize from native resolution to all scales
                    for scale in range(self.num_scales):
                        inputs[(name, frame_id, scale)] = self.resize[scale](inputs[key])
            
            if "plane" in key or "line" in key:
                name, frame_id, current_scale = key
                if current_scale == -1:
                    # Resize segmentation maps to all scales (use NEAREST)
                    for scale in range(self.num_scales):
                        inputs[(name, frame_id, scale)] = self.pl_resize[scale](inputs[key])

        # Step 2: Apply color augmentation and convert to tensors
        for key in list(inputs.keys()):
            if "color" in key:
                name, frame_id, scale = key
                if scale == -1:
                    continue  # Skip native resolution (will be processed separately)
                
                # Get the resized image (created in step 1)
                resized_image = inputs[key]
                
                # Convert to tensor
                inputs[key] = self.to_tensor(resized_image)
                
                # Apply color augmentation and convert to tensor
                # Note: Apply augmentation to the resized image, not the native one
                augmented = color_aug(resized_image)
                inputs[(name + "_aug", frame_id, scale)] = self.to_tensor(augmented)

        # Step 3: Preprocess geometric data for each scale
        for scale in range(self.num_scales):
            self._preprocess_geometric_data(inputs, scale)
        
        # Step 4: In test mode, also process native resolution (-1) geometric data
        # This matches MonoLDP's behavior where -1 versions are kept in test mode
        if self.is_test:
            for struct_type in ["plane", "line"]:
                struct_key = (struct_type, 0, -1)
                if struct_key in inputs:
                    # Convert to numpy array
                    struct_map_array = np.expand_dims(np.array(inputs[struct_key]), 0)
                    # Store both float and long versions (matching MonoLDP)
                    inputs[(struct_type + "_float", 0, -1)] = torch.from_numpy(struct_map_array).float()
                    inputs[(struct_type, 0, -1)] = torch.from_numpy(struct_map_array).long()

    def _cleanup_native_resolution(self, inputs):
        """Remove native resolution (-1) data that's no longer needed.
        
        Args:
            inputs: Dictionary to clean up
        """
        # Remove native resolution color images (kept only in test mode)
        for frame_id in self.frame_idxs:
            key = ("color", frame_id, -1)
            if key in inputs:
                if self.is_test:
                    # 确保 PIL Image 被正确转换为 tensor
                    # 使用 transforms.ToTensor() 直接转换，避免序列化问题
                    if isinstance(inputs[key], Image.Image):
                        # 直接使用 transforms.ToTensor() 转换，避免使用 self.to_tensor（可能序列化问题）
                        to_tensor = transforms.ToTensor()
                        inputs[key] = to_tensor(inputs[key])
                    elif not isinstance(inputs[key], torch.Tensor):
                        # 如果不是 PIL Image 也不是 Tensor，尝试转换
                        to_tensor = transforms.ToTensor()
                        inputs[key] = to_tensor(inputs[key])
                else:
                    del inputs[key]

        # Remove native resolution geometric data (not needed after preprocessing)
        # 注意：在测试模式下，plane 和 line 的 -1 版本会被保留并转换为 tensor（在 preprocess 中处理）
        # 在非测试模式下，删除 -1 版本（只保留 scale 0, 1, 2... 的版本）
        if self.load_plane and not self.is_test:
            if ("plane", 0, -1) in inputs:
                del inputs[("plane", 0, -1)]

        if self.load_line and not self.is_test:
            if ("line", 0, -1) in inputs:
                del inputs[("line", 0, -1)]

    def _add_stereo_extrinsics(self, inputs, side, do_flip):
        """Add stereo camera extrinsics if using stereo pairs.
        
        Args:
            inputs: Dictionary to add extrinsics to
            side: Stereo side ('l' or 'r')
            do_flip: Whether images are flipped
        """
        if "s" in self.frame_idxs:
            # Create stereo transformation matrix
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1  # Baseline = 0.1m
            inputs["stereo_T"] = torch.from_numpy(stereo_T)

    def __len__(self):
        """Return dataset size."""
        return len(self.filenames)

    def __getitem__(self, index):
        """Load and preprocess a single training item.
        
        Returns a dictionary containing:
        - ("color", frame_id, scale): Raw color images at different scales
        - ("color_aug", frame_id, scale): Augmented color images
        - ("K", scale): Camera intrinsics matrix at each scale
        - ("norm_pix_coords", scale): Normalized pixel coordinates
        - "depth_gt": Ground truth depth (if available)
        - ("plane", 0, scale): Plane segmentation maps
        - ("plane_keysets", 0, scale): Plane keysets for regularization
        - ("line", 0, scale): Line segmentation maps
        - ("line_keysets", 0, scale): Line keysets for regularization
        - "stereo_T": Stereo camera extrinsics (if using stereo)
        
        Args:
            index: Dataset index
            
        Returns:
            dict: Preprocessed training item
        """
        inputs = {}

        # Decide on augmentations (only during training)
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        # Parse filename to get folder, frame_index, and side
        folder, frame_index, side = self._parse_filename(index)

        # Load frame images
        frame_images = self._load_frame_images(folder, frame_index, side, do_flip)
        inputs.update(frame_images)

        # Compute camera intrinsics for all scales
        intrinsics = self._compute_camera_intrinsics(do_flip)
        inputs.update(intrinsics)

        # Create color augmentation transform
        color_aug = self._create_color_augmentation(do_color_aug)

        # Load auxiliary data (depth, plane, line)
        self._load_auxiliary_data(folder, frame_index, side, do_flip, inputs)

        # Preprocess all data (resize, augment, generate keysets)
        self.preprocess(inputs, color_aug)

        # Cleanup native resolution data
        self._cleanup_native_resolution(inputs)
        
        # 额外检查：确保在测试模式下，所有 ("color", i, -1) 都是 tensor
        # 这在多进程环境下特别重要
        if self.is_test:
            for frame_id in self.frame_idxs:
                key = ("color", frame_id, -1)
                if key in inputs:
                    if isinstance(inputs[key], Image.Image):
                        # 如果仍然是 PIL Image，强制转换
                        to_tensor = transforms.ToTensor()
                        inputs[key] = to_tensor(inputs[key])
                    elif not isinstance(inputs[key], torch.Tensor):
                        # 如果不是 tensor，也转换
                        to_tensor = transforms.ToTensor()
                        inputs[key] = to_tensor(inputs[key])

        # Add stereo extrinsics if needed
        self._add_stereo_extrinsics(inputs, side, do_flip)

        return inputs

    # Abstract methods that must be implemented by subclasses
    def get_color(self, folder, frame_index, side, do_flip):
        """Load color image.
        
        Args:
            folder: Scene/folder name
            frame_index: Frame index
            side: Stereo side ('l' or 'r'), None if monocular
            do_flip: Whether to horizontally flip
            
        Returns:
            PIL Image
        """
        raise NotImplementedError

    def check_depth(self):
        """Check if depth ground truth is available.
        
        Returns:
            bool: True if depth files exist
        """
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        """Load depth ground truth.
        
        Args:
            folder: Scene/folder name
            frame_index: Frame index
            side: Stereo side ('l' or 'r'), None if monocular
            do_flip: Whether to horizontally flip
            
        Returns:
            numpy array: Depth map (H, W)
        """
        raise NotImplementedError

    def check_plane(self):
        """Check if plane segmentation is available.
        
        Returns:
            bool: True if plane files exist
        """
        raise NotImplementedError

    def get_plane(self, folder, frame_index, side, do_flip):
        """Load plane segmentation map.
        
        Args:
            folder: Scene/folder name
            frame_index: Frame index
            side: Stereo side ('l' or 'r'), None if monocular
            do_flip: Whether to horizontally flip
            
        Returns:
            PIL Image: Plane segmentation map
        """
        raise NotImplementedError

    def check_line(self):
        """Check if line segmentation is available.
        
        Returns:
            bool: True if line files exist
        """
        raise NotImplementedError

    def get_line(self, folder, frame_index, side, do_flip):
        """Load line segmentation map.
        
        Args:
            folder: Scene/folder name
            frame_index: Frame index
            side: Stereo side ('l' or 'r'), None if monocular
            do_flip: Whether to horizontally flip
            
        Returns:
            PIL Image: Line segmentation map
        """
        raise NotImplementedError
