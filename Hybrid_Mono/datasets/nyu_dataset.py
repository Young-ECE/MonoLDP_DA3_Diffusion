from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import cv2

from .mono_dataset import MonoDataset


class NYUDataset(MonoDataset):
    """NYU Depth Dataset loader for indoor depth estimation.
    
    The NYU dataset contains RGB-D images of indoor scenes.
    This loader supports:
    - RGB images (640x480)
    - Depth ground truth
    - Plane segmentation maps (optional)
    - Line segmentation maps (optional)
    
    Dataset structure:
    - Training: scene_name/frame_index.jpg (RGB), frame_index.png (depth)
    - Testing: scene_name/XXXXX_colors.png (RGB), XXXXX_depth.png (depth)
    
    Edge cropping: 16 pixels from each edge (removes invalid regions)
    Full resolution after cropping: 608x448 (640-32, 480-32)
    """
    
    # Dataset-specific constants
    edge_crop = 16
    full_res_shape = (640 - 2*edge_crop, 480 - 2*edge_crop)  # (608, 448)
    default_crop = [40 - edge_crop, 601 - edge_crop, 44 - edge_crop, 471 - edge_crop]
    min_depth = 0.01
    max_depth = 10.0

    def __init__(self, *args, **kwargs):
        """Initialize NYU dataset.
        
        Sets up camera intrinsics matrix (normalized by image size).
        """
        super(NYUDataset, self).__init__(*args, **kwargs)

        # Compute normalized camera intrinsics
        # NOTE: Intrinsics are normalized by the cropped image size
        w, h = self.full_res_shape

        # Original NYU camera intrinsics (before cropping)
        fx_original = 5.1885790117450188e+02
        fy_original = 5.1946961112127485e+02
        cx_original = 3.2558244941119034e+02
        cy_original = 2.5373616633400465e+02

        # Normalize by cropped image size and adjust for cropping
        fx = fx_original / w
        fy = fy_original / h
        cx = (cx_original - self.edge_crop) / w
        cy = (cy_original - self.edge_crop) / h

        # Create intrinsics matrix (4x4 homogeneous)
        self.K = np.array([
            [fx, 0,  cx, 0],
            [0,  fy, cy, 0],
            [0,  0,  1,  0],
            [0,  0,  0,  1]
        ], dtype=np.float32)

    def _get_file_info(self, index=0):
        """Parse filename to extract scene name and frame index.
        
        Args:
            index: Filename index (default: 0 for check methods)
            
        Returns:
            tuple: (scene_name, frame_index_str)
        """
        line = self.filenames[index].split()
        scene_name = line[0]
        frame_index_str = line[1] if len(line) >= 2 else "0"
        return scene_name, frame_index_str

    def _format_frame_index(self, frame_index):
        """Format frame index for file paths.
        
        Training: use frame_index as-is (string)
        Testing: format as 5-digit zero-padded (e.g., "00123")
        
        Args:
            frame_index: Frame index (int or string)
            
        Returns:
            str: Formatted frame index
        """
        if self.is_test:
            return "{:05d}".format(int(frame_index))
        else:
            return str(frame_index)

    def _apply_edge_crop(self, image):
        """Apply edge cropping to remove invalid regions.
        
        Crops 16 pixels from each edge: (16, 16, 624, 464)
        
        Args:
            image: PIL Image
            
        Returns:
            PIL Image: Cropped image
        """
        return image.crop((
            self.edge_crop,
            self.edge_crop,
            640 - self.edge_crop,
            480 - self.edge_crop
        ))

    def _apply_flip(self, image):
        """Apply horizontal flip if needed.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Image: Flipped image (same type as input)
        """
        if isinstance(image, np.ndarray):
            return np.fliplr(image)
        else:
            return image.transpose(pil.FLIP_LEFT_RIGHT)

    def get_image_path(self, folder, frame_index, side):
        """Get path to RGB image file.
        
        Training format: folder/frame_index.jpg
        Testing format: folder/XXXXX_colors.png
        
        Args:
            folder: Scene/folder name
            frame_index: Frame index
            side: Not used for NYU (always None)
            
        Returns:
            str: Image file path
        """
        frame_index_str = self._format_frame_index(frame_index)
        
        if self.is_test:
            image_path = os.path.join(
                self.data_path, folder, frame_index_str + "_colors.png"
            )
        else:
            image_path = os.path.join(
                self.data_path, folder, frame_index_str + ".jpg"
            )
        return image_path

    def get_color(self, folder, frame_index, side, do_flip):
        """Load and preprocess color image.
        
        Steps:
        1. Load image from disk
        2. Apply edge cropping
        3. Apply horizontal flip if needed
        
        Args:
            folder: Scene/folder name
            frame_index: Frame index
            side: Not used for NYU (monocular dataset)
            do_flip: Whether to horizontally flip
            
        Returns:
            PIL Image: Preprocessed color image
        """
        image_path = self.get_image_path(folder, frame_index, side)
        color = self.loader(image_path)
        
        # Apply edge cropping
        color = self._apply_edge_crop(color)
        
        # Apply flip if needed
        if do_flip:
            color = self._apply_flip(color)
        
        return color

    def check_depth(self):
        """Check if depth ground truth files exist.
        
        Returns:
            bool: True if depth files are available
        """
        scene_name, frame_index_str = self._get_file_info(0)
        frame_index_formatted = self._format_frame_index(frame_index_str)
        
        if self.is_test:
            depth_filename = os.path.join(
                self.data_path, scene_name, frame_index_formatted + "_depth.png"
            )
        else:
            depth_filename = os.path.join(
                self.data_path, scene_name, frame_index_str + ".png"
            )
        return os.path.isfile(depth_filename)

    def get_depth(self, folder, frame_index, side, do_flip):
        """Load and preprocess depth ground truth.
        
        Training: depth values are stored in mm, divided by 25.6
        Testing: depth values are stored in mm, divided by 1000
        
        Args:
            folder: Scene/folder name
            frame_index: Frame index
            side: Not used for NYU
            do_flip: Whether to horizontally flip
            
        Returns:
            numpy array: Depth map in meters (H, W)
        """
        frame_index_str = self._format_frame_index(frame_index)
        
        if self.is_test:
            depth_path = os.path.join(
                self.data_path, folder, frame_index_str + "_depth.png"
            )
            # Test depth is in mm, convert to meters
            depth_scale = 1000.0
        else:
            depth_path = os.path.join(
                self.data_path, folder, str(frame_index) + ".png"
            )
            # Training depth is in special units, convert to meters
            depth_scale = 25.6

        # Load depth image
        depth_gt = pil.open(depth_path)
        
        # Apply edge cropping
        depth_gt = self._apply_edge_crop(depth_gt)
        
        # Convert to numpy array and scale to meters
        depth_gt = np.array(depth_gt).astype(np.float32) / depth_scale

        # Apply flip if needed
        if do_flip:
            depth_gt = self._apply_flip(depth_gt)

        return depth_gt

    def check_plane(self):
        """Check if plane segmentation files exist.
        
        Returns:
            bool: True if plane files are available
        """
        scene_name, frame_index_str = self._get_file_info(0)
        frame_index_formatted = self._format_frame_index(frame_index_str)
        
        if self.is_test:
            plane_filename = os.path.join(
                self.data_path, scene_name, frame_index_formatted + "_seg.png"
            )
        else:
            plane_filename = os.path.join(
                self.data_path, scene_name, frame_index_str + "_seg.png"
            )
        return os.path.isfile(plane_filename)

    def get_plane_path(self, folder, frame_index, side):
        """Get path to plane segmentation file.
        
        Training format: folder/frame_index_seg.png
        Testing format: folder/XXXXX_seg.png
        
        Args:
            folder: Scene/folder name
            frame_index: Frame index
            side: Not used for NYU
            
        Returns:
            str: Plane segmentation file path
        """
        frame_index_str = self._format_frame_index(frame_index)
        
        if self.is_test:
            plane_path = os.path.join(
                self.data_path, folder, frame_index_str + "_seg.png"
            )
        else:
            plane_path = os.path.join(
                self.data_path, folder, frame_index_str + "_seg.png"
            )
        return plane_path

    def get_plane(self, folder, frame_index, side, do_flip):
        """Load and preprocess plane segmentation map.
        
        Args:
            folder: Scene/folder name
            frame_index: Frame index
            side: Not used for NYU
            do_flip: Whether to horizontally flip
            
        Returns:
            PIL Image: Plane segmentation map
        """
        plane_path = self.get_plane_path(folder, frame_index, side)
        plane = pil.open(plane_path)
        
        # Apply edge cropping
        plane = self._apply_edge_crop(plane)
        
        # Apply flip if needed
        if do_flip:
            plane = self._apply_flip(plane)
        
        return plane

    def check_line(self):
        """Check if line segmentation files exist.
        
        Returns:
            bool: True if line files are available
        """
        scene_name, frame_index_str = self._get_file_info(0)
        frame_index_formatted = self._format_frame_index(frame_index_str)
        
        if self.is_test:
            line_filename = os.path.join(
                self.data_path, scene_name, frame_index_formatted + "_line.png"
            )
        else:
            line_filename = os.path.join(
                self.data_path, scene_name, frame_index_str + "_line.png"
            )
        return os.path.isfile(line_filename)

    def get_line_path(self, folder, frame_index, side):
        """Get path to line segmentation file.
        
        Training format: folder/frame_index_line.png
        Testing format: folder/XXXXX_line.png
        
        Args:
            folder: Scene/folder name
            frame_index: Frame index
            side: Not used for NYU
            
        Returns:
            str: Line segmentation file path
        """
        frame_index_str = self._format_frame_index(frame_index)
        
        if self.is_test:
            line_path = os.path.join(
                self.data_path, folder, frame_index_str + "_line.png"
            )
        else:
            line_path = os.path.join(
                self.data_path, folder, frame_index_str + "_line.png"
            )
        return line_path

    def get_line(self, folder, frame_index, side, do_flip):
        """Load and preprocess line segmentation map.
        
        Args:
            folder: Scene/folder name
            frame_index: Frame index
            side: Not used for NYU
            do_flip: Whether to horizontally flip
            
        Returns:
            PIL Image: Line segmentation map
        """
        line_path = self.get_line_path(folder, frame_index, side)
        line = pil.open(line_path)
        
        # Apply edge cropping
        line = self._apply_edge_crop(line)
        
        # Apply flip if needed
        if do_flip:
            line = self._apply_flip(line)
        
        return line

    def get_norm_pix_coords(self):
        """Get normalized pixel coordinates for full resolution.
        
        Computes normalized pixel coordinates used for 3D point backprojection.
        This is a utility method, not used in the main data loading pipeline.
        
        Returns:
            numpy array: Normalized pixel coordinates (2, H, W)
        """
        w, h = self.full_res_shape

        Us, Vs = np.meshgrid(
            np.linspace(0, w - 1, w, dtype=np.float32),
            np.linspace(0, h - 1, h, dtype=np.float32),
            indexing='xy'
        )
        Us /= w  # Normalize to [0, 1]
        Vs /= h  # Normalize to [0, 1]
        
        norm_pix_coords = np.stack((
            (Us - self.K[0, 2]) / self.K[0, 0],  # (u - cx) / fx
            (Vs - self.K[1, 2]) / self.K[1, 1]   # (v - cy) / fy
        ), axis=0)

        return norm_pix_coords
