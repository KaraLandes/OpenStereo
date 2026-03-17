"""
IRS (Indoor Robotics Stereo) dataset implementation.
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import logging

from .base_dataset import BaseStereoDataset

logger = logging.getLogger(__name__)


class IRSDataset(BaseStereoDataset):
    """
    IRS dataset (Home, Office, Restaurant, Store).
    Handles PNG stereo images and .exr disparity files.
    
    Primary mode: pseudo-foundationstereo-online
    (provided mode raises NotImplementedError)
    """
    
    def __init__(self, samples, config, split='train'):
        """
        Initialize IRS dataset.
        
        Config keys:
            - resolution: [H, W] target resolution
            - target_source: 'pseudo-foundationstereo-online', 'ssl', etc.
            - return_right_disp: bool
            - augmentation: dict of augmentation configs
        """
        super().__init__(samples, config, split)
        
        self.return_right_disp = config.get('return_right_disp', False)
        self.augmentation_config = config.get('augmentation', {})
    
    def _load_image(self, path: str) -> np.ndarray:
        """
        Load image from path.
        
        Args:
            path: Path to image file
            
        Returns:
            numpy array [H, W, 3] in RGB format, float32
        """
        img = Image.open(path).convert('RGB')
        img = np.array(img, dtype=np.float32)
        return img
    
    def _load_disparity(self, path: str) -> np.ndarray:
        """
        Load disparity from file.
        
        Supports:
        - .npy files (presaved pseudo-GT disparities)
        - .exr files (original ground truth disparities)
        
        Args:
            path: Path to disparity file (.npy or .exr)
            
        Returns:
            Disparity array [H, W]
        """
        path = Path(path)
        
        if path.suffix == '.npy':
            # Load presaved pseudo-GT disparity
            disparity = np.load(str(path))
            return disparity.astype(np.float32)
        
        elif path.suffix == '.exr':
            # Load original ground truth from .exr
            from .exr_loader import load_exr_disparity
            disparity = load_exr_disparity(str(path))
            return disparity
        
        else:
            raise ValueError(
                f"Unsupported disparity file format: {path.suffix}. "
                f"Supported formats: .npy (presaved pseudo-GT), .exr (original GT)"
            )
    
    def _resize(self, left_img: np.ndarray, right_img: np.ndarray, 
                disparity: np.ndarray) -> tuple:
        """
        Resize images and disparity to target resolution.
        
        Args:
            left_img: Left image [H, W, 3]
            right_img: Right image [H, W, 3]
            disparity: Disparity map [H, W]
            
        Returns:
            Tuple of resized (left, right, disparity)
        """
        if self.resolution is None:
            return left_img, right_img, disparity
        
        target_h, target_w = self.resolution
        orig_h, orig_w = left_img.shape[:2]
        
        left_img = cv2.resize(left_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        right_img = cv2.resize(right_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        disparity = cv2.resize(disparity, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        disparity = disparity * (target_w / orig_w)
        
        return left_img, right_img, disparity
    
    def _compute_valid_mask(self, disparity: np.ndarray) -> np.ndarray:
        """
        Compute valid pixel mask for IRS.
        
        Args:
            disparity: Disparity numpy array [H, W]
            
        Returns:
            Valid mask [H, W] (bool numpy array)
        """
        valid = (disparity > 0) & (disparity < 512)
        return valid.astype(bool)
