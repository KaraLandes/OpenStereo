"""
KITTI 2015 stereo dataset implementation
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch

from .base_dataset import BaseStereoDataset

logger = logging.getLogger(__name__)


class KITTI15Dataset(BaseStereoDataset):
    """
    KITTI 2015 stereo dataset.
    
    Loads colored stereo pairs and disparity maps from KITTI 2015.
    Disparity maps are stored as 16-bit PNG with values scaled by 256.
    
    Difference from KITTI12: Uses image_2/image_3 instead of image_0/image_1
    """
    
    def _load_image(self, path: str) -> np.ndarray:
        """
        Load RGB image from PNG file.
        
        Args:
            path: Path to image file
            
        Returns:
            Image as numpy array [H, W, 3] in range [0, 255]
        """
        img = Image.open(path).convert('RGB')
        return np.array(img, dtype=np.float32)
    
    def _load_disparity(self, path: str) -> np.ndarray:
        """
        Load disparity map from PNG file.
        
        KITTI disparity maps are stored as 16-bit PNG where:
        - disp_value = pixel_value / 256.0
        - 0 means invalid/unknown disparity
        
        Args:
            path: Path to disparity file
            
        Returns:
            Disparity map as numpy array [H, W]
        """
        disp_img = Image.open(path)
        disp = np.array(disp_img, dtype=np.float32)
        
        # KITTI stores disparity scaled by 256
        disp = disp / 256.0
        
        return disp
    
    def _compute_valid_mask(self, disparity: np.ndarray) -> np.ndarray:
        """
        Compute valid pixel mask for KITTI15.
        
        Args:
            disparity: Disparity numpy array [H, W]
            
        Returns:
            Valid mask [H, W] (bool numpy array)
        """
        valid = (disparity > 0) & (disparity < 256)
        return valid.astype(bool)
    
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
        
        import cv2
        left_img = cv2.resize(left_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        right_img = cv2.resize(right_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        disparity = cv2.resize(disparity, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        disparity = disparity * (target_w / orig_w)
        
        return left_img, right_img, disparity
    
