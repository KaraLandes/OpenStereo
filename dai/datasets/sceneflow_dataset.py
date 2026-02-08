"""
SceneFlow dataset implementation
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import logging
import time

from .base_dataset import BaseStereoDataset

logger = logging.getLogger(__name__)

# Global counter for profiling
_profile_counter = 0


class SceneFlowDataset(BaseStereoDataset):
    """
    SceneFlow dataset (Monkaa, Driving, FlyingThings3D).
    Handles cleanpass/finalpass images and .pfm disparity files.
    """
    
    def __init__(self, samples, config, split='train'):
        """
        Initialize SceneFlow dataset.
        
        Config keys:
            - resolution: [H, W] target resolution
            - target_source: 'ground_truth' or 'pseudo'
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
        Load disparity from .pfm file.
        
        Args:
            path: Path to .pfm file
            
        Returns:
            numpy array [H, W], float32
        """
        disp = self._read_pfm(path)
        
        if np.isnan(disp).any():
            logger.warning(f"NaN values found in disparity: {path}")
            disp = np.nan_to_num(disp, nan=0.0)
        
        return disp.astype(np.float32)
    
    @staticmethod
    def _read_pfm(file_path: str) -> np.ndarray:
        """
        Read PFM file.
        
        Args:
            file_path: Path to .pfm file
            
        Returns:
            numpy array
        """
        with open(file_path, 'rb') as f:
            header = f.readline().decode('utf-8').rstrip()
            
            if header == 'PF':
                color = True
            elif header == 'Pf':
                color = False
            else:
                raise ValueError(f'Invalid PFM file: {file_path}')
            
            dim_match = f.readline().decode('utf-8')
            width, height = map(int, dim_match.split())
            
            scale = float(f.readline().decode('utf-8').rstrip())
            endian = '<' if scale < 0 else '>'
            scale = abs(scale)
            
            data = np.fromfile(f, endian + 'f')
            
            if color:
                shape = (height, width, 3)
            else:
                shape = (height, width)
            
            data = np.reshape(data, shape)
            data = np.flipud(data)
            
            return data
    
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
        Compute valid pixel mask for SceneFlow.
        
        Args:
            disparity: Disparity numpy array [H, W]
            
        Returns:
            Valid mask [H, W] (bool numpy array)
        """
        valid = (disparity > 0) & (disparity < 512)
        return valid.astype(bool)
