"""
Postprocessing wrapper dataset that applies transforms to base datasets
"""

import logging
from typing import Dict, Any
from torch.utils.data import Dataset

from .transforms import Compose

logger = logging.getLogger(__name__)


class PostprocessingDataset(Dataset):
    """
    Wrapper dataset that applies transforms to samples from base dataset.
    
    Flow:
    1. Base dataset (SceneFlow/KITTI12) loads raw data [0-255]
    2. PostprocessingDataset applies transforms (crop, normalize, etc.)
    3. DataLoader batches the transformed samples
    """
    
    def __init__(self, base_dataset: Dataset, transforms: Compose):
        """
        Args:
            base_dataset: Base dataset (SceneFlow, KITTI12, etc.)
            transforms: Compose object with transforms to apply
        """
        self.base_dataset = base_dataset
        self.transforms = transforms
        
        logger.info(
            f"Initialized PostprocessingDataset wrapping {base_dataset.__class__.__name__} "
            f"with {len(transforms.transforms)} transforms"
        )
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get sample from base dataset and apply transforms.
        
        Args:
            idx: Sample index
            
        Returns:
            Transformed sample dict with:
                - left: [3, H, W] normalized tensor
                - right: [3, H, W] normalized tensor
                - disparity: [H, W] tensor
                - valid_mask: [H, W] bool tensor
                - metadata: dict
        """
        # Get raw sample from base dataset
        sample = self.base_dataset[idx]
        
        # Apply transforms
        sample = self.transforms(sample)
        
        return sample
