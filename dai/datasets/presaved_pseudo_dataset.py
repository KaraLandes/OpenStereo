"""
Wrapper dataset for loading presaved pseudo-GT disparities based on crop coordinates.

This dataset wraps a base dataset and handles the two-pass loading:
1. First pass: Load images, apply transforms to get crop coordinates
2. Second pass: Load presaved pseudo disparity using crop coordinates
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PresavedPseudoGTDataset(Dataset):
    """
    Wrapper dataset that loads presaved pseudo-GT disparities based on crop coordinates.
    
    Flow:
    1. Base dataset loads raw images
    2. Transforms are applied (including RandomCrop which stores coordinates)
    3. This wrapper loads the presaved disparity matching the crop coordinates
    4. Returns final sample with correct presaved disparity
    
    File naming convention:
        pseudo_d_{y}_{x}_{frame_id}.npy
        
    Example:
        For frame l_00001.png cropped at (y=28, x=190):
        -> pseudo_d_28_190_00001.npy
    """
    
    def __init__(self, base_dataset: Dataset, transforms, resize_size: list = None):
        """
        Args:
            base_dataset: Base dataset (e.g., IRSDataset)
            transforms: Transform pipeline (Compose object)
            resize_size: [height, width] of final disparity size (after resize)
        """
        self.base_dataset = base_dataset
        self.transforms = transforms
        self.resize_size = resize_size
        
        logger.info(
            f"Initialized PresavedPseudoGTDataset wrapping {base_dataset.__class__.__name__} "
            f"with {len(transforms.transforms)} transforms"
        )
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get sample with presaved pseudo-GT disparity.
        
        Args:
            idx: Sample index
            
        Returns:
            Transformed sample dict with presaved disparity
        """
        # Get raw sample from base dataset
        sample = self.base_dataset[idx]
        
        # Store original left path before transforms for presaved loading
        if 'metadata' in sample:
            # Try to get left path from base dataset
            left_path = self._get_sample_left_path(idx)
            if left_path:
                sample['metadata']['left_path'] = left_path
        
        # Apply transforms (this generates crop coordinates)
        sample = self.transforms(sample)
        
        # Load presaved disparity if crop coordinates are available
        if 'crop_coords' in sample and sample['crop_coords']:
            try:
                presaved_disp = self._load_presaved_disparity(sample, idx)
                
                # Replace the dummy disparity with presaved one
                sample['disparity'] = presaved_disp
                
                # Update valid mask based on presaved disparity
                import torch
                if isinstance(presaved_disp, torch.Tensor):
                    sample['valid_mask'] = (presaved_disp > 0) & (presaved_disp < 512)
                else:
                    sample['valid_mask'] = (presaved_disp > 0) & (presaved_disp < 512)
                
                sample['needs_generation'] = False
                
            except FileNotFoundError as e:
                # Flag for trainer to generate and save on-the-fly
                sample['needs_generation'] = True
                logger.debug(f"Cache miss: {e}")
            except Exception as e:
                sample['needs_generation'] = True
                logger.error(f"Error loading presaved disparity: {e}. Flagged for generation.")
        else:
            sample['needs_generation'] = False
        
        return sample
    
    def _get_sample_left_path(self, idx: int) -> str:
        """
        Get the left image path for a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Path to left image, or None if not found
        """
        # Try to access samples attribute from base dataset
        if hasattr(self.base_dataset, 'samples'):
            return self.base_dataset.samples[idx]['left']
        
        # Try CombinedStereoDataset structure
        if hasattr(self.base_dataset, 'datasets'):
            # Find which dataset this index belongs to
            cumulative = 0
            for dataset in self.base_dataset.datasets:
                if idx < cumulative + len(dataset):
                    local_idx = idx - cumulative
                    if hasattr(dataset, 'samples'):
                        return dataset.samples[local_idx]['left']
                    break
                cumulative += len(dataset)
        
        return None
    
    def _load_presaved_disparity(self, sample: Dict[str, Any], idx: int) -> Any:
        """
        Load presaved pseudo disparity based on crop coordinates.
        
        Args:
            sample: Sample dict with metadata and crop_coords
            idx: Sample index for fallback path lookup
            
        Returns:
            Presaved disparity (numpy array or tensor, matching sample format)
            
        Raises:
            FileNotFoundError: If presaved file doesn't exist
        """
        import torch
        
        # Extract metadata
        metadata = sample['metadata']
        crop_coords = sample['crop_coords']
        
        # Get frame_id from metadata
        frame_id = metadata.get('frame_id', '00000')
        
        # Get scene directory from left image path
        left_path = metadata.get('left_path', None)
        if left_path is None:
            # Fallback: get from base dataset samples
            left_path = self.base_dataset.samples[idx]['left']
        
        scene_dir = Path(left_path).parent
        
        # Build presaved disparity path
        y = crop_coords['y']
        x = crop_coords['x']
        presaved_filename = f"pseudo_d_{y}_{x}_{frame_id}.npy"
        presaved_path = scene_dir / presaved_filename
        
        # Load presaved disparity
        if not presaved_path.exists():
            raise FileNotFoundError(
                f"Presaved disparity not found: {presaved_path}\n"
                f"Expected format: pseudo_d_{{y}}_{{x}}_{{frame_id}}.npy\n"
                f"Crop coords: y={y}, x={x}, frame_id={frame_id}"
            )
        
        disparity = np.load(str(presaved_path)).squeeze()  # Remove batch dim if present
        disparity = disparity.astype(np.float32)  # Ensure float32 for training
        
        # Verify size matches expected resize dimensions
        if self.resize_size is not None:
            expected_h, expected_w = self.resize_size
            actual_h, actual_w = disparity.shape[-2:]
            if actual_h != expected_h or actual_w != expected_w:
                logger.warning(
                    f"Presaved disparity size mismatch: expected {expected_h}x{expected_w}, "
                    f"got {actual_h}x{actual_w}. File: {presaved_path}"
                )
        
        # Convert to tensor if sample disparity is a tensor
        if 'disparity' in sample and isinstance(sample['disparity'], torch.Tensor):
            disparity = torch.from_numpy(disparity).float()
        
        return disparity
