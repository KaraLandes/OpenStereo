"""
Sequential + Presaved Pseudo-GT dataset wrapper.

Combines sequential temporal data with presaved FoundationStereo disparities.
Applies transforms with shared crop across all frames, then loads presaved
disparities for each frame using the crop coordinates.
"""

import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from torch.utils.data import Dataset

from .transforms import Compose

logger = logging.getLogger(__name__)


class SequentialPresavedDataset(Dataset):
    """
    Wraps a sequential dataset (SequentialSceneFlowDataset) and loads
    presaved pseudo-GT disparities for each frame after transforms.
    
    Data flow:
    1. SequentialSceneFlowDataset returns raw sequence of frames
    2. Compose transforms applied with shared crop across all frames
    3. For each frame, presaved disparity loaded using crop coordinates
    4. Returns sequence with real presaved disparities
    
    File naming convention:
        pseudo_d_{y}_{x}_{frame_id}.npy
    """
    
    def __init__(self, sequential_dataset: Dataset, transforms: Compose,
                 resize_size: Optional[List[int]] = None):
        """
        Args:
            sequential_dataset: SequentialSceneFlowDataset instance
            transforms: Compose transform pipeline
            resize_size: [height, width] of final disparity size (after resize)
        """
        self.sequential_dataset = sequential_dataset
        self.transforms = transforms
        self.resize_size = resize_size
        
        # Expose attributes for pipeline compatibility
        self.target_source = getattr(sequential_dataset, 'target_source', 'provided')
        
        logger.info(
            f"Initialized SequentialPresavedDataset wrapping "
            f"{sequential_dataset.__class__.__name__} with {len(self)} sequences"
        )
    
    def __len__(self) -> int:
        return len(self.sequential_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get sequence with presaved pseudo-GT disparities.
        
        Returns:
            Dict with 'sequence' (list of frame dicts) and 'sequence_metadata'
        """
        # Get raw sequence
        data = self.sequential_dataset[idx]
        
        # Store left paths in metadata before transforms
        for i, frame in enumerate(data['sequence']):
            frame_idx = self.sequential_dataset.sequences[idx][i]
            left_path = self._get_left_path(frame_idx)
            if left_path and 'metadata' in frame:
                frame['metadata']['left_path'] = left_path
        
        # Apply transforms (Compose handles sequential data with shared crop)
        data = self.transforms(data)
        
        # Load presaved disparities for each frame
        for frame in data['sequence']:
            self._load_frame_presaved(frame)
        
        return data
    
    def _get_left_path(self, frame_idx: int) -> Optional[str]:
        """Get left image path for a frame index in the base dataset."""
        base = self.sequential_dataset.base_dataset
        if hasattr(base, 'samples') and frame_idx < len(base.samples):
            return base.samples[frame_idx].get('left', None)
        return None
    
    def _load_frame_presaved(self, frame: Dict[str, Any]):
        """
        Load presaved disparity for a single frame using its crop_coords.
        Modifies frame dict in-place.
        """
        crop_coords = frame.get('crop_coords', None)
        metadata = frame.get('metadata', {})
        
        if not crop_coords or 'y' not in crop_coords:
            frame['needs_generation'] = True
            return
        
        try:
            # Get frame info
            frame_id = metadata.get('frame_id', '00000')
            left_path = metadata.get('left_path', None)
            
            if left_path is None:
                frame['needs_generation'] = True
                return
            
            scene_dir = Path(left_path).parent
            y = crop_coords['y']
            x = crop_coords['x']
            
            # Load presaved disparity
            presaved_filename = f"pseudo_d_{y}_{x}_{frame_id}.npy"
            presaved_path = scene_dir / presaved_filename
            
            if not presaved_path.exists():
                frame['needs_generation'] = True
                logger.debug(f"Cache miss: {presaved_path}")
                return
            
            disparity = np.load(str(presaved_path)).squeeze()
            disparity = disparity.astype(np.float32)
            
            # Verify size if resize_size is set
            if self.resize_size is not None:
                expected_h, expected_w = self.resize_size
                actual_h, actual_w = disparity.shape[-2:]
                if actual_h != expected_h or actual_w != expected_w:
                    logger.warning(
                        f"Presaved disparity size mismatch: "
                        f"expected {expected_h}x{expected_w}, got {actual_h}x{actual_w}"
                    )
            
            # Convert to tensor if frame disparity is a tensor
            if isinstance(frame.get('disparity'), torch.Tensor):
                disparity = torch.from_numpy(disparity).float()
            
            # Replace dummy disparity with presaved
            frame['disparity'] = disparity
            frame['disparity_foundationstereo'] = disparity
            
            # Update valid mask
            frame['valid_mask'] = (disparity > 0) & (disparity < 512)
            frame['needs_generation'] = False
            
        except Exception as e:
            frame['needs_generation'] = True
            logger.error(f"Error loading presaved disparity: {e}")
