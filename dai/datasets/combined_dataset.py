"""
Combined dataset that merges multiple name-specific datasets
"""

from typing import List, Dict, Any
from torch.utils.data import Dataset
import logging

from .base_dataset import BaseStereoDataset

logger = logging.getLogger(__name__)


class CombinedStereoDataset(Dataset):
    """
    Parent dataset that combines multiple name-specific datasets.
    Handles indexing across datasets and delegates loading to child datasets.
    """
    
    def __init__(self, datasets: List[BaseStereoDataset]):
        """
        Initialize combined dataset.
        
        Args:
            datasets: List of name-specific dataset instances
        """
        self.datasets = datasets
        self.dataset_lengths = [len(ds) for ds in datasets]
        self.cumulative_lengths = self._compute_cumulative_lengths()
        
        total_samples = sum(self.dataset_lengths)
        logger.info(
            f"Initialized CombinedStereoDataset with {len(datasets)} datasets "
            f"and {total_samples} total samples"
        )
        
        for i, ds in enumerate(datasets):
            logger.info(f"  Dataset {i}: {ds.__class__.__name__} with {len(ds)} samples")
    
    def _compute_cumulative_lengths(self) -> List[int]:
        """Compute cumulative lengths for indexing"""
        cumulative = []
        total = 0
        for length in self.dataset_lengths:
            cumulative.append(total)
            total += length
        return cumulative
    
    def __len__(self) -> int:
        """Return total number of samples across all datasets"""
        return sum(self.dataset_lengths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get sample by global index.
        Maps global index to (dataset_idx, local_idx) and delegates to child dataset.
        
        Args:
            idx: Global index across all datasets
            
        Returns:
            Sample dictionary from appropriate child dataset
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        dataset_idx = self._get_dataset_idx(idx)
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        
        return self.datasets[dataset_idx][local_idx]
    
    def _get_dataset_idx(self, global_idx: int) -> int:
        """
        Find which dataset a global index belongs to.
        
        Args:
            global_idx: Global index
            
        Returns:
            Dataset index
        """
        for i in range(len(self.datasets) - 1, -1, -1):
            if global_idx >= self.cumulative_lengths[i]:
                return i
        return 0
    
    def get_dataset_info(self) -> List[Dict[str, Any]]:
        """
        Get information about constituent datasets.
        
        Returns:
            List of dicts with dataset info
        """
        info = []
        for i, ds in enumerate(self.datasets):
            info.append({
                'index': i,
                'name': ds.__class__.__name__,
                'num_samples': len(ds),
                'start_idx': self.cumulative_lengths[i],
                'end_idx': self.cumulative_lengths[i] + len(ds) - 1
            })
        return info
