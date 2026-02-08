"""
Base discovery strategy and dataset-specific implementations
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path


class BaseDiscoveryStrategy(ABC):
    """
    Base class for dataset-specific file discovery.
    Each dataset implements this to tell splitter how to find samples.
    """
    
    @abstractmethod
    def discover_samples(self, root: Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Discover all samples in the dataset from its file structure.
        
        Args:
            root: Root directory of dataset
            config: Dataset-specific configuration (subsets, pass_type, etc.)
            
        Returns:
            List of sample dictionaries with keys:
            - 'left': Path to left image
            - 'right': Path to right image
            - 'disparity': Path to disparity map (None if pseudo-target)
            - 'metadata': Dict with additional info (scene, frame_id, etc.)
        """
        pass
    
    @abstractmethod
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        Validate that a sample has all required files.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            True if sample is valid
        """
        pass
