"""
Dataset registry system for managing dataset-specific discovery strategies
"""

from typing import Dict, Type, List


class DatasetRegistry:
    """
    Registry for dataset-specific discovery strategies.
    Each dataset type registers how to discover its samples from file structure.
    """
    _registry: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a dataset discovery strategy
        
        Args:
            name: Unique name for the dataset (e.g., 'sceneflow', 'kitti12')
            
        Example:
            @DatasetRegistry.register('sceneflow')
            class SceneFlowDiscovery(BaseDiscoveryStrategy):
                ...
        """
        def decorator(discovery_class):
            if name in cls._registry:
                raise ValueError(f"Dataset '{name}' already registered")
            cls._registry[name] = discovery_class
            return discovery_class
        return decorator
    
    @classmethod
    def get_discovery_strategy(cls, name: str):
        """
        Get discovery strategy class for a dataset
        
        Args:
            name: Dataset name
            
        Returns:
            Discovery strategy class
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Dataset '{name}' not registered. "
                f"Available datasets: {available}"
            )
        return cls._registry[name]
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered dataset names"""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a dataset is registered"""
        return name in cls._registry
