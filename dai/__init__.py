"""
DAI (Data, Architecture, Infrastructure) Package
Training infrastructure for LightStereo-S model with multi-dataset support
"""

from .config import load_config, save_config
from .datasets import (
    DatasetRegistry,
    DatasetSplitter,
    BaseStereoDataset,
    CombinedStereoDataset,
    SceneFlowDataset,
)

__version__ = "0.1.0"
__all__ = [
    'load_config',
    'save_config',
    'DatasetRegistry',
    'DatasetSplitter',
    'BaseStereoDataset',
    'CombinedStereoDataset',
    'SceneFlowDataset',
]
