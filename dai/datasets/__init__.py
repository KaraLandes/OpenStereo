"""
Dataset implementations for stereo training with multi-dataset support
"""

from .registry import DatasetRegistry
from .splitter import DatasetSplitter
from .base_dataset import BaseStereoDataset
from .combined_dataset import CombinedStereoDataset
from .sceneflow_dataset import SceneFlowDataset
from .sequential_sceneflow_dataset import SequentialSceneFlowDataset
from .kitti12_dataset import KITTI12Dataset
from .kitti15_dataset import KITTI15Dataset
from .irs_dataset import IRSDataset
from .postprocessing_dataset import PostprocessingDataset
from .collate import sequential_collate_fn
from .transforms import build_transforms, Compose
from .presaved_pseudo_dataset import PresavedPseudoGTDataset
from .sequential_presaved_dataset import SequentialPresavedDataset

# Import discovery strategies to populate registry
# This must happen after DatasetRegistry is imported
from .discovery_strategies import SceneFlowDiscovery, KITTI12Discovery, KITTI15Discovery, IRSDiscovery

__all__ = [
    'DatasetRegistry',
    'DatasetSplitter',
    'BaseStereoDataset',
    'CombinedStereoDataset',
    'SceneFlowDataset',
    'SequentialSceneFlowDataset',
    'KITTI12Dataset',
    'KITTI15Dataset',
    'IRSDataset',
    'PostprocessingDataset',
    'PresavedPseudoGTDataset',
    'sequential_collate_fn',
    'build_transforms',
    'Compose',
]
