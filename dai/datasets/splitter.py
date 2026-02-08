"""
Dataset splitter for creating train/val/test splits
"""

import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from .registry import DatasetRegistry

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """
    Configurable dataset splitter that uses registry to discover samples
    and split them into train/val/test sets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize splitter with configuration.
        
        Args:
            config: Splitter configuration with keys:
                - train: {'enabled': bool, 'proportion': float}
                - val: {'enabled': bool, 'proportion': float}
                - test: {'enabled': bool, 'proportion': float}
                - seed: int (random seed)
                - strategy: 'random', 'sequential', or 'stratified'
                - per_dataset: Optional dict of per-dataset overrides
        """
        self.config = config
        self.seed = config.get('seed', 42)
        self.strategy = config.get('strategy', 'stratified')
        
        self.train_config = config.get('train', {'enabled': True, 'proportion': 0.8})
        self.val_config = config.get('val', {'enabled': True, 'proportion': 0.1})
        self.test_config = config.get('test', {'enabled': True, 'proportion': 0.1})
        
        self._validate_proportions()
    
    def _validate_proportions(self):
        """Validate that proportions sum to 1.0"""
        total = 0.0
        if self.train_config.get('enabled', False):
            total += self.train_config.get('proportion', 0.0)
        if self.val_config.get('enabled', False):
            total += self.val_config.get('proportion', 0.0)
        if self.test_config.get('enabled', False):
            total += self.test_config.get('proportion', 0.0)
        
        if abs(total - 1.0) > 1e-6:
            logger.warning(f"Split proportions sum to {total}, not 1.0. Will normalize.")
    
    def split_dataset(
        self, 
        dataset_name: str,
        root: Path,
        dataset_config: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Discover and split a dataset.
        
        Args:
            dataset_name: Name of registered dataset (e.g., 'sceneflow')
            root: Root directory of dataset
            dataset_config: Dataset-specific configuration
            
        Returns:
            Dictionary with keys 'train', 'val', 'test' containing sample lists
        """
        discovery_class = DatasetRegistry.get_discovery_strategy(dataset_name)
        discovery = discovery_class()
        
        logger.info(f"Discovering samples for dataset: {dataset_name}")
        samples = discovery.discover_samples(Path(root), dataset_config)
        
        if len(samples) == 0:
            logger.warning(f"No samples found for dataset: {dataset_name}")
            return {'train': [], 'val': [], 'test': []}
        
        per_dataset_config = self.config.get('per_dataset', {}).get(dataset_name, {})
        
        train_prop = per_dataset_config.get('train', self.train_config.get('proportion', 0.8))
        val_prop = per_dataset_config.get('val', self.val_config.get('proportion', 0.1))
        test_prop = per_dataset_config.get('test', self.test_config.get('proportion', 0.1))
        
        total = train_prop + val_prop + test_prop
        if abs(total - 1.0) > 1e-6:
            train_prop /= total
            val_prop /= total
            test_prop /= total
        
        # Use stratified splitting for SceneFlow, simple splitting for others
        if dataset_name == 'sceneflow' and self.strategy == 'stratified':
            splits = self._split_samples_stratified_sceneflow(samples, train_prop, val_prop, test_prop)
        else:
            splits = self._split_samples(samples, train_prop, val_prop, test_prop)
        
        logger.info(
            f"Split {dataset_name}: "
            f"train={len(splits['train'])}, "
            f"val={len(splits['val'])}, "
            f"test={len(splits['test'])}"
        )
        
        return splits
    
    def _split_samples(
        self,
        samples: List[Dict[str, Any]],
        train_prop: float,
        val_prop: float,
        test_prop: float
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Split samples into train/val/test.
        
        Args:
            samples: List of samples
            train_prop: Train proportion
            val_prop: Validation proportion
            test_prop: Test proportion
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        n_samples = len(samples)
        
        if self.strategy == 'random':
            random.seed(self.seed)
            indices = list(range(n_samples))
            random.shuffle(indices)
        else:
            indices = list(range(n_samples))
        
        n_train = int(n_samples * train_prop)
        n_val = int(n_samples * val_prop)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        splits = {
            'train': [samples[i] for i in train_indices] if self.train_config.get('enabled', True) else [],
            'val': [samples[i] for i in val_indices] if self.val_config.get('enabled', True) else [],
            'test': [samples[i] for i in test_indices] if self.test_config.get('enabled', True) else [],
        }
        
        return splits
    
    def _split_samples_stratified_sceneflow(
        self,
        samples: List[Dict[str, Any]],
        train_prop: float,
        val_prop: float,
        test_prop: float
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Stratified split for SceneFlow datasets.
        
        Strategy:
        - Monkaa: Split by scene folder (entire scenes go to train/val/test)
        - Driving/FlyingThings3D: Split sequentially within each video sequence
        
        Args:
            samples: List of samples with metadata containing 'subset' and 'scene'
            train_prop: Train proportion
            val_prop: Validation proportion
            test_prop: Test proportion
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        from collections import defaultdict
        
        # Group samples by subset and scene/sequence
        subset_groups = defaultdict(lambda: defaultdict(list))
        
        for sample in samples:
            subset = sample['metadata']['subset']
            scene = sample['metadata']['scene']
            subset_groups[subset][scene].append(sample)
        
        train_samples = []
        val_samples = []
        test_samples = []
        
        random.seed(self.seed)
        
        for subset, scene_dict in subset_groups.items():
            if subset == 'Monkaa':
                # Monkaa: Split by scene folder (entire scenes to train/val/test)
                scenes = list(scene_dict.keys())
                random.shuffle(scenes)
                
                n_scenes = len(scenes)
                n_train = int(n_scenes * train_prop)
                n_val = int(n_scenes * val_prop)
                
                train_scenes = scenes[:n_train]
                val_scenes = scenes[n_train:n_train + n_val]
                test_scenes = scenes[n_train + n_val:]
                
                for scene in train_scenes:
                    train_samples.extend(scene_dict[scene])
                for scene in val_scenes:
                    val_samples.extend(scene_dict[scene])
                for scene in test_scenes:
                    test_samples.extend(scene_dict[scene])
                    
                logger.info(f"Monkaa stratified split: {len(train_scenes)} train scenes, "
                           f"{len(val_scenes)} val scenes, {len(test_scenes)} test scenes")
            
            elif subset == 'FlyingThings3D':
                # FlyingThings3D: TRAIN folder → train split, TEST folder → val/test split
                train_scenes = []
                test_scenes = []
                
                for scene, scene_samples in scene_dict.items():
                    # Check if scene is in TRAIN or TEST subdirectory
                    if '/TRAIN/' in scene or scene.startswith('TRAIN/'):
                        train_scenes.append((scene, scene_samples))
                    elif '/TEST/' in scene or scene.startswith('TEST/'):
                        test_scenes.append((scene, scene_samples))
                    else:
                        logger.warning(f"FlyingThings3D scene not in TRAIN or TEST: {scene}")
                
                # All TRAIN samples go to training split
                for scene, scene_samples in train_scenes:
                    train_samples.extend(scene_samples)
                
                # TEST samples are split between val and test proportionally
                # Calculate relative proportions of val and test
                val_test_total = val_prop + test_prop
                if val_test_total > 0:
                    val_ratio = val_prop / val_test_total
                    test_ratio = test_prop / val_test_total
                else:
                    val_ratio = 0.5
                    test_ratio = 0.5
                
                # Collect all TEST samples and shuffle them
                all_test_samples = []
                for scene, scene_samples in test_scenes:
                    all_test_samples.extend(scene_samples)
                
                random.shuffle(all_test_samples)
                
                # Split TEST samples between val and test
                n_test_samples = len(all_test_samples)
                n_val_from_test = int(n_test_samples * val_ratio)
                
                val_samples.extend(all_test_samples[:n_val_from_test])
                test_samples.extend(all_test_samples[n_val_from_test:])
                
                logger.info(f"FlyingThings3D stratified split: "
                           f"{len(train_scenes)} TRAIN scenes → train ({len([s for _, ss in train_scenes for s in ss])} samples), "
                           f"{len(test_scenes)} TEST scenes → val/test ({len(val_samples)} val, {len(all_test_samples[n_val_from_test:])} test)")
            
            else:
                # Driving: Sequential split within each video sequence
                for scene, scene_samples in scene_dict.items():
                    # Sort samples by frame_id to ensure temporal order
                    scene_samples.sort(key=lambda x: x['metadata']['frame_id'])
                    
                    n_samples = len(scene_samples)
                    n_train = int(n_samples * train_prop)
                    n_val = int(n_samples * val_prop)
                    
                    # Sequential split: first frames to train, middle to val, last to test
                    train_samples.extend(scene_samples[:n_train])
                    val_samples.extend(scene_samples[n_train:n_train + n_val])
                    test_samples.extend(scene_samples[n_train + n_val:])
                
                logger.info(f"{subset} stratified split: {len(scene_dict)} video sequences, "
                           f"sequential split within each")
        
        splits = {
            'train': train_samples if self.train_config.get('enabled', True) else [],
            'val': val_samples if self.val_config.get('enabled', True) else [],
            'test': test_samples if self.test_config.get('enabled', True) else [],
        }
        
        return splits
