"""
Main execution pipeline for DAI training infrastructure
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any
import random
import numpy as np

# Add parent directory to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from dai.config import load_config
from dai.datasets import (
    DatasetRegistry,
    DatasetSplitter,
    SceneFlowDataset,
    SequentialSceneFlowDataset,
    KITTI12Dataset,
    KITTI15Dataset,
    IRSDataset,
    CombinedStereoDataset,
    PostprocessingDataset,
    PresavedPseudoGTDataset,
    SequentialPresavedDataset,
    build_transforms,
    sequential_collate_fn
)
from dai.training import build_trainer
from dai.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DAIPipeline:
    """
    Main orchestrator for the entire training pipeline.
    Uses different services: dataset loading, splitting, postprocessing, dataloaders, and trainer.
    """
    
    def __init__(self, config_path: str, device: str = 'cuda'):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to YAML configuration file
            device: Device to train on ('cuda' or 'cpu')
        """
        logger.info("DAI Pipeline - Initializing")
        
        self.config = load_config(config_path)
        self.device = device
        
        # Set random seed for reproducibility
        seed = self.config.get('seed', 28)
        set_seed(seed)
        
        # Services
        self.splitter = DatasetSplitter(self.config['splitter'])
        
        # Data containers
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        
        self.combined_train = None
        self.combined_val = None
        self.combined_test = None
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Trainer (initialized after data is ready)
        self.trainer = None
    
    def load_datasets(self):
        """Discover, split, and load all configured datasets."""
        logger.info("Loading datasets...")
        
        for dataset_config in self.config['datasets']:
            try:
                self._load_single_dataset(dataset_config)
            except Exception as e:
                logger.error(f"Failed to load {dataset_config['name']}: {e}")
                continue
        
        total_train = sum(len(ds) for ds in self.train_datasets)
        total_val = sum(len(ds) for ds in self.val_datasets)
        logger.info(f"Loaded {total_train} train, {total_val} val samples")
    
    def _load_single_dataset(self, dataset_config: Dict[str, Any]):
        """Load a single dataset (discover, split, create PyTorch dataset)."""
        dataset_name = dataset_config['name']
        root = Path(dataset_config['root'])
        
        if not root.exists():
            logger.warning(f"{dataset_name}: root not found at {root}")
            return
        
        splits = self.splitter.split_dataset(dataset_name, root, dataset_config)
        
        if len(splits['train']) == 0 and len(splits['val']) == 0:
            logger.warning(f"{dataset_name}: no samples found")
            return
        
        if len(splits['train']) > 0:
            train_ds = self._create_dataset_instance(dataset_name, splits['train'], dataset_config, 'train')
            self.train_datasets.append(train_ds)
        
        if len(splits['val']) > 0:
            val_ds = self._create_dataset_instance(dataset_name, splits['val'], dataset_config, 'val')
            self.val_datasets.append(val_ds)
        
        if len(splits['test']) > 0:
            test_ds = self._create_dataset_instance(dataset_name, splits['test'], dataset_config, 'test')
            self.test_datasets.append(test_ds)
    
    def _create_dataset_instance(self, dataset_name: str, samples: List[Dict[str, Any]], 
                                 config: Dict[str, Any], split: str):
        """
        Create appropriate dataset instance based on dataset name.
        
        Args:
            dataset_name: Name of dataset
            samples: List of samples
            config: Dataset configuration
            split: 'train', 'val', or 'test'
            
        Returns:
            Dataset instance
        """
        # Create base dataset
        if dataset_name == 'sceneflow':
            base_dataset = SceneFlowDataset(samples, config, split)
        elif dataset_name == 'kitti12':
            base_dataset = KITTI12Dataset(samples, config, split)
        elif dataset_name == 'kitti15':
            base_dataset = KITTI15Dataset(samples, config, split)
        elif dataset_name == 'irs':
            base_dataset = IRSDataset(samples, config, split)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Wrap with sequential dataset if enabled (works for any dataset)
        sequential_config = config.get('sequential', {})
        if sequential_config.get('enabled', False):
            sequence_length = sequential_config.get('sequence_length', 4)
            stride = sequential_config.get('stride', 1)
            skip_incomplete = sequential_config.get('skip_incomplete', True)
            
            logger.info(
                f"Wrapping {dataset_name} with SequentialSceneFlowDataset "
                f"(seq_len={sequence_length}, stride={stride})"
            )
            
            return SequentialSceneFlowDataset(
                base_dataset,
                sequence_length=sequence_length,
                stride=stride,
                skip_incomplete=skip_incomplete
            )
        else:
            return base_dataset
    
    def _wrap_dataset(self, datasets, transforms, use_presaved, is_sequential, resize_size):
        """
        Wrap dataset(s) with the appropriate transform and presaved pipeline.
        
        Three modes:
        1. Sequential + presaved: SequentialPresavedDataset (handles transforms + presaved)
        2. Non-sequential + presaved: PresavedPseudoGTDataset (handles transforms + presaved)
        3. Neither: PostprocessingDataset (handles transforms only)
        """
        if len(datasets) == 0:
            return None
        
        base = CombinedStereoDataset(datasets) if len(datasets) > 1 else datasets[0]
        
        if is_sequential and use_presaved:
            # Sequential + presaved: single wrapper handles both
            return SequentialPresavedDataset(base, transforms, resize_size)
        elif use_presaved:
            # Non-sequential presaved
            return PresavedPseudoGTDataset(base, transforms, resize_size)
        else:
            # Standard: just apply transforms
            return PostprocessingDataset(base, transforms)
    
    def combine_datasets(self):
        """Combine multiple datasets and apply transforms."""
        logger.info("Applying transforms and combining datasets...")
        
        transform_config = self.config.get('transforms', {})
        train_transforms = build_transforms(transform_config, 'train')
        val_transforms = build_transforms(transform_config, 'val')
        test_transforms = build_transforms(transform_config, 'test')
        
        # Check if any dataset uses presaved mode
        use_presaved = any(
            getattr(ds, 'target_source', '') == 'pseudo-foundationstereo-presaved' 
            for ds in self.train_datasets + self.val_datasets + self.test_datasets
        )
        
        # Check if using sequential training
        is_sequential = any(
            isinstance(ds, SequentialSceneFlowDataset)
            for ds in self.train_datasets + self.val_datasets + self.test_datasets
        )
        
        # Get resize size for presaved wrapper (from transforms config)
        resize_size = self._get_resize_size_from_transforms(transform_config, 'train')
        
        self.combined_train = self._wrap_dataset(
            self.train_datasets, train_transforms, use_presaved, is_sequential, resize_size
        )
        self.combined_val = self._wrap_dataset(
            self.val_datasets, val_transforms, use_presaved, is_sequential, resize_size
        )
        self.combined_test = self._wrap_dataset(
            self.test_datasets, test_transforms, use_presaved, is_sequential, resize_size
        )
    
    def _get_resize_size_from_transforms(self, transform_config: dict, split: str) -> list:
        """Extract resize dimensions from transform config."""
        if split not in transform_config:
            return None
        
        for transform in transform_config[split]:
            if transform['name'] == 'Resize':
                return transform['size']
        
        return None
    
    def create_dataloaders(self):
        """Create PyTorch DataLoaders from combined datasets."""
        dataloader_config = self.config.get('dataloader', {})
        batch_size = dataloader_config.get('batch_size', 4)
        num_workers = dataloader_config.get('num_workers', 4)
        shuffle = dataloader_config.get('shuffle', True)
        pin_memory = dataloader_config.get('pin_memory', True)
        drop_last = dataloader_config.get('drop_last', True)
        persistent_workers = dataloader_config.get('persistent_workers', False)
        
        # Worker init function for reproducibility
        seed = self.config.get('seed', 28)
        def worker_init_fn(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        
        # Create generator for reproducible shuffling
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        logger.info(f"Creating dataloaders (batch_size={batch_size}, workers={num_workers})...")
        
        # Check if we're using sequential datasets (need custom collate)
        training_config = self.config.get('training', {})
        training_mode = training_config.get('mode', 'normal')
        is_sequential = training_mode in ['sequential', 'sequential_fused_features', 'sequential_warped_disparity']
        collate_fn = sequential_collate_fn if is_sequential else None
        
        if self.combined_train is not None:
            self.train_loader = DataLoader(
                self.combined_train,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                persistent_workers=persistent_workers if num_workers > 0 else False,
                worker_init_fn=worker_init_fn,
                generator=generator,
                collate_fn=collate_fn
            )
        
        if self.combined_val is not None:
            self.val_loader = DataLoader(
                self.combined_val,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
                persistent_workers=persistent_workers if num_workers > 0 else False,
                worker_init_fn=worker_init_fn,
                collate_fn=collate_fn
            )
        
        if self.combined_test is not None:
            self.test_loader = DataLoader(
                self.combined_test,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
                persistent_workers=persistent_workers if num_workers > 0 else False,
                worker_init_fn=worker_init_fn,
                collate_fn=collate_fn
            )
    
    def test_sample_loading(self):
        """Test loading a batch from train dataloader."""
        if self.train_loader is None:
            return
        
        try:
            batch = next(iter(self.train_loader))
            logger.info(f"Test batch: {batch['left'].shape}, disparity range [{batch['disparity'][0].min():.1f}, {batch['disparity'][0].max():.1f}]")
        except Exception as e:
            logger.error(f"Batch loading test failed: {e}")
    
    def setup_data(self):
        """Setup data pipeline: load datasets, combine them, create dataloaders."""
        try:
            self.load_datasets()
            
            if len(self.train_datasets) == 0 and len(self.val_datasets) == 0:
                logger.error("No datasets loaded")
                return False
            
            self.combine_datasets()
            self.create_dataloaders()
            self.test_sample_loading()
            
            logger.info("Data pipeline ready")
            return True
            
        except Exception as e:
            logger.error(f"Data setup failed: {e}")
            return False
    
    def setup_trainer(self, resume_checkpoint: str = None, pretrained_path: str = None):
        """Initialize trainer with data loaders."""
        logger.info("Initializing trainer...")
        
        # Use build_trainer factory to handle all training modes
        training_mode = self.config.get('training', {}).get('mode', 'normal')
        logger.info(f"Training mode: {training_mode}")
        
        self.trainer = build_trainer(
            config=self.config,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device
        )
        
        if resume_checkpoint:
            logger.info(f"Resuming from {resume_checkpoint}")
            self.trainer.load_checkpoint(Path(resume_checkpoint))
        elif pretrained_path:
            logger.info(f"Loading pretrained model from {pretrained_path} (fine-tuning mode)")
            self.trainer.load_pretrained(Path(pretrained_path))
    
    def train(self, resume_checkpoint: str = None, pretrained_path: str = None):
        """Run the full training pipeline."""
        logger.info("="*80)
        logger.info("DAI Training Pipeline")
        logger.info("="*80)
        
        # Setup data
        if not self.setup_data():
            return False
        
        # Setup trainer
        self.setup_trainer(resume_checkpoint, pretrained_path)
        
        # Train
        num_epochs = self.config.get('optimization', {}).get('num_epochs', 90)
        save_dir = Path(self.config.get('checkpoints', {}).get('save_dir', './checkpoints'))
        
        logger.info(f"Training for {num_epochs} epochs")
        logger.info(f"Checkpoints: {save_dir}")
        logger.info("="*80)
        
        self.trainer.train(num_epochs=num_epochs, save_dir=save_dir)
        
        logger.info("="*80)
        logger.info("Training complete")
        logger.info("="*80)
        return True


def main():
    """Main entry point for training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DAI Training Pipeline')
    # parser.add_argument('--config', type=str, default='dai/configs/stereo_flow.yaml', help='Path to config YAML')
    parser.add_argument('--config', type=str, default='dai/configs/temporal_stereo_flow.yaml', help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained model for fine-tuning (resets optimizer/scheduler)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    # Parse known args to allow WandB sweep to pass additional parameters
    args, unknown = parser.parse_known_args()
    
    # Check if sweep is overriding the config file path
    config_path = args.config
    config_overrides = []
    
    if unknown:
        for i in range(0, len(unknown), 1):
            if unknown[i].startswith('--'):
                param = unknown[i][2:]  # Remove '--'
                if '=' in param:
                    key, value = param.split('=', 1)
                    # Special handling for 'config' parameter - use it to load base config
                    if key == 'config':
                        config_path = value
                        logger.info(f"Using sweep-specified config: {config_path}")
                    else:
                        config_overrides.append((key, value))
    
    # Initialize pipeline with base config (potentially from sweep)
    pipeline = DAIPipeline(config_path, device=args.device)
    
    # Apply other sweep parameter overrides
    if config_overrides:
        logger.info("Applying sweep parameter overrides:")
        for key, value in config_overrides:
            # Convert value to appropriate type
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                pass  # Keep as string
            
            # Apply nested config override
            keys = key.split('.')
            config_ref = pipeline.config
            for k in keys[:-1]:
                if k not in config_ref:
                    config_ref[k] = {}
                config_ref = config_ref[k]
            config_ref[keys[-1]] = value
            logger.info(f"  {key} = {value}")
    
    success = pipeline.train(resume_checkpoint=args.resume, pretrained_path=args.pretrained)
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
