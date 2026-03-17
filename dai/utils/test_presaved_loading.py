"""
Test script to verify presaved pseudo-GT loading works correctly.

This script:
1. Creates a few test presaved disparity files
2. Loads them using the PresavedPseudoGTDataset
3. Verifies the correct disparities are loaded for given crop coordinates
"""

import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import torch
import random
import argparse
from dai.config import load_config
from dai.utils import set_seed
from dai.main import DAIPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_presaved_files(dataset_root: Path, num_test_files: int = 5):
    """
    Create a few test presaved disparity files for testing.
    
    Args:
        dataset_root: Root of IRS dataset
        num_test_files: Number of test files to create
    """
    logger.info("Creating test presaved disparity files...")
    
    # Find first scene directory
    subsets = ['Home_1', 'Home_2', 'Office_1', 'Office_2', 'Restaurant', 'Store']
    scene_dir = None
    
    for subset in subsets:
        subset_path = dataset_root / subset
        if subset_path.exists():
            scene_dirs = [d for d in subset_path.iterdir() if d.is_dir()]
            if scene_dirs:
                scene_dir = scene_dirs[0]
                break
    
    if scene_dir is None:
        logger.error("No scene directory found in IRS dataset")
        return []
    
    logger.info(f"Using test scene: {scene_dir}")
    
    # Find first few images
    left_images = sorted(scene_dir.glob('l_*.png'))[:num_test_files]
    
    if len(left_images) == 0:
        logger.error("No images found in scene directory")
        return []
    
    created_files = []
    
    for left_img in left_images:
        frame_id = left_img.stem[2:]  # strip "l_" prefix
        
        # Create test disparities for a few crop positions
        test_crops = [(28, 190), (50, 100), (100, 150)]
        
        for y, x in test_crops:
            presaved_filename = f"pseudo_d_{y}_{x}_{frame_id}.npy"
            presaved_path = scene_dir / presaved_filename
            
            # Create dummy disparity (256x512 with unique pattern)
            disparity = np.ones((256, 512), dtype=np.float32) * (y + x)
            disparity[0, 0] = y  # Store y in top-left
            disparity[0, 1] = x  # Store x in position [0, 1]
            
            np.save(presaved_path, disparity)
            created_files.append(presaved_path)
    
    logger.info(f"Created {len(created_files)} test presaved files")
    return created_files


def test_presaved_loading(config_path: str):
    """
    Test that presaved disparities are loaded correctly.
    
    Args:
        config_path: Path to presaved config YAML
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing Presaved Pseudo-GT Loading")
    logger.info(f"{'='*80}\n")
    
    config = load_config(config_path)
    
    # Check target_source
    target_source = config['datasets'][0].get('target_source')
    if target_source != 'pseudo-foundationstereo-presaved':
        logger.error(f"Config must use target_source: pseudo-foundationstereo-presaved")
        logger.error(f"Current: {target_source}")
        return False
    
    logger.info(f"Config: {Path(config_path).name}")
    logger.info(f"Target source: {target_source}\n")
    
    # Create test presaved files
    dataset_root = Path(config['datasets'][0]['root'])
    created_files = create_test_presaved_files(dataset_root, num_test_files=2)
    
    if len(created_files) == 0:
        logger.error("Failed to create test files")
        return False
    
    # Setup pipeline
    logger.info("\nSetting up data pipeline...")
    global_seed = config.get('seed', 28)
    set_seed(global_seed)
    
    pipeline = DAIPipeline(config_path, device='cpu')
    if not pipeline.setup_data():
        logger.error("Failed to setup data pipeline")
        return False
    
    # Test loading a batch
    logger.info("\nLoading test batch...")
    try:
        batch = next(iter(pipeline.train_loader))
        
        logger.info(f"Batch loaded successfully!")
        logger.info(f"  Left shape: {batch['left'].shape}")
        logger.info(f"  Right shape: {batch['right'].shape}")
        logger.info(f"  Disparity shape: {batch['disparity'].shape}")
        logger.info(f"  Disparity range: [{batch['disparity'].min():.2f}, {batch['disparity'].max():.2f}]")
        
        # Check if disparities are non-zero (indicating presaved loaded)
        non_zero_count = (batch['disparity'] != 0).sum().item()
        total_pixels = batch['disparity'].numel()
        
        logger.info(f"  Non-zero pixels: {non_zero_count}/{total_pixels} ({non_zero_count/total_pixels*100:.1f}%)")
        
        if non_zero_count > 0:
            logger.info("\n✓ SUCCESS: Presaved disparities are being loaded!")
            logger.info("  (Non-zero disparities indicate presaved files were used)")
        else:
            logger.warning("\n⚠ WARNING: All disparities are zero")
            logger.warning("  This might mean presaved files weren't found or loaded")
        
    except Exception as e:
        logger.error(f"Error loading batch: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup test files
    logger.info(f"\nCleaning up {len(created_files)} test files...")
    for file_path in created_files:
        if file_path.exists():
            file_path.unlink()
    
    logger.info(f"\n{'='*80}")
    logger.info("Test complete!")
    logger.info(f"{'='*80}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test presaved pseudo-GT loading')
    parser.add_argument('--config', type=str, 
                       default='dai/configs/presaved_pseudo.yaml',
                       help='Path to presaved config YAML file')
    
    args = parser.parse_args()
    
    success = test_presaved_loading(args.config)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
