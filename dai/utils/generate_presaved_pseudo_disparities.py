"""
Generate presaved pseudo-GT disparities for all unique crop positions.

This script:
1. Loads config and discovers IRS dataset samples
2. Simulates all crop positions that will occur during training
3. For each unique crop position per image, generates FoundationStereo pseudo GT
4. Saves disparities as: pseudo_d_{y}_{x}_{frame_id}.npy

Usage:
    python dai/utils/generate_presaved_pseudo_disparities.py dai/configs/presaved_pseudo.yaml
"""

import sys
from pathlib import Path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import yaml
import random
import numpy as np
import torch
import cv2
from PIL import Image
from typing import Dict, Any, List, Tuple
import argparse
from tqdm import tqdm
from collections import defaultdict
import logging

from dai.datasets.discovery_strategies import IRSDiscovery
from dai.datasets.splitter import DatasetSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_foundation_stereo_model(model_size: str = 'small'):
    """
    Load FoundationStereo model for pseudo GT generation.
    
    Args:
        model_size: 'small', 'base', or 'large'
        
    Returns:
        Dictionary with model, device, and config
    """
    from stereo.modeling.models.foundationstereo.core.foundation_stereo import FoundationStereo
    from easydict import EasyDict
    
    checkpoint_dir = repo_root / "src_data" / "checkpoints" / "foundationstereo" / model_size
    cfg_path = checkpoint_dir / "cfg.yaml"
    
    if not cfg_path.exists():
        raise FileNotFoundError(f"FoundationStereo config not found: {cfg_path}")
    
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    args = EasyDict(cfg_dict)
    
    logger.info(f"Loading FoundationStereo ({model_size}) model...")
    model = FoundationStereo(args)
    
    checkpoint_path = checkpoint_dir / "model_best_bp2.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"FoundationStereo checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    logger.info(f"FoundationStereo loaded on {device}")
    return {'model': model, 'device': device, 'args': args}


def generate_pseudo_disparity(model_dict: Dict, left_img: np.ndarray, right_img: np.ndarray,
                              inference_iters: int = 12, use_amp: bool = True) -> np.ndarray:
    """
    Generate pseudo disparity using FoundationStereo.
    
    Args:
        model_dict: Dictionary with model, device, args
        left_img: Left image [H, W, 3] in range [0, 255]
        right_img: Right image [H, W, 3] in range [0, 255]
        inference_iters: Number of GRU iterations
        use_amp: Use automatic mixed precision
        
    Returns:
        Disparity map [H, W]
    """
    from stereo.modeling.models.foundationstereo.core.foundation_stereo import normalize_image
    from stereo.modeling.models.foundationstereo.core.utils.utils import InputPadder
    
    model = model_dict['model']
    device = model_dict['device']
    
    # Convert to tensors [1, 3, H, W]
    left_tensor = torch.from_numpy(left_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    right_tensor = torch.from_numpy(right_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    
    # Normalize
    left_tensor = normalize_image(left_tensor)
    right_tensor = normalize_image(right_tensor)
    
    # Pad to multiple of 32
    padder = InputPadder(left_tensor.shape[-2:], divis_by=32, force_square=False)
    left_padded, right_padded = padder.pad(left_tensor, right_tensor)
    
    data = {'left': left_padded, 'right': right_padded}
    
    # Run inference
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(data, iters=inference_iters)
        else:
            output = model(data, iters=inference_iters)
    
    # Get disparity and unpad
    disp_pred = output['disp_pred']
    disp_pred = padder.unpad(disp_pred)
    
    # Convert to numpy [H, W]
    disp_np = disp_pred.squeeze().cpu().numpy()
    
    return disp_np


def simulate_crop_positions(num_images: int, image_size: Tuple[int, int], 
                           crop_size: List[int], num_epochs: int, seed: int) -> set:
    """
    Simulate all unique crop positions that will occur.
    
    Returns:
        Set of (y, x) tuples representing unique crop positions
    """
    img_h, img_w = image_size
    crop_h, crop_w = crop_size
    
    random.seed(seed)
    np.random.seed(seed)
    
    unique_positions = set()
    
    for epoch in range(num_epochs):
        for img_idx in range(num_images):
            y = random.randint(0, img_h - crop_h)
            x = random.randint(0, img_w - crop_w)
            unique_positions.add((y, x))
    
    return unique_positions


def generate_presaved_disparities(config_path: str, split: str = 'train', 
                                  dry_run: bool = False, max_images: int = None):
    """
    Generate presaved pseudo disparities for a dataset split.
    
    Args:
        config_path: Path to config YAML file
        split: 'train', 'val', or 'test'
        dry_run: If True, only simulate without generating
        max_images: Maximum number of images to process (for testing)
    """
    config = load_config(config_path)
    
    logger.info(f"{'='*80}")
    logger.info(f"Generating Presaved Pseudo Disparities - {split.upper()} Split")
    logger.info(f"{'='*80}")
    
    # Extract config
    global_seed = config.get('seed', 28)
    splitter_config = config.get('splitter', {})
    datasets_config = config.get('datasets', [])
    transforms_config = config.get('transforms', {})
    optimization_config = config.get('optimization', {})
    foundationstereo_config = config.get('foundationstereo', {})
    
    num_epochs = optimization_config.get('num_epochs', 90)
    model_size = foundationstereo_config.get('model_size', 'small')
    inference_iters = foundationstereo_config.get('inference_iters', 12)
    use_amp = foundationstereo_config.get('inference_amp', True)
    
    logger.info(f"Configuration:")
    logger.info(f"  Global seed: {global_seed}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  FoundationStereo: {model_size}, iters={inference_iters}, amp={use_amp}")
    
    # Get crop and resize sizes
    crop_size = None
    resize_size = None
    for transform in transforms_config.get(split, []):
        if transform['name'] == 'RandomCrop':
            crop_size = transform['size']
        elif transform['name'] == 'Resize':
            resize_size = transform['size']
    
    if crop_size is None:
        logger.error(f"No RandomCrop found in {split} transforms")
        return
    
    logger.info(f"  Crop size: {crop_size[0]} × {crop_size[1]}")
    logger.info(f"  Resize to: {resize_size[0]} × {resize_size[1]}" if resize_size else "  No resize")
    
    # Load FoundationStereo model (unless dry run)
    model_dict = None
    if not dry_run:
        model_dict = load_foundation_stereo_model(model_size)
    
    # Process each dataset
    for dataset_config in datasets_config:
        dataset_name = dataset_config['name']
        root = Path(dataset_config['root'])
        
        logger.info(f"\nDataset: {dataset_name}")
        logger.info(f"Root: {root}")
        
        # Discover samples
        discovery = IRSDiscovery()
        all_samples = discovery.discover_samples(root, dataset_config)
        
        if len(all_samples) == 0:
            logger.warning("No samples found, skipping")
            continue
        
        # Split samples
        splitter = DatasetSplitter(splitter_config)
        splits = splitter.split_dataset(dataset_name, root, dataset_config)
        
        split_samples = splits[split]
        if len(split_samples) == 0:
            logger.warning(f"No samples in {split} split, skipping")
            continue
        
        # Limit for testing
        if max_images:
            split_samples = split_samples[:max_images]
        
        logger.info(f"Processing {len(split_samples)} images in {split} split")
        
        # Simulate crop positions
        image_size = (540, 960)
        unique_positions = simulate_crop_positions(
            len(split_samples), image_size, crop_size, num_epochs, global_seed
        )
        
        logger.info(f"Unique crop positions to generate: {len(unique_positions)}")
        logger.info(f"Total disparities to generate: {len(split_samples) * len(unique_positions)}")
        
        if dry_run:
            logger.info("DRY RUN - Skipping actual generation")
            continue
        
        # Generate disparities for each image and crop position
        total_generated = 0
        total_skipped = 0
        
        for sample in tqdm(split_samples, desc=f"Processing {split} images"):
            left_path = Path(sample['left'])
            right_path = Path(sample['right'])
            frame_id = sample['metadata']['frame_id']
            scene_dir = left_path.parent
            
            # Load original images
            left_img = np.array(Image.open(left_path).convert('RGB'), dtype=np.float32)
            right_img = np.array(Image.open(right_path).convert('RGB'), dtype=np.float32)
            
            # Generate disparity for each unique crop position
            for y, x in unique_positions:
                presaved_filename = f"pseudo_d_{y}_{x}_{frame_id}.npy"
                presaved_path = scene_dir / presaved_filename
                
                # Skip if already exists
                if presaved_path.exists():
                    total_skipped += 1
                    continue
                
                # Crop images
                crop_h, crop_w = crop_size
                left_crop = left_img[y:y+crop_h, x:x+crop_w, :]
                right_crop = right_img[y:y+crop_h, x:x+crop_w, :]
                
                # Resize if needed
                if resize_size:
                    target_h, target_w = resize_size
                    left_crop = cv2.resize(left_crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    right_crop = cv2.resize(right_crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                
                # Generate pseudo disparity
                pseudo_disp = generate_pseudo_disparity(
                    model_dict, left_crop, right_crop, 
                    inference_iters=inference_iters, use_amp=use_amp
                )
                
                # Save as .npy
                np.save(presaved_path, pseudo_disp.astype(np.float16))
                total_generated += 1
        
        logger.info(f"Generated: {total_generated}, Skipped (already exist): {total_skipped}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Presaved disparity generation complete!")
    logger.info(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Generate presaved pseudo-GT disparities')
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument('--split', type=str, default='train', 
                       choices=['train', 'val', 'test'],
                       help='Dataset split to process')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate without generating files')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process (for testing)')
    
    args = parser.parse_args()
    
    generate_presaved_disparities(
        args.config,
        split=args.split,
        dry_run=args.dry_run,
        max_images=args.max_images
    )


if __name__ == '__main__':
    main()
