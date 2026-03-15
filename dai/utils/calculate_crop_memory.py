"""
Utility to calculate memory requirements for presaved pseudo GT crops.

This script:
1. Loads the config file
2. Simulates the dataset iteration with the same seed
3. Counts unique crop positions that will be sampled
4. Estimates total memory requirements
"""

import yaml
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple
import argparse


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_image_size(dataset_name: str, dataset_config: Dict[str, Any]) -> Tuple[int, int]:
    """
    Get the native image size for a dataset.
    
    Returns:
        (height, width) tuple
    """
    if dataset_name == 'irs':
        return (540, 960)
    elif dataset_name == 'sceneflow':
        return (540, 960)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def count_dataset_images(dataset_config: Dict[str, Any]) -> int:
    """
    Count total images in a dataset by scanning the directory.
    
    Args:
        dataset_config: Dataset configuration dict
        
    Returns:
        Total number of image pairs
    """
    root = Path(dataset_config['root'])
    subsets = dataset_config.get('subsets', [])
    
    if not root.exists():
        print(f"  Warning: Dataset root not found: {root}")
        print(f"  Using estimated count for {dataset_config['name']}")
        if dataset_config['name'] == 'irs':
            return 76000
        return 0
    
    total_count = 0
    found_any = False
    
    for subset in subsets:
        subset_path = root / subset / 'left'
        if subset_path.exists():
            # Count image files
            image_files = list(subset_path.glob('*.png')) + list(subset_path.glob('*.jpg'))
            total_count += len(image_files)
            print(f"  {subset}: {len(image_files)} images")
            found_any = True
        else:
            print(f"  Warning: Subset not found: {subset_path}")
    
    # If no subsets found, use estimate
    if not found_any and dataset_config['name'] == 'irs':
        print(f"  Using estimated count: 76,000 images")
        return 76000
    
    return total_count


def calculate_possible_crops(image_size: Tuple[int, int], crop_size: List[int]) -> int:
    """
    Calculate number of possible crop positions.
    
    Args:
        image_size: (height, width) of original image
        crop_size: [height, width] of crop
        
    Returns:
        Number of possible crop positions
    """
    img_h, img_w = image_size
    crop_h, crop_w = crop_size
    
    if img_h < crop_h or img_w < crop_w:
        return 0
    
    num_vertical = img_h - crop_h + 1
    num_horizontal = img_w - crop_w + 1
    
    return num_vertical * num_horizontal


def simulate_crops_per_epoch(num_images: int, seed: int, 
                             image_size: Tuple[int, int], 
                             crop_size: List[int],
                             num_epochs: int = 1) -> Dict[Tuple[int, int], int]:
    """
    Simulate the random crops that will occur during training.
    
    Args:
        num_images: Total number of images in dataset
        seed: Random seed
        image_size: (height, width) of original images
        crop_size: [height, width] of crop
        num_epochs: Number of epochs to simulate
        
    Returns:
        Dictionary mapping (y, x) crop positions to occurrence count
    """
    img_h, img_w = image_size
    crop_h, crop_w = crop_size
    
    if img_h < crop_h or img_w < crop_w:
        return {}
    
    crop_positions = defaultdict(int)
    
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    for epoch in range(num_epochs):
        for img_idx in range(num_images):
            # Simulate random crop (same logic as RandomCrop transform)
            y = random.randint(0, img_h - crop_h)
            x = random.randint(0, img_w - crop_w)
            
            crop_positions[(y, x)] += 1
    
    return crop_positions


def estimate_memory(num_disparity_maps: int, 
                   disparity_size: List[int],
                   dtype_bytes: int = 4) -> Dict[str, float]:
    """
    Estimate memory requirements for storing disparity maps.
    
    Args:
        num_disparity_maps: Total number of disparity maps
        disparity_size: [height, width] of each disparity map
        dtype_bytes: Bytes per pixel (4 for float32, 2 for float16)
        
    Returns:
        Dictionary with memory estimates in different units
    """
    disp_h, disp_w = disparity_size
    bytes_per_map = disp_h * disp_w * dtype_bytes
    total_bytes = num_disparity_maps * bytes_per_map
    
    return {
        'bytes_per_map': bytes_per_map,
        'mb_per_map': bytes_per_map / (1024 ** 2),
        'total_bytes': total_bytes,
        'total_mb': total_bytes / (1024 ** 2),
        'total_gb': total_bytes / (1024 ** 3),
        'total_tb': total_bytes / (1024 ** 4),
    }


def analyze_config(config_path: str, num_epochs: int = 90, verbose: bool = True):
    """
    Analyze configuration and estimate memory requirements.
    
    Args:
        config_path: Path to config YAML file
        num_epochs: Number of training epochs to simulate
        verbose: Print detailed information
    """
    config = load_config(config_path)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Configuration Analysis: {Path(config_path).name}")
        print(f"{'='*70}\n")
    
    # Extract relevant config
    seed = config.get('seed', 42)
    splitter_seed = config.get('splitter', {}).get('seed', 42)
    datasets_config = config.get('datasets', [])
    transforms_config = config.get('transforms', {})
    optimization_config = config.get('optimization', {})
    
    # Get number of epochs from config if available
    config_epochs = optimization_config.get('num_epochs', num_epochs)
    
    if verbose:
        print(f"Seeds:")
        print(f"  Global seed: {seed}")
        print(f"  Splitter seed: {splitter_seed}")
        print(f"  Training epochs: {config_epochs}\n")
    
    # Analyze each dataset
    total_crops_all_datasets = 0
    total_memory_gb = 0
    
    for dataset_config in datasets_config:
        dataset_name = dataset_config['name']
        
        if verbose:
            print(f"\nDataset: {dataset_name}")
            print(f"-" * 70)
        
        # Count images
        num_images = count_dataset_images(dataset_config)
        if verbose:
            print(f"Total images: {num_images:,}")
        
        # Get image size
        image_size = get_image_size(dataset_name, dataset_config)
        if verbose:
            print(f"Image size: {image_size[0]} × {image_size[1]}")
        
        # Analyze transforms for each split
        for split in ['train', 'val', 'test']:
            if split not in transforms_config:
                continue
            
            # Find RandomCrop and Resize transforms
            crop_size = None
            resize_size = None
            
            for transform in transforms_config[split]:
                if transform['name'] == 'RandomCrop':
                    crop_size = transform['size']
                elif transform['name'] == 'Resize':
                    resize_size = transform['size']
            
            if crop_size is None:
                if verbose:
                    print(f"  {split}: No RandomCrop found, skipping")
                continue
            
            if verbose:
                print(f"\n  Split: {split}")
                print(f"    Crop size: {crop_size[0]} × {crop_size[1]}")
                if resize_size:
                    print(f"    Resize to: {resize_size[0]} × {resize_size[1]}")
            
            # Calculate possible crop positions
            num_possible_crops = calculate_possible_crops(image_size, crop_size)
            if verbose:
                print(f"    Possible crop positions: {num_possible_crops:,}")
            
            # Get split proportion
            splitter_config = config.get('splitter', {})
            split_proportion = splitter_config.get(split, {}).get('proportion', 0.0)
            split_images = int(num_images * split_proportion)
            
            if verbose:
                print(f"    Split proportion: {split_proportion} ({split_images:,} images)")
            
            # Simulate crops for this split
            if verbose:
                print(f"    Simulating {config_epochs} epochs...")
            
            crop_positions = simulate_crops_per_epoch(
                split_images, seed, image_size, crop_size, config_epochs
            )
            
            unique_crops = len(crop_positions)
            total_crops = sum(crop_positions.values())
            
            if verbose:
                print(f"    Unique crop positions sampled: {unique_crops:,}")
                print(f"    Total crops (across epochs): {total_crops:,}")
            
            # Calculate memory for this split
            final_size = resize_size if resize_size else crop_size
            
            # Memory for unique crops (presaved)
            memory_unique = estimate_memory(unique_crops, final_size)
            
            # Memory for all crops (if saving per epoch)
            memory_total = estimate_memory(total_crops, final_size)
            
            if verbose:
                print(f"\n    Memory (unique crops only):")
                print(f"      Per disparity map: {memory_unique['mb_per_map']:.2f} MB")
                print(f"      Total: {memory_unique['total_gb']:.2f} GB ({memory_unique['total_tb']:.4f} TB)")
                
                print(f"\n    Memory (all crops per epoch):")
                print(f"      Total: {memory_total['total_gb']:.2f} GB ({memory_total['total_tb']:.4f} TB)")
            
            total_crops_all_datasets += unique_crops
            total_memory_gb += memory_unique['total_gb']
    
    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"Total unique crops across all datasets/splits: {total_crops_all_datasets:,}")
        print(f"Total memory required (float32): {total_memory_gb:.2f} GB ({total_memory_gb/1024:.4f} TB)")
        print(f"Total memory required (float16): {total_memory_gb/2:.2f} GB ({total_memory_gb/2048:.4f} TB)")
        print(f"{'='*70}\n")
    
    return {
        'total_unique_crops': total_crops_all_datasets,
        'total_memory_gb_float32': total_memory_gb,
        'total_memory_gb_float16': total_memory_gb / 2,
    }


def main():
    parser = argparse.ArgumentParser(description='Calculate memory requirements for presaved pseudo GT crops')
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument('--epochs', type=int, default=None, 
                       help='Number of epochs to simulate (default: read from config)')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Load config to get epochs if not specified
    config = load_config(args.config)
    num_epochs = args.epochs if args.epochs is not None else config.get('optimization', {}).get('num_epochs', 90)
    
    results = analyze_config(args.config, num_epochs=num_epochs, verbose=not args.quiet)
    
    return results


if __name__ == '__main__':
    main()
