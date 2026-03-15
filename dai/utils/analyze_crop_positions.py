"""
Utility to analyze actual crop positions that will occur during training.

This script:
1. Loads the config and discovers actual IRS dataset samples
2. Simulates the splitter with correct seed to get train/val/test splits
3. Simulates the RandomCrop transform for each epoch with correct seed
4. Logs all crop positions and calculates memory requirements
"""

import yaml
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.discovery_strategies import IRSDiscovery
from datasets.splitter import DatasetSplitter


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def simulate_random_crop(image_size: Tuple[int, int], crop_size: List[int]) -> Tuple[int, int]:
    """
    Simulate a single random crop using the same logic as RandomCrop transform.
    
    Args:
        image_size: (height, width) of original image
        crop_size: [height, width] of crop
        
    Returns:
        (y, x) crop position
    """
    img_h, img_w = image_size
    crop_h, crop_w = crop_size
    
    if img_h < crop_h or img_w < crop_w:
        return (0, 0)
    
    y = random.randint(0, img_h - crop_h)
    x = random.randint(0, img_w - crop_w)
    
    return (y, x)


def analyze_crops_for_split(
    samples: List[Dict[str, Any]],
    split_name: str,
    image_size: Tuple[int, int],
    crop_size: List[int],
    resize_size: List[int],
    num_epochs: int,
    global_seed: int
) -> Dict[str, Any]:
    """
    Simulate crops for a specific split across multiple epochs.
    
    Args:
        samples: List of sample dictionaries
        split_name: 'train', 'val', or 'test'
        image_size: (height, width) of original images
        crop_size: [height, width] of crop
        resize_size: [height, width] after resize
        num_epochs: Number of epochs to simulate
        global_seed: Global random seed
        
    Returns:
        Dictionary with analysis results
    """
    num_images = len(samples)
    
    if num_images == 0:
        return {
            'num_images': 0,
            'num_epochs': num_epochs,
            'unique_crops': 0,
            'total_crops': 0,
            'crop_positions': {},
            'per_image_crops': {}
        }
    
    # Track crop positions globally and per image
    all_crop_positions = defaultdict(int)
    per_image_crops = defaultdict(lambda: defaultdict(int))
    
    # Set seed for reproducibility
    random.seed(global_seed)
    np.random.seed(global_seed)
    
    print(f"  Simulating {num_epochs} epochs for {num_images} images...")
    
    for epoch in range(num_epochs):
        for img_idx, sample in enumerate(samples):
            # Simulate random crop
            y, x = simulate_random_crop(image_size, crop_size)
            
            # Track globally
            all_crop_positions[(y, x)] += 1
            
            # Track per image
            image_path = sample['left']
            per_image_crops[image_path][(y, x)] += 1
    
    unique_crops = len(all_crop_positions)
    total_crops = sum(all_crop_positions.values())
    
    return {
        'num_images': num_images,
        'num_epochs': num_epochs,
        'unique_crops': unique_crops,
        'total_crops': total_crops,
        'crop_positions': dict(all_crop_positions),
        'per_image_crops': {k: dict(v) for k, v in per_image_crops.items()}
    }


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


def analyze_config(config_path: str, num_epochs: int = None, 
                  save_crop_log: str = None, verbose: bool = True):
    """
    Analyze configuration with actual dataset discovery and crop simulation.
    
    Args:
        config_path: Path to config YAML file
        num_epochs: Number of epochs to simulate (default: read from config)
        save_crop_log: Optional path to save detailed crop log
        verbose: Print detailed information
    """
    config = load_config(config_path)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Crop Position Analysis: {Path(config_path).name}")
        print(f"{'='*80}\n")
    
    # Extract config
    global_seed = config.get('seed', 42)
    splitter_config = config.get('splitter', {})
    datasets_config = config.get('datasets', [])
    transforms_config = config.get('transforms', {})
    optimization_config = config.get('optimization', {})
    
    # Get number of epochs
    if num_epochs is None:
        num_epochs = optimization_config.get('num_epochs', 90)
    
    if verbose:
        print(f"Configuration:")
        print(f"  Global seed: {global_seed}")
        print(f"  Splitter seed: {splitter_config.get('seed', 42)}")
        print(f"  Training epochs: {num_epochs}")
        print(f"  Splitter strategy: {splitter_config.get('strategy', 'stratified')}\n")
    
    # Initialize splitter
    splitter = DatasetSplitter(splitter_config)
    
    # Analyze each dataset
    all_results = {}
    total_unique_crops = 0
    total_memory_gb_float32 = 0
    total_memory_gb_float16 = 0
    
    for dataset_config in datasets_config:
        dataset_name = dataset_config['name']
        root = Path(dataset_config['root'])
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*80}")
            print(f"Root: {root}")
        
        # Discover samples using the actual discovery strategy
        discovery = IRSDiscovery()
        all_samples = discovery.discover_samples(root, dataset_config)
        
        if len(all_samples) == 0:
            if verbose:
                print(f"  No samples found. Skipping.\n")
            continue
        
        if verbose:
            print(f"  Total samples discovered: {len(all_samples)}")
        
        # Split samples using the actual splitter
        splits = splitter.split_dataset(dataset_name, root, dataset_config)
        
        # Get image size (IRS is 540x960)
        image_size = (540, 960)
        
        if verbose:
            print(f"  Image size: {image_size[0]} × {image_size[1]}")
            print(f"\n  Split sizes:")
            print(f"    Train: {len(splits['train'])} images")
            print(f"    Val: {len(splits['val'])} images")
            print(f"    Test: {len(splits['test'])} images")
        
        dataset_results = {}
        
        # Analyze each split
        for split_name in ['train', 'val', 'test']:
            if split_name not in transforms_config:
                continue
            
            split_samples = splits[split_name]
            if len(split_samples) == 0:
                continue
            
            # Find RandomCrop and Resize transforms
            crop_size = None
            resize_size = None
            
            for transform in transforms_config[split_name]:
                if transform['name'] == 'RandomCrop':
                    crop_size = transform['size']
                elif transform['name'] == 'Resize':
                    resize_size = transform['size']
            
            if crop_size is None:
                if verbose:
                    print(f"\n  {split_name.upper()}: No RandomCrop found, skipping")
                continue
            
            if verbose:
                print(f"\n  {split_name.upper()} Split Analysis:")
                print(f"    Crop size: {crop_size[0]} × {crop_size[1]}")
                if resize_size:
                    print(f"    Resize to: {resize_size[0]} × {resize_size[1]}")
            
            # Calculate possible crop positions
            img_h, img_w = image_size
            crop_h, crop_w = crop_size
            num_possible_crops = (img_h - crop_h + 1) * (img_w - crop_w + 1)
            
            if verbose:
                print(f"    Possible crop positions: {num_possible_crops:,}")
            
            # Simulate crops for this split
            split_results = analyze_crops_for_split(
                split_samples,
                split_name,
                image_size,
                crop_size,
                resize_size if resize_size else crop_size,
                num_epochs,
                global_seed
            )
            
            dataset_results[split_name] = split_results
            
            if verbose:
                print(f"    Unique crop positions sampled: {split_results['unique_crops']:,}")
                print(f"    Total crops (across {num_epochs} epochs): {split_results['total_crops']:,}")
                print(f"    Coverage: {split_results['unique_crops'] / num_possible_crops * 100:.2f}% of possible positions")
            
            # Calculate memory
            final_size = resize_size if resize_size else crop_size
            
            # Memory for unique crops
            memory_unique = estimate_memory(split_results['unique_crops'], final_size, dtype_bytes=4)
            memory_unique_fp16 = estimate_memory(split_results['unique_crops'], final_size, dtype_bytes=2)
            
            # Memory for all crops (per epoch)
            memory_total = estimate_memory(split_results['total_crops'], final_size, dtype_bytes=4)
            
            if verbose:
                print(f"\n    Memory (unique crops - RECOMMENDED):")
                print(f"      Per disparity: {memory_unique['mb_per_map']:.2f} MB")
                print(f"      Total (float32): {memory_unique['total_gb']:.2f} GB")
                print(f"      Total (float16): {memory_unique_fp16['total_gb']:.2f} GB")
                
                print(f"\n    Memory (all crops per epoch - NOT RECOMMENDED):")
                print(f"      Total (float32): {memory_total['total_gb']:.2f} GB ({memory_total['total_tb']:.4f} TB)")
            
            total_unique_crops += split_results['unique_crops']
            total_memory_gb_float32 += memory_unique['total_gb']
            total_memory_gb_float16 += memory_unique_fp16['total_gb']
        
        all_results[dataset_name] = dataset_results
    
    # Summary
    if verbose:
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Total unique crops across all datasets/splits: {total_unique_crops:,}")
        print(f"Total memory required (float32): {total_memory_gb_float32:.2f} GB ({total_memory_gb_float32/1024:.4f} TB)")
        print(f"Total memory required (float16): {total_memory_gb_float16:.2f} GB ({total_memory_gb_float16/1024:.4f} TB)")
        print(f"{'='*80}\n")
    
    # Save detailed crop log if requested
    if save_crop_log:
        save_crop_log_file(all_results, save_crop_log)
        if verbose:
            print(f"Detailed crop log saved to: {save_crop_log}\n")
    
    return {
        'total_unique_crops': total_unique_crops,
        'total_memory_gb_float32': total_memory_gb_float32,
        'total_memory_gb_float16': total_memory_gb_float16,
        'detailed_results': all_results
    }


def save_crop_log_file(results: Dict[str, Any], output_path: str):
    """Save detailed crop position log to file"""
    with open(output_path, 'w') as f:
        f.write("Crop Position Analysis Log\n")
        f.write("="*80 + "\n\n")
        
        for dataset_name, dataset_results in results.items():
            f.write(f"Dataset: {dataset_name}\n")
            f.write("-"*80 + "\n")
            
            for split_name, split_results in dataset_results.items():
                f.write(f"\n{split_name.upper()} Split:\n")
                f.write(f"  Images: {split_results['num_images']}\n")
                f.write(f"  Epochs: {split_results['num_epochs']}\n")
                f.write(f"  Unique crops: {split_results['unique_crops']}\n")
                f.write(f"  Total crops: {split_results['total_crops']}\n\n")
                
                f.write(f"  Crop position distribution (top 20 most frequent):\n")
                crop_positions = split_results['crop_positions']
                sorted_crops = sorted(crop_positions.items(), key=lambda x: x[1], reverse=True)
                
                for i, ((y, x), count) in enumerate(sorted_crops[:20]):
                    f.write(f"    {i+1}. Position (y={y}, x={x}): {count} occurrences\n")
                
                f.write(f"\n  Per-image crop statistics (first 10 images):\n")
                per_image = split_results['per_image_crops']
                for i, (image_path, crops) in enumerate(list(per_image.items())[:10]):
                    f.write(f"    Image: {Path(image_path).name}\n")
                    f.write(f"      Unique crops: {len(crops)}\n")
                    f.write(f"      Total crops: {sum(crops.values())}\n")
                
                f.write("\n")
            
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze actual crop positions for presaved pseudo GT'
    )
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument('--epochs', type=int, default=None, 
                       help='Number of epochs to simulate (default: read from config)')
    parser.add_argument('--save-log', type=str, default=None,
                       help='Save detailed crop log to file')
    parser.add_argument('--quiet', action='store_true', 
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    results = analyze_config(
        args.config, 
        num_epochs=args.epochs,
        save_crop_log=args.save_log,
        verbose=not args.quiet
    )
    
    return results


if __name__ == '__main__':
    main()
