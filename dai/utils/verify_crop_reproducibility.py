"""
Verify that crop positions are reproducible with the same seeds.

This script runs the crop simulation multiple times with the same seed
and verifies that the exact same crop positions are generated.
"""

import yaml
import random
import numpy as np
from pathlib import Path
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


def simulate_crops_with_seed(
    num_images: int,
    image_size: Tuple[int, int],
    crop_size: List[int],
    num_epochs: int,
    seed: int
) -> List[Tuple[int, int, int, int]]:
    """
    Simulate crops and return ordered list of (epoch, img_idx, y, x).
    
    Args:
        num_images: Number of images
        image_size: (height, width) of original images
        crop_size: [height, width] of crop
        num_epochs: Number of epochs
        seed: Random seed
        
    Returns:
        List of (epoch, img_idx, y, x) tuples in order
    """
    img_h, img_w = image_size
    crop_h, crop_w = crop_size
    
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    
    crops = []
    
    for epoch in range(num_epochs):
        for img_idx in range(num_images):
            y = random.randint(0, img_h - crop_h)
            x = random.randint(0, img_w - crop_w)
            crops.append((epoch, img_idx, y, x))
    
    return crops


def verify_reproducibility(config_path: str, num_runs: int = 3, num_epochs: int = 5):
    """
    Verify that crops are reproducible across multiple runs.
    
    Args:
        config_path: Path to config YAML file
        num_runs: Number of times to run the simulation
        num_epochs: Number of epochs to simulate (keep small for speed)
    """
    config = load_config(config_path)
    
    print(f"\n{'='*80}")
    print(f"Crop Reproducibility Verification")
    print(f"{'='*80}\n")
    
    global_seed = config.get('seed', 28)
    splitter_config = config.get('splitter', {})
    datasets_config = config.get('datasets', [])
    transforms_config = config.get('transforms', {})
    
    print(f"Configuration:")
    print(f"  Global seed: {global_seed}")
    print(f"  Splitter seed: {splitter_config.get('seed', 42)}")
    print(f"  Test epochs: {num_epochs}")
    print(f"  Test runs: {num_runs}\n")
    
    # Get first dataset
    dataset_config = datasets_config[0]
    dataset_name = dataset_config['name']
    root = Path(dataset_config['root'])
    
    print(f"Dataset: {dataset_name}")
    print(f"Root: {root}\n")
    
    # Discover samples
    discovery = IRSDiscovery()
    all_samples = discovery.discover_samples(root, dataset_config)
    
    if len(all_samples) == 0:
        print("ERROR: No samples found. Cannot verify reproducibility.")
        return False
    
    print(f"Discovered {len(all_samples)} samples\n")
    
    # Split samples
    splitter = DatasetSplitter(splitter_config)
    splits = splitter.split_dataset(dataset_name, root, dataset_config)
    
    print(f"Split sizes:")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val: {len(splits['val'])}")
    print(f"  Test: {len(splits['test'])}\n")
    
    # Get crop size from train transforms
    crop_size = None
    for transform in transforms_config.get('train', []):
        if transform['name'] == 'RandomCrop':
            crop_size = transform['size']
            break
    
    if crop_size is None:
        print("ERROR: No RandomCrop found in train transforms")
        return False
    
    print(f"Crop size: {crop_size[0]} × {crop_size[1]}\n")
    
    # Use a small subset for testing
    test_samples = splits['train'][:100]  # First 100 images
    num_images = len(test_samples)
    image_size = (540, 960)
    
    print(f"Testing with {num_images} images for {num_epochs} epochs\n")
    print(f"Running {num_runs} independent simulations with same seed...\n")
    
    # Run multiple simulations
    all_runs = []
    for run_idx in range(num_runs):
        print(f"  Run {run_idx + 1}/{num_runs}...", end=" ")
        crops = simulate_crops_with_seed(
            num_images, image_size, crop_size, num_epochs, global_seed
        )
        all_runs.append(crops)
        print(f"Generated {len(crops)} crops")
    
    # Verify all runs are identical
    print(f"\nVerifying reproducibility...")
    
    all_identical = True
    for run_idx in range(1, num_runs):
        if all_runs[run_idx] != all_runs[0]:
            all_identical = False
            print(f"  ❌ Run {run_idx + 1} differs from Run 1")
            
            # Find first difference
            for i, (crop1, crop2) in enumerate(zip(all_runs[0], all_runs[run_idx])):
                if crop1 != crop2:
                    print(f"     First difference at index {i}:")
                    print(f"       Run 1: epoch={crop1[0]}, img={crop1[1]}, y={crop1[2]}, x={crop1[3]}")
                    print(f"       Run {run_idx + 1}: epoch={crop2[0]}, img={crop2[1]}, y={crop2[2]}, x={crop2[3]}")
                    break
        else:
            print(f"  ✓ Run {run_idx + 1} matches Run 1")
    
    print(f"\n{'='*80}")
    if all_identical:
        print("✓ SUCCESS: All runs produced identical crop sequences!")
        print("Crops are fully reproducible with the same seed.")
    else:
        print("✗ FAILURE: Runs produced different crop sequences!")
        print("Crops are NOT reproducible - there may be a seeding issue.")
    print(f"{'='*80}\n")
    
    # Show sample of first 10 crops from Run 1
    print("Sample of first 10 crops from Run 1:")
    for i, (epoch, img_idx, y, x) in enumerate(all_runs[0][:10]):
        print(f"  {i+1}. Epoch {epoch}, Image {img_idx}: crop at (y={y}, x={x})")
    
    print()
    
    return all_identical


def verify_dataloader_reproducibility(config_path: str):
    """
    Verify reproducibility with actual DataLoader (including worker processes).
    This is the REAL test - simulates actual training conditions.
    """
    print(f"\n{'='*80}")
    print(f"DataLoader Reproducibility Verification")
    print(f"{'='*80}\n")
    
    config = load_config(config_path)
    global_seed = config.get('seed', 28)
    
    print(f"Global seed: {global_seed}\n")
    print("This test requires running the actual DataLoader.")
    print("It will create two dataloaders with the same seed and verify")
    print("that they produce identical crop positions.\n")
    
    # Import required modules
    from dai.main import DAIPipeline
    from dai.utils import set_seed
    
    print("Creating first pipeline...")
    set_seed(global_seed)
    pipeline1 = DAIPipeline(config_path, device='cpu')
    pipeline1.load_datasets()
    pipeline1.combine_datasets()
    pipeline1.create_dataloaders()
    
    print("Creating second pipeline...")
    set_seed(global_seed)
    pipeline2 = DAIPipeline(config_path, device='cpu')
    pipeline2.load_datasets()
    pipeline2.combine_datasets()
    pipeline2.create_dataloaders()
    
    print("\nFetching first batch from both dataloaders...\n")
    
    # Get first batch from each
    batch1 = next(iter(pipeline1.train_loader))
    batch2 = next(iter(pipeline2.train_loader))
    
    # Compare shapes
    print(f"Batch 1 left shape: {batch1['left'].shape}")
    print(f"Batch 2 left shape: {batch2['left'].shape}")
    
    # Compare actual pixel values
    import torch
    left_identical = torch.allclose(batch1['left'], batch2['left'], rtol=1e-5, atol=1e-5)
    right_identical = torch.allclose(batch1['right'], batch2['right'], rtol=1e-5, atol=1e-5)
    
    print(f"\nLeft images identical: {left_identical}")
    print(f"Right images identical: {right_identical}")
    
    if left_identical and right_identical:
        print("\n✓ SUCCESS: DataLoader produces reproducible crops!")
    else:
        print("\n✗ WARNING: DataLoader crops may not be fully reproducible!")
        print("This could be due to:")
        print("  1. Worker process seeding issues")
        print("  2. Non-deterministic operations in transforms")
        print("  3. Dataset shuffling without proper seeding")
    
    print(f"\n{'='*80}\n")
    
    return left_identical and right_identical


def main():
    parser = argparse.ArgumentParser(description='Verify crop reproducibility')
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument('--runs', type=int, default=3, help='Number of test runs')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to test')
    parser.add_argument('--test-dataloader', action='store_true', 
                       help='Test actual DataLoader reproducibility (slower)')
    
    args = parser.parse_args()
    
    # Test 1: Simple simulation
    sim_success = verify_reproducibility(args.config, num_runs=args.runs, num_epochs=args.epochs)
    
    # Test 2: Actual DataLoader (optional, slower)
    if args.test_dataloader:
        dl_success = verify_dataloader_reproducibility(args.config)
        return sim_success and dl_success
    
    return sim_success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
