"""
Test DataLoader reproducibility with actual worker processes.

This is the definitive test - it runs the actual DataLoader with multiple workers
and verifies that crops are reproducible across runs.
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import torch
import random
import numpy as np
import argparse
from dai.config import load_config
from dai.utils import set_seed


def test_dataloader_reproducibility(config_path: str, num_batches: int = 3):
    """
    Test that DataLoader produces reproducible crops with worker processes.
    
    Args:
        config_path: Path to config YAML file
        num_batches: Number of batches to compare
    """
    from dai.main import DAIPipeline
    
    config = load_config(config_path)
    global_seed = config.get('seed', 28)
    num_workers = config.get('dataloader', {}).get('num_workers', 4)
    
    print(f"\n{'='*80}")
    print(f"DataLoader Reproducibility Test")
    print(f"{'='*80}\n")
    print(f"Config: {Path(config_path).name}")
    print(f"Global seed: {global_seed}")
    print(f"Num workers: {num_workers}")
    print(f"Test batches: {num_batches}\n")
    
    # Run 1
    print("Creating Pipeline 1...")
    set_seed(global_seed)
    pipeline1 = DAIPipeline(config_path, device='cpu')
    if not pipeline1.setup_data():
        print("ERROR: Failed to setup data")
        return False
    
    # Run 2
    print("Creating Pipeline 2...")
    set_seed(global_seed)
    pipeline2 = DAIPipeline(config_path, device='cpu')
    if not pipeline2.setup_data():
        print("ERROR: Failed to setup data")
        return False
    
    print(f"\nFetching {num_batches} batches from each dataloader...\n")
    
    # Collect batches
    batches1 = []
    batches2 = []
    
    train_iter1 = iter(pipeline1.train_loader)
    train_iter2 = iter(pipeline2.train_loader)
    
    for i in range(num_batches):
        print(f"  Batch {i+1}/{num_batches}...", end=" ")
        batch1 = next(train_iter1)
        batch2 = next(train_iter2)
        batches1.append(batch1)
        batches2.append(batch2)
        print("✓")
    
    # Compare batches
    print(f"\nComparing batches...\n")
    
    all_match = True
    for i in range(num_batches):
        batch1 = batches1[i]
        batch2 = batches2[i]
        
        # Compare left images
        left_match = torch.allclose(batch1['left'], batch2['left'], rtol=1e-5, atol=1e-5)
        right_match = torch.allclose(batch1['right'], batch2['right'], rtol=1e-5, atol=1e-5)
        
        # Check if any pixels differ
        if not left_match:
            left_diff = torch.abs(batch1['left'] - batch2['left']).max().item()
            print(f"  Batch {i+1}: Left images differ (max diff: {left_diff:.6f})")
            all_match = False
        elif not right_match:
            right_diff = torch.abs(batch1['right'] - batch2['right']).max().item()
            print(f"  Batch {i+1}: Right images differ (max diff: {right_diff:.6f})")
            all_match = False
        else:
            print(f"  Batch {i+1}: ✓ Identical")
    
    print(f"\n{'='*80}")
    if all_match:
        print("✓ SUCCESS: DataLoader is fully reproducible!")
        print(f"All {num_batches} batches matched perfectly across runs.")
        print("\nConclusion: Crop positions are deterministic and can be presaved.")
    else:
        print("✗ WARNING: DataLoader shows non-deterministic behavior!")
        print("\nPossible causes:")
        print("  1. Worker processes not properly seeded")
        print("  2. Dataset shuffling without proper generator")
        print("  3. Non-deterministic operations in transforms")
        print("\nRecommendation: Fix seeding before presaving crops.")
    print(f"{'='*80}\n")
    
    return all_match


def main():
    parser = argparse.ArgumentParser(description='Test DataLoader reproducibility')
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument('--batches', type=int, default=3, help='Number of batches to test')
    
    args = parser.parse_args()
    
    success = test_dataloader_reproducibility(args.config, num_batches=args.batches)
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
