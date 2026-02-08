#!/usr/bin/env python3
"""
Quick script to verify current training configuration
"""

import yaml
from pathlib import Path

config_path = Path("dai/configs/stereo_flow.yaml")

with open(config_path) as f:
    config = yaml.safe_load(f)

print("="*60)
print("CURRENT CONFIGURATION CHECK")
print("="*60)

# Check critical performance settings
print("\n📊 DataLoader Settings:")
print(f"  batch_size: {config['dataloader']['batch_size']}")
print(f"  num_workers: {config['dataloader']['num_workers']}")
print(f"  persistent_workers: {config['dataloader']['persistent_workers']}")

print("\n⚡ Optimization Settings:")
print(f"  amp (mixed precision): {config['optimization']['amp']}")
print(f"  num_epochs: {config['optimization']['num_epochs']}")
print(f"  learning_rate: {config['optimization']['optimizer']['lr']}")

print("\n📝 Logging Settings:")
print(f"  log_interval: {config['logging']['log_interval']}")

print("\n💾 Dataset Settings:")
for ds in config['datasets']:
    print(f"  {ds['name']}:")
    print(f"    subsets: {ds['subsets']}")
    print(f"    target_source: {ds['target_source']}")

print("\n" + "="*60)
print("EXPECTED PERFORMANCE:")
print("="*60)

# Calculate expected performance
train_samples = 23800  # Approximate for 3 SceneFlow subsets
batch_size = config['dataloader']['batch_size']
steps_per_epoch = train_samples / batch_size

print(f"\nEstimated steps per epoch: {steps_per_epoch:.0f}")
print(f"\nWith current settings:")
print(f"  - AMP={'✅ ENABLED' if config['optimization']['amp'] else '❌ DISABLED'}")
print(f"  - Workers: {config['dataloader']['num_workers']}")
print(f"  - Batch size: {batch_size}")

if config['optimization']['amp'] and config['dataloader']['num_workers'] >= 8:
    print(f"\n✅ Expected speed: ~0.2-0.3s/step")
    print(f"✅ Expected epoch time: ~{steps_per_epoch * 0.25 / 60:.1f} minutes")
else:
    print(f"\n⚠️  Suboptimal settings detected!")
    if not config['optimization']['amp']:
        print(f"   - AMP disabled (enable for 1.5-2x speedup)")
    if config['dataloader']['num_workers'] < 8:
        print(f"   - Only {config['dataloader']['num_workers']} workers (use 8-12)")
    print(f"\n❌ Current speed: ~1-2s/step")
    print(f"❌ Current epoch time: ~{steps_per_epoch * 1.5 / 60:.1f} minutes")

print("\n" + "="*60)
