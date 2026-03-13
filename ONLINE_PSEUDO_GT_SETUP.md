# Online Pseudo Ground Truth Setup Guide

## Overview
The `pseudo-foundationstereo-online` mode requires FoundationStereo model dependencies to be installed.

## Required Dependencies

### Already Installed ✓
- `trimesh`
- `joblib`
- `open3d`
- `imageio`
- `transformations`

### Still Needed
- `flash-attn` (requires CUDA compilation)

## Installing flash-attn

### Option 1: Install with CUDA (Recommended)

1. **Set CUDA_HOME environment variable:**
```bash
export CUDA_HOME=/usr/local/cuda  # Adjust path to your CUDA installation
# Or find it with: which nvcc | sed 's/\/bin\/nvcc//'
```

2. **Install flash-attn:**
```bash
source .venv/bin/activate
pip install flash-attn --no-build-isolation
```

### Option 2: Install pre-built wheel (Faster)

Check if a pre-built wheel is available for your CUDA version:
```bash
pip install flash-attn --no-build-isolation --find-links https://github.com/Dao-AILab/flash-attention/releases
```

### Option 3: Skip flash-attn (Fallback)

If flash-attn installation fails, you can modify FoundationStereo to use standard attention instead of flash attention. This will be slower but will work.

## Testing the Implementation

Once dependencies are installed, run:

```bash
# Using VS Code debugger
# Press F5 and select "Test Online Pseudo GT Mode"

# Or via command line:
python dai/main.py --config dai/configs/test_online_pseudo.yaml
```

## Expected Behavior

1. **Initialization:**
   - Loads LightStereo model (student)
   - Loads FoundationStereo model (teacher, frozen)
   - FoundationStereo converted to bfloat16 for memory efficiency

2. **Training Loop:**
   - For each batch:
     - Denormalize images (normalized → [0-255])
     - Generate pseudo GT with FoundationStereo
     - Train LightStereo to match pseudo GT
     - Only LightStereo weights updated

3. **Logging:**
   - Metrics logged every 10 iterations
   - WandB tracking enabled
   - Checkpoints saved to `src_data/test_online_pseudo/`

## Troubleshooting

### CUDA_HOME not set
```bash
# Find CUDA installation
which nvcc
# Set CUDA_HOME (add to ~/.bashrc for persistence)
export CUDA_HOME=/usr/local/cuda
```

### nvcc not found
Install CUDA toolkit or use a Docker image with CUDA development tools:
```bash
# Example: PyTorch Docker with CUDA
docker pull pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
```

### Out of Memory
- Reduce batch size in config (currently 2)
- Use smaller FoundationStereo model size
- Disable bfloat16 conversion (edit online_pseudo_trainer.py)

## Configuration

Edit `dai/configs/test_online_pseudo.yaml`:

```yaml
datasets:
  - target_source: pseudo-foundationstereo-online  # Enable online mode

foundationstereo:
  model_size: small  # Options: small, base, large
```

## Files Modified

1. **New Trainer:** `dai/training/online_pseudo_trainer.py`
2. **Dataset Support:** `dai/datasets/base_dataset.py` (added online mode)
3. **Trainer Factory:** `dai/training/__init__.py` (auto-detection)
4. **Config:** `dai/configs/stereo_flow.yaml` (documentation)
5. **Test Config:** `dai/configs/test_online_pseudo.yaml`
6. **Debugger:** `.vscode/launch.json`
