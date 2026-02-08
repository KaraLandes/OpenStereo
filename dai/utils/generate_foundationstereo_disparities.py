#!/usr/bin/env python3
"""
Standalone script to pre-generate FoundationStereo disparity predictions for datasets.

This script:
1. Discovers samples from a dataset (SceneFlow, KITTI12, etc.)
2. Loads FoundationStereo model
3. Runs inference on each stereo pair
4. Saves disparity predictions in a mirrored directory structure

Usage:
    python generate_foundationstereo_disparities.py --dataset sceneflow --root ./src_data/sceneflow
    python generate_foundationstereo_disparities.py --dataset kitti12 --root ./src_data/kitti12
"""

import sys
import logging
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import yaml
from easydict import EasyDict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add repo to path (script is in dai/utils/, so go up 2 levels to repo root)
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))


def load_foundation_stereo_model(model_size='small'):
    """Load FoundationStereo model from checkpoint"""
    from stereo.modeling.models.foundationstereo.core.foundation_stereo import FoundationStereo
    
    checkpoint_dir = repo_root / "src_data" / "checkpoints" / "foundationstereo" / model_size
    cfg_path = checkpoint_dir / "cfg.yaml"
    
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found at {cfg_path}")
    
    logger.info(f"Loading {model_size} FoundationStereo model config from {cfg_path}")
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    args = EasyDict(cfg_dict)
    
    # Create model
    logger.info("Creating FoundationStereo model...")
    model = FoundationStereo(args)
    
    # Load checkpoint
    checkpoint_path = checkpoint_dir / "model_best_bp2.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    logger.info("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    # Move to CUDA if available (no multiprocessing issues in standalone script)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Moving model to device: {device}")
    model = model.to(device)
    
    # Convert eligible layers to bfloat16 for memory efficiency
    # Keep normalization layers (BatchNorm, LayerNorm, GroupNorm) in float32 for stability
    if device.type == 'cuda':
        logger.info("Converting eligible layers to bfloat16...")
        for name, module in model.named_modules():
            # Skip normalization layers
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d, 
                                  torch.nn.LayerNorm, torch.nn.GroupNorm,
                                  torch.nn.InstanceNorm2d)):
                continue
            # Convert linear and conv layers to bfloat16
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d, 
                                  torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
                module.to(torch.bfloat16)
        logger.info("✓ Model converted to bfloat16 (normalization layers kept in float32)")
    
    model.eval()
    
    logger.info("✓ FoundationStereo model loaded successfully")
    return model, device, args


def run_inference(model, device, left_img, right_img, downscale_factor=1.0):
    """
    Run FoundationStereo inference on a stereo pair.
    
    Args:
        model: FoundationStereo model
        device: torch device
        left_img: Left image [H, W, 3] in range [0, 255]
        right_img: Right image [H, W, 3] in range [0, 255]
        downscale_factor: Factor to downscale images (e.g., 0.5 for half size)
        
    Returns:
        Disparity map [H, W] at original resolution
    """
    import cv2
    from stereo.modeling.models.foundationstereo.core.foundation_stereo import normalize_image
    from stereo.modeling.models.foundationstereo.core.utils.utils import InputPadder
    
    # Store original size
    orig_h, orig_w = left_img.shape[:2]
    
    # Downscale images if needed
    if downscale_factor != 1.0:
        new_h = int(orig_h * downscale_factor)
        new_w = int(orig_w * downscale_factor)
        left_img = cv2.resize(left_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        right_img = cv2.resize(right_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensors [1, 3, H, W]
    left_tensor = torch.from_numpy(left_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    right_tensor = torch.from_numpy(right_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    
    # Normalize images
    left_tensor = normalize_image(left_tensor)
    right_tensor = normalize_image(right_tensor)
    
    # Convert to bfloat16 if model uses it (check first conv layer)
    first_conv = None
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            first_conv = module
            break
    if first_conv is not None and first_conv.weight.dtype == torch.bfloat16:
        left_tensor = left_tensor.to(torch.bfloat16)
        right_tensor = right_tensor.to(torch.bfloat16)
    
    # Pad to multiple of 32
    padder = InputPadder(left_tensor.shape[-2:], divis_by=32, force_square=False)
    left_padded, right_padded = padder.pad(left_tensor, right_tensor)
    
    # Prepare data dict
    data = {
        'left': left_padded,
        'right': right_padded
    }
    
    # Run inference
    with torch.no_grad():
        output = model(data)
    
    # Get disparity and unpad
    disp_pred = output['disp_pred']
    disp_pred = padder.unpad(disp_pred)
    
    # Convert to numpy [H, W]
    disp_np = disp_pred.squeeze().cpu().numpy()
    
    # Upscale disparity back to original resolution if it was downscaled
    if downscale_factor != 1.0:
        disp_np = cv2.resize(disp_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        # Scale disparity values proportionally
        disp_np = disp_np / downscale_factor
    
    return disp_np


def save_disparity_pfm(filepath, disparity):
    """Save disparity as PFM file (SceneFlow format)"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # PFM format: header + binary data
    height, width = disparity.shape
    scale = -1  # Little endian
    
    with open(filepath, 'wb') as f:
        # Write header
        f.write(b'Pf\n')
        f.write(f'{width} {height}\n'.encode())
        f.write(f'{scale}\n'.encode())
        # Write data (flip vertically for PFM format)
        disparity_flipped = np.flipud(disparity).astype(np.float32)
        disparity_flipped.tofile(f)
    
    logger.debug(f"Saved PFM: {filepath}")


def save_disparity_png(filepath, disparity):
    """Save disparity as PNG file (KITTI format)"""
    import cv2
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # KITTI format: disparity * 256 as uint16
    disparity_uint16 = (disparity * 256.0).astype(np.uint16)
    cv2.imwrite(str(filepath), disparity_uint16)
    
    logger.debug(f"Saved PNG: {filepath}")


def discover_sceneflow_samples(root, subsets, pass_type='cleanpass'):
    """Discover SceneFlow samples"""
    from dai.datasets.discovery_strategies import SceneFlowDiscovery
    
    root = Path(root)
    strategy = SceneFlowDiscovery()
    
    config = {
        'root': str(root),
        'subsets': subsets,
        'pass_type': pass_type
    }
    
    samples = strategy.discover_samples(root, config)
    logger.info(f"Discovered {len(samples)} SceneFlow samples")
    return samples


def discover_kitti12_samples(root, split='training'):
    """Discover KITTI12 samples"""
    from dai.datasets.discovery_strategies import KITTI12Discovery
    
    root = Path(root)
    strategy = KITTI12Discovery()
    
    config = {
        'root': str(root),
        'split': split
    }
    
    samples = strategy.discover_samples(root, config)
    logger.info(f"Discovered {len(samples)} KITTI12 samples")
    return samples


def process_sceneflow(model, device, root, subsets, pass_type='cleanpass', output_suffix='foundationstereo', downscale_factor=1.0, testing_mode=False):
    """Process SceneFlow dataset and save FoundationStereo disparities"""
    from PIL import Image
    import matplotlib.pyplot as plt
    
    root = Path(root)
    samples = discover_sceneflow_samples(root, subsets, pass_type)
    
    logger.info(f"Processing {len(samples)} SceneFlow samples...")
    if downscale_factor != 1.0:
        logger.info(f"Images will be downscaled by factor {downscale_factor} to reduce memory usage")
    if testing_mode:
        logger.info(f"Testing mode: Will save visualizations to root directory")
    
    for sample in tqdm(samples, desc="Generating disparities"):
        # Load images
        left_img = np.array(Image.open(sample['left']).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(sample['right']).convert('RGB'), dtype=np.float32)
        
        # Run inference with OOM handling
        try:
            disparity = run_inference(model, device, left_img, right_img, downscale_factor)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"GPU OOM on {sample['left']}, retrying on CPU...")
            torch.cuda.empty_cache()
            # Move model to CPU temporarily
            model_cpu = model.cpu()
            disparity = run_inference(model_cpu, torch.device('cpu'), left_img, right_img, downscale_factor)
            # Move model back to GPU
            model.to(device)
            torch.cuda.empty_cache()
        
        # Determine output path (mirror structure with foundationstereo suffix)
        # Example: monkaa/frames_cleanpass/... -> monkaa/disparity_foundationstereo/...
        left_path = Path(sample['left'])
        
        # Find the subset name (monkaa, driving, flyingthings3d) - case insensitive
        subset = None
        subset_actual = None
        path_str_lower = str(left_path).lower()
        for s in subsets:
            if s.lower() in path_str_lower:
                subset = s
                # Find actual capitalization in path
                for part in left_path.parts:
                    if part.lower() == s.lower():
                        subset_actual = part
                        break
                break
        
        if subset is None or subset_actual is None:
            logger.warning(f"Could not determine subset for {left_path}, skipping")
            continue
        
        # Build output path
        # Replace frames_cleanpass/frames_finalpass with disparity_foundationstereo
        relative_path = left_path.relative_to(root / subset_actual)
        parts = list(relative_path.parts)
        
        # Replace frames_* with disparity_foundationstereo
        if parts[0].startswith('frames_'):
            parts[0] = f'disparity_{output_suffix}'
        
        # Change extension to .pfm
        parts[-1] = parts[-1].replace('.png', '.pfm')
        
        output_path = root / subset_actual / Path(*parts)
        
        # Testing mode: Save visualization to root
        if testing_mode:
            frame_name = left_path.stem
            vis_path = Path('.') / f'test_{frame_name}_disparity.png'
            
            # Create figure with left image and disparity using gridspec for equal sizes
            from matplotlib import gridspec
            fig = plt.figure(figsize=(18, 6))
            gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.05)
            
            # Left image
            ax0 = fig.add_subplot(gs[0])
            ax0.imshow(left_img.astype(np.uint8))
            ax0.set_title('Left Image', fontsize=14)
            ax0.axis('off')
            
            # Disparity
            ax1 = fig.add_subplot(gs[1])
            im = ax1.imshow(disparity, cmap='turbo')
            ax1.set_title('FoundationStereo Disparity', fontsize=14)
            ax1.axis('off')
            
            # Colorbar in separate axis
            cax = fig.add_subplot(gs[2])
            plt.colorbar(im, cax=cax)
            
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved test visualization to {vis_path}")
            
            # Only process one sample in testing mode
            break
        
        # Save disparity
        save_disparity_pfm(output_path, disparity)
    
    logger.info(f"✓ Processed {len(samples)} SceneFlow samples")


def process_kitti12(model, device, root, split='training', output_suffix='foundationstereo', downscale_factor=1.0):
    """Process KITTI12 dataset and save FoundationStereo disparities"""
    from PIL import Image
    
    root = Path(root)
    samples = discover_kitti12_samples(root, split)
    
    logger.info(f"Processing {len(samples)} KITTI12 samples...")
    if downscale_factor != 1.0:
        logger.info(f"Images will be downscaled by factor {downscale_factor} to reduce memory usage")
    
    for sample in tqdm(samples, desc="Generating disparities"):
        # Load images
        left_img = np.array(Image.open(sample['left']).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(sample['right']).convert('RGB'), dtype=np.float32)
        
        # Run inference with OOM handling
        try:
            disparity = run_inference(model, device, left_img, right_img, downscale_factor)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"GPU OOM on {sample['left']}, retrying on CPU...")
            torch.cuda.empty_cache()
            # Move model to CPU temporarily
            model_cpu = model.cpu()
            disparity = run_inference(model_cpu, torch.device('cpu'), left_img, right_img, downscale_factor)
            # Move model back to GPU
            model.to(device)
            torch.cuda.empty_cache()
        
        # Determine output path
        # KITTI12 has disp_occ and disp_noc directories
        # We create disp_foundationstereo directory
        left_path = Path(sample['left'])
        
        # Get image filename (e.g., 000000_10.png)
        filename = left_path.name
        
        # Create output directory
        output_dir = root / split / f'disp_{output_suffix}'
        output_path = output_dir / filename
        
        # Save disparity
        save_disparity_png(output_path, disparity)
    
    logger.info(f"✓ Processed {len(samples)} KITTI12 samples")


def main():
    # Hardcoded configuration
    dataset = 'sceneflow'
    root = './src_data/sceneflow'
    subsets = ['driving','flyingthings3d']
    pass_type = 'cleanpass'
    split = 'training'
    model_size = 'small'
    output_suffix = 'foundationstereo'
    downscale_factor = 0.75
    testing_mode = False  # Save visualization to root for testing  
    
    logger.info("=" * 80)
    logger.info("FoundationStereo Disparity Pre-Generation")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Root: {root}")
    logger.info(f"Model size: {model_size}")
    logger.info(f"Output suffix: {output_suffix}")
    
    # Load model
    model, device, model_args = load_foundation_stereo_model(model_size)
    
    # Process dataset
    if dataset == 'sceneflow':
        logger.info(f"Subsets: {subsets}")
        logger.info(f"Pass type: {pass_type}")
        process_sceneflow(
            model, device, root, subsets, pass_type, output_suffix, downscale_factor, testing_mode
        )
    elif dataset == 'kitti12':
        logger.info(f"Split: {split}")
        process_kitti12(
            model, device, root, split, output_suffix, downscale_factor
        )
    
    logger.info("=" * 80)
    logger.info("✓ ALL DISPARITIES GENERATED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
