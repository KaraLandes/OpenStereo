#!/usr/bin/env python3
"""
Generate presaved pseudo-GT disparities for all expected crop positions.

This script:
1. Analyzes the training configuration to determine all crop positions
2. Loads FoundationStereo model
3. For each image and unique crop position:
   - Crops and resizes the image pair
   - Generates pseudo-GT disparity
   - Saves as .npy file with naming: pseudo_d_{y}_{x}_{frame_id}.npy

Usage:
    python generate_presaved_disparities.py dai/configs/presaved_pseudo.yaml --split train
    python generate_presaved_disparities.py dai/configs/presaved_pseudo.yaml --split val
    python generate_presaved_disparities.py dai/configs/presaved_pseudo.yaml --split test
"""

import sys
from pathlib import Path
import yaml
import random
import numpy as np
import torch
import cv2
from PIL import Image
from typing import Dict, Any, List, Tuple, Set
import argparse
from tqdm import tqdm
import logging
import json
from datetime import datetime

# Add repo to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from dai.datasets.discovery_strategies import IRSDiscovery
from dai.datasets.splitter import DatasetSplitter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PresavedDisparityGenerator:
    """Generator for presaved pseudo-GT disparities."""
    
    def __init__(self, config_path: str, split: str = 'train'):
        """
        Initialize generator.
        
        Args:
            config_path: Path to training config YAML
            split: 'train', 'val', or 'test'
        """
        self.config_path = Path(config_path)
        self.split = split
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract settings
        self.global_seed = self.config.get('seed', 28)
        self.num_epochs = self.config.get('optimization', {}).get('num_epochs', 90)
        self.splitter_config = self.config.get('splitter', {})
        self.datasets_config = self.config.get('datasets', [])
        self.transforms_config = self.config.get('transforms', {})
        self.foundationstereo_config = self.config.get('foundationstereo', {})
        
        # FoundationStereo settings
        self.model_size = self.foundationstereo_config.get('model_size', 'small')
        self.inference_iters = self.foundationstereo_config.get('inference_iters', 12)
        self.use_amp = self.foundationstereo_config.get('inference_amp', True)
        self.batch_size = self.foundationstereo_config.get('inference_batch_size', 8)
        
        # Get crop and resize sizes
        self.crop_size = None
        self.resize_size = None
        self.stride = None
        for transform in self.transforms_config.get(split, []):
            if transform['name'] == 'RandomCrop':
                self.crop_size = transform['size']
            elif transform['name'] == 'StridedRandomCrop':
                self.crop_size = transform['size']
                self.stride = transform.get('stride', 50)
            elif transform['name'] == 'Resize':
                self.resize_size = transform['size']
        
        if self.crop_size is None:
            raise ValueError(f"No RandomCrop or StridedRandomCrop found in {split} transforms")
        
        # Model (loaded lazily)
        self.model_dict = None
        
        logger.info(f"Initialized generator for {split} split")
        logger.info(f"  Config: {self.config_path.name}")
        logger.info(f"  Seed: {self.global_seed}")
        logger.info(f"  Epochs: {self.num_epochs}")
        logger.info(f"  Crop size: {self.crop_size}")
        logger.info(f"  Crop stride: {self.stride if self.stride else 'fully random'}")
        logger.info(f"  Resize size: {self.resize_size}")
        logger.info(f"  FoundationStereo: {self.model_size}")
        logger.info(f"  Batch size: {self.batch_size}")
    
    def load_foundation_stereo(self):
        """Load FoundationStereo model."""
        if self.model_dict is not None:
            return
        
        logger.info(f"Loading FoundationStereo ({self.model_size}) model...")
        
        from stereo.modeling.models.foundationstereo.core.foundation_stereo import FoundationStereo
        from easydict import EasyDict
        
        checkpoint_dir = repo_root / "src_data" / "checkpoints" / "foundationstereo" / self.model_size
        cfg_path = checkpoint_dir / "cfg.yaml"
        
        if not cfg_path.exists():
            raise FileNotFoundError(f"FoundationStereo config not found: {cfg_path}")
        
        with open(cfg_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        args = EasyDict(cfg_dict)
        
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
        
        self.model_dict = {'model': model, 'device': device, 'args': args}
        logger.info(f"FoundationStereo loaded on {device}")
    
    def generate_pseudo_disparity_batch(self, left_imgs: List[np.ndarray], right_imgs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate pseudo disparities for a batch of image pairs.
        
        Args:
            left_imgs: List of left images, each [H, W, 3] in range [0, 255]
            right_imgs: List of right images, each [H, W, 3] in range [0, 255]
            
        Returns:
            List of disparity maps, each [H, W]
        """
        from stereo.modeling.models.foundationstereo.core.foundation_stereo import normalize_image
        from stereo.modeling.models.foundationstereo.core.utils.utils import InputPadder
        
        model = self.model_dict['model']
        device = self.model_dict['device']
        
        # Convert to tensors [B, 3, H, W]
        left_tensors = []
        right_tensors = []
        
        for left_img, right_img in zip(left_imgs, right_imgs):
            left_tensor = torch.from_numpy(left_img.transpose(2, 0, 1)).float()
            right_tensor = torch.from_numpy(right_img.transpose(2, 0, 1)).float()
            left_tensors.append(left_tensor)
            right_tensors.append(right_tensor)
        
        left_batch = torch.stack(left_tensors).to(device).contiguous()
        right_batch = torch.stack(right_tensors).to(device).contiguous()
        
        # Normalize
        left_batch = normalize_image(left_batch).contiguous()
        right_batch = normalize_image(right_batch).contiguous()
        
        # Pad to multiple of 32
        padder = InputPadder(left_batch.shape[-2:], divis_by=32, force_square=False)
        padded = padder.pad(left_batch, right_batch)
        left_padded = padded[0].contiguous()
        right_padded = padded[1].contiguous()
        
        data = {'left': left_padded, 'right': right_padded}
        
        # Run inference (iters determined by model config: args.valid_iters)
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = model(data)
            else:
                output = model(data)
        
        # Get disparity and unpad
        disp_pred = output['disp_pred']
        disp_pred = padder.unpad(disp_pred)
        
        # Convert to list of numpy arrays
        disp_list = [disp_pred[i].cpu().numpy() for i in range(len(disp_pred))]
        
        return disp_list
    
    def generate_pseudo_disparity(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """
        Generate pseudo disparity using FoundationStereo.
        
        Args:
            left_img: Left image [H, W, 3] in range [0, 255]
            right_img: Right image [H, W, 3] in range [0, 255]
            
        Returns:
            Disparity map [H, W]
        """
        from stereo.modeling.models.foundationstereo.core.foundation_stereo import normalize_image
        from stereo.modeling.models.foundationstereo.core.utils.utils import InputPadder
        
        model = self.model_dict['model']
        device = self.model_dict['device']
        
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
        
        # Run inference (iters determined by model config: args.valid_iters)
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = model(data)
            else:
                output = model(data)
        
        # Get disparity and unpad
        disp_pred = output['disp_pred']
        disp_pred = padder.unpad(disp_pred)
        
        # Convert to numpy [H, W]
        disp_np = disp_pred.squeeze().cpu().numpy()
        
        return disp_np
    
    def get_strided_positions(self) -> List[Tuple[int, int]]:
        """
        Get all valid strided crop positions.
        
        Returns:
            List of (y, x) tuples representing all grid positions
        """
        image_size = (540, 960)  # IRS native resolution
        img_h, img_w = image_size
        crop_h, crop_w = self.crop_size
        
        if self.stride is None:
            # Fully random mode - not supported for presaved generation
            raise ValueError(
                "Presaved generation requires StridedRandomCrop with a stride parameter. "
                "Please use StridedRandomCrop in your config instead of RandomCrop."
            )
        
        # Calculate all valid strided positions
        max_y = img_h - crop_h
        max_x = img_w - crop_w
        
        y_positions = list(range(0, max_y + 1, self.stride))
        x_positions = list(range(0, max_x + 1, self.stride))
        
        # Ensure we include the maximum valid position if not already included
        if y_positions[-1] < max_y:
            y_positions.append(max_y)
        if x_positions[-1] < max_x:
            x_positions.append(max_x)
        
        # Generate all combinations
        positions = [(y, x) for y in y_positions for x in x_positions]
        
        return positions
    
    def generate_for_dataset(self, dataset_config: Dict[str, Any], 
                           dry_run: bool = False, max_images: int = None,
                           resume: bool = True):
        """
        Generate presaved disparities for a dataset.
        
        Args:
            dataset_config: Dataset configuration dict
            dry_run: If True, only simulate without generating
            max_images: Maximum number of images to process
            resume: If True, skip already generated files
        """
        dataset_name = dataset_config['name']
        root = Path(dataset_config['root'])
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Root: {root}")
        logger.info(f"{'='*80}")
        
        # Discover samples
        discovery = IRSDiscovery()
        all_samples = discovery.discover_samples(root, dataset_config)
        
        if len(all_samples) == 0:
            logger.warning("No samples found, skipping")
            return
        
        # Split samples
        splitter = DatasetSplitter(self.splitter_config)
        splits = splitter.split_dataset(dataset_name, root, dataset_config)
        
        split_samples = splits[self.split]
        if len(split_samples) == 0:
            logger.warning(f"No samples in {self.split} split, skipping")
            return
        
        # Limit for testing
        if max_images:
            split_samples = split_samples[:max_images]
        
        logger.info(f"Processing {len(split_samples)} images in {self.split} split")
        
        # Get all strided positions (same for every image)
        strided_positions = self.get_strided_positions()
        
        # Calculate total crops to generate
        total_crops = len(split_samples) * len(strided_positions)
        
        logger.info(f"Strided positions per image: {len(strided_positions)}")
        logger.info(f"Total disparities to generate: {total_crops:,} ({len(split_samples):,} images × {len(strided_positions)} positions)")
        
        # Estimate storage
        bytes_per_disp = 256 * 512 * 4  # float32
        storage_gb = (total_crops * bytes_per_disp) / (1024**3)
        logger.info(f"Estimated storage: {storage_gb:.2f} GB")
        
        if dry_run:
            logger.info("DRY RUN - Skipping actual generation")
            return
        
        # Load model
        self.load_foundation_stereo()
        
        # Generate disparities
        total_generated = 0
        total_skipped = 0
        total_errors = 0
        
        # Progress tracking
        progress_file = root / f"presaved_generation_{self.split}_progress.json"
        
        with tqdm(total=len(split_samples), desc=f"Processing {self.split} images") as pbar:
            for sample_idx, sample in enumerate(split_samples):
                left_path = Path(sample['left'])
                right_path = Path(sample['right'])
                frame_id = sample['metadata']['frame_id']
                scene_dir = left_path.parent
                
                try:
                    # Load original images
                    left_img = np.array(Image.open(left_path).convert('RGB'), dtype=np.float32)
                    right_img = np.array(Image.open(right_path).convert('RGB'), dtype=np.float32)
                    
                    # Prepare batch of crops for this image
                    batch_left_crops = []
                    batch_right_crops = []
                    batch_metadata = []  # Store (y, x, presaved_path) for each crop
                    
                    for y, x in strided_positions:
                        presaved_filename = f"pseudo_d_{y}_{x}_{frame_id}.npy"
                        presaved_path = scene_dir / presaved_filename
                        
                        # Skip if already exists and resume enabled
                        if resume and presaved_path.exists():
                            total_skipped += 1
                            continue
                        
                        # Crop images
                        crop_h, crop_w = self.crop_size
                        left_crop = left_img[y:y+crop_h, x:x+crop_w, :]
                        right_crop = right_img[y:y+crop_h, x:x+crop_w, :]
                        
                        # Resize if needed
                        if self.resize_size:
                            target_h, target_w = self.resize_size
                            left_crop = cv2.resize(left_crop, (target_w, target_h), 
                                                  interpolation=cv2.INTER_LINEAR)
                            right_crop = cv2.resize(right_crop, (target_w, target_h), 
                                                   interpolation=cv2.INTER_LINEAR)
                        
                        batch_left_crops.append(left_crop)
                        batch_right_crops.append(right_crop)
                        batch_metadata.append((y, x, presaved_path))
                    
                    # Process crops in batches
                    if len(batch_left_crops) > 0:
                        for i in range(0, len(batch_left_crops), self.batch_size):
                            batch_end = min(i + self.batch_size, len(batch_left_crops))
                            
                            try:
                                # Generate batch of disparities
                                pseudo_disps = self.generate_pseudo_disparity_batch(
                                    batch_left_crops[i:batch_end],
                                    batch_right_crops[i:batch_end]
                                )
                                
                                # Save each disparity
                                for j, pseudo_disp in enumerate(pseudo_disps):
                                    _, _, presaved_path = batch_metadata[i + j]
                                    np.save(presaved_path, pseudo_disp.astype(np.float16))
                                    total_generated += 1
                                    
                            except Exception as e:
                                logger.error(f"Error generating batch {i}-{batch_end}: {e}")
                                total_errors += (batch_end - i)
                    
                except Exception as e:
                    logger.error(f"Error loading images for {left_path.name}: {e}")
                    total_errors += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    'generated': total_generated,
                    'skipped': total_skipped,
                    'errors': total_errors
                })
                
                # Save progress periodically
                if (sample_idx + 1) % 100 == 0:
                    progress = {
                        'last_processed': sample_idx + 1,
                        'total_generated': total_generated,
                        'total_skipped': total_skipped,
                        'total_errors': total_errors,
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(progress_file, 'w') as f:
                        json.dump(progress, f, indent=2)
        
        # Final summary
        logger.info(f"\n{'='*80}")
        logger.info(f"Generation Complete - {self.split.upper()} Split")
        logger.info(f"{'='*80}")
        logger.info(f"Generated: {total_generated:,}")
        logger.info(f"Skipped (already exist): {total_skipped:,}")
        logger.info(f"Errors: {total_errors:,}")
        logger.info(f"{'='*80}\n")
        
        # Save final progress
        progress = {
            'completed': True,
            'total_generated': total_generated,
            'total_skipped': total_skipped,
            'total_errors': total_errors,
            'timestamp': datetime.now().isoformat()
        }
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def run(self, dry_run: bool = False, max_images: int = None, resume: bool = True):
        """
        Run generation for all configured datasets.
        
        Args:
            dry_run: If True, only simulate without generating
            max_images: Maximum number of images to process per dataset
            resume: If True, skip already generated files
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Presaved Pseudo-GT Generation - {self.split.upper()} Split")
        logger.info(f"{'='*80}\n")
        
        for dataset_config in self.datasets_config:
            self.generate_for_dataset(dataset_config, dry_run, max_images, resume)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"All datasets processed!")
        logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate presaved pseudo-GT disparities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for training split
  python generate_presaved_disparities.py dai/configs/presaved_pseudo.yaml --split train
  
  # Test with first 10 images
  python generate_presaved_disparities.py dai/configs/presaved_pseudo.yaml --split train --max-images 10
  
  # Dry run to see what would be generated
  python generate_presaved_disparities.py dai/configs/presaved_pseudo.yaml --split train --dry-run
  
  # Generate from scratch (don't skip existing files)
  python generate_presaved_disparities.py dai/configs/presaved_pseudo.yaml --split train --no-resume
        """
    )
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to process (default: train)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate without generating files')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to process (for testing)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Regenerate all files (do not skip existing)')
    
    args = parser.parse_args()
    
    try:
        generator = PresavedDisparityGenerator(args.config, args.split)
        generator.run(
            dry_run=args.dry_run,
            max_images=args.max_images,
            resume=not args.no_resume
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
