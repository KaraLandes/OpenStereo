"""
Warping sequential trainer for temporal stereo matching with disparity warping.
Implements training loop with optical flow computation and disparity warping.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import sys
from tqdm import tqdm

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from stereo.modeling.models.lightstereo.warping_lightstereo import WarpingLightStereo
from stereo.modeling.models.lightstereo.warping_utils import prepare_frame_input
from .trainer import DAITrainer
from .metrics import StereoMetrics

logger = logging.getLogger(__name__)


class WarpingSequentialTrainer(DAITrainer):
    """
    Trainer for temporal stereo matching with disparity warping.
    
    Extends DAITrainer to handle:
    - Sequential data loading (sequences of frames)
    - Optical flow computation between frames
    - Disparity warping using flow
    - 6-channel input (RGB + warped disparity + flow hints)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda'
    ):
        """
        Initialize warping sequential trainer.
        
        Args:
            config: Training configuration dict
            train_loader: Training data loader (returns sequences)
            val_loader: Validation data loader (returns sequences, optional)
            device: Device to train on ('cuda' or 'cpu')
        """
        super().__init__(config, train_loader, val_loader, device)
        
        # Initialize visualization attributes if not set by parent
        if not hasattr(self, 'viz_samples_cache'):
            self.viz_samples_cache = None
        if not hasattr(self, 'viz_sample_indices'):
            self.viz_sample_indices = [0, 1, 2]
        
        # Frame 0 initialization setting
        self.use_frame0_init = config.get('training', {}).get('use_frame0_init', True)
        
        # Check if using presaved pseudo-GT (need generation for cache misses)
        self._uses_presaved = any(
            ds.get('target_source') == 'pseudo-foundationstereo-presaved'
            for ds in config.get('datasets', [])
        )
        
        if self._uses_presaved:
            self._init_foundation_stereo(config)
            self._cache_hits = 0
            self._cache_misses = 0
        
        logger.info("Initialized WarpingSequentialTrainer for temporal stereo with disparity warping")
        logger.info(f"Frame 0 initialization: {'enabled' if self.use_frame0_init else 'disabled (zero-init)'}")
        if self._uses_presaved:
            logger.info("Cache-or-generate mode: will generate pseudo-GT for cache misses")
    
    def _init_foundation_stereo(self, config):
        """Load FoundationStereo model for cache-miss generation."""
        from dai.training.online_pseudo_trainer import OnlinePseudoGTTrainer
        
        logger.info("Loading FoundationStereo for cache-miss generation...")
        # Reuse the loading logic from OnlinePseudoGTTrainer
        self.foundation_model, self.foundation_device = (
            OnlinePseudoGTTrainer._load_foundation_stereo_model(self)
        )
        
        fs_config = config.get('foundationstereo', {})
        self.foundation_batch_size = fs_config.get('inference_batch_size', 1)
        self.foundation_use_amp = fs_config.get('inference_amp', True)
        self.foundation_iters = fs_config.get('inference_iters', 12)
        
        if hasattr(self.foundation_model, 'args'):
            self.foundation_model.args.valid_iters = self.foundation_iters
        
        # Normalization constants for denormalization
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    def _denormalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Convert normalized images back to [0-255] range for FoundationStereo."""
        mean = self.norm_mean.to(images.device)
        std = self.norm_std.to(images.device)
        return (images * std + mean) * 255.0
    
    def _generate_pseudo_gt(self, left_img: torch.Tensor, right_img: torch.Tensor) -> torch.Tensor:
        """Generate pseudo GT disparity using FoundationStereo."""
        from stereo.modeling.models.foundationstereo.core.foundation_stereo import normalize_image
        from stereo.modeling.models.foundationstereo.core.utils.utils import InputPadder
        
        B = left_img.shape[0]
        left_denorm = self._denormalize_images(left_img)
        right_denorm = self._denormalize_images(right_img)
        
        if left_denorm.device != self.foundation_device:
            left_denorm = left_denorm.to(self.foundation_device)
            right_denorm = right_denorm.to(self.foundation_device)
        
        disp_list = []
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.foundation_use_amp):
            for i in range(0, B, self.foundation_batch_size):
                left_norm = normalize_image(left_denorm[i:i+self.foundation_batch_size])
                right_norm = normalize_image(right_denorm[i:i+self.foundation_batch_size])
                padder = InputPadder(left_norm.shape[-2:], divis_by=32, force_square=False)
                left_pad, right_pad = padder.pad(left_norm, right_norm)
                output = self.foundation_model({'left': left_pad, 'right': right_pad})
                disp_pred = padder.unpad(output['disp_pred'])
                disp_list.append(disp_pred.squeeze(1).float())
            disp_pred = torch.cat(disp_list, dim=0)
            if disp_pred.device != self.device:
                disp_pred = disp_pred.to(self.device)
        return disp_pred
    
    def _handle_frame_cache_miss(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check frame for cache misses and generate + save missing disparities.
        Works on batched frame data (all samples at timestep t).
        """
        if not self._uses_presaved:
            return frame_data
        
        needs_gen = frame_data.get('needs_generation', None)
        if needs_gen is None:
            return frame_data
        
        if not isinstance(needs_gen, torch.Tensor):
            needs_gen = torch.tensor(needs_gen)
        
        num_misses = needs_gen.sum().item()
        num_hits = (~needs_gen).sum().item()
        self._cache_hits += num_hits
        self._cache_misses += num_misses
        
        if num_misses == 0:
            return frame_data
        
        gen_indices = needs_gen.nonzero(as_tuple=True)[0]
        
        # Generate pseudo GT for missing samples
        left_subset = frame_data['left'][gen_indices].to(self.device)
        right_subset = frame_data['right'][gen_indices].to(self.device)
        
        try:
            pseudo_gt = self._generate_pseudo_gt(left_subset, right_subset)
            
            # Fill in generated disparities
            frame_data['disparity'][gen_indices] = pseudo_gt.cpu()
            frame_data['disparity_foundationstereo'] = frame_data.get(
                'disparity_foundationstereo', frame_data['disparity'].clone()
            )
            if frame_data['disparity_foundationstereo'] is not None:
                frame_data['disparity_foundationstereo'][gen_indices] = pseudo_gt.cpu()
            else:
                frame_data['disparity_foundationstereo'] = frame_data['disparity'].clone()
                frame_data['disparity_foundationstereo'][gen_indices] = pseudo_gt.cpu()
            frame_data['valid_mask'][gen_indices] = (pseudo_gt.cpu() > 0) & (pseudo_gt.cpu() < 512)
            frame_data['needs_generation'] = torch.zeros_like(needs_gen)
            
            # Save for future cache hits
            self._save_frame_disparities(frame_data, gen_indices, pseudo_gt)
            
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM during FoundationStereo generation, skipping cache miss")
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Failed to generate pseudo GT: {e}")
        
        return frame_data
    
    def _save_frame_disparities(self, frame_data, gen_indices, pseudo_gt):
        """Save generated disparities to disk as .npy files."""
        metadata = frame_data.get('metadata', None)
        crop_coords = frame_data.get('crop_coords', None)
        
        if metadata is None or crop_coords is None:
            return
        
        for i, batch_idx in enumerate(gen_indices):
            idx = batch_idx.item()
            try:
                # metadata is a list of dicts (one per batch item)
                if isinstance(metadata, list) and idx < len(metadata):
                    frame_id = metadata[idx].get('frame_id', '00000')
                    left_path = metadata[idx].get('left_path', None)
                else:
                    continue
                
                if left_path is None:
                    continue
                
                # crop_coords is a list of dicts (one per batch item) from collate
                if isinstance(crop_coords, list) and idx < len(crop_coords):
                    cc = crop_coords[idx]
                    if isinstance(cc, dict):
                        y = cc.get('y', 0)
                        x = cc.get('x', 0)
                    else:
                        continue
                else:
                    continue
                
                scene_dir = Path(left_path).parent
                presaved_path = scene_dir / f"pseudo_d_{y}_{x}_{frame_id}.npy"
                
                disp_np = pseudo_gt[i].cpu().numpy().astype(np.float16)
                np.save(str(presaved_path), disp_np)
                logger.debug(f"Saved generated disparity: {presaved_path}")
                
            except Exception as e:
                logger.error(f"Failed to save disparity for idx {idx}: {e}")
    
    def _save_visualizations(self, save_dir: Path, epoch: int, prefix: str = 'train'):
        """
        Override parent's visualization to handle sequential data.
        Visualizes the last frame of each sequence.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import numpy as np
        
        if self.val_loader is None:
            return
        
        viz_dir = save_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        if self.viz_samples_cache is None:
            self.viz_samples_cache = []
            dataset = self.val_loader.dataset
            for idx in self.viz_sample_indices:
                if idx < len(dataset):
                    self.viz_samples_cache.append(dataset[idx])
        
        for sample_idx, sample in enumerate(self.viz_samples_cache):
            sequence = sample['sequence']
            last_frame = sequence[-1]
            
            batch = {
                'left': last_frame['left'].unsqueeze(0).to(self.device),
                'right': last_frame['right'].unsqueeze(0).to(self.device)
            }
            
            # Prepare 6-channel input with zero disparity and flow for single-frame inference
            B, C, H, W = batch['left'].shape
            zero_disp = torch.zeros(B, 1, H, W, device=self.device)
            zero_flow = torch.zeros(B, 2, H, W, device=self.device)
            
            left_input = prepare_frame_input(batch['left'], zero_disp, zero_flow, 
                                            use_motion_hints=self.model.use_motion_hints)
            right_input = prepare_frame_input(batch['right'], zero_disp, zero_flow,
                                             use_motion_hints=self.model.use_motion_hints)
            
            with autocast('cuda', enabled=self.use_amp):
                model_output = self.model({'left': left_input, 'right': right_input})
            
            pred_disp = model_output['disp_pred'][0].detach().cpu().numpy().squeeze()
            gt_disp = last_frame['disparity'].detach().cpu().numpy().squeeze()
            valid_mask = last_frame['valid_mask'].detach().cpu().numpy().squeeze()
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Custom colormap with uneven distribution for bins: 0-75, 75-250, 250-512
            bounds = [0, 50, 175, 512]
            # Map bounds to [0, 1] interval for colormap positions
            norm_bounds = np.interp(bounds, [bounds[0], bounds[-1]], [0, 1])
            # Sample full turbo colormap at 256 positions
            turbo_cmap = plt.cm.turbo
            turbo_colors = turbo_cmap(np.linspace(0, 1, 256))
            # Create color list with uneven distribution
            color_positions = []
            for i in range(len(bounds) - 1):
                n_colors = int(256 * (norm_bounds[i+1] - norm_bounds[i]))
                color_positions.extend(np.linspace(i/(len(bounds)-1), (i+1)/(len(bounds)-1), n_colors))
            custom_colors = turbo_cmap(np.array(color_positions))
            custom_cmap = LinearSegmentedColormap.from_list('custom_disp', custom_colors, N=256)
            
            im0 = axes[0].imshow(gt_disp, cmap=custom_cmap, vmin=0, vmax=512)
            axes[0].set_title(f'GT Disparity (Seq {sample_idx}, Last Frame)', fontsize=12)
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            im1 = axes[1].imshow(pred_disp, cmap=custom_cmap, vmin=0, vmax=512)
            axes[1].set_title(f'Predicted Disparity (Seq {sample_idx}, Last Frame)', fontsize=12)
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            error = np.abs(pred_disp - gt_disp) * valid_mask
            im2 = axes[2].imshow(error, cmap='hot', vmin=0, vmax=10)
            axes[2].set_title(f'Error Map (Seq {sample_idx}, Last Frame)', fontsize=12)
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            save_path = viz_dir / f'{prefix}_epoch{epoch:03d}_seq{sample_idx}.png'
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
        
        logger.info(f"Saved {len(self.viz_samples_cache)} {prefix} visualizations to {viz_dir}")
    
    def _build_model(self) -> nn.Module:
        """Build WarpingLightStereo model from config"""
        model_config = self.config.get('model', {})
        
        from easydict import EasyDict
        model_cfg = EasyDict({
            'MAX_DISP': model_config.get('max_disp', 192),
            'LEFT_ATT': model_config.get('left_att', True),
            'AGGREGATION_BLOCKS': model_config.get('aggregation_blocks', [1, 2, 4]),
            'EXPANSE_RATIO': model_config.get('expanse_ratio', 4),
            'BACKBONE': model_config.get('backbone', 'MobileNetv2'),
            'TEMPORAL': {
                'use_motion_hints': model_config.get('temporal', {}).get('use_motion_hints', True),
                'flow_max_displacement': model_config.get('temporal', {}).get('flow_max_displacement', 4),
                'flow_feature_channels': model_config.get('temporal', {}).get('flow_feature_channels', 16),
            }
        })
        
        model = WarpingLightStereo(model_cfg)
        logger.info(f"Built WarpingLightStereo model with config: {model_cfg}")
        
        # Load pretrained LightStereo weights if specified
        pretrained_path = model_config.get('pretrained_checkpoint', None)
        if pretrained_path:
            logger.info(f"Loading pretrained LightStereo weights from: {pretrained_path}")
            model.load_pretrained_lightstereo(pretrained_path)
        else:
            logger.info("No pretrained checkpoint specified. Using random initialization for all weights.")
            logger.info("To use pretrained weights, add 'pretrained_checkpoint' to model config.")
        
        return model
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with sequential data and disparity warping.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of average metrics for the epoch
        """
        self.model.train()
        
        epoch_metrics = {
            'loss': 0.0,
            'epe': 0.0,
            'd1': 0.0
        }
        
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            disable=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            sequence = batch['sequence']
            sequence_length = len(sequence)
            batch_size = sequence[0]['left'].shape[0]
            
            # Handle cache misses for first frame before init
            first_frame = self._handle_frame_cache_miss(sequence[0])
            sequence[0] = first_frame
            
            B, _, H, W = first_frame['left'].shape
            
            if self.use_frame0_init:
                # Use GT or FoundationStereo disparity for initialization
                if 'disparity_foundationstereo' in first_frame and first_frame['disparity_foundationstereo'] is not None:
                    gt_disparity = first_frame['disparity_foundationstereo'].to(self.device)
                    if gt_disparity.dim() == 3:
                        gt_disparity = gt_disparity.unsqueeze(1)
                elif 'disparity' in first_frame and first_frame['disparity'] is not None:
                    gt_disparity = first_frame['disparity'].to(self.device)
                    if gt_disparity.dim() == 3:
                        gt_disparity = gt_disparity.unsqueeze(1)
                else:
                    gt_disparity = torch.zeros(B, 1, H, W, device=self.device)
            else:
                gt_disparity = torch.zeros(B, 1, H, W, device=self.device)
            
            prev_disparity = gt_disparity
            prev_left = first_frame['left'].to(self.device)
            
            total_loss = 0.0
            seq_epe = 0.0
            seq_d1 = 0.0
            
            # Process each frame in the sequence
            for t, frame_data in enumerate(sequence):
                # Handle cache misses: generate pseudo-GT if needed (frame 0 already handled)
                if t > 0:
                    frame_data = self._handle_frame_cache_miss(frame_data)
                
                left_rgb = frame_data['left'].to(self.device)
                right_rgb = frame_data['right'].to(self.device)
                disp_gt = frame_data['disparity'].to(self.device)
                valid_mask = frame_data['valid_mask'].to(self.device)
                
                if t == 0:
                    # Frame 0: Use GT disparity for left, zeros for right
                    zero_flow = torch.zeros(batch_size, 2, left_rgb.shape[2], left_rgb.shape[3], device=self.device)
                    
                    left_input = prepare_frame_input(
                        left_rgb, gt_disparity, zero_flow,
                        use_motion_hints=self.model.use_motion_hints
                    )
                    # Right: zero temporal information (consistent with t>0)
                    zero_disp_right = torch.zeros_like(gt_disparity)
                    right_input = prepare_frame_input(
                        right_rgb, zero_disp_right, zero_flow,
                        use_motion_hints=self.model.use_motion_hints
                    )
                else:
                    # Frame t>0: Compute flow and warp disparity (LEFT ONLY)
                    with torch.no_grad():
                        flow_left = self.model.flow_estimator(prev_left, left_rgb)
                    
                    # Warp previous disparity (left side only)
                    warped_disp_left, mask_left = self.model.disparity_warper(prev_disparity, flow_left)
                    
                    # Prepare 6-channel input for left with temporal information
                    left_input = prepare_frame_input(
                        left_rgb, warped_disp_left, flow_left,
                        use_motion_hints=self.model.use_motion_hints
                    )
                    
                    # Right image: zero temporal information (no warping, no flow)
                    # This maintains 6-channel input but with zeros for temporal channels
                    zero_disp_right = torch.zeros_like(warped_disp_left)
                    zero_flow_right = torch.zeros(batch_size, 2, left_rgb.shape[2], left_rgb.shape[3], device=self.device)
                    right_input = prepare_frame_input(
                        right_rgb, zero_disp_right, zero_flow_right,
                        use_motion_hints=self.model.use_motion_hints
                    )
                
                # Forward pass
                data_dict = {'left': left_input, 'right': right_input}
                output = self.model(data_dict)
                
                # Prepare data dict with ground truth for loss computation
                data_with_gt = {
                    'left': left_rgb,
                    'right': right_rgb,
                    'disparity': disp_gt,
                    'valid_mask': valid_mask
                }
                
                # Compute loss for this frame
                frame_loss, loss_dict = self.loss_fn(output, data_with_gt)
                
                # Add temporal consistency loss if t > 0
                if t > 0 and hasattr(self.loss_fn, 'temporal_consistency_weight'):
                    temporal_loss = self.loss_fn.compute_temporal_consistency(
                        output['disp_pred'], warped_disp_left, mask_left
                    )
                    frame_loss += temporal_loss
                
                total_loss += frame_loss
                
                # Per-frame metrics
                with torch.no_grad():
                    frame_metrics = self.metrics.compute_all_metrics(
                        output['disp_pred'].squeeze(1), disp_gt, valid_mask
                    )
                    seq_epe += frame_metrics['epe']
                    seq_d1 += frame_metrics['d1_all']
                
                # Update for next iteration
                prev_disparity = output['disp_pred'].detach()
                prev_left = left_rgb
            
            # Average loss over sequence
            avg_loss = total_loss / sequence_length
            
            # Scale loss by accumulation steps for correct gradient magnitude
            avg_loss = avg_loss / self.accumulation_steps
            
            # Backpropagation
            avg_loss.backward()
            
            # Only update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.grad_clip_value > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip_value)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if self.scheduler is not None:
                    sched_config = self.config.get('optimization', {}).get('scheduler', {})
                    if not sched_config.get('on_epoch', False):
                        self.scheduler.step()
            
            # Average metrics over sequence
            avg_epe = seq_epe / sequence_length
            avg_d1 = seq_d1 / sequence_length
            
            epoch_metrics['loss'] += avg_loss.item() * self.accumulation_steps
            epoch_metrics['epe'] += avg_epe
            epoch_metrics['d1'] += avg_d1
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f"{avg_loss.item() * self.accumulation_steps:.4f}",
                'epe': f"{avg_epe:.4f}",
                'd1': f"{avg_d1:.4f}"
            })
            
            log_interval = self.config.get('logging', {}).get('log_interval', 250)
            if (batch_idx + 1) % log_interval == 0:
                try:
                    import wandb
                    wandb.log({
                        'train/loss': avg_loss.item() * self.accumulation_steps,
                        'train/epe': avg_epe,
                        'train/d1_all': avg_d1,
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'step': self.global_step
                    })
                except:
                    pass
            
            self.global_step += 1
        
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        if 'd1' in epoch_metrics:
            epoch_metrics['d1_all'] = epoch_metrics.pop('d1')
        
        if self.scheduler is not None:
            sched_config = self.config.get('optimization', {}).get('scheduler', {})
            if sched_config.get('on_epoch', False):
                self.scheduler.step()
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate on validation set with sequential data and disparity warping.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        val_metrics = {
            'loss': 0.0,
            'epe': 0.0,
            'd1': 0.0
        }
        
        num_batches = 0
        
        progress_bar = tqdm(
            self.val_loader,
            desc=f"Validation",
            disable=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            sequence = batch['sequence']
            sequence_length = len(sequence)
            batch_size = sequence[0]['left'].shape[0]
            
            # Handle cache misses for first frame before init
            first_frame = self._handle_frame_cache_miss(sequence[0])
            sequence[0] = first_frame
            
            if 'disparity_foundationstereo' in first_frame and first_frame['disparity_foundationstereo'] is not None:
                gt_disparity = first_frame['disparity_foundationstereo'].to(self.device)
                if gt_disparity.dim() == 3:
                    gt_disparity = gt_disparity.unsqueeze(1)
            else:
                B, _, H, W = first_frame['left'].shape
                gt_disparity = torch.zeros(B, 1, H, W, device=self.device)
            
            prev_disparity = gt_disparity
            prev_left = first_frame['left'].to(self.device)
            
            total_loss = 0.0
            seq_epe = 0.0
            seq_d1 = 0.0
            
            for t, frame_data in enumerate(sequence):
                # Handle cache misses (frame 0 already handled)
                if t > 0:
                    frame_data = self._handle_frame_cache_miss(frame_data)
                
                left_rgb = frame_data['left'].to(self.device)
                right_rgb = frame_data['right'].to(self.device)
                disp_gt = frame_data['disparity'].to(self.device)
                valid_mask = frame_data['valid_mask'].to(self.device)
                
                if t == 0:
                    # Frame 0: Use GT disparity for left, zeros for right
                    zero_flow = torch.zeros(batch_size, 2, left_rgb.shape[2], left_rgb.shape[3], device=self.device)
                    left_input = prepare_frame_input(left_rgb, gt_disparity, zero_flow,
                                                    use_motion_hints=self.model.use_motion_hints)
                    # Right: zero temporal information
                    zero_disp_right = torch.zeros_like(gt_disparity)
                    right_input = prepare_frame_input(right_rgb, zero_disp_right, zero_flow,
                                                     use_motion_hints=self.model.use_motion_hints)
                else:
                    # Frame t>0: LEFT ONLY temporal information
                    flow_left = self.model.flow_estimator(prev_left, left_rgb)
                    warped_disp_left, _ = self.model.disparity_warper(prev_disparity, flow_left)
                    
                    left_input = prepare_frame_input(left_rgb, warped_disp_left, flow_left,
                                                    use_motion_hints=self.model.use_motion_hints)
                    
                    # Right: zero temporal information
                    zero_disp_right = torch.zeros_like(warped_disp_left)
                    zero_flow_right = torch.zeros(batch_size, 2, left_rgb.shape[2], left_rgb.shape[3], device=self.device)
                    right_input = prepare_frame_input(right_rgb, zero_disp_right, zero_flow_right,
                                                     use_motion_hints=self.model.use_motion_hints)
                
                output = self.model({'left': left_input, 'right': right_input})
                
                data_with_gt = {
                    'left': left_rgb,
                    'right': right_rgb,
                    'disparity': disp_gt,
                    'valid_mask': valid_mask
                }
                
                frame_loss, loss_dict = self.loss_fn(output, data_with_gt)
                total_loss += frame_loss
                
                # Per-frame metrics
                frame_metrics = self.metrics.compute_all_metrics(
                    output['disp_pred'].squeeze(1), disp_gt, valid_mask
                )
                seq_epe += frame_metrics['epe']
                seq_d1 += frame_metrics['d1_all']
                
                prev_disparity = output['disp_pred'].detach()
                prev_left = left_rgb
            
            # Average over sequence
            avg_loss = total_loss / sequence_length
            avg_epe = seq_epe / sequence_length
            avg_d1 = seq_d1 / sequence_length
            
            val_metrics['loss'] += avg_loss.item()
            val_metrics['epe'] += avg_epe
            val_metrics['d1'] += avg_d1
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f"{avg_loss.item():.4f}",
                'epe': f"{avg_epe:.4f}",
                'd1': f"{avg_d1:.4f}"
            })
        
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        if 'd1' in val_metrics:
            d1_value = val_metrics.pop('d1')
            val_metrics['d1_all'] = d1_value
        else:
            d1_value = val_metrics.get('d1_all', 0.0)
        
        try:
            import wandb
            wandb.log({
                'val/loss': val_metrics['loss'],
                'val/epe': val_metrics['epe'],
                'val/d1_all': val_metrics['d1_all'],
                'epoch': epoch
            })
        except:
            pass
        
        logger.info(
            f"Validation - Loss: {val_metrics['loss']:.4f}, "
            f"EPE: {val_metrics['epe']:.4f}, D1: {d1_value:.4f}"
        )
        
        return val_metrics
