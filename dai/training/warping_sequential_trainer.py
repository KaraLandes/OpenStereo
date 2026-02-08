"""
Warping sequential trainer for temporal stereo matching with disparity warping.
Implements training loop with optical flow computation and disparity warping.
"""

import torch
import torch.nn as nn
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
        
        logger.info("Initialized WarpingSequentialTrainer for temporal stereo with disparity warping")
    
    def _save_visualizations(self, save_dir: Path, epoch: int, prefix: str = 'train'):
        """
        Override parent's visualization to handle sequential data.
        Visualizes the last frame of each sequence.
        """
        import matplotlib.pyplot as plt
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
            
            im0 = axes[0].imshow(gt_disp, cmap='turbo', vmin=0, vmax=192)
            axes[0].set_title(f'GT Disparity (Seq {sample_idx}, Last Frame)', fontsize=12)
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            im1 = axes[1].imshow(pred_disp, cmap='turbo', vmin=0, vmax=192)
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
            
            # Frame 0: Use GT disparity with zero flow
            first_frame = sequence[0]
            if 'disparity_foundationstereo' in first_frame and first_frame['disparity_foundationstereo'] is not None:
                gt_disparity = first_frame['disparity_foundationstereo'].to(self.device)
                if gt_disparity.dim() == 3:
                    gt_disparity = gt_disparity.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            elif 'disparity' in first_frame and first_frame['disparity'] is not None:
                gt_disparity = first_frame['disparity'].to(self.device)
                if gt_disparity.dim() == 3:
                    gt_disparity = gt_disparity.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            else:
                B, _, H, W = first_frame['left'].shape
                gt_disparity = torch.zeros(B, 1, H, W, device=self.device)
            
            # Initialize previous disparity
            prev_disparity = gt_disparity
            
            # Store previous frame for flow computation
            prev_left = first_frame['left'].to(self.device)
            prev_right = first_frame['right'].to(self.device)
            
            total_loss = 0.0
            
            # Process each frame in the sequence
            for t, frame_data in enumerate(sequence):
                left_rgb = frame_data['left'].to(self.device)
                right_rgb = frame_data['right'].to(self.device)
                disp_gt = frame_data['disparity'].to(self.device)
                valid_mask = frame_data['valid_mask'].to(self.device)
                
                if t == 0:
                    # Frame 0: Use GT disparity with zero flow
                    zero_flow = torch.zeros(batch_size, 2, left_rgb.shape[2], left_rgb.shape[3], device=self.device)
                    
                    left_input = prepare_frame_input(
                        left_rgb, gt_disparity, zero_flow,
                        use_motion_hints=self.model.use_motion_hints
                    )
                    right_input = prepare_frame_input(
                        right_rgb, gt_disparity, zero_flow,
                        use_motion_hints=self.model.use_motion_hints
                    )
                else:
                    # Frame t>0: Compute flow and warp disparity
                    with torch.no_grad():
                        flow_left = self.model.flow_estimator(prev_left, left_rgb)
                        flow_right = self.model.flow_estimator(prev_right, right_rgb)
                    
                    # Warp previous disparity
                    warped_disp_left, mask_left = self.model.disparity_warper(prev_disparity, flow_left)
                    warped_disp_right, mask_right = self.model.disparity_warper(prev_disparity, flow_right)
                    
                    # Prepare 6-channel input
                    left_input = prepare_frame_input(
                        left_rgb, warped_disp_left, flow_left,
                        use_motion_hints=self.model.use_motion_hints
                    )
                    right_input = prepare_frame_input(
                        right_rgb, warped_disp_right, flow_right,
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
                
                # Update for next iteration
                prev_disparity = output['disp_pred'].detach()
                prev_left = left_rgb
                prev_right = right_rgb
            
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
            
            # Compute metrics on last frame
            with torch.no_grad():
                last_frame = sequence[-1]
                last_output = output
                
                disp_pred = last_output['disp_pred'].squeeze(1)
                disp_gt = last_frame['disparity'].to(self.device)
                valid_mask = last_frame['valid_mask'].to(self.device)
                
                metrics = self.metrics.compute_all_metrics(disp_pred, disp_gt, valid_mask)
            
            epoch_metrics['loss'] += avg_loss.item() * self.accumulation_steps
            epoch_metrics['epe'] += metrics['epe']
            epoch_metrics['d1'] += metrics['d1_all']
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f"{avg_loss.item() * self.accumulation_steps:.4f}",
                'epe': f"{metrics['epe']:.4f}",
                'd1': f"{metrics['d1_all']:.4f}"
            })
            
            log_interval = self.config.get('logging', {}).get('log_interval', 250)
            if (batch_idx + 1) % log_interval == 0:
                try:
                    import wandb
                    wandb.log({
                        'train/loss': avg_loss.item() * self.accumulation_steps,
                        'train/epe': metrics['epe'],
                        'train/d1_all': metrics['d1_all'],
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
            
            first_frame = sequence[0]
            if 'disparity_foundationstereo' in first_frame and first_frame['disparity_foundationstereo'] is not None:
                gt_disparity = first_frame['disparity_foundationstereo'].to(self.device)
                if gt_disparity.dim() == 3:
                    gt_disparity = gt_disparity.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            else:
                B, _, H, W = first_frame['left'].shape
                gt_disparity = torch.zeros(B, 1, H, W, device=self.device)
            
            prev_disparity = gt_disparity
            prev_left = first_frame['left'].to(self.device)
            prev_right = first_frame['right'].to(self.device)
            
            total_loss = 0.0
            
            for t, frame_data in enumerate(sequence):
                left_rgb = frame_data['left'].to(self.device)
                right_rgb = frame_data['right'].to(self.device)
                disp_gt = frame_data['disparity'].to(self.device)
                valid_mask = frame_data['valid_mask'].to(self.device)
                
                if t == 0:
                    zero_flow = torch.zeros(batch_size, 2, left_rgb.shape[2], left_rgb.shape[3], device=self.device)
                    left_input = prepare_frame_input(left_rgb, gt_disparity, zero_flow,
                                                    use_motion_hints=self.model.use_motion_hints)
                    right_input = prepare_frame_input(right_rgb, gt_disparity, zero_flow,
                                                     use_motion_hints=self.model.use_motion_hints)
                else:
                    flow_left = self.model.flow_estimator(prev_left, left_rgb)
                    flow_right = self.model.flow_estimator(prev_right, right_rgb)
                    
                    warped_disp_left, _ = self.model.disparity_warper(prev_disparity, flow_left)
                    warped_disp_right, _ = self.model.disparity_warper(prev_disparity, flow_right)
                    
                    left_input = prepare_frame_input(left_rgb, warped_disp_left, flow_left,
                                                    use_motion_hints=self.model.use_motion_hints)
                    right_input = prepare_frame_input(right_rgb, warped_disp_right, flow_right,
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
                
                prev_disparity = output['disp_pred']
                prev_left = left_rgb
                prev_right = right_rgb
            
            avg_loss = total_loss / sequence_length
            
            last_frame = sequence[-1]
            disp_pred = output['disp_pred'].squeeze(1)
            disp_gt = last_frame['disparity'].to(self.device)
            valid_mask = last_frame['valid_mask'].to(self.device)
            
            metrics = self.metrics.compute_all_metrics(disp_pred, disp_gt, valid_mask)
            
            val_metrics['loss'] += avg_loss.item()
            val_metrics['epe'] += metrics['epe']
            val_metrics['d1'] += metrics['d1_all']
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f"{avg_loss.item():.4f}",
                'epe': f"{metrics['epe']:.4f}",
                'd1': f"{metrics['d1_all']:.4f}"
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
