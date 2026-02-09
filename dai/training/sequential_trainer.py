"""
Sequential trainer for temporal stereo matching.
Handles sequence-based training with hidden state propagation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import sys
from tqdm import tqdm

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from stereo.modeling.models.lightstereo.temporal_lightstereo import TemporalLightStereo
from .trainer import DAITrainer
from .metrics import StereoMetrics

logger = logging.getLogger(__name__)


class SequentialTrainer(DAITrainer):
    """
    Trainer for temporal stereo matching with sequence-based training.
    
    Extends DAITrainer to handle:
    - Sequential data loading (sequences of frames)
    - Hidden state propagation between frames
    - Backpropagation through time (BPTT)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda'
    ):
        """
        Initialize sequential trainer.
        
        Args:
            config: Training configuration dict
            train_loader: Training data loader (returns sequences)
            val_loader: Validation data loader (returns sequences, optional)
            device: Device to train on ('cuda' or 'cpu')
        """
        # Initialize parent trainer (will build model, optimizer, etc.)
        super().__init__(config, train_loader, val_loader, device)
        
        logger.info("Initialized SequentialTrainer for temporal stereo matching")
    
    def _save_visualizations(self, save_dir: Path, epoch: int, prefix: str = 'train'):
        """
        Override parent's visualization to handle sequential data.
        Visualizes the last frame of each sequence.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if self.val_loader is None:
            return
        
        # Create visualization directory
        viz_dir = save_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache samples on first call
        if self.viz_samples_cache is None:
            self.viz_samples_cache = []
            dataset = self.val_loader.dataset
            for idx in self.viz_sample_indices:
                if idx < len(dataset):
                    self.viz_samples_cache.append(dataset[idx])
        
        # Process each cached sample (sequence)
        for sample_idx, sample in enumerate(self.viz_samples_cache):
            # Extract last frame from sequence for visualization
            sequence = sample['sequence']
            last_frame = sequence[-1]
            
            # Move to device and add batch dimension
            batch = {
                'left': last_frame['left'].unsqueeze(0).to(self.device),
                'right': last_frame['right'].unsqueeze(0).to(self.device)
            }
            
            # Forward pass (without hidden state - single frame inference)
            with autocast('cuda', enabled=self.use_amp):
                model_output = self.model(batch)
            
            # Get predictions and GT
            pred_disp = model_output['disp_pred'][0].detach().cpu().numpy().squeeze()  # [H, W]
            gt_disp = last_frame['disparity'].detach().cpu().numpy().squeeze()  # [H, W]
            valid_mask = last_frame['valid_mask'].detach().cpu().numpy().squeeze()  # [H, W]
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot GT disparity
            im0 = axes[0].imshow(gt_disp, cmap='turbo', vmin=0, vmax=192)
            axes[0].set_title(f'GT Disparity (Seq {sample_idx}, Last Frame)', fontsize=12)
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            # Plot predicted disparity
            im1 = axes[1].imshow(pred_disp, cmap='turbo', vmin=0, vmax=192)
            axes[1].set_title(f'Predicted Disparity (Seq {sample_idx}, Last Frame)', fontsize=12)
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Plot error map (absolute difference where valid)
            error = np.abs(pred_disp - gt_disp) * valid_mask
            im2 = axes[2].imshow(error, cmap='hot', vmin=0, vmax=10)
            axes[2].set_title(f'Error Map (Seq {sample_idx}, Last Frame)', fontsize=12)
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            # Compute EPE for this sample
            epe = np.sum(error) / np.sum(valid_mask) if np.sum(valid_mask) > 0 else 0
            
            fig.suptitle(f'Epoch {epoch} - {prefix.upper()} Sequence {sample_idx} (Last Frame) - EPE: {epe:.2f}px', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save figure
            save_path = viz_dir / f'{prefix}_epoch{epoch:03d}_seq{sample_idx}.png'
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Log to WandB
            try:
                import wandb
                wandb.log({
                    f'{prefix}/viz_sample_{sample_idx}': wandb.Image(str(save_path)),
                    'epoch': epoch,
                })
            except:
                pass  # WandB not available or not initialized
        
        logger.info(f"Saved {len(self.viz_samples_cache)} {prefix} visualizations to {viz_dir}")
    
    def _build_model(self) -> nn.Module:
        """Build TemporalLightStereo model from config"""
        model_config = self.config.get('model', {})
        
        # Create config object for TemporalLightStereo
        from easydict import EasyDict
        model_cfg = EasyDict({
            'MAX_DISP': model_config.get('max_disp', 192),
            'LEFT_ATT': model_config.get('left_att', True),
            'AGGREGATION_BLOCKS': model_config.get('aggregation_blocks', [1, 2, 4]),
            'EXPANSE_RATIO': model_config.get('expanse_ratio', 4),
            'BACKCONE': model_config.get('backbone', 'MobileNetv2'),
            # Temporal-specific configs
            'TEMPORAL_FUSION_TYPE': model_config.get('temporal', {}).get('fusion_type', 'convolutional'),
            'TEMPORAL_HIDDEN_CHANNELS': model_config.get('temporal', {}).get('hidden_channels', 64),
            'USE_TEMPORAL_FUSION': model_config.get('temporal', {}).get('enabled', True),
        })
        
        model = TemporalLightStereo(model_cfg)
        logger.info(f"Built TemporalLightStereo model with config: {model_cfg}")
        logger.info(f"Fusion type: {model_cfg.TEMPORAL_FUSION_TYPE}")
        
        return model
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with sequential data.
        
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
            # batch['sequence'] is a list of frames, each frame has batch dimension
            # batch['sequence_metadata'] is a list of metadata (one per sequence in batch)
            
            sequence = batch['sequence']
            sequence_length = len(sequence)
            batch_size = sequence[0]['left'].shape[0]
            
            # Initialize: no previous disparity for frame 0
            prev_disparity = None
            hidden_state = None
            
            # Accumulate loss over the sequence
            total_loss = 0.0
            
            # Process each frame in the sequence
            for t, frame_data in enumerate(sequence):
                # Move frame data to device (now batched: [batch_size, C, H, W])
                left = frame_data['left'].to(self.device)
                right = frame_data['right'].to(self.device)
                disp_gt = frame_data['disparity'].to(self.device)
                valid_mask = frame_data['valid_mask'].to(self.device)
                
                # Prepare model input
                data = {
                    'left': left,
                    'right': right
                }
                
                # Forward pass with temporal context
                # Frame 0: prev_disparity=None, hidden_state=None (no fusion)
                # Frame t>0: prev_disparity from t-1, hidden_state from t-1
                output = self.model(data, prev_disparity=prev_disparity, hidden_state=hidden_state)
                
                # Prepare data dict with ground truth for loss computation
                data_with_gt = {
                    'left': left,
                    'right': right,
                    'disparity': disp_gt,
                    'valid_mask': valid_mask
                }
                
                # Compute loss for this frame
                frame_loss, loss_dict = self.loss_fn(output, data_with_gt)
                
                # Accumulate loss
                total_loss += frame_loss
                
                # Update for next frame (NO DETACH - keep gradients!)
                prev_disparity = output['disp_pred']
                hidden_state = output.get('hidden_state', None)
            
            # Average loss over sequence
            avg_loss = total_loss / sequence_length
            
            # Scale loss by accumulation steps for correct gradient magnitude
            avg_loss = avg_loss / self.accumulation_steps
            
            # Backpropagation
            avg_loss.backward()
            
            # Only update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.grad_clip_value > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip_value)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update learning rate scheduler (if per-step) - only after actual optimizer step
                if self.scheduler is not None:
                    sched_config = self.config.get('optimization', {}).get('scheduler', {})
                    if not sched_config.get('on_epoch', False):
                        self.scheduler.step()
            
            # Compute metrics on last frame of sequence
            with torch.no_grad():
                last_frame = sequence[-1]
                last_output = output  # Already have output from last frame
                
                disp_pred = last_output['disp_pred'].squeeze(1)
                disp_gt = last_frame['disparity'].to(self.device)
                valid_mask = last_frame['valid_mask'].to(self.device)
                
                metrics = self.metrics.compute_all_metrics(disp_pred, disp_gt, valid_mask)
            
            # Update epoch metrics (multiply loss back for correct logging)
            epoch_metrics['loss'] += avg_loss.item() * self.accumulation_steps
            epoch_metrics['epe'] += metrics['epe']
            epoch_metrics['d1'] += metrics['d1_all']
            num_batches += 1
            
            # Update progress bar (multiply loss back for display)
            progress_bar.set_postfix({
                'loss': f"{avg_loss.item() * self.accumulation_steps:.4f}",
                'epe': f"{metrics['epe']:.4f}",
                'd1': f"{metrics['d1_all']:.4f}"
            })
            
            # Log to WandB
            log_interval = self.config.get('logging', {}).get('log_interval', 250)
            if (batch_idx + 1) % log_interval == 0:
                try:
                    import wandb
                    log_dict = {
                        'train/loss': avg_loss.item() * self.accumulation_steps,
                        'train/epe': metrics['epe'],
                        'train/d1_all': metrics['d1_all'],
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'step': self.global_step
                    }
                    # Log fusion type for tracking
                    if hasattr(self.model, 'fusion_type'):
                        log_dict['model/fusion_type'] = self.model.fusion_type
                    wandb.log(log_dict)
                except:
                    pass  # WandB not available or not initialized
            
            self.global_step += 1
            # break#!
        
        # Average metrics over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Rename 'd1' to 'd1_all' to match parent class expectations
        if 'd1' in epoch_metrics:
            epoch_metrics['d1_all'] = epoch_metrics.pop('d1')
        
        # Update scheduler (if per-epoch)
        if self.scheduler is not None:
            sched_config = self.config.get('optimization', {}).get('scheduler', {})
            if sched_config.get('on_epoch', False):
                self.scheduler.step()
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate on validation set with sequential data.
        
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
            
            # Initialize: no previous disparity for frame 0
            prev_disparity = None
            hidden_state = None
            
            total_loss = 0.0
            
            # Process sequence
            for t, frame_data in enumerate(sequence):
                left = frame_data['left'].to(self.device)
                right = frame_data['right'].to(self.device)
                disp_gt = frame_data['disparity'].to(self.device)
                valid_mask = frame_data['valid_mask'].to(self.device)
                
                data = {
                    'left': left,
                    'right': right
                }
                
                output = self.model(data, prev_disparity=prev_disparity, hidden_state=hidden_state)
                disp_pred = output['disp_pred']
                prev_disparity = disp_pred
                hidden_state = output.get('hidden_state', None)
                
                # Prepare data dict with ground truth for loss computation
                data_with_gt = {
                    'left': left,
                    'right': right,
                    'disparity': disp_gt,
                    'valid_mask': valid_mask
                }
                
                frame_loss, loss_dict = self.loss_fn(output, data_with_gt)
                total_loss += frame_loss
            
            avg_loss = total_loss / sequence_length
            
            # Compute metrics on last frame
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
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        # Rename 'd1' to 'd1_all' to match parent class expectations
        if 'd1' in val_metrics:
            d1_value = val_metrics.pop('d1')
            val_metrics['d1_all'] = d1_value
        else:
            d1_value = val_metrics.get('d1_all', 0.0)
        
        # Log to WandB
        try:
            import wandb
            wandb.log({
                'val/loss': val_metrics['loss'],
                'val/epe': val_metrics['epe'],
                'val/d1_all': val_metrics['d1_all'],
                'epoch': epoch
            })
        except:
            pass  # WandB not available or not initialized
        
        logger.info(
            f"Validation - Loss: {val_metrics['loss']:.4f}, "
            f"EPE: {val_metrics['epe']:.4f}, D1: {d1_value:.4f}"
        )
        
        return val_metrics
