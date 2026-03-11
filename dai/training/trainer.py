"""
Main trainer class for DAI pipeline
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
import matplotlib.pyplot as plt
import numpy as np

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from stereo.modeling.models.lightstereo.lightstereo import LightStereo
from .metrics import StereoMetrics
from .losses import build_loss_function

logger = logging.getLogger(__name__)

# WandB import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("WandB not available. Install with: pip install wandb")


class DAITrainer:
    """
    Trainer for LightStereo-S model.
    
    Features:
    - Configurable optimizer, scheduler, loss function
    - Two training modes: supervised (with provided/pseudo targets) and SSL (placeholder)
    - WandB logging with grouped runs for sweeps
    - Metrics tracking: EPE, D1-all, D1-bg, D1-fg
    - Mixed precision training
    - Gradient clipping
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda'
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dict
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            device: Device to train on ('cuda' or 'cpu')
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Get executable_epochs for early stopping at epoch level (for testing)
        self.executable_epochs = config.get('optimization', {}).get('executable_epochs', None)
        
        # Determine training mode from dataset config
        self.training_mode = self._determine_training_mode()
        logger.info(f"Training mode: {self.training_mode}")
        
        # Build model
        self.model = self._build_model()
        self.model = self.model.to(self.device)
        
        # Freeze backbone if configured
        self._freeze_backbone_if_configured()
        
        # Build loss function
        self.loss_fn = self._build_loss_function()
        
        # Build optimizer
        self.optimizer = self._build_optimizer()
        
        # Gradient accumulation (must be set before scheduler)
        self.accumulation_steps = config.get('optimization', {}).get('accumulation_steps', 1)
        if self.accumulation_steps > 1:
            logger.info(f"Gradient accumulation enabled: {self.accumulation_steps} steps")
            logger.info(f"Effective batch size: {config.get('dataloader', {}).get('batch_size', 1)} × {self.accumulation_steps} = {config.get('dataloader', {}).get('batch_size', 1) * self.accumulation_steps}")
        
        # Build scheduler (needs accumulation_steps)
        self.scheduler = self._build_scheduler()
        
        # Metrics calculator
        self.metrics = StereoMetrics()
        
        # Mixed precision training
        self.use_amp = config.get('optimization', {}).get('amp', True)
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        
        # Gradient clipping
        self.grad_clip_value = config.get('optimization', {}).get('grad_clip_value', 0.1)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        # Visualization: select 3 fixed samples for tracking
        self.viz_sample_indices = [0, len(val_loader.dataset) // 2, len(val_loader.dataset) - 1] if val_loader else []
        self.viz_samples_cache = None
        
        # WandB setup (always enabled)
        self._setup_wandb()
        
        logger.info(f"Initialized DAITrainer on device: {self.device}")
        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        logger.info(f"Model parameters: {total_params:.2f}M total, {trainable_params:.2f}M trainable")
    
    def _determine_training_mode(self) -> str:
        """
        Determine training mode from dataset configuration.
        If target_source='ssl', use SSL mode. Otherwise, supervised.
        """
        datasets_config = self.config.get('datasets', [])
        for dataset_cfg in datasets_config:
            if dataset_cfg.get('target_source') == 'ssl':
                return 'ssl'
        return 'supervised'
    
    def _build_model(self) -> nn.Module:
        """Build LightStereo-S model from config"""
        model_config = self.config.get('model', {})
        
        # Create config object for LightStereo
        from easydict import EasyDict
        model_cfg = EasyDict({
            'MAX_DISP': model_config.get('max_disp', 192),
            'LEFT_ATT': model_config.get('left_att', True),
            'AGGREGATION_BLOCKS': model_config.get('aggregation_blocks', [1, 2, 4]),
            'EXPANSE_RATIO': model_config.get('expanse_ratio', 4),
            'BACKCONE': model_config.get('backbone', 'MobileNetv2'),
        })
        
        model = LightStereo(model_cfg)
        logger.info(f"Built LightStereo model with config: {model_cfg}")
        
        return model
    
    def _freeze_backbone_if_configured(self):
        """Freeze backbone parameters if freeze_backbone is enabled in config"""
        freeze_backbone = self.config.get('model', {}).get('freeze_backbone', True)
        
        if freeze_backbone:
            # Freeze all parameters in the backbone
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            
            # Count frozen vs trainable parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            
            logger.info(f"🔒 Backbone FROZEN: {frozen_params/1e6:.2f}M parameters frozen")
            logger.info(f"🔓 Trainable parameters: {trainable_params/1e6:.2f}M (excluding backbone)")
        else:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"🔓 Backbone NOT frozen: all {trainable_params/1e6:.2f}M parameters trainable")
    
    def _build_loss_function(self) -> nn.Module:
        """Build loss function based on training mode"""
        loss_config = self.config.get('loss', {})
        loss_config['max_disp'] = self.config.get('model', {}).get('max_disp', 192)
        
        loss_fn = build_loss_function(loss_config, self.training_mode)
        return loss_fn
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer from config (only for trainable parameters)"""
        opt_config = self.config.get('optimization', {}).get('optimizer', {})
        
        optimizer_name = opt_config.get('name', 'AdamW')
        lr = opt_config.get('lr', 0.0024)
        weight_decay = opt_config.get('weight_decay', 1e-5)
        
        # Only optimize parameters that require gradients (excludes frozen backbone)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay,
                eps=opt_config.get('eps', 1e-8)
            )
        elif optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(
                trainable_params,
                lr=lr,
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        logger.info(f"Built {optimizer_name} optimizer with lr={lr} for {len(trainable_params)} parameter groups")
        return optimizer
    
    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler from config"""
        sched_config = self.config.get('optimization', {}).get('scheduler', {})
        
        if not sched_config:
            return None
        
        scheduler_name = sched_config.get('name', 'OneCycleLR')
        
        if scheduler_name == 'OneCycleLR':
            # Use optimizer's lr as max_lr (shared configuration)
            optimizer_lr = self.config.get('optimization', {}).get('optimizer', {}).get('lr', 0.0024)
            epochs = self.config.get('optimization', {}).get('num_epochs', 90)
            
            # Account for gradient accumulation: effective batch size = batch_size × accumulation_steps
            # Scheduler should see steps based on effective batches, not physical batches
            steps_per_epoch = len(self.train_loader) // self.accumulation_steps
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=optimizer_lr,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=sched_config.get('pct_start', 0.01),
                anneal_strategy='cos',
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=25.0,
                final_div_factor=10000.0
            )
            logger.info(f"Built OneCycleLR scheduler with max_lr={optimizer_lr}, steps_per_epoch={steps_per_epoch} (accounting for accumulation_steps={self.accumulation_steps})")
        
        elif scheduler_name == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
            logger.info(f"Built StepLR scheduler")
        
        elif scheduler_name == 'CosineAnnealingLR':
            epochs = self.config.get('optimization', {}).get('num_epochs', 90)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=sched_config.get('eta_min', 0)
            )
            logger.info(f"Built CosineAnnealingLR scheduler")
        
        else:
            logger.warning(f"Unknown scheduler: {scheduler_name}, using None")
            return None
        
        return scheduler
    
    def _setup_wandb(self):
        """
        Setup WandB logging with grouped runs for sweeps.
        
        WandB Sweeps workflow:
        1. Define sweep config (YAML or dict) with hyperparameters to search
        2. Initialize sweep: sweep_id = wandb.sweep(sweep_config)
        3. Run agents: wandb.agent(sweep_id, function=train_function)
        
        The sweep agent automatically:
        - Sets wandb.config with hyperparameters for this run
        - Sets group name to group all runs in the sweep
        - Manages parallel runs across multiple machines/GPUs
        """
        if not WANDB_AVAILABLE:
            raise ImportError(
                "WandB is required but not installed. "
                "Install with: pip install wandb"
            )
        
        wandb_config = self.config.get('logging', {}).get('wandb', {})
        
        # Generate dynamic run name from hyperparameters if not explicitly set
        run_name = wandb_config.get('run_name', None)
        if run_name is None:
            # Auto-generate name from key hyperparameters
            batch_size = self.config.get('dataloader', {}).get('batch_size', 'unk')
            lr = self.config.get('optimization', {}).get('optimizer', {}).get('lr', 'unk')
            pct_start = self.config.get('optimization', {}).get('scheduler', {}).get('pct_start', 'unk')
            run_name = f"bs{batch_size}_lr{lr}_pct{pct_start}"
        
        # Initialize WandB run
        # Group will be set by sweep agent or manually
        wandb.init(
            project=wandb_config.get('project', 'lightstereo-training'),
            entity=wandb_config.get('entity', None),
            name=run_name,
            group=wandb_config.get('group', None),  # For sweep grouping
            config=self.config,
            tags=wandb_config.get('tags', []),
        )
        
        # Watch model (disabled for performance - gradients logging is expensive)
        # wandb.watch(self.model, log='all', log_freq=100)
        
        logger.info(f"Initialized WandB logging: {wandb.run.name}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dict with average metrics for the epoch
        """
        self.model.train()
        self.current_epoch = epoch
        
        epoch_metrics = {
            'loss': 0.0,
            'epe': 0.0,
            'd1_all': 0.0,
            'd1_bg': 0.0,
            'd1_fg': 0.0,
        }
        metrics_count = 0  # Track how many times we computed metrics
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        # Timing instrumentation (only for first 100 iterations)
        # import time
        # if not hasattr(self, '_timing_done'):
        #     self._timing_done = False
        
        for batch_idx, batch in enumerate(pbar):
            # enable_timing = batch_idx < 100 if hasattr(self, '_timing_done') and not self._timing_done else True
            
            # batch_ready_time = time.time()
            # iter_start = time.time() if enable_timing and batch_idx < 100 else None
            
            # # Measure data loading wait time (time between iterations)
            # if batch_idx > 0 and iter_start:
            #     data_wait_time = batch_ready_time - prev_iter_end
            # else:
            #     data_wait_time = 0
            
            # Move batch to device
            # t0 = time.time() if iter_start else None
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            # if t0:
            #     data_transfer_time = time.time() - t0
            
            # Forward pass with mixed precision
            # t1 = time.time() if iter_start else None
            with autocast('cuda', enabled=self.use_amp):
                model_output = self.model(batch)
                loss, loss_dict = self.loss_fn(model_output, batch)
                
                # Scale loss by accumulation steps for correct gradient magnitude
                loss = loss / self.accumulation_steps
            # if t1:
            #     forward_time = time.time() - t1
            
            # Backward pass
            # t2 = time.time() if iter_start else None
            self.scaler.scale(loss).backward()
            
            # Only update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.grad_clip_value > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip_value)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Scheduler step (if per-iteration) - only after actual optimizer step
                if self.scheduler is not None:
                    sched_config = self.config.get('optimization', {}).get('scheduler', {})
                    if not sched_config.get('on_epoch', False):
                        self.scheduler.step()
                
                # LR tracking: Log every 50 steps and at key warmup milestones
                current_lr = self.optimizer.param_groups[0]['lr']
                if (self.global_step % 50 == 0 or 
                    self.global_step < 10 or 
                    self.global_step in [1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180]):
                    logger.info(f"[LR] Step {self.global_step}, LR: {current_lr:.8f}")
            
            # if t2:
            #     backward_time = time.time() - t2 - optim_time if 'optim_time' in locals() else 0
            #     optim_time = time.time() - t2
            
            # Accumulate loss (detach to prevent memory leak)
            # Note: loss is already scaled by accumulation_steps, so multiply back for logging
            loss_value = loss_dict['loss_total']
            if torch.is_tensor(loss_value):
                epoch_metrics['loss'] += loss_value.detach().item() * self.accumulation_steps
            else:
                epoch_metrics['loss'] += loss_value * self.accumulation_steps
            
            # Compute metrics only at log intervals (expensive operation)
            log_interval = self.config.get('logging', {}).get('log_interval', 10)
            if batch_idx % log_interval == 0:
                # t4 = time.time() if iter_start else None
                with torch.no_grad():
                    metrics = self.metrics.compute_all_metrics(
                        model_output['disp_pred'],
                        batch['disparity'],
                        batch['valid_mask'],
                        max_disp=self.config.get('model', {}).get('max_disp', 192)
                    )
                # if t4:
                #     metrics_time = time.time() - t4
                
                # Accumulate metrics (already scalars from metrics computation)
                for key in ['epe', 'd1_all', 'd1_bg', 'd1_fg']:
                    epoch_metrics[key] += metrics[key]
                metrics_count += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['loss_total']:.4f}",
                    'epe': f"{metrics['epe']:.2f}",
                    'd1': f"{metrics['d1_all']:.2f}%"
                })
                
                # Log to WandB (ensure loss is scalar)
                loss_log = loss_dict['loss_total'].item() if torch.is_tensor(loss_dict['loss_total']) else loss_dict['loss_total']
                wandb.log({
                    'train/loss': loss_log,
                    'train/epe': metrics['epe'],
                    'train/d1_all': metrics['d1_all'],
                    'train/d1_bg': metrics['d1_bg'],
                    'train/d1_fg': metrics['d1_fg'],
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                    'step': self.global_step,
                })
            else:
                # Just update progress bar with loss
                pbar.set_postfix({'loss': f"{loss_dict['loss_total']:.4f}"})
            
            # # Log timing breakdown for first 100 iterations
            # if iter_start and batch_idx < 100:
            #     total_time = time.time() - iter_start
            #     logger.info(f"\n⏱️  Iteration {batch_idx} timing breakdown:")
            #     if batch_idx > 0:
            #         logger.info(f"  Data wait:     {data_wait_time*1000:.1f}ms ⚠️ (DataLoader delay)")
            #     logger.info(f"  Data transfer: {data_transfer_time*1000:.1f}ms")
            #     logger.info(f"  Forward pass:  {forward_time*1000:.1f}ms")
            #     logger.info(f"  Backward pass: {backward_time*1000:.1f}ms")
            #     logger.info(f"  Optimizer:     {optim_time*1000:.1f}ms")
            #     if batch_idx % log_interval == 0:
            #         logger.info(f"  Metrics:       {metrics_time*1000:.1f}ms")
            #     logger.info(f"  TOTAL:         {total_time*1000:.1f}ms ({total_time:.3f}s)")
            #     if batch_idx == 99:
            #         self._timing_done = True
            
            # # Store end time for next iteration's wait calculation
            # if iter_start:
            #     prev_iter_end = time.time()
            
            # Periodic CUDA cache clearing to prevent memory fragmentation
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.global_step += 1
            # break
        
        # Average metrics over epoch
        num_batches = len(self.train_loader)
        epoch_metrics['loss'] /= num_batches
        # Average other metrics by the number of times we computed them
        if metrics_count > 0:
            for key in ['epe', 'd1_all', 'd1_bg', 'd1_fg']:
                epoch_metrics[key] /= metrics_count
        
        # Scheduler step (if per-epoch)
        if self.scheduler is not None:
            sched_config = self.config.get('optimization', {}).get('scheduler', {})
            if sched_config.get('on_epoch', False):
                self.scheduler.step()
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dict with average validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        val_metrics = {
            'loss': 0.0,
            'epe': 0.0,
            'd1_all': 0.0,
            'd1_bg': 0.0,
            'd1_fg': 0.0,
        }
        
        pbar = tqdm(self.val_loader, desc=f"Validation")
        
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            with autocast('cuda', enabled=self.use_amp):
                model_output = self.model(batch)
                loss, loss_dict = self.loss_fn(model_output, batch)
            
            # Compute metrics
            metrics = self.metrics.compute_all_metrics(
                model_output['disp_pred'],
                batch['disparity'],
                batch['valid_mask'],
                max_disp=self.config.get('model', {}).get('max_disp', 192)
            )
            
            # Accumulate (handle both tensor and float cases)
            loss_value = loss_dict['loss_total']
            if torch.is_tensor(loss_value):
                val_metrics['loss'] += loss_value.detach().item()
            else:
                val_metrics['loss'] += loss_value
            for key in ['epe', 'd1_all', 'd1_bg', 'd1_fg']:
                val_metrics[key] += metrics[key]
            
            pbar.set_postfix({
                'loss': f"{loss_dict['loss_total']:.4f}",
                'epe': f"{metrics['epe']:.2f}",
                'd1': f"{metrics['d1_all']:.2f}%"
            })
        
        # Average metrics
        num_batches = len(self.val_loader)
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        # Log to WandB
        wandb.log({
                'val/loss': val_metrics['loss'],
                'val/epe': val_metrics['epe'],
                'val/d1_all': val_metrics['d1_all'],
                'val/d1_bg': val_metrics['d1_bg'],
                'val/d1_fg': val_metrics['d1_fg'],
                'epoch': epoch,
            })
        
        return val_metrics
    
    @torch.no_grad()
    def _save_visualizations(self, save_dir: Path, epoch: int, prefix: str = 'val'):
        """
        Save visualizations of 3 fixed samples (GT vs Predicted disparity).
        
        Args:
            save_dir: Directory to save visualizations
            epoch: Current epoch number
            prefix: Prefix for saved files ('val' or 'test')
        """
        if self.val_loader is None or not self.viz_sample_indices:
            return
        
        self.model.eval()
        
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
        
        # Process each cached sample
        for sample_idx, sample in enumerate(self.viz_samples_cache):
            # Move to device and add batch dimension
            batch = {k: v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v 
                    for k, v in sample.items()}
            
            # Forward pass
            with autocast('cuda', enabled=self.use_amp):
                model_output = self.model(batch)
            
            # Get predictions and GT
            pred_disp = model_output['disp_pred'][0].detach().cpu().numpy().squeeze()  # [H, W]
            gt_disp = batch['disparity'][0].detach().cpu().numpy().squeeze()  # [H, W]
            valid_mask = batch['valid_mask'][0].detach().cpu().numpy().squeeze()  # [H, W]
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot GT disparity
            im0 = axes[0].imshow(gt_disp, cmap='turbo', vmin=0, vmax=192)
            axes[0].set_title(f'GT Disparity (Sample {sample_idx})', fontsize=12)
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            # Plot predicted disparity
            im1 = axes[1].imshow(pred_disp, cmap='turbo', vmin=0, vmax=192)
            axes[1].set_title(f'Predicted Disparity (Sample {sample_idx})', fontsize=12)
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Plot error map (absolute difference where valid)
            error = np.abs(pred_disp - gt_disp) * valid_mask
            im2 = axes[2].imshow(error, cmap='hot', vmin=0, vmax=10)
            axes[2].set_title(f'Error Map (Sample {sample_idx})', fontsize=12)
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            # Compute EPE for this sample
            epe = np.sum(error) / np.sum(valid_mask) if np.sum(valid_mask) > 0 else 0
            
            fig.suptitle(f'Epoch {epoch} - {prefix.upper()} Sample {sample_idx} - EPE: {epe:.2f}px', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save figure
            save_path = viz_dir / f'{prefix}_sample_{sample_idx}_epoch_{epoch:03d}.png'
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Log to WandB
            wandb.log({
                f'{prefix}/viz_sample_{sample_idx}': wandb.Image(str(save_path)),
                'epoch': epoch,
            })
        
        logger.info(f"Saved {len(self.viz_samples_cache)} visualization(s) to {viz_dir}")
    
    def train(self, num_epochs: int, save_dir: Optional[Path] = None):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints (optional)
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        if save_dir:
            save_dir = Path(save_dir)
            # Create unique subdirectory for each sweep run using wandb run name
            if WANDB_AVAILABLE and wandb.run is not None:
                save_dir = save_dir / wandb.run.name
                logger.info(f"Saving checkpoints to sweep-specific directory: {save_dir}")
            save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_epe = float('inf')
        
        # Determine actual number of epochs to execute (for early stopping/testing)
        epochs_to_run = self.executable_epochs if self.executable_epochs is not None else num_epochs
        if self.executable_epochs is not None:
            logger.info(f"Early stopping enabled: running {epochs_to_run} epochs (configured: {num_epochs})")
        
        # Run initial validation before training (epoch 0)
        logger.info(f"\n{'='*80}")
        logger.info(f"Initial Validation (Epoch 0) - Before Training")
        logger.info(f"{'='*80}")
        val_metrics = self.validate(epoch=0)
        if val_metrics:
            logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                       f"EPE: {val_metrics['epe']:.2f}, "
                       f"D1-all: {val_metrics['d1_all']:.2f}%")
            if save_dir:
                self._save_visualizations(save_dir, epoch=0, prefix='val')
        
        for epoch in range(1, epochs_to_run + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'='*80}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"EPE: {train_metrics['epe']:.2f}, "
                       f"D1-all: {train_metrics['d1_all']:.2f}%")
            
            # Validate
            val_metrics = self.validate(epoch)
            if val_metrics:
                logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                           f"EPE: {val_metrics['epe']:.2f}, "
                           f"D1-all: {val_metrics['d1_all']:.2f}%")
                
                # Save visualizations
                if save_dir:
                    self._save_visualizations(save_dir, epoch, prefix='val')
                
                # Save best model
                if save_dir and val_metrics['epe'] < best_val_epe:
                    best_val_epe = val_metrics['epe']
                    self.save_checkpoint(save_dir / 'best_model.pth', epoch, val_metrics)
                    logger.info(f"✓ Saved best model (EPE: {best_val_epe:.2f})")
            
            # Save periodic checkpoint
            if save_dir and epoch % 10 == 0:
                self.save_checkpoint(save_dir / f'checkpoint_epoch_{epoch}.pth', epoch, train_metrics)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Training complete! Best val EPE: {best_val_epe:.2f}")
        logger.info(f"{'='*80}")
        
        wandb.finish()
    
    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
