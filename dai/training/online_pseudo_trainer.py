"""
Online Pseudo Ground Truth Trainer

This trainer generates pseudo ground truth disparities in real-time during training
using a frozen FoundationStereo model as a teacher. The student model (LightStereo)
is trained to match the teacher's predictions.

Usage:
    Set target_source: 'pseudo-foundationstereo-online' in dataset config
"""

import torch
import torch.nn as nn
from torch.amp import autocast
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import threading
import queue
from tqdm import tqdm

from .trainer import DAITrainer

logger = logging.getLogger(__name__)


class PseudoGTPrefetcher:
    """
    Prefetches pseudo GT disparities by running FoundationStereo inference
    in a background thread on a separate CUDA stream.
    
    While the main thread trains LightStereo on batch N, this prefetcher
    runs FoundationStereo on batch N+1 concurrently.
    """
    
    def __init__(self, dataloader, generate_fn, device, buffer_size=1):
        """
        Args:
            dataloader: Training dataloader to iterate over
            generate_fn: Callable(left, right) -> disparity tensor
            device: CUDA device for data transfer
            buffer_size: Number of batches to buffer in queue (default: 1)
        """
        self.dataloader = dataloader
        self.generate_fn = generate_fn
        self.device = device
        self.buffer_size = buffer_size
        self.stream = torch.cuda.Stream(device=device)
        self._result_queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._thread = None
        self._iterator = None
        self._pbar = None  # Progress bar for prefetcher
        self._current_epoch = None  # Track which epoch we're prefetching
    
    def _prefetch_worker(self):
        """Background thread: fetch next batch, run FS inference on secondary stream."""
        try:
            for batch in self._iterator:
                if self._stop_event.is_set():
                    break
                
                # Move batch to device on the secondary stream
                with torch.cuda.stream(self.stream):
                    batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                             for k, v in batch.items()}
                    
                    # Run FoundationStereo inference
                    try:
                        pseudo_gt = self.generate_fn(batch['left'], batch['right'])
                        batch['disparity'] = pseudo_gt
                        batch['valid_mask'] = torch.ones_like(pseudo_gt, dtype=torch.bool)
                        error = None
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        error = 'oom'
                    except Exception as e:
                        error = str(e)
                
                # Record event so main stream can synchronize
                event = self.stream.record_event()
                self._result_queue.put((batch, event, error))
                
                # Update prefetcher progress bar
                if self._pbar is not None:
                    self._pbar.update(1)
            
            # Close progress bar when done
            if self._pbar is not None:
                self._pbar.close()
                self._pbar = None
            
            # Sentinel: end of epoch
            self._result_queue.put(None)
        except Exception as e:
            if self._pbar is not None:
                self._pbar.close()
                self._pbar = None
            logger.error(f"Prefetcher thread error: {e}")
            self._result_queue.put(None)
    
    def __iter__(self):
        # Only start thread once - if already running, just return self
        if self._thread is None or not self._thread.is_alive():
            self._iterator = iter(self.dataloader)
            self._stop_event.clear()
            
            # Create progress bar for prefetcher
            if self._current_epoch is not None:
                self._pbar = tqdm(
                    total=len(self.dataloader),
                    desc=f"Prefetching Epoch {self._current_epoch}",
                    position=0,  # Top position
                    leave=False,
                    colour='blue'
                )
            
            self._thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self._thread.start()
        return self
    
    def __next__(self):
        result = self._result_queue.get()
        if result is None:
            raise StopIteration
        
        batch, event, error = result
        # Wait for the secondary stream's work to complete before using the batch
        event.wait()
        return batch, error
    
    def __len__(self):
        return len(self.dataloader)
    
    def stop(self):
        """Signal the prefetcher to stop early."""
        self._stop_event.set()
        # Drain the queue to unblock the worker thread
        try:
            while not self._result_queue.empty():
                self._result_queue.get_nowait()
        except queue.Empty:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5)
        # Close progress bar if exists
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
    
    def reset(self):
        """Reset prefetcher for next epoch without reloading model."""
        # Stop current thread if running
        if self._thread is not None and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=5)
        
        # Drain queue
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break
        
        # Reset state for next epoch
        self._stop_event.clear()
        self._thread = None
        self._iterator = None
        
        # Close progress bar if exists
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
    
    def set_epoch(self, epoch: int):
        """Set which epoch this prefetcher is preparing for."""
        self._current_epoch = epoch


# WandB import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class OnlinePseudoGTTrainer(DAITrainer):
    """
    Trainer that generates pseudo ground truth using FoundationStereo in real-time.
    
    Workflow:
    1. Load frozen FoundationStereo model (teacher)
    2. For each batch:
       - Denormalize images (normalized -> [0-255])
       - Generate pseudo GT disparity using FoundationStereo
       - Train LightStereo (student) to match pseudo GT
    3. Only LightStereo parameters are updated
    
    Features:
    - Inherits all DAITrainer functionality (WandB, metrics, checkpointing)
    - Works with all datasets (SceneFlow, KITTI, etc.)
    - Applies to train, val, and test splits
    - Memory efficient (bfloat16 for FoundationStereo)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        train_loader,
        val_loader=None,
        device: str = 'cuda'
    ):
        """
        Initialize trainer with FoundationStereo model.
        
        Args:
            config: Training configuration dict
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            device: Device to train on ('cuda' or 'cpu')
        """
        # Initialize parent trainer (builds LightStereo, optimizer, etc.)
        super().__init__(config, train_loader, val_loader, device)
        
        # Load FoundationStereo model
        logger.info("Loading FoundationStereo model for online pseudo GT generation...")
        self.foundation_model, self.foundation_device = self._load_foundation_stereo_model()
        
        # FoundationStereo inference settings
        fs_config = config.get('foundationstereo', {})
        self.foundation_batch_size = fs_config.get('inference_batch_size', 4)
        self.foundation_use_amp = fs_config.get('inference_amp', True)
        self.prefetch_buffer_size = fs_config.get('prefetch_buffer_size', 1)
        
        # Override GRU iterations for faster pseudo GT (default 32 -> 12)
        self.foundation_iters = fs_config.get('inference_iters', 12)
        if hasattr(self.foundation_model, 'args'):
            original_iters = self.foundation_model.args.valid_iters
            self.foundation_model.args.valid_iters = self.foundation_iters
            logger.info(f"FoundationStereo GRU iters: {original_iters} -> {self.foundation_iters}")
        
        logger.info(f"FoundationStereo inference: batch_size={self.foundation_batch_size}, "
                    f"iters={self.foundation_iters}, amp={self.foundation_use_amp}, "
                    f"prefetch_buffer={self.prefetch_buffer_size}")
        
        # Normalization constants for denormalization
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        logger.info("✓ OnlinePseudoGTTrainer initialized successfully")
    
    def _load_foundation_stereo_model(self):
        """
        Load FoundationStereo model from checkpoint.
        
        Returns:
            (model, device) tuple
        """
        # Get model size from config (default: small)
        model_size = self.config.get('foundationstereo', {}).get('model_size', 'small')
        
        # Add repo root to path for imports
        repo_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(repo_root))
        
        try:
            from stereo.modeling.models.foundationstereo.core.foundation_stereo import FoundationStereo
            from easydict import EasyDict
            import yaml
        except ImportError as e:
            error_msg = str(e)
            if 'trimesh' in error_msg:
                raise ImportError(
                    f"Failed to import FoundationStereo: {e}\n\n"
                    "FoundationStereo requires additional dependencies. Please install:\n"
                    "  pip install trimesh\n\n"
                    "Or install all FoundationStereo dependencies if available."
                )
            elif 'easydict' in error_msg:
                raise ImportError(
                    f"Failed to import FoundationStereo: {e}\n\n"
                    "Missing required dependency. Please install:\n"
                    "  pip install easydict\n"
                )
            else:
                raise ImportError(
                    f"Failed to import FoundationStereo: {e}\n\n"
                    "Make sure the stereo.modeling package is available and all dependencies are installed."
                )
        
        # Load config
        checkpoint_dir = repo_root / "src_data" / "checkpoints" / "foundationstereo" / model_size
        cfg_path = checkpoint_dir / "cfg.yaml"
        
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"FoundationStereo config not found at {cfg_path}. "
                f"For 'pseudo-foundationstereo-online' mode, the FoundationStereo model must be available. "
                f"Please ensure checkpoint is in src_data/checkpoints/foundationstereo/{model_size}/"
            )
        
        logger.info(f"Loading FoundationStereo ({model_size}) config from {cfg_path}")
        with open(cfg_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        args = EasyDict(cfg_dict)
        
        # Create model
        logger.info("Creating FoundationStereo model...")
        model = FoundationStereo(args)
        
        # Load checkpoint
        checkpoint_path = checkpoint_dir / "model_best_bp2.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"FoundationStereo checkpoint not found at {checkpoint_path}. "
                f"For 'pseudo-foundationstereo-online' mode, the model checkpoint must be available."
            )
        
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        elif 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Moving FoundationStereo to device: {device}")
        model = model.to(device)
        
        # Note: We do NOT manually convert to bfloat16 here because grid_sample
        # does not support bfloat16 inputs. FoundationStereo's own mixed_precision
        # autocast (configured via cfg.yaml) handles dtype casting safely.
        
        # Set to eval mode and freeze parameters
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        logger.info("✓ FoundationStereo loaded successfully (frozen, eval mode)")
        return model, device
    
    def _denormalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized images back to [0-255] range for FoundationStereo.
        
        Images come from DataLoader already normalized with:
            (img / 255.0 - mean) / std
        
        We reverse this to get back to [0-255]:
            img = (normalized * std + mean) * 255.0
        
        Args:
            images: [B, 3, H, W] normalized tensor
            
        Returns:
            [B, 3, H, W] tensor in range [0-255]
        """
        mean = self.norm_mean.to(images.device)
        std = self.norm_std.to(images.device)
        
        # Reverse normalization
        denormalized = (images * std + mean) * 255.0
        
        return denormalized
    
    def _generate_pseudo_gt(self, left_img: torch.Tensor, right_img: torch.Tensor) -> torch.Tensor:
        """
        Generate pseudo ground truth disparity using FoundationStereo.
        
        Processes one sample at a time to avoid cuDNN grid_sample errors
        caused by huge batch dimensions in FoundationStereo's internal
        geometry encoding volume (b*h*w reshape).
        
        Args:
            left_img: [B, 3, H, W] normalized left images
            right_img: [B, 3, H, W] normalized right images
            
        Returns:
            [B, H, W] pseudo ground truth disparity
        """
        from stereo.modeling.models.foundationstereo.core.foundation_stereo import normalize_image
        from stereo.modeling.models.foundationstereo.core.utils.utils import InputPadder
        
        B = left_img.shape[0]
        
        # Denormalize images (normalized -> [0-255])
        left_denorm = self._denormalize_images(left_img)
        right_denorm = self._denormalize_images(right_img)
        
        # Move to foundation model device if different
        if left_denorm.device != self.foundation_device:
            left_denorm = left_denorm.to(self.foundation_device)
            right_denorm = right_denorm.to(self.foundation_device)
        
        # Process in sub-batches to avoid cuDNN grid_sample issues with large batches
        sub_bs = self.foundation_batch_size
        disp_list = []
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.foundation_use_amp):
            for i in range(0, B, sub_bs):
                # Normalize for FoundationStereo (expects [0-255] input)
                left_norm = normalize_image(left_denorm[i:i+sub_bs])
                right_norm = normalize_image(right_denorm[i:i+sub_bs])
                
                # Pad to multiple of 32
                padder = InputPadder(left_norm.shape[-2:], divis_by=32, force_square=False)
                left_pad, right_pad = padder.pad(left_norm, right_norm)
                
                # Run inference
                output = self.foundation_model({'left': left_pad, 'right': right_pad})
                
                # Get disparity and unpad
                disp_pred = padder.unpad(output['disp_pred'])
                disp_list.append(disp_pred.squeeze(1).float())
            
            # Stack back to [B, H, W]
            disp_pred = torch.cat(disp_list, dim=0)
            
            # Move back to original device if needed
            if disp_pred.device != self.device:
                disp_pred = disp_pred.to(self.device)
        
        return disp_pred
    
    def train_epoch(self, epoch: int, prefetcher: 'PseudoGTPrefetcher' = None) -> Dict[str, float]:
        """
        Train for one epoch with online pseudo GT generation.
        
        Uses PseudoGTPrefetcher to overlap FoundationStereo inference on batch N+1
        with LightStereo training on batch N via a background thread + CUDA stream.
        
        Args:
            epoch: Current epoch number
            prefetcher: Optional pre-created prefetcher (started during validation overlap)
            
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
        metrics_count = 0
        
        # Use pre-created prefetcher if provided (started during validation),
        # otherwise create a new one
        if prefetcher is None:
            prefetcher = PseudoGTPrefetcher(
                self.train_loader, self._generate_pseudo_gt, self.device,
                buffer_size=self.prefetch_buffer_size
            )
        pbar = tqdm(total=len(self.train_loader), desc=f"Training Epoch {epoch}", position=1, leave=False)
        
        for batch_idx, (batch, prefetch_error) in enumerate(prefetcher):
            # Handle prefetch errors
            if prefetch_error == 'oom':
                logger.error(f"CUDA OOM during pseudo GT generation at epoch {epoch}, batch {batch_idx}")
                logger.error("Consider reducing batch size or using a smaller FoundationStereo model")
                torch.cuda.empty_cache()
                pbar.update(1)
                continue
            elif prefetch_error is not None:
                logger.error(f"Prefetch error at epoch {epoch}, batch {batch_idx}: {prefetch_error}")
                pbar.update(1)
                continue
            
            # Forward pass with mixed precision
            with autocast('cuda', enabled=self.use_amp):
                model_output = self.model(batch)
                loss, loss_dict = self.loss_fn(model_output, batch)
                
                # Scale loss by accumulation steps
                loss = loss / self.accumulation_steps
            
            # NaN detection
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}")
                logger.error(f"Loss dict: {loss_dict}")
                pbar.update(1)
                continue
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.grad_clip_value > 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        logger.error(f"NaN/Inf gradient norm detected at epoch {epoch}, batch {batch_idx}")
                        self.optimizer.zero_grad()
                        pbar.update(1)
                        continue
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Scheduler step (if per-iteration)
                if self.scheduler is not None:
                    sched_config = self.config.get('optimization', {}).get('scheduler', {})
                    if not sched_config.get('on_epoch', False):
                        self.scheduler.step()
                
                # LR tracking
                current_lr = self.optimizer.param_groups[0]['lr']
                if (self.global_step % 50 == 0 or 
                    self.global_step < 10 or 
                    self.global_step in [1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180]):
                    logger.info(f"[LR] Step {self.global_step}, LR: {current_lr:.8f}")
            
            # Accumulate loss
            loss_value = loss_dict['loss_total']
            if torch.is_tensor(loss_value):
                epoch_metrics['loss'] += loss_value.detach().item() * self.accumulation_steps
            else:
                epoch_metrics['loss'] += loss_value * self.accumulation_steps
            
            # Compute metrics at log intervals
            log_interval = self.config.get('logging', {}).get('log_interval', 10)
            if batch_idx % log_interval == 0:
                with torch.no_grad():
                    metrics = self.metrics.compute_all_metrics(
                        model_output['disp_pred'],
                        batch['disparity'],
                        batch['valid_mask'],
                        max_disp=self.config.get('model', {}).get('max_disp', 192)
                    )
                
                # Accumulate metrics
                for key in ['epe', 'd1_all', 'd1_bg', 'd1_fg']:
                    epoch_metrics[key] += metrics[key]
                metrics_count += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['loss_total']:.4f}",
                    'epe': f"{metrics['epe']:.2f}",
                    'd1': f"{metrics['d1_all']:.2f}%"
                })
                
                # Log to WandB
                loss_log = loss_dict['loss_total'].item() if torch.is_tensor(loss_dict['loss_total']) else loss_dict['loss_total']
                if WANDB_AVAILABLE:
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
                pbar.set_postfix({'loss': f"{loss_dict['loss_total']:.4f}"})
            
            # Periodic CUDA cache clearing
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.global_step += 1
            pbar.update(1)
        
        pbar.close()
        
        # Average metrics over epoch
        num_batches = len(self.train_loader)
        epoch_metrics['loss'] /= num_batches
        if metrics_count > 0:
            for key in ['epe', 'd1_all', 'd1_bg', 'd1_fg']:
                epoch_metrics[key] /= metrics_count
        
        # Scheduler step (if per-epoch)
        if self.scheduler is not None:
            sched_config = self.config.get('optimization', {}).get('scheduler', {})
            if sched_config.get('on_epoch', False):
                self.scheduler.step()
        
        return epoch_metrics
    
    def train(self, num_epochs: int, save_dir: Optional[Path] = None):
        """
        Main training loop with async validation overlap.
        
        Starts the next epoch's PseudoGTPrefetcher BEFORE validation, so
        FoundationStereo inference runs concurrently with the validation pass.
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        if save_dir:
            save_dir = Path(save_dir)
            if WANDB_AVAILABLE and wandb.run is not None:
                save_dir = save_dir / wandb.run.name
                logger.info(f"Saving checkpoints to sweep-specific directory: {save_dir}")
            save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_epe = float('inf')
        
        epochs_to_run = self.executable_epochs if self.executable_epochs is not None else num_epochs
        if self.executable_epochs is not None:
            logger.info(f"Early stopping enabled: running {epochs_to_run} epochs (configured: {num_epochs})")
        
        # Create prefetcher ONCE (reused throughout training)
        prefetcher = PseudoGTPrefetcher(
            self.train_loader, self._generate_pseudo_gt, self.device,
            buffer_size=self.prefetch_buffer_size
        )
        
        # Initial validation (epoch 0)
        logger.info(f"\n{'='*80}")
        logger.info(f"Initial Validation (Epoch 0) - Before Training")
        logger.info(f"{'='*80}")
        
        # Start prefetching for epoch 1 BEFORE initial validation
        prefetcher.set_epoch(1)
        prefetcher.__iter__()
        logger.info("Started prefetching for epoch 1 (async with initial validation)")
        
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
            
            # Train (prefetcher already running from previous validation)
            train_metrics = self.train_epoch(epoch, prefetcher=prefetcher)
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"EPE: {train_metrics['epe']:.2f}, "
                       f"D1-all: {train_metrics['d1_all']:.2f}%")
            
            # Reset and restart prefetcher for NEXT epoch BEFORE validation
            if epoch < epochs_to_run:
                prefetcher.reset()
                prefetcher.set_epoch(epoch + 1)
                prefetcher.__iter__()
                logger.info(f"Started prefetching for epoch {epoch + 1} (async with validation)")
            
            # Validate (while prefetcher runs in background)
            val_metrics = self.validate(epoch)
            if val_metrics:
                logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                           f"EPE: {val_metrics['epe']:.2f}, "
                           f"D1-all: {val_metrics['d1_all']:.2f}%")
                
                if save_dir:
                    self._save_visualizations(save_dir, epoch, prefix='val')
                
                if save_dir and val_metrics['epe'] < best_val_epe:
                    best_val_epe = val_metrics['epe']
                    self.save_checkpoint(save_dir / 'best_model.pth', epoch, val_metrics)
                    logger.info(f"✓ Saved best model (EPE: {best_val_epe:.2f})")
            
            if save_dir and epoch % 10 == 0:
                self.save_checkpoint(save_dir / f'checkpoint_epoch_{epoch}.pth', epoch, train_metrics)
        
        # Cleanup
        prefetcher.stop()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Training complete! Best val EPE: {best_val_epe:.2f}")
        logger.info(f"{'='*80}")
        
        if WANDB_AVAILABLE:
            wandb.finish()
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate on validation set with online pseudo GT generation.
        
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
        
        pbar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch}", position=1, leave=False)
        
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Generate pseudo GT disparity using FoundationStereo
            try:
                pseudo_gt = self._generate_pseudo_gt(batch['left'], batch['right'])
                batch['disparity'] = pseudo_gt
                batch['valid_mask'] = torch.ones_like(pseudo_gt, dtype=torch.bool)
            except torch.cuda.OutOfMemoryError:
                logger.error(f"CUDA OOM during validation pseudo GT generation")
                torch.cuda.empty_cache()
                continue
            
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
            
            # Accumulate
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
        if WANDB_AVAILABLE:
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
        Save visualizations with online pseudo GT generation.
        
        Overrides parent to generate pseudo GT disparity before visualizing,
        since the dataset returns dummy zeros for 'pseudo-foundationstereo-online' mode.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
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
            
            # Generate pseudo GT disparity using FoundationStereo
            pseudo_gt = self._generate_pseudo_gt(batch['left'], batch['right'])
            batch['disparity'] = pseudo_gt
            batch['valid_mask'] = torch.ones_like(pseudo_gt, dtype=torch.bool)
            
            # Forward pass through student model
            with autocast('cuda', enabled=self.use_amp):
                model_output = self.model(batch)
            
            # Get predictions and GT
            pred_disp = model_output['disp_pred'][0].detach().cpu().numpy().squeeze()
            gt_disp = batch['disparity'][0].detach().cpu().numpy().squeeze()
            valid_mask = batch['valid_mask'][0].detach().cpu().numpy().squeeze()
            
            # Create figure with 3 subplots
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
            
            # Plot GT disparity (pseudo GT from FoundationStereo)
            im0 = axes[0].imshow(gt_disp, cmap=custom_cmap, vmin=0, vmax=512)
            axes[0].set_title(f'Pseudo GT Disparity (Sample {sample_idx})', fontsize=12)
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            # Plot predicted disparity
            im1 = axes[1].imshow(pred_disp, cmap=custom_cmap, vmin=0, vmax=512)
            axes[1].set_title(f'Predicted Disparity (Sample {sample_idx})', fontsize=12)
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Plot error map
            error = np.abs(pred_disp - gt_disp) * valid_mask
            im2 = axes[2].imshow(error, cmap='hot', vmin=0, vmax=10)
            axes[2].set_title(f'Error Map (Sample {sample_idx})', fontsize=12)
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            # Compute EPE
            epe = np.sum(error) / np.sum(valid_mask) if np.sum(valid_mask) > 0 else 0
            
            fig.suptitle(f'Epoch {epoch} - {prefix.upper()} Sample {sample_idx} - EPE: {epe:.2f}px', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save figure
            save_path = viz_dir / f'{prefix}_sample_{sample_idx}_epoch_{epoch:03d}.png'
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Log to WandB
            if WANDB_AVAILABLE:
                wandb.log({
                    f'{prefix}/viz_sample_{sample_idx}': wandb.Image(str(save_path)),
                    'epoch': epoch,
                })
        
        logger.info(f"Saved {len(self.viz_samples_cache)} visualization(s) to {viz_dir}")
