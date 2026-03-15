"""
Presaved Pseudo Ground Truth Trainer

Extends OnlinePseudoGTTrainer to work with presaved disparities.
On cache hit: uses presaved disparity directly (fast).
On cache miss: generates using FoundationStereo, saves to disk, then uses it.

Usage:
    Set target_source: 'pseudo-foundationstereo-presaved' in dataset config
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from torch.amp import autocast
from tqdm import tqdm

from .online_pseudo_trainer import OnlinePseudoGTTrainer

logger = logging.getLogger(__name__)

# WandB import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class PresavedPseudoGTTrainer(OnlinePseudoGTTrainer):
    """
    Trainer that uses presaved pseudo-GT disparities with on-demand generation.
    
    Workflow:
    1. DataLoader returns batch with presaved disparities (cache hits)
       and flagged samples needing generation (cache misses)
    2. For cache misses: generate using FoundationStereo, save to disk, use for training
    3. For cache hits: use presaved disparity directly
    
    This avoids the overhead of running FoundationStereo for every sample,
    while still handling missing presaved files gracefully.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        train_loader,
        val_loader=None,
        device: str = 'cuda'
    ):
        super().__init__(config, train_loader, val_loader, device)
        
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info("PresavedPseudoGTTrainer initialized (cache-or-generate mode)")
    
    def _handle_cache_misses(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check batch for cache misses and generate + save missing disparities.
        
        Args:
            batch: Collated batch from DataLoader
            
        Returns:
            Updated batch with generated disparities filled in
        """
        needs_gen = batch.get('needs_generation', None)
        if needs_gen is None:
            return batch
        
        # needs_gen is a tensor of bools after collation
        if not isinstance(needs_gen, torch.Tensor):
            needs_gen = torch.tensor(needs_gen)
        
        num_hits = (~needs_gen).sum().item()
        num_misses = needs_gen.sum().item()
        self._cache_hits += num_hits
        self._cache_misses += num_misses
        
        if num_misses == 0:
            return batch
        
        # Get indices of samples needing generation
        gen_indices = needs_gen.nonzero(as_tuple=True)[0]
        
        logger.debug(f"Cache misses: {num_misses}/{len(needs_gen)} samples in batch")
        
        # Generate pseudo GT for missing samples
        left_subset = batch['left'][gen_indices]
        right_subset = batch['right'][gen_indices]
        
        pseudo_gt = self._generate_pseudo_gt(left_subset, right_subset)
        
        # Fill in the generated disparities
        batch['disparity'][gen_indices] = pseudo_gt
        batch['valid_mask'][gen_indices] = (pseudo_gt > 0) & (pseudo_gt < 512)
        
        # Save generated disparities to disk for future cache hits
        self._save_generated_disparities(batch, gen_indices, pseudo_gt)
        
        return batch
    
    def _save_generated_disparities(
        self, batch: Dict[str, Any], gen_indices: torch.Tensor, pseudo_gt: torch.Tensor
    ):
        """
        Save generated disparities to disk as .npy files.
        
        Args:
            batch: Full batch with metadata
            gen_indices: Indices of generated samples
            pseudo_gt: Generated disparity tensors [N, H, W]
        """
        metadata = batch.get('metadata', None)
        crop_coords = batch.get('crop_coords', None)
        
        if metadata is None or crop_coords is None:
            logger.warning("Cannot save generated disparities: missing metadata or crop_coords")
            return
        
        for i, batch_idx in enumerate(gen_indices):
            idx = batch_idx.item()
            try:
                # Extract per-sample metadata
                if isinstance(metadata, list):
                    frame_id = metadata[idx].get('frame_id', '00000')
                    left_path = metadata[idx].get('left_path', None)
                elif isinstance(metadata, dict):
                    frame_id = metadata['frame_id'][idx] if isinstance(metadata['frame_id'], list) else metadata['frame_id']
                    left_path = metadata['left_path'][idx] if isinstance(metadata['left_path'], list) else metadata['left_path']
                else:
                    continue
                
                if left_path is None:
                    continue
                
                # Extract crop coordinates
                if isinstance(crop_coords, dict):
                    y = crop_coords['y'][idx].item() if torch.is_tensor(crop_coords['y']) else crop_coords['y']
                    x = crop_coords['x'][idx].item() if torch.is_tensor(crop_coords['x']) else crop_coords['x']
                else:
                    continue
                
                # Build save path
                scene_dir = Path(left_path).parent
                presaved_filename = f"pseudo_d_{y}_{x}_{frame_id}.npy"
                presaved_path = scene_dir / presaved_filename
                
                # Save as float16 (consistent with generation script)
                disp_np = pseudo_gt[i].cpu().numpy().astype(np.float16)
                np.save(str(presaved_path), disp_np)
                
                logger.debug(f"Saved generated disparity: {presaved_path}")
                
            except Exception as e:
                logger.error(f"Failed to save generated disparity for batch idx {idx}: {e}")
    
    def train_epoch(self, epoch: int, prefetcher=None) -> Dict[str, float]:
        """
        Train for one epoch with presaved pseudo GT + on-demand generation.
        
        Unlike OnlinePseudoGTTrainer which generates ALL disparities online,
        this only generates for cache misses.
        """
        self.model.train()
        self.current_epoch = epoch
        
        # Reset cache stats per epoch
        self._cache_hits = 0
        self._cache_misses = 0
        
        epoch_metrics = {
            'loss': 0.0,
            'epe': 0.0,
            'd1_all': 0.0,
            'd1_bg': 0.0,
            'd1_fg': 0.0,
        }
        metrics_count = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}", position=0, leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            
            # Handle cache misses: generate + save missing disparities
            batch = self._handle_cache_misses(batch)
            
            # Forward pass with mixed precision
            with autocast('cuda', enabled=self.use_amp):
                model_output = self.model(batch)
                loss, loss_dict = self.loss_fn(model_output, batch)
                
                # Scale loss by accumulation steps
                loss = loss / self.accumulation_steps
            
            # NaN detection
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss at epoch {epoch}, batch {batch_idx}")
                pbar.update(1)
                continue
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.grad_clip_value > 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        self.optimizer.zero_grad()
                        continue
                
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
                if self.global_step % 50 == 0:
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
                
                for key in ['epe', 'd1_all', 'd1_bg', 'd1_fg']:
                    epoch_metrics[key] += metrics[key]
                metrics_count += 1
                
                cache_rate = self._cache_hits / max(1, self._cache_hits + self._cache_misses) * 100
                pbar.set_postfix({
                    'loss': f"{loss_dict['loss_total']:.4f}",
                    'epe': f"{metrics['epe']:.2f}",
                    'd1': f"{metrics['d1_all']:.2f}%",
                    'cache': f"{cache_rate:.0f}%"
                })
                
                # Log to WandB
                loss_log = loss_dict['loss_total'].item() if torch.is_tensor(loss_dict['loss_total']) else loss_dict['loss_total']
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({
                        'train/loss': loss_log,
                        'train/epe': metrics['epe'],
                        'train/d1_all': metrics['d1_all'],
                        'train/d1_bg': metrics['d1_bg'],
                        'train/d1_fg': metrics['d1_fg'],
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'train/cache_hit_rate': cache_rate,
                        'epoch': epoch,
                        'step': self.global_step,
                    })
            else:
                pbar.set_postfix({'loss': f"{loss_dict['loss_total']:.4f}"})
            
            # Periodic CUDA cache clearing
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.global_step += 1
        
        pbar.close()
        
        # Log cache stats
        total = self._cache_hits + self._cache_misses
        cache_rate = self._cache_hits / max(1, total) * 100
        logger.info(f"Epoch {epoch} cache stats: {self._cache_hits}/{total} hits ({cache_rate:.1f}%), "
                    f"{self._cache_misses} misses (generated + saved)")
        
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
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate with presaved pseudo GT + on-demand generation.
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
        
        pbar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch}", position=0, leave=False)
        
        for batch in pbar:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            
            # Handle cache misses
            batch = self._handle_cache_misses(batch)
            
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
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                'val/loss': val_metrics['loss'],
                'val/epe': val_metrics['epe'],
                'val/d1_all': val_metrics['d1_all'],
                'val/d1_bg': val_metrics['d1_bg'],
                'val/d1_fg': val_metrics['d1_fg'],
                'epoch': epoch,
            })
        
        return val_metrics
    
    def train(self, num_epochs: int, save_dir: Optional[Path] = None):
        """
        Main training loop for presaved pseudo GT mode.
        
        Unlike OnlinePseudoGTTrainer, this does NOT use a prefetcher since
        most samples are loaded from cache (fast) and only cache misses
        trigger FoundationStereo inference.
        """
        logger.info(f"Starting presaved pseudo GT training for {num_epochs} epochs")
        
        if save_dir:
            save_dir = Path(save_dir)
            if WANDB_AVAILABLE and wandb.run is not None:
                save_dir = save_dir / wandb.run.name
                logger.info(f"Saving checkpoints to: {save_dir}")
            save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_epe = float('inf')
        
        epochs_to_run = self.executable_epochs if self.executable_epochs is not None else num_epochs
        if self.executable_epochs is not None:
            logger.info(f"Running {epochs_to_run} epochs (configured: {num_epochs})")
        
        # Initial validation
        logger.info(f"\n{'='*80}")
        logger.info(f"Initial Validation (Epoch 0)")
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
                
                if save_dir:
                    self._save_visualizations(save_dir, epoch, prefix='val')
                
                if save_dir and val_metrics['epe'] < best_val_epe:
                    best_val_epe = val_metrics['epe']
                    self.save_checkpoint(save_dir / 'best_model.pth', epoch, val_metrics)
                    logger.info(f"Saved best model (EPE: {best_val_epe:.2f})")
            
            if save_dir and epoch % 10 == 0:
                self.save_checkpoint(save_dir / f'checkpoint_epoch_{epoch}.pth', epoch, train_metrics)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Training complete! Best val EPE: {best_val_epe:.2f}")
        logger.info(f"{'='*80}")
        
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
