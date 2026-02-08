"""
Training infrastructure for DAI pipeline
"""

from .trainer import DAITrainer
from .sequential_trainer import SequentialTrainer
from .warping_sequential_trainer import WarpingSequentialTrainer
from .metrics import StereoMetrics
from .losses import build_loss_function


def build_trainer(config, train_loader, val_loader=None, device='cuda'):
    """
    Build trainer based on training mode in config.
    
    Args:
        config: Training configuration dict
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        device: Device to train on
    
    Returns:
        Trainer instance
    """
    mode = config.get('training', {}).get('mode', 'normal')
    
    if mode == 'normal':
        # Standard single-frame training
        return DAITrainer(config, train_loader, val_loader, device)
    
    elif mode in ['sequential', 'sequential_fused_features']:
        # Temporal with feature fusion (current implementation)
        return SequentialTrainer(config, train_loader, val_loader, device)
    
    elif mode == 'sequential_warped_disparity':
        # Temporal with disparity warping (new implementation)
        return WarpingSequentialTrainer(config, train_loader, val_loader, device)
    
    else:
        raise ValueError(
            f"Unknown training mode: {mode}. "
            f"Use 'normal', 'sequential_fused_features', or 'sequential_warped_disparity'."
        )


__all__ = [
    'DAITrainer',
    'SequentialTrainer',
    'WarpingSequentialTrainer',
    'StereoMetrics',
    'build_loss_function',
    'build_trainer',
    'SupervisedLoss',
    'SSLLoss',
]
