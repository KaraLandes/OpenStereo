"""
Training infrastructure for DAI pipeline
"""

from .trainer import DAITrainer
from .sequential_trainer import SequentialTrainer
from .warping_sequential_trainer import WarpingSequentialTrainer
from .online_pseudo_trainer import OnlinePseudoGTTrainer
from .presaved_pseudo_trainer import PresavedPseudoGTTrainer
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
    # Check if any dataset uses online or presaved pseudo GT generation
    datasets_config = config.get('datasets', [])
    uses_online_pseudo = any(
        ds.get('target_source') == 'pseudo-foundationstereo-online' 
        for ds in datasets_config
    )
    uses_presaved_pseudo = any(
        ds.get('target_source') == 'pseudo-foundationstereo-presaved' 
        for ds in datasets_config
    )
    
    if uses_presaved_pseudo:
        # Presaved pseudo GT with on-demand generation for cache misses
        return PresavedPseudoGTTrainer(config, train_loader, val_loader, device)
    
    if uses_online_pseudo:
        # Online pseudo ground truth generation with FoundationStereo
        return OnlinePseudoGTTrainer(config, train_loader, val_loader, device)
    
    # Standard mode detection
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
    'OnlinePseudoGTTrainer',
    'PresavedPseudoGTTrainer',
    'StereoMetrics',
    'build_loss_function',
    'build_trainer',
    'SupervisedLoss',
    'SSLLoss',
]
