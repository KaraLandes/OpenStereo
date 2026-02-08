"""
Loss functions for stereo matching training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SupervisedLoss(nn.Module):
    """
    Supervised loss for stereo matching.
    Uses Smooth L1 loss on predicted disparity vs ground truth.
    Matches LightStereo paper implementation.
    
    Optionally includes temporal consistency loss for sequential training.
    """
    
    def __init__(self, max_disp: float = 192.0, loss_weights: dict = None):
        """
        Args:
            max_disp: Maximum disparity value
            loss_weights: Dict with weights for different outputs
                         e.g., {'disp_pred': 1.0, 'disp_4': 0.3, 'temporal_consistency': 0.1}
        """
        super().__init__()
        self.max_disp = max_disp
        self.loss_weights = loss_weights or {'disp_pred': 1.0, 'disp_4': 0.3}
        
        # Temporal consistency weight (0.0 = disabled)
        self.temporal_consistency_weight = self.loss_weights.get('temporal_consistency', 0.0)
        
        logger.info(f"Initialized SupervisedLoss with max_disp={max_disp}, weights={self.loss_weights}")
    
    def forward(self, model_output: dict, batch: dict) -> tuple:
        """
        Compute supervised loss.
        
        Args:
            model_output: Dict from model forward pass
                         {'disp_pred': [B, 1, H, W], 'disp_4': [B, 1, H, W]}
            batch: Batch dict with 'disparity' and 'valid_mask'
            
        Returns:
            (total_loss, loss_dict)
        """
        disp_gt = batch['disparity']  # [B, H, W]
        valid_mask = batch['valid_mask']  # [B, H, W]
        
        # Add channel dimension if needed
        if disp_gt.dim() == 3:
            disp_gt = disp_gt.unsqueeze(1)  # [B, 1, H, W]
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]
        
        # Create mask: valid pixels within disparity range
        mask = valid_mask & (disp_gt < self.max_disp) & (disp_gt > 0)
        
        total_loss = 0.0
        loss_dict = {}
        
        # Loss on final prediction
        if 'disp_pred' in model_output:
            disp_pred = model_output['disp_pred']
            weight = self.loss_weights.get('disp_pred', 1.0)
            loss = weight * F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction='mean')
            total_loss += loss
            loss_dict['loss_disp_pred'] = loss.item()
        
        # Loss on intermediate prediction (1/4 resolution)
        if 'disp_4' in model_output:
            disp_4 = model_output['disp_4']
            weight = self.loss_weights.get('disp_4', 0.3)
            loss = weight * F.smooth_l1_loss(disp_4[mask], disp_gt[mask], reduction='mean')
            total_loss += loss
            loss_dict['loss_disp_4'] = loss.item()
        
        loss_dict['loss_total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def compute_temporal_consistency(self, current_disp, warped_prev_disp, valid_mask):
        """
        Compute temporal consistency loss between current and warped previous disparity.
        
        Args:
            current_disp: [B, 1, H, W] - current frame disparity prediction
            warped_prev_disp: [B, 1, H, W] - warped previous frame disparity
            valid_mask: [B, 1, H, W] - mask indicating valid warped regions
        
        Returns:
            temporal_loss: Scalar tensor
        """
        if self.temporal_consistency_weight == 0.0:
            return torch.tensor(0.0, device=current_disp.device, dtype=current_disp.dtype)
        
        # L1 difference weighted by valid mask
        diff = torch.abs(current_disp - warped_prev_disp) * valid_mask
        
        # Mean over valid pixels only
        num_valid = valid_mask.sum() + 1e-6
        temporal_loss = self.temporal_consistency_weight * (diff.sum() / num_valid)
        
        return temporal_loss


class SSLLoss(nn.Module):
    """
    Self-supervised loss for stereo matching.
    Placeholder for future implementation.
    """
    
    def __init__(self, config: dict = None):
        """
        Args:
            config: SSL loss configuration
        """
        super().__init__()
        self.config = config or {}
        
        logger.warning("SSLLoss is a placeholder - not implemented yet!")
    
    def forward(self, model_output: dict, batch: dict) -> tuple:
        """
        Compute self-supervised loss.
        
        Args:
            model_output: Dict from model forward pass
            batch: Batch dict with left/right images
            
        Returns:
            (total_loss, loss_dict)
        """
        raise NotImplementedError(
            "Self-supervised loss not implemented yet. "
            "Use target_source='provided' or 'pseudo-foundationstereo' for supervised training."
        )


def build_loss_function(config: dict, training_mode: str = 'supervised') -> nn.Module:
    """
    Build loss function based on configuration.
    
    Args:
        config: Loss configuration dict
        training_mode: 'supervised' or 'ssl'
        
    Returns:
        Loss function module
    """
    if training_mode == 'supervised':
        max_disp = config.get('max_disp', 192.0)
        loss_weights = config.get('loss_weights', {'disp_pred': 1.0, 'disp_4': 0.3})
        return SupervisedLoss(max_disp=max_disp, loss_weights=loss_weights)
    
    elif training_mode == 'ssl':
        return SSLLoss(config=config)
    
    else:
        raise ValueError(f"Unknown training_mode: {training_mode}. Use 'supervised' or 'ssl'.")
