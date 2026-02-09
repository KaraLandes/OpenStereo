"""
Temporal LightStereo with configurable post-prediction fusion.

Supports three fusion strategies:
- Convolutional: Confidence-weighted blending
- Recurrent: ConvGRU-based temporal memory
- Residual: Predict disparity delta/correction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lightstereo import LightStereo
from .disparity_fusion import build_fusion_module


class TemporalLightStereo(nn.Module):
    """
    Temporal extension of LightStereo with configurable post-prediction fusion.
    
    Architecture:
    1. Run base LightStereo on current frame → current_disp
    2. Fuse current_disp with prev_disp using selected strategy
    3. Return fused prediction
    """
    
    def __init__(self, cfgs):
        """
        Initialize temporal LightStereo.
        
        Args:
            cfgs: Configuration object
                - TEMPORAL_FUSION_TYPE: 'convolutional', 'recurrent', or 'residual'
                - TEMPORAL_HIDDEN_CHANNELS: Hidden state channels (default: 64)
                - USE_TEMPORAL_FUSION: Enable/disable fusion (default: True)
        """
        super().__init__()
        
        self.max_disp = cfgs.MAX_DISP
        
        # Base LightStereo model (unchanged)
        self.base_model = LightStereo(cfgs)
        
        # Post-prediction fusion module
        fusion_type = cfgs.get('TEMPORAL_FUSION_TYPE', 'convolutional')
        hidden_channels = cfgs.get('TEMPORAL_HIDDEN_CHANNELS', 64)
        self.fusion = build_fusion_module(fusion_type, hidden_channels)
        
        self.use_temporal_fusion = cfgs.get('USE_TEMPORAL_FUSION', True)
        self.fusion_type = fusion_type
        
        # Track if fusion is stateful (recurrent)
        self.is_stateful = (fusion_type.lower() == 'recurrent')
    
    
    def forward(self, data: dict, prev_disparity: torch.Tensor = None, hidden_state: torch.Tensor = None) -> dict:
        """
        Forward pass with post-prediction temporal fusion.
        
        Args:
            data: Dictionary with 'left' and 'right' images [B, 3, H, W]
            prev_disparity: Previous disparity prediction [B, 1, H, W] (optional)
            hidden_state: Hidden state for recurrent fusion [B, C, H, W] (optional)
            
        Returns:
            Dictionary with:
            - 'disp_pred': Fused disparity prediction [B, 1, H, W]
            - 'disp_pred_current': Current frame only prediction [B, 1, H, W]
            - 'hidden_state': Updated hidden state (recurrent fusion only)
            - 'disp_4': Intermediate disparity at 1/4 resolution (training only)
        """
        # 1. Run base LightStereo (full forward pass)
        base_output = self.base_model(data)
        current_disp = base_output['disp_pred']  # [B, 1, H, W]
        
        # 2. Fuse with previous prediction if available
        if self.use_temporal_fusion and prev_disparity is not None:
            fused_disp, new_hidden = self.fusion(
                current_disp, 
                prev_disparity,
                hidden_state
            )
        else:
            # First frame or fusion disabled: no fusion
            fused_disp = current_disp
            new_hidden = None
        
        # 3. Prepare output
        result = {
            'disp_pred': fused_disp,
            'disp_pred_current': current_disp,
            'hidden_state': new_hidden
        }
        
        if self.training:
            result['disp_4'] = base_output['disp_4']
        
        return result
    
    
    def get_loss(self, model_pred, input_data):
        """
        Compute loss (same as base LightStereo).
        
        Args:
            model_pred: Model predictions
            input_data: Input data with ground truth
            
        Returns:
            Tuple of (loss, loss_info)
        """
        disp_gt = input_data["disp"]
        disp_gt = disp_gt.unsqueeze(1)
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)
        
        disp_pred = model_pred['disp_pred']
        loss = 1.0 * F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction='mean')
        
        if 'disp_4' in model_pred:
            disp_4 = model_pred['disp_4']
            loss += 0.3 * F.smooth_l1_loss(disp_4[mask], disp_gt[mask], reduction='mean')
        
        loss_info = {'scalar/train/loss_disp': loss.item()}
        
        return loss, loss_info
