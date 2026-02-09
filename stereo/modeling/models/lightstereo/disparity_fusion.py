"""
Configurable disparity fusion modules for temporal stereo matching.

Supports three fusion strategies:
1. Convolutional Fusion: Confidence-weighted blending
2. Recurrent Fusion: ConvGRU-based temporal memory
3. Residual Fusion: Predict disparity delta/correction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    """
    Convolutional GRU cell for spatial-temporal fusion.
    """
    
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        
        padding = kernel_size // 2
        
        # Reset gate
        self.conv_reset = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=padding
        )
        
        # Update gate
        self.conv_update = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=padding
        )
        
        # New gate
        self.conv_new = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=padding
        )
    
    def forward(self, x, h):
        """
        Args:
            x: Input tensor [B, C_in, H, W]
            h: Hidden state [B, C_hidden, H, W]
        
        Returns:
            new_h: Updated hidden state [B, C_hidden, H, W]
        """
        combined = torch.cat([x, h], dim=1)
        
        # Compute gates
        reset_gate = torch.sigmoid(self.conv_reset(combined))
        update_gate = torch.sigmoid(self.conv_update(combined))
        
        # Compute new hidden state candidate
        combined_reset = torch.cat([x, reset_gate * h], dim=1)
        new_candidate = torch.tanh(self.conv_new(combined_reset))
        
        # Update hidden state
        new_h = (1 - update_gate) * h + update_gate * new_candidate
        
        return new_h


class ConvolutionalFusion(nn.Module):
    """
    Confidence-weighted fusion of current and previous disparity.
    
    Learns per-pixel weights to blend predictions.
    Simple, stable, and effective.
    """
    
    def __init__(self, hidden_channels=64):
        super().__init__()
        
        # Encode both disparities
        self.current_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden_channels, 3, 1, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        self.prev_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden_channels, 3, 1, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Confidence network: outputs weights for [current, prev]
        self.confidence_net = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1, 1, 0),
            nn.Softmax(dim=1)  # [B, 2, H, W]
        )
    
    def forward(self, current_disp, prev_disp, hidden_state=None):
        """
        Args:
            current_disp: [B, 1, H, W] - current frame prediction
            prev_disp: [B, 1, H, W] - previous frame prediction
            hidden_state: Not used (for API consistency)
        
        Returns:
            fused_disp: [B, 1, H, W]
            new_hidden: None (stateless)
        """
        # Encode both disparities
        current_feat = self.current_encoder(current_disp)
        prev_feat = self.prev_encoder(prev_disp)
        
        # Compute confidence weights
        combined = torch.cat([current_feat, prev_feat], dim=1)
        weights = self.confidence_net(combined)  # [B, 2, H, W]
        
        # Weighted fusion
        fused_disp = (weights[:, 0:1] * current_disp + 
                     weights[:, 1:2] * prev_disp)
        
        return fused_disp, None


class RecurrentFusion(nn.Module):
    """
    GRU-based temporal fusion with hidden state memory.
    
    Accumulates temporal context across frames.
    More complex but handles longer sequences better.
    """
    
    def __init__(self, hidden_channels=64):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        
        # Encode current disparity
        self.current_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden_channels, 3, 1, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encode previous disparity (for initialization)
        self.prev_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden_channels, 3, 1, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # ConvGRU cell
        self.gru_cell = ConvGRUCell(hidden_channels, hidden_channels)
        
        # Decode to disparity
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1)
        )
    
    def forward(self, current_disp, prev_disp, hidden_state=None):
        """
        Args:
            current_disp: [B, 1, H, W] - current frame prediction
            prev_disp: [B, 1, H, W] - previous frame prediction
            hidden_state: [B, C, H, W] - GRU hidden state (optional)
        
        Returns:
            fused_disp: [B, 1, H, W]
            new_hidden: [B, C, H, W]
        """
        # Encode current observation
        current_feat = self.current_encoder(current_disp)
        
        # Initialize hidden state if None
        if hidden_state is None:
            prev_feat = self.prev_encoder(prev_disp)
            hidden_state = prev_feat
        
        # Update hidden state with current observation
        new_hidden = self.gru_cell(current_feat, hidden_state)
        
        # Decode to disparity
        delta_disp = self.decoder(new_hidden)
        
        # Residual connection with current prediction
        fused_disp = current_disp + delta_disp
        
        return fused_disp, new_hidden


class ResidualFusion(nn.Module):
    """
    Residual fusion: predict disparity correction/delta.
    
    Exploits temporal smoothness - predicts small changes.
    Works well for smooth motion and static scenes.
    """
    
    def __init__(self, hidden_channels=64):
        super().__init__()
        
        # Encode both disparities
        self.current_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden_channels, 3, 1, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        self.prev_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden_channels, 3, 1, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual prediction network
        self.residual_net = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1)
        )
    
    def forward(self, current_disp, prev_disp, hidden_state=None):
        """
        Args:
            current_disp: [B, 1, H, W] - current frame prediction
            prev_disp: [B, 1, H, W] - previous frame prediction
            hidden_state: Not used (for API consistency)
        
        Returns:
            fused_disp: [B, 1, H, W]
            new_hidden: None (stateless)
        """
        # Encode both disparities
        current_feat = self.current_encoder(current_disp)
        prev_feat = self.prev_encoder(prev_disp)
        
        # Predict residual/delta
        combined = torch.cat([current_feat, prev_feat], dim=1)
        delta = self.residual_net(combined)
        
        # Apply residual to current prediction
        # Can also apply to previous: prev_disp + delta
        fused_disp = current_disp + delta
        
        return fused_disp, None


def build_fusion_module(fusion_type='convolutional', hidden_channels=64):
    """
    Factory function to build fusion module based on type.
    
    Args:
        fusion_type: One of ['convolutional', 'recurrent', 'residual']
        hidden_channels: Hidden state channels
    
    Returns:
        Fusion module instance
    """
    fusion_type = fusion_type.lower()
    
    if fusion_type == 'convolutional':
        return ConvolutionalFusion(hidden_channels)
    elif fusion_type == 'recurrent':
        return RecurrentFusion(hidden_channels)
    elif fusion_type == 'residual':
        return ResidualFusion(hidden_channels)
    else:
        raise ValueError(
            f"Unknown fusion type: {fusion_type}. "
            f"Choose from ['convolutional', 'recurrent', 'residual']"
        )
