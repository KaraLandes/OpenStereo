"""
Lightweight optical flow estimator using correlation.
Computationally efficient for temporal stereo matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightFlowEstimator(nn.Module):
    """
    Lightweight optical flow estimator using correlation.
    Computationally efficient, suitable for real-time applications.
    
    Architecture:
    - Lightweight feature extractor (2 conv layers, 16 channels)
    - Correlation volume computation
    - Flow regression head
    
    Total parameters: ~7K
    Total FLOPs: ~1 GFLOP (for 320x736 image)
    """
    
    def __init__(self, max_displacement=4, feature_channels=16):
        """
        Initialize flow estimator.
        
        Args:
            max_displacement: Maximum displacement to search in each direction
            feature_channels: Number of feature channels (default: 16 for efficiency)
        """
        super().__init__()
        self.max_displacement = max_displacement
        self.feature_channels = feature_channels
        
        # Very lightweight feature extractor (only 2 layers)
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, feature_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )
        
        # Flow regression from correlation
        # Correlation volume has (2*max_disp+1)^2 channels
        corr_channels = (2 * max_displacement + 1) ** 2
        
        self.flow_head = nn.Sequential(
            nn.Conv2d(corr_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        )
    
    def compute_correlation_volume(self, feat1, feat2):
        """
        Compute correlation volume efficiently.
        
        Args:
            feat1: [B, C, H, W] - features from previous frame
            feat2: [B, C, H, W] - features from current frame
        
        Returns:
            correlation: [B, (2*D+1)^2, H, W] where D=max_displacement
        """
        B, C, H, W = feat1.shape
        D = self.max_displacement
        
        # Normalize features for better correlation
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        
        # Pad feat2 for shifting
        feat2_padded = F.pad(feat2, [D, D, D, D], mode='replicate')
        
        # Compute correlation for each displacement
        correlations = []
        for dy in range(-D, D + 1):
            for dx in range(-D, D + 1):
                # Extract shifted region
                y_start = D + dy
                x_start = D + dx
                shifted_feat2 = feat2_padded[:, :, y_start:y_start+H, x_start:x_start+W]
                
                # Compute correlation (dot product)
                corr = (feat1 * shifted_feat2).sum(dim=1, keepdim=True)  # [B, 1, H, W]
                correlations.append(corr)
        
        correlation = torch.cat(correlations, dim=1)  # [B, (2D+1)^2, H, W]
        return correlation
    
    def forward(self, img1, img2):
        """
        Estimate optical flow between two frames.
        
        Args:
            img1: [B, 3, H, W] - previous frame (RGB)
            img2: [B, 3, H, W] - current frame (RGB)
        
        Returns:
            flow: [B, 2, H, W] - optical flow (flow_x, flow_y) in pixels
        """
        # Extract lightweight features
        feat1 = self.feature_net(img1)  # [B, C, H, W]
        feat2 = self.feature_net(img2)  # [B, C, H, W]
        
        # Compute correlation volume
        correlation = self.compute_correlation_volume(feat1, feat2)
        
        # Regress flow from correlation
        flow = self.flow_head(correlation)  # [B, 2, H, W]
        
        return flow
