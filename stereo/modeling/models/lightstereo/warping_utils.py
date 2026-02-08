"""
Warping utilities for temporal stereo matching.
Includes disparity warping and flow normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_flow_median(flow, percentile=50.0):
    """
    Normalize flow using median (50th percentile) to handle outliers.
    Robust to extreme values while preserving typical motion patterns.
    
    Args:
        flow: [B, 2, H, W] - optical flow (flow_x, flow_y)
        percentile: Percentile to use for normalization (default: 50 = median)
    
    Returns:
        normalized_flow: [B, 2, H, W] - normalized to approximately [-1, 1]
    """
    # Compute magnitude
    magnitude = torch.sqrt(flow[:, 0]**2 + flow[:, 1]**2)  # [B, H, W]
    
    # Get median magnitude (robust to outliers)
    median_magnitude = torch.quantile(magnitude.flatten(), percentile / 100.0)
    median_magnitude = torch.clamp(median_magnitude, min=1.0)  # Avoid division by zero
    
    # Normalize by median and clamp to [-1, 1]
    normalized_flow = flow / median_magnitude
    normalized_flow = torch.clamp(normalized_flow, -1.0, 1.0)
    
    return normalized_flow


class DisparityWarper(nn.Module):
    """
    Warp disparity map using optical flow.
    Handles out-of-frame pixels with border padding.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, disparity, flow, padding_mode='border'):
        """
        Warp disparity using optical flow.
        
        Args:
            disparity: [B, 1, H, W] - previous frame disparity
            flow: [B, 2, H, W] - optical flow from previous to current frame
            padding_mode: Padding mode for out-of-frame pixels
                         'border' - replicate edge values (recommended)
                         'zeros' - use zero for out-of-frame
        
        Returns:
            warped_disparity: [B, 1, H, W] - warped disparity
            valid_mask: [B, 1, H, W] - mask indicating valid warped regions
        """
        B, _, H, W = disparity.shape
        device = disparity.device
        
        # Create base sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0)  # [2, H, W]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]
        
        # Apply flow to grid
        warped_grid = grid + flow  # [B, 2, H, W]
        
        # Normalize to [-1, 1] for grid_sample
        warped_grid[:, 0] = 2.0 * warped_grid[:, 0] / (W - 1) - 1.0  # x
        warped_grid[:, 1] = 2.0 * warped_grid[:, 1] / (H - 1) - 1.0  # y
        
        # Permute to [B, H, W, 2] for grid_sample
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        # Warp disparity
        warped_disparity = F.grid_sample(
            disparity, 
            warped_grid,
            mode='bilinear', 
            padding_mode=padding_mode, 
            align_corners=True
        )
        
        # Compute valid mask (pixels that came from within the frame)
        mask = torch.ones_like(disparity)
        valid_mask = F.grid_sample(
            mask, 
            warped_grid,
            mode='bilinear', 
            padding_mode='zeros',  # Use zeros for mask to detect out-of-frame
            align_corners=True
        )
        valid_mask = (valid_mask > 0.9999).float()  # Threshold to binary mask
        
        return warped_disparity, valid_mask


def prepare_frame_input(rgb, disparity, flow, use_motion_hints=True):
    """
    Prepare input tensor for warping model.
    
    Args:
        rgb: [B, 3, H, W] - RGB image
        disparity: [B, 1, H, W] - disparity map (GT for frame 0, warped for frame t>0)
        flow: [B, 2, H, W] - optical flow (zero for frame 0, computed for frame t>0)
        use_motion_hints: Whether to include normalized flow as motion hints
    
    Returns:
        input_tensor: [B, 6, H, W] if use_motion_hints else [B, 4, H, W]
    """
    if use_motion_hints:
        # Normalize flow using median
        normalized_flow = normalize_flow_median(flow, percentile=50.0)
        
        # Concatenate: RGB (3) + disparity (1) + normalized flow (2)
        input_tensor = torch.cat([rgb, disparity, normalized_flow], dim=1)
    else:
        # Concatenate: RGB (3) + disparity (1)
        input_tensor = torch.cat([rgb, disparity], dim=1)
    
    return input_tensor
