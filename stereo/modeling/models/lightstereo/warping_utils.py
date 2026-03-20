"""
Warping utilities for temporal stereo matching.
Includes disparity warping and flow normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_flow_fixed(flow, scale=10.0):
    """
    Normalize flow using a fixed scale parameter.
    Preserves absolute motion magnitude information.
    
    Args:
        flow: [B, 2, H, W] - optical flow (flow_x, flow_y)
        scale: Fixed normalization scale (default: 10.0 pixels)
               Typical motion of 10px will normalize to 1.0
    
    Returns:
        normalized_flow: [B, 2, H, W] - normalized by fixed scale
    """
    # Normalize by fixed scale (no clamping to preserve large motions)
    normalized_flow = flow / scale
    
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
        
        # Normalize to [-1, 1] for grid_sample with align_corners=False
        # align_corners=False: grid [-1, 1] maps to pixel centers
        warped_grid[:, 0] = 2.0 * (warped_grid[:, 0] + 0.5) / W - 1.0  # x
        warped_grid[:, 1] = 2.0 * (warped_grid[:, 1] + 0.5) / H - 1.0  # y
        
        # Permute to [B, H, W, 2] for grid_sample
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        # Warp disparity
        warped_disparity = F.grid_sample(
            disparity, 
            warped_grid,
            mode='bilinear', 
            padding_mode=padding_mode, 
            align_corners=False
        )
        
        # Compute valid mask (pixels that came from within the frame)
        mask = torch.ones_like(disparity)
        valid_mask = F.grid_sample(
            mask, 
            warped_grid,
            mode='bilinear', 
            padding_mode='zeros',  # Use zeros for mask to detect out-of-frame
            align_corners=False
        )
        valid_mask = (valid_mask > 0.9999).float()  # Threshold to binary mask
        
        return warped_disparity, valid_mask


def prepare_frame_input(rgb, disparity, flow, use_motion_hints=True, flow_scale=10.0):
    """
    Prepare input tensor for warping model.
    
    Args:
        rgb: [B, 3, H, W] - RGB image
        disparity: [B, 1, H, W] - disparity map (GT for frame 0, warped for frame t>0)
        flow: [B, 2, H, W] - optical flow (zero for frame 0, computed for frame t>0)
        use_motion_hints: Whether to include normalized flow as motion hints
        flow_scale: Fixed scale for flow normalization (default: 10.0 pixels)
    
    Returns:
        input_tensor: [B, 6, H, W] if use_motion_hints else [B, 4, H, W]
    """
    if use_motion_hints:
        # Normalize flow using fixed scale
        normalized_flow = normalize_flow_fixed(flow, scale=flow_scale)
        
        # Concatenate: RGB (3) + disparity (1) + normalized flow (2)
        input_tensor = torch.cat([rgb, disparity, normalized_flow], dim=1)
    else:
        # Concatenate: RGB (3) + disparity (1)
        input_tensor = torch.cat([rgb, disparity], dim=1)
    
    return input_tensor
