"""
Stereo matching metrics: EPE, D1-all, D1-bg, D1-fg
"""

import torch
import logging

logger = logging.getLogger(__name__)


class StereoMetrics:
    """
    Compute stereo matching metrics.
    Reuses metric implementations from stereo.evaluation.
    """
    
    def __init__(self):
        """Initialize metrics calculator"""
        pass
    
    @staticmethod
    def compute_epe(disp_pred: torch.Tensor, disp_gt: torch.Tensor, 
                    valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute End-Point-Error (EPE).
        
        Args:
            disp_pred: Predicted disparity [B, H, W] or [B, 1, H, W]
            disp_gt: Ground truth disparity [B, H, W] or [B, 1, H, W]
            valid_mask: Valid pixel mask [B, H, W] or [B, 1, H, W]
            
        Returns:
            EPE per image [B]
        """
        # Ensure 3D tensors [B, H, W]
        if disp_pred.dim() == 4:
            disp_pred = disp_pred.squeeze(1)
        if disp_gt.dim() == 4:
            disp_gt = disp_gt.squeeze(1)
        if valid_mask.dim() == 4:
            valid_mask = valid_mask.squeeze(1)
        
        # Compute absolute error
        E = torch.abs(disp_gt - disp_pred)
        E_masked = torch.where(valid_mask, E, torch.zeros_like(E))
        
        # Average over valid pixels
        E_sum = E_masked.sum(dim=[1, 2])
        num_valid = valid_mask.sum(dim=[1, 2])
        epe = E_sum / torch.clamp(num_valid, min=1.0)
        
        return epe
    
    @staticmethod
    def compute_d1(disp_pred: torch.Tensor, disp_gt: torch.Tensor,
                   valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute D1-all metric (percentage of bad pixels).
        Bad pixel: error > 3px AND error > 5% of ground truth.
        
        Args:
            disp_pred: Predicted disparity [B, H, W] or [B, 1, H, W]
            disp_gt: Ground truth disparity [B, H, W] or [B, 1, H, W]
            valid_mask: Valid pixel mask [B, H, W] or [B, 1, H, W]
            
        Returns:
            D1-all per image [B] (percentage)
        """
        # Ensure 3D tensors [B, H, W]
        if disp_pred.dim() == 4:
            disp_pred = disp_pred.squeeze(1)
        if disp_gt.dim() == 4:
            disp_gt = disp_gt.squeeze(1)
        if valid_mask.dim() == 4:
            valid_mask = valid_mask.squeeze(1)
        
        # Compute error
        E = torch.abs(disp_gt - disp_pred)
        
        # Bad pixel criterion: (error > 3px) AND (error > 5% of gt)
        err_mask = (E > 3.0) & (E / torch.clamp(torch.abs(disp_gt), min=1e-6) > 0.05)
        err_mask = err_mask & valid_mask
        
        # Compute percentage
        num_errors = err_mask.sum(dim=[1, 2])
        num_valid = valid_mask.sum(dim=[1, 2])
        d1 = num_errors.float() / torch.clamp(num_valid.float(), min=1.0) * 100.0
        
        return d1
    
    @staticmethod
    def compute_d1_bg(disp_pred: torch.Tensor, disp_gt: torch.Tensor,
                      valid_mask: torch.Tensor, threshold: float = 192.0) -> torch.Tensor:
        """
        Compute D1-bg (background) metric.
        Background: disparity < threshold (far from camera).
        
        Args:
            disp_pred: Predicted disparity [B, H, W] or [B, 1, H, W]
            disp_gt: Ground truth disparity [B, H, W] or [B, 1, H, W]
            valid_mask: Valid pixel mask [B, H, W] or [B, 1, H, W]
            threshold: Disparity threshold for background (default: 192)
            
        Returns:
            D1-bg per image [B] (percentage)
        """
        # Ensure 3D tensors
        if disp_pred.dim() == 4:
            disp_pred = disp_pred.squeeze(1)
        if disp_gt.dim() == 4:
            disp_gt = disp_gt.squeeze(1)
        if valid_mask.dim() == 4:
            valid_mask = valid_mask.squeeze(1)
        
        # Background mask: disparity < threshold
        bg_mask = (disp_gt < threshold) & valid_mask
        
        # Compute D1 only on background pixels
        E = torch.abs(disp_gt - disp_pred)
        err_mask = (E > 3.0) & (E / torch.clamp(torch.abs(disp_gt), min=1e-6) > 0.05)
        err_mask = err_mask & bg_mask
        
        num_errors = err_mask.sum(dim=[1, 2])
        num_valid = bg_mask.sum(dim=[1, 2])
        d1_bg = num_errors.float() / torch.clamp(num_valid.float(), min=1.0) * 100.0
        
        # Set to 0 if no background pixels
        d1_bg = torch.where(num_valid > 0, d1_bg, torch.zeros_like(d1_bg))
        
        return d1_bg
    
    @staticmethod
    def compute_d1_fg(disp_pred: torch.Tensor, disp_gt: torch.Tensor,
                      valid_mask: torch.Tensor, threshold: float = 192.0) -> torch.Tensor:
        """
        Compute D1-fg (foreground) metric.
        Foreground: disparity >= threshold (close to camera).
        
        Args:
            disp_pred: Predicted disparity [B, H, W] or [B, 1, H, W]
            disp_gt: Ground truth disparity [B, H, W] or [B, 1, H, W]
            valid_mask: Valid pixel mask [B, H, W] or [B, 1, H, W]
            threshold: Disparity threshold for foreground (default: 192)
            
        Returns:
            D1-fg per image [B] (percentage)
        """
        # Ensure 3D tensors
        if disp_pred.dim() == 4:
            disp_pred = disp_pred.squeeze(1)
        if disp_gt.dim() == 4:
            disp_gt = disp_gt.squeeze(1)
        if valid_mask.dim() == 4:
            valid_mask = valid_mask.squeeze(1)
        
        # Foreground mask: disparity >= threshold
        fg_mask = (disp_gt >= threshold) & valid_mask
        
        # Compute D1 only on foreground pixels
        E = torch.abs(disp_gt - disp_pred)
        err_mask = (E > 3.0) & (E / torch.clamp(torch.abs(disp_gt), min=1e-6) > 0.05)
        err_mask = err_mask & fg_mask
        
        num_errors = err_mask.sum(dim=[1, 2])
        num_valid = fg_mask.sum(dim=[1, 2])
        d1_fg = num_errors.float() / torch.clamp(num_valid.float(), min=1.0) * 100.0
        
        # Set to 0 if no foreground pixels
        d1_fg = torch.where(num_valid > 0, d1_fg, torch.zeros_like(d1_fg))
        
        return d1_fg
    
    def compute_all_metrics(self, disp_pred: torch.Tensor, disp_gt: torch.Tensor,
                           valid_mask: torch.Tensor, max_disp: float = 192.0) -> dict:
        """
        Compute all metrics at once.
        
        Args:
            disp_pred: Predicted disparity [B, H, W] or [B, 1, H, W]
            disp_gt: Ground truth disparity [B, H, W] or [B, 1, H, W]
            valid_mask: Valid pixel mask [B, H, W] or [B, 1, H, W]
            max_disp: Maximum disparity for fg/bg split
            
        Returns:
            Dictionary with all metrics (batch-averaged)
        """
        with torch.no_grad():
            epe = self.compute_epe(disp_pred, disp_gt, valid_mask)
            d1_all = self.compute_d1(disp_pred, disp_gt, valid_mask)
            d1_bg = self.compute_d1_bg(disp_pred, disp_gt, valid_mask, max_disp)
            d1_fg = self.compute_d1_fg(disp_pred, disp_gt, valid_mask, max_disp)
            
            return {
                'epe': epe.mean().item(),
                'd1_all': d1_all.mean().item(),
                'd1_bg': d1_bg.mean().item(),
                'd1_fg': d1_fg.mean().item(),
            }
