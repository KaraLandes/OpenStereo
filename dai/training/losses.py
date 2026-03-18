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
        
        # Extended loss component weights (0.0 = disabled)
        self.gradient_weight = self.loss_weights.get('gradient', 0.0)
        self.edge_smooth_weight = self.loss_weights.get('edge_smooth', 0.0)
        self.second_order_smooth_weight = self.loss_weights.get('second_order_smooth', 0.0)
        
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
        
        # Guard against empty mask (prevents NaN from mean() on empty tensors)
        if mask.sum() == 0:
            logger.warning("Empty loss mask detected - skipping batch")
            return torch.tensor(0.0, device=disp_gt.device, requires_grad=True), {
                'loss_total': 0.0,
                'loss_disp_pred': 0.0,
                'loss_disp_4': 0.0
            }
        
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
        
        # Penalty for high disparity in invalid regions (sky, etc.)
        invalid_reg_weight = self.loss_weights.get('invalid_region_penalty', 0.0)
        if invalid_reg_weight > 0 and 'disp_pred' in model_output:
            invalid_mask = ~valid_mask
            if invalid_mask.any():
                disp_pred = model_output['disp_pred']
                invalid_disp = disp_pred[invalid_mask]
                # Penalize predictions above threshold (default 5px)
                threshold = self.loss_weights.get('invalid_region_threshold', 5.0)
                penalty = F.relu(invalid_disp - threshold).mean()
                total_loss = total_loss + invalid_reg_weight * penalty
                loss_dict['loss_invalid_reg'] = (invalid_reg_weight * penalty).item()
        
        # --- Extended loss components (applied to disp_pred only) ---
        if 'disp_pred' in model_output:
            disp_pred = model_output['disp_pred']  # [B, 1, H, W]
            
            # Gradient matching loss
            if self.gradient_weight > 0:
                grad_loss = self._gradient_loss(disp_pred, disp_gt, mask)
                total_loss = total_loss + self.gradient_weight * grad_loss
                loss_dict['loss_gradient'] = (self.gradient_weight * grad_loss).item()
            
            # Edge-aware first-order smoothness
            if self.edge_smooth_weight > 0 and 'left' in batch:
                smooth_loss = self._edge_aware_smoothness(disp_pred, batch['left'])
                total_loss = total_loss + self.edge_smooth_weight * smooth_loss
                loss_dict['loss_edge_smooth'] = (self.edge_smooth_weight * smooth_loss).item()
            
            # Edge-aware second-order smoothness
            if self.second_order_smooth_weight > 0 and 'left' in batch:
                smooth2_loss = self._second_order_smoothness(disp_pred, batch['left'])
                total_loss = total_loss + self.second_order_smooth_weight * smooth2_loss
                loss_dict['loss_second_order_smooth'] = (self.second_order_smooth_weight * smooth2_loss).item()
        
        loss_dict['loss_total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    # --- Gradient and smoothness helpers ---
    
    @staticmethod
    def _grad_x(t):
        """Horizontal finite difference: [B, C, H, W] -> [B, C, H, W-1]"""
        return t[:, :, :, 1:] - t[:, :, :, :-1]
    
    @staticmethod
    def _grad_y(t):
        """Vertical finite difference: [B, C, H, W] -> [B, C, H-1, W]"""
        return t[:, :, 1:, :] - t[:, :, :-1, :]
    
    @staticmethod
    def _image_grad_mag_x(image):
        """Mean absolute horizontal gradient across channels: [B, C, H, W] -> [B, 1, H, W-1]"""
        gx = image[:, :, :, 1:] - image[:, :, :, :-1]
        return gx.abs().mean(dim=1, keepdim=True)
    
    @staticmethod
    def _image_grad_mag_y(image):
        """Mean absolute vertical gradient across channels: [B, C, H, W] -> [B, 1, H-1, W]"""
        gy = image[:, :, 1:, :] - image[:, :, :-1, :]
        return gy.abs().mean(dim=1, keepdim=True)
    
    def _gradient_loss(self, pred, gt, mask):
        """
        Gradient matching loss: L1 on horizontal/vertical disparity gradients.
        
        Args:
            pred: [B, 1, H, W] predicted disparity
            gt: [B, 1, H, W] ground-truth disparity
            mask: [B, 1, H, W] valid pixel mask
        
        Returns:
            Scalar loss tensor
        """
        pred_dx = self._grad_x(pred)
        pred_dy = self._grad_y(pred)
        gt_dx = self._grad_x(gt)
        gt_dy = self._grad_y(gt)
        
        # Both neighboring pixels must be valid
        mask_x = mask[:, :, :, 1:] & mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] & mask[:, :, :-1, :]
        
        loss_x = torch.abs(pred_dx - gt_dx)
        loss_y = torch.abs(pred_dy - gt_dy)
        
        num_valid_x = mask_x.sum().clamp(min=1)
        num_valid_y = mask_y.sum().clamp(min=1)
        
        return (loss_x[mask_x].sum() / num_valid_x) + (loss_y[mask_y].sum() / num_valid_y)
    
    def _edge_aware_smoothness(self, pred, image):
        """
        Edge-aware first-order smoothness loss.
        Penalizes disparity gradients, weighted down near image edges.
        
        Args:
            pred: [B, 1, H, W] predicted disparity
            image: [B, C, H, W] left image (normalized)
        
        Returns:
            Scalar loss tensor
        """
        pred_dx = self._grad_x(pred)
        pred_dy = self._grad_y(pred)
        
        weight_x = torch.exp(-self._image_grad_mag_x(image))
        weight_y = torch.exp(-self._image_grad_mag_y(image))
        
        loss_x = (torch.abs(pred_dx) * weight_x).mean()
        loss_y = (torch.abs(pred_dy) * weight_y).mean()
        
        return loss_x + loss_y
    
    def _second_order_smoothness(self, pred, image):
        """
        Edge-aware second-order smoothness loss.
        Penalizes curvature (change of slope), weighted down near image edges.
        Encourages planar surfaces.
        
        Args:
            pred: [B, 1, H, W] predicted disparity
            image: [B, C, H, W] left image (normalized)
        
        Returns:
            Scalar loss tensor
        """
        pred_dx = self._grad_x(pred)
        pred_dy = self._grad_y(pred)
        pred_dxx = self._grad_x(pred_dx)  # [B, 1, H, W-2]
        pred_dyy = self._grad_y(pred_dy)  # [B, 1, H-2, W]
        
        img_grad_x = self._image_grad_mag_x(image)  # [B, 1, H, W-1]
        img_grad_y = self._image_grad_mag_y(image)  # [B, 1, H-1, W]
        
        # Slice image gradients to match second-derivative sizes
        weight_x = torch.exp(-img_grad_x[:, :, :, :-1])  # [B, 1, H, W-2]
        weight_y = torch.exp(-img_grad_y[:, :, :-1, :])  # [B, 1, H-2, W]
        
        loss_x = (torch.abs(pred_dxx) * weight_x).mean()
        loss_y = (torch.abs(pred_dyy) * weight_y).mean()
        
        return loss_x + loss_y
    
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
