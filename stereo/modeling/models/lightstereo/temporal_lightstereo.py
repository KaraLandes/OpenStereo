"""
Temporal LightStereo with state fusion for video stereo matching.
Implements Option 2: Temporal State Fusion with GRU-based memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lightstereo import LightStereo


class DisparityFusion(nn.Module):
    """
    Disparity-based temporal fusion module.
    Fuses previous disparity prediction with current cost volume.
    """
    
    def __init__(self, max_disp: int = 192):
        """
        Initialize disparity fusion module.
        
        Args:
            max_disp: Maximum disparity value
        """
        super().__init__()
        self.max_disp = max_disp
        
        # Learnable fusion weights
        # Input: concatenated [cost_volume, prev_disp_features]
        # Output: fused cost volume
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(48 + 16, 48, 3, 1, 1),  # 48 from cost vol, 16 from disp features
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.BatchNorm2d(48)
        )
        
        # Encode previous disparity into features
        self.disp_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16)
        )
    
    def forward(self, cost_volume: torch.Tensor, prev_disp: torch.Tensor) -> torch.Tensor:
        """
        Fuse cost volume with previous disparity.
        
        Args:
            cost_volume: Current cost volume [B, 48, H/4, W/4]
            prev_disp: Previous disparity prediction [B, 1, H, W]
            
        Returns:
            Fused cost volume [B, 48, H/4, W/4]
        """
        # Downsample previous disparity to match cost volume resolution
        prev_disp_small = F.interpolate(
            prev_disp, 
            size=cost_volume.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        # Scale disparity to match resolution
        prev_disp_small = prev_disp_small / 4.0
        
        # Encode previous disparity into features
        disp_features = self.disp_encoder(prev_disp_small)
        
        # Concatenate and fuse
        combined = torch.cat([cost_volume, disp_features], dim=1)
        fused_volume = self.fusion_conv(combined)
        
        # Residual connection
        fused_volume = fused_volume + cost_volume
        
        return fused_volume


class TemporalLightStereo(nn.Module):
    """
    Temporal extension of LightStereo with state fusion.
    
    Wraps base LightStereo and adds temporal fusion between frames.
    """
    
    def __init__(self, cfgs):
        """
        Initialize temporal LightStereo.
        
        Args:
            cfgs: Configuration object (same as LightStereo)
                Additional temporal configs:
                - TEMPORAL_HIDDEN_CHANNELS: Hidden state channels (default: 64)
                - TEMPORAL_USE_FLOW: Whether to use optical flow warping (default: False)
        """
        super().__init__()
        
        self.max_disp = cfgs.MAX_DISP
        self.left_att = cfgs.LEFT_ATT
        
        # Base LightStereo components (reuse architecture)
        from .backbone import Backbone, FPNLayer
        from .aggregation import Aggregation
        from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
        
        # Backbone (same as LightStereo)
        self.backbone = Backbone(cfgs.get('BACKCONE', 'MobileNetv2'))
        
        # Disparity-based temporal fusion (NEW)
        self.disparity_fusion = DisparityFusion(max_disp=self.max_disp)
        self.use_temporal_fusion = cfgs.get('USE_TEMPORAL_FUSION', True)
        
        # Cost aggregation (same as base LightStereo - expects 48 channels from correlation)
        self.cost_agg = Aggregation(
            in_channels=48,  # Correlation volume channels (24 * 2 = 48 for concat, or just disparity levels)
            left_att=self.left_att,
            blocks=cfgs.AGGREGATION_BLOCKS,
            expanse_ratio=cfgs.EXPANSE_RATIO,
            backbone_channels=self.backbone.output_channels
        )
        
        # Disparity refinement (same as LightStereo)
        self.refine_1 = nn.Sequential(
            BasicConv2d(self.backbone.output_channels[0], 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU))
        
        self.stem_2 = nn.Sequential(
            BasicConv2d(3, 16, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.refine_2 = FPNLayer(24, 16)
        
        self.refine_3 = BasicDeconv2d(16, 9, kernel_size=4, stride=2, padding=1)
        
        # Optional: Optical flow warping
        self.use_flow = cfgs.get('TEMPORAL_USE_FLOW', False)
        if self.use_flow:
            # Placeholder for lightweight flow network
            # Can be implemented later if needed
            pass
    
    def initialize_disparity(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """
        Initialize disparity for first frame (will be replaced by FoundationStereo in trainer).
        
        Args:
            batch_size: Number of sequences in batch
            height: Image height
            width: Image width
            device: Device to create tensor on
            
        Returns:
            Zero disparity [batch_size, 1, height, width]
        """
        return torch.zeros(batch_size, 1, height, width, device=device)
    
    def forward(self, data: dict, prev_disparity: torch.Tensor = None) -> dict:
        """
        Forward pass with disparity-based temporal fusion.
        
        Args:
            data: Dictionary with 'left' and 'right' images [B, 3, H, W]
            prev_disparity: Previous disparity prediction [B, 1, H, W] (optional)
            
        Returns:
            Dictionary with:
            - 'disp_pred': Predicted disparity [B, 1, H, W]
            - 'disp_4': Intermediate disparity at 1/4 resolution (training only)
        """
        from stereo.modeling.cost_volume.cost_volume import correlation_volume
        from stereo.modeling.disp_pred.disp_regression import disparity_regression
        from stereo.modeling.disp_refinement.disp_refinement import context_upsample
        
        image1 = data['left']
        image2 = data['right']
        
        # Extract features from both images
        features_left = self.backbone(image1)
        features_right = self.backbone(image2)
        
        # Build cost volume
        gwc_volume = correlation_volume(features_left[0], features_right[0], self.max_disp // 4)
        
        # Fuse with previous disparity if available
        if self.use_temporal_fusion and prev_disparity is not None:
            gwc_volume = self.disparity_fusion(gwc_volume, prev_disparity)
        else:
            # First frame: use cost volume as-is
            pass
        
        # Cost aggregation
        encoding_volume = self.cost_agg(gwc_volume, features_left)
        squeezed_encoding = encoding_volume[0].reshape(
            encoding_volume[0].size(0), -1, 
            encoding_volume[0].size(2), encoding_volume[0].size(3)
        )
        
        # Disparity regression
        prob = F.softmax(squeezed_encoding, dim=1)
        init_disp = disparity_regression(prob, self.max_disp // 4)
        
        # Disparity refinement
        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, self.stem_2(image1))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)
        disp_pred = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)
        
        result = {
            'disp_pred': disp_pred
        }
        
        if self.training:
            disp_4 = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=False)
            disp_4 *= 4
            result['disp_4'] = disp_4
        
        return result
    
    def _warp_features(self, features: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp features using optical flow.
        
        Args:
            features: Features to warp [B, C, H, W]
            flow: Optical flow [B, 2, H, W]
            
        Returns:
            Warped features [B, C, H, W]
        """
        B, C, H, W = features.shape
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=features.device),
            torch.arange(W, device=features.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Add flow to grid
        warped_grid = grid + flow
        
        # Normalize to [-1, 1]
        warped_grid[:, 0] = 2.0 * warped_grid[:, 0] / (W - 1) - 1.0
        warped_grid[:, 1] = 2.0 * warped_grid[:, 1] / (H - 1) - 1.0
        
        # Permute to [B, H, W, 2]
        warped_grid = warped_grid.permute(0, 2, 3, 1)
        
        # Sample features
        warped_features = F.grid_sample(
            features, warped_grid, 
            mode='bilinear', padding_mode='border', align_corners=True
        )
        
        return warped_features
    
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
