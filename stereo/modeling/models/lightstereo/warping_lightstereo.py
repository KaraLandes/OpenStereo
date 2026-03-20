"""
Warping LightStereo for temporal stereo matching with disparity warping.
Extends LightStereo to accept 6-channel input (RGB + warped disparity + flow hints).
Uses the same Backbone class as normal LightStereo, with a dual-path stem adapter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from stereo.modeling.cost_volume.cost_volume import correlation_volume
from stereo.modeling.disp_pred.disp_regression import disparity_regression
from stereo.modeling.disp_refinement.disp_refinement import context_upsample

from .backbone import Backbone, FPNLayer
from .aggregation import Aggregation
from .flow_estimator import LightweightFlowEstimator
from .warping_utils import DisparityWarper, prepare_frame_input


class WarpingLightStereo(nn.Module):
    """
    LightStereo variant for temporal stereo matching with disparity warping.
    
    Uses the same Backbone as normal LightStereo, with an additional dual-path
    stem adapter that handles 6-channel input: RGB (3) + warped disparity (1) + flow (2).
    """
    
    def __init__(self, cfgs):
        super().__init__()
        
        self.max_disp = cfgs.MAX_DISP
        self.left_att = cfgs.LEFT_ATT
        
        # Temporal configuration
        temporal_cfg = cfgs.get('TEMPORAL', {})
        self.use_motion_hints = temporal_cfg.get('use_motion_hints', True)
        self.input_channels = 6 if self.use_motion_hints else 4
        
        # --- Temporal modules (new) ---
        
        # Optical flow estimator
        flow_max_disp = temporal_cfg.get('flow_max_displacement', 4)
        flow_feat_channels = temporal_cfg.get('flow_feature_channels', 16)
        self.flow_estimator = LightweightFlowEstimator(
            max_displacement=flow_max_disp,
            feature_channels=flow_feat_channels
        )
        
        # Disparity warper
        self.disparity_warper = DisparityWarper()
        
        # Dual-path stem adapter for 6-channel input
        temporal_channels = 3 if self.use_motion_hints else 1
        self.temporal_stem = nn.Conv2d(temporal_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.fusion = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        
        # --- Standard LightStereo components (same as normal LS) ---
        
        # Backbone (identical to normal LightStereo)
        self.backbone = Backbone(cfgs.get('BACKBONE', 'MobileNetv2'))
        
        # Cost aggregation
        self.cost_agg = Aggregation(
            in_channels=self.max_disp // 4,
            left_att=self.left_att,
            blocks=cfgs.AGGREGATION_BLOCKS,
            expanse_ratio=cfgs.EXPANSE_RATIO,
            backbone_channels=self.backbone.output_channels
        )
        
        # Disparity refinement
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
    
    def load_pretrained_lightstereo(self, checkpoint_path):
        """
        Load pretrained LightStereo weights.
        Since we use the same Backbone class, most weights load directly.
        Temporal stem is initialized randomly.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                pretrained_state = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                pretrained_state = checkpoint['state_dict']
            else:
                pretrained_state = checkpoint
            
            # Load matching keys (backbone, cost_agg, refine, etc.)
            own_state = self.state_dict()
            loaded_keys = []
            for name, param in pretrained_state.items():
                if name in own_state and own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    loaded_keys.append(name)
            
            logger.info(f"Loaded {len(loaded_keys)}/{len(pretrained_state)} pretrained weights")
            logger.info("Temporal stem and fusion layers initialized randomly")
            
        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")
            logger.warning("Continuing with random initialization")
    
    def extract_features(self, input_tensor):
        """
        Extract features from 6-channel input using dual-path stem + standard backbone.
        
        Args:
            input_tensor: [B, 6, H, W] - RGB (3) + disparity (1) + flow (2)
        
        Returns:
            features: List of feature maps at different scales
        """
        # Split into RGB and temporal channels
        rgb = input_tensor[:, :3, :, :]
        temporal = input_tensor[:, 3:, :, :]
        
        # Dual-path stem: RGB through backbone's conv_stem, temporal through ours
        rgb_features = self.backbone.conv_stem(rgb)           # [B, 32, H/2, W/2]
        temporal_features = self.temporal_stem(temporal)       # [B, 32, H/2, W/2]
        
        # Fuse
        fused = self.fusion(torch.cat([rgb_features, temporal_features], dim=1))  # [B, 32, H/2, W/2]
        
        # Continue through backbone (bn1, act1, blocks, FPN)
        c1 = self.backbone.act1(self.backbone.bn1(fused))
        c1 = self.backbone.block0(c1)
        c2 = self.backbone.block1(c1)
        c3 = self.backbone.block2(c2)
        c4 = self.backbone.block3(c3)
        c5 = self.backbone.block4(c4)
        
        p4 = self.backbone.fpn_layer4(c5, c4)
        p3 = self.backbone.fpn_layer3(p4, c3)
        p2 = self.backbone.fpn_layer2(p3, c2)
        p2 = self.backbone.out_conv(p2)
        
        return [p2, p3, p4, c5]
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: Dictionary with:
                - 'left': [B, 6, H, W] - left image with disparity and flow
                - 'right': [B, 6, H, W] - right image with disparity and flow
        """
        left_input = data['left']
        right_input = data['right']
        left_rgb = left_input[:, :3, :, :]
        
        features_left = self.extract_features(left_input)
        features_right = self.extract_features(right_input)
        
        gwc_volume = correlation_volume(features_left[0], features_right[0], self.max_disp // 4)
        
        encoding_volume = self.cost_agg(gwc_volume, features_left)
        squeezed_encoding = encoding_volume[0].reshape(
            encoding_volume[0].size(0), -1,
            encoding_volume[0].size(2), encoding_volume[0].size(3)
        )
        
        prob = F.softmax(squeezed_encoding, dim=1)
        init_disp = disparity_regression(prob, self.max_disp // 4)
        
        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, self.stem_2(left_rgb))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)
        disp_pred = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)
        
        result = {'disp_pred': disp_pred}
        
        if self.training:
            disp_4 = F.interpolate(init_disp, left_rgb.shape[2:], mode='bilinear', align_corners=False)
            disp_4 *= 4
            result['disp_4'] = disp_4
        
        return result
