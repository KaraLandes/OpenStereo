"""
Warping LightStereo for temporal stereo matching with disparity warping.
Extends LightStereo to accept 6-channel input (RGB + warped disparity + flow hints).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_estimator import LightweightFlowEstimator
from .warping_utils import DisparityWarper, prepare_frame_input


class WarpingLightStereo(nn.Module):
    """
    LightStereo variant for temporal stereo matching with disparity warping.
    
    Key differences from base LightStereo:
    - Accepts 6-channel input: RGB (3) + warped disparity (1) + flow hints (2)
    - Integrates optical flow estimator
    - Includes disparity warping module
    """
    
    def __init__(self, cfgs):
        """
        Initialize warping LightStereo.
        
        Args:
            cfgs: Configuration object with:
                - MAX_DISP: Maximum disparity
                - LEFT_ATT: Use left attention
                - AGGREGATION_BLOCKS: Aggregation block counts
                - EXPANSE_RATIO: Channel expansion ratio
                - BACKCONE: Backbone name
                - TEMPORAL: Temporal config dict
                    - use_motion_hints: Whether to use flow as input (default: True)
                    - flow_max_displacement: Max displacement for flow (default: 4)
                    - flow_feature_channels: Feature channels for flow (default: 16)
        """
        super().__init__()
        
        self.max_disp = cfgs.MAX_DISP
        self.left_att = cfgs.LEFT_ATT
        
        # Temporal configuration
        temporal_cfg = cfgs.get('TEMPORAL', {})
        self.use_motion_hints = temporal_cfg.get('use_motion_hints', True)
        
        # Determine input channels
        self.input_channels = 6 if self.use_motion_hints else 4
        
        # Optical flow estimator
        flow_max_disp = temporal_cfg.get('flow_max_displacement', 4)
        flow_feat_channels = temporal_cfg.get('flow_feature_channels', 16)
        self.flow_estimator = LightweightFlowEstimator(
            max_displacement=flow_max_disp,
            feature_channels=flow_feat_channels
        )
        
        # Disparity warper
        self.disparity_warper = DisparityWarper()
        
        # Modified backbone for 6-channel input
        from .aggregation import Aggregation
        from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
        
        # DUAL-PATH ADAPTER: Separate stems for RGB and temporal channels
        # RGB path - will use pretrained weights
        self.rgb_stem = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Temporal path - learns from scratch (disparity + flow)
        temporal_channels = 3 if self.use_motion_hints else 1  # 1 (disp) + 2 (flow) or just 1 (disp)
        self.temporal_stem = nn.Conv2d(temporal_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Fusion layer to combine RGB and temporal features
        self.fusion = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.SiLU(inplace=True)
        
        # Load pretrained backbone
        import timm
        backbone_name = cfgs.get('BACKBONE', 'MobileNetv2')
        
        if backbone_name == 'MobileNetv2':
            model = timm.create_model('mobilenetv2_100', pretrained=True, features_only=False)
            channels = [160, 96, 32, 24]
        elif backbone_name == 'EfficientNetv2':
            model = timm.create_model('efficientnetv2_rw_s', pretrained=True, features_only=False)
            channels = [272, 160, 64, 48]
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not supported")
        
        # Copy blocks from pretrained model
        self.block0 = model.blocks[0]
        self.block1 = model.blocks[1]
        self.block2 = model.blocks[2]
        self.block3 = model.blocks[3:5]
        self.block4 = model.blocks[5]
        
        # FPN layers
        from .backbone import FPNLayer
        self.fpn_layer4 = FPNLayer(channels[0], channels[1])
        self.fpn_layer3 = FPNLayer(channels[1], channels[2])
        self.fpn_layer2 = FPNLayer(channels[2], channels[3])
        
        self.out_conv = BasicConv2d(channels[3], channels[3],
                                    kernel_size=3, padding=1, padding_mode="replicate",
                                    norm_layer=nn.InstanceNorm2d)
        
        self.output_channels = channels[::-1]
        
        # Cost aggregation (same as base LightStereo)
        self.cost_agg = Aggregation(
            in_channels=48,
            left_att=self.left_att,
            blocks=cfgs.AGGREGATION_BLOCKS,
            expanse_ratio=cfgs.EXPANSE_RATIO,
            backbone_channels=self.output_channels
        )
        
        # Disparity refinement (same as LightStereo)
        self.refine_1 = nn.Sequential(
            BasicConv2d(self.output_channels[0], 24, kernel_size=3, stride=1, padding=1,
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
        Load pretrained LightStereo weights into the dual-path architecture.
        
        Args:
            checkpoint_path: Path to pretrained LightStereo checkpoint
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Load pretrained checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract state dict (handle different checkpoint formats)
            if 'model_state_dict' in checkpoint:
                pretrained_state = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                pretrained_state = checkpoint['state_dict']
            else:
                pretrained_state = checkpoint
            
            # Load RGB stem weights from pretrained backbone.conv_stem or first conv
            rgb_stem_keys = ['backbone.conv_stem.weight', 'conv_stem.weight', 'stem.0.weight']
            loaded_rgb_stem = False
            for key in rgb_stem_keys:
                if key in pretrained_state:
                    self.rgb_stem.weight.data = pretrained_state[key].clone()
                    logger.info(f"Loaded RGB stem weights from '{key}'")
                    loaded_rgb_stem = True
                    break
            
            if not loaded_rgb_stem:
                logger.warning("Could not find RGB stem weights in checkpoint, using random initialization")
            
            # Load backbone blocks (block0-block4)
            for i in range(5):
                block_name = f'block{i}' if i < 3 else 'block3' if i == 3 else 'block4'
                target_block = getattr(self, block_name)
                
                # Try to load from pretrained
                block_state = {}
                for k, v in pretrained_state.items():
                    if f'backbone.{block_name}' in k or f'{block_name}' in k:
                        new_key = k.replace(f'backbone.{block_name}.', '').replace(f'{block_name}.', '')
                        block_state[new_key] = v
                
                if block_state:
                    target_block.load_state_dict(block_state, strict=False)
                    logger.info(f"Loaded {block_name} weights")
            
            # Load FPN layers
            for fpn_name in ['fpn_layer4', 'fpn_layer3', 'fpn_layer2']:
                fpn_state = {}
                for k, v in pretrained_state.items():
                    if fpn_name in k:
                        new_key = k.replace(f'{fpn_name}.', '')
                        fpn_state[new_key] = v
                
                if fpn_state:
                    getattr(self, fpn_name).load_state_dict(fpn_state, strict=False)
                    logger.info(f"Loaded {fpn_name} weights")
            
            # Load cost aggregation
            cost_agg_state = {}
            for k, v in pretrained_state.items():
                if 'cost_agg' in k:
                    new_key = k.replace('cost_agg.', '')
                    cost_agg_state[new_key] = v
            
            if cost_agg_state:
                self.cost_agg.load_state_dict(cost_agg_state, strict=False)
                logger.info("Loaded cost aggregation weights")
            
            # Load refinement layers
            for refine_name in ['refine_1', 'refine_2', 'refine_3', 'stem_2', 'out_conv']:
                if hasattr(self, refine_name):
                    refine_state = {}
                    for k, v in pretrained_state.items():
                        if refine_name in k:
                            new_key = k.replace(f'{refine_name}.', '')
                            refine_state[new_key] = v
                    
                    if refine_state:
                        getattr(self, refine_name).load_state_dict(refine_state, strict=False)
                        logger.info(f"Loaded {refine_name} weights")
            
            logger.info(f"Successfully loaded pretrained LightStereo weights from {checkpoint_path}")
            logger.info("Temporal stem initialized randomly and will learn from scratch")
            
        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")
            logger.warning("Continuing with random initialization")
    
    def extract_features(self, input_tensor):
        """
        Extract features from 6-channel input using dual-path adapter.
        
        Args:
            input_tensor: [B, 6, H, W] - RGB (3) + disparity (1) + flow (2)
        
        Returns:
            features: List of feature maps at different scales
        """
        # Split input into RGB and temporal channels
        rgb = input_tensor[:, :3, :, :]  # [B, 3, H, W]
        temporal = input_tensor[:, 3:, :, :]  # [B, 3, H, W] (disp + flow) or [B, 1, H, W] (disp only)
        
        # Process RGB path (pretrained)
        rgb_features = self.rgb_stem(rgb)  # [B, 32, H/2, W/2]
        
        # Process temporal path (learns from scratch)
        temporal_features = self.temporal_stem(temporal)  # [B, 32, H/2, W/2]
        
        # Fuse both paths
        combined = torch.cat([rgb_features, temporal_features], dim=1)  # [B, 64, H/2, W/2]
        fused = self.fusion(combined)  # [B, 32, H/2, W/2]
        
        # Apply batch norm and activation
        c1 = self.act1(self.bn1(fused))  # [B, 32, H/2, W/2]
        
        # Continue with pretrained backbone blocks
        c1 = self.block0(c1)  # [B, 16, H/2, W/2]
        c2 = self.block1(c1)  # [B, 24, H/4, W/4]
        c3 = self.block2(c2)  # [B, 32, H/8, W/8]
        c4 = self.block3(c3)  # [B, 96, H/16, W/16]
        c5 = self.block4(c4)  # [B, 160, H/32, W/32]
        
        # FPN
        p4 = self.fpn_layer4(c5, c4)  # [B, 96, H/16, W/16]
        p3 = self.fpn_layer3(p4, c3)  # [B, 32, H/8, W/8]
        p2 = self.fpn_layer2(p3, c2)  # [B, 24, H/4, W/4]
        p2 = self.out_conv(p2)
        
        return [p2, p3, p4, c5]
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: Dictionary with:
                - 'left': [B, 6, H, W] - left image with disparity and flow
                - 'right': [B, 6, H, W] - right image with disparity and flow
        
        Returns:
            Dictionary with:
                - 'disp_pred': [B, 1, H, W] - predicted disparity
                - 'disp_4': [B, 1, H, W] - intermediate disparity (training only)
        """
        from stereo.modeling.cost_volume.cost_volume import correlation_volume
        from stereo.modeling.disp_pred.disp_regression import disparity_regression
        from stereo.modeling.disp_refinement.disp_refinement import context_upsample
        
        # Input is already 6-channel (prepared by trainer)
        left_input = data['left']
        right_input = data['right']
        
        # Extract RGB for refinement (first 3 channels)
        left_rgb = left_input[:, :3, :, :]
        
        # Extract features from both images
        features_left = self.extract_features(left_input)
        features_right = self.extract_features(right_input)
        
        # Build cost volume
        gwc_volume = correlation_volume(features_left[0], features_right[0], self.max_disp // 4)
        
        # Cost aggregation
        encoding_volume = self.cost_agg(gwc_volume, features_left)
        squeezed_encoding = encoding_volume[0].reshape(
            encoding_volume[0].size(0), -1,
            encoding_volume[0].size(2), encoding_volume[0].size(3)
        )
        
        # Disparity regression
        prob = F.softmax(squeezed_encoding, dim=1)
        init_disp = disparity_regression(prob, self.max_disp // 4)
        
        # Disparity refinement (uses RGB only)
        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, self.stem_2(left_rgb))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)
        disp_pred = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)
        
        result = {
            'disp_pred': disp_pred
        }
        
        if self.training:
            disp_4 = F.interpolate(init_disp, left_rgb.shape[2:], mode='bilinear', align_corners=False)
            disp_4 *= 4
            result['disp_4'] = disp_4
        
        return result
