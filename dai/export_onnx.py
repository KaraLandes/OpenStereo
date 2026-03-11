"""
ONNX Export Utility for LightStereo

Exports PyTorch checkpoint to ONNX format with numerical validation.
Uses ONNX-friendly correlation volume (no loops) for clean export.

Usage:
    python dai/export_onnx.py --checkpoint <path_to_checkpoint.pth> --output <output.onnx>
"""

import argparse
import os
import sys
from pathlib import Path

# Force legacy ONNX exporter (TorchScript-based) instead of new dynamo-based one
os.environ['TORCH_ONNX_USE_NEW_EXPORTER'] = '0'

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

from easydict import EasyDict


class LightStereoONNX(nn.Module):
    """
    ONNX-exportable wrapper for LightStereo.
    
    Changes from original:
    - Accepts (left, right) tensors instead of dict
    - Uses correlation_volume_onnx (no loops) instead of correlation_volume
    - Returns single disparity tensor instead of dict
    """
    
    def __init__(self, base_model):
        super().__init__()
        self.max_disp = base_model.max_disp
        self.left_att = base_model.left_att
        self.backbone = base_model.backbone
        self.cost_agg = base_model.cost_agg
        self.refine_1 = base_model.refine_1
        self.stem_2 = base_model.stem_2
        self.refine_2 = base_model.refine_2
        self.refine_3 = base_model.refine_3
    
    def correlation_volume_onnx(self, left_feature, right_feature, max_disp):
        """ONNX-friendly correlation volume without unfold (uses explicit slicing)."""
        B, C, H, W = left_feature.shape
        
        # Pad right feature on the left side with zeros
        right_padded = F.pad(right_feature, (max_disp - 1, 0), mode='constant', value=0)
        
        # Build cost volume by explicit slicing (avoids unfold)
        # For disparity d: correlate left with right shifted by d
        cost_slices = []
        for d in range(max_disp):
            # Get shifted right feature for this disparity
            # disparity 0 -> right_padded[:, :, :, max_disp-1 : max_disp-1+W]
            # disparity d -> right_padded[:, :, :, max_disp-1-d : max_disp-1-d+W]
            start_idx = max_disp - 1 - d
            right_shifted = right_padded[:, :, :, start_idx:start_idx + W]
            
            # Correlation: element-wise multiply and mean over channels
            corr = (left_feature * right_shifted).mean(dim=1, keepdim=True)
            cost_slices.append(corr)
        
        # Stack along disparity dimension: [B, max_disp, H, W]
        cost_volume = torch.cat(cost_slices, dim=1)
        
        return cost_volume.contiguous()
    
    def context_upsample_onnx(self, disp_low, up_weights, scale_factor=4):
        """Context-aware upsampling without F.unfold (ONNX compatible)."""
        b, c, h, w = disp_low.shape
        
        # Pad input for 3x3 window extraction (replicate padding)
        disp_padded = F.pad(disp_low, (1, 1, 1, 1), mode='replicate')
        
        # Extract 3x3 patches manually using slicing (avoids F.unfold)
        # This creates 9 shifted versions of the input
        patches = []
        for dy in range(3):
            for dx in range(3):
                patches.append(disp_padded[:, :, dy:dy+h, dx:dx+w])
        
        # Stack patches: [b, 9, h, w] (since c=1 for disparity)
        disp_unfold = torch.cat(patches, dim=1)
        
        # Upsample to full resolution
        disp_unfold = F.interpolate(disp_unfold, (h * scale_factor, w * scale_factor), mode='nearest')
        
        # Apply learned upsampling weights and sum
        disp = (disp_unfold * up_weights).sum(1)
        
        return disp
    
    def forward(self, left, right):
        """
        Forward pass for ONNX export.
        
        Args:
            left: [B, 3, H, W] left image tensor
            right: [B, 3, H, W] right image tensor
            
        Returns:
            disp_pred: [B, 1, H, W] disparity prediction
        """
        # Backbone feature extraction
        features_left = self.backbone(left)
        features_right = self.backbone(right)
        
        # Cost volume with ONNX-friendly implementation
        gwc_volume = self.correlation_volume_onnx(
            features_left[0], features_right[0], self.max_disp // 4
        )
        
        # Cost aggregation
        encoding_volume = self.cost_agg(gwc_volume, features_left)
        squeezed_encoding = encoding_volume[0].reshape(
            encoding_volume[0].size(0), -1, 
            encoding_volume[0].size(2), encoding_volume[0].size(3)
        )
        
        # Disparity regression
        prob = F.softmax(squeezed_encoding, dim=1)
        disp_values = torch.arange(0, self.max_disp // 4, dtype=prob.dtype, device=prob.device)
        disp_values = disp_values.view(1, self.max_disp // 4, 1, 1)
        init_disp = torch.sum(prob * disp_values, 1, keepdim=True)
        
        # Refinement
        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, self.stem_2(left))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)
        
        # Context upsample
        disp_pred = self.context_upsample_onnx(init_disp * 4., spx_pred.float())
        disp_pred = disp_pred.unsqueeze(1)
        
        return disp_pred


def load_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """
    Load LightStereo model from DAI checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: Device to load model on
        
    Returns:
        model: LightStereo model with loaded weights
        config: Model configuration from checkpoint
    """
    from stereo.modeling.models.lightstereo.lightstereo import LightStereo
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    # Build model config
    model_cfg = EasyDict({
        'MAX_DISP': model_config.get('max_disp', 192),
        'LEFT_ATT': model_config.get('left_att', True),
        'AGGREGATION_BLOCKS': model_config.get('aggregation_blocks', [1, 2, 4]),
        'EXPANSE_RATIO': model_config.get('expanse_ratio', 4),
        'BACKCONE': model_config.get('backbone', 'MobileNetv2'),
    })
    
    print(f"Model config: MAX_DISP={model_cfg.MAX_DISP}, BACKBONE={model_cfg.BACKCONE}")
    
    # Create model and load weights
    model = LightStereo(model_cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, model_cfg


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    height: int = 256,
    width: int = 512,
    opset_version: int = 17,
    simplify: bool = True,
    device: str = 'cpu'
):
    """
    Export LightStereo model to ONNX format.
    
    Args:
        model: LightStereo model (will be wrapped for ONNX)
        output_path: Path for output .onnx file
        height: Input image height
        width: Input image width
        opset_version: ONNX opset version (minimum 17 for PyTorch 2.x)
        simplify: Whether to simplify the ONNX model
        device: Device to use for export
    """
    import onnx
    
    # Wrap model for ONNX export
    onnx_model = LightStereoONNX(model)
    onnx_model.eval()
    onnx_model.to(device)
    
    # Create dummy inputs
    left = torch.randn(1, 3, height, width, device=device)
    right = torch.randn(1, 3, height, width, device=device)
    
    print(f"Exporting to ONNX with input shape: [1, 3, {height}, {width}]")
    print(f"Opset version: {opset_version}")
    
    # Use legacy TorchScript-based exporter (dynamo=False)
    # The new dynamo-based exporter has compatibility issues
    torch.onnx.export(
        onnx_model,
        (left, right),
        output_path,
        input_names=['left', 'right'],
        output_names=['disparity'],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes=None,  # Fixed shape for TensorRT compatibility
        verbose=False,
        export_params=True,
        dynamo=False  # Use legacy TorchScript exporter
    )
    
    # Verify export
    onnx_model_loaded = onnx.load(output_path)
    onnx.checker.check_model(onnx_model_loaded)
    print(f"✓ ONNX model saved to: {output_path}")
    
    # Simplify if requested
    if simplify:
        try:
            import onnxsim
            print("Simplifying ONNX model...")
            model_simplified, check = onnxsim.simplify(onnx_model_loaded)
            if check:
                onnx.save(model_simplified, output_path)
                print("✓ ONNX model simplified")
            else:
                print("⚠ Simplification check failed, keeping original")
        except ImportError:
            print("⚠ onnx-simplifier not installed, skipping simplification")
        except Exception as e:
            print(f"⚠ Simplification failed: {e}")
    
    return onnx_model


def load_image(path: str, height: int = None, width: int = None, device: str = 'cpu') -> tuple:
    """
    Load and preprocess image for inference with aspect-ratio preserving resize.
    
    Strategy:
    - If original dimension < target: resize (squeeze) to target
    - If original dimension >= target: pad to target
    
    Args:
        path: Path to image file
        height: Target height (None = use original, will be padded to multiple of 32)
        width: Target width (None = use original, will be padded to multiple of 32)
        device: Device to load tensor on
        
    Returns:
        tuple: (img_tensor, pad_info)
            - img_tensor: Preprocessed image [1, 3, H, W]
            - pad_info: Dict with padding info for cropping output
    """
    img = Image.open(path).convert('RGB')
    orig_w, orig_h = img.size
    
    # Use original size if not specified
    if height is None or width is None:
        # Pad to multiple of 32 for network compatibility
        height = ((orig_h + 31) // 32) * 32
        width = ((orig_w + 31) // 32) * 32
    
    # Compute resize/pad strategy
    # If dimension < target: resize (squeeze) to target
    # If dimension >= target: pad to target
    resize_h = min(orig_h, height)
    resize_w = min(orig_w, width)
    
    # Resize if needed (only squeeze, never upscale beyond target)
    if (orig_w, orig_h) != (resize_w, resize_h):
        img = img.resize((resize_w, resize_h), Image.BILINEAR)
    
    img_np = np.array(img).astype(np.float32)
    
    # Pad to target size if needed (right and bottom padding)
    pad_h = height - resize_h
    pad_w = width - resize_w
    
    if pad_h > 0 or pad_w > 0:
        # Pad with zeros (will be normalized to mean later)
        img_padded = np.zeros((height, width, 3), dtype=np.float32)
        img_padded[:resize_h, :resize_w, :] = img_np
        img_np = img_padded
    
    # ImageNet normalization (same as training)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img_np = (img_np / 255.0 - mean) / std
    
    # HWC -> CHW and add batch dimension
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0)
    
    # Store padding info for output cropping
    pad_info = {
        'orig_h': orig_h,
        'orig_w': orig_w,
        'resize_h': resize_h,
        'resize_w': resize_w,
        'pad_h': pad_h,
        'pad_w': pad_w
    }
    
    return img_tensor.to(device), pad_info


def validate_onnx(
    pytorch_model: nn.Module,
    onnx_path: str,
    height: int = 256,
    width: int = 512,
    device: str = 'cpu',
    tolerance: float = 1e-4,
    left_image: str = None,
    right_image: str = None
):
    """
    Validate ONNX model output matches PyTorch model output.
    
    Args:
        pytorch_model: Original LightStereo model (will be wrapped)
        onnx_path: Path to exported ONNX model
        height: Input image height
        width: Input image width
        device: Device to run validation on
        tolerance: Maximum acceptable absolute difference
        left_image: Path to left test image (optional, uses random if not provided)
        right_image: Path to right test image (optional, uses random if not provided)
        
    Returns:
        bool: True if validation passes
    """
    import onnxruntime as ort
    
    print("\n" + "="*60)
    print("Validating ONNX vs PyTorch outputs")
    print("="*60)
    
    # Create ONNX wrapper with same structure
    onnx_wrapper = LightStereoONNX(pytorch_model)
    onnx_wrapper.eval()
    onnx_wrapper.to(device)
    
    # Load test images or create random input
    pad_info = None
    if left_image and right_image:
        print(f"Using test images:")
        print(f"  Left:  {left_image}")
        print(f"  Right: {right_image}")
        left, left_pad = load_image(left_image, height, width, device)
        right, right_pad = load_image(right_image, height, width, device)
        pad_info = left_pad  # Use left padding info (should be same as right)
        print(f"  Original size: {pad_info['orig_h']}x{pad_info['orig_w']}")
        print(f"  Resized to: {pad_info['resize_h']}x{pad_info['resize_w']}")
        print(f"  Padded to: {height}x{width}")
    else:
        print("Using random dummy input")
        torch.manual_seed(42)
        left = torch.randn(1, 3, height, width, device=device)
        right = torch.randn(1, 3, height, width, device=device)
    
    # PyTorch inference
    with torch.no_grad():
        pytorch_output = onnx_wrapper(left, right)
    pytorch_output_np = pytorch_output.cpu().numpy()
    
    # ONNX Runtime inference
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    
    ort_inputs = {
        'left': left.cpu().numpy(),
        'right': right.cpu().numpy()
    }
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # Crop padded regions from output if we have padding info
    if pad_info is not None and (pad_info['pad_h'] > 0 or pad_info['pad_w'] > 0):
        crop_h = pad_info['resize_h']
        crop_w = pad_info['resize_w']
        pytorch_output_np = pytorch_output_np[:, :, :crop_h, :crop_w]
        ort_output = ort_output[:, :, :crop_h, :crop_w]
        print(f"  Cropped output to: {crop_h}x{crop_w} (removed padding)")
    
    # Compare outputs
    abs_diff = np.abs(pytorch_output_np - ort_output)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    
    print(f"\nPyTorch output shape: {pytorch_output_np.shape}")
    print(f"ONNX output shape:    {ort_output.shape}")
    print(f"\nPyTorch disparity range: [{pytorch_output_np.min():.4f}, {pytorch_output_np.max():.4f}]")
    print(f"ONNX disparity range:    [{ort_output.min():.4f}, {ort_output.max():.4f}]")
    print(f"\nMax absolute difference:  {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    
    # Save visualization plot
    matplotlib.use('Agg')  # Non-interactive backend
    plot_path = str(Path(onnx_path).with_suffix('.png'))
    
    disp_pth = pytorch_output_np[0, 0]
    disp_onnx = ort_output[0, 0]
    epe_diff = np.abs(disp_pth - disp_onnx)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Dynamic min/max from actual data
    vmin = min(disp_pth.min(), disp_onnx.min())
    vmax = max(disp_pth.max(), disp_onnx.max())
    
    # Use turbo colormap (purple-blue-cyan-green-yellow-red)
    im0 = axes[0].imshow(disp_pth, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'PyTorch\n[{disp_pth.min():.2f}, {disp_pth.max():.2f}]')
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label='Disparity (px)')
    
    im1 = axes[1].imshow(disp_onnx, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'ONNX\n[{disp_onnx.min():.2f}, {disp_onnx.max():.2f}]')
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label='Disparity (px)')
    
    im2 = axes[2].imshow(epe_diff, cmap='turbo', vmin=0, vmax=max(epe_diff.max(), 0.01))
    axes[2].set_title(f'EPE Difference\nMax: {epe_diff.max():.4f} | Mean: {epe_diff.mean():.4f}')
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], shrink=0.8, label='EPE (px)')
    
    fig.suptitle(f'PyTorch vs ONNX Disparity Comparison', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved comparison plot: {plot_path}")
    
    if max_diff < tolerance:
        print(f"\n✓ VALIDATION PASSED (max diff {max_diff:.2e} < tolerance {tolerance:.2e})")
        return True
    else:
        print(f"\n✗ VALIDATION FAILED (max diff {max_diff:.2e} >= tolerance {tolerance:.2e})")
        return False


def main():
    parser = argparse.ArgumentParser(description='Export LightStereo to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to PyTorch checkpoint (.pth)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX path (default: same as checkpoint with .onnx)')
    parser.add_argument('--height', type=int, default=256,
                        help='Input image height (default: 256)')
    parser.add_argument('--width', type=int, default=512,
                        help='Input image width (default: 512)')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version (default: 17, minimum for PyTorch 2.x)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device for export (default: cpu)')
    parser.add_argument('--no-simplify', action='store_true',
                        help='Skip ONNX simplification')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip validation')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='Validation tolerance (default: 1e-4)')
    parser.add_argument('--left-image', type=str, default=None,
                        help='Path to left test image for validation')
    parser.add_argument('--right-image', type=str, default=None,
                        help='Path to right test image for validation')
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        args.output = str(Path(args.checkpoint).with_suffix('.onnx'))
    
    print("="*60)
    print("LightStereo ONNX Export")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output:     {args.output}")
    print(f"Resolution: {args.height}x{args.width}")
    print(f"Opset:      {args.opset}")
    print(f"Device:     {args.device}")
    print("="*60)
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    model, config = load_checkpoint(args.checkpoint, args.device)
    model.to(args.device)
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    onnx_wrapper = export_to_onnx(
        model,
        args.output,
        height=args.height,
        width=args.width,
        opset_version=args.opset,
        simplify=not args.no_simplify,
        device=args.device
    )
    
    # Validate
    if not args.no_validate:
        success = validate_onnx(
            model,
            args.output,
            height=args.height,
            width=args.width,
            device=args.device,
            tolerance=args.tolerance,
            left_image=args.left_image,
            right_image=args.right_image
        )
        if not success:
            sys.exit(1)
    
    print("\n" + "="*60)
    print("Export complete!")
    print("="*60)


if __name__ == '__main__':
    main()
