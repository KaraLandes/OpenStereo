"""
Test different resize interpolation methods for disparity maps.
Compares various OpenCV and PyTorch interpolation methods on Monkaa eating_x2_0000.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F

# Paths
LEFT_IMG_PATH = "src_data/sceneflow/Monkaa/frames_cleanpass/eating_x2/left/0000.png"
RIGHT_IMG_PATH = "src_data/sceneflow/Monkaa/frames_cleanpass/eating_x2/right/0000.png"
DISP_PATH = "src_data/sceneflow/Monkaa/disparity/eating_x2/left/0000.pfm"

# Target size
TARGET_SIZE = (256, 512)  # (H, W)

def read_pfm(file_path):
    """Read PFM file."""
    with open(file_path, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise ValueError(f'Invalid PFM file: {file_path}')
        
        dim_match = f.readline().decode('utf-8')
        width, height = map(int, dim_match.split())
        
        scale = float(f.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)
        
        data = np.fromfile(f, endian + 'f')
        
        if color:
            shape = (height, width, 3)
        else:
            shape = (height, width)
        
        data = np.reshape(data, shape)
        data = np.flipud(data)
        
        return data

def resize_disparity_cv2(disp, target_size, interpolation, method_name):
    """Resize disparity using OpenCV."""
    orig_h, orig_w = disp.shape
    target_h, target_w = target_size
    
    resized = cv2.resize(disp, (target_w, target_h), interpolation=interpolation)
    # Scale disparity values
    resized = resized * (target_w / orig_w)
    
    return resized

def resize_disparity_torch(disp, target_size, mode, method_name):
    """Resize disparity using PyTorch."""
    orig_h, orig_w = disp.shape
    target_h, target_w = target_size
    
    # Convert to tensor
    disp_tensor = torch.from_numpy(disp).unsqueeze(0).unsqueeze(0).float()
    
    # Resize
    if mode == 'nearest':
        resized = F.interpolate(disp_tensor, size=target_size, mode='nearest')
    elif mode == 'bilinear':
        resized = F.interpolate(disp_tensor, size=target_size, mode='bilinear', align_corners=False)
    elif mode == 'bicubic':
        resized = F.interpolate(disp_tensor, size=target_size, mode='bicubic', align_corners=False)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    resized = resized.squeeze(0).squeeze(0).numpy()
    # Scale disparity values
    resized = resized * (target_w / orig_w)
    
    return resized

def compute_statistics(disp_original, disp_resized_back, disp_resized):
    """Compute error statistics."""
    # For fair comparison, resize back to original size
    orig_h, orig_w = disp_original.shape
    resized_h, resized_w = disp_resized.shape
    
    # Resize back using nearest neighbor
    disp_back = cv2.resize(disp_resized, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    disp_back = disp_back * (orig_w / resized_w)
    
    # Compute errors
    abs_error = np.abs(disp_original - disp_back)
    
    # Compute statistics on valid pixels (disp > 0)
    valid_mask = disp_original > 0
    
    if valid_mask.sum() > 0:
        mae = np.mean(abs_error[valid_mask])
        rmse = np.sqrt(np.mean((abs_error[valid_mask]) ** 2))
        max_error = np.max(abs_error[valid_mask])
        
        # Percentage of pixels with error > 3
        bad_3 = (abs_error[valid_mask] > 3).sum() / valid_mask.sum() * 100
    else:
        mae = rmse = max_error = bad_3 = 0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'Max Error': max_error,
        'Bad 3.0 (%)': bad_3
    }

def visualize_disparity(disp, title, vmin=None, vmax=None):
    """Visualize disparity map."""
    if vmin is None:
        vmin = disp[disp > 0].min() if (disp > 0).any() else 0
    if vmax is None:
        vmax = disp.max()
    
    plt.imshow(disp, cmap='turbo', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar(label='Disparity')
    plt.axis('off')

def main():
    print("=" * 80)
    print("Testing Resize Interpolation Methods for Disparity Maps")
    print("=" * 80)
    print(f"\nTest image: {LEFT_IMG_PATH}")
    print(f"Target size: {TARGET_SIZE} (H, W)")
    print()
    
    # Load disparity
    print("Loading disparity map...")
    disp_original = read_pfm(DISP_PATH).astype(np.float32)
    print(f"Original shape: {disp_original.shape}")
    print(f"Disparity range: [{disp_original.min():.2f}, {disp_original.max():.2f}]")
    print()
    
    # Define methods to test
    methods = [
        # OpenCV methods
        ('CV2 NEAREST', lambda d: resize_disparity_cv2(d, TARGET_SIZE, cv2.INTER_NEAREST, 'CV2_NEAREST')),
        ('CV2 LINEAR', lambda d: resize_disparity_cv2(d, TARGET_SIZE, cv2.INTER_LINEAR, 'CV2_LINEAR')),
        ('CV2 CUBIC', lambda d: resize_disparity_cv2(d, TARGET_SIZE, cv2.INTER_CUBIC, 'CV2_CUBIC')),
        ('CV2 AREA', lambda d: resize_disparity_cv2(d, TARGET_SIZE, cv2.INTER_AREA, 'CV2_AREA')),
        ('CV2 LANCZOS4', lambda d: resize_disparity_cv2(d, TARGET_SIZE, cv2.INTER_LANCZOS4, 'CV2_LANCZOS4')),
        
        # PyTorch methods
        ('Torch NEAREST', lambda d: resize_disparity_torch(d, TARGET_SIZE, 'nearest', 'Torch_NEAREST')),
        ('Torch BILINEAR', lambda d: resize_disparity_torch(d, TARGET_SIZE, 'bilinear', 'Torch_BILINEAR')),
        ('Torch BICUBIC', lambda d: resize_disparity_torch(d, TARGET_SIZE, 'bicubic', 'Torch_BICUBIC')),
    ]
    
    # Test each method
    results = {}
    resized_maps = {}
    
    print("Testing interpolation methods...")
    print("-" * 80)
    
    for method_name, resize_func in methods:
        # Resize
        disp_resized = resize_func(disp_original)
        resized_maps[method_name] = disp_resized
        
        # Compute statistics
        stats = compute_statistics(disp_original, None, disp_resized)
        results[method_name] = stats
        
        print(f"{method_name:20s} | MAE: {stats['MAE']:6.3f} | RMSE: {stats['RMSE']:6.3f} | "
              f"Max: {stats['Max Error']:7.2f} | Bad3: {stats['Bad 3.0 (%)']:5.2f}%")
    
    print("-" * 80)
    
    # Find best method (lowest MAE)
    best_method = min(results.items(), key=lambda x: x[1]['MAE'])
    print(f"\n✓ BEST METHOD (lowest MAE): {best_method[0]}")
    print(f"  MAE: {best_method[1]['MAE']:.3f}")
    print()
    
    # Visualize results
    print("Creating visualization...")
    
    # Compute global vmin/vmax for consistent colormap
    vmin = disp_original[disp_original > 0].min()
    vmax = disp_original.max()
    
    # Create figure with subplots
    n_methods = len(methods) + 1  # +1 for original
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    # Plot original
    plt.sca(axes[0])
    visualize_disparity(disp_original, f'Original ({disp_original.shape[0]}x{disp_original.shape[1]})', vmin, vmax)
    
    # Plot resized versions
    for idx, (method_name, _) in enumerate(methods):
        plt.sca(axes[idx + 1])
        disp_resized = resized_maps[method_name]
        stats = results[method_name]
        title = f'{method_name}\nMAE: {stats["MAE"]:.3f}, Bad3: {stats["Bad 3.0 (%)"]:.2f}%'
        visualize_disparity(disp_resized, title, vmin, vmax)
    
    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'resize_interpolation_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    
    # Show plot
    plt.show()
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    print(f"Use {best_method[0]} for disparity resizing.")
    print("This method preserves disparity values best with minimal error.")
    print()

if __name__ == '__main__':
    main()
