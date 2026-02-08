"""
Data augmentation transforms for stereo matching
"""

import random
import torch
import numpy as np
from typing import Dict, Any, List
from PIL import Image
from torchvision.transforms import ColorJitter


class Compose:
    """Compose multiple transforms together"""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Check if this is a sequential sample (has 'sequence' key)
        if 'sequence' in sample:
            # Apply transforms to each frame in the sequence
            transformed_sequence = []
            for frame in sample['sequence']:
                transformed_frame = frame
                for transform in self.transforms:
                    transformed_frame = transform(transformed_frame)
                transformed_sequence.append(transformed_frame)
            sample['sequence'] = transformed_sequence
            return sample
        else:
            # Regular single-frame sample
            for transform in self.transforms:
                sample = transform(sample)
            return sample


class ToTensor:
    """
    Convert numpy arrays to torch tensors.
    Converts images from [H, W, C] to [C, H, W] format.
    """
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Convert images from numpy [H, W, C] to tensor [C, H, W]
        if not isinstance(sample['left'], torch.Tensor):
            sample['left'] = torch.from_numpy(sample['left'].transpose(2, 0, 1)).float()
        if not isinstance(sample['right'], torch.Tensor):
            sample['right'] = torch.from_numpy(sample['right'].transpose(2, 0, 1)).float()
        
        # Convert disparity and mask (already [H, W])
        if not isinstance(sample['disparity'], torch.Tensor):
            sample['disparity'] = torch.from_numpy(sample['disparity']).float()
        if not isinstance(sample['valid_mask'], torch.Tensor):
            sample['valid_mask'] = torch.from_numpy(sample['valid_mask']).bool()
        
        # Convert disparity_foundationstereo if present
        if 'disparity_foundationstereo' in sample and sample['disparity_foundationstereo'] is not None:
            if not isinstance(sample['disparity_foundationstereo'], torch.Tensor):
                sample['disparity_foundationstereo'] = torch.from_numpy(sample['disparity_foundationstereo']).float().unsqueeze(0)  # [1, H, W]
        
        return sample


class NormalizeImage:
    """
    Normalize images with mean and std.
    Assumes images are in range [0, 255].
    """
    
    def __init__(self, mean: List[float] = [0.485, 0.456, 0.406], 
                 std: List[float] = [0.229, 0.224, 0.225]):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Normalize: (img / 255 - mean) / std
        sample['left'] = (sample['left'] / 255.0 - self.mean) / self.std
        sample['right'] = (sample['right'] / 255.0 - self.mean) / self.std
        return sample


class RandomCrop:
    """
    Random crop for stereo pairs.
    Crops left and right images at the same y position.
    """
    
    def __init__(self, size: List[int]):
        """
        Args:
            size: [height, width] of crop
        """
        self.crop_h, self.crop_w = size
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Handle both numpy arrays [H, W, C] and tensors [C, H, W]
        if isinstance(sample['left'], np.ndarray):
            # Numpy array: [H, W, C]
            h, w = sample['left'].shape[:2]
        else:
            # Tensor: [C, H, W]
            _, h, w = sample['left'].shape
        
        # If image smaller than crop, return as-is
        if h < self.crop_h or w < self.crop_w:
            return sample
        
        # Random crop position
        y = random.randint(0, h - self.crop_h)
        x = random.randint(0, w - self.crop_w)
        
        # Crop based on data type
        if isinstance(sample['left'], np.ndarray):
            # Numpy arrays: [H, W, C]
            sample['left'] = sample['left'][y:y+self.crop_h, x:x+self.crop_w, :]
            sample['right'] = sample['right'][y:y+self.crop_h, x:x+self.crop_w, :]
            sample['disparity'] = sample['disparity'][y:y+self.crop_h, x:x+self.crop_w]
            sample['valid_mask'] = sample['valid_mask'][y:y+self.crop_h, x:x+self.crop_w]
        else:
            # Tensors: [C, H, W]
            sample['left'] = sample['left'][:, y:y+self.crop_h, x:x+self.crop_w]
            sample['right'] = sample['right'][:, y:y+self.crop_h, x:x+self.crop_w]
            sample['disparity'] = sample['disparity'][y:y+self.crop_h, x:x+self.crop_w]
            sample['valid_mask'] = sample['valid_mask'][y:y+self.crop_h, x:x+self.crop_w]
        
        return sample


class StereoColorJitter:
    """
    Color jittering for stereo pairs.
    Can apply symmetric (same jitter to both) or asymmetric (different jitter).
    """
    
    def __init__(self, brightness: List[float] = [0.6, 1.4],
                 contrast: List[float] = [0.6, 1.4],
                 saturation: List[float] = [0.6, 1.4],
                 hue: float = 0.5,
                 asymmetric_prob: float = 0.2):
        """
        Args:
            brightness: Range for brightness adjustment
            contrast: Range for contrast adjustment
            saturation: Range for saturation adjustment
            hue: Range for hue adjustment
            asymmetric_prob: Probability of asymmetric color jitter
        """
        self.asymmetric_prob = asymmetric_prob
        # Convert hue to range expected by ColorJitter
        hue_range = [-hue / 3.14, hue / 3.14]
        self.color_jitter = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue_range
        )
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Convert tensors back to numpy for PIL operations
        left = sample['left'].permute(1, 2, 0).numpy().astype(np.uint8)
        right = sample['right'].permute(1, 2, 0).numpy().astype(np.uint8)
        
        # Asymmetric: apply different jitter to left and right
        if random.random() < self.asymmetric_prob:
            left = np.array(self.color_jitter(Image.fromarray(left)), dtype=np.uint8)
            right = np.array(self.color_jitter(Image.fromarray(right)), dtype=np.uint8)
        # Symmetric: apply same jitter to both
        else:
            stacked = np.concatenate([left, right], axis=0)
            stacked = np.array(self.color_jitter(Image.fromarray(stacked)), dtype=np.uint8)
            left, right = np.split(stacked, 2, axis=0)
        
        # Convert back to tensors
        sample['left'] = torch.from_numpy(left).permute(2, 0, 1).float()
        sample['right'] = torch.from_numpy(right).permute(2, 0, 1).float()
        
        return sample


class RandomErase:
    """
    Random erasing augmentation for stereo pairs.
    Randomly erases rectangular regions in the right image.
    """
    
    def __init__(self, prob: float = 0.5, max_time: int = 2,
                 bounds: List[int] = [50, 100]):
        """
        Args:
            prob: Probability of applying random erase
            max_time: Maximum number of erase operations
            bounds: [min_size, max_size] of erased rectangles
        """
        self.prob = prob
        self.max_time = max_time
        self.bounds = bounds
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.prob:
            return sample
        
        # Convert to numpy for easier manipulation
        right = sample['right'].permute(1, 2, 0).numpy()
        h, w = right.shape[:2]
        
        # Compute mean color
        mean_color = np.mean(right.reshape(-1, 3), axis=0)
        
        # Apply random erases
        num_erases = random.randint(1, self.max_time)
        for _ in range(num_erases):
            x0 = random.randint(0, w - 1)
            y0 = random.randint(0, h - 1)
            dx = random.randint(self.bounds[0], self.bounds[1])
            dy = random.randint(self.bounds[0], self.bounds[1])
            
            # Ensure bounds
            x1 = min(x0 + dx, w)
            y1 = min(y0 + dy, h)
            
            # Erase region
            right[y0:y1, x0:x1, :] = mean_color
        
        # Convert back to tensor
        sample['right'] = torch.from_numpy(right).permute(2, 0, 1).float()
        
        return sample


class RightBottomPad:
    """
    Pad images to ensure dimensions are divisible by a factor.
    Pads on the right and bottom sides.
    """
    
    def __init__(self, divisor: int = 32):
        """
        Args:
            divisor: Ensure height and width are divisible by this value
        """
        self.divisor = divisor
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Handle both numpy arrays [H, W, C] and tensors [C, H, W]
        if isinstance(sample['left'], np.ndarray):
            # Numpy array: [H, W, C]
            h, w = sample['left'].shape[:2]
        else:
            # Tensor: [C, H, W]
            _, h, w = sample['left'].shape
        
        # Calculate padding needed
        pad_h = (self.divisor - h % self.divisor) % self.divisor
        pad_w = (self.divisor - w % self.divisor) % self.divisor
        
        if pad_h == 0 and pad_w == 0:
            return sample
        
        # Pad based on data type
        if isinstance(sample['left'], np.ndarray):
            # Numpy arrays: [H, W, C]
            sample['left'] = np.pad(sample['left'], ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            sample['right'] = np.pad(sample['right'], ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            sample['disparity'] = np.pad(sample['disparity'], ((0, pad_h), (0, pad_w)), mode='constant')
            sample['valid_mask'] = np.pad(sample['valid_mask'], ((0, pad_h), (0, pad_w)), mode='constant', constant_values=False)
        else:
            # Tensors: [C, H, W]
            import torch.nn.functional as F
            # F.pad expects (left, right, top, bottom)
            sample['left'] = F.pad(sample['left'], (0, pad_w, 0, pad_h), mode='constant', value=0)
            sample['right'] = F.pad(sample['right'], (0, pad_w, 0, pad_h), mode='constant', value=0)
            sample['disparity'] = F.pad(sample['disparity'], (0, pad_w, 0, pad_h), mode='constant', value=0)
            sample['valid_mask'] = F.pad(sample['valid_mask'], (0, pad_w, 0, pad_h), mode='constant', value=False)
        
        return sample


class RandomScale:
    """
    Random scale augmentation for stereo pairs.
    """
    
    def __init__(self, min_scale: float = -0.2, max_scale: float = 0.4, 
                 scale_prob: float = 0.8):
        """
        Args:
            min_scale: Minimum scale factor (negative = shrink)
            max_scale: Maximum scale factor (positive = grow)
            scale_prob: Probability of applying scaling
        """
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_prob = scale_prob
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.scale_prob:
            return sample
        
        # Random scale factor
        scale = 1.0 + random.uniform(self.min_scale, self.max_scale)
        
        # Get current size
        _, h, w = sample['left'].shape
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize using interpolation
        import torch.nn.functional as F
        
        sample['left'] = F.interpolate(
            sample['left'].unsqueeze(0), 
            size=(new_h, new_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        sample['right'] = F.interpolate(
            sample['right'].unsqueeze(0), 
            size=(new_h, new_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        sample['disparity'] = F.interpolate(
            sample['disparity'].unsqueeze(0).unsqueeze(0), 
            size=(new_h, new_w), 
            mode='nearest'
        ).squeeze(0).squeeze(0) * scale
        
        sample['valid_mask'] = F.interpolate(
            sample['valid_mask'].unsqueeze(0).unsqueeze(0).float(), 
            size=(new_h, new_w), 
            mode='nearest'
        ).squeeze(0).squeeze(0).bool()
        
        return sample


def build_transforms(config: Dict[str, Any], split: str = 'train') -> Compose:
    """
    Build transform pipeline from config.
    
    Args:
        config: Transform configuration dict
        split: 'train', 'val', or 'test'
    
    Returns:
        Compose object with transforms
    """
    if config is None or split not in config:
        # Default: just convert to tensor
        return Compose([ToTensor()])
    
    transform_list = []
    
    for t_config in config[split]:
        name = t_config['name']
        
        if name == 'ToTensor':
            transform_list.append(ToTensor())
        
        elif name == 'NormalizeImage':
            mean = t_config.get('mean', [0.485, 0.456, 0.406])
            std = t_config.get('std', [0.229, 0.224, 0.225])
            transform_list.append(NormalizeImage(mean, std))
        
        elif name == 'RandomCrop':
            size = t_config['size']
            transform_list.append(RandomCrop(size))
        
        elif name == 'RandomScale':
            min_scale = t_config.get('min_scale', -0.2)
            max_scale = t_config.get('max_scale', 0.4)
            scale_prob = t_config.get('scale_prob', 0.8)
            transform_list.append(RandomScale(min_scale, max_scale, scale_prob))
        
        elif name == 'StereoColorJitter':
            brightness = t_config.get('brightness', [0.6, 1.4])
            contrast = t_config.get('contrast', [0.6, 1.4])
            saturation = t_config.get('saturation', [0.6, 1.4])
            hue = t_config.get('hue', 0.5)
            asymmetric_prob = t_config.get('asymmetric_prob', 0.2)
            transform_list.append(StereoColorJitter(brightness, contrast, saturation, hue, asymmetric_prob))
        
        elif name == 'RandomErase':
            prob = t_config.get('prob', 0.5)
            max_time = t_config.get('max_time', 2)
            bounds = t_config.get('bounds', [50, 100])
            transform_list.append(RandomErase(prob, max_time, bounds))
        
        elif name == 'RightBottomPad':
            divisor = t_config.get('divisor', 32)
            transform_list.append(RightBottomPad(divisor))
        
        else:
            raise ValueError(f"Unknown transform: {name}")
    
    return Compose(transform_list)
