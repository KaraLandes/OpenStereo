"""
OpenEXR file loader for IRS dataset disparity maps.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_exr(path: str) -> np.ndarray:
    """
    Load an OpenEXR file and return as numpy array.
    
    For 1-channel files (disparity): returns [H, W] float32
    For 3-channel files (normals): returns [H, W, 3] float32
    
    Args:
        path: Path to .exr file
        
    Returns:
        numpy array, float32
    """
    import OpenEXR
    import Imath
    
    exr_file = OpenEXR.InputFile(str(path))
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    channel_keys = list(exr_file.header()['channels'].keys())
    num_channels = len(channel_keys)
    
    if num_channels > 1:
        # 3-channel (e.g. normals): read R, G, B
        channels = ['R', 'G', 'B']
        pixels = [
            np.frombuffer(exr_file.channel(c, pixel_type), dtype=np.float32)
            for c in channels
        ]
        result = np.zeros((height, width, 3), dtype=np.float32)
        result[:, :, 0] = pixels[0].reshape(height, width)
        result[:, :, 1] = pixels[1].reshape(height, width)
        result[:, :, 2] = pixels[2].reshape(height, width)
    else:
        # 1-channel (e.g. disparity): read the single channel
        channel_name = channel_keys[0]
        raw = exr_file.channel(channel_name, pixel_type)
        result = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
    
    return result
