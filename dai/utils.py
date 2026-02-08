"""
Utility functions for DAI pipeline
"""

import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set random seed to {seed} for reproducibility")
