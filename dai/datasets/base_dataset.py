"""
Base dataset class for stereo matching
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

# Lazy import for foundation stereo (only when needed)
_foundation_stereo_model = None


class BaseStereoDataset(Dataset, ABC):
    """
    Abstract base class for stereo datasets.
    All name-specific datasets inherit from this.
    """
    
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        config: Dict[str, Any],
        split: str = 'train'
    ):
        """
        Initialize base stereo dataset.
        
        Args:
            samples: List of sample dictionaries from splitter
            config: Dataset configuration
            split: 'train', 'val', or 'test'
        """
        self.samples = samples
        self.config = config
        self.split = split
        
        self.resolution = config.get('resolution', None)
        self.target_source = config.get('target_source', 'provided')
        
        # Validate target_source
        valid_sources = ['provided', 'pseudo-foundationstereo', 'ssl']
        if self.target_source not in valid_sources:
            raise ValueError(
                f"Invalid target_source '{self.target_source}'. "
                f"Must be one of {valid_sources}"
            )
        
        logger.info(
            f"Initialized {self.__class__.__name__} with "
            f"{len(samples)} samples for {split} split, "
            f"target_source={self.target_source}"
        )
    
    def __len__(self) -> int:
        """Return number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample by index.
        
        Returns raw numpy arrays - PostprocessingDataset handles tensor conversion.
        
        Returns:
            Dictionary with keys:
            - 'left': np.ndarray [H, W, 3] float32 in [0, 255]
            - 'right': np.ndarray [H, W, 3] float32 in [0, 255]
            - 'disparity': np.ndarray [H, W] float32
            - 'valid_mask': np.ndarray [H, W] bool
            - 'metadata': Dict with sample info
        """
        from concurrent.futures import ThreadPoolExecutor
        
        sample = self.samples[idx]
        
        # Load all 3 files in parallel using threads (3x faster than sequential)
        with ThreadPoolExecutor(max_workers=3) as executor:
            # For 'provided' target_source, we can load disparity directly from file
            left_future = executor.submit(self._load_image, sample['left'])
            right_future = executor.submit(self._load_image, sample['right'])
            
            # Load disparity directly if target_source is 'provided'
            if self.target_source == 'provided' and sample['disparity'] is not None:
                disp_future = executor.submit(self._load_disparity, sample['disparity'])
            else:
                disp_future = None
            
            # Wait for all to complete
            left_img = left_future.result()
            right_img = right_future.result()
            
            if disp_future is not None:
                disparity = disp_future.result()
            else:
                # Fall back to _get_disparity for other target sources
                disparity = self._get_disparity(sample, left_img, right_img)
        
        # Resize if needed
        if self.resolution is not None:
            left_img, right_img, disparity = self._resize(
                left_img, right_img, disparity
            )
        
        # Compute valid mask (numpy array)
        valid_mask = self._compute_valid_mask(disparity)
        
        # Load FoundationStereo disparity if available (for temporal fusion initialization)
        disparity_foundationstereo = None
        if 'disparity_foundationstereo' in sample and sample['disparity_foundationstereo'] is not None:
            try:
                disparity_foundationstereo = self._load_disparity(sample['disparity_foundationstereo'])
                # Resize to match current resolution
                if self.resolution is not None:
                    h, w = self.resolution
                    scale_h = h / disparity_foundationstereo.shape[0]
                    scale_w = w / disparity_foundationstereo.shape[1]
                    disparity_foundationstereo = cv2.resize(
                        disparity_foundationstereo, (w, h), interpolation=cv2.INTER_LINEAR
                    )
                    disparity_foundationstereo = disparity_foundationstereo * scale_w
            except Exception as e:
                logger.warning(f"Failed to load disparity_foundationstereo: {e}")
                disparity_foundationstereo = None
        
        result = {
            'left': left_img,
            'right': right_img,
            'disparity': disparity,
            'valid_mask': valid_mask,
            'metadata': sample['metadata']
        }
        
        if disparity_foundationstereo is not None:
            result['disparity_foundationstereo'] = disparity_foundationstereo
        
        return result
    
    @abstractmethod
    def _load_image(self, path: str) -> Any:
        """Load image from path (dataset-specific format)"""
        pass
    
    @abstractmethod
    def _load_disparity(self, path: str) -> Any:
        """Load disparity from path (dataset-specific format)"""
        pass
    
    def _get_disparity(self, sample: Dict[str, Any], left_img: Any, right_img: Any) -> Any:
        """
        Get disparity based on target_source strategy.
        
        Strategies:
        1. 'provided': Load ground truth disparity from dataset files
        2. 'pseudo-foundationstereo': Generate using foundation stereo model
        3. 'ssl': Return zeros (for self-supervised learning)
        
        Args:
            sample: Sample dict with paths
            left_img: Left image (numpy array)
            right_img: Right image (numpy array)
            
        Returns:
            Disparity map (numpy array)
        """
        if self.target_source == 'provided':
            # Load ground truth disparity from dataset
            if sample['disparity'] is not None:
                return self._load_disparity(sample['disparity'])
            else:
                raise ValueError(
                    f"target_source='provided' but no disparity path in sample. "
                    f"Use 'ssl' or 'pseudo-foundationstereo' for samples without ground truth."
                )
        
        elif self.target_source == 'pseudo-foundationstereo':
            # Load pre-calculated FoundationStereo disparity if available
            if 'disparity_foundationstereo' in sample and sample['disparity_foundationstereo'] is not None:
                return self._load_disparity(sample['disparity_foundationstereo'])
            else:
                # Fallback: generate on-the-fly (not recommended due to performance)
                logger.warning(
                    "Pre-calculated FoundationStereo disparity not found. "
                    "Consider running generate_foundationstereo_disparities.py first."
                )
                return self._generate_pseudo_disparity(left_img, right_img)
        
        elif self.target_source == 'ssl':
            # Return zeros for self-supervised learning
            import numpy as np
            h, w = left_img.shape[:2]
            return np.zeros((h, w), dtype=np.float32)
        
        else:
            raise ValueError(f"Unknown target_source: {self.target_source}")
    
    def _generate_pseudo_disparity(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """
        Generate pseudo disparity using foundation stereo model.
        Automatically loads model and runs inference.
        
        Args:
            left_img: Left image (numpy array [H, W, 3] in range [0, 255])
            right_img: Right image (numpy array [H, W, 3] in range [0, 255])
            
        Returns:
            Pseudo disparity map (numpy array [H, W])
        """
        global _foundation_stereo_model
        
        # Lazy load foundation stereo model
        if _foundation_stereo_model is None:
            _foundation_stereo_model = self._load_foundation_stereo_model()
        
        # Run inference
        return self._run_foundation_stereo_inference(
            _foundation_stereo_model, left_img, right_img
        )
    
    def _load_foundation_stereo_model(self):
        """
        Load foundation stereo model from checkpoint.
        Uses checkpoint from src_data/checkpoints/foundationstereo/
        """
        import sys
        from pathlib import Path
        
        # Add stereo modeling to path
        repo_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(repo_root))
        
        try:
            from stereo.modeling.models.foundationstereo.core.foundation_stereo import FoundationStereo
            from easydict import EasyDict
            import yaml
        except ImportError as e:
            raise ImportError(
                f"Failed to import FoundationStereo: {e}. "
                "Make sure the stereo.modeling package is available."
            )
        
        # Load config - use small (vits) configuration
        checkpoint_dir = repo_root / "src_data" / "checkpoints" / "foundationstereo" / "small"
        cfg_path = checkpoint_dir / "cfg.yaml"
        
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Foundation stereo config not found at {cfg_path}. "
                f"Please ensure checkpoint is in src_data/checkpoints/foundationstereo/small/"
            )
        
        with open(cfg_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        args = EasyDict(cfg_dict)
        
        # Create model
        logger.info("Loading FoundationStereo model...")
        model = FoundationStereo(args)
        
        # Load checkpoint (small model uses model_best_bp2.pth)
        checkpoint_path = checkpoint_dir / "model_best_bp2.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Foundation stereo checkpoint not found at {checkpoint_path}"
            )
        
        # Load checkpoint to CPU and keep it on CPU to avoid CUDA re-initialization
        # in forked DataLoader worker processes
        logger.info("Loading checkpoint to CPU...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        elif 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        # Keep model on CPU to avoid CUDA initialization in worker processes
        device = torch.device('cpu')
        logger.info(f"Model will run on CPU (required for DataLoader multiprocessing)")
        model.eval()
        
        logger.info("FoundationStereo model loaded successfully")
        return {'model': model, 'device': device, 'args': args}
    
    def _run_foundation_stereo_inference(
        self, 
        model_dict: Dict[str, Any], 
        left_img: np.ndarray, 
        right_img: np.ndarray
    ) -> np.ndarray:
        """
        Run foundation stereo inference on image pair.
        
        Args:
            model_dict: Dict with 'model', 'device', 'args'
            left_img: Left image [H, W, 3] in range [0, 255]
            right_img: Right image [H, W, 3] in range [0, 255]
            
        Returns:
            Disparity map [H, W]
        """
        model = model_dict['model']
        device = model_dict['device']
        
        # Convert to tensors [1, 3, H, W]
        left_tensor = torch.from_numpy(left_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        right_tensor = torch.from_numpy(right_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        
        # Normalize images (FoundationStereo expects normalized input)
        from stereo.modeling.models.foundationstereo.core.foundation_stereo import normalize_image
        left_tensor = normalize_image(left_tensor)
        right_tensor = normalize_image(right_tensor)
        
        # Pad to multiple of 32
        from stereo.modeling.models.foundationstereo.core.utils.utils import InputPadder
        padder = InputPadder(left_tensor.shape[-2:], divis_by=32, force_square=False)
        left_padded, right_padded = padder.pad(left_tensor, right_tensor)
        
        # Prepare data dict for FoundationStereo (expects dict with 'left' and 'right' keys)
        data = {
            'left': left_padded,
            'right': right_padded
        }
        
        # Run inference
        with torch.no_grad():
            output = model(data)
        
        # Get disparity and unpad
        disp_pred = output['disp_pred']
        disp_pred = padder.unpad(disp_pred)
        
        # Convert to numpy [H, W]
        disp_np = disp_pred.squeeze().cpu().numpy()
        
        return disp_np
    
    def _resize(self, left_img: Any, right_img: Any, disparity: Any) -> tuple:
        """
        Resize images and disparity to target resolution.
        Override in subclass for specific implementation.
        """
        raise NotImplementedError("Resize not implemented in base class")
    
    def _apply_transforms(self, left_img: Any, right_img: Any, disparity: Any) -> tuple:
        """
        Apply data augmentation transforms.
        Override in subclass for specific implementation.
        """
        return left_img, right_img, disparity
    
    def _compute_valid_mask(self, disparity: Any) -> Any:
        """
        Compute valid pixel mask.
        Override in subclass for specific implementation.
        """
        if isinstance(disparity, torch.Tensor):
            return (disparity > 0) & (disparity < 512)
        else:
            import numpy as np
            return (disparity > 0) & (disparity < 512)
