"""
Dataset-specific discovery strategies
"""

from pathlib import Path
from typing import List, Dict, Any
import logging
from tqdm import tqdm

from .discovery import BaseDiscoveryStrategy
from .registry import DatasetRegistry

logger = logging.getLogger(__name__)


@DatasetRegistry.register('sceneflow')
class SceneFlowDiscovery(BaseDiscoveryStrategy):
    """
    Discovery strategy for SceneFlow dataset.
    
    File structure:
    root/
        Monkaa/
            frames_cleanpass/
                scene_name/
                    left/0001.png
                    right/0001.png
            frames_finalpass/
                scene_name/
                    left/0001.png
                    right/0001.png
            disparity/
                scene_name/
                    left/0001.pfm
                    right/0001.pfm
        Driving/
            ...
        FlyingThings3D/
            ...
    """
    
    def discover_samples(self, root: Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Discover SceneFlow samples.
        
        Config keys:
            - subsets: List of subsets to use ['monkaa', 'driving', 'flyingthings3d']
            - pass_type: 'cleanpass' or 'finalpass'
            - target_source: 'provided', 'pseudo-foundationstereo', or 'ssl'
        """
        root = Path(root)
        subsets = config.get('subsets', ['monkaa'])
        pass_type = config.get('pass_type', 'cleanpass')
        target_source = config.get('target_source', 'provided')
        
        samples = []
        
        for subset in subsets:
            subset_name = self._normalize_subset_name(subset)
            subset_path = root / subset_name
            
            if not subset_path.exists():
                logger.warning(f"Subset path not found: {subset_path}")
                continue
            
            frames_dir = subset_path / f'frames_{pass_type}'
            disparity_dir = subset_path / 'disparity'
            
            if not frames_dir.exists():
                logger.warning(f"Frames directory not found: {frames_dir}")
                continue
            
            # Find all left/right directory pairs (handles nested structures in Driving/FlyingThings3D)
            logger.info(f"Scanning {subset_name} for samples (this may take 1-2 minutes for large datasets)...")
            left_dirs = list(tqdm(frames_dir.rglob('left'), desc=f"Scanning {subset_name}", unit=" dirs"))
            logger.info(f"Found {len(left_dirs)} scene directories in {subset_name}, processing samples...")
            
            for left_dir in tqdm(left_dirs, desc=f"Processing {subset_name} scenes", unit=" scene"):
                if not left_dir.is_dir():
                    continue
                    
                right_dir = left_dir.parent / 'right'
                if not right_dir.exists() or not right_dir.is_dir():
                    continue
                
                # Get scene name as relative path from frames_dir
                scene_dir = left_dir.parent
                scene_name = str(scene_dir.relative_to(frames_dir))
                
                left_images = sorted(left_dir.glob('*.png'))
                
                for left_img in left_images:
                    frame_id = left_img.stem
                    right_img = right_dir / left_img.name
                    
                    if not right_img.exists():
                        continue
                    
                    # Check for ground truth disparity
                    if target_source == 'provided':
                        disp_path = disparity_dir / scene_name / 'left' / f'{frame_id}.pfm'
                        if not disp_path.exists():
                            continue
                    else:
                        disp_path = None
                    
                    # Check for pre-calculated FoundationStereo disparity
                    disp_foundationstereo_path = None
                    if target_source == 'pseudo-foundationstereo':
                        foundationstereo_dir = subset_path / 'disparity_foundationstereo'
                        disp_foundationstereo_path = foundationstereo_dir / scene_name / 'left' / f'{frame_id}.pfm'
                        if not disp_foundationstereo_path.exists():
                            disp_foundationstereo_path = None
                    
                    sample = {
                        'left': str(left_img),
                        'right': str(right_img),
                        'disparity': str(disp_path) if disp_path else None,
                        'disparity_foundationstereo': str(disp_foundationstereo_path) if disp_foundationstereo_path else None,
                        'metadata': {
                            'dataset': 'sceneflow',
                            'subset': subset_name,
                            'scene': scene_name,
                            'frame_id': frame_id,
                            'pass_type': pass_type,
                        }
                    }
                    
                    if self.validate_sample(sample):
                        samples.append(sample)
        
        logger.info(f"Discovered {len(samples)} SceneFlow samples")
        return samples
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate SceneFlow sample"""
        left_path = Path(sample['left'])
        right_path = Path(sample['right'])
        
        if not left_path.exists() or not right_path.exists():
            return False
        
        if sample['disparity'] is not None:
            disp_path = Path(sample['disparity'])
            if not disp_path.exists():
                return False
        
        return True
    
    @staticmethod
    def _normalize_subset_name(subset: str) -> str:
        """Normalize subset name to match directory structure"""
        mapping = {
            'monkaa': 'Monkaa',
            'driving': 'Driving',
            'flyingthings3d': 'FlyingThings3D',
            'flyingthings3d_subset': 'FlyingThings3D_subset',
        }
        return mapping.get(subset.lower(), subset)


@DatasetRegistry.register('kitti12')
class KITTI12Discovery(BaseDiscoveryStrategy):
    """
    Discovery strategy for KITTI 2012 dataset.
    
    File structure:
    root/
        data_stereo_flow/
            training/
                colored_0/000000_10.png, 000000_11.png, ...
                colored_1/000000_10.png, 000000_11.png, ...
                disp_noc/000000_10.png, ...
                disp_occ/000000_10.png, ...
            testing/
                colored_0/
                colored_1/
    """
    
    def discover_samples(self, root: Path, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Discover KITTI12 samples.
        
        Config keys:
            - split: 'training' or 'testing'
            - target_source: 'provided', 'pseudo-foundationstereo', or 'ssl'
        """
        root = Path(root)
        split = config.get('split', 'training')
        target_source = config.get('target_source', 'provided')
        
        samples = []
        
        # KITTI12 has data_stereo_flow subdirectory
        data_dir = root / 'data_stereo_flow'
        if not data_dir.exists():
            # Try without subdirectory (alternative structure)
            data_dir = root
        
        split_dir = data_dir / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            return samples
        
        left_dir = split_dir / 'colored_0'
        right_dir = split_dir / 'colored_1'
        
        if not left_dir.exists() or not right_dir.exists():
            logger.warning(f"Image directories not found in {split_dir}")
            return samples
        
        left_images = sorted(left_dir.glob('*_10.png'))
        
        for left_img in left_images:
            frame_id = left_img.stem
            right_img = right_dir / left_img.name
            
            if not right_img.exists():
                continue
            
            # Check for ground truth disparity
            if target_source == 'provided' and split == 'training':
                disp_path = split_dir / 'disp_noc' / left_img.name
                if not disp_path.exists():
                    disp_path = split_dir / 'disp_occ' / left_img.name
                if not disp_path.exists():
                    continue
            else:
                disp_path = None
            
            # Check for pre-calculated FoundationStereo disparity
            disp_foundationstereo_path = None
            if target_source == 'pseudo-foundationstereo':
                foundationstereo_dir = split_dir / 'disp_foundationstereo'
                disp_foundationstereo_path = foundationstereo_dir / left_img.name
                if not disp_foundationstereo_path.exists():
                    disp_foundationstereo_path = None
            
            sample = {
                'left': str(left_img),
                'right': str(right_img),
                'disparity': str(disp_path) if disp_path else None,
                'disparity_foundationstereo': str(disp_foundationstereo_path) if disp_foundationstereo_path else None,
                'metadata': {
                    'dataset': 'kitti12',
                    'split': split,
                    'frame_id': frame_id,
                }
            }
            
            if self.validate_sample(sample):
                samples.append(sample)
        
        logger.info(f"Discovered {len(samples)} KITTI12 samples")
        return samples
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate KITTI12 sample"""
        left_path = Path(sample['left'])
        right_path = Path(sample['right'])
        
        if not left_path.exists() or not right_path.exists():
            return False
        
        if sample['disparity'] is not None:
            disp_path = Path(sample['disparity'])
            if not disp_path.exists():
                return False
        
        return True
