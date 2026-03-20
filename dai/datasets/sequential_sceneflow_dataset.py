"""
Sequential SceneFlow dataset for temporal stereo matching.
Wraps existing SceneFlowDataset to return sequences of frames instead of single frames.
"""

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict
from torch.utils.data import Dataset

from .sceneflow_dataset import SceneFlowDataset

logger = logging.getLogger(__name__)


class SequentialSceneFlowDataset(Dataset):
    """
    Sequential wrapper for SceneFlowDataset.
    
    Groups samples by scene and creates sliding window sequences.
    Preserves folder-based stratified splits by only creating sequences within scenes.
    """
    
    def __init__(
        self,
        base_dataset: SceneFlowDataset,
        sequence_length: int = 4,
        stride: int = 1,
        skip_incomplete: bool = True
    ):
        """
        Initialize sequential dataset.
        
        Args:
            base_dataset: Base SceneFlowDataset instance (already split)
            sequence_length: Number of frames per sequence
            stride: Sliding window stride (1 = no overlap, 2 = 50% overlap)
            skip_incomplete: If True, skip sequences shorter than sequence_length
        """
        self.base_dataset = base_dataset
        self.sequence_length = sequence_length
        self.stride = stride
        self.skip_incomplete = skip_incomplete
        
        # Expose base dataset attributes for pipeline compatibility
        self.target_source = getattr(base_dataset, 'target_source', 'provided')
        
        # Group samples by scene to create sequences
        self.sequences = self._create_sequences()
        
        logger.info(
            f"Created SequentialSceneFlowDataset with {len(self.sequences)} sequences "
            f"(sequence_length={sequence_length}, stride={stride}, "
            f"skip_incomplete={skip_incomplete})"
        )
    
    def _create_sequences(self) -> List[List[int]]:
        """
        Create sequences by grouping samples by scene and applying sliding window.
        
        Returns:
            List of sequences, where each sequence is a list of indices into base_dataset
        """
        # Group samples by scene
        scene_groups = defaultdict(list)
        
        for idx in range(len(self.base_dataset)):
            sample = self.base_dataset.samples[idx]
            scene_key = self._get_scene_key(sample)
            scene_groups[scene_key].append(idx)
        
        # Sort each scene's samples by frame_id to ensure temporal order
        for scene_key in scene_groups:
            scene_groups[scene_key].sort(
                key=lambda idx: self.base_dataset.samples[idx]['metadata']['frame_id']
            )
        
        # Create sliding window sequences within each scene
        sequences = []
        
        for scene_key, indices in scene_groups.items():
            scene_sequences = self._create_scene_sequences(indices)
            sequences.extend(scene_sequences)
            
            logger.debug(
                f"Scene {scene_key}: {len(indices)} frames → {len(scene_sequences)} sequences"
            )
        
        return sequences
    
    def _get_scene_key(self, sample: Dict[str, Any]) -> str:
        """
        Get unique scene identifier from sample metadata.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Scene key string
        """
        metadata = sample['metadata']
        subset = metadata['subset']
        scene = metadata['scene']
        return f"{subset}/{scene}"
    
    def _create_scene_sequences(self, indices: List[int]) -> List[List[int]]:
        """
        Create sliding window sequences from a single scene's frame indices.
        
        Args:
            indices: List of frame indices (already sorted by frame_id)
            
        Returns:
            List of sequences (each sequence is a list of indices)
        """
        sequences = []
        
        # Sliding window
        for start_idx in range(0, len(indices), self.stride):
            end_idx = start_idx + self.sequence_length
            
            # Check if we have enough frames
            if end_idx > len(indices):
                if self.skip_incomplete:
                    break  # Skip incomplete sequences
                else:
                    end_idx = len(indices)  # Use what we have
            
            sequence = indices[start_idx:end_idx]
            
            # Only add if sequence meets minimum length requirement
            if len(sequence) == self.sequence_length or not self.skip_incomplete:
                sequences.append(sequence)
        
        return sequences
    
    def __len__(self) -> int:
        """Return number of sequences"""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sequence by index.
        
        Args:
            idx: Sequence index
            
        Returns:
            Dictionary with:
            - 'sequence': List of frame dicts (length = sequence_length)
            - 'sequence_metadata': Dict with sequence info
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        sequence_indices = self.sequences[idx]
        
        # Load all frames in the sequence
        frames = []
        for frame_idx in sequence_indices:
            frame = self.base_dataset[frame_idx]
            frames.append(frame)
        
        # Extract sequence metadata from first frame
        first_frame_metadata = frames[0]['metadata']
        sequence_metadata = {
            'scene_key': self._get_scene_key(self.base_dataset.samples[sequence_indices[0]]),
            'subset': first_frame_metadata['subset'],
            'scene': first_frame_metadata['scene'],
            'start_frame_id': first_frame_metadata['frame_id'],
            'sequence_length': len(frames),
            'frame_indices': sequence_indices
        }
        
        return {
            'sequence': frames,
            'sequence_metadata': sequence_metadata
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        # Count sequences per scene
        scene_counts = defaultdict(int)
        for sequence in self.sequences:
            sample = self.base_dataset.samples[sequence[0]]
            scene_key = self._get_scene_key(sample)
            scene_counts[scene_key] += 1
        
        return {
            'total_sequences': len(self.sequences),
            'total_frames': len(self.base_dataset),
            'num_scenes': len(scene_counts),
            'avg_sequences_per_scene': len(self.sequences) / len(scene_counts) if scene_counts else 0,
            'sequence_length': self.sequence_length,
            'stride': self.stride
        }
