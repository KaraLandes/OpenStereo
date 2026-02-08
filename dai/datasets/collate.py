"""
Custom collate functions for sequential datasets
"""

import torch
from typing import List, Dict, Any


def sequential_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for sequential datasets.

    Stacks sequences along batch dimension while maintaining temporal structure.

    Args:
        batch: List of samples, where each sample is:
            {
                'sequence': [frame0, frame1, ..., frameN],
                'sequence_metadata': {...}
            }

    Returns:
        Dictionary with:
            'sequence': List of frames, where each frame has batch dimension
            'sequence_metadata': List of metadata dicts

    Example:
        Input batch (batch_size=8, sequence_length=4):
            [
                {'sequence': [f0, f1, f2, f3], 'metadata': m0},  # seq0
                {'sequence': [f0, f1, f2, f3], 'metadata': m1},  # seq1
                ...
                {'sequence': [f0, f1, f2, f3], 'metadata': m7}   # seq7
            ]

        Output:
            {
                'sequence': [
                    {batched_frame0},  # [8, 3, H, W] - frame 0 from all 8 seqs
                    {batched_frame1},  # [8, 3, H, W] - frame 1 from all 8 seqs
                    {batched_frame2},  # [8, 3, H, W]
                    {batched_frame3}   # [8, 3, H, W]
                ],
                'sequence_metadata': [m0, m1, ..., m7]
            }
    """
    batch_size = len(batch)
    sequence_length = len(batch[0]['sequence'])

    # Collect all frames at each timestep
    batched_sequence = []
    for t in range(sequence_length):
        # Collect frame t from all sequences
        frames_at_t = [batch[i]['sequence'][t] for i in range(batch_size)]

        # Stack tensors for this timestep
        batched_frame = {
            'left': torch.stack([f['left'] for f in frames_at_t], dim=0),
            'right': torch.stack([f['right'] for f in frames_at_t], dim=0),
            'disparity': torch.stack([f['disparity'] for f in frames_at_t], dim=0),
            'valid_mask': torch.stack([f['valid_mask'] for f in frames_at_t], dim=0),
            'metadata': [f['metadata'] for f in frames_at_t]
        }
        
        # Include disparity_foundationstereo if available
        if 'disparity_foundationstereo' in frames_at_t[0]:
            disp_fs_list = [f.get('disparity_foundationstereo') for f in frames_at_t]
            if all(d is not None for d in disp_fs_list):
                batched_frame['disparity_foundationstereo'] = torch.stack(disp_fs_list, dim=0)
            else:
                batched_frame['disparity_foundationstereo'] = None
        else:
            batched_frame['disparity_foundationstereo'] = None
        batched_sequence.append(batched_frame)

    # Collect metadata
    metadata_list = [batch[i]['sequence_metadata'] for i in range(batch_size)]

    return {
        'sequence': batched_sequence,
        'sequence_metadata': metadata_list
    }
