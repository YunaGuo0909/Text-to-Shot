"""
Dataset classes for training the camera trajectory diffusion model.

Supports loading camera trajectory data extracted from film shots
(e.g., ShotDeck) with text annotations, shot types, and motion labels.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from typing import Optional, Dict, List


class CameraTrajectoryDataset(Dataset):
    """
    Dataset for camera trajectory generation.

    Each sample contains:
    - Camera trajectory: (num_frames, 6) Toric parameters over time
    - Text description of the shot/scene
    - Shot type label (close-up, medium-shot, wide-shot, etc.)
    - Camera motion type label (dolly-in, pan-left, crane-up, etc.)

    Data is sourced from ShotDeck film shots with camera parameters
    extracted via camera estimation methods.
    """

    # Maps for converting string labels to indices
    SHOT_TYPE_MAP = {
        "close-up": 0, "medium-shot": 1, "wide-shot": 2,
        "over-the-shoulder": 3, "two-shot": 4,
    }
    MOTION_TYPE_MAP = {
        "static": 0, "dolly-in": 1, "dolly-out": 2,
        "pan-left": 3, "pan-right": 4, "crane-up": 5,
        "crane-down": 6, "track": 7, "orbit": 8,
    }

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        num_frames: int = 48,
        toric_dim: int = 6,
    ):
        """
        Args:
            data_root: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test')
            num_frames: Expected number of frames per trajectory
            toric_dim: Dimension of Toric camera state (6)
        """
        self.data_root = data_root
        self.split = split
        self.num_frames = num_frames
        self.toric_dim = toric_dim
        self.total_dim = num_frames * toric_dim

        self.samples = self._load_index()

    def _load_index(self) -> List[Dict]:
        """Load dataset index file."""
        index_path = os.path.join(self.data_root, f'{self.split}_index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"Warning: Index file not found at {index_path}")
            return []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load camera trajectory
        if 'trajectory' in sample:
            trajectory = np.array(sample['trajectory'], dtype=np.float32)
        elif 'trajectory_path' in sample:
            traj_path = os.path.join(self.data_root, sample['trajectory_path'])
            trajectory = np.load(traj_path).astype(np.float32)
        else:
            trajectory = np.zeros((self.num_frames, self.toric_dim), dtype=np.float32)

        # Resample to fixed number of frames if needed
        if trajectory.shape[0] != self.num_frames:
            trajectory = self._resample_trajectory(trajectory, self.num_frames)

        # Flatten to 1D vector for diffusion model
        y = torch.tensor(trajectory.flatten(), dtype=torch.float32)

        # Text description
        text = sample.get('text', sample.get('description', ''))

        # Shot type
        shot_type_str = sample.get('shot_type', '')
        shot_type = self.SHOT_TYPE_MAP.get(shot_type_str, -1)

        # Camera motion type
        motion_type_str = sample.get('camera_motion', sample.get('motion_type', ''))
        motion_type = self.MOTION_TYPE_MAP.get(motion_type_str, -1)

        return {
            'y': y,                    # (num_frames * toric_dim,) flattened trajectory
            'text': text,
            'shot_type': shot_type,
            'motion_type': motion_type,
            'sample_id': sample.get('id', idx),
        }

    def _resample_trajectory(self, trajectory: np.ndarray, target_frames: int) -> np.ndarray:
        """Resample a trajectory to a fixed number of frames via linear interpolation."""
        src_frames = trajectory.shape[0]
        src_t = np.linspace(0, 1, src_frames)
        tgt_t = np.linspace(0, 1, target_frames)

        resampled = np.zeros((target_frames, self.toric_dim), dtype=np.float32)
        for dim in range(self.toric_dim):
            resampled[:, dim] = np.interp(tgt_t, src_t, trajectory[:, dim])

        return resampled


def collate_fn(batch):
    """Custom collate function for the dataloader."""
    y = torch.stack([item['y'] for item in batch])
    texts = [item['text'] for item in batch]
    shot_types = torch.tensor([item['shot_type'] for item in batch], dtype=torch.long)
    motion_types = torch.tensor([item['motion_type'] for item in batch], dtype=torch.long)
    sample_ids = [item['sample_id'] for item in batch]

    return {
        'y': y,
        'texts': texts,
        'shot_types': shot_types,
        'motion_types': motion_types,
        'sample_ids': sample_ids,
    }
