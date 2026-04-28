"""
Dataset for joint person-camera trajectory diffusion training.

Each sample contains:
- Person root trajectory: (T, 4) world positions + yaw [px, py, pz, yaw]
- Camera trajectory: (T, 6) camera state (tx, ty, tz, azimuth, elevation, roll)
- Text description
- Shot type label
- Camera motion type label

Joint representation: y = [person_flat (T*4), camera_flat (T*6)]

Backward compatible: old (T, 3) person files are zero-padded to (T, 4).
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from typing import Optional, Dict, List, Tuple


class JointTrajectoryDataset(Dataset):
    """
    Dataset for joint person-camera trajectory generation.

    Loads paired (person, camera) trajectories with text annotations.
    """

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
        person_dim: int = 3,
        camera_dim: int = 6,
        index_file: Optional[str] = None,
        norm_stats_path: Optional[str] = None,
    ):
        self.data_root = data_root
        self.split = split
        self.num_frames = num_frames
        self.person_dim = person_dim
        self.camera_dim = camera_dim
        self.person_total = person_dim * num_frames
        self.camera_total = camera_dim * num_frames
        self.total_dim = self.person_total + self.camera_total
        self.index_file = index_file

        self.norm_mean: Optional[torch.Tensor] = None
        self.norm_std: Optional[torch.Tensor] = None
        if norm_stats_path and os.path.exists(norm_stats_path):
            with open(norm_stats_path, 'r') as f:
                stats = json.load(f)
            self.norm_mean = torch.tensor(stats['mean'], dtype=torch.float32)
            self.norm_std = torch.tensor(stats['std'], dtype=torch.float32)
            print(f"Loaded norm stats from {norm_stats_path} ({stats.get('n_samples', '?')} samples)")
        elif norm_stats_path:
            print(f"Warning: norm_stats_path not found: {norm_stats_path}")

        self.samples = self._load_index()

    def _load_index(self) -> List[Dict]:
        index_name = self.index_file if self.index_file else f'{self.split}_index.json'
        index_path = os.path.join(self.data_root, index_name)
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

        # Load camera trajectory (T, 6)
        camera_traj = self._load_trajectory(
            sample, key='camera_trajectory_path', fallback_key='trajectory_path',
            dim=self.camera_dim)

        # Load person trajectory (T, 3)
        person_traj = self._load_trajectory(
            sample, key='person_trajectory_path', fallback_key=None,
            dim=self.person_dim)

        # Resample if needed
        if camera_traj.shape[0] != self.num_frames:
            camera_traj = self._resample(camera_traj, self.num_frames)
        if person_traj.shape[0] != self.num_frames:
            person_traj = self._resample(person_traj, self.num_frames)

        # Flatten and concatenate: [person_flat, camera_flat]
        person_flat = torch.tensor(person_traj.flatten(), dtype=torch.float32)
        camera_flat = torch.tensor(camera_traj.flatten(), dtype=torch.float32)
        y = torch.cat([person_flat, camera_flat], dim=0)

        # Normalize if stats available
        if self.norm_mean is not None:
            y = (y - self.norm_mean) / self.norm_std

        text = sample.get('text', sample.get('description', ''))
        shot_type = self.SHOT_TYPE_MAP.get(sample.get('shot_type', ''), -1)
        motion_type = self.MOTION_TYPE_MAP.get(
            sample.get('camera_motion', sample.get('motion_type', '')), -1)

        return {
            'y': y,
            'text': text,
            'shot_type': shot_type,
            'motion_type': motion_type,
            'sample_id': sample.get('id', idx),
        }

    def _load_trajectory(self, sample, key, fallback_key, dim):
        """Load a trajectory array from the sample entry.

        Handles backward compatibility: if loaded array has fewer columns
        than expected dim (e.g., old (T,3) files when person_dim=4),
        pads with zeros for missing columns.
        """
        path_key = key
        if path_key not in sample and fallback_key and fallback_key in sample:
            path_key = fallback_key

        if path_key in sample:
            traj_path = os.path.join(self.data_root, sample[path_key])
            if os.path.exists(traj_path):
                traj = np.load(traj_path).astype(np.float32)
                if traj.ndim == 1:
                    # Try to reshape; if loaded dim < expected, pad first
                    loaded_total = traj.shape[0]
                    loaded_dim = loaded_total // (loaded_total // dim) if dim > 0 else dim
                    # Best guess: reshape with the smaller dim, then pad
                    for try_dim in [dim, dim - 1, 3]:
                        if try_dim > 0 and loaded_total % try_dim == 0:
                            traj = traj.reshape(-1, try_dim)
                            break
                # Pad columns if loaded has fewer dims than expected
                if traj.ndim == 2 and traj.shape[1] < dim:
                    pad_width = dim - traj.shape[1]
                    traj = np.concatenate(
                        [traj, np.zeros((traj.shape[0], pad_width), dtype=np.float32)],
                        axis=1)
                return traj

        # Fallback: zeros
        return np.zeros((self.num_frames, dim), dtype=np.float32)

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        """Undo normalization. y: (..., total_dim)"""
        if self.norm_mean is None:
            return y
        return y * self.norm_std.to(y.device) + self.norm_mean.to(y.device)

    def _resample(self, trajectory: np.ndarray, target_frames: int) -> np.ndarray:
        src_frames = trajectory.shape[0]
        dim = trajectory.shape[1]
        src_t = np.linspace(0, 1, src_frames)
        tgt_t = np.linspace(0, 1, target_frames)
        resampled = np.zeros((target_frames, dim), dtype=np.float32)
        for d in range(dim):
            resampled[:, d] = np.interp(tgt_t, src_t, trajectory[:, d])
        return resampled


def collate_fn(batch):
    """Custom collate function."""
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
