"""
Camera Utility Functions.

Provides helper functions for camera parameter processing,
rotation representations, and trajectory manipulation.

Reference:
- Zhou, Y., et al. (2019). On the Continuity of Rotation Representations
  in Neural Networks. CVPR.
"""

import numpy as np
import torch
from typing import Tuple


def rotation_matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to 6D continuous representation.

    Args:
        R: Rotation matrix (3, 3)

    Returns:
        r6d: 6D rotation representation (6,)
    """
    return R[:, :2].T.flatten()


def rotation_6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D continuous representation back to rotation matrix.

    Args:
        r6d: 6D rotation (6,)

    Returns:
        R: Rotation matrix (3, 3)
    """
    a1 = r6d[:3]
    a2 = r6d[3:6]

    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=1)


def rotation_6d_to_matrix_torch(r6d: torch.Tensor) -> torch.Tensor:
    """
    Batch conversion of 6D rotation to rotation matrices.

    Args:
        r6d: (batch, 6) 6D rotations

    Returns:
        R: (batch, 3, 3) rotation matrices
    """
    a1 = r6d[..., :3]
    a2 = r6d[..., 3:6]

    b1 = torch.nn.functional.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def normalize_trajectory(trajectory: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Normalize a camera trajectory to zero mean and unit variance.

    Args:
        trajectory: (T, 6) raw trajectory

    Returns:
        normalized: (T, 6) normalized trajectory
        stats: dict with 'mean' and 'std' for denormalization
    """
    mean = trajectory.mean(axis=0)
    std = trajectory.std(axis=0)
    std[std < 1e-8] = 1.0

    normalized = (trajectory - mean) / std
    stats = {'mean': mean, 'std': std}
    return normalized, stats


def denormalize_trajectory(trajectory: np.ndarray, stats: dict) -> np.ndarray:
    """
    Denormalize a camera trajectory.

    Args:
        trajectory: (T, 6) normalized trajectory
        stats: dict with 'mean' and 'std'

    Returns:
        (T, 6) denormalized trajectory
    """
    return trajectory * stats['std'] + stats['mean']
