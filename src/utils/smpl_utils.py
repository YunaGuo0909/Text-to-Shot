"""
SMPL-related Utility Functions.

Handles conversion between SMPL parameters and joint positions,
6D rotation representations, and pose vector construction.

Reference:
- Loper, M., et al. (2015). SMPL: A Skinned Multi-Person Linear Model.
- Zhou, Y., et al. (2019). On the Continuity of Rotation Representations 
  in Neural Networks. CVPR.
"""

import numpy as np
import torch
from typing import Tuple


def rotation_matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to 6D continuous representation.
    
    Uses the first two columns of the rotation matrix as the 
    representation, following Zhou et al. (2019).
    
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
    
    # Gram-Schmidt orthogonalization
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


def construct_pose_vector(
    joint_rotations_6d: np.ndarray,
    root_translation: np.ndarray,
    facing_direction_6d: np.ndarray,
    global_translation: np.ndarray,
) -> np.ndarray:
    """
    Construct the full 150-dim pose vector for one character.
    
    Structure:
    - Root translation embedded as 6D vector: [tx, ty, tz, 0, 0, 0] → (6,)
    - 22 joint rotations in 6D: (22 × 6 = 132)
    - Placement vector D: facing_6d + global_translation + zero_pad → (12,)
    Total: 6 + 132 + 12 = 150
    
    Args:
        joint_rotations_6d: (22, 6) joint rotations
        root_translation: (3,) root position
        facing_direction_6d: (6,) global facing in 6D
        global_translation: (3,) global root translation
        
    Returns:
        pose_vector: (150,) complete pose vector
    """
    # Root translation as 6D (pad with zeros)
    root_6d = np.concatenate([root_translation, np.zeros(3)])
    
    # Flatten joint rotations
    joints_flat = joint_rotations_6d.flatten()  # (132,)
    
    # Placement vector D (9-dim → 12-dim with zero padding)
    D = np.concatenate([facing_direction_6d, global_translation, np.zeros(3)])
    
    # Full vector
    pose_vector = np.concatenate([root_6d, joints_flat, D])
    
    assert pose_vector.shape == (150,), f"Expected 150-dim, got {pose_vector.shape}"
    return pose_vector


def parse_pose_vector(pose_vector: np.ndarray) -> dict:
    """
    Parse a 150-dim pose vector into its components.
    
    Args:
        pose_vector: (150,) complete pose vector
        
    Returns:
        Dictionary with parsed components
    """
    return {
        'root_translation': pose_vector[:3],
        'root_6d': pose_vector[:6],
        'joint_rotations_6d': pose_vector[6:138].reshape(22, 6),
        'placement_vector': pose_vector[138:150],
        'facing_direction_6d': pose_vector[138:144],
        'global_translation': pose_vector[144:147],
    }
