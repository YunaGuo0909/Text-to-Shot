"""
Toric Camera Parameterization Utilities.

The Toric space parameterizes camera viewpoints relative to two reference
subjects, making it naturally suited for two-character scenarios.

Reference:
- Lino, C., & Christie, M. (2015). Intuitive and efficient camera control
  with the toric space. ACM Transactions on Graphics, 34(4), 1-12.
"""

import numpy as np
import torch
from typing import Tuple


def toric_to_camera_extrinsics(
    theta: float,
    phi: float,
    p_A: np.ndarray,
    p_B: np.ndarray,
    head_A_3d: np.ndarray,
    head_B_3d: np.ndarray,
    focal_length: float = 50.0,
    sensor_width: float = 36.0,
    image_width: int = 1920,
    image_height: int = 1080,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Toric camera parameters to camera extrinsics (rotation + translation).
    
    Args:
        theta: Azimuth (yaw) angle in Toric space
        phi: Elevation (pitch) angle in Toric space
        p_A: Normalized screen position of character A head (2,)
        p_B: Normalized screen position of character B head (2,)
        head_A_3d: 3D position of character A head (3,)
        head_B_3d: 3D position of character B head (3,)
        focal_length: Camera focal length in mm
        sensor_width: Camera sensor width in mm
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
    """
    # Midpoint between two subjects
    midpoint = (head_A_3d + head_B_3d) / 2.0
    
    # Distance between subjects
    dist = np.linalg.norm(head_A_3d - head_B_3d)
    
    # Camera position in Toric coordinates
    # Toric space places camera on a torus around the two subjects
    r = dist * 1.5  # Base distance from midpoint
    
    cam_x = midpoint[0] + r * np.cos(theta) * np.cos(phi)
    cam_y = midpoint[1] + r * np.sin(phi)
    cam_z = midpoint[2] + r * np.sin(theta) * np.cos(phi)
    
    cam_pos = np.array([cam_x, cam_y, cam_z])
    
    # Look-at direction (camera looks at midpoint)
    forward = midpoint - cam_pos
    forward = forward / np.linalg.norm(forward)
    
    # Up vector (world up)
    up = np.array([0.0, 1.0, 0.0])
    
    # Right vector
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # Recompute up
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Rotation matrix (camera-to-world)
    R = np.stack([right, up, -forward], axis=1)
    t = cam_pos
    
    return R, t


def camera_to_toric(
    R: np.ndarray,
    t: np.ndarray,
    head_A_3d: np.ndarray,
    head_B_3d: np.ndarray,
    intrinsics: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Convert camera extrinsics to Toric parameters.
    
    Args:
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
        head_A_3d: 3D position of character A head (3,)
        head_B_3d: 3D position of character B head (3,)
        intrinsics: Camera intrinsic matrix (3, 3)
        
    Returns:
        theta: Azimuth angle
        phi: Elevation angle
        p_A: Normalized screen position of A
        p_B: Normalized screen position of B
    """
    # Project heads to screen
    p_A_homo = intrinsics @ (R.T @ (head_A_3d - t))
    p_B_homo = intrinsics @ (R.T @ (head_B_3d - t))
    
    p_A = p_A_homo[:2] / p_A_homo[2]
    p_B = p_B_homo[:2] / p_B_homo[2]
    
    # Normalize to [0, 1]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    p_A_norm = np.array([(p_A[0] - cx) / fx, (p_A[1] - cy) / fy])
    p_B_norm = np.array([(p_B[0] - cx) / fx, (p_B[1] - cy) / fy])
    
    # Compute Toric angles
    midpoint = (head_A_3d + head_B_3d) / 2.0
    cam_to_mid = midpoint - t
    dist_xz = np.sqrt(cam_to_mid[0]**2 + cam_to_mid[2]**2)
    
    theta = np.arctan2(cam_to_mid[2], cam_to_mid[0])
    phi = np.arctan2(cam_to_mid[1], dist_xz)
    
    return theta, phi, p_A_norm, p_B_norm


def pack_toric_state(
    p_A: np.ndarray,
    p_B: np.ndarray,
    theta: float,
    phi: float,
) -> np.ndarray:
    """Pack Toric parameters into a 6D vector: x_C = {p_A, p_B, θ, φ}."""
    return np.array([p_A[0], p_A[1], p_B[0], p_B[1], theta, phi])


def unpack_toric_state(
    x_C: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Unpack 6D vector into Toric parameters."""
    p_A = x_C[:2]
    p_B = x_C[2:4]
    theta = x_C[4]
    phi = x_C[5]
    return p_A, p_B, theta, phi
