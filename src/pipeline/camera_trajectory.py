"""
Camera Motion Trajectory Generator.

Extends static Toric camera states into temporal camera motion trajectories.
Given a starting camera configuration and a motion type, generates a sequence
of keyframe Toric parameters and interpolates them into a smooth trajectory.

Camera motion types supported:
  - static: No camera movement
  - dolly-in: Push camera toward subjects (推)
  - dolly-out: Pull camera away from subjects (拉)
  - pan-left: Rotate camera leftward (左摇)
  - pan-right: Rotate camera rightward (右摇)
  - crane-up: Raise camera upward (升)
  - crane-down: Lower camera downward (降)
  - track: Lateral tracking shot following action (跟踪)
  - orbit: Orbit around subjects (环绕)

This is a NEW module not present in the original paper, representing a
key extension from static shot generation to dynamic camera trajectories.

Reference:
- Lino, C., & Christie, M. (2015). Toric space. ACM TOG.
- Wang, Z., et al. (2024). DanceCamera3D. AAAI.
"""

import numpy as np
from scipy import interpolate
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch


# ============================================================
# Camera motion type definitions
# ============================================================

CAMERA_MOTION_TYPES = [
    "static",
    "dolly-in",
    "dolly-out",
    "pan-left",
    "pan-right",
    "crane-up",
    "crane-down",
    "track",
    "orbit",
]

CAMERA_MOTION_MAP = {name: idx for idx, name in enumerate(CAMERA_MOTION_TYPES)}


@dataclass
class CameraMotionProfile:
    """
    Defines how Toric parameters change over the duration of a shot
    for a given camera motion type.
    
    Toric state: x_C = (p_A_x, p_A_y, p_B_x, p_B_y, theta, phi)
    
    delta_* fields define the total change applied from start to end.
    ease_type controls the acceleration curve.
    """
    delta_theta: float = 0.0     # Change in azimuth (yaw)
    delta_phi: float = 0.0       # Change in elevation (pitch)
    delta_pA_x: float = 0.0     # Change in character A screen x
    delta_pA_y: float = 0.0     # Change in character A screen y
    delta_pB_x: float = 0.0     # Change in character B screen x
    delta_pB_y: float = 0.0     # Change in character B screen y
    ease_type: str = "ease-in-out"  # ease-in, ease-out, ease-in-out, linear
    num_keyframes: int = 4       # Number of keyframes to generate


# Pre-defined motion profiles for each camera motion type
MOTION_PROFILES = {
    "static": CameraMotionProfile(
        delta_theta=0, delta_phi=0,
        num_keyframes=2,
    ),
    "dolly-in": CameraMotionProfile(
        # Dolly in: characters get larger on screen → pA, pB move outward
        delta_pA_x=-0.08, delta_pB_x=0.08,
        delta_pA_y=-0.03, delta_pB_y=-0.03,
        ease_type="ease-in-out",
        num_keyframes=4,
    ),
    "dolly-out": CameraMotionProfile(
        # Dolly out: characters get smaller → pA, pB move inward
        delta_pA_x=0.06, delta_pB_x=-0.06,
        delta_pA_y=0.02, delta_pB_y=0.02,
        ease_type="ease-in-out",
        num_keyframes=4,
    ),
    "pan-left": CameraMotionProfile(
        delta_theta=-0.35,
        delta_pA_x=0.12, delta_pB_x=0.12,
        ease_type="ease-in-out",
        num_keyframes=4,
    ),
    "pan-right": CameraMotionProfile(
        delta_theta=0.35,
        delta_pA_x=-0.12, delta_pB_x=-0.12,
        ease_type="ease-in-out",
        num_keyframes=4,
    ),
    "crane-up": CameraMotionProfile(
        delta_phi=0.25,
        delta_pA_y=0.08, delta_pB_y=0.08,
        ease_type="ease-out",
        num_keyframes=4,
    ),
    "crane-down": CameraMotionProfile(
        delta_phi=-0.25,
        delta_pA_y=-0.08, delta_pB_y=-0.08,
        ease_type="ease-in",
        num_keyframes=4,
    ),
    "track": CameraMotionProfile(
        delta_theta=0.2,
        delta_pA_x=-0.05, delta_pB_x=-0.05,
        ease_type="linear",
        num_keyframes=5,
    ),
    "orbit": CameraMotionProfile(
        delta_theta=0.6,
        delta_phi=0.1,
        ease_type="ease-in-out",
        num_keyframes=6,
    ),
}


@dataclass
class CameraTrajectory:
    """A generated camera trajectory over time."""
    motion_type: str
    num_frames: int
    keyframes: np.ndarray       # (K, 6) Toric keyframe states
    trajectory: np.ndarray      # (T, 6) Interpolated smooth trajectory
    timestamps: np.ndarray      # (T,) Normalized time [0, 1]


class CameraTrajectoryGenerator:
    """
    Generates smooth camera motion trajectories from a starting Toric
    camera state and a specified motion type.
    
    Pipeline:
    1. Start with diffusion-generated static Toric state x_C
    2. Apply motion profile to generate K keyframes
    3. Use Catmull-Rom / cubic spline interpolation for smooth trajectory
    4. Apply easing functions for cinematic acceleration curves
    """

    def __init__(self, default_num_frames: int = 60):
        """
        Args:
            default_num_frames: Default trajectory length (frames at 24fps → ~2.5s)
        """
        self.default_num_frames = default_num_frames

    def generate(
        self,
        start_state: np.ndarray,
        motion_type: str = "static",
        num_frames: Optional[int] = None,
        intensity: float = 1.0,
    ) -> CameraTrajectory:
        """
        Generate a camera trajectory from a starting Toric state.
        
        Args:
            start_state: (6,) initial Toric camera state
                         [p_A_x, p_A_y, p_B_x, p_B_y, theta, phi]
            motion_type: Type of camera motion to generate
            num_frames: Number of output frames (None = use default)
            intensity: Scale factor for motion magnitude (0.5 = subtle, 2.0 = dramatic)
            
        Returns:
            CameraTrajectory with keyframes and interpolated trajectory
        """
        if num_frames is None:
            num_frames = self.default_num_frames

        if motion_type not in MOTION_PROFILES:
            print(f"Warning: Unknown motion type '{motion_type}', using 'static'")
            motion_type = "static"

        profile = MOTION_PROFILES[motion_type]

        # Step 1: Generate keyframes
        keyframes = self._generate_keyframes(start_state, profile, intensity)

        # Step 2: Interpolate to smooth trajectory
        trajectory, timestamps = self._interpolate_trajectory(
            keyframes, num_frames, profile.ease_type
        )

        return CameraTrajectory(
            motion_type=motion_type,
            num_frames=num_frames,
            keyframes=keyframes,
            trajectory=trajectory,
            timestamps=timestamps,
        )

    def generate_from_torch(
        self,
        camera_state_tensor: torch.Tensor,
        motion_type: str = "static",
        num_frames: Optional[int] = None,
        intensity: float = 1.0,
    ) -> CameraTrajectory:
        """Generate trajectory from a PyTorch tensor camera state."""
        start_state = camera_state_tensor.detach().cpu().numpy()
        return self.generate(start_state, motion_type, num_frames, intensity)

    def _generate_keyframes(
        self,
        start_state: np.ndarray,
        profile: CameraMotionProfile,
        intensity: float,
    ) -> np.ndarray:
        """
        Generate K keyframe Toric states based on motion profile.
        
        Keyframes are spaced along the motion arc with appropriate
        ease-in/out distribution.
        
        Args:
            start_state: (6,) starting Toric state
            profile: Motion profile defining parameter changes
            intensity: Scaling factor
            
        Returns:
            keyframes: (K, 6) Toric keyframe states
        """
        K = profile.num_keyframes
        keyframes = np.zeros((K, 6))

        # Compute delta vector
        delta = np.array([
            profile.delta_pA_x,
            profile.delta_pA_y,
            profile.delta_pB_x,
            profile.delta_pB_y,
            profile.delta_theta,
            profile.delta_phi,
        ]) * intensity

        # Distribute keyframes along the motion arc
        t_keyframes = np.linspace(0, 1, K)

        for i, t in enumerate(t_keyframes):
            # Apply easing to the progress value
            t_eased = self._apply_easing(t, profile.ease_type)
            keyframes[i] = start_state + delta * t_eased

        return keyframes

    def _interpolate_trajectory(
        self,
        keyframes: np.ndarray,
        num_frames: int,
        ease_type: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate keyframes into smooth trajectory using cubic splines.
        
        Uses scipy's CubicSpline for C2-continuous interpolation,
        ensuring smooth velocity and acceleration transitions.
        
        Args:
            keyframes: (K, 6) keyframe Toric states
            num_frames: Number of output frames
            ease_type: Easing function type
            
        Returns:
            trajectory: (T, 6) smooth trajectory
            timestamps: (T,) normalized time values
        """
        K = keyframes.shape[0]

        if K < 2:
            # Single keyframe → static
            trajectory = np.tile(keyframes[0], (num_frames, 1))
            timestamps = np.linspace(0, 1, num_frames)
            return trajectory, timestamps

        # Keyframe time positions
        t_key = np.linspace(0, 1, K)
        # Output time positions
        t_out = np.linspace(0, 1, num_frames)

        trajectory = np.zeros((num_frames, 6))

        # Interpolate each Toric parameter independently
        for dim in range(6):
            if K >= 4:
                # Cubic spline interpolation (C2 continuous)
                cs = interpolate.CubicSpline(
                    t_key, keyframes[:, dim],
                    bc_type='clamped'  # Zero velocity at endpoints → smooth start/stop
                )
                trajectory[:, dim] = cs(t_out)
            else:
                # Fall back to linear for very few keyframes
                trajectory[:, dim] = np.interp(t_out, t_key, keyframes[:, dim])

        timestamps = t_out
        return trajectory, timestamps

    def _apply_easing(self, t: float, ease_type: str) -> float:
        """
        Apply easing function to a normalized time value [0, 1].
        
        Easing functions create natural-looking acceleration curves
        that mimic real camera movements.
        
        Args:
            t: Normalized time in [0, 1]
            ease_type: Type of easing function
            
        Returns:
            Eased time value in [0, 1]
        """
        if ease_type == "linear":
            return t
        elif ease_type == "ease-in":
            # Quadratic ease in (slow start)
            return t * t
        elif ease_type == "ease-out":
            # Quadratic ease out (slow end)
            return 1 - (1 - t) * (1 - t)
        elif ease_type == "ease-in-out":
            # Smooth-step (slow start and end)
            return 3 * t * t - 2 * t * t * t
        else:
            return t

    @staticmethod
    def compute_trajectory_smoothness(trajectory: np.ndarray) -> dict:
        """
        Compute smoothness metrics for a camera trajectory.
        
        Useful for evaluation — lower jerk = smoother motion.
        
        Args:
            trajectory: (T, 6) camera trajectory
            
        Returns:
            Dictionary with smoothness metrics
        """
        # Velocity (first derivative)
        velocity = np.diff(trajectory, axis=0)
        # Acceleration (second derivative)
        acceleration = np.diff(velocity, axis=0)
        # Jerk (third derivative)
        jerk = np.diff(acceleration, axis=0)

        return {
            "mean_velocity": np.mean(np.linalg.norm(velocity, axis=1)),
            "max_velocity": np.max(np.linalg.norm(velocity, axis=1)),
            "mean_acceleration": np.mean(np.linalg.norm(acceleration, axis=1)),
            "max_acceleration": np.max(np.linalg.norm(acceleration, axis=1)),
            "mean_jerk": np.mean(np.linalg.norm(jerk, axis=1)),
            "max_jerk": np.max(np.linalg.norm(jerk, axis=1)),
            "total_path_length": np.sum(np.linalg.norm(velocity, axis=1)),
        }
