"""
Person (character) motion trajectory generator.

Produces a simple 3D trajectory (T, 3) from a text description of person motion.
Rule-based: keywords (walk, run, stand, move left/right, etc.) map to
straight lines, arcs, or static points. Intended for single-person dynamic
storyboard: person is represented as a cube or point in 3D; no skeletal animation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# Default person position in world (used when static). Camera often at ~(0.3, 0.4, 0.7) looking toward scene
DEFAULT_PERSON_ORIGIN = np.array([0.0, 0.0, 0.0], dtype=np.float32)
# Scale of movement (world units per "unit" of motion)
MOTION_SCALE = 0.5


def _normalized_lerp(t: np.ndarray, ease: str = "linear") -> np.ndarray:
    """Map t in [0,1] to eased value in [0,1]."""
    t = np.clip(t, 0.0, 1.0)
    if ease == "linear":
        return t
    if ease == "ease-in-out":
        return t * t * (3.0 - 2.0 * t)
    if ease == "ease-in":
        return t * t
    if ease == "ease-out":
        return 1.0 - (1.0 - t) ** 2
    return t


def classify_person_motion(description: str) -> str:
    """
    Classify person_motion_description into a motion type for rule-based trajectory.

    Returns one of: static, walk_forward, walk_backward, walk_left, walk_right,
    run_forward, move_left, move_right, turn, arc_left, arc_right.
    """
    if not description or not description.strip():
        return "static"
    text = description.lower().strip()

    if "stand" in text or "still" in text or "static" in text or "stays" in text or "remain" in text:
        return "static"
    if "run" in text:
        if "forward" in text or "toward" in text or "ahead" in text:
            return "run_forward"
        if "back" in text or "away" in text:
            return "walk_backward"
        return "run_forward"
    if "walk" in text or "move" in text or "go" in text:
        if "forward" in text or "toward" in text or "ahead" in text or "in" in text:
            return "walk_forward"
        if "back" in text or "away" in text or "backward" in text:
            return "walk_backward"
        if "left" in text:
            return "walk_left"
        if "right" in text:
            return "walk_right"
        return "walk_forward"
    if "left" in text and ("move" in text or "step" in text or "go" in text):
        return "walk_left"
    if "right" in text and ("move" in text or "step" in text or "go" in text):
        return "walk_right"
    if "turn" in text or "rotate" in text:
        if "left" in text:
            return "arc_left"
        if "right" in text:
            return "arc_right"
        return "arc_right"
    if "circle" in text or "arc" in text:
        if "left" in text:
            return "arc_left"
        return "arc_right"
    if "forward" in text or "toward" in text or "camera" in text:
        return "walk_forward"
    if "back" in text or "away" in text:
        return "walk_backward"

    return "static"


class PersonTrajectoryGenerator:
    """
    Generates (T, 3) person trajectory in world coordinates from a text description.

    World convention: X right, Y up or forward, Z depth (or Y forward, Z up depending on
    camera_trajectory / visualize_3d). We use same as visualize_3d: positions (tx, ty, tz)
    from camera trajectory are camera positions. Person at origin (0,0,0) is typically
    "in front" of default camera. So person moving "forward" can be +Z or +Y depending
    on project convention. From camera_trajectory and typical 3D: often X right, Y up, Z
    toward scene. So "person walks toward camera" = person moves in -Z. "Person walks
    forward" (into scene) = +Z. We'll use: X right, Y up, Z forward (into scene).
    Walk forward -> +Z, walk backward -> -Z, walk left -> -X, walk right -> +X.
    """

    def __init__(
        self,
        num_frames: int = 48,
        origin: Optional[np.ndarray] = None,
        motion_scale: float = MOTION_SCALE,
    ):
        self.num_frames = num_frames
        self.origin = origin if origin is not None else DEFAULT_PERSON_ORIGIN.copy()
        self.motion_scale = motion_scale

    def generate(
        self,
        person_motion_description: str,
        num_frames: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate (T, 3) trajectory from text description.

        Args:
            person_motion_description: e.g. "person walks toward camera", "character stands still"
            num_frames: Override default T (default 48).

        Returns:
            trajectory: (T, 3) float32 array of world positions (x, y, z).
        """
        T = num_frames if num_frames is not None else self.num_frames
        motion_type = classify_person_motion(person_motion_description or "")

        t = np.linspace(0.0, 1.0, T, dtype=np.float32)
        s = _normalized_lerp(t, "ease-in-out") * self.motion_scale

        if motion_type == "static":
            return np.tile(self.origin, (T, 1))

        if motion_type == "walk_forward":
            # +Z into scene
            traj = self.origin + np.stack([np.zeros(T), np.zeros(T), s], axis=1)
            return traj.astype(np.float32)
        if motion_type == "walk_backward":
            traj = self.origin + np.stack([np.zeros(T), np.zeros(T), -s], axis=1)
            return traj.astype(np.float32)
        if motion_type == "run_forward":
            traj = self.origin + np.stack([np.zeros(T), np.zeros(T), s * 1.5], axis=1)
            return traj.astype(np.float32)
        if motion_type == "walk_left":
            traj = self.origin + np.stack([-s, np.zeros(T), np.zeros(T)], axis=1)
            return traj.astype(np.float32)
        if motion_type == "walk_right":
            traj = self.origin + np.stack([s, np.zeros(T), np.zeros(T)], axis=1)
            return traj.astype(np.float32)
        if motion_type == "move_left":
            traj = self.origin + np.stack([-s, np.zeros(T), np.zeros(T)], axis=1)
            return traj.astype(np.float32)
        if motion_type == "move_right":
            traj = self.origin + np.stack([s, np.zeros(T), np.zeros(T)], axis=1)
            return traj.astype(np.float32)
        if motion_type == "arc_left":
            # Small arc: move in X and Z
            theta = t * np.pi * 0.5
            traj = self.origin + self.motion_scale * np.stack([
                -np.sin(theta),
                np.zeros(T),
                np.cos(theta) - 1.0,
            ], axis=1)
            return traj.astype(np.float32)
        if motion_type == "arc_right":
            theta = t * np.pi * 0.5
            traj = self.origin + self.motion_scale * np.stack([
                np.sin(theta),
                np.zeros(T),
                np.cos(theta) - 1.0,
            ], axis=1)
            return traj.astype(np.float32)

        return np.tile(self.origin, (T, 1)).astype(np.float32)
