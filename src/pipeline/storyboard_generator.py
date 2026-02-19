"""
Camera Trajectory Generation Pipeline.

Orchestrates the full pipeline from scene description to multi-shot
camera trajectory generation. Handles sequential shot generation with
inter-shot transition smoothing.
"""

import torch
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field

from .shot_decomposer import ShotConfig, StoryboardPlan, SHOT_TYPE_MAP, CAMERA_MOTION_MAP
from .camera_trajectory import CameraTrajectory, CameraTrajectoryGenerator


@dataclass
class GeneratedShot:
    """A generated shot with camera trajectory."""
    shot_config: ShotConfig
    camera_trajectory: CameraTrajectory  # Generated camera motion trajectory
    toric_start: np.ndarray = None       # (6,) starting Toric state
    toric_end: np.ndarray = None         # (6,) ending Toric state


@dataclass
class GeneratedStoryboard:
    """Complete generated sequence with all shots."""
    plan: StoryboardPlan
    shots: List[GeneratedShot]


class TrajectoryPipeline:
    """
    Generates multi-shot camera trajectories from a StoryboardPlan.

    Uses the camera trajectory diffusion model to generate each shot's
    trajectory, with optional inter-shot smoothing to maintain spatial
    continuity across cuts.
    """

    def __init__(self, diffusion_model=None, text_encoder=None, device='cuda'):
        """
        Args:
            diffusion_model: Trained GaussianDiffusion model (None for rule-based mode)
            text_encoder: Text encoder (CLIP) for conditioning
            device: Computation device
        """
        self.diffusion_model = diffusion_model
        self.text_encoder = text_encoder
        self.device = device
        self.rule_based_generator = CameraTrajectoryGenerator(default_num_frames=48)

    @torch.no_grad()
    def generate(
        self,
        plan: StoryboardPlan,
        mode: str = "rule_based",
        smooth_transitions: bool = True,
    ) -> GeneratedStoryboard:
        """
        Generate camera trajectories for all shots.

        Args:
            plan: StoryboardPlan from ShotDecomposer
            mode: "diffusion" (learned model) or "rule_based" (motion profiles)
            smooth_transitions: Whether to smooth inter-shot transitions

        Returns:
            GeneratedStoryboard with all generated trajectories
        """
        generated_shots = []
        prev_shot = None

        for shot_config in plan.shots:
            if mode == "diffusion" and self.diffusion_model is not None:
                trajectory = self._generate_diffusion(shot_config, prev_shot)
            else:
                trajectory = self._generate_rule_based(shot_config, prev_shot)

            # Smooth transition from previous shot
            if smooth_transitions and prev_shot is not None:
                trajectory = self._smooth_transition(prev_shot, trajectory)

            shot = GeneratedShot(
                shot_config=shot_config,
                camera_trajectory=trajectory,
                toric_start=trajectory.trajectory[0].copy(),
                toric_end=trajectory.trajectory[-1].copy(),
            )
            generated_shots.append(shot)
            prev_shot = shot

        return GeneratedStoryboard(plan=plan, shots=generated_shots)

    def _generate_diffusion(self, shot_config: ShotConfig, prev_shot: Optional[GeneratedShot]):
        """Generate trajectory using the trained diffusion model."""
        text_embed = self._encode_text(shot_config.description)

        shot_type_idx = SHOT_TYPE_MAP.get(shot_config.shot_type, 1)
        shot_type = torch.tensor([shot_type_idx], device=self.device)

        motion_idx = CAMERA_MOTION_MAP.get(shot_config.camera_motion, 0)
        motion_type = torch.tensor([motion_idx], device=self.device)

        # Generate via diffusion
        y_0 = self.diffusion_model.sample(
            text_embed=text_embed,
            shot_type=shot_type,
            motion_type=motion_type,
            device=self.device,
        )

        # Reshape to trajectory
        toric_dim = self.diffusion_model.denoiser.toric_dim
        num_frames = self.diffusion_model.denoiser.num_frames
        traj_data = y_0.squeeze(0).cpu().numpy().reshape(num_frames, toric_dim)

        return CameraTrajectory(
            motion_type=shot_config.camera_motion,
            num_frames=num_frames,
            keyframes=traj_data[np.linspace(0, num_frames - 1, 4, dtype=int)],
            trajectory=traj_data,
            timestamps=np.linspace(0, 1, num_frames),
        )

    def _generate_rule_based(self, shot_config: ShotConfig, prev_shot: Optional[GeneratedShot]):
        """Generate trajectory using rule-based motion profiles."""
        if prev_shot is not None:
            start_state = prev_shot.toric_end
        else:
            start_state = np.array([0.3, 0.4, 0.7, 0.4, 0.0, 0.1])

        num_frames = int(shot_config.duration_hint * 24)  # 24 fps
        return self.rule_based_generator.generate(
            start_state=start_state,
            motion_type=shot_config.camera_motion,
            num_frames=max(num_frames, 12),
            intensity=1.0,
        )

    def _smooth_transition(self, prev_shot: GeneratedShot, curr_trajectory: CameraTrajectory):
        """
        Smooth the transition between two consecutive shots.

        Blends the first few frames of the current trajectory toward
        the end state of the previous shot for continuity.
        """
        blend_frames = min(6, curr_trajectory.num_frames // 4)
        prev_end = prev_shot.toric_end

        for i in range(blend_frames):
            alpha = i / blend_frames  # 0 → 1
            curr_trajectory.trajectory[i] = (
                (1 - alpha) * prev_end + alpha * curr_trajectory.trajectory[i]
            )

        return curr_trajectory

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text using CLIP text encoder."""
        if self.text_encoder is not None:
            tokens = self.text_encoder.tokenize([text])
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            return self.text_encoder.encode(tokens)
        else:
            return torch.randn(1, 512, device=self.device)
