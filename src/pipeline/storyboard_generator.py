"""
Storyboard Generator Module.

Orchestrates the full pipeline from scene description to multi-shot
storyboard generation. Handles sequential shot generation with 
inter-shot coherence constraints.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field

from .shot_decomposer import ShotConfig, StoryboardPlan, SHOT_TYPE_MAP
from .camera_trajectory import CameraTrajectory, CameraTrajectoryGenerator


@dataclass
class GeneratedShot:
    """A generated shot with 3D character poses, camera configuration, and motion trajectory."""
    shot_config: ShotConfig
    char_a_pose: torch.Tensor           # (150,) SMPL pose vector
    char_b_pose: torch.Tensor           # (150,) SMPL pose vector
    camera_state: torch.Tensor          # (6,) Toric camera parameters (static start pose)
    camera_trajectory: Optional[CameraTrajectory] = None  # 🆕 Camera motion trajectory
    

@dataclass
class GeneratedStoryboard:
    """Complete generated storyboard with all shots."""
    plan: StoryboardPlan
    shots: List[GeneratedShot]


class StoryboardGenerator:
    """
    Generates a multi-shot storyboard from a StoryboardPlan.
    
    Uses the joint character-camera diffusion model to generate each shot,
    with optional inter-shot coherence constraints to maintain spatial
    continuity across the storyboard.
    """

    def __init__(self, diffusion_model, text_encoder, device='cuda'):
        """
        Args:
            diffusion_model: Trained GaussianDiffusion model
            text_encoder: Text encoder (CLIP) for conditioning
            device: Computation device
        """
        self.diffusion_model = diffusion_model
        self.text_encoder = text_encoder
        self.device = device
        self.char_pose_dim = diffusion_model.denoiser.char_pose_dim

    @torch.no_grad()
    def generate(
        self,
        plan: StoryboardPlan,
        coherence_weight: float = 0.3,
        use_coherence: bool = True,
    ) -> GeneratedStoryboard:
        """
        Generate all shots in a storyboard plan.
        
        Args:
            plan: StoryboardPlan from ShotDecomposer
            coherence_weight: Weight for inter-shot coherence guidance
            use_coherence: Whether to apply coherence constraints
            
        Returns:
            GeneratedStoryboard with all generated shots
        """
        generated_shots = []
        prev_shot = None

        for shot_config in plan.shots:
            # Encode text description
            text_embed = self._encode_text(shot_config.description)
            
            # Get shot type index
            shot_type_idx = SHOT_TYPE_MAP.get(shot_config.shot_type, 1)
            shot_type = torch.tensor([shot_type_idx], device=self.device)

            # Generate shot
            if use_coherence and prev_shot is not None:
                generated = self._generate_with_coherence(
                    text_embed, shot_type, prev_shot, coherence_weight
                )
            else:
                generated = self._generate_single(text_embed, shot_type)

            # Parse output
            char_a_pose = generated[:, :self.char_pose_dim].squeeze(0)
            char_b_pose = generated[:, self.char_pose_dim:self.char_pose_dim * 2].squeeze(0)
            camera_state = generated[:, self.char_pose_dim * 2:].squeeze(0)

            shot = GeneratedShot(
                shot_config=shot_config,
                char_a_pose=char_a_pose,
                char_b_pose=char_b_pose,
                camera_state=camera_state,
            )
            generated_shots.append(shot)
            prev_shot = shot

        return GeneratedStoryboard(plan=plan, shots=generated_shots)

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text description using CLIP text encoder."""
        # Placeholder - will be replaced with actual CLIP encoding
        # text_embed shape: (1, text_dim)
        tokens = self.text_encoder.tokenize([text])
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        text_embed = self.text_encoder.encode(tokens)
        return text_embed

    def _generate_single(self, text_embed, shot_type):
        """Generate a single shot without coherence constraints."""
        return self.diffusion_model.sample(
            text_embed=text_embed,
            shot_type=shot_type,
            device=self.device,
        )

    def _generate_with_coherence(
        self,
        text_embed: torch.Tensor,
        shot_type: torch.Tensor,
        prev_shot: GeneratedShot,
        coherence_weight: float,
    ) -> torch.Tensor:
        """
        Generate a shot with inter-shot coherence guidance.
        
        Applies soft constraints during the reverse diffusion process
        to maintain spatial continuity with the previous shot:
        1. Character positions should be roughly consistent
        2. Camera should respect the 180-degree rule
        
        Args:
            text_embed: Text conditioning
            shot_type: Shot type index
            prev_shot: Previously generated shot
            coherence_weight: Strength of coherence guidance
            
        Returns:
            Generated shot configuration tensor
        """
        batch_size = text_embed.shape[0]
        total_dim = self.diffusion_model.denoiser.total_dim
        model = self.diffusion_model

        # Start from noise
        y_t = torch.randn(batch_size, total_dim, device=self.device)

        # Previous shot reference for coherence
        prev_char_a = prev_shot.char_a_pose.unsqueeze(0)
        prev_char_b = prev_shot.char_b_pose.unsqueeze(0)

        for t in reversed(range(model.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # Standard denoising step
            y_0_pred = model.denoiser(y_t, t_batch, text_embed, shot_type=shot_type)

            # Apply coherence guidance on the prediction
            if t > model.num_timesteps * 0.3:  # Only in early (noisy) steps
                # Extract predicted character global positions (last 12 dims = placement vector)
                pred_a_pos = y_0_pred[:, self.char_pose_dim - 12:self.char_pose_dim]
                pred_b_pos = y_0_pred[:, self.char_pose_dim * 2 - 12:self.char_pose_dim * 2]
                prev_a_pos = prev_char_a[:, -12:]
                prev_b_pos = prev_char_b[:, -12:]

                # Soft position coherence
                pos_diff_a = coherence_weight * (prev_a_pos - pred_a_pos)
                pos_diff_b = coherence_weight * (prev_b_pos - pred_b_pos)

                # Apply correction to placement vectors
                y_0_pred[:, self.char_pose_dim - 12:self.char_pose_dim] += pos_diff_a
                y_0_pred[:, self.char_pose_dim * 2 - 12:self.char_pose_dim * 2] += pos_diff_b

            # Compute posterior mean and sample
            posterior_mean = (
                model.posterior_mean_coef1[t_batch].unsqueeze(-1) * y_0_pred +
                model.posterior_mean_coef2[t_batch].unsqueeze(-1) * y_t
            )

            if t > 0:
                noise = torch.randn_like(y_t)
                posterior_var = model.betas[t_batch].unsqueeze(-1)
                y_t = posterior_mean + torch.sqrt(posterior_var) * noise
            else:
                y_t = posterior_mean

        return y_t
