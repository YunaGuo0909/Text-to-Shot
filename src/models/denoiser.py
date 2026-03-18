"""
Joint Person-Camera Trajectory Denoiser.

Dual-branch Transformer that jointly denoises person root trajectory (T, 3)
and camera trajectory (T, 6) with cross-attention between branches.

The person branch and camera branch each have self-attention for temporal
coherence, plus cross-attention so camera motion is aware of person motion
and vice versa. Both branches are conditioned on text (CLIP), diffusion
timestep, shot type, and camera motion type via FiLM modulation.

Reference:
- Tevet, G., et al. (2022). Human Motion Diffusion Model (MDM). ICLR.
- Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
"""

import torch
import torch.nn as nn
import math
from .film import FiLMLayer


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1).float() * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class DualBranchBlock(nn.Module):
    """
    One layer of the dual-branch Transformer.

    Each branch (person, camera) has:
    1. Self-attention over time
    2. Cross-attention to the other branch
    3. FiLM-conditioned feed-forward network
    """

    def __init__(self, hidden_dim, condition_dim, num_heads=4, ff_mult=4, dropout=0.1):
        super().__init__()

        # Person branch
        self.person_norm1 = nn.LayerNorm(hidden_dim)
        self.person_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.person_norm2 = nn.LayerNorm(hidden_dim)
        self.person_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.person_norm3 = nn.LayerNorm(hidden_dim)
        self.person_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
            nn.Dropout(dropout),
        )
        self.person_film = FiLMLayer(hidden_dim, condition_dim)

        # Camera branch
        self.camera_norm1 = nn.LayerNorm(hidden_dim)
        self.camera_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.camera_norm2 = nn.LayerNorm(hidden_dim)
        self.camera_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.camera_norm3 = nn.LayerNorm(hidden_dim)
        self.camera_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
            nn.Dropout(dropout),
        )
        self.camera_film = FiLMLayer(hidden_dim, condition_dim)

    def forward(self, person_h, camera_h, condition):
        """
        Args:
            person_h: (B, T, H) person branch features
            camera_h: (B, T, H) camera branch features
            condition: (B, C) conditioning vector

        Returns:
            person_h, camera_h: updated features
        """
        B, T, H = person_h.shape

        # --- Person branch ---
        # Self-attention
        h = self.person_norm1(person_h)
        h, _ = self.person_self_attn(h, h, h)
        person_h = person_h + h

        # Cross-attention (person attends to camera)
        h = self.person_norm2(person_h)
        cam_kv = self.camera_norm2(camera_h)
        h, _ = self.person_cross_attn(h, cam_kv, cam_kv)
        person_h = person_h + h

        # FiLM-conditioned FFN
        h = self.person_norm3(person_h)
        h = self.person_ff(h)
        h = h.reshape(B * T, H)
        cond_exp = condition.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
        h = self.person_film(h, cond_exp)
        person_h = person_h + h.reshape(B, T, H)

        # --- Camera branch ---
        # Self-attention
        h = self.camera_norm1(camera_h)
        h, _ = self.camera_self_attn(h, h, h)
        camera_h = camera_h + h

        # Cross-attention (camera attends to person)
        h = self.camera_norm2(camera_h)
        per_kv = self.person_norm2(person_h)
        h, _ = self.camera_cross_attn(h, per_kv, per_kv)
        camera_h = camera_h + h

        # FiLM-conditioned FFN
        h = self.camera_norm3(camera_h)
        h = self.camera_ff(h)
        h = h.reshape(B * T, H)
        h = self.camera_film(h, cond_exp)
        camera_h = camera_h + h.reshape(B, T, H)

        return person_h, camera_h


class JointTrajectoryDenoiser(nn.Module):
    """
    Dual-branch denoiser for joint person-camera trajectory generation.

    Takes a noisy joint trajectory y_t = [person_flat, camera_flat] and
    predicts the clean trajectory y_0. Person branch (T, 3) and camera
    branch (T, 6) interact via cross-attention at every layer.

    Input/Output: flattened joint vector (B, T*person_dim + T*camera_dim)
    """

    def __init__(
        self,
        person_dim=3,
        camera_dim=6,
        num_frames=48,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        text_dim=512,
        timestep_dim=128,
        num_shot_types=5,
        shot_type_dim=64,
        num_motion_types=9,
        motion_type_dim=64,
        dropout=0.1,
    ):
        super().__init__()
        self.person_dim = person_dim
        self.camera_dim = camera_dim
        self.num_frames = num_frames
        self.person_total = person_dim * num_frames
        self.camera_total = camera_dim * num_frames
        self.total_dim = self.person_total + self.camera_total

        # Conditioning
        condition_dim = text_dim + timestep_dim + shot_type_dim + motion_type_dim

        # Timestep embedding
        self.timestep_embed = SinusoidalPositionEmbedding(timestep_dim)
        self.timestep_proj = nn.Sequential(
            nn.Linear(timestep_dim, timestep_dim),
            nn.SiLU(),
            nn.Linear(timestep_dim, timestep_dim),
        )

        # Shot type embedding
        self.shot_type_embed = nn.Embedding(num_shot_types, shot_type_dim)
        self.no_shot_type = nn.Parameter(torch.zeros(shot_type_dim))

        # Camera motion type embedding
        self.motion_type_embed = nn.Embedding(num_motion_types, motion_type_dim)
        self.no_motion_type = nn.Parameter(torch.zeros(motion_type_dim))

        # Per-frame input projections
        self.person_input_proj = nn.Linear(person_dim, hidden_dim)
        self.camera_input_proj = nn.Linear(camera_dim, hidden_dim)

        # Shared temporal positional encoding
        self.temporal_pe = nn.Parameter(torch.randn(1, num_frames, hidden_dim) * 0.02)

        # Condition projection (injected as bias)
        self.cond_proj = nn.Linear(condition_dim, hidden_dim)

        # Dual-branch Transformer blocks
        self.blocks = nn.ModuleList([
            DualBranchBlock(
                hidden_dim=hidden_dim,
                condition_dim=condition_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output projections
        self.person_final_norm = nn.LayerNorm(hidden_dim)
        self.camera_final_norm = nn.LayerNorm(hidden_dim)
        self.person_output_proj = nn.Linear(hidden_dim, person_dim)
        self.camera_output_proj = nn.Linear(hidden_dim, camera_dim)

    def forward(self, y_t, t, text_embed, shot_type=None, motion_type=None):
        """
        Predict clean joint trajectory y_0 from noisy y_t.

        Args:
            y_t: (B, total_dim) noisy joint trajectory
                 [person_flat (T*3), camera_flat (T*6)]
            t: (B,) diffusion timestep
            text_embed: (B, text_dim)
            shot_type: (B,) or None
            motion_type: (B,) or None

        Returns:
            y_0_pred: (B, total_dim) predicted clean joint trajectory
        """
        B = y_t.shape[0]

        # Split into person and camera
        person_flat = y_t[:, :self.person_total]
        camera_flat = y_t[:, self.person_total:]

        person_x = person_flat.reshape(B, self.num_frames, self.person_dim)
        camera_x = camera_flat.reshape(B, self.num_frames, self.camera_dim)

        # Build conditioning
        t_emb = self.timestep_proj(self.timestep_embed(t))

        if shot_type is not None:
            s_emb = self.shot_type_embed(shot_type)
        else:
            s_emb = self.no_shot_type.unsqueeze(0).expand(B, -1)

        if motion_type is not None:
            m_emb = self.motion_type_embed(motion_type)
        else:
            m_emb = self.no_motion_type.unsqueeze(0).expand(B, -1)

        condition = torch.cat([text_embed, t_emb, s_emb, m_emb], dim=-1)

        # Project to hidden space + temporal PE + condition bias
        cond_bias = self.cond_proj(condition).unsqueeze(1)
        pe = self.temporal_pe[:, :self.num_frames, :]

        person_h = self.person_input_proj(person_x) + pe + cond_bias
        camera_h = self.camera_input_proj(camera_x) + pe + cond_bias

        # Dual-branch Transformer
        for block in self.blocks:
            person_h, camera_h = block(person_h, camera_h, condition)

        # Decode
        person_out = self.person_output_proj(self.person_final_norm(person_h))
        camera_out = self.camera_output_proj(self.camera_final_norm(camera_h))

        # Flatten and concatenate
        person_flat = person_out.reshape(B, self.person_total)
        camera_flat = camera_out.reshape(B, self.camera_total)

        return torch.cat([person_flat, camera_flat], dim=-1)
