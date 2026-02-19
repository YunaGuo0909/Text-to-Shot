"""
Camera Trajectory Denoiser Network.

A temporal Transformer-based denoiser that predicts clean camera trajectories
in Toric parameter space from noisy inputs. Conditioned on text embeddings,
shot type, and camera motion type via FiLM modulation.

The network processes per-frame Toric states (6-dim) through temporal
self-attention layers, enabling it to learn smooth, cinematographically
plausible camera motions.

Reference:
- Tevet, G., et al. (2022). Human Motion Diffusion Model (MDM). ICLR.
- Wang, Z., et al. (2024). DanceCamera3D. AAAI.
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


class TemporalTransformerBlock(nn.Module):
    """
    Transformer block with FiLM conditioning for temporal sequences.

    Applies multi-head self-attention across the time dimension,
    followed by a FiLM-conditioned feed-forward network.
    """

    def __init__(self, hidden_dim, condition_dim, num_heads=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
            nn.Dropout(dropout),
        )
        self.film = FiLMLayer(hidden_dim, condition_dim)

    def forward(self, x, condition):
        """
        Args:
            x: (batch, T, hidden_dim) temporal features
            condition: (batch, condition_dim) conditioning signal

        Returns:
            (batch, T, hidden_dim) updated features
        """
        B, T, D = x.shape

        # Self-attention with residual
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h

        # FiLM-conditioned feed-forward with residual
        h = self.norm2(x)
        h = self.ff(h)
        # Apply FiLM per frame
        h = h.reshape(B * T, D)
        cond_expanded = condition.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
        h = self.film(h, cond_expanded)
        h = h.reshape(B, T, D)
        x = x + h

        return x


class CameraTrajectoryDenoiser(nn.Module):
    """
    Denoiser for camera trajectory generation in Toric space.

    Generates smooth camera motion trajectories conditioned on textual
    scene descriptions, shot types, and camera motion types.

    Architecture:
    1. Per-frame linear projection from Toric space (6-dim) to hidden space
    2. Learnable temporal positional encoding
    3. N Transformer blocks with FiLM conditioning
    4. Per-frame linear projection back to Toric space

    Input/Output: flattened trajectory (batch, num_frames * toric_dim)
    Internally reshaped to (batch, num_frames, toric_dim) for temporal processing.
    """

    def __init__(
        self,
        toric_dim=6,
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
        """
        Args:
            toric_dim: Dimension of Toric camera state (6)
            num_frames: Number of trajectory frames
            hidden_dim: Hidden dimension for Transformer
            num_layers: Number of Transformer blocks
            num_heads: Number of attention heads
            text_dim: Dimension of text embedding (CLIP)
            timestep_dim: Dimension of diffusion timestep embedding
            num_shot_types: Number of shot type categories
            shot_type_dim: Dimension of shot type embedding
            num_motion_types: Number of camera motion type categories
            motion_type_dim: Dimension of motion type embedding
            dropout: Dropout rate
        """
        super().__init__()
        self.toric_dim = toric_dim
        self.num_frames = num_frames
        self.total_dim = toric_dim * num_frames

        # Conditioning dimensions
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

        # Per-frame input/output projections
        self.input_proj = nn.Linear(toric_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, toric_dim)

        # Learnable temporal positional encoding
        self.temporal_pe = nn.Parameter(torch.randn(1, num_frames, hidden_dim) * 0.02)

        # Condition projection into hidden space (injected at input)
        self.cond_proj = nn.Linear(condition_dim, hidden_dim)

        # Temporal Transformer blocks
        self.blocks = nn.ModuleList([
            TemporalTransformerBlock(
                hidden_dim=hidden_dim,
                condition_dim=condition_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, y_t, t, text_embed, shot_type=None, motion_type=None):
        """
        Predict clean trajectory y_0 from noisy y_t.

        Args:
            y_t: Noisy trajectory (batch, total_dim) where total_dim = num_frames * toric_dim
            t: Diffusion timestep (batch,)
            text_embed: Text embedding (batch, text_dim)
            shot_type: Shot type index (batch,) or None
            motion_type: Camera motion type index (batch,) or None

        Returns:
            y_0_pred: Predicted clean trajectory (batch, total_dim)
        """
        B = y_t.shape[0]

        # Reshape to per-frame representation
        x = y_t.reshape(B, self.num_frames, self.toric_dim)  # (B, T, 6)

        # Build conditioning signal
        t_emb = self.timestep_proj(self.timestep_embed(t))  # (B, timestep_dim)

        if shot_type is not None:
            s_emb = self.shot_type_embed(shot_type)
        else:
            s_emb = self.no_shot_type.unsqueeze(0).expand(B, -1)

        if motion_type is not None:
            m_emb = self.motion_type_embed(motion_type)
        else:
            m_emb = self.no_motion_type.unsqueeze(0).expand(B, -1)

        condition = torch.cat([text_embed, t_emb, s_emb, m_emb], dim=-1)  # (B, cond_dim)

        # Per-frame encoding + temporal position + global condition token
        h = self.input_proj(x)  # (B, T, hidden)
        h = h + self.temporal_pe[:, :self.num_frames, :]

        # Add condition as a bias to all frames
        cond_bias = self.cond_proj(condition).unsqueeze(1)  # (B, 1, hidden)
        h = h + cond_bias

        # Temporal Transformer blocks
        for block in self.blocks:
            h = block(h, condition)

        # Decode
        h = self.final_norm(h)
        y_0_pred = self.output_proj(h)  # (B, T, 6)

        # Flatten back
        return y_0_pred.reshape(B, self.total_dim)
