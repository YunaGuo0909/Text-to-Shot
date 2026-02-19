"""
Temporal Interaction Modules for Camera Trajectory Generation.

Provides auxiliary modules for modeling temporal dependencies in
camera trajectories. These can be composed with the main denoiser
for enhanced trajectory coherence.

Note: The primary temporal modeling is handled by the Transformer
self-attention in CameraTrajectoryDenoiser. This module provides
optional inter-shot coherence mechanisms.
"""

import torch
import torch.nn as nn
from .film import FiLMLayer


class TemporalSmoothingModule(nn.Module):
    """
    Learnable temporal smoothing module.

    Applies a 1D convolution along the time axis to encourage
    smooth camera transitions, followed by FiLM conditioning.
    """

    def __init__(self, hidden_dim, condition_dim, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.temporal_conv = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=1,
        )
        self.film = FiLMLayer(hidden_dim, condition_dim)
        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, condition):
        """
        Args:
            x: (batch, T, hidden_dim) temporal features
            condition: (batch, condition_dim) conditioning

        Returns:
            (batch, T, hidden_dim) smoothed features
        """
        B, T, D = x.shape
        residual = x

        # Conv1d expects (B, C, T)
        h = x.permute(0, 2, 1)
        h = self.temporal_conv(h)
        h = h.permute(0, 2, 1)  # back to (B, T, D)

        h = self.activation(h)
        h = h.reshape(B * T, D)
        cond_expanded = condition.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
        h = self.film(h, cond_expanded)
        h = h.reshape(B, T, D)

        return self.norm(residual + h)


class InterShotCoherenceModule(nn.Module):
    """
    Ensures smooth camera transitions between consecutive shots.

    Takes the ending state of the previous shot's trajectory and
    produces a bias that guides the beginning of the next shot.
    """

    def __init__(self, toric_dim=6, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(toric_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.transition_gate = nn.Sequential(
            nn.Linear(hidden_dim, toric_dim),
            nn.Tanh(),
        )

    def forward(self, prev_end_state, next_start_state):
        """
        Compute a correction to smooth the shot transition.

        Args:
            prev_end_state: (batch, toric_dim) end of previous trajectory
            next_start_state: (batch, toric_dim) start of next trajectory

        Returns:
            correction: (batch, toric_dim) additive correction for next start
        """
        h = self.encoder(prev_end_state)
        correction = self.transition_gate(h)
        return correction
