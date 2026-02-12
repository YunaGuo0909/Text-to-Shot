"""
Joint Character-Camera Denoiser Network.

Three parallel Transformer branches encode intermediate embeddings for
Character A, Character B, and the Camera. Three pairwise interaction
modules exchange residual messages between entity pairs (A↔B, A↔C, B↔C)
to capture mutual dependencies under textual guidance.

Reference:
- "From Script to Shot" (SIGGRAPH 2026) Section 3.3-3.4
"""

import torch
import torch.nn as nn
import math
from .film import FiLMLayer
from .interaction import CharacterCharacterInteraction, CameraCharacterInteraction


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

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


class EntityBranch(nn.Module):
    """
    Single entity processing branch (Transformer-based encoder-decoder).
    
    Processes one entity (Character A, Character B, or Camera) through
    a series of MLP layers with FiLM conditioning.
    """

    def __init__(self, input_dim, hidden_dim, condition_dim, num_layers=4):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
            self.film_layers.append(FiLMLayer(hidden_dim, condition_dim))

        self.decoder = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, condition):
        """Encode input to hidden embedding."""
        h = self.encoder(x)
        for layer, film in zip(self.layers, self.film_layers):
            residual = h
            h = layer(h)
            h = film(h, condition)
            h = h + residual  # Residual connection
        return h

    def decode(self, h):
        """Decode hidden embedding to output."""
        return self.decoder(h)


class JointDenoiser(nn.Module):
    """
    Joint denoiser for simultaneous character-camera generation.
    
    Architecture:
    1. Three parallel branches: Character A, Character B, Camera
    2. Each branch encodes input → hidden embedding
    3. Pairwise interaction modules refine embeddings
    4. Decoders predict clean outputs
    """

    def __init__(
        self,
        char_pose_dim=150,
        camera_dim=6,
        hidden_dim=512,
        num_layers=4,
        text_dim=512,
        timestep_dim=128,
        num_shot_types=5,
        shot_type_dim=64,
    ):
        """
        Args:
            char_pose_dim: Dimension of character pose vector (25 joints × 6D = 150)
            camera_dim: Dimension of camera state in Toric space (6)
            hidden_dim: Hidden dimension for all branches
            num_layers: Number of layers per branch
            text_dim: Dimension of text embedding from CLIP
            timestep_dim: Dimension of timestep embedding
            num_shot_types: Number of shot type categories
            shot_type_dim: Dimension of shot type embedding
        """
        super().__init__()
        self.char_pose_dim = char_pose_dim
        self.camera_dim = camera_dim
        self.total_dim = char_pose_dim * 2 + camera_dim  # 150 + 150 + 6 = 306

        # Conditioning
        condition_dim = text_dim + timestep_dim + shot_type_dim
        self.timestep_embed = SinusoidalPositionEmbedding(timestep_dim)
        self.timestep_proj = nn.Linear(timestep_dim, timestep_dim)
        self.shot_type_embed = nn.Embedding(num_shot_types, shot_type_dim)
        self.no_shot_type_embed = nn.Parameter(torch.zeros(shot_type_dim))

        # Three parallel branches
        self.branch_A = EntityBranch(char_pose_dim, hidden_dim, condition_dim, num_layers)
        self.branch_B = EntityBranch(char_pose_dim, hidden_dim, condition_dim, num_layers)
        self.branch_C = EntityBranch(camera_dim, hidden_dim, condition_dim, num_layers)

        # Pairwise interaction modules
        self.interaction_HH = CharacterCharacterInteraction(hidden_dim, condition_dim)
        self.interaction_AC = CameraCharacterInteraction(hidden_dim, condition_dim)
        self.interaction_BC = CameraCharacterInteraction(hidden_dim, condition_dim)

    def forward(self, y_t, t, text_embed, shot_type=None):
        """
        Predict clean sample y_0 from noisy y_t.
        
        Args:
            y_t: Noisy sample (batch_size, total_dim)
            t: Timestep (batch_size,)
            text_embed: Text embedding (batch_size, text_dim)
            shot_type: Shot type index (batch_size,) or None
            
        Returns:
            y_0_pred: Predicted clean sample (batch_size, total_dim)
        """
        batch_size = y_t.shape[0]

        # Split into entity components
        x_A_t = y_t[:, :self.char_pose_dim]
        x_B_t = y_t[:, self.char_pose_dim:self.char_pose_dim * 2]
        x_C_t = y_t[:, self.char_pose_dim * 2:]

        # Build conditioning signal
        t_emb = self.timestep_proj(self.timestep_embed(t))
        if shot_type is not None:
            s_emb = self.shot_type_embed(shot_type)
        else:
            s_emb = self.no_shot_type_embed.unsqueeze(0).expand(batch_size, -1)
        condition = torch.cat([text_embed, t_emb, s_emb], dim=-1)

        # Encode each entity
        h_A = self.branch_A.encode(x_A_t, condition)
        h_B = self.branch_B.encode(x_B_t, condition)
        h_C = self.branch_C.encode(x_C_t, condition)

        # Pairwise interactions
        delta_B_to_A, delta_A_to_B = self.interaction_HH(h_A, h_B, condition)
        delta_C_to_A, delta_A_to_C = self.interaction_AC(h_A, h_C, condition)
        delta_C_to_B, delta_B_to_C = self.interaction_BC(h_B, h_C, condition)

        # Refine embeddings (Eq. 10-12 in paper)
        h_A_refined = h_A + delta_B_to_A + delta_C_to_A
        h_B_refined = h_B + delta_A_to_B + delta_C_to_B
        h_C_refined = h_C + delta_A_to_C + delta_B_to_C

        # Decode to predicted clean sample
        x_A_pred = self.branch_A.decode(h_A_refined)
        x_B_pred = self.branch_B.decode(h_B_refined)
        x_C_pred = self.branch_C.decode(h_C_refined)

        # Concatenate back
        y_0_pred = torch.cat([x_A_pred, x_B_pred, x_C_pred], dim=-1)
        return y_0_pred
