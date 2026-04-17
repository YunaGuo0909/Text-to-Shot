import torch
import torch.nn as nn
import math
from src.models.film import FiLMLayer


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1).float() * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Stage2Block(nn.Module):
    """Self-attention -> Cross-attention to person -> FiLM-conditioned FFN."""

    def __init__(self, hidden_dim, condition_dim, num_heads=4, ff_mult=4, dropout=0.1):
        super().__init__()
        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Cross-attention to person trajectory
        self.norm_cross_q = nn.LayerNorm(hidden_dim)
        self.norm_cross_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # FiLM-conditioned FFN
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
            nn.Dropout(dropout),
        )
        self.film = FiLMLayer(hidden_dim, condition_dim)

    def forward(self, x, person_h, condition):
        B, T, H = x.shape

        # Self-attention
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h)
        x = x + h

        # Cross-attention: camera queries attend to person keys/values
        q = self.norm_cross_q(x)
        kv = self.norm_cross_kv(person_h)
        h, _ = self.cross_attn(q, kv, kv)
        x = x + h

        # FiLM-conditioned FFN
        h = self.norm2(x)
        h = self.ff(h)
        h = h.reshape(B * T, H)
        cond_exp = condition.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
        h = self.film(h, cond_exp)
        x = x + h.reshape(B, T, H)

        return x


class Stage2CameraDenoiser(nn.Module):
    """Text + Person trajectory -> Camera trajectory denoiser (Stage 2)."""

    def __init__(
        self,
        camera_dim=6,
        person_dim=3,
        num_frames=48,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        text_dim=512,
        timestep_dim=128,
        num_motion_types=9,
        motion_type_dim=64,
        dropout=0.1,
    ):
        super().__init__()
        self.camera_dim = camera_dim
        self.person_dim = person_dim
        self.num_frames = num_frames
        self.total_dim = camera_dim * num_frames  # 288

        # Condition: text(512) + timestep(128) + motion_type(64) = 704
        condition_dim = text_dim + timestep_dim + motion_type_dim

        # Timestep embedding
        self.timestep_embed = SinusoidalPositionEmbedding(timestep_dim)
        self.timestep_proj = nn.Sequential(
            nn.Linear(timestep_dim, timestep_dim),
            nn.SiLU(),
            nn.Linear(timestep_dim, timestep_dim),
        )

        # Motion type embedding
        self.motion_type_embed = nn.Embedding(num_motion_types, motion_type_dim)
        self.no_motion_type = nn.Parameter(torch.zeros(motion_type_dim))

        # Person trajectory projection (condition, not noisy)
        self.person_proj = nn.Linear(person_dim, hidden_dim)
        self.person_pe = nn.Parameter(torch.randn(1, num_frames, hidden_dim) * 0.02)

        # Camera input projection
        self.camera_proj = nn.Linear(camera_dim, hidden_dim)
        self.camera_pe = nn.Parameter(torch.randn(1, num_frames, hidden_dim) * 0.02)

        # Condition bias
        self.cond_proj = nn.Linear(condition_dim, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Stage2Block(hidden_dim, condition_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, camera_dim)

    def forward(self, y_t, t, text_embed, shot_type=None, motion_type=None,
                person_traj=None):
        B = y_t.shape[0]

        # Reshape noisy camera to per-frame
        cam_x = y_t.reshape(B, self.num_frames, self.camera_dim)

        # Build condition vector
        t_emb = self.timestep_proj(self.timestep_embed(t))

        if motion_type is not None:
            m_emb = self.motion_type_embed(motion_type)
        else:
            m_emb = self.no_motion_type.unsqueeze(0).expand(B, -1)

        condition = torch.cat([text_embed, t_emb, m_emb], dim=-1)
        cond_bias = self.cond_proj(condition).unsqueeze(1)

        # Person trajectory as cross-attention context
        if person_traj is not None:
            person_frames = person_traj.reshape(B, self.num_frames, self.person_dim)
            person_h = self.person_proj(person_frames) + self.person_pe
        else:
            # Null person condition (zeros)
            person_h = torch.zeros(B, self.num_frames, self.camera_proj.out_features,
                                   device=y_t.device)

        # Camera hidden states
        cam_h = self.camera_proj(cam_x) + self.camera_pe + cond_bias

        # Transformer with cross-attention
        for block in self.blocks:
            cam_h = block(cam_h, person_h, condition)

        # Output
        out = self.output_proj(self.final_norm(cam_h))
        return out.reshape(B, self.total_dim)
