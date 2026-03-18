"""
Gaussian Diffusion for Joint Person-Camera Trajectory Generation.

Based on DDPM (Ho et al., 2020), operates on the concatenated joint
trajectory vector y = [person_traj_flat, camera_traj_flat].
"""

import torch
import torch.nn as nn
import numpy as np


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine beta schedule (Nichol & Dhariwal, 2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """Linear beta schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion for joint person-camera trajectory.

    Operates on flattened joint vectors:
        y = [person_traj (T*3), camera_traj (T*6)]
    """

    def __init__(self, denoiser, num_timesteps=1000, beta_schedule='cosine'):
        super().__init__()
        self.denoiser = denoiser
        self.num_timesteps = num_timesteps

        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(num_timesteps)
        elif beta_schedule == 'linear':
            betas = linear_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def q_sample(self, y_0, t, noise=None):
        """Forward process: sample y_t from q(y_t | y_0)."""
        if noise is None:
            noise = torch.randn_like(y_0)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_alpha_t * y_0 + sqrt_one_minus_alpha_t * noise

    def p_losses(self, y_0, text_embed, shot_type=None, motion_type=None):
        """
        Training loss: L = E[||y_0 - f_theta(y_t, t, c)||^2]

        Args:
            y_0: (B, total_dim) clean joint trajectory
            text_embed: (B, text_dim) text conditioning
            shot_type: (B,) or None
            motion_type: (B,) or None
        """
        batch_size = y_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=y_0.device)
        noise = torch.randn_like(y_0)
        y_t = self.q_sample(y_0, t, noise)

        y_0_pred = self.denoiser(y_t, t, text_embed,
                                 shot_type=shot_type, motion_type=motion_type)

        loss = nn.functional.mse_loss(y_0_pred, y_0)
        return loss

    @torch.no_grad()
    def p_sample(self, y_t, t, text_embed, shot_type=None, motion_type=None):
        """Reverse step: sample y_{t-1} from p_theta(y_{t-1} | y_t)."""
        y_0_pred = self.denoiser(y_t, t, text_embed,
                                 shot_type=shot_type, motion_type=motion_type)

        posterior_mean = (
            self.posterior_mean_coef1[t].unsqueeze(-1) * y_0_pred +
            self.posterior_mean_coef2[t].unsqueeze(-1) * y_t
        )

        if t[0] > 0:
            noise = torch.randn_like(y_t)
            posterior_var = self.betas[t].unsqueeze(-1)
            return posterior_mean + torch.sqrt(posterior_var) * noise
        else:
            return posterior_mean

    @torch.no_grad()
    def sample(self, text_embed, shot_type=None, motion_type=None, device='cuda'):
        """
        Generate joint person-camera trajectory.

        Returns:
            y_0: (B, total_dim) = [person_flat (T*3), camera_flat (T*6)]
        """
        batch_size = text_embed.shape[0]
        total_dim = self.denoiser.total_dim

        y_t = torch.randn(batch_size, total_dim, device=device)

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            y_t = self.p_sample(y_t, t_batch, text_embed,
                                shot_type=shot_type, motion_type=motion_type)

        return y_t
