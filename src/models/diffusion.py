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
    def p_sample(self, y_t, t, text_embed, shot_type=None, motion_type=None,
                 guidance_scale=1.0):
        """Reverse step: sample y_{t-1} from p_theta(y_{t-1} | y_t)."""
        if guidance_scale > 1.0:
            # Classifier-Free Guidance: run denoiser with null and text embed
            null_embed = torch.zeros_like(text_embed)
            y_0_null = self.denoiser(y_t, t, null_embed,
                                     shot_type=None, motion_type=None)
            y_0_text = self.denoiser(y_t, t, text_embed,
                                     shot_type=shot_type, motion_type=motion_type)
            y_0_pred = y_0_null + guidance_scale * (y_0_text - y_0_null)
        else:
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
    def sample(self, text_embed, shot_type=None, motion_type=None, device='cuda',
               guidance_scale=1.0, use_ddim=False, ddim_steps=50, ddim_eta=0.0):
        """
        Generate joint person-camera trajectory.

        Args:
            guidance_scale: CFG scale. 1.0 = no CFG. 3.0-7.0 = typical range.
            use_ddim: If True, use DDIM deterministic sampling (much smoother).
            ddim_steps: Number of DDIM sampling steps (default 50).
            ddim_eta: DDIM stochasticity. 0.0 = fully deterministic.
        Returns:
            y_0: (B, total_dim) = [person_flat (T*3), camera_flat (T*6)]
        """
        if use_ddim:
            return self.ddim_sample(text_embed, shot_type=shot_type,
                                    motion_type=motion_type, device=device,
                                    guidance_scale=guidance_scale,
                                    ddim_steps=ddim_steps, ddim_eta=ddim_eta)

        batch_size = text_embed.shape[0]
        total_dim = self.denoiser.total_dim

        y_t = torch.randn(batch_size, total_dim, device=device)

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            y_t = self.p_sample(y_t, t_batch, text_embed,
                                shot_type=shot_type, motion_type=motion_type,
                                guidance_scale=guidance_scale)

        return y_t

    @torch.no_grad()
    def _predict_y0(self, y_t, t_batch, text_embed, shot_type=None,
                    motion_type=None, guidance_scale=1.0):
        """Predict y_0 from y_t with optional CFG."""
        if guidance_scale > 1.0:
            null_embed = torch.zeros_like(text_embed)
            y_0_null = self.denoiser(y_t, t_batch, null_embed,
                                     shot_type=None, motion_type=None)
            y_0_text = self.denoiser(y_t, t_batch, text_embed,
                                     shot_type=shot_type, motion_type=motion_type)
            return y_0_null + guidance_scale * (y_0_text - y_0_null)
        else:
            return self.denoiser(y_t, t_batch, text_embed,
                                 shot_type=shot_type, motion_type=motion_type)

    @torch.no_grad()
    def ddim_sample(self, text_embed, shot_type=None, motion_type=None,
                    device='cuda', guidance_scale=1.0, ddim_steps=50,
                    ddim_eta=0.0):
        """
        DDIM sampling (Song et al., 2020). Deterministic when eta=0.

        Much smoother than DDPM because it skips most timesteps and
        avoids adding random noise at each step.
        """
        batch_size = text_embed.shape[0]
        total_dim = self.denoiser.total_dim

        # Build sub-sequence of timesteps: evenly spaced
        step_size = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = sorted(timesteps, reverse=True)

        y_t = torch.randn(batch_size, total_dim, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            y_0_pred = self._predict_y0(y_t, t_batch, text_embed,
                                        shot_type=shot_type,
                                        motion_type=motion_type,
                                        guidance_scale=guidance_scale)

            # Clip predicted y_0 to reasonable range for stability
            y_0_pred = y_0_pred.clamp(-5.0, 5.0)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]

                alpha_t = self.alphas_cumprod[t]
                alpha_next = self.alphas_cumprod[t_next]

                # Predicted noise from y_0 prediction
                eps_pred = (y_t - torch.sqrt(alpha_t) * y_0_pred) / torch.sqrt(1.0 - alpha_t)

                # DDIM update
                sigma = ddim_eta * torch.sqrt(
                    (1.0 - alpha_next) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_next)
                )
                dir_xt = torch.sqrt(1.0 - alpha_next - sigma ** 2) * eps_pred
                y_t = torch.sqrt(alpha_next) * y_0_pred + dir_xt

                if ddim_eta > 0:
                    y_t = y_t + sigma * torch.randn_like(y_t)
            else:
                y_t = y_0_pred

        return y_t
