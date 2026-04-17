import torch
import torch.nn as nn


class ConditionalFlowMatching(nn.Module):
    """
    Conditional Flow Matching (Lipman et al., 2023).

    Learns a velocity field v(x_t, t) that transports samples along straight
    paths from noise (t=0) to data (t=1).

    Forward interpolant: x_t = (1 - t) * epsilon + t * x_0
    Target velocity:     v_target = x_0 - epsilon
    Sampling:            Euler ODE integration from t=0 to t=1
    """

    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    def flow_loss(self, x_0, text_embed, shot_type=None, motion_type=None):
        B = x_0.shape[0]
        device = x_0.device

        t = torch.rand(B, device=device)
        epsilon = torch.randn_like(x_0)

        x_t = (1.0 - t.unsqueeze(-1)) * epsilon + t.unsqueeze(-1) * x_0
        v_target = x_0 - epsilon

        t_scaled = (t * 999).long()
        v_pred = self.denoiser(x_t, t_scaled, text_embed,
                               shot_type=shot_type, motion_type=motion_type)

        return nn.functional.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(self, text_embed, shot_type=None, motion_type=None,
               device='cuda', guidance_scale=1.0, num_steps=50):
        B = text_embed.shape[0]
        total_dim = self.denoiser.total_dim

        x = torch.randn(B, total_dim, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = i * dt
            t_batch = torch.full((B,), int(t * 999), device=device, dtype=torch.long)

            if guidance_scale > 1.0:
                null_embed = torch.zeros_like(text_embed)
                v_null = self.denoiser(x, t_batch, null_embed,
                                       shot_type=None, motion_type=None)
                v_cond = self.denoiser(x, t_batch, text_embed,
                                       shot_type=shot_type, motion_type=motion_type)
                v = v_null + guidance_scale * (v_cond - v_null)
            else:
                v = self.denoiser(x, t_batch, text_embed,
                                  shot_type=shot_type, motion_type=motion_type)

            x = x + v * dt

        return x
