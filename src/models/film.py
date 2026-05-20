"""
FiLM (Feature-wise Linear Modulation) Layer.

Used to inject text conditioning and timestep information into the 
denoising network, following:
- Perez, E., et al. (2018). FiLM: Visual reasoning with a general 
  conditioning layer. AAAI.
"""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    
    Applies affine transformation conditioned on external signal:
        output = gamma * input + beta
    where gamma and beta are predicted from the conditioning signal.
    """

    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        self.film_generator = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, feature_dim * 2)  # gamma and beta
        )

    def forward(self, x, condition):
        film_params = self.film_generator(condition)
        gamma, beta = film_params.chunk(2, dim=-1)
        return gamma * x + beta
