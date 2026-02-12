"""
Pairwise Interaction Modules for Joint Character-Camera Generation.

Implements three types of interaction:
1. Character-Character (I_HH): Captures coordination between two people
2. Camera-Character (I_CH): Couples camera framing with character poses

These modules exchange residual messages between entity pairs (A↔B, A↔C, B↔C)
to capture mutual dependencies under textual guidance.

Reference:
- "From Script to Shot" (SIGGRAPH 2026) Section 3.4
"""

import torch
import torch.nn as nn
from .film import FiLMLayer


class InteractionModule(nn.Module):
    """
    Directed residual interaction module between two entities.
    
    Given source and target embeddings, produces a directed residual 
    update that refines the target representation. FiLM modulation 
    enables text-dependent interaction behavior.
    """

    def __init__(self, embed_dim, condition_dim):
        """
        Args:
            embed_dim: Dimension of entity embeddings
            condition_dim: Dimension of text conditioning
        """
        super().__init__()
        self.mlp1 = nn.Linear(embed_dim * 2, embed_dim)
        self.film = FiLMLayer(embed_dim, condition_dim)
        self.mlp2 = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.SiLU()

    def forward(self, h_source, h_target, condition):
        """
        Compute directed residual update from source to target.
        
        Args:
            h_source: Source entity embedding (batch_size, embed_dim)
            h_target: Target entity embedding (batch_size, embed_dim)
            condition: Text conditioning (batch_size, condition_dim)
            
        Returns:
            delta_h: Residual update for target (batch_size, embed_dim)
        """
        # Concatenate source and target
        h_concat = torch.cat([h_source, h_target], dim=-1)
        h = self.activation(self.mlp1(h_concat))
        h = self.film(h, condition)
        delta_h = self.mlp2(h)
        return delta_h


class CharacterCharacterInteraction(nn.Module):
    """
    Bidirectional character-character interaction module (I_HH).
    
    Captures coordination and spatial relations between two characters
    by predicting bidirectional residuals.
    """

    def __init__(self, embed_dim, condition_dim):
        super().__init__()
        self.interaction = InteractionModule(embed_dim, condition_dim)

    def forward(self, h_A, h_B, condition):
        """
        Args:
            h_A: Character A embedding
            h_B: Character B embedding
            condition: Text conditioning
            
        Returns:
            delta_A: Residual update for A (from B)
            delta_B: Residual update for B (from A)
        """
        delta_A_to_B = self.interaction(h_A, h_B, condition)
        delta_B_to_A = self.interaction(h_B, h_A, condition)
        return delta_B_to_A, delta_A_to_B


class CameraCharacterInteraction(nn.Module):
    """
    Bidirectional camera-character interaction module (I_CH).
    
    Couples camera framing with character configurations by computing
    residuals in both directions.
    """

    def __init__(self, embed_dim, condition_dim):
        super().__init__()
        self.interaction = InteractionModule(embed_dim, condition_dim)

    def forward(self, h_char, h_cam, condition):
        """
        Args:
            h_char: Character embedding
            h_cam: Camera embedding
            condition: Text conditioning
            
        Returns:
            delta_char: Residual update for character (from camera)
            delta_cam: Residual update for camera (from character)
        """
        delta_char_to_cam = self.interaction(h_char, h_cam, condition)
        delta_cam_to_char = self.interaction(h_cam, h_char, condition)
        return delta_cam_to_char, delta_char_to_cam
