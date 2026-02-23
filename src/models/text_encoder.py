"""
CLIP Text Encoder wrapper for conditioning the diffusion model.

Provides a frozen CLIP text encoder that converts scene descriptions
into 512-dim embeddings for trajectory generation conditioning.

Reference:
- Radford, A., et al. (2021). Learning Transferable Visual Models
  From Natural Language Supervision. ICML.
"""

import torch
import torch.nn as nn
from typing import List


class CLIPTextEncoder(nn.Module):
    """
    Frozen CLIP text encoder for extracting text embeddings.

    Wraps HuggingFace's CLIPModel to provide a simple interface
    for encoding text descriptions into conditioning vectors.
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda", max_length=77):
        super().__init__()
        from transformers import CLIPModel, CLIPTokenizer

        self.device = device
        self.max_length = max_length

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.text_dim = self.model.config.text_config.hidden_size  # 512

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of text strings into CLIP embeddings.

        Args:
            texts: List of text descriptions

        Returns:
            text_embed: (batch_size, 512) normalized text embeddings
        """
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            text_embed = self.model.get_text_features(**tokens)

        # L2 normalize
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        return text_embed

    def to(self, device):
        self.device = device if isinstance(device, str) else str(device)
        self.model = self.model.to(device)
        return super().to(device)
