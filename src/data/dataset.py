"""
Dataset classes for training the joint character-camera diffusion model.

Supports loading dual-human interaction data with text annotations
and camera parameters.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from typing import Optional, Dict, List


class CharacterCameraDataset(Dataset):
    """
    Dataset for joint character-camera generation.
    
    Each sample contains:
    - Two character poses (SMPL-based, 150-dim each)
    - Camera state (Toric parameterization, 6-dim)
    - Text description
    - Optional shot type label
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        text_encoder=None,
        max_text_length: int = 77,
    ):
        """
        Args:
            data_root: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test')
            text_encoder: Pre-initialized text encoder
            max_text_length: Maximum text token length
        """
        self.data_root = data_root
        self.split = split
        self.text_encoder = text_encoder
        self.max_text_length = max_text_length

        # Load data index
        self.samples = self._load_index()

    def _load_index(self) -> List[Dict]:
        """Load dataset index file."""
        index_path = os.path.join(self.data_root, f'{self.split}_index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: Index file not found at {index_path}")
            return []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load character poses
        char_a_pose = torch.tensor(sample['char_a_pose'], dtype=torch.float32)
        char_b_pose = torch.tensor(sample['char_b_pose'], dtype=torch.float32)

        # Load camera state
        camera_state = torch.tensor(sample['camera_state'], dtype=torch.float32)

        # Concatenate full state: y = (x_A, x_B, x_C)
        y = torch.cat([char_a_pose, char_b_pose, camera_state], dim=-1)

        # Text description
        text = sample.get('text', '')

        # Shot type (optional)
        shot_type = sample.get('shot_type', -1)

        return {
            'y': y,
            'text': text,
            'shot_type': shot_type,
            'sample_id': sample.get('id', idx),
        }


def collate_fn(batch):
    """Custom collate function for the dataloader."""
    y = torch.stack([item['y'] for item in batch])
    texts = [item['text'] for item in batch]
    shot_types = torch.tensor([item['shot_type'] for item in batch], dtype=torch.long)
    sample_ids = [item['sample_id'] for item in batch]

    return {
        'y': y,
        'texts': texts,
        'shot_types': shot_types,
        'sample_ids': sample_ids,
    }
