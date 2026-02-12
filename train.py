"""
Training script for the Joint Character-Camera Diffusion Model.

Usage:
    python train.py --config configs/default.yaml
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.denoiser import JointDenoiser
from src.models.diffusion import GaussianDiffusion
from src.data.dataset import CharacterCameraDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description='Train Joint Character-Camera Model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(config, device):
    """Main training loop."""
    
    # Create model
    denoiser = JointDenoiser(
        char_pose_dim=config['model']['char_pose_dim'],
        camera_dim=config['model']['camera_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        text_dim=512,  # CLIP embedding dim
        timestep_dim=128,
        num_shot_types=len(config['shot_types']['categories']),
        shot_type_dim=config['shot_types']['embedding_dim'],
    ).to(device)

    diffusion = GaussianDiffusion(
        denoiser=denoiser,
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
    ).to(device)

    # Create dataset and dataloader
    dataset = CharacterCameraDataset(
        data_root=config['data']['data_root'],
        split='train',
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_fn,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )

    # Training loop
    num_epochs = config['training']['num_epochs']
    save_interval = config['training']['save_interval']
    checkpoint_dir = config['paths']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Starting training for {num_epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in diffusion.parameters()):,}")

    for epoch in range(num_epochs):
        diffusion.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            y = batch['y'].to(device)
            
            # TODO: Replace with actual CLIP text encoding
            text_embed = torch.randn(y.shape[0], 512, device=device)
            
            # Shot type conditioning (use -1 for no shot type)
            shot_types = batch['shot_types'].to(device)
            shot_type = shot_types if (shot_types >= 0).all() else None

            # Compute loss
            loss = diffusion.p_losses(y, text_embed, shot_type=shot_type)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                diffusion.parameters(),
                config['training']['gradient_clip']
            )
            
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    print("Training complete!")


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train(config, device)
