"""
Training script for Joint Person-Camera Trajectory Diffusion Model.

Usage:
    python train.py --config configs/default.yaml --device cuda
    python train.py --config configs/default.yaml --device cpu --no-clip
    python train.py --config configs/default.yaml --device cuda --single-person
"""

import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from src.models.denoiser import JointTrajectoryDenoiser
from src.models.diffusion import GaussianDiffusion
from src.data.dataset import JointTrajectoryDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description='Train Joint Person-Camera Model')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-clip', action='store_true',
                        help='Use random text embeddings instead of CLIP')
    parser.add_argument('--single-person', action='store_true',
                        help='Use single-person subset')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_text_encoder(config, device, use_clip=True):
    if use_clip:
        try:
            from src.models.text_encoder import CLIPTextEncoder
            model_name = config['text_encoder']['model_name']
            print(f"Loading CLIP text encoder: {model_name}")
            encoder = CLIPTextEncoder(model_name=model_name, device=device)
            encoder = encoder.to(device)
            print("CLIP text encoder loaded.")
            return encoder
        except Exception as e:
            print(f"Warning: Failed to load CLIP ({e}). Using random embeddings.")
            return None
    else:
        print("Using random text embeddings (--no-clip).")
        return None


def train(config, args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model_cfg = config['model']
    traj_cfg = config['trajectory']

    text_encoder = build_text_encoder(config, device, use_clip=not args.no_clip)

    # Build joint denoiser
    denoiser = JointTrajectoryDenoiser(
        person_dim=model_cfg['person_dim'],
        camera_dim=model_cfg['camera_dim'],
        num_frames=traj_cfg['default_num_frames'],
        hidden_dim=model_cfg['hidden_dim'],
        num_layers=model_cfg['num_layers'],
        num_heads=model_cfg['num_heads'],
        text_dim=512,
        timestep_dim=128,
        num_shot_types=len(config['shot_types']['categories']),
        shot_type_dim=config['shot_types']['embedding_dim'],
        num_motion_types=len(traj_cfg['motion_types']),
        motion_type_dim=traj_cfg.get('motion_type_dim', 64),
        dropout=model_cfg.get('dropout', 0.1),
    ).to(device)

    diffusion = GaussianDiffusion(
        denoiser=denoiser,
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
    ).to(device)

    # Resume
    start_epoch = 0
    optimizer_state = None
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        diffusion.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        optimizer_state = ckpt.get('optimizer_state_dict', None)
        print(f"Resumed from epoch {start_epoch}")

    # Dataset
    train_index = config['data'].get('train_index_file', 'train_index.json')
    if args.single_person:
        train_index = 'train_index_single_person.json'
        print(f"Using single-person subset: {train_index}")

    norm_stats_path = config['data'].get('norm_stats_path', None)

    dataset = JointTrajectoryDataset(
        data_root=config['data']['data_root'],
        split='train',
        num_frames=traj_cfg['default_num_frames'],
        person_dim=model_cfg['person_dim'],
        camera_dim=model_cfg['camera_dim'],
        index_file=train_index,
        norm_stats_path=norm_stats_path,
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
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # CFG dropout probability
    cfg_dropout_prob = config['training'].get('cfg_dropout_prob', 0.1)
    print(f"  CFG dropout prob: {cfg_dropout_prob}")

    # Training
    num_epochs = config['training']['num_epochs']
    save_interval = config['training']['save_interval']
    checkpoint_dir = config['paths']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    total_params = sum(p.numel() for p in diffusion.parameters())
    trainable_params = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
    T = traj_cfg['default_num_frames']
    p_dim = model_cfg['person_dim']
    c_dim = model_cfg['camera_dim']

    print(f"\n{'='*60}")
    print(f"Joint Person-Camera Diffusion Training")
    print(f"{'='*60}")
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"  Joint dim: person ({T}x{p_dim}={T*p_dim}) + camera ({T}x{c_dim}={T*c_dim}) = {T*(p_dim+c_dim)}")
    print(f"  Dataset: {len(dataset)} samples | Batch: {config['training']['batch_size']}")
    print(f"  Text: {'CLIP' if text_encoder else 'Random'}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, num_epochs):
        diffusion.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            y = batch['y'].to(device)

            # Text encoding
            if text_encoder is not None:
                text_embed = text_encoder(batch['texts'])
            else:
                text_embed = torch.randn(y.shape[0], 512, device=device)

            # Classifier-Free Guidance: randomly drop text conditioning
            if cfg_dropout_prob > 0:
                drop_mask = torch.rand(y.shape[0], device=device) < cfg_dropout_prob
                text_embed = text_embed.clone()
                text_embed[drop_mask] = 0.0

            shot_types = batch['shot_types'].to(device)
            shot_type = shot_types if (shot_types >= 0).all() else None

            motion_types = batch['motion_types'].to(device)
            motion_type = motion_types if (motion_types >= 0).all() else None

            loss = diffusion.p_losses(y, text_embed,
                                      shot_type=shot_type,
                                      motion_type=motion_type)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                diffusion.parameters(),
                config['training']['gradient_clip'],
            )
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}")

        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'stc_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # Final
    final_path = os.path.join(checkpoint_dir, 'stc_final.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': diffusion.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': config,
    }, final_path)
    print(f"\nTraining complete! Final model: {final_path}")


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    train(config, args)
