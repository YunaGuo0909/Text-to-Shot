"""
Train Stage 1: Text -> Person root trajectory.

Usage:
    PYTHONPATH=. python experiments/two_stage/train_stage1.py --config experiments/two_stage/configs/stage1.yaml
    PYTHONPATH=. python experiments/two_stage/train_stage1.py --config experiments/two_stage/configs/stage1.yaml --no-clip
"""

import argparse
import json
import os
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import JointTrajectoryDataset, collate_fn
from experiments.two_stage.models.stage1_denoiser import Stage1PersonDenoiser
from experiments.two_stage.models.diffusion import StageDiffusion


def parse_args():
    parser = argparse.ArgumentParser(description='Train Stage 1: Person Trajectory')
    parser.add_argument('--config', type=str, default='experiments/two_stage/configs/stage1.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-clip', action='store_true')
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_text_encoder(config, device, use_clip=True):
    if use_clip:
        try:
            from src.models.text_encoder import CLIPTextEncoder
            model_name = config['text_encoder']['model_name']
            print(f"Loading CLIP: {model_name}")
            encoder = CLIPTextEncoder(model_name=model_name, device=device).to(device)
            print("CLIP loaded.")
            return encoder
        except Exception as e:
            print(f"CLIP failed ({e}), using random embeddings.")
            return None
    print("Using random text embeddings.")
    return None


def train(config, args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model_cfg = config['model']
    traj_cfg = config['trajectory']
    person_dim = model_cfg['person_dim']
    num_frames = traj_cfg['default_num_frames']
    person_total = person_dim * num_frames  # 240

    text_encoder = build_text_encoder(config, device, use_clip=not args.no_clip)

    # Build Stage 1 denoiser
    denoiser = Stage1PersonDenoiser(
        person_dim=person_dim,
        num_frames=num_frames,
        hidden_dim=model_cfg['hidden_dim'],
        num_layers=model_cfg['num_layers'],
        num_heads=model_cfg['num_heads'],
        text_dim=512,
        timestep_dim=128,
        num_motion_types=len(traj_cfg['motion_types']),
        motion_type_dim=traj_cfg.get('motion_type_dim', 64),
        dropout=model_cfg.get('dropout', 0.1),
    ).to(device)

    diffusion = StageDiffusion(
        denoiser=denoiser,
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
    ).to(device)

    # Resume
    start_epoch = 0
    optimizer_state = None
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        diffusion.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        optimizer_state = ckpt.get('optimizer_state_dict', None)
        print(f"Resumed from epoch {start_epoch}")

    # Dataset (reuse JointTrajectoryDataset, we extract person part only)
    norm_stats_path = config['data'].get('norm_stats_path', None)
    train_dataset = JointTrajectoryDataset(
        data_root=config['data']['data_root'],
        split='train',
        num_frames=num_frames,
        person_dim=person_dim,
        camera_dim=6,
        index_file=config['data'].get('train_index_file', 'train_index.json'),
        norm_stats_path=norm_stats_path,
    )
    val_dataset = JointTrajectoryDataset(
        data_root=config['data']['data_root'],
        split='test',
        num_frames=num_frames,
        person_dim=person_dim,
        camera_dim=6,
        index_file=config['data'].get('test_index_file', 'test_index.json'),
        norm_stats_path=norm_stats_path,
    )

    # Weighted sampler by motion_type
    motion_labels = [s.get('camera_motion', 'static') for s in train_dataset.samples]
    label_counts = Counter(motion_labels)
    weight_per_label = {label: 1.0 / count for label, count in label_counts.items()}
    sample_weights = [weight_per_label[label] for label in motion_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=config['training']['batch_size'],
        sampler=sampler, num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'], collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['training']['batch_size'],
        shuffle=False, num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'], collate_fn=collate_fn,
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    num_epochs = config['training']['num_epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 10)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    cfg_dropout_prob = config['training'].get('cfg_dropout_prob', 0.25)
    save_interval = config['training']['save_interval']
    eval_interval = config['training'].get('eval_interval', 10)
    checkpoint_dir = config['paths']['checkpoint_dir']
    log_dir = config['paths'].get('log_dir', '/transfer/two-stage-logs/stage1')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    total_params = sum(p.numel() for p in diffusion.parameters())
    trainable_params = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"Stage 1: Text -> Person Trajectory ({num_frames}x{person_dim}={person_total})")
    print(f"{'='*60}")
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"  Batch: {config['training']['batch_size']} | CFG dropout: {cfg_dropout_prob}")
    print(f"  Text: {'CLIP' if text_encoder else 'Random'}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"{'='*60}\n")

    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, num_epochs):
        diffusion.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            y = batch['y'].to(device)
            # Extract person part only
            person_y = y[:, :person_total]

            if text_encoder is not None:
                text_embed = text_encoder(batch['texts'])
            else:
                text_embed = torch.randn(y.shape[0], 512, device=device)

            # CFG dropout on text
            if cfg_dropout_prob > 0:
                drop_mask = torch.rand(y.shape[0], device=device) < cfg_dropout_prob
                text_embed = text_embed.clone()
                text_embed[drop_mask] = 0.0

            motion_types = batch['motion_types'].to(device)
            motion_type = motion_types if (motion_types >= 0).all() else None

            loss = diffusion.p_losses(person_y, text_embed, motion_type=motion_type)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(),
                                           config['training']['gradient_clip'])
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)

        # Validation
        avg_val_loss = float('nan')
        if len(val_dataset) > 0 and (epoch + 1) % eval_interval == 0:
            diffusion.eval()
            val_total = 0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    y = batch['y'].to(device)
                    person_y = y[:, :person_total]

                    if text_encoder is not None:
                        text_embed = text_encoder(batch['texts'])
                    else:
                        text_embed = torch.randn(y.shape[0], 512, device=device)

                    motion_types = batch['motion_types'].to(device)
                    motion_type = motion_types if (motion_types >= 0).all() else None

                    loss = diffusion.p_losses(person_y, text_embed, motion_type=motion_type)
                    val_total += loss.item()
                    val_batches += 1

            avg_val_loss = val_total / max(val_batches, 1)

        val_losses.append(avg_val_loss)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        if np.isnan(avg_val_loss):
            print(f"Epoch [{epoch+1}/{num_epochs}] Train: {avg_train_loss:.6f}  lr: {current_lr:.2e}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train: {avg_train_loss:.6f}  Val: {avg_val_loss:.6f}  lr: {current_lr:.2e}")

        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'stage1_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': config,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # Final save
    final_path = os.path.join(checkpoint_dir, 'stage1_final.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': diffusion.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config,
    }, final_path)
    print(f"\nStage 1 training complete! Final model: {final_path}")

    # Save loss history
    loss_json_path = os.path.join(log_dir, 'loss_history.json')
    with open(loss_json_path, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'eval_interval': eval_interval,
            'num_epochs': num_epochs,
        }, f, indent=2)

    # Plot
    plot_loss_curves(train_losses, val_losses, eval_interval, num_epochs, log_dir, 'Stage 1')


def plot_loss_curves(train_losses, val_losses, eval_interval, num_epochs, save_dir, title_prefix):
    epochs = list(range(1, len(train_losses) + 1))
    val_epochs = [e for e, v in zip(epochs, val_losses) if not np.isnan(v)]
    val_vals = [v for v in val_losses if not np.isnan(v)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(epochs, train_losses, color='#FF6B6B', linewidth=1.2, alpha=0.85, label='Train')
    if val_vals:
        ax.plot(val_epochs, val_vals, color='#4ECDC4', linewidth=2, marker='o',
                markersize=4, label=f'Val (every {eval_interval} ep)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(f'{title_prefix} Loss ({num_epochs} epochs)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    train(config, args)
