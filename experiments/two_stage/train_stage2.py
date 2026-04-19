"""
Train Stage 2: Text + Person trajectory -> Camera trajectory.

Usage:
    PYTHONPATH=. python experiments/two_stage/train_stage2.py --config experiments/two_stage/configs/stage2.yaml
    PYTHONPATH=. python experiments/two_stage/train_stage2.py --config experiments/two_stage/configs/stage2.yaml --no-clip
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
from experiments.two_stage.models.stage2_denoiser import Stage2CameraDenoiser
from experiments.two_stage.models.diffusion import StageDiffusion


def parse_args():
    parser = argparse.ArgumentParser(description='Train Stage 2: Camera Trajectory')
    parser.add_argument('--config', type=str, default='experiments/two_stage/configs/stage2.yaml')
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
    camera_dim = model_cfg['camera_dim']
    person_dim = model_cfg['person_dim']
    num_frames = traj_cfg['default_num_frames']
    person_total = person_dim * num_frames  # 144
    camera_total = camera_dim * num_frames  # 288

    text_encoder = build_text_encoder(config, device, use_clip=not args.no_clip)

    # Build Stage 2 denoiser
    denoiser = Stage2CameraDenoiser(
        camera_dim=camera_dim,
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

    # Dataset
    norm_stats_path = config['data'].get('norm_stats_path', None)
    train_dataset = JointTrajectoryDataset(
        data_root=config['data']['data_root'],
        split='train',
        num_frames=num_frames,
        person_dim=person_dim,
        camera_dim=camera_dim,
        index_file=config['data'].get('train_index_file', 'train_index.json'),
        norm_stats_path=norm_stats_path,
    )
    val_dataset = JointTrajectoryDataset(
        data_root=config['data']['data_root'],
        split='test',
        num_frames=num_frames,
        person_dim=person_dim,
        camera_dim=camera_dim,
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
    log_dir = config['paths'].get('log_dir', '/transfer/two-stage-logs/stage2')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    total_params = sum(p.numel() for p in diffusion.parameters())
    trainable_params = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"Stage 2: Text + Person -> Camera ({num_frames}x{camera_dim}={camera_total})")
    print(f"{'='*60}")
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"  Person condition: {num_frames}x{person_dim}={person_total} (GT during training)")
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
            # Split: person as CONDITION, camera as TARGET
            person_traj = y[:, :person_total]
            camera_y = y[:, person_total:]

            if text_encoder is not None:
                text_embed = text_encoder(batch['texts'])
            else:
                text_embed = torch.randn(y.shape[0], 512, device=device)

            # CFG dropout on BOTH text AND person trajectory
            if cfg_dropout_prob > 0:
                text_drop = torch.rand(y.shape[0], device=device) < cfg_dropout_prob
                person_drop = torch.rand(y.shape[0], device=device) < cfg_dropout_prob

                text_embed = text_embed.clone()
                text_embed[text_drop] = 0.0

                person_traj = person_traj.clone()
                person_traj[person_drop] = 0.0

            motion_types = batch['motion_types'].to(device)
            motion_type = motion_types if (motion_types >= 0).all() else None

            diffusion_loss = diffusion.p_losses(camera_y, text_embed,
                                               motion_type=motion_type,
                                               person_traj=person_traj)

            # === Auxiliary loss: distance direction consistency ===
            # For low-noise timesteps, the model's y_0 prediction should have
            # correct distance-change direction relative to motion_type.
            # This prevents the model from collapsing to a mean trajectory.
            aux_loss = torch.tensor(0.0, device=device)
            aux_weight = config['training'].get('aux_loss_weight', 0.1)

            if motion_type is not None and aux_weight > 0:
                # Get model prediction at RANDOM noise levels (including high noise
                # where mode collapse actually happens)
                B = camera_y.shape[0]
                t_aux = torch.randint(200, 800, (B,), device=device)
                noise = torch.randn_like(camera_y)
                y_t = diffusion.q_sample(camera_y, t_aux, noise)
                y_0_pred = diffusion.denoiser(y_t, t_aux, text_embed,
                                              motion_type=motion_type,
                                              person_traj=person_traj)

                # Reshape predicted camera to (B, T, 6)
                cam_pred = y_0_pred.reshape(B, num_frames, camera_dim)
                per_cond = person_traj.reshape(B, num_frames, person_dim)

                # Distance at first and last quarter
                quarter = num_frames // 4
                dist_start = torch.norm(cam_pred[:, :quarter, :3] - per_cond[:, :quarter, :], dim=-1).mean(dim=1)
                dist_end = torch.norm(cam_pred[:, -quarter:, :3] - per_cond[:, -quarter:, :], dim=-1).mean(dim=1)
                dist_change = dist_end - dist_start  # positive = moving away

                # Target direction per motion_type:
                #   dolly-in (1) → distance should decrease → target = -1
                #   dolly-out (2) → distance should increase → target = +1
                #   static (0) → distance should stay → target = 0
                #   others → no constraint → target = 0 (soft)
                target_dir = torch.zeros(B, device=device)
                target_dir[motion_type == 1] = -1.0   # dolly-in
                target_dir[motion_type == 2] = 1.0    # dolly-out

                # Only apply loss where we have a direction target
                has_target = (motion_type == 1) | (motion_type == 2)
                if has_target.any():
                    # Hinge-like: penalize if sign is wrong
                    margin = 0.1
                    direction_error = torch.clamp(margin - dist_change * target_dir, min=0.0)
                    aux_loss = direction_error[has_target].mean()

            loss = diffusion_loss + aux_weight * aux_loss

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
                    person_traj = y[:, :person_total]
                    camera_y = y[:, person_total:]

                    if text_encoder is not None:
                        text_embed = text_encoder(batch['texts'])
                    else:
                        text_embed = torch.randn(y.shape[0], 512, device=device)

                    motion_types = batch['motion_types'].to(device)
                    motion_type = motion_types if (motion_types >= 0).all() else None

                    loss = diffusion.p_losses(camera_y, text_embed,
                                              motion_type=motion_type,
                                              person_traj=person_traj)
                    val_total += loss.item()
                    val_batches += 1

            avg_val_loss = val_total / max(val_batches, 1)

        val_losses.append(avg_val_loss)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        if np.isnan(avg_val_loss):
            print(f"Epoch [{epoch+1}/{num_epochs}] Train: {avg_train_loss:.6f}  lr: {current_lr:.2e}  aux: {aux_loss.item():.4f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train: {avg_train_loss:.6f}  Val: {avg_val_loss:.6f}  lr: {current_lr:.2e}  aux: {aux_loss.item():.4f}")

        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'stage2_epoch{epoch+1}.pth')
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
    final_path = os.path.join(checkpoint_dir, 'stage2_final.pth')
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
    print(f"\nStage 2 training complete! Final model: {final_path}")

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
    plot_loss_curves(train_losses, val_losses, eval_interval, num_epochs, log_dir, 'Stage 2')


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
