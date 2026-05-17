"""
Training script for Flow Matching + Data Augmentation experiment.

Usage:
    PYTHONPATH=. python experiments/flow_matching/train.py
    PYTHONPATH=. python experiments/flow_matching/train.py --config experiments/flow_matching/configs/default.yaml --device cuda
    PYTHONPATH=. python experiments/flow_matching/train.py --no-clip --device cpu
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

from src.models.denoiser import JointTrajectoryDenoiser
from src.data.dataset import collate_fn
from experiments.flow_matching.models.flow_model import ConditionalFlowMatching
from experiments.flow_matching.data.augmented_dataset import AugmentedTrajectoryDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train Flow Matching Model')
    parser.add_argument('--config', type=str,
                        default='experiments/flow_matching/configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-clip', action='store_true')
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
    train_cfg = config['training']
    data_cfg = config['data']

    text_encoder = build_text_encoder(config, device, use_clip=not args.no_clip)

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

    flow = ConditionalFlowMatching(denoiser=denoiser).to(device)

    start_epoch = 0
    optimizer_state = None
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        flow.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        optimizer_state = ckpt.get('optimizer_state_dict', None)
        print(f"Resumed from epoch {start_epoch}")

    use_augmentation = train_cfg.get('augmentation', True)
    norm_stats_path = data_cfg.get('norm_stats_path', None)

    train_dataset = AugmentedTrajectoryDataset(
        augment=use_augmentation,
        data_root=data_cfg['data_root'],
        split='train',
        num_frames=traj_cfg['default_num_frames'],
        person_dim=model_cfg['person_dim'],
        camera_dim=model_cfg['camera_dim'],
        index_file=data_cfg.get('train_index_file', 'train_index.json'),
        norm_stats_path=norm_stats_path,
    )

    val_dataset = AugmentedTrajectoryDataset(
        augment=False,
        data_root=data_cfg['data_root'],
        split='test',
        num_frames=traj_cfg['default_num_frames'],
        person_dim=model_cfg['person_dim'],
        camera_dim=model_cfg['camera_dim'],
        index_file=data_cfg.get('test_index_file', 'test_index.json'),
        norm_stats_path=norm_stats_path,
    )

    motion_labels = [s.get('camera_motion', 'static') for s in train_dataset.samples]
    label_counts = Counter(motion_labels)
    weight_per_label = {label: 1.0 / count for label, count in label_counts.items()}
    sample_weights = [weight_per_label[label] for label in motion_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset),
                                    replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        sampler=sampler,
        num_workers=data_cfg['num_workers'],
        pin_memory=data_cfg['pin_memory'],
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=data_cfg['num_workers'],
        pin_memory=data_cfg['pin_memory'],
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        flow.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    num_epochs = train_cfg['num_epochs']
    warmup_epochs = train_cfg.get('warmup_epochs', 10)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    cfg_dropout_prob = train_cfg.get('cfg_dropout_prob', 0.25)
    save_interval = train_cfg['save_interval']
    eval_interval = train_cfg.get('eval_interval', 10)
    checkpoint_dir = config['paths']['checkpoint_dir']
    log_dir = config['paths'].get('log_dir', '/transfer/fm-logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    total_params = sum(p.numel() for p in flow.parameters())
    trainable_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)
    T = traj_cfg['default_num_frames']
    p_dim = model_cfg['person_dim']
    c_dim = model_cfg['camera_dim']

    print(f"\n{'='*60}")
    print(f"Flow Matching + Data Augmentation Training")
    print(f"{'='*60}")
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"  Joint dim: person ({T}x{p_dim}={T*p_dim}) + camera ({T}x{c_dim}={T*c_dim}) = {T*(p_dim+c_dim)}")
    print(f"  Train set: {len(train_dataset)} samples | Val set: {len(val_dataset)} samples")
    print(f"  Batch: {train_cfg['batch_size']} | Augmentation: {use_augmentation}")
    print(f"  CFG dropout: {cfg_dropout_prob}")
    print(f"  Sampling steps: {config['flow_matching']['num_steps']}")
    print(f"  Text: {'CLIP' if text_encoder else 'Random'}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Logs: {log_dir}")
    print(f"{'='*60}\n")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        flow.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            y = batch['y'].to(device)

            if text_encoder is not None:
                text_embed = text_encoder(batch['texts'])
            else:
                text_embed = torch.randn(y.shape[0], 512, device=device)

            if cfg_dropout_prob > 0:
                drop_mask = torch.rand(y.shape[0], device=device) < cfg_dropout_prob
                text_embed = text_embed.clone()
                text_embed[drop_mask] = 0.0

            shot_types = batch['shot_types'].to(device)
            shot_type = shot_types if (shot_types >= 0).all() else None

            motion_types = batch['motion_types'].to(device)
            motion_type = motion_types if (motion_types >= 0).all() else None

            # Flow loss + get predicted velocity for smooth regularization
            B = y.shape[0]
            person_total = T * p_dim

            t_fm = torch.rand(B, device=device)
            epsilon = torch.randn_like(y)
            x_t = (1.0 - t_fm.unsqueeze(-1)) * epsilon + t_fm.unsqueeze(-1) * y
            v_target = y - epsilon

            t_scaled = ((1.0 - t_fm) * 999).long()
            v_pred = flow.denoiser(x_t, t_scaled, text_embed,
                                   shot_type=shot_type, motion_type=motion_type)

            # Separate person/camera loss with weighting to compensate
            # dimension imbalance (person=144 dims vs camera=288 dims)
            person_weight = train_cfg.get('person_loss_weight', 2.0)
            v_pred_person = v_pred[:, :person_total]
            v_pred_camera = v_pred[:, person_total:]
            v_target_person = v_target[:, :person_total]
            v_target_camera = v_target[:, person_total:]
            loss_person = torch.nn.functional.mse_loss(v_pred_person, v_target_person)
            loss_camera = torch.nn.functional.mse_loss(v_pred_camera, v_target_camera)
            flow_loss = person_weight * loss_person + loss_camera

            # === Smooth loss on PREDICTED velocity (not GT) ===
            # Penalizes non-smooth predicted trajectories to encourage
            # cinematically plausible camera motion.
            smooth_weight = train_cfg.get('smooth_loss_weight', 0.05)
            smooth_loss = torch.tensor(0.0, device=device)
            if smooth_weight > 0:
                # Reconstruct predicted x_0 from v_pred
                x0_pred = x_t + (1.0 - t_fm.unsqueeze(-1)) * v_pred

                # Camera smoothness on predicted trajectory
                cam_pred = x0_pred[:, person_total:].reshape(B, T, c_dim)
                cam_angle_diff = cam_pred[:, 1:, 3:] - cam_pred[:, :-1, 3:]
                cam_pos_diff = cam_pred[:, 1:, :3] - cam_pred[:, :-1, :3]
                angle_smooth = (cam_angle_diff ** 2).mean()
                pos_smooth = (cam_pos_diff ** 2).mean()

                # Person smoothness on predicted trajectory
                per_pred = x0_pred[:, :person_total].reshape(B, T, p_dim)
                per_diff = per_pred[:, 1:] - per_pred[:, :-1]
                per_smooth = (per_diff ** 2).mean()

                smooth_loss = angle_smooth + 0.5 * pos_smooth + 0.5 * per_smooth

            # === Loss 2: Camera look-at loss ===
            # Camera forward vector should roughly point toward person position.
            # This directly penalises the model when it generates cameras facing away.
            lookat_weight = train_cfg.get('lookat_loss_weight', 0.02)
            lookat_loss = torch.tensor(0.0, device=device)
            if lookat_weight > 0:
                B = y.shape[0]
                person_total = T * p_dim
                per_gt = y[:, :person_total].reshape(B, T, p_dim)           # (B, T, 3)
                cam_gt = y[:, person_total:].reshape(B, T, c_dim)           # (B, T, 6)
                cam_pos = cam_gt[:, :, :3]
                az  = cam_gt[:, :, 3]
                el  = cam_gt[:, :, 4]
                # Camera forward direction (world space)
                fwd_x = torch.cos(el) * torch.sin(az)
                fwd_y = -torch.sin(el)
                fwd_z = -torch.cos(el) * torch.cos(az)
                fwd = torch.stack([fwd_x, fwd_y, fwd_z], dim=-1)  # (B, T, 3)
                # Direction from camera to person
                to_person = per_gt - cam_pos
                dist = to_person.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                to_person_norm = to_person / dist
                # 1 - cosine similarity (0 = perfect alignment, 2 = opposite)
                cos_sim = (fwd * to_person_norm).sum(dim=-1)   # (B, T)
                lookat_loss = (1.0 - cos_sim).mean()

            loss = flow_loss + smooth_weight * smooth_loss + lookat_weight * lookat_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), train_cfg['gradient_clip'])
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)

        avg_val_loss = float('nan')
        if len(val_dataset) > 0 and (epoch + 1) % eval_interval == 0:
            flow.eval()
            val_total = 0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    y = batch['y'].to(device)

                    if text_encoder is not None:
                        text_embed = text_encoder(batch['texts'])
                    else:
                        text_embed = torch.randn(y.shape[0], 512, device=device)

                    shot_types = batch['shot_types'].to(device)
                    shot_type = shot_types if (shot_types >= 0).all() else None
                    motion_types = batch['motion_types'].to(device)
                    motion_type = motion_types if (motion_types >= 0).all() else None

                    loss = flow.flow_loss(y, text_embed,
                                          shot_type=shot_type,
                                          motion_type=motion_type)
                    val_total += loss.item()
                    val_batches += 1

            avg_val_loss = val_total / max(val_batches, 1)

        val_losses.append(avg_val_loss)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        if np.isnan(avg_val_loss):
            print(f"Epoch [{epoch+1}/{num_epochs}] Train: {avg_train_loss:.6f}  lr: {current_lr:.2e}  smooth: {smooth_loss.item():.4f}  lookat: {lookat_loss.item():.4f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] Train: {avg_train_loss:.6f}  Val: {avg_val_loss:.6f}  lr: {current_lr:.2e}  smooth: {smooth_loss.item():.4f}  lookat: {lookat_loss.item():.4f}")

        # Save best val checkpoint
        if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(checkpoint_dir, 'fm_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': flow.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': config,
            }, best_path)
            print(f"  New best val: {avg_val_loss:.6f} (epoch {epoch+1})")

        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'fm_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': flow.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': config,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    final_path = os.path.join(checkpoint_dir, 'fm_final.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': flow.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config,
    }, final_path)
    print(f"\nTraining complete! Final model: {final_path}")

    loss_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'eval_interval': eval_interval,
        'num_epochs': num_epochs,
    }
    loss_json_path = os.path.join(log_dir, 'loss_history.json')
    with open(loss_json_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"Loss history saved: {loss_json_path}")

    plot_loss_curves(train_losses, val_losses, eval_interval, num_epochs,
                     save_dir=log_dir)


def plot_loss_curves(train_losses, val_losses, eval_interval, num_epochs,
                     save_dir='.'):
    epochs = list(range(1, len(train_losses) + 1))
    val_epochs = [e for e, v in zip(epochs, val_losses) if not np.isnan(v)]
    val_vals = [v for v in val_losses if not np.isnan(v)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(epochs, train_losses, color='#FF6B6B', linewidth=1.2,
            alpha=0.85, label='Train Loss')
    if val_vals:
        ax.plot(val_epochs, val_vals, color='#4ECDC4', linewidth=2,
                marker='o', markersize=4, label=f'Val Loss (every {eval_interval} ep)')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title(f'Flow Matching Training ({num_epochs} epochs)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    if val_vals:
        best_idx = int(np.argmin(val_vals))
        best_ep, best_val = val_epochs[best_idx], val_vals[best_idx]
        ax.annotate(f'Best val: {best_val:.4f} (ep {best_ep})',
                    xy=(best_ep, best_val),
                    xytext=(best_ep + num_epochs * 0.05, best_val + 0.02),
                    arrowprops=dict(arrowstyle='->', color='#4ECDC4'),
                    fontsize=10, color='#4ECDC4')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'loss_curve.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Loss curve saved: {save_path}")


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    train(config, args)
