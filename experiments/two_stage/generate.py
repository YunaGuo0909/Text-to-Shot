"""
Two-stage inference: Text -> Person trajectory -> Camera trajectory.

Usage:
    PYTHONPATH=. python experiments/two_stage/generate.py \
        --stage1-ckpt /transfer/two-stage-checkpoints/stage1/stage1_final.pth \
        --stage2-ckpt /transfer/two-stage-checkpoints/stage2/stage2_final.pth \
        --text "A person walks toward camera" --motion dolly-in

    PYTHONPATH=. python experiments/two_stage/generate.py \
        --stage1-ckpt /transfer/two-stage-checkpoints/stage1/stage1_final.pth \
        --stage2-ckpt /transfer/two-stage-checkpoints/stage2/stage2_final.pth \
        --text "Wide shot, person stands still" --ddim --guidance-scale 5.0
"""

import argparse
import os
import time
import json
import torch
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.two_stage.models.stage1_denoiser import Stage1PersonDenoiser
from experiments.two_stage.models.stage2_denoiser import Stage2CameraDenoiser
from experiments.two_stage.models.diffusion import StageDiffusion


MOTION_TYPE_MAP = {
    "static": 0, "dolly-in": 1, "dolly-out": 2,
    "pan-left": 3, "pan-right": 4, "crane-up": 5,
    "crane-down": 6, "track": 7, "orbit": 8,
}


def load_stage1(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model_cfg = config['model']
    traj_cfg = config['trajectory']

    denoiser = Stage1PersonDenoiser(
        person_dim=model_cfg['person_dim'],
        num_frames=traj_cfg['default_num_frames'],
        hidden_dim=model_cfg['hidden_dim'],
        num_layers=model_cfg['num_layers'],
        num_heads=model_cfg['num_heads'],
        text_dim=512, timestep_dim=128,
        num_motion_types=len(traj_cfg['motion_types']),
        motion_type_dim=traj_cfg.get('motion_type_dim', 64),
        dropout=0.0,
    ).to(device)

    diffusion = StageDiffusion(
        denoiser=denoiser,
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
    ).to(device)

    diffusion.load_state_dict(ckpt['model_state_dict'])
    diffusion.eval()
    return diffusion, config


def load_stage2(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model_cfg = config['model']
    traj_cfg = config['trajectory']

    denoiser = Stage2CameraDenoiser(
        camera_dim=model_cfg['camera_dim'],
        person_dim=model_cfg['person_dim'],
        num_frames=traj_cfg['default_num_frames'],
        hidden_dim=model_cfg['hidden_dim'],
        num_layers=model_cfg['num_layers'],
        num_heads=model_cfg['num_heads'],
        text_dim=512, timestep_dim=128,
        num_motion_types=len(traj_cfg['motion_types']),
        motion_type_dim=traj_cfg.get('motion_type_dim', 64),
        dropout=0.0,
    ).to(device)

    diffusion = StageDiffusion(
        denoiser=denoiser,
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
    ).to(device)

    diffusion.load_state_dict(ckpt['model_state_dict'])
    diffusion.eval()
    return diffusion, config


def smooth_trajectory(traj, window=7, polyorder=2):
    if traj.shape[0] < window:
        return traj
    smoothed = np.zeros_like(traj)
    for d in range(traj.shape[1]):
        smoothed[:, d] = savgol_filter(traj[:, d], window_length=window,
                                       polyorder=polyorder)
    return smoothed


def _camera_forward(az, el):
    dx = np.cos(el) * np.sin(az)
    dy = -np.sin(el)
    dz = -np.cos(el) * np.cos(az)
    return np.array([dx, dy, dz])


def visualize_joint(person_traj, camera_traj, text, motion, save_path):
    num_frames = len(camera_traj)
    t_axis = np.linspace(0, 1, num_frames)

    fig = plt.figure(figsize=(20, 10), facecolor='#1a1a2e')

    # 3D trajectories
    ax1 = fig.add_subplot(231, projection='3d', facecolor='#1a1a2e')
    ax1.plot3D(camera_traj[:, 0], camera_traj[:, 1], camera_traj[:, 2],
               color='#FFE66D', linewidth=2, label='Camera', alpha=0.9)
    ax1.plot3D(person_traj[:, 0], person_traj[:, 1], person_traj[:, 2],
               color='#4ECDC4', linewidth=2, label='Person', alpha=0.9)
    ax1.scatter(*camera_traj[0, :3], color='#FFE66D', s=80, marker='o',
                edgecolors='white', linewidths=1.5, zorder=5)
    ax1.scatter(*camera_traj[-1, :3], color='#FF6B6B', s=80, marker='s',
                edgecolors='white', linewidths=1.5, zorder=5)
    ax1.scatter(*person_traj[0], color='#4ECDC4', s=80, marker='^',
                edgecolors='white', linewidths=1.5, zorder=5)
    ax1.scatter(*person_traj[-1], color='#95E66D', s=80, marker='v',
                edgecolors='white', linewidths=1.5, zorder=5)

    arrow_interval = max(1, num_frames // 6)
    arrow_len = 0.25 * max(np.ptp(camera_traj[:, :3], axis=0).max(),
                           np.ptp(person_traj, axis=0).max(), 0.5)
    for i in range(0, num_frames, arrow_interval):
        cx, cy, cz = camera_traj[i, :3]
        fwd = _camera_forward(camera_traj[i, 3], camera_traj[i, 4]) * arrow_len
        ax1.quiver(cx, cy, cz, fwd[0], fwd[1], fwd[2],
                   color='#FF9F43', arrow_length_ratio=0.35, linewidth=2.5, alpha=0.9)

    ax1.set_title('3D Trajectories (Two-Stage)', color='white', fontsize=10)
    ax1.legend(fontsize=7, labelcolor='white', framealpha=0.3)
    ax1.tick_params(colors='gray', labelsize=6)

    # Top-down XZ
    ax2 = fig.add_subplot(232, facecolor='#2C3E50')
    ax2.plot(camera_traj[:, 0], camera_traj[:, 2], color='#FFE66D', linewidth=2, label='Camera')
    ax2.plot(person_traj[:, 0], person_traj[:, 2], color='#4ECDC4', linewidth=2, label='Person')
    ax2.scatter(camera_traj[0, 0], camera_traj[0, 2], color='#FFE66D', s=70,
                marker='o', edgecolors='white', zorder=5)
    ax2.scatter(camera_traj[-1, 0], camera_traj[-1, 2], color='#FF6B6B', s=70,
                marker='s', edgecolors='white', zorder=5)
    ax2.set_xlabel('X', color='gray')
    ax2.set_ylabel('Z', color='gray')
    ax2.set_title('Top-Down (XZ)', color='white', fontsize=10)
    ax2.legend(fontsize=7, labelcolor='white', framealpha=0.3)
    ax2.tick_params(colors='gray', labelsize=7)
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.grid(alpha=0.15)

    # Camera position
    ax3 = fig.add_subplot(233, facecolor='#2C3E50')
    for i, (name, c) in enumerate(zip(['tx', 'ty', 'tz'],
                                       ['#FF6B6B', '#FFE66D', '#4ECDC4'])):
        ax3.plot(t_axis, camera_traj[:, i], color=c, linewidth=1.5, label=name)
    ax3.set_title('Camera Position', color='white', fontsize=10)
    ax3.legend(fontsize=8, labelcolor='white', framealpha=0.3)
    ax3.tick_params(colors='gray', labelsize=7)
    ax3.grid(alpha=0.15)

    # Camera orientation
    ax4 = fig.add_subplot(234, facecolor='#2C3E50')
    for i, (name, c) in enumerate(zip(['azimuth', 'elevation', 'roll'],
                                       ['#C44ECD', '#95E66D', '#FF9F43'])):
        ax4.plot(t_axis, np.degrees(camera_traj[:, 3 + i]), color=c, linewidth=1.5, label=name)
    ax4.set_title('Camera Orientation (deg)', color='white', fontsize=10)
    ax4.legend(fontsize=8, labelcolor='white', framealpha=0.3)
    ax4.tick_params(colors='gray', labelsize=7)
    ax4.grid(alpha=0.15)

    # Person position
    ax5 = fig.add_subplot(235, facecolor='#2C3E50')
    for i, (name, c) in enumerate(zip(['px', 'py', 'pz'],
                                       ['#4ECDC4', '#95E66D', '#C44ECD'])):
        ax5.plot(t_axis, person_traj[:, i], color=c, linewidth=1.5, label=name)
    ax5.set_title('Person Position', color='white', fontsize=10)
    ax5.legend(fontsize=8, labelcolor='white', framealpha=0.3)
    ax5.tick_params(colors='gray', labelsize=7)
    ax5.grid(alpha=0.15)

    # Distance
    ax6 = fig.add_subplot(236, facecolor='#2C3E50')
    dist = np.linalg.norm(camera_traj[:, :3] - person_traj[:, :3], axis=1)
    ax6.plot(t_axis, dist, color='#FF6B6B', linewidth=2)
    ax6.fill_between(t_axis, 0, dist, color='#FF6B6B', alpha=0.15)
    ax6.set_title('Camera-Person Distance', color='white', fontsize=10)
    ax6.tick_params(colors='gray', labelsize=7)
    ax6.grid(alpha=0.15)

    title = f'Two-Stage | {motion} | "{text[:70]}"'
    fig.suptitle(title, color='white', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Visualization saved: {save_path}")


@torch.no_grad()
def generate(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load both stages
    print("Loading Stage 1...")
    stage1_diffusion, stage1_config = load_stage1(args.stage1_ckpt, device)
    print("Loading Stage 2...")
    stage2_diffusion, stage2_config = load_stage2(args.stage2_ckpt, device)

    person_dim = stage1_config['model']['person_dim']
    camera_dim = stage2_config['model']['camera_dim']
    num_frames = stage1_config['trajectory']['default_num_frames']
    person_total = person_dim * num_frames
    camera_total = camera_dim * num_frames

    # Text encoder
    text_encoder = None
    try:
        from src.models.text_encoder import CLIPTextEncoder
        model_name = stage1_config['text_encoder']['model_name']
        text_encoder = CLIPTextEncoder(model_name=model_name, device=device).to(device)
        print("CLIP loaded.")
    except Exception:
        print("CLIP unavailable, using random embeddings.")

    # Encode text
    if text_encoder:
        text_embed = text_encoder([args.text])
    else:
        text_embed = torch.randn(1, 512, device=device)

    motion_idx = MOTION_TYPE_MAP.get(args.motion, 0)
    motion_type = torch.tensor([motion_idx], device=device)

    # Load norm stats
    norm_mean = norm_std = None
    norm_stats_path = stage1_config['data'].get('norm_stats_path', None)
    if norm_stats_path and os.path.exists(norm_stats_path):
        with open(norm_stats_path, 'r') as f:
            stats = json.load(f)
        norm_mean = torch.tensor(stats['mean'], dtype=torch.float32, device=device)
        norm_std = torch.tensor(stats['std'], dtype=torch.float32, device=device)
        print(f"Norm stats loaded from {norm_stats_path}")

    sampler_name = "DDIM" if args.ddim else "DDPM"
    steps = args.ddim_steps if args.ddim else 1000
    print(f"\nGenerating: \"{args.text}\"")
    print(f"  motion={args.motion}, guidance_s1={args.guidance_scale_s1}, guidance_s2={args.guidance_scale_s2}")
    print(f"  sampler={sampler_name}, steps={steps}")

    # Stage 1: Text -> Person trajectory
    print("\n--- Stage 1: Generating person trajectory ---")
    t0 = time.time()
    person_flat = stage1_diffusion.sample(
        text_embed, motion_type=motion_type, device=device,
        guidance_scale=args.guidance_scale_s1,
        use_ddim=args.ddim, ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta,
    )
    print(f"  Stage 1 done in {time.time() - t0:.1f}s")

    # Stage 2: Text + Person -> Camera trajectory
    print("--- Stage 2: Generating camera trajectory ---")
    t0 = time.time()
    camera_flat = stage2_diffusion.sample(
        text_embed, motion_type=motion_type, device=device,
        guidance_scale=args.guidance_scale_s2,
        use_ddim=args.ddim, ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta,
        person_traj=person_flat,
    )
    print(f"  Stage 2 done in {time.time() - t0:.1f}s")

    # Denormalize
    if norm_mean is not None:
        person_mean = norm_mean[:person_total]
        person_std = norm_std[:person_total]
        camera_mean = norm_mean[person_total:]
        camera_std = norm_std[person_total:]
        person_flat = person_flat * person_std + person_mean
        camera_flat = camera_flat * camera_std + camera_mean

    person_np = person_flat[0].cpu().numpy().reshape(num_frames, person_dim)
    camera_np = camera_flat[0].cpu().numpy().reshape(num_frames, camera_dim)

    # Smooth
    if not args.no_smooth:
        person_np = smooth_trajectory(person_np, window=args.smooth_window)
        camera_np = smooth_trajectory(camera_np, window=args.smooth_window)
        print(f"  Smoothing applied (window={args.smooth_window})")

    # Save
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    tag = f"2stage_{args.motion}_{time.strftime('%m%d_%H%M%S')}"
    np.save(os.path.join(output_dir, f'gen_person_{tag}.npy'), person_np)
    np.save(os.path.join(output_dir, f'gen_camera_{tag}.npy'), camera_np)

    # Visualize
    visualize_joint(person_np, camera_np, args.text, args.motion,
                    save_path=os.path.join(output_dir, f'gen_joint_{tag}.png'))

    print(f"\nOutputs saved to {output_dir}/  [tag: {tag}]")


def main():
    parser = argparse.ArgumentParser(description='Two-Stage Generation')
    parser.add_argument('--stage1-ckpt', type=str, required=True)
    parser.add_argument('--stage2-ckpt', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--motion', type=str, default='static',
                        choices=list(MOTION_TYPE_MAP.keys()))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--guidance-scale-s1', type=float, default=3.0,
                        help='CFG scale for Stage 1 (person)')
    parser.add_argument('--guidance-scale-s2', type=float, default=3.0,
                        help='CFG scale for Stage 2 (camera)')
    parser.add_argument('--output-dir', type=str, default='/transfer/two-stage-outputs')
    # DDIM
    parser.add_argument('--ddim', action='store_true')
    parser.add_argument('--ddim-steps', type=int, default=50)
    parser.add_argument('--ddim-eta', type=float, default=0.0)
    # Smoothing
    parser.add_argument('--no-smooth', action='store_true')
    parser.add_argument('--smooth-window', type=int, default=7)
    args = parser.parse_args()
    generate(args)


if __name__ == '__main__':
    main()
