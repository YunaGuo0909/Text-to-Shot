"""
Inference script: generate joint person-camera trajectories from text.

Usage:
    python generate.py --checkpoint /transfer/stc-checkpoints/stc_final.pth --text "A person walks toward camera"
    python generate.py --checkpoint /transfer/stc-checkpoints/stc_final.pth --text "Close-up, person stands still" --motion dolly-in
"""

import argparse
import os
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_model(checkpoint_path, device):
    from src.models.denoiser import JointTrajectoryDenoiser
    from src.models.diffusion import GaussianDiffusion

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model_cfg = config['model']
    traj_cfg = config['trajectory']

    denoiser = JointTrajectoryDenoiser(
        person_dim=model_cfg['person_dim'],
        camera_dim=model_cfg['camera_dim'],
        num_frames=traj_cfg['default_num_frames'],
        hidden_dim=model_cfg['hidden_dim'],
        num_layers=model_cfg['num_layers'],
        num_heads=model_cfg['num_heads'],
        text_dim=512, timestep_dim=128,
        num_shot_types=len(config['shot_types']['categories']),
        shot_type_dim=config['shot_types']['embedding_dim'],
        num_motion_types=len(traj_cfg['motion_types']),
        motion_type_dim=traj_cfg.get('motion_type_dim', 64),
        dropout=0.0,
    ).to(device)

    diffusion = GaussianDiffusion(
        denoiser=denoiser,
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
    ).to(device)

    diffusion.load_state_dict(ckpt['model_state_dict'])
    diffusion.eval()
    return diffusion, config


SHOT_TYPE_MAP = {
    "close-up": 0, "medium-shot": 1, "wide-shot": 2,
    "over-the-shoulder": 3, "two-shot": 4,
}
MOTION_TYPE_MAP = {
    "static": 0, "dolly-in": 1, "dolly-out": 2,
    "pan-left": 3, "pan-right": 4, "crane-up": 5,
    "crane-down": 6, "track": 7, "orbit": 8,
}


@torch.no_grad()
def generate(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    diffusion, config = load_model(args.checkpoint, device)

    model_cfg = config['model']
    person_dim = model_cfg['person_dim']
    camera_dim = model_cfg['camera_dim']
    num_frames = config['trajectory']['default_num_frames']
    person_total = person_dim * num_frames

    # Text encoder
    text_encoder = None
    try:
        from src.models.text_encoder import CLIPTextEncoder
        text_encoder = CLIPTextEncoder(
            model_name=config['text_encoder']['model_name'], device=device
        ).to(device)
    except Exception:
        print("CLIP unavailable, using random embeddings.")

    # Encode
    if text_encoder:
        text_embed = text_encoder([args.text])
    else:
        text_embed = torch.randn(1, 512, device=device)

    shot_idx = SHOT_TYPE_MAP.get(args.shot_type, 1)
    motion_idx = MOTION_TYPE_MAP.get(args.motion, 0)
    shot_type = torch.tensor([shot_idx], device=device)
    motion_type = torch.tensor([motion_idx], device=device)

    # Load norm stats for denormalization
    norm_mean = norm_std = None
    norm_stats_path = config['data'].get('norm_stats_path', None)
    if norm_stats_path and os.path.exists(norm_stats_path):
        import json
        with open(norm_stats_path, 'r') as f:
            stats = json.load(f)
        import torch as _torch
        norm_mean = _torch.tensor(stats['mean'], dtype=torch.float32, device=device)
        norm_std = _torch.tensor(stats['std'], dtype=torch.float32, device=device)
        print(f"  Using norm stats from {norm_stats_path}")

    # Generate
    print(f"Generating: \"{args.text}\"")
    print(f"  shot={args.shot_type}, motion={args.motion}, guidance_scale={args.guidance_scale}")

    y = diffusion.sample(text_embed, shot_type=shot_type,
                         motion_type=motion_type, device=device,
                         guidance_scale=args.guidance_scale)

    # Denormalize
    if norm_mean is not None:
        y = y * norm_std + norm_mean

    y_np = y[0].cpu().numpy()
    person_traj = y_np[:person_total].reshape(num_frames, person_dim)
    camera_traj = y_np[person_total:].reshape(num_frames, camera_dim)

    # Save trajectories
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    tag = f"{args.motion}_{args.shot_type}_{time.strftime('%m%d_%H%M%S')}"
    np.save(os.path.join(output_dir, f'gen_person_{tag}.npy'), person_traj)
    np.save(os.path.join(output_dir, f'gen_camera_{tag}.npy'), camera_traj)

    # Visualize
    visualize_joint(person_traj, camera_traj, args.text, args.motion,
                    save_path=os.path.join(output_dir, f'gen_joint_{tag}.png'))

    print(f"Outputs saved to {output_dir}/  [tag: {tag}]")


def visualize_joint(person_traj, camera_traj, text, motion, save_path):
    """Plot person and camera trajectories in 3D and parameter curves."""
    fig = plt.figure(figsize=(16, 6), facecolor='#1a1a2e')

    # 3D paths
    ax1 = fig.add_subplot(131, projection='3d', facecolor='#1a1a2e')
    ax1.plot3D(camera_traj[:, 0], camera_traj[:, 1], camera_traj[:, 2],
               color='#FFE66D', linewidth=2, label='Camera')
    ax1.plot3D(person_traj[:, 0], person_traj[:, 1], person_traj[:, 2],
               color='#4ECDC4', linewidth=2, label='Person')
    ax1.scatter(*camera_traj[0, :3], color='#FFE66D', s=60, marker='o', edgecolors='white')
    ax1.scatter(*person_traj[0], color='#4ECDC4', s=60, marker='^', edgecolors='white')
    ax1.set_title('3D Trajectories', color='white', fontsize=11)
    ax1.legend(fontsize=8, labelcolor='white', framealpha=0.3)
    ax1.tick_params(colors='gray', labelsize=7)

    # Camera parameters
    ax2 = fig.add_subplot(132, facecolor='#2C3E50')
    t = np.linspace(0, 1, len(camera_traj))
    names = ['tx', 'ty', 'tz', 'az', 'el', 'roll']
    colors = ['#FF6B6B', '#FFE66D', '#4ECDC4', '#C44ECD', '#95E66D', '#FF9F43']
    for i, (name, c) in enumerate(zip(names, colors)):
        ax2.plot(t, camera_traj[:, i], color=c, linewidth=1.5, label=name, alpha=0.8)
    ax2.set_title('Camera Parameters', color='white', fontsize=11)
    ax2.legend(fontsize=7, labelcolor='white', framealpha=0.3, ncol=2)
    ax2.tick_params(colors='gray', labelsize=7)
    ax2.grid(alpha=0.15)

    # Person position
    ax3 = fig.add_subplot(133, facecolor='#2C3E50')
    pnames = ['px', 'py', 'pz']
    pcolors = ['#4ECDC4', '#95E66D', '#C44ECD']
    for i, (name, c) in enumerate(zip(pnames, pcolors)):
        ax3.plot(t, person_traj[:, i], color=c, linewidth=1.5, label=name)
    ax3.set_title('Person Position', color='white', fontsize=11)
    ax3.legend(fontsize=8, labelcolor='white', framealpha=0.3)
    ax3.tick_params(colors='gray', labelsize=7)
    ax3.grid(alpha=0.15)

    title = f'{motion} | "{text[:60]}"'
    fig.suptitle(title, color='white', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Visualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Joint Person-Camera Trajectory')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--shot-type', type=str, default='medium-shot',
                        choices=list(SHOT_TYPE_MAP.keys()))
    parser.add_argument('--motion', type=str, default='static',
                        choices=list(MOTION_TYPE_MAP.keys()))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--guidance-scale', type=float, default=3.0,
                        help='Classifier-Free Guidance scale (1.0=off, 3-7=typical)')
    args = parser.parse_args()
    generate(args)


if __name__ == '__main__':
    main()
