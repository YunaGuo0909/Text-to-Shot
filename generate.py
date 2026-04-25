"""
Inference script: generate joint person-camera trajectories from text.

Usage:
    python generate.py --checkpoint /transfer/stc-checkpoints/stc_final.pth --text "A person walks toward camera"
    python generate.py --checkpoint /transfer/stc-checkpoints/stc_final.pth --text "Close-up, person stands still" --motion dolly-in
    python generate.py --checkpoint /transfer/stc-checkpoints/stc_final.pth --text "Wide shot" --ddim --ddim-steps 50
"""

import argparse
import os
import time
import torch
import numpy as np
from scipy.signal import savgol_filter
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
    sampler = "DDIM" if args.ddim else "DDPM"
    print(f"Generating: \"{args.text}\"")
    print(f"  shot={args.shot_type}, motion={args.motion}, guidance_scale={args.guidance_scale}")
    print(f"  sampler={sampler}, steps={args.ddim_steps if args.ddim else 1000}")

    y = diffusion.sample(text_embed, shot_type=shot_type,
                         motion_type=motion_type, device=device,
                         guidance_scale=args.guidance_scale,
                         use_ddim=args.ddim,
                         ddim_steps=args.ddim_steps,
                         ddim_eta=args.ddim_eta)

    # Denormalize
    if norm_mean is not None:
        y = y * norm_std + norm_mean

    y_np = y[0].cpu().numpy()
    person_traj = y_np[:person_total].reshape(num_frames, person_dim)
    camera_traj = y_np[person_total:].reshape(num_frames, camera_dim)

    # Post-processing: smooth trajectories
    if not args.no_smooth:
        person_traj = smooth_trajectory(person_traj, window=args.smooth_window)
        # Camera: angles (dims 3,4,5 = az,el,roll) get stronger smoothing
        camera_traj = smooth_trajectory(camera_traj, window=args.smooth_window,
                                        angle_dims=[3, 4, 5], angle_window=31)
        print(f"  Smoothing applied (position window={args.smooth_window}, angle window=31)")

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


def smooth_trajectory(traj, window=7, polyorder=2, angle_dims=None, angle_window=21):
    """
    Savitzky-Golay filter. Angle dimensions get a larger window for extra smoothness
    (camera orientation should change slowly in real cinematography).

    angle_dims: list of column indices that are angles (e.g. [3,4,5] for az/el/roll)
    angle_window: larger window for angle dims (must be odd, >= window)
    """
    if traj.shape[0] < window:
        return traj
    smoothed = np.zeros_like(traj)
    for d in range(traj.shape[1]):
        if angle_dims and d in angle_dims:
            w = min(angle_window, traj.shape[0] if traj.shape[0] % 2 == 1 else traj.shape[0] - 1)
            w = w if w % 2 == 1 else w - 1
            w = max(w, window)
            smoothed[:, d] = savgol_filter(traj[:, d], window_length=w, polyorder=polyorder)
        else:
            smoothed[:, d] = savgol_filter(traj[:, d], window_length=window,
                                           polyorder=polyorder)
    return smoothed


def _camera_forward(az, el):
    """Compute camera forward direction from azimuth and elevation (radians)."""
    dx = np.cos(el) * np.sin(az)
    dy = -np.sin(el)
    dz = -np.cos(el) * np.cos(az)
    return np.array([dx, dy, dz])


def visualize_joint(person_traj, camera_traj, text, motion, save_path):
    """
    Enhanced visualization: 3D view with camera orientation arrows,
    top-down view, camera parameter curves, and person position curves.
    """
    num_frames = len(camera_traj)
    t_axis = np.linspace(0, 1, num_frames)

    fig = plt.figure(figsize=(20, 10), facecolor='#1a1a2e')

    # ---- Panel 1: 3D trajectories with camera orientation ----
    ax1 = fig.add_subplot(231, projection='3d', facecolor='#1a1a2e')

    # Trajectory lines
    ax1.plot3D(camera_traj[:, 0], camera_traj[:, 1], camera_traj[:, 2],
               color='#FFE66D', linewidth=2, label='Camera path', alpha=0.9)
    ax1.plot3D(person_traj[:, 0], person_traj[:, 1], person_traj[:, 2],
               color='#4ECDC4', linewidth=2, label='Person path', alpha=0.9)

    # Start and end markers
    ax1.scatter(*camera_traj[0, :3], color='#FFE66D', s=80, marker='o',
                edgecolors='white', linewidths=1.5, zorder=5, label='Cam start')
    ax1.scatter(*camera_traj[-1, :3], color='#FF6B6B', s=80, marker='s',
                edgecolors='white', linewidths=1.5, zorder=5, label='Cam end')
    ax1.scatter(*person_traj[0], color='#4ECDC4', s=80, marker='^',
                edgecolors='white', linewidths=1.5, zorder=5)
    ax1.scatter(*person_traj[-1], color='#95E66D', s=80, marker='v',
                edgecolors='white', linewidths=1.5, zorder=5)

    # Camera orientation arrows in 3D (every few frames, orange)
    arrow_interval = max(1, num_frames // 6)
    arrow_len_factor = 0.25 * max(
        np.ptp(camera_traj[:, :3], axis=0).max(),
        np.ptp(person_traj, axis=0).max(),
        0.5
    )
    for i in range(0, num_frames, arrow_interval):
        cx, cy, cz = camera_traj[i, :3]
        az, el = camera_traj[i, 3], camera_traj[i, 4]
        fwd = _camera_forward(az, el) * arrow_len_factor
        ax1.quiver(cx, cy, cz, fwd[0], fwd[1], fwd[2],
                   color='#FF9F43', arrow_length_ratio=0.35,
                   linewidth=2.5, alpha=0.9)

    # Camera-to-person line at start and end
    for idx, ls, alpha in [(0, '-', 0.4), (-1, '--', 0.3)]:
        ax1.plot3D(
            [camera_traj[idx, 0], person_traj[idx, 0]],
            [camera_traj[idx, 1], person_traj[idx, 1]],
            [camera_traj[idx, 2], person_traj[idx, 2]],
            color='white', linewidth=1, linestyle=ls, alpha=alpha
        )

    ax1.set_title('3D Trajectories + Camera Direction', color='white', fontsize=10)
    ax1.legend(fontsize=7, labelcolor='white', framealpha=0.3, loc='upper left')
    ax1.tick_params(colors='gray', labelsize=6)

    # ---- Panel 2: Top-down (XZ) view ----
    ax2 = fig.add_subplot(232, facecolor='#2C3E50')

    # Paths
    ax2.plot(camera_traj[:, 0], camera_traj[:, 2], color='#FFE66D',
             linewidth=2, label='Camera', alpha=0.9)
    ax2.plot(person_traj[:, 0], person_traj[:, 2], color='#4ECDC4',
             linewidth=2, label='Person', alpha=0.9)

    # Start/end markers
    ax2.scatter(camera_traj[0, 0], camera_traj[0, 2], color='#FFE66D',
                s=70, marker='o', edgecolors='white', zorder=5)
    ax2.scatter(camera_traj[-1, 0], camera_traj[-1, 2], color='#FF6B6B',
                s=70, marker='s', edgecolors='white', zorder=5)
    ax2.scatter(person_traj[0, 0], person_traj[0, 2], color='#4ECDC4',
                s=70, marker='^', edgecolors='white', zorder=5)
    ax2.scatter(person_traj[-1, 0], person_traj[-1, 2], color='#95E66D',
                s=70, marker='v', edgecolors='white', zorder=5)

    # Camera frustum triangles in top-down (shows where camera is looking)
    scene_extent = max(
        np.ptp(camera_traj[:, 0]), np.ptp(camera_traj[:, 2]),
        np.ptp(person_traj[:, 0]), np.ptp(person_traj[:, 2]),
        0.5
    )
    frust_len = scene_extent * 0.15     # how far the triangle tip extends
    frust_half_w = frust_len * 0.4      # half-width of the triangle base
    for i in range(0, num_frames, arrow_interval):
        cx, cz = camera_traj[i, 0], camera_traj[i, 2]
        az = camera_traj[i, 3]
        # forward direction in XZ plane
        fx, fz = np.sin(az), -np.cos(az)
        # perpendicular direction
        rx, rz = fz, -fx
        # triangle: tip = forward, two base corners behind
        tip_x = cx + fx * frust_len
        tip_z = cz + fz * frust_len
        bl_x = cx - rx * frust_half_w
        bl_z = cz - rz * frust_half_w
        br_x = cx + rx * frust_half_w
        br_z = cz + rz * frust_half_w
        tri = plt.Polygon(
            [[tip_x, tip_z], [bl_x, bl_z], [br_x, br_z]],
            color='#FF9F43', alpha=0.5, edgecolor='white', linewidth=0.5
        )
        ax2.add_patch(tri)

    # Camera-to-person line at start and end
    for idx, ls, alpha in [(0, ':', 0.4), (-1, ':', 0.25)]:
        ax2.plot([camera_traj[idx, 0], person_traj[idx, 0]],
                 [camera_traj[idx, 2], person_traj[idx, 2]],
                 color='white', linewidth=1, linestyle=ls, alpha=alpha)

    ax2.set_xlabel('X', color='gray', fontsize=9)
    ax2.set_ylabel('Z', color='gray', fontsize=9)
    ax2.set_title('Top-Down View (XZ) + Look Direction', color='white', fontsize=10)
    ax2.legend(fontsize=7, labelcolor='white', framealpha=0.3)
    ax2.tick_params(colors='gray', labelsize=7)
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.grid(alpha=0.15)

    # ---- Panel 3: Camera position over time ----
    ax3 = fig.add_subplot(233, facecolor='#2C3E50')
    for i, (name, c) in enumerate(zip(['tx', 'ty', 'tz'],
                                       ['#FF6B6B', '#FFE66D', '#4ECDC4'])):
        ax3.plot(t_axis, camera_traj[:, i], color=c, linewidth=1.5, label=name)
    ax3.set_title('Camera Position', color='white', fontsize=10)
    ax3.set_xlabel('time', color='gray', fontsize=8)
    ax3.legend(fontsize=8, labelcolor='white', framealpha=0.3)
    ax3.tick_params(colors='gray', labelsize=7)
    ax3.grid(alpha=0.15)

    # ---- Panel 4: Camera orientation over time ----
    ax4 = fig.add_subplot(234, facecolor='#2C3E50')
    for i, (name, c) in enumerate(zip(['azimuth', 'elevation', 'roll'],
                                       ['#C44ECD', '#95E66D', '#FF9F43'])):
        ax4.plot(t_axis, np.degrees(camera_traj[:, 3 + i]), color=c,
                 linewidth=1.5, label=name)
    ax4.set_title('Camera Orientation (degrees)', color='white', fontsize=10)
    ax4.set_xlabel('time', color='gray', fontsize=8)
    ax4.set_ylabel('degrees', color='gray', fontsize=8)
    ax4.legend(fontsize=8, labelcolor='white', framealpha=0.3)
    ax4.tick_params(colors='gray', labelsize=7)
    ax4.grid(alpha=0.15)

    # ---- Panel 5: Person position over time ----
    ax5 = fig.add_subplot(235, facecolor='#2C3E50')
    for i, (name, c) in enumerate(zip(['px', 'py', 'pz'],
                                       ['#4ECDC4', '#95E66D', '#C44ECD'])):
        ax5.plot(t_axis, person_traj[:, i], color=c, linewidth=1.5, label=name)
    ax5.set_title('Person Position', color='white', fontsize=10)
    ax5.set_xlabel('time', color='gray', fontsize=8)
    ax5.legend(fontsize=8, labelcolor='white', framealpha=0.3)
    ax5.tick_params(colors='gray', labelsize=7)
    ax5.grid(alpha=0.15)

    # ---- Panel 6: Camera-to-person distance over time ----
    ax6 = fig.add_subplot(236, facecolor='#2C3E50')
    dist = np.linalg.norm(camera_traj[:, :3] - person_traj[:, :3], axis=1)
    ax6.plot(t_axis, dist, color='#FF6B6B', linewidth=2, label='distance')
    ax6.fill_between(t_axis, 0, dist, color='#FF6B6B', alpha=0.15)
    ax6.set_title('Camera-Person Distance', color='white', fontsize=10)
    ax6.set_xlabel('time', color='gray', fontsize=8)
    ax6.set_ylabel('metres', color='gray', fontsize=8)
    ax6.legend(fontsize=8, labelcolor='white', framealpha=0.3)
    ax6.tick_params(colors='gray', labelsize=7)
    ax6.grid(alpha=0.15)

    # Title
    title = f'{motion} | "{text[:70]}"'
    fig.suptitle(title, color='white', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
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
    # DDIM sampling
    parser.add_argument('--ddim', action='store_true',
                        help='Use DDIM deterministic sampling (recommended)')
    parser.add_argument('--ddim-steps', type=int, default=50,
                        help='Number of DDIM sampling steps')
    parser.add_argument('--ddim-eta', type=float, default=0.0,
                        help='DDIM stochasticity (0=deterministic, 1=DDPM-like)')
    # Smoothing
    parser.add_argument('--no-smooth', action='store_true',
                        help='Disable trajectory smoothing post-processing')
    parser.add_argument('--smooth-window', type=int, default=7,
                        help='Savitzky-Golay smoothing window size (odd number)')
    args = parser.parse_args()
    generate(args)


if __name__ == '__main__':
    main()
