"""
3D Camera Trajectory Animation Visualizer.

Generates an animated GIF showing the camera moving through 3D space
along a generated trajectory. Supports both model-generated and
rule-based trajectories.

Usage:
    # From trained model
    python visualize_3d.py --scene "Camera dollies in slowly" --checkpoint checkpoints/checkpoint_final.pth --motion dolly-in

    # From demo (rule-based)
    python visualize_3d.py --demo --motion orbit

    # All motion types comparison
    python visualize_3d.py --compare-motions --checkpoint checkpoints/checkpoint_final.pth
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional

from src.pipeline.camera_trajectory import CameraTrajectory, CameraTrajectoryGenerator


def trajectory_to_positions(traj_data: np.ndarray):
    """
    Convert 6D trajectory (tx, ty, tz, azimuth, elevation, roll)
    to 3D positions and look-at directions for visualization.

    Args:
        traj_data: (T, 6) array

    Returns:
        positions: (T, 3) camera XYZ positions
        directions: (T, 3) camera forward direction vectors
    """
    positions = traj_data[:, :3].copy()
    azimuth = traj_data[:, 3]
    elevation = traj_data[:, 4]

    # Forward direction from azimuth & elevation
    dx = np.cos(elevation) * np.cos(azimuth)
    dy = np.cos(elevation) * np.sin(azimuth)
    dz = np.sin(elevation)
    directions = np.stack([dx, dy, dz], axis=-1)

    return positions, directions


def draw_camera_frustum(ax, pos, direction, size=0.15, color='cyan', alpha=0.6):
    """Draw a small camera frustum at given position and orientation."""
    d = direction / (np.linalg.norm(direction) + 1e-8)

    # Build local coordinate frame
    up = np.array([0, 0, 1.0])
    if abs(np.dot(d, up)) > 0.99:
        up = np.array([0, 1.0, 0])
    right = np.cross(d, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    cam_up = np.cross(right, d)

    # Frustum corners (front face)
    hw, hh = size * 0.6, size * 0.4
    depth = size
    corners = [
        pos + depth * d + hw * right + hh * cam_up,
        pos + depth * d - hw * right + hh * cam_up,
        pos + depth * d - hw * right - hh * cam_up,
        pos + depth * d + hw * right - hh * cam_up,
    ]

    # Draw front face
    verts = [corners]
    face = Poly3DCollection(verts, alpha=alpha * 0.3, facecolor=color, edgecolor=color, linewidths=0.8)
    ax.add_collection3d(face)

    # Draw edges from camera to corners
    for c in corners:
        ax.plot3D(*zip(pos, c), color=color, alpha=alpha * 0.5, linewidth=0.5)

    # Draw direction arrow
    arrow_end = pos + depth * 1.5 * d
    ax.plot3D(*zip(pos, arrow_end), color=color, alpha=alpha, linewidth=1.5)


def draw_ground_plane(ax, center, size=3.0, color='#334455'):
    """Draw a semi-transparent ground plane grid."""
    grid_n = 8
    half = size / 2
    cx, cy = center[0], center[1]
    z_ground = center[2] - size * 0.3

    for i in range(grid_n + 1):
        t = -half + i * size / grid_n
        ax.plot3D([cx + t, cx + t], [cy - half, cy + half], [z_ground, z_ground],
                  color=color, alpha=0.3, linewidth=0.5)
        ax.plot3D([cx - half, cx + half], [cy + t, cy + t], [z_ground, z_ground],
                  color=color, alpha=0.3, linewidth=0.5)


def create_trajectory_animation(
    traj_data: np.ndarray,
    title: str = "Camera Trajectory",
    save_path: str = "outputs/trajectory_3d.gif",
    fps: int = 12,
    trail_length: int = 10,
):
    """
    Create an animated GIF of the camera moving through 3D space.

    Args:
        traj_data: (T, 6) trajectory array
        title: Plot title
        save_path: Output path (.gif or .mp4)
        fps: Frames per second
        trail_length: Number of past frames to show as trail
    """
    positions, directions = trajectory_to_positions(traj_data)
    T = len(positions)

    center = positions.mean(axis=0)
    spread = max(np.ptp(positions, axis=0).max(), 0.5) * 1.2

    fig = plt.figure(figsize=(10, 8), facecolor='#1a1a2e')
    ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')

    # Styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#334455')
    ax.yaxis.pane.set_edgecolor('#334455')
    ax.zaxis.pane.set_edgecolor('#334455')
    ax.tick_params(colors='gray', labelsize=7)
    ax.set_xlabel('X', color='gray', fontsize=9)
    ax.set_ylabel('Y', color='gray', fontsize=9)
    ax.set_zlabel('Z', color='gray', fontsize=9)

    # Fixed axis limits
    ax.set_xlim(center[0] - spread, center[0] + spread)
    ax.set_ylim(center[1] - spread, center[1] + spread)
    ax.set_zlim(center[2] - spread, center[2] + spread)

    draw_ground_plane(ax, center, size=spread * 1.5)

    # Full path (faded)
    ax.plot3D(positions[:, 0], positions[:, 1], positions[:, 2],
              color='#FFE66D', alpha=0.15, linewidth=1)

    # Start marker
    ax.scatter(*positions[0], color='#4ECDC4', s=60, marker='o',
               edgecolors='white', linewidths=1, zorder=10, label='Start')

    # Dynamic elements
    trail_line, = ax.plot3D([], [], [], color='#FFE66D', linewidth=2.5, alpha=0.8)
    current_dot = ax.scatter([], [], [], color='#FF6B6B', s=80, marker='D',
                             edgecolors='white', linewidths=1.5, zorder=10)

    title_text = ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=15)

    # Info text
    info_text = fig.text(0.02, 0.02, '', color='#95E66D', fontsize=9,
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))

    # Param labels
    param_names = ['tx', 'ty', 'tz', 'azimuth', 'elevation', 'roll']
    param_text = fig.text(0.98, 0.02, '', color='#4ECDC4', fontsize=8,
                          fontfamily='monospace', ha='right',
                          bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))

    ax.legend(loc='upper left', fontsize=8, framealpha=0.3, labelcolor='white')

    def init():
        trail_line.set_data_3d([], [], [])
        current_dot._offsets3d = ([], [], [])
        return trail_line, current_dot

    def update(frame):
        # Clear previous frustum artists
        while len(ax.collections) > 2:
            ax.collections[-1].remove()

        # Trail
        start = max(0, frame - trail_length)
        seg = positions[start:frame + 1]
        trail_line.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])

        # Current position
        current_dot._offsets3d = ([positions[frame, 0]],
                                  [positions[frame, 1]],
                                  [positions[frame, 2]])

        # Camera frustum
        draw_camera_frustum(ax, positions[frame], directions[frame],
                            size=spread * 0.08, color='cyan')

        # Rotate viewpoint slowly
        ax.view_init(elev=25, azim=30 + frame * 0.8)

        # Info
        info_text.set_text(f"Frame {frame+1}/{T}  |  t = {frame/T:.2f}")

        # Parameters
        vals = traj_data[frame]
        lines = [f"{n:>10s}: {v:+.3f}" for n, v in zip(param_names, vals)]
        param_text.set_text('\n'.join(lines))

        return trail_line, current_dot

    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=T, interval=1000 // fps, blit=False,
    )

    os.makedirs(os.path.dirname(save_path) or 'outputs', exist_ok=True)

    if save_path.endswith('.mp4'):
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(save_path, writer=writer, dpi=120)
    else:
        anim.save(save_path, writer='pillow', fps=fps, dpi=120)

    plt.close(fig)
    print(f"Animation saved to: {save_path}")


def create_static_3d_view(
    traj_data: np.ndarray,
    title: str = "Camera Trajectory (3D)",
    save_path: str = "outputs/trajectory_3d_static.png",
    num_cameras: int = 6,
):
    """
    Create a static 3D view with multiple camera frustums along the path.
    Useful for paper figures.
    """
    positions, directions = trajectory_to_positions(traj_data)
    T = len(positions)

    center = positions.mean(axis=0)
    spread = max(np.ptp(positions, axis=0).max(), 0.5) * 1.2

    fig = plt.figure(figsize=(12, 9), facecolor='#1a1a2e')
    ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#334455')
    ax.yaxis.pane.set_edgecolor('#334455')
    ax.zaxis.pane.set_edgecolor('#334455')
    ax.tick_params(colors='gray', labelsize=7)
    ax.set_xlabel('X', color='gray', fontsize=9)
    ax.set_ylabel('Y', color='gray', fontsize=9)
    ax.set_zlabel('Z', color='gray', fontsize=9)

    ax.set_xlim(center[0] - spread, center[0] + spread)
    ax.set_ylim(center[1] - spread, center[1] + spread)
    ax.set_zlim(center[2] - spread, center[2] + spread)

    draw_ground_plane(ax, center, size=spread * 1.5)

    # Full path with gradient color
    for i in range(T - 1):
        frac = i / T
        color = plt.cm.plasma(frac)
        ax.plot3D(positions[i:i+2, 0], positions[i:i+2, 1], positions[i:i+2, 2],
                  color=color, linewidth=2.5, alpha=0.8)

    # Camera frustums at evenly spaced positions
    indices = np.linspace(0, T - 1, num_cameras, dtype=int)
    for i, idx in enumerate(indices):
        frac = idx / T
        color = plt.cm.plasma(frac)
        draw_camera_frustum(ax, positions[idx], directions[idx],
                            size=spread * 0.07, color=color, alpha=0.9)
        ax.text(positions[idx, 0], positions[idx, 1],
                positions[idx, 2] + spread * 0.06,
                f"t={frac:.1f}", color='white', fontsize=7, ha='center')

    # Start / End markers
    ax.scatter(*positions[0], color='#4ECDC4', s=100, marker='o',
               edgecolors='white', linewidths=2, zorder=10, label='Start')
    ax.scatter(*positions[-1], color='#FF6B6B', s=100, marker='s',
               edgecolors='white', linewidths=2, zorder=10, label='End')

    ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.3, labelcolor='white')
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Static 3D view saved to: {save_path}")


def create_motion_comparison(
    trajectories: dict,
    save_path: str = "outputs/motion_comparison_3d.png",
):
    """
    Create a comparison figure showing multiple motion types side by side.

    Args:
        trajectories: dict of {motion_name: (T, 6) array}
    """
    n = len(trajectories)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 6, rows * 5), facecolor='#1a1a2e')

    for i, (motion_name, traj_data) in enumerate(trajectories.items()):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d', facecolor='#1a1a2e')

        positions, directions = trajectory_to_positions(traj_data)
        T = len(positions)
        center = positions.mean(axis=0)
        spread = max(np.ptp(positions, axis=0).max(), 0.5) * 1.2

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#334455')
        ax.yaxis.pane.set_edgecolor('#334455')
        ax.zaxis.pane.set_edgecolor('#334455')
        ax.tick_params(colors='gray', labelsize=6)

        ax.set_xlim(center[0] - spread, center[0] + spread)
        ax.set_ylim(center[1] - spread, center[1] + spread)
        ax.set_zlim(center[2] - spread, center[2] + spread)

        draw_ground_plane(ax, center, size=spread * 1.2)

        for j in range(T - 1):
            color = plt.cm.plasma(j / T)
            ax.plot3D(positions[j:j+2, 0], positions[j:j+2, 1], positions[j:j+2, 2],
                      color=color, linewidth=2, alpha=0.8)

        indices = np.linspace(0, T - 1, 4, dtype=int)
        for idx in indices:
            draw_camera_frustum(ax, positions[idx], directions[idx],
                                size=spread * 0.06, color='cyan', alpha=0.7)

        ax.scatter(*positions[0], color='#4ECDC4', s=60, marker='o',
                   edgecolors='white', linewidths=1.5, zorder=10)
        ax.scatter(*positions[-1], color='#FF6B6B', s=60, marker='s',
                   edgecolors='white', linewidths=1.5, zorder=10)

        ax.set_title(motion_name, color='white', fontsize=12, fontweight='bold')
        ax.view_init(elev=25, azim=45)

    plt.suptitle('Camera Motion Types — 3D Trajectory Comparison',
                 color='white', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Motion comparison saved to: {save_path}")


def demo_rule_based(motion_type='orbit', num_frames=48):
    """
    Generate a synthetic 6D camera trajectory in world space for visualization.
    Uses analytic motion profiles that look good in 3D.
    """
    t = np.linspace(0, 1, num_frames)
    traj = np.zeros((num_frames, 6), dtype=np.float32)

    profiles = {
        'static': lambda: _set_traj(traj, t,
            tx=np.full_like(t, 2.0), ty=np.full_like(t, 0.0), tz=np.full_like(t, 1.5),
            az=np.full_like(t, 0.0), el=np.full_like(t, 0.0), ro=np.full_like(t, 0.0)),
        'dolly-in': lambda: _set_traj(traj, t,
            tx=2.0 - 1.5 * t, ty=np.zeros_like(t), tz=1.5 - 0.3 * t,
            az=np.zeros_like(t), el=-0.1 * t, ro=np.zeros_like(t)),
        'dolly-out': lambda: _set_traj(traj, t,
            tx=0.5 + 1.5 * t, ty=np.zeros_like(t), tz=1.2 + 0.3 * t,
            az=np.zeros_like(t), el=0.05 * t, ro=np.zeros_like(t)),
        'pan-left': lambda: _set_traj(traj, t,
            tx=np.full_like(t, 2.0), ty=np.zeros_like(t), tz=np.full_like(t, 1.5),
            az=-0.8 * t, el=np.zeros_like(t), ro=np.zeros_like(t)),
        'pan-right': lambda: _set_traj(traj, t,
            tx=np.full_like(t, 2.0), ty=np.zeros_like(t), tz=np.full_like(t, 1.5),
            az=0.8 * t, el=np.zeros_like(t), ro=np.zeros_like(t)),
        'crane-up': lambda: _set_traj(traj, t,
            tx=np.full_like(t, 2.0), ty=np.zeros_like(t), tz=1.0 + 2.0 * t,
            az=np.zeros_like(t), el=0.4 * t, ro=np.zeros_like(t)),
        'crane-down': lambda: _set_traj(traj, t,
            tx=np.full_like(t, 2.0), ty=np.zeros_like(t), tz=3.0 - 2.0 * t,
            az=np.zeros_like(t), el=-0.4 * t, ro=np.zeros_like(t)),
        'track': lambda: _set_traj(traj, t,
            tx=np.full_like(t, 2.0), ty=-1.5 + 3.0 * t, tz=np.full_like(t, 1.5),
            az=0.3 * np.sin(t * np.pi), el=np.zeros_like(t), ro=np.zeros_like(t)),
        'orbit': lambda: _set_traj(traj, t,
            tx=2.0 * np.cos(t * 2 * np.pi * 0.8), ty=2.0 * np.sin(t * 2 * np.pi * 0.8),
            tz=1.5 + 0.3 * np.sin(t * np.pi),
            az=t * 2 * np.pi * 0.8 + np.pi, el=-0.1 * np.sin(t * 2 * np.pi),
            ro=0.05 * np.sin(t * 4 * np.pi)),
    }

    builder = profiles.get(motion_type, profiles['static'])
    builder()
    return traj


def _set_traj(traj, t, tx, ty, tz, az, el, ro):
    """Helper to fill trajectory array."""
    traj[:, 0] = tx
    traj[:, 1] = ty
    traj[:, 2] = tz
    traj[:, 3] = az
    traj[:, 4] = el
    traj[:, 5] = ro


def main():
    parser = argparse.ArgumentParser(description='3D Camera Trajectory Visualizer')
    parser.add_argument('--scene', type=str, default=None,
                        help='Scene description for model-based generation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint path')
    parser.add_argument('--motion', type=str, default='dolly-in',
                        help='Camera motion type')
    parser.add_argument('--shot-type', type=str, default='medium-shot')
    parser.add_argument('--output', type=str, default='outputs/trajectory_3d.gif')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--demo', action='store_true',
                        help='Use rule-based trajectory')
    parser.add_argument('--static', action='store_true',
                        help='Generate static PNG instead of GIF')
    parser.add_argument('--compare-motions', action='store_true',
                        help='Generate comparison of all motion types')
    parser.add_argument('--fps', type=int, default=12)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or 'outputs', exist_ok=True)

    if args.compare_motions:
        motion_types = ['static', 'dolly-in', 'dolly-out', 'pan-left',
                        'pan-right', 'crane-up', 'crane-down', 'track', 'orbit']

        if args.checkpoint:
            import torch
            from generate_storyboard import load_model, load_text_encoder, generate_from_text
            device = args.device if torch.cuda.is_available() else 'cpu'
            diffusion, config = load_model(args.checkpoint, device)
            text_encoder = load_text_encoder(config, device)

            trajectories = {}
            scene = args.scene or "A person standing in a room"
            for mt in motion_types:
                trajs = generate_from_text(scene, diffusion, text_encoder,
                                           device=device, camera_motion=mt)
                trajectories[mt] = trajs[0].trajectory
        else:
            trajectories = {}
            for mt in motion_types:
                trajectories[mt] = demo_rule_based(mt)

        base = args.output.rsplit('.', 1)[0]
        create_motion_comparison(trajectories, save_path=f"{base}_comparison.png")

        print("\nAlso generating individual GIFs...")
        for mt in motion_types:
            create_trajectory_animation(
                trajectories[mt], title=f"Motion: {mt}",
                save_path=f"{base}_{mt}.gif", fps=args.fps,
            )
        return

    # Single trajectory
    if args.scene and args.checkpoint:
        import torch
        from generate_storyboard import load_model, load_text_encoder, generate_from_text
        device = args.device if torch.cuda.is_available() else 'cpu'
        diffusion, config = load_model(args.checkpoint, device)
        text_encoder = load_text_encoder(config, device)

        trajs = generate_from_text(args.scene, diffusion, text_encoder,
                                   device=device, camera_motion=args.motion,
                                   shot_type=args.shot_type)
        traj_data = trajs[0].trajectory
        title = f"{args.motion} | \"{args.scene[:50]}\""
    elif args.demo:
        traj_data = demo_rule_based(args.motion)
        title = f"Demo: {args.motion}"
    else:
        print("Usage:")
        print("  Demo:    python visualize_3d.py --demo --motion orbit")
        print("  Model:   python visualize_3d.py --scene '...' --checkpoint ... --motion dolly-in")
        print("  Compare: python visualize_3d.py --compare-motions --checkpoint ...")
        return

    if args.static:
        base = args.output.rsplit('.', 1)[0]
        create_static_3d_view(traj_data, title=title, save_path=f"{base}_static.png")
    else:
        create_trajectory_animation(traj_data, title=title,
                                    save_path=args.output, fps=args.fps)

    # Always also save a static view
    base = args.output.rsplit('.', 1)[0]
    create_static_3d_view(traj_data, title=title, save_path=f"{base}_static.png")


if __name__ == '__main__':
    main()
