"""
Animated 3D trajectory visualization + camera POV rendering.

Produces:
  1. trajectory_anim.gif  - 3D bird's-eye view with trajectories drawn over time
  2. camera_pov.gif       - first-person view from the generated camera
  3. combined.gif         - side-by-side split screen

Usage:
    python scripts/visualize_animated.py --person path/to/person.npy --camera path/to/camera.npy --output-dir ./output
    python scripts/visualize_animated.py --person path/to/person.npy --camera path/to/camera.npy --title "dolly-in" --fps 12
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa
from PIL import Image


def camera_forward(az, el):
    fx = np.cos(el) * np.sin(az)
    fy = -np.sin(el)
    fz = -np.cos(el) * np.cos(az)
    return np.array([fx, fy, fz])


def draw_ground_grid(ax, center, size=4, n=10, color='gray', alpha=0.15):
    """Draw a flat grid on the XZ plane at y=center[1]-0.5."""
    y = center[1] - 0.5
    for i in np.linspace(-size/2, size/2, n):
        ax.plot([center[0]+i, center[0]+i], [y, y],
                [center[2]-size/2, center[2]+size/2], color=color, alpha=alpha, lw=0.5)
        ax.plot([center[0]-size/2, center[0]+size/2], [y, y],
                [center[2]+i, center[2]+i], color=color, alpha=alpha, lw=0.5)


def draw_person_marker(ax, pos, color='#4ECDC4', size=80):
    """Draw person as a sphere + vertical line."""
    ax.scatter(*pos, color=color, s=size, alpha=0.9, edgecolors='white', linewidths=0.5)
    ax.plot([pos[0], pos[0]], [pos[1]-0.3, pos[1]+0.3], [pos[2], pos[2]],
            color=color, lw=2, alpha=0.6)


def create_trajectory_animation(person_traj, camera_traj, title='', fps=12):
    """3D bird's-eye animation showing trajectories drawn over time."""
    T = len(person_traj)
    fig = plt.figure(figsize=(8, 8), facecolor='#1a1a2e')
    ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')

    all_pts = np.vstack([person_traj, camera_traj[:, :3]])
    mid = all_pts.mean(axis=0)
    span = max(all_pts.ptp(axis=0).max() * 0.6, 1.0)

    frames = []
    for t in range(T):
        ax.cla()
        ax.set_facecolor('#1a1a2e')
        ax.set_xlim(mid[0]-span, mid[0]+span)
        ax.set_ylim(mid[1]-span, mid[1]+span)
        ax.set_zlim(mid[2]-span, mid[2]+span)
        ax.tick_params(colors='gray', labelsize=6)
        ax.set_title(f'{title}  [frame {t+1}/{T}]', color='white', fontsize=11)

        draw_ground_grid(ax, mid, size=span*1.5)

        # Trajectory trails
        if t > 0:
            ax.plot3D(camera_traj[:t+1, 0], camera_traj[:t+1, 1], camera_traj[:t+1, 2],
                      color='#FFE66D', linewidth=2, alpha=0.7)
            ax.plot3D(person_traj[:t+1, 0], person_traj[:t+1, 1], person_traj[:t+1, 2],
                      color='#4ECDC4', linewidth=2, alpha=0.7)

        # Current positions
        cx, cy, cz = camera_traj[t, :3]
        ax.scatter(cx, cy, cz, color='#FFE66D', s=100, marker='o',
                   edgecolors='white', linewidths=1.5, zorder=5)

        draw_person_marker(ax, person_traj[t])

        # Camera direction arrow
        az, el = camera_traj[t, 3], camera_traj[t, 4]
        fwd = camera_forward(az, el) * span * 0.2
        ax.quiver(cx, cy, cz, fwd[0], fwd[1], fwd[2],
                  color='#FF9F43', arrow_length_ratio=0.3, linewidth=2.5)

        # Camera-to-person line
        px, py, pz = person_traj[t]
        ax.plot3D([cx, px], [cy, py], [cz, pz],
                  color='white', linewidth=0.8, linestyle=':', alpha=0.3)

        ax.view_init(elev=25, azim=-60 + t * 0.5)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames.append(Image.fromarray(img))

    plt.close(fig)
    return frames


def create_pov_animation(person_traj, camera_traj, title='', fps=12):
    """First-person camera POV: render what the camera sees each frame."""
    T = len(person_traj)

    all_pts = np.vstack([person_traj, camera_traj[:, :3]])
    mid = all_pts.mean(axis=0)
    scene_span = max(all_pts.ptp(axis=0).max(), 2.0)

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(8, 6), facecolor='#0d1117')
        ax = fig.add_subplot(111, projection='3d', facecolor='#0d1117')

        cx, cy, cz = camera_traj[t, :3]
        az_rad, el_rad = camera_traj[t, 3], camera_traj[t, 4]
        az_deg = np.degrees(az_rad)
        el_deg = np.degrees(el_rad)

        # Scene bounds centered on camera look-at target
        fwd = camera_forward(az_rad, el_rad)
        look_target = np.array([cx, cy, cz]) + fwd * 3.0
        view_span = scene_span * 0.8

        ax.set_xlim(look_target[0]-view_span, look_target[0]+view_span)
        ax.set_ylim(look_target[1]-view_span, look_target[1]+view_span)
        ax.set_zlim(look_target[2]-view_span, look_target[2]+view_span)

        # Ground grid
        draw_ground_grid(ax, look_target, size=view_span*2, n=15, alpha=0.1)

        # Person marker (sphere + vertical line)
        px, py, pz = person_traj[t]
        dist_to_cam = np.linalg.norm(person_traj[t] - camera_traj[t, :3])
        marker_size = max(20, min(300, 150 / max(dist_to_cam, 0.5)))
        draw_person_marker(ax, person_traj[t], size=marker_size)

        # Person trajectory trail (faded)
        if t > 0:
            ax.plot3D(person_traj[:t+1, 0], person_traj[:t+1, 1], person_traj[:t+1, 2],
                      color='#4ECDC4', linewidth=1, alpha=0.3)

        # Set camera viewpoint to match generated camera orientation
        # matplotlib view_init: elev = angle above XY plane, azim = rotation around Z
        # Our convention: azimuth = rotation in XZ plane, elevation = angle from horizon
        ax.view_init(elev=-el_deg + 10, azim=-az_deg - 90)

        ax.set_axis_off()
        dist_str = f'{dist_to_cam:.1f}m'
        ax.set_title(f'Camera POV | {title} | frame {t+1}/{T} | dist={dist_str}',
                     color='white', fontsize=10)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames.append(Image.fromarray(img))
        plt.close(fig)

    return frames


def combine_frames(traj_frames, pov_frames):
    """Side-by-side combination of trajectory and POV frames."""
    combined = []
    for tf, pf in zip(traj_frames, pov_frames):
        tw, th = tf.size
        pw, ph = pf.size
        # Resize to same height
        target_h = min(th, ph)
        tf_r = tf.resize((int(tw * target_h / th), target_h), Image.LANCZOS)
        pf_r = pf.resize((int(pw * target_h / ph), target_h), Image.LANCZOS)
        total_w = tf_r.size[0] + pf_r.size[0]
        canvas = Image.new('RGB', (total_w, target_h), (13, 17, 23))
        canvas.paste(tf_r, (0, 0))
        canvas.paste(pf_r, (tf_r.size[0], 0))
        combined.append(canvas)
    return combined


def save_gif(frames, path, fps=12):
    duration = int(1000 / fps)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)
    print(f"  Saved: {path} ({len(frames)} frames, {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description='Animated trajectory visualization')
    parser.add_argument('--person', type=str, required=True, help='Person trajectory .npy (T, 3)')
    parser.add_argument('--camera', type=str, required=True, help='Camera trajectory .npy (T, 6)')
    parser.add_argument('--output-dir', type=str, default='.')
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--no-pov', action='store_true', help='Skip POV animation (faster)')
    parser.add_argument('--no-combined', action='store_true', help='Skip combined animation')
    args = parser.parse_args()

    person_traj = np.load(args.person).astype(np.float32)
    camera_traj = np.load(args.camera).astype(np.float32)

    assert person_traj.shape == (48, 3), f"Expected person (48,3), got {person_traj.shape}"
    assert camera_traj.shape == (48, 6), f"Expected camera (48,6), got {camera_traj.shape}"

    os.makedirs(args.output_dir, exist_ok=True)
    title = args.title or 'Generated'

    print(f"Generating trajectory animation ({len(person_traj)} frames)...")
    traj_frames = create_trajectory_animation(person_traj, camera_traj, title, args.fps)
    save_gif(traj_frames, os.path.join(args.output_dir, 'trajectory_anim.gif'), args.fps)

    if not args.no_pov:
        print(f"Generating camera POV animation...")
        pov_frames = create_pov_animation(person_traj, camera_traj, title, args.fps)
        save_gif(pov_frames, os.path.join(args.output_dir, 'camera_pov.gif'), args.fps)

        if not args.no_combined:
            print(f"Generating combined split-screen...")
            combined = combine_frames(traj_frames, pov_frames)
            save_gif(combined, os.path.join(args.output_dir, 'combined.gif'), args.fps)

    print("Done!")


if __name__ == '__main__':
    main()
