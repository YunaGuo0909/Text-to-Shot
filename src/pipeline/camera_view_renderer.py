"""
Camera view renderer: what the camera sees.

Renders the view from the camera at each frame given camera trajectory (T, 6)
and person trajectory (T, 3). The person is drawn as a simple cube (rectangle)
in 2D. Outputs image sequence or GIF for single-person dynamic storyboard.

Supports both world-space camera (tx, ty, tz, azimuth, elevation, roll) and
Toric-space camera (pA_x, pA_y, pB_x, pB_y, theta, phi) when person_trajectory
is provided for conversion.
"""

import os
import numpy as np
from typing import Optional, Tuple
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation


def _toric_state_to_world_camera(
    toric_state: np.ndarray,
    person_pos: np.ndarray,
) -> np.ndarray:
    """
    Convert one Toric camera state (pA_x, pA_y, pB_x, pB_y, theta, phi) to
    world camera (tx, ty, tz, azimuth, elevation, roll) for single-person.
    """
    from src.utils.toric import toric_to_camera_extrinsics, unpack_toric_state
    p_A, p_B, theta, phi = unpack_toric_state(toric_state)
    head_A = person_pos.astype(np.float64)
    head_B = head_A + np.array([0.05, 0.0, 0.0], dtype=np.float64)  # avoid zero dist
    R, t = toric_to_camera_extrinsics(
        float(theta), float(phi),
        p_A, p_B,
        head_A, head_B,
    )
    cam_pos = t
    forward = -R[:, 2]
    az = float(np.arctan2(forward[1], forward[0]))
    el = float(np.arcsin(np.clip(forward[2], -1.0, 1.0)))
    return np.array([cam_pos[0], cam_pos[1], cam_pos[2], az, el, 0.0], dtype=np.float32)


def _ensure_world_trajectory(
    cam_trajectory: np.ndarray,
    person_trajectory: np.ndarray,
    is_toric: bool,
) -> np.ndarray:
    """If is_toric, convert (T,6) Toric to world (T,6) using person_trajectory."""
    if not is_toric:
        return cam_trajectory
    T = min(cam_trajectory.shape[0], person_trajectory.shape[0])
    out = np.zeros((T, 6), dtype=np.float32)
    for t in range(T):
        out[t] = _toric_state_to_world_camera(cam_trajectory[t], person_trajectory[t])
    return out


def camera_view_directions(cam_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From 6D camera state (tx, ty, tz, azimuth, elevation, roll), compute
    camera position and right/up/forward vectors in world space.

    Returns:
        pos: (3,) camera position
        right: (3,) unit vector
        up: (3,) unit vector
        forward: (3,) unit vector (view direction)
    """
    tx, ty, tz = cam_state[0], cam_state[1], cam_state[2]
    az, el, roll = cam_state[3], cam_state[4], cam_state[5]
    pos = np.array([tx, ty, tz], dtype=np.float64)

    # Forward direction (same as visualize_3d trajectory_to_positions)
    dx = np.cos(el) * np.cos(az)
    dy = np.cos(el) * np.sin(az)
    dz = np.sin(el)
    forward = np.array([dx, dy, dz], dtype=np.float64)
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    # World up
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    n = np.linalg.norm(right)
    if n < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / n
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)

    return pos, right, up, forward


def project_world_to_screen(
    cam_state: np.ndarray,
    point_world: np.ndarray,
    focal: float = 500.0,
    image_size: Tuple[int, int] = (256, 256),
) -> Optional[Tuple[float, float, float]]:
    """
    Project a 3D world point into camera image plane.

    Returns:
        (x_pixel, y_pixel, depth) or None if behind camera.
    """
    pos, right, up, forward = camera_view_directions(cam_state)
    v = point_world.astype(np.float64) - pos
    z_cam = np.dot(v, forward)
    if z_cam <= 1e-4:
        return None
    x_cam = np.dot(v, right)
    y_cam = np.dot(v, up)
    # Perspective
    w, h = image_size
    scale = focal / z_cam
    x_ndc = x_cam * scale
    y_ndc = y_cam * scale
    px = w / 2.0 + x_ndc
    py = h / 2.0 - y_ndc  # flip Y for image
    return (float(px), float(py), float(z_cam))


def render_frame(
    cam_state: np.ndarray,
    person_pos: np.ndarray,
    image_size: Tuple[int, int] = (256, 256),
    focal: float = 500.0,
    person_size: float = 25.0,
    bg_color: str = "#1a1a2e",
    person_color: str = "#4ECDC4",
    border_color: str = "#334455",
) -> np.ndarray:
    """
    Render a single frame: camera view with person as a rectangle (cube proxy).

    Returns:
        (H, W, 3) RGB array in 0-255.
    """
    w, h = image_size
    fig, ax = plt.subplots(1, 1, figsize=(w / 100, h / 100), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    proj = project_world_to_screen(cam_state, person_pos, focal=focal, image_size=image_size)
    if proj is not None:
        px, py, depth = proj
        # Scale person size by inverse depth (closer = larger)
        size = person_size * 2.0 / (depth + 0.5)
        size = max(8, min(80, size))
        rect = mpatches.Rectangle(
            (px - size / 2, py - size / 2),
            size,
            size * 1.5,
            facecolor=person_color,
            edgecolor=border_color,
            linewidth=2,
        )
        ax.add_patch(rect)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape((int(fig.canvas.get_renderer().height), int(fig.canvas.get_renderer().width), 3))
    plt.close(fig)
    # Resize to exact image_size if needed
    if buf.shape[0] != h or buf.shape[1] != w:
        from PIL import Image as PILImage
        img = PILImage.fromarray(buf)
        img = img.resize((w, h), PILImage.Resampling.LANCZOS)
        buf = np.array(img)
    return buf


def render_camera_view_animation(
    cam_trajectory: np.ndarray,
    person_trajectory: np.ndarray,
    save_path: str = "outputs/camera_view.gif",
    fps: int = 12,
    image_size: Tuple[int, int] = (256, 256),
    focal: float = 500.0,
    person_size: float = 25.0,
    title: str = "Camera view",
    cam_trajectory_is_toric: bool = False,
) -> None:
    """
    Render an animated GIF showing what the camera sees over time (person as cube).

    Args:
        cam_trajectory: (T, 6) camera states (world or Toric)
        person_trajectory: (T, 3) person positions in world
        save_path: output path (.gif)
        fps, image_size, focal, person_size: rendering options
        cam_trajectory_is_toric: if True, convert Toric (pA,pB,theta,phi) to world per frame
    """
    T = min(cam_trajectory.shape[0], person_trajectory.shape[0])
    cam_trajectory = cam_trajectory[:T].copy()
    person_trajectory = person_trajectory[:T]
    cam_trajectory = _ensure_world_trajectory(
        cam_trajectory, person_trajectory, cam_trajectory_is_toric
    )

    os.makedirs(os.path.dirname(save_path) or "outputs", exist_ok=True)

    frames = []
    for t in range(T):
        frame_rgb = render_frame(
            cam_trajectory[t],
            person_trajectory[t],
            image_size=image_size,
            focal=focal,
            person_size=person_size,
        )
        frames.append(Image.fromarray(frame_rgb))

    if not frames:
        return
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0,
    )
    print(f"Camera view animation saved to: {save_path}")


def render_camera_view_static(
    cam_trajectory: np.ndarray,
    person_trajectory: np.ndarray,
    save_path: str = "outputs/camera_view_static.png",
    frame_indices: Optional[list] = None,
    image_size: Tuple[int, int] = (256, 256),
    cam_trajectory_is_toric: bool = False,
) -> None:
    """
    Render a grid of static frames (what the camera sees at selected frames).

    Args:
        frame_indices: which frames to show (default: 6 evenly spaced)
        cam_trajectory_is_toric: if True, convert Toric to world using person_trajectory
    """
    T = min(cam_trajectory.shape[0], person_trajectory.shape[0])
    cam_trajectory = _ensure_world_trajectory(
        cam_trajectory[:T].copy(), person_trajectory[:T], cam_trajectory_is_toric
    )
    if frame_indices is None:
        frame_indices = np.linspace(0, T - 1, 6, dtype=int).tolist()
    n = len(frame_indices)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    fig.patch.set_facecolor("#1a1a2e")

    for i, idx in enumerate(frame_indices):
        if idx >= T:
            continue
        ax = axes.flat[i]
        frame_rgb = render_frame(
            cam_trajectory[idx],
            person_trajectory[idx],
            image_size=image_size,
        )
        ax.imshow(frame_rgb)
        ax.set_title(f"Frame {idx + 1}", color="white", fontsize=10)
        ax.axis("off")
    for j in range(i + 1, len(axes.flat)):
        axes.flat[j].axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Camera view static grid saved to: {save_path}")
