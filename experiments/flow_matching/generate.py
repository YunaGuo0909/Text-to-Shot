"""
Inference script for Flow Matching experiment.

Usage:
    PYTHONPATH=. python experiments/flow_matching/generate.py --checkpoint /transfer/fm-checkpoints/fm_final.pth --text "A person walks toward camera"
    PYTHONPATH=. python experiments/flow_matching/generate.py --checkpoint /transfer/fm-checkpoints/fm_final.pth --text "Wide shot, person stands still" --motion dolly-in --guidance-scale 5.0
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

from src.models.denoiser import JointTrajectoryDenoiser
from experiments.flow_matching.models.flow_model import ConditionalFlowMatching


SHOT_TYPE_MAP = {
    "close-up": 0, "medium-shot": 1, "wide-shot": 2,
    "over-the-shoulder": 3, "two-shot": 4,
}
MOTION_TYPE_MAP = {
    "static": 0, "dolly-in": 1, "dolly-out": 2,
    "pan-left": 3, "pan-right": 4, "crane-up": 5,
    "crane-down": 6, "track": 7, "orbit": 8,
}


def load_model(checkpoint_path, device):
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

    flow = ConditionalFlowMatching(denoiser=denoiser).to(device)
    flow.load_state_dict(ckpt['model_state_dict'])
    flow.eval()
    return flow, config


def smooth_trajectory(traj, window=7, polyorder=2, angle_dims=None, angle_window=21):
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


def freeze_static_dims(traj, threshold=0.05):
    """
    Per-dimension: if the total range of a dimension is below threshold,
    treat it as static and replace with its mean value (completely freeze it).
    Eliminates residual oscillation on near-static trajectories.
    """
    result = traj.copy()
    for d in range(traj.shape[1]):
        if traj[:, d].max() - traj[:, d].min() < threshold:
            result[:, d] = traj[:, d].mean()
    return result


def regularize_person_trajectory(traj, static_threshold=0.15, segment_cost=0.3):
    """
    Make person trajectory physically plausible:
    - If overall displacement is small, freeze to stationary.
    - Otherwise, detect segments of roughly linear motion and straighten each.
    - Connect segments with smooth cubic interpolation for turns.
    - If sin/cos yaw (dims 3-4) are present, smooth them and renormalize.

    Args:
        traj: (T, 3) or (T, 5) person root positions [+ sin_yaw, cos_yaw]
        static_threshold: if total displacement < this, person is stationary
        segment_cost: penalty for adding a new segment (higher = fewer segments)

    Returns:
        (T, D) regularized trajectory (same shape as input)
    """
    T = traj.shape[0]
    D = traj.shape[1]
    pos_dims = min(D, 3)  # position dims are 0-2

    # Separate sin/cos yaw if present (dim >= 5)
    has_yaw = D >= 5
    sin_yaw = traj[:, 3].copy() if has_yaw else None
    cos_yaw = traj[:, 4].copy() if has_yaw else None
    pos = traj[:, :pos_dims]

    # 1. If person barely moves, freeze completely
    total_disp = np.linalg.norm(pos[-1] - pos[0])
    max_range = np.max(np.ptp(pos, axis=0))
    if total_disp < static_threshold and max_range < static_threshold:
        result = np.tile(pos.mean(axis=0), (T, 1))
        if has_yaw:
            # Smooth sin/cos yaw and renormalize for static person
            s_smooth, c_smooth = _smooth_sincos(sin_yaw, cos_yaw, window=31)
            result = np.concatenate([result, s_smooth.reshape(-1, 1), c_smooth.reshape(-1, 1)], axis=1)
        return result

    # 2. Find optimal breakpoints using simple greedy segmentation
    #    Each segment is approximated by a straight line (start -> end).
    #    Add a breakpoint when the max deviation from the line exceeds a threshold.
    breakpoints = [0]
    i = 0
    while i < T - 1:
        best_end = T - 1
        for j in range(i + 1, T):
            if j == i:
                continue
            frac = np.linspace(0, 1, j - i + 1).reshape(-1, 1)
            line = pos[i] + frac * (pos[j] - pos[i])
            actual = pos[i:j+1]
            max_dev = np.max(np.linalg.norm(actual - line, axis=1))
            if max_dev > segment_cost:
                best_end = max(j - 1, i + 1)
                break
        breakpoints.append(best_end)
        i = best_end
    if breakpoints[-1] != T - 1:
        breakpoints.append(T - 1)
    breakpoints = sorted(set(breakpoints))

    # 3. Build keyframe positions at breakpoints
    keyframe_times = np.array(breakpoints)
    keyframe_positions = pos[keyframe_times].copy()

    # 4. Per-dimension: freeze dimensions that don't change much within each segment
    for seg_idx in range(len(breakpoints) - 1):
        s, e = breakpoints[seg_idx], breakpoints[seg_idx + 1]
        seg = pos[s:e+1]
        for d in range(pos_dims):
            if np.ptp(seg[:, d]) < static_threshold:
                keyframe_positions[seg_idx, d] = seg[:, d].mean()
                keyframe_positions[seg_idx + 1, d] = seg[:, d].mean()

    # 5. Interpolate between keyframes with cubic spline for smooth transitions
    from scipy.interpolate import CubicSpline
    result_pos = np.zeros((T, pos_dims), dtype=traj.dtype)
    t_all = np.arange(T, dtype=np.float64)

    if len(keyframe_times) >= 3:
        for d in range(pos_dims):
            cs = CubicSpline(keyframe_times.astype(np.float64),
                             keyframe_positions[:, d],
                             bc_type='clamped')
            result_pos[:, d] = cs(t_all)
    elif len(keyframe_times) == 2:
        for d in range(pos_dims):
            result_pos[:, d] = np.linspace(keyframe_positions[0, d],
                                            keyframe_positions[1, d], T)
    else:
        result_pos = pos.copy()

    # 6. Handle sin/cos yaw smoothing
    if has_yaw:
        s_smooth, c_smooth = _smooth_sincos(sin_yaw, cos_yaw, window=15)
        result = np.concatenate([result_pos, s_smooth.reshape(-1, 1), c_smooth.reshape(-1, 1)], axis=1)
    else:
        result = result_pos

    return result


def _smooth_sincos(sin_vals, cos_vals, window=15):
    """
    Smooth sin/cos yaw components independently, then renormalize to unit circle.

    Since sin and cos are already continuous (no wraparound), standard
    Savitzky-Golay smoothing works directly.
    """
    w = min(window, len(sin_vals) if len(sin_vals) % 2 == 1 else len(sin_vals) - 1)
    w = max(w, 3)
    if w % 2 == 0:
        w -= 1
    sin_smooth = savgol_filter(sin_vals, window_length=w, polyorder=2)
    cos_smooth = savgol_filter(cos_vals, window_length=w, polyorder=2)
    # Renormalize to unit circle
    norm = np.sqrt(sin_smooth**2 + cos_smooth**2) + 1e-8
    sin_smooth = sin_smooth / norm
    cos_smooth = cos_smooth / norm
    return sin_smooth.astype(np.float32), cos_smooth.astype(np.float32)


@torch.no_grad()
def generate(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    flow, config = load_model(args.checkpoint, device)

    model_cfg = config['model']
    person_dim = model_cfg['person_dim']
    camera_dim = model_cfg['camera_dim']
    num_frames = config['trajectory']['default_num_frames']
    person_total = person_dim * num_frames
    num_steps = args.steps if args.steps is not None else config['flow_matching']['num_steps']

    text_encoder = None
    try:
        from src.models.text_encoder import CLIPTextEncoder
        text_encoder = CLIPTextEncoder(
            model_name=config['text_encoder']['model_name'], device=device
        ).to(device)
    except Exception:
        print("CLIP unavailable, using random embeddings.")

    norm_mean = norm_std = None
    norm_stats_path = config['data'].get('norm_stats_path', None)
    if norm_stats_path and os.path.exists(norm_stats_path):
        with open(norm_stats_path, 'r') as f:
            stats = json.load(f)
        norm_mean = torch.tensor(stats['mean'], dtype=torch.float32, device=device)
        norm_std = torch.tensor(stats['std'], dtype=torch.float32, device=device)
        print(f"  Using norm stats from {norm_stats_path}")

    prompts = args.text if isinstance(args.text, list) else [args.text]

    output_dir = args.output_dir if args.output_dir else config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    for prompt in prompts:
        if text_encoder:
            text_embed = text_encoder([prompt])
        else:
            text_embed = torch.randn(1, 512, device=device)

        shot_idx = SHOT_TYPE_MAP.get(args.shot_type, 1)
        motion_idx = MOTION_TYPE_MAP.get(args.motion, 0)
        shot_type = torch.tensor([shot_idx], device=device)
        motion_type = torch.tensor([motion_idx], device=device)

        print(f"Generating: \"{prompt}\"")
        print(f"  shot={args.shot_type}, motion={args.motion}, "
              f"guidance_scale={args.guidance_scale}, steps={num_steps}")

        y = flow.sample(text_embed, shot_type=shot_type, motion_type=motion_type,
                        device=device, guidance_scale=args.guidance_scale,
                        num_steps=num_steps)

        if norm_mean is not None:
            y = y * norm_std + norm_mean

        y_np = y[0].cpu().numpy()
        person_traj = y_np[:person_total].reshape(num_frames, person_dim)
        camera_traj = y_np[person_total:].reshape(num_frames, camera_dim)

        if not args.no_smooth:
            # Person trajectory: smooth → regularize to piecewise-linear → freeze static
            person_smooth_window = min(31, person_traj.shape[0] if person_traj.shape[0] % 2 == 1 else person_traj.shape[0] - 1)
            person_traj = smooth_trajectory(person_traj, window=person_smooth_window)
            person_traj = regularize_person_trajectory(person_traj,
                                                        static_threshold=0.08,
                                                        segment_cost=0.3)
            person_traj = freeze_static_dims(person_traj, threshold=0.05)

            # Camera: position dims get window=21, angle dims get window=31
            # then freeze near-static dims (e.g. static shot camera won't drift)
            camera_traj = smooth_trajectory(camera_traj, window=21,
                                            angle_dims=[3, 4, 5], angle_window=31)
            camera_traj = freeze_static_dims(camera_traj, threshold=0.05)
            print(f"  Smoothing + regularize + freeze applied")

        tag = f"{args.motion}_{args.shot_type}_{time.strftime('%m%d_%H%M%S')}"
        np.save(os.path.join(output_dir, f'fm_person_{tag}.npy'), person_traj)
        np.save(os.path.join(output_dir, f'fm_camera_{tag}.npy'), camera_traj)

        import importlib, sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        _gen = importlib.import_module('generate')
        visualize_joint = _gen.visualize_joint
        visualize_joint(person_traj, camera_traj, prompt, args.motion,
                        save_path=os.path.join(output_dir, f'fm_joint_{tag}.png'))

        print(f"  Outputs saved to {output_dir}/  [tag: {tag}]")


def main():
    parser = argparse.ArgumentParser(description='Generate with Flow Matching')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--text', type=str, nargs='+', required=True)
    parser.add_argument('--shot-type', type=str, default='medium-shot',
                        choices=list(SHOT_TYPE_MAP.keys()))
    parser.add_argument('--motion', type=str, default='static',
                        choices=list(MOTION_TYPE_MAP.keys()))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--guidance-scale', type=float, default=3.0)
    parser.add_argument('--steps', type=int, default=None,
                        help='Euler ODE steps (overrides config, e.g. 200 for smoother paths)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory (default: from config)')
    parser.add_argument('--no-smooth', action='store_true')
    parser.add_argument('--smooth-window', type=int, default=7)
    args = parser.parse_args()
    generate(args)


if __name__ == '__main__':
    main()
