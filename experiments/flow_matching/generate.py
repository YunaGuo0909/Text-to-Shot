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

    output_dir = config['paths']['output_dir']
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
            # Person trajectory: strong smoothing + freeze near-static dims
            person_smooth_window = min(31, person_traj.shape[0] if person_traj.shape[0] % 2 == 1 else person_traj.shape[0] - 1)
            person_traj = smooth_trajectory(person_traj, window=person_smooth_window)
            person_traj = freeze_static_dims(person_traj, threshold=0.05)

            # Camera: position dims get window=21, angle dims get window=31
            # then freeze near-static dims (e.g. static shot camera won't drift)
            camera_traj = smooth_trajectory(camera_traj, window=21,
                                            angle_dims=[3, 4, 5], angle_window=31)
            camera_traj = freeze_static_dims(camera_traj, threshold=0.05)
            print(f"  Smoothing + freeze applied (cam_pos=21, angle=31, person={person_smooth_window})")

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
    parser.add_argument('--no-smooth', action='store_true')
    parser.add_argument('--smooth-window', type=int, default=7)
    args = parser.parse_args()
    generate(args)


if __name__ == '__main__':
    main()
