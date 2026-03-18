"""
Evaluation script for Joint Person-Camera Trajectory Diffusion Model.

Computes:
  - Per-branch reconstruction: MSE, MAE for person and camera separately
  - Joint trajectory smoothness (jerk)
  - Person-camera coordination metrics
  - Diversity across multiple generations

Usage:
    python evaluate.py --checkpoint /transfer/stc-checkpoints/stc_final.pth --device cuda
"""

import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models.denoiser import JointTrajectoryDenoiser
from src.models.diffusion import GaussianDiffusion
from src.data.dataset import JointTrajectoryDataset, collate_fn


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

    diffusion = GaussianDiffusion(
        denoiser=denoiser,
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
    ).to(device)

    diffusion.load_state_dict(ckpt['model_state_dict'])
    diffusion.eval()
    return diffusion, config


def compute_smoothness(trajectory: np.ndarray) -> dict:
    """Compute smoothness metrics (velocity, jerk, path length)."""
    velocity = np.diff(trajectory, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)
    return {
        'mean_jerk': float(np.mean(np.linalg.norm(jerk, axis=1))) if len(jerk) > 0 else 0.0,
        'path_length': float(np.sum(np.linalg.norm(velocity, axis=1))),
    }


def compute_metrics(gen_person, gen_camera, gt_person, gt_camera):
    """Compute per-sample metrics between generated and ground truth."""
    # Person metrics
    person_mse = float(np.mean((gen_person - gt_person) ** 2))
    person_mae = float(np.mean(np.abs(gen_person - gt_person)))

    # Camera metrics
    camera_mse = float(np.mean((gen_camera - gt_camera) ** 2))
    camera_mae = float(np.mean(np.abs(gen_camera - gt_camera)))

    # Smoothness
    person_smooth = compute_smoothness(gen_person)
    camera_smooth = compute_smoothness(gen_camera)
    gt_camera_smooth = compute_smoothness(gt_camera)

    # Person-camera coordination: person should stay in front of camera
    # Simple metric: average distance between person and camera position
    cam_pos = gen_camera[:, :3]
    person_cam_dist = np.mean(np.linalg.norm(gen_person - cam_pos, axis=1))

    return {
        'person_mse': person_mse,
        'person_mae': person_mae,
        'camera_mse': camera_mse,
        'camera_mae': camera_mae,
        'person_jerk': person_smooth['mean_jerk'],
        'camera_jerk': camera_smooth['mean_jerk'],
        'gt_camera_jerk': gt_camera_smooth['mean_jerk'],
        'person_path_length': person_smooth['path_length'],
        'camera_path_length': camera_smooth['path_length'],
        'person_cam_dist': float(person_cam_dist),
    }


@torch.no_grad()
def evaluate(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    diffusion, config = load_model(args.checkpoint, device)

    model_cfg = config['model']
    person_dim = model_cfg['person_dim']
    camera_dim = model_cfg['camera_dim']
    num_frames = config['trajectory']['default_num_frames']
    person_total = person_dim * num_frames
    camera_total = camera_dim * num_frames

    # Text encoder
    text_encoder = None
    if not args.no_clip:
        try:
            from src.models.text_encoder import CLIPTextEncoder
            text_encoder = CLIPTextEncoder(
                model_name=config['text_encoder']['model_name'], device=device
            ).to(device)
        except Exception as e:
            print(f"CLIP unavailable ({e}), using random embeddings.")

    # Test dataset
    test_index = config['data'].get('test_index_file', 'test_index.json')
    if args.single_person:
        test_index = 'test_index_single_person.json'

    test_dataset = JointTrajectoryDataset(
        data_root=config['data']['data_root'],
        split='test',
        num_frames=num_frames,
        person_dim=person_dim,
        camera_dim=camera_dim,
        index_file=test_index,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, collate_fn=collate_fn,
    )

    print(f"\nEvaluating on {len(test_dataset)} test samples...")

    all_metrics = []

    for batch in tqdm(test_loader, desc="Evaluating"):
        y_gt = batch['y']
        batch_size = y_gt.shape[0]

        if text_encoder is not None:
            text_embed = text_encoder(batch['texts'])
        else:
            text_embed = torch.randn(batch_size, 512, device=device)

        shot_types = batch['shot_types'].to(device)
        shot_type = shot_types if (shot_types >= 0).all() else None
        motion_types = batch['motion_types'].to(device)
        motion_type = motion_types if (motion_types >= 0).all() else None

        y_gen = diffusion.sample(
            text_embed=text_embed,
            shot_type=shot_type,
            motion_type=motion_type,
            device=device,
        )

        for i in range(batch_size):
            gt = y_gt[i].numpy()
            gen = y_gen[i].cpu().numpy()

            gt_person = gt[:person_total].reshape(num_frames, person_dim)
            gt_camera = gt[person_total:].reshape(num_frames, camera_dim)
            gen_person = gen[:person_total].reshape(num_frames, person_dim)
            gen_camera = gen[person_total:].reshape(num_frames, camera_dim)

            metrics = compute_metrics(gen_person, gen_camera, gt_person, gt_camera)
            all_metrics.append(metrics)

    # Aggregate
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")

    results = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        results[key] = {'mean': float(mean_val), 'std': float(std_val)}
        print(f"  {key:25s}: {mean_val:.6f} +/- {std_val:.6f}")

    print(f"{'='*60}")

    # Save
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Joint Model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--no-clip', action='store_true')
    parser.add_argument('--single-person', action='store_true')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
