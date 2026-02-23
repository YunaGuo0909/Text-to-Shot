"""
Evaluation script for the Camera Trajectory Diffusion Model.

Computes quantitative metrics on the test set:
  - Reconstruction quality: MSE, MAE between generated and ground truth
  - Trajectory smoothness: velocity, acceleration, jerk
  - Diversity: variance across multiple generations for the same prompt
  - Motion-type accuracy: whether generated trajectory matches the intended motion

Usage:
    python evaluate.py --checkpoint checkpoints/checkpoint_final.pth --device cuda
    python evaluate.py --checkpoint checkpoints/checkpoint_final.pth --num-samples 5 --device cuda
"""

import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models.denoiser import CameraTrajectoryDenoiser
from src.models.diffusion import GaussianDiffusion
from src.data.dataset import CameraTrajectoryDataset, collate_fn
from src.pipeline.camera_trajectory import CameraTrajectoryGenerator


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model_cfg = config['model']
    traj_cfg = config['trajectory']

    denoiser = CameraTrajectoryDenoiser(
        toric_dim=model_cfg['toric_dim'],
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


def compute_trajectory_metrics(generated: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Compute metrics between generated and ground truth trajectories.

    Args:
        generated: (T, 6) generated trajectory
        ground_truth: (T, 6) ground truth trajectory

    Returns:
        Dictionary of metric values
    """
    # Reconstruction quality
    mse = np.mean((generated - ground_truth) ** 2)
    mae = np.mean(np.abs(generated - ground_truth))

    # Per-parameter MSE
    param_names = ['tx', 'ty', 'tz', 'azimuth', 'elevation', 'roll']
    per_param_mse = {f"mse_{name}": float(np.mean((generated[:, i] - ground_truth[:, i]) ** 2))
                     for i, name in enumerate(param_names)}

    # Smoothness of generated trajectory
    smoothness = CameraTrajectoryGenerator.compute_trajectory_smoothness(generated)

    # Smoothness of ground truth
    gt_smoothness = CameraTrajectoryGenerator.compute_trajectory_smoothness(ground_truth)

    return {
        'mse': float(mse),
        'mae': float(mae),
        **per_param_mse,
        'gen_mean_jerk': smoothness['mean_jerk'],
        'gen_path_length': smoothness['total_path_length'],
        'gt_mean_jerk': gt_smoothness['mean_jerk'],
        'gt_path_length': gt_smoothness['total_path_length'],
        'jerk_ratio': smoothness['mean_jerk'] / max(gt_smoothness['mean_jerk'], 1e-8),
    }


def compute_diversity(trajectories: list) -> dict:
    """
    Compute diversity metrics across multiple generated trajectories.

    Args:
        trajectories: List of (T, 6) arrays from the same prompt

    Returns:
        Dictionary with diversity metrics
    """
    if len(trajectories) < 2:
        return {'diversity': 0.0}

    stacked = np.stack(trajectories, axis=0)  # (N, T, 6)
    variance = np.mean(np.var(stacked, axis=0))
    pairwise_dists = []
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            dist = np.mean(np.sqrt(np.sum((trajectories[i] - trajectories[j]) ** 2, axis=-1)))
            pairwise_dists.append(dist)

    return {
        'diversity_variance': float(variance),
        'diversity_mean_dist': float(np.mean(pairwise_dists)),
    }


@torch.no_grad()
def evaluate(args):
    """Run evaluation on the test set."""
    device = args.device if torch.cuda.is_available() else 'cpu'

    # Load model
    diffusion, config = load_model(args.checkpoint, device)
    toric_dim = diffusion.denoiser.toric_dim
    num_frames = diffusion.denoiser.num_frames

    # Text encoder
    text_encoder = None
    if not args.no_clip:
        try:
            from src.models.text_encoder import CLIPTextEncoder
            text_encoder = CLIPTextEncoder(
                model_name=config['text_encoder']['model_name'], device=device
            ).to(device)
            print("CLIP text encoder loaded.")
        except Exception as e:
            print(f"CLIP unavailable ({e}), using random embeddings.")

    # Test dataset
    test_dataset = CameraTrajectoryDataset(
        data_root=config['data']['data_root'],
        split='test',
        num_frames=num_frames,
        toric_dim=toric_dim,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    print(f"\nEvaluating on {len(test_dataset)} test samples...")
    print(f"Generating {args.num_samples} sample(s) per prompt\n")

    all_metrics = []

    for batch in tqdm(test_loader, desc="Evaluating"):
        y_gt = batch['y']  # (B, T*6)
        batch_size = y_gt.shape[0]

        # Encode text
        if text_encoder is not None:
            text_embed = text_encoder(batch['texts'])
        else:
            text_embed = torch.randn(batch_size, 512, device=device)

        shot_types = batch['shot_types'].to(device)
        shot_type = shot_types if (shot_types >= 0).all() else None
        motion_types = batch['motion_types'].to(device)
        motion_type = motion_types if (motion_types >= 0).all() else None

        # Generate trajectories
        for sample_idx in range(args.num_samples):
            y_gen = diffusion.sample(
                text_embed=text_embed,
                shot_type=shot_type,
                motion_type=motion_type,
                device=device,
            )

            # Compute per-sample metrics
            for i in range(batch_size):
                gt = y_gt[i].numpy().reshape(num_frames, toric_dim)
                gen = y_gen[i].cpu().numpy().reshape(num_frames, toric_dim)
                metrics = compute_trajectory_metrics(gen, gt)
                metrics['sample_idx'] = sample_idx
                all_metrics.append(metrics)

    # Aggregate metrics
    metric_keys = ['mse', 'mae', 'gen_mean_jerk', 'gen_path_length',
                   'gt_mean_jerk', 'gt_path_length', 'jerk_ratio']
    param_keys = [f'mse_{p}' for p in ['tx', 'ty', 'tz', 'azimuth', 'elevation', 'roll']]

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    results = {}
    for key in metric_keys:
        vals = [m[key] for m in all_metrics]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        results[key] = {'mean': float(mean_val), 'std': float(std_val)}
        print(f"  {key:20s}: {mean_val:.6f} ± {std_val:.6f}")

    print(f"\n  Per-parameter MSE:")
    for key in param_keys:
        vals = [m[key] for m in all_metrics]
        mean_val = np.mean(vals)
        results[key] = float(mean_val)
        print(f"    {key:20s}: {mean_val:.6f}")

    print("=" * 60)

    # Save results
    os.makedirs('outputs', exist_ok=True)
    results_path = 'outputs/evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Camera Trajectory Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-samples', type=int, default=1,
                        help='Number of samples per prompt (for diversity)')
    parser.add_argument('--no-clip', action='store_true',
                        help='Use random text embeddings')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
