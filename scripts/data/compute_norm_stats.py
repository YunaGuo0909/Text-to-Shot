"""
Compute per-dimension normalization statistics from training data.

Run ONCE before training:
    python scripts/data/compute_norm_stats.py
    python scripts/data/compute_norm_stats.py --index-file train_index_single_person.json
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Compute normalization stats for joint trajectories')
    parser.add_argument('--data-root', type=str, default='/transfer/stc-data')
    parser.add_argument('--index-file', type=str, default='train_index.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: <data-root>/norm_stats.json)')
    parser.add_argument('--num-frames', type=int, default=48)
    parser.add_argument('--person-dim', type=int, default=5)
    parser.add_argument('--camera-dim', type=int, default=6)
    args = parser.parse_args()

    index_path = os.path.join(args.data_root, args.index_file)
    if not os.path.exists(index_path):
        # fallback to full train index
        index_path = os.path.join(args.data_root, 'train_index.json')
    with open(index_path, 'r') as f:
        samples = json.load(f)

    print(f"Computing stats over {len(samples)} training samples from {index_path}")

    all_y = []
    skipped = 0
    for sample in tqdm(samples, desc='Loading'):
        cam_path = os.path.join(args.data_root, sample['camera_trajectory_path'])
        person_path = os.path.join(args.data_root, sample['person_trajectory_path'])
        if not os.path.exists(cam_path) or not os.path.exists(person_path):
            skipped += 1
            continue
        cam = np.load(cam_path).astype(np.float32)
        person = np.load(person_path).astype(np.float32)
        # Resample if needed
        if cam.shape[0] != args.num_frames:
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, cam.shape[0])
            x_new = np.linspace(0, 1, args.num_frames)
            cam = np.stack([np.interp(x_new, x_old, cam[:, d]) for d in range(cam.shape[1])], axis=1)
        if person.shape[0] != args.num_frames:
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, person.shape[0])
            x_new = np.linspace(0, 1, args.num_frames)
            person = np.stack([np.interp(x_new, x_old, person[:, d]) for d in range(person.shape[1])], axis=1)
        # Joint vector: [person_flat, camera_flat]
        y = np.concatenate([person.flatten(), cam.flatten()])
        # Skip samples with NaN/Inf (SLAHMR tracking failures)
        if not np.isfinite(y).all():
            skipped += 1
            continue
        all_y.append(y)

    if skipped:
        print(f"Skipped {skipped} samples (missing files or NaN/Inf)")

    all_y = np.stack(all_y, axis=0)  # (N, total_dim)
    mean = all_y.mean(axis=0)        # (total_dim,)
    std = all_y.std(axis=0)          # (total_dim,)
    std = np.where(std < 1e-6, 1.0, std)  # avoid division by zero for constant dims

    output_path = args.output or os.path.join(args.data_root, 'norm_stats.json')
    stats = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'n_samples': int(len(all_y)),
        'total_dim': int(all_y.shape[1]),
        'num_frames': args.num_frames,
        'person_dim': args.person_dim,
        'camera_dim': args.camera_dim,
    }
    with open(output_path, 'w') as f:
        json.dump(stats, f)

    print(f"\nNorm stats saved to {output_path}")
    print(f"  Samples: {len(all_y)}")
    print(f"  Total dim: {all_y.shape[1]}")
    print(f"  Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  Std  range: [{std.min():.3f}, {std.max():.3f}]")
    print(f"\nNext: re-run training with norm_stats_path set in config.")


if __name__ == '__main__':
    main()
