"""
Evaluate Flow Matching trajectory generation.

Metrics: FTD, motion type accuracy, smoothness (jerk),
distance monotonicity, diversity, camera-person distance.

Usage:
    PYTHONPATH=. python experiments/flow_matching/evaluate.py \
        --checkpoint /transfer/fm-v9-checkpoints/fm_final.pth --device cuda
"""

import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models.denoiser import JointTrajectoryDenoiser
from src.data.dataset import JointTrajectoryDataset, collate_fn
from experiments.flow_matching.models.flow_model import ConditionalFlowMatching


MOTION_TYPE_MAP = {
    "static": 0, "dolly-in": 1, "dolly-out": 2,
    "pan-left": 3, "pan-right": 4, "crane-up": 5,
    "crane-down": 6, "track": 7, "orbit": 8,
}
MOTION_TYPE_NAMES = {v: k for k, v in MOTION_TYPE_MAP.items()}


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


# ======================================================================
# Metric 1: FTD (Frechet Trajectory Distance)
# ======================================================================

def compute_trajectory_statistics(trajectories):
    """Compute mean and covariance of flattened trajectories."""
    flat = np.array([t.flatten() for t in trajectories])
    mu = flat.mean(axis=0)
    sigma = np.cov(flat, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2):
    """Compute Frechet distance between two multivariate Gaussians."""
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.dot(diff, diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


# ======================================================================
# Metric 2: Motion Type Accuracy (rule-based classifier)
# ======================================================================

def classify_motion_from_trajectory(cam_traj, person_traj):
    """
    Classify the motion type of a generated trajectory using rules.
    Returns the predicted motion type string.
    """
    T = cam_traj.shape[0]
    cam_pos = cam_traj[:, :3]
    person_pos = person_traj[:, :3]

    distances = np.linalg.norm(cam_pos - person_pos, axis=1)
    dist_start = distances[:T // 4].mean()
    dist_end = distances[-T // 4:].mean()
    dist_change = dist_end - dist_start

    azimuth = cam_traj[:, 3]
    az_change = azimuth[-1] - azimuth[0]
    az_change = (az_change + np.pi) % (2 * np.pi) - np.pi

    cam_y_change = cam_pos[-1, 1] - cam_pos[0, 1]
    cam_pos_var = np.var(cam_pos, axis=0).sum()
    cam_xz_disp = np.linalg.norm(cam_pos[-1, [0, 2]] - cam_pos[0, [0, 2]])
    person_xz_disp = np.linalg.norm(person_pos[-1, [0, 2]] - person_pos[0, [0, 2]])
    dist_std = np.std(distances)

    if cam_pos_var < 0.05:
        return 'static'
    if abs(az_change) > np.radians(30) and dist_std < 0.5:
        return 'orbit'
    if abs(dist_change) > 0.5:
        return 'dolly-in' if dist_change < 0 else 'dolly-out'
    if abs(cam_y_change) > 0.3:
        return 'crane-up' if cam_y_change > 0 else 'crane-down'
    if cam_xz_disp > 0.3 and person_xz_disp > 0.3 and dist_std < 0.5:
        return 'track'
    if abs(az_change) > np.radians(10) and cam_xz_disp < 0.3:
        return 'pan-left' if az_change < 0 else 'pan-right'
    if abs(dist_change) > 0.2:
        return 'dolly-in' if dist_change < 0 else 'dolly-out'
    return 'static'


# ======================================================================
# Metric 3: Smoothness
# ======================================================================

def compute_jerk(traj_3d):
    if traj_3d.shape[0] < 4:
        return 0.0
    jerk = np.diff(np.diff(np.diff(traj_3d, axis=0), axis=0), axis=0)
    return float(np.mean(np.linalg.norm(jerk, axis=1)))


# ======================================================================
# Metric 4: Distance Monotonicity
# ======================================================================

def distance_monotonicity(distances, direction='decrease'):
    """Fraction of consecutive frame pairs where distance changes correctly."""
    diffs = np.diff(distances)
    if direction == 'decrease':
        correct = (diffs < 0).sum()
    else:
        correct = (diffs > 0).sum()
    return float(correct / max(len(diffs), 1))


# ======================================================================
# Metric 5: Diversity
# ======================================================================

def compute_diversity(trajectories):
    """Mean pairwise L2 distance between generated trajectories."""
    if len(trajectories) < 2:
        return 0.0
    flat = np.array([t.flatten() for t in trajectories])
    n = len(flat)
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += np.linalg.norm(flat[i] - flat[j])
            count += 1
    return float(total_dist / max(count, 1))


# ======================================================================
# Main evaluation
# ======================================================================

@torch.no_grad()
def evaluate(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    flow, config = load_model(args.checkpoint, device)

    model_cfg = config['model']
    person_dim = model_cfg['person_dim']
    camera_dim = model_cfg['camera_dim']
    num_frames = config['trajectory']['default_num_frames']
    person_total = person_dim * num_frames
    num_steps = config['flow_matching']['num_steps']

    # Text encoder
    text_encoder = None
    try:
        from src.models.text_encoder import CLIPTextEncoder
        text_encoder = CLIPTextEncoder(
            model_name=config['text_encoder']['model_name'], device=device
        ).to(device)
    except Exception as e:
        print(f"CLIP unavailable ({e}), using random embeddings.")

    # Norm stats for denormalization
    norm_mean = norm_std = None
    norm_stats_path = config['data'].get('norm_stats_path', None)
    if norm_stats_path and os.path.exists(norm_stats_path):
        with open(norm_stats_path, 'r') as f:
            stats = json.load(f)
        norm_mean = torch.tensor(stats['mean'], dtype=torch.float32, device=device)
        norm_std = torch.tensor(stats['std'], dtype=torch.float32, device=device)

    # Test dataset
    test_dataset = JointTrajectoryDataset(
        data_root=config['data']['data_root'],
        split='test',
        num_frames=num_frames,
        person_dim=person_dim,
        camera_dim=camera_dim,
        index_file=config['data'].get('test_index_file', 'test_index.json'),
        norm_stats_path=norm_stats_path,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, collate_fn=collate_fn,
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {config['data']['data_root']}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Guidance scale: {args.guidance_scale}")
    print()

    # ---- Collect GT and generated trajectories ----
    gt_joints = []
    gen_joints = []
    per_sample_metrics = []
    motion_type_results = {}  # motion_type -> list of (predicted, correct)

    for batch in tqdm(test_loader, desc="Generating on test set"):
        y_gt = batch['y'].to(device)
        B = y_gt.shape[0]

        if text_encoder is not None:
            text_embed = text_encoder(batch['texts'])
        else:
            text_embed = torch.randn(B, 512, device=device)

        shot_types = batch['shot_types'].to(device)
        shot_type = shot_types if (shot_types >= 0).all() else None
        motion_types = batch['motion_types'].to(device)
        motion_type = motion_types if (motion_types >= 0).all() else None

        y_gen = flow.sample(text_embed, shot_type=shot_type, motion_type=motion_type,
                            device=device, guidance_scale=args.guidance_scale,
                            num_steps=num_steps)

        # Denormalize both GT and generated
        if norm_mean is not None:
            y_gt_denorm = y_gt * norm_std + norm_mean
            y_gen_denorm = y_gen * norm_std + norm_mean
        else:
            y_gt_denorm = y_gt
            y_gen_denorm = y_gen

        for i in range(B):
            gt = y_gt_denorm[i].cpu().numpy()
            gen = y_gen_denorm[i].cpu().numpy()

            gt_person = gt[:person_total].reshape(num_frames, person_dim)
            gt_camera = gt[person_total:].reshape(num_frames, camera_dim)
            gen_person = gen[:person_total].reshape(num_frames, person_dim)
            gen_camera = gen[person_total:].reshape(num_frames, camera_dim)

            gt_joints.append(gt)
            gen_joints.append(gen)

            # Per-sample metrics
            mt_idx = motion_types[i].item() if motion_type is not None else -1
            mt_name = MOTION_TYPE_NAMES.get(mt_idx, 'unknown')

            # Smoothness
            person_jerk = compute_jerk(gen_person[:, :3])
            camera_jerk = compute_jerk(gen_camera[:, :3])
            gt_camera_jerk = compute_jerk(gt_camera[:, :3])

            # Camera-person distance
            distances = np.linalg.norm(gen_camera[:, :3] - gen_person[:, :3], axis=1)
            mean_dist = float(distances.mean())

            # Motion type accuracy
            predicted_mt = classify_motion_from_trajectory(gen_camera, gen_person)
            is_correct = predicted_mt == mt_name

            if mt_name not in motion_type_results:
                motion_type_results[mt_name] = []
            motion_type_results[mt_name].append(is_correct)

            # Distance monotonicity for dolly types
            mono_score = None
            if mt_name == 'dolly-in':
                mono_score = distance_monotonicity(distances, 'decrease')
            elif mt_name == 'dolly-out':
                mono_score = distance_monotonicity(distances, 'increase')

            per_sample_metrics.append({
                'motion_type': mt_name,
                'person_jerk': person_jerk,
                'camera_jerk': camera_jerk,
                'gt_camera_jerk': gt_camera_jerk,
                'mean_cam_person_dist': mean_dist,
                'motion_correct': is_correct,
                'mono_score': mono_score,
            })

        if len(gt_joints) >= args.max_samples:
            break

    # ---- Compute aggregate metrics ----
    print(f"\n{'='*65}")
    print(f"  EVALUATION RESULTS ({len(gt_joints)} samples)")
    print(f"{'='*65}")

    # 1. FTD
    print("\n[1] Frechet Trajectory Distance (FTD)")
    # Joint FTD
    mu_gt, sigma_gt = compute_trajectory_statistics(gt_joints)
    mu_gen, sigma_gen = compute_trajectory_statistics(gen_joints)
    ftd_joint = frechet_distance(mu_gt, sigma_gt, mu_gen, sigma_gen)
    print(f"  Joint FTD:  {ftd_joint:.2f}")

    # Person-only FTD
    gt_persons = [g[:person_total] for g in gt_joints]
    gen_persons = [g[:person_total] for g in gen_joints]
    mu_gtp, sigma_gtp = compute_trajectory_statistics(gt_persons)
    mu_genp, sigma_genp = compute_trajectory_statistics(gen_persons)
    ftd_person = frechet_distance(mu_gtp, sigma_gtp, mu_genp, sigma_genp)
    print(f"  Person FTD: {ftd_person:.2f}")

    # Camera-only FTD
    gt_cameras = [g[person_total:] for g in gt_joints]
    gen_cameras = [g[person_total:] for g in gen_joints]
    mu_gtc, sigma_gtc = compute_trajectory_statistics(gt_cameras)
    mu_genc, sigma_genc = compute_trajectory_statistics(gen_cameras)
    ftd_camera = frechet_distance(mu_gtc, sigma_gtc, mu_genc, sigma_genc)
    print(f"  Camera FTD: {ftd_camera:.2f}")

    # 2. Motion Type Accuracy
    print("\n[2] Motion Type Accuracy")
    total_correct = 0
    total_count = 0
    for mt in sorted(motion_type_results.keys()):
        results = motion_type_results[mt]
        correct = sum(results)
        count = len(results)
        acc = 100 * correct / max(count, 1)
        total_correct += correct
        total_count += count
        marker = " !!!" if acc < 50 else ""
        print(f"  {mt:15s}: {acc:5.1f}% ({correct}/{count}){marker}")
    overall_acc = 100 * total_correct / max(total_count, 1)
    print(f"  {'OVERALL':15s}: {overall_acc:5.1f}%")

    # 3. Smoothness
    print("\n[3] Smoothness (Jerk, lower = smoother)")
    person_jerks = [m['person_jerk'] for m in per_sample_metrics]
    camera_jerks = [m['camera_jerk'] for m in per_sample_metrics]
    gt_cam_jerks = [m['gt_camera_jerk'] for m in per_sample_metrics]
    print(f"  Gen person jerk:  {np.mean(person_jerks):.6f} +/- {np.std(person_jerks):.6f}")
    print(f"  Gen camera jerk:  {np.mean(camera_jerks):.6f} +/- {np.std(camera_jerks):.6f}")
    print(f"  GT  camera jerk:  {np.mean(gt_cam_jerks):.6f} +/- {np.std(gt_cam_jerks):.6f}")

    # 4. Distance Monotonicity
    print("\n[4] Distance Monotonicity (dolly-in/out)")
    for mt in ['dolly-in', 'dolly-out']:
        scores = [m['mono_score'] for m in per_sample_metrics
                  if m['motion_type'] == mt and m['mono_score'] is not None]
        if scores:
            print(f"  {mt:15s}: {100*np.mean(scores):.1f}% frames correct "
                  f"(n={len(scores)})")
        else:
            print(f"  {mt:15s}: no samples")

    # 5. Camera-Person Distance
    print("\n[5] Camera-Person Distance")
    dists = [m['mean_cam_person_dist'] for m in per_sample_metrics]
    print(f"  Mean: {np.mean(dists):.2f}m +/- {np.std(dists):.2f}m")
    print(f"  Range: [{np.min(dists):.2f}, {np.max(dists):.2f}]m")

    # 6. Diversity (generate multiple times for same prompts)
    print("\n[6] Diversity (same prompt, multiple generations)")
    diversity_prompts = [
        ("The camera pushes in while the character walks forward", "dolly-in"),
        ("The camera orbits around the character", "orbit"),
        ("A static shot as the character stands still", "static"),
    ]
    for prompt, mt in diversity_prompts:
        if text_encoder is not None:
            text_embed = text_encoder([prompt])
        else:
            text_embed = torch.randn(1, 512, device=device)
        mt_idx = MOTION_TYPE_MAP[mt]
        shot_t = torch.tensor([1], device=device)  # medium-shot
        motion_t = torch.tensor([mt_idx], device=device)

        gen_trajs = []
        for _ in range(args.num_diversity):
            y = flow.sample(text_embed, shot_type=shot_t, motion_type=motion_t,
                            device=device, guidance_scale=args.guidance_scale,
                            num_steps=num_steps)
            if norm_mean is not None:
                y = y * norm_std + norm_mean
            gen_trajs.append(y[0].cpu().numpy())

        div = compute_diversity(gen_trajs)
        print(f"  {mt:15s}: diversity = {div:.4f}")

    # ---- Summary ----
    print(f"\n{'='*65}")
    summary = {
        'ftd_joint': ftd_joint,
        'ftd_person': ftd_person,
        'ftd_camera': ftd_camera,
        'motion_type_accuracy': overall_acc,
        'person_jerk_mean': float(np.mean(person_jerks)),
        'camera_jerk_mean': float(np.mean(camera_jerks)),
        'gt_camera_jerk_mean': float(np.mean(gt_cam_jerks)),
        'cam_person_dist_mean': float(np.mean(dists)),
        'num_samples': len(gt_joints),
        'guidance_scale': args.guidance_scale,
        'checkpoint': args.checkpoint,
        'per_motion_accuracy': {
            mt: 100 * sum(r) / max(len(r), 1)
            for mt, r in motion_type_results.items()
        },
    }

    # Add monotonicity
    for mt in ['dolly-in', 'dolly-out']:
        scores = [m['mono_score'] for m in per_sample_metrics
                  if m['motion_type'] == mt and m['mono_score'] is not None]
        if scores:
            summary[f'{mt}_monotonicity'] = float(100 * np.mean(scores))

    # Save
    output_dir = config['paths'].get('output_dir', '/transfer/fm-v9-outputs')
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {results_path}")
    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Flow Matching Model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--guidance-scale', type=float, default=3.0)
    parser.add_argument('--max-samples', type=int, default=2000,
                        help='Max test samples to evaluate')
    parser.add_argument('--num-diversity', type=int, default=10,
                        help='Number of generations per prompt for diversity')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
