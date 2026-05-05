"""
Diagnose v6 output quality issues.

Checks:
1. E.T. label accuracy: does the actual trajectory match the motion type label?
2. Person trajectory quality per source (smoothness, range, zero-ratio)
3. Data balance and source mixing
4. Normalization stats sanity
5. Person .npy file dimensions (3 vs 5)

Usage:
    python scripts/diagnose_v6_issues.py --data-root /transfer/merged-v8
    python scripts/diagnose_v6_issues.py --data-root /transfer/merged-v7
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict


def load_index(data_root, split='train'):
    path = os.path.join(data_root, f'{split}_index.json')
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return []
    with open(path, 'r') as f:
        return json.load(f)


def compute_smoothness(traj):
    """Mean per-frame acceleration magnitude (lower = smoother)."""
    if traj.shape[0] < 3:
        return 0.0
    vel = np.diff(traj, axis=0)
    acc = np.diff(vel, axis=0)
    return float(np.mean(np.linalg.norm(acc, axis=1)))


def compute_jerk(traj):
    """Mean per-frame jerk magnitude (lower = smoother)."""
    if traj.shape[0] < 4:
        return 0.0
    vel = np.diff(traj, axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)
    return float(np.mean(np.linalg.norm(jerk, axis=1)))


def verify_motion_label(cam_traj, person_traj, label):
    """
    Check if the actual trajectory matches the motion type label.
    Returns (is_correct, reason).
    """
    T = cam_traj.shape[0]
    cam_pos = cam_traj[:, :3]
    person_pos = person_traj[:, :3]

    # Camera-person distance over time
    distances = np.linalg.norm(cam_pos - person_pos, axis=1)
    dist_start = distances[:T // 4].mean()
    dist_end = distances[-T // 4:].mean()
    dist_change = dist_end - dist_start

    # Camera azimuth change
    azimuth = cam_traj[:, 3]
    az_change = azimuth[-1] - azimuth[0]

    # Camera elevation change
    elevation = cam_traj[:, 4]
    el_change = elevation[-1] - elevation[0]

    # Camera Y position change
    cam_y_change = cam_pos[-1, 1] - cam_pos[0, 1]

    # Camera XZ displacement
    cam_xz_disp = np.linalg.norm(cam_pos[-1, [0, 2]] - cam_pos[0, [0, 2]])

    # Person XZ displacement
    person_xz_disp = np.linalg.norm(person_pos[-1, [0, 2]] - person_pos[0, [0, 2]])

    # Camera position variance (should be ~0 for static)
    cam_pos_var = np.var(cam_pos, axis=0).sum()

    if label == 'static':
        ok = cam_pos_var < 0.1
        return ok, f"cam_var={cam_pos_var:.4f}"

    elif label == 'dolly-in':
        ok = dist_change < -0.3
        return ok, f"dist_change={dist_change:.2f}"

    elif label == 'dolly-out':
        ok = dist_change > 0.3
        return ok, f"dist_change={dist_change:.2f}"

    elif label == 'pan-left':
        ok = az_change < -np.radians(10)
        return ok, f"az_change={np.degrees(az_change):.1f}deg"

    elif label == 'pan-right':
        ok = az_change > np.radians(10)
        return ok, f"az_change={np.degrees(az_change):.1f}deg"

    elif label == 'crane-up':
        ok = cam_y_change > 0.2
        return ok, f"cam_y_change={cam_y_change:.2f}"

    elif label == 'crane-down':
        ok = cam_y_change < -0.2
        return ok, f"cam_y_change={cam_y_change:.2f}"

    elif label == 'track':
        # Camera should follow person laterally, distance roughly constant
        dist_std = np.std(distances)
        ok = cam_xz_disp > 0.2 and dist_std < 0.5
        return ok, f"cam_xz={cam_xz_disp:.2f}, dist_std={dist_std:.2f}"

    elif label == 'orbit':
        # Azimuth should change significantly
        ok = abs(az_change) > np.radians(20)
        return ok, f"az_change={np.degrees(az_change):.1f}deg"

    return True, "unknown_type"


def check_person_dim(data_root, samples, max_check=500):
    """Check actual dimensions of person .npy files."""
    dim_counts = defaultdict(int)
    for s in samples[:max_check]:
        path = os.path.join(data_root, s['person_trajectory_path'])
        if not os.path.exists(path):
            continue
        arr = np.load(path)
        dim_counts[arr.shape] += 1
    return dict(dim_counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='/transfer/merged-v8')
    parser.add_argument('--max-samples', type=int, default=5000,
                        help='Max samples to check per source (0=all)')
    args = parser.parse_args()

    samples = load_index(args.data_root, 'train')
    if not samples:
        print("No training samples found!")
        return

    print(f"Total training samples: {len(samples)}")
    print(f"Data root: {args.data_root}")
    print("=" * 70)

    # --- 1. Source distribution ---
    print("\n[1] SOURCE DISTRIBUTION")
    source_counts = defaultdict(int)
    for s in samples:
        src = s.get('source', 'unknown')
        source_counts[src] += 1
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / len(samples)
        print(f"  {src:20s}: {cnt:7d} ({pct:5.1f}%)")

    # --- 2. Motion type distribution per source ---
    print("\n[2] MOTION TYPE DISTRIBUTION PER SOURCE")
    motion_by_source = defaultdict(lambda: defaultdict(int))
    for s in samples:
        src = s.get('source', 'unknown')
        mt = s.get('camera_motion', 'unknown')
        motion_by_source[src][mt] += 1

    for src in sorted(motion_by_source):
        print(f"\n  {src}:")
        total_src = sum(motion_by_source[src].values())
        for mt, cnt in sorted(motion_by_source[src].items(), key=lambda x: -x[1]):
            pct = 100 * cnt / total_src
            print(f"    {mt:15s}: {cnt:6d} ({pct:5.1f}%)")

    # --- 3. Person file dimensions ---
    print("\n[3] PERSON TRAJECTORY DIMENSIONS (sample check)")
    for src in sorted(source_counts):
        src_samples = [s for s in samples if s.get('source', 'unknown') == src]
        dims = check_person_dim(args.data_root, src_samples, max_check=200)
        print(f"  {src}:")
        for shape, cnt in sorted(dims.items()):
            print(f"    shape={shape}: {cnt} files")

    # --- 4. Label accuracy verification ---
    print("\n[4] LABEL ACCURACY VERIFICATION")
    label_results = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'wrong': 0, 'errors': []}))

    for src in sorted(source_counts):
        src_samples = [s for s in samples if s.get('source', 'unknown') == src]
        if args.max_samples > 0:
            check_samples = src_samples[:args.max_samples]
        else:
            check_samples = src_samples

        for s in check_samples:
            cam_path = os.path.join(args.data_root, s['camera_trajectory_path'])
            per_path = os.path.join(args.data_root, s['person_trajectory_path'])
            if not os.path.exists(cam_path) or not os.path.exists(per_path):
                continue

            cam = np.load(cam_path).astype(np.float32)
            per = np.load(per_path).astype(np.float32)

            if cam.shape[0] < 10 or per.shape[0] < 10:
                continue

            # Use only xyz for person
            if per.shape[1] > 3:
                per_xyz = per[:, :3]
            else:
                per_xyz = per

            mt = s.get('camera_motion', 'unknown')
            ok, reason = verify_motion_label(cam, per_xyz, mt)
            if ok:
                label_results[src][mt]['correct'] += 1
            else:
                label_results[src][mt]['wrong'] += 1
                if len(label_results[src][mt]['errors']) < 3:
                    label_results[src][mt]['errors'].append(
                        f"{s['id']}: {reason}")

    for src in sorted(label_results):
        print(f"\n  {src}:")
        total_correct = 0
        total_wrong = 0
        for mt in sorted(label_results[src]):
            c = label_results[src][mt]['correct']
            w = label_results[src][mt]['wrong']
            total_correct += c
            total_wrong += w
            total = c + w
            acc = 100 * c / max(total, 1)
            marker = " !!!" if acc < 70 else ""
            print(f"    {mt:15s}: {acc:5.1f}% ({c}/{total}){marker}")
            if label_results[src][mt]['errors']:
                for err in label_results[src][mt]['errors']:
                    print(f"      e.g. {err}")
        overall_acc = 100 * total_correct / max(total_correct + total_wrong, 1)
        print(f"    {'OVERALL':15s}: {overall_acc:5.1f}%")

    # --- 5. Trajectory quality per source ---
    print("\n[5] TRAJECTORY QUALITY PER SOURCE")
    quality_stats = defaultdict(lambda: {
        'person_smoothness': [], 'camera_smoothness': [],
        'person_range': [], 'camera_range': [],
        'person_speed': [], 'person_zero_frac': 0, 'person_count': 0,
        'cam_jerk': [], 'person_jerk': [],
    })

    for src in sorted(source_counts):
        src_samples = [s for s in samples if s.get('source', 'unknown') == src]
        check_n = min(args.max_samples, len(src_samples)) if args.max_samples > 0 else len(src_samples)

        for s in src_samples[:check_n]:
            cam_path = os.path.join(args.data_root, s['camera_trajectory_path'])
            per_path = os.path.join(args.data_root, s['person_trajectory_path'])
            if not os.path.exists(cam_path) or not os.path.exists(per_path):
                continue

            cam = np.load(cam_path).astype(np.float32)
            per = np.load(per_path).astype(np.float32)
            per_xyz = per[:, :3] if per.shape[1] > 3 else per

            qs = quality_stats[src]
            qs['person_count'] += 1

            # Smoothness (acceleration)
            qs['person_smoothness'].append(compute_smoothness(per_xyz))
            qs['camera_smoothness'].append(compute_smoothness(cam[:, :3]))

            # Jerk
            qs['person_jerk'].append(compute_jerk(per_xyz))
            qs['cam_jerk'].append(compute_jerk(cam[:, :3]))

            # Range
            qs['person_range'].append(np.ptp(per_xyz, axis=0).max())
            qs['camera_range'].append(np.ptp(cam[:, :3], axis=0).max())

            # Speed
            vel = np.diff(per_xyz, axis=0)
            speed = np.linalg.norm(vel, axis=1).mean()
            qs['person_speed'].append(speed)

            # Zero check
            if np.allclose(per_xyz, 0, atol=1e-4):
                qs['person_zero_frac'] += 1

    for src in sorted(quality_stats):
        qs = quality_stats[src]
        n = qs['person_count']
        if n == 0:
            continue
        print(f"\n  {src} ({n} samples checked):")
        print(f"    Person smoothness (accel):  mean={np.mean(qs['person_smoothness']):.6f}, "
              f"max={np.max(qs['person_smoothness']):.6f}")
        print(f"    Camera smoothness (accel):  mean={np.mean(qs['camera_smoothness']):.6f}, "
              f"max={np.max(qs['camera_smoothness']):.6f}")
        print(f"    Person jerk:                mean={np.mean(qs['person_jerk']):.6f}, "
              f"max={np.max(qs['person_jerk']):.6f}")
        print(f"    Camera jerk:                mean={np.mean(qs['cam_jerk']):.6f}, "
              f"max={np.max(qs['cam_jerk']):.6f}")
        print(f"    Person range (max axis):    mean={np.mean(qs['person_range']):.4f}, "
              f"p95={np.percentile(qs['person_range'], 95):.4f}")
        print(f"    Camera range (max axis):    mean={np.mean(qs['camera_range']):.4f}, "
              f"p95={np.percentile(qs['camera_range'], 95):.4f}")
        print(f"    Person speed (mean/frame):  mean={np.mean(qs['person_speed']):.6f}, "
              f"p95={np.percentile(qs['person_speed'], 95):.6f}")
        zero_pct = 100 * qs['person_zero_frac'] / n
        if zero_pct > 0:
            print(f"    Person all-zero:            {qs['person_zero_frac']} ({zero_pct:.1f}%) !!!")

    # --- 6. Normalization stats check ---
    print("\n[6] NORMALIZATION STATS CHECK")
    for fname in ['norm_stats.json', 'norm_stats_v9.json']:
        norm_path = os.path.join(args.data_root, fname)
        if not os.path.exists(norm_path):
            continue
        with open(norm_path, 'r') as f:
            stats = json.load(f)
        mean = np.array(stats['mean'])
        std = np.array(stats['std'])
        print(f"\n  {fname} ({stats.get('n_samples', '?')} samples):")
        print(f"    Total dim: {len(mean)}")

        # Figure out person_dim from total_dim
        total_dim = len(mean)
        cam_total = 6 * 48  # 288
        person_total = total_dim - cam_total
        person_dim = person_total // 48
        print(f"    Inferred person_dim: {person_dim}")

        p_mean = mean[:person_total]
        p_std = std[:person_total]
        c_mean = mean[person_total:]
        c_std = std[person_total:]

        print(f"    Person mean: [{p_mean.min():.4f}, {p_mean.max():.4f}]")
        print(f"    Person std:  [{p_std.min():.4f}, {p_std.max():.4f}]")
        print(f"    Camera mean: [{c_mean.min():.4f}, {c_mean.max():.4f}]")
        print(f"    Camera std:  [{c_std.min():.4f}, {c_std.max():.4f}]")

        # Check for suspicious values
        n_tiny_std = (std < 0.01).sum()
        n_clamped = (std == 1.0).sum()
        if n_tiny_std > 0:
            print(f"    WARNING: {n_tiny_std} dims with std < 0.01")
        if n_clamped > 0:
            print(f"    WARNING: {n_clamped} dims clamped to std=1.0 (constant features)")

        # Per-channel analysis for person
        print(f"\n    Person per-channel stats:")
        for ch in range(person_dim):
            ch_mean = p_mean[ch::person_dim]
            ch_std = p_std[ch::person_dim]
            ch_names = ['px', 'py', 'pz', 'sin_yaw', 'cos_yaw']
            name = ch_names[ch] if ch < len(ch_names) else f'ch{ch}'
            print(f"      {name}: mean=[{ch_mean.min():.4f}, {ch_mean.max():.4f}], "
                  f"std=[{ch_std.min():.4f}, {ch_std.max():.4f}]")

        # Per-channel analysis for camera
        print(f"    Camera per-channel stats:")
        cam_names = ['tx', 'ty', 'tz', 'azimuth', 'elevation', 'roll']
        for ch in range(6):
            ch_mean = c_mean[ch::6]
            ch_std = c_std[ch::6]
            print(f"      {cam_names[ch]}: mean=[{ch_mean.min():.4f}, {ch_mean.max():.4f}], "
                  f"std=[{ch_std.min():.4f}, {ch_std.max():.4f}]")

    # --- 7. E.T. has_real_person check ---
    print("\n[7] E.T. PERSON DATA SOURCE")
    et_samples = [s for s in samples if s.get('source', 'unknown') == 'et']
    if et_samples:
        real = sum(1 for s in et_samples if s.get('has_real_person', False))
        proxy = len(et_samples) - real
        print(f"  Real person (SMPL-H): {real} ({100*real/len(et_samples):.1f}%)")
        print(f"  Look-at proxy:        {proxy} ({100*proxy/len(et_samples):.1f}%)")
        if proxy > 0:
            print(f"  WARNING: {proxy} E.T. samples use estimated person position from camera look-at!")
            print(f"           These are NOT real person trajectories - they're camera-derived.")

    # --- 8. Caption diversity check ---
    print("\n[8] CAPTION DIVERSITY PER SOURCE")
    for src in sorted(source_counts):
        src_samples = [s for s in samples if s.get('source', 'unknown') == src]
        texts = [s.get('text', '') for s in src_samples[:2000]]
        unique = len(set(texts))
        total = len(texts)
        print(f"  {src:20s}: {unique:6d} unique / {total:6d} checked "
              f"({100*unique/max(total,1):.1f}% unique)")
        # Show a few examples
        examples = list(set(texts))[:3]
        for ex in examples:
            print(f"    e.g. \"{ex[:100]}\"")

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
