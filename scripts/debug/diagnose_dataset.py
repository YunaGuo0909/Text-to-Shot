"""
Deep dataset diagnostic for /transfer/stc-data.

Checks:
  1. Angle range and smoothness (raw, before normalization)
  2. Camera-person distance distribution per motion_type
  3. Camera look-at alignment (does camera actually face person?)
  4. Label consistency (do dolly-in samples have decreasing distance?)
  5. Person trajectory quality (displacement, speed)
  6. Outlier detection per split

Usage:
    python scripts/debug/diagnose_dataset.py --data-root /transfer/stc-data --n 2000
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict

RAD2DEG = 180.0 / np.pi


def camera_forward(az, el):
    """Unit forward vector from azimuth + elevation (radians)."""
    fx = np.cos(el) * np.sin(az)
    fy = -np.sin(el)
    fz = -np.cos(el) * np.cos(az)
    return np.stack([fx, fy, fz], axis=-1)


def lookat_alignment(cam_traj, per_traj):
    """Mean cosine similarity between camera forward and camera-to-person direction."""
    cam_pos = cam_traj[:, :3]
    az, el = cam_traj[:, 3], cam_traj[:, 4]
    fwd = camera_forward(az, el)                       # (T, 3)
    to_person = per_traj - cam_pos                     # (T, 3)
    dist = np.linalg.norm(to_person, axis=-1, keepdims=True).clip(1e-6)
    to_person_norm = to_person / dist
    cos = (fwd * to_person_norm).sum(axis=-1)          # (T,)
    return cos.mean()


def angle_smoothness(traj):
    """Max and mean absolute frame-to-frame change in angle dims (az, el, roll)."""
    angles = traj[:, 3:]  # (T, 3)
    diff = np.abs(np.diff(angles, axis=0))  # (T-1, 3)
    return diff.mean(), diff.max()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/transfer/stc-data')
    parser.add_argument('--n', type=int, default=2000,
                        help='Max samples to analyse per split')
    args = parser.parse_args()

    # ── Load index ───────────────────────────────────────────────────
    idx_path = os.path.join(args.data_root, 'train_index.json')
    with open(idx_path, encoding='utf-8') as f:
        samples = json.load(f)
    print(f"Total train samples: {len(samples)}")

    # Sample evenly across motion types
    by_motion = defaultdict(list)
    for s in samples:
        by_motion[s.get('camera_motion', 'unknown')].append(s)

    selected = []
    per_class = max(1, args.n // len(by_motion))
    for motion, slist in sorted(by_motion.items()):
        selected.extend(slist[:per_class])
    print(f"Analysing {len(selected)} samples ({per_class} per motion type)\n")

    # ── Per-motion statistics ────────────────────────────────────────
    stats = defaultdict(lambda: {
        'dist_changes': [],        # end_dist - start_dist
        'lookat_cos': [],          # camera alignment with person
        'angle_mean_diff': [],     # mean frame-to-frame angle change (rad)
        'angle_max_diff': [],      # max frame-to-frame angle change (rad)
        'person_displacement': [], # total person movement
        'cam_displacement': [],    # total camera translation movement
        'cam_angle_range': [],     # total angle range (max-min)
    })

    skipped = 0
    for s in selected:
        cam_path = os.path.join(args.data_root, s.get('camera_trajectory_path', ''))
        per_path = os.path.join(args.data_root, s.get('person_trajectory_path', ''))
        if not (os.path.exists(cam_path) and os.path.exists(per_path)):
            skipped += 1
            continue

        cam = np.load(cam_path).astype(np.float32)   # (48, 6)
        per = np.load(per_path).astype(np.float32)   # (48, 3)

        if cam.shape != (48, 6) or per.shape != (48, 3):
            skipped += 1
            continue
        if not (np.isfinite(cam).all() and np.isfinite(per).all()):
            skipped += 1
            continue

        motion = s.get('camera_motion', 'unknown')
        st = stats[motion]

        # Distance change
        q = 12  # quarter of 48
        dist_start = np.linalg.norm(cam[:q, :3] - per[:q, :], axis=-1).mean()
        dist_end   = np.linalg.norm(cam[-q:, :3] - per[-q:, :], axis=-1).mean()
        st['dist_changes'].append(dist_end - dist_start)

        # Look-at alignment
        st['lookat_cos'].append(lookat_alignment(cam, per))

        # Angle smoothness
        mean_diff, max_diff = angle_smoothness(cam)
        st['angle_mean_diff'].append(mean_diff)
        st['angle_max_diff'].append(max_diff)

        # Person displacement
        st['person_displacement'].append(
            np.linalg.norm(per[-1] - per[0]))

        # Camera translation displacement
        st['cam_displacement'].append(
            np.linalg.norm(cam[-1, :3] - cam[0, :3]))

        # Camera angle range (max - min per angle dim)
        st['cam_angle_range'].append(
            (cam[:, 3:].max(axis=0) - cam[:, 3:].min(axis=0)).mean())

    if skipped:
        print(f"Skipped {skipped} samples (missing/bad files)\n")

    # ── Print report ─────────────────────────────────────────────────
    print(f"{'='*80}")
    print(f"{'Motion':<12} {'N':>5}  "
          f"{'dist_chg':>10}  "
          f"{'lookat':>8}  "
          f"{'ang_mean':>10}  "
          f"{'ang_max':>9}  "
          f"{'per_disp':>10}  "
          f"{'cam_disp':>10}")
    print(f"{'':12} {'':>5}  "
          f"{'(end-start)':>10}  "
          f"{'cosine':>8}  "
          f"{'(rad/frm)':>10}  "
          f"{'(rad/frm)':>9}  "
          f"{'(m)':>10}  "
          f"{'(m)':>10}")
    print(f"{'-'*80}")

    for motion in sorted(stats.keys()):
        st = stats[motion]
        n = len(st['dist_changes'])
        if n == 0:
            continue
        dc  = np.array(st['dist_changes'])
        lk  = np.array(st['lookat_cos'])
        amd = np.array(st['angle_mean_diff'])
        amx = np.array(st['angle_max_diff'])
        pd_ = np.array(st['person_displacement'])
        cd  = np.array(st['cam_displacement'])

        print(f"{motion:<12} {n:>5}  "
              f"{dc.mean():>+10.4f}  "
              f"{lk.mean():>8.4f}  "
              f"{amd.mean():>10.6f}  "
              f"{amx.mean():>9.4f}  "
              f"{pd_.mean():>10.4f}  "
              f"{cd.mean():>10.4f}")

    print(f"{'='*80}")

    # ── Dolly label consistency ──────────────────────────────────────
    print("\n--- Dolly label consistency ---")
    for motion in ['dolly-in', 'dolly-out']:
        if motion not in stats:
            continue
        dc = np.array(stats[motion]['dist_changes'])
        n_correct = (dc < 0).sum() if motion == 'dolly-in' else (dc > 0).sum()
        print(f"  {motion}: {n_correct}/{len(dc)} samples have "
              f"{'decreasing' if motion=='dolly-in' else 'increasing'} distance "
              f"({100*n_correct/max(len(dc),1):.1f}% consistent)")

    # ── Look-at quality summary ──────────────────────────────────────
    all_cos = []
    for st in stats.values():
        all_cos.extend(st['lookat_cos'])
    all_cos = np.array(all_cos)
    print(f"\n--- Camera look-at alignment (cos similarity, 1=perfect) ---")
    print(f"  Mean: {all_cos.mean():.4f}")
    print(f"  <0.5 (bad alignment): {(all_cos < 0.5).sum()}/{len(all_cos)} "
          f"({100*(all_cos < 0.5).mean():.1f}%)")
    print(f"  <0.0 (facing away):   {(all_cos < 0.0).sum()}/{len(all_cos)} "
          f"({100*(all_cos < 0.0).mean():.1f}%)")

    # ── Angle smoothness summary ─────────────────────────────────────
    all_amd = []
    for st in stats.values():
        all_amd.extend(st['angle_mean_diff'])
    all_amd = np.array(all_amd)
    print(f"\n--- Angle smoothness (mean frame-to-frame change, radians) ---")
    print(f"  Mean:   {all_amd.mean():.6f} rad = {all_amd.mean()*RAD2DEG:.4f} deg/frame")
    print(f"  Median: {np.median(all_amd):.6f} rad = {np.median(all_amd)*RAD2DEG:.4f} deg/frame")
    print(f"  95th %: {np.percentile(all_amd, 95):.6f} rad = {np.percentile(all_amd, 95)*RAD2DEG:.4f} deg/frame")

    # ── Additional: check if angles are radians or degrees ───────────
    print(f"\n--- Angle value range check (to confirm radians vs degrees) ---")
    sample_angles = []
    for s in selected[:200]:
        cam_path = os.path.join(args.data_root, s.get('camera_trajectory_path', ''))
        if os.path.exists(cam_path):
            cam = np.load(cam_path).astype(np.float32)
            if cam.shape == (48, 6) and np.isfinite(cam).all():
                sample_angles.append(cam[:, 3:])  # az, el, roll
    if sample_angles:
        angles = np.concatenate(sample_angles, axis=0)
        print(f"  Raw angle values: min={angles.min():.4f}, max={angles.max():.4f}, "
              f"mean: {angles.mean():.4f}")
        print(f"  Interpretation: {'RADIANS (within ±π)' if abs(angles).max() < 4.0 else 'DEGREES or other'}")


if __name__ == '__main__':
    main()
