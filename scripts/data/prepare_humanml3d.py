"""
Prepare HumanML3D data as a training source for joint person-camera trajectory.

Reads HumanML3D's index.csv to map motion IDs to AMASS source .npz files,
extracts root position + yaw, pairs with human-written text annotations,
and generates synthetic camera trajectories.

Only processes motions whose source AMASS files exist locally.

Outputs (same format as other prepare_* scripts):
  - <output-root>/camera_trajectories/*.npy  (48, 6)
  - <output-root>/person_trajectories/*.npy  (48, 5)
  - <output-root>/train_index.json

Usage:
    python scripts/data/prepare_humanml3d.py --amass-root /transfer/amassdata --humanml3d-root /transfer/HumanML3D --output-root /transfer/humanml3d-stc-data-v7
"""

import os
import re
import csv
import json
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Reuse camera generation and action inference from prepare_amass
from scripts.prepare_amass import (
    generate_camera_for_person,
    infer_action,
    resample_trajectory,
    is_valid_trajectory,
    ALL_MOTION_TYPES,
    COMBINED_MOTION_TYPES,
    CAPTION_TEMPLATES,
    SHOT_TYPES,
)


def axis_angle_to_yaw(aa):
    """Convert (3,) axis-angle to yaw (float) via rotation matrix."""
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return 0.0
    axis = aa / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return float(np.arctan2(R[0, 2], R[2, 2]))


def load_index_csv(index_path):
    """Load HumanML3D index.csv → list of (new_name, source_path, start, end)."""
    entries = []
    with open(index_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            source_path, start_str, end_str, new_name = row[0], row[1], row[2], row[3]
            if source_path == 'source_path':
                continue  # header
            motion_id = new_name.replace('.npy', '')
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                continue
            entries.append({
                'motion_id': motion_id,
                'source_path': source_path,
                'start_frame': start,
                'end_frame': end,
            })
    return entries


def parse_captions(texts_dir, motion_id):
    """Read captions from texts/<motion_id>.txt."""
    txt_path = os.path.join(texts_dir, f'{motion_id}.txt')
    if not os.path.exists(txt_path):
        return []
    captions = []
    with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            caption = line.split('#')[0].strip()
            if caption:
                captions.append(caption)
    return captions


def resolve_amass_path(source_path, amass_root):
    """
    Map HumanML3D source_path to actual AMASS .npz file.

    HumanML3D uses:  ./pose_data/CMU/80/80_63_poses.npy
    AMASS might be:  /amassdata/CMU/CMU/80/80_63_poses.npz

    Tries multiple patterns to find the file.
    """
    # Strip ./pose_data/ prefix
    clean = source_path
    for prefix in ['./pose_data/', 'pose_data/', './', '/']:
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break

    # Skip humanact12 (not AMASS)
    if 'humanact12' in clean.lower():
        return None

    # Base name without extension
    stem = clean.replace('.npy', '')

    # Try multiple patterns
    candidates = [
        # Direct: CMU/80/80_63_poses.npz
        os.path.join(amass_root, stem + '.npz'),
        # Double dir: CMU/CMU/80/80_63_poses.npz
        os.path.join(amass_root, stem.split('/')[0], stem + '.npz'),
        # Without _poses suffix
        os.path.join(amass_root, stem.replace('_poses', '') + '.npz'),
        os.path.join(amass_root, stem.split('/')[0], stem.replace('_poses', '') + '.npz'),
        # _stageii variant
        os.path.join(amass_root, stem.replace('_poses', '_stageii') + '.npz'),
        os.path.join(amass_root, stem.split('/')[0], stem.replace('_poses', '_stageii') + '.npz'),
    ]

    for c in candidates:
        if os.path.exists(c):
            return c

    return None


def extract_person_traj(npz_path, start_frame, end_frame, num_frames=48):
    """
    Load AMASS .npz, extract root position + sin/cos yaw for specified frame range.
    Returns (num_frames, 5) or None.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception:
        return None

    # Root translation
    trans = None
    for key in ['trans', 'transl', 'root_translation']:
        if key in data:
            trans = np.array(data[key], dtype=np.float32)
            break
    if trans is None or trans.ndim != 2 or trans.shape[1] < 3:
        return None
    trans = trans[:, :3]

    # Root orientation (axis-angle)
    root_orient = None
    if 'root_orient' in data:
        root_orient = np.array(data['root_orient'], dtype=np.float32)
    elif 'poses' in data:
        poses = np.array(data['poses'], dtype=np.float32)
        if poses.ndim == 2 and poses.shape[1] >= 3:
            root_orient = poses[:, :3]

    # Apply frame range
    T = trans.shape[0]
    if end_frame == -1:
        end_frame = T
    end_frame = min(end_frame, T)
    start_frame = max(0, min(start_frame, T - 1))

    if end_frame - start_frame < 4:
        return None

    trans_clip = trans[start_frame:end_frame]

    # Compute yaw
    yaw = np.zeros(trans_clip.shape[0], dtype=np.float32)
    if root_orient is not None and root_orient.shape[0] >= end_frame:
        ro_clip = root_orient[start_frame:end_frame]
        for i in range(len(ro_clip)):
            if ro_clip.shape[1] >= 3:
                yaw[i] = axis_angle_to_yaw(ro_clip[i, :3])

    # Combine to (T, 5) with sin/cos yaw
    sin_yaw = np.sin(yaw).reshape(-1, 1)
    cos_yaw = np.cos(yaw).reshape(-1, 1)
    person_traj = np.concatenate([trans_clip, sin_yaw, cos_yaw], axis=1)

    # Resample to target frames
    person_traj = resample_trajectory(person_traj, num_frames)

    if not is_valid_trajectory(person_traj):
        return None

    return person_traj


def main():
    parser = argparse.ArgumentParser(
        description='Prepare HumanML3D data for joint person-camera trajectory training.'
    )
    parser.add_argument('--amass-root', type=str, default='/transfer/amassdata')
    parser.add_argument('--humanml3d-root', type=str, default='/transfer/HumanML3D')
    parser.add_argument('--output-root', type=str, default='/transfer/humanml3d-stc-data-v7')
    parser.add_argument('--num-frames', type=int, default=48)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Find texts and index
    texts_dir = None
    for candidate in [
        os.path.join(args.humanml3d_root, 'HumanML3D', 'texts'),
        os.path.join(args.humanml3d_root, 'texts'),
    ]:
        if os.path.isdir(candidate):
            texts_dir = candidate
            break
    if texts_dir is None:
        print(f"Cannot find texts/ under {args.humanml3d_root}")
        return

    index_path = os.path.join(args.humanml3d_root, 'index.csv')
    if not os.path.exists(index_path):
        print(f"Cannot find index.csv at {index_path}")
        return

    print(f"Texts: {texts_dir}")
    print(f"Index: {index_path}")
    print(f"AMASS: {args.amass_root}")

    # Load index
    entries = load_index_csv(index_path)
    print(f"Index entries: {len(entries)}")

    # Output dirs
    cam_out = os.path.join(args.output_root, 'camera_trajectories')
    per_out = os.path.join(args.output_root, 'person_trajectories')
    os.makedirs(cam_out, exist_ok=True)
    os.makedirs(per_out, exist_ok=True)

    train_index = []
    stats = {'matched': 0, 'unmatched': 0, 'no_text': 0, 'load_fail': 0, 'samples': 0}
    motion_counts = {mt: 0 for mt in ALL_MOTION_TYPES}

    for entry in tqdm(entries, desc="Processing HumanML3D"):
        motion_id = entry['motion_id']

        # Get captions
        captions = parse_captions(texts_dir, motion_id)
        if not captions:
            stats['no_text'] += 1
            continue

        # Resolve AMASS file
        npz_path = resolve_amass_path(entry['source_path'], args.amass_root)
        if npz_path is None:
            stats['unmatched'] += 1
            continue

        stats['matched'] += 1

        # Extract person trajectory
        person_traj = extract_person_traj(
            npz_path, entry['start_frame'], entry['end_frame'], args.num_frames
        )
        if person_traj is None:
            stats['load_fail'] += 1
            continue

        # Generate camera for each motion type (balanced)
        for motion_type in ALL_MOTION_TYPES:
            sample_id = f"hml3d_{motion_id}_{motion_type}"

            camera_traj, shot_type = generate_camera_for_person(
                person_traj, motion_type, args.num_frames
            )
            if camera_traj is None or not is_valid_trajectory(camera_traj):
                continue

            np.save(os.path.join(cam_out, f'{sample_id}.npy'), camera_traj)
            np.save(os.path.join(per_out, f'{sample_id}.npy'), person_traj)

            # Use human-written caption + camera motion description + shot type prefix
            person_caption = random.choice(captions)
            cam_template = random.choice(CAPTION_TEMPLATES[motion_type])
            shot_prefix = f"{shot_type.replace('-', ' ').title()}. " if shot_type != 'medium-shot' else ""
            cam_text = f"{shot_prefix}{cam_template.format(action=person_caption.rstrip('.').lower())}"

            train_index.append({
                'id': sample_id,
                'text': cam_text,
                'shot_type': shot_type,
                'camera_motion': motion_type,
                'camera_trajectory_path': f'camera_trajectories/{sample_id}.npy',
                'person_trajectory_path': f'person_trajectories/{sample_id}.npy',
                'has_real_person': True,
                'source': 'humanml3d',
            })

            motion_counts[motion_type] += 1
            stats['samples'] += 1

    # Save index
    index_out = os.path.join(args.output_root, 'train_index.json')
    with open(index_out, 'w', encoding='utf-8') as f:
        json.dump(train_index, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("HumanML3D Preparation Complete!")
    print(f"{'='*60}")
    print(f"  Index entries: {len(entries)}")
    print(f"  AMASS matched: {stats['matched']}")
    print(f"  Unmatched (no AMASS file): {stats['unmatched']}")
    print(f"  No text file: {stats['no_text']}")
    print(f"  Load failures: {stats['load_fail']}")
    print(f"  Total samples: {stats['samples']}")
    print(f"\n  Motion type distribution:")
    for mt, count in sorted(motion_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(stats['samples'], 1)
        print(f"    {mt:15s}: {count:6d} ({pct:5.1f}%)")
    print(f"\n  Output: {args.output_root}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
