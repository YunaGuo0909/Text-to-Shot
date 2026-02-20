"""
Preprocess E.T. (Exceptional Trajectories) dataset into the format
required by the Script-to-Camera training pipeline.

Converts:
  - 3×4 camera extrinsic matrices → 6D camera state (tx, ty, tz, azimuth, elevation, roll)
  - caption_cam text files → text descriptions
  - Keywords in captions → camera_motion type labels

Outputs:
  - data/trajectories/*.npy  (T, 6) per-sample trajectory arrays
  - data/train_index.json    training index file
  - data/test_index.json     test index file

Usage:
    python scripts/preprocess_et_data.py
    python scripts/preprocess_et_data.py --et-root data/et-data --output-root data --num-frames 48
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm


def parse_extrinsic_line(line: str) -> np.ndarray:
    """Parse a line of 12 floats into a 3×4 [R|t] matrix."""
    vals = [float(x) for x in line.strip().split()]
    assert len(vals) == 12, f"Expected 12 values, got {len(vals)}"
    return np.array(vals).reshape(3, 4)


def extrinsic_to_6d(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Convert camera extrinsic (R, t) to 6D state:
    (tx, ty, tz, azimuth, elevation, roll)

    - azimuth (θ):  rotation around Y axis (left-right panning)
    - elevation (φ): rotation around X axis (up-down tilting)
    - roll (ψ):     rotation around Z axis (camera tilt/dutch angle)
    """
    tx, ty, tz = t[0], t[1], t[2]

    # Extract Euler angles from rotation matrix (ZYX convention)
    # R = Rz(ψ) @ Ry(θ) @ Rx(φ)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    if sy > 1e-6:
        elevation = np.arctan2(-R[2, 0], sy)             # φ
        azimuth = np.arctan2(R[1, 0], R[0, 0])           # θ
        roll = np.arctan2(R[2, 1], R[2, 2])              # ψ
    else:
        elevation = np.arctan2(-R[2, 0], sy)
        azimuth = np.arctan2(-R[1, 2], R[1, 1])
        roll = 0.0

    return np.array([tx, ty, tz, azimuth, elevation, roll], dtype=np.float32)


def load_trajectory(traj_path: str) -> np.ndarray:
    """
    Load a trajectory file and convert each frame to 6D camera state.

    Returns:
        trajectory: (T, 6) array
    """
    with open(traj_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    frames = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            mat = parse_extrinsic_line(line)
            R = mat[:, :3]
            t = mat[:, 3]
            state = extrinsic_to_6d(R, t)
            frames.append(state)
        except Exception:
            continue

    if len(frames) == 0:
        return np.zeros((1, 6), dtype=np.float32)

    return np.stack(frames, axis=0)


def resample_trajectory(trajectory: np.ndarray, target_frames: int) -> np.ndarray:
    """Resample trajectory to fixed number of frames via linear interpolation."""
    src_frames = trajectory.shape[0]
    if src_frames == target_frames:
        return trajectory

    src_t = np.linspace(0, 1, src_frames)
    tgt_t = np.linspace(0, 1, target_frames)

    resampled = np.zeros((target_frames, 6), dtype=np.float32)
    for dim in range(6):
        resampled[:, dim] = np.interp(tgt_t, src_t, trajectory[:, dim])

    return resampled


def classify_camera_motion(caption: str) -> str:
    """
    Classify camera motion type from caption text using keyword matching.
    Maps E.T. terminology to our motion categories.
    """
    text = caption.lower()

    # Order matters: check specific terms before general ones
    if 'static' in text or 'stationary' in text or 'remains still' in text:
        return 'static'
    if 'push-in' in text or 'push in' in text or 'pushes in' in text:
        return 'dolly-in'
    if 'pull-out' in text or 'pull out' in text or 'pulls out' in text or 'pull back' in text:
        return 'dolly-out'
    if 'dolly' in text:
        if 'in' in text or 'forward' in text:
            return 'dolly-in'
        elif 'out' in text or 'back' in text:
            return 'dolly-out'
        return 'dolly-in'
    if 'pan' in text:
        if 'left' in text:
            return 'pan-left'
        elif 'right' in text:
            return 'pan-right'
        return 'pan-right'
    if 'tilt' in text:
        if 'up' in text:
            return 'crane-up'
        elif 'down' in text:
            return 'crane-down'
        return 'crane-up'
    if 'crane' in text or 'pedestal' in text or 'boom' in text:
        if 'up' in text or 'top' in text or 'rise' in text:
            return 'crane-up'
        elif 'down' in text or 'bottom' in text or 'lower' in text:
            return 'crane-down'
        return 'crane-up'
    if 'orbit' in text or 'arc' in text or 'circular' in text:
        return 'orbit'
    if 'truck' in text or 'lateral' in text or 'tracking' in text:
        if 'left' in text:
            return 'track'
        elif 'right' in text:
            return 'track'
        return 'track'
    if 'follow' in text or 'track' in text:
        return 'track'
    if 'zoom' in text:
        if 'in' in text:
            return 'dolly-in'
        elif 'out' in text:
            return 'dolly-out'
        return 'dolly-in'
    if 'move' in text:
        if 'left' in text:
            return 'pan-left'
        elif 'right' in text:
            return 'pan-right'
        elif 'up' in text:
            return 'crane-up'
        elif 'down' in text:
            return 'crane-down'
        elif 'forward' in text:
            return 'dolly-in'
        elif 'back' in text:
            return 'dolly-out'
        return 'track'

    return 'static'


def infer_shot_type(caption: str) -> str:
    """Infer shot type from caption text. Default to medium-shot."""
    text = caption.lower()
    if 'close-up' in text or 'closeup' in text or 'close up' in text:
        return 'close-up'
    if 'wide' in text or 'establish' in text:
        return 'wide-shot'
    if 'over the shoulder' in text or 'over-the-shoulder' in text:
        return 'over-the-shoulder'
    if 'two-shot' in text or 'two shot' in text:
        return 'two-shot'
    return 'medium-shot'


def main():
    parser = argparse.ArgumentParser(description='Preprocess E.T. dataset')
    parser.add_argument('--et-root', type=str, default='data/et-data',
                        help='Path to E.T. dataset root')
    parser.add_argument('--output-root', type=str, default='data',
                        help='Output directory for processed data')
    parser.add_argument('--num-frames', type=int, default=48,
                        help='Number of frames to resample each trajectory to')
    parser.add_argument('--min-frames', type=int, default=10,
                        help='Minimum number of raw frames to include a sample')
    args = parser.parse_args()

    et_root = args.et_root
    output_root = args.output_root
    num_frames = args.num_frames

    traj_dir = os.path.join(et_root, 'traj')
    caption_dir = os.path.join(et_root, 'caption')
    caption_cam_dir = os.path.join(et_root, 'caption_cam')

    # Output directories
    out_traj_dir = os.path.join(output_root, 'trajectories')
    os.makedirs(out_traj_dir, exist_ok=True)

    # Load split files
    splits = {}
    for split_name in ['train', 'test']:
        split_file = os.path.join(et_root, f'full_{split_name}_split.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                splits[split_name] = set(line.strip() for line in f if line.strip())
            print(f"Loaded {split_name} split: {len(splits[split_name])} samples")
        else:
            print(f"Warning: {split_file} not found")
            splits[split_name] = set()

    # Get all sample IDs from trajectory directory
    all_sample_ids = sorted(
        f[:-4] for f in os.listdir(traj_dir) if f.endswith('.txt')
    )
    print(f"\nTotal trajectory files: {len(all_sample_ids)}")

    # Process samples
    train_index = []
    test_index = []
    motion_counts = {}
    skipped = 0

    for sample_id in tqdm(all_sample_ids, desc="Processing trajectories"):
        traj_path = os.path.join(traj_dir, f'{sample_id}.txt')

        # Load and convert trajectory
        trajectory = load_trajectory(traj_path)

        if trajectory.shape[0] < args.min_frames:
            skipped += 1
            continue

        # Resample to fixed length
        trajectory = resample_trajectory(trajectory, num_frames)

        # Save as .npy
        npy_filename = f'{sample_id}.npy'
        npy_path = os.path.join(out_traj_dir, npy_filename)
        np.save(npy_path, trajectory)

        # Load caption (camera-specific)
        caption_cam_path = os.path.join(caption_cam_dir, f'{sample_id}.txt')
        caption_path = os.path.join(caption_dir, f'{sample_id}.txt')

        caption_cam = ''
        caption_full = ''
        if os.path.exists(caption_cam_path):
            try:
                with open(caption_cam_path, 'r', encoding='utf-8', errors='replace') as f:
                    caption_cam = f.read().strip()
            except Exception:
                pass
        if os.path.exists(caption_path):
            try:
                with open(caption_path, 'r', encoding='utf-8', errors='replace') as f:
                    caption_full = f.read().strip()
            except Exception:
                pass

        # Use camera caption as primary, fall back to full caption
        text = caption_cam if caption_cam else caption_full

        # Classify motion type and shot type
        camera_motion = classify_camera_motion(text)
        shot_type = infer_shot_type(caption_full)

        motion_counts[camera_motion] = motion_counts.get(camera_motion, 0) + 1

        # Build sample entry
        entry = {
            'id': sample_id,
            'text': text,
            'shot_type': shot_type,
            'camera_motion': camera_motion,
            'trajectory_path': f'trajectories/{npy_filename}',
            'num_raw_frames': int(trajectory.shape[0]),
        }

        # Assign to split
        if sample_id in splits.get('test', set()):
            test_index.append(entry)
        else:
            train_index.append(entry)

    # Save index files
    train_path = os.path.join(output_root, 'train_index.json')
    test_path = os.path.join(output_root, 'test_index.json')

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_index, f, indent=2, ensure_ascii=False)

    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_index, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"  Train samples: {len(train_index)}")
    print(f"  Test samples:  {len(test_index)}")
    print(f"  Skipped (< {args.min_frames} frames): {skipped}")
    print(f"  Frames per trajectory: {num_frames}")
    print(f"  State dimension: 6 (tx, ty, tz, azimuth, elevation, roll)")
    print(f"\n  Camera motion distribution:")
    for motion, count in sorted(motion_counts.items(), key=lambda x: -x[1]):
        pct = count / (len(train_index) + len(test_index)) * 100
        print(f"    {motion:15s}: {count:6d} ({pct:5.1f}%)")
    print(f"\n  Output files:")
    print(f"    {train_path}")
    print(f"    {test_path}")
    print(f"    {out_traj_dir}/ ({len(train_index) + len(test_index)} .npy files)")
    print("=" * 60)


if __name__ == '__main__':
    main()
