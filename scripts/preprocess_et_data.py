"""
Preprocess E.T. dataset for joint person-camera trajectory training.

Extracts BOTH camera trajectory (T, 6) and person trajectory (T, 5) per sample.

Camera: 3x4 extrinsic matrices → (tx, ty, tz, azimuth, elevation, roll)
Person: If character joint data exists, extract root position + sin/cos yaw.
        Otherwise, estimate look-at point from camera as person proxy (sin_yaw=0, cos_yaw=1).

Outputs:
  - stc-data/camera_trajectories/*.npy  (T, 6)
  - stc-data/person_trajectories/*.npy  (T, 5)
  - stc-data/train_index.json
  - stc-data/test_index.json

Usage:
    python scripts/preprocess_et_data.py
    python scripts/preprocess_et_data.py --et-root /transfer/et-data --output-root /transfer/stc-data
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm


def parse_extrinsic_line(line: str) -> np.ndarray:
    """Parse 12 floats into a 3x4 [R|t] matrix."""
    vals = [float(x) for x in line.strip().split()]
    assert len(vals) == 12, f"Expected 12 values, got {len(vals)}"
    return np.array(vals).reshape(3, 4)


def extrinsic_to_6d(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Convert (R, t) to 6D: (tx, ty, tz, azimuth, elevation, roll)."""
    tx, ty, tz = t[0], t[1], t[2]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        elevation = np.arctan2(-R[2, 0], sy)
        azimuth = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        elevation = np.arctan2(-R[2, 0], sy)
        azimuth = np.arctan2(-R[1, 2], R[1, 1])
        roll = 0.0
    return np.array([tx, ty, tz, azimuth, elevation, roll], dtype=np.float32)


def extrinsic_to_lookat(R: np.ndarray, t: np.ndarray, distance: float = 3.0) -> np.ndarray:
    """Estimate look-at point from camera extrinsic (used as person position proxy)."""
    # Camera forward direction (negative Z in camera frame → world)
    forward = -R[:, 2]
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    lookat = t + forward * distance
    return lookat.astype(np.float32)


def load_camera_trajectory(traj_path: str):
    """Load trajectory file → list of (R, t) pairs."""
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
            frames.append((R, t))
        except Exception:
            continue
    return frames


def rotation_matrix_to_yaw(R: np.ndarray) -> float:
    """Extract yaw angle from a 3x3 rotation matrix (Y-up convention)."""
    return float(np.arctan2(R[0, 2], R[2, 2]))


def load_person_joints(joints_dir: str, sample_id: str):
    """
    Load person root position + yaw from E.T. dataset character data.

    E.T. smplh/ files are .pkl (pickle) dicts with torch.Tensor values:
        betas, global_orient, body_pose, left_hand_pose, right_hand_pose, transl

    We extract 'transl' as (T, 3) root translation and 'global_orient' as
    (T, 1, 3, 3) root rotation matrix to derive yaw.

    Also supports .npy/.npz formats as fallback.
    Returns (T, 5) [px, py, pz, sin_yaw, cos_yaw] or (T, 3) [px, py, pz] if no orientation, or None.
    """
    import torch

    # === Try .pkl first (E.T. smplh/ format) ===
    pkl_path = os.path.join(joints_dir, f'{sample_id}.pkl')
    if os.path.exists(pkl_path):
        try:
            import pickle
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'transl' in data:
                transl = data['transl']
                if isinstance(transl, torch.Tensor):
                    # E.T. smplh tensors have requires_grad=True, must detach
                    transl = transl.detach().cpu().numpy()
                transl = np.array(transl, dtype=np.float32)

                # Extract global_orient for yaw if available
                global_orient = None
                if 'global_orient' in data:
                    go = data['global_orient']
                    if isinstance(go, torch.Tensor):
                        go = go.detach().cpu().numpy()
                    go = np.array(go, dtype=np.float32)
                    global_orient = go

                if transl.ndim == 2 and transl.shape[1] >= 3:
                    positions = transl[:, :3]
                elif transl.ndim == 1 and transl.shape[0] >= 3:
                    positions = transl[:3].reshape(1, 3)
                else:
                    positions = None

                if positions is not None and global_orient is not None:
                    T_frames = positions.shape[0]
                    yaws = np.zeros(T_frames, dtype=np.float32)
                    try:
                        # global_orient is (T, 1, 3, 3) or (T, 3, 3)
                        if global_orient.ndim == 4:
                            global_orient = global_orient[:, 0, :, :]  # (T, 3, 3)
                        if global_orient.ndim == 3 and global_orient.shape[1:] == (3, 3):
                            for t in range(min(T_frames, global_orient.shape[0])):
                                yaws[t] = rotation_matrix_to_yaw(global_orient[t])
                    except Exception:
                        pass  # yaws stays zeros
                    sin_yaw = np.sin(yaws[:T_frames]).reshape(-1, 1)
                    cos_yaw = np.cos(yaws[:T_frames]).reshape(-1, 1)
                    result = np.concatenate([positions, sin_yaw, cos_yaw], axis=1)
                    return result  # (T, 5)
                elif positions is not None:
                    return positions  # (T, 3) fallback
        except Exception as e:
            print(f"[smplh load failed for {sample_id}] {type(e).__name__}: {e}")

    # === Fallback: .npy / .npz ===
    for ext in ['.npy', '.npz']:
        path = os.path.join(joints_dir, f'{sample_id}{ext}')
        if not os.path.exists(path):
            continue
        try:
            if ext == '.npy':
                data = np.load(path, allow_pickle=True)
                if data.ndim == 0:
                    data = data.item()
                if isinstance(data, dict):
                    for key in ['transl', 'trans', 'root_translation']:
                        if key in data:
                            arr = np.array(data[key], dtype=np.float32)
                            if isinstance(data[key], torch.Tensor):
                                arr = data[key].cpu().numpy().astype(np.float32)
                            if arr.ndim == 2 and arr.shape[1] >= 3:
                                return arr[:, :3]
                elif isinstance(data, np.ndarray):
                    if data.ndim >= 2 and data.shape[-1] >= 3:
                        if data.ndim == 3:
                            return data[:, 0, :3].astype(np.float32)
                        return data[:, :3].astype(np.float32)
            elif ext == '.npz':
                data = np.load(path)
                for key in ['transl', 'trans', 'root_translation']:
                    if key in data and data[key].ndim == 2:
                        return data[key][:, :3].astype(np.float32)
        except Exception:
            continue
    return None


def resample_trajectory(trajectory: np.ndarray, target_frames: int) -> np.ndarray:
    src_frames = trajectory.shape[0]
    if src_frames == target_frames:
        return trajectory
    dim = trajectory.shape[1]
    src_t = np.linspace(0, 1, src_frames)
    tgt_t = np.linspace(0, 1, target_frames)
    resampled = np.zeros((target_frames, dim), dtype=np.float32)
    for d in range(dim):
        resampled[:, d] = np.interp(tgt_t, src_t, trajectory[:, d])
    return resampled


def classify_camera_motion(caption: str) -> str:
    text = caption.lower()
    if 'static' in text or 'stationary' in text or 'remains still' in text:
        return 'static'
    if 'push-in' in text or 'push in' in text or 'pushes in' in text:
        return 'dolly-in'
    if 'pull-out' in text or 'pull out' in text or 'pull back' in text:
        return 'dolly-out'
    if 'dolly' in text:
        return 'dolly-in' if ('in' in text or 'forward' in text) else 'dolly-out'
    if 'pan' in text:
        return 'pan-left' if 'left' in text else 'pan-right'
    if 'tilt' in text or 'crane' in text or 'pedestal' in text:
        return 'crane-up' if ('up' in text or 'rise' in text) else 'crane-down'
    if 'orbit' in text or 'arc' in text or 'circular' in text:
        return 'orbit'
    if 'truck' in text or 'lateral' in text or 'tracking' in text or 'follow' in text or 'track' in text:
        return 'track'
    if 'zoom' in text:
        return 'dolly-in' if 'in' in text else 'dolly-out'
    return 'static'


def infer_shot_type(caption: str) -> str:
    text = caption.lower()
    if 'close-up' in text or 'closeup' in text:
        return 'close-up'
    if 'wide' in text or 'establish' in text:
        return 'wide-shot'
    if 'over the shoulder' in text or 'over-the-shoulder' in text:
        return 'over-the-shoulder'
    if 'two-shot' in text or 'two shot' in text:
        return 'two-shot'
    return 'medium-shot'


def main():
    parser = argparse.ArgumentParser(description='Preprocess E.T. for joint training')
    parser.add_argument('--et-root', type=str, default='/transfer/et-data')
    parser.add_argument('--output-root', type=str, default='/transfer/stc-data')
    parser.add_argument('--num-frames', type=int, default=48)
    parser.add_argument('--min-frames', type=int, default=10)
    parser.add_argument('--lookat-distance', type=float, default=3.0,
                        help='Default distance for look-at person proxy')
    parser.add_argument('--require-person', action='store_true',
                        help='Skip samples without real person (smplh) data')
    args = parser.parse_args()

    et_root = args.et_root
    output_root = args.output_root

    # Check dataset exists
    if not os.path.isdir(et_root):
        print(f"Error: E.T. dataset not found at {et_root}")
        print("Run: python scripts/download_et_data.py")
        return

    traj_dir = os.path.join(et_root, 'traj')
    if not os.path.isdir(traj_dir):
        print(f"Error: No traj/ directory in {et_root}")
        return

    caption_dir = os.path.join(et_root, 'caption')
    caption_cam_dir = os.path.join(et_root, 'caption_cam')

    # Look for person joint/character data (E.T. has smplh/ and char/)
    joints_dir = None
    for candidate in ['smplh', 'char', 'joints', 'body', 'smpl', 'character', 'motion']:
        d = os.path.join(et_root, candidate)
        if os.path.isdir(d):
            # Verify it has actual files
            entries = os.listdir(d)
            if len(entries) > 0:
                joints_dir = d
                print(f"Found person data directory: {d} ({len(entries)} entries)")
                break
    if joints_dir is None:
        print("No person joint directory found. Will estimate person position from camera look-at.")

    # Output directories
    cam_out_dir = os.path.join(output_root, 'camera_trajectories')
    person_out_dir = os.path.join(output_root, 'person_trajectories')
    os.makedirs(cam_out_dir, exist_ok=True)
    os.makedirs(person_out_dir, exist_ok=True)

    # Load splits
    splits = {}
    for split_name in ['train', 'test']:
        split_file = os.path.join(et_root, f'full_{split_name}_split.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                splits[split_name] = set(line.strip() for line in f if line.strip())
            print(f"Loaded {split_name} split: {len(splits[split_name])} samples")
        else:
            splits[split_name] = set()

    # Process
    all_sample_ids = sorted(f[:-4] for f in os.listdir(traj_dir) if f.endswith('.txt'))
    print(f"\nTotal trajectory files: {len(all_sample_ids)}")

    train_index = []
    test_index = []
    stats = {'total': 0, 'skipped': 0, 'has_joints': 0, 'lookat_proxy': 0}
    motion_counts = {}

    for sample_id in tqdm(all_sample_ids, desc="Processing"):
        traj_path = os.path.join(traj_dir, f'{sample_id}.txt')
        frames = load_camera_trajectory(traj_path)

        if len(frames) < args.min_frames:
            stats['skipped'] += 1
            continue

        # Camera trajectory (T, 6)
        camera_traj = np.stack([extrinsic_to_6d(R, t) for R, t in frames], axis=0)
        camera_traj = resample_trajectory(camera_traj, args.num_frames)

        # Skip camera outliers (extrinsic divergence)
        if not np.isfinite(camera_traj).all() or np.abs(camera_traj[:, :3]).max() > 100.0:
            stats['skipped'] += 1
            continue

        # Person trajectory (T, 5) = [px, py, pz, sin_yaw, cos_yaw]
        person_traj = None
        if joints_dir:
            person_traj = load_person_joints(joints_dir, sample_id)

        if person_traj is not None:
            # Skip if NaN/Inf or extreme outliers (SLAHMR divergence)
            if not np.isfinite(person_traj).all() or np.abs(person_traj).max() > 100.0:
                person_traj = None

        if person_traj is not None:
            person_traj = resample_trajectory(person_traj, args.num_frames)
            # Pad to (T, 5) if only (T, 3) was returned (no orientation)
            if person_traj.shape[1] == 3:
                zeros = np.zeros((person_traj.shape[0], 1), dtype=np.float32)
                ones = np.ones((person_traj.shape[0], 1), dtype=np.float32)
                person_traj = np.concatenate(
                    [person_traj, zeros, ones], axis=1)  # sin(0)=0, cos(0)=1
            stats['has_joints'] += 1
            has_real_person = True
        else:
            if args.require_person:
                stats['skipped'] += 1
                continue
            # Estimate from camera look-at (yaw=0 proxy)
            person_positions = np.stack(
                [extrinsic_to_lookat(R, t, args.lookat_distance) for R, t in frames],
                axis=0)
            person_positions = resample_trajectory(person_positions, args.num_frames)
            # Pad with sin(0)=0, cos(0)=1 for yaw → (T, 5)
            zeros = np.zeros((person_positions.shape[0], 1), dtype=np.float32)
            ones = np.ones((person_positions.shape[0], 1), dtype=np.float32)
            person_traj = np.concatenate(
                [person_positions, zeros, ones], axis=1)
            stats['lookat_proxy'] += 1
            has_real_person = False

        # Save
        np.save(os.path.join(cam_out_dir, f'{sample_id}.npy'), camera_traj)
        np.save(os.path.join(person_out_dir, f'{sample_id}.npy'), person_traj)

        # Caption
        caption_cam = ''
        caption_full = ''
        for path, target in [
            (os.path.join(caption_cam_dir, f'{sample_id}.txt'), 'cam'),
            (os.path.join(caption_dir, f'{sample_id}.txt'), 'full'),
        ]:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8', errors='replace') as f:
                        text = f.read().strip()
                    if target == 'cam':
                        caption_cam = text
                    else:
                        caption_full = text
                except Exception:
                    pass

        # Prefer full caption (describes both character + camera) for joint model
        text = caption_full if caption_full else caption_cam
        camera_motion = classify_camera_motion(text)
        shot_type = infer_shot_type(caption_full)
        motion_counts[camera_motion] = motion_counts.get(camera_motion, 0) + 1

        entry = {
            'id': sample_id,
            'text': text,
            'shot_type': shot_type,
            'camera_motion': camera_motion,
            'camera_trajectory_path': f'camera_trajectories/{sample_id}.npy',
            'person_trajectory_path': f'person_trajectories/{sample_id}.npy',
            'has_real_person': has_real_person,
        }

        if sample_id in splits.get('test', set()):
            test_index.append(entry)
        else:
            train_index.append(entry)

        stats['total'] += 1

    # Save index files
    for name, index in [('train_index.json', train_index), ('test_index.json', test_index)]:
        path = os.path.join(output_root, name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*60}")
    print("Preprocessing Complete!")
    print(f"{'='*60}")
    print(f"  Train: {len(train_index)}  |  Test: {len(test_index)}")
    print(f"  Skipped: {stats['skipped']}  |  Frames/traj: {args.num_frames}")
    print(f"  Person data: {stats['has_joints']} from joints, {stats['lookat_proxy']} from look-at proxy")
    print(f"\n  Camera motion distribution:")
    total = len(train_index) + len(test_index)
    for motion, count in sorted(motion_counts.items(), key=lambda x: -x[1]):
        print(f"    {motion:15s}: {count:6d} ({100*count/max(total,1):5.1f}%)")
    print(f"\n  Output: {output_root}")
    print(f"    camera_trajectories/ ({total} .npy)")
    print(f"    person_trajectories/ ({total} .npy)")
    print(f"    train_index.json, test_index.json")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
