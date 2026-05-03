"""
Prepare AMASS human motion data for joint person-camera trajectory training.

Downloads AMASS motion capture data (or uses a local copy), extracts root
translations, generates synthetic camera trajectories using cinematography
rules, and produces captions from templates.

Each person trajectory chunk is paired with EVERY camera motion type to
produce balanced augmentation data.

Outputs (same format as E.T. preprocessed data):
  - <output-root>/camera_trajectories/*.npy  (48, 6)
  - <output-root>/person_trajectories/*.npy  (48, 4)  [px, py, pz, yaw]
  - <output-root>/train_index.json

Usage:
    python scripts/prepare_amass.py --amass-root /transfer/amass --output-root /transfer/amass-stc-data
    python scripts/prepare_amass.py --output-root /transfer/amass-stc-data  # auto-download CMU subset
"""

import os
import re
import sys
import json
import argparse
import random
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Caption templates
# ---------------------------------------------------------------------------

CAPTION_TEMPLATES = {
    'static': [
        "The camera remains static while the character {action}.",
        "A static shot as the character {action}.",
        "The camera holds still as the character {action}.",
    ],
    'dolly-in': [
        "As the character {action}, the camera pushes in.",
        "The camera gradually closes in on the character who {action}.",
        "The camera dollies in while the character {action}.",
    ],
    'dolly-out': [
        "As the character {action}, the camera pulls out.",
        "The camera slowly pulls back as the character {action}.",
        "The camera dollies out while the character {action}.",
    ],
    'pan-left': [
        "The camera pans left as the character {action}.",
        "A pan to the left while the character {action}.",
    ],
    'pan-right': [
        "The camera pans right as the character {action}.",
        "A pan to the right while the character {action}.",
    ],
    'crane-up': [
        "The camera cranes up as the character {action}.",
        "The camera rises while the character {action}.",
    ],
    'crane-down': [
        "The camera cranes down as the character {action}.",
        "The camera lowers while the character {action}.",
    ],
    'track': [
        "The camera tracks the character as they {action}.",
        "As the character {action}, the camera trucks laterally to follow.",
        "A tracking shot follows the character who {action}.",
    ],
    'orbit': [
        "The camera orbits around the character as they {action}.",
        "An orbiting shot circles the character who {action}.",
    ],
}

ALL_MOTION_TYPES = list(CAPTION_TEMPLATES.keys())

SHOT_TYPES = ['close-up', 'medium-shot', 'wide-shot']
SHOT_DISTANCE = {
    'close-up': (1.5, 2.5),
    'medium-shot': (2.5, 4.0),
    'wide-shot': (4.0, 6.0),
}


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------

def axis_angle_to_rotation_matrix(aa: np.ndarray) -> np.ndarray:
    """Convert (3,) axis-angle to (3,3) rotation matrix via Rodrigues formula."""
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3)
    axis = aa / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rotation_matrix_to_yaw(R: np.ndarray) -> float:
    """Extract yaw angle from a 3x3 rotation matrix (Y-up convention)."""
    return float(np.arctan2(R[0, 2], R[2, 2]))


def extract_yaw_from_axis_angle(aa_sequence: np.ndarray) -> np.ndarray:
    """
    Convert (T, 3) axis-angle root orientations to (T,) yaw angles.
    """
    T = aa_sequence.shape[0]
    yaws = np.zeros(T, dtype=np.float32)
    for t in range(T):
        R = axis_angle_to_rotation_matrix(aa_sequence[t])
        yaws[t] = rotation_matrix_to_yaw(R)
    return yaws


# ---------------------------------------------------------------------------
# Action inference from person trajectory
# ---------------------------------------------------------------------------

def _direction_label(dx: float, dz: float, threshold: float = 0.15) -> str:
    """Convert XZ displacement into a direction word."""
    if abs(dx) < threshold and abs(dz) < threshold:
        return None   # no clear direction
    if abs(dx) >= abs(dz):
        return "to the right" if dx > 0 else "to the left"
    return "forward" if dz > 0 else "backward"


def infer_action(person_traj: np.ndarray) -> str:
    """
    Infer action from person trajectory with multi-phase awareness.

    Accepts (T, 3) or (T, 4) trajectories. If dim 4 is present, uses yaw
    to detect turning and enriches the description.

    Strategy:
    1. Check overall speed — if very slow → stationary.
    2. Split into thirds and get dominant XZ direction for each third.
    3. If all thirds agree → single-direction description.
    4. If two distinct directions appear → two-phase description.
    5. If three distinct directions → "moves in multiple directions".
    6. If yaw changes significantly (>45 deg), append turning info.
    """
    T = len(person_traj)

    # Overall per-frame speed in XZ plane
    vel_xz = np.diff(person_traj[:, [0, 2]], axis=0)      # (T-1, 2)
    frame_speeds = np.linalg.norm(vel_xz, axis=1)          # (T-1,)
    avg_speed = frame_speeds.mean()

    if avg_speed < 0.008:   # ~0.4 m/s × 1/48 s/frame
        base = random.choice(["stands still", "remains in place", "stays stationary"])
    else:
        # Split into three equal segments, compute dominant XZ displacement per segment
        seg = T // 3
        segments = [
            person_traj[0:seg],
            person_traj[seg:2*seg],
            person_traj[2*seg:],
        ]
        dirs = []
        for s in segments:
            d = s[-1] - s[0]
            label = _direction_label(d[0], d[2], threshold=0.10)
            dirs.append(label)

        # Filter out None (stationary segments)
        active_dirs = [d for d in dirs if d is not None]
        if not active_dirs:
            base = random.choice(["stands still", "remains in place"])
        else:
            unique_dirs = list(dict.fromkeys(active_dirs))   # preserve order, deduplicate
            if len(unique_dirs) == 1:
                base = f"moves {unique_dirs[0]}"
            elif len(unique_dirs) == 2:
                base = f"moves {unique_dirs[0]} then {unique_dirs[1]}"
            else:
                base = "moves in multiple directions"

    # Enrich with yaw information if available (dim 4)
    if person_traj.shape[1] >= 4:
        yaw = person_traj[:, 3]
        # Total yaw change (handle wraparound)
        yaw_diff = np.diff(yaw)
        # Wrap to [-pi, pi]
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
        total_yaw_change = np.sum(yaw_diff)
        if abs(total_yaw_change) > np.radians(45):
            if total_yaw_change > 0:
                base += " while turning left"
            else:
                base += " while turning right"

    return base


# ---------------------------------------------------------------------------
# Synthetic camera generation
# ---------------------------------------------------------------------------

def look_at_angles(cam_pos: np.ndarray, target: np.ndarray):
    """Compute azimuth and elevation from camera position looking at target."""
    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    dist_xz = np.sqrt(dx ** 2 + dz ** 2) + 1e-8
    azimuth = np.arctan2(dx, -dz)
    elevation = np.arctan2(dy, dist_xz)
    return azimuth, elevation


def generate_camera_for_person(person_traj: np.ndarray, motion_type: str,
                               num_frames: int = 48) -> tuple:
    """
    Generate a synthetic camera trajectory for a person trajectory.

    Args:
        person_traj: (T, 3) or (T, 4) person root positions (only xyz used).
        motion_type: one of ALL_MOTION_TYPES.
        num_frames: number of frames (should match person_traj.shape[0]).

    Returns:
        camera_traj: (T, 6) camera trajectory (tx, ty, tz, azimuth, elevation, roll).
        shot_type: inferred shot type string.
    """
    T = num_frames
    # Use only position dims (first 3) for camera generation
    if person_traj.shape[1] > 3:
        person_pos = person_traj[:, :3]
    else:
        person_pos = person_traj
    assert person_pos.shape == (T, 3), f"Expected ({T}, 3), got {person_pos.shape}"
    # Use person_pos (xyz only) throughout this function
    person_traj = person_pos

    centroid = person_traj.mean(axis=0)
    noise_sigma = 0.01

    # Pick a random shot type which determines base distance
    shot_type = random.choice(SHOT_TYPES)
    dist_lo, dist_hi = SHOT_DISTANCE[shot_type]
    base_distance = random.uniform(dist_lo, dist_hi)

    # Random initial angle around the person (in XZ plane)
    init_angle = random.uniform(0, 2 * np.pi)
    cam_height_offset = random.uniform(-0.3, 0.5)  # camera slightly above person

    camera_traj = np.zeros((T, 6), dtype=np.float32)

    if motion_type == 'static':
        # Fixed camera position, facing person centroid
        offset_dir = np.array([np.sin(init_angle), 0.0, np.cos(init_angle)])
        cam_pos = centroid + offset_dir * base_distance
        cam_pos[1] += cam_height_offset
        for t in range(T):
            az, el = look_at_angles(cam_pos, person_traj[t])
            camera_traj[t, :3] = cam_pos
            camera_traj[t, 3] = az
            camera_traj[t, 4] = el
            camera_traj[t, 5] = 0.0

    elif motion_type == 'dolly-in':
        start_dist = max(base_distance, 4.0) + random.uniform(0.5, 1.5)
        end_dist = max(1.5, base_distance - 1.5)
        offset_dir = np.array([np.sin(init_angle), 0.0, np.cos(init_angle)])
        for t in range(T):
            alpha = t / max(T - 1, 1)
            dist = start_dist + (end_dist - start_dist) * alpha
            cam_pos = person_traj[t] + offset_dir * dist
            cam_pos[1] += cam_height_offset
            az, el = look_at_angles(cam_pos, person_traj[t])
            camera_traj[t, :3] = cam_pos
            camera_traj[t, 3] = az
            camera_traj[t, 4] = el
            camera_traj[t, 5] = 0.0

    elif motion_type == 'dolly-out':
        start_dist = max(1.5, base_distance - 1.5)
        end_dist = max(base_distance, 4.0) + random.uniform(0.5, 1.5)
        offset_dir = np.array([np.sin(init_angle), 0.0, np.cos(init_angle)])
        for t in range(T):
            alpha = t / max(T - 1, 1)
            dist = start_dist + (end_dist - start_dist) * alpha
            cam_pos = person_traj[t] + offset_dir * dist
            cam_pos[1] += cam_height_offset
            az, el = look_at_angles(cam_pos, person_traj[t])
            camera_traj[t, :3] = cam_pos
            camera_traj[t, 3] = az
            camera_traj[t, 4] = el
            camera_traj[t, 5] = 0.0

    elif motion_type in ('pan-left', 'pan-right'):
        # Camera stays in place, azimuth rotates
        offset_dir = np.array([np.sin(init_angle), 0.0, np.cos(init_angle)])
        cam_pos = centroid + offset_dir * base_distance
        cam_pos[1] += cam_height_offset
        pan_range = random.uniform(np.radians(30), np.radians(60))
        if motion_type == 'pan-left':
            pan_range = -pan_range  # negative azimuth change = pan left
        base_az, base_el = look_at_angles(cam_pos, centroid)
        for t in range(T):
            alpha = t / max(T - 1, 1)
            az = base_az + pan_range * alpha
            camera_traj[t, :3] = cam_pos
            camera_traj[t, 3] = az
            camera_traj[t, 4] = base_el
            camera_traj[t, 5] = 0.0

    elif motion_type in ('crane-up', 'crane-down'):
        # Camera stays in XZ place, elevation changes
        offset_dir = np.array([np.sin(init_angle), 0.0, np.cos(init_angle)])
        cam_pos = centroid + offset_dir * base_distance
        cam_pos[1] += cam_height_offset
        el_range = random.uniform(np.radians(20), np.radians(40))
        if motion_type == 'crane-down':
            el_range = -el_range
        base_az, base_el = look_at_angles(cam_pos, centroid)
        for t in range(T):
            alpha = t / max(T - 1, 1)
            # Move camera vertically to change elevation
            height_delta = np.tan(el_range * alpha) * base_distance
            cur_pos = cam_pos.copy()
            cur_pos[1] += height_delta
            az, el = look_at_angles(cur_pos, person_traj[t])
            camera_traj[t, :3] = cur_pos
            camera_traj[t, 3] = az
            camera_traj[t, 4] = el
            camera_traj[t, 5] = 0.0

    elif motion_type == 'track':
        # Camera follows person laterally, maintaining constant distance
        offset_dir = np.array([np.sin(init_angle), 0.0, np.cos(init_angle)])
        for t in range(T):
            cam_pos = person_traj[t] + offset_dir * base_distance
            cam_pos[1] += cam_height_offset
            az, el = look_at_angles(cam_pos, person_traj[t])
            camera_traj[t, :3] = cam_pos
            camera_traj[t, 3] = az
            camera_traj[t, 4] = el
            camera_traj[t, 5] = 0.0

    elif motion_type == 'orbit':
        # Camera orbits around person centroid
        orbit_range = random.uniform(np.radians(60), np.radians(180))
        orbit_dir = random.choice([-1, 1])
        for t in range(T):
            alpha = t / max(T - 1, 1)
            angle = init_angle + orbit_dir * orbit_range * alpha
            offset = np.array([np.sin(angle), 0.0, np.cos(angle)])
            cam_pos = centroid + offset * base_distance
            cam_pos[1] += cam_height_offset
            az, el = look_at_angles(cam_pos, person_traj[t])
            camera_traj[t, :3] = cam_pos
            camera_traj[t, 3] = az
            camera_traj[t, 4] = el
            camera_traj[t, 5] = 0.0

    else:
        raise ValueError(f"Unknown motion type: {motion_type}")

    # Add small Gaussian noise for realism
    camera_traj += np.random.normal(0, noise_sigma, camera_traj.shape).astype(np.float32)

    return camera_traj, shot_type


# ---------------------------------------------------------------------------
# Trajectory utilities
# ---------------------------------------------------------------------------

def resample_trajectory(trajectory: np.ndarray, target_frames: int) -> np.ndarray:
    """Linearly resample a trajectory to target_frames."""
    src_frames = trajectory.shape[0]
    if src_frames == target_frames:
        return trajectory.astype(np.float32)
    dim = trajectory.shape[1]
    src_t = np.linspace(0, 1, src_frames)
    tgt_t = np.linspace(0, 1, target_frames)
    resampled = np.zeros((target_frames, dim), dtype=np.float32)
    for d in range(dim):
        resampled[:, d] = np.interp(tgt_t, src_t, trajectory[:, d])
    return resampled


def is_valid_trajectory(traj: np.ndarray, max_val: float = 100.0) -> bool:
    """Check trajectory for NaN/Inf and extreme outliers."""
    if not np.isfinite(traj).all():
        return False
    if np.abs(traj).max() > max_val:
        return False
    return True


# ---------------------------------------------------------------------------
# AMASS data loading
# ---------------------------------------------------------------------------

def find_npz_files(root_dir: str) -> list:
    """Recursively find all .npz files under root_dir."""
    npz_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('.npz'):
                npz_files.append(os.path.join(dirpath, fname))
    return sorted(npz_files)


def extract_chunks(trans: np.ndarray, source_fps: float, target_fps: int = 24,
                   chunk_frames: int = 48, root_orient: np.ndarray = None) -> list:
    """
    Extract fixed-length trajectory chunks from a long motion sequence.

    Downsamples from source_fps to target_fps, then splits into chunks
    of chunk_frames frames.

    Args:
        trans: (T_source, 3) root translation at source_fps.
        source_fps: frames per second of the source data.
        target_fps: target fps (24 for our model).
        chunk_frames: number of frames per chunk (48).
        root_orient: (T_source, 3) axis-angle root orientation, or None.

    Returns:
        List of (chunk_frames, 4) numpy arrays [px, py, pz, yaw].
    """
    T_src = trans.shape[0]
    duration_sec = T_src / source_fps
    T_target = int(duration_sec * target_fps)

    if T_target < chunk_frames:
        return []

    # Extract yaw from root orientation if available
    if root_orient is not None and root_orient.shape[0] == T_src:
        yaws = extract_yaw_from_axis_angle(root_orient)  # (T_src,)
        # Combine trans + yaw → (T_src, 4)
        combined = np.concatenate([trans, yaws.reshape(-1, 1)], axis=1)
    else:
        # No orientation: pad with zeros for yaw
        combined = np.concatenate(
            [trans, np.zeros((T_src, 1), dtype=np.float32)], axis=1)

    # Resample to target_fps (all 4 dims)
    resampled = resample_trajectory(combined, T_target)

    # Split into non-overlapping chunks
    chunks = []
    for start in range(0, T_target - chunk_frames + 1, chunk_frames):
        chunk = resampled[start:start + chunk_frames]
        if is_valid_trajectory(chunk):
            chunks.append(chunk)

    return chunks


def download_amass_subset(output_dir: str, subset: str = 'CMU') -> bool:
    """
    Attempt to download an AMASS sub-dataset from HuggingFace.

    Returns True on success, False on failure.
    """
    print(f"\nAttempting to download AMASS {subset} subset from HuggingFace...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    # AMASS subsets on HuggingFace (community mirrors)
    hf_repos = {
        'CMU': 'smplx/amass-cmu',
        'BMLrub': 'smplx/amass-bmlrub',
        'HDM05': 'smplx/amass-hdm05',
    }

    # Try the specific subset first
    repo_id = hf_repos.get(subset)
    if repo_id:
        try:
            print(f"Trying HuggingFace repo: {repo_id}")
            subset_dir = os.path.join(output_dir, subset)
            os.makedirs(subset_dir, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                repo_type='dataset',
                local_dir=subset_dir,
                local_dir_use_symlinks=False,
            )
            print(f"Downloaded {subset} to {subset_dir}")
            return True
        except Exception as e:
            print(f"HuggingFace download failed for {repo_id}: {e}")

    # Try the full AMASS dataset repo
    try:
        print("Trying HuggingFace repo: smplx/amass")
        os.makedirs(output_dir, exist_ok=True)
        snapshot_download(
            repo_id='smplx/amass',
            repo_type='dataset',
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            allow_patterns=[f"*{subset}*/**/*.npz", f"*{subset}*/*.npz"],
        )
        print(f"Downloaded AMASS ({subset} subset) to {output_dir}")
        return True
    except Exception as e:
        print(f"Full AMASS download also failed: {e}")

    return False


def print_manual_download_instructions():
    """Print instructions for manually downloading AMASS."""
    print("\n" + "=" * 60)
    print("AMASS MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("""
1. Visit https://amass.is.tue.mpg.de/ and create an account.
2. Download one or more sub-datasets (e.g., CMU, BMLrub, HDM05).
3. Extract the downloaded archives into a single directory, e.g.:
       /transfer/amass/
       +-- CMU/
       |   +-- 01/
       |   |   +-- 01_01_stageii.npz
       |   |   +-- ...
       +-- BMLrub/
       |   +-- ...

4. Re-run this script with:
       python scripts/prepare_amass.py --amass-root /transfer/amass --output-root /transfer/amass-stc-data

Alternative: Some AMASS sub-datasets are mirrored on HuggingFace:
    pip install huggingface_hub
    python -c "from huggingface_hub import snapshot_download; snapshot_download('smplx/amass', repo_type='dataset', local_dir='/transfer/amass')"
""")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Prepare AMASS data for joint person-camera trajectory training.'
    )
    parser.add_argument('--amass-root', type=str, default='/transfer/amass',
                        help='Path to local AMASS dataset directory.')
    parser.add_argument('--output-root', type=str, default='/transfer/amass-stc-data',
                        help='Output directory for processed data.')
    parser.add_argument('--num-frames', type=int, default=48,
                        help='Number of frames per trajectory chunk.')
    parser.add_argument('--target-fps', type=int, default=24,
                        help='Target frames per second.')
    parser.add_argument('--default-source-fps', type=float, default=120.0,
                        help='Default source FPS if not specified in .npz metadata.')
    parser.add_argument('--max-chunks-per-sequence', type=int, default=50,
                        help='Maximum chunks to extract from a single long sequence.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--download-subset', type=str, default='CMU',
                        help='AMASS subset to auto-download if amass-root is missing.')
    parser.add_argument('--humanml3d-texts', type=str, default=None,
                        help='Path to humanml3d_captions.json (from download_humanml3d_texts.py). '
                             'If provided, uses real captions instead of synthetic templates.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load HumanML3D captions if provided
    humanml3d_captions = {}
    if args.humanml3d_texts and os.path.exists(args.humanml3d_texts):
        with open(args.humanml3d_texts, 'r', encoding='utf-8') as f:
            humanml3d_captions = json.load(f)
        print(f"Loaded HumanML3D captions: {len(humanml3d_captions)} AMASS sequences")

    amass_root = args.amass_root

    # Check if AMASS data exists locally
    if not os.path.isdir(amass_root):
        print(f"AMASS directory not found at: {amass_root}")
        print("Attempting automatic download...")
        os.makedirs(amass_root, exist_ok=True)
        success = download_amass_subset(amass_root, subset=args.download_subset)
        if not success:
            print_manual_download_instructions()
            sys.exit(1)

    # Find all .npz files
    print(f"\nScanning for .npz files in {amass_root} ...")
    npz_files = find_npz_files(amass_root)
    if len(npz_files) == 0:
        print(f"No .npz files found in {amass_root}")
        print_manual_download_instructions()
        sys.exit(1)

    print(f"Found {len(npz_files)} .npz files")

    # Create output directories
    cam_out_dir = os.path.join(args.output_root, 'camera_trajectories')
    person_out_dir = os.path.join(args.output_root, 'person_trajectories')
    os.makedirs(cam_out_dir, exist_ok=True)
    os.makedirs(person_out_dir, exist_ok=True)

    train_index = []
    stats = {
        'files_processed': 0,
        'files_skipped': 0,
        'chunks_total': 0,
        'samples_total': 0,
    }
    motion_counts = {mt: 0 for mt in ALL_MOTION_TYPES}
    shot_counts = {st: 0 for st in SHOT_TYPES}

    sample_counter = 0

    for npz_path in tqdm(npz_files, desc="Processing AMASS files"):
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            stats['files_skipped'] += 1
            continue

        # Extract root translation
        trans = None
        for key in ['trans', 'transl', 'root_translation']:
            if key in data:
                trans = data[key]
                break

        if trans is None:
            stats['files_skipped'] += 1
            continue

        trans = np.array(trans, dtype=np.float32)
        if trans.ndim != 2 or trans.shape[1] < 3:
            stats['files_skipped'] += 1
            continue
        trans = trans[:, :3]

        # Extract root orientation (axis-angle) if available
        root_orient = None
        if 'root_orient' in data:
            ro = np.array(data['root_orient'], dtype=np.float32)
            if ro.ndim == 2 and ro.shape[1] >= 3 and ro.shape[0] == trans.shape[0]:
                root_orient = ro[:, :3]
        elif 'poses' in data:
            poses = np.array(data['poses'], dtype=np.float32)
            if poses.ndim == 2 and poses.shape[1] >= 3 and poses.shape[0] == trans.shape[0]:
                root_orient = poses[:, :3]

        # Determine source FPS
        source_fps = args.default_source_fps
        if 'mocap_framerate' in data:
            fps_val = float(data['mocap_framerate'])
            if fps_val > 0:
                source_fps = fps_val
        elif 'frame_rate' in data:
            fps_val = float(data['frame_rate'])
            if fps_val > 0:
                source_fps = fps_val

        # Extract chunks (now returns (chunk_frames, 4) with yaw)
        chunks = extract_chunks(trans, source_fps, args.target_fps, args.num_frames,
                                root_orient=root_orient)
        if not chunks:
            stats['files_skipped'] += 1
            continue

        stats['files_processed'] += 1

        # Limit chunks per sequence
        if len(chunks) > args.max_chunks_per_sequence:
            chunks = random.sample(chunks, args.max_chunks_per_sequence)

        stats['chunks_total'] += len(chunks)

        # Derive a base name from the file path
        rel_path = os.path.relpath(npz_path, amass_root)
        base_name = rel_path.replace(os.sep, '_').replace('/', '_').replace('.npz', '')

        # Check for HumanML3D captions for this AMASS sequence
        # Try matching against the relative path (without extension)
        amass_key = rel_path.replace(os.sep, '/').replace('.npz', '')
        # Also try without _stageii or _poses suffix
        amass_key_variants = [
            amass_key,
            re.sub(r'_stageii$', '', amass_key),
            re.sub(r'_poses$', '', amass_key),
        ]
        real_captions = []
        for variant in amass_key_variants:
            if variant in humanml3d_captions:
                real_captions = humanml3d_captions[variant]
                break

        for chunk_idx, person_chunk in enumerate(chunks):
            action = infer_action(person_chunk)

            # Generate one camera variant for EACH motion type (balanced data)
            for motion_type in ALL_MOTION_TYPES:
                sample_id = f"amass_{base_name}_c{chunk_idx:04d}_{motion_type}"

                camera_traj, shot_type = generate_camera_for_person(
                    person_chunk, motion_type, args.num_frames
                )

                # Validate
                if not is_valid_trajectory(camera_traj):
                    continue

                # Save .npy files
                cam_path = os.path.join(cam_out_dir, f'{sample_id}.npy')
                person_path = os.path.join(person_out_dir, f'{sample_id}.npy')
                np.save(cam_path, camera_traj)
                np.save(person_path, person_chunk)

                # Generate caption: prefer real HumanML3D captions, fall back to templates
                shot_prefix = f"{shot_type.replace('-', ' ').title()}. " if shot_type != 'medium-shot' else ""
                if real_captions:
                    person_caption = random.choice(real_captions)
                    cam_desc = f"The camera {motion_type.replace('-', ' ')}s."
                    caption = f"{shot_prefix}{person_caption} {cam_desc}"
                else:
                    template = random.choice(CAPTION_TEMPLATES[motion_type])
                    caption = f"{shot_prefix}{template.format(action=action)}"

                entry = {
                    'id': sample_id,
                    'text': caption,
                    'shot_type': shot_type,
                    'camera_motion': motion_type,
                    'camera_trajectory_path': f'camera_trajectories/{sample_id}.npy',
                    'person_trajectory_path': f'person_trajectories/{sample_id}.npy',
                    'has_real_person': True,
                    'source': 'amass',
                }
                train_index.append(entry)

                motion_counts[motion_type] += 1
                shot_counts[shot_type] += 1
                sample_counter += 1

    stats['samples_total'] = sample_counter

    # Save train index
    index_path = os.path.join(args.output_root, 'train_index.json')
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(train_index, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'=' * 60}")
    print("AMASS Preparation Complete!")
    print(f"{'=' * 60}")
    print(f"  Files processed: {stats['files_processed']}")
    print(f"  Files skipped:   {stats['files_skipped']}")
    print(f"  Chunks extracted: {stats['chunks_total']}")
    print(f"  Total samples:   {stats['samples_total']}")
    print(f"\n  Camera motion distribution:")
    for motion, count in sorted(motion_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(stats['samples_total'], 1)
        print(f"    {motion:15s}: {count:6d} ({pct:5.1f}%)")
    print(f"\n  Shot type distribution:")
    for st, count in sorted(shot_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(stats['samples_total'], 1)
        print(f"    {st:20s}: {count:6d} ({pct:5.1f}%)")
    print(f"\n  Output: {args.output_root}")
    print(f"    camera_trajectories/ ({stats['samples_total']} .npy)")
    print(f"    person_trajectories/ ({stats['samples_total']} .npy)")
    print(f"    train_index.json ({len(train_index)} entries)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
