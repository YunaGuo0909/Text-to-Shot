"""
Prepare DanceCamera3D dataset for joint person-camera trajectory training.

Downloads (or uses a local copy of) the DanceCamera3D dataset, extracts
paired dance + camera trajectories, converts to the project's 6D camera
format, and generates captions from templates.

Outputs (same format as E.T. preprocessed data):
  - <output-root>/camera_trajectories/*.npy  (48, 6)
  - <output-root>/person_trajectories/*.npy  (48, 3)
  - <output-root>/train_index.json

Usage:
    python scripts/data/prepare_dancecamera3d.py --data-root /transfer/dancecamera3d --output-root /transfer/dance-stc-data
"""

import os
import sys
import json
import argparse
import random
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Caption templates for dance data
# ---------------------------------------------------------------------------

DANCE_CAPTION_TEMPLATES = {
    'static': [
        "The camera remains static as the character dances.",
        "A fixed shot captures the character dancing.",
    ],
    'dolly-in': [
        "The camera pushes in as the character dances.",
        "The camera closes in on the dancing character.",
    ],
    'dolly-out': [
        "The camera pulls back as the character dances.",
        "The camera slowly pulls out while the character dances.",
    ],
    'pan-left': [
        "The camera pans left as the character dances.",
    ],
    'pan-right': [
        "The camera pans right as the character dances.",
    ],
    'crane-up': [
        "The camera cranes up as the character dances.",
        "The camera rises while the character dances.",
    ],
    'crane-down': [
        "The camera cranes down as the character dances.",
        "The camera lowers while the character dances.",
    ],
    'track': [
        "The camera tracks the character as they dance.",
        "A tracking shot follows the dancing character.",
    ],
    'orbit': [
        "The camera orbits around the character as they dance.",
        "An orbiting shot circles the dancing character.",
    ],
}


# ---------------------------------------------------------------------------
# Camera motion classification from trajectory
# ---------------------------------------------------------------------------

def classify_camera_motion_from_trajectory(cam_traj: np.ndarray) -> str:
    """
    Infer camera motion type from a (T, 6) camera trajectory.

    cam_traj columns: tx, ty, tz, azimuth, elevation, roll
    """
    T = cam_traj.shape[0]
    if T < 2:
        return 'static'

    pos = cam_traj[:, :3]  # (T, 3)
    az = cam_traj[:, 3]    # (T,)
    el = cam_traj[:, 4]    # (T,)

    # Position displacement
    pos_disp = np.linalg.norm(pos[-1] - pos[0])
    # Azimuth change
    az_change = az[-1] - az[0]
    # Elevation change
    el_change = el[-1] - el[0]

    # Height change
    height_change = pos[-1, 1] - pos[0, 1]

    # Distance from centroid of person trajectory is not available here,
    # so use heuristics on camera motion alone
    if pos_disp < 0.3 and abs(az_change) < np.radians(10) and abs(el_change) < np.radians(10):
        return 'static'
    elif abs(az_change) > np.radians(50):
        # Large azimuth change with position change -> orbit, without -> pan
        if pos_disp > 1.0:
            return 'orbit'
        elif az_change > 0:
            return 'pan-right'
        else:
            return 'pan-left'
    elif abs(el_change) > np.radians(15):
        if el_change > 0:
            return 'crane-up'
        else:
            return 'crane-down'
    elif pos_disp > 0.5:
        # Check if camera moves toward/away or laterally
        forward_dir = pos[-1] - pos[0]
        forward_dir_norm = forward_dir / (np.linalg.norm(forward_dir) + 1e-8)
        # Use azimuth to estimate facing direction
        mean_az = az.mean()
        facing = np.array([np.sin(mean_az), 0.0, -np.cos(mean_az)])
        dot = np.dot(forward_dir_norm, facing)
        if dot > 0.5:
            return 'dolly-in'
        elif dot < -0.5:
            return 'dolly-out'
        else:
            return 'track'
    else:
        return 'static'


def infer_shot_type_from_distance(cam_traj: np.ndarray, person_traj: np.ndarray) -> str:
    """Infer shot type from average camera-to-person distance."""
    cam_pos = cam_traj[:, :3]
    avg_dist = np.linalg.norm(cam_pos - person_traj, axis=1).mean()
    if avg_dist < 2.0:
        return 'close-up'
    elif avg_dist < 4.0:
        return 'medium-shot'
    else:
        return 'wide-shot'


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
# Rotation conversion utilities
# ---------------------------------------------------------------------------

def rotation_matrix_to_euler(R: np.ndarray) -> tuple:
    """
    Convert 3x3 rotation matrix to (azimuth, elevation, roll) in radians.
    Uses the same convention as preprocess_et_data.py.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        elevation = np.arctan2(-R[2, 0], sy)
        azimuth = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        elevation = np.arctan2(-R[2, 0], sy)
        azimuth = np.arctan2(-R[1, 2], R[1, 1])
        roll = 0.0
    return azimuth, elevation, roll


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


def euler_xyz_to_rotation_matrix(angles: np.ndarray) -> np.ndarray:
    """Convert Euler angles (rx, ry, rz) in radians to 3x3 rotation matrix."""
    rx, ry, rz = angles[0], angles[1], angles[2]
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def look_at_angles(cam_pos: np.ndarray, target: np.ndarray) -> tuple:
    """Compute azimuth and elevation from camera position looking at target."""
    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    dist_xz = np.sqrt(dx ** 2 + dz ** 2) + 1e-8
    azimuth = np.arctan2(dx, -dz)
    elevation = np.arctan2(dy, dist_xz)
    return azimuth, elevation


# ---------------------------------------------------------------------------
# DanceCamera3D data loading
# ---------------------------------------------------------------------------

def try_download_dancecamera3d(data_root: str) -> bool:
    """
    Attempt to download the DanceCamera3D dataset.
    Returns True on success, False on failure.
    """
    print("\nAttempting to download DanceCamera3D dataset...")

    # Try git clone
    repo_url = "https://github.com/Carmenw1203/DanceCamera3D-Dataset.git"
    try:
        print(f"Cloning {repo_url} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, data_root],
            check=True, capture_output=True, text=True, timeout=300,
        )
        print(f"Cloned to {data_root}")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Git clone failed: {e}")

    # Try HuggingFace
    try:
        from huggingface_hub import snapshot_download
        print("Trying HuggingFace mirror...")
        snapshot_download(
            repo_id='Carmenw1203/DanceCamera3D',
            repo_type='dataset',
            local_dir=data_root,
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded to {data_root}")
        return True
    except Exception as e:
        print(f"HuggingFace download failed: {e}")

    return False


def print_manual_download_instructions():
    """Print instructions for manually downloading DanceCamera3D."""
    print("\n" + "=" * 60)
    print("DanceCamera3D MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("""
1. Visit https://github.com/Carmenw1203/DanceCamera3D-Dataset
2. Follow the download instructions in the repository README.
3. The dataset may require downloading from Google Drive links
   provided in the repository.
4. Extract the data into a directory, e.g.:
       /transfer/dancecamera3d/

5. Re-run this script with:
       python scripts/data/prepare_dancecamera3d.py \\
           --data-root /transfer/dancecamera3d \\
           --output-root /transfer/dance-stc-data

Expected directory structure (flexible -- the script will search recursively):
    /transfer/dancecamera3d/
    +-- <any subdirectory structure>/
        +-- *.json, *.pkl, *.npy, or *.npz files
            containing camera and motion data
""")
    print("=" * 60)


def load_json_data(json_path: str) -> dict:
    """Load a JSON file, handling common encoding issues."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(json_path, 'r', encoding='latin-1') as f:
            return json.load(f)


def find_data_files(data_root: str) -> dict:
    """
    Search for DanceCamera3D data files under data_root.

    Returns a dict with lists of found file paths, keyed by type.
    DanceCamera3D stores data in various formats; we try to support all.
    """
    result = {
        'json': [],
        'pkl': [],
        'npy': [],
        'npz': [],
        'csv': [],
    }
    for dirpath, dirnames, filenames in os.walk(data_root):
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext == '.json' and 'readme' not in fname.lower():
                result['json'].append(full_path)
            elif ext == '.pkl':
                result['pkl'].append(full_path)
            elif ext == '.npy':
                result['npy'].append(full_path)
            elif ext == '.npz':
                result['npz'].append(full_path)
            elif ext == '.csv':
                result['csv'].append(full_path)
    return result


def extract_camera_6d_from_pos_rot(cam_pos: np.ndarray, cam_rot: np.ndarray) -> np.ndarray:
    """
    Convert camera position + rotation to our 6D format.

    Args:
        cam_pos: (T, 3) camera positions.
        cam_rot: (T, 3) Euler angles, (T, 4) quaternions, or (T, 3, 3) rotation matrices.

    Returns:
        (T, 6) camera trajectory: tx, ty, tz, azimuth, elevation, roll.
    """
    T = cam_pos.shape[0]
    cam_6d = np.zeros((T, 6), dtype=np.float32)
    cam_6d[:, :3] = cam_pos

    if cam_rot.ndim == 2 and cam_rot.shape[1] == 3:
        # Euler angles -- convert via rotation matrix
        for t in range(T):
            R = euler_xyz_to_rotation_matrix(cam_rot[t])
            az, el, roll = rotation_matrix_to_euler(R)
            cam_6d[t, 3] = az
            cam_6d[t, 4] = el
            cam_6d[t, 5] = roll
    elif cam_rot.ndim == 2 and cam_rot.shape[1] == 4:
        # Quaternions (w, x, y, z)
        for t in range(T):
            R = quaternion_to_rotation_matrix(cam_rot[t])
            az, el, roll = rotation_matrix_to_euler(R)
            cam_6d[t, 3] = az
            cam_6d[t, 4] = el
            cam_6d[t, 5] = roll
    elif cam_rot.ndim == 3 and cam_rot.shape[1:] == (3, 3):
        # Rotation matrices
        for t in range(T):
            az, el, roll = rotation_matrix_to_euler(cam_rot[t])
            cam_6d[t, 3] = az
            cam_6d[t, 4] = el
            cam_6d[t, 5] = roll
    else:
        # Fallback: treat as raw angles
        cols = min(cam_rot.shape[1], 3)
        cam_6d[:, 3:3 + cols] = cam_rot[:, :cols]

    return cam_6d


def process_json_file(json_path: str, num_frames: int) -> list:
    """
    Try to extract (camera_traj, person_traj) pairs from a DanceCamera3D JSON file.

    Returns list of (camera_6d, person_xyz, source_name) tuples.
    """
    results = []
    try:
        data = load_json_data(json_path)
    except Exception:
        return results

    if isinstance(data, dict):
        # Look for camera and motion keys
        cam_data = None
        person_data = None

        # Camera keys
        for key in ['camera', 'cam', 'camera_position', 'cam_pos',
                     'camera_trajectory', 'cam_traj']:
            if key in data:
                cam_data = data[key]
                break

        # Person/motion keys
        for key in ['motion', 'dancer', 'person', 'body', 'joints',
                     'root_translation', 'trans', 'transl', 'position',
                     'dance', 'smpl_trans']:
            if key in data:
                person_data = data[key]
                break

        if cam_data is not None and person_data is not None:
            cam_arr = np.array(cam_data, dtype=np.float32)
            person_arr = np.array(person_data, dtype=np.float32)

            # Extract camera position and rotation
            if cam_arr.ndim == 2:
                if cam_arr.shape[1] >= 6:
                    # Assume pos(3) + rot(3+)
                    cam_pos = cam_arr[:, :3]
                    cam_rot = cam_arr[:, 3:6]
                    cam_6d = extract_camera_6d_from_pos_rot(cam_pos, cam_rot)
                elif cam_arr.shape[1] == 3:
                    # Position only, compute look-at from person
                    cam_6d = np.zeros((cam_arr.shape[0], 6), dtype=np.float32)
                    cam_6d[:, :3] = cam_arr
                else:
                    return results
            else:
                return results

            # Extract person root position
            if person_arr.ndim == 2 and person_arr.shape[1] >= 3:
                person_xyz = person_arr[:, :3]
            elif person_arr.ndim == 3:
                # (T, J, 3) -- take root joint (index 0)
                person_xyz = person_arr[:, 0, :3]
            else:
                return results

            # If camera is position-only, fill in look-at angles
            if cam_arr.ndim == 2 and cam_arr.shape[1] == 3:
                for t in range(cam_6d.shape[0]):
                    az, el = look_at_angles(cam_6d[t, :3], person_xyz[t])
                    cam_6d[t, 3] = az
                    cam_6d[t, 4] = el

            # Align lengths
            T = min(cam_6d.shape[0], person_xyz.shape[0])
            cam_6d = cam_6d[:T]
            person_xyz = person_xyz[:T].astype(np.float32)

            # Split into chunks of num_frames
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            for start in range(0, T - num_frames + 1, num_frames):
                cam_chunk = cam_6d[start:start + num_frames]
                person_chunk = person_xyz[start:start + num_frames]
                if is_valid_trajectory(cam_chunk) and is_valid_trajectory(person_chunk):
                    chunk_name = f"{base_name}_f{start:05d}"
                    results.append((cam_chunk, person_chunk, chunk_name))

    elif isinstance(data, list):
        # List of frames or list of sequences
        for i, item in enumerate(data):
            if isinstance(item, dict):
                # Try to recursively parse each item
                tmp_path = json_path + f"_item{i}"
                # Write a temporary approach: try extracting from the dict
                cam_data = None
                person_data = None
                for key in ['camera', 'cam', 'camera_position', 'cam_pos']:
                    if key in item:
                        cam_data = item[key]
                        break
                for key in ['motion', 'dancer', 'person', 'trans', 'transl']:
                    if key in item:
                        person_data = item[key]
                        break
                if cam_data is not None and person_data is not None:
                    cam_arr = np.array(cam_data, dtype=np.float32)
                    person_arr = np.array(person_data, dtype=np.float32)
                    if cam_arr.ndim == 2 and person_arr.ndim >= 2:
                        # Use same logic as above (simplified)
                        if cam_arr.shape[1] >= 6:
                            cam_6d = extract_camera_6d_from_pos_rot(
                                cam_arr[:, :3], cam_arr[:, 3:6])
                        elif cam_arr.shape[1] == 3:
                            cam_6d = np.zeros((cam_arr.shape[0], 6), dtype=np.float32)
                            cam_6d[:, :3] = cam_arr
                        else:
                            continue
                        if person_arr.ndim == 3:
                            person_xyz = person_arr[:, 0, :3]
                        else:
                            person_xyz = person_arr[:, :3]
                        if cam_arr.shape[1] == 3:
                            for t in range(min(cam_6d.shape[0], person_xyz.shape[0])):
                                az, el = look_at_angles(cam_6d[t, :3], person_xyz[t])
                                cam_6d[t, 3] = az
                                cam_6d[t, 4] = el
                        T = min(cam_6d.shape[0], person_xyz.shape[0])
                        cam_6d = cam_6d[:T]
                        person_xyz = person_xyz[:T].astype(np.float32)
                        base_name = os.path.splitext(os.path.basename(json_path))[0]
                        for start in range(0, T - num_frames + 1, num_frames):
                            cam_chunk = cam_6d[start:start + num_frames]
                            person_chunk = person_xyz[start:start + num_frames]
                            if is_valid_trajectory(cam_chunk) and is_valid_trajectory(person_chunk):
                                chunk_name = f"{base_name}_i{i}_f{start:05d}"
                                results.append((cam_chunk, person_chunk, chunk_name))

    return results


def process_pkl_file(pkl_path: str, num_frames: int) -> list:
    """
    Try to extract (camera_traj, person_traj) pairs from a .pkl file.

    Returns list of (camera_6d, person_xyz, source_name) tuples.
    """
    import pickle
    results = []
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception:
        return results

    if not isinstance(data, dict):
        return results

    # Look for camera data
    cam_pos = None
    cam_rot = None
    person_xyz = None

    for key in ['camera_position', 'cam_pos', 'camera', 'cam_translation']:
        if key in data:
            arr = np.array(data[key], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                cam_pos = arr[:, :3]
                break

    for key in ['camera_rotation', 'cam_rot', 'cam_euler', 'camera_orientation']:
        if key in data:
            arr = np.array(data[key], dtype=np.float32)
            if arr.ndim >= 2:
                cam_rot = arr
                break

    for key in ['transl', 'trans', 'root_translation', 'dancer_position',
                'motion_translation', 'smpl_trans', 'body_trans']:
        if key in data:
            arr = np.array(data[key], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                person_xyz = arr[:, :3]
                break

    # If we have a combined camera array
    if cam_pos is None:
        for key in ['camera', 'cam', 'camera_trajectory']:
            if key in data:
                arr = np.array(data[key], dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 6:
                    cam_pos = arr[:, :3]
                    cam_rot = arr[:, 3:6]
                    break
                elif arr.ndim == 2 and arr.shape[1] == 3:
                    cam_pos = arr
                    break

    if cam_pos is None or person_xyz is None:
        return results

    # Build camera 6D
    if cam_rot is not None:
        cam_6d = extract_camera_6d_from_pos_rot(cam_pos, cam_rot)
    else:
        cam_6d = np.zeros((cam_pos.shape[0], 6), dtype=np.float32)
        cam_6d[:, :3] = cam_pos
        for t in range(min(cam_6d.shape[0], person_xyz.shape[0])):
            az, el = look_at_angles(cam_6d[t, :3], person_xyz[t])
            cam_6d[t, 3] = az
            cam_6d[t, 4] = el

    # Align lengths and chunk
    T = min(cam_6d.shape[0], person_xyz.shape[0])
    cam_6d = cam_6d[:T]
    person_xyz = person_xyz[:T].astype(np.float32)

    base_name = os.path.splitext(os.path.basename(pkl_path))[0]
    for start in range(0, T - num_frames + 1, num_frames):
        cam_chunk = cam_6d[start:start + num_frames]
        person_chunk = person_xyz[start:start + num_frames]
        if is_valid_trajectory(cam_chunk) and is_valid_trajectory(person_chunk):
            chunk_name = f"{base_name}_f{start:05d}"
            results.append((cam_chunk, person_chunk, chunk_name))

    return results


def process_npy_npz_pair(data_root: str, npy_files: list, npz_files: list,
                         num_frames: int) -> list:
    """
    Try to find paired camera/person .npy or .npz files.

    DanceCamera3D may store camera and motion in separate files with
    matching names or in .npz archives.
    """
    results = []

    # Group .npy files by directory
    npy_by_dir = {}
    for path in npy_files:
        dirpath = os.path.dirname(path)
        fname = os.path.basename(path).lower()
        if dirpath not in npy_by_dir:
            npy_by_dir[dirpath] = {}
        npy_by_dir[dirpath][fname] = path

    # Look for camera/person pairs in the same directory
    for dirpath, files in npy_by_dir.items():
        cam_file = None
        person_file = None
        for fname, full_path in files.items():
            if any(k in fname for k in ['camera', 'cam_pos', 'cam_traj']):
                cam_file = full_path
            elif any(k in fname for k in ['motion', 'person', 'dancer', 'trans',
                                           'body', 'root', 'smpl']):
                person_file = full_path

        if cam_file and person_file:
            try:
                cam_arr = np.load(cam_file).astype(np.float32)
                person_arr = np.load(person_file).astype(np.float32)

                if cam_arr.ndim == 2 and person_arr.ndim >= 2:
                    if cam_arr.shape[1] >= 6:
                        cam_6d = extract_camera_6d_from_pos_rot(
                            cam_arr[:, :3], cam_arr[:, 3:6])
                    elif cam_arr.shape[1] == 3:
                        cam_6d = np.zeros((cam_arr.shape[0], 6), dtype=np.float32)
                        cam_6d[:, :3] = cam_arr
                    else:
                        continue

                    if person_arr.ndim == 3:
                        person_xyz = person_arr[:, 0, :3]
                    else:
                        person_xyz = person_arr[:, :3]

                    if cam_arr.shape[1] == 3:
                        for t in range(min(cam_6d.shape[0], person_xyz.shape[0])):
                            az, el = look_at_angles(cam_6d[t, :3], person_xyz[t])
                            cam_6d[t, 3] = az
                            cam_6d[t, 4] = el

                    T = min(cam_6d.shape[0], person_xyz.shape[0])
                    cam_6d = cam_6d[:T]
                    person_xyz = person_xyz[:T].astype(np.float32)
                    base_name = os.path.basename(dirpath)
                    for start in range(0, T - num_frames + 1, num_frames):
                        cam_chunk = cam_6d[start:start + num_frames]
                        person_chunk = person_xyz[start:start + num_frames]
                        if is_valid_trajectory(cam_chunk) and is_valid_trajectory(person_chunk):
                            chunk_name = f"{base_name}_f{start:05d}"
                            results.append((cam_chunk, person_chunk, chunk_name))
            except Exception:
                continue

    # Process .npz files
    for npz_path in npz_files:
        try:
            data = np.load(npz_path, allow_pickle=True)
            cam_arr = None
            person_arr = None
            for key in data.files:
                kl = key.lower()
                if any(k in kl for k in ['camera', 'cam_pos', 'cam_traj']):
                    cam_arr = data[key].astype(np.float32)
                elif any(k in kl for k in ['transl', 'trans', 'root', 'motion',
                                            'person', 'dancer', 'body']):
                    person_arr = data[key].astype(np.float32)

            if cam_arr is not None and person_arr is not None:
                if cam_arr.ndim == 2 and person_arr.ndim >= 2:
                    if cam_arr.shape[1] >= 6:
                        cam_6d = extract_camera_6d_from_pos_rot(
                            cam_arr[:, :3], cam_arr[:, 3:6])
                    else:
                        cam_6d = np.zeros((cam_arr.shape[0], 6), dtype=np.float32)
                        cam_6d[:, :3] = cam_arr[:, :3]

                    if person_arr.ndim == 3:
                        person_xyz = person_arr[:, 0, :3]
                    else:
                        person_xyz = person_arr[:, :3]

                    if cam_arr.shape[1] < 6:
                        for t in range(min(cam_6d.shape[0], person_xyz.shape[0])):
                            az, el = look_at_angles(cam_6d[t, :3], person_xyz[t])
                            cam_6d[t, 3] = az
                            cam_6d[t, 4] = el

                    T = min(cam_6d.shape[0], person_xyz.shape[0])
                    cam_6d = cam_6d[:T]
                    person_xyz = person_xyz[:T].astype(np.float32)
                    base_name = os.path.splitext(os.path.basename(npz_path))[0]
                    for start in range(0, T - num_frames + 1, num_frames):
                        cam_chunk = cam_6d[start:start + num_frames]
                        person_chunk = person_xyz[start:start + num_frames]
                        if is_valid_trajectory(cam_chunk) and is_valid_trajectory(person_chunk):
                            chunk_name = f"{base_name}_f{start:05d}"
                            results.append((cam_chunk, person_chunk, chunk_name))
        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Prepare DanceCamera3D data for joint person-camera trajectory training.'
    )
    parser.add_argument('--data-root', type=str, default='/transfer/dancecamera3d',
                        help='Path to DanceCamera3D dataset directory.')
    parser.add_argument('--output-root', type=str, default='/transfer/dance-stc-data',
                        help='Output directory for processed data.')
    parser.add_argument('--num-frames', type=int, default=48,
                        help='Number of frames per trajectory chunk.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_root = args.data_root

    # Check if data exists locally
    if not os.path.isdir(data_root):
        print(f"DanceCamera3D directory not found at: {data_root}")
        print("Attempting automatic download...")
        os.makedirs(os.path.dirname(data_root) or '.', exist_ok=True)
        success = try_download_dancecamera3d(data_root)
        if not success:
            print_manual_download_instructions()
            sys.exit(1)

    # Find all data files
    print(f"\nScanning for data files in {data_root} ...")
    file_map = find_data_files(data_root)
    total_files = sum(len(v) for v in file_map.values())
    print(f"Found files: {', '.join(f'{k}={len(v)}' for k, v in file_map.items() if v)}")

    if total_files == 0:
        print(f"No data files found in {data_root}")
        print_manual_download_instructions()
        sys.exit(1)

    # Create output directories
    cam_out_dir = os.path.join(args.output_root, 'camera_trajectories')
    person_out_dir = os.path.join(args.output_root, 'person_trajectories')
    os.makedirs(cam_out_dir, exist_ok=True)
    os.makedirs(person_out_dir, exist_ok=True)

    all_pairs = []  # list of (cam_6d, person_xyz, name) tuples

    # Process JSON files
    if file_map['json']:
        print(f"\nProcessing {len(file_map['json'])} JSON files...")
        for json_path in tqdm(file_map['json'], desc="JSON files"):
            pairs = process_json_file(json_path, args.num_frames)
            all_pairs.extend(pairs)
        print(f"  Extracted {len(all_pairs)} chunks from JSON files")

    # Process PKL files
    count_before = len(all_pairs)
    if file_map['pkl']:
        print(f"\nProcessing {len(file_map['pkl'])} PKL files...")
        for pkl_path in tqdm(file_map['pkl'], desc="PKL files"):
            pairs = process_pkl_file(pkl_path, args.num_frames)
            all_pairs.extend(pairs)
        print(f"  Extracted {len(all_pairs) - count_before} chunks from PKL files")

    # Process NPY/NPZ files
    count_before = len(all_pairs)
    if file_map['npy'] or file_map['npz']:
        print(f"\nProcessing {len(file_map['npy'])} NPY + {len(file_map['npz'])} NPZ files...")
        pairs = process_npy_npz_pair(data_root, file_map['npy'], file_map['npz'],
                                     args.num_frames)
        all_pairs.extend(pairs)
        print(f"  Extracted {len(all_pairs) - count_before} chunks from NPY/NPZ files")

    if len(all_pairs) == 0:
        print("\nNo valid camera+person trajectory pairs could be extracted.")
        print("The DanceCamera3D data may be in an unexpected format.")
        print("Please check the data files manually and ensure they contain")
        print("camera position/rotation and person root position data.")
        print_manual_download_instructions()
        sys.exit(1)

    # Deduplicate by name
    seen_names = set()
    unique_pairs = []
    for cam_6d, person_xyz, name in all_pairs:
        if name not in seen_names:
            seen_names.add(name)
            unique_pairs.append((cam_6d, person_xyz, name))
    all_pairs = unique_pairs

    print(f"\nTotal unique trajectory chunks: {len(all_pairs)}")

    # Save to output
    train_index = []
    motion_counts = {}
    shot_counts = {}

    for cam_6d, person_xyz, name in tqdm(all_pairs, desc="Saving"):
        sample_id = f"dance_{name}"

        # Save .npy files
        cam_path = os.path.join(cam_out_dir, f'{sample_id}.npy')
        person_path = os.path.join(person_out_dir, f'{sample_id}.npy')
        np.save(cam_path, cam_6d)
        np.save(person_path, person_xyz)

        # Classify camera motion from trajectory
        motion_type = classify_camera_motion_from_trajectory(cam_6d)
        shot_type = infer_shot_type_from_distance(cam_6d, person_xyz)

        # Generate caption
        templates = DANCE_CAPTION_TEMPLATES.get(motion_type, DANCE_CAPTION_TEMPLATES['static'])
        caption = random.choice(templates)

        entry = {
            'id': sample_id,
            'text': caption,
            'shot_type': shot_type,
            'camera_motion': motion_type,
            'camera_trajectory_path': f'camera_trajectories/{sample_id}.npy',
            'person_trajectory_path': f'person_trajectories/{sample_id}.npy',
            'has_real_person': True,
            'source': 'dancecamera3d',
        }
        train_index.append(entry)

        motion_counts[motion_type] = motion_counts.get(motion_type, 0) + 1
        shot_counts[shot_type] = shot_counts.get(shot_type, 0) + 1

    # Save train index
    index_path = os.path.join(args.output_root, 'train_index.json')
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(train_index, f, indent=2, ensure_ascii=False)

    # Summary
    total = len(train_index)
    print(f"\n{'=' * 60}")
    print("DanceCamera3D Preparation Complete!")
    print(f"{'=' * 60}")
    print(f"  Total samples: {total}")
    print(f"\n  Camera motion distribution:")
    for motion, count in sorted(motion_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(total, 1)
        print(f"    {motion:15s}: {count:6d} ({pct:5.1f}%)")
    print(f"\n  Shot type distribution:")
    for st, count in sorted(shot_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(total, 1)
        print(f"    {st:20s}: {count:6d} ({pct:5.1f}%)")
    print(f"\n  Output: {args.output_root}")
    print(f"    camera_trajectories/ ({total} .npy)")
    print(f"    person_trajectories/ ({total} .npy)")
    print(f"    train_index.json ({total} entries)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
