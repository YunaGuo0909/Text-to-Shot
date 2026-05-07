"""
Verify camera generation quality BEFORE full data pipeline.

Generates sample trajectories for all 9 motion types using the actual
generate_camera_for_person function, then checks:
1. Dolly-in: distance monotonically decreasing every frame
2. Dolly-out: distance monotonically increasing every frame
3. Static: camera position variance near zero
4. Pan: camera position static, azimuth changes
5. Crane: camera Y changes, XZ static
6. Track: camera follows person, distance stable
7. Orbit: azimuth sweeps, distance stable
8. All types: camera path smoothness (low jerk)
9. All types: camera-person distance > 0.5m (no collision)

Usage:
    PYTHONPATH=. python scripts/verify_camera_generation.py
"""

import numpy as np
import random
import sys

random.seed(42)
np.random.seed(42)

from scripts.prepare_amass import generate_camera_for_person, ALL_MOTION_TYPES


def make_person_stationary(T=48):
    """Person stands still at origin."""
    traj = np.zeros((T, 3), dtype=np.float32)
    traj[:, 1] = 0.5  # slight Y offset (standing height proxy)
    return traj


def make_person_walking_forward(T=48):
    """Person walks forward along +Z."""
    traj = np.zeros((T, 3), dtype=np.float32)
    traj[:, 2] = np.linspace(0, 2.0, T)  # 2m forward
    traj[:, 1] = 0.5
    return traj


def make_person_walking_right(T=48):
    """Person walks to the right along +X."""
    traj = np.zeros((T, 3), dtype=np.float32)
    traj[:, 0] = np.linspace(0, 1.5, T)
    traj[:, 1] = 0.5
    return traj


def make_person_zigzag(T=48):
    """Person with slight zigzag (MoCap-like noise)."""
    traj = np.zeros((T, 3), dtype=np.float32)
    traj[:, 2] = np.linspace(0, 1.5, T)
    traj[:, 0] = 0.05 * np.sin(np.linspace(0, 4 * np.pi, T))
    traj[:, 1] = 0.5
    return traj


PERSON_TRAJS = {
    'stationary': make_person_stationary,
    'walk_forward': make_person_walking_forward,
    'walk_right': make_person_walking_right,
    'zigzag': make_person_zigzag,
}


def compute_smoothness(traj_3d):
    """Mean jerk magnitude (lower = smoother)."""
    if traj_3d.shape[0] < 4:
        return 0.0
    jerk = np.diff(np.diff(np.diff(traj_3d, axis=0), axis=0), axis=0)
    return float(np.mean(np.linalg.norm(jerk, axis=1)))


def check_monotonic_decreasing(values):
    """Check if values decrease every frame. Returns (ok, worst_violation)."""
    diffs = np.diff(values)
    violations = diffs[diffs > 0]
    if len(violations) == 0:
        return True, 0.0
    return False, float(violations.max())


def check_monotonic_increasing(values):
    diffs = np.diff(values)
    violations = diffs[diffs < 0]
    if len(violations) == 0:
        return True, 0.0
    return False, float(abs(violations.min()))


def verify_one(motion_type, person_traj, person_name):
    """Generate and verify one camera trajectory."""
    result = generate_camera_for_person(person_traj, motion_type, 48)
    cam_traj, shot_type = result

    if cam_traj is None:
        return {'status': 'SKIPPED', 'reason': 'incompatible (e.g. track+stationary)'}

    cam_pos = cam_traj[:, :3]
    person_pos = person_traj[:, :3]
    distances = np.linalg.norm(cam_pos - person_pos, axis=1)
    cam_jerk = compute_smoothness(cam_pos)
    cam_pos_var = np.var(cam_pos, axis=0).sum()
    az = cam_traj[:, 3]
    az_change = az[-1] - az[0]
    cam_y_change = cam_pos[-1, 1] - cam_pos[0, 1]
    min_dist = distances.min()

    issues = []

    # Universal checks
    if cam_jerk > 0.1:
        issues.append(f'HIGH_JERK={cam_jerk:.4f}')
    if min_dist < 0.5:
        issues.append(f'TOO_CLOSE={min_dist:.2f}m')
    if not np.isfinite(cam_traj).all():
        issues.append('NaN/Inf')

    # Motion-specific checks
    if motion_type == 'dolly-in':
        ok, worst = check_monotonic_decreasing(distances)
        if not ok:
            issues.append(f'DIST_NOT_MONOTONIC_DEC(worst={worst:.4f})')
        if distances[-1] >= distances[0]:
            issues.append(f'DIST_DID_NOT_DECREASE({distances[0]:.2f}->{distances[-1]:.2f})')

    elif motion_type == 'dolly-out':
        ok, worst = check_monotonic_increasing(distances)
        if not ok:
            issues.append(f'DIST_NOT_MONOTONIC_INC(worst={worst:.4f})')
        if distances[-1] <= distances[0]:
            issues.append(f'DIST_DID_NOT_INCREASE({distances[0]:.2f}->{distances[-1]:.2f})')

    elif motion_type == 'static':
        if cam_pos_var > 0.05:
            issues.append(f'CAM_MOVED(var={cam_pos_var:.4f})')

    elif motion_type in ('pan-left', 'pan-right'):
        cam_xz_disp = np.linalg.norm(cam_pos[-1, [0, 2]] - cam_pos[0, [0, 2]])
        if cam_xz_disp > 0.1:
            issues.append(f'CAM_POS_MOVED(xz_disp={cam_xz_disp:.4f})')
        if abs(az_change) < np.radians(10):
            issues.append(f'AZ_TOO_SMALL({np.degrees(az_change):.1f}deg)')

    elif motion_type in ('crane-up', 'crane-down'):
        if motion_type == 'crane-up' and cam_y_change < 0.1:
            issues.append(f'CAM_Y_DID_NOT_RISE({cam_y_change:.2f})')
        if motion_type == 'crane-down' and cam_y_change > -0.1:
            issues.append(f'CAM_Y_DID_NOT_DROP({cam_y_change:.2f})')

    elif motion_type == 'track':
        dist_std = np.std(distances)
        if dist_std > 0.5:
            issues.append(f'DIST_UNSTABLE(std={dist_std:.2f})')

    elif motion_type == 'orbit':
        if abs(az_change) < np.radians(20):
            issues.append(f'AZ_TOO_SMALL({np.degrees(az_change):.1f}deg)')

    return {
        'status': 'PASS' if not issues else 'FAIL',
        'issues': issues,
        'dist_range': f'{distances[0]:.2f}->{distances[-1]:.2f}',
        'cam_jerk': f'{cam_jerk:.6f}',
        'cam_pos_var': f'{cam_pos_var:.4f}',
        'az_change': f'{np.degrees(az_change):.1f}deg',
        'cam_y_change': f'{cam_y_change:.2f}',
        'min_dist': f'{min_dist:.2f}m',
    }


def main():
    print("=" * 70)
    print("CAMERA GENERATION VERIFICATION")
    print("=" * 70)

    total_pass = 0
    total_fail = 0
    total_skip = 0
    N_TRIALS = 20  # trials per (motion_type, person_type) combination

    for motion_type in ALL_MOTION_TYPES:
        print(f"\n--- {motion_type} ---")
        for person_name, person_fn in PERSON_TRAJS.items():
            pass_count = 0
            fail_count = 0
            skip_count = 0
            all_issues = []

            for trial in range(N_TRIALS):
                random.seed(42 + trial * 100)
                np.random.seed(42 + trial * 100)
                person_traj = person_fn()
                result = verify_one(motion_type, person_traj, person_name)

                if result['status'] == 'PASS':
                    pass_count += 1
                    total_pass += 1
                elif result['status'] == 'SKIPPED':
                    skip_count += 1
                    total_skip += 1
                else:
                    fail_count += 1
                    total_fail += 1
                    all_issues.extend(result['issues'])

            status_str = f"PASS={pass_count} FAIL={fail_count} SKIP={skip_count}"
            marker = " !!!" if fail_count > 0 else ""
            print(f"  {person_name:15s}: {status_str}{marker}")
            if all_issues:
                # Show unique issues
                unique_issues = list(set(all_issues))
                for iss in unique_issues[:3]:
                    print(f"    -> {iss}")

    print(f"\n{'=' * 70}")
    print(f"TOTAL: PASS={total_pass} FAIL={total_fail} SKIP={total_skip}")
    if total_fail == 0:
        print("ALL CHECKS PASSED - safe to generate full dataset")
    else:
        print(f"WARNING: {total_fail} failures - FIX BEFORE GENERATING DATA")
    print(f"{'=' * 70}")

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == '__main__':
    main()
