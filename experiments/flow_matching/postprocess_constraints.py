"""
Motion-type-aware post-processing constraints.

Optional module: apply hard constraints based on motion type AFTER model
generates raw output. Enabled via --enforce-constraints flag in generate.py.

Each constraint function takes (person_traj, camera_traj, num_frames) and
returns (person_traj, camera_traj) with constraints applied.

Design: this module is standalone. If it causes issues, simply don't pass
--enforce-constraints and the generate pipeline works exactly as before.
"""

import numpy as np


def look_at_angles(cam_pos, target):
    """Compute azimuth and elevation from camera to target."""
    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    dist_xz = np.sqrt(dx ** 2 + dz ** 2) + 1e-8
    azimuth = np.arctan2(dx, -dz)
    elevation = np.arctan2(dy, dist_xz)
    return azimuth, elevation


def enforce_static_camera(person_traj, camera_traj):
    """
    Static: freeze camera position to its mean, recompute look-at each frame.
    Camera stays perfectly still but tracks person with orientation.
    """
    T = camera_traj.shape[0]
    mean_pos = camera_traj[:, :3].mean(axis=0)

    for t in range(T):
        camera_traj[t, :3] = mean_pos
        az, el = look_at_angles(mean_pos, person_traj[t, :3])
        camera_traj[t, 3] = az
        camera_traj[t, 4] = el
        camera_traj[t, 5] = 0.0

    return person_traj, camera_traj


def enforce_static_person(person_traj, camera_traj):
    """
    If person speed is very low, freeze person to mean position.
    """
    vel = np.diff(person_traj[:, :3], axis=0)
    avg_speed = np.mean(np.linalg.norm(vel, axis=1))

    if avg_speed < 0.01:  # ~0.5 m/s at 24fps
        mean_pos = person_traj[:, :3].mean(axis=0)
        person_traj[:, :3] = mean_pos

    return person_traj, camera_traj


def enforce_dolly_in(person_traj, camera_traj, min_dist=0.5):
    """
    Dolly-in: enforce distance monotonically decreasing every frame.
    If a frame's distance > previous frame, pull camera closer.
    Floor at min_dist to prevent camera entering person.
    """
    T = camera_traj.shape[0]
    cam_pos = camera_traj[:, :3]
    person_pos = person_traj[:, :3]

    distances = np.linalg.norm(cam_pos - person_pos, axis=1)

    for t in range(1, T):
        if distances[t] >= distances[t - 1]:
            target_dist = max(distances[t - 1] - 1e-3, min_dist)
            direction = person_pos[t] - cam_pos[t]
            current_dist = np.linalg.norm(direction) + 1e-8
            direction_norm = direction / current_dist
            camera_traj[t, :3] = person_pos[t] - direction_norm * target_dist
            distances[t] = target_dist

    # Recompute orientation to face person
    for t in range(T):
        az, el = look_at_angles(camera_traj[t, :3], person_pos[t])
        camera_traj[t, 3] = az
        camera_traj[t, 4] = el

    return person_traj, camera_traj


def enforce_dolly_out(person_traj, camera_traj, max_dist=20.0):
    """
    Dolly-out: enforce distance monotonically increasing every frame.
    """
    T = camera_traj.shape[0]
    cam_pos = camera_traj[:, :3]
    person_pos = person_traj[:, :3]

    distances = np.linalg.norm(cam_pos - person_pos, axis=1)

    for t in range(1, T):
        if distances[t] <= distances[t - 1]:
            target_dist = min(distances[t - 1] + 1e-3, max_dist)
            direction = person_pos[t] - cam_pos[t]
            current_dist = np.linalg.norm(direction) + 1e-8
            direction_norm = direction / current_dist
            camera_traj[t, :3] = person_pos[t] - direction_norm * target_dist
            distances[t] = target_dist

    for t in range(T):
        az, el = look_at_angles(camera_traj[t, :3], person_pos[t])
        camera_traj[t, 3] = az
        camera_traj[t, 4] = el

    return person_traj, camera_traj


def enforce_crane_up(person_traj, camera_traj):
    """
    Crane-up: enforce camera Y monotonically increasing.
    Keep XZ position stable.
    """
    T = camera_traj.shape[0]
    person_pos = person_traj[:, :3]

    # Freeze XZ to mean
    mean_x = camera_traj[:, 0].mean()
    mean_z = camera_traj[:, 2].mean()
    camera_traj[:, 0] = mean_x
    camera_traj[:, 2] = mean_z

    # Enforce Y monotonically increasing
    for t in range(1, T):
        if camera_traj[t, 1] <= camera_traj[t - 1, 1]:
            camera_traj[t, 1] = camera_traj[t - 1, 1] + 1e-3

    # Recompute orientation
    for t in range(T):
        az, el = look_at_angles(camera_traj[t, :3], person_pos[t])
        camera_traj[t, 3] = az
        camera_traj[t, 4] = el

    return person_traj, camera_traj


def enforce_crane_down(person_traj, camera_traj):
    """
    Crane-down: enforce camera Y monotonically decreasing.
    Keep XZ position stable.
    """
    T = camera_traj.shape[0]
    person_pos = person_traj[:, :3]

    mean_x = camera_traj[:, 0].mean()
    mean_z = camera_traj[:, 2].mean()
    camera_traj[:, 0] = mean_x
    camera_traj[:, 2] = mean_z

    for t in range(1, T):
        if camera_traj[t, 1] >= camera_traj[t - 1, 1]:
            camera_traj[t, 1] = camera_traj[t - 1, 1] - 1e-3

    for t in range(T):
        az, el = look_at_angles(camera_traj[t, :3], person_pos[t])
        camera_traj[t, 3] = az
        camera_traj[t, 4] = el

    return person_traj, camera_traj


def _wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def enforce_pan(person_traj, camera_traj, direction='left'):
    """
    Pan: freeze camera position, enforce azimuth monotonically changing.
    Angles wrapped to [-pi, pi].
    """
    T = camera_traj.shape[0]

    # Freeze position to mean
    mean_pos = camera_traj[:, :3].mean(axis=0)
    camera_traj[:, :3] = mean_pos

    # Unwrap azimuth for monotonic enforcement, then wrap back
    az = camera_traj[:, 3].copy()
    az = np.unwrap(az)  # remove discontinuities for processing

    if direction == 'left':
        for t in range(1, T):
            if az[t] >= az[t - 1]:
                az[t] = az[t - 1] - 1e-3
    else:  # right
        for t in range(1, T):
            if az[t] <= az[t - 1]:
                az[t] = az[t - 1] + 1e-3

    # Wrap back to [-pi, pi]
    camera_traj[:, 3] = _wrap_angle(az)

    # Freeze elevation and roll
    camera_traj[:, 4] = camera_traj[:, 4].mean()
    camera_traj[:, 5] = 0.0

    return person_traj, camera_traj


def enforce_orbit(person_traj, camera_traj):
    """
    Orbit: enforce roughly constant distance to person centroid.
    """
    T = camera_traj.shape[0]
    person_pos = person_traj[:, :3]
    centroid = person_pos.mean(axis=0)

    distances = np.linalg.norm(camera_traj[:, :3] - centroid, axis=1)
    target_dist = distances.mean()

    for t in range(T):
        direction = camera_traj[t, :3] - centroid
        current_dist = np.linalg.norm(direction) + 1e-8
        camera_traj[t, :3] = centroid + (direction / current_dist) * target_dist

    # Recompute orientation
    for t in range(T):
        az, el = look_at_angles(camera_traj[t, :3], person_pos[t])
        camera_traj[t, 3] = az
        camera_traj[t, 4] = el

    return person_traj, camera_traj


def enforce_track(person_traj, camera_traj):
    """
    Track: enforce roughly constant distance between camera and person.
    Camera should follow person's XZ movement.
    """
    T = camera_traj.shape[0]
    person_pos = person_traj[:, :3]

    distances = np.linalg.norm(camera_traj[:, :3] - person_pos, axis=1)
    target_dist = distances.mean()

    for t in range(T):
        direction = camera_traj[t, :3] - person_pos[t]
        current_dist = np.linalg.norm(direction) + 1e-8
        camera_traj[t, :3] = person_pos[t] + (direction / current_dist) * target_dist

    for t in range(T):
        az, el = look_at_angles(camera_traj[t, :3], person_pos[t])
        camera_traj[t, 3] = az
        camera_traj[t, 4] = el

    return person_traj, camera_traj


# Dispatch table
CONSTRAINT_FNS = {
    'static': enforce_static_camera,
    'dolly-in': enforce_dolly_in,
    'dolly-out': enforce_dolly_out,
    'crane-up': enforce_crane_up,
    'crane-down': enforce_crane_down,
    'pan-left': lambda p, c: enforce_pan(p, c, 'left'),
    'pan-right': lambda p, c: enforce_pan(p, c, 'right'),
    'orbit': enforce_orbit,
    'track': enforce_track,
}


def apply_constraints(person_traj, camera_traj, motion_type):
    """
    Apply motion-type-aware constraints.

    Args:
        person_traj: (T, person_dim) person trajectory
        camera_traj: (T, 6) camera trajectory
        motion_type: string motion type name

    Returns:
        (person_traj, camera_traj) with constraints applied
    """
    # Always check if person should be frozen
    person_traj, camera_traj = enforce_static_person(person_traj, camera_traj)

    # Apply motion-specific constraint
    fn = CONSTRAINT_FNS.get(motion_type)
    if fn is not None:
        person_traj, camera_traj = fn(person_traj, camera_traj)

    return person_traj, camera_traj
