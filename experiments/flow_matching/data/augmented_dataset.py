import torch
import numpy as np
from src.data.dataset import JointTrajectoryDataset


class AugmentedTrajectoryDataset(JointTrajectoryDataset):
    """
    Wraps JointTrajectoryDataset with trajectory augmentations applied
    between loading and normalization.

    Augmentations:
    1. Mirror X (50%): flip person/camera X coords, negate azimuth
    2. Mirror Z (50%): flip person/camera Z coords, azimuth -> pi - azimuth
    3. Random spatial offset (always): shared (dx, dy, dz) ~ N(0, 0.1)
    4. Temporal speed jitter (30%): resample at 0.8x-1.2x speed
    """

    def __init__(self, augment=True, **kwargs):
        super().__init__(**kwargs)
        self.augment = augment

    def __getitem__(self, idx):
        sample = self.samples[idx]

        camera_traj = self._load_trajectory(
            sample, key='camera_trajectory_path', fallback_key='trajectory_path',
            dim=self.camera_dim)
        person_traj = self._load_trajectory(
            sample, key='person_trajectory_path', fallback_key=None,
            dim=self.person_dim)

        if camera_traj.shape[0] != self.num_frames:
            camera_traj = self._resample(camera_traj, self.num_frames)
        if person_traj.shape[0] != self.num_frames:
            person_traj = self._resample(person_traj, self.num_frames)

        if self.augment:
            person_traj, camera_traj = self._augment(person_traj, camera_traj)

        person_flat = torch.tensor(person_traj.flatten(), dtype=torch.float32)
        camera_flat = torch.tensor(camera_traj.flatten(), dtype=torch.float32)
        y = torch.cat([person_flat, camera_flat], dim=0)

        if self.norm_mean is not None:
            y = (y - self.norm_mean) / self.norm_std

        text = sample.get('text', sample.get('description', ''))
        shot_type = self.SHOT_TYPE_MAP.get(sample.get('shot_type', ''), -1)
        motion_type = self.MOTION_TYPE_MAP.get(
            sample.get('camera_motion', sample.get('motion_type', '')), -1)

        return {
            'y': y,
            'text': text,
            'shot_type': shot_type,
            'motion_type': motion_type,
            'sample_id': sample.get('id', idx),
        }

    def _augment(self, person_traj, camera_traj):
        # person_traj: (T, 5) = [px, py, pz, sin_yaw, cos_yaw]
        # camera_traj: (T, 6) = [tx, ty, tz, azimuth, elevation, roll]
        person_traj = person_traj.copy()
        camera_traj = camera_traj.copy()
        has_yaw = person_traj.shape[1] >= 5

        # 1. Mirror X (50%): px -> -px, azimuth -> -azimuth, yaw -> -yaw
        #    sin(-yaw) = -sin(yaw), cos(-yaw) = cos(yaw)
        if np.random.rand() < 0.5:
            person_traj[:, 0] *= -1.0
            camera_traj[:, 0] *= -1.0
            camera_traj[:, 3] *= -1.0  # negate azimuth
            if has_yaw:
                person_traj[:, 3] *= -1.0  # negate sin_yaw

        # 2. Mirror Z (50%): pz -> -pz, azimuth -> pi - azimuth, yaw -> pi - yaw
        #    sin(pi - yaw) = sin(yaw), cos(pi - yaw) = -cos(yaw)
        if np.random.rand() < 0.5:
            person_traj[:, 2] *= -1.0
            camera_traj[:, 2] *= -1.0
            camera_traj[:, 3] = np.pi - camera_traj[:, 3]  # azimuth -> pi - azimuth
            if has_yaw:
                person_traj[:, 4] *= -1.0  # negate cos_yaw

        # 3. Random spatial offset (position dims only, yaw unaffected)
        offset = np.random.randn(3).astype(np.float32) * 0.1
        person_traj[:, 0] += offset[0]
        person_traj[:, 1] += offset[1]
        person_traj[:, 2] += offset[2]
        camera_traj[:, 0] += offset[0]
        camera_traj[:, 1] += offset[1]
        camera_traj[:, 2] += offset[2]

        # 4. Temporal speed jitter (30%)
        if np.random.rand() < 0.3:
            speed = np.random.uniform(0.8, 1.2)
            T = person_traj.shape[0]
            src_t = np.linspace(0, 1, T)
            new_duration = 1.0 / speed
            tgt_t = np.linspace(0, min(new_duration, 1.0), T)

            new_person = np.zeros_like(person_traj)
            for d in range(person_traj.shape[1]):
                new_person[:, d] = np.interp(tgt_t, src_t, person_traj[:, d])

            new_camera = np.zeros_like(camera_traj)
            for d in range(camera_traj.shape[1]):
                new_camera[:, d] = np.interp(tgt_t, src_t, camera_traj[:, d])

            person_traj = new_person
            camera_traj = new_camera

        return person_traj, camera_traj
