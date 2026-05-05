"""Compute normalization statistics for merged dataset."""
import json
import os
import numpy as np
from tqdm import tqdm

import sys

DATA_ROOT = '/transfer/merged-v8'
PERSON_DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 5
CAMERA_DIM = 6
NUM_FRAMES = 48

index_path = os.path.join(DATA_ROOT, 'train_index.json')
with open(index_path, 'r') as f:
    samples = json.load(f)

print(f"Computing norm stats over {len(samples)} samples...")

all_y = []
for s in tqdm(samples):
    cam_path = os.path.join(DATA_ROOT, s['camera_trajectory_path'])
    per_path = os.path.join(DATA_ROOT, s['person_trajectory_path'])

    if not os.path.exists(cam_path) or not os.path.exists(per_path):
        continue

    cam = np.load(cam_path).astype(np.float32)
    per = np.load(per_path).astype(np.float32)

    if cam.shape != (NUM_FRAMES, CAMERA_DIM):
        continue
    if per.shape[0] != NUM_FRAMES:
        continue
    if per.shape[1] < PERSON_DIM:
        if per.shape[1] == 3 and PERSON_DIM == 5:
            # sin(0)=0, cos(0)=1
            zeros = np.zeros((NUM_FRAMES, 1), dtype=np.float32)
            ones = np.ones((NUM_FRAMES, 1), dtype=np.float32)
            per = np.concatenate([per, zeros, ones], axis=1)
        else:
            pad = np.zeros((NUM_FRAMES, PERSON_DIM - per.shape[1]), dtype=np.float32)
            per = np.concatenate([per, pad], axis=1)
    per = per[:, :PERSON_DIM]

    y = np.concatenate([per.flatten(), cam.flatten()])
    all_y.append(y)

all_y = np.array(all_y)
print(f"Loaded {len(all_y)} valid samples, dim={all_y.shape[1]}")

mean = all_y.mean(axis=0)
std = all_y.std(axis=0)
std[std < 1e-6] = 1.0  # avoid div by zero

suffix = f'_v9' if PERSON_DIM == 3 else ''
out_path = os.path.join(DATA_ROOT, f'norm_stats{suffix}.json')
with open(out_path, 'w') as f:
    json.dump({
        'mean': mean.tolist(),
        'std': std.tolist(),
        'n_samples': len(all_y),
    }, f)

print(f"Saved to {out_path}")
print(f"  mean len: {len(mean)}, std len: {len(std)}")
print(f"  person mean range: [{mean[:PERSON_DIM*NUM_FRAMES].min():.4f}, {mean[:PERSON_DIM*NUM_FRAMES].max():.4f}]")
print(f"  camera mean range: [{mean[PERSON_DIM*NUM_FRAMES:].min():.4f}, {mean[PERSON_DIM*NUM_FRAMES:].max():.4f}]")
print(f"  person std range:  [{std[:PERSON_DIM*NUM_FRAMES].min():.4f}, {std[:PERSON_DIM*NUM_FRAMES].max():.4f}]")
print(f"  camera std range:  [{std[PERSON_DIM*NUM_FRAMES:].min():.4f}, {std[PERSON_DIM*NUM_FRAMES:].max():.4f}]")
