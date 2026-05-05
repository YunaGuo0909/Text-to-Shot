import json
import torch
import numpy as np
import glob
import os

# 1. Check norm_stats
print("=== norm_stats ===")
ns = json.load(open('/transfer/merged-v7/norm_stats.json'))
print(f"mean len: {len(ns['mean'])}, std len: {len(ns['std'])}")

# 2. Check a person .npy file shape
print("\n=== person .npy samples ===")
pdir = '/transfer/merged-v7/person_trajectories'
if os.path.isdir(pdir):
    files = os.listdir(pdir)[:5]
    for f in files:
        arr = np.load(os.path.join(pdir, f))
        print(f"  {f}: shape={arr.shape}")
else:
    print(f"  {pdir} not found")

# 3. Check checkpoint config
print("\n=== checkpoint config ===")
ckpt_dir = '/transfer/fm-v8-checkpoints'
if os.path.isdir(ckpt_dir):
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pth')])
    if ckpts:
        p = os.path.join(ckpt_dir, ckpts[-1])
        print(f"Loading: {p}")
        c = torch.load(p, map_location='cpu', weights_only=False)
        cfg = c['config']
        pd = cfg['model']['person_dim']
        cd = cfg['model']['camera_dim']
        nf = cfg['trajectory']['default_num_frames']
        print(f"  person_dim={pd}, camera_dim={cd}, frames={nf}")
        print(f"  total_dim={pd*nf + cd*nf}")
        print(f"  norm_stats_path={cfg['data'].get('norm_stats_path')}")
        print(f"  data_root={cfg['data'].get('data_root')}")
    else:
        print("  no .pth files")
else:
    print(f"  {ckpt_dir} not found")
