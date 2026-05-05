import json
import numpy as np
import os

# 1. Check ACTUAL v8 norm_stats
print("=== /transfer/merged-v8/norm_stats.json ===")
p = '/transfer/merged-v8/norm_stats.json'
if os.path.exists(p):
    ns = json.load(open(p))
    print(f"mean len: {len(ns['mean'])}, std len: {len(ns['std'])}")
else:
    print("NOT FOUND!")

# 2. Check v8 person .npy shapes
print("\n=== /transfer/merged-v8/person_trajectories ===")
pdir = '/transfer/merged-v8/person_trajectories'
if os.path.isdir(pdir):
    files = os.listdir(pdir)[:5]
    for f in files:
        arr = np.load(os.path.join(pdir, f))
        print(f"  {f}: shape={arr.shape}")
else:
    print("NOT FOUND!")

# 3. List all checkpoints
print("\n=== checkpoints ===")
cdir = '/transfer/fm-v8-checkpoints'
if os.path.isdir(cdir):
    for f in sorted(os.listdir(cdir)):
        if f.endswith('.pth'):
            size = os.path.getsize(os.path.join(cdir, f)) / 1e6
            print(f"  {f}  ({size:.1f} MB)")
else:
    print("NOT FOUND!")
