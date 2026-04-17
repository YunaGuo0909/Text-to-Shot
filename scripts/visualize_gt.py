"""
Visualize a few ground-truth samples from the training set.

Usage:
    python scripts/visualize_gt.py
    python scripts/visualize_gt.py --motion dolly-out --n 3
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/transfer/stc-data')
    parser.add_argument('--index-file', default='train_index.json')
    parser.add_argument('--output-dir', default='/transfer/stc-outputs')
    parser.add_argument('--motion', default=None, help='Filter by camera_motion type')
    parser.add_argument('--n', type=int, default=5, help='Number of samples to visualize')
    args = parser.parse_args()

    with open(os.path.join(args.data_root, args.index_file), encoding='utf-8') as f:
        index = json.load(f)

    if args.motion:
        index = [s for s in index if s.get('camera_motion') == args.motion]
        print(f"Filtered to {len(index)} samples with motion={args.motion}")

    os.makedirs(args.output_dir, exist_ok=True)

    for i, sample in enumerate(index[:args.n]):
        cam = np.load(os.path.join(args.data_root, sample['camera_trajectory_path']))
        per = np.load(os.path.join(args.data_root, sample['person_trajectory_path']))
        text = sample.get('text', '')[:80]
        motion = sample.get('camera_motion', '?')
        sid = sample.get('id', str(i))

        # Use the same enhanced visualisation as generate.py
        import sys, os as _os
        sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from generate import visualize_joint

        save_path = os.path.join(args.output_dir, f'gt_{motion}_{i}.png')
        title_text = f'[GT] {text}'
        visualize_joint(per, cam, title_text, motion, save_path)

        print(f"[{i+1}] {sid}: {text}")
        print(f"    cam range: {cam.min():.2f} ~ {cam.max():.2f}")
        print(f"    per range: {per.min():.2f} ~ {per.max():.2f}")
        print(f"    -> {save_path}")


if __name__ == '__main__':
    main()
