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

        fig = plt.figure(figsize=(16, 6), facecolor='#1a1a2e')

        # 3D
        ax1 = fig.add_subplot(131, projection='3d', facecolor='#1a1a2e')
        ax1.plot3D(cam[:, 0], cam[:, 1], cam[:, 2], color='#FFE66D', lw=2, label='Camera')
        ax1.plot3D(per[:, 0], per[:, 1], per[:, 2], color='#4ECDC4', lw=2, label='Person')
        ax1.scatter(*cam[0, :3], color='#FFE66D', s=60, marker='o', edgecolors='white')
        ax1.scatter(*per[0], color='#4ECDC4', s=60, marker='^', edgecolors='white')
        ax1.set_title('3D Trajectories', color='white', fontsize=11)
        ax1.legend(fontsize=8, labelcolor='white', framealpha=0.3)
        ax1.tick_params(colors='gray', labelsize=7)

        # Camera params
        ax2 = fig.add_subplot(132, facecolor='#2C3E50')
        t = np.linspace(0, 1, len(cam))
        names = ['tx', 'ty', 'tz', 'az', 'el', 'roll']
        colors = ['#FF6B6B', '#FFE66D', '#4ECDC4', '#C44ECD', '#95E66D', '#FF9F43']
        for j, (name, c) in enumerate(zip(names, colors)):
            ax2.plot(t, cam[:, j], color=c, lw=1.5, label=name, alpha=0.8)
        ax2.set_title('Camera Parameters', color='white', fontsize=11)
        ax2.legend(fontsize=7, labelcolor='white', framealpha=0.3, ncol=2)
        ax2.tick_params(colors='gray', labelsize=7)
        ax2.grid(alpha=0.15)

        # Person
        ax3 = fig.add_subplot(133, facecolor='#2C3E50')
        pnames = ['px', 'py', 'pz']
        pcolors = ['#4ECDC4', '#95E66D', '#C44ECD']
        for j, (name, c) in enumerate(zip(pnames, pcolors)):
            ax3.plot(t, per[:, j], color=c, lw=1.5, label=name)
        ax3.set_title('Person Position', color='white', fontsize=11)
        ax3.legend(fontsize=8, labelcolor='white', framealpha=0.3)
        ax3.tick_params(colors='gray', labelsize=7)
        ax3.grid(alpha=0.15)

        title = f'GT | {motion} | "{text}"'
        fig.suptitle(title, color='white', fontsize=11, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        save_path = os.path.join(args.output_dir, f'gt_{motion}_{i}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[{i+1}] {sid}: {text}")
        print(f"    cam range: {cam.min():.2f} ~ {cam.max():.2f}")
        print(f"    per range: {per.min():.2f} ~ {per.max():.2f}")
        print(f"    → {save_path}")


if __name__ == '__main__':
    main()
