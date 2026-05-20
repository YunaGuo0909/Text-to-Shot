"""
Merge multiple processed datasets into one unified dataset for training.

Reads train_index.json from each source directory, copies trajectory .npy
files to a unified directory, merges index entries (adding a 'source' field),
optionally computes normalization statistics, and prints a summary.

Outputs:
  - <output-root>/camera_trajectories/*.npy
  - <output-root>/person_trajectories/*.npy
  - <output-root>/train_index.json
  - <output-root>/test_index.json  (if any source has one)
  - <output-root>/norm_stats.json  (with --compute-norm-stats)

Usage:
    python scripts/data/merge_datasets.py \\
        --sources /transfer/stc-data /transfer/amass-stc-data /transfer/dance-stc-data \\
        --output-root /transfer/merged-stc-data \\
        --compute-norm-stats
"""

import os
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict


# Default source names derived from directory names
SOURCE_NAMES = {
    'stc-data': 'et',
    'amass-stc-data': 'amass',
    'dance-stc-data': 'dancecamera3d',
}


def infer_source_name(source_dir: str) -> str:
    """Infer a human-readable source name from a directory path."""
    basename = os.path.basename(os.path.normpath(source_dir))
    if basename in SOURCE_NAMES:
        return SOURCE_NAMES[basename]
    # Try common patterns
    lower = basename.lower()
    if 'amass' in lower:
        return 'amass'
    if 'dance' in lower:
        return 'dancecamera3d'
    if 'et' in lower or 'stc' in lower:
        return 'et'
    return basename


def copy_or_symlink(src: str, dst: str, use_symlinks: bool = False):
    """Copy or symlink a file from src to dst."""
    if os.path.exists(dst):
        return  # already exists
    dst_dir = os.path.dirname(dst)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    if use_symlinks:
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple processed datasets into one unified dataset.'
    )
    parser.add_argument('--sources', type=str, nargs='+', required=True,
                        help='List of source data directories, each containing '
                             'train_index.json, camera_trajectories/, person_trajectories/')
    parser.add_argument('--output-root', type=str, default='/transfer/merged-stc-data',
                        help='Output directory for the merged dataset.')
    parser.add_argument('--use-symlinks', action='store_true',
                        help='Use symlinks instead of copying .npy files (saves disk space).')
    parser.add_argument('--compute-norm-stats', action='store_true',
                        help='Compute normalization statistics over the merged training data.')
    parser.add_argument('--test-split-ratio', type=float, default=0.0,
                        help='If > 0 and a source has no test_index.json, randomly hold out '
                             'this fraction as test data.')
    parser.add_argument('--num-frames', type=int, default=48)
    parser.add_argument('--person-dim', type=int, default=5)
    parser.add_argument('--camera-dim', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Output directories
    cam_out_dir = os.path.join(args.output_root, 'camera_trajectories')
    person_out_dir = os.path.join(args.output_root, 'person_trajectories')
    os.makedirs(cam_out_dir, exist_ok=True)
    os.makedirs(person_out_dir, exist_ok=True)

    merged_train = []
    merged_test = []
    source_stats = {}  # source_name -> {train: N, test: N}
    motion_counts = defaultdict(int)
    shot_counts = defaultdict(int)
    id_set = set()  # for deduplication

    for source_dir in args.sources:
        source_dir = os.path.normpath(source_dir)
        source_name = infer_source_name(source_dir)

        if not os.path.isdir(source_dir):
            print(f"[Warning] Source directory not found, skipping: {source_dir}")
            continue

        train_index_path = os.path.join(source_dir, 'train_index.json')
        test_index_path = os.path.join(source_dir, 'test_index.json')

        if not os.path.exists(train_index_path):
            print(f"[Warning] No train_index.json in {source_dir}, skipping.")
            continue

        with open(train_index_path, 'r', encoding='utf-8') as f:
            train_entries = json.load(f)

        test_entries = []
        if os.path.exists(test_index_path):
            with open(test_index_path, 'r', encoding='utf-8') as f:
                test_entries = json.load(f)

        print(f"\nSource: {source_name} ({source_dir})")
        print(f"  Train: {len(train_entries)}  Test: {len(test_entries)}")

        src_train_count = 0
        src_test_count = 0

        for split_name, entries, target_list in [
            ('train', train_entries, merged_train),
            ('test', test_entries, merged_test),
        ]:
            for entry in tqdm(entries, desc=f"  {source_name}/{split_name}", leave=False):
                sample_id = entry.get('id', '')

                # Handle ID collisions across sources by prefixing with source
                unique_id = f"{source_name}_{sample_id}" if sample_id in id_set else sample_id
                if unique_id in id_set:
                    # Still collides (unlikely but handle it)
                    unique_id = f"{source_name}_{sample_id}_{len(id_set)}"
                id_set.add(unique_id)

                # Resolve source file paths
                cam_src = os.path.join(source_dir, entry['camera_trajectory_path'])
                person_src = os.path.join(source_dir, entry['person_trajectory_path'])

                if not os.path.exists(cam_src) or not os.path.exists(person_src):
                    continue

                # Copy/symlink to output
                cam_dst = os.path.join(cam_out_dir, f'{unique_id}.npy')
                person_dst = os.path.join(person_out_dir, f'{unique_id}.npy')
                copy_or_symlink(cam_src, cam_dst, args.use_symlinks)
                copy_or_symlink(person_src, person_dst, args.use_symlinks)

                # Build merged entry
                merged_entry = {
                    'id': unique_id,
                    'text': entry.get('text', ''),
                    'shot_type': entry.get('shot_type', 'medium-shot'),
                    'camera_motion': entry.get('camera_motion', 'static'),
                    'camera_trajectory_path': f'camera_trajectories/{unique_id}.npy',
                    'person_trajectory_path': f'person_trajectories/{unique_id}.npy',
                    'has_real_person': entry.get('has_real_person', False),
                    'source': entry.get('source', source_name),
                }
                target_list.append(merged_entry)

                motion_counts[merged_entry['camera_motion']] += 1
                shot_counts[merged_entry['shot_type']] += 1

                if split_name == 'train':
                    src_train_count += 1
                else:
                    src_test_count += 1

        source_stats[source_name] = {'train': src_train_count, 'test': src_test_count}

    # Optional: split some train data into test if no test data exists from that source
    if args.test_split_ratio > 0 and len(merged_test) == 0:
        n_test = int(len(merged_train) * args.test_split_ratio)
        if n_test > 0:
            np.random.shuffle(merged_train)
            merged_test = merged_train[:n_test]
            merged_train = merged_train[n_test:]
            print(f"\nHeld out {n_test} samples as test set ({args.test_split_ratio:.1%})")

    # Save merged indices
    train_path = os.path.join(args.output_root, 'train_index.json')
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(merged_train, f, indent=2, ensure_ascii=False)

    test_path = os.path.join(args.output_root, 'test_index.json')
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(merged_test, f, indent=2, ensure_ascii=False)

    # Compute normalization statistics
    if args.compute_norm_stats:
        print("\nComputing normalization statistics over merged training data...")
        all_y = []
        skipped = 0
        for entry in tqdm(merged_train, desc="Loading for norm stats"):
            cam_path = os.path.join(args.output_root, entry['camera_trajectory_path'])
            person_path = os.path.join(args.output_root, entry['person_trajectory_path'])
            if not os.path.exists(cam_path) or not os.path.exists(person_path):
                skipped += 1
                continue
            try:
                cam = np.load(cam_path).astype(np.float32)
                person = np.load(person_path).astype(np.float32)

                # Resample if needed
                if cam.shape[0] != args.num_frames:
                    src_t = np.linspace(0, 1, cam.shape[0])
                    tgt_t = np.linspace(0, 1, args.num_frames)
                    cam = np.stack([np.interp(tgt_t, src_t, cam[:, d])
                                    for d in range(cam.shape[1])], axis=1)
                if person.shape[0] != args.num_frames:
                    src_t = np.linspace(0, 1, person.shape[0])
                    tgt_t = np.linspace(0, 1, args.num_frames)
                    person = np.stack([np.interp(tgt_t, src_t, person[:, d])
                                       for d in range(person.shape[1])], axis=1)

                y = np.concatenate([person.flatten(), cam.flatten()])
                if not np.isfinite(y).all():
                    skipped += 1
                    continue
                all_y.append(y)
            except Exception:
                skipped += 1
                continue

        if skipped:
            print(f"  Skipped {skipped} samples (missing files or NaN/Inf)")

        if len(all_y) > 0:
            all_y = np.stack(all_y, axis=0)
            mean = all_y.mean(axis=0)
            std = all_y.std(axis=0)
            std = np.where(std < 1e-6, 1.0, std)

            norm_stats = {
                'mean': mean.tolist(),
                'std': std.tolist(),
                'n_samples': int(len(all_y)),
                'total_dim': int(all_y.shape[1]),
                'num_frames': args.num_frames,
                'person_dim': args.person_dim,
                'camera_dim': args.camera_dim,
            }
            norm_path = os.path.join(args.output_root, 'norm_stats.json')
            with open(norm_path, 'w') as f:
                json.dump(norm_stats, f)
            print(f"  Saved norm_stats.json ({len(all_y)} samples)")
            print(f"  Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
            print(f"  Std  range: [{std.min():.3f}, {std.max():.3f}]")
        else:
            print("  [Warning] No valid samples for norm stats computation.")

    # Summary
    total_train = len(merged_train)
    total_test = len(merged_test)
    total = total_train + total_test
    print(f"\n{'=' * 60}")
    print("Dataset Merge Complete!")
    print(f"{'=' * 60}")
    print(f"  Total: {total}  (Train: {total_train}  Test: {total_test})")
    print(f"\n  Source breakdown:")
    for source_name, counts in sorted(source_stats.items()):
        src_total = counts['train'] + counts['test']
        pct = 100 * src_total / max(total, 1)
        print(f"    {source_name:20s}: {src_total:7d} "
              f"(train={counts['train']}, test={counts['test']}) "
              f"[{pct:5.1f}%]")
    print(f"\n  Camera motion distribution:")
    for motion, count in sorted(motion_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(total, 1)
        print(f"    {motion:15s}: {count:7d} ({pct:5.1f}%)")
    print(f"\n  Shot type distribution:")
    for st, count in sorted(shot_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(total, 1)
        print(f"    {st:20s}: {count:7d} ({pct:5.1f}%)")
    print(f"\n  Output: {args.output_root}")
    print(f"    camera_trajectories/")
    print(f"    person_trajectories/")
    print(f"    train_index.json ({total_train} entries)")
    print(f"    test_index.json ({total_test} entries)")
    if args.compute_norm_stats:
        print(f"    norm_stats.json")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
