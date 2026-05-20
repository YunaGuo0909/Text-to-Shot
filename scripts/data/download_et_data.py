"""
Download E.T. (Exceptional Trajectories) dataset to /transfer/et-data.

IMPORTANT: This script checks for existing data BEFORE doing anything.
If /transfer/et-data/traj/ already exists, it will NOT download.

Source: https://huggingface.co/datasets/robin-courant/et-data

Usage:
    python scripts/data/download_et_data.py
    python scripts/data/download_et_data.py --download-dir /path/to/et-data
"""

import os
import sys
import argparse
import subprocess


# Key subdirectories that indicate a complete E.T. dataset
REQUIRED_DIRS = ['traj', 'caption']
EXPECTED_DIRS = ['traj', 'caption', 'caption_cam', 'smplh', 'char']


def check_dataset_exists(download_dir: str) -> bool:
    """
    Check if the E.T. dataset already exists at the given path.
    Returns True if the dataset appears complete (has traj/ and caption/).
    """
    if not os.path.isdir(download_dir):
        return False

    for d in REQUIRED_DIRS:
        if not os.path.isdir(os.path.join(download_dir, d)):
            return False

    return True


def print_dataset_status(download_dir: str):
    """Print which subdirectories exist in the dataset."""
    print(f"\nDataset location: {download_dir}")
    print("  Subdirectories found:")
    for d in EXPECTED_DIRS:
        path = os.path.join(download_dir, d)
        if os.path.isdir(path):
            # Count files
            try:
                n = len(os.listdir(path))
            except OSError:
                n = '?'
            print(f"    [OK] {d}/ ({n} entries)")
        else:
            print(f"    [--] {d}/ (not found)")


def main():
    parser = argparse.ArgumentParser(
        description="Download E.T. dataset from Hugging Face."
    )
    parser.add_argument(
        "--download-dir", type=str, default=None,
        help="Target directory (default: /transfer/et-data)",
    )
    parser.add_argument(
        "--repo", type=str, default="robin-courant/et-data",
    )
    parser.add_argument(
        "--skip-untar", action="store_true",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download even if dataset exists.",
    )
    args = parser.parse_args()

    download_dir = args.download_dir or os.environ.get("ET_DATA_DOWNLOAD_DIR", "/transfer/et-data")
    download_dir = os.path.abspath(download_dir)

    # === Check if dataset already exists ===
    if check_dataset_exists(download_dir) and not args.force:
        print(f"[Skip] E.T. dataset already exists at: {download_dir}")
        print_dataset_status(download_dir)
        _print_next_steps(download_dir)
        return

    if os.path.isdir(download_dir) and not args.force:
        # Directory exists but incomplete
        print(f"[Warning] Directory exists but missing required subdirs: {download_dir}")
        print("  Use --force to re-download, or check the directory manually.")
        print_dataset_status(download_dir)
        return

    print(f"Downloading E.T. dataset to: {download_dir}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    os.makedirs(download_dir, exist_ok=True)
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=download_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to {download_dir}")

    untar_script = os.path.join(download_dir, "untar_and_move.sh")
    if not args.skip_untar and os.path.isfile(untar_script):
        print("Running untar_and_move.sh ...")
        subprocess.run(["sh", untar_script], cwd=download_dir, check=False)

    print_dataset_status(download_dir)
    _print_next_steps(download_dir)


def _print_next_steps(download_dir):
    print("\nNext steps:")
    print(f"  1. Preprocess:    python scripts/data/preprocess_et_data.py --et-root {download_dir}")
    print(f"  2. Filter single: python scripts/data/filter_et_single_person.py --data-root /transfer/stc-data")
    print(f"  3. Train:         python train.py --config configs/default.yaml --device cuda --single-person")


if __name__ == "__main__":
    main()
