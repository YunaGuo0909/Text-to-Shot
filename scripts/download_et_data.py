"""
Download E.T. (Exceptional Trajectories) dataset to /transfer/et-data.

Checks if the dataset already exists before downloading.
Source: https://huggingface.co/datasets/robin-courant/et-data

Usage:
    python scripts/download_et_data.py
    python scripts/download_et_data.py --download-dir /path/to/et-data
    python scripts/download_et_data.py --skip-untar
"""

import os
import sys
import argparse
import subprocess


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
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--skip-untar", action="store_true",
        help="Do not run untar_and_move.sh after download.",
    )
    args = parser.parse_args()

    download_dir = args.download_dir or os.environ.get("ET_DATA_DOWNLOAD_DIR", "/transfer/et-data")
    download_dir = os.path.abspath(download_dir)

    # Check if dataset already exists
    if os.path.isdir(download_dir):
        # Verify it has actual data (not just an empty dir)
        has_traj = os.path.isdir(os.path.join(download_dir, 'traj'))
        has_files = len(os.listdir(download_dir)) > 0
        if has_traj or has_files:
            print(f"[Skip] Dataset already exists at: {download_dir}")
            _print_next_steps(download_dir)
            return
        else:
            print(f"[Info] Directory exists but appears empty: {download_dir}")

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

    # Run untar if available
    untar_script = os.path.join(download_dir, "untar_and_move.sh")
    if not args.skip_untar and os.path.isfile(untar_script):
        print("Running untar_and_move.sh ...")
        subprocess.run(["sh", untar_script], cwd=download_dir, check=False)
    elif not args.skip_untar:
        print("No untar_and_move.sh found; if the dataset uses tarballs, extract manually.")

    _print_next_steps(download_dir)


def _print_next_steps(download_dir):
    print("\nNext steps:")
    print(f"  1. Preprocess: python scripts/preprocess_et_data.py --et-root {download_dir}")
    print(f"  2. (Optional) Filter: python scripts/filter_et_single_person.py --data-root /transfer/stc-data")
    print(f"  3. Train: python train.py --config configs/default.yaml --device cuda")


if __name__ == "__main__":
    main()
