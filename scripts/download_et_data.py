"""
Download E.T. (Exceptional Trajectories) dataset to a configurable location.

Use this after cloning the project so the dataset lives under /transfer.
Preprocessing (preprocess_et_data.py) then uses --et-root pointing to the downloaded path.

Source: https://huggingface.co/datasets/robin-courant/et-data

Usage:
    # Default: download to /transfer/et-data，然后执行 untar_and_move.sh；若目录已存在则跳过下载
    PYTHONPATH=. python scripts/download_et_data.py

    # 只下载、不解压
    PYTHONPATH=. python scripts/download_et_data.py --skip-untar

    # Custom location
    PYTHONPATH=. python scripts/download_et_data.py --download-dir /path/to/et-data
"""

import os
import sys
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Download E.T. dataset from Hugging Face to a configurable directory."
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="Target directory for E.T. data (e.g. /otherlocation/transfer/et-data). "
             "Defaults to env ET_DATA_DOWNLOAD_DIR or ./data/et-data.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="robin-courant/et-data",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--skip-untar",
        action="store_true",
        help="Do not run untar_and_move.sh after download (run manually if needed).",
    )
    args = parser.parse_args()

    download_dir = args.download_dir or os.environ.get("ET_DATA_DOWNLOAD_DIR")
    if not download_dir:
        download_dir = "/transfer/et-data"

    download_dir = os.path.abspath(download_dir)

    # 若目标目录 et-data 已存在，则跳过下载
    if os.path.isdir(download_dir):
        print(f"[Skip] {download_dir} already exists — skipping download.")
        print("\nNext steps:")
        print(f"  1. Preprocess: python scripts/preprocess_et_data.py --et-root {download_dir} --output-root /transfer/data")
        print(f"  2. (Optional) Single-person: python scripts/filter_et_single_person.py --data-root /transfer/data")
        print(f"  3. Train: python train.py --config configs/default.yaml --device cuda")
        return

    print(f"E.T. dataset will be downloaded to: {download_dir}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub: pip install huggingface_hub")
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
    elif not args.skip_untar:
        print("No untar_and_move.sh found; if the dataset uses tarballs, run it manually in the download dir.")

    print("\nNext steps:")
    print(f"  1. Preprocess: python scripts/preprocess_et_data.py --et-root {download_dir} --output-root /transfer/data")
    print(f"  2. (Optional) Single-person subset: python scripts/filter_et_single_person.py --data-root /transfer/data")
    print(f"  3. Train: python train.py --config configs/default.yaml --device cuda")


if __name__ == "__main__":
    main()
