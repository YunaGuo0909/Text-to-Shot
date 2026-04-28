"""
Download HumanML3D text annotations and build AMASS caption mapping.

HumanML3D (https://github.com/EricGuo5513/HumanML3D) provides text
descriptions for motions sourced from AMASS. This script:

1. Clones the HumanML3D repo (sparse checkout: only texts/ and index.csv)
2. Parses each .txt file to extract captions (text before first '#')
3. Uses index.csv to map motion IDs to AMASS source sequence identifiers
4. Outputs a JSON mapping: AMASS path fragment -> list of captions

Output format (saved to /transfer/humanml3d_captions.json):
{
  "CMU/01/01_01": ["a person walks forward slowly.", "someone walks ahead."],
  "Eyes_Japan_Dataset/hamada/...": ["a person jumps.", ...],
  ...
}

Usage:
    python scripts/download_humanml3d_texts.py
    python scripts/download_humanml3d_texts.py --output /transfer/humanml3d_captions.json
    python scripts/download_humanml3d_texts.py --humanml3d-root /path/to/existing/HumanML3D
"""

import os
import re
import csv
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict


HUMANML3D_REPO = "https://github.com/EricGuo5513/HumanML3D.git"


def clone_humanml3d(target_dir: str) -> bool:
    """
    Clone HumanML3D repo with sparse checkout (only texts/ and index.csv).

    Returns True on success, False on failure.
    """
    if os.path.isdir(os.path.join(target_dir, '.git')):
        print(f"HumanML3D repo already exists at {target_dir}")
        return True

    print(f"Cloning HumanML3D (sparse) into {target_dir} ...")
    os.makedirs(target_dir, exist_ok=True)

    try:
        # Initialize sparse checkout
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--sparse", HUMANML3D_REPO, target_dir],
            check=True, capture_output=True, text=True
        )
        subprocess.run(
            ["git", "-C", target_dir, "sparse-checkout", "set", "HumanML3D/texts", "index.csv"],
            check=True, capture_output=True, text=True
        )
        print("Sparse clone complete.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Sparse clone failed: {e.stderr}")

    # Fallback: full clone (shallow)
    try:
        print("Trying shallow full clone...")
        subprocess.run(
            ["git", "clone", "--depth", "1", HUMANML3D_REPO, target_dir],
            check=True, capture_output=True, text=True
        )
        print("Shallow clone complete.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Clone failed: {e.stderr}")
        return False


def parse_text_file(txt_path: str) -> list:
    """
    Parse a HumanML3D text file. Each line has format:
        caption#annotation1#annotation2#...
    We extract just the caption (before the first #).
    """
    captions = []
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Extract caption (before first #)
                caption = line.split('#')[0].strip()
                if caption:
                    captions.append(caption)
    except Exception as e:
        print(f"  [Warning] Failed to parse {txt_path}: {e}")
    return captions


def load_index_csv(index_path: str) -> dict:
    """
    Load HumanML3D index.csv that maps motion IDs to AMASS source paths.

    Expected CSV format (may vary):
        motion_id, source_path, start_frame, end_frame
    or:
        new_name, source_path, start_frame, end_frame

    Returns dict: motion_id (str, zero-padded 6 digits) -> AMASS path fragment
    """
    mapping = {}
    if not os.path.exists(index_path):
        return mapping

    try:
        with open(index_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                motion_id = row[0].strip()
                source_path = row[1].strip()
                # Skip header rows
                if motion_id.lower() in ('new_name', 'motion_id', 'id', 'name', ''):
                    continue
                # Normalize motion ID to 6-digit zero-padded string
                # Could be "000001" or "M000001" or just "1"
                id_digits = re.sub(r'[^0-9]', '', motion_id)
                if id_digits:
                    motion_key = id_digits.zfill(6)
                else:
                    motion_key = motion_id
                # Clean source path: remove .npz extension and leading slashes
                source_path = source_path.replace('\\', '/')
                source_path = re.sub(r'\.npz$', '', source_path)
                source_path = source_path.strip('/')
                # Remove common prefixes like "pose_data/" or "smplh/"
                for prefix in ['pose_data/', 'smplh/', 'smpl_data/']:
                    if source_path.startswith(prefix):
                        source_path = source_path[len(prefix):]
                        break
                mapping[motion_key] = source_path
    except Exception as e:
        print(f"  [Warning] Failed to load index: {e}")

    return mapping


def find_texts_dir(root: str) -> str:
    """Find the texts/ directory within the HumanML3D repo."""
    # Common locations
    candidates = [
        os.path.join(root, 'HumanML3D', 'texts'),
        os.path.join(root, 'texts'),
        os.path.join(root, 'HumanML3D', 'HumanML3D', 'texts'),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def find_index_csv(root: str) -> str:
    """Find the index.csv within the HumanML3D repo."""
    candidates = [
        os.path.join(root, 'index.csv'),
        os.path.join(root, 'HumanML3D', 'index.csv'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Download HumanML3D text annotations and build AMASS caption mapping.'
    )
    parser.add_argument('--humanml3d-root', type=str, default=None,
                        help='Path to existing HumanML3D repo (skip download).')
    parser.add_argument('--clone-dir', type=str, default='/transfer/HumanML3D',
                        help='Where to clone HumanML3D if not already present.')
    parser.add_argument('--output', type=str, default='/transfer/humanml3d_captions.json',
                        help='Output JSON mapping file path.')
    args = parser.parse_args()

    # Step 1: Get HumanML3D data
    repo_root = args.humanml3d_root or args.clone_dir
    if args.humanml3d_root and os.path.isdir(args.humanml3d_root):
        print(f"Using existing HumanML3D at: {args.humanml3d_root}")
    else:
        success = clone_humanml3d(args.clone_dir)
        if not success:
            print("\nFailed to download HumanML3D.")
            print("You can manually clone it:")
            print(f"  git clone https://github.com/EricGuo5513/HumanML3D.git {args.clone_dir}")
            print("Then re-run with --humanml3d-root")
            return
        repo_root = args.clone_dir

    # Step 2: Find texts directory
    texts_dir = find_texts_dir(repo_root)
    if texts_dir is None:
        print(f"Could not find texts/ directory under {repo_root}")
        return
    print(f"Texts directory: {texts_dir}")

    # Step 3: Parse all text files
    print("Parsing text files...")
    motion_captions = {}  # motion_id -> list of captions
    txt_files = sorted(Path(texts_dir).glob('*.txt'))
    print(f"  Found {len(txt_files)} text files")

    for txt_path in txt_files:
        motion_id = txt_path.stem  # e.g., "000001"
        captions = parse_text_file(str(txt_path))
        if captions:
            motion_captions[motion_id] = captions

    print(f"  Parsed captions for {len(motion_captions)} motions")
    total_captions = sum(len(v) for v in motion_captions.values())
    print(f"  Total captions: {total_captions}")

    # Step 4: Load index mapping (motion_id -> AMASS path)
    index_path = find_index_csv(repo_root)
    if index_path is None:
        print("No index.csv found. Will output motion_id-based mapping only.")
        # Save as-is with motion IDs
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(motion_captions, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(motion_captions)} entries to {args.output}")
        return

    print(f"Index file: {index_path}")
    id_to_amass = load_index_csv(index_path)
    print(f"  Index maps {len(id_to_amass)} motion IDs to AMASS paths")

    # Step 5: Build AMASS path -> captions mapping
    amass_captions = defaultdict(list)
    mapped = 0
    unmapped = 0

    for motion_id, captions in motion_captions.items():
        # Normalize to 6-digit key
        id_key = re.sub(r'[^0-9]', '', motion_id).zfill(6)
        if id_key in id_to_amass:
            amass_path = id_to_amass[id_key]
            # Deduplicate captions for same path
            existing = set(amass_captions[amass_path])
            for cap in captions:
                if cap not in existing:
                    amass_captions[amass_path].append(cap)
                    existing.add(cap)
            mapped += 1
        else:
            unmapped += 1

    print(f"\n  Mapped: {mapped} motions -> {len(amass_captions)} unique AMASS sequences")
    if unmapped > 0:
        print(f"  Unmapped (no index entry): {unmapped}")

    # Step 6: Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(dict(amass_captions), f, indent=2, ensure_ascii=False)

    total_mapped_captions = sum(len(v) for v in amass_captions.values())
    print(f"\nSaved {len(amass_captions)} AMASS sequences with {total_mapped_captions} captions")
    print(f"Output: {args.output}")

    # Print a few examples
    print("\nExample entries:")
    for i, (k, v) in enumerate(list(amass_captions.items())[:5]):
        print(f"  \"{k}\": {v[:2]}{'...' if len(v) > 2 else ''}")


if __name__ == '__main__':
    main()
