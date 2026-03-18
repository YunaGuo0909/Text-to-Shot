"""
Filter E.T. (Exceptional Trajectories) dataset to single-person subset.

Uses caption text to classify samples as single-person vs multi-person.
Reads either preprocessed index files (train_index.json / test_index.json)
or raw E.T. root (caption + caption_cam). Outputs filtered index files
for training on single-person shots.

Usage:
    # From preprocessed index (after preprocess_et_data.py):
    python scripts/filter_et_single_person.py --data-root data

    # From raw E.T. root (no preprocess needed):
    python scripts/filter_et_single_person.py --et-root data/et-data --output-root data

    # Include unknown in single-person subset:
    python scripts/filter_et_single_person.py --data-root data --keep-unknown
"""

import os
import json
import argparse
from typing import List, Dict, Tuple

# Single-person indicators (caption suggests one subject)
SINGLE_PERSON_KEYWORDS = [
    "a person", "the person", "the character", "single subject", "one person",
    "the subject", "the main character", "a character", "one character",
    "a man ", "a woman ", "the man ", "the woman ",
    "person moves", "character moves", "subject moves",
]

# Multi-person indicators (caption suggests two or more)
MULTI_PERSON_KEYWORDS = [
    "two people", "two persons", "both characters", "two characters",
    "two men", "two women", "the two", "they ", "them ",
    "dialogue", "conversation", "between two", "both people",
    "two-shot", "two shot", "over the shoulder",
    "characters ", "people ", "persons ",
]


def classify_person_count(text: str) -> str:
    """
    Classify caption as single-person, multi-person, or unknown.

    Returns:
        "single", "multi", or "unknown"
    """
    if not text or not text.strip():
        return "unknown"

    lower = text.lower().strip()

    # Check multi first (explicit multi overrides single)
    for kw in MULTI_PERSON_KEYWORDS:
        if kw in lower:
            return "multi"

    # Then single
    for kw in SINGLE_PERSON_KEYWORDS:
        if kw in lower:
            return "single"

    return "unknown"


def load_index_entries(data_root: str, split: str) -> List[Dict]:
    """Load entries from preprocessed train_index.json or test_index.json."""
    path = os.path.join(data_root, f"{split}_index.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text_from_et_root(et_root: str, sample_id: str) -> str:
    """Load combined caption text from E.T. raw root (caption_cam + caption)."""
    caption_cam_path = os.path.join(et_root, "caption_cam", f"{sample_id}.txt")
    caption_path = os.path.join(et_root, "caption", f"{sample_id}.txt")
    text = ""
    for p in (caption_cam_path, caption_path):
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    text = (text + " " + f.read().strip()).strip()
            except Exception:
                pass
    return text


def filter_from_index(
    data_root: str,
    output_root: str,
    output_suffix: str = "_single_person",
    keep_unknown: bool = False,
) -> Tuple[int, int, int]:
    """
    Filter using preprocessed index files. Write filtered indices.

    Returns:
        (single_count, multi_count, unknown_count)
    """
    single_count = multi_count = unknown_count = 0
    for split in ("train", "test"):
        entries = load_index_entries(data_root, split)
        if not entries:
            continue

        single_entries = []
        for e in entries:
            text = e.get("text", e.get("description", ""))
            label = classify_person_count(text)
            if label == "single":
                single_count += 1
                single_entries.append(e)
            elif label == "multi":
                multi_count += 1
            else:
                unknown_count += 1
                if keep_unknown:
                    single_entries.append(e)

        out_path = os.path.join(
            output_root,
            f"{split}_index{suffix_to_filename(output_suffix)}.json",
        )
        os.makedirs(output_root, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(single_entries, f, indent=2, ensure_ascii=False)
        print(f"  {split}: {len(single_entries)} / {len(entries)} -> {out_path}")

    return single_count, multi_count, unknown_count


def suffix_to_filename(suffix: str) -> str:
    """e.g. _single_person -> _single_person (for train_index_single_person.json)."""
    return suffix if suffix.startswith("_") else f"_{suffix}"


def filter_from_et_root(
    et_root: str,
    output_root: str,
    output_suffix: str = "_single_person",
    keep_unknown: bool = False,
    split_file_train: str = "full_train_split.txt",
    split_file_test: str = "full_test_split.txt",
) -> Tuple[int, int, int]:
    """
    Build entries from E.T. root (traj dir as source of IDs), classify by caption,
    write filtered index. Entries have id, text, trajectory_path; no trajectories
    are copied (call preprocess_et_data first if you need .npy). This mode
    only produces filtered lists of IDs + text for downstream preprocessing or
    filtering of existing data.
    """
    traj_dir = os.path.join(et_root, "traj")
    if not os.path.exists(traj_dir):
        raise FileNotFoundError(f"E.T. traj directory not found: {traj_dir}")

    all_ids = sorted(
        f[:-4] for f in os.listdir(traj_dir) if f.endswith(".txt")
    )

    # Load splits if present
    train_ids = set()
    test_ids = set()
    for name, path in (
        ("train", os.path.join(et_root, split_file_train)),
        ("test", os.path.join(et_root, split_file_test)),
    ):
        if os.path.exists(path):
            with open(path, "r") as f:
                s = set(line.strip() for line in f if line.strip())
            if name == "train":
                train_ids = s
            else:
                test_ids = s

    single_count = multi_count = unknown_count = 0
    single_by_split = {"train": [], "test": []}

    for sample_id in all_ids:
        text = load_text_from_et_root(et_root, sample_id)
        label = classify_person_count(text)
        entry = {
            "id": sample_id,
            "text": text,
            "trajectory_path": f"trajectories/{sample_id}.npy",
        }
        if label == "single":
            single_count += 1
            if sample_id in test_ids:
                single_by_split["test"].append(entry)
            else:
                single_by_split["train"].append(entry)
        elif label == "multi":
            multi_count += 1
        else:
            unknown_count += 1
            if keep_unknown:
                if sample_id in test_ids:
                    single_by_split["test"].append(entry)
                else:
                    single_by_split["train"].append(entry)

    os.makedirs(output_root, exist_ok=True)
    for split in ("train", "test"):
        out_path = os.path.join(
            output_root,
            f"{split}_index{suffix_to_filename(output_suffix)}.json",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(single_by_split[split], f, indent=2, ensure_ascii=False)
        print(f"  {split}: {len(single_by_split[split])} -> {out_path}")

    return single_count, multi_count, unknown_count


def main():
    parser = argparse.ArgumentParser(
        description="Filter E.T. dataset to single-person subset by caption."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root of preprocessed data (train_index.json, test_index.json).",
    )
    parser.add_argument(
        "--et-root",
        type=str,
        default=None,
        help="Raw E.T. dataset root (traj/, caption/, caption_cam/).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/transfer/stc-data",
        help="Output directory for filtered index files (default: /transfer/stc-data).",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_single_person",
        help="Suffix for index filenames (e.g. train_index_single_person.json).",
    )
    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="Treat unknown as single-person and include in output.",
    )
    args = parser.parse_args()

    if args.data_root and args.et_root:
        print("Use either --data-root or --et-root, not both.")
        return
    if not args.data_root and not args.et_root:
        args.data_root = "/transfer/stc-data"

    print("Filter E.T. to single-person subset")
    print("=" * 50)

    if args.data_root:
        single, multi, unknown = filter_from_index(
            args.data_root,
            args.output_root,
            output_suffix=args.output_suffix,
            keep_unknown=args.keep_unknown,
        )
    else:
        single, multi, unknown = filter_from_et_root(
            args.et_root,
            args.output_root,
            output_suffix=args.output_suffix,
            keep_unknown=args.keep_unknown,
        )

    total = single + multi + unknown
    print()
    print("Summary:")
    print(f"  single-person: {single}")
    print(f"  multi-person:  {multi}")
    print(f"  unknown:      {unknown}")
    print(f"  total:        {total}")
    if total:
        print(f"  single %:      {100 * single / total:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
