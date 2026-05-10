# Text-to-Shot

Text-to-joint person-camera trajectory generation.  
This repository currently maintains only the **Flow Matching** pipeline.

> Note: historical approaches, experiment history, and comparisons are consolidated in `report_v2.md`. This README only keeps the currently supported workflow.

---

## Task Definition

Given a text prompt, generate a joint trajectory of length `T=48`:

- Person trajectory: `(T, Dp)`, with current training commonly using `Dp=3` (`px, py, pz`)
- Camera trajectory: `(T, 6)`, `[tx, ty, tz, azimuth, elevation, roll]`
- Joint vector: `[person_flat, camera_flat]`

---

## Installation

Requirements:

- Python >= 3.10
- CUDA GPU recommended (for training)

Install:

```bash
pip install -e .
```

---

## Quick Start

The default examples use `experiments/flow_matching/configs/v9.yaml`.

### 1) Prepare Data (skip if already available)

Organize your training data as:

- `<data_root>/train_index.json`
- `<data_root>/test_index.json`
- Person/camera `.npy` trajectories referenced by those index files

If you need to rebuild data from E.T. / AMASS / HumanML3D, use the data scripts under `scripts/`.

### 2) Compute Normalization Statistics (recommended)

```bash
python scripts/compute_norm_stats.py \
  --data-root /transfer/merged-v9b \
  --index-file train_index.json \
  --person-dim 3 \
  --camera-dim 6
```

Default output: `/transfer/merged-v9b/norm_stats.json`

### 3) Train

```bash
PYTHONPATH=. python experiments/flow_matching/train.py \
  --config experiments/flow_matching/configs/v9.yaml \
  --device cuda
```

### 4) Inference

```bash
PYTHONPATH=. python experiments/flow_matching/generate.py \
  --checkpoint /transfer/fm-v9b-checkpoints/fm_final.pth \
  --text "A person walks toward camera" \
  --motion dolly-in \
  --shot-type medium-shot \
  --guidance-scale 3.0
```

Optional: enable hard-constraint postprocessing

```bash
--enforce-constraints
```

---

## Output Files

The default output directory is controlled by `paths.output_dir` in config. Typical outputs:

- `fm_person_<tag>.npy`
- `fm_camera_<tag>.npy`
- `fm_joint_<tag>.png`

---

## Key Directories

```text
experiments/flow_matching/
  configs/v9.yaml            # Current primary config
  train.py                   # Training entry point
  generate.py                # Inference entry point
  postprocess_constraints.py # Optional hard-constraint postprocessing

src/
  data/dataset.py            # Data loading and normalization
  models/                    # Core model components

scripts/
  compute_norm_stats.py      # Normalization stats
  preprocess_et_data.py      # Data preprocessing (as needed)
  prepare_amass.py           # Data preparation (as needed)
  prepare_humanml3d.py       # Data preparation (as needed)
  merge_datasets.py          # Dataset merge (as needed)
```

---

## Documentation

- Main report: `report_v2.md`
- Legacy report: `report_v1.md`
