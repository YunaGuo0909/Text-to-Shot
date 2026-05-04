# Text-to-Shot (Script-to-Camera)

Joint person-camera trajectory generation from text.

This repository contains three active pipelines:

1. **Baseline joint DDPM** (`train.py`, `generate.py`, `evaluate.py`)
2. **Flow Matching experiment** (`experiments/flow_matching`)
3. **Two-stage experiment** (`experiments/two_stage`)

The project is data-centric and includes scripts to prepare E.T., AMASS, HumanML3D, and DanceCamera3D into one unified training format.

---

## Current trajectory format

- **Person trajectory**: `(T, 5)` = `[px, py, pz, sin_yaw, cos_yaw]`
- **Camera trajectory**: `(T, 6)` = `[tx, ty, tz, azimuth, elevation, roll]`
- **Joint vector**: `[person_flat, camera_flat]` with `T=48` by default

> Note: old `(T, 3)` person files are still supported and padded to `(T, 5)` in `src/data/dataset.py`.

---

## Repository layout

```text
Text-to-Shot/
├── configs/
│   └── default.yaml
├── src/
│   ├── data/
│   │   └── dataset.py
│   ├── models/
│   │   ├── denoiser.py
│   │   ├── diffusion.py
│   │   ├── text_encoder.py
│   │   └── film.py
│   └── utils/
├── scripts/
│   ├── download_et_data.py
│   ├── preprocess_et_data.py
│   ├── filter_et_single_person.py
│   ├── compute_norm_stats.py
│   ├── prepare_amass.py
│   ├── prepare_humanml3d.py
│   ├── prepare_dancecamera3d.py
│   └── merge_datasets.py
├── experiments/
│   ├── flow_matching/
│   └── two_stage/
├── train.py
├── generate.py
├── evaluate.py
├── Text-to-Shot_Pipeline.ipynb
└── report_v2.md
```

---

## Installation

Requirements:

- Python >= 3.10
- CUDA GPU recommended for training

Install:

```bash
pip install -e .
```

---

## Quick start (baseline E.T. pipeline)

All default configs/scripts assume data under `/transfer`.

### 1) Download E.T.

```bash
python scripts/download_et_data.py
```

Default target: `/transfer/et-data` (skips download if core folders already exist).

### 2) Preprocess E.T. into training format

```bash
python scripts/preprocess_et_data.py \
  --et-root /transfer/et-data \
  --output-root /transfer/stc-data \
  --num-frames 48
```

Output:

- `/transfer/stc-data/camera_trajectories/*.npy`
- `/transfer/stc-data/person_trajectories/*.npy`
- `/transfer/stc-data/train_index.json`
- `/transfer/stc-data/test_index.json`

### 3) (Optional) filter to single-person subset

```bash
python scripts/filter_et_single_person.py --data-root /transfer/stc-data
```

Creates:

- `train_index_single_person.json`
- `test_index_single_person.json`

### 4) Compute normalization stats (recommended)

```bash
python scripts/compute_norm_stats.py \
  --data-root /transfer/stc-data \
  --index-file train_index.json
```

Writes `/transfer/stc-data/norm_stats.json`.

---

## Optional data augmentation pipeline

### Prepare AMASS synthetic camera pairs

```bash
python scripts/prepare_amass.py \
  --amass-root /transfer/amass \
  --output-root /transfer/amass-stc-data
```

### Prepare HumanML3D-derived pairs

```bash
python scripts/prepare_humanml3d.py \
  --amass-root /transfer/amassdata \
  --humanml3d-root /transfer/HumanML3D \
  --output-root /transfer/humanml3d-stc-data-v7
```

### Prepare DanceCamera3D pairs

```bash
python scripts/prepare_dancecamera3d.py \
  --data-root /transfer/dancecamera3d \
  --output-root /transfer/dance-stc-data
```

### Merge multiple prepared datasets

```bash
python scripts/merge_datasets.py \
  --sources /transfer/stc-data /transfer/amass-stc-data /transfer/dance-stc-data \
  --output-root /transfer/merged-stc-data \
  --compute-norm-stats
```

---

## Training

### 1) Baseline joint DDPM

```bash
# Full set
python train.py --config configs/default.yaml --device cuda

# Single-person subset
python train.py --config configs/default.yaml --device cuda --single-person

# Resume
python train.py --config configs/default.yaml --device cuda \
  --resume /transfer/stc-checkpoints/stc_epoch50.pth
```

### 2) Flow Matching experiment

```bash
PYTHONPATH=. python experiments/flow_matching/train.py \
  --config experiments/flow_matching/configs/default.yaml \
  --device cuda
```

### 3) Two-stage experiment

```bash
# Stage 1: text -> person
PYTHONPATH=. python experiments/two_stage/train_stage1.py \
  --config experiments/two_stage/configs/stage1.yaml --device cuda

# Stage 2: text + person -> camera
PYTHONPATH=. python experiments/two_stage/train_stage2.py \
  --config experiments/two_stage/configs/stage2.yaml --device cuda
```

---

## Inference

### Baseline DDPM

```bash
python generate.py \
  --checkpoint /transfer/stc-checkpoints/stc_final.pth \
  --text "A person walks toward camera" \
  --motion dolly-in \
  --shot-type medium-shot \
  --guidance-scale 3.0 \
  --ddim --ddim-steps 50
```

### Flow Matching

```bash
PYTHONPATH=. python experiments/flow_matching/generate.py \
  --checkpoint /transfer/fm-v8-checkpoints/fm_final.pth \
  --text "A person walks toward camera" \
  --motion dolly-in \
  --guidance-scale 3.0
```

### Two-stage

```bash
PYTHONPATH=. python experiments/two_stage/generate.py \
  --stage1-ckpt /transfer/two-stage-checkpoints/stage1/stage1_final.pth \
  --stage2-ckpt /transfer/two-stage-checkpoints/stage2/stage2_final.pth \
  --text "A person walks toward camera" \
  --motion dolly-in \
  --ddim --ddim-steps 50
```

Typical outputs are saved as tagged files in the configured output directory, e.g.:

- `gen_person_<tag>.npy` / `fm_person_<tag>.npy`
- `gen_camera_<tag>.npy` / `fm_camera_<tag>.npy`
- `gen_joint_<tag>.png` / `fm_joint_<tag>.png`

---

## Evaluation

`evaluate.py` currently evaluates the baseline joint DDPM checkpoints:

```bash
python evaluate.py \
  --checkpoint /transfer/stc-checkpoints/stc_final.pth \
  --device cuda \
  --single-person
```

Metrics include:

- person/camera MSE and MAE
- person/camera jerk
- path length
- camera-person distance statistics

Results are saved to `evaluation_results.json` in the configured output directory.

---

## Default paths in configs

### Baseline (`configs/default.yaml`)

- data root: `/transfer/stc-data`
- checkpoints: `/transfer/stc-checkpoints`
- outputs: `/transfer/stc-outputs`
- logs: `/transfer/stc-logs`

### Flow Matching (`experiments/flow_matching/configs/default.yaml`)

- data root: `/transfer/merged-v8`
- checkpoints: `/transfer/fm-v8-checkpoints`
- outputs: `/transfer/fm-v8-outputs`
- logs: `/transfer/fm-v8-logs`

### Two-stage

- stage1 ckpt: `/transfer/two-stage-checkpoints/stage1`
- stage2 ckpt: `/transfer/two-stage-checkpoints/stage2`
- outputs: `/transfer/two-stage-outputs`

---

## Documentation and reports

- Main report: `report_v2.md`
- Previous report: `report_v1.md`
- Notebook: `Text-to-Shot_Pipeline.ipynb`
- Additional analysis: `docs/v7_yaw_failure_analysis.md`

---

## References

- Courant et al., E.T. the Exceptional Trajectories (ECCV 2024)
- Ho et al., Denoising Diffusion Probabilistic Models (NeurIPS 2020)
- Ho and Salimans, Classifier-Free Guidance (NeurIPS Workshops 2021)
- Lipman et al., Flow Matching for Generative Modeling (ICLR 2023)
- Radford et al., CLIP (ICML 2021)
