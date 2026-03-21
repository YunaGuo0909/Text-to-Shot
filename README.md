# Script-to-Camera

**Joint Person-Camera Trajectory Generation from Text via Dual-Branch Diffusion**

---

## Overview

Given a text scene description, this system **simultaneously generates** a person's 3D motion trajectory and a cinematic camera trajectory — using a dual-branch diffusion model with cross-attention.

```
Text Description ──▶ CLIP Encoder ──▶ Dual-Branch Diffusion ──▶ Person Trajectory (T, 3)
                                       ├─ Person Branch ◄──►─┤     +
                                       └─ Camera Branch ◄──►─┘   Camera Trajectory (T, 6)
```

**Key distinction from prior work:** E.T./DIRECTOR (Wang et al., 2024) generates camera trajectories *conditioned on pre-given character motion* — the person's movement must already exist as input. Our system requires **no pre-existing character data**: both person and camera trajectories are generated from text alone. This enables fully automated scene generation without an upstream motion capture or motion synthesis step.

Training data: [E.T. (Exceptional Trajectories)](https://github.com/robincourant/DIRECTOR) dataset — real film camera trajectories + SMPL-H character data.

---

## Project Structure

```
Text-to-Shot/
├── configs/
│   └── default.yaml                 # Model, training & path configuration
├── src/
│   ├── models/
│   │   ├── denoiser.py              # JointTrajectoryDenoiser (dual-branch Transformer + cross-attention)
│   │   ├── diffusion.py             # Gaussian diffusion (DDPM)
│   │   ├── text_encoder.py          # Frozen CLIP text encoder
│   │   └── film.py                  # FiLM conditioning layer
│   ├── data/
│   │   └── dataset.py               # JointTrajectoryDataset (person + camera paired)
│   └── utils/
│       ├── toric.py                 # Toric camera parameterization
│       └── smpl_utils.py            # Rotation & trajectory utilities
├── scripts/
│   ├── download_et_data.py          # Download E.T. dataset (with existence check)
│   ├── preprocess_et_data.py        # E.T. → joint training data (person + camera .npy)
│   ├── filter_et_single_person.py   # Filter to single-person subset
│   └── verify_data.py               # Data & model sanity check
├── train.py                         # Training loop
├── evaluate.py                      # Quantitative evaluation
├── generate.py                      # Inference: text → joint trajectory + visualization
└── pyproject.toml
```

---

## Architecture

### Dual-Branch Transformer Denoiser

The core model (`JointTrajectoryDenoiser`) processes person and camera trajectories through parallel Transformer branches with cross-attention:

```
Input: y_t = [person_flat (T×3), camera_flat (T×6)]   (noisy joint trajectory)

  ┌─────────────────────────────────────────────────┐
  │              Conditioning                        │
  │  CLIP text (512) + timestep (128)               │
  │  + shot_type (64) + motion_type (64)            │
  └──────────────────┬──────────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────────┐
  │          × N Dual-Branch Blocks                  │
  │                                                  │
  │  Person Branch          Camera Branch            │
  │  ┌──────────┐          ┌──────────┐             │
  │  │Self-Attn │          │Self-Attn │             │
  │  └────┬─────┘          └────┬─────┘             │
  │       │    Cross-Attention  │                    │
  │       │◄───────────────────►│                    │
  │  ┌────▼─────┐          ┌────▼─────┐             │
  │  │FiLM + FF │          │FiLM + FF │             │
  │  └──────────┘          └──────────┘             │
  └──────────────────────────────────────────────────┘

Output: y_0 = [person_pred (T×3), camera_pred (T×6)]  (denoised joint trajectory)
```

### Data Representation

| Component | Dimension | Description |
|-----------|-----------|-------------|
| Person trajectory | (T, 3) | Root position (px, py, pz) from SMPL-H `transl` |
| Camera trajectory | (T, 6) | (tx, ty, tz, azimuth, elevation, roll) |
| Joint vector | T×3 + T×6 = 432 | Concatenated, flattened for diffusion |

---

## Setup

### Prerequisites

- Python ≥ 3.10
- CUDA GPU (for training)

### Installation

```bash
git clone <repo-url>
cd Text-to-Shot
pip install -e .
```

---

## Data Preparation

All data lives under `/transfer/` on the training machine.

### 1. Download E.T. Dataset

```bash
python scripts/download_et_data.py
```

Downloads to `/transfer/et-data`. **Skips automatically if data already exists** (checks for `traj/` and `caption/` directories).

### 2. Preprocess

Extracts paired person + camera trajectories:

```bash
python scripts/preprocess_et_data.py --et-root /transfer/et-data --output-root /transfer/stc-data
```

- Camera: 3×4 extrinsics → 6D state (tx, ty, tz, azimuth, elevation, roll)
- Person: SMPL-H `.pkl` → root translation `transl` (T, 3)
- All trajectories resampled to 48 frames (2s @ 24fps)

Output:
```
/transfer/stc-data/
├── camera_trajectories/*.npy   (T, 6) per sample
├── person_trajectories/*.npy   (T, 3) per sample
├── train_index.json
└── test_index.json
```

### 3. Filter Single-Person Subset

```bash
python scripts/filter_et_single_person.py --data-root /transfer/stc-data
```

Outputs `train_index_single_person.json` and `test_index_single_person.json`.

### 4. Verify

```bash
python scripts/verify_data.py
```

---

## Training

```bash
# Single-person subset with CLIP conditioning
python train.py --config configs/default.yaml --device cuda --single-person

# Without CLIP (quick test)
python train.py --config configs/default.yaml --device cpu --no-clip

# Resume
python train.py --config configs/default.yaml --device cuda --single-person \
  --resume /transfer/stc-checkpoints/stc_epoch50.pth
```

Checkpoints saved to `/transfer/stc-checkpoints/stc_epoch*.pth`.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Frames (T) | 48 (2s @ 24fps) |
| Person dim | 3 |
| Camera dim | 6 |
| Joint diffusion dim | 432 (48×9) |
| Diffusion timesteps | 1000 (cosine schedule) |
| Hidden dim | 256 |
| Dual-branch layers | 6 |
| Attention heads | 4 |
| Batch size | 64 |
| Learning rate | 1e-4 (AdamW) |
| Epochs | 500 |

---

## Inference

```bash
python generate.py \
  --checkpoint /transfer/stc-checkpoints/stc_final.pth \
  --text "A person walks toward camera" \
  --motion dolly-in --shot-type medium-shot
```

Outputs to `/transfer/stc-outputs/`:
- `gen_person.npy` — person trajectory (48, 3)
- `gen_camera.npy` — camera trajectory (48, 6)
- `gen_joint.png` — visualization (3D paths + parameter curves)

---

## Evaluation

```bash
python evaluate.py \
  --checkpoint /transfer/stc-checkpoints/stc_final.pth \
  --device cuda --single-person
```

| Metric | Description |
|--------|-------------|
| Person MSE / MAE | Person trajectory reconstruction error |
| Camera MSE / MAE | Camera trajectory reconstruction error |
| Person jerk | Person motion smoothness |
| Camera jerk | Camera motion smoothness |
| Person-camera distance | Average person-to-camera distance (coordination) |

---

## Key Paths

| Content | Path |
|---------|------|
| E.T. raw dataset | `/transfer/et-data` |
| Preprocessed data | `/transfer/stc-data` |
| Checkpoints | `/transfer/stc-checkpoints` |
| Outputs | `/transfer/stc-outputs` |

---

## References

- **E.T. / DIRECTOR**: Courant et al. (2024), "E.T. the Exceptional Trajectories" — dataset + camera diffusion model that generates camera trajectories *given* character motion as input
- **DDPM**: Ho, Jain & Abbeel (2020), NeurIPS — diffusion framework
- **CLIP**: Radford et al. (2021), ICML — text encoder
- **FiLM**: Perez et al. (2018), AAAI — conditioning layer
- **MDM**: Tevet et al. (2022), ICLR — motion diffusion model
- **Toric Space**: Lino & Christie (2015), ACM TOG — camera parameterization
