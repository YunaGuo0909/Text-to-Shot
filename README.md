# Script-to-Camera

**Generating Cinematic Camera Motion Trajectories from Screenplays via Diffusion Models**

---

## Overview

This project proposes a diffusion-based framework that automatically generates cinematic camera motion trajectories from textual scene descriptions. Given a screenplay excerpt, the system:

1. **Decomposes** the scene into a sequence of cinematic shots (via LLM)
2. **Generates** smooth camera motion trajectories via a text-conditioned diffusion model
3. **Visualizes** the trajectories as 3D animations, parameter curves, and camera path diagrams

```
Screenplay ──▶ Shot Decomposer ──▶ Trajectory Diffusion Model ──▶ 3D Trajectory Visualizer
  (text)          (LLM-based)       (CLIP + DDPM denoiser)        (GIF + curves + 3D paths)
```

Training data is sourced from the [E.T. (Exceptional Trajectories)](https://github.com/robincourant/DIRECTOR) dataset, which provides real film camera trajectories paired with textual descriptions.

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) package manager (or pip)

### Installation

```bash
git clone https://github.com/YunaGuo0909/Text-to-Shot.git
cd Text-to-Shot
uv sync
```

### Run Demo (no trained model needed)

```bash
# 2D parameter curves + multi-shot grid
PYTHONPATH=. python generate_storyboard.py --demo

# 3D camera trajectory animation (GIF)
PYTHONPATH=. python visualize_3d.py --demo --motion orbit

# All 9 motion types comparison
PYTHONPATH=. python visualize_3d.py --compare-motions
```
(The `--demo` run also generates single-person camera-view GIFs; see Single-Person Dynamic Storyboard below.)

### Demo Outputs

| Output | Description |
|---|---|
| `outputs/demo_trajectory_storyboard.png` | 6-panel grid with per-shot trajectory curves |
| `outputs/demo_trajectory_detail.png` | Detailed parameter evolution for one shot |
| `outputs/demo_camera_path.png` | Top-down camera path |
| `outputs/test_orbit.gif` | 3D animated camera trajectory (GIF) |
| `outputs/test_orbit_static.png` | 3D static view with camera frustums |
| `outputs/all_motions_comparison.png` | 9 motion types side-by-side comparison |
| `outputs/demo_camera_view_shot*.gif` | What the camera sees (person as cube) per shot |

---

## Single-Person Dynamic Storyboard

The pipeline can be extended to **single-person dynamic storyboard**: in addition to camera trajectory, the system generates a **person motion trajectory** and renders **what the camera sees** (camera view) with the person represented as a cube.

- **Shot Decomposer**: Each shot can include `person_motion_description` (e.g. "person walks toward camera", "character stands still"), produced by the LLM or set manually.
- **Person trajectory**: A rule-based `PersonTrajectoryGenerator` turns the description into a (T, 3) world-space trajectory. No skeletal animation—the person is a single point or cube.
- **Camera view renderer**: Given camera trajectory and person trajectory, the module renders a 2D view from the camera at each frame (person drawn as a rectangle/cube) and outputs a GIF.

Use `--with-person` and `--render-camera-view` when running `generate_storyboard.py` (e.g. with `--demo` or with `--scene` + `--checkpoint`). The demo run already generates `outputs/demo_camera_view_shot*.gif`.

---

## E.T. Dataset: Single-Person Subset

To train or evaluate on **single-person** shots only, use the filter script on the E.T. dataset. It classifies samples by caption text (single vs multi-person keywords) and writes filtered index files.

```bash
# After preprocessing E.T. (preprocess_et_data.py), filter to single-person index
PYTHONPATH=. python scripts/filter_et_single_person.py --data-root data

# Or from raw E.T. root (no preprocess)
PYTHONPATH=. python scripts/filter_et_single_person.py --et-root data/et-data --output-root data

# Include unknown captions in the single-person set
PYTHONPATH=. python scripts/filter_et_single_person.py --data-root data --keep-unknown
```

Outputs: `data/train_index_single_person.json`, `data/test_index_single_person.json`. Point your training data loader at these index files to use the single-person subset.

---

## Project Structure

```
Text-to-Shot/
├── configs/
│   └── default.yaml                 # Model, training & trajectory configuration
├── src/
│   ├── models/                      # Neural network modules
│   │   ├── diffusion.py             # Gaussian diffusion process (DDPM)
│   │   ├── denoiser.py              # Temporal Transformer denoiser
│   │   ├── text_encoder.py          # Frozen CLIP text encoder wrapper
│   │   ├── film.py                  # FiLM conditioning layer
│   │   └── interaction.py           # Temporal smoothing & inter-shot coherence
│   ├── pipeline/                    # Generation pipeline
│   │   ├── shot_decomposer.py       # LLM-based scene → shot decomposition
│   │   ├── storyboard_generator.py  # Multi-shot trajectory generation pipeline
│   │   ├── camera_trajectory.py     # Rule-based camera trajectory generation
│   │   ├── person_trajectory.py     # Rule-based person (single) trajectory from text
│   │   ├── camera_view_renderer.py # Render camera view (person as cube) → GIF
│   │   └── storyboard_renderer.py   # 2D trajectory visualization & rendering
│   ├── data/
│   │   └── dataset.py               # Camera trajectory dataset & dataloader
│   └── utils/
│       ├── toric.py                 # Toric camera parameterization utilities
│       └── smpl_utils.py            # Camera & rotation utility functions
├── scripts/
│   ├── download_et_data.py         # Download E.T. from Hugging Face (e.g. to /otherlocation/transfer)
│   ├── preprocess_et_data.py        # E.T. dataset → training data preprocessing
│   ├── filter_et_single_person.py   # Filter E.T. index to single-person subset by caption
│   └── verify_data.py               # Data & model forward pass verification
├── train.py                         # Training script (CLIP text conditioning)
├── generate_storyboard.py           # Inference: text → trajectory + visualization
├── evaluate.py                      # Quantitative evaluation on test set
├── visualize_3d.py                  # 3D camera trajectory animation (GIF/PNG)
├── pyproject.toml                   # Dependencies
└── README.md
```

---

## Technical Pipeline

### Stage 1 — Shot Decomposition (`shot_decomposer.py`)

Uses an LLM (GPT-4 / local model) to break a screenplay into a structured shot list. Each shot specifies:

- **Shot type**: close-up, medium-shot, wide-shot, over-the-shoulder, two-shot
- **Camera motion**: static, dolly-in, dolly-out, pan-left, pan-right, crane-up, crane-down, track, orbit
- **Emotional tone**: tense, calm, dramatic, intimate, etc.
- **Duration hint**: estimated shot length in seconds
- **Person motion description** (optional): how the character moves in the shot (e.g. "person walks toward camera", "character stands still") for single-person dynamic storyboard

### Stage 2 — Camera Trajectory Generation (Diffusion Model)

A **Gaussian Diffusion Model (DDPM)** generates smooth camera trajectories conditioned on text, shot type, and camera motion type.

**Data representation** — Each trajectory is a sequence of T frames, each a 6D camera state:

```
x_C(t) = (tx, ty, tz, azimuth, elevation, roll)  ∈ R^6    for t = 1, ..., T
```

- `tx, ty, tz`: Camera position in world coordinates
- `azimuth` (θ): Yaw rotation (left-right)
- `elevation` (φ): Pitch rotation (up-down)
- `roll` (ψ): Dutch angle rotation

The trajectory is flattened to a (T × 6)-dim vector for the diffusion process.

**Network architecture** — `CameraTrajectoryDenoiser`:

- Per-frame linear projection: 6D → hidden (256-dim)
- Learnable temporal positional encoding
- N Temporal Transformer blocks, each containing:
  - Multi-head self-attention across time axis
  - FiLM-conditioned feed-forward network
- Per-frame linear projection: hidden → 6D

**Conditioning signals**:

| Signal | Method | Dimension |
|--------|--------|-----------|
| Text (scene description) | Frozen CLIP encoder | 512 |
| Diffusion timestep | Sinusoidal + MLP | 128 |
| Shot type | Learnable embedding | 64 |
| Camera motion type | Learnable embedding | 64 |

### Stage 3 — Visualization

**2D Visualization** (`storyboard_renderer.py`):
1. Multi-shot grid with per-shot parameter curves
2. Detailed 6-parameter evolution view
3. Top-down camera path (θ vs φ)

**3D Visualization** (`visualize_3d.py`):
1. Animated GIF: camera moves through 3D space with frustum, trail, and real-time parameter display
2. Static 3D view: camera path with frustums at keyframes (for paper figures)
3. Motion type comparison: 9 motion types in a single figure

---

## Training Data

Training data is derived from the **E.T. (Exceptional Trajectories)** dataset:

| Component | Source | Usage |
|---|---|---|
| Camera trajectories | `traj/*.txt` (3×4 extrinsic matrices) | Converted to 6D camera state |
| Text descriptions | `caption_cam/*.txt` | CLIP text conditioning |
| Train/test split | `full_train_split.txt` / `full_test_split.txt` | Official E.T. splits |

**Download E.T.** (`scripts/download_et_data.py`): After each clone, download the dataset to a fixed location (e.g. outside the repo) so you don’t re-download into the project:

```bash
# Download to a fixed path (e.g. /otherlocation/transfer/et-data)
PYTHONPATH=. python scripts/download_et_data.py --download-dir /otherlocation/transfer/et-data

# Or set once: export ET_DATA_DOWNLOAD_DIR=/otherlocation/transfer/et-data
# Then: PYTHONPATH=. python scripts/download_et_data.py
```

Then run preprocessing with `--et-root` pointing to that path.

**Preprocessing** (`scripts/preprocess_et_data.py`):
1. Parse 3×4 extrinsic matrices → extract rotation (Euler angles) and translation
2. Convert to 6D state: `(tx, ty, tz, azimuth, elevation, roll)`
3. Resample all trajectories to 48 frames (2s @ 24fps)
4. Auto-classify camera motion type from text via keyword matching
5. Output: `train_index.json` (103k samples) + `test_index.json` (11k samples) + `.npy` files

**Data format** (`data/train_index.json`):

```json
[
  {
    "id": "2015_YmoTJu2iOfc_00039_00000",
    "text": "The camera pushes in slowly toward the subject...",
    "shot_type": "medium-shot",
    "camera_motion": "dolly-in",
    "trajectory_path": "trajectories/2015_YmoTJu2iOfc_00039_00000.npy"
  }
]
```

Each `.npy` file contains a `(48, 6)` float32 NumPy array.

---

## Training

```bash
# With CLIP text conditioning
PYTHONPATH=. python train.py --config configs/default.yaml --device cuda

# Without CLIP (random embeddings, for quick testing)
PYTHONPATH=. python train.py --config configs/default.yaml --device cuda --no-clip

# Resume from checkpoint
PYTHONPATH=. python train.py --config configs/default.yaml --device cuda --resume checkpoints/checkpoint_epoch50.pth
```

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Trajectory frames (T) | 48 (2s @ 24fps) |
| Camera state dimension | 6 |
| Total diffusion dim | 288 (48 × 6) |
| Diffusion timesteps | 1000 |
| Beta schedule | Cosine |
| Hidden dim | 256 |
| Transformer layers | 6 |
| Attention heads | 4 |
| Batch size | 64 |
| Learning rate | 1e-4 |
| Epochs | 500 |

---

## Inference

```bash
# Generate trajectory from text (requires trained checkpoint)
PYTHONPATH=. python generate_storyboard.py \
  --scene "Camera slowly dollies in toward two people at a table" \
  --checkpoint checkpoints/checkpoint_final.pth \
  --motion dolly-in --shot-type medium-shot

# 3D animated visualization
PYTHONPATH=. python visualize_3d.py \
  --scene "Camera orbits around two people" \
  --checkpoint checkpoints/checkpoint_final.pth \
  --motion orbit
```

Outputs: parameter curves (PNG) + 3D animation (GIF) + 3D static view (PNG).

---

## Evaluation

```bash
PYTHONPATH=. python evaluate.py --checkpoint checkpoints/checkpoint_final.pth --device cuda
```

| Metric | Description |
|---|---|
| MSE / MAE | Reconstruction error (generated vs ground truth) |
| Per-parameter MSE | Error breakdown for tx, ty, tz, azimuth, elevation, roll |
| Jerk | Trajectory smoothness (lower = smoother) |
| Jerk ratio | Generated / ground truth smoothness (closer to 1.0 = better) |
| Path length | Total camera displacement |

---

## Camera Motion Types

| Type | Description | 3D Movement |
|------|-------------|-------------|
| `static` | Fixed camera | No movement |
| `dolly-in` | Push toward subject | Forward along Z axis |
| `dolly-out` | Pull away from subject | Backward along Z axis |
| `pan-left` | Rotate camera left | Azimuth decreases |
| `pan-right` | Rotate camera right | Azimuth increases |
| `crane-up` | Raise camera | Upward along Y axis |
| `crane-down` | Lower camera | Downward along Y axis |
| `track` | Lateral follow | Sideways translation |
| `orbit` | Circle around subject | Circular XY path |

---

## Acknowledgments

- E.T. dataset: Wang et al. (2024), "E.T. the Exceptional Trajectories"
- DDPM: Ho, Jain & Abbeel (2020), NeurIPS
- CLIP: Radford et al. (2021), ICML
- FiLM conditioning: Perez et al. (2018), AAAI
- Toric camera space: Lino & Christie (2015), ACM TOG
- DanceCamera3D: Wang et al. (2024), AAAI
- MDM (Human Motion Diffusion): Tevet et al. (2022), ICLR
