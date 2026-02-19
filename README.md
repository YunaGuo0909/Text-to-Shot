# Script-to-Camera

**Generating Cinematic Camera Motion Trajectories from Screenplays via Diffusion Models**

---

## Overview

This project proposes a diffusion-based framework that automatically generates cinematic camera motion trajectories from textual scene descriptions. Given a screenplay excerpt, the system:

1. **Decomposes** the scene into a sequence of cinematic shots (via LLM)
2. **Generates** smooth camera motion trajectories in Toric parameter space (via diffusion model)
3. **Visualizes** the trajectories as parameter curves, camera path diagrams, and multi-shot grids

```
Screenplay ──▶ Shot Decomposer ──▶ Trajectory Diffusion Model ──▶ Trajectory Visualizer
  (text)          (LLM-based)         (Toric space DDPM)           (curves + paths)
```

Training data is constructed by extracting camera parameters from real film shots sourced from **ShotDeck**, bridging computational cinematography and generative AI.

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone https://github.com/YunaGuo0909/Text-to-Shot.git
cd Text-to-Shot
uv sync
```

### Run Demo

```bash
# Generate trajectories with rule-based motion profiles (no trained model needed)
uv run python generate_storyboard.py --demo
```

### Output

The demo generates three visualizations:

| Output File | Description |
|---|---|
| `outputs/demo_trajectory_storyboard.png` | 6-panel grid with per-shot trajectory curves |
| `outputs/demo_trajectory_detail.png` | Detailed Toric parameter evolution for one shot |
| `outputs/demo_camera_path.png` | Top-down camera path in Toric space |

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
│   │   ├── film.py                  # FiLM conditioning layer
│   │   └── interaction.py           # Temporal smoothing & inter-shot coherence
│   ├── pipeline/                    # Generation pipeline
│   │   ├── shot_decomposer.py       # LLM-based scene → shot decomposition
│   │   ├── storyboard_generator.py  # Multi-shot trajectory generation pipeline
│   │   ├── camera_trajectory.py     # Rule-based camera trajectory generation
│   │   └── storyboard_renderer.py   # Trajectory visualization & rendering
│   ├── data/
│   │   └── dataset.py               # Camera trajectory dataset loading
│   └── utils/
│       ├── toric.py                 # Toric camera parameterization utilities
│       └── smpl_utils.py            # Camera & rotation utility functions
├── train.py                         # Model training script
├── generate_storyboard.py           # Trajectory generation entry point
├── pyproject.toml                   # Dependencies (uv)
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

### Stage 2 — Camera Trajectory Generation (Diffusion Model)

A **Gaussian Diffusion Model** generates smooth camera trajectories in Toric parameter space.

**Data representation** — Each trajectory is a sequence of T frames, each a 6-dim Toric state:

```
x_C(t) = (pA_x, pA_y, pB_x, pB_y, θ, φ)  ∈ R^6    for t = 1, ..., T
```

- `pA_x, pA_y`: Normalized screen position of reference point A
- `pB_x, pB_y`: Normalized screen position of reference point B
- `θ` (theta): Camera azimuth (yaw) in Toric space
- `φ` (phi): Camera elevation (pitch) in Toric space

The trajectory is flattened to a (T × 6)-dim vector for the diffusion process.

**Network architecture** — `CameraTrajectoryDenoiser`:

- Per-frame linear projection: Toric (6-dim) → hidden (256-dim)
- Learnable temporal positional encoding
- N Temporal Transformer blocks, each containing:
  - Multi-head self-attention across time axis
  - FiLM-conditioned feed-forward network
- Per-frame linear projection: hidden → Toric (6-dim)

**Conditioning signals**:

| Signal | Method | Dimension |
|--------|--------|-----------|
| Text (scene description) | CLIP embedding | 512 |
| Diffusion timestep | Sinusoidal + MLP | 128 |
| Shot type | Learnable embedding | 64 |
| Camera motion type | Learnable embedding | 64 |

### Stage 3 — Visualization (`storyboard_renderer.py`)

Renders trajectories as:

1. **Multi-shot grid**: Each panel shows θ/φ curves and screen position evolution
2. **Parameter detail view**: All 6 Toric parameters with keyframe markers
3. **Top-down camera path**: θ vs φ plot showing spatial camera movement across shots

---

## Training

### Data Preparation

Training data is extracted from film shots on ShotDeck:

1. **Scrape** film shot clips/frames from ShotDeck
2. **Estimate camera parameters** for each frame using camera estimation methods
3. **Extract Toric parameters**: Convert extrinsics to (pA_x, pA_y, pB_x, pB_y, θ, φ)
4. **Annotate** with shot type, camera motion type, and text descriptions

Expected data format (`data/train_index.json`):

```json
[
  {
    "id": "shot_001",
    "text": "Medium shot, camera slowly dollies in toward two people at a table",
    "shot_type": "medium-shot",
    "camera_motion": "dolly-in",
    "trajectory_path": "trajectories/shot_001.npy"
  }
]
```

Each `.npy` file contains a `(T, 6)` NumPy array of Toric camera states.

### Run Training

```bash
uv run python train.py --config configs/default.yaml --device cuda
```

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Trajectory frames (T) | 48 (2s @ 24fps) |
| Toric dimension | 6 |
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

## Camera Motion Types

| Type | Description | Primary Toric Change |
|------|-------------|---------------------|
| `static` | Fixed camera | No change |
| `dolly-in` | Push toward subjects | pA, pB spread outward |
| `dolly-out` | Pull away from subjects | pA, pB move inward |
| `pan-left` | Rotate camera left | θ decreases |
| `pan-right` | Rotate camera right | θ increases |
| `crane-up` | Raise camera | φ increases |
| `crane-down` | Lower camera | φ decreases |
| `track` | Follow action laterally | θ + slight dolly |
| `orbit` | Circle around subjects | Large θ change |

---

## Acknowledgments

- Toric camera space: Lino & Christie (2015), ACM TOG
- DDPM: Ho, Jain & Abbeel (2020)
- FiLM conditioning: Perez et al. (2018)
- 6D rotation: Zhou et al. (2019), CVPR
- DanceCamera3D: Wang et al. (2024), AAAI
- MDM (Human Motion Diffusion): Tevet et al. (2022), ICLR
