# AI-Driven Storyboard Generation

**Extending Text-to-Shot Diffusion for Multi-Shot Visual Pre-production with Camera Motion Trajectories**



---

## Overview

This project extends prior work on joint character-camera generation — which produces **a single static shot** (camera pose + two-character 3D poses) from text — into a full **multi-shot storyboard pipeline** with **dynamic camera motion trajectories**.

Given a scene description like *"Two people meet at a cafe, shake hands and sit down"*, the system automatically:

1. **Decomposes** the scene into a sequence of cinematic shots (via LLM)
2. **Generates** 3D character poses + camera framing for each shot (via diffusion model)
3. **Creates camera motion trajectories** for each shot (dolly, pan, track, crane, orbit, etc.)
4. **Renders** the complete storyboard as annotated panels with motion path overlays

```
Scene Description ──▶ Shot Decomposer ──▶ Diffusion Generator ──▶ Trajectory Generator ──▶ Storyboard Renderer
     (text)             (LLM-based)        (per-shot 3D)         (keyframe → spline)        (panels + paths)
```

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/YunaGuo0909/Text-to-Shot.git
cd Text-to-Shot

# Install dependencies with uv
uv sync

# Run demo (no trained model needed)
uv run python generate_storyboard.py --demo
```

### Output

The demo generates:
- `outputs/demo_storyboard.png` — 6-panel storyboard with camera motion annotations
- `outputs/demo_trajectory_dolly_in.png` — Detailed trajectory parameter curves

---

## Project Structure

```
Text-to-Shot/
├── configs/
│   └── default.yaml              # Model, training & rendering configuration
├── src/
│   ├── models/                   # Neural network modules
│   │   ├── diffusion.py          # Gaussian diffusion process (DDPM)
│   │   ├── denoiser.py           # Joint 3-branch denoiser network
│   │   ├── film.py               # FiLM conditioning layer
│   │   └── interaction.py        # Character-character & camera-character interaction
│   ├── pipeline/                 # Generation pipeline modules
│   │   ├── shot_decomposer.py    # LLM-based scene → shot decomposition
│   │   ├── storyboard_generator.py  # Multi-shot generation with coherence
│   │   ├── camera_trajectory.py  # Camera motion trajectory generation ★
│   │   └── storyboard_renderer.py   # Visual storyboard rendering
│   ├── data/
│   │   └── dataset.py            # Dataset loading for training
│   └── utils/
│       ├── toric.py              # Toric camera parameterization utilities
│       └── smpl_utils.py         # SMPL body model & 6D rotation utilities
├── train.py                      # Model training script
├── generate_storyboard.py        # Storyboard generation entry point
├── pyproject.toml                # Dependencies (uv)
├── PROJECT_PLAN.md               # 10-week project timeline
└── TECHNICAL_DESIGN.md           # Detailed technical design document
```

---

## Technical Pipeline

### Stage 1 — Shot Decomposition (`shot_decomposer.py`)

Uses an LLM (GPT-4 / local model) to break a scene description into a structured shot list. Each shot specifies:
- **Shot type**: close-up, medium-shot, wide-shot, over-the-shoulder, two-shot
- **Camera motion**: static, dolly-in, dolly-out, pan-left, pan-right, crane-up, crane-down, track, orbit
- Character actions for both persons A and B

### Stage 2 — Joint Character-Camera Generation (`diffusion.py` + `denoiser.py`)

A **Gaussian Diffusion Model** generates the 3D configuration for each shot:

**Data representation** — Each shot is a 306-dim vector `y = (x_A, x_B, x_C)`:
- `x_A ∈ R^150`: Character A pose (22 joints × 6D rotation + placement vector)
- `x_B ∈ R^150`: Character B pose (same structure)
- `x_C ∈ R^6`: Camera state in **Toric space** (pA_x, pA_y, pB_x, pB_y, θ, φ)

**Network architecture** — `JointDenoiser` with three parallel branches:
- Branch A (Character A), Branch B (Character B), Branch C (Camera)
- Each branch: Linear → [MLP + FiLM conditioning + residual] × 4 → Linear
- Three **pairwise interaction modules** exchange messages between entities:
  - `I_HH`: Character A ↔ Character B (e.g., handshake coordination)
  - `I_AC`: Character A ↔ Camera (e.g., framing follows action)
  - `I_BC`: Character B ↔ Camera

**Conditioning** — The denoiser is conditioned on:
- Text embedding (CLIP, 512-dim)
- Diffusion timestep (sinusoidal, 128-dim)
- Shot type embedding (learnable, 64-dim) ★ *extension*

### Stage 3 — Camera Trajectory Generation (`camera_trajectory.py`) ★ New

Extends the static Toric camera state into a temporal motion trajectory:

1. **Motion profile**: Each motion type (dolly, pan, crane, etc.) defines delta changes to the 6 Toric parameters
2. **Keyframe generation**: K keyframes are generated with easing functions (ease-in-out, quadratic)
3. **Spline interpolation**: Cubic spline (C² continuous) interpolation produces smooth T-frame trajectories
4. **Evaluation metrics**: Velocity, acceleration, jerk (smoothness) are computed per trajectory

### Stage 4 — Storyboard Rendering (`storyboard_renderer.py`)

Renders all shots into a visual storyboard grid:
- Stick figure characters (A = red, B = cyan) from SMPL joint positions
- Camera motion arrows and path overlays (yellow)
- Shot type labels, camera motion type badges, descriptive text

---

## Training

```bash
# Train the diffusion model
uv run python train.py --config configs/default.yaml --device cuda
```

**Training process:**

1. Load dataset of `(text, character_A_pose, character_B_pose, camera_state)` tuples
2. For each batch:
   - Encode text with CLIP → `text_embed (B, 512)`
   - Sample random timestep `t ~ Uniform(0, T)`
   - Add noise: `y_t = √ᾱ_t · y_0 + √(1-ᾱ_t) · ε`
   - Predict clean sample: `ŷ_0 = f_θ(y_t, t, text_embed, shot_type)`
   - Compute loss: `L = MSE(ŷ_0, y_0)`
3. Backpropagate with AdamW optimizer + gradient clipping
4. Save checkpoints every N epochs

**Key hyperparameters** (see `configs/default.yaml`):

| Parameter | Value |
|-----------|-------|
| Diffusion timesteps | 1000 |
| Beta schedule | Cosine |
| Hidden dim | 512 |
| Transformer layers | 4 × 3 branches |
| Batch size | 64 |
| Learning rate | 1e-4 |
| Epochs | 500 |

---

## Camera Motion Types

| Type | Description | Toric Parameter Change |
|------|-------------|----------------------|
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
