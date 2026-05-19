# Script to Camera

Joint person-camera trajectory generation from text via Conditional Flow Matching.

Given a text prompt (e.g. "the camera orbits around the character as they stand still"), the system jointly generates a person root trajectory and a 6-DoF camera trajectory, 48 frames at 24 fps.

## Architecture

- Dual-branch Transformer denoiser (24.7M params, 6 layers, 256 hidden dim)
- Conditional Flow Matching (OT-CFM) with 50-step Euler ODE sampling
- CLIP ViT-B/32 text encoder (frozen)
- 9 motion types: static, dolly-in/out, pan-left/right, crane-up/down, track, orbit

## Installation

```bash
pip install -e .
```

Requires Python >= 3.10 and a CUDA GPU for training.

## Data Preparation

Training uses 609k samples from three sources:

1. **E.T.** (56k) - real film camera trajectories with SMPL-H person tracks
2. **AMASS** (477k) - motion capture with rule-based camera supervision
3. **HumanML3D** (76k) - motion capture with human-written captions

Prepare each source, then merge:

```bash
python scripts/preprocess_et_data.py
python scripts/prepare_amass.py
python scripts/prepare_humanml3d.py
python scripts/merge_datasets.py
```

Compute normalization statistics:

```bash
python compute_norm_stats.py 3 /path/to/merged-data
```

## Training

```bash
PYTHONPATH=. python experiments/flow_matching/train.py \
    --config experiments/flow_matching/configs/v9.yaml \
    --device cuda
```

Config: `experiments/flow_matching/configs/v9.yaml`

## Inference

```bash
PYTHONPATH=. python experiments/flow_matching/generate.py \
    --checkpoint /path/to/fm_best.pth \
    --text "The camera orbits around the character as they stand still" \
    --motion orbit \
    --shot-type medium-shot \
    --guidance-scale 3.0 \
    --lookat
```

## Evaluation

```bash
PYTHONPATH=. python experiments/flow_matching/evaluate.py \
    --checkpoint /path/to/fm_best.pth \
    --device cuda \
    --max-samples 1024
```

## Web Demo

```bash
PYTHONPATH=. python app.py \
    --checkpoint /path/to/fm_best.pth \
    --port 7861
```

## Project Structure

```
experiments/flow_matching/
    configs/v9.yaml                # Training config
    train.py                       # Training
    generate.py                    # Inference + post-processing
    evaluate.py                    # Quantitative evaluation
    models/flow_model.py           # Flow Matching implementation
    postprocess_constraints.py     # Motion-type-aware constraints

src/
    data/dataset.py                # Dataset and data loading
    models/denoiser.py             # Dual-branch Transformer
    models/text_encoder.py         # CLIP encoder wrapper
    models/film.py                 # FiLM conditioning

scripts/
    preprocess_et_data.py          # E.T. data preprocessing
    prepare_amass.py               # AMASS camera generation
    prepare_humanml3d.py           # HumanML3D preparation
    merge_datasets.py              # Dataset merging
    verify_camera_generation.py    # Pre-training data verification
    gen_cross_version.sh           # Cross-version comparison generation

app.py                             # Flask web demo
generate.py                        # Legacy DDPM inference (unused)
compute_norm_stats.py              # Normalization statistics
ScriptToCamera.ipynb               # End-to-end notebook
```
