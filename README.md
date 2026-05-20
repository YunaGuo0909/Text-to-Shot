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
python scripts/data/preprocess_et_data.py
python scripts/data/prepare_amass.py
python scripts/data/prepare_humanml3d.py
python scripts/data/merge_datasets.py
```

Verify camera label quality before training (optional but recommended):

```bash
PYTHONPATH=. python scripts/data/verify_camera_generation.py
```

Compute normalization statistics:

```bash
python scripts/data/compute_norm_stats.py --data-root /path/to/merged-data
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
    configs/v9.yaml                    # Training config (used for v11)
    train.py                           # Training
    generate.py                        # Inference + post-processing
    evaluate.py                        # Quantitative evaluation
    models/flow_model.py               # OT-CFM implementation
    data/augmented_dataset.py          # Dataset with augmentations
    postprocess_constraints.py         # Motion-type-aware hard constraints

src/
    data/dataset.py                    # Base dataset and collate_fn
    models/denoiser.py                 # Dual-branch Transformer
    models/text_encoder.py             # CLIP encoder wrapper
    models/film.py                     # FiLM conditioning layer

scripts/
    data/
        preprocess_et_data.py          # E.T. preprocessing
        prepare_amass.py               # AMASS camera generation
        prepare_humanml3d.py           # HumanML3D preparation
        prepare_dancecamera3d.py       # DanceCamera3D preparation (unused source)
        filter_et_single_person.py     # Filter E.T. to single-person
        merge_datasets.py              # Merge all sources
        compute_norm_stats.py          # Normalization statistics
        verify_camera_generation.py    # Pre-training label quality check
        download_et_data.py            # Download E.T. dataset
        download_humanml3d_texts.py    # Download HumanML3D captions
        rebuild_dataset_v9.sh          # [legacy] Full data rebuild script
    generate/
        gen_v11_all.sh                 # Batch generation, all 9 motion types
        gen_cross_version.sh           # Cross-version comparison
        gen_all_nosmooth.sh            # [legacy] v10 raw output comparison
        generate_v9_all.sh             # [legacy] v9 batch generation
        generate_v9b_all.sh            # [legacy] v9b batch generation
    eval/
        eval_all_versions.sh           # Evaluate all model versions
    viz/
        visualize_gt.py                # Visualize ground-truth trajectories
        visualize_animated.py          # Animated 3D + camera POV GIFs
    debug/
        diagnose_v6_issues.py          # Label accuracy & data quality check
        diagnose_dataset.py            # Dataset statistics diagnostic
        diagnose_smplh.py              # SMPL-H file format check
        check_smplh.py                 # SMPL-H loading check
        verify_data.py                 # Data verification

app.py                                 # Flask web demo
generate.py                            # [legacy] DDPM inference; visualize_joint used by FM pipeline
ScriptToCamera.ipynb                   # End-to-end notebook
```
