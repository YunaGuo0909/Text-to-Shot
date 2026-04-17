# Joint Person-Camera Trajectory Generation from Text via Dual-Branch Diffusion

## 1. Introduction

[TODO: Project motivation, contributions summary, report outline]

## 2. Background

[TODO: Literature review covering diffusion models, motion generation, camera trajectory synthesis, SMPL body model, Toric camera parameterization, E.T./DIRECTOR baseline]

## 3. Methodology

### 3.1 Problem Formulation

The goal is to generate, from a text description alone, a joint trajectory consisting of a person root position sequence and a camera state sequence. Unlike DIRECTOR (Courant et al., 2024), which requires pre-existing character motion as input, this system produces both outputs simultaneously. The joint vector is defined as y = [person_flat, camera_flat], where person_flat is T x 3 (root XYZ per frame) and camera_flat is T x 6 (translation, azimuth, elevation, roll per frame), giving a total dimension of 432 for T = 48 frames at 24 fps.

### 3.2 Model Architecture

The denoiser is a dual-branch Transformer with six layers, each containing a person branch and a camera branch. Within each layer, both branches perform temporal self-attention over the frame dimension, followed by cross-attention where the person branch attends to the camera branch and vice versa. This cross-attention mechanism allows camera motion to remain aware of person motion throughout the generation process.

Conditioning is injected through two mechanisms. First, a condition vector is formed by concatenating the CLIP text embedding (512 dimensions), a sinusoidal timestep embedding (128 dimensions), a learned shot type embedding (64 dimensions), and a learned motion type embedding (64 dimensions). This vector is projected and added as a bias to the input of both branches. Second, each feed-forward sub-layer uses FiLM (Feature-wise Linear Modulation), where the condition vector produces per-feature scale and shift parameters that modulate intermediate representations.

The model has approximately 24.7 million trainable parameters.

### 3.3 Diffusion Framework

Training follows the DDPM framework (Ho et al., 2020) with a cosine noise schedule over 1000 timesteps. The training objective predicts the clean trajectory y_0 directly from the noised input y_t, minimising MSE loss. During inference, both DDPM full-chain sampling and DDIM deterministic sampling (Song et al., 2020) are supported, with Classifier-Free Guidance (CFG) to strengthen text alignment.

### 3.4 Data Preparation

Training data comes from the E.T. (Exceptional Trajectories) dataset (Courant et al., 2024), which contains 114,603 samples extracted from real film footage using SLAHMR (Ye et al., 2023). Each sample pairs a camera extrinsic trajectory with SMPL-H body parameters and natural language captions describing both camera and character behaviour.

[TODO: Expand with preprocessing pipeline details, caption types, data statistics]

## 4. Implementation Challenges and Solutions

This section documents the technical problems encountered during development, along with the diagnostic process and solutions applied. These issues span data preprocessing, numerical stability, and training dynamics.

### 4.1 Missing Person Trajectory Data

The first training run (500 epochs, 103,173 samples) produced results where generated person positions bore no relation to the text description. Camera trajectories showed some learned structure but person trajectories appeared random.

Investigation began with the dataset loading code (`src/data/dataset.py`). The `__getitem__` method loads person trajectories by looking up the key `person_trajectory_path` in each sample entry. When this key is absent, the loader calls `_load_trajectory`, which falls back to returning an array of zeros with shape (48, 3). Inspecting the training index file (`train_index.json`) revealed that none of the 103,173 entries contained the `person_trajectory_path` field. Every sample used the field name `trajectory_path`, which pointed to a camera-only file of shape (48, 6).

The cause was straightforward. The training data had been produced by an earlier version of the preprocessing script that extracted only camera trajectories from the E.T. dataset. The current version of `preprocess_et_data.py` was designed to output separate `camera_trajectories/` and `person_trajectories/` directories, but the data on disk predated this script. The model had trained for 500 epochs with person trajectories set to zero for every sample, learning nothing meaningful about person motion.

The fix required rerunning the preprocessing pipeline with the updated script. However, this exposed a chain of further issues described below.

### 4.2 Silent Exception Swallowing in SMPL-H Loading

After rerunning `preprocess_et_data.py` with the `smplh/` directory available (65,308 pickle files), the output summary reported that only 2,470 out of 114,603 samples (2.2%) successfully loaded person data from SMPL-H files. The remaining 97.8% fell back to a look-at proxy: a heuristic that estimates person position as a point 3 metres along the camera's forward direction.

A filename comparison confirmed that 65,308 sample IDs matched between `traj/` and `smplh/`. The mismatch rate should have been at most 43%, not 98%. Something was failing silently during the actual pickle loading.

The loading function `load_person_joints` wrapped all file I/O and tensor operations in a broad `try/except Exception: pass` block. Any error during loading was discarded without logging, and the function simply returned `None`, triggering the look-at fallback.

A diagnostic script (`scripts/diagnose_smplh.py`) was written to bypass this exception handler and load a single SMPL-H file directly. The true error was immediately revealed:

```
RuntimeError: Can't call numpy() on Tensor that requires grad.
Use tensor.detach().numpy() instead.
```

The E.T. dataset stores SMPL-H parameters as PyTorch tensors with `requires_grad=True`, likely a residual from the SLAHMR optimisation process. The original code called `.cpu().numpy()` on these tensors, which PyTorch rejects when gradient tracking is active. The fix was to insert `.detach()` before the conversion: `.detach().cpu().numpy()`.

This single missing method call, hidden by a bare `except: pass`, was responsible for losing 62,838 valid person trajectory samples during preprocessing.

### 4.3 NaN and Extreme Values from SLAHMR Tracking Failures

With the `detach()` fix applied and a `--require-person` flag added to skip samples lacking SMPL-H data, the preprocessing pipeline successfully extracted 65,308 paired trajectories. However, computing normalisation statistics over the training set produced:

```
Mean range: [nan, nan]
Std  range: [nan, nan]
```

A single NaN value in any sample propagates through the mean computation and corrupts the entire statistic. The source was SLAHMR: as an optimisation-based method for jointly estimating camera and human pose from video, SLAHMR occasionally fails on occluded or ambiguous frames, producing NaN or numerically extreme translation values.

After adding a `np.isfinite()` check and filtering samples with absolute translation values exceeding 100 metres, the normalisation statistics became:

```
Mean range: [-1.240, 0.202]
Std  range: [0.098, 1.389]
```

This reduced the usable dataset from 65,308 to 64,948 samples (360 removed, 0.6%). The final training set contained 58,553 samples after applying the official train/test split.

### 4.4 Caption Priority Error

The E.T. dataset provides two types of text annotations per sample. The `caption/` directory contains full descriptions covering both character actions and camera movement (e.g., "As the character moves left, the camera trucks left in sync, followed by a pull-out"). The `caption_cam/` directory contains camera-only descriptions (e.g., "The camera remains static during the entire shot").

The preprocessing script selected captions with the following priority:

```python
text = caption_cam if caption_cam else caption_full
```

This prioritised camera-only captions, discarding all character motion information from the text. For a model that jointly generates person and camera trajectories, this meant the text conditioning contained no signal about what the person should be doing. The fix was to reverse the priority to prefer full captions.

### 4.5 Severe Class Imbalance in Motion Types

The filtered training set exhibited heavy class imbalance across camera motion types:

| Motion Type | Samples | Percentage |
|-------------|---------|------------|
| static      | 43,329  | 66.7%      |
| track       | 13,610  | 21.0%      |
| dolly-in    | 6,902   | 10.6%      |
| pan-right   | 548     | 0.8%       |
| pan-left    | 246     | 0.4%       |
| dolly-out   | 232     | 0.4%       |
| crane-up    | 68      | 0.1%       |
| crane-down  | 11      | <0.1%      |
| orbit       | 2       | <0.1%      |

With uniform random sampling, the model sees static shots in two out of every three batches. The learned trajectory distribution collapses toward the mean of the dominant class. At inference time, varying the motion type conditioning produces near-identical outputs regardless of the requested motion, because the model has learned that "the average trajectory" is the safest prediction for minimising MSE.

The first training run (500 epochs, uniform sampling, CFG dropout 0.1) converged to a train loss of 0.089 with a validation loss of 0.471, indicating significant overfitting. Generated trajectories showed the same camera motion pattern (a bell curve in tx/ty with decreasing tz) for dolly-in, track, and static prompts alike.

Two changes were applied for the second training run. A `WeightedRandomSampler` was introduced to equalise the sampling probability across motion types, so that rare classes like dolly-out (232 samples) are drawn as frequently as static (43,329 samples) per epoch. The CFG dropout probability was increased from 0.1 to 0.25 to create a wider gap between conditional and unconditional predictions, making guidance more effective at inference. Model dropout was also raised from 0.1 to 0.2 and the total epochs reduced from 500 to 250 to mitigate overfitting.

[TODO: Report results of second training run]

### 4.6 Summary of Fixes

| Problem | Root Cause | Fix Applied |
|---------|------------|-------------|
| Person trajectory all zeros | Old preprocessing output lacked person data | Reran preprocessing with updated script |
| 98% of SMPL-H loads failing | `.cpu().numpy()` on tensor with `requires_grad=True`, hidden by `except: pass` | Added `.detach()`, replaced silent exception with logging |
| NaN in normalisation stats | SLAHMR tracking failures producing NaN translations | Filtered non-finite and extreme (>100m) values |
| Normalisation mean at 5e10 | SLAHMR divergence producing astronomically large translations | Same outlier filter |
| Text describes only camera | Preprocessing preferred `caption_cam` over full `caption` | Reversed caption priority |
| All motion types produce identical output | 67% static data dominates training; weak CFG dropout | Weighted sampling by motion type; CFG dropout 0.1 to 0.25 |
| Overfitting (train 0.09 vs val 0.47) | 500 epochs on 58k samples; low dropout | Reduced epochs to 250; dropout 0.1 to 0.2 |

## 5. Experiments

[TODO: Training curves for both runs, qualitative comparison (GT vs generated), quantitative metrics (MSE, jerk, motion type accuracy), ablation studies, guidance scale comparison]

## 6. Results

[TODO: Generated trajectory visualisations, comparison across motion types, GT vs predicted side by side]

## 7. Critical Self-Evaluation

### What Worked

The dual-branch architecture with cross-attention successfully couples person and camera generation in a single forward pass. The preprocessing pipeline, once corrected, reliably extracts paired trajectories from the E.T. dataset. The diagnostic methodology of checking normalisation statistics as a first sanity test proved effective at catching data issues early.

### What Did Not Work Well

The most significant limitation is data imbalance. With six of nine motion types each accounting for less than 1% of training data, the model cannot learn to differentiate them reliably. Weighted sampling alleviates but does not solve this, because the absolute number of examples for rare classes (e.g., 2 orbit samples) is too small for a 24M parameter model to generalise from.

The silent exception handling in the preprocessing code cost substantial debugging time. The `except: pass` pattern is especially dangerous in data pipelines because corrupted or missing data produces no error, only degraded training performance that manifests much later.

[TODO: Expand with discussion of overfitting, text-trajectory alignment, comparison to DIRECTOR baseline, future work directions]

## References

[TODO: Full bibliography]

- Courant, R., Dufour, N., Wang, X., Christie, M., & Kalogeiton, V. (2024). E.T. the Exceptional Trajectories: Text-to-Camera-Trajectory Generation with Character Awareness. ECCV 2024.
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS 2020.
- Song, J., Meng, C., & Ermon, S. (2020). Denoising Diffusion Implicit Models. ICLR 2021.
- Radford, A., et al. (2021). Learning Transferable Visual Models from Natural Language Supervision. ICML 2021.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Bengio, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. AAAI 2018.
- Tevet, G., Raab, S., Gordon, B., Shafir, Y., Cohen-Or, D., & Bermano, A. H. (2022). Human Motion Diffusion Model. ICLR 2023.
- Ye, V., Pavlakos, G., Malik, J., & Kanazawa, A. (2023). Decoupling Human and Camera Motion from Videos in the Wild. CVPR 2023.
