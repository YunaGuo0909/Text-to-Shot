# Joint Person-Camera Trajectory Generation from Text via Conditional Flow Matching

**[INSERT: Project title image or pipeline overview diagram]**

---

## 1. Introduction

[TODO: motivation, problem statement, contributions summary]

This project addresses the task of generating paired person and camera trajectories simultaneously from a natural language description. Given a text such as "As the character moves forward, the camera pushes in", the system should produce a 2-second sequence of 3D camera states and person root positions that are spatially coherent and semantically consistent with the description.

The baseline approach (DIRECTOR, Courant et al., 2024) conditions camera generation on pre-existing character motion. This project removes that dependency, generating both outputs from text alone, enabling fully automated scene generation without upstream motion capture.

---

## 2. Background

[TODO: expand with full literature review]

- **DDPM** (Ho et al., 2020): denoising diffusion probabilistic models
- **Flow Matching** (Lipman et al., 2023): straight-path ODE transport from noise to data, better mode coverage than DDPM
- **E.T. / DIRECTOR** (Courant et al., 2024): camera trajectory diffusion conditioned on character motion, ECCV 2024
- **SMPL-H** (Romero et al., 2017): parametric human body model, 22 body joints + hands
- **SLAHMR** (Ye et al., 2023): joint camera + human pose estimation from video, CVPR 2023
- **AMASS** (Mahmood et al., 2019): large-scale motion capture database, 40+ hours, SMPL parameters
- **CLIP** (Radford et al., 2021): contrastive language-image model, used as frozen text encoder
- **FiLM** (Perez et al., 2018): feature-wise linear modulation for conditioning

---

## 3. Method

### 3.1 Problem Formulation

The model generates a joint trajectory vector $y = [\text{person\_flat}, \text{camera\_flat}]$. Camera state per frame is $(t_x, t_y, t_z, \text{azimuth}, \text{elevation}, \text{roll})$ (6 dimensions). Person state per frame is $(p_x, p_y, p_z, \sin(\text{yaw}), \cos(\text{yaw}))$ (5 dimensions), where yaw encodes the root body facing direction using a continuous sin/cos representation to avoid angular discontinuities. With 48 frames at 24 fps (2 seconds), the total joint vector dimension is $48 \times (5 + 6) = 528$.

### 3.2 Architecture

The denoiser is a dual-branch Transformer with 6 layers, 256 hidden dimensions, and 4 attention heads. The person branch and camera branch each perform temporal self-attention over the 48-frame sequence, then cross-attend to each other at every layer. Conditioning is injected via FiLM modulation (Perez et al., 2018) using a concatenation of CLIP text embedding (512d), sinusoidal timestep embedding (128d), shot type embedding (64d), and camera motion type embedding (64d). The model has approximately 24.7 million parameters.

**[INSERT: Architecture diagram showing dual-branch transformer with cross-attention and FiLM conditioning]**

### 3.3 Conditional Flow Matching

Rather than DDPM, this work uses Conditional Flow Matching (Lipman et al., 2023). The forward interpolant is:

$$x_t = (1 - t) \cdot \epsilon + t \cdot x_0, \quad \epsilon \sim \mathcal{N}(0, I), \quad t \in [0, 1]$$

The target velocity field is $v = x_0 - \epsilon$ (constant along the straight-line path). The network is trained to predict this velocity, and sampling uses Euler integration from $t=0$ (noise) to $t=1$ (data). Classifier-Free Guidance (Ho et al., 2022) is applied at inference with a null text embedding.

### 3.4 Training Losses

Two loss terms are combined:

**Flow matching loss**: MSE between predicted and target velocity,
$$\mathcal{L}_{\text{flow}} = \mathbb{E}_{t, \epsilon}[\| v_\theta(x_t, t, c) - (x_0 - \epsilon) \|^2]$$

**Temporal smoothness loss**: penalises large frame-to-frame angle changes in the camera orientation dimensions (azimuth, elevation, roll), motivated by dataset analysis showing ground-truth cameras have a median angular change of 0.27 degrees per frame:
$$\mathcal{L}_{\text{smooth}} = \frac{1}{T-1}\sum_{t=1}^{T-1} \|\theta_{t} - \theta_{t-1}\|^2$$

Total loss: $\mathcal{L} = \mathcal{L}_{\text{flow}} + 0.05 \cdot \mathcal{L}_{\text{smooth}}$

---

## 4. Data

### 4.1 E.T. Dataset

The E.T. (Exceptional Trajectories) dataset (Courant et al., 2024) contains 114,603 samples extracted from real film footage using SLAHMR (Ye et al., 2023). Each sample pairs a camera extrinsic trajectory with SMPL-H character data and natural language captions. After filtering for samples with valid SMPL-H person trajectories and removing SLAHMR outliers (NaN values and displacements exceeding 100m), 64,948 usable samples remained.

**[INSERT: E.T. dataset motion type distribution bar chart - before and after filtering]**

### 4.2 Dataset Diagnostic Findings

A systematic analysis of the E.T. training data (`scripts/diagnose_dataset.py`) revealed several important properties:

**Label noise in motion types.** Camera motion labels are inferred by keyword matching on caption text, which introduces significant noise. For the "dolly-in" class, only 57.4% of samples actually show decreasing camera-to-person distance — essentially random. This explains why models trained on E.T. alone could not learn to distinguish dolly-in from dolly-out behaviour.

**[INSERT: Table showing dolly-in/dolly-out label consistency statistics]**

**Camera look-at alignment.** The mean cosine similarity between camera forward direction and camera-to-person direction is −0.67, with 93.4% of frames showing negative alignment. This is attributable to a coordinate system convention difference between the camera extrinsic decomposition and the SMPL-H world frame, not a genuine data quality issue.

**Angle smoothness.** Ground-truth camera angles change by a median of 0.27 degrees per frame (mean 0.65 deg/frame), confirming that professionally filmed camera movements are smooth. This motivates the temporal smoothness loss.

**Severe class imbalance.** The static class comprises 66.7% of E.T. samples, while dolly-out accounts for 0.4% and orbit for 0%. This imbalance causes mode collapse when training on E.T. alone.

**[INSERT: Class distribution pie chart or bar chart for E.T. alone]**

### 4.3 AMASS Augmentation

To address the label noise and class imbalance in E.T., the AMASS motion capture dataset (Mahmood et al., 2019) was used to generate synthetic training data. AMASS provides accurate 3D root translations extracted from real human motion capture. For each 2-second clip, synthetic camera trajectories were generated deterministically for all 9 motion types using cinematography rules: dolly-in linearly decreases camera-to-person distance from 5m to 2m; orbit maintains constant distance while azimuth increases linearly; static holds the camera at a fixed position facing the person centroid; and so on. Captions were generated from templates using trajectory-based action inference.

This approach produces labels that are 100% accurate by construction — unlike the keyword-matched E.T. labels — and ensures perfect class balance (11.1% per motion type).

After processing 5,401 AMASS files (CMU, ACCAD, BMLrub subsets), 268,317 samples were generated. In subsequent iterations, additional AMASS subsets (KIT, Eyes_Japan_Dataset) were added, increasing the AMASS contribution to 517,797 samples from 10,383 source files. Caption generation was also enriched with shot type prefixes (e.g., "Close Up. The camera pushes in while the character walks forward.") and person turning descriptions inferred from yaw changes exceeding 45 degrees (e.g., "moves forward while turning left").

### 4.4 HumanML3D Integration

The AMASS-generated captions, while structurally accurate, are produced from templates and lack the lexical diversity of human language. To address this, the HumanML3D dataset (Guo et al., 2022) was integrated as a third data source. HumanML3D provides 14,616 motions with 44,970 human-written text descriptions averaging three descriptions per motion, covering locomotion, gestures, and complex multi-phase actions.

HumanML3D motions are derived from AMASS but stored in a processed format that requires re-extraction from the original AMASS source files. Using the dataset's `index.csv`, which maps each motion ID to an AMASS source path and frame range, a matching script (`scripts/prepare_humanml3d.py`) was implemented. This script resolves HumanML3D source paths (e.g., `./pose_data/CMU/80/80_63_poses.npy`) to actual AMASS files on disk (e.g., `/transfer/amassdata/CMU/CMU/80/80_63_poses.npz`), handling naming convention differences (double directory nesting, `.npy` to `.npz` extension, `_poses` to `_stageii` suffix variants).

With five AMASS subsets available locally (CMU, ACCAD, BMLrub, KIT, Eyes_Japan_Dataset), 9,303 of 14,616 HumanML3D motions were successfully matched (63.6%). The remaining 36.4% reference AMASS subsets not downloaded (BMLmovi, HDM05, SSM_synced) or the non-AMASS HumanAct12 subset. Each matched motion was paired with all 9 camera motion types, producing 83,700 samples with human-written captions.

The final merged training set for v8 contains 660,050 samples from three sources:

| Source | Samples | Caption Quality | Person Data |
|--------|---------|----------------|-------------|
| E.T. | 64,948 | Film captions (moderate) | SLAHMR-estimated |
| AMASS | 517,797 | Template-generated | MoCap ground truth |
| HumanML3D | 83,700 | **Human-written** | MoCap ground truth |
| **Total** | **666,445** | | |

**[INSERT: Class distribution comparison across E.T. only, merged v5 (326k), merged v8 (666k)]**

| Motion Type | E.T. only | Merged v5 | Merged v8 |
|-------------|-----------|-----------|-----------|
| static | 66.7% | 21.9% | 16.5% |
| dolly-out | 0.4% | 9.0% | 10.1% |
| orbit | 0.0% | 8.9% | 10.0% |
| dolly-in | 10.6% | 11.0% | 11.0% |

### 4.5 Person Trajectory Representation

The person trajectory representation evolved through three stages during development:

**v1-v6 (position only):** $(p_x, p_y, p_z)$ per frame, 3 dimensions. The root translation is extracted from the SMPL-H `transl` field for E.T. data and from the `trans` or `root_orient` fields for AMASS data. This captures where the person is but not which direction they face.

**v7 (raw yaw, failed):** $(p_x, p_y, p_z, \theta)$ per frame, 4 dimensions. The yaw angle $\theta$ was extracted from the SMPL-H `global_orient` rotation matrix via $\theta = \text{atan2}(R_{02}, R_{22})$ and from AMASS axis-angle root orientation via Rodrigues conversion. This model suffered complete mode collapse due to the angular wraparound discontinuity at $\pm\pi$ (see Section 5.6).

**v8 (sin/cos yaw):** $(p_x, p_y, p_z, \sin\theta, \cos\theta)$ per frame, 5 dimensions. The same yaw angle is encoded as its sine and cosine components, removing the discontinuity. This is the standard approach in the motion generation literature. The total joint vector dimension is $48 \times (5 + 6) = 528$.

For samples without orientation data (E.T. look-at proxy fallback), the yaw columns are set to $(\sin 0, \cos 0) = (0, 1)$, representing a default forward-facing direction.

---

## 5. Implementation Challenges

This section documents the technical problems encountered during development and the solutions applied.

### 5.1 Missing Person Trajectory Data

The first training run (500 epochs, E.T.) produced outputs where person positions were random and unrelated to the text. Investigation of the dataset loading code (`src/data/dataset.py`) revealed that the `person_trajectory_path` field was absent from all 103,173 index entries. The fallback behaviour on missing paths returned a zero array of shape (48, 3), so every training sample had a person trajectory of exactly zero. The model spent 500 epochs learning that all persons are located at the origin, regardless of text conditioning.

The root cause was that the training data had been preprocessed by an earlier version of `preprocess_et_data.py` that extracted only camera trajectories. The data on disk predated the addition of person trajectory extraction logic.

### 5.2 Silent Exception Handling Concealing a Critical Bug

After re-running preprocessing with the updated script, the output summary reported that only 2,470 of 114,603 samples (2.2%) successfully loaded person data from SMPL-H files. The remaining 97.8% fell back to a heuristic look-at proxy.

A diagnostic script (`scripts/diagnose_smplh.py`) was written to bypass the exception handler and load a single SMPL-H file directly. The true error was:

```
RuntimeError: Can't call numpy() on Tensor that requires grad.
Use tensor.detach().numpy() instead.
```

E.T. stores SMPL-H tensors with `requires_grad=True` as a residual of the SLAHMR optimisation process. The original code called `.cpu().numpy()` directly, which PyTorch rejects on gradient-tracked tensors. The fix required inserting `.detach()` before the conversion. This single missing method call, hidden by a bare `except: pass`, was responsible for losing 62,838 valid person trajectory samples.

This incident highlights the danger of broad exception suppression in data pipelines. A `except: pass` block produces no error output when a critical operation fails, only degraded downstream behaviour that may not manifest until training time.

**[INSERT: Diagram showing data pipeline before and after the fix, with sample counts at each stage]**

### 5.3 Caption Priority Inversion

The preprocessing script selected captions with the following priority:

```python
text = caption_cam if caption_cam else caption_full
```

This preferred the camera-only caption (e.g., "The camera remains static during the entire shot") over the full caption that describes both character and camera behaviour (e.g., "As the character moves left, the camera trucks left in sync"). For a model generating joint person-camera trajectories, discarding character motion descriptions eliminated a major source of semantic signal. The fix was to reverse the priority.

### 5.4 Numerical Instability in SLAHMR Outputs

After extracting person trajectories successfully, normalisation statistics computed over the training set produced `mean = NaN` and `std = NaN`. A single NaN value in any sample propagates through the mean computation and corrupts all statistics. The source was SLAHMR: as an iterative optimisation-based method, it occasionally produces NaN or astronomically large translation values (up to 5×10¹⁰) when tracking fails on occluded or ambiguous footage. Filtering with `np.isfinite()` and an outlier threshold of 100m removed 360 samples (0.6%) and restored normal statistics: mean ∈ [−1.24, 0.20], std ∈ [0.10, 1.39].

### 5.5 Flow Matching Timestep Direction Bug

The first Flow Matching model (v1) produced pure noise at inference despite completing training without errors and achieving a training loss of 0.295. The root cause was a sign error in the mapping from Flow Matching's continuous time parameter $t \in [0, 1]$ to the integer timestep expected by the Transformer's sinusoidal embedding (designed for DDPM's 0–999 range).

The Flow Matching convention uses $t=0$ for noise and $t=1$ for clean data. The DDPM sinusoidal embedding convention uses timestep 0 for clean data and timestep 999 for noise. The original mapping was:

```python
t_scaled = (t * 999).long()   # wrong: t=0 (noise) → timestep 0 (clean)
```

This meant that whenever the model received a noisy input (early in the ODE trajectory), it was told via the timestep embedding that the input was clean, and vice versa. The model received contradictory signals throughout training, causing the velocity field to be incoherent. The fix:

```python
t_scaled = ((1.0 - t) * 999).long()   # correct: t=0 (noise) → timestep 999 (noise)
```

One line of code. The post-fix model (v2) produced structured outputs with visible motion type differentiation for the first time in the project.

**[INSERT: Side-by-side comparison of FM v1 (noise) vs FM v2 (structured) generated trajectories]**

### 5.6 Angular Wraparound in Person Yaw Representation

To support person facing direction, the person trajectory was extended from $(p_x, p_y, p_z)$ to $(p_x, p_y, p_z, \text{yaw})$, where yaw is the root body orientation around the vertical axis, extracted from the SMPL-H `global_orient` field. This increased person_dim from 3 to 4 and total joint dimension from 432 to 480.

The extended model (v7, trained on 660k samples including HumanML3D annotations) converged to a training loss of 0.241 but produced completely flat outputs at inference. All motion types generated identical stationary trajectories, a total mode collapse that was absent in the previous version (v6, person_dim=3) which had successfully differentiated orbit, dolly-in, static, and track.

The root cause was the angular wraparound discontinuity. Yaw in radians has range $[-\pi, \pi]$, where $-\pi$ and $+\pi$ represent the same physical direction but differ numerically by $2\pi \approx 6.28$. Flow Matching interpolates between noise and data along straight lines in the data space:

$$x_t = (1-t)\epsilon + t \cdot x_0$$

This interpolation assumes the data space is Euclidean. For position dimensions this holds, but for yaw it does not. When the training set contains samples with yaw near $+\pi$ and samples with yaw near $-\pi$, the model receives contradictory gradient signals. The MSE-optimal compromise is to predict yaw $\approx 0$ for all inputs, collapsing the yaw dimension and destabilising the other dimensions through the shared hidden representation.

Inspection of the training data confirmed the issue. One sample showed yaw ranging from $-2.98$ to $+3.10$ radians within a single 48-frame clip. In angle space this is a 9-degree turn. In the raw representation it is a 6.08-unit jump that the model cannot interpolate through correctly.

The fix was to replace the raw yaw angle with its sine and cosine components:

$$(\text{yaw}) \rightarrow (\sin(\text{yaw}), \cos(\text{yaw}))$$

This changes person_dim from 4 to 5. The sin/cos representation is continuous everywhere, with no discontinuity at $\pm\pi$. Two directions that are close in angle space (e.g., $-2.98$ and $+3.10$ radians) map to nearby points in sin/cos space: $(-0.16, -0.99)$ and $(0.04, -1.00)$, a Euclidean distance of only 0.20. Linear interpolation between these points correctly traces through the "backward-facing" region of the circle.

This is the standard approach for encoding angular quantities in neural networks, used in MDM (Tevet et al., 2023), MotionDiffuse (Zhang et al., 2022), and other motion generation works. At inference time, the original angle is recovered via $\text{yaw} = \text{atan2}(\sin, \cos)$.

This failure illustrates a broader principle: the topology of each data dimension must match the generative model's interpolation assumptions. Flow matching and diffusion models assume Euclidean geometry. Circular quantities require explicit embedding into a Euclidean space (sin/cos) before they can be modelled by these frameworks.

### 5.7 Mode Collapse in Joint Diffusion Models

All DDPM-based joint generation models collapsed to producing nearly identical outputs regardless of motion type conditioning. Analysis identified three contributing factors:

**MSE-optimal mean prediction.** With 66.7% of training samples labelled as static, the MSE-minimising strategy is to predict a near-static trajectory for every input. The difference in loss between predicting the mean and predicting a correctly directed dolly-in trajectory is small in absolute terms.

**Weak CFG conditioning.** With CFG dropout probability of 0.1, the model was exposed to unconditional training only 10% of the time, creating a small gap between conditional and unconditional outputs. At inference, even with guidance scale 5.0, the amplified direction was nearly zero. Increasing CFG dropout to 0.25 improved but did not resolve this issue.

**Joint generation difficulty.** Generating person (144d) and camera (288d) simultaneously requires the model to learn a 432-dimensional joint distribution from text. DIRECTOR (Courant et al., 2024) avoids this by conditioning camera generation on pre-given character motion, reducing the search space substantially. Two-stage decomposition was implemented (Stage 1: text → person; Stage 2: text + person → camera) with promising training dynamics (val loss 0.11, gap 2.3×) but remaining spatial decoupling at inference time.

**[INSERT: Comparison table of all experiments - method, data, val loss, qualitative result]**

**[INSERT: Generated trajectory comparison - DDPM mode collapse vs FM v5 motion type differentiation]**

---

## 6. Experiments

### 6.1 Experimental Setup

All models were trained on an NVIDIA RTX 4080 (16 GB). The Adam-W optimiser was used with learning rate 1×10⁻⁴, cosine decay with 10-epoch warmup, and gradient clipping at 1.0. Batch size was 64. Motion type class-balanced sampling (WeightedRandomSampler) was used for all experiments.

### 6.2 Ablation: E.T. Only vs Merged Dataset

**[INSERT: Training loss curves comparing FM v2 (E.T. 58k) and FM v5 (merged 326k)]**

| Model | Data | person_dim | Val Loss | Train/Val Gap | Notes |
|-------|------|-----------|----------|---------------|-------|
| DDPM run 1 | E.T. 58k | 3 | 0.471 | 5.3× | Mode collapse |
| DDPM run 2 | E.T. 58k | 3 | 0.657 | 9.7× | Balanced sampling, still collapse |
| FM v1 | E.T. 58k | 3 | noise | — | Timestep direction bug |
| FM v2 | E.T. 58k | 3 | 0.822 | 2.8× | First working FM |
| FM v5 | Merged 326k | 3 | 0.388 | 1.8× | Best position-only model |
| FM v6 | Merged 326k | 3 | 0.466 | 1.9× | Smooth loss added |
| FM v7 | Merged 660k | **4 (raw yaw)** | 0.466 | 1.9× | **Mode collapse from yaw wraparound** |
| FM v8 | Merged 660k | **5 (sin/cos)** | TBD | TBD | HumanML3D captions + sin/cos fix |

The addition of AMASS data reduced val loss by 53% and halved the overfitting gap. The improvement is attributable to both increased data volume and improved label quality: AMASS-derived dolly-in samples have 100% label accuracy, compared to 57% in E.T. The v7 regression demonstrates that representation choices for individual dimensions can override the benefit of additional data, underscoring the importance of matching data topology to model assumptions (Section 5.6).

### 6.3 Qualitative Evaluation by Motion Type

**[INSERT: 3×3 grid of generated visualisations for all 9 motion types (orbit, dolly-in, dolly-out, static, track, pan-left, pan-right, crane-up, crane-down)]**

Selected results:

**Orbit**: The camera follows a clear arc trajectory in the top-down view, with azimuth increasing linearly from 15° to 70° over 48 frames. Camera-to-person distance remains constant at approximately 3.5m throughout. This is the most visually convincing result, made possible by the 29,813 orbit samples contributed by AMASS (E.T. contributed only 2 orbit samples).

**[INSERT: Orbit result - 3D view + top-down + distance curve]**

**Dolly-in**: Camera-to-person distance decreases monotonically from approximately 5m to 1.5m. The camera orientation remains stable (azimuth constant at ≈175°), consistent with a pure push-in movement. This behaviour was absent in all E.T.-only models, where the distance showed no directional trend.

**[INSERT: Dolly-in result - 3D view + distance curve showing monotonic decrease]**

**Static**: All camera position and orientation dimensions remain within 0.05 units of their initial values across all 48 frames, verified by the post-processing freeze step. Person trajectory also shows no net displacement.

**[INSERT: Static result - Camera Position panel showing flat lines]**

### 6.4 Text Alignment Test

Three different text prompts for the same motion type (dolly-in) were used to test whether the model responds to semantic variation in the text:

1. "As the character moves forward, the camera pushes in"
2. "The camera slowly dollies in on the standing character"
3. "A close-up push in as the character stands still"

**[INSERT: Three dolly-in results with different text prompts, showing Camera-Person Distance curves side by side]**

All three produce decreasing distance curves, suggesting the motion type embedding and text conditioning jointly drive the dolly-in behaviour.

### 6.5 Post-Processing Analysis

Raw model outputs contain high-frequency oscillations in the person trajectory (median amplitude 0.05 units, period 6-10 frames) caused by accumulated Euler integration error over 50 ODE steps. Four post-processing stages are applied sequentially:

1. **Savitzky-Golay smoothing**: person trajectory window 31, camera position window 21, camera angle window 31. Removes high-frequency jitter while preserving the overall trajectory shape.

2. **Person trajectory regularisation**: a greedy segmentation algorithm detects piecewise-linear segments in the person position (dimensions 0-2). Each segment is straightened by fitting a line between its start and end points; dimensions with range below 0.08m within a segment are locked to their mean. Segments are connected by cubic spline interpolation, producing smooth transitions at turning points. If the total displacement is below 0.08m, the person is frozen as stationary. This enforces physically plausible motion: a person described as "walking forward" produces a straight-line trajectory, while "walks right then turns left" produces two connected line segments with a smooth turn.

3. **Sin/cos yaw smoothing**: the sin and cos yaw components are smoothed independently with Savitzky-Golay filtering, then renormalised to the unit circle ($\sin^2 + \cos^2 = 1$). Unlike angular unwrapping, this approach is numerically stable and introduces no phase ambiguity.

4. **Static dimension freezing**: any trajectory dimension with range < 0.05 units is replaced by its mean, eliminating residual drift in nominally static shots.

**[INSERT: Before/after smoothing comparison on person trajectory (showing oscillation removal)]**

**[INSERT: Example of person trajectory regularisation - raw zigzag vs straightened piecewise-linear path]**

---

## 7. Critical Self-Evaluation

### What Worked

**Flow Matching outperformed DDPM** for this task. The straight-path ODE formulation produced better mode coverage with fewer training samples. The critical factor was correctly mapping the continuous time parameter to the DDPM-style timestep embedding, a subtle implementation detail that took multiple iterations to identify.

**AMASS augmentation was the single largest improvement.** The 53% reduction in validation loss came primarily from replacing noisy E.T. labels with AMASS samples whose camera behaviour was generated by construction. This addresses a fundamental limitation of the E.T. dataset for this task: camera motion labels derived from natural language are inherently imprecise.

**Systematic data pipeline debugging.** The project uncovered six distinct data pipeline failures (zero person trajectories, silent exception swallowing, caption priority inversion, numerical outliers, timestep direction reversal, angular wraparound) that collectively produced models that appeared to train correctly but generated meaningless outputs. The debugging methodology — normalisation statistics as an early sanity check, targeted diagnostic scripts, exception handler removal, and mid-training generation tests — proved more effective than adjusting model hyperparameters.

**Multi-source data integration.** Combining E.T. (real film data with noisy labels), AMASS (motion capture with synthetic but accurate labels), and HumanML3D (motion capture with human-written captions) proved complementary. Each source addresses a different weakness: E.T. provides naturalistic camera work, AMASS provides clean labels and class balance, and HumanML3D provides lexically diverse text descriptions.

### What Did Not Work Well

**The joint generation formulation is fundamentally difficult.** Generating 432-dimensional joint trajectories from text alone requires the model to simultaneously satisfy position, orientation, and semantic constraints. DIRECTOR sidesteps this by conditioning on ground-truth character motion. The two-stage decomposition (Stage 1: person generation; Stage 2: camera given person) showed promising training metrics but did not produce convincing spatial coupling at inference time. The person branch suffered severe overfitting (train 0.08, val 2.02) when trained on the imbalanced E.T. dataset.

**Camera tracking of person is not learned.** The camera and person trajectories are generated from a shared latent representation but their spatial relationship is not explicitly constrained. For motion types requiring the camera to follow the person (track), the model produces a camera that moves independently. Incorporating an explicit spatial constraint loss (e.g., maintaining constant camera-to-person distance) is straightforward in principle but requires clean ground-truth distance data, which E.T. does not reliably provide.

**E.T. label quality limits the ceiling.** With dolly-in labels only 57% accurate in E.T., any model trained primarily on E.T. cannot exceed random performance on distance-direction metrics for dolly-in/out. AMASS augmentation addresses this but covers only the camera trajectory dimension; the person trajectories in AMASS come from laboratory motion capture and do not reflect the diversity of naturalistic human motion in film.

### Future Work

1. **HumanML3D full integration**: The current training set includes 9,303 HumanML3D motions matched to locally available AMASS subsets (CMU, ACCAD, BMLrub, KIT, Eyes_Japan_Dataset). Downloading the remaining AMASS subsets (BMLmovi, HDM05, SSM_synced) would increase HumanML3D coverage from 64% to approximately 90%, adding high-quality human-written captions for over 13,000 motions.
2. **LLM prompt normalisation**: Add a lightweight LLM rewriting layer (Gemini Flash free tier) at inference time to normalise arbitrary user input to the training caption style before CLIP encoding, reducing the train-inference text distribution gap.
3. **Camera-relative-position conditioning**: Encode the initial camera-to-person relative position (front, behind, left, right) in the text caption during training, enabling users to specify "camera behind the person" at inference.
4. **3D visualisation with human mesh**: Replace stick-figure/point visualisation with SMPL body meshes or bounding-box proxies, using the generated yaw to orient the person model correctly.
5. **Spatial constraint loss**: Add a differentiable camera-to-person distance loss using AMASS samples where distance ground truth is exact.
6. **Temporal consistency loss**: Add a velocity smoothness term directly on the generated trajectories to reduce Euler integration oscillations without relying on post-processing.

---

## 8. References

- Courant, R., Dufour, N., Wang, X., Christie, M., & Kalogeiton, V. (2024). E.T. the Exceptional Trajectories: Text-to-Camera-Trajectory Generation with Character Awareness. *ECCV 2024*.
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS 2020*.
- Ho, J., & Salimans, T. (2022). Classifier-Free Diffusion Guidance. *NeurIPS Workshops 2021*.
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow Matching for Generative Modeling. *ICLR 2023*.
- Mahmood, N., Ghorbani, N., Troje, N. F., Pons-Moll, G., & Black, M. J. (2019). AMASS: Archive of Motion Capture as Surface Shapes. *ICCV 2019*.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Bengio, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI 2018*.
- Radford, A., et al. (2021). Learning Transferable Visual Models from Natural Language Supervision. *ICML 2021*.
- Romero, J., Tzionas, D., & Black, M. J. (2017). Embodied Hands: Modeling and Capturing Hands and Bodies Together. *SIGGRAPH Asia 2017*.
- Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models. *ICLR 2021*.
- Tevet, G., Raab, S., Gordon, B., Shafir, Y., Cohen-Or, D., & Bermano, A. H. (2023). Human Motion Diffusion Model. *ICLR 2023*.
- Ye, V., Pavlakos, G., Malik, J., & Kanazawa, A. (2023). Decoupling Human and Camera Motion from Videos in the Wild. *CVPR 2023*.
- Guo, C., Zou, S., Zuo, X., Wang, S., Ji, T., Li, X., & Cheng, L. (2022). Generating Diverse and Natural 3D Human Motions from Text. *CVPR 2022*.
- Zhang, M., Cai, Z., Pan, L., Hong, F., Guo, X., Yang, L., & Liu, Z. (2022). MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model. *arXiv preprint arXiv:2208.15001*.
