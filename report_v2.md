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

The model generates a joint trajectory vector $y = [\text{person\_flat}, \text{camera\_flat}]$ of dimension 432 (48 frames × 3 person + 48 frames × 6 camera). Camera state is $(t_x, t_y, t_z, \text{azimuth}, \text{elevation}, \text{roll})$. Person state is the SMPL-H root translation $(p_x, p_y, p_z)$.

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

After processing 5,401 AMASS files (CMU, ACCAD, BMLrub subsets), 268,317 samples were generated. Merging with E.T. yielded a final training set of 326,870 samples.

**[INSERT: Class distribution comparison - E.T. only vs merged dataset (stacked bar or side-by-side)]**

| Motion Type | E.T. only | Merged |
|-------------|-----------|--------|
| static | 66.7% | 21.9% |
| dolly-out | 0.4% | 9.0% |
| orbit | 0.0% | 8.9% |
| dolly-in | 10.6% | 11.0% |

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

### 5.6 Mode Collapse in Joint Diffusion Models

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

| Model | Data | Val Loss | Train/Val Gap |
|-------|------|----------|---------------|
| FM v2 | E.T. 58k | 0.822 | 2.8× |
| FM v5 | Merged 326k | **0.388** | **1.8×** |

The addition of AMASS data reduced val loss by 53% and halved the overfitting gap. The improvement is attributable to both increased data volume and improved label quality: AMASS-derived dolly-in samples have 100% label accuracy, compared to 57% in E.T.

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

Raw model outputs contain high-frequency oscillations in the person trajectory (median amplitude 0.05 units, period 6-10 frames) caused by accumulated Euler integration error over 50 ODE steps. Three post-processing stages were applied:

1. **Savitzky-Golay smoothing**: person trajectory window 31, camera position window 21, camera angle window 31.
2. **Static dimension freezing**: any trajectory dimension with range < 0.05 units is replaced by its mean, eliminating residual drift in nominally static shots.

**[INSERT: Before/after smoothing comparison on person trajectory (showing oscillation removal)]**

---

## 7. Critical Self-Evaluation

### What Worked

**Flow Matching outperformed DDPM** for this task. The straight-path ODE formulation produced better mode coverage with fewer training samples. The critical factor was correctly mapping the continuous time parameter to the DDPM-style timestep embedding, a subtle implementation detail that took multiple iterations to identify.

**AMASS augmentation was the single largest improvement.** The 53% reduction in validation loss came primarily from replacing noisy E.T. labels with AMASS samples whose camera behaviour was generated by construction. This addresses a fundamental limitation of the E.T. dataset for this task: camera motion labels derived from natural language are inherently imprecise.

**Systematic data pipeline debugging.** The project uncovered four distinct data pipeline failures (zero person trajectories, silent exception swallowing, caption priority inversion, numerical outliers) that collectively produced a model that appeared to train correctly but generated meaningless outputs. The debugging methodology — normalisation statistics as an early sanity check, targeted diagnostic scripts, exception handler removal — proved more effective than adjusting model hyperparameters.

### What Did Not Work Well

**The joint generation formulation is fundamentally difficult.** Generating 432-dimensional joint trajectories from text alone requires the model to simultaneously satisfy position, orientation, and semantic constraints. DIRECTOR sidesteps this by conditioning on ground-truth character motion. The two-stage decomposition (Stage 1: person generation; Stage 2: camera given person) showed promising training metrics but did not produce convincing spatial coupling at inference time. The person branch suffered severe overfitting (train 0.08, val 2.02) when trained on the imbalanced E.T. dataset.

**Camera tracking of person is not learned.** The camera and person trajectories are generated from a shared latent representation but their spatial relationship is not explicitly constrained. For motion types requiring the camera to follow the person (track), the model produces a camera that moves independently. Incorporating an explicit spatial constraint loss (e.g., maintaining constant camera-to-person distance) is straightforward in principle but requires clean ground-truth distance data, which E.T. does not reliably provide.

**E.T. label quality limits the ceiling.** With dolly-in labels only 57% accurate in E.T., any model trained primarily on E.T. cannot exceed random performance on distance-direction metrics for dolly-in/out. AMASS augmentation addresses this but covers only the camera trajectory dimension; the person trajectories in AMASS come from laboratory motion capture and do not reflect the diversity of naturalistic human motion in film.

### Future Work

1. **Full AMASS integration**: Re-run preprocessing with improved multi-phase action inference, re-merge, and retrain.
2. **Spatial constraint loss**: Add a differentiable camera-to-person distance loss using AMASS samples where distance ground truth is exact.
3. **LLM prompt normalisation**: Add a lightweight LLM rewriting layer (Gemini Flash free tier) at inference time to normalise arbitrary user input to the E.T./AMASS caption style before CLIP encoding, reducing the train-inference text distribution gap.
4. **Larger model**: The current 24.7M parameter model may be under-parameterised for a 432-dimensional joint generation task. Scaling to 512 hidden dimensions would increase capacity with modest compute cost.
5. **Temporal consistency loss**: Add a velocity smoothness term directly on the generated trajectories to reduce Euler integration oscillations without relying on post-processing.

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
