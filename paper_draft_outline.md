# Paper Draft: Text-to-Shot — Joint Person-Camera Trajectory Generation via Flow Matching

## Target Venue: CVEU Workshop @ CVPR 2026 / BMVC 2026 / MIG 2026

---

## Title Options

1. "Text-to-Shot: Joint Person-Camera Trajectory Generation from Text via Conditional Flow Matching"
2. "Flow Matching for Joint Cinematographic Trajectory Generation from Natural Language"
3. "Generating Coordinated Person and Camera Motion from Text for Pre-production Storyboarding"

---

## Abstract (~150 words)

We present Text-to-Shot, a system that jointly generates person root trajectories and cinematic camera trajectories from natural language descriptions. Unlike prior work that conditions camera generation on pre-existing character motion (DIRECTOR, Courant et al., 2024), our approach generates both outputs simultaneously from text alone, enabling fully automated scene pre-visualisation. We employ Conditional Flow Matching with a dual-branch Transformer denoiser featuring cross-attention between person and camera branches. Person orientation is encoded as (sin, cos) components to avoid angular discontinuities in the flow matching interpolation. Training data is assembled from three complementary sources: the E.T. film dataset for naturalistic camera work, AMASS motion capture for accurate person trajectories with balanced camera labels, and HumanML3D for human-written motion descriptions. We demonstrate controllable generation across 9 camera motion types (orbit, dolly-in, static, track, etc.) and analyse failure modes including mode collapse, label noise, and angular representation issues encountered during development.

---

## 1. Introduction (~1 page)

### Motivation
- Pre-production storyboarding is time-consuming and requires cinematography expertise
- Automating "text to camera + person motion" reduces this barrier
- Existing tools generate camera only (DIRECTOR) or require pre-existing character animation

### Gap
- DIRECTOR (ECCV 2024): camera FROM person motion + text. Person must already exist.
- PULP MOTION (2025): joint generation, but uses framing as auxiliary modality. Our approach is text-only.
- No prior work addresses: text → (person root + yaw) + (camera 6DoF) jointly via flow matching

### Contributions
1. Joint person-camera trajectory generation from text alone, without pre-existing character data
2. First application of Conditional Flow Matching to cinematographic trajectory generation, with analysis of angular representation requirements
3. Multi-source training pipeline combining E.T. (film), AMASS (MoCap), and HumanML3D (human-annotated) datasets with systematic label quality analysis
4. Comprehensive ablation across DDPM vs Flow Matching, data sources, and representation choices, documenting 8 failure modes and their solutions

---

## 2. Related Work (~1 page)

### 2.1 Camera Trajectory Generation
- E.T./DIRECTOR (Courant et al., ECCV 2024): camera conditioned on character
- GenDoP (ICCV 2025): auto-regressive camera from text
- Director3D (NeurIPS 2024): camera + 3D scene from text
- CameraCtrl, MotionCtrl: camera control in video diffusion

### 2.2 Human Motion Generation
- MDM (Tevet et al., ICLR 2023): motion diffusion model
- MotionDiffuse (Zhang et al., 2022): text-driven motion diffusion
- MLD (Chen et al., CVPR 2023): motion latent diffusion

### 2.3 Joint Person-Camera Generation
- PULP MOTION (2025): closest to our work, uses framing modality
- SymphoMotion (2025): joint camera + object dynamics for video
- Ours: text-only conditioning, flow matching, sin/cos orientation

### 2.4 Flow Matching
- Lipman et al. (ICLR 2023): OT-CFM
- Applications to motion: FLOAT (ICCV 2025), GoalFlow (CVPR 2025)

---

## 3. Method (~2 pages)

### 3.1 Problem Formulation
- Joint trajectory: y = [person(T×5), camera(T×6)], T=48, total dim=528
- Person: (px, py, pz, sin_yaw, cos_yaw)
- Camera: (tx, ty, tz, azimuth, elevation, roll)
- Input: text description + shot type + camera motion type

### 3.2 Architecture
- Dual-branch Transformer (6 layers, 256 hidden, 4 heads)
- Person branch ↔ Camera branch cross-attention
- Conditioning: CLIP text (512d) + timestep (128d) + shot type (64d) + motion type (64d)
- FiLM modulation at every layer
- ~24.7M parameters

**[Figure 2: Architecture diagram]**

### 3.3 Conditional Flow Matching
- Interpolant: x_t = (1-t)ε + t·x_0
- Velocity target: v = x_0 - ε
- Loss: MSE(v_pred, v_target) + λ·smooth_loss
- Sampling: Euler ODE integration, 50 steps
- CFG: 25% text dropout, guidance scale 1.0-2.0

### 3.4 Angular Representation
- Motivation: raw yaw in radians causes mode collapse in flow matching
- Solution: sin/cos encoding removes discontinuity at ±π
- Recovery: yaw = atan2(sin, cos)
- Empirical validation: v7 (raw yaw) total collapse vs v8 (sin/cos) successful

**[Figure 3: Angular wraparound illustration — raw vs sin/cos interpolation paths]**

### 3.5 Post-Processing
- Savitzky-Golay smoothing (person: w=31, camera position: w=21, angles: w=31)
- Person trajectory regularisation: greedy segmentation → piecewise-linear → cubic spline
- Sin/cos renormalisation after smoothing
- Static dimension freezing (threshold < 0.05)

---

## 4. Data (~1.5 pages)

### 4.1 E.T. Dataset
- 64,948 samples from real film (SLAHMR extraction)
- Severe class imbalance (67% static)
- Label noise: dolly-in accuracy only 57%
- Data quality issues: NaN, requires_grad, caption priority

### 4.2 AMASS Augmentation
- 517,797 synthetic samples from 10,383 MoCap files (CMU, ACCAD, BMLrub, KIT, Eyes_Japan)
- Deterministic camera generation: 100% label accuracy
- Perfect class balance (11.1% per motion type)
- Template captions with inferred action and turning descriptions

### 4.3 HumanML3D Integration
- 83,700 samples with human-written captions
- 9,303/14,616 motions matched to local AMASS subsets (63.6%)
- Average 3 captions per motion, lexically diverse
- Combined with synthetic camera trajectories

### 4.4 Final Dataset
- 660,050 training samples from 3 sources
- Shot type conditioning via caption prefixes

**[Table 1: Dataset source comparison — size, caption quality, label accuracy, person data source]**

**[Table 2: Class distribution evolution — E.T. only → +AMASS → +HumanML3D]**

---

## 5. Experiments (~2 pages)

### 5.1 Setup
- NVIDIA RTX 4080 (16 GB), AdamW, lr=1e-4, cosine schedule, 250 epochs
- WeightedRandomSampler for motion type balance
- Evaluation: qualitative (visualisation) + quantitative (val loss, distance curves)

### 5.2 DDPM vs Flow Matching
- DDPM: mode collapse regardless of sampling strategy or CFG tuning
- Flow Matching: first successful motion type differentiation

**[Table 3: Full experiment history — 8 versions, val loss, qualitative result]**

### 5.3 Data Ablation
- E.T. only (58k): val 0.822, 2.8× gap
- + AMASS (326k): val 0.388, 1.8× gap (53% improvement)
- + HumanML3D (660k): val TBD

### 5.4 Representation Ablation
- person_dim=3 (position only): works
- person_dim=4 (raw yaw): total collapse
- person_dim=5 (sin/cos yaw): works, with orientation

### 5.5 Qualitative Results
- Per-motion-type visualisation: orbit, dolly-in, dolly-out, static, track, pan, crane
- Camera-person distance curves
- Person yaw evolution (facing direction)

**[Figure 4: 3×3 grid of generated trajectories for 9 motion types]**

**[Figure 5: Orbit result — top-down view with camera frustums and person yaw arrow]**

### 5.6 Failure Analysis
- Track: camera does not maintain lateral offset from person
- Dolly-out: limited training data (232 E.T. samples) limits accuracy
- Text alignment: model responds to motion type embedding more than free-form text

---

## 6. Limitations and Future Work (~0.5 page)

- Person representation limited to root position + yaw (no full body pose)
- Camera-person spatial coupling not explicitly constrained
- Inference text style must match training caption distribution
- Future: LLM prompt normalisation, camera-relative-position conditioning, 3D mesh visualisation, full AMASS integration for HumanML3D coverage

---

## 7. Conclusion (~0.25 page)

We presented Text-to-Shot, a system for joint person-camera trajectory generation from text using Conditional Flow Matching. Through systematic data integration and representation design, we demonstrated controllable generation across multiple camera motion types. Our analysis of failure modes — from silent data pipeline bugs to angular representation mismatches — provides practical guidance for applying generative models to mixed Euclidean-angular trajectory data.

---

## Figures List
1. Pipeline overview (text → CLIP → dual-branch FM → person + camera trajectories)
2. Architecture diagram (dual-branch transformer with cross-attention + FiLM)
3. Angular wraparound problem illustration (raw yaw vs sin/cos interpolation)
4. Generated trajectories grid (9 motion types)
5. Detailed orbit/dolly-in results with 6-panel visualisation
6. Data source comparison (E.T. vs AMASS vs HumanML3D statistics)
7. Training loss curves across versions
8. Failure case examples (mode collapse, track spatial decoupling)

## Tables List
1. Dataset composition (source, size, caption quality, label accuracy)
2. Class distribution evolution across dataset versions
3. Full experiment history (8 versions, method, data, val loss, result)
4. Ablation: DDPM vs FM vs Two-stage
5. Ablation: person_dim=3 vs 4 vs 5
