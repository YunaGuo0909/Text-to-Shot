# AI-Driven Storyboard Generation
## Extending Text-to-Shot Diffusion for Multi-Shot Visual Pre-production

### Project Overview
This project extends the SIGGRAPH 2026 paper "From Script to Shot: Joint Generation of Camera Pose and Dual-Human 3D Actions" toward automated storyboard generation with **camera motion trajectories** for media pre-production. The original work generates a single static shot configuration (camera pose + dual-human 3D poses) from text. This project extends it in two key directions:
1. **Multi-shot pipeline**: Takes a full scene description and produces a coherent sequence of visual storyboard panels
2. **Camera motion trajectory generation**: Extends static camera pose to temporal camera trajectories (dolly, pan, track, etc.)

This project aligns with Mo-sys's AI/ML innovation area: **Pre-production → Storyboarding + Camera Motion**.

---

### 10-Week Timeline (Feb 12 - Apr 23, 2026)

#### Phase 1: Research & Foundation (Week 1-2, Feb 12 - Feb 25)

| Week | Tasks | Deliverables |
|------|-------|-------------|
| Week 1 (Feb 12-18) | - Literature review: storyboard generation, cinematic AI, diffusion models, DanceCamera3D<br>- Deep dive into the original paper's codebase<br>- Study Toric camera parameterization, SMPL representation<br>- Set up uv environment | Literature notes, dev environment ready |
| Week 2 (Feb 19-25) | - Reproduce baseline single-shot generation model<br>- Validate results against paper's reported outputs<br>- Study camera trajectory representations (Bézier, spline, keyframe)<br>- Weekly tutorial discussion | Baseline reproduction results |

#### Phase 2: Core Development (Week 3-6, Feb 26 - Mar 25)

| Week | Tasks | Deliverables |
|------|-------|-------------|
| Week 3 (Feb 26-Mar 4) | - Design multi-shot pipeline + camera trajectory architecture<br>- Implement LLM-based Shot Decomposer module<br>- Define shot transition logic & camera motion types | Architecture design document |
| Week 4 (Mar 5-11) | - Implement camera trajectory generation module (Toric keyframe → spline interpolation)<br>- Add shot type conditioning (close-up, medium, wide, etc.)<br>- Weekly tutorial discussion | Camera trajectory prototype |
| Week 5 (Mar 12-18) | - Implement storyboard visualization/rendering with camera path overlay<br>- 3D-to-2D panel rendering with stick figures + camera motion arrows<br>- Integrate: Script → Decomposition → Generation → Trajectory → Visualization | Storyboard renderer + trajectory viz |
| Week 6 (Mar 19-25) | - Add inter-shot coherence constraints (spatial continuity, 180° rule)<br>- Camera trajectory smoothness & cinematic constraints<br>- End-to-end testing and bug fixes<br>- Weekly tutorial discussion | Integrated pipeline v1 |

#### Phase 3: Evaluation & Refinement (Week 7-8, Mar 26 - Apr 8)

| Week | Tasks | Deliverables |
|------|-------|-------------|
| Week 7 (Mar 26-Apr 1) | - Design evaluation metrics (FID, shot type accuracy, trajectory smoothness)<br>- Run quantitative experiments + ablation studies<br>- Weekly tutorial discussion | Experimental results |
| Week 8 (Apr 2-8) | - Additional experiments, qualitative results<br>- Compare against baseline approaches<br>- Polish visualization outputs | Final experimental results |

#### Phase 4: Documentation & Submission (Week 9-10, Apr 9 - Apr 23)

| Week | Tasks | Deliverables |
|------|-------|-------------|
| Week 9 (Apr 9-15) | - Write research report: background, methodology, experiments, camera trajectory analysis<br>- Critical self-evaluation section<br>- Weekly tutorial discussion | Report draft |
| Week 10 (Apr 16-23) | - Finalize report<br>- Record 8-min project explanation video<br>- Final code cleanup and documentation on GitHub<br>- Submit via Brightspace + GitHub | **Final Submission** |

---

### Assessment Alignment

| Assessment Criteria | Weight | How This Project Addresses It |
|---------------------|--------|-------------------------------|
| Clear structure of project report | 5% | Well-organized report with introduction, background, methodology, experiments, evaluation |
| In-depth background research | 15% | Comprehensive review of diffusion models, cinematography AI, storyboarding, SMPL, Toric space, camera trajectory generation |
| In-depth methodology and experiments | 25% | Novel multi-shot pipeline, camera trajectory generation, shot decomposition, coherence constraints, quantitative evaluation |
| Critical self-evaluation | 15% | Honest analysis of limitations, comparison with professional storyboarding, future work |
| Weekly tutorial engagement | 10% | Regular participation and progress sharing |
| Clear implementation on GitHub | 15% | Well-structured codebase with README, clear commits, modular design, uv environment |
| 8-min voiceover recording | 15% | Walkthrough of architecture, code, and results with camera trajectory demo |

---

### Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Baseline model reproduction fails | High | Start Week 2, seek help from original authors, use pretrained weights if available |
| Training data unavailable | High | Use publicly available datasets (InterHuman, InterGen), synthetic data |
| Camera trajectory generation quality is poor | Medium | Fallback to keyframe interpolation (Bézier/Catmull-Rom) without learned model |
| Multi-shot coherence is poor | Medium | Simplify to sequential independent shots as fallback |
| Time overrun (only 10 weeks!) | High | Prioritize: baseline→trajectory→pipeline→eval→report; cut Nice-to-Have features early |
| Insufficient compute resources | Medium | Use cloud GPU (Colab Pro, university HPC), reduce model size |
