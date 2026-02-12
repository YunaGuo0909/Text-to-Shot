# Technical Design: AI-Driven Storyboard Generation

## 1. System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    STORYBOARD GENERATION PIPELINE                        │
│                                                                          │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────┐  ┌─────┐│
│  │  Script   │─▶│Shot Decomposer│─▶│ Shot-Level  │─▶│ Camera   │─▶│Board││
│  │  Input    │  │ (LLM-based)  │  │ Generator   │  │Trajectory│  │Render││
│  └──────────┘  └──────────────┘  └─────────────┘  └──────────┘  └─────┘│
│                       │                 │               │           │    │
│                  Shot prompts      Per-shot 3D    Keyframe→     Panels  │
│                 + shot types     configurations   Spline path  + paths  │
│                 + camera motion                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

## 2. Module Design

### Module 1: Shot Decomposer (新增模块)

**目标**: 将一段完整的场景描述自动分解为多个镜头级别的提示词。

**方法**: 使用 LLM (如 GPT-4 / open-source LLM) 进行剧本分析：
- 输入: 场景描述文本 (e.g., "Two people meet at a cafe. Person A waves and walks toward Person B. They shake hands and sit down together.")
- 输出: 结构化的镜头列表，每个镜头包含：
  - `shot_description`: 镜头描述 (e.g., "Person A waves at Person B")
  - `shot_type`: 镜头类型 (close-up / medium / wide / over-shoulder)
  - `shot_index`: 镜头顺序编号
  - `duration_hint`: 预估时长提示
  - `camera_motion`: **镜头运动类型** (static / dolly-in / dolly-out / pan-left / pan-right / track / crane-up / crane-down)

**Prompt Engineering 示例**:
```
You are a professional film storyboard artist. Given a scene description,
decompose it into a sequence of cinematic shots. For each shot, specify:
1. A concise action description for two characters (A and B)
2. The recommended shot type (close-up, medium-shot, wide-shot, over-the-shoulder)
3. The camera motion (static, dolly-in, dolly-out, pan-left, pan-right, track, crane-up, crane-down)
4. The shot order

Scene: "{scene_description}"

Output as JSON array.
```

### Module 2: Shot-Level Generator (基于原论文模型扩展)

**基础**: 直接复用/微调原论文的 Joint Character-Camera Diffusion Model

**扩展点**:

#### 2a. Shot Type Conditioning (镜头类型条件控制)
- 在原始的 text conditioning 基础上，增加 **shot type embedding**
- Shot types 编码为 learnable embeddings，通过 FiLM 注入网络
- 这样可以控制生成的镜头构图符合特定类型（如近景人物更大、远景人物更小）

```python
# Shot type conditioning
shot_types = ['close-up', 'medium-shot', 'wide-shot', 'over-the-shoulder', 'two-shot']
shot_type_embedding = nn.Embedding(len(shot_types), embed_dim)

# Inject into FiLM alongside text and timestep
film_params = film_generator(text_embed + shot_type_embed + timestep_embed)
```

#### 2b. Inter-Shot Coherence (镜头间连贯性)
- **空间连续性**: 前一个镜头的角色全局位置作为下一个镜头的初始化约束
- **180度规则**: 确保摄影机不越过两个角色之间的轴线
- **实现方式**: 在反向扩散过程中加入 guidance：

```python
def coherence_guidance(prev_shot, current_noisy, t):
    """Guide denoising to maintain spatial coherence with previous shot"""
    # Soft constraint on character positions
    position_loss = mse(current_noisy.char_positions, prev_shot.char_positions)
    # 180-degree rule constraint
    angle_loss = axis_crossing_penalty(prev_shot.camera, current_noisy.camera)
    return gradient(position_loss + angle_loss)
```

### Module 3: Camera Motion Trajectory Generator (🆕 核心创新模块)

**目标**: 将原论文生成的静态 Toric 镜头位姿扩展为时序上的镜头运动轨迹。

**核心思路**:
原论文生成的是单一静态镜头配置 `x_C ∈ R^6`（Toric参数），这里将其扩展为生成 **T 个关键帧**组成的时序轨迹 `X_C ∈ R^(K×6)`，并通过样条插值得到连续平滑的镜头运动路径。

**方法**: 两阶段方案（更可控、更稳定）：

#### Stage 1: 关键帧生成 (Keyframe Generation)
基于镜头运动类型和文本描述，生成 K 个关键帧的 Toric 参数：

```python
class CameraTrajectoryGenerator:
    """
    Generates camera motion trajectories from static shot configuration.
    
    Given:
    - Start Toric state x_C_start (from diffusion model)
    - Camera motion type (dolly-in, pan-left, etc.)
    - Duration T
    
    Produces: K keyframe Toric states → smooth spline trajectory
    """
    
    # 运动类型到Toric参数变化的映射
    MOTION_PROFILES = {
        'static':     {'theta': 0, 'phi': 0, 'scale': 0},       # 固定
        'dolly-in':   {'theta': 0, 'phi': 0, 'scale': -0.3},    # 推近
        'dolly-out':  {'theta': 0, 'phi': 0, 'scale': +0.3},    # 拉远
        'pan-left':   {'theta': -0.4, 'phi': 0, 'scale': 0},    # 左摇
        'pan-right':  {'theta': +0.4, 'phi': 0, 'scale': 0},    # 右摇
        'crane-up':   {'theta': 0, 'phi': +0.3, 'scale': 0},    # 升
        'crane-down': {'theta': 0, 'phi': -0.3, 'scale': 0},    # 降
        'track':      {'theta': +0.2, 'phi': 0, 'scale': -0.1}, # 跟踪
    }
```

#### Stage 2: 样条插值 (Spline Interpolation)
关键帧之间用 Catmull-Rom 样条插值，确保轨迹平滑且专业：

```python
def interpolate_trajectory(keyframes, num_frames, method='catmull-rom'):
    """
    Interpolate between Toric keyframes to produce smooth trajectory.
    
    Args:
        keyframes: (K, 6) array of Toric keyframe states
        num_frames: Total number of output frames (T)
        method: Interpolation method
    
    Returns:
        trajectory: (T, 6) smooth camera trajectory in Toric space
    """
```

#### 可选进阶: 学习型轨迹生成 (Learned Trajectory Generation)
如果时间允许，可以训练一个小型条件扩散模型直接生成轨迹：
- 输入：文本描述 + 镜头运动类型 + 角色动作
- 输出：`X_C ∈ R^(T×6)` 时序轨迹
- 参考：DanceCamera3D (Wang et al., 2024) 的镜头轨迹生成架构

### Module 4: Storyboard Renderer (新增模块)

**目标**: 将生成的3D配置 + 镜头运动轨迹渲染为2D故事板面板。

**方法**:
1. **Stick Figure Rendering**: 使用 matplotlib/Open3D 将 SMPL 关节位置绘制为简笔画人物
2. **Camera Framing**: 根据 Toric camera parameters 确定画面裁剪和透视
3. **Camera Path Overlay**: 🆕 在面板上叠加镜头运动轨迹路径（箭头、运动方向）
4. **Panel Layout**: 将多个镜头排列为漫画式的故事板布局
5. **Annotation**: 添加镜头编号、描述文字、镜头类型、运动类型标注

```python
class StoryboardRenderer:
    def render_panel(self, char_a_pose, char_b_pose, camera_params, 
                     trajectory, shot_info):
        """Render a single storyboard panel with camera path overlay"""
        # 1. Transform poses to camera view
        # 2. Project 3D joints to 2D
        # 3. Draw stick figures
        # 4. Draw camera motion trajectory as arrow overlay
        # 5. Add frame border and annotations
        return panel_image
```

## 3. Data Pipeline

### 训练数据
- **InterHuman Dataset**: 双人交互动作数据集，包含文本标注
- **InterGen Dataset**: 带有文本描述的多人动作数据
- **CineScale2**: 镜头类型标注（用于 shot type conditioning）
- **Movie clip datasets**: MovieNet 用于学习镜头序列模式
- **DanceCamera3D Dataset**: 🆕 镜头运动轨迹数据（用于学习型轨迹生成）

### 数据处理流程
```
Raw Motion Data (BVH/SMPL) 
    → SMPL Parameter Extraction (22 joints × 6D rotation)
    → Global Placement Vector Computation
    → Toric Camera Parameter Computation
    → Camera Motion Type Classification (if available)
    → Text-Shot Pair Construction
    → Training Data
```

## 4. Evaluation Plan

### 定量指标
| Metric | What it measures |
|--------|-----------------|
| FID (Fréchet Inception Distance) | Quality of generated poses |
| Shot Type Accuracy | Whether generated camera matches target shot type |
| Spatial Coherence Score | Position consistency between consecutive shots |
| R-Precision | Text-motion alignment quality |
| Diversity Score | Variety of generated storyboards from same script |
| **Trajectory Smoothness** | 🆕 Jerk (三阶导数) of camera trajectory |
| **Motion Type Accuracy** | 🆕 Whether trajectory matches requested motion type |
| **Trajectory-Action Consistency** | 🆕 How well camera motion follows character actions |

### 定性评估
- **User Study**: 让影视专业人士评价故事板的专业性和可用性
- **Visual Comparison**: 与手动创建的故事板对比
- **Trajectory Visualization**: 🆕 展示不同运动类型的轨迹效果
- **Ablation Visualization**: 展示各模块的贡献

## 5. Technology Stack

| Component | Technology |
|-----------|------------|
| **Package Manager** | **uv** |
| Deep Learning Framework | PyTorch |
| Human Body Model | SMPL (smplx library) |
| Diffusion Model | Custom (based on MDM architecture) |
| Text Encoder | CLIP / Sentence-BERT |
| LLM for Shot Decomposition | OpenAI API / Local LLM (Llama) |
| Spline Interpolation | scipy.interpolate (CubicSpline / CatmullRom) |
| 3D Visualization | matplotlib 3D / Open3D / PyRender |
| Storyboard Rendering | Pillow / matplotlib |
| Experiment Tracking | Weights & Biases / TensorBoard |
| Version Control | Git + GitHub |

## 6. Key Innovation Points (创新点)

1. **首个从剧本到完整故事板的端到端 AI 管线** — 将单镜头生成扩展为多镜头序列
2. **镜头运动轨迹生成** 🆕 — 将静态镜头位姿扩展为时序相机轨迹（推/拉/摇/移/升/降）
3. **镜头类型条件控制** — 通过 shot type embedding 实现细粒度的摄影构图控制
4. **镜头间连贯性约束** — 通过 guidance 机制确保故事板空间逻辑一致
5. **可视化故事板渲染** — 将 3D 配置 + 轨迹自动转化为专业级故事板面板（含运动路径标注）

## 7. Scope Management (范围管理)

### Must Have (必须完成) — Week 1-6
- [ ] 基线模型复现
- [ ] Shot Decomposer 模块
- [ ] 多镜头序列生成
- [ ] **基于规则的镜头运动轨迹生成（关键帧 + 样条插值）**
- [ ] 基础故事板可视化（含轨迹路径标注）
- [ ] 定量评估

### Should Have (应该完成) — Week 7-8
- [ ] Shot type conditioning
- [ ] Inter-shot coherence guidance
- [ ] Professional storyboard panel rendering
- [ ] **多种镜头运动类型支持（推/拉/摇/移/升/降/跟踪）**

### Nice to Have (锦上添花) — 仅时间充裕时
- [ ] **学习型轨迹生成（条件扩散模型）**
- [ ] Interactive web demo (Gradio/Streamlit)
- [ ] 轨迹3D可视化动画
- [ ] Video generation from storyboard
