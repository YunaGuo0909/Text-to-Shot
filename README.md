# Text-to-Shot

文本到联合人物-相机轨迹生成，当前仅维护 **Flow Matching** 一条主线。

> 说明：历史方案、实验心路和对比分析统一放在 `report_v2.md`，README 只保留当前可用流程。

---

## 任务定义

输入一段文本，生成一段长度为 `T=48` 的联合轨迹：

- 人物轨迹：`(T, Dp)`，当前训练配置常用 `Dp=3`（`px, py, pz`）
- 相机轨迹：`(T, 6)`，`[tx, ty, tz, azimuth, elevation, roll]`
- 联合向量：`[person_flat, camera_flat]`

---

## 安装

要求：

- Python >= 3.10
- 推荐 CUDA GPU（训练）

安装：

```bash
pip install -e .
```

---

## 快速开始

默认示例使用 `experiments/flow_matching/configs/v9.yaml`。

### 1) 准备数据（如果已有可跳过）

将训练数据整理为：

- `<data_root>/train_index.json`
- `<data_root>/test_index.json`
- 索引中指向的人物/相机 `.npy` 轨迹文件

如果需要从 E.T. / AMASS / HumanML3D 重建数据，可用 `scripts/` 下的数据脚本。

### 2) 计算归一化统计（推荐）

```bash
python scripts/compute_norm_stats.py \
  --data-root /transfer/merged-v9b \
  --index-file train_index.json \
  --person-dim 3 \
  --camera-dim 6
```

默认输出：`/transfer/merged-v9b/norm_stats.json`

### 3) 训练

```bash
PYTHONPATH=. python experiments/flow_matching/train.py \
  --config experiments/flow_matching/configs/v9.yaml \
  --device cuda
```

### 4) 推理

```bash
PYTHONPATH=. python experiments/flow_matching/generate.py \
  --checkpoint /transfer/fm-v9b-checkpoints/fm_final.pth \
  --text "A person walks toward camera" \
  --motion dolly-in \
  --shot-type medium-shot \
  --guidance-scale 3.0
```

可选：开启硬约束后处理

```bash
--enforce-constraints
```

---

## 输出文件

默认输出目录由配置文件 `paths.output_dir` 指定，典型产物：

- `fm_person_<tag>.npy`
- `fm_camera_<tag>.npy`
- `fm_joint_<tag>.png`

---

## 关键目录

```text
experiments/flow_matching/
  configs/v9.yaml            # 当前主配置
  train.py                   # 训练入口
  generate.py                # 推理入口
  postprocess_constraints.py # 可选硬约束后处理

src/
  data/dataset.py            # 数据加载与归一化
  models/                    # 核心模型组件

scripts/
  compute_norm_stats.py      # 归一化统计
  preprocess_et_data.py      # 数据预处理（按需）
  prepare_amass.py           # 数据准备（按需）
  prepare_humanml3d.py       # 数据准备（按需）
  merge_datasets.py          # 数据合并（按需）
```

---

## 文档

- 主报告：`report_v2.md`
- 旧版报告：`report_v1.md`
