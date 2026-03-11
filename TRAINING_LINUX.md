# 在 Linux 上跑训练 — 具体步骤

在另一台 Linux 电脑上从零到跑通训练，按下面顺序做即可。

---

## 1. 确认环境

- Python ≥ 3.10  
- 有 GPU 时用 CUDA，没有可用 CPU（加 `--device cpu`）

```bash
cd Text-to-Shot-main   # 或你的项目目录
python3 --version      # 确认 >= 3.10
```

---

## 2. 安装依赖

**方式 A：用 pip**

```bash
pip install -e .
# 或
pip install torch torchvision transformers sentence-transformers openai matplotlib Pillow tensorboard wandb scipy numpy tqdm PyYAML einops
```

**方式 B：用 uv（若已安装）**

```bash
uv sync
```

---

## 3. 下载 E.T. 数据集（每次 clone 后执行）

用脚本把 E.T. 下到**固定目录**（如 `/otherlocation/transfer`），不放进仓库，这样每次 clone 后只需重新跑一遍下载即可。

```bash
export PYTHONPATH=.

# 下载到固定路径（例如 /otherlocation/transfer/et-data）
python scripts/download_et_data.py --download-dir /otherlocation/transfer/et-data

# 或用环境变量（写入 .env 或 shell 配置）
export ET_DATA_DOWNLOAD_DIR=/otherlocation/transfer/et-data
python scripts/download_et_data.py
```

脚本会从 Hugging Face 拉取 `robin-courant/et-data` 并解包（若有 `untar_and_move.sh`）。完成后 E.T. 根目录即为 `--download-dir`（或 `ET_DATA_DOWNLOAD_DIR`）。

---

## 4. 预处理数据

在项目根目录执行，`--et-root` 指向上一步的下载目录：

```bash
export PYTHONPATH=.

python scripts/preprocess_et_data.py \
  --et-root /otherlocation/transfer/et-data \
  --output-root data \
  --num-frames 48
```

会生成：

- `data/train_index.json`
- `data/test_index.json`
- `data/trajectories/*.npy`

（若 E.T. 已在别处预处理过，只需把 `data/` 里上述文件拷到当前项目的 `data/` 下即可，可跳过本步。）

---

## 5. （可选）单人子集：添加而非替换

若要用「单人」子集训练，在预处理之后执行：

```bash
python scripts/filter_et_single_person.py --data-root data
```

会**新增** `data/train_index_single_person.json` 和 `data/test_index_single_person.json`，**不会覆盖**原来的 `train_index.json` / `test_index.json`。

- **原来 500 epoch 的相机轨迹训练**：继续用 `train_index.json`（全量或你之前的「不确定」子集），配置里保持 `train_index_file: "train_index.json"` 即可。
- **用单人子集训练**：在 `configs/default.yaml` 的 `data` 下改为：
  - `train_index_file: "train_index_single_person.json"`
  - `test_index_file: "test_index_single_person.json"`
  训练和评估会按配置读对应索引，无需覆盖任何文件，原有流程完全保留。

---

## 6. 启动训练

在项目根目录：

```bash
export PYTHONPATH=.

# GPU
python train.py --config configs/default.yaml --device cuda

# 无 GPU 时用 CPU（较慢，仅调试用）
python train.py --config configs/default.yaml --device cpu --no-clip
```

- `--no-clip`：不加载 CLIP，用随机文本向量，适合快速试跑或没有外网/显存紧张时。
- 断点续训：

```bash
python train.py --config configs/default.yaml --device cuda --resume checkpoints/checkpoint_epoch50.pth
```

---

## 7. 输出与检查

- 权重会保存在 `checkpoints/`（由 `configs/default.yaml` 里 `paths.checkpoint_dir` 决定）。
- 默认每 50 个 epoch 存一次，结束会写 `checkpoint_final.pth`。
- 用 TensorBoard（若装了）：

```bash
tensorboard --logdir logs
```

---

## 8. 常见问题

| 情况 | 处理 |
|------|------|
| `ModuleNotFoundError: src...` | 在项目根目录执行，并先 `export PYTHONPATH=.` 再跑 `train.py`。 |
| `FileNotFoundError: train_index.json` | 先完成步骤 4 预处理，或把已有 `data/train_index.json` 放到 `data/`。 |
| CUDA out of memory | 在 `configs/default.yaml` 里把 `training.batch_size` 调小（如 32 或 16）。 |
| 没有 GPU | 使用 `--device cpu --no-clip`，或减小 batch_size。 |

---

## 9. 一步到位（下载到固定目录时）

若使用 `otherlocation/transfer` 等固定目录，可顺序执行：

```bash
cd Text-to-Shot-main
export PYTHONPATH=.
pip install -e .   # 或 uv sync
python scripts/download_et_data.py --download-dir /otherlocation/transfer/et-data
python scripts/preprocess_et_data.py --et-root /otherlocation/transfer/et-data --output-root data
python train.py --config configs/default.yaml --device cuda
```

之后用 `checkpoints/checkpoint_final.pth` 做推理或评估即可。
