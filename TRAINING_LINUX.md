# 在 Linux 上跑训练 — 详细步骤

下面按顺序做，**每一步都依赖前一步**。所有路径默认都在 **`/transfer`** 下。

---

## 目录结构总览（默认都在 /transfer 下）

| 路径 | 是什么 | 谁创建的 |
|------|--------|----------|
| `/transfer/et-data/` | E.T. 原始数据（traj/, caption/, caption_cam/ 等） | 步骤 3 下载脚本 |
| `/transfer/data/` | 预处理后的训练数据（train_index.json + test_index.json + trajectories/） | 步骤 4 预处理脚本 |
| `/transfer/checkpoints/` | 训练得到的权重（.pth） | 步骤 6 训练 |
| `/transfer/outputs/` | 推理/可视化生成的图、GIF | generate_storyboard、visualize_3d |
| `/transfer/logs/` | TensorBoard 等日志 | 训练时可选 |

**预处理之后**的数据在 **`/transfer/data/`**：`train_index.json`、`test_index.json`、`trajectories/*.npy`。训练时读 **`/transfer/data/`**，权重写到 **`/transfer/checkpoints/`**。

---

## 0. 先建好 /transfer（没有的话）

**为什么**：所有数据、权重、输出都放在 `/transfer`，不放进项目目录，这样 clone 代码不会动到数据和权重。

```bash
sudo mkdir -p /transfer
# 若没有 sudo，改用你有写权限的目录，并改 configs/default.yaml 里所有 /transfer 为你的路径
```

---

## 1. 确认环境

**为什么**：保证 Python 版本和 GPU 可用。

```bash
cd Text-to-Shot-main
python3 --version   # 需要 >= 3.10
export PYTHONPATH=.
```

---

## 2. 安装依赖

**为什么**：训练和预处理依赖 PyTorch、transformers 等，必须装好。

```bash
pip install -e .
# 或: pip install torch torchvision transformers sentence-transformers ...
```

---

## 3. 下载 E.T. 数据集（必须先做，再预处理）

**为什么**：预处理脚本要读「原始 E.T.」里的 `traj/*.txt`、`caption/*.txt`、`caption_cam/*.txt`。这些文件只有下载后才有。你之前的报错 `No such file or directory: '/transfer/et-data/traj'` 就是因为 **还没执行这一步**，`/transfer/et-data/` 里没有内容或结构不对。

**会得到什么**：  
在 **`/transfer/et-data/`** 下出现 E.T. 仓库内容，**必须**包含子目录：
- `traj/`（每帧相机外参的 .txt）
- `caption/`
- `caption_cam/`  
以及（若有）`full_train_split.txt`、`full_test_split.txt`。  
若 Hugging Face 下载后是压缩包，需在 **`/transfer/et-data/`** 里执行仓库里的 `untar_and_move.sh`（或按官方说明解压），直到出现上述 `traj/` 等目录。

```bash
python scripts/download_et_data.py
```

若下载到了别处，预处理时用 `--et-root` 指过去，例如：
```bash
python scripts/preprocess_et_data.py --et-root /你的路径/et-data --output-root /transfer/data --num-frames 48
```

---

## 4. 预处理：把 E.T. 转成训练用的格式

**为什么**：原始 E.T. 是 3×4 外参的 .txt，训练需要的是 48 帧×6 维的 .npy + 索引。这一步会：
- 从 `/transfer/et-data/traj/` 读轨迹，转成 6D，重采样到 48 帧，写成 .npy
- 从 caption 读文本，并推断 shot_type、camera_motion
- 写出索引文件，方便 DataLoader 按条读

**输出在哪儿**：**`/transfer/data/`**，文件夹和文件名为：
- `train_index.json`、`test_index.json`
- `trajectories/*.npy`（每个样本一个 .npy）

**命令**（默认就是从 `/transfer/et-data` 读、写到 `/transfer/data`）：

```bash
python scripts/preprocess_et_data.py --num-frames 48
```

等价于：
```bash
python scripts/preprocess_et_data.py --et-root /transfer/et-data --output-root /transfer/data --num-frames 48
```

若你 E.T. 在别的路径，用 `--et-root` 和 `--output-root` 指定；**只要 `--output-root` 写成 `/transfer/data`，预处理之后的数据就在 /transfer/data**。

---

## 5. （可选）单人子集：只训练「单人」样本

**为什么**：若你想用「单人」子集训练，需要从索引里筛出 caption 像单人的样本，得到新的索引；不改动原来的 `train_index.json`，所以是「添加」而不是替换。

**会得到什么**：在 **`/transfer/data/`** 下**新增**：
- `train_index_single_person.json`
- `test_index_single_person.json`  
轨迹文件还是用原来的 `trajectories/*.npy`，不重复生成。

```bash
python scripts/filter_et_single_person.py --data-root /transfer/data
```

要用单人训练时，改 `configs/default.yaml` 里：
- `train_index_file: "train_index_single_person.json"`
- `test_index_file: "test_index_single_person.json"`  
训练就会读 `/transfer/data/` 下这两个索引。

---

## 6. 启动训练

**为什么**：用 `/transfer/data/` 里的索引和 .npy 训练扩散模型，得到可用的权重。

**读哪里**：`configs/default.yaml` 里 `data_root: "/transfer/data"`，所以会读 **`/transfer/data/train_index.json`** 和 **`/transfer/data/trajectories/*.npy`**。

**写哪里**：`paths.checkpoint_dir: "/transfer/checkpoints"`，所以权重在 **`/transfer/checkpoints/`**，例如：
- `checkpoint_epoch50.pth`、`checkpoint_epoch100.pth`（按 save_interval 存）
- `checkpoint_final.pth`（训练结束）

```bash
# GPU
python train.py --config configs/default.yaml --device cuda

# 无 GPU 或快速试跑
python train.py --config configs/default.yaml --device cpu --no-clip
```

断点续训（权重必须在 /transfer/checkpoints 下）：
```bash
python train.py --config configs/default.yaml --device cuda --resume /transfer/checkpoints/checkpoint_epoch50.pth
```

---

## 7. 输出与检查

- **权重**：在 **`/transfer/checkpoints/`**（如 `checkpoint_final.pth`）。
- **日志**：若写 TensorBoard，在 **`/transfer/logs/`**，可：
  ```bash
  tensorboard --logdir /transfer/logs
  ```

---

## 8. 常见问题

| 情况 | 原因 / 处理 |
|------|----------------|
| `FileNotFoundError: .../transfer/et-data/traj` | 还没做步骤 3 下载，或下载后没有 `traj/`。先跑 `download_et_data.py`，并在 `/transfer/et-data/` 下确认有 `traj/`、`caption/`、`caption_cam/`；若 HF 是压缩包，按说明解压或运行 `untar_and_move.sh`。 |
| `full_train_split.txt not found` | 官方 E.T. 可能没提供 split 文件；脚本会继续用 `traj/` 下所有 .txt 当样本，train/test 可能全当 train。若你有自己的 split，可放到 `/transfer/et-data/` 下同名文件。 |
| `FileNotFoundError: train_index.json` | 还没做步骤 4 预处理，或 `data_root` 不是 `/transfer/data`。先跑预处理并确认 `/transfer/data/train_index.json` 存在。 |
| CUDA out of memory | 在 `configs/default.yaml` 里把 `training.batch_size` 改小（如 32、16）。 |
| 没有 GPU | 使用 `--device cpu --no-clip`。 |

---

## 9. 顺序小结（从 0 开始：下载 → 预处理 → 训练）

```bash
cd Text-to-Shot-main
export PYTHONPATH=.
pip install -e .

python scripts/download_et_data.py
# 下载到 /transfer/et-data；若 HF 是压缩包，在 /transfer/et-data 里解压或运行 untar_and_move.sh，直到出现 traj/、caption/、caption_cam/

python scripts/preprocess_et_data.py --num-frames 48
# 从 /transfer/et-data 读，写到 /transfer/data（train_index.json、test_index.json、trajectories/）

python train.py --config configs/default.yaml --device cuda
# 从 /transfer/data 读，权重写到 /transfer/checkpoints/
```

推理时用 `/transfer/checkpoints/checkpoint_final.pth`，生成结果默认在 `/transfer/outputs/`。
