# 在 Google Colab 上训练（NVIDIA GPU）

本地是 Apple MPS，单轮 ~226s、跑满要 7–8 小时；Colab 的 T4/A100 会快很多。
本目录的 `colab_train.ipynb` 已把整套流程做成可以逐格运行的 Notebook。

## 0. 一次性准备

1. **把仓库推到 GitHub**（这样 Colab 直接 `git clone` 就能拿到代码）。
2. **把数据集打包上传到 Google Drive**：本地在**仓库根目录**执行
   ```bash
   zip -qr datasets.zip dataset1 dataset2     # 两个数据集一起打包（约 1.6GB）
   ```
   然后把 `datasets.zip` 传到 Drive 的 `MyDrive` 根目录。

> 只想要其中一个数据集也行：把 zip 里解压出的目录名保持为 `dataset1` / `dataset2` 即可，
> `build_dataset.py` 默认会把存在的那些都合进来，也可以用 `--src dataset2` 显式指定。

> 也可以用 Colab 的“上传”按钮直接传 zip，只是大文件用 Drive 更稳。

## 1. 打开 Notebook 并选 GPU

- 打开 <https://colab.research.google.com/>
- `文件 → 打开笔记本 → GitHub`，粘贴仓库地址 `https://github.com/diyanqi/rm_armor.git`，
  选择 `deep_learning/colab_train.ipynb`；或者直接 `文件 → 上传笔记本` 传这个 ipynb。
- 菜单 `代码执行程序 → 更改运行时类型 → 硬件加速器 = T4 GPU → 保存`。

## 2. 逐格运行（Notebook 里就是这些命令）

```bash
# 1) 检查 GPU
nvidia-smi

# 2) 克隆仓库
git clone https://github.com/diyanqi/rm_armor.git rm_armor
cd rm_armor

# 3) 装依赖（Colab 已自带 torch/cv2/numpy，只需补 ultralytics）
pip -q install "ultralytics>=8.4"

# 4) 准备数据（从 Drive 解压，包含 dataset1 + dataset2）
cp /content/drive/MyDrive/datasets.zip . && unzip -q -o datasets.zip

# 5) 构建增广数据集（合并两个数据集，约 4 倍）
python deep_learning/apps/build_dataset.py --format jpg --clean

# 6) 训练
python deep_learning/apps/train.py \
  --model yolo26n-obb.pt \
  --epochs 150 --imgsz 1024 --batch 16 --device 0 --workers 2 \
  --project /content/drive/MyDrive/rm_runs --name colab_yolo26n
```

`train.py` 的 `--device` 默认就是 `auto`：在 Colab 上会自动选中 CUDA（等价 `--device 0`），
在本机自动选 MPS，所以同一条命令两处都能跑。

## 3. 参数建议（T4 16G）

| 参数 | 建议 | 说明 |
| --- | --- | --- |
| `--model` | `yolo26n-obb.pt`（先跑通）→ `yolo26s-obb.pt`（追精度） | s/m 更吃显存 |
| `--imgsz` | `1024` | 目标偏小，别降到 640 |
| `--batch` | `16` | OOM 就 8 / 4；A100 可 32 |
| `--epochs` | `150` | 配 `--patience 30` 早停 |
| `--workers` | `2` | Colab 上别开太大 |
| `--project` | 指向 Drive | 防断线丢产物 |

## 4. 断线与续训

Colab 免费档会因空闲/超时断开。两个保险：

- **产物写进 Drive**：`--project /content/drive/MyDrive/rm_runs`（上面已这么写）。
- **续训**：重新运行到第 6 格时加 `--resume`：
  ```bash
  python deep_learning/apps/train.py \
    --project /content/drive/MyDrive/rm_runs --name colab_yolo26n --resume
  ```

## 5. 查看结果 / 把权重带回本机

- 训练曲线：`/content/drive/MyDrive/rm_runs/*/results.png`（Notebook 里有展示单元格）。
- 下载：`zip -qr results.zip /content/drive/MyDrive/rm_runs preview` 再 `files.download(...)`。
- 回到本机后，用交互式查看器看标注效果：
  ```bash
  deep_learning/.venv/bin/python deep_learning/apps/viewer.py \
      --weights /path/to/best.pt --source dataset2/images --limit 200 --device mps
  ```

## 6. 常见问题

- **`CUDA out of memory`**：调小 `--batch`（16 → 8 → 4）或 `--imgsz`。
- **`nvidia-smi` 报错 / `torch.cuda.is_available()` 为 False**：运行时类型没选 GPU，重选后重启运行时。
- **Drive 挂载后看不到刚传的文件**：Drive 有同步延迟，等几十秒或重新挂载。
- **`ultralytics` 找不到 `yolo26*-obb`**：版本太旧，`pip -q install -U ultralytics`。
- **数据在 Drive 上直接训练很慢**：先把 zip 复制到 `/content`（本地盘）再解压，训练读图更快。
