# deep_learning — 装甲板 YOLO26-OBB 检测

RoboMaster 装甲板检测的**深度学习方案**：YOLO26 旋转框（OBB）检测。
与 `traditional_cv/` 平级、互不耦合，复用根目录 `dataset2/` 数据集（不随仓库分发）。

## 目录结构

```
deep_learning/
├── armor_det/            # 核心库
│   ├── config.py         #   路径、类别、增广默认参数
│   ├── augment.py        #   离线增广：噪点 / 人为遮挡 / 亮度扰动
│   └── pose.py           #   PnP 位姿解算（距离 / 朝向，真实尺寸常量）
├── apps/                 # 可执行入口（均在仓库根目录执行）
│   ├── build_dataset.py  #   标签转换 + 近重复隔离划分 + 离线增广 → data/
│   ├── train.py          #   YOLO26-OBB 训练（自动选 CUDA/MPS/CPU）
│   ├── infer.py          #   推理与可视化
│   └── viewer.py         #   交互式查看器（弹窗浏览标注 + 训练曲线）
├── colab_train.ipynb     # Google Colab GPU 一键 Notebook
├── COLAB.md              # Colab 训练教程
├── utils/files.py        # 自然排序、图片收集
├── data/                 # 生成数据集（不入库，仅 .gitkeep 占位）
├── runs/                 # 训练产物（不入库）
└── requirements.txt
```

## 数据格式与类别

`dataset1`（1197 张）与 `dataset2`（593 张）标签格式相同，均为 10 列：

```
color_id  vehicle_id  x1 y1 x2 y2 x3 y3 x4 y4      # 归一化四点旋转框
```

> 注意：`dataset1/README.md` 把前两列写成 `class_id color_id`，但实际文件里
> 第 1 列只取 0–2（颜色）、第 2 列取 0–7（车型），与 dataset2 一致。

- **颜色**：`0=blue` 蓝方、`1=red` 红方、`2=gray` 已熄灭（灯灭/血量条消失）。
- **车型**：`0=sentry, 1=hero, 2=engineer, 3=infantry3, 4=infantry4, 5=outpost, 6=base_small, 7=base_big`。
- **24 类** = 颜色 × 车型，组合 id `= color_id * 8 + vehicle_id`（如 `1*8+3=11` = `red_infantry3`）。

`build_dataset.py` 会合并两个数据集，并转成 Ultralytics 需要的 9 列
`class x1 y1 ... x4 y4`；文件名带来源前缀（`dataset1_000001`），避免同名帧互相覆盖。

## 安装

仓库目前用 Python 3.12 虚拟环境（3.14 暂缺 torch wheel）：

```bash
uv venv --python 3.12 deep_learning/.venv
uv pip install --python deep_learning/.venv/bin/python -r deep_learning/requirements.txt
```

> YOLO26 于 2026-01 随 ultralytics 发布，需较新版本（已验证 `ultralytics==8.4.149`）。

## 使用

以下命令均在**仓库根目录**执行。

### 1. 构建数据集（转换 + 划分 + 离线增广）

```bash
deep_learning/.venv/bin/python deep_learning/apps/build_dataset.py
```

默认**合并 dataset1 + dataset2**，每张训练原图生成 **3 个增广变体**，整体约 **4 倍**。
验证集只放**未增广原图**，且与训练集做**近重复隔离**（见下），避免指标被高估。

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--src dataset1 dataset2` | 原始数据集目录，可多个（默认自动带上存在的 dataset1/dataset2） |
| `--variants 3` | 每张原图生成的增广变体数（3 → 约 4 倍） |
| `--val-ratio 0.15` | 验证集比例 |
| `--split dup` | 划分方式：`dup`=近重复帧分组（默认，防泄漏）；`random`=随机；`block`=序号尾部 |
| `--dup-threshold 8` | 近重复判定阈值（64×48 灰度平均绝对差） |
| `--format jpg` | 输出格式，`jpg` 体积更小（默认） |
| `--limit N` | 每个数据集仅取前 N 张原图（冒烟用） |
| `--no-augment` | 只转换与划分，不做增广（对照基线） |
| `--clean` | 构建前清空输出目录 |
| `--seed 42` | 随机种子，保证可复现 |

#### 为什么要做近重复隔离

两个数据集都是从视频密集抽帧得到的，**大量帧内容近乎相同、但序号并不相邻**
（dataset2 实测：593 帧里超过 90% 的帧都能在别处找到"双胞胎"，最近邻灰度差中位数仅 4.5/255）。
若直接随机划分，这些近重复帧会同时进入训练与验证，指标被严重高估（最初随机划分时，
1 个 epoch 的 mAP50-95 就虚高到 0.89）。

`--split dup` 会先按图像签名把近重复帧聚成一组，划分时**整组一起走**，
保证同一组不会跨 train/val。默认阈值 8 下，验证集已无"训练集近邻 < 8"的帧。

> 注意：这是密集抽帧数据，帧与帧本身高度连续，即使做了分组隔离，
> 验证指标仍偏乐观；如需更严格的泛化估计，可用 `--split block` 或留出独立视频。

### 2. 训练

```bash
deep_learning/.venv/bin/python deep_learning/apps/train.py
```

默认 `yolo26n-obb.pt` / `imgsz=1024` / `batch=8` / `device=auto` / `epochs=150` / `patience=30`。
`--device auto` 会按平台自动选择：**Colab/NVIDIA → CUDA(`0`)**、Apple → `mps`、否则 `cpu`；
也可以显式写 `--device 0`（CUDA）、`--device 0,1`（多卡）、`--device mps`、`--device cpu`。
产物在 `--project`（默认 `deep_learning/runs/`）下的 `<name>/`（含 `weights/best.pt`、`results.png`）。

- 精度优先可换 `--model yolo26s-obb.pt`。
- 显存不足（OOM）就降低 `--batch`；GPU 上可开到 `--batch 16`/`32`。
- 本地（Apple MPS）太慢时，按 [COLAB.md](COLAB.md) 到 Colab GPU 上跑。

### 3. 推理

```bash
# 默认自动取 runs/ 下最新的 best.pt
deep_learning/.venv/bin/python deep_learning/apps/infer.py --source dataset2/images --limit 50
deep_learning/.venv/bin/python deep_learning/apps/infer.py --source autoaim_all.mp4 --name video
```

结果写入 `preview/obb_infer/`。

### 4. 交互式查看器（弹窗看标注、位姿与训练结果）

```bash
# 图片：浏览验证集 + 最新权重，并弹出训练曲线窗口
deep_learning/.venv/bin/python deep_learning/apps/viewer.py

# 视频：逐帧实时标注 + PnP 位姿（--source 直接给 mp4 即可）
deep_learning/.venv/bin/python deep_learning/apps/viewer.py --source autoaim_all.mp4 --play

# 指定权重/数据集，或只看原图与真值
deep_learning/.venv/bin/python deep_learning/apps/viewer.py --split train --limit 200
deep_learning/.venv/bin/python deep_learning/apps/viewer.py --source dataset2/images --no-detect
```

每个检出目标会用 **PnP 解算位姿**：按 `--hfov`（默认 60°）估算内参，或 `--calib` 加载标定文件；
在框上叠加距离与坐标轴，左上角面板列出 `dist / yaw / pitch / x-y-z / 重投影误差`，
并同时给出「按已知板高反推」的独立距离估计用于交叉验证。

按键：`空格` 图片=开关检测 / 视频=播放暂停、`n/p` 上/下一张(帧)、`,/.` 前后跳、`r` 回到开头、
`x` 开关检测、`z` 开关 PnP 位姿、`g` 真值(GT)、`c` 置信度、`v` 训练曲线窗口、`f` 适应窗口、
`s` 保存、`h` 帮助、`q` 退出。曲线优先用 `results.png`，缺失时由 `results.csv` 现画。
预测/真值框按颜色分组着色（蓝/红/灰），标签显示 24 类名（如 `red_infantry3`）。

> PnP 相关参数：`--hfov`（水平视场角，默认 60°，未标定时据此估算内参）/
> `--calib`（标定文件，优先于 `--hfov`）/ `--focal`（直接给焦距像素，优先于 `--hfov`，
> 距离与之成正比）/ `--size-from {class,aspect}` / `--min-aspect`（宽高比可信度门限，
> 默认 0.7，低于该值的框标为不可信）/ `--no-pose`。
> 装甲板尺寸取自类别（`hero`、`base_big` 按大板 0.231m，其余按小板 0.136m，板高 0.05603m），
> 也可用 `--size-from aspect` 改由观测宽高比推断。

## 在 Colab（NVIDIA GPU）上训练

见 [COLAB.md](COLAB.md)，或直接用 [colab_train.ipynb](colab_train.ipynb)：
克隆仓库 → 装 ultralytics → 从 Drive 解压 `dataset2` → 构建数据集 → 训练。
`train.py` 的 `--device auto` 在 Colab 上会自动走 CUDA，命令与本地一致。

## 增广设计

噪点、亮度、遮挡都是**几何不变**的，因此原始 OBB 四点标签可原样沿用，无需重标注：

- **亮度**：随机整体变暗（×0.40–0.80）或变亮（×1.20–1.80），并叠加对比度/伽马抖动。
- **噪点**：高斯噪点（σ=5–25）或椒盐噪点（0.1%–1%）。
- **人为遮挡（落在甲板上）**：不是随便乱遮——先随机挑一块装甲板，在其标注多边形**内部**采样落点，
  再按该板短边的 30%–80% 生成遮挡块，因此遮挡**一定压在甲板像素上**（模拟挡住灯条/装甲号）。
  同时限制对目标的遮挡面积在 **5%–60%** 之间：遮得太少或太多都会重采样，保证目标仍可辨识。
