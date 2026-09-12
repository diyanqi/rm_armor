# rm_armor — RoboMaster 装甲板检测

按技术路线分目录组织的 RoboMaster 装甲板检测实验仓库。

## 目录结构

```
.
├── traditional_cv/     # 传统视觉方案：颜色分割 + 灯带配对 + PnP 位姿
│   ├── rm_armor/       #   核心检测库
│   ├── apps/           #   可执行入口（检测 / 查看器 / 评估 / 预览 / 自检）
│   ├── utils/          #   通用工具
│   ├── requirements.txt
│   └── README.md
├── images/             # 数据集占位目录（内容不入库）
├── preview/            # 预览输出占位目录（内容不入库）
└── autoaim_all.mp4     # 测试视频（不入库，需自行放置）
```

## 数据与媒体

为控制仓库体积，数据集、测试视频与生成物均不纳入版本控制，仓库内只保留
`images/`、`preview/` 两个空目录作为占位：

- `images/`：放入从视频抽帧得到的图片（如 `frame_000001.jpg`）。
- `autoaim_all.mp4`：测试视频，放在仓库根目录。
- `preview/`、`out/`、`viewer_out/`：脚本生成的输出，会被自动忽略。

## 快速开始

```bash
pip install -r traditional_cv/requirements.txt

# 合成数据自检（无需数据集）
python3 traditional_cv/apps/selfcheck.py

# 单图 / 图片目录 / 视频检测
python3 traditional_cv/apps/main.py images/frame_000117.jpg --out out
python3 traditional_cv/apps/main.py images --limit 200 --out out
python3 traditional_cv/apps/main.py autoaim_all.mp4 --out out/annotated.mp4 --stride 5
```

各脚本均带命令行帮助（`-h`），完整用法见 [traditional_cv/README.md](traditional_cv/README.md)。

## 深度学习方案

`deep_learning/` 为与 `traditional_cv/` 平级的深度学习方案（YOLO26-OBB 旋转框检测），
自带 `utils/`，不引用传统视觉代码，两套方案互不耦合。合并 `dataset1` + `dataset2`
（共 1790 张，24 类 = 颜色 × 车型）训练。

```bash
pip install -r deep_learning/requirements.txt

# 1. 构建数据集：合并两个数据集 + 标签转换 + 近重复隔离划分 + 离线增广（噪点/遮挡/亮度）到约 4 倍
python3 deep_learning/apps/build_dataset.py

# 2. 训练 YOLO26-OBB（--device auto：NVIDIA 走 CUDA、Apple 走 MPS）
python3 deep_learning/apps/train.py

# 3. 推理与可视化
python3 deep_learning/apps/infer.py --source dataset2/images --limit 50

# 4. 交互式查看器：弹窗浏览标注结果 + 训练曲线
python3 deep_learning/apps/viewer.py
```

本地训练太慢时，可用 [deep_learning/colab_train.ipynb](deep_learning/colab_train.ipynb)
在 Google Colab 的 NVIDIA GPU 上训练，步骤见 [deep_learning/COLAB.md](deep_learning/COLAB.md)。

详见 [deep_learning/README.md](deep_learning/README.md)。
