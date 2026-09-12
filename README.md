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

## 后续计划

深度学习方案将放在与 `traditional_cv/` 平级的新目录（如 `deep_learning/`）中，
复用根目录下的数据集与测试视频，两套方案代码互不耦合。
