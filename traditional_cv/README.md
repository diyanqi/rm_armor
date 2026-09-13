# traditional_cv

RoboMaster 装甲板检测的**传统视觉方案**：颜色分割 → 灯带提取 → 灯带配对 → PnP 位姿解算，
不依赖任何训练权重。入口脚本统一从仓库根目录运行；数据集（放入 `images/`）与测试视频
（`autoaim_all.mp4`）不随仓库分发，需自行放在仓库根目录。

## 目录结构

```
traditional_cv/
├── rm_armor/           # 核心检测库
│   ├── config.py       # 全局配置、真实尺寸、各类阈值
│   ├── color.py        # 灯带颜色分割（橙红 / 蓝）
│   ├── lightbar.py     # 灯带提取与外观筛选
│   ├── armor.py        # 装甲板数据结构
│   ├── matcher.py      # 灯带配对与板面校验
│   ├── camera.py       # 相机模型 / 内参估算与标定
│   ├── pose.py         # PnP 位姿解算（距离 / 朝向）
│   ├── detector.py     # 检测流水线（以上模块的编排）
│   └── visualize.py    # 检测结果可视化
├── apps/               # 可执行入口
│   ├── main.py             # 命令行：单图 / 图片目录 / 视频
│   ├── viewer.py           # 交互式查看器（调参、逐帧浏览，支持视频）
│   ├── evaluate_dataset.py # 数据集召回 / 误报 / 耗时统计
│   ├── make_preview.py     # 生成拼图与标注视频
│   └── selfcheck.py        # 合成数据自检（无需图片）
├── utils/              # 通用工具（文件、路径、自然排序）
├── requirements.txt
└── README.md
```

## 安装

```bash
pip install -r traditional_cv/requirements.txt
```

## 运行

以下命令均在**仓库根目录**执行（`images/`、`autoaim_all.mp4` 在根目录）。

```bash
# 单张图片
python3 traditional_cv/apps/main.py images/frame_000117.jpg --out out

# 图片目录（--limit 限制数量）
python3 traditional_cv/apps/main.py images --limit 200 --out out

# 视频（--stride 为帧间隔）
python3 traditional_cv/apps/main.py autoaim_all.mp4 --out out/annotated.mp4 --stride 5

# 交互式查看器
python3 traditional_cv/apps/viewer.py --source images

# 生成预览拼图 / 标注视频
python3 traditional_cv/apps/make_preview.py stills --source images --out preview
python3 traditional_cv/apps/make_preview.py video --video autoaim_all.mp4 --start 2400 --end 3600

# 数据集评估
python3 traditional_cv/apps/evaluate_dataset.py images --json

# 合成数据自检（不需要数据集）
python3 traditional_cv/apps/selfcheck.py
```

各脚本自身带有完整的命令行帮助，例如 `python3 traditional_cv/apps/main.py -h`。

## 查看器按键

```text
n / d / 右方向键   下一张            p / a / 左方向键   上一张
space              切换“是否运行识别”（先看原图 / 看识别结果）
r                  重新识别当前帧     b   切换灯带边框
[ / ]              配对得分门限 -/+ 0.05（自动重识别）
, / .              灯带亮度门限 -/+ 5（自动重识别）
x                  切换位姿坐标轴     i   切换板面内部区域
m                  切换颜色掩膜视图   k   切换灯带颜色指标
f                  适应窗口 / 原始尺寸    + / -   缩放
s                  保存当前标注图到 --save-dir
h                  切换按键提示        q / ESC   退出
```
