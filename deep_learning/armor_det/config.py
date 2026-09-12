"""全局配置：路径、类别与增广默认参数。

类别体系（24 类 = 颜色 × 车型）：
  标签为 10 列 `color_id vehicle_id x1 y1 x2 y2 x3 y3 x4 y4`
  （dataset1/dataset2 两套数据都是这个顺序，dataset1 的 README 把两列名字写反了）。
  组合类别 id = color_id * 8 + vehicle_id。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

# deep_learning/armor_det/config.py -> 上三级为仓库根目录
DEEP_LEARNING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(DEEP_LEARNING_DIR)

# 颜色：0=blue 蓝方，1=red 红方，2=gray 已熄灭（灯灭/血量条消失）
COLOR_NAMES: Dict[int, str] = {0: "blue", 1: "red", 2: "gray"}

# 车型：0-7
VEHICLE_NAMES: Dict[int, str] = {
    0: "sentry",
    1: "hero",
    2: "engineer",
    3: "infantry3",
    4: "infantry4",
    5: "outpost",
    6: "base_small",
    7: "base_big",
}

NC_COLOR = len(COLOR_NAMES)
NC_VEHICLE = len(VEHICLE_NAMES)
# 24 类：组合 id = color * 8 + vehicle
CLASS_NAMES: Dict[int, str] = {
    color * NC_VEHICLE + vehicle: f"{COLOR_NAMES[color]}_{VEHICLE_NAMES[vehicle]}"
    for color in sorted(COLOR_NAMES)
    for vehicle in sorted(VEHICLE_NAMES)
}

# 可视化配色（BGR），按颜色索引
COLOR_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (255, 150, 40),   # blue
    1: (60, 60, 255),    # red
    2: (190, 190, 190),  # gray
}


def class_color_bgr(cls: int) -> Tuple[int, int, int]:
    """类别 id -> BGR 颜色（按颜色分组着色）。"""
    return COLOR_BGR.get(cls // NC_VEHICLE, (0, 255, 0))


# 原始数据集（可多个，合并训练）；不随仓库分发
DEFAULT_SOURCES: List[str] = [
    path for path in (os.path.join(REPO_ROOT, "dataset1"), os.path.join(REPO_ROOT, "dataset2"))
    if os.path.isdir(path)
] or [os.path.join(REPO_ROOT, "dataset2")]

DEFAULT_OUT = os.path.join(DEEP_LEARNING_DIR, "data", "armor_obb")
DEFAULT_DATA_YAML = os.path.join(DEEP_LEARNING_DIR, "data", "armor_obb.yaml")
DEFAULT_RUNS = os.path.join(DEEP_LEARNING_DIR, "runs")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class AugmentConfig:
    """离线增广参数。所有增广均保持几何不变，因此原始 OBB 标签可直接沿用。"""

    # 各手段被触发的概率
    brightness_prob: float = 0.70
    noise_prob: float = 0.60
    occlusion_prob: float = 0.60

    # 亮度：整体缩放系数区间（"随机增大暗度 / 亮度"）
    darken_range: Tuple[float, float] = (0.40, 0.80)
    brighten_range: Tuple[float, float] = (1.20, 1.80)
    # 对比度与伽马抖动，模拟光照变化
    contrast_range: Tuple[float, float] = (0.85, 1.15)
    gamma_range: Tuple[float, float] = (0.85, 1.15)

    # 噪点：高斯 sigma / 椒盐像素占比
    gaussian_sigma: Tuple[float, float] = (5.0, 25.0)
    salt_pepper_ratio: Tuple[float, float] = (0.001, 0.010)

    # 人为遮挡：只在装甲板（标注框）内部落块，模拟局部遮挡
    occlusion_min: int = 1           # 每图最少遮挡块数
    occlusion_max: int = 3           # 最多遮挡块数
    occlusion_patch_scale: Tuple[float, float] = (0.30, 0.80)  # 边长占目标框短边比例
    occlusion_min_cover: float = 0.05   # 至少要遮住目标这么多面积，才算"遮到甲板"
    max_target_cover: float = 0.60      # 最多遮住目标这么多，保证仍可辨识
    occlusion_tries: int = 30           # 每块遮挡的最大尝试次数

    def as_dict(self) -> dict:
        return dict(self.__dict__)


DEFAULT_AUG = AugmentConfig()
