"""全局配置：装甲板真实尺寸、相机模型、灯带与配对阈值。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence


class ArmorColor(Enum):
    """灯带颜色（红方 / 蓝方）。"""

    RED = "red"
    BLUE = "blue"


class ArmorSize(Enum):
    """装甲板规格。"""

    SMALL = "small"
    LARGE = "large"


# ---------------------------------------------------------------- 真实尺寸 --
# 单位为米，取自 RoboMaster 官方装甲板规格。
ARMOR_WIDTH_M = {
    ArmorSize.SMALL: 0.136,
    ArmorSize.LARGE: 0.231,
}
ARMOR_HEIGHT_M = 0.05603

# 宽高比 = 两条灯带外沿间距 / 灯带高度，是区分大小装甲板的核心特征。
ARMOR_ASPECT = {size: width / ARMOR_HEIGHT_M for size, width in ARMOR_WIDTH_M.items()}

# 大小装甲板宽高比的几何平均数，作为分类决策边界。
# 注意：该比值随偏航角（yaw）按 cos 衰减，大装甲板大角度倾斜时会退化成小装甲板的比值。
ARMOR_ASPECT_SPLIT = math.sqrt(ARMOR_ASPECT[ArmorSize.SMALL] * ARMOR_ASPECT[ArmorSize.LARGE])


class ArmorSizeEstimator:
    """把观测到的宽高比映射为装甲板规格（含置信度）。"""

    @staticmethod
    def from_aspect(aspect: float) -> tuple[ArmorSize, float]:
        """返回 (规格, 置信度)。置信度为 0~1，越接近决策边界越低。"""
        size = ArmorSize.LARGE if aspect >= ARMOR_ASPECT_SPLIT else ArmorSize.SMALL
        nominal = ARMOR_ASPECT[size]
        # 用对数距离衡量“离决策边界有多远”，距离为 0 时置信度 = 0。
        margin = abs(math.log(aspect / ARMOR_ASPECT_SPLIT))
        span = abs(math.log(nominal / ARMOR_ASPECT_SPLIT))
        confidence = 1.0 if span <= 1e-6 else min(1.0, margin / span)
        return size, confidence


# ------------------------------------------------------------------ 相机模型 --
@dataclass
class CameraConfig:
    """相机内参配置。未标定时按视场角估算，绝对距离会有偏差但角度仍然可用。"""

    hfov_deg: float = 60.0
    calibration_file: Optional[str] = None
    dist_coeffs: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    # 标定时的图像分辨率，用于自动缩放到实际输入分辨率。
    calib_width: Optional[int] = None
    calib_height: Optional[int] = None


# ------------------------------------------------------------------ 颜色阈值 --
@dataclass
class ColorParams:
    """灯带颜色分割参数。

    先用通道差 ``R - max(G, B)``（蓝灯为 ``B - max(R, G)``）取亮色，再用
    HSV 色相把范围收紧到“橙红”和“蓝”两类，其余颜色（绿、黄、紫、白）直接丢弃。
    偏深红的状态指示灯与彩色背景噪点主要靠灯带级的亮度/周边暗度门限排除。
    """

    # --- 通道差门限（按颜色区分）---
    # 橙红灯带的 ``R - max(G, B)`` 峰值可达 100+，取 30 就能得到完整的灯带。
    # 蓝色灯带经镜头扩散与白平衡后大量漏光到 G/R 通道，``B - max(R, G)`` 的峰值
    # 只有 60~90，光晕又宽又暗。若沿用 30，蓝色掩膜会把光晕一起吞进来，相邻光斑
    # 连成不规则大块，几何筛选几乎全军覆没。取 65 只保留最亮的核心，反而能得到
    # 干净细长的竖条——实测蓝甲板召回从 10/66 提升到 21/66，且无误报增加。
    diff_threshold_red: int = 30
    diff_threshold_blue: int = 65
    min_channel: int = 80
    morph_close: int = 3
    morph_open: int = 0

    # --- 色相门限（OpenCV 色相 0~179）---
    # 色相门限主要用来排除非红非蓝的颜色（绿/黄/紫/白）；深红状态指示灯主要靠
    # 亮度（peak V）与“灯带是否明显亮于四周”来排除，因为过曝的橙红灯带边缘
    # 色相会漂到 0 附近。
    use_hue_gate: bool = True
    red_hue_min: float = 0.0
    red_hue_max: float = 40.0
    blue_hue_min: float = 90.0
    blue_hue_max: float = 140.0
    min_saturation: int = 100     # 灯带是高饱和光源，灰白反光排除
    min_value: int = 60           # 太暗的像素不可能是点亮的灯带
    blown_out_value: int = 230    # 极亮像素（过曝核心）允许饱和度偏低
    blown_out_min_saturation: int = 40

    def diff_threshold_for(self, color: "ArmorColor") -> int:
        return self.diff_threshold_red if color is ArmorColor.RED else self.diff_threshold_blue


# ------------------------------------------------------------------ 灯带筛选 --
@dataclass
class LightBarParams:
    """单条灯带的几何与外观筛选条件。"""

    min_height_px: float = 8.0
    max_height_px: float = 1e9
    min_aspect: float = 1.4          # 长边 / 短边
    min_fill_ratio: float = 0.35     # 轮廓面积 / 最小外接矩形面积（灯带有光晕，不会太方正）
    max_tilt_deg: float = 40.0       # 灯带长轴与图像竖直方向的夹角上限

    # --- 必须是“点亮的灯带”，而不是状态指示灯/反光/背景噪点 ---
    # 两个判据配合使用：
    # 1) 相对判据（对比度/比值）：真灯带一定明显亮于紧邻的四周。蓝色背景上偶然变亮
    #    的噪点虽然自身够亮，但四周同样是蓝色背景，对比度很低，会被这里挡掉。
    # 2) 绝对亮度下限：摄像头上更暗的红色状态指示灯靠这一条排除；该下限必须按颜色
    #    区分，因为蓝色灯带的 V（=max(R,G,B)）天生比橙红低得多。
    min_ring_contrast: float = 50.0    # v90(灯带) - V中位(四周)
    min_ring_ratio: float = 1.35       # v90(灯带) / V中位(四周)
    min_peak_value_red: float = 180.0
    min_peak_value_blue: float = 110.0
    # 灯带四周应当紧邻暗色（装甲板黑面/黑边框）
    min_ring_dark_ratio: float = 0.40
    ring_dark_v_max: int = 110
    ring_pad_ratio: float = 0.9        # 环宽 = 灯带宽 * 该系数

    def min_peak_value_for(self, color: "ArmorColor") -> float:
        return self.min_peak_value_red if color is ArmorColor.RED else self.min_peak_value_blue


# ------------------------------------------------------------------ 配对条件 --
@dataclass
class PairWeights:
    """配对得分的各分量权重。未启用的证据会自动从分母中剔除。"""

    aspect: float = 0.30       # 宽高比与标称值的接近程度
    height: float = 0.18       # 两条灯带等高程度
    parallel: float = 0.18     # 平行 / 垂直关系
    number: float = 0.14       # 板面“黑底白字”证据（数字看不清时自动降级）
    brightness: float = 0.20   # 灯带亮度（区分真灯带与反光）


@dataclass
class PairParams:
    """两条灯带配成一块装甲板的约束与权重。

    判定主体是灯带的几何关系；板面内部区域只做“是不是黑底”的弱校验，
    白字只是可选加分项——远距离或欠曝时数字常常完全看不清。
    """

    # --- 几何约束 ---
    # 宽高比随偏航角按 cos 衰减，下限必须留足余量，否则斜视的小装甲板会被漏掉。
    min_aspect: float = 1.2
    max_aspect: float = 5.6
    max_height_ratio: float = 1.7    # 两条灯带高度之比上限
    # 两条灯带中心的竖直错位上限（占灯带平均高度）。不能取太紧：甲板带一点
    # 横滚时两灯带中心就会有明显高度差（roll 10° 时已达 0.42 个灯带高）。
    # 真正的倾斜约束由 max_perp_dev_deg 负责，这里只需挡住跨板的离谱错位。
    max_center_dy_ratio: float = 1.0
    max_tilt_diff_deg: float = 18.0  # 两条灯带倾斜角之差
    max_perp_dev_deg: float = 22.0   # 中心连线与灯带长轴的垂直偏差

    # --- 灯带自身质量 ---
    min_bar_brightness: float = 30.0  # 通道差低于该值的“灯带”更像反光，直接丢弃
    brightness_ref: float = 110.0     # 通道差达到该值即认为灯带足够亮

    # --- 板面校验：黑底 + 居中的白色数字 ---
    # 这是区分“真甲板”和“跨板误配”的关键：跨板配对的中间只有一条窄亮缝，
    # 而真甲板的白字在板宽里占明显比例（kappa 见下）。
    use_plate_validation: bool = True
    plate_margin_x: float = 0.06      # 裁掉左右边缘，避开灯带本体与光晕
    plate_margin_y: float = 0.04
    plate_min_dark_ratio: float = 0.40  # 板面必须大部分是暗的
    plate_dark_v_max: int = 110
    # 白字判据用“绝对”亮度与饱和度：白字又亮又几乎不饱和，灯带光晕与彩色背景都不满足
    digit_v_min: int = 150
    digit_sat_max: int = 80
    # kappa = 白字像素宽度 / 灯带像素高度 = 白字真实宽度 / 装甲板真实高度，
    # 与偏航角和装甲板尺寸都无关，因此可以稳定地区分白字与窄亮缝。
    digit_min_kappa: float = 0.45
    digit_max_kappa: float = 2.6           # 白字宽度 / 灯带高度的合理上限
    digit_min_height_ratio: float = 0.35   # 白字高度 / 灯带高度
    digit_max_offset: float = 0.18         # 白字中心相对板面中心的横向偏移（占板宽比例）
    digit_min_area_ratio: float = 0.02     # 白字面积 / 板面面积
    digit_max_area_ratio: float = 0.60
    digit_check_min_bar_height: float = 18.0  # 灯带太短则看不清数字，跳过数字校验
    require_digit_when_evaluated: bool = False  # True 时 SMALL 也必须有白字（召回会下降）
    large_requires_digit: bool = True      # LARGE 必须有白字证据，防止跨板误配

    # --- 拆解仲裁：优先把小甲板解释出来，而不是拼成一块“大甲板” ---
    # 相邻两块小甲板的内侧灯带会被误配成一块大甲板。若这两条灯带各自
    # 还能与旁边的灯带配成合理的“小甲板”，就说明当前 LARGE 是跨板组合，
    # 直接丢弃它，把灯带让给两块真正的小甲板。
    decompose_large: bool = True
    decompose_small_max_aspect: float = 3.0  # 子配对算作“可能是小甲板”的宽高比上限
    # 斜视时同一块甲板的两条灯带会有可观的竖直错位，子配对的容差要更宽
    decompose_small_max_center_dy_ratio: float = 0.9

    weights: PairWeights = field(default_factory=PairWeights)
    min_score: float = 0.35


# ------------------------------------------------------------------ 总配置 --
@dataclass
class DetectorConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    color: ColorParams = field(default_factory=ColorParams)
    lightbar: LightBarParams = field(default_factory=LightBarParams)
    pair: PairParams = field(default_factory=PairParams)
    colors: Sequence[ArmorColor] = (ArmorColor.RED, ArmorColor.BLUE)
    undistort: bool = False
