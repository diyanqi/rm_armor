"""灯带（装甲板两侧竖条灯）检测。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .color import build_mask, color_score
from .config import ArmorColor, DetectorConfig


@dataclass
class LightBar:
    """一条灯带的几何与外观描述。

    ``corners`` 顺序固定为 ``[左上, 右上, 右下, 左下]``，且保证长边是竖直的
    （即 ``左上->左下`` 为长边），这样配对时无需再判断朝向。
    """

    color: ArmorColor
    corners: np.ndarray          # (4, 2) float32
    height_px: float             # 长边长度
    width_px: float              # 短边长度
    center: np.ndarray           # (2,) float32
    tilt_deg: float              # 长轴相对图像竖直方向的夹角，顺时针为正
    fill_ratio: float
    mean_diff: float             # 灯带区域的平均通道差，反映亮度质量
    peak_value: float            # 灯带区域 V 通道的 90 分位（状态灯/反光明显更低）
    ring_value: float            # 灯带外围一圈 V 通道中位数（用于相对亮度判据）
    hue_med: float               # 灯带区域色相中位数（橙红 vs 蓝）
    ring_dark_ratio: float       # 灯带外围一圈中暗像素占比（真灯带紧邻黑甲板/黑边框）

    @property
    def aspect(self) -> float:
        return self.height_px / max(self.width_px, 1e-6)

    @property
    def ring_contrast(self) -> float:
        """灯带比四周亮多少（V 差值）。背景噪点周围同样是背景，该值很低。"""
        return self.peak_value - self.ring_value

    @property
    def ring_ratio(self) -> float:
        """灯带与四周的亮度比，与颜色和曝光无关。"""
        return self.peak_value / max(self.ring_value, 1.0)

    @property
    def top_center(self) -> np.ndarray:
        return (self.corners[0] + self.corners[1]) / 2.0

    @property
    def bottom_center(self) -> np.ndarray:
        return (self.corners[2] + self.corners[3]) / 2.0

    @property
    def axis(self) -> np.ndarray:
        """单位向量，指向灯带“下端”。"""
        v = self.bottom_center - self.top_center
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else np.array([0.0, 1.0], dtype=np.float32)


def _weighted_axis(points: np.ndarray, weights: np.ndarray) -> Optional[
        Tuple[np.ndarray, np.ndarray]]:
    """强度加权 PCA，返回 ``(中心, 指向图像下方的单位长轴向量)``。

    权重取通道差强度，使得灯带最亮的芯部主导方向估计。外接矩形会把光晕
    一起框进去，光晕左右不对称时矩形就跟着歪；加权后这种偏差被大幅压低。
    """
    total = float(weights.sum())
    if total <= 1e-6:
        return None
    center = np.einsum("n,nk->k", weights, points) / total
    delta = points - center
    cov = (delta * weights[:, None]).T @ delta / total
    values, vectors = np.linalg.eigh(cov)
    axis = vectors[:, int(np.argmax(values))].astype(np.float32)
    if axis[1] < 0:
        axis = -axis
    return center.astype(np.float32), axis


def _bar_geometry(contour: np.ndarray, diff: np.ndarray) -> Optional[
        Tuple[np.ndarray, float, float, float]]:
    """由轮廓求灯带长轴方向与贴合的四角点。

    返回 ``(corners, tilt_deg, height_px, width_px)``，``corners`` 为
    ``[左上, 右上, 右下, 左下]``，``tilt_deg`` 为长轴相对图像竖直方向的夹角。
    """
    x, y, w, h = cv2.boundingRect(contour)
    x0, y0 = max(0, x - 1), max(0, y - 1)
    x1 = min(diff.shape[1], x + w + 1)
    y1 = min(diff.shape[0], y + h + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None

    local = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(local, [np.round(contour - [x0, y0]).astype(np.int32)], 255)
    rows, cols = np.nonzero(local)
    if rows.size < 6:
        return None

    points = np.stack([cols + x0, rows + y0], axis=1).astype(np.float32)
    weights = diff[y0:y1, x0:x1][rows, cols].astype(np.float32)
    weights = np.clip(weights - float(weights.min()), 0.0, None)
    if float(weights.sum()) <= 1e-6:
        weights = np.ones(rows.size, dtype=np.float32)

    axis = _weighted_axis(points, weights)
    if axis is None:
        return None
    center, axis = axis
    normal = np.array([axis[1], -axis[0]], dtype=np.float32)  # 指向图像右侧

    # 用轮廓点（而非掩膜像素）确定沿轴范围，边缘位置更贴合灯带真实边界。
    edge_points = contour.reshape(-1, 2).astype(np.float32)
    rel = edge_points - center
    t = rel @ axis
    s = rel @ normal
    t0, t1 = float(t.min()), float(t.max())
    s0, s1 = float(s.min()), float(s.max())
    if t1 - t0 < 1e-6 or s1 - s0 < 1e-6:
        return None

    corners = np.array([
        center + t0 * axis + s0 * normal,
        center + t0 * axis + s1 * normal,
        center + t1 * axis + s1 * normal,
        center + t1 * axis + s0 * normal,
    ], dtype=np.float32)

    tilt_deg = math.degrees(math.atan2(float(axis[0]), float(axis[1])))
    return corners, tilt_deg, t1 - t0, s1 - s0


def _appearance_metrics(corners: np.ndarray, gray: np.ndarray, value: np.ndarray,
                        hue: np.ndarray, diff: np.ndarray,
                        cfg: DetectorConfig) -> Optional[Tuple[float, float, float, float, float]]:
    """在灯带局部窗口内统计外观指标。

    返回 ``(mean_diff, peak_value, ring_value, hue_med, ring_dark_ratio)``。
    只在灯带外接框（按灯带宽扩一圈）内计算，避免整图运算。
    """
    params = cfg.lightbar
    height, width = gray.shape[:2]

    pad = max(3, int(round(max(1.0, _min_side(corners)) * params.ring_pad_ratio)))
    x0 = int(np.clip(corners[:, 0].min() - pad, 0, width))
    x1 = int(np.clip(corners[:, 0].max() + pad + 1, 0, width))
    y0 = int(np.clip(corners[:, 1].min() - pad, 0, height))
    y1 = int(np.clip(corners[:, 1].max() + pad + 1, 0, height))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None

    local = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(local, [np.round(corners - [x0, y0]).astype(np.int32)], 255)
    inside = local > 0
    if not inside.any():
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
    ring = (cv2.dilate(local, kernel) > 0) & ~inside

    local_gray = gray[y0:y1, x0:x1]
    local_value = value[y0:y1, x0:x1]
    local_hue = hue[y0:y1, x0:x1]
    local_diff = diff[y0:y1, x0:x1]

    ring_dark = 1.0
    ring_value = 0.0
    if ring.any():
        ring_dark = float(np.mean(local_gray[ring] <= params.ring_dark_v_max))
        ring_value = float(np.median(local_value[ring]))

    return (
        float(np.mean(local_diff[inside])),
        float(np.percentile(local_value[inside], 90)),
        ring_value,
        float(np.median(local_hue[inside])),
        ring_dark,
    )


def _min_side(corners: np.ndarray) -> float:
    side_a = float(np.linalg.norm(corners[1] - corners[0]))
    side_b = float(np.linalg.norm(corners[2] - corners[1]))
    return min(side_a, side_b)


def _make_lightbar(contour: np.ndarray, color: ArmorColor, diff: np.ndarray,
                   gray: np.ndarray, value: np.ndarray, hue: np.ndarray,
                   cfg: DetectorConfig) -> LightBar | None:
    params = cfg.lightbar

    geometry = _bar_geometry(contour, diff)
    if geometry is None:
        return None
    corners, tilt_deg, height_px, width_px = geometry

    long_side, short_side = max(height_px, width_px), min(height_px, width_px)
    if short_side < 1e-6:
        return None
    if not (params.min_height_px <= long_side <= params.max_height_px):
        return None
    if long_side / short_side < params.min_aspect:
        return None

    area = cv2.contourArea(contour)
    fill_ratio = area / (long_side * short_side)
    if fill_ratio < params.min_fill_ratio:
        return None

    # 长轴必须接近竖直：灯带平躺时不能作为装甲板候选。
    if abs(tilt_deg) > params.max_tilt_deg:
        return None
    if width_px > height_px:
        return None

    top_center = (corners[0] + corners[1]) / 2.0
    bottom_center = (corners[2] + corners[3]) / 2.0

    metrics = _appearance_metrics(corners, gray, value, hue, diff, cfg)
    if metrics is None:
        return None
    mean_diff, peak_value, ring_value, hue_med, ring_dark_ratio = metrics

    # 必须是“点亮的灯带”：
    # 1) 绝对亮度下限按颜色区分（蓝色灯带的 V 天生低于橙红）
    # 2) 相对判据：必须明显亮于紧邻四周（背景噪点周围也是背景，会被挡掉）
    # 3) 四周紧邻暗色（黑色甲板/黑边框）
    if peak_value < params.min_peak_value_for(color):
        return None
    contrast = peak_value - ring_value
    ratio = peak_value / max(ring_value, 1.0)
    if contrast < params.min_ring_contrast or ratio < params.min_ring_ratio:
        return None
    if ring_dark_ratio < params.min_ring_dark_ratio:
        return None

    return LightBar(
        color=color,
        corners=corners,
        height_px=height_px,
        width_px=width_px,
        center=(top_center + bottom_center) / 2.0,
        tilt_deg=tilt_deg,
        fill_ratio=float(fill_ratio),
        mean_diff=mean_diff,
        peak_value=peak_value,
        ring_value=ring_value,
        hue_med=hue_med,
        ring_dark_ratio=ring_dark_ratio,
    )


def detect_for_color(bgr: np.ndarray, color: ArmorColor, cfg: DetectorConfig,
                     hsv: np.ndarray, gray: np.ndarray,
                     mask: Optional[np.ndarray] = None) -> List[LightBar]:
    """在单张图像上检测指定颜色的所有灯带。"""
    if mask is None:
        mask = build_mask(bgr, color, cfg.color, hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diff = color_score(bgr, color)
    value = hsv[:, :, 2]
    hue = hsv[:, :, 0]

    bars: List[LightBar] = []
    for contour in contours:
        bar = _make_lightbar(contour, color, diff, gray, value, hue, cfg)
        if bar is not None:
            bars.append(bar)
    return bars


def detect_light_bars(bgr: np.ndarray, cfg: DetectorConfig,
                      hsv: Optional[np.ndarray] = None,
                      gray: Optional[np.ndarray] = None) -> List[LightBar]:
    """检测图像中所有允许颜色（橙红/蓝）的灯带。"""
    if hsv is None:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if gray is None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    bars: List[LightBar] = []
    for color in cfg.colors:
        bars.extend(detect_for_color(bgr, color, cfg, hsv, gray))
    return bars
