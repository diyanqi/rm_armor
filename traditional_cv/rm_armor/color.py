"""灯带颜色分割。

只保留两类灯带颜色：

- 橙红：R 明显高于 G/B，且色相落在橙红区间（偏深红的状态指示灯被排除）
- 蓝：B 明显高于 R/G，色相落在蓝色区间

其余颜色（绿、黄、紫、白）以及灰白反光都会被丢弃。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import cv2
import numpy as np

from .config import ArmorColor, ColorParams


def color_score(bgr: np.ndarray, color: ArmorColor) -> np.ndarray:
    """返回灯带颜色的“通道差”强度图（int16）。

    红：``R - max(G, B)``；蓝：``B - max(R, G)``。
    通道差比 HSV 更抗白平衡漂移和低曝光。
    """
    b, g, r = cv2.split(bgr.astype(np.int16))
    if color is ArmorColor.RED:
        return r - np.maximum(g, b)
    return b - np.maximum(r, g)


def _hue_range_mask(hsv: np.ndarray, color: ArmorColor, params: ColorParams) -> np.ndarray:
    """色相落在目标区间的像素。红色区间不跨 0，因此深红会落在区间之外。"""
    hue = hsv[:, :, 0].astype(np.float32)
    if color is ArmorColor.RED:
        return (hue >= params.red_hue_min) & (hue <= params.red_hue_max)
    return (hue >= params.blue_hue_min) & (hue <= params.blue_hue_max)


def build_mask(bgr: np.ndarray, color: ArmorColor, params: ColorParams,
               hsv: Optional[np.ndarray] = None) -> np.ndarray:
    """生成该颜色的二值灯带掩膜（uint8，0/255）。"""
    score = color_score(bgr, color)

    if color is ArmorColor.RED:
        base = bgr[:, :, 2].astype(np.int16)
    else:
        base = bgr[:, :, 0].astype(np.int16)

    mask = (score >= params.diff_threshold_for(color)) & (base >= params.min_channel)

    if params.use_hue_gate:
        if hsv is None:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask &= _hue_range_mask(hsv, color, params)
        # 过曝的灯带核心饱和度会掉下来，这类极亮像素单独放行
        saturated = sat >= params.min_saturation
        blown_out = (val >= params.blown_out_value) & (sat >= params.blown_out_min_saturation)
        mask &= saturated | blown_out
        mask &= val >= params.min_value

    binary = mask.astype(np.uint8) * 255

    if params.morph_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (params.morph_close, params.morph_close))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    if params.morph_open > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (params.morph_open, params.morph_open))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    return binary


def build_masks(bgr: np.ndarray, colors: Sequence[ArmorColor], params: ColorParams,
                hsv: Optional[np.ndarray] = None) -> Dict[ArmorColor, np.ndarray]:
    """一次算好多色掩膜，避免重复做 BGR->HSV 转换。"""
    if params.use_hue_gate and hsv is None:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return {color: build_mask(bgr, color, params, hsv) for color in colors}
