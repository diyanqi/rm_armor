"""灯带配对：把属于同一块装甲板的两条灯带组合起来。"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .armor import Armor, PlateEvidence
from .config import ARMOR_ASPECT, ArmorSize, ArmorSizeEstimator, DetectorConfig, PairParams
from .lightbar import LightBar
from .pose import mean_horizontal_edge, mean_vertical_edge


def build_quad(left: LightBar, right: LightBar) -> np.ndarray:
    """由左右灯带的外沿角点构成装甲板正面四边形 [左上, 右上, 右下, 左下]。"""
    return np.array(
        [left.corners[0], right.corners[1], right.corners[2], left.corners[3]],
        dtype=np.float32,
    )


@dataclass
class _Candidate:
    """一条候选配对，保留灯带下标以便去重。"""

    armor: Armor
    left_idx: int
    right_idx: int


# ------------------------------------------------------------------ 几何约束 --
def _geometry_features(left: LightBar, right: LightBar, quad: np.ndarray,
                       params: PairParams) -> Optional[Dict[str, float]]:
    """计算两条灯带的几何一致性特征，不满足硬约束时返回 None。"""
    h1, h2 = left.height_px, right.height_px
    h_mean = (h1 + h2) / 2.0
    if max(h1, h2) / min(h1, h2) > params.max_height_ratio:
        return None
    if abs(float(left.center[1] - right.center[1])) > params.max_center_dy_ratio * h_mean:
        return None

    tilt_diff = abs(left.tilt_deg - right.tilt_deg)
    if tilt_diff > params.max_tilt_diff_deg:
        return None

    width_px = mean_horizontal_edge(quad)
    aspect = width_px / h_mean
    if not (params.min_aspect <= aspect <= params.max_aspect):
        return None

    # 两条灯带的中心连线应垂直于灯带长轴（否则是同一辆车上离得很远的两个装甲板）。
    perp_dev = abs(_angle_between(right.center - left.center, (left.axis + right.axis) / 2.0) - 90.0)
    if perp_dev > params.max_perp_dev_deg:
        return None

    return {
        "height_ratio": max(h1, h2) / min(h1, h2),
        "tilt_diff": tilt_diff,
        "perp_dev": perp_dev,
        "width_px": width_px,
        "height_px": h_mean,
        "aspect": aspect,
    }


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cos))


# --------------------------------------------------------- 板面校验（黑底白字）--
def warp_plate(gray: np.ndarray, quad: np.ndarray, params: PairParams,
               target_w: int = 160) -> Optional[np.ndarray]:
    """把装甲板正面四边形矫正成正视矩形，并裁掉左右边缘的灯带与光晕。"""
    width_px = mean_horizontal_edge(quad)
    height_px = mean_vertical_edge(quad)
    if width_px < 6.0 or height_px < 5.0:
        return None

    target_h = int(round(target_w * height_px / width_px))
    target_h = max(8, min(240, target_h))

    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(np.asarray(quad, dtype=np.float32), dst)
    warp = cv2.warpPerspective(gray, M, (target_w, target_h), flags=cv2.INTER_LINEAR)

    mx = int(round(target_w * params.plate_margin_x))
    my = min(int(round(target_h * params.plate_margin_y)), max(0, (target_h - 4) // 2))
    mx = min(mx, max(0, (target_w - 4) // 2))
    plate = warp[my:target_h - my, mx:target_w - mx]
    if plate.size < 24:
        return None
    return plate


def _largest_digit_blob(white: np.ndarray) -> Optional[Dict[str, float]]:
    """取面积最大的白色连通域，返回其几何描述（占板面的比例）。"""
    height, width = white.shape[:2]
    count, _, stats, centroids = cv2.connectedComponentsWithStats(white.astype(np.uint8),
                                                                   connectivity=8)
    if count <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    index = int(np.argmax(areas)) + 1
    return {
        "area": float(stats[index, cv2.CC_STAT_AREA]),
        "w": float(stats[index, cv2.CC_STAT_WIDTH]),
        "h": float(stats[index, cv2.CC_STAT_HEIGHT]),
        "cx": float(centroids[index][0]),
        "area_ratio": float(stats[index, cv2.CC_STAT_AREA]) / float(height * width),
        "w_ratio": float(stats[index, cv2.CC_STAT_WIDTH]) / width,
        "h_ratio": float(stats[index, cv2.CC_STAT_HEIGHT]) / height,
        "offset": abs(float(centroids[index][0]) / width - 0.5),
    }


def plate_evidence(gray: np.ndarray, sat: np.ndarray, quad: np.ndarray,
                   bar_height_px: float, params: PairParams) -> Optional[PlateEvidence]:
    """统计板面证据：是否黑底、是否有居中的白色数字。

    白字用**绝对**判据（够亮 + 几乎不饱和），因此灯带光晕、彩色背景、
    以及跨板配对中间的窄亮缝都不会被当成数字。
    """
    plate = warp_plate(gray, quad, params)
    if plate is None:
        return None
    plate_sat = warp_plate(sat, quad, params, target_w=plate.shape[1])
    if plate_sat is None or plate_sat.shape != plate.shape:
        plate_sat = np.zeros_like(plate)

    evidence = PlateEvidence(
        dark_ratio=float(np.mean(plate <= params.plate_dark_v_max)),
        contrast=float(np.percentile(plate, 99) - np.median(plate)),
    )

    if bar_height_px < params.digit_check_min_bar_height:
        # 灯带太短，白字在像素上已经糊掉，不做数字校验。
        return evidence

    evidence.evaluated = True
    white = (plate >= params.digit_v_min) & (plate_sat <= params.digit_sat_max)
    blob = _largest_digit_blob(white)
    if blob is None:
        return evidence

    evidence.digit_ratio = blob["area_ratio"]
    evidence.digit_offset = blob["offset"]
    # 矫正变换保持比例，所以 warped 空间的宽度比就是原图的宽度比：
    # kappa = 白字像素宽度 / 装甲板像素高度
    evidence.kappa = blob["w"] / max(plate.shape[0], 1e-6)
    evidence.digit_height_ratio = blob["h_ratio"]

    evidence.digit_present = (
        params.digit_min_area_ratio <= blob["area_ratio"] <= params.digit_max_area_ratio
        and blob["offset"] <= params.digit_max_offset
        and blob["h_ratio"] >= params.digit_min_height_ratio
        and params.digit_min_kappa <= evidence.kappa <= params.digit_max_kappa
        and blob["w_ratio"] <= 0.85
    )
    return evidence


def _digit_score(evidence: PlateEvidence, params: PairParams) -> float:
    """把板面证据折算成 0~1 的分数。"""
    s_dark = float(np.clip(evidence.dark_ratio / 0.85, 0.0, 1.0))
    if not evidence.digit_present:
        return 0.35 * s_dark
    s_kappa = float(np.clip(evidence.kappa / (params.digit_min_kappa * 2.0), 0.0, 1.0))
    s_area = float(np.clip(evidence.digit_ratio / 0.12, 0.0, 1.0))
    s_center = float(np.clip(1.0 - evidence.digit_offset / params.digit_max_offset, 0.0, 1.0))
    return 0.3 * s_dark + 0.25 * s_kappa + 0.25 * s_area + 0.2 * s_center


def _brightness_score(left: LightBar, right: LightBar, params: PairParams) -> float:
    """灯带亮度分：反光/环境光斑的通道差通常远低于真正的灯带。"""
    mean_diff = (left.mean_diff + right.mean_diff) / 2.0
    return float(np.clip(mean_diff / params.brightness_ref, 0.0, 1.0))


def _small_pair_aspect(left: LightBar, right: LightBar, params: PairParams) -> Optional[float]:
    """两条灯带能否配成一块“小甲板”。可以则返回宽高比，否则 None。

    拆解仲裁时对竖直错位的容忍度放宽：同一块甲板在斜视下两条灯带的中心高度
    本来就会有明显差异，而误配跨板灯带的代价更高。
    """
    if max(left.height_px, right.height_px) / min(left.height_px, right.height_px) \
            > params.max_height_ratio:
        return None
    if abs(float(left.center[1] - right.center[1])) \
            > params.decompose_small_max_center_dy_ratio * (left.height_px + right.height_px) / 2:
        return None

    quad = build_quad(left, right)
    geo = _geometry_features(left, right, quad, params)
    if geo is None or geo["aspect"] > params.decompose_small_max_aspect:
        return None
    return geo["aspect"]


def _has_small_decomposition(left_idx: int, right_idx: int,
                             bars: Sequence[LightBar], params: PairParams) -> bool:
    """判断这对灯带是否更像“两块相邻小甲板”而不是“一块大甲板”。

    若左灯带能与其左侧的某条灯带、右灯带能与其右侧的某条灯带各自配成
    合理的小甲板，那么当前的 LARGE 就是跨板组合，应当丢弃。
    """
    left, right = bars[left_idx], bars[right_idx]
    left_ok = any(
        bar.color is left.color and _small_pair_aspect(bar, left, params) is not None
        for index, bar in enumerate(bars) if index != left_idx and index != right_idx
        and bar.center[0] < left.center[0]
    )
    if not left_ok:
        return False
    return any(
        bar.color is right.color and _small_pair_aspect(right, bar, params) is not None
        for index, bar in enumerate(bars) if index != left_idx and index != right_idx
        and bar.center[0] > right.center[0]
    )



# ------------------------------------------------------------------ 遮挡检查 --
def _bar_between(quad: np.ndarray, bars: Sequence[LightBar],
                 skip: Tuple[int, int]) -> bool:
    """若第三条灯带的中心落在该四边形内部，说明这是跨车/跨板的错误配对。"""
    quad32 = np.asarray(quad, dtype=np.float32)
    for index, bar in enumerate(bars):
        if index in skip:
            continue
        if cv2.pointPolygonTest(quad32, (float(bar.center[0]), float(bar.center[1])), False) >= 0:
            return True
    return False


# ------------------------------------------------------------------ 主流程 --
def _try_pair(left_idx: int, right_idx: int, bars: Sequence[LightBar],
              gray: np.ndarray, sat: np.ndarray,
              cfg: DetectorConfig) -> Optional[_Candidate]:
    left, right = bars[left_idx], bars[right_idx]
    params = cfg.pair
    quad = build_quad(left, right)

    if min(left.mean_diff, right.mean_diff) < params.min_bar_brightness:
        return None
    geo = _geometry_features(left, right, quad, params)
    if geo is None:
        return None
    if _bar_between(quad, bars, (left_idx, right_idx)):
        return None

    size, size_conf = ArmorSizeEstimator.from_aspect(geo["aspect"])

    # 拆解仲裁：跨板误配出来的“大甲板”优先让位给两块真正的小甲板。
    if (params.decompose_large and size is ArmorSize.LARGE
            and _has_small_decomposition(left_idx, right_idx, bars, params)):
        return None

    # 板面校验：必须是黑底，并且（能看清数字时）中间有居中的白色数字。
    # 跨板误配的中间只有一条窄亮缝，kappa 很小，会被这里挡掉。
    evidence = None
    if params.use_plate_validation:
        evidence = plate_evidence(gray, sat, quad, geo["height_px"], params)
        if evidence is not None:
            if evidence.dark_ratio < params.plate_min_dark_ratio:
                return None
            if evidence.evaluated and not evidence.digit_present:
                # 宽高比落在 LARGE 区间却没有白字 → 典型的“左板右边灯带 + 右板左边灯带”
                if params.large_requires_digit and size is ArmorSize.LARGE:
                    return None
                if params.require_digit_when_evaluated:
                    return None

    s_height = 1.0 - (geo["height_ratio"] - 1.0) / max(params.max_height_ratio - 1.0, 1e-6)
    s_tilt = 1.0 - geo["tilt_diff"] / max(params.max_tilt_diff_deg, 1e-6)
    s_perp = 1.0 - geo["perp_dev"] / max(params.max_perp_dev_deg, 1e-6)

    sub_scores: Dict[str, float] = {
        "aspect": _aspect_score(geo["aspect"], size),
        "height": float(np.clip(s_height, 0.0, 1.0)),
        "parallel": float(np.clip(0.5 * (s_tilt + s_perp), 0.0, 1.0)),
        "brightness": _brightness_score(left, right, params),
    }
    weights = {
        "aspect": params.weights.aspect,
        "height": params.weights.height,
        "parallel": params.weights.parallel,
        "brightness": params.weights.brightness,
    }

    if evidence is not None:
        sub_scores.update({
            "dark_ratio": evidence.dark_ratio,
            "digit_ratio": evidence.digit_ratio,
            "kappa": evidence.kappa,
            "plate": _digit_score(evidence, params),
        })
        weights["plate"] = params.weights.number

    total_weight = sum(weights[name] for name in sub_scores if name in weights)
    score = sum(weights[name] * sub_scores[name] for name in sub_scores if name in weights)
    score /= max(total_weight, 1e-6)
    if score < params.min_score:
        return None

    armor = Armor(
        left=left,
        right=right,
        color=left.color,
        size=size,
        size_confidence=size_conf,
        quad=quad,
        width_px=geo["width_px"],
        height_px=geo["height_px"],
        aspect=geo["aspect"],
        score=float(score),
        sub_scores=sub_scores,
        evidence=evidence,
    )
    return _Candidate(armor=armor, left_idx=left_idx, right_idx=right_idx)


def _aspect_score(aspect: float, size) -> float:
    """宽高比与该规格标称值的接近程度（相对偏差 45% 时归零）。"""
    nominal = ARMOR_ASPECT[size]
    deviation = abs(aspect - nominal) / nominal
    return float(np.clip(1.0 - deviation / 0.45, 0.0, 1.0))


def match_armors(bars: Sequence[LightBar], gray: np.ndarray, sat: np.ndarray,
                 cfg: DetectorConfig) -> List[Armor]:
    """枚举所有灯带组合，返回互相不共用灯带的装甲板列表（按得分降序）。"""
    ordered = list(bars)
    candidates: List[_Candidate] = []
    for left_idx, right_idx in itertools.combinations(range(len(ordered)), 2):
        left, right = ordered[left_idx], ordered[right_idx]
        if left.color is not right.color:
            continue
        if left.center[0] > right.center[0]:
            left_idx, right_idx = right_idx, left_idx
        candidate = _try_pair(left_idx, right_idx, ordered, gray, sat, cfg)
        if candidate is not None:
            candidates.append(candidate)

    # 贪心去重：得分高的配对优先占用灯带，保证每条灯带只归属一块装甲板。
    candidates.sort(key=lambda c: c.armor.score, reverse=True)
    used: set[int] = set()
    armors: List[Armor] = []
    for candidate in candidates:
        if candidate.left_idx in used or candidate.right_idx in used:
            continue
        used.add(candidate.left_idx)
        used.add(candidate.right_idx)
        armors.append(candidate.armor)
    return armors
