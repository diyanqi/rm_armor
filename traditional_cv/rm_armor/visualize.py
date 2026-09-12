"""检测结果可视化。"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .armor import Armor
from .camera import Camera
from .config import ARMOR_WIDTH_M, DetectorConfig, PairParams
from .detector import DetectionResult
from .lightbar import LightBar
from .pose import ArmorPose

# BGR
_COLOR_BGR = {
    "red": (60, 60, 255),
    "blue": (255, 140, 40),
}
_SIZE_COLOR = {"small": (0, 220, 255), "large": (0, 255, 120)}


def draw_light_bar(image: np.ndarray, bar: LightBar, thickness: int = 1,
                   show_metrics: bool = False) -> None:
    color = _COLOR_BGR.get(bar.color.value, (0, 255, 0))
    cv2.polylines(image, [np.round(bar.corners).astype(np.int32)], True, color, thickness)
    cv2.circle(image, (int(bar.center[0]), int(bar.center[1])), 2, color, -1)
    if show_metrics:
        text = (f"{bar.color.value} hue{bar.hue_med:.0f} V{bar.peak_value:.0f} "
                f"ring{bar.ring_dark_ratio:.2f} d{bar.mean_diff:.0f}")
        _put_text(image, text, (int(bar.center[0]) - 40, int(bar.center[1])), color)


def draw_pose_axes(image: np.ndarray, pose: ArmorPose, camera: Camera,
                   length_m: float = 0.08) -> None:
    """画出装甲板坐标系三轴在当前位姿下的投影。"""
    axes_obj = np.array(
        [[0, 0, 0], [length_m, 0, 0], [0, length_m, 0], [0, 0, -length_m]],
        dtype=np.float64,
    )
    pts, _ = cv2.projectPoints(axes_obj, pose.rvec, pose.tvec, camera.K, camera.dist)
    pts = pts.reshape(-1, 2).astype(int)
    origin = tuple(pts[0])
    cv2.line(image, origin, tuple(pts[1]), (0, 0, 255), 2)
    cv2.line(image, origin, tuple(pts[2]), (0, 255, 0), 2)
    cv2.line(image, origin, tuple(pts[3]), (255, 0, 0), 2)


def draw_armor(image: np.ndarray, armor: Armor, camera: Optional[Camera] = None,
               show_label: bool = True, thickness: int = 2) -> None:
    quad = np.round(armor.quad).astype(np.int32)
    color = _SIZE_COLOR.get(armor.size.value, (255, 255, 255))
    cv2.polylines(image, [quad], True, color, thickness)

    # 对角连线标出装甲板中心，即开火瞄准点。
    cv2.line(image, tuple(quad[0]), tuple(quad[2]), color, 1)
    cv2.line(image, tuple(quad[1]), tuple(quad[3]), color, 1)
    center = tuple(np.round(armor.center).astype(int))
    cv2.circle(image, center, 3, (0, 0, 255), -1)

    if show_label:
        size_m = ARMOR_WIDTH_M[armor.size]
        text = f"{armor.size.value.upper()} {size_m * 100:.1f}cm  s={armor.score:.2f}"
        if armor.pose is not None:
            text += f"  {armor.pose.distance_m:.2f}m yaw={armor.pose.yaw_deg:+.1f}deg"
        _put_text(image, text, (int(quad[0][0]), int(quad[0][1]) - 6), color)

    if camera is not None and armor.pose is not None:
        draw_pose_axes(image, armor.pose, camera)


def draw_interior_region(image: np.ndarray, armor: Armor, params: PairParams) -> None:
    """勾出用于“黑底白字”校验的区域（调试用）。"""
    quad = armor.quad
    cx = float(quad[:, 0].mean())
    cy = float(quad[:, 1].mean())
    margin_x = params.plate_margin_x
    margin_y = params.plate_margin_y

    inner = np.empty_like(quad)
    for i, (x, y) in enumerate(quad):
        inner[i] = (cx + (x - cx) * (1.0 - 2.0 * margin_x),
                    cy + (y - cy) * (1.0 - 2.0 * margin_y))
    cv2.polylines(image, [np.round(inner).astype(np.int32)], True, (255, 255, 0), 1)


def draw_result(image: np.ndarray, result: DetectionResult,
                config: Optional[DetectorConfig] = None,
                show_light_bars: bool = True,
                show_pose_axes: bool = True,
                show_interior: bool = False,
                show_panel: bool = False,
                show_bar_metrics: bool = False,
                header: str = "") -> np.ndarray:
    """把检测结果画到图像副本上并返回。"""
    canvas = image.copy()
    params = (config or DetectorConfig()).pair

    if show_light_bars:
        for bar in result.light_bars:
            draw_light_bar(canvas, bar, show_metrics=show_bar_metrics)

    for armor in result.armors:
        draw_armor(canvas, armor,
                   camera=result.camera if show_pose_axes else None)
        if show_interior:
            draw_interior_region(canvas, armor, params)

    if show_panel:
        draw_info_panel(canvas, result, header=header)
    else:
        _put_text(canvas, f"armors: {len(result.armors)}  bars: {len(result.light_bars)}",
                  (10, 24), (255, 255, 255))
    return canvas


def render_mask(mask: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    """把单通道掩膜转成可视化的三通道图。"""
    colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if size is not None:
        colored = cv2.resize(colored, size, interpolation=cv2.INTER_NEAREST)
    return colored


# ------------------------------------------------------------------ 信息面板 --
PANEL_LINE_H = 18
PANEL_FONT = 0.44


def armor_lines(armor: Armor, index: int) -> list[str]:
    """把一块装甲板的关键信息整理成多行文本。

    坐标系为相机系：x 右、y 下、z 前，单位为米。
    """
    head = (f"#{index} {armor.color.value.upper()} {armor.size.value.upper()} "
            f"w={ARMOR_WIDTH_M[armor.size] * 100:.1f}cm score={armor.score:.2f}")
    cx, cy = armor.center
    lines = [head, f"   px=({cx:.0f},{cy:.0f}) aspect={armor.aspect:.2f}"]

    evidence = armor.evidence
    if evidence is not None:
        flag = "digit Y" if evidence.digit_present else ("digit n" if evidence.evaluated else "digit -")
        lines.append(f"   plate dark={evidence.dark_ratio:.2f} "
                     f"kappa={evidence.kappa:.2f} digit={evidence.digit_ratio:.3f} "
                     f"off={evidence.digit_offset:.2f} {flag}")
    lines.append(f"   bar hue={armor.left.hue_med:.0f}/{armor.right.hue_med:.0f} "
                 f"V={armor.left.peak_value:.0f}/{armor.right.peak_value:.0f} "
                 f"ring={armor.left.ring_dark_ratio:.2f}/{armor.right.ring_dark_ratio:.2f}")

    if armor.pose is None:
        lines.append("   pose: solvePnP failed")
        return lines

    pose = armor.pose
    x, y, z = pose.center_cam
    lines.append(f"   pos x={x:+.3f} y={y:+.3f} z={z:+.3f} m   dist={pose.distance_m:.3f} m")
    lines.append(f"   yaw={pose.yaw_deg:+.1f} pitch={pose.pitch_deg:+.1f} "
                 f"roll={pose.roll_deg:+.1f} deg  bearing={pose.bearing_deg:+.1f}")
    lines.append(f"   reproj={pose.reproj_error_px:.2f} px   z_by_height={pose.distance_from_height_m:.3f} m")
    return lines


def text_width(text: str, scale: float) -> int:
    """估算单行文字宽度（Hershey 字体的平均字宽约为字号的 0.6 倍）。"""
    return int(len(text) * scale * 12.0)


def panel_scale(texts: Sequence[str], max_width: int, preferred: float) -> float:
    """在不超过给定宽度前提下，选取尽量接近期望值的字号。"""
    longest = max((len(t) for t in texts), default=1)
    if longest <= 0 or max_width <= 0:
        return preferred
    return float(min(preferred, max(0.26, max_width / (longest * 12.0))))


def draw_info_panel(image: np.ndarray, result: DetectionResult,
                    header: str = "",
                    show_armor_details: bool = True,
                    origin: tuple[int, int] = (8, 8)) -> None:
    """在左上角画半透明信息面板：耗时、灯带数、以及每块装甲板的位姿。"""
    lines: list[str] = []
    if header:
        lines.append(header)

    timings = result.timings
    if timings:
        lines.append(
            f"detect= {timings.get('total', 0.0):.1f} ms  "
            f"(lightbar {timings.get('lightbar', 0.0):.1f} / "
            f"match {timings.get('match', 0.0):.1f} / "
            f"pose {timings.get('pose', 0.0):.1f})"
        )
    lines.append(f"armors={len(result.armors)}  light_bars={len(result.light_bars)}")

    if show_armor_details:
        for index, armor in enumerate(result.armors, start=1):
            lines.extend(armor_lines(armor, index))
    elif not result.armors:
        lines.append("no armor matched")

    height, width = image.shape[:2]
    scale = panel_scale(lines, int(width * 0.55), PANEL_FONT)
    line_h = int(PANEL_LINE_H * scale / PANEL_FONT)
    box_w = text_width(max(lines, key=len), scale) + 16
    box_h = line_h * len(lines) + 12

    overlay = image.copy()
    cv2.rectangle(overlay, origin, (origin[0] + box_w, origin[1] + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.86, image, 0.14, 0.0, image)

    for row, line in enumerate(lines):
        y = origin[1] + line_h + row * line_h
        colour = (255, 255, 255) if row else (120, 255, 120)
        cv2.putText(image, line, (origin[0] + 8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (origin[0] + 8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, 1, cv2.LINE_AA)


def draw_text_block(image: np.ndarray, lines: Sequence[str], origin: tuple[int, int],
                    max_width: int, colour=(215, 215, 215), preferred: float = 0.42,
                    align_right: bool = False, margin: int = 8) -> None:
    """画一块半透明底衬的多行文字，自动缩放以适配给定宽度。

    ``align_right`` 为真时把 ``origin[0]`` 当作右边界锚点，文字块向左展开。
    """
    if not lines:
        return
    scale = panel_scale(lines, max_width - 16, preferred)
    line_h = max(int(14 * scale / preferred), 11)
    box_w = text_width(max(lines, key=len), scale) + 16
    box_h = line_h * len(lines) + 12
    x0 = max(origin[0] - box_w - margin, 4) if align_right else origin[0]
    y0 = origin[1]

    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.86, image, 0.14, 0.0, image)
    for row, text in enumerate(lines):
        cv2.putText(image, text, (x0 + 8, y0 + line_h + row * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def draw_key_hints(image: np.ndarray, hints: str) -> None:
    """在底部画一行操作提示，超宽时自动缩小字号。"""
    height, width = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, height - 24), (width, height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.86, image, 0.14, 0.0, image)

    scale = panel_scale([hints], width - 20, 0.46)
    cv2.putText(image, hints, (10, height - 7), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (230, 230, 230), 1, cv2.LINE_AA)


def _put_text(image: np.ndarray, text: str, org: tuple[int, int], color) -> None:
    """带黑描边的文字，保证在亮暗背景上都可读。"""
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
