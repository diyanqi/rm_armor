"""合成数据自检：不需要图片即可验证几何配对与 PnP 位姿解算是否正确。

运行::

    python selfcheck.py
"""

from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rm_armor import (  # noqa: E402
    ARMOR_HEIGHT_M,
    ARMOR_WIDTH_M,
    ArmorColor,
    ArmorDetector,
    ArmorSize,
    CameraConfig,
    DetectorConfig,
    build_camera,
    object_points,
    solve_armor_pose,
)

WIDTH, HEIGHT = 1280, 720
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}{(' -> ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def camera() -> object:
    return build_camera(CameraConfig(hfov_deg=60.0), WIDTH, HEIGHT)


def project(quad_obj: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, cam) -> np.ndarray:
    pts, _ = cv2.projectPoints(quad_obj, rvec, tvec, cam.K, cam.dist)
    return pts.reshape(4, 2).astype(np.float32)


def test_object_geometry() -> None:
    for size in (ArmorSize.SMALL, ArmorSize.LARGE):
        quad = object_points(size)
        width = float(np.linalg.norm(quad[1] - quad[0]))
        height = float(np.linalg.norm(quad[3] - quad[0]))
        check(f"{size.value} 装甲板宽度", math.isclose(width, ARMOR_WIDTH_M[size], rel_tol=1e-9),
              f"{width:.4f} m")
        check(f"{size.value} 装甲板高度", math.isclose(height, ARMOR_HEIGHT_M, rel_tol=1e-9),
              f"{height:.5f} m")


def test_pose_roundtrip() -> None:
    """给定已知位姿投影出角点，再用 PnP 反解，检查角度/距离是否复原。"""
    cam = camera()
    cases = [
        # (规格, 距离, yaw, pitch, roll)  角度单位为度
        (ArmorSize.LARGE, 1.50, 0.0, 0.0, 0.0),
        (ArmorSize.LARGE, 2.50, 18.0, 0.0, 0.0),
        (ArmorSize.SMALL, 1.20, -25.0, 0.0, 0.0),
        (ArmorSize.SMALL, 3.00, 8.0, 5.0, 12.0),
        (ArmorSize.LARGE, 4.00, -12.0, -8.0, -6.0),
    ]
    for size, dist, yaw, pitch, roll in cases:
        obj = object_points(size)
        rvec = np.array([math.radians(pitch), math.radians(yaw), math.radians(roll)],
                        dtype=np.float64)
        tvec = np.array([0.0, 0.0, dist], dtype=np.float64)
        quad = project(obj, rvec, tvec, cam)

        pose = solve_armor_pose(quad, size, cam)
        label = f"{size.value} d={dist} yaw={yaw} pitch={pitch}"
        if pose is None:
            check(f"位姿解算 {label}", False, "solvePnP 返回 None")
            continue

        check(f"位姿距离 {label}", abs(pose.distance_m - dist) < 0.03 * dist + 0.01,
              f"{pose.distance_m:.3f} m")
        check(f"位姿 yaw {label}", abs(pose.yaw_deg - yaw) < 2.5, f"{pose.yaw_deg:+.1f} deg")
        check(f"位姿 pitch {label}", abs(pose.pitch_deg - pitch) < 2.5, f"{pose.pitch_deg:+.1f} deg")
        check(f"法线朝向相机 {label}", pose.rotation[2, 2] > 0.0,
              f"R[2,2]={pose.rotation[2, 2]:.3f}")
        check(f"重投影误差 {label}", pose.reproj_error_px < 1.0,
              f"{pose.reproj_error_px:.3f} px")


def test_size_aspect_split() -> None:
    """尺寸分类边界应能把大小装甲板的标称宽高比分开。"""
    from rm_armor.config import ARMOR_ASPECT, ArmorSizeEstimator

    small, small_conf = ArmorSizeEstimator.from_aspect(ARMOR_ASPECT[ArmorSize.SMALL])
    large, large_conf = ArmorSizeEstimator.from_aspect(ARMOR_ASPECT[ArmorSize.LARGE])
    check("小装甲板宽高比判定", small is ArmorSize.SMALL, f"confidence={small_conf:.2f}")
    check("大装甲板宽高比判定", large is ArmorSize.LARGE, f"confidence={large_conf:.2f}")


def test_empty_frame() -> None:
    """没有任何灯带的图像必须返回空结果而不是报错。"""
    detector = ArmorDetector(DetectorConfig())
    for name, image in (
        ("全黑图", np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)),
        ("全白图", np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)),
    ):
        result = detector.detect(image)
        check(f"无灯带图像({name}) 返回空", not result.armors and not result.light_bars,
              f"armors={len(result.armors)} bars={len(result.light_bars)}")


def test_synthetic_light_bars() -> None:
    """橙红与蓝色各画一块装甲板，验证两种颜色的检测 + 配对 + 位姿都能走通。"""
    detector = ArmorDetector(DetectorConfig())
    bar_h = 200
    bar_w = 16
    # 让两灯带外沿间距与灯带高度之比接近大装甲板的标称宽高比
    gap = max(bar_w, int(round(ARMOR_WIDTH_M[ArmorSize.LARGE] / ARMOR_HEIGHT_M * bar_h)) - 2 * bar_w)

    cases = [
        ("橙红", ArmorColor.RED, (30, 90, 255)),
        ("蓝", ArmorColor.BLUE, (255, 90, 30)),
    ]
    for name, color, colour in cases:
        image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        x0, y0 = WIDTH // 2 - gap // 2 - bar_w, HEIGHT // 2 - bar_h // 2
        cv2.rectangle(image, (x0, y0), (x0 + bar_w, y0 + bar_h), colour, -1)
        cv2.rectangle(image, (x0 + bar_w + gap, y0),
                      (x0 + bar_w + gap + bar_w, y0 + bar_h), colour, -1)

        # 板面中间的白色数字：kappa = 字宽 / 灯带高 ≈ 0.8，与真实装甲板一致
        digit_w = int(round(0.8 * bar_h))
        digit_h = int(round(0.7 * bar_h))
        cx = x0 + bar_w + gap // 2
        cy = y0 + bar_h // 2
        cv2.rectangle(image, (cx - digit_w // 2, cy - digit_h // 2),
                      (cx + digit_w // 2, cy + digit_h // 2), (245, 245, 245), -1)

        result = detector.detect(image)
        bars = [b for b in result.light_bars if b.color is color]
        armors = [a for a in result.armors if a.color is color]
        check(f"合成灯带检测({name})", len(bars) == 2, f"bars={len(bars)}")
        check(f"合成装甲板配对({name})", len(armors) == 1, f"armors={len(armors)}")
        if armors:
            armor = armors[0]
            check(f"合成装甲板规格({name})", armor.size is ArmorSize.LARGE, armor.size.value)
            check(f"合成装甲板位姿已解算({name})", armor.pose is not None)
            if armor.pose is not None:
                check(f"合成装甲板正对相机({name})", abs(armor.pose.yaw_deg) < 3.0,
                      f"yaw={armor.pose.yaw_deg:+.1f} deg")


def test_blue_channel_signal() -> None:
    """蓝色灯带在传感器里大量漏光到 G/R，验证较窄的通道差仍能被正确分离。"""
    detector = ArmorDetector(DetectorConfig())
    # 模拟真实蓝色灯带：B 高，但 G/R 也被抬起来，B-max(R,G) 只有 ~70
    image = np.zeros((240, 240, 3), dtype=np.uint8)
    cv2.rectangle(image, (110, 40), (126, 200), (200, 130, 60), -1)
    bars = detector.detect(image).light_bars
    check("低通道差蓝色灯带可检出", len(bars) == 1, f"bars={len(bars)}")

    # 光晕更大但更暗的蓝色块不应被当作灯带（通道差不足）
    halo = np.zeros((240, 240, 3), dtype=np.uint8)
    cv2.rectangle(halo, (100, 30), (140, 210), (120, 95, 65), -1)
    check("暗蓝光晕不认作灯带", len(detector.detect(halo).light_bars) == 0,
          f"bars={len(detector.detect(halo).light_bars)}")


def test_color_gate() -> None:
    """只认橙红与蓝：绿色、黄色、灰白反光都不应被当作灯带。"""
    detector = ArmorDetector(DetectorConfig())
    cases = [
        ("橙红灯带", (30, 90, 255), True),
        ("蓝色灯带", (255, 90, 30), True),
        ("绿色灯", (0, 255, 0), False),
        ("黄色灯", (0, 255, 255), False),
        ("灰白反光", (210, 210, 210), False),
        ("紫色灯", (200, 40, 200), False),
    ]
    for name, colour, expected in cases:
        image = np.zeros((240, 240, 3), dtype=np.uint8)
        cv2.rectangle(image, (110, 40), (126, 200), colour, -1)
        found = len(detector.detect(image).light_bars) > 0
        check(f"颜色门限({name})", found == expected,
              f"检出={found} 期望={expected}")


def test_dim_bar_rejected() -> None:
    """暗的色块（如摄像头上更暗的状态指示灯）不算灯带。"""
    detector = ArmorDetector(DetectorConfig())
    for name, colour in (("深红状态灯", (0, 0, 110)), ("暗红灯带", (10, 20, 120))):
        image = np.zeros((240, 240, 3), dtype=np.uint8)
        cv2.rectangle(image, (110, 40), (126, 200), colour, -1)
        found = len(detector.detect(image).light_bars) > 0
        check(f"暗色块不认作灯带({name})", not found, f"检出={found}")


def test_dark_surround_required() -> None:
    """灯带四周必须有暗色衬托：悬在亮色背景上的色块不算灯带。"""
    detector = ArmorDetector(DetectorConfig())
    bright_bg = np.full((240, 240, 3), (250, 250, 250), dtype=np.uint8)
    cv2.rectangle(bright_bg, (110, 40), (126, 200), (30, 90, 255), -1)
    check("亮背景上的色块不认作灯带",
          len(detector.detect(bright_bg).light_bars) == 0,
          f"bars={len(detector.detect(bright_bg).light_bars)}")

    dark_bg = np.zeros((240, 240, 3), dtype=np.uint8)
    cv2.rectangle(dark_bg, (110, 40), (126, 200), (30, 90, 255), -1)
    check("暗背景上的灯带被检出",
          len(detector.detect(dark_bg).light_bars) == 1,
          f"bars={len(detector.detect(dark_bg).light_bars)}")


def test_plate_validation() -> None:
    """跨板误配校验：两条灯带之间必须有黑底甲板与居中的白色数字。"""
    detector = ArmorDetector(DetectorConfig())
    bar_h, bar_w = 200, 16
    gap = int(round(ARMOR_WIDTH_M[ArmorSize.LARGE] / ARMOR_HEIGHT_M * bar_h)) - 2 * bar_w
    x0, y0 = 200, 200
    colour = (30, 90, 255)

    def build(with_digit: bool) -> np.ndarray:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(image, (x0, y0), (x0 + bar_w, y0 + bar_h), colour, -1)
        cv2.rectangle(image, (x0 + bar_w + gap, y0),
                      (x0 + bar_w + gap + bar_w, y0 + bar_h), colour, -1)
        if with_digit:
            dw, dh = int(round(0.8 * bar_h)), int(round(0.7 * bar_h))
            cx, cy = x0 + bar_w + gap // 2, y0 + bar_h // 2
            cv2.rectangle(image, (cx - dw // 2, cy - dh // 2),
                          (cx + dw // 2, cy + dh // 2), (245, 245, 245), -1)
        return image

    check("有黑底白字 → 配对成 LARGE",
          len(detector.detect(build(True)).armors) == 1,
          f"armors={len(detector.detect(build(True)).armors)}")
    check("中间没有白字 → LARGE 被拒绝",
          len(detector.detect(build(False)).armors) == 0,
          f"armors={len(detector.detect(build(False)).armors)}")


def main() -> int:
    test_object_geometry()
    test_size_aspect_split()
    test_pose_roundtrip()
    test_color_gate()
    test_dim_bar_rejected()
    test_dark_surround_required()
    test_empty_frame()
    test_synthetic_light_bars()
    test_blue_channel_signal()
    test_plate_validation()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} 项失败: {', '.join(FAILURES)}")
        return 1
    print("全部自检通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
