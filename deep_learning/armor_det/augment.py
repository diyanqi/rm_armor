"""离线增广：噪点、人为遮挡、亮度扰动。

设计要点：三种手段都是**几何不变**的——目标的位置/形状/朝向都不改变，
因此原始 OBB 四点标签可以原样沿用，零标注风险。
遮挡特意限制对单个目标的覆盖比例，既模拟实战中的部分遮挡，又保证目标仍可辨识。
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import AugmentConfig


def obb_norm_to_px(poly_norm, width: int, height: int) -> np.ndarray:
    """归一化四点 (x1,y1,...,x4,y4) -> 像素坐标 (4,2) float32。"""
    pts = np.asarray(poly_norm, dtype=np.float32).reshape(-1, 2).copy()
    pts[:, 0] *= width
    pts[:, 1] *= height
    return pts


def add_gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """叠加高斯噪点。"""
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper(img: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """叠加椒盐噪点（部分像素置为纯黑/纯白）。"""
    out = img.copy()
    h, w = img.shape[:2]
    n = int(ratio * h * w)
    if n <= 0:
        return out
    ys = rng.integers(0, h, n)
    xs = rng.integers(0, w, n)
    vals = (rng.integers(0, 2, n) * 255).astype(np.uint8)
    out[ys, xs] = vals[:, None] if out.ndim == 3 else vals
    return out


def adjust_brightness_contrast(img: np.ndarray, mode: str, rng: np.random.Generator,
                               cfg: AugmentConfig) -> np.ndarray:
    """亮度/对比度/伽马扰动。mode 为 'darken' 或 'brighten'。"""
    lo, hi = cfg.darken_range if mode == "darken" else cfg.brighten_range
    factor = float(rng.uniform(lo, hi))
    out = img.astype(np.float32) * factor

    contrast = float(rng.uniform(*cfg.contrast_range))
    if abs(contrast - 1.0) > 1e-3:
        mean = float(out.mean())
        out = (out - mean) * contrast + mean

    gamma = float(rng.uniform(*cfg.gamma_range))
    if abs(gamma - 1.0) > 1e-3:
        out = np.power(np.clip(out, 0, 255) / 255.0, gamma) * 255.0

    return np.clip(out, 0, 255).astype(np.uint8)


def _poly_cover_ratio(poly_px: np.ndarray, rect) -> float:
    """目标多边形被矩形遮挡的面积比例。"""
    x1, y1, x2, y2 = rect
    box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    area = float(cv2.contourArea(poly_px))
    if area <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(poly_px.astype(np.float32), box)
    return float(inter) / area


def _sample_occluder_color(img: np.ndarray, rect, rng: np.random.Generator):
    """遮挡块颜色：偏暗，贴近夜间赛场的阴影/结构件干扰。

    夜色场景里如果用亮色块，会显得很突兀，也容易引入虚假的"亮目标"，
    因此以纯黑/暗灰为主，少量用区域均值。
    """
    mode = float(rng.random())
    if mode < 0.5:
        return (0, 0, 0)
    if mode < 0.8:
        v = int(rng.integers(0, 60))
        return (v, v, v)
    x1, y1, x2, y2 = rect
    region = img[y1:y2, x1:x2]
    if region.size == 0:
        return (0, 0, 0)
    mean = region.reshape(-1, img.shape[2] if img.ndim == 3 else 1).mean(axis=0)
    return tuple(int(c) for c in np.atleast_1d(mean)[: (img.shape[2] if img.ndim == 3 else 1)])


def add_occlusion(img: np.ndarray, polys_px, rng: np.random.Generator,
                  cfg: AugmentConfig) -> np.ndarray:
    """在装甲板（标注框）内部随机落遮挡块。

    与"随手乱遮"不同：这里先随机挑一块装甲板，在其多边形内部采样一个中心点，
    再按该板尺寸的固定比例生成遮挡块——保证遮挡**一定落在甲板像素上**，
    同时限制对目标的遮挡比例（既要遮到，又不能把目标整个盖没）。
    """
    out = img.copy()
    if not polys_px:
        return out
    height, width = out.shape[:2]
    count = int(rng.integers(cfg.occlusion_min, cfg.occlusion_max + 1))

    for _ in range(count):
        for _try in range(cfg.occlusion_tries):
            poly = polys_px[int(rng.integers(0, len(polys_px)))].astype(np.float32)
            xs, ys = poly[:, 0], poly[:, 1]
            short_side = max(float(min(xs.max() - xs.min(), ys.max() - ys.min())), 4.0)

            side = short_side * float(rng.uniform(*cfg.occlusion_patch_scale))
            patch_w = int(round(side * float(rng.uniform(0.8, 1.2))))
            patch_h = int(round(side * float(rng.uniform(0.8, 1.2))))
            if patch_w < 3 or patch_h < 3:
                continue

            # 在板的多边形内部采样一个落点作为遮挡块中心
            center = None
            for _ in range(20):
                px = float(rng.uniform(xs.min(), xs.max()))
                py = float(rng.uniform(ys.min(), ys.max()))
                if cv2.pointPolygonTest(poly, (px, py), False) >= 0:
                    center = (px, py)
                    break
            if center is None:
                continue

            x1 = int(round(center[0] - patch_w / 2))
            y1 = int(round(center[1] - patch_h / 2))
            x1 = max(0, min(x1, width - patch_w - 1))
            y1 = max(0, min(y1, height - patch_h - 1))
            rect = (x1, y1, x1 + patch_w, y1 + patch_h)

            cover = _poly_cover_ratio(poly, rect)
            if not cfg.occlusion_min_cover <= cover <= cfg.max_target_cover:
                continue  # 遮得太少或太多，重采样

            color = _sample_occluder_color(out, rect, rng)
            cv2.rectangle(out, (x1, y1), (x1 + patch_w, y1 + patch_h), color, -1)
            break
    return out


def augment_image(img: np.ndarray, polys_px, rng: np.random.Generator,
                  cfg: AugmentConfig | None = None) -> np.ndarray:
    """对单张图随机组合施加：亮度扰动 / 噪点 / 遮挡。

    保证至少施加一种手段；返回值几何与原图一致，标签无需改动。
    """
    cfg = cfg or AugmentConfig()
    out = img
    applied = False

    if rng.random() < cfg.brightness_prob:
        mode = "darken" if rng.random() < 0.5 else "brighten"
        out = adjust_brightness_contrast(out, mode, rng, cfg)
        applied = True

    if rng.random() < cfg.noise_prob:
        if rng.random() < 0.5:
            out = add_gaussian_noise(out, float(rng.uniform(*cfg.gaussian_sigma)), rng)
        else:
            out = add_salt_pepper(out, float(rng.uniform(*cfg.salt_pepper_ratio)), rng)
        applied = True

    if polys_px and rng.random() < cfg.occlusion_prob:
        out = add_occlusion(out, polys_px, rng, cfg)
        applied = True

    if not applied:
        mode = "darken" if rng.random() < 0.5 else "brighten"
        out = adjust_brightness_contrast(out, mode, rng, cfg)

    return out
