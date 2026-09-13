"""装甲板位姿解算（PnP）：由 OBB 四角点 + 已知真实尺寸求位置与朝向。

真实尺寸、内参估算方式与求解流程参考 ``traditional_cv/rm_armor``
（``ARMOR_WIDTH_M`` / ``ARMOR_HEIGHT_M``、按水平视场角估算内参、
``solvePnPGeneric(IPPE)`` 取"正面朝向相机"的解）。
为保持两套方案互不耦合，这里自带一份精简实现，不 import traditional_cv。

相机坐标系：x 右、y 下、z 前（单位：米）。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------- 真实尺寸 --
# 单位：米，取自 RoboMaster 官方装甲板规格（与 traditional_cv 一致）。
ARMOR_WIDTH_M = {"small": 0.136, "large": 0.231}
ARMOR_HEIGHT_M = 0.05603

# 车辆类型（0-7: sentry/hero/engineer/infantry3/infantry4/outpost/base_small/base_big）
# 中挂"大装甲板"的类型；其余按小装甲板算。
LARGE_VEHICLES = (1, 7)          # hero、base_big


def vehicle_of(class_id: int) -> int:
    """24 类组合 id -> 车辆类型 id（0-7）。"""
    return int(class_id) % 8


def armor_size_of(class_id: int) -> str:
    """按车辆类型推断装甲板规格（'small' / 'large'）。"""
    return "large" if vehicle_of(class_id) in LARGE_VEHICLES else "small"


def nominal_aspect(size: str) -> float:
    """该规格装甲板标称宽高比（宽 / 高）。"""
    return ARMOR_WIDTH_M[size] / ARMOR_HEIGHT_M


def size_from_aspect(width_px: float, height_px: float) -> Tuple[str, float]:
    """按观测宽高比推断规格（备用判据，返回 (规格, 置信度)）。"""
    if height_px <= 1e-6:
        return "small", 0.0
    aspect = width_px / height_px
    split = math.sqrt((ARMOR_WIDTH_M["small"] / ARMOR_HEIGHT_M)
                      * (ARMOR_WIDTH_M["large"] / ARMOR_HEIGHT_M))
    size = "large" if aspect >= split else "small"
    nominal = ARMOR_WIDTH_M[size] / ARMOR_HEIGHT_M
    margin = abs(math.log(max(aspect, 1e-6) / split))
    span = abs(math.log(nominal / split)) or 1.0
    return size, float(min(1.0, margin / span))


# ------------------------------------------------------------------ 相机模型 --
@dataclass
class Camera:
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int

    @property
    def focal_px(self) -> float:
        return float(self.K[0, 0])

    def distance_from_height(self, height_px: float, real_height_m: float) -> float:
        """由已知真实高度反推深度：z = f * H / h_px（用于交叉验证 PnP）。"""
        if height_px <= 1e-6:
            return float("inf")
        return self.focal_px * real_height_m / height_px


def guess_intrinsics(width: int, height: int, hfov_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """按水平视场角估算内参（假设方像素、主点居中）。"""
    focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    K = np.array([[focal, 0.0, width / 2.0],
                  [0.0, focal, height / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, np.zeros(5, dtype=np.float64)


def load_calibration(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取 JSON / YAML 标定文件，支持 OpenCV 键名或 fx/fy/cx/cy 简写。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"标定文件不存在: {path}")
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        import yaml  # ultralytics 已依赖
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

    if "camera_matrix" in data:
        K = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
    elif {"fx", "fy", "cx", "cy"} <= set(data):
        K = np.array([[data["fx"], 0.0, data["cx"]],
                      [0.0, data["fy"], data["cy"]],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    else:
        raise KeyError(f"{path} 中缺少 camera_matrix 或 fx/fy/cx/cy")
    coeffs = data.get("dist_coeffs", data.get("distortion_coefficients", [0.0] * 5))
    dist = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    if dist.size < 5:
        dist = np.pad(dist, (0, 5 - dist.size))
    return K, dist[:5]


def build_camera(width: int, height: int, hfov_deg: float = 60.0,
                 calib_path: Optional[str] = None,
                 focal_px: Optional[float] = None) -> Camera:
    """构造相机模型。

    优先级：``calib_path`` > ``focal_px``（直接给焦距像素值）> ``hfov_deg``（按视场角估算）。
    注意：**绝对距离与焦距成正比**，未标定时距离只在该视场角假设下成立。
    """
    if calib_path:
        K, dist = load_calibration(calib_path)
    elif focal_px:
        K = np.array([[focal_px, 0.0, width / 2.0],
                      [0.0, focal_px, height / 2.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
    else:
        K, dist = guess_intrinsics(width, height, hfov_deg)
    return Camera(K=K, dist=dist, width=width, height=height)


# ------------------------------------------------------------------ 位姿解算 --
@dataclass
class ArmorPose:
    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    distance_m: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    bearing_deg: float
    reproj_error_px: float
    distance_from_height_m: float

    @property
    def center_cam(self) -> np.ndarray:
        return self.tvec.reshape(3)


def order_quad(corners: np.ndarray) -> np.ndarray:
    """把任意顺序的 4 角点整理成 [左上, 右上, 右下, 左下]。

    装甲板总是"宽 > 高"，所以先在**相邻边**里找最长的那条作为上下边（注意不能
    拿对角线比——对角线和边长是一个量级，会比边长还长），再取中点 y 较小的那条
    作为上边；上边两点按 x 分左右，剩下两点按离左上角的远近分左下 / 右下。
    """
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    box = cv2.boxPoints(cv2.minAreaRect(points)).astype(np.float64)   # 保证是环序

    edges = [float(np.linalg.norm(box[(i + 1) % 4] - box[i])) for i in range(4)]
    start = int(np.argmax(edges))              # 最长相邻边 = 装甲板宽边
    opposite = (start + 2) % 4

    def mid_y(k: int) -> float:
        return float((box[k][1] + box[(k + 1) % 4][1]) * 0.5)

    if mid_y(opposite) < mid_y(start):         # 取中点更靠上的那条长边
        start = opposite
    corner_a, corner_b = box[start], box[(start + 1) % 4]
    top_left, top_right = ((corner_a, corner_b) if corner_a[0] <= corner_b[0]
                           else (corner_b, corner_a))

    rest = [p for k, p in enumerate(box) if k not in (start, (start + 1) % 4)]
    rest.sort(key=lambda p: float(np.linalg.norm(p - top_left)))
    bottom_left, bottom_right = rest
    return np.stack([top_left, top_right, bottom_right, bottom_left]).astype(np.float64)


def object_points(size: str) -> np.ndarray:
    """装甲板正面四角点（物体坐标系），顺序 [左上, 右上, 右下, 左下]。

    y 轴向下（与图像一致），z 轴由 x × y 决定、指向板内；因此板面外法线（朝相机）是 ``-R[:, 2]``。
    """
    half_w = ARMOR_WIDTH_M[size] / 2.0
    half_h = ARMOR_HEIGHT_M / 2.0
    return np.array([[-half_w, -half_h, 0.0],
                     [half_w, -half_h, 0.0],
                     [half_w, half_h, 0.0],
                     [-half_w, half_h, 0.0]], dtype=np.float64)


def mean_vertical_edge(quad_px: np.ndarray) -> float:
    """左右两条竖边的平均长度（≈ 装甲板像素高度）。"""
    quad = np.asarray(quad_px, dtype=np.float64).reshape(4, 2)
    return float((np.linalg.norm(quad[0] - quad[3]) + np.linalg.norm(quad[1] - quad[2])) / 2.0)


def _reproj_error(obj, rvec, tvec, K, dist, image_pts) -> float:
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    diff = projected.reshape(-1, 2) - image_pts.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def solve_pose(quad_px: np.ndarray, size: str, camera: Camera) -> Optional[ArmorPose]:
    """求解装甲板位姿；``quad_px`` 为 [左上, 右上, 右下, 左下] 四个像素点。"""
    image_pts = np.asarray(quad_px, dtype=np.float64).reshape(4, 1, 2)
    obj = object_points(size)

    rvec = tvec = None
    try:
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, image_pts, camera.K, camera.dist,
                                                 flags=cv2.SOLVEPNP_IPPE)
        candidates = []
        for candidate_r, candidate_t in zip(rvecs or [], tvecs or []):
            candidate_r = np.asarray(candidate_r, dtype=np.float64).reshape(3)
            candidate_t = np.asarray(candidate_t, dtype=np.float64).reshape(3)
            if not (np.all(np.isfinite(candidate_r)) and np.all(np.isfinite(candidate_t))):
                continue                          # IPPE 偶尔会给出 NaN 解，直接丢
            rotation, _ = cv2.Rodrigues(candidate_r)
            error = _reproj_error(obj, candidate_r, candidate_t, camera.K, camera.dist,
                                  image_pts)
            if not np.isfinite(error):
                continue
            facing = float(rotation[2, 2]) > 0.0  # 板内 z 轴指向前方 = 正面朝相机
            candidates.append((not facing, error, candidate_r, candidate_t))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]))
            rvec, tvec = candidates[0][2], candidates[0][3]
    except cv2.error:
        rvec = tvec = None

    if rvec is None:                              # 退化 / 全部非法时的兜底
        ok, rvec_iter, tvec_iter = cv2.solvePnP(obj, image_pts, camera.K, camera.dist,
                                                flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        rvec, tvec = rvec_iter.reshape(3), tvec_iter.reshape(3)
    if not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))):
        return None

    rotation, _ = cv2.Rodrigues(rvec)
    normal = -rotation[:, 2]                          # 板面外法线（朝相机）
    x_axis = rotation[:, 0].reshape(3)
    return ArmorPose(
        rvec=rvec,
        tvec=tvec,
        rotation=rotation,
        distance_m=float(np.linalg.norm(tvec)),
        yaw_deg=math.degrees(math.atan2(float(-normal[0]), float(-normal[2]))),
        pitch_deg=math.degrees(math.asin(float(np.clip(normal[1], -1.0, 1.0)))),
        roll_deg=math.degrees(math.atan2(float(x_axis[1]), float(x_axis[0]))),
        bearing_deg=math.degrees(math.atan2(float(tvec[0]), float(tvec[2]))),
        reproj_error_px=_reproj_error(obj, rvec, tvec, camera.K, camera.dist, image_pts),
        distance_from_height_m=camera.distance_from_height(mean_vertical_edge(quad_px),
                                                           ARMOR_HEIGHT_M),
    )


def project_axes(pose: ArmorPose, camera: Camera, length_m: float = 0.08) -> np.ndarray:
    """把装甲板坐标系三轴投影到图像上，返回 (4, 2) 像素点 [原点, X, Y, Z]。"""
    axes = np.array([[0, 0, 0], [length_m, 0, 0], [0, length_m, 0], [0, 0, -length_m]],
                    dtype=np.float64)
    points, _ = cv2.projectPoints(axes, pose.rvec, pose.tvec, camera.K, camera.dist)
    return points.reshape(-1, 2)
