"""装甲板位姿解算：由四角点 + 已知真实尺寸求位置与朝向。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .camera import Camera
from .config import ARMOR_HEIGHT_M, ARMOR_WIDTH_M, ArmorSize


@dataclass
class ArmorPose:
    """装甲板相对相机坐标系的位姿（相机系：x 右, y 下, z 前）。

    角度约定（板面法线由板内指向相机）：

    - ``yaw_deg``：0 表示板面正对相机；为正表示法线偏向画面右侧，
      因此正对相机的装甲板满足 ``yaw ≈ bearing``。
    - ``pitch_deg``：0 表示板面竖直；为正表示法线朝下（相机位于装甲板下方）。
    - ``roll_deg``：板面在图像内的自转。
    """

    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray          # (3, 3)
    distance_m: float             # 装甲板中心到光心的距离
    yaw_deg: float                # 水平朝向角，正对相机时为 0
    pitch_deg: float              # 板面俯仰角
    roll_deg: float               # 绕法线的自转（图像内倾斜）
    bearing_deg: float            # 装甲板中心相对光轴的方位角
    reproj_error_px: float
    distance_from_height_m: float  # 由灯带像素高度反推的深度，用于交叉验证

    @property
    def center_cam(self) -> np.ndarray:
        return self.tvec.reshape(3)


def object_points(size: ArmorSize) -> np.ndarray:
    """装甲板正面的四角点，顺序为 [左上, 右上, 右下, 左下]（与图像点对应）。

    坐标系：x 向右、y 向下（与图像一致），z 由 x × y 决定，指向板内（背离观察者）。
    因此“板面外法线（朝向相机）”是 ``-R[:, 2]``。
    """
    w = ARMOR_WIDTH_M[size] / 2.0
    h = ARMOR_HEIGHT_M / 2.0
    return np.array(
        [[-w, -h, 0.0],
         [w, -h, 0.0],
         [w, h, 0.0],
         [-w, h, 0.0]],
        dtype=np.float64,
    )


def _reprojection_error(obj: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                        K: np.ndarray, dist: np.ndarray, image_pts: np.ndarray) -> float:
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    diff = projected.reshape(-1, 2) - image_pts.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def _pick_facing_solution(obj: np.ndarray, image_pts: np.ndarray, K: np.ndarray,
                          dist: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """求解平面靶标位姿，并挑出“板面朝向相机”的那个解。

    装甲板是左右对称的平面矩形，仅凭四个角点无法区分正反面：IPPE 会给出两个
    投影误差几乎相同的解，其中一个把板面法线指向背面。实际比赛中我们只能看到
    灯带发光的正面，因此按“物体 z 轴（指向板内）朝向相机”来剔除镜像解。
    """
    retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj, image_pts, K, dist, flags=cv2.SOLVEPNP_IPPE
    )
    candidates = []
    for rvec, tvec in zip(rvecs or [], tvecs or []):
        rvec = np.asarray(rvec).reshape(3)
        tvec = np.asarray(tvec).reshape(3)
        rotation, _ = cv2.Rodrigues(rvec)
        error = _reprojection_error(obj, rvec, tvec, K, dist, image_pts)
        facing = float(rotation[2, 2]) > 0.0  # 板内 z 轴指向光心前方 = 正面朝向相机
        candidates.append((not facing, error, rvec, tvec))

    if not candidates:
        return None
    # 先取朝向相机的解，再看重投影误差。
    candidates.sort(key=lambda c: (c[0], c[1]))
    return candidates[0][2], candidates[0][3]


def solve_armor_pose(quad_px: np.ndarray, size: ArmorSize, camera: Camera) -> Optional[ArmorPose]:
    """求解装甲板位姿。

    ``quad_px`` 为 ``[左上, 右上, 右下, 左下]`` 四个图像点（两灯带外沿构成的矩形）。
    """
    image_pts = np.asarray(quad_px, dtype=np.float64).reshape(4, 1, 2)
    obj = object_points(size)

    picked = _pick_facing_solution(obj, image_pts, camera.K, camera.dist)
    if picked is None:
        ok, rvec, tvec = cv2.solvePnP(
            obj, image_pts, camera.K, camera.dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None
        picked = (rvec.reshape(3), tvec.reshape(3))

    rvec, tvec = picked
    rotation, _ = cv2.Rodrigues(rvec)
    normal = -rotation[:, 2]  # 板面外法线（朝向相机）
    x_axis = rotation[:, 0].reshape(3)

    distance = float(np.linalg.norm(tvec))
    distance_from_height = camera.distance_from_height(mean_vertical_edge(quad_px), ARMOR_HEIGHT_M)

    return ArmorPose(
        rvec=rvec,
        tvec=tvec,
        rotation=rotation,
        distance_m=distance,
        yaw_deg=math.degrees(math.atan2(float(-normal[0]), float(-normal[2]))),
        pitch_deg=math.degrees(math.asin(float(np.clip(normal[1], -1.0, 1.0)))),
        roll_deg=math.degrees(math.atan2(float(x_axis[1]), float(x_axis[0]))),
        bearing_deg=math.degrees(math.atan2(float(tvec[0]), float(tvec[2]))),
        reproj_error_px=_reprojection_error(obj, rvec, tvec, camera.K, camera.dist, image_pts),
        distance_from_height_m=distance_from_height,
    )


def mean_vertical_edge(quad_px: np.ndarray) -> float:
    """四角点中左右两条竖边的平均长度，即灯带在图像中的高度。"""
    quad = np.asarray(quad_px, dtype=np.float64).reshape(4, 2)
    left = np.linalg.norm(quad[0] - quad[3])
    right = np.linalg.norm(quad[1] - quad[2])
    return float((left + right) / 2.0)


def mean_horizontal_edge(quad_px: np.ndarray) -> float:
    """四角点中上下两条横边的平均长度，即两灯带外沿间距。"""
    quad = np.asarray(quad_px, dtype=np.float64).reshape(4, 2)
    top = np.linalg.norm(quad[0] - quad[1])
    bottom = np.linalg.norm(quad[3] - quad[2])
    return float((top + bottom) / 2.0)
