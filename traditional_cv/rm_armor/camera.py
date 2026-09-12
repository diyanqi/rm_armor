"""相机模型：内参估算 / 标定文件加载 / 参数随分辨率缩放。"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import CameraConfig


@dataclass
class Camera:
    """已解析的相机模型（针对某一确定分辨率）。"""

    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int

    @property
    def focal_px(self) -> float:
        return float(self.K[0, 0])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    def undistort(self, image: np.ndarray) -> np.ndarray:
        if not np.any(self.dist):
            return image
        return cv2.undistort(image, self.K, self.dist)

    def distance_from_height(self, height_px: float, real_height_m: float) -> float:
        """由“已知真实高度”反推深度：z = f * H / h_px。用于交叉验证 PnP 结果。"""
        if height_px <= 1e-6:
            return float("inf")
        return self.focal_px * real_height_m / height_px

    def scaled(self, width: int, height: int) -> "Camera":
        """把内参缩放到另一分辨率（假设同一视场角）。"""
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        K = self.K.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        return Camera(K=K, dist=self.dist.copy(), width=width, height=height)


def guess_intrinsics(width: int, height: int, hfov_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """按水平视场角估算内参（假设方形像素、主点居中）。"""
    focal = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    K = np.array(
        [[focal, 0.0, width / 2.0],
         [0.0, focal, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K, np.zeros(5, dtype=np.float64)


def load_calibration(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """从 JSON / YAML 标定文件读取内参。

    支持 OpenCV 风格的键名（``camera_matrix`` / ``dist_coeffs`` / ``image_width``
    等）以及自定义的 ``fx/fy/cx/cy`` 简写。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"标定文件不存在: {path}")

    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    else:
        data = _load_yaml(path)

    if "camera_matrix" in data:
        K = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
    elif {"fx", "fy", "cx", "cy"} <= set(data):
        K = np.array(
            [[data["fx"], 0.0, data["cx"]],
             [0.0, data["fy"], data["cy"]],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    else:
        raise KeyError(f"{path} 中缺少 camera_matrix 或 fx/fy/cx/cy")

    coeffs = data.get("dist_coeffs", data.get("distortion_coefficients", [0.0] * 5))
    dist = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    if dist.size < 5:
        dist = np.pad(dist, (0, 5 - dist.size))
    return K, dist[:5]


def _load_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise ImportError("读取 YAML 标定文件需要安装 PyYAML: pip install pyyaml") from exc

    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def build_camera(cfg: CameraConfig, width: int, height: int) -> Camera:
    """按配置与图像尺寸构造相机模型。"""
    if cfg.calibration_file:
        K, dist = load_calibration(cfg.calibration_file)
    else:
        K, dist = guess_intrinsics(width, height, cfg.hfov_deg)

    camera = Camera(K=K, dist=dist, width=width, height=height)

    # 标定通常在某一个分辨率下完成，这里按实际输入自动缩放。
    if cfg.calibration_file:
        calib_w = getattr(cfg, "calib_width", None)
        calib_h = getattr(cfg, "calib_height", None)
        if calib_w and calib_h and (calib_w, calib_h) != (width, height):
            camera = camera.scaled(int(calib_w), int(calib_h)).scaled(width, height)
    return camera


def resolve_camera(cfg: CameraConfig, image_size: Tuple[int, int],
                   cached: Optional[Camera] = None) -> Camera:
    """复用已构造的相机模型，避免每帧重复解析标定文件。"""
    width, height = image_size
    if cached is not None and (cached.width, cached.height) == (width, height):
        return cached
    return build_camera(cfg, width, height)
