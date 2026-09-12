"""检测流水线：颜色分割 -> 灯带提取 -> 配对 -> 位姿解算。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .armor import Armor
from .camera import Camera, build_camera
from .color import build_masks
from .config import ARMOR_WIDTH_M, ArmorColor, DetectorConfig
from .lightbar import LightBar, detect_light_bars
from .matcher import match_armors
from .pose import solve_armor_pose


@dataclass
class DetectionResult:
    """一帧图像的检测结果，同时保留中间量以便调试可视化。"""

    armors: List[Armor] = field(default_factory=list)
    light_bars: List[LightBar] = field(default_factory=list)
    masks: Dict[ArmorColor, np.ndarray] = field(default_factory=dict)
    gray: Optional[np.ndarray] = None
    camera: Optional[Camera] = None
    image_size: Tuple[int, int] = (0, 0)
    # 各阶段耗时（毫秒），键为 lightbar / match / pose / masks / total
    timings: Dict[str, float] = field(default_factory=dict)


class ArmorDetector:
    """装甲板检测器。配置与相机模型可复用，适合逐帧调用。"""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        self._camera: Optional[Camera] = None

    # ------------------------------------------------------------------ 相机 --
    @property
    def camera(self) -> Optional[Camera]:
        return self._camera

    def _resolve_camera(self, image: np.ndarray) -> Camera:
        height, width = image.shape[:2]
        if self._camera is None or (self._camera.width, self._camera.height) != (width, height):
            self._camera = build_camera(self.config.camera, width, height)
        return self._camera

    # ------------------------------------------------------------------ 检测 --
    def detect(self, bgr: np.ndarray) -> DetectionResult:
        """对单帧 BGR 图像做完整检测。"""
        started = time.perf_counter()
        camera = self._resolve_camera(bgr)
        image = camera.undistort(bgr) if self.config.undistort else bgr
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mark = time.perf_counter()
        light_bars = detect_light_bars(image, self.config, hsv, gray)
        t_lightbar = time.perf_counter()
        armors = match_armors(light_bars, gray, hsv[:, :, 1], self.config)
        t_match = time.perf_counter()
        for armor in armors:
            armor.pose = solve_armor_pose(armor.quad, armor.size, camera)
        t_pose = time.perf_counter()
        masks = build_masks(image, self.config.colors, self.config.color, hsv)
        t_masks = time.perf_counter()

        height, width = image.shape[:2]
        return DetectionResult(
            armors=armors,
            light_bars=light_bars,
            masks=masks,
            gray=gray,
            camera=camera,
            image_size=(width, height),
            timings={
                "preprocess": (mark - started) * 1e3,
                "lightbar": (t_lightbar - mark) * 1e3,
                "match": (t_match - t_lightbar) * 1e3,
                "pose": (t_pose - t_match) * 1e3,
                "masks": (t_masks - t_pose) * 1e3,
                "total": (t_masks - started) * 1e3,
            },
        )

    def detect_pose(self, bgr: np.ndarray) -> List[Armor]:
        """只关心装甲板（含位姿）时的便捷入口。"""
        return self.detect(bgr).armors

    # ------------------------------------------------------------------ 辅助 --
    @staticmethod
    def armor_width_m(armor: Armor) -> float:
        return ARMOR_WIDTH_M[armor.size]
