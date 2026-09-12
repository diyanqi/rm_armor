"""RM 装甲板检测：由灯带配对得到可攻击的装甲板位置与朝向。"""

from .armor import Armor
from .camera import Camera, build_camera, guess_intrinsics, load_calibration
from .config import (
    ARMOR_HEIGHT_M,
    ARMOR_WIDTH_M,
    ArmorColor,
    ArmorSize,
    CameraConfig,
    ColorParams,
    DetectorConfig,
    LightBarParams,
    PairParams,
)
from .detector import ArmorDetector, DetectionResult
from .lightbar import LightBar, detect_light_bars
from .matcher import match_armors
from .pose import ArmorPose, object_points, solve_armor_pose
from .visualize import draw_result

__all__ = [
    "ARMOR_HEIGHT_M",
    "ARMOR_WIDTH_M",
    "Armor",
    "ArmorColor",
    "ArmorDetector",
    "ArmorPose",
    "ArmorSize",
    "Camera",
    "CameraConfig",
    "ColorParams",
    "DetectionResult",
    "DetectorConfig",
    "LightBar",
    "LightBarParams",
    "PairParams",
    "build_camera",
    "detect_light_bars",
    "draw_result",
    "guess_intrinsics",
    "load_calibration",
    "match_armors",
    "object_points",
    "solve_armor_pose",
]

__version__ = "0.1.0"
