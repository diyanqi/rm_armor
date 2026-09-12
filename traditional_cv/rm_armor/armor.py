"""装甲板数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np

from .config import ArmorColor, ArmorSize
from .lightbar import LightBar

if TYPE_CHECKING:  # 避免与 pose 模块循环导入
    from .pose import ArmorPose


@dataclass
class PlateEvidence:
    """板面证据：黑底 + 居中的白色数字。

    ``kappa`` = 白字像素宽度 / 装甲板像素高度。白字的真实宽度与装甲板真实高度都是
    常数，因此 kappa 与偏航角、以及大/小装甲板都无关，可以稳定地区分“白字”与
    “跨板配对中间那条窄亮缝”。
    """

    dark_ratio: float = 0.0
    digit_ratio: float = 0.0        # 白字面积 / 板面面积
    kappa: float = 0.0
    digit_height_ratio: float = 0.0
    digit_offset: float = 1.0       # 白字中心相对板面中心的横向偏移（占板宽比例）
    contrast: float = 0.0
    evaluated: bool = False         # 灯带足够长、做过数字校验
    digit_present: bool = False

    def as_dict(self) -> Dict[str, float]:
        return {
            "dark_ratio": self.dark_ratio,
            "digit_ratio": self.digit_ratio,
            "kappa": self.kappa,
            "digit_height": self.digit_height_ratio,
            "digit_offset": self.digit_offset,
            "digit_present": float(self.digit_present),
        }


@dataclass
class Armor:
    """由两条灯带配对得到的一块装甲板。"""

    left: LightBar
    right: LightBar
    color: ArmorColor
    size: ArmorSize
    size_confidence: float
    quad: np.ndarray            # (4, 2) [左上, 右上, 右下, 左下]，即装甲板正面外沿
    width_px: float             # 两条灯带外沿间距
    height_px: float            # 灯带高度（≈ 装甲板高度）
    aspect: float               # width_px / height_px
    score: float
    sub_scores: Dict[str, float] = field(default_factory=dict)
    evidence: Optional[PlateEvidence] = None
    pose: Optional["ArmorPose"] = None

    @property
    def center(self) -> np.ndarray:
        """装甲板中心（四角点均值）。"""
        return self.quad.mean(axis=0)

    @property
    def light_bars(self) -> tuple[LightBar, LightBar]:
        return self.left, self.right

    def __str__(self) -> str:  # pragma: no cover - 仅用于打印
        return (
            f"Armor({self.color.value}/{self.size.value} "
            f"score={self.score:.2f} aspect={self.aspect:.2f})"
        )
