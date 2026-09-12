"""交互式装甲板检测查看器：弹窗浏览图片，随时重新识别并查看标注信息。

用法::

    python viewer.py --source images
    python viewer.py --source images/frame_003156.jpg
    python viewer.py --video autoaim_all.mp4 --start 2400

按键::

    n / d / 右方向键   下一张
    p / a / 左方向键   上一张
    space              切换“是否运行识别”（先看原图 / 看识别结果）
    r                  重新识别当前帧
    [ / ]              配对得分门限 -/+ 0.05（自动重识别）
    , / .              灯带亮度门限 -/+ 5（自动重识别）
    b                  切换灯带边框
    x                  切换位姿坐标轴
    i                  切换板面内部区域
    m                  切换颜色掩膜视图
    f                  切换“适应窗口 / 原始尺寸”
    s                  保存当前标注图到 --save-dir
    h                  切换按键提示
    q / ESC            退出
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rm_armor import ArmorDetector, ArmorColor, DetectorConfig
from rm_armor.detector import DetectionResult
from rm_armor.visualize import draw_key_hints, draw_result, draw_text_block, render_mask
from utils.files import collect_images, natural_key

WINDOW = "RM Armor Viewer"

KEY_HINTS = ("n/p prev-next  space detect on/off  r redetect  [/] score  ,/. brightness  "
             "b bars  k bar-info  x axes  i interior  m masks  f fit  s save  h help  q quit")

HINT_LEGEND = [
    "green box  = matched armor (attackable plate)",
    "thin bars  = light bars (orange-red / blue only)",
    "cyan box   = plate face used for dark+digit check",
    "RGB axes   = armor frame X/Y/Z",
    "pos        = armor center in camera frame (x right, y down, z fwd), m",
    "kappa      = white digit width / plate height (cross-plate guard)",
]


# ------------------------------------------------------------------ 视图状态 --
@dataclass
class ViewState:
    """当前显示开关与缩放。"""

    detect: bool = True
    show_bars: bool = True
    show_bar_metrics: bool = False
    show_axes: bool = True
    show_interior: bool = False
    show_masks: bool = False
    fit_window: bool = True
    help_visible: bool = True
    zoom: float = 1.0


class FrameCache:
    """按（文件, 门限）缓存检测结果，来回翻页时不必重复计算。"""

    def __init__(self, detector: ArmorDetector, capacity: int = 24) -> None:
        self.detector = detector
        self.capacity = capacity
        self._store: Dict[Tuple[str, float, float], DetectionResult] = {}
        self._order: List[Tuple[str, float, float]] = []

    def key(self, path: str) -> Tuple[str, float, float]:
        return (path, self.detector.config.pair.min_score,
                self.detector.config.pair.min_bar_brightness)

    def get(self, path: str, image: np.ndarray, use_cache: bool = True) -> DetectionResult:
        key = self.key(path)
        if use_cache and key in self._store:
            return self._store[key]
        result = self.detector.detect(image)
        self._store[key] = result
        self._order.append(key)
        while len(self._order) > self.capacity:
            self._store.pop(self._order.pop(0), None)
        return result

    def invalidate(self) -> None:
        self._store.clear()
        self._order.clear()


# ------------------------------------------------------------------ 画面合成 --
def compose(result: Optional[DetectionResult], image: np.ndarray,
            state: ViewState, header: str) -> np.ndarray:
    """合成待显示的画面。``result`` 为 None 时只显示原图。"""
    if result is None:
        canvas = image.copy()
        _banner(canvas, "detection OFF  (press space to detect)")
        return canvas

    canvas = draw_result(image, result, None,
                         show_light_bars=state.show_bars,
                         show_pose_axes=state.show_axes,
                         show_interior=state.show_interior,
                         show_panel=True, header=header,
                         show_bar_metrics=state.show_bar_metrics)
    if state.help_visible:
        _legend(canvas)
    if state.show_masks:
        canvas = _append_masks(canvas, result)
    return canvas


def _banner(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(image, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (120, 255, 120), 1, cv2.LINE_AA)


def _legend(image: np.ndarray) -> None:
    """右上角画图例说明，让标注含义一目了然。"""
    draw_text_block(image, HINT_LEGEND, (image.shape[1], 8),
                    max_width=int(image.shape[1] * 0.42), align_right=True)


def _append_masks(canvas: np.ndarray, result: DetectionResult) -> np.ndarray:
    """在原图右侧并排拼接红/蓝颜色掩膜，并标注每列内容。"""
    height, width = canvas.shape[:2]
    tiles = [canvas]
    for color in (ArmorColor.RED, ArmorColor.BLUE):
        mask = result.masks.get(color)
        if mask is None:
            continue
        tile = render_mask(mask, (width, height))
        cv2.rectangle(tile, (0, 0), (width, 30), (20, 20, 20), -1)
        cv2.putText(tile, f"{color.value.upper()} mask", (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 1, cv2.LINE_AA)
        tiles.append(tile)
    return np.hstack(tiles)


def fit_to_box(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """等比缩放到不超过给定尺寸；本来就够小则原样返回。"""
    height, width = image.shape[:2]
    scale = min(max_w / width, max_h / height, 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                      interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------ 主循环 --
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="交互式装甲板检测查看器")
    parser.add_argument("--source", default="images", help="图片文件或图片目录")
    parser.add_argument("--video", default=None, help="改为浏览视频帧")
    parser.add_argument("--start", type=int, default=0, help="视频起始帧")
    parser.add_argument("--calib", default=None, help="相机标定文件（JSON/YAML）")
    parser.add_argument("--hfov", type=float, default=60.0, help="未标定时的水平视场角（度）")
    parser.add_argument("--max-score", type=float, default=None, help="初始配对得分门限")
    parser.add_argument("--no-detect", action="store_true", help="启动时不运行识别")
    parser.add_argument("--save-dir", default="viewer_out", help="按 s 保存标注图的目录")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> DetectorConfig:
    config = DetectorConfig()
    config.camera.hfov_deg = args.hfov
    config.camera.calibration_file = args.calib
    if args.max_score is not None:
        config.pair.min_score = args.max_score
    return config


# ------------------------------------------------------------------ 查看器 --
class Viewer:
    """在窗口中浏览图片/视频帧，并对当前帧运行检测与标注。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.detector = ArmorDetector(build_config(args))
        self.cache = FrameCache(self.detector)
        self.state = ViewState(detect=not args.no_detect)

        self.capture = None
        self.paths: List[str] = []
        if args.video:
            self.capture = cv2.VideoCapture(args.video)
            if not self.capture.isOpened():
                raise RuntimeError(f"无法打开视频: {args.video}")
            self.index = args.start
        else:
            self.paths = collect_images(args.source)
            if not self.paths:
                raise RuntimeError(f"找不到图片: {args.source}")
            self.index = 0

        self.image: Optional[np.ndarray] = None
        self.result: Optional[DetectionResult] = None
        self._logged: Optional[Tuple[str, float]] = None

    # ------------------------------------------------------------- 基本属性 --
    @property
    def total(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) if self.capture else len(self.paths)

    @property
    def current_path(self) -> str:
        if self.capture:
            return f"{self.args.video}#{self.index}"
        return self.paths[self.index]

    # ------------------------------------------------------------- 帧的读取 --
    def load(self) -> bool:
        my_result = None
        if self.capture:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.index)
            ok, frame = self.capture.read()
            self.image = frame if ok else None
        else:
            self.image = cv2.imread(self.paths[self.index])
        if self.image is None:
            return False
        self.result = self.cache.get(self.current_path, self.image) if self.state.detect else None
        return True

    def step(self, delta: int) -> bool:
        target = self.index + delta
        if not 0 <= target < self.total:
            return False
        self.index = target
        return True

    # --------------------------------------------------------------- 重新识别 --
    def redetect(self, invalidate: bool = False) -> None:
        if invalidate:
            self.cache.invalidate()
        if self.state.detect and self.image is not None:
            self.result = self.cache.get(self.current_path, self.image, use_cache=not invalidate)

    def tune_score(self, delta: float) -> None:
        pair = self.detector.config.pair
        pair.min_score = float(np.clip(round(pair.min_score + delta, 2), 0.0, 1.0))
        self.redetect(invalidate=True)

    def tune_brightness(self, delta: float) -> None:
        pair = self.detector.config.pair
        pair.min_bar_brightness = float(max(0.0, round(pair.min_bar_brightness + delta, 1)))
        self.redetect(invalidate=True)

    # ----------------------------------------------------------------- 显示 --
    def header(self) -> str:
        name = os.path.basename(self.current_path.rsplit("#", 1)[0])
        text = f"{name}  [{self.index + 1}/{self.total}]"
        if self.state.detect:
            pair = self.detector.config.pair
            text += f"  score>={pair.min_score:.2f}  bar_bri>={pair.min_bar_brightness:.0f}"
        return text

    def draw(self) -> np.ndarray:
        canvas = compose(self.result, self.image, self.state, self.header())
        if self.state.help_visible:
            draw_key_hints(canvas, KEY_HINTS)
        return _scale(canvas, self.state)

    def save(self) -> None:
        os.makedirs(self.args.save_dir, exist_ok=True)
        stem = re.sub(r"[^\w.-]+", "_", os.path.basename(self.current_path.rsplit("#", 1)[0]))
        out = os.path.join(self.args.save_dir, stem or "frame")
        if not os.path.splitext(out)[1]:
            out += ".jpg"
        cv2.imwrite(out, compose(self.result, self.image, self.state, self.header()))
        print(f"[save] {out}")

    # ----------------------------------------------------------------- 主循环 --
    def run(self) -> int:
        if not self.load():
            print("[error] 无法读取首帧", file=sys.stderr)
            return 1

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 760)

        while True:
            cv2.imshow(WINDOW, self.draw())
            self.log_once()

            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            if not self.handle_key(key):
                continue
            if not self.load():
                print("[warn] 已到末尾或读取失败")
                self.index = max(0, min(self.index, self.total - 1))

        if self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()
        return 0

    def log_once(self) -> None:
        """同一帧只在结果变化时打印一次，避免刷屏。"""
        signature = (self.current_path, self.result.timings.get("total", 0.0)
                     if self.result else -1.0)
        if signature == self._logged:
            return
        self._logged = signature
        _log(self.current_path, self.result)

    def handle_key(self, key: int) -> bool:
        """处理按键；返回 True 表示需要刷新画面。"""
        state = self.state
        if key in (ord("n"), ord("d"), 83, 3):
            return self.step(1)
        if key in (ord("p"), ord("a"), 81, 2):
            return self.step(-1)
        if key == ord(" "):
            state.detect = not state.detect
            self.result = None
            if state.detect and self.image is not None:
                self.redetect()
            return True
        if key == ord("r"):
            self.redetect(invalidate=True)
        elif key == ord("["):
            self.tune_score(-0.05)
        elif key == ord("]"):
            self.tune_score(0.05)
        elif key == ord(","):
            self.tune_brightness(-5.0)
        elif key == ord("."):
            self.tune_brightness(5.0)
        elif key == ord("b"):
            state.show_bars = not state.show_bars
        elif key == ord("k"):
            state.show_bar_metrics = not state.show_bar_metrics
        elif key == ord("x"):
            state.show_axes = not state.show_axes
        elif key == ord("i"):
            state.show_interior = not state.show_interior
        elif key == ord("m"):
            state.show_masks = not state.show_masks
        elif key == ord("f"):
            state.fit_window = not state.fit_window
        elif key == ord("h"):
            state.help_visible = not state.help_visible
        elif key in (ord("+"), ord("=")):
            state.zoom = min(state.zoom * 1.25, 4.0)
        elif key in (ord("-"), ord("_")):
            state.zoom = max(state.zoom / 1.25, 0.25)
        elif key == ord("s"):
            self.save()
        return True


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        viewer = Viewer(args)
    except RuntimeError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1
    return viewer.run()


# ------------------------------------------------------------------ 辅助函数 --
def _render(window: str, image: np.ndarray, result: Optional[DetectionResult],
            state: ViewState, header: str) -> None:
    canvas = compose(result, image, state, header)
    if state.help_visible:
        draw_key_hints(canvas, KEY_HINTS)
    cv2.imshow(window, _scale(canvas, state))


def _scale(canvas: np.ndarray, state: ViewState) -> np.ndarray:
    """按需缩放：适应窗口时限制到 1280x760，否则按 zoom 缩放。"""
    if state.zoom != 1.0:
        return cv2.resize(canvas, None, fx=state.zoom, fy=state.zoom,
                          interpolation=cv2.INTER_AREA if state.zoom < 1 else cv2.INTER_LINEAR)
    if state.fit_window:
        return fit_to_box(canvas, 1280, 760)
    return canvas


def _log(path: str, result: Optional[DetectionResult]) -> None:
    """把结果同时打到终端，方便复制数值。"""
    if result is None:
        print(f"{os.path.basename(path)}: 未识别")
        return
    total = result.timings.get("total", 0.0)
    print(f"{os.path.basename(path)}: {total:.1f} ms, bars={len(result.light_bars)}, "
          f"armors={len(result.armors)}")
    for index, armor in enumerate(result.armors, start=1):
        if armor.pose is None:
            print(f"   #{index} {armor.color.value}/{armor.size.value} score={armor.score:.2f} (无位姿)")
            continue
        pose = armor.pose
        print(f"   #{index} {armor.color.value}/{armor.size.value} score={armor.score:.2f} "
              f"dist={pose.distance_m:.3f}m yaw={pose.yaw_deg:+.1f} pitch={pose.pitch_deg:+.1f} "
              f"px=({armor.center[0]:.0f},{armor.center[1]:.0f}) reproj={pose.reproj_error_px:.2f}px")


if __name__ == "__main__":
    sys.exit(main())
