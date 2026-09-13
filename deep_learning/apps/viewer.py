"""交互式装甲板 OBB 查看器：弹窗浏览图片 / 视频标注，并可视化训练结果。

用法::

    # 图片：默认浏览验证集 + runs/ 下最新的 best.pt，并弹出训练曲线窗口
    python deep_learning/apps/viewer.py
    python deep_learning/apps/viewer.py --split train --limit 200

    # 视频：逐帧实时标注（用 --source 直接给 mp4 即可）
    python deep_learning/apps/viewer.py --source autoaim_all.mp4 --play
    python deep_learning/apps/viewer.py --source autoaim_all.mp4 --start 1000 --conf 0.3

每个检出目标都会用 PnP 解算位姿（按 --hfov 估算内参，或用 --calib 加载标定文件），
在框上叠加距离（米）、坐标轴，并在左上角面板列出 dist / yaw / pitch / x-y-z / 重投影误差。

图片按键::

    n / d / 右方向键   下一张        p / a / 左方向键   上一张
    , / .              前/后跳 --jump 张
    space              切换"是否运行检测"（看原图 / 看预测）
    z                  切换 PnP 位姿（距离/角度 + 坐标轴）
    g                  切换真值标注（GT）        c  切换置信度数值
    v                  切换训练曲线窗口          f  切换"适应窗口 / 原始尺寸"
    s                  保存当前标注图            h  切换按键提示        q / ESC 退出

视频按键::

    space              播放 / 暂停
    n / p              下一帧 / 上一帧（自动暂停）
    , / .              前/后跳 --jump 帧
    r                  回到开头
    x                  切换"是否运行检测"
    z / g / c / v / f / s / h / q   同上
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from armor_det import pose as armor_pose  # noqa: E402
from armor_det.config import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_OUT,
    DEFAULT_RUNS,
    NC_VEHICLE,
    REPO_ROOT,
    class_color_bgr,
)
from utils.files import collect_images, is_video, stem  # noqa: E402

WINDOW_IMAGE = "RM Armor OBB Viewer"
WINDOW_CURVES = "Training Results"

KEY_HINTS_IMAGE = ("n/p prev-next  ,/. jump  space detect  z pose  g GT  c conf  "
                   "v curves  f fit  s save  h help  q quit")
KEY_HINTS_VIDEO = ("space play/pause  n/p frame  ,/. jump  r restart  x detect  z pose  "
                   "c conf  v curves  f fit  s save  h help  q quit")

DEFAULT_SOURCE = (os.path.join(DEFAULT_OUT, "images", "val")
                  if os.path.isdir(os.path.join(DEFAULT_OUT, "images", "val"))
                  else os.path.join(REPO_ROOT, "dataset2", "images"))


@dataclass
class ViewState:
    detect: bool = True
    show_gt: bool = True
    show_conf: bool = True
    show_pose: bool = True
    show_curves: bool = True
    fit_window: bool = True
    help_visible: bool = True


# ------------------------------------------------------------------ 辅助函数 --
def resolve_weights(runs_dir: str) -> Optional[str]:
    """在 runs/ 下查找最新的 best.pt。"""
    candidates = glob.glob(os.path.join(runs_dir, "**", "weights", "best.pt"), recursive=True)
    return max(candidates, key=os.path.getmtime) if candidates else None


def guess_labels_dir(images_dir: str) -> Optional[str]:
    """根据图片目录推断同名标签目录（built dataset 或原始数据集）。"""
    path = os.path.abspath(images_dir)
    candidates = (
        os.path.join(os.path.dirname(os.path.dirname(path)), "labels", os.path.basename(path)),
        os.path.join(os.path.dirname(path), "labels"),
    )
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def read_gt(label_path: str, width: int, height: int) -> List[Tuple[int, np.ndarray]]:
    """读取真值 OBB。

    9 列 = YOLO-OBB（已是 24 类组合 id）；
    10 列 = 原始 `color_id vehicle_id x1..y4`，按 `color*8+vehicle` 组合成 24 类。
    """
    boxes: List[Tuple[int, np.ndarray]] = []
    if not label_path or not os.path.exists(label_path):
        return boxes
    with open(label_path, encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.split()
            if len(parts) == 9:
                cls, coords = int(parts[0]), parts[1:9]
            elif len(parts) == 10:
                cls, coords = int(parts[0]) * NC_VEHICLE + int(parts[1]), parts[2:10]
            else:
                continue
            points = np.array([float(v) for v in coords], dtype=np.float32).reshape(4, 2)
            points[:, 0] *= width
            points[:, 1] *= height
            boxes.append((cls, points))
    return boxes


def draw_poly(image: np.ndarray, points: np.ndarray, color, text: str,
              thickness: int = 2) -> None:
    quad = np.round(points).astype(np.int32)
    cv2.polylines(image, [quad], True, color, thickness)
    if text:
        x = max(int(points[:, 0].min()), 0)
        y = max(int(points[:, 1].min()) - 6, 14)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)


def plot_curves(run_dir: str, width: int = 1100, height: int = 700) -> Optional[np.ndarray]:
    """从 results.csv 用 OpenCV 画训练曲线（results.png 不存在时的兜底）。"""
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row]
    if not rows:
        return None

    panels = [("train/box_loss", "train box loss"), ("metrics/mAP50-95(B)", "mAP50-95"),
              ("metrics/mAP50(B)", "mAP50"), ("metrics/precision(B)", "precision")]
    canvas = np.full((height, width, 3), 32, np.uint8)
    cols, row_count = 2, 2
    panel_w, panel_h = width // cols, (height - 50) // row_count
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(canvas, f"training curves  ({os.path.basename(run_dir)}, epochs={len(rows)})",
                (14, 32), font, 0.7, (140, 255, 140), 1, cv2.LINE_AA)
    for i, (key, title) in enumerate(panels):
        values = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except (KeyError, ValueError, TypeError):
                values.append(None)
        finite = [v for v in values if v is not None]

        ox, oy = (i % cols) * panel_w, 50 + (i // cols) * panel_h
        cv2.rectangle(canvas, (ox + 10, oy + 8), (ox + panel_w - 10, oy + panel_h - 10),
                      (48, 48, 48), -1)
        cv2.putText(canvas, title, (ox + 20, oy + 32), font, 0.55, (200, 255, 200), 1, cv2.LINE_AA)
        if len(finite) < 2:
            continue
        lo, hi = min(finite), max(finite)
        span = (hi - lo) or 1.0
        points = []
        for j, value in enumerate(values):
            if value is None:
                continue
            x = ox + 24 + int((panel_w - 48) * j / max(len(values) - 1, 1))
            y = oy + panel_h - 18 - int((panel_h - 70) * (value - lo) / span)
            points.append((x, y))
        cv2.polylines(canvas, [np.array(points, np.int32)], False, (80, 220, 255), 2)
        cv2.putText(canvas, f"{finite[-1]:.4g}", (ox + panel_w - 110, oy + 32),
                    font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def resolve_run_dir(runs_dir: str) -> str:
    """在 runs/ 下查找最近修改过的训练结果目录（含 results.csv）。"""
    candidates = glob.glob(os.path.join(runs_dir, "**", "results.csv"), recursive=True)
    return os.path.dirname(max(candidates, key=os.path.getmtime)) if candidates else runs_dir


def load_curves(run_dir: str) -> Optional[np.ndarray]:
    """优先用 ultralytics 生成的 results.png，缺失时用 results.csv 现画。"""
    png = os.path.join(run_dir, "results.png")
    if os.path.exists(png):
        image = cv2.imread(png)
        if image is not None:
            return image
    return plot_curves(run_dir)


def fit_to_box(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_w / width, max_h / height, 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                      interpolation=cv2.INTER_AREA)


def draw_panel(image: np.ndarray, lines: List[str], origin: Tuple[int, int] = (8, 40),
               font_scale: float = 0.44) -> None:
    """在图像上画一块半透明底的文字面板（用于显示各目标的位姿）。"""
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = int(font_scale * 30)
    box_w = max(int(len(line) * font_scale * 11.5) for line in lines) + 16
    box_h = line_h * len(lines) + 12
    x0, y0 = origin
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0.0, image)
    for row, text in enumerate(lines):
        y = y0 + line_h + row * line_h
        cv2.putText(image, text, (x0 + 8, y), font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (x0 + 8, y), font, font_scale,
                    (215, 255, 215) if row else (140, 255, 140), 1, cv2.LINE_AA)


def draw_axes(image: np.ndarray, points: np.ndarray) -> None:
    """画装甲板坐标系三轴：[原点, X, Y, Z]。"""
    origin = tuple(np.round(points[0]).astype(int))
    cv2.line(image, origin, tuple(np.round(points[1]).astype(int)), (0, 0, 255), 2)
    cv2.line(image, origin, tuple(np.round(points[2]).astype(int)), (0, 255, 0), 2)
    cv2.line(image, origin, tuple(np.round(points[3]).astype(int)), (255, 0, 0), 2)


# ------------------------------------------------------------------ 查看器 --
class Viewer:
    """在窗口中浏览图片 / 播放视频，并对当前帧做 OBB 检测与标注。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.source_path = args.video or args.source
        self.is_video = bool(args.video) or (os.path.isfile(self.source_path)
                                             and is_video(self.source_path))

        self.capture = None
        self.paths: List[str] = []
        self.fps = 0.0
        self._next_seq = -1

        if self.is_video:
            self.capture = cv2.VideoCapture(self.source_path)
            if not self.capture.isOpened():
                raise RuntimeError(f"无法打开视频: {self.source_path}")
            self.total = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            self.fps = float(self.capture.get(cv2.CAP_PROP_FPS)) or 30.0
        else:
            self.paths = (collect_images(args.source) if os.path.isdir(args.source)
                          else [args.source])
            if not self.paths:
                raise RuntimeError(f"找不到图片: {args.source}")
            if args.limit > 0:
                self.paths = self.paths[:args.limit]
            self.total = len(self.paths)

        self.index = min(max(args.start, 0), self.total - 1)
        self.playing = False
        self.state = ViewState(detect=not args.no_detect, show_pose=not args.no_pose)
        self.labels_dir = None if self.is_video else (
            args.labels or (guess_labels_dir(args.source) if os.path.isdir(args.source) else None))
        self._camera: Optional[armor_pose.Camera] = None
        self._camera_size: Tuple[int, int] = (0, 0)

        self.run_dir = args.run or (os.path.dirname(os.path.dirname(args.weights))
                                    if args.weights else resolve_run_dir(DEFAULT_RUNS))
        self.model = None
        if not args.no_detect:
            if not args.weights or not os.path.exists(args.weights):
                raise RuntimeError("未找到权重，请用 --weights 指定，或先训练模型")
            from ultralytics import YOLO
            self.model = YOLO(args.weights)
        self._cache: Dict[int, object] = {}
        self.curves = None if args.no_curves else load_curves(self.run_dir)

    # -------------------------------------------------------------- 帧的读取 --
    def read_frame(self) -> Optional[np.ndarray]:
        if self.capture is None:
            return cv2.imread(self.paths[self.index])
        if self.index != self._next_seq:            # 顺序播放时不必反复 seek
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.index)
        ok, frame = self.capture.read()
        self._next_seq = self.index + 1
        return frame if ok else None

    # -------------------------------------------------------------- 相机模型 --
    def camera_for(self, width: int, height: int) -> armor_pose.Camera:
        """按当前帧尺寸构造/复用相机模型（未标定时按 --hfov 估算）。"""
        if self._camera is None or self._camera_size != (width, height):
            self._camera = armor_pose.build_camera(width, height, self.args.hfov,
                                                   self.args.calib, self.args.focal)
            self._camera_size = (width, height)
        return self._camera

    def resolve_size(self, cls: int, quad: np.ndarray) -> str:
        """装甲板规格：按类别推断，或按观测宽高比推断。"""
        if self.args.size_from == "class":
            return armor_pose.armor_size_of(cls)
        width_px = float(np.linalg.norm(quad[0] - quad[1]))
        return armor_pose.size_from_aspect(width_px, armor_pose.mean_vertical_edge(quad))[0]

    # --------------------------------------------------------------- 推理 --
    def predict(self, frame: np.ndarray):
        if self.model is None:
            return None
        if self.is_video:                            # 视频逐帧播放，不做缓存
            return self.model.predict(frame, imgsz=self.args.imgsz, conf=self.args.conf,
                                      device=self.args.device, verbose=False)[0]
        if self.index not in self._cache:
            self._cache[self.index] = self.model.predict(
                frame, imgsz=self.args.imgsz, conf=self.args.conf,
                device=self.args.device, verbose=False)[0]
        return self._cache[self.index]

    # ----------------------------------------------------------------- 显示 --
    def header(self, infer_ms: float = 0.0) -> str:
        if self.is_video:
            text = (f"{os.path.basename(self.source_path)}  [{self.index + 1}/{self.total}]  "
                    f"{'PLAY' if self.playing else 'PAUSE'}")
            if infer_ms:
                text += f"  {infer_ms:.0f} ms/frame ({1000.0 / max(infer_ms, 1e-3):.0f} FPS)"
        else:
            text = f"{os.path.basename(self.paths[self.index])}  [{self.index + 1}/{self.total}]"
        if self.state.detect and self.model is not None:
            text += f"  conf>={self.args.conf:.2f}"
        if self.labels_dir and self.state.show_gt:
            text += "  GT:on"
        return text

    def draw(self, frame: Optional[np.ndarray]) -> np.ndarray:
        if frame is None:
            return np.full((480, 640, 3), 40, np.uint8)
        height, width = frame.shape[:2]
        canvas = frame.copy()

        if self.state.show_gt and self.labels_dir:
            label_path = os.path.join(self.labels_dir, stem(self.paths[self.index]) + ".txt")
            for cls, points in read_gt(label_path, width, height):
                draw_poly(canvas, points, class_color_bgr(cls),
                          f"GT {CLASS_NAMES.get(cls, cls)}", thickness=1)

        started = time.perf_counter()
        result = self.predict(frame) if self.state.detect else None
        infer_ms = (time.perf_counter() - started) * 1000.0

        camera = self.camera_for(width, height) if self.state.show_pose else None
        rows: List[Tuple[float, int, int, str, bool, float, object]] = []
        if result is not None and getattr(result, "obb", None) is not None and len(result.obb):
            boxes = result.obb.xyxyxyxy.cpu().numpy()
            classes = result.obb.cls.cpu().numpy().astype(int)
            confs = result.obb.conf.cpu().numpy()
            for index, (points, cls, conf) in enumerate(zip(boxes, classes, confs), start=1):
                quad = armor_pose.order_quad(points.reshape(4, 2))
                edges = [float(np.linalg.norm(quad[(k + 1) % 4] - quad[k])) for k in range(4)]
                observed = max(edges) / max(min(edges), 1e-6)

                text = CLASS_NAMES.get(cls, str(cls))
                if self.state.show_conf:
                    text += f" {conf:.2f}"

                pose = None
                size = ""
                ratio = 0.0
                reliable = True
                if camera is not None:
                    size = self.resolve_size(cls, quad)
                    # 框必须"像一块甲板"：观测宽高比要接近该规格的标称值，
                    # 否则（例如误检到灯条/半个板）PnP 会解出一个虚假的位姿。
                    ratio = observed / armor_pose.nominal_aspect(size)
                    reliable = ratio >= self.args.min_aspect
                    pose = armor_pose.solve_pose(quad, size, camera)
                    if pose is not None:
                        text = ("?" if not reliable else "") + text + f"  {pose.distance_m:.2f}m"
                        if reliable:
                            draw_axes(canvas, armor_pose.project_axes(pose, camera))

                draw_poly(canvas, quad,
                          class_color_bgr(cls) if reliable else (150, 150, 150), text)
                if pose is not None:
                    rows.append((pose.distance_m, index, cls, size, reliable, ratio, pose))

        if rows:
            rows.sort(key=lambda row: row[0])        # 由近到远
            panel = ["     target              size  AR    dist    yaw/pitch     x/y/z (m)     reproj"]
            for _, index, cls, size, reliable, ratio, pose in rows[:6]:
                x, y, z = pose.center_cam
                panel.append(f"{' ' if reliable else '!'}#{index} {CLASS_NAMES.get(cls, cls):<16} "
                             f"{size[0].upper()}  {ratio:4.2f}  {pose.distance_m:5.2f}m  "
                             f"{pose.yaw_deg:+5.0f}/{pose.pitch_deg:+4.0f}  "
                             f"({x:+.2f},{y:+.2f},{z:+.2f})  {pose.reproj_error_px:4.1f}px")
            panel.append("! = 框宽高比偏离标称值，位姿不可信   AR = 实测/标称宽高比")
            draw_panel(canvas, panel, origin=(8, 40))

        self._banner(canvas, self.header(infer_ms))
        if self.state.help_visible:
            self._hints(canvas)
        return fit_to_box(canvas, 1280, 760) if self.state.fit_window else canvas

    @staticmethod
    def _banner(image: np.ndarray, text: str) -> None:
        cv2.rectangle(image, (0, 0), (image.shape[1], 34), (20, 20, 20), -1)
        cv2.putText(image, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (140, 255, 140), 1, cv2.LINE_AA)

    def _hints(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        cv2.rectangle(image, (0, height - 24), (width, height), (20, 20, 20), -1)
        cv2.putText(image, KEY_HINTS_VIDEO if self.is_video else KEY_HINTS_IMAGE,
                    (10, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (230, 230, 230), 1, cv2.LINE_AA)

    def save(self, canvas: Optional[np.ndarray] = None) -> None:
        os.makedirs(self.args.save_dir, exist_ok=True)
        name = (f"frame_{self.index:06d}" if self.is_video
                else stem(self.paths[self.index]))
        out = os.path.join(self.args.save_dir, name + ".jpg")
        cv2.imwrite(out, canvas if canvas is not None else self.draw(self.read_frame()))
        print(f"[save] {out}")

    # ---------------------------------------------------------------- 主循环 --
    def run(self) -> int:
        cv2.namedWindow(WINDOW_IMAGE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_IMAGE, 1280, 760)
        self.playing = bool(self.args.play) and self.is_video

        while True:
            frame = self.read_frame()
            if frame is None:
                print("[warn] 读取失败或已到末尾", file=sys.stderr)
                break
            canvas = self.draw(frame)
            cv2.imshow(WINDOW_IMAGE, canvas)
            if self.state.show_curves and self.curves is not None:
                cv2.imshow(WINDOW_CURVES, fit_to_box(self.curves, 1100, 700))
            elif self.curves is not None:
                cv2.destroyWindow(WINDOW_CURVES)

            key = cv2.waitKey(1 if self.playing else 0) & 0xFF
            if key in (ord("q"), 27):
                break
            self.handle_key(key, canvas)

            if self.playing:                          # 顺序播放
                self.index += 1
                if self.index >= self.total:
                    self.index = self.total - 1
                    self.playing = False

        if self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()
        return 0

    def handle_key(self, key: int, canvas: Optional[np.ndarray] = None) -> None:
        jump = max(self.args.jump, 1)
        if key in (ord("n"), ord("d"), 83, 3):        # 下一帧/下一张
            self.playing = False
            self.index = min(self.index + 1, self.total - 1)
        elif key in (ord("p"), ord("a"), 81, 2):      # 上一帧/上一张
            self.playing = False
            self.index = max(self.index - 1, 0)
        elif key == ord(","):
            self.playing = False
            self.index = max(self.index - jump, 0)
        elif key == ord("."):
            self.playing = False
            self.index = min(self.index + jump, self.total - 1)
        elif key == ord("r"):
            self.playing = False
            self.index = 0
        elif key == ord(" "):
            if self.is_video:
                self.playing = not self.playing
            else:
                self.state.detect = not self.state.detect
        elif key == ord("x"):
            self.state.detect = not self.state.detect
        elif key == ord("z"):
            self.state.show_pose = not self.state.show_pose
        elif key == ord("g"):
            self.state.show_gt = not self.state.show_gt
        elif key == ord("c"):
            self.state.show_conf = not self.state.show_conf
        elif key == ord("v"):
            self.state.show_curves = not self.state.show_curves
        elif key == ord("f"):
            self.state.fit_window = not self.state.fit_window
        elif key == ord("h"):
            self.state.help_visible = not self.state.help_visible
        elif key == ord("s"):
            self.save(canvas)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="交互式装甲板 OBB 查看器（图片浏览 / 视频实时标注 + 训练曲线）")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="图片文件、图片目录或视频文件")
    parser.add_argument("--video", default=None, help="视频文件（等价于 --source 传视频）")
    parser.add_argument("--start", type=int, default=0, help="视频起始帧")
    parser.add_argument("--jump", type=int, default=30, help=", / . 键的跳帧步长")
    parser.add_argument("--play", action="store_true", help="视频打开后立即播放")
    parser.add_argument("--labels", default=None, help="真值标签目录（默认按 source 自动推断）")
    parser.add_argument("--weights", default=None, help="模型权重，默认取 runs/ 下最新的 best.pt")
    parser.add_argument("--run", default=None, help="训练产物目录（用其中的 results.png/csv）")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="mps", help="mps / cpu / 0(CUDA)")
    parser.add_argument("--hfov", type=float, default=60.0,
                        help="未标定时用于估算内参的水平视场角（度）")
    parser.add_argument("--calib", default=None, help="相机标定文件（JSON/YAML），优先于 --hfov")
    parser.add_argument("--focal", type=float, default=None,
                        help="直接指定焦距（像素），优先于 --hfov；距离与之成正比")
    parser.add_argument("--min-aspect", type=float, default=0.7,
                        help="位姿可信度门限：观测/标称宽高比低于该值的框标为不可信")
    parser.add_argument("--size-from", choices=("class", "aspect"), default="class",
                        help="装甲板规格来源：class=按类别推断（默认）；aspect=按观测宽高比")
    parser.add_argument("--no-pose", action="store_true", help="关闭 PnP 位姿显示")
    parser.add_argument("--split", choices=("val", "train"), default=None,
                        help="快捷切换 built dataset 的 val/train 子集")
    parser.add_argument("--limit", type=int, default=0, help="图片模式下仅浏览前 N 张")
    parser.add_argument("--no-curves", action="store_true", help="不弹出训练曲线窗口")
    parser.add_argument("--no-detect", action="store_true", help="不加载模型，只看原图/真值")
    parser.add_argument("--save-dir", default=os.path.join(REPO_ROOT, "viewer_out"),
                        help="按 s 保存标注图的目录")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.split:
        candidate = os.path.join(DEFAULT_OUT, "images", args.split)
        if os.path.isdir(candidate):
            args.source = candidate
    if args.weights is None and not args.no_detect:
        args.weights = resolve_weights(args.run or DEFAULT_RUNS)
    try:
        return Viewer(args).run()
    except RuntimeError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
