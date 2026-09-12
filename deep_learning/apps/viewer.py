"""交互式装甲板 OBB 查看器：弹窗浏览图片标注，并可视化训练结果。

用法::

    # 默认：浏览验证集 + 自动取 runs/ 下最新的 best.pt，并弹出训练曲线窗口
    python deep_learning/apps/viewer.py

    # 浏览训练集前 200 张，指定权重
    python deep_learning/apps/viewer.py --split train --limit 200 \
        --weights deep_learning/runs/armor_obb_yolo26n/weights/best.pt

    # 只看原始数据集（无预测）
    python deep_learning/apps/viewer.py --source dataset2/images --no-detect

按键::

    n / d / 右方向键   下一张
    p / a / 左方向键   上一张
    space              切换"是否运行检测"（看原图 / 看预测）
    g                  切换真值标注（GT）
    c                  切换置信度数值
    v                  切换训练曲线窗口
    f                  切换"适应窗口 / 原始尺寸"
    s                  保存当前标注图到 --save-dir
    h                  切换按键提示
    q / ESC            退出
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from armor_det.config import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_OUT,
    DEFAULT_RUNS,
    NC_VEHICLE,
    REPO_ROOT,
    class_color_bgr,
)
from utils.files import collect_images, stem  # noqa: E402

WINDOW_IMAGE = "RM Armor OBB Viewer"
WINDOW_CURVES = "Training Results"

KEY_HINTS = ("n/p prev-next  space detect  g GT  c conf  v curves  f fit  s save  h help  q quit")

DEFAULT_SOURCE = (os.path.join(DEFAULT_OUT, "images", "val")
                  if os.path.isdir(os.path.join(DEFAULT_OUT, "images", "val"))
                  else os.path.join(REPO_ROOT, "dataset2", "images"))


@dataclass
class ViewState:
    detect: bool = True
    show_gt: bool = True
    show_conf: bool = True
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


# ------------------------------------------------------------------ 查看器 --
class Viewer:
    """在窗口中浏览图片并对当前帧做 OBB 检测与标注。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.paths: List[str] = collect_images(args.source) if os.path.isdir(args.source) else [args.source]
        if not self.paths:
            raise RuntimeError(f"找不到图片: {args.source}")
        if args.limit > 0:
            self.paths = self.paths[:args.limit]
        self.index = 0
        self.state = ViewState(detect=not args.no_detect)
        self.labels_dir = args.labels or (
            guess_labels_dir(args.source) if os.path.isdir(args.source) else None)

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

    # --------------------------------------------------------------- 推理 --
    def predict(self, image: np.ndarray):
        if self.model is None:
            return None
        if self.index not in self._cache:
            self._cache[self.index] = self.model.predict(
                image, imgsz=self.args.imgsz, conf=self.args.conf,
                device=self.args.device, verbose=False)[0]
        return self._cache[self.index]

    # --------------------------------------------------------------- 显示 --
    def header(self) -> str:
        name = os.path.basename(self.paths[self.index])
        text = f"{name}  [{self.index + 1}/{len(self.paths)}]"
        if self.state.detect and self.model is not None:
            text += f"  conf>={self.args.conf:.2f}"
        if self.labels_dir and self.state.show_gt:
            text += "  GT:on"
        return text

    def draw(self) -> np.ndarray:
        image = cv2.imread(self.paths[self.index])
        if image is None:
            return np.full((480, 640, 3), 40, np.uint8)
        height, width = image.shape[:2]
        canvas = image.copy()

        if self.state.show_gt and self.labels_dir:
            label_path = os.path.join(self.labels_dir, stem(self.paths[self.index]) + ".txt")
            for cls, points in read_gt(label_path, width, height):
                draw_poly(canvas, points, class_color_bgr(cls),
                          f"GT {CLASS_NAMES.get(cls, cls)}", thickness=1)

        result = self.predict(image) if self.state.detect else None
        if result is not None and getattr(result, "obb", None) is not None and len(result.obb):
            boxes = result.obb.xyxyxyxy.cpu().numpy()
            classes = result.obb.cls.cpu().numpy().astype(int)
            confs = result.obb.conf.cpu().numpy()
            for points, cls, conf in zip(boxes, classes, confs):
                text = CLASS_NAMES.get(cls, str(cls))
                if self.state.show_conf:
                    text += f" {conf:.2f}"
                draw_poly(canvas, points.reshape(4, 2), class_color_bgr(cls), text)

        self._banner(canvas, self.header())
        if self.state.help_visible:
            self._hints(canvas)
        return fit_to_box(canvas, 1280, 760) if self.state.fit_window else canvas

    @staticmethod
    def _banner(image: np.ndarray, text: str) -> None:
        cv2.rectangle(image, (0, 0), (image.shape[1], 34), (20, 20, 20), -1)
        cv2.putText(image, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (140, 255, 140), 1, cv2.LINE_AA)

    @staticmethod
    def _hints(image: np.ndarray) -> None:
        height, width = image.shape[:2]
        cv2.rectangle(image, (0, height - 24), (width, height), (20, 20, 20), -1)
        cv2.putText(image, KEY_HINTS, (10, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (230, 230, 230), 1, cv2.LINE_AA)

    def save(self) -> None:
        os.makedirs(self.args.save_dir, exist_ok=True)
        out = os.path.join(self.args.save_dir, stem(self.paths[self.index]) + ".jpg")
        cv2.imwrite(out, self.draw())
        print(f"[save] {out}")

    # --------------------------------------------------------------- 主循环 --
    def run(self) -> int:
        cv2.namedWindow(WINDOW_IMAGE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_IMAGE, 1280, 760)
        while True:
            cv2.imshow(WINDOW_IMAGE, self.draw())
            if self.state.show_curves and self.curves is not None:
                cv2.imshow(WINDOW_CURVES, fit_to_box(self.curves, 1100, 700))
            elif self.curves is not None:
                cv2.destroyWindow(WINDOW_CURVES)

            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            self.handle_key(key)

        cv2.destroyAllWindows()
        return 0

    def handle_key(self, key: int) -> None:
        if key in (ord("n"), ord("d"), 83, 3):
            self.index = min(self.index + 1, len(self.paths) - 1)
        elif key in (ord("p"), ord("a"), 81, 2):
            self.index = max(self.index - 1, 0)
        elif key == ord(" "):
            self.state.detect = not self.state.detect
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
            self.save()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="交互式装甲板 OBB 查看器（弹窗浏览标注与训练曲线）")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="图片文件或目录")
    parser.add_argument("--labels", default=None, help="真值标签目录（默认按 source 自动推断）")
    parser.add_argument("--weights", default=None, help="模型权重，默认取 runs/ 下最新的 best.pt")
    parser.add_argument("--run", default=None, help="训练产物目录（用其中的 results.png/csv）")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="mps", help="mps / cpu / 0(CUDA)")
    parser.add_argument("--split", choices=("val", "train"), default=None,
                        help="快捷切换 built dataset 的 val/train 子集")
    parser.add_argument("--limit", type=int, default=0, help="仅浏览前 N 张")
    parser.add_argument("--no-curves", action="store_true", help="不弹出训练曲线窗口")
    parser.add_argument("--no-detect", action="store_true", help="不加载模型，只看原图/真值")
    parser.add_argument("--save-dir", default=os.path.join(REPO_ROOT, "viewer_out"), help="按 s 保存标注图")
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
