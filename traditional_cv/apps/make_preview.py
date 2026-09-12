"""检测结果可视化：生成静态帧拼图与标注视频。

示例::

    python make_preview.py stills --source images --out preview
    python make_preview.py video --video autoaim_all.mp4 --start 2400 --end 3600 --stride 3
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rm_armor import ArmorDetector, DetectorConfig, draw_result
from rm_armor.config import ArmorColor
from rm_armor.detector import DetectionResult
from rm_armor.visualize import draw_text_block, render_mask
from utils.files import frame_number, natural_key

TILE_W = 470
CAPTION_H = 24
COLS = 3

GREEN = (120, 255, 120)
ORANGE = (80, 190, 255)
GREY = (200, 200, 200)


@dataclass
class Case:
    """一帧的检测摘要，用于挑选用例。"""

    path: str
    frame: int
    n_armors: int
    n_bars: int
    best_score: float
    best_label: str


# ------------------------------------------------------------------ 基础绘制 --
def caption_of(case: Case) -> Tuple[str, Tuple[int, int, int]]:
    """生成图块标题与颜色：有检出为绿色，只有灯带为橙色。"""
    name = os.path.basename(case.path)
    if case.n_armors:
        return f"{name}  bars={case.n_bars}  {case.best_label}", GREEN
    if case.n_bars:
        return f"{name}  bars={case.n_bars}  未配对", ORANGE
    return f"{name}  无灯带", GREY


def make_tile(image: np.ndarray, caption: str, colour: Tuple[int, int, int],
              info: Sequence[str] = ()) -> np.ndarray:
    """把画面缩放到统一宽度，加上标题条，并在图内重绘紧凑信息块。

    信息块必须在缩放之后绘制，否则在缩略图里会小到看不清。
    """
    height, width = image.shape[:2]
    scale = TILE_W / width
    resized = cv2.resize(image, (TILE_W, int(round(height * scale))), interpolation=cv2.INTER_AREA)
    bar = np.full((CAPTION_H, TILE_W, 3), 28, dtype=np.uint8)
    cv2.putText(bar, caption, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
    tile = np.vstack([bar, resized])
    if info:
        draw_text_block(tile, list(info), (6, CAPTION_H + 6),
                        max_width=TILE_W - 12, preferred=0.42)
    return tile


def compact_info(result: DetectionResult) -> List[str]:
    """缩略图用的紧凑信息：耗时 + 每块装甲板的关键数值。"""
    lines = [f"detect {result.timings.get('total', 0.0):.1f} ms   bars={len(result.light_bars)}"
             f"   armors={len(result.armors)}"]
    for index, armor in enumerate(result.armors[:2], start=1):
        text = f"#{index} {armor.color.value.upper()} {armor.size.value.upper()} s={armor.score:.2f}"
        if armor.pose is not None:
            text += f"  {armor.pose.distance_m:.2f}m yaw={armor.pose.yaw_deg:+.0f}"
        lines.append(text)
    return lines


def montage(tiles: Sequence[np.ndarray], cols: int = COLS) -> Optional[np.ndarray]:
    if not tiles:
        return None
    rows = []
    for start in range(0, len(tiles), cols):
        row = list(tiles[start:start + cols])
        while len(row) < cols:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def bar_of(result: DetectionResult, image: np.ndarray) -> Optional[Tuple[str, Tuple[int, int, int]]]:
    """取该帧得分最高的装甲板的摘要文字。"""
    if not result.armors:
        return None
    armor = max(result.armors, key=lambda a: a.score)
    text = f"{armor.color.value.upper()[:1]}{armor.size.value.upper()[:1]} s={armor.score:.2f}"
    if armor.pose is not None:
        text += f" {armor.pose.distance_m:.2f}m yaw={armor.pose.yaw_deg:+.0f}"
    return text, GREEN


def _panel_header(case: Case, result: DetectionResult) -> str:
    return (f"{os.path.basename(case.path)}  armors={len(result.armors)} "
            f"bars={len(result.light_bars)}")


# ------------------------------------------------------------------ 用例挑选 --
def collect_cases(detector: ArmorDetector, paths: Sequence[str], step: int) -> List[Case]:
    cases: List[Case] = []
    for path in paths[::step]:
        image = cv2.imread(path)
        if image is None:
            continue
        result = detector.detect(image)
        summary = bar_of(result, image)
        cases.append(Case(
            path=path,
            frame=frame_number(path),
            n_armors=len(result.armors),
            n_bars=len(result.light_bars),
            best_score=max((a.score for a in result.armors), default=0.0),
            best_label=summary[0] if summary else "",
        ))
    return cases


def pick(cases: Sequence[Case], limit: int) -> List[Tuple[str, Case]]:
    """挑选有代表性的用例：高分检出、多目标、仅灯带未配对、无灯带。"""
    chosen: List[Tuple[str, Case]] = []
    used: set[str] = set()

    def take(tag: str, pool: Sequence[Case], count: int) -> None:
        taken = 0
        for case in pool:
            if taken >= count:
                break
            if case.path in used:
                continue
            used.add(case.path)
            chosen.append((tag, case))
            taken += 1

    detected = sorted((c for c in cases if c.n_armors), key=lambda c: -c.best_score)
    take("高分检出", detected, limit)
    take("多目标", [c for c in detected if c.n_armors >= 2], 3)
    take("中等得分", list(reversed(detected)), 3)
    take("仅灯带未配对", [c for c in cases if c.n_bars and not c.n_armors], 3)
    take("单条灯带", [c for c in cases if c.n_bars == 1], 3)
    take("无灯带帧", [c for c in cases if not c.n_bars], 3)
    return chosen


# ------------------------------------------------------------------ 静态拼图 --
def run_stills(args: argparse.Namespace) -> int:
    detector = ArmorDetector(DetectorConfig())
    paths = sorted(glob.glob(os.path.join(args.source, "*.jpg")), key=natural_key)
    if not paths:
        print(f"目录中没有图片: {args.source}", file=sys.stderr)
        return 1

    cases = collect_cases(detector, paths, args.sample_step)
    chosen = pick(cases, args.cases)
    print(f"采样 {len(cases)} 帧（每 {args.sample_step} 帧取 1），选中 {len(chosen)} 个用例")

    os.makedirs(args.out, exist_ok=True)
    tiles: List[np.ndarray] = []
    mask_tiles: List[np.ndarray] = []

    for index, (tag, case) in enumerate(chosen, start=1):
        image = cv2.imread(case.path)
        result = detector.detect(image)
        # 单张原尺寸输出带完整信息面板，便于打开细看
        canvas = draw_result(image, result, detector.config, show_interior=True,
                             show_panel=True, header=_panel_header(case, result))

        stem = f"{index:02d}_{tag}_{os.path.basename(case.path)}"
        cv2.imwrite(os.path.join(args.out, stem), canvas)

        caption, colour = caption_of(case)
        tiles.append(make_tile(canvas, f"[{tag}] {caption}", colour, compact_info(result)))

        red = render_mask(result.masks[ArmorColor.RED], (image.shape[1], image.shape[0]))
        blue = render_mask(result.masks[ArmorColor.BLUE], (image.shape[1], image.shape[0]))
        strip = np.hstack([image, red, blue])
        mask_tiles.append(make_tile(strip, f"[{tag}] 原图 | 红掩膜 | 蓝掩膜  {os.path.basename(case.path)}",
                                    colour))

    board = montage(tiles)
    if board is not None:
        path = os.path.join(args.out, "montage_detections.jpg")
        cv2.imwrite(path, board)
        print(f"已写出 {path}  ({board.shape[1]}x{board.shape[0]})")

    mask_board = montage(mask_tiles)
    if mask_board is not None:
        path = os.path.join(args.out, "montage_masks.jpg")
        cv2.imwrite(path, mask_board)
        print(f"已写出 {path}  ({mask_board.shape[1]}x{mask_board.shape[0]})")

    print(f"用例缩略图位于 {args.out}/")
    return 0


# ------------------------------------------------------------------ 标注视频 --
def open_writer(path: str, fps: float, size: Tuple[int, int]):
    for tag in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*tag), fps, size)
        if writer.isOpened():
            return writer, tag
        writer.release()
    return None, None


def run_video(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.video):
        print(f"[error] 找不到视频: {args.video}", file=sys.stderr)
        return 1

    detector = ArmorDetector(DetectorConfig())
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        print(f"[error] 无法打开视频: {args.video}", file=sys.stderr)
        return 1

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    end = min(args.end, total) if total else args.end
    out_fps = max(fps / max(args.stride, 1), 1.0)

    writer, tag = open_writer(args.out, out_fps, (width, height))
    if writer is None:
        print("[error] 无法创建输出视频", file=sys.stderr)
        capture.release()
        return 1

    span = max(end - args.start, 1)
    samples: List[Tuple[int, DetectionResult, np.ndarray]] = []

    frame = 0
    written = 0
    hits = 0
    while True:
        ok, image = capture.read()
        if not ok or frame >= end:
            break
        if frame >= args.start and (frame - args.start) % args.stride == 0:
            result = detector.detect(image)
            canvas = draw_result(image, result, detector.config, show_interior=True)
            writer.write(canvas)
            written += 1
            hits += 1 if result.armors else 0
            if len(samples) < 6 and written % max(1, span // args.stride // 6) == 1:
                samples.append((frame, result, canvas.copy()))
        frame += 1

    capture.release()
    writer.release()
    print(f"区间 [{args.start}, {end}) 每 {args.stride} 帧处理一张，共写出 {written} 帧，"
          f"其中检出 {hits} ({hits / max(written, 1):.1%}) -> {args.out}  编码={tag}")

    tiles = []
    for index, (number, result, canvas) in enumerate(samples, start=1):
        best = max(result.armors, key=lambda a: a.score) if result.armors else None
        if best is not None and best.pose is not None:
            label = (f"#{number} {best.size.value} {best.pose.distance_m:.2f}m "
                     f"yaw={best.pose.yaw_deg:+.0f} s={best.score:.2f}")
            colour = GREEN
        elif result.light_bars:
            label = f"#{number} bars={len(result.light_bars)} 未配对"
            colour = ORANGE
        else:
            label = f"#{number} 无灯带"
            colour = GREY
        tiles.append(make_tile(canvas, label, colour))

    board = montage(tiles)
    if board is not None:
        path = os.path.splitext(args.out)[0] + "_samples.jpg"
        cv2.imwrite(path, board)
        print(f"已写出抽样拼图 {path}")
    return 0


# ------------------------------------------------------------------ 入口 --
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="装甲板检测结果可视化")
    sub = parser.add_subparsers(dest="mode", required=True)

    stills = sub.add_parser("stills", help="从图片目录生成拼图")
    stills.add_argument("--source", default="images")
    stills.add_argument("--out", default="preview")
    stills.add_argument("--cases", type=int, default=6, help="高分检出用例数")
    stills.add_argument("--sample-step", type=int, default=8, help="采样间隔，越小越慢但覆盖越全")

    video = sub.add_parser("video", help="导出标注视频")
    video.add_argument("--video", default="autoaim_all.mp4")
    video.add_argument("--out", default="preview/annotated.mp4")
    video.add_argument("--start", type=int, default=0)
    video.add_argument("--end", type=int, default=10 ** 9)
    video.add_argument("--stride", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    return run_stills(args) if args.mode == "stills" else run_video(args)


if __name__ == "__main__":
    sys.exit(main())
