"""数据集评估脚本：统计召回、误报与耗时。

注意：数据集中大量帧根本没有机器人/灯带，这些帧检出为空是正确行为，
因此这里把“有灯带”和“无灯带”的帧分开统计，避免把空结果算成漏检。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import List

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rm_armor import ArmorDetector, DetectorConfig  # noqa: E402
from rm_armor.color import color_score  # noqa: E402
from rm_armor.config import ArmorColor  # noqa: E402
from utils.files import frame_number  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在图片目录上评估装甲板检测")
    parser.add_argument("directory", help="图片目录")
    parser.add_argument("--halt-threshold", type=int, default=70,
                        help="判定“灯带很亮”的通道差门限")
    parser.add_argument("--halt-pixels", type=int, default=200,
                        help="一帧中很亮像素数超过该值即认为机器人强在场")
    parser.add_argument("--json", action="store_true", help="逐帧打印结果")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths: List[str] = sorted(
        glob.glob(os.path.join(args.directory, "*.jpg")), key=frame_number
    )
    if not paths:
        print(f"目录中没有图片: {args.directory}", file=sys.stderr)
        return 1

    detector = ArmorDetector(DetectorConfig())

    with_bars = det_with_bars = 0
    no_bars = fp_no_bars = 0
    armors = 0
    strong_present = strong_hit = 0
    missed: List[str] = []

    start = time.perf_counter()
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        result = detector.detect(image)
        count = len(result.armors)
        armors += count

        halo = max(
            int((color_score(image, ArmorColor.RED) >= args.halt_threshold).sum()),
            int((color_score(image, ArmorColor.BLUE) >= args.halt_threshold).sum()),
        )
        if result.light_bars:
            with_bars += 1
            det_with_bars += count > 0
        else:
            no_bars += 1
            fp_no_bars += count > 0

        if halo >= args.halt_pixels:
            strong_present += 1
            strong_hit += count > 0
            if count == 0:
                missed.append(os.path.basename(path))

        if args.json:
            print(f"{os.path.basename(path)} armors={count} bars={len(result.light_bars)}")

    elapsed = time.perf_counter() - start
    total = len(paths)
    print(f"总帧数 {total}  耗时 {elapsed:.1f}s ({elapsed / total * 1000:.1f} ms/帧)")
    print(f"检出灯带的帧 {with_bars}，其中产出装甲板 {det_with_bars} "
          f"({det_with_bars / max(with_bars, 1):.1%})")
    print(f"未检出灯带的帧 {no_bars}，其中仍产出装甲板 {fp_no_bars}")
    print(f"装甲板总数 {armors}")
    print(f"[独立亮度判据] 机器人强在场帧 {strong_present}，检出 {strong_hit} "
          f"({strong_hit / max(strong_present, 1):.1%})")
    if missed:
        print(f"强在场但未检出 {len(missed)} 帧，例如: {', '.join(missed[:10])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
