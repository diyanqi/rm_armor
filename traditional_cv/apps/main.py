"""装甲板检测命令行入口：支持单图 / 图片目录 / 视频。

示例::

    python main.py images/frame_000117.jpg --out out
    python main.py images --limit 200 --out out
    python main.py autoaim_all.mp4 --out out/annotated.mp4 --stride 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Optional

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rm_armor import ArmorDetector, DetectorConfig, draw_result
from rm_armor.armor import Armor
from rm_armor.config import ARMOR_ASPECT, ArmorSize
from rm_armor.detector import DetectionResult
from rm_armor.visualize import render_mask
from utils.files import IMAGE_SUFFIXES, VIDEO_SUFFIXES, ensure_parent, natural_key


# ------------------------------------------------------------------ 参数解析 --
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoboMaster 装甲板检测（灯带配对 + PnP 位姿）")
    parser.add_argument("source", help="图片路径、图片目录或视频文件")
    parser.add_argument("--out", default=None, help="输出路径：图片模式为目录，视频模式为文件")
    parser.add_argument("--limit", type=int, default=None, help="目录模式最多处理多少张图片")
    parser.add_argument("--stride", type=int, default=1, help="视频模式帧间隔")
    parser.add_argument("--calib", default=None, help="相机标定文件（JSON/YAML）")
    parser.add_argument("--hfov", type=float, default=60.0, help="未标定时的水平视场角（度）")
    parser.add_argument("--min-score", type=float, default=None, help="配对得分下限")
    parser.add_argument("--no-plate-check", action="store_true",
                        help="关闭板面“黑底+白字”校验（调参用，会引入跨板误配）")
    parser.add_argument("--no-large", action="store_true",
                        help="只输出小装甲板（数据集中没有大装甲板时用）")
    parser.add_argument("--no-decompose", action="store_true",
                        help="关闭“跨板组合拆解”仲裁")
    parser.add_argument("--no-hue-gate", action="store_true",
                        help="关闭色相门限（会重新引入深红状态灯与彩色背景）")
    parser.add_argument("--min-bar-height", type=float, default=None, help="灯带最小像素高度")
    parser.add_argument("--min-peak-red", type=float, default=None, help="橙红灯带最亮处 V 下限")
    parser.add_argument("--min-peak-blue", type=float, default=None, help="蓝色灯带最亮处 V 下限")
    parser.add_argument("--min-ring-contrast", type=float, default=None,
                        help="灯带相对四周的最小亮度差")
    parser.add_argument("--min-ring-dark", type=float, default=None,
                        help="灯带四周暗像素占比下限")
    parser.add_argument("--json", action="store_true", help="逐帧打印检测结果 JSON")
    parser.add_argument("--dump-masks", default=None, help="把颜色掩膜写到该目录")
    parser.add_argument("--show-interior", action="store_true", help="勾出黑底白字校验区域")
    parser.add_argument("--show-bar-metrics", action="store_true", help="在每条灯带旁标出颜色指标")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> DetectorConfig:
    cfg = DetectorConfig()
    cfg.camera.hfov_deg = args.hfov
    cfg.camera.calibration_file = args.calib
    if args.min_score is not None:
        cfg.pair.min_score = args.min_score
    if args.min_bar_height is not None:
        cfg.lightbar.min_height_px = args.min_bar_height
    if args.min_peak_red is not None:
        cfg.lightbar.min_peak_value_red = args.min_peak_red
    if args.min_peak_blue is not None:
        cfg.lightbar.min_peak_value_blue = args.min_peak_blue
    if args.min_ring_contrast is not None:
        cfg.lightbar.min_ring_contrast = args.min_ring_contrast
    if args.min_ring_dark is not None:
        cfg.lightbar.min_ring_dark_ratio = args.min_ring_dark
    if args.no_plate_check:
        cfg.pair.use_plate_validation = False
    if args.no_decompose:
        cfg.pair.decompose_large = False
    if args.no_hue_gate:
        cfg.color.use_hue_gate = False
    if args.no_large:
        # 靠把 LARGE 的宽高比下限抬到不可达来只保留小装甲板
        cfg.pair.max_aspect = ARMOR_ASPECT[ArmorSize.SMALL] * 1.15
    return cfg


# ------------------------------------------------------------------ 结果导出 --
def armor_to_dict(armor: Armor) -> Dict:
    record = {
        "color": armor.color.value,
        "size": armor.size.value,
        "size_confidence": round(armor.size_confidence, 3),
        "score": round(armor.score, 3),
        "aspect": round(armor.aspect, 3),
        "center_px": [round(float(v), 1) for v in armor.center],
        "bars": [
            {
                "px": [round(float(v), 1) for v in armor.left.center],
                "hue": round(armor.left.hue_med, 1),
                "peak_value": round(armor.left.peak_value, 1),
                "ring_dark_ratio": round(armor.left.ring_dark_ratio, 3),
                "height_px": round(armor.left.height_px, 1),
            },
            {
                "px": [round(float(v), 1) for v in armor.right.center],
                "hue": round(armor.right.hue_med, 1),
                "peak_value": round(armor.right.peak_value, 1),
                "ring_dark_ratio": round(armor.right.ring_dark_ratio, 3),
                "height_px": round(armor.right.height_px, 1),
            },
        ],
    }
    if armor.evidence is not None:
        record["plate_evidence"] = {
            "dark_ratio": round(armor.evidence.dark_ratio, 3),
            "kappa": round(armor.evidence.kappa, 3),
            "digit_ratio": round(armor.evidence.digit_ratio, 4),
            "digit_offset": round(armor.evidence.digit_offset, 3),
            "digit_present": armor.evidence.digit_present,
            "evaluated": armor.evidence.evaluated,
        }
    if armor.pose is not None:
        pose = armor.pose
        record["pose"] = {
            "distance_m": round(pose.distance_m, 3),
            "distance_from_height_m": round(pose.distance_from_height_m, 3),
            "yaw_deg": round(pose.yaw_deg, 2),
            "pitch_deg": round(pose.pitch_deg, 2),
            "roll_deg": round(pose.roll_deg, 2),
            "bearing_deg": round(pose.bearing_deg, 2),
            "center_cam_m": [round(float(v), 3) for v in pose.center_cam],
            "reproj_error_px": round(pose.reproj_error_px, 3),
        }
    return record


def result_to_dict(result: DetectionResult) -> Dict:
    return {
        "armors": [armor_to_dict(a) for a in result.armors],
        "light_bars": len(result.light_bars),
    }


# ------------------------------------------------------------------ 处理流程 --
def process_image(detector: ArmorDetector, path: str, args: argparse.Namespace,
                  out_path: Optional[str] = None) -> Optional[Dict]:
    image = cv2.imread(path)
    if image is None:
        print(f"[warn] 无法读取图片: {path}", file=sys.stderr)
        return None

    result = detector.detect(image)
    if out_path:
        ensure_parent(out_path)
        cv2.imwrite(out_path, draw_result(image, result, detector.config,
                                          show_interior=args.show_interior,
                                          show_bar_metrics=args.show_bar_metrics))
    if args.dump_masks:
        _dump_masks(result, args.dump_masks, os.path.basename(path))
    if args.json:
        print(json.dumps({"frame": os.path.basename(path), **result_to_dict(result)},
                         ensure_ascii=False))
    return result_to_dict(result)


def process_directory(detector: ArmorDetector, directory: str,
                      args: argparse.Namespace) -> None:
    # 按数字后缀排序，避免 frame_010000 排在 frame_002933 前面。
    paths = sorted(
        (os.path.join(directory, name) for name in os.listdir(directory)
         if name.lower().endswith(IMAGE_SUFFIXES)),
        key=natural_key,
    )
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print(f"[warn] 目录中没有图片: {directory}", file=sys.stderr)
        return

    out_dir = args.out
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total = Counter()
    hit_images = 0
    for index, path in enumerate(paths, start=1):
        out_path = os.path.join(out_dir, os.path.basename(path)) if out_dir else None
        record = process_image(detector, path, args, out_path)
        if record is None:
            continue
        if record["armors"]:
            hit_images += 1
        for armor in record["armors"]:
            total[f"{armor['color']}/{armor['size']}"] += 1
        if index % 50 == 0:
            print(f"  已处理 {index}/{len(paths)} 张", file=sys.stderr)

    print(f"图片总数: {len(paths)}")
    print(f"检出装甲板的图片: {hit_images} ({hit_images / len(paths):.1%})")
    print(f"装甲板总数: {sum(total.values())}")
    for key, count in sorted(total.items()):
        print(f"  {key}: {count}")


def process_video(detector: ArmorDetector, path: str, args: argparse.Namespace) -> None:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        print(f"[warn] 无法打开视频: {path}", file=sys.stderr)
        return

    writer = None
    if args.out:
        ensure_parent(args.out)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / max(args.stride, 1), size)

    frame_index = 0
    processed = 0
    detected = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % args.stride == 0:
                result = detector.detect(frame)
                processed += 1
                detected += 1 if result.armors else 0
                if writer is not None:
                    writer.write(draw_result(frame, result, detector.config,
                                             show_interior=args.show_interior,
                                          show_bar_metrics=args.show_bar_metrics))
                if args.dump_masks:
                    _dump_masks(result, args.dump_masks, f"frame_{frame_index:06d}.jpg")
                if args.json:
                    print(json.dumps({"frame": frame_index, **result_to_dict(result)},
                                     ensure_ascii=False))
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    print(f"处理帧数: {processed}")
    if processed:
        print(f"检出装甲板的帧: {detected} ({detected / processed:.1%})")


def _dump_masks(result: DetectionResult, directory: str, name: str) -> None:
    os.makedirs(directory, exist_ok=True)
    stem, ext = os.path.splitext(name)
    for color, mask in result.masks.items():
        path = os.path.join(directory, f"{stem}_{color.value}{ext}")
        cv2.imwrite(path, render_mask(mask))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    detector = ArmorDetector(build_config(args))

    source = args.source
    if os.path.isdir(source):
        process_directory(detector, source, args)
    elif source.lower().endswith(VIDEO_SUFFIXES):
        process_video(detector, source, args)
    elif os.path.isfile(source):
        record = process_image(detector, source, args, args.out)
        if record is not None and not args.json:
            print(json.dumps(record, ensure_ascii=False, indent=2))
    else:
        print(f"[error] 找不到输入: {source}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
