"""推理：在图片 / 目录 / 视频上跑装甲板 OBB 检测并保存可视化结果。"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from armor_det.config import DEFAULT_RUNS, REPO_ROOT  # noqa: E402
from utils.files import collect_images  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="装甲板 OBB 推理与可视化")
    parser.add_argument("--weights", default=None,
                        help="模型权重，默认自动查找 runs/ 下最新的 best.pt")
    parser.add_argument("--source", default=os.path.join(REPO_ROOT, "dataset2", "images"),
                        help="图片 / 目录 / 视频")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "preview"), help="输出目录")
    parser.add_argument("--name", default="obb_infer", help="输出子目录名")
    parser.add_argument("--save-txt", action="store_true", help="同时导出预测 OBB 文本")
    parser.add_argument("--limit", type=int, default=0, help="目录输入时仅取前 N 张")
    return parser.parse_args()


def find_latest_weights(runs_dir: str) -> str | None:
    """在 runs/ 下查找最新的 best.pt。"""
    candidates = glob.glob(os.path.join(runs_dir, "**", "weights", "best.pt"), recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def main() -> int:
    args = parse_args()
    weights = args.weights or find_latest_weights(DEFAULT_RUNS)
    if not weights or not os.path.exists(weights):
        print("未找到权重，请用 --weights 指定，或先训练模型。", file=sys.stderr)
        return 1

    source = args.source
    if args.limit > 0 and os.path.isdir(source):
        source = collect_images(source)[:args.limit]

    from ultralytics import YOLO  # 延迟导入

    print(f"权重: {weights}\n输入: {source}\n输出: {os.path.join(args.out, args.name)}")
    model = YOLO(weights)
    model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        project=args.out,
        name=args.name,
        stream=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
