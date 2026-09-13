"""训练 YOLO26-OBB 装甲板检测模型。

先用 build_dataset.py 生成 data/armor_obb.yaml，再跑本脚本。
默认面向本机 Apple MPS；若 MPS 报不支持算子，用 --device cpu 回退。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from armor_det.config import DEFAULT_DATA_YAML, DEFAULT_RUNS  # noqa: E402


def resolve_device(name: str) -> str:
    """把 --device auto 解析成当前平台可用的设备：CUDA -> 0，Apple -> mps，否则 cpu。"""
    if name != "auto":
        return name
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 YOLO26-OBB 装甲板检测模型")
    parser.add_argument("--data", default=DEFAULT_DATA_YAML, help="数据集 data.yaml")
    parser.add_argument("--model", default="yolo26n-obb.pt",
                        help="预训练权重或模型 yaml（如 yolo26s-obb.pt 提升精度）")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=1024, help="目标偏小，建议 1024")
    parser.add_argument("--batch", type=int, default=8,
                        help="显存/内存受限就降低；GPU 上可开到 16/32")
    parser.add_argument("--device", default="auto",
                        help="auto(默认，自动选 CUDA/MPS/CPU) / cpu / 0 / 0,1 / mps")
    parser.add_argument("--patience", type=int, default=30, help="早停轮数")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default=DEFAULT_RUNS, help="训练产物根目录")
    parser.add_argument("--name", default=None, help="本次运行名，默认按模型与时间戳生成")
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not os.path.exists(args.data):
        print(f"未找到数据集配置: {args.data}\n"
              f"请先运行: python3 deep_learning/apps/build_dataset.py", file=sys.stderr)
        return 1

    from ultralytics import YOLO  # 延迟导入，未装依赖时给出清晰报错

    device = resolve_device(args.device)
    model_tag = os.path.splitext(os.path.basename(args.model))[0]
    name = args.name or f"armor_obb_{model_tag}_{datetime.now():%Y%m%d-%H%M%S}"

    print(f"模型: {args.model}  数据: {args.data}")
    print(f"device={args.device} -> {device}  epochs={args.epochs} imgsz={args.imgsz} "
          f"batch={args.batch} patience={args.patience}")

    model = YOLO(args.model)
    if args.workers > 0:
        # ultralytics 在 device 为 cpu/mps 时会把 workers 强制置 0（BaseTrainer.__init__），
        # 这样图像解码/增广全挤在单进程里、成为瓶颈（实测吞吐与 batch 无关）。
        # 该回调在 dataloader 构建之前触发，正好把 workers 改回来。
        model.add_callback("on_pretrain_routine_start",
                           lambda trainer: setattr(trainer.args, "workers", args.workers))
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        project=args.project,
        name=name,
        cache=False,
        resume=args.resume,
    )
    print(f"训练完成，最佳权重: {os.path.join(args.project, name, 'weights', 'best.pt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
