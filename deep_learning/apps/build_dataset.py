"""构建 YOLO-OBB 数据集：多数据集合并 + 标签转换 + 划分 + 离线增广。

原始标签为 10 列 `color_id vehicle_id x1 y1 x2 y2 x3 y3 x4 y4`
（dataset1 / dataset2 同格式）。组合成 24 类 `class = color_id * 8 + vehicle_id`，
再转成 YOLO-OBB 的 9 列 `class x1 y1 ... x4 y4`。

来源之间用目录名前缀区分（如 `dataset1_000001`），避免同名帧互相覆盖；
划分按**原图**进行（同一张原图的所有变体只落在同一侧），并做近重复隔离，避免数据泄漏。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from typing import Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from armor_det.augment import augment_image, obb_norm_to_px  # noqa: E402
from armor_det.config import (  # noqa: E402
    CLASS_NAMES,
    COLOR_NAMES,
    DEFAULT_DATA_YAML,
    DEFAULT_OUT,
    DEFAULT_SOURCES,
    NC_VEHICLE,
    VEHICLE_NAMES,
    AugmentConfig,
)
from utils.files import collect_images, natural_key, stem  # noqa: E402

Label = Tuple[int, List[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 YOLO-OBB 数据集（合并 + 转换 + 划分 + 离线增广）")
    parser.add_argument("--src", nargs="+", default=DEFAULT_SOURCES,
                        help="原始数据集目录，可多个（每个含 images/ 与 labels/），默认 dataset1 + dataset2")
    parser.add_argument("--out", default=DEFAULT_OUT, help="输出数据集根目录")
    parser.add_argument("--yaml", default=DEFAULT_DATA_YAML, help="输出的 data.yaml 路径")
    parser.add_argument("--variants", type=int, default=3,
                        help="每张训练原图生成的增广变体数（3 张变体 -> 整体约 4 倍）")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="验证集比例（按原图划分，防泄漏）")
    parser.add_argument("--split", choices=("dup", "random", "block"), default="dup",
                        help="划分方式：dup=近似重复帧分组（默认，防泄漏）；random=随机；block=按序号尾部")
    parser.add_argument("--dup-threshold", type=float, default=8.0,
                        help="近似重复判定阈值（64x48 灰度平均绝对差，越小越严格）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（保证可复现）")
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg",
                        help="输出图片格式，jpg 体积更小")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 张原图（冒烟用，0=全部）")
    parser.add_argument("--no-augment", action="store_true", help="只转换与划分，不做增广（对照基线）")
    parser.add_argument("--clean", action="store_true", help="构建前清空输出目录")
    return parser.parse_args()


def load_obb_label(path: str) -> List[Label]:
    """读取 10 列原始标签，返回 [(组合类别, [x1..y4]), ...]；文件不存在视为空（背景帧）。

    组合类别 = color_id * 8 + vehicle_id。
    """
    items: List[Label] = []
    if not os.path.exists(path):
        return items
    with open(path, encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            parts = raw.split()
            if not parts:
                continue
            if len(parts) != 10:
                print(f"[warn] {path}:{lineno} 期望 10 列，实际 {len(parts)} 列，已跳过",
                      file=sys.stderr)
                continue
            color, vehicle = int(parts[0]), int(parts[1])
            if color not in COLOR_NAMES or vehicle not in VEHICLE_NAMES:
                print(f"[warn] {path}:{lineno} 非法 color/vehicle {color}/{vehicle}，已跳过",
                      file=sys.stderr)
                continue
            coords = [float(v) for v in parts[2:10]]
            if any(c < 0.0 or c > 1.0 for c in coords):
                print(f"[warn] {path}:{lineno} 坐标越界 [0,1]，已跳过", file=sys.stderr)
                continue
            items.append((color * NC_VEHICLE + vehicle, coords))
    return items


def to_yolo_obb_line(cls: int, coords: List[float]) -> str:
    return f"{cls} " + " ".join(f"{c:.6f}" for c in coords)


def split_random(stems: List[str], has_target: Dict[str, bool], val_ratio: float,
                 rng: np.random.Generator) -> Tuple[List[str], List[str]]:
    """随机划分，并对"是否含目标"分层，保证两类都进入训练集。"""
    train: List[str] = []
    val: List[str] = []
    for group in (True, False):
        pool = sorted(s for s in stems if has_target[s] is group)
        rng.shuffle(pool)
        n_val = int(round(len(pool) * val_ratio))
        val.extend(pool[:n_val])
        train.extend(pool[n_val:])
    return sorted(train), sorted(val)


def split_block(stems: List[str], val_ratio: float) -> Tuple[List[str], List[str]]:
    """按序号顺序划分，尾部 val_ratio 作为验证集（时间留出）。"""
    ordered = sorted(stems)
    n_val = int(round(len(ordered) * val_ratio))
    return ordered[:len(ordered) - n_val], ordered[len(ordered) - n_val:]


def _signature(path: str, size: Tuple[int, int] = (64, 48)) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return np.full(size[0] * size[1], np.inf, dtype=np.float32)
    return cv2.resize(image, size).astype(np.float32).ravel()


def group_duplicates(stems: List[str], by_stem, threshold: float) -> List[List[str]]:
    """把近似重复的帧聚成一组（单链接：与"已有全部帧"中最接近的一张比较）。

    数据集是抽帧得到的，存在大量内容几乎相同、但序号不相邻的帧；
    若随机划分会把这些"近重复"同时放进训练与验证，指标被严重高估。
    这里先把近重复帧归组，划分时整组一起走。
    """
    signatures = np.stack([_signature(by_stem[name][0]) for name in stems])
    group_of = np.full(len(stems), -1)
    groups: List[List[str]] = []
    for i, name in enumerate(stems):
        if i == 0:
            group_of[i] = 0
            groups.append([name])
            continue
        dist = np.abs(signatures[:i] - signatures[i]).mean(axis=1)
        nearest = int(dist.argmin())
        if dist[nearest] < threshold:
            group_of[i] = group_of[nearest]
            groups[group_of[i]].append(name)
        else:
            group_of[i] = len(groups)
            groups.append([name])
    return groups


def split_dup(stems: List[str], by_stem, val_ratio: float, threshold: float,
              rng: np.random.Generator) -> Tuple[List[str], List[str], List[List[str]]]:
    """按"近重复组"划分：同一组的所有帧只落到同一侧，避免近重复泄漏。"""
    groups = group_duplicates(stems, by_stem, threshold)
    order = list(range(len(groups)))
    rng.shuffle(order)
    target = int(round(len(stems) * val_ratio))
    val_groups = set()
    picked = 0
    for gi in order:
        if picked >= target:
            break
        val_groups.add(gi)
        picked += len(groups[gi])
    train: List[str] = []
    val: List[str] = []
    for gi, members in enumerate(groups):
        (val if gi in val_groups else train).extend(members)
    return sorted(train), sorted(val), groups


def main() -> int:
    args = parse_args()
    sources = [os.path.abspath(s) for s in args.src]
    for src in sources:
        if not os.path.isdir(src):
            print(f"未找到数据集目录: {src}", file=sys.stderr)
            return 1

    if args.clean and os.path.isdir(args.out):
        shutil.rmtree(args.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    ext = ".jpg" if args.format == "jpg" else ".png"

    records: List[Tuple[str, str, List[Label]]] = []
    per_source: Counter = Counter()
    for src in sources:
        tag = os.path.basename(src)
        img_dir = os.path.join(src, "images")
        lbl_dir = os.path.join(src, "labels")
        images = collect_images(img_dir)
        if not images:
            print(f"[warn] 目录中没有图片，已跳过: {img_dir}", file=sys.stderr)
            continue
        if args.limit > 0:
            images = images[:args.limit]
        for path in images:
            # 加来源前缀，避免不同数据集的同名帧互相覆盖
            name = f"{tag}_{stem(path)}"
            records.append((name, path,
                            load_obb_label(os.path.join(lbl_dir, stem(path) + ".txt"))))
            per_source[tag] += 1
    if not records:
        print("没有收集到任何图片", file=sys.stderr)
        return 1
    records.sort(key=lambda record: natural_key(record[0]))

    by_stem = {name: (path, items) for name, path, items in records}
    has_target = {name: len(items) > 0 for name, _, items in records}
    class_counts: Counter = Counter()
    empty_files = 0
    for _, _, items in records:
        if not items:
            empty_files += 1
        for cls, _ in items:
            class_counts[cls] += 1

    rng = np.random.default_rng(args.seed)
    all_stems = [name for name, _, _ in records]
    if args.split == "dup":
        train, val, groups = split_dup(all_stems, by_stem, args.val_ratio,
                                       args.dup_threshold, rng)
        largest = max((len(g) for g in groups), default=0)
        print(f"划分方式: 近重复分组（{len(groups)} 组，最大组 {largest} 张，"
              f"阈值 {args.dup_threshold}）")
    elif args.split == "block":
        train, val = split_block(all_stems, args.val_ratio)
        print("划分方式: 序号尾部 block")
    else:
        train, val = split_random(all_stems, has_target, args.val_ratio, rng)
        print("划分方式: 随机（按原图）")

    aug_cfg = AugmentConfig()

    def save(dst_img: str, dst_lbl: str, image: np.ndarray, items: List[Label]) -> None:
        if args.format == "jpg":
            cv2.imwrite(dst_img, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            cv2.imwrite(dst_img, image)
        with open(dst_lbl, "w", encoding="utf-8") as handle:
            for cls, coords in items:
                handle.write(to_yolo_obb_line(cls, coords) + "\n")

    n_val = 0
    for name in val:  # 验证集：仅原图，不增广，保证指标可比
        path, items = by_stem[name]
        image = cv2.imread(path)
        if image is None:
            print(f"[warn] 读取失败，跳过: {path}", file=sys.stderr)
            continue
        save(os.path.join(args.out, "images/val", name + ext),
             os.path.join(args.out, "labels/val", name + ".txt"), image, items)
        n_val += 1

    n_train = 0
    for index, name in enumerate(train):  # 训练集：原图 + variants 个增广变体
        path, items = by_stem[name]
        image = cv2.imread(path)
        if image is None:
            print(f"[warn] 读取失败，跳过: {path}", file=sys.stderr)
            continue
        save(os.path.join(args.out, "images/train", name + ext),
             os.path.join(args.out, "labels/train", name + ".txt"), image, items)
        n_train += 1

        if args.no_augment or args.variants <= 0:
            continue
        height, width = image.shape[:2]
        polys = [obb_norm_to_px(coords, width, height) for _, coords in items]
        for k in range(1, args.variants + 1):
            # 每张变体用独立且可复现的种子
            vrng = np.random.default_rng(args.seed * 100003 + index * 11 + k)
            augmented = augment_image(image, polys, vrng, aug_cfg)
            aug_name = f"{name}_aug{k}"
            save(os.path.join(args.out, "images/train", aug_name + ext),
                 os.path.join(args.out, "labels/train", aug_name + ".txt"), augmented, items)
            n_train += 1

    data_yaml = args.yaml or os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                          os.path.basename(os.path.abspath(args.out)) + ".yaml")
    os.makedirs(os.path.dirname(os.path.abspath(data_yaml)), exist_ok=True)
    with open(data_yaml, "w", encoding="utf-8") as handle:
        handle.write("# 由 deep_learning/apps/build_dataset.py 自动生成\n")
        handle.write(f"path: {os.path.abspath(args.out)}\n")
        handle.write("train: images/train\n")
        handle.write("val: images/val\n")
        handle.write("names:\n")
        for index in sorted(CLASS_NAMES):
            handle.write(f"  {index}: {CLASS_NAMES[index]}\n")

    total_orig = len(records)
    src_text = ", ".join(f"{tag}={count}" for tag, count in per_source.items())
    color_counts: Counter = Counter()
    for cls, count in class_counts.items():
        color_counts[cls // NC_VEHICLE] += count
    print(f"原始图片 {total_orig} 张（{src_text}），空标签(背景) {empty_files} 张")
    print("颜色目标数: " + ", ".join(f"{COLOR_NAMES[c]}={color_counts.get(c, 0)}"
                                     for c in sorted(COLOR_NAMES)))
    print(f"类别数 {len(CLASS_NAMES)}（颜色×车型），目标总数 {sum(class_counts.values())}")
    print("每类目标数: " + ", ".join(f"{CLASS_NAMES[c]}={class_counts.get(c, 0)}"
                                     for c in sorted(CLASS_NAMES)))
    print(f"划分: train 原图 {len(train)} 张, val 原图 {len(val)} 张 "
          f"(val-ratio={args.val_ratio})")
    print(f"输出: train {n_train} 张, val {n_val} 张, 合计 {n_train + n_val} 张 "
          f"(≈ {(n_train + n_val) / max(total_orig, 1):.2f} 倍)")
    print(f"数据集目录: {os.path.abspath(args.out)}")
    print(f"配置文件:   {os.path.abspath(data_yaml)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
