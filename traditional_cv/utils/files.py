"""通用文件与路径工具：自然排序、帧号解析、目录收集。"""

from __future__ import annotations

import glob
import os
import re
from typing import List, Sequence

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


def natural_key(path: str) -> List:
    """把文件名里的数字按数值比较，其余部分按字符串比较。

    避免 frame_010000 排在 frame_002933 前面。
    """
    return [int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", os.path.basename(path))]


def frame_number(path: str) -> int:
    """取文件名中的第一个数字作为帧号，没有则返回 0。"""
    numbers = re.findall(r"\d+", os.path.basename(path))
    return int(numbers[0]) if numbers else 0


def ensure_parent(path: str) -> None:
    """确保目标文件的父目录存在。"""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def collect_images(source: str, suffixes: Sequence[str] = IMAGE_SUFFIXES) -> List[str]:
    """收集图片文件（或目录下指定后缀的图片），按自然顺序排序。"""
    if os.path.isfile(source):
        return [source]
    files: List[str] = []
    for suffix in suffixes:
        files.extend(glob.glob(os.path.join(source, f"*{suffix}")))
    return sorted(files, key=natural_key)
