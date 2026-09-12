"""通用文件与路径工具：自然排序、图片收集。

与 traditional_cv/utils/files.py 保持一致的语义，但刻意各自独立，
避免两套方案（传统视觉 / 深度学习）相互耦合。
"""

from __future__ import annotations

import glob
import os
import re
from typing import List, Sequence

from armor_det.config import IMAGE_SUFFIXES


def natural_key(path: str) -> List:
    """把文件名里的数字按数值比较，其余部分按字符串比较。

    避免 frame_010000 排在 frame_002933 前面。
    """
    return [int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", os.path.basename(path))]


def stem(path: str) -> str:
    """取不含扩展名的文件名，用于图片与标签同名配对。"""
    return os.path.splitext(os.path.basename(path))[0]


def collect_images(source: str, suffixes: Sequence[str] = IMAGE_SUFFIXES) -> List[str]:
    """收集图片文件（或目录下指定后缀的图片），按自然顺序排序。"""
    if os.path.isfile(source):
        return [source]
    files: List[str] = []
    for suffix in suffixes:
        files.extend(glob.glob(os.path.join(source, f"*{suffix}")))
    return sorted(files, key=natural_key)
