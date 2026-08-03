# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""电动自行车备案图与实拍图的部件外观特征比对 Demo。"""

from dataclasses import dataclass
from math import ceil, floor
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

Box = Tuple[float, float, float, float]
ClippedBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    """保存一个部件检测框及其业务匹配字段。"""

    class_id: int
    class_name: str
    confidence: float
    box: Box


def classify_similarity(
    score: float, similar_threshold: float, review_threshold: float
) -> Tuple[str, str]:
    """按技术方案的两个阈值返回结果代码和中文判定。"""
    if score >= similar_threshold:
        return "consistent", "部件一致"
    if score >= review_threshold:
        return "review", "疑似更换，需人工复核"
    return "inconsistent", "部件外观不一致"


def clip_box(box: Box, width: int, height: int) -> ClippedBox:
    """将浮点检测框向外取整并限制在图片边界内。"""
    x1, y1, x2, y2 = box
    clipped = (
        max(0, min(width, floor(x1))),
        max(0, min(height, floor(y1))),
        max(0, min(width, ceil(x2))),
        max(0, min(height, ceil(y2))),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f"无效检测框: 原始坐标={box}, 裁剪坐标={clipped}")
    return clipped


def select_best_detections(detections: Iterable[Detection]) -> Dict[str, Detection]:
    """每个类别只保留最高置信度框，形成确定的单部件匹配基线。"""
    selected: Dict[str, Detection] = {}
    for detection in detections:
        current = selected.get(detection.class_name)
        if current is None or detection.confidence > current.confidence:
            selected[detection.class_name] = detection
    return selected


def pair_detections(
    reference: Dict[str, Detection], actual: Dict[str, Detection]
) -> Tuple[List[Tuple[str, Detection, Detection]], List[Detection], List[Detection]]:
    """拆分双方共同部件和单侧部件，单侧部件不进入 Level 2。"""
    common_names = sorted(reference.keys() & actual.keys())
    matched = [(name, reference[name], actual[name]) for name in common_names]
    reference_only = [reference[name] for name in sorted(reference.keys() - actual.keys())]
    actual_only = [actual[name] for name in sorted(actual.keys() - reference.keys())]
    return matched, reference_only, actual_only


def cosine_similarity(reference: torch.Tensor, actual: torch.Tensor) -> float:
    """显式归一化两个一维特征向量并计算余弦相似度。"""
    if reference.ndim != 1 or actual.ndim != 1 or reference.shape != actual.shape:
        raise ValueError("余弦相似度要求两个形状相同的一维特征向量")
    reference_normalized = F.normalize(reference.float(), dim=0)
    actual_normalized = F.normalize(actual.float(), dim=0)
    return float(torch.dot(reference_normalized, actual_normalized).item())
