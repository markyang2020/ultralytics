# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ebike_part_feature_compare import (  # noqa: E402
    Detection,
    classify_similarity,
    clip_box,
    cosine_similarity,
    pair_detections,
    select_best_detections,
)


@pytest.mark.parametrize(
    ("score", "expected_code"),
    [(0.80, "consistent"), (0.60, "review"), (0.5999, "inconsistent")],
)
def test_classify_similarity_uses_closed_threshold_boundaries(score, expected_code):
    """防止阈值边界因比较符号错误落入相邻分档。"""
    assert classify_similarity(score, 0.80, 0.60)[0] == expected_code


def test_select_best_detections_keeps_highest_confidence_per_class():
    """防止同类低置信框覆盖更可靠的检测结果。"""
    detections = [
        Detection(0, "ebike_full", 0.70, (1, 1, 8, 8)),
        Detection(0, "ebike_full", 0.95, (2, 2, 9, 9)),
        Detection(5, "saddle", 0.85, (4, 2, 7, 5)),
    ]

    selected = select_best_detections(detections)

    assert selected["ebike_full"].confidence == 0.95
    assert selected["ebike_full"].box == (2, 2, 9, 9)
    assert selected["saddle"].confidence == 0.85


def test_clip_box_clamps_coordinates_and_rounds_outward():
    """防止浮点检测框裁剪时丢失边缘像素或越过图像边界。"""
    assert clip_box((-2.2, 1.1, 12.8, 9.2), width=10, height=8) == (0, 1, 10, 8)


def test_clip_box_rejects_empty_box():
    """防止零宽检测框进入特征提取并产生难以解释的模型异常。"""
    with pytest.raises(ValueError, match="无效检测框"):
        clip_box((4, 4, 4, 7), width=10, height=8)


def test_pair_detections_separates_common_and_single_side_parts():
    """防止把单侧检测部件错误送入 Level 2 外观特征比对。"""
    reference = {
        "ebike_full": Detection(0, "ebike_full", 0.95, (1, 1, 9, 9)),
        "rack": Detection(9, "rack", 0.90, (5, 5, 8, 8)),
    }
    actual = {
        "ebike_full": Detection(0, "ebike_full", 0.93, (2, 2, 9, 9)),
        "rear_box": Detection(7, "rear_box", 0.88, (4, 4, 7, 7)),
    }

    matched, reference_only, actual_only = pair_detections(reference, actual)

    assert [part_name for part_name, _, _ in matched] == ["ebike_full"]
    assert [detection.class_name for detection in reference_only] == ["rack"]
    assert [detection.class_name for detection in actual_only] == ["rear_box"]


def test_cosine_similarity_normalizes_inputs():
    """防止直接点积使相似度随特征向量长度变化。"""
    score = cosine_similarity(torch.tensor([2.0, 0.0]), torch.tensor([4.0, 0.0]))

    assert score == pytest.approx(1.0)
