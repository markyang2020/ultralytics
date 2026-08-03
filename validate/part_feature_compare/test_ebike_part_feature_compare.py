# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ebike_part_feature_compare import (  # noqa: E402
    Detection,
    DinoV2FeatureExtractor,
    build_report,
    classify_similarity,
    clip_box,
    compare_matched_parts,
    cosine_similarity,
    crop_detection,
    parse_yolo_result,
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


def test_parse_yolo_result_selects_highest_confidence_box_per_class():
    """防止 YOLO 张量字段解析错误或同类框选择规则失效。"""
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            cls=torch.tensor([0.0, 0.0, 5.0]),
            conf=torch.tensor([0.70, 0.95, 0.85]),
            xyxy=torch.tensor([[1, 1, 8, 8], [2, 2, 9, 9], [4, 2, 7, 5]], dtype=torch.float32),
        ),
        names={0: "ebike_full", 5: "saddle"},
    )

    selected = parse_yolo_result(result)

    assert set(selected) == {"ebike_full", "saddle"}
    assert selected["ebike_full"].confidence == pytest.approx(0.95)
    assert selected["ebike_full"].box == (2.0, 2.0, 9.0, 9.0)


def test_crop_detection_uses_clipped_box():
    """防止部件裁剪绕过已验证的边界限制。"""
    image = Image.new("RGB", (10, 8), "white")
    detection = Detection(0, "part", 0.9, (-2, 1, 12, 7))

    crop = crop_detection(image, detection)

    assert crop.size == (10, 6)


class _FakeDinoModel(torch.nn.Module):
    """仅替代外部 DINOv2 网络计算，保留真实预处理和特征后处理。"""

    def forward_features(self, batch):
        features = torch.zeros((batch.shape[0], 768), device=batch.device)
        features[:, 0] = 2.0
        return {"x_norm_clstoken": features}


def test_dinov2_extractor_returns_normalized_768d_features():
    """防止读取错误 token、维度漂移或遗漏输出归一化。"""
    extractor = DinoV2FeatureExtractor(device="cpu", model=_FakeDinoModel())

    features = extractor.extract([Image.new("RGB", (20, 10), "white"), Image.new("RGB", (8, 16), "black")])

    assert features.shape == (2, 768)
    assert torch.linalg.vector_norm(features, dim=1).tolist() == pytest.approx([1.0, 1.0])


def test_build_report_records_models_thresholds_and_unmatched_parts():
    """防止报告漏掉复现实验所需的模型、阈值或单侧检测证据。"""
    report = build_report(
        reference_path=Path("reference.jpg"),
        actual_path=Path("actual.jpg"),
        detector_path=Path("model.pt"),
        detection_threshold=0.6,
        similar_threshold=0.8,
        review_threshold=0.6,
        matched_parts=[],
        reference_only=[Detection(9, "rack", 0.9, (1, 1, 4, 4))],
        actual_only=[],
    )

    assert report["feature_model"] == "dinov2_vitb14"
    assert report["thresholds"] == {"detection": 0.6, "consistent": 0.8, "review": 0.6}
    assert report["unmatched_parts"][0]["part"] == "rack"
    assert report["unmatched_parts"][0]["side"] == "reference_only"


class _FixedFeatureExtractor:
    """用确定向量隔离外部模型，验证部件裁剪到相似度分档的真实编排。"""

    def extract(self, images):
        assert [image.size for image in images] == [(4, 4), (4, 4)]
        return torch.tensor([[1.0, 0.0], [0.0, 1.0]])


def test_compare_matched_parts_uses_paired_crops_and_similarity_thresholds():
    """防止双方部件裁剪顺序错位或相似度结果未写入匹配项。"""
    reference_detection = Detection(5, "saddle", 0.93, (1, 1, 5, 5))
    actual_detection = Detection(5, "saddle", 0.79, (2, 2, 6, 6))

    compared = compare_matched_parts(
        [("saddle", reference_detection, actual_detection)],
        Image.new("RGB", (8, 8), "white"),
        Image.new("RGB", (8, 8), "black"),
        _FixedFeatureExtractor(),
        similar_threshold=0.8,
        review_threshold=0.6,
    )

    assert len(compared) == 1
    assert compared[0].part_name == "saddle"
    assert compared[0].similarity == pytest.approx(0.0)
    assert compared[0].verdict == "inconsistent"
