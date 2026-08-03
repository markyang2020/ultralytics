# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ebike_part_feature_compare import (
    Detection,
    DinoV2FeatureExtractor,
    _resolve_device,
    build_report,
    classify_similarity,
    clip_box,
    compare_matched_parts,
    cosine_similarity,
    crop_detection,
    pair_detections,
    parse_yolo_result,
    place_label_box,
    resolve_inputs,
    run_comparison,
    save_report,
    select_best_detections,
    validate_thresholds,
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


def test_dinov2_load_error_preserves_upstream_reason(monkeypatch):
    """防止网络或缓存异常被包装成无法继续排查的笼统错误。"""

    def fail_to_load(*args, **kwargs):
        raise OSError("download timed out")

    monkeypatch.setattr(torch.hub, "load", fail_to_load)

    with pytest.raises(RuntimeError, match="download timed out"):
        DinoV2FeatureExtractor(device="cpu")


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


def test_resolve_inputs_reads_fixed_order_filenames(tmp_path):
    """防止订单模式读取到错误文件名或交换备案图和实拍图。"""
    reference = tmp_path / "reference_3c.jpg"
    actual = tmp_path / "actual_left_front_45.jpg"
    reference.touch()
    actual.touch()

    resolved_reference, resolved_actual = resolve_inputs(tmp_path, None, None)

    assert resolved_reference == reference
    assert resolved_actual == actual


def test_resolve_inputs_rejects_mixed_input_modes(tmp_path):
    """防止订单模式和显式图片模式混用后产生不明确输入。"""
    with pytest.raises(ValueError, match="二选一"):
        resolve_inputs(tmp_path, Path("reference.jpg"), Path("actual.jpg"))


def test_validate_thresholds_rejects_inverted_similarity_bands():
    """防止一致阈值低于复核阈值导致分档分支不可达。"""
    with pytest.raises(ValueError, match="一致阈值"):
        validate_thresholds(0.6, similar_threshold=0.5, review_threshold=0.6)


def test_save_report_writes_utf8_json(tmp_path):
    """防止中文报告被 ASCII 转义或使用非 UTF-8 编码。"""
    output = save_report({"verdict_text": "需人工复核"}, tmp_path)

    assert json.loads(output.read_text(encoding="utf-8"))["verdict_text"] == "需人工复核"
    assert "需人工复核" in output.read_text(encoding="utf-8")


class _FakeDetector:
    """替代外部 YOLO 推理，返回完整 Results 边界结构。"""

    names: ClassVar = {0: "ebike_full", 5: "saddle", 7: "rear_box"}

    def predict(self, **kwargs):
        reference = SimpleNamespace(
            boxes=SimpleNamespace(
                cls=torch.tensor([0.0, 5.0]),
                conf=torch.tensor([0.95, 0.90]),
                xyxy=torch.tensor([[1, 1, 19, 19], [5, 5, 12, 12]], dtype=torch.float32),
            ),
            names=self.names,
        )
        actual = SimpleNamespace(
            boxes=SimpleNamespace(
                cls=torch.tensor([0.0, 5.0, 7.0]),
                conf=torch.tensor([0.94, 0.85, 0.88]),
                xyxy=torch.tensor([[2, 2, 20, 20], [6, 6, 13, 13], [12, 3, 19, 9]], dtype=torch.float32),
            ),
            names=self.names,
        )
        return [reference, actual]


class _SamePairFeatureExtractor:
    """为每一对裁剪返回相同方向的特征，预期全部判为一致。"""

    def extract(self, images):
        features = torch.zeros((len(images), 768))
        features[:, 0] = 1.0
        return features


def test_run_comparison_writes_report_summary_and_part_evidence(tmp_path):
    """防止完整流水线只打印分数而没有生成可回溯的文件证据。"""
    reference_path = tmp_path / "reference_3c.jpg"
    actual_path = tmp_path / "actual_left_front_45.jpg"
    detector_path = tmp_path / "model.pt"
    output_dir = tmp_path / "output"
    Image.new("RGB", (24, 24), "white").save(reference_path)
    Image.new("RGB", (24, 24), "gray").save(actual_path)
    detector_path.touch()

    report_path = run_comparison(
        reference_path=reference_path,
        actual_path=actual_path,
        detector_path=detector_path,
        output_dir=output_dir,
        detection_threshold=0.6,
        similar_threshold=0.8,
        review_threshold=0.6,
        device="cpu",
        detector_model=_FakeDetector(),
        feature_extractor=_SamePairFeatureExtractor(),
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "consistent"
    assert [part["part"] for part in report["matched_parts"]] == ["ebike_full", "saddle"]
    assert report["unmatched_parts"][0]["part"] == "rear_box"
    assert (output_dir / "comparison_summary.jpg").is_file()
    assert (output_dir / "crops" / "saddle" / "reference.jpg").is_file()
    assert (output_dir / "crops" / "saddle" / "actual.jpg").is_file()
    assert (output_dir / "crops" / "saddle" / "comparison.jpg").is_file()


def test_place_label_box_keeps_label_inside_canvas_and_avoids_collision():
    """防止右侧部件标签被画布裁断或与已有标签重叠。"""
    occupied = [(50, 0, 100, 16)]

    label_box = place_label_box(
        anchor_box=(90, 18, 100, 40),
        label_size=(50, 16),
        canvas_size=(100, 60),
        occupied=occupied,
    )

    assert label_box == (50, 18, 100, 34)
    assert label_box[0] >= 0 and label_box[2] <= 100
    assert label_box[1] >= 0 and label_box[3] <= 60


def test_resolve_device_normalizes_cuda_index_for_pytorch():
    """防止 YOLO 可识别的设备编号传给 torch.device 后变成无效字符串。"""
    assert _resolve_device("0") == "cuda:0"
