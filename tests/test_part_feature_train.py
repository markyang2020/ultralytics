import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from validate.part_feature_train.domain_adapt import ComponentPairDataset
from validate.part_feature_train import prepare_data
from validate.part_feature_train.prepare_data import build_training_pairs, crop_components_from_image, verify_dataset


MODEL_NAMES = {
    0: "ebike_full",
    1: "front_headlight",
    2: "front_basket",
    3: "mirror",
    4: "pedal",
    5: "saddle",
    6: "reflector",
    7: "rear_box",
    8: "seat",
    9: "rack",
    10: "backrest",
    11: "center_box",
}


class FakeYolo:
    """返回真实权重类别表和每个类别一个检测框的最小 YOLO 替身。"""

    names = MODEL_NAMES

    def __init__(self, class_ids):
        self.class_ids = class_ids

    def __call__(self, image_path, conf, verbose):
        boxes = [
            SimpleNamespace(
                cls=np.array([class_id]),
                conf=np.array([0.9]),
                xyxy=np.array([[1, 1, 10, 10]]),
            )
            for class_id in self.class_ids
        ]
        return [SimpleNamespace(boxes=boxes)]


def write_image(path):
    """写入可供裁剪或数据集加载的最小 RGB 图像。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((16, 16, 3), 127, dtype=np.uint8))


def test_crop_uses_model_class_names_and_keeps_only_target_components(tmp_path):
    """12 类权重必须按模型类别名保存，且仅保留训练目标部件。"""
    image_path = tmp_path / "source.jpg"
    output_dir = tmp_path / "crops"
    write_image(image_path)

    target_components = {"ebike_full", "front_headlight", "front_basket", "saddle", "seat", "rack", "backrest"}
    for class_id, component_name in MODEL_NAMES.items():
        saved = crop_components_from_image(str(image_path), FakeYolo([class_id]), str(output_dir / str(class_id)))

        assert set(saved) == ({component_name} if component_name in target_components else set())


def test_numeric_order_directories_are_verified_with_two_common_components(tmp_path, capsys):
    """订单号目录必须读取，且少于两个公共部件时不计为有效车辆。"""
    pairs_root = tmp_path / "data" / "pairs"
    valid_root = pairs_root / "2071976149890498561_348822600832821"
    invalid_root = pairs_root / "2071976149890498562_348822600832822"
    for component in ["rack", "saddle"]:
        write_image(valid_root / "reference" / f"{component}.jpg")
        write_image(valid_root / "actual" / f"{component}.jpg")
    write_image(invalid_root / "reference" / "rack.jpg")
    write_image(invalid_root / "actual" / "rack.jpg")

    verify_dataset(str(pairs_root))
    output = capsys.readouterr().out

    assert "Total vehicles with paired data: 1" in output
    assert "seat                    0 pairs" in output
    assert "Need more data for:" in output


def test_dataset_excludes_single_domain_groups_and_has_one_item_per_paired_group(tmp_path):
    """只保留参考图和实拍图齐全的部件组，每个组每轮仅生成一个训练项。"""
    pairs_root = tmp_path / "data" / "pairs"
    paired_root = pairs_root / "2071976149890498561_348822600832821"
    for component in ["rack", "saddle"]:
        write_image(paired_root / "reference" / f"{component}.jpg")
        write_image(paired_root / "actual" / f"{component}.jpg")
    write_image(pairs_root / "2071976149890498562_348822600832822" / "reference" / "seat.jpg")

    dataset = ComponentPairDataset(str(tmp_path / "data"))

    assert len(dataset.samples) == 5
    assert set(dataset.keys) == {
        ("2071976149890498561_348822600832821", "rack"),
        ("2071976149890498561_348822600832821", "saddle"),
    }
    assert len(dataset) == 2


def test_build_training_pairs_writes_manifest_with_input_hashes(tmp_path, monkeypatch):
    """构建训练对时必须保存不可变的权重、清单、参数和统计溯源信息。"""
    records_path = tmp_path / "inspection_records.json"
    model_path = tmp_path / "detector.pt"
    output_root = tmp_path / "pairs"
    records_path.write_text(json.dumps([{"vehicle_id": "order_1", "cert_image": "cert.jpg", "actual_images": ["actual.jpg"]}]))
    model_path.write_bytes(b"detector-weights")

    monkeypatch.setattr(prepare_data, "YOLO", lambda _: object())
    def fake_crop_components(*args, **kwargs):
        Path(args[2]).mkdir(parents=True, exist_ok=True)
        return {
            "rack": {"path": "rack.jpg", "conf": 0.9},
            "saddle": {"path": "saddle.jpg", "conf": 0.8},
        }

    monkeypatch.setattr(prepare_data, "crop_components_from_image", fake_crop_components)

    build_training_pairs(str(records_path), str(model_path), str(output_root), min_components=2)

    manifest = json.loads((output_root / "build_manifest.json").read_text())
    assert manifest["yolo_model_path"] == str(model_path.resolve())
    assert manifest["yolo_model_sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert manifest["inspection_records_json"] == str(records_path.resolve())
    assert manifest["inspection_records_sha256"] == hashlib.sha256(records_path.read_bytes()).hexdigest()
    assert manifest["conf_threshold"] == 0.50
    assert manifest["min_components"] == 2
    assert manifest["stats"] == {"total": 1, "ok": 1, "skipped": 0}
