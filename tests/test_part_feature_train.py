from types import SimpleNamespace

import cv2
import numpy as np

from validate.part_feature_train.domain_adapt import ComponentPairDataset
from validate.part_feature_train.prepare_data import crop_components_from_image, verify_dataset


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


def test_numeric_order_directories_are_loaded_and_zero_coverage_is_reported(tmp_path, capsys):
    """订单号目录应被验证和训练数据集读取，0 对目标类别也必须显示为不足。"""
    pairs_root = tmp_path / "data" / "pairs"
    write_image(pairs_root / "2071976149890498561_348822600832821" / "reference" / "rack.jpg")
    write_image(pairs_root / "2071976149890498561_348822600832821" / "actual" / "rack.jpg")

    verify_dataset(str(pairs_root))
    output = capsys.readouterr().out
    dataset = ComponentPairDataset(str(tmp_path / "data"))

    assert "Total vehicles with paired data: 1" in output
    assert "saddle                  0 pairs" in output
    assert "Need more data for:" in output
    assert len(dataset.samples) == 2
    assert len(dataset) == 8
