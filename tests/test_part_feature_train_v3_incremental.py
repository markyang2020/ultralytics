# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import numpy as np
from PIL import Image

from validate.part_feature_train_v3_incremental import build_dataset
from validate.part_feature_train_v3_incremental.train_common import WholeVehicleTripletDataset


def write_image(path: Path, value: int = 127) -> None:
    """写入可被训练数据读取的最小RGB图片。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)


def test_discover_source_pairs_reads_batches_and_records_missing_images(tmp_path, monkeypatch):
    """两批工单应统一发现，缺少合格证整车图的工单进入排除清单。"""
    complete = tmp_path / "orders_1000/order_1"
    write_image(complete / build_dataset.REFERENCE_NAME)
    write_image(complete / build_dataset.ACTUAL_NAME)
    missing = tmp_path / "orders_200/order_2"
    write_image(missing / build_dataset.ACTUAL_NAME)

    samples, excluded = build_dataset.discover_source_pairs(tmp_path)

    assert [sample.vin for sample in samples] == ["order_1"]
    assert excluded == [{"order_id": "order_2", "reason": "缺少图片：reference_3c.jpg"}]


def test_split_keeps_duplicate_reference_group_in_one_partition():
    """复用同一合格证图的工单不能跨训练、验证和验收集合。"""
    samples = [
        build_dataset.SourcePair(
            vin=f"order_{index}",
            reference_path=Path(f"ref_{index}"),
            actual_path=Path(f"actual_{index}"),
            reference_sha256="shared" if index in (0, 1) else f"ref_{index}",
            actual_sha256=f"actual_{index}",
        )
        for index in range(8)
    ]

    split_counts = {"train": 4, "validation": 2, "acceptance": 2}
    splits = build_dataset.stratified_split(samples, split_counts)
    split_by_order = {sample.vin: split for split, values in splits.items() for sample in values}

    assert {name: len(values) for name, values in splits.items()} == {"train": 4, "validation": 2, "acceptance": 2}
    assert split_by_order["order_0"] == split_by_order["order_1"]


def test_training_epoch_keeps_every_order_with_shared_reference(tmp_path):
    """多个工单复用同一合格证时，每个工单仍应在每轮各参与一次训练。"""
    pairs_root = tmp_path / "pairs"
    for order_id, reference_value, actual_value in (
        ("order_1", 10, 20),
        ("order_2", 10, 30),
        ("order_3", 40, 50),
    ):
        write_image(pairs_root / order_id / "reference/ebike_full.jpg", reference_value)
        write_image(pairs_root / order_id / "actual/ebike_full.jpg", actual_value)

    dataset = WholeVehicleTripletDataset(tmp_path, image_size=16)

    assert len(dataset.samples) == 3
    assert len(dataset.groups) == 2
    assert len(dataset) == 3
    assert dataset.group_index_by_vehicle["order_1"] == dataset.group_index_by_vehicle["order_2"]
