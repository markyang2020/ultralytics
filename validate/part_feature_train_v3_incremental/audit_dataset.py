# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""审计DINOv3整车训练数据的文件哈希、拆分边界和固定正负配对。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from PIL import Image

try:
    from .build_dataset import file_sha256
except ImportError:
    from build_dataset import file_sha256

SCRIPT_DIR = Path(__file__).resolve().parent


def audit_dataset(data_root: Path) -> dict:
    """校验上传或复制后的数据包，任何缺图、串集或哈希变化都立即失败。"""
    manifest_path = data_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"数据清单不存在：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    split_vins = {name: set() for name in ("train", "validation", "acceptance")}
    domain_counts = Counter()
    source_hash_splits = {}
    crop_hash_splits = {}
    duplicate_group_splits = {}
    for record in manifest["records"]:
        split_name = record["split"]
        if split_name not in split_vins:
            raise ValueError(f"未知数据分区：{split_name}")
        split_vins[split_name].add(record["vin"])
        domain_counts[(split_name, record["domain"])] += 1
        source_hash_splits.setdefault(record["source_sha256"], set()).add(split_name)
        crop_hash_splits.setdefault(record["crop_sha256"], set()).add(split_name)
        duplicate_group_splits.setdefault(record["duplicate_group"], set()).add(split_name)
        crop_path = data_root / record["crop_path"]
        if not crop_path.is_file():
            raise FileNotFoundError(f"裁剪图不存在：{crop_path}")
        if file_sha256(crop_path) != record["crop_sha256"]:
            raise ValueError(f"裁剪图哈希变化：{crop_path}")
        with Image.open(crop_path) as image:
            image.verify()

    for split_name, expected_count in manifest["split_counts"].items():
        if len(split_vins[split_name]) != expected_count:
            raise ValueError(f"{split_name}车辆数异常：{len(split_vins[split_name])} != {expected_count}")
        for domain in ("reference", "actual"):
            if domain_counts[(split_name, domain)] != expected_count:
                raise ValueError(f"{split_name}/{domain}图片数异常")
    if split_vins["train"] & split_vins["validation"]:
        raise ValueError("训练集和验证集存在车辆交叉")
    if split_vins["train"] & split_vins["acceptance"]:
        raise ValueError("训练集和验收集存在车辆交叉")
    if split_vins["validation"] & split_vins["acceptance"]:
        raise ValueError("验证集和验收集存在车辆交叉")
    leaked_source_hashes = [key for key, splits in source_hash_splits.items() if len(splits) > 1]
    leaked_crop_hashes = [key for key, splits in crop_hash_splits.items() if len(splits) > 1]
    leaked_duplicate_groups = [group for group, splits in duplicate_group_splits.items() if len(splits) > 1]
    if leaked_source_hashes:
        raise ValueError(f"原图哈希跨分区泄漏：{len(leaked_source_hashes)}组")
    if leaked_crop_hashes:
        raise ValueError(f"裁剪图哈希跨分区泄漏：{len(leaked_crop_hashes)}组")
    if leaked_duplicate_groups:
        raise ValueError(f"重复图关联车辆跨分区泄漏：{len(leaked_duplicate_groups)}组")

    pair_counts = {}
    for split_name in ("validation", "acceptance"):
        records_path = data_root / split_name / "labeled_pairs.json"
        records = json.loads(records_path.read_text(encoding="utf-8"))
        labels = Counter(record["label"] for record in records)
        expected = manifest["split_counts"][split_name]
        if labels != Counter({0: expected, 1: expected}):
            raise ValueError(f"{split_name}正负样本数量异常：{dict(labels)}")
        negative_actual_vins = set()
        for record in records:
            reference_path = data_root / split_name / record["ref"]
            actual_path = data_root / split_name / record["actual"]
            if not reference_path.is_file() or not actual_path.is_file():
                raise FileNotFoundError(f"固定配对引用不存在：{record['pair_id']}")
            if record["label"] == 1 and record["reference_vin"] != record["actual_vin"]:
                raise ValueError(f"正样本不是同车：{record['pair_id']}")
            if record["label"] == 0:
                if record["reference_vin"] == record["actual_vin"]:
                    raise ValueError(f"负样本错误配回同车：{record['pair_id']}")
                positive_actual_path = data_root / split_name / f"pairs/{record['reference_vin']}/actual/ebike_full.jpg"
                negative_hash = file_sha256(actual_path)
                if negative_hash in {file_sha256(reference_path), file_sha256(positive_actual_path)}:
                    raise ValueError(f"负样本与本车正样本图片完全相同：{record['pair_id']}")
                negative_actual_vins.add(record["actual_vin"])
        if len(negative_actual_vins) != expected:
            raise ValueError(f"{split_name}负样本实拍图没有保持一一使用")
        pair_counts[split_name] = {"positive": labels[1], "negative": labels[0]}

    report = {
        "status": "passed",
        "valid_vehicles": sum(len(vins) for vins in split_vins.values()),
        "excluded_vehicles": manifest["excluded_pair_count"],
        "split_counts": {name: len(vins) for name, vins in split_vins.items()},
        "pair_counts": pair_counts,
        "verified_crop_images": sum(domain_counts.values()),
        "duplicate_groups": manifest["duplicate_group_count"],
        "cross_split_source_hashes": 0,
        "cross_split_crop_hashes": 0,
        "train_validation_overlap": len(split_vins["train"] & split_vins["validation"]),
        "train_acceptance_overlap": len(split_vins["train"] & split_vins["acceptance"]),
        "validation_acceptance_overlap": len(split_vins["validation"] & split_vins["acceptance"]),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    """审计脚本同目录下已经拆分好的YOLO整车裁剪数据。"""
    audit_dataset(SCRIPT_DIR / "data_bbox")


if __name__ == "__main__":
    main()
