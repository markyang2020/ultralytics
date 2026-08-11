# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""从三批工单构建可复现的DINOv3增量训练、验证和验收数据集。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from math import ceil, floor
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_NAME = "reference_3c.jpg"
ACTUAL_NAME = "ebikeFront.jpg"
COMPONENT_NAME = "ebike_full"
SPLIT_SEED = 20260811
DETECTION_THRESHOLD = 0.05


@dataclass(frozen=True)
class SourcePair:
    """保存同一车辆的合格证整车图和E码通正视图。"""

    vin: str
    reference_path: Path
    actual_path: Path
    reference_sha256: str | None = None
    actual_sha256: str | None = None
    reference_crop_sha256: str | None = None
    actual_crop_sha256: str | None = None


def file_sha256(path: Path) -> str:
    """计算文件SHA-256，保证拆分及裁剪结果可审计。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_image(path: Path) -> None:
    """校验图片可由Pillow完整解码。"""
    with Image.open(path) as image:
        image.verify()


def discover_source_pairs(source_root: Path) -> tuple[list[SourcePair], list[dict[str, str]]]:
    """发现全部工单，缺少左正主训练正样本的工单进入排除清单。"""
    if not source_root.is_dir():
        raise FileNotFoundError(f"原始配对目录不存在：{source_root}")

    samples = []
    excluded = []
    vehicle_dirs = sorted(
        vehicle_dir
        for batch_dir in source_root.iterdir()
        if batch_dir.is_dir()
        for vehicle_dir in batch_dir.iterdir()
        if vehicle_dir.is_dir()
    )
    for vehicle_dir in vehicle_dirs:
        reference_path = vehicle_dir / REFERENCE_NAME
        actual_path = vehicle_dir / ACTUAL_NAME
        missing = [path.name for path in (reference_path, actual_path) if not path.is_file()]
        if missing:
            excluded.append({"order_id": vehicle_dir.name, "reason": f"缺少图片：{', '.join(missing)}"})
            continue
        verify_image(reference_path)
        verify_image(actual_path)
        samples.append(
            SourcePair(
                vin=vehicle_dir.name,
                reference_path=reference_path,
                actual_path=actual_path,
                reference_sha256=file_sha256(reference_path),
                actual_sha256=file_sha256(actual_path),
            )
        )

    if len({sample.vin for sample in samples}) != len(samples):
        raise ValueError("原始数据存在重复工单目录名")
    return samples, excluded


def build_duplicate_groups(samples: Sequence[SourcePair]) -> list[list[SourcePair]]:
    """任一侧图片哈希相同即合并车辆，防止重复图片跨分区泄漏。"""
    parent = {sample.vin: sample.vin for sample in samples}

    def find(vin: str) -> str:
        while parent[vin] != vin:
            parent[vin] = parent[parent[vin]]
            vin = parent[vin]
        return vin

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_hash: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        hashes = (
            sample.reference_sha256 or f"unique-reference-{sample.vin}",
            sample.actual_sha256 or f"unique-actual-{sample.vin}",
            sample.reference_crop_sha256 or f"unique-reference-crop-{sample.vin}",
            sample.actual_crop_sha256 or f"unique-actual-crop-{sample.vin}",
        )
        for image_hash in hashes:
            by_hash[image_hash].append(sample.vin)
    for vins in by_hash.values():
        for vin in vins[1:]:
            union(vins[0], vin)

    samples_by_root: dict[str, list[SourcePair]] = defaultdict(list)
    for sample in samples:
        samples_by_root[find(sample.vin)].append(sample)
    return sorted(
        (sorted(group, key=lambda sample: sample.vin) for group in samples_by_root.values()),
        key=lambda group: group[0].vin,
    )


def duplicate_group_id(group: Sequence[SourcePair]) -> str:
    """根据组内车辆编号生成稳定重复组标识。"""
    digest = hashlib.sha256("\n".join(sample.vin for sample in group).encode()).hexdigest()[:12]
    return f"dup_{digest}"


def _select_exact_groups(groups: Sequence[list[SourcePair]], target: int, seed: int) -> list[list[SourcePair]] | None:
    """用子集和动态规划选择车辆总数恰好等于目标值的完整重复组。"""
    candidates = list(groups)
    random.Random(seed).shuffle(candidates)
    choices: dict[int, list[list[SourcePair]]] = {0: []}
    for group in candidates:
        group_size = len(group)
        for current in sorted(choices, reverse=True):
            new_total = current + group_size
            if new_total <= target and new_total not in choices:
                choices[new_total] = [*choices[current], group]
    return choices.get(target)


def _split_balance_score(split_groups: dict[str, Sequence[list[SourcePair]]]) -> float:
    """计算前6位分布相对全量分布的偏差，选择更有代表性的确定性拆分。"""
    all_samples = [sample for groups in split_groups.values() for group in groups for sample in group]
    all_prefixes = Counter(sample.vin[:6] for sample in all_samples)
    total = len(all_samples)
    score = 0.0
    for split_name in ("validation", "acceptance"):
        samples = [sample for group in split_groups[split_name] for sample in group]
        counts = Counter(sample.vin[:6] for sample in samples)
        ratio = len(samples) / total
        score += sum(abs(counts[prefix] - count * ratio) for prefix, count in all_prefixes.items())
        # 验证和验收优先容纳更多独立图组，避免少数重复图片在指标中占过高权重。
        score += 2.0 * sum((len(group) - 1) ** 2 for group in split_groups[split_name])
    return score


def calculate_split_counts(sample_count: int) -> dict[str, int]:
    """按约80%/10%/10%计算训练、验证和独立验收工单数量。"""
    holdout_count = round(sample_count * 0.10)
    return {
        "train": sample_count - 2 * holdout_count,
        "validation": holdout_count,
        "acceptance": holdout_count,
    }


def stratified_split(samples: Sequence[SourcePair], split_counts: dict[str, int]) -> dict[str, list[SourcePair]]:
    """重复图组不拆散，在固定工单配额下搜索代表性较好的确定性拆分。"""
    expected_count = sum(split_counts.values())
    if len(samples) != expected_count:
        raise ValueError(f"有效样本数量应为{expected_count}，实际为{len(samples)}")

    duplicate_groups = build_duplicate_groups(samples)
    best = None
    best_score = float("inf")
    for attempt in range(512):
        acceptance_groups = _select_exact_groups(duplicate_groups, split_counts["acceptance"], SPLIT_SEED + attempt)
        if acceptance_groups is None:
            continue
        acceptance_ids = {id(group) for group in acceptance_groups}
        remaining_groups = [group for group in duplicate_groups if id(group) not in acceptance_ids]
        validation_groups = _select_exact_groups(
            remaining_groups, split_counts["validation"], SPLIT_SEED + 1000 + attempt
        )
        if validation_groups is None:
            continue
        validation_ids = {id(group) for group in validation_groups}
        train_groups = [group for group in remaining_groups if id(group) not in validation_ids]
        candidate = {
            "train": train_groups,
            "validation": validation_groups,
            "acceptance": acceptance_groups,
        }
        score = _split_balance_score(candidate)
        if score < best_score:
            best = candidate
            best_score = score
    if best is None:
        raise RuntimeError(f"无法在不拆散重复图组的前提下满足拆分配额：{split_counts}")

    result = {
        split_name: sorted((sample for group in best[split_name] for sample in group), key=lambda sample: sample.vin)
        for split_name in split_counts
    }
    actual_counts = {name: len(values) for name, values in result.items()}
    if actual_counts != split_counts:
        raise RuntimeError(f"拆分数量异常：{actual_counts}")
    return result


def build_derangement(samples: Sequence[SourcePair], seed: int) -> list[SourcePair]:
    """生成一一固定负样本，并禁止同车或同一重复图组互相配对。"""
    if len(samples) < 2:
        raise ValueError("生成负样本至少需要2组车辆")
    groups = build_duplicate_groups(samples)
    group_by_vin = {sample.vin: index for index, group in enumerate(groups) for sample in group}
    randomizer = random.Random(seed)
    candidate_indexes = {}
    for reference_index, reference in enumerate(samples):
        candidates = [
            actual_index
            for actual_index, actual in enumerate(samples)
            if group_by_vin[reference.vin] != group_by_vin[actual.vin]
        ]
        randomizer.shuffle(candidates)
        candidate_indexes[reference_index] = candidates

    matched_reference_by_actual: dict[int, int] = {}

    def assign(reference_index: int, visited: set[int]) -> bool:
        for actual_index in candidate_indexes[reference_index]:
            if actual_index in visited:
                continue
            visited.add(actual_index)
            current_reference = matched_reference_by_actual.get(actual_index)
            if current_reference is None or assign(current_reference, visited):
                matched_reference_by_actual[actual_index] = reference_index
                return True
        return False

    reference_order = sorted(range(len(samples)), key=lambda index: len(candidate_indexes[index]))
    for reference_index in reference_order:
        if not assign(reference_index, set()):
            raise RuntimeError("无法在排除重复图片的前提下生成一一负样本")
    actual_by_reference = {reference: actual for actual, reference in matched_reference_by_actual.items()}
    negatives = [samples[actual_by_reference[index]] for index in range(len(samples))]
    return negatives


def load_cached_detections(path: Path | None) -> dict[str, Any]:
    """读取可续跑检测缓存；首次运行或未配置时返回空字典。"""
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    detections = payload.get("detections")
    if not isinstance(detections, dict):
        raise TypeError(f"检测缓存缺少detections：{path}")
    return detections


def save_detection_cache(path: Path, detections: dict[str, Any]) -> None:
    """原子写入阶段性检测结果，长任务中断后可以继续。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps({"schema_version": "1.0", "detections": detections}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def run_detector(
    image_paths: Sequence[Path],
    detector_path: Path,
    device: str,
    detections: dict[str, Any],
    cache_path: Path | None,
) -> dict[str, Any]:
    """补齐缺失的YOLO整车检测结果，并在每批完成后更新可续跑缓存。"""
    if not detector_path.is_file():
        raise FileNotFoundError(f"YOLO权重不存在：{detector_path}")
    from ultralytics import YOLO

    paths = [path for path in image_paths if str(path.resolve()) not in detections]
    if not paths:
        return detections
    detector = YOLO(detector_path)
    for start in range(0, len(paths), 24):
        chunk = paths[start : start + 24]
        results = detector.predict(
            source=[str(path) for path in chunk],
            imgsz=640,
            # 使用低阈值保证小目标整车也能进入裁剪，低置信度样本会写入质量复核清单。
            conf=DETECTION_THRESHOLD,
            device=device,
            batch=8,
            verbose=False,
        )
        if len(results) != len(chunk):
            raise RuntimeError(f"YOLO结果数量异常：期望{len(chunk)}，实际{len(results)}")
        for image_path, result in zip(chunk, results):
            best = None
            for class_id, confidence, box in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
                class_id_value = int(class_id.item())
                class_name = str(result.names.get(class_id_value, class_id_value))
                if class_name != COMPONENT_NAME:
                    continue
                confidence_value = float(confidence.item())
                if best is None or confidence_value > best["confidence"]:
                    best = {
                        "detected": True,
                        "confidence": confidence_value,
                        "box_xyxy": [float(value) for value in box.tolist()],
                    }
            detections[str(image_path.resolve())] = best or {"detected": False}
        if cache_path is not None:
            save_detection_cache(cache_path, detections)
        print(f"[检测] {min(start + len(chunk), len(paths))}/{len(paths)}", flush=True)
    return detections


def resolve_detection(detections: dict[str, Any], image_path: Path) -> dict[str, Any]:
    """兼容检测摘要中的绝对路径，并确保当前图片存在有效整车框。"""
    key = str(image_path.resolve())
    detection = detections.get(key)
    if detection is None:
        # 服务器目录变化时可按车辆编号和文件名回退匹配，但必须唯一。
        suffix = f"/{image_path.parent.name}/{image_path.name}"
        matches = [value for cached_path, value in detections.items() if cached_path.endswith(suffix)]
        if len(matches) == 1:
            detection = matches[0]
    if not isinstance(detection, dict) or not detection.get("detected"):
        raise RuntimeError(f"没有可用整车检测结果：{image_path}")
    box = detection.get("box_xyxy")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"整车检测框格式错误：{image_path}")
    return detection


def has_detection(detections: dict[str, Any], image_path: Path) -> bool:
    """判断图片是否存在落在图像范围内的有效整车框。"""
    detection = detections.get(str(image_path.resolve()))
    if not isinstance(detection, dict) or not detection.get("detected"):
        return False
    box = detection.get("box_xyxy")
    if not isinstance(box, list) or len(box) != 4:
        return False
    with Image.open(image_path) as image:
        width, height = image.size
    x1, y1, x2, y2 = (float(value) for value in box)
    return min(width, x2) > max(0.0, x1) and min(height, y2) > max(0.0, y1)


def extract_vehicle_crop(source_path: Path, detection: dict[str, Any], margin: float) -> Image.Image:
    """按整车检测框外扩并返回RGB裁剪图。"""
    with Image.open(source_path) as source:
        image = source.convert("RGB")
        x1, y1, x2, y2 = (float(value) for value in detection["box_xyxy"])
        margin_x = (x2 - x1) * margin
        margin_y = (y2 - y1) * margin
        box = (
            max(0, floor(x1 - margin_x)),
            max(0, floor(y1 - margin_y)),
            min(image.width, ceil(x2 + margin_x)),
            min(image.height, ceil(y2 + margin_y)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"整车裁剪框无效：{source_path} {box}")
        return image.crop(box)


def crop_vehicle(source_path: Path, destination_path: Path, detection: dict[str, Any], margin: float) -> None:
    """保存与训练输入一致的RGB整车裁剪。"""
    crop = extract_vehicle_crop(source_path, detection, margin)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(destination_path, format="JPEG", quality=95, subsampling=0)


def calculate_crop_sha256(source_path: Path, detection: dict[str, Any], margin: float) -> str:
    """在拆分前计算最终JPEG裁剪哈希，防止不同原图裁出相同车辆后跨分区。"""
    buffer = io.BytesIO()
    extract_vehicle_crop(source_path, detection, margin).save(buffer, format="JPEG", quality=95, subsampling=0)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def calculate_box_area_ratio(source_path: Path, detection: dict[str, Any]) -> float:
    """计算整车框占原图比例，用于发现车辆过小等需要人工复核的样本。"""
    with Image.open(source_path) as image:
        width, height = image.size
    x1, y1, x2, y2 = (float(value) for value in detection["box_xyxy"])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1, width * height)


def build_quality_flags(domain: str, detection: dict[str, Any], box_area_ratio: float) -> list[str]:
    """只标记疑似质量问题，不凭检测分数自动删除业务正样本。"""
    flags = []
    if float(detection["confidence"]) < 0.60:
        flags.append(f"{domain}_low_detection_confidence")
    if box_area_ratio < 0.08:
        flags.append(f"{domain}_vehicle_too_small")
    return flags


def build_labeled_pairs(split_root: Path, samples: Sequence[SourcePair], seed: int) -> list[dict[str, Any]]:
    """为验证或验收分区生成数量相等的同车正样本和固定错车负样本。"""
    negatives = build_derangement(samples, seed)
    records = []
    for sample, negative in zip(samples, negatives):
        reference = f"pairs/{sample.vin}/reference/{COMPONENT_NAME}.jpg"
        positive_actual = f"pairs/{sample.vin}/actual/{COMPONENT_NAME}.jpg"
        negative_actual = f"pairs/{negative.vin}/actual/{COMPONENT_NAME}.jpg"
        records.extend(
            (
                {
                    "pair_id": f"{sample.vin}_positive",
                    "reference_vin": sample.vin,
                    "actual_vin": sample.vin,
                    "ref": reference,
                    "actual": positive_actual,
                    "component": COMPONENT_NAME,
                    "label": 1,
                },
                {
                    "pair_id": f"{sample.vin}_negative",
                    "reference_vin": sample.vin,
                    "actual_vin": negative.vin,
                    "ref": reference,
                    "actual": negative_actual,
                    "component": COMPONENT_NAME,
                    "label": 0,
                },
            )
        )
    (split_root / "labeled_pairs.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return records


def prefix_distribution(samples: Sequence[SourcePair]) -> dict[str, int]:
    """统计车辆编号前6位分布，便于复核拆分代表性。"""
    return dict(sorted(Counter(sample.vin[:6] for sample in samples).items()))


def build_dataset(
    source_root: Path,
    output_root: Path,
    detector_path: Path,
    *,
    detector_device: str = "cpu",
    cached_detections_path: Path | None = None,
    crop_margin: float = 0.03,
) -> dict[str, Any]:
    """完成源数据审核、确定性拆分、整车裁剪、固定负样本和清单落盘。"""
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，为防止覆盖请先更换路径：{output_root}")
    if not 0.0 <= crop_margin <= 0.25:
        raise ValueError("crop_margin必须在0到0.25之间")

    all_samples, excluded_records = discover_source_pairs(source_root)
    if len(all_samples) < 10:
        raise ValueError(f"有效左正正样本不足10组，当前发现{len(all_samples)}组")
    source_order_count = len(all_samples) + len(excluded_records)
    detections = load_cached_detections(cached_detections_path)
    main_paths = [path for sample in all_samples for path in (sample.reference_path, sample.actual_path)]
    detections = run_detector(main_paths, detector_path, detector_device, detections, cached_detections_path)

    valid_samples = []
    for sample in all_samples:
        missing_domains = [
            domain
            for domain, path in (("reference_3c", sample.reference_path), ("ebikeFront", sample.actual_path))
            if not has_detection(detections, path)
        ]
        if missing_domains:
            excluded_records.append(
                {"order_id": sample.vin, "reason": f"YOLO未检测到整车：{', '.join(missing_domains)}"}
            )
            continue
        valid_samples.append(sample)

    valid_samples = [
        replace(
            sample,
            reference_crop_sha256=calculate_crop_sha256(
                sample.reference_path, resolve_detection(detections, sample.reference_path), crop_margin
            ),
            actual_crop_sha256=calculate_crop_sha256(
                sample.actual_path, resolve_detection(detections, sample.actual_path), crop_margin
            ),
        )
        for sample in valid_samples
    ]

    split_counts = calculate_split_counts(len(valid_samples))
    split_samples = stratified_split(valid_samples, split_counts)
    valid_samples = [sample for values in split_samples.values() for sample in values]
    duplicate_groups = build_duplicate_groups(valid_samples)
    duplicate_group_by_vin = {sample.vin: duplicate_group_id(group) for group in duplicate_groups for sample in group}
    temporary_parent = output_root.parent
    temporary_parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=temporary_parent))
    try:
        manifest_records = []
        quality_by_vin: dict[str, set[str]] = defaultdict(set)
        for split_name, samples in split_samples.items():
            for sample in sorted(samples, key=lambda item: item.vin):
                for domain, source_path in (("reference", sample.reference_path), ("actual", sample.actual_path)):
                    detection = resolve_detection(detections, source_path)
                    destination = temporary_root / split_name / "pairs" / sample.vin / domain / f"{COMPONENT_NAME}.jpg"
                    crop_vehicle(source_path, destination, detection, crop_margin)
                    box_area_ratio = calculate_box_area_ratio(source_path, detection)
                    quality_flags = build_quality_flags(domain, detection, box_area_ratio)
                    quality_by_vin[sample.vin].update(quality_flags)
                    manifest_records.append(
                        {
                            "vin": sample.vin,
                            "duplicate_group": duplicate_group_by_vin[sample.vin],
                            "split": split_name,
                            "domain": domain,
                            "source_path": str(source_path.resolve()),
                            "source_sha256": file_sha256(source_path),
                            "crop_path": str(destination.relative_to(temporary_root)),
                            "crop_sha256": file_sha256(destination),
                            "detector_confidence": round(float(detection["confidence"]), 6),
                            "box_area_ratio": round(box_area_ratio, 6),
                            "quality_flags": quality_flags,
                            "box_xyxy": [round(float(value), 2) for value in detection["box_xyxy"]],
                        }
                    )

        validation_records = build_labeled_pairs(
            temporary_root / "validation", split_samples["validation"], SPLIT_SEED + 1
        )
        acceptance_records = build_labeled_pairs(
            temporary_root / "acceptance", split_samples["acceptance"], SPLIT_SEED + 2
        )

        manifest = {
            "schema_version": "1.0",
            "source_root": str(source_root.resolve()),
            "source_order_count": source_order_count,
            "source_pair_count": len(all_samples),
            "valid_pair_count": len(valid_samples),
            "excluded_pair_count": len(excluded_records),
            "split_seed": SPLIT_SEED,
            "split_method": "任一侧原图哈希相同的工单先合组，再满足固定车辆配额；同组禁止跨分区",
            "split_counts": {name: len(samples) for name, samples in split_samples.items()},
            "split_prefix_distribution": {
                name: prefix_distribution(samples) for name, samples in split_samples.items()
            },
            "validation_pairs": {
                "positive": sum(record["label"] == 1 for record in validation_records),
                "negative": sum(record["label"] == 0 for record in validation_records),
            },
            "acceptance_pairs": {
                "positive": sum(record["label"] == 1 for record in acceptance_records),
                "negative": sum(record["label"] == 0 for record in acceptance_records),
            },
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_group_size_distribution": dict(
                sorted(Counter(len(group) for group in duplicate_groups).items())
            ),
            "negative_pairing": "固定一一最大匹配，排除同车和同一重复图组",
            "detector_path": str(detector_path.resolve()),
            "detector_sha256": file_sha256(detector_path),
            "detector_device": detector_device,
            "detection_threshold": DETECTION_THRESHOLD,
            "crop_margin": crop_margin,
            "preprocessing": "YOLO ebike_full检测框外扩3%后保存RGB矩形裁剪，不执行前景分割",
            "quality_review_count": sum(bool(flags) for flags in quality_by_vin.values()),
            "component": COMPONENT_NAME,
            "excluded": excluded_records,
            "records": manifest_records,
        }
        (temporary_root / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (temporary_root / "split_assignments.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("vin", "split", "prefix6", "duplicate_group"))
            for split_name, samples in split_samples.items():
                for sample in sorted(samples, key=lambda item: item.vin):
                    writer.writerow((sample.vin, split_name, sample.vin[:6], duplicate_group_by_vin[sample.vin]))
            for record in sorted(excluded_records, key=lambda item: item["order_id"]):
                order_id = record["order_id"]
                writer.writerow((order_id, "excluded", order_id[:6], "excluded"))
        with (temporary_root / "quality_review.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("order_id", "split", "quality_flags"))
            split_by_vin = {sample.vin: split for split, samples in split_samples.items() for sample in samples}
            for vin, flags in sorted(quality_by_vin.items()):
                if flags:
                    writer.writerow((vin, split_by_vin[vin], ";".join(sorted(flags))))
        temporary_root.replace(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(
        "[完成] "
        f"source={len(all_samples)}, train={len(split_samples['train'])}, "
        f"validation={len(split_samples['validation'])}, acceptance={len(split_samples['acceptance'])}, "
        f"excluded={len(excluded_records)}, quality_review={sum(bool(flags) for flags in quality_by_vin.values())}, "
        f"output={output_root}",
        flush=True,
    )
    return manifest


def main() -> None:
    """使用本机三批工单直接构建YOLO整车裁剪版增量训练数据。"""
    source_checkout = Path("/Users/mark/Workspace/ultralytics-ebike-part-feature-demo")
    source_root = source_checkout / "validate/datasets"
    detector_path = source_checkout / "validate/part_feature_compare/models/20260731_yolo11l_best.pt"
    output_root = SCRIPT_DIR / "data_bbox"
    detector_device = "mps"
    cached_detections_path = SCRIPT_DIR / "detections_1400.json"

    # 当前torchvision的NMS没有MPS实现，允许该算子回退CPU，其余YOLO计算仍使用MPS。
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    build_dataset(
        source_root,
        output_root,
        detector_path,
        detector_device=detector_device,
        cached_detections_path=cached_detections_path,
    )


if __name__ == "__main__":
    main()
