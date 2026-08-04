"""
训练数据准备
核心：你们不需要额外标注任何东西。
     现有的"合格证图 + 查验现场图（哪怕当前比对分数很低的）"就是训练数据。

操作流程：
1. 对合格证图跑YOLO分割 → 裁剪出各部件 → 存入 reference/
2. 对现场查验图跑YOLO分割 → 裁剪出各部件 → 存入 actual/
3. 以 vehicle_id 为文件夹名配对 → 这就是正样本
4. 不同 vehicle_id 之间就是负样本（代码自动生成，不需要手动操作）
"""

import os
import cv2
import json
import shutil
import hashlib
from pathlib import Path
from ultralytics import YOLO


# DINOv2 训练当前需要对齐的部件，类别 ID 必须以实际 YOLO 权重的 names 为准。
TARGET_COMPONENT_NAMES = (
    "ebike_full",
    "front_headlight",
    "front_basket",
    "saddle",
    "seat",
    "rack",
    "backrest",
)
CONF_THRESHOLD = 0.50


def calculate_sha256(file_path: Path) -> str:
    """计算构建输入文件的 SHA-256，保证训练数据可追溯。"""
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_build_manifest(
    output_root: Path, yolo_model_path: Path, inspection_records_path: Path, min_components: int, stats: dict
):
    """保存训练对构建时使用的不可变输入、参数和结果统计。"""
    manifest = {
        "yolo_model_path": str(yolo_model_path.resolve()),
        "yolo_model_sha256": calculate_sha256(yolo_model_path),
        "inspection_records_json": str(inspection_records_path.resolve()),
        "inspection_records_sha256": calculate_sha256(inspection_records_path),
        "conf_threshold": CONF_THRESHOLD,
        "min_components": min_components,
        "stats": stats,
    }
    with (output_root / "build_manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)


def crop_components_from_image(
    image_path: str,
    yolo_model: YOLO,
    output_dir:  str,
    conf_threshold: float = 0.50,
    padding_ratio:  float = 0.10,   # 裁剪时四周留10%边距
):
    """
    对单张图用YOLO分割，裁剪出各部件图像，保存到 output_dir。

    输出文件名：{component_name}.jpg
    如果同一部件检出多个（低概率），取置信度最高的那个。
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [WARN] Cannot read: {image_path}")
        return {}

    h, w = img.shape[:2]
    results = yolo_model(image_path, conf=conf_threshold, verbose=False)

    best_per_comp = {}   # component_name → (conf, crop_img)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        comp_name = yolo_model.names.get(cls_id) if isinstance(yolo_model.names, dict) else yolo_model.names[cls_id]
        if comp_name not in TARGET_COMPONENT_NAMES:
            continue

        # 只保留每个部件置信度最高的检测框
        if comp_name in best_per_comp and conf <= best_per_comp[comp_name][0]:
            continue

        # 裁剪 + padding
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        pad_x = int((x2 - x1) * padding_ratio)
        pad_y = int((y2 - y1) * padding_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = img[y1:y2, x1:x2]
        best_per_comp[comp_name] = (conf, crop)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    saved = {}
    for comp_name, (conf, crop) in best_per_comp.items():
        save_path = os.path.join(output_dir, f"{comp_name}.jpg")
        cv2.imwrite(save_path, crop)
        saved[comp_name] = {"path": save_path, "conf": round(conf, 3)}

    return saved


def build_training_pairs(
    inspection_records_json: str,
    yolo_model_path:         str,
    output_root:             str = "pairs",
    min_components:          int = 2,   # 每辆车至少检出2个部件才纳入训练
):
    """
    从查验记录批量构建训练对。

    inspection_records_json 格式：
    [
      {
        "vehicle_id": "vehicle_001",
        "cert_image":   "/path/to/cert_001.jpg",      ← 合格证图
        "actual_images": ["/path/to/actual_001a.jpg", ← 现场图（可以多张）
                          "/path/to/actual_001b.jpg"]
      },
      ...
    ]
    """
    inspection_records_path = Path(inspection_records_json)
    yolo_model_path = Path(yolo_model_path)
    with inspection_records_path.open() as f:
        records = json.load(f)

    yolo = YOLO(yolo_model_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "ok": 0, "skipped": 0}

    for rec in records:
        vid = rec["vehicle_id"]
        stats["total"] += 1

        vehicle_dir = output_root / vid
        ref_dir     = vehicle_dir / "reference"
        actual_dir  = vehicle_dir / "actual"

        print(f"Processing {vid} ...")

        # 1. 处理合格证图
        ref_saved = crop_components_from_image(
            rec["cert_image"], yolo, str(ref_dir), conf_threshold=CONF_THRESHOLD
        )
        if len(ref_saved) < min_components:
            print(f"  [SKIP] {vid}: only {len(ref_saved)} components in cert image")
            stats["skipped"] += 1
            continue

        # 2. 处理现场图（可多张，部件取置信度最高的）
        all_actual = {}
        for actual_path in rec.get("actual_images", []):
            saved = crop_components_from_image(
                actual_path, yolo, str(actual_dir), conf_threshold=CONF_THRESHOLD
            )
            # 同一部件多张图：保留置信度最高的（覆盖写入，YOLO会选最优）
            all_actual.update(saved)

        if len(all_actual) < min_components:
            print(f"  [SKIP] {vid}: only {len(all_actual)} components in actual images")
            shutil.rmtree(str(vehicle_dir), ignore_errors=True)
            stats["skipped"] += 1
            continue

        # 3. 保存摘要
        summary = {
            "vehicle_id":        vid,
            "cert_image":        rec["cert_image"],
            "ref_components":    ref_saved,
            "actual_components": all_actual,
        }
        with open(str(vehicle_dir / "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        stats["ok"] += 1
        print(f"  ✓ ref={list(ref_saved)}, actual={list(all_actual)}")

    write_build_manifest(output_root, yolo_model_path, inspection_records_path, min_components, stats)
    print(f"\n[Done] total={stats['total']} ok={stats['ok']} "
          f"skipped={stats['skipped']}")
    print(f"Training pairs saved to: {output_root}/")
    print(f"Now run: python train/domain_adapt.py")


def verify_dataset(pairs_root: str):
    """
    验证数据集质量，输出统计报告。
    运行 prepare_data.py 之后先跑这个检查一遍。
    """
    pairs_root = Path(pairs_root)
    vehicle_dirs = [path for path in pairs_root.iterdir() if path.is_dir()]

    comp_stats = {component: 0 for component in TARGET_COMPONENT_NAMES}
    total_vehicles = 0

    for vdir in vehicle_dirs:
        ref_comps    = {p.stem for p in (vdir / "reference").glob("*.jpg")} \
                       if (vdir / "reference").exists() else set()
        actual_comps = {p.stem for p in (vdir / "actual").glob("*.jpg")} \
                       if (vdir / "actual").exists() else set()
        common = ref_comps & actual_comps

        if len(common) < 2:
            continue

        total_vehicles += 1
        for c in common:
            comp_stats[c] = comp_stats.get(c, 0) + 1

    print(f"\n=== Dataset Stats ===")
    print(f"Total vehicles with paired data: {total_vehicles}")
    print(f"Component coverage:")
    for comp, count in sorted(comp_stats.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 2)
        print(f"  {comp:<20} {count:>4} pairs  {bar}")

    # 建议：每个部件至少50对才能收敛
    print(f"\nRecommendation: each component needs >= 50 pairs")
    insufficient = [c for c, n in comp_stats.items() if n < 50]
    if insufficient:
        print(f"  Need more data for: {insufficient}")
    else:
        print(f"  ✓ All components have sufficient data")


if __name__ == "__main__":
    # 示例运行
    build_training_pairs(
        inspection_records_json = "inspection_records.json",
        yolo_model_path         = "models/ebike_yolo.pt",
        output_root             = "data/pairs",
        min_components          = 2,
    )

    verify_dataset("data/pairs")
