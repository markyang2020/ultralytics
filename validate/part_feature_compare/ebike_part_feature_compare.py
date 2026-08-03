# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""电动自行车备案图与实拍图的部件外观特征比对 Demo。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil, floor
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms

Box = Tuple[float, float, float, float]
ClippedBox = Tuple[int, int, int, int]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DETECTOR = Path("/Users/mark/Workspace/ultralytics/yunst_ai/validate_best_model/yolo11l_12000_best.pt")
DEFAULT_ORDER_DIR = Path("/Users/mark/Workspace/ultralytics/validate/orders/2071943118060388353_214522621506984")
VERDICT_COLORS = {
    "consistent": (34, 139, 94),
    "review": (222, 143, 0),
    "inconsistent": (196, 55, 55),
}


@dataclass(frozen=True)
class Detection:
    """保存一个部件检测框及其业务匹配字段。"""

    class_id: int
    class_name: str
    confidence: float
    box: Box


@dataclass(frozen=True)
class MatchedPart:
    """保存一对同类部件的裁剪、相似度和分档结果。"""

    part_name: str
    reference_detection: Detection
    actual_detection: Detection
    reference_crop: Image.Image
    actual_crop: Image.Image
    similarity: float
    verdict: str
    verdict_text: str


def classify_similarity(score: float, similar_threshold: float, review_threshold: float) -> tuple[str, str]:
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


def select_best_detections(detections: Iterable[Detection]) -> dict[str, Detection]:
    """每个类别只保留最高置信度框，形成确定的单部件匹配基线。"""
    selected: dict[str, Detection] = {}
    for detection in detections:
        current = selected.get(detection.class_name)
        if current is None or detection.confidence > current.confidence:
            selected[detection.class_name] = detection
    return selected


def pair_detections(
    reference: dict[str, Detection], actual: dict[str, Detection]
) -> tuple[list[tuple[str, Detection, Detection]], list[Detection], list[Detection]]:
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


def parse_yolo_result(result: Any) -> dict[str, Detection]:
    """把 Ultralytics Results 转成按类别选择后的轻量检测对象。"""
    detections = []
    for class_id, confidence, box in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
        class_id_value = int(class_id.item())
        detections.append(
            Detection(
                class_id=class_id_value,
                class_name=str(result.names.get(class_id_value, class_id_value)),
                confidence=float(confidence.item()),
                box=tuple(float(value) for value in box.tolist()),
            )
        )
    return select_best_detections(detections)


def crop_detection(image: Image.Image, detection: Detection) -> Image.Image:
    """按检测框从原图裁剪一个有效 RGB 部件图。"""
    rgb_image = image.convert("RGB")
    box = clip_box(detection.box, width=rgb_image.width, height=rgb_image.height)
    return rgb_image.crop(box)


class DinoV2FeatureExtractor:
    """使用官方 DINOv2 ViT-B/14 提取并归一化 CLS token。"""

    model_name = "dinov2_vitb14"
    feature_dimension = 768

    def __init__(self, device: str, model: torch.nn.Module | None = None):
        """初始化特征模型；测试可注入模型以隔离网络下载。"""
        self.device = torch.device(device)
        if model is None:
            try:
                model = torch.hub.load("facebookresearch/dinov2", self.model_name, trust_repo=True)
            except Exception as error:
                message = f"DINOv2 ViT-B/14 加载失败，请检查网络或 ~/.cache/torch/hub 缓存: {error}"
                raise RuntimeError(message) from error
        self.model = model.to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @torch.inference_mode()
    def extract(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """批量提取部件图的 768 维 L2 归一化特征。"""
        if not images:
            raise ValueError("至少需要一张部件图片才能提取 DINOv2 特征")
        batch = torch.stack([self.transform(image.convert("RGB")) for image in images]).to(self.device)
        output = self.model.forward_features(batch)
        features = output.get("x_norm_clstoken") if isinstance(output, dict) else None
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise RuntimeError("DINOv2 未返回二维 x_norm_clstoken 特征")
        if features.shape[1] != self.feature_dimension:
            raise RuntimeError(f"DINOv2 特征维度异常: 期望 {self.feature_dimension}, 实际 {features.shape[1]}")
        return F.normalize(features.float(), dim=1).cpu()


def compare_matched_parts(
    matched: Sequence[tuple[str, Detection, Detection]],
    reference_image: Image.Image,
    actual_image: Image.Image,
    feature_extractor: Any,
    similar_threshold: float,
    review_threshold: float,
) -> list[MatchedPart]:
    """裁剪公共部件并按固定顺序批量提取、比较双方特征。"""
    if not matched:
        return []

    crops = []
    paired_crops = []
    for part_name, reference_detection, actual_detection in matched:
        reference_crop = crop_detection(reference_image, reference_detection)
        actual_crop = crop_detection(actual_image, actual_detection)
        paired_crops.append((part_name, reference_detection, actual_detection, reference_crop, actual_crop))
        crops.extend((reference_crop, actual_crop))

    features = feature_extractor.extract(crops)
    if features.shape[0] != len(crops):
        raise RuntimeError(f"DINOv2 特征数量异常: 期望 {len(crops)}, 实际 {features.shape[0]}")

    compared = []
    for index, (part_name, reference_detection, actual_detection, reference_crop, actual_crop) in enumerate(
        paired_crops
    ):
        similarity = cosine_similarity(features[index * 2], features[index * 2 + 1])
        verdict, verdict_text = classify_similarity(similarity, similar_threshold, review_threshold)
        compared.append(
            MatchedPart(
                part_name=part_name,
                reference_detection=reference_detection,
                actual_detection=actual_detection,
                reference_crop=reference_crop,
                actual_crop=actual_crop,
                similarity=similarity,
                verdict=verdict,
                verdict_text=verdict_text,
            )
        )
    return compared


def _detection_payload(detection: Detection) -> dict[str, Any]:
    """把检测对象转换为可直接写入 JSON 的字段。"""
    return {
        "class_id": detection.class_id,
        "part": detection.class_name,
        "confidence": round(detection.confidence, 6),
        "box_xyxy": [round(value, 2) for value in detection.box],
    }


def build_report(
    reference_path: Path,
    actual_path: Path,
    detector_path: Path,
    detection_threshold: float,
    similar_threshold: float,
    review_threshold: float,
    matched_parts: Sequence[MatchedPart],
    reference_only: Sequence[Detection],
    actual_only: Sequence[Detection],
) -> dict[str, Any]:
    """构建包含复现实验信息和全部比对证据的结构化报告。"""
    if any(part.verdict == "inconsistent" for part in matched_parts):
        verdict, verdict_text = "inconsistent", "存在部件外观不一致"
    elif any(part.verdict == "review" for part in matched_parts):
        verdict, verdict_text = "review", "存在疑似更换部件，需人工复核"
    elif matched_parts:
        verdict, verdict_text = "consistent", "已比较部件外观一致"
    else:
        verdict, verdict_text = "unable_to_compare", "没有双方共同检测到的部件"

    matched_payload = []
    for part in matched_parts:
        matched_payload.append(
            {
                "part": part.part_name,
                "reference_detection": _detection_payload(part.reference_detection),
                "actual_detection": _detection_payload(part.actual_detection),
                "similarity": round(part.similarity, 6),
                "verdict": part.verdict,
                "verdict_text": part.verdict_text,
            }
        )

    unmatched_payload = [{**_detection_payload(detection), "side": "reference_only"} for detection in reference_only]
    unmatched_payload.extend({**_detection_payload(detection), "side": "actual_only"} for detection in actual_only)
    return {
        "reference_image": str(reference_path.resolve()),
        "actual_image": str(actual_path.resolve()),
        "detector": str(detector_path.resolve()),
        "feature_model": DinoV2FeatureExtractor.model_name,
        "thresholds": {
            "detection": detection_threshold,
            "consistent": similar_threshold,
            "review": review_threshold,
        },
        "verdict": verdict,
        "verdict_text": verdict_text,
        "matched_parts": matched_payload,
        "unmatched_parts": unmatched_payload,
    }


def resolve_inputs(order_dir: Path | None, reference_path: Path | None, actual_path: Path | None) -> tuple[Path, Path]:
    """解析订单或显式图片输入，两个输入模式必须二选一。"""
    has_explicit_input = reference_path is not None or actual_path is not None
    if order_dir is not None and has_explicit_input:
        raise ValueError("--order-dir 与 --reference/--actual 必须二选一")
    if order_dir is None and not has_explicit_input:
        raise ValueError("必须使用 --order-dir，或同时提供 --reference 和 --actual")

    if order_dir is not None:
        reference_path = order_dir / "reference_3c.jpg"
        actual_path = order_dir / "actual_left_front_45.jpg"
    elif reference_path is None or actual_path is None:
        raise ValueError("显式图片模式必须同时提供 --reference 和 --actual")

    for path, description in ((reference_path, "备案图"), (actual_path, "实拍图")):
        if not path.is_file():
            raise FileNotFoundError(f"{description}不存在: {path}")
    return reference_path, actual_path


def validate_thresholds(detection_threshold: float, similar_threshold: float, review_threshold: float) -> None:
    """校验检测和相似度阈值范围及分档顺序。"""
    if not 0.0 <= detection_threshold <= 1.0:
        raise ValueError("检测置信度阈值必须在 0 到 1 之间")
    if not 0.0 <= review_threshold <= 1.0 or not 0.0 <= similar_threshold <= 1.0:
        raise ValueError("相似度阈值必须在 0 到 1 之间")
    if similar_threshold < review_threshold:
        raise ValueError("一致阈值不能低于人工复核阈值")


def save_report(report: dict[str, Any], output_dir: Path) -> Path:
    """使用 UTF-8 保存结构化 JSON 报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comparison.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    return output_path


def _load_font(size: int) -> ImageFont.ImageFont:
    """优先加载可显示中文的系统字体，缺失时退回 Pillow 默认字体。"""
    font_paths = (
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for font_path in font_paths:
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _fit_panel(image: Image.Image, size: tuple[int, int], background: str = "white") -> Image.Image:
    """保持宽高比把图片居中放入固定尺寸面板。"""
    panel = Image.new("RGB", size, background)
    fitted = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    panel.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return panel


def place_label_box(
    anchor_box: ClippedBox,
    label_size: tuple[int, int],
    canvas_size: tuple[int, int],
    occupied: Sequence[ClippedBox],
) -> ClippedBox:
    """在检测框附近放置不越界且尽量不与其他标签重叠的文本框。"""
    canvas_width, canvas_height = canvas_size
    label_width = min(label_size[0], canvas_width)
    label_height = min(label_size[1], canvas_height)
    x1 = max(0, min(anchor_box[0], canvas_width - label_width))

    def build_box(y1: int) -> ClippedBox:
        y1 = max(0, min(y1, canvas_height - label_height))
        return x1, y1, x1 + label_width, y1 + label_height

    def overlaps(candidate: ClippedBox) -> bool:
        return any(
            not (
                candidate[2] <= existing[0]
                or candidate[0] >= existing[2]
                or candidate[3] <= existing[1]
                or candidate[1] >= existing[3]
            )
            for existing in occupied
        )

    candidates = (
        build_box(anchor_box[1] - label_height - 2),
        build_box(anchor_box[1]),
        build_box(anchor_box[3] + 2),
    )
    for candidate in candidates:
        if not overlaps(candidate):
            return candidate

    for y1 in range(0, canvas_height - label_height + 1, label_height + 2):
        candidate = build_box(y1)
        if not overlaps(candidate):
            return candidate
    return build_box(anchor_box[1])


def _annotate_detections(
    image: Image.Image,
    detections: dict[str, Detection],
    matched_parts: dict[str, MatchedPart],
) -> Image.Image:
    """在原图坐标系绘制检测框和本次实际相似度。"""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = _load_font(max(16, min(32, annotated.width // 45)))
    line_width = max(2, annotated.width // 400)
    occupied: list[ClippedBox] = []
    for part_name, detection in sorted(detections.items(), key=lambda item: (item[1].box[1], item[1].box[0])):
        matched_part = matched_parts.get(part_name)
        color = VERDICT_COLORS.get(matched_part.verdict, (80, 140, 210)) if matched_part else (110, 110, 110)
        box = clip_box(detection.box, annotated.width, annotated.height)
        draw.rectangle(box, outline=color, width=line_width)
        label = f"{part_name} conf={detection.confidence:.2f}"
        if matched_part:
            label += f" sim={matched_part.similarity:.3f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        label_size = (text_box[2] - text_box[0] + 6, text_box[3] - text_box[1] + 6)
        label_box = place_label_box(box, label_size, annotated.size, occupied)
        draw.rectangle(label_box, fill=color)
        draw.text((label_box[0] + 3, label_box[1] + 2), label, fill="white", font=font)
        occupied.append(label_box)
    return annotated


def save_visualizations(
    reference_image: Image.Image,
    actual_image: Image.Image,
    reference_detections: dict[str, Detection],
    actual_detections: dict[str, Detection],
    matched_parts: Sequence[MatchedPart],
    output_dir: Path,
) -> dict[str, str]:
    """保存公共部件裁剪、逐部件对照图和双图检测汇总。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    matched_by_name = {part.part_name: part for part in matched_parts}
    font = _load_font(24)
    small_font = _load_font(20)

    for part in matched_parts:
        part_dir = output_dir / "crops" / part.part_name
        part_dir.mkdir(parents=True, exist_ok=True)
        part.reference_crop.save(part_dir / "reference.jpg", quality=95)
        part.actual_crop.save(part_dir / "actual.jpg", quality=95)

        canvas = Image.new("RGB", (1000, 500), "white")
        draw = ImageDraw.Draw(canvas)
        color = VERDICT_COLORS[part.verdict]
        draw.text((20, 14), f"{part.part_name}  similarity={part.similarity:.4f}", fill=color, font=font)
        draw.text((20, 48), part.verdict_text, fill=color, font=small_font)
        canvas.paste(_fit_panel(part.reference_crop, (480, 390)), (10, 95))
        canvas.paste(_fit_panel(part.actual_crop, (480, 390)), (510, 95))
        draw.text((20, 98), f"Reference conf={part.reference_detection.confidence:.3f}", fill="black", font=small_font)
        draw.text((520, 98), f"Actual conf={part.actual_detection.confidence:.3f}", fill="black", font=small_font)
        canvas.save(part_dir / "comparison.jpg", quality=95)

    reference_annotated = _annotate_detections(reference_image, reference_detections, matched_by_name)
    actual_annotated = _annotate_detections(actual_image, actual_detections, matched_by_name)
    summary = Image.new("RGB", (1620, 720), "white")
    summary.paste(_fit_panel(reference_annotated, (790, 650)), (10, 60))
    summary.paste(_fit_panel(actual_annotated, (790, 650)), (820, 60))
    summary_draw = ImageDraw.Draw(summary)
    summary_draw.text((20, 15), "Reference 3C", fill="black", font=font)
    summary_draw.text((830, 15), "Actual left-front 45", fill="black", font=font)
    summary_path = output_dir / "comparison_summary.jpg"
    summary.save(summary_path, quality=95)
    return {
        "summary": str(summary_path.resolve()),
        "crops_dir": str((output_dir / "crops").resolve()),
    }


def _print_report(report: dict[str, Any], report_path: Path) -> None:
    """把最常用结果输出为便于实验核对的中文表格。"""
    print("\n===== 部件外观特征比对结果 =====")
    print(f"{'部件':<20} {'备案置信度':>10} {'实拍置信度':>10} {'相似度':>10}  判定")
    for part in report["matched_parts"]:
        print(
            f"{part['part']:<20} "
            f"{part['reference_detection']['confidence']:>10.4f} "
            f"{part['actual_detection']['confidence']:>10.4f} "
            f"{part['similarity']:>10.4f}  {part['verdict_text']}"
        )
    if report["unmatched_parts"]:
        names = ", ".join(f"{part['part']}({part['side']})" for part in report["unmatched_parts"])
        print(f"未进入 Level 2: {names}")
    print(f"综合结果: {report['verdict_text']}")
    print(f"JSON 报告: {report_path}")
    print(f"汇总图: {report['artifacts']['summary']}")


def run_comparison(
    reference_path: Path,
    actual_path: Path,
    detector_path: Path,
    output_dir: Path,
    detection_threshold: float = 0.60,
    similar_threshold: float = 0.80,
    review_threshold: float = 0.60,
    device: str = "cpu",
    image_size: int = 640,
    detector_model: Any | None = None,
    feature_extractor: Any | None = None,
) -> Path:
    """执行单对图片检测、部件特征比对并保存全部实验结果。"""
    validate_thresholds(detection_threshold, similar_threshold, review_threshold)
    for path, description in (
        (reference_path, "备案图"),
        (actual_path, "实拍图"),
        (detector_path, "YOLO 权重"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description}不存在: {path}")

    try:
        reference_image = Image.open(reference_path).convert("RGB")
        actual_image = Image.open(actual_path).convert("RGB")
    except Exception as error:
        raise ValueError(f"图片无法解码: {error}") from error

    if detector_model is None:
        from ultralytics import YOLO

        detector_model = YOLO(detector_path)
    results = detector_model.predict(
        source=[str(reference_path), str(actual_path)],
        imgsz=image_size,
        conf=detection_threshold,
        device=device,
        verbose=False,
    )
    if len(results) != 2:
        raise RuntimeError(f"YOLO 推理结果数量异常: 期望 2, 实际 {len(results)}")

    reference_detections = parse_yolo_result(results[0])
    actual_detections = parse_yolo_result(results[1])
    matched, reference_only, actual_only = pair_detections(reference_detections, actual_detections)

    matched_parts = []
    if matched:
        if feature_extractor is None:
            feature_extractor = DinoV2FeatureExtractor(device=device)
        matched_parts = compare_matched_parts(
            matched,
            reference_image,
            actual_image,
            feature_extractor,
            similar_threshold,
            review_threshold,
        )

    report = build_report(
        reference_path=reference_path,
        actual_path=actual_path,
        detector_path=detector_path,
        detection_threshold=detection_threshold,
        similar_threshold=similar_threshold,
        review_threshold=review_threshold,
        matched_parts=matched_parts,
        reference_only=reference_only,
        actual_only=actual_only,
    )
    report["artifacts"] = save_visualizations(
        reference_image,
        actual_image,
        reference_detections,
        actual_detections,
        matched_parts,
        output_dir,
    )
    report_path = save_report(report, output_dir)
    _print_report(report, report_path)
    return report_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析单订单部件特征比对命令行参数。"""
    parser = argparse.ArgumentParser(description="电动自行车备案图与实拍图部件外观特征比对 Demo")
    parser.add_argument("--order-dir", type=Path, help="订单目录，读取 reference_3c.jpg 和 actual_left_front_45.jpg")
    parser.add_argument("--reference", type=Path, help="显式指定 3C 合格证官方图")
    parser.add_argument("--actual", type=Path, help="显式指定现场实拍图")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR, help="YOLO 部件检测权重")
    parser.add_argument("--det-conf", type=float, default=0.60, help="YOLO 检测置信度阈值，默认 0.60")
    parser.add_argument("--similar-threshold", type=float, default=0.80, help="部件一致阈值，默认 0.80")
    parser.add_argument("--review-threshold", type=float, default=0.60, help="人工复核阈值，默认 0.60")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理尺寸，默认 640")
    parser.add_argument("--device", default="auto", help="推理设备：auto、cpu、mps 或 CUDA 设备编号")
    parser.add_argument("--output-dir", type=Path, help="结果目录，默认按订单号和运行时间创建")
    return parser.parse_args(argv)


def _resolve_device(device: str) -> str:
    """根据本机能力把 auto 解析为 YOLO 和 DINOv2 均支持的设备。"""
    if device != "auto":
        return f"cuda:{device}" if device.isdigit() else device
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv: Sequence[str] | None = None) -> int:
    """运行命令行 Demo，并把可操作错误转换为非零退出码。"""
    args = parse_args(argv)
    order_dir = args.order_dir
    if order_dir is None and args.reference is None and args.actual is None:
        order_dir = DEFAULT_ORDER_DIR

    try:
        reference_path, actual_path = resolve_inputs(order_dir, args.reference, args.actual)
        device = _resolve_device(args.device)
        order_name = order_dir.name if order_dir is not None else reference_path.stem
        timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir or SCRIPT_DIR / "runs" / order_name / timestamp
        run_comparison(
            reference_path=reference_path,
            actual_path=actual_path,
            detector_path=args.detector,
            output_dir=output_dir,
            detection_threshold=args.det_conf,
            similar_threshold=args.similar_threshold,
            review_threshold=args.review_threshold,
            device=device,
            image_size=args.imgsz,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
