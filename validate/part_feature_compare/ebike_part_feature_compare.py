# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""电动自行车备案图与实拍图的部件外观特征比对 Demo。"""

from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

Box = Tuple[float, float, float, float]
ClippedBox = Tuple[int, int, int, int]


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


def classify_similarity(
    score: float, similar_threshold: float, review_threshold: float
) -> Tuple[str, str]:
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


def select_best_detections(detections: Iterable[Detection]) -> Dict[str, Detection]:
    """每个类别只保留最高置信度框，形成确定的单部件匹配基线。"""
    selected: Dict[str, Detection] = {}
    for detection in detections:
        current = selected.get(detection.class_name)
        if current is None or detection.confidence > current.confidence:
            selected[detection.class_name] = detection
    return selected


def pair_detections(
    reference: Dict[str, Detection], actual: Dict[str, Detection]
) -> Tuple[List[Tuple[str, Detection, Detection]], List[Detection], List[Detection]]:
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


def parse_yolo_result(result: Any) -> Dict[str, Detection]:
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

    def __init__(self, device: str, model: Optional[torch.nn.Module] = None):
        """初始化特征模型；测试可注入模型以隔离网络下载。"""
        self.device = torch.device(device)
        if model is None:
            try:
                model = torch.hub.load("facebookresearch/dinov2", self.model_name, trust_repo=True)
            except Exception as error:
                raise RuntimeError(
                    "DINOv2 ViT-B/14 加载失败，请检查网络或 ~/.cache/torch/hub 缓存"
                ) from error
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
            raise RuntimeError(
                f"DINOv2 特征维度异常: 期望 {self.feature_dimension}, 实际 {features.shape[1]}"
            )
        return F.normalize(features.float(), dim=1).cpu()


def compare_matched_parts(
    matched: Sequence[Tuple[str, Detection, Detection]],
    reference_image: Image.Image,
    actual_image: Image.Image,
    feature_extractor: Any,
    similar_threshold: float,
    review_threshold: float,
) -> List[MatchedPart]:
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


def _detection_payload(detection: Detection) -> Dict[str, Any]:
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
) -> Dict[str, Any]:
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

    unmatched_payload = [
        {**_detection_payload(detection), "side": "reference_only"} for detection in reference_only
    ]
    unmatched_payload.extend(
        {**_detection_payload(detection), "side": "actual_only"} for detection in actual_only
    )
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
