# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""电动自行车部件实验特征的统一预处理和提取。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


@dataclass(frozen=True)
class PreprocessedCrop:
    """保存实验通道输入及可审计的前景、边缘证据。"""

    gray_image: Image.Image
    mask: np.ndarray
    edges: np.ndarray
    foreground_fallback: str | None


@dataclass(frozen=True)
class ExtractedFeatureBatch:
    """保存同一批裁剪的原始基线、灰度局部特征和预处理证据。"""

    baseline: torch.Tensor
    gray_dino: torch.Tensor
    shape: torch.Tensor
    shape_available: tuple[bool, ...]
    preprocessed: tuple[PreprocessedCrop, ...]


@dataclass(frozen=True)
class ChannelScores:
    """保存基线和实验通道的独立分数及本次实际融合权重。"""

    baseline_similarity: float
    gray_dino_similarity: float
    shape_similarity: float | None
    fused_similarity: float
    channel_gap: float | None
    dino_weight: float
    shape_weight: float
    shape_available: bool


CHANNEL_WEIGHTS = {
    "saddle": (0.35, 0.65),
    "seat": (0.35, 0.65),
    "backrest": (0.40, 0.60),
    "front_basket": (0.45, 0.55),
    "ebike_full": (0.60, 0.40),
    "front_wheel": (0.30, 0.70),
    "rear_wheel": (0.30, 0.70),
}

SHAPE_FEATURE_DIMENSION = 2020
HOG_ORIENTATIONS = 9
HOG_CELL_SIZE = 32
HOG_BLOCK_CELLS = 2


def extract_foreground_mask(image: Image.Image) -> tuple[np.ndarray, str | None]:
    """以白色补边和中心矩形为先验提取部件前景，失败时显式回退为完整裁剪。"""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("部件裁剪不能为空")
    if float(rgb.std()) < 1.0:
        return np.full((height, width), 255, dtype=np.uint8), "foreground_mask_empty"

    padding = max(2, round(min(height, width) * 0.1))
    padded = cv2.copyMakeBorder(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    mask = np.full(padded.shape[:2], cv2.GC_BGD, dtype=np.uint8)
    original_y = slice(padding, padding + height)
    original_x = slice(padding, padding + width)
    mask[original_y, original_x] = cv2.GC_PR_BGD
    center_margin_x = max(1, round(width * 0.25))
    center_margin_y = max(1, round(height * 0.25))
    mask[
        padding + center_margin_y : padding + height - center_margin_y,
        padding + center_margin_x : padding + width - center_margin_x,
    ] = cv2.GC_PR_FGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(padded, mask, None, background_model, foreground_model, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.full((height, width), 255, dtype=np.uint8), "foreground_mask_failed"

    foreground = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    foreground = foreground[padding : padding + height, padding : padding + width]
    if not np.any(foreground):
        return np.full((height, width), 255, dtype=np.uint8), "foreground_mask_empty"
    if np.count_nonzero(foreground) / foreground.size > 0.98:
        return np.full((height, width), 255, dtype=np.uint8), "foreground_mask_full"
    return foreground, None


def letterbox_square(
    gray: np.ndarray,
    mask: np.ndarray,
    size: int = 256,
    content_size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    """保持原始长宽比缩放灰度图和掩码，并居中放入正方形画布。"""
    if gray.ndim != 2 or mask.ndim != 2 or gray.shape != mask.shape:
        raise ValueError("灰度图和前景掩码必须是形状相同的二维数组")
    if not 0 < content_size <= size:
        raise ValueError("内容尺寸必须大于 0 且不能超过画布尺寸")
    height, width = gray.shape
    if height == 0 or width == 0:
        raise ValueError("灰度图不能为空")

    scale = min(content_size / width, content_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized_gray = cv2.resize(gray, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)

    gray_canvas = np.full((size, size), 255, dtype=np.uint8)
    mask_canvas = np.zeros((size, size), dtype=np.uint8)
    offset_x = (size - resized_width) // 2
    offset_y = (size - resized_height) // 2
    gray_canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized_gray
    mask_canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized_mask
    return gray_canvas, mask_canvas


def preprocess_crop(image: Image.Image) -> PreprocessedCrop:
    """把双方部件裁剪统一为灰度、白底、等比例输入，并生成掩码和边缘证据。"""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    foreground_mask, fallback = extract_foreground_mask(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    equalized = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    foreground_gray = np.where(foreground_mask > 0, blurred, 255).astype(np.uint8)
    boxed_gray, boxed_mask = letterbox_square(foreground_gray, foreground_mask)
    edges = cv2.Canny(boxed_gray, threshold1=20, threshold2=80)
    edges[boxed_mask == 0] = 0
    return PreprocessedCrop(
        gray_image=Image.fromarray(boxed_gray, mode="L"),
        mask=boxed_mask,
        edges=edges,
        foreground_fallback=fallback,
    )


def _compute_hog(gray: np.ndarray) -> np.ndarray:
    """使用现有 OpenCV 梯度算子计算 9 方向、2 x 2 block 的 L2-Hys HOG。"""
    image = gray.astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)
    angle = np.mod(angle, 180.0)

    cell_count = gray.shape[0] // HOG_CELL_SIZE
    histograms = np.zeros((cell_count, cell_count, HOG_ORIENTATIONS), dtype=np.float32)
    bin_position = angle / (180.0 / HOG_ORIENTATIONS)
    lower_bin = np.floor(bin_position).astype(np.int32) % HOG_ORIENTATIONS
    upper_bin = (lower_bin + 1) % HOG_ORIENTATIONS
    upper_weight = bin_position - np.floor(bin_position)

    for cell_y in range(cell_count):
        y1 = cell_y * HOG_CELL_SIZE
        y2 = y1 + HOG_CELL_SIZE
        for cell_x in range(cell_count):
            x1 = cell_x * HOG_CELL_SIZE
            x2 = x1 + HOG_CELL_SIZE
            cell_magnitude = magnitude[y1:y2, x1:x2].reshape(-1)
            cell_lower = lower_bin[y1:y2, x1:x2].reshape(-1)
            cell_upper = upper_bin[y1:y2, x1:x2].reshape(-1)
            cell_upper_weight = upper_weight[y1:y2, x1:x2].reshape(-1)
            histograms[cell_y, cell_x] += np.bincount(
                cell_lower,
                weights=cell_magnitude * (1.0 - cell_upper_weight),
                minlength=HOG_ORIENTATIONS,
            ).astype(np.float32)
            histograms[cell_y, cell_x] += np.bincount(
                cell_upper,
                weights=cell_magnitude * cell_upper_weight,
                minlength=HOG_ORIENTATIONS,
            ).astype(np.float32)

    blocks = []
    for block_y in range(cell_count - HOG_BLOCK_CELLS + 1):
        for block_x in range(cell_count - HOG_BLOCK_CELLS + 1):
            block = histograms[
                block_y : block_y + HOG_BLOCK_CELLS,
                block_x : block_x + HOG_BLOCK_CELLS,
            ].reshape(-1)
            block /= np.sqrt(float(np.dot(block, block)) + 1e-12)
            block = np.minimum(block, 0.2)
            block /= np.sqrt(float(np.dot(block, block)) + 1e-12)
            blocks.append(block)
    return np.concatenate(blocks).astype(np.float32)


def extract_shape_feature(gray: np.ndarray, edges: np.ndarray) -> tuple[torch.Tensor, bool]:
    """提取 HOG 和边缘空间分布；零信息输入显式标记为不可用。"""
    if gray.shape != (256, 256) or edges.shape != gray.shape:
        raise ValueError("形状特征要求灰度图和边缘图均为 256 x 256")
    hog_feature = _compute_hog(gray)
    edge_feature = cv2.resize(edges, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1).astype(np.float32)
    feature = torch.from_numpy(np.concatenate((hog_feature, edge_feature)))
    if feature.numel() != SHAPE_FEATURE_DIMENSION:
        raise RuntimeError(f"形状特征维度异常: 期望 {SHAPE_FEATURE_DIMENSION}, 实际 {feature.numel()}")
    norm = torch.linalg.vector_norm(feature)
    if float(norm.item()) == 0.0:
        return feature, False
    return feature / norm, True


def cosine_feature_similarity(reference: torch.Tensor, actual: torch.Tensor) -> float:
    """计算两个一维特征的原始余弦，零向量返回 0 而不进行分数平移。"""
    if reference.ndim != 1 or actual.ndim != 1 or reference.shape != actual.shape:
        raise ValueError("余弦相似度要求两个形状相同的一维特征向量")
    reference_norm = torch.linalg.vector_norm(reference.float())
    actual_norm = torch.linalg.vector_norm(actual.float())
    if float(reference_norm.item()) == 0.0 or float(actual_norm.item()) == 0.0:
        return 0.0
    return float(torch.dot(reference.float() / reference_norm, actual.float() / actual_norm).item())


def compute_channel_scores(
    baseline_ref: torch.Tensor,
    baseline_actual: torch.Tensor,
    gray_ref: torch.Tensor,
    gray_actual: torch.Tensor,
    shape_ref: torch.Tensor,
    shape_actual: torch.Tensor,
    component: str,
    shape_available: bool,
) -> ChannelScores:
    """计算独立通道分数，并仅在形状通道有效时使用部件实验权重。"""
    baseline_similarity = cosine_feature_similarity(baseline_ref, baseline_actual)
    gray_dino_similarity = cosine_feature_similarity(gray_ref, gray_actual)
    if not shape_available:
        return ChannelScores(
            baseline_similarity=baseline_similarity,
            gray_dino_similarity=gray_dino_similarity,
            shape_similarity=None,
            fused_similarity=gray_dino_similarity,
            channel_gap=None,
            dino_weight=1.0,
            shape_weight=0.0,
            shape_available=False,
        )

    shape_similarity = cosine_feature_similarity(shape_ref, shape_actual)
    dino_weight, shape_weight = CHANNEL_WEIGHTS.get(component, (0.5, 0.5))
    return ChannelScores(
        baseline_similarity=baseline_similarity,
        gray_dino_similarity=gray_dino_similarity,
        shape_similarity=shape_similarity,
        fused_similarity=dino_weight * gray_dino_similarity + shape_weight * shape_similarity,
        channel_gap=abs(gray_dino_similarity - shape_similarity),
        dino_weight=dino_weight,
        shape_weight=shape_weight,
        shape_available=True,
    )


class DinoV2FeatureExtractor:
    """使用同一个 DINOv2 ViT-B/14 实例提取 RGB 基线和灰度局部特征。"""

    model_name = "dinov2_vitb14"
    feature_dimension = 768
    patch_grid_size = 16

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

    def _validate_outputs(
        self,
        output: object,
        expected_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """校验官方 forward_features 字段和本 Demo 固定的 ViT-B/14 维度。"""
        cls_tokens = output.get("x_norm_clstoken") if isinstance(output, dict) else None
        patch_tokens = output.get("x_norm_patchtokens") if isinstance(output, dict) else None
        if not isinstance(cls_tokens, torch.Tensor) or cls_tokens.ndim != 2:
            raise RuntimeError("DINOv2 未返回二维 x_norm_clstoken 特征")
        if not isinstance(patch_tokens, torch.Tensor) or patch_tokens.ndim != 3:
            raise RuntimeError("DINOv2 未返回三维 x_norm_patchtokens 特征")
        if cls_tokens.shape != (expected_count, self.feature_dimension):
            raise RuntimeError(
                "DINOv2 CLS 特征维度异常: "
                f"期望 {(expected_count, self.feature_dimension)}, 实际 {tuple(cls_tokens.shape)}"
            )
        expected_patches = self.patch_grid_size**2
        if patch_tokens.shape != (expected_count, expected_patches, self.feature_dimension):
            raise RuntimeError(
                "DINOv2 patch 特征维度异常: "
                f"期望 {(expected_count, expected_patches, self.feature_dimension)}, "
                f"实际 {tuple(patch_tokens.shape)}"
            )
        return cls_tokens, patch_tokens

    def _build_patch_weights(self, preprocessed: Sequence[PreprocessedCrop]) -> torch.Tensor:
        """把可审计前景掩码缩放到 DINOv2 patch 网格并归一化。"""
        weights = []
        for item in preprocessed:
            resized = cv2.resize(
                item.mask.astype(np.float32) / 255.0,
                (self.patch_grid_size, self.patch_grid_size),
                interpolation=cv2.INTER_AREA,
            ).reshape(-1)
            if not np.any(resized):
                resized.fill(1.0)
            resized /= resized.sum()
            weights.append(torch.from_numpy(resized))
        return torch.stack(weights).to(self.device)

    @torch.inference_mode()
    def extract(self, images: Sequence[Image.Image]) -> ExtractedFeatureBatch:
        """批量提取原 RGB CLS 基线和灰度前景加权 patch 融合特征。"""
        if not images:
            raise ValueError("至少需要一张部件图片才能提取 DINOv2 特征")

        preprocessed = tuple(preprocess_crop(image) for image in images)
        rgb_batch = torch.stack([self.transform(image.convert("RGB")) for image in images])
        gray_batch = torch.stack([self.transform(item.gray_image.convert("RGB")) for item in preprocessed])
        combined_batch = torch.cat((rgb_batch, gray_batch)).to(self.device)
        cls_tokens, patch_tokens = self._validate_outputs(
            self.model.forward_features(combined_batch),
            expected_count=len(images) * 2,
        )

        count = len(images)
        baseline = F.normalize(cls_tokens[:count].float(), dim=1)
        gray_cls = cls_tokens[count:].float()
        gray_patches = patch_tokens[count:].float()
        patch_weights = self._build_patch_weights(preprocessed)
        weighted_patches = torch.sum(gray_patches * patch_weights.unsqueeze(-1), dim=1)
        gray_dino = F.normalize(0.4 * gray_cls + 0.6 * weighted_patches, dim=1)
        shape_results = [
            extract_shape_feature(np.asarray(item.gray_image, dtype=np.uint8), item.edges) for item in preprocessed
        ]
        return ExtractedFeatureBatch(
            baseline=baseline.cpu(),
            gray_dino=gray_dino.cpu(),
            shape=torch.stack([feature for feature, _ in shape_results]),
            shape_available=tuple(available for _, available in shape_results),
            preprocessed=preprocessed,
        )
