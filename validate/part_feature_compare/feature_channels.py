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
    preprocessed: tuple[PreprocessedCrop, ...]


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
    mask = np.zeros(padded.shape[:2], dtype=np.uint8)
    inset = max(1, padding // 2)
    rect = (inset, inset, padded.shape[1] - 2 * inset, padded.shape[0] - 2 * inset)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(padded, mask, rect, background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT)
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
        return ExtractedFeatureBatch(
            baseline=baseline.cpu(),
            gray_dino=gray_dino.cpu(),
            preprocessed=preprocessed,
        )
