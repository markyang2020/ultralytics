# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""电动自行车部件实验特征的统一预处理和提取。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessedCrop:
    """保存实验通道输入及可审计的前景、边缘证据。"""

    gray_image: Image.Image
    mask: np.ndarray
    edges: np.ndarray
    foreground_fallback: str | None


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
