# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_channels import letterbox_square, preprocess_crop


def test_preprocess_crop_is_identical_for_identical_inputs():
    """防止备案图和实拍图经过同一预处理时产生非确定性差异。"""
    image = Image.new("RGB", (80, 40), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 10, 60, 30), fill="black")

    first = preprocess_crop(image)
    second = preprocess_crop(image)

    assert np.array_equal(np.asarray(first.gray_image), np.asarray(second.gray_image))
    assert np.array_equal(first.mask, second.mask)
    assert np.array_equal(first.edges, second.edges)


def test_letterbox_square_preserves_foreground_aspect_ratio():
    """防止部件被直接拉伸成正方形后产生虚假的形状差异。"""
    gray = np.full((40, 80), 255, dtype=np.uint8)
    gray[10:30, 10:70] = 0
    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[10:30, 10:70] = 255

    _, boxed_mask = letterbox_square(gray, mask, size=256, content_size=224)
    ys, xs = np.where(boxed_mask > 0)

    ratio = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
    assert ratio == pytest.approx(3.0, rel=0.03)


def test_preprocess_crop_records_fallback_for_empty_foreground():
    """防止前景提取失败后静默输出空白特征。"""
    result = preprocess_crop(Image.new("RGB", (40, 40), "white"))

    assert result.foreground_fallback == "foreground_mask_empty"
    assert result.mask[128, 128] == 255
    assert result.mask[0, 0] == 0
