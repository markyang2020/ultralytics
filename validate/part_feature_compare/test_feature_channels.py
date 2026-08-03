# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_channels import DinoV2FeatureExtractor, letterbox_square, preprocess_crop


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


class _FakeDinoModel(torch.nn.Module):
    """返回官方 forward_features 字段，隔离真实权重和网络。"""

    def forward_features(self, batch):
        count = batch.shape[0]
        cls = torch.zeros((count, 768), device=batch.device)
        cls[:, 0] = 2.0
        patches = torch.zeros((count, 256, 768), device=batch.device)
        patches[:, :, 1] = 1.0
        return {"x_norm_clstoken": cls, "x_norm_patchtokens": patches}


def test_extractor_returns_parallel_normalized_features():
    """防止V2覆盖RGB基线，或灰度局部特征遗漏归一化。"""
    extractor = DinoV2FeatureExtractor(device="cpu", model=_FakeDinoModel())

    result = extractor.extract([Image.new("RGB", (40, 20), "black")])

    assert result.baseline.shape == (1, 768)
    assert result.gray_dino.shape == (1, 768)
    assert torch.linalg.vector_norm(result.baseline, dim=1).tolist() == pytest.approx([1.0])
    assert torch.linalg.vector_norm(result.gray_dino, dim=1).tolist() == pytest.approx([1.0])
    assert result.gray_dino[0, 1] > 0


def test_extractor_rejects_missing_patch_tokens():
    """防止官方模型接口漂移后静默退化成另一套算法。"""

    class MissingPatchModel(_FakeDinoModel):
        def forward_features(self, batch):
            return {"x_norm_clstoken": super().forward_features(batch)["x_norm_clstoken"]}

    extractor = DinoV2FeatureExtractor(device="cpu", model=MissingPatchModel())

    with pytest.raises(RuntimeError, match="x_norm_patchtokens"):
        extractor.extract([Image.new("RGB", (20, 20), "white")])
