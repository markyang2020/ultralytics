# 电动自行车部件特征比对 Demo V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不覆盖原始 DINOv2 RGB CLS 基线的前提下，增加统一灰度预处理、前景加权 patch token、HOG/边缘形状特征和可审计的实验融合结果。

**Architecture:** 新建 `feature_channels.py` 集中负责图像预处理、DINOv2 双路特征和形状融合，删除主脚本中被替代的单一特征提取实现。现有主脚本继续拥有 YOLO 检测、部件配对、分档、JSON、可视化和 CLI，并同时编排基线与实验结果。

**Tech Stack:** Python、Ultralytics YOLO、PyTorch、torchvision、DINOv2 ViT-B/14、OpenCV、NumPy、Pillow、pytest。

## Global Constraints

- 不修改 `ultralytics/` 核心包，不复制现有 YOLO、输入解析或结果落盘流水线。
- 保留用户已有的默认实拍文件名 `actual_bike_weight.jpg` 修改并同步测试、README，不回退为 `actual_left_front_45.jpg`。
- 双方部件采用同一预处理过程，原始 RGB CLS 数值路径保持不变。
- 使用当前官方 `forward_features()` 返回的 `x_norm_clstoken` 和 `x_norm_patchtokens`，不调用 `get_last_selfattention()`。
- DINOv2 保持 `dinov2_vitb14` 和 768 维，不切换 ViT-L，不增加 `scikit-image` 依赖。
- 不对非负形状余弦执行 `(cos + 1) / 2`，所有实验权重和 fallback 必须写入 JSON。
- 原有 `similarity`、`verdict`、`verdict_text` 继续表示基线；实验结果进入独立 `experimental` 节点。
- 所有新增行为先写失败测试并确认按预期失败，再实现最小代码。

---

### Task 1: 统一部件预处理和可审计证据

**Files:**

- Create: `validate/part_feature_compare/feature_channels.py`
- Create: `validate/part_feature_compare/test_feature_channels.py`

**Interfaces:**

- Produces: `PreprocessedCrop`、`extract_foreground_mask()`、`letterbox_square()`、`preprocess_crop()`。
- `PreprocessedCrop` fields: `gray_image: Image.Image`、`mask: np.ndarray`、`edges: np.ndarray`、`foreground_fallback: str | None`。

- [ ] **Step 1: 写相同输入对称处理、长宽比保持和空前景 fallback 的失败测试**

```python
def test_preprocess_crop_is_identical_for_identical_inputs():
    image = Image.new("RGB", (80, 40), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 10, 60, 30), fill="black")

    first = preprocess_crop(image)
    second = preprocess_crop(image)

    assert np.array_equal(np.asarray(first.gray_image), np.asarray(second.gray_image))
    assert np.array_equal(first.mask, second.mask)
    assert np.array_equal(first.edges, second.edges)


def test_letterbox_square_preserves_foreground_aspect_ratio():
    gray = np.full((40, 80), 255, dtype=np.uint8)
    gray[10:30, 10:70] = 0
    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[10:30, 10:70] = 255

    _, boxed_mask = letterbox_square(gray, mask, size=256, content_size=224)
    ys, xs = np.where(boxed_mask > 0)

    assert (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1) == pytest.approx(3.0, rel=0.03)


def test_preprocess_crop_records_fallback_for_empty_foreground():
    result = preprocess_crop(Image.new("RGB", (40, 40), "white"))

    assert result.foreground_fallback == "foreground_mask_empty"
    assert np.all(result.mask == 255)
```

- [ ] **Step 2: 运行三个测试并确认因接口不存在而失败**

Run:

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest \
    validate/part_feature_compare/test_feature_channels.py \
    -k "preprocess or letterbox" -v
```

Expected: collection FAIL，提示无法导入 `PreprocessedCrop` 或目标函数。

- [ ] **Step 3: 实现最小统一预处理**

```python
@dataclass(frozen=True)
class PreprocessedCrop:
    gray_image: Image.Image
    mask: np.ndarray
    edges: np.ndarray
    foreground_fallback: str | None


def letterbox_square(gray: np.ndarray, mask: np.ndarray, size: int = 256, content_size: int = 224):
    scale = min(content_size / gray.shape[1], content_size / gray.shape[0])
    resized_size = (max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale)))
    resized_gray = cv2.resize(gray, resized_size, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, resized_size, interpolation=cv2.INTER_NEAREST)
    gray_canvas = np.full((size, size), 255, dtype=np.uint8)
    mask_canvas = np.zeros((size, size), dtype=np.uint8)
    x = (size - resized_size[0]) // 2
    y = (size - resized_size[1]) // 2
    gray_canvas[y : y + resized_size[1], x : x + resized_size[0]] = resized_gray
    mask_canvas[y : y + resized_size[1], x : x + resized_size[0]] = resized_mask
    return gray_canvas, mask_canvas
```

`extract_foreground_mask()` 将 RGB 转 BGR，先增加白边，再以包含原裁剪的中心矩形初始化 OpenCV GrabCut，最后裁回原尺寸。前景像素为空时返回全前景掩码和 `foreground_mask_empty`；成功时 fallback 为 `None`。

`preprocess_crop()` 按相同顺序执行前景掩码、灰度转换、CLAHE、`3 x 3` 高斯去噪、白底填充、等比例补边和 Canny。边缘图在掩码外清零。

- [ ] **Step 4: 运行新增测试并确认通过**

Run: 使用 Step 2 相同命令。

Expected: 3 tests PASS，输出无警告。

- [ ] **Step 5: 提交统一预处理**

```bash
git add validate/part_feature_compare/feature_channels.py \
    validate/part_feature_compare/test_feature_channels.py
git commit -m "feat: normalize e-bike part crops"
```

### Task 2: DINOv2 基线和前景加权局部特征

**Files:**

- Modify: `validate/part_feature_compare/feature_channels.py`
- Modify: `validate/part_feature_compare/test_feature_channels.py`
- Modify: `validate/part_feature_compare/ebike_part_feature_compare.py`
- Modify: `validate/part_feature_compare/test_ebike_part_feature_compare.py`

**Interfaces:**

- Consumes: `preprocess_crop()`、`PreprocessedCrop`。
- Produces: `ExtractedFeatureBatch`、`DinoV2FeatureExtractor.extract(images: Sequence[Image.Image]) -> ExtractedFeatureBatch`。
- `ExtractedFeatureBatch` fields: `baseline: torch.Tensor`、`gray_dino: torch.Tensor`、`shape: torch.Tensor`、`shape_available: tuple[bool, ...]`、`preprocessed: tuple[PreprocessedCrop, ...]`。

- [ ] **Step 1: 写基线保持、patch 前景加权和官方字段校验的失败测试**

```python
class _FakeDinoModel(torch.nn.Module):
    def forward_features(self, batch):
        count = batch.shape[0]
        cls = torch.zeros((count, 768), device=batch.device)
        cls[:, 0] = 2.0
        patches = torch.zeros((count, 256, 768), device=batch.device)
        patches[:, :, 1] = 1.0
        return {"x_norm_clstoken": cls, "x_norm_patchtokens": patches}


def test_extractor_returns_parallel_normalized_features():
    extractor = DinoV2FeatureExtractor(device="cpu", model=_FakeDinoModel())

    result = extractor.extract([Image.new("RGB", (40, 20), "black")])

    assert result.baseline.shape == (1, 768)
    assert result.gray_dino.shape == (1, 768)
    assert torch.linalg.vector_norm(result.baseline, dim=1).tolist() == pytest.approx([1.0])
    assert torch.linalg.vector_norm(result.gray_dino, dim=1).tolist() == pytest.approx([1.0])
    assert result.gray_dino[0, 1] > 0


def test_extractor_rejects_missing_patch_tokens():
    class MissingPatchModel(_FakeDinoModel):
        def forward_features(self, batch):
            return {"x_norm_clstoken": super().forward_features(batch)["x_norm_clstoken"]}

    with pytest.raises(RuntimeError, match="x_norm_patchtokens"):
        DinoV2FeatureExtractor(device="cpu", model=MissingPatchModel()).extract([Image.new("RGB", (20, 20))])
```

- [ ] **Step 2: 运行测试并确认旧提取器返回 Tensor 或缺少新接口而失败**

Run:

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest \
    validate/part_feature_compare/test_feature_channels.py \
    -k "extractor" -v
```

Expected: FAIL，提示 `ExtractedFeatureBatch` 或并行字段不存在。

- [ ] **Step 3: 将 DINOv2 提取器移动到特征模块并实现单批次双路推理**

删除主脚本中的 `DinoV2FeatureExtractor` 类及其专用 `torchvision.transforms` 导入，改为从 `feature_channels` 导入。

```python
@dataclass(frozen=True)
class ExtractedFeatureBatch:
    baseline: torch.Tensor
    gray_dino: torch.Tensor
    shape: torch.Tensor
    shape_available: tuple[bool, ...]
    preprocessed: tuple[PreprocessedCrop, ...]


@torch.inference_mode()
def extract(self, images):
    preprocessed = tuple(preprocess_crop(image) for image in images)
    rgb_batch = torch.stack([self.transform(image.convert("RGB")) for image in images])
    gray_batch = torch.stack([self.transform(item.gray_image.convert("RGB")) for item in preprocessed])
    output = self.model.forward_features(torch.cat((rgb_batch, gray_batch)).to(self.device))
```

将输出前半部分 CLS 作为基线。后半部分 CLS 与 patch token 用 `0.4/0.6` 融合；把每个 `256 x 256` 掩码用 `INTER_AREA` 缩放为 `16 x 16`，归一化后作为 patch 权重。校验批量维度、768 维和 256 个 patch，最后对两路特征执行 L2 归一化并返回 CPU Tensor。

- [ ] **Step 4: 更新原有提取器测试并运行全部特征测试**

原测试不再断言 `extract()` 直接返回 Tensor，改为断言 `result.baseline` 数值和维度。模型加载失败测试继续从主脚本导入的重新导出类执行，保证外部错误原因仍保留。

Run:

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest \
    validate/part_feature_compare/test_feature_channels.py \
    validate/part_feature_compare/test_ebike_part_feature_compare.py \
    -k "extractor or dinov2" -v
```

Expected: all selected tests PASS。

- [ ] **Step 5: 提交双路特征提取**

```bash
git add validate/part_feature_compare/feature_channels.py \
    validate/part_feature_compare/test_feature_channels.py \
    validate/part_feature_compare/ebike_part_feature_compare.py \
    validate/part_feature_compare/test_ebike_part_feature_compare.py
git commit -m "feat: extract parallel DINOv2 part features"
```

### Task 3: 形状特征和部件实验融合

**Files:**

- Modify: `validate/part_feature_compare/feature_channels.py`
- Modify: `validate/part_feature_compare/test_feature_channels.py`

**Interfaces:**

- Consumes: `PreprocessedCrop`。
- Produces: `CHANNEL_WEIGHTS`、`ChannelScores`、`extract_shape_feature()`、`cosine_feature_similarity()`、`compute_channel_scores()`。
- `ChannelScores` fields: `baseline_similarity`、`gray_dino_similarity`、`shape_similarity: float | None`、`fused_similarity`、`channel_gap: float | None`、`dino_weight`、`shape_weight`、`shape_available`。

- [ ] **Step 1: 写形状归一化、禁止虚高映射、权重融合和不可用通道重归一化测试**

```python
def test_shape_feature_is_normalized_without_score_remapping():
    gray = np.full((256, 256), 255, dtype=np.uint8)
    cv2.rectangle(gray, (48, 96), (208, 160), 0, thickness=4)
    edges = cv2.Canny(gray, 20, 80)

    feature, available = extract_shape_feature(gray, edges)

    assert available is True
    assert torch.linalg.vector_norm(feature).item() == pytest.approx(1.0)
    assert cosine_feature_similarity(feature, torch.zeros_like(feature)) == 0.0


def test_compute_channel_scores_uses_component_weights():
    scores = compute_channel_scores(
        baseline_ref=torch.tensor([1.0, 0.0]),
        baseline_actual=torch.tensor([0.8, 0.6]),
        gray_ref=torch.tensor([1.0, 0.0]),
        gray_actual=torch.tensor([0.8, 0.6]),
        shape_ref=torch.tensor([1.0, 0.0]),
        shape_actual=torch.tensor([0.4, 0.916515]),
        component="saddle",
        shape_available=True,
    )

    assert scores.gray_dino_similarity == pytest.approx(0.8)
    assert scores.shape_similarity == pytest.approx(0.4, abs=1e-5)
    assert scores.fused_similarity == pytest.approx(0.54, abs=1e-5)
    assert scores.channel_gap == pytest.approx(0.4)


def test_compute_channel_scores_ignores_unavailable_shape_channel():
    scores = compute_channel_scores(
        baseline_ref=torch.tensor([1.0, 0.0]),
        baseline_actual=torch.tensor([0.8, 0.6]),
        gray_ref=torch.tensor([1.0, 0.0]),
        gray_actual=torch.tensor([0.8, 0.6]),
        shape_ref=torch.tensor([1.0, 0.0]),
        shape_actual=torch.tensor([0.4, 0.916515]),
        component="saddle",
        shape_available=False,
    )
    assert scores.fused_similarity == pytest.approx(scores.gray_dino_similarity)
    assert scores.shape_similarity is None
    assert scores.dino_weight == 1.0
    assert scores.shape_weight == 0.0
```

- [ ] **Step 2: 运行测试并确认因形状和融合接口不存在而失败**

Run:

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest \
    validate/part_feature_compare/test_feature_channels.py \
    -k "shape or channel_scores" -v
```

Expected: FAIL，提示目标函数或数据类不存在。

- [ ] **Step 3: 使用 OpenCV 实现 HOG、边缘空间特征和有效通道融合**

```python
CHANNEL_WEIGHTS = {
    "saddle": (0.35, 0.65),
    "seat": (0.35, 0.65),
    "backrest": (0.40, 0.60),
    "front_basket": (0.45, 0.55),
    "ebike_full": (0.60, 0.40),
    "front_wheel": (0.30, 0.70),
    "rear_wheel": (0.30, 0.70),
}
```

`extract_shape_feature()` 使用 `cv2.HOGDescriptor((256, 256), (64, 64), (32, 32), (32, 32), 9)` 生成 1764 维 HOG，再把 Canny 图以 `INTER_AREA` 缩小到 `16 x 16` 并拼接为 2020 维向量。零范数返回零向量和 `False`，否则 L2 归一化并返回 `True`。

`compute_channel_scores()` 对三路归一化向量计算原始余弦。形状可用时使用部件权重；不可用时使用 `(1.0, 0.0)`。`channel_gap` 仅在形状可用时返回绝对差值，不再构造未经标定的 `confidence`。

- [ ] **Step 4: 运行完整特征模块测试并确认通过**

Run:

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest \
    validate/part_feature_compare/test_feature_channels.py -v
```

Expected: all tests PASS。

- [ ] **Step 5: 提交形状与融合逻辑**

```bash
git add validate/part_feature_compare/feature_channels.py \
    validate/part_feature_compare/test_feature_channels.py
git commit -m "feat: fuse semantic and shape part scores"
```

### Task 4: 集成双结果报告、可视化和默认订单输入

**Files:**

- Modify: `validate/part_feature_compare/ebike_part_feature_compare.py`
- Modify: `validate/part_feature_compare/test_ebike_part_feature_compare.py`

**Interfaces:**

- Consumes: `ExtractedFeatureBatch`、`ChannelScores`、`compute_channel_scores()`。
- Produces: 扩展后的 `MatchedPart`、并行 JSON 字段、并行控制台输出和预处理证据文件。

- [ ] **Step 1: 写部件并行分数、报告兼容字段和证据文件的失败测试**

```python
def _build_preprocessed_crop() -> PreprocessedCrop:
    return PreprocessedCrop(
        gray_image=Image.new("L", (256, 256), 255),
        mask=np.full((256, 256), 255, dtype=np.uint8),
        edges=np.zeros((256, 256), dtype=np.uint8),
        foreground_fallback=None,
    )


def build_fixed_matched_part(reference_detection: Detection, actual_detection: Detection) -> MatchedPart:
    return MatchedPart(
        part_name="saddle",
        reference_detection=reference_detection,
        actual_detection=actual_detection,
        reference_crop=Image.new("RGB", (4, 4), "white"),
        actual_crop=Image.new("RGB", (4, 4), "black"),
        similarity=0.8,
        verdict="review",
        verdict_text="疑似更换，需人工复核",
        experimental=ChannelScores(
            baseline_similarity=0.8,
            gray_dino_similarity=0.6,
            shape_similarity=1.0,
            fused_similarity=0.86,
            channel_gap=0.4,
            dino_weight=0.35,
            shape_weight=0.65,
            shape_available=True,
        ),
        experimental_verdict="consistent",
        experimental_verdict_text="部件一致",
        reference_preprocessed=_build_preprocessed_crop(),
        actual_preprocessed=_build_preprocessed_crop(),
    )


class _FixedParallelExtractor:
    def extract(self, images):
        assert len(images) == 2
        return ExtractedFeatureBatch(
            baseline=torch.tensor([[1.0, 0.0], [0.8, 0.6]]),
            gray_dino=torch.tensor([[1.0, 0.0], [0.6, 0.8]]),
            shape=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            shape_available=(True, True),
            preprocessed=(_build_preprocessed_crop(), _build_preprocessed_crop()),
        )


def test_compare_matched_parts_keeps_baseline_and_adds_experimental_scores():
    reference_detection = Detection(5, "saddle", 0.93, (1, 1, 5, 5))
    actual_detection = Detection(5, "saddle", 0.79, (2, 2, 6, 6))
    compared = compare_matched_parts(
        [("saddle", reference_detection, actual_detection)],
        Image.new("RGB", (8, 8), "white"),
        Image.new("RGB", (8, 8), "black"),
        _FixedParallelExtractor(),
        similar_threshold=0.85,
        review_threshold=0.60,
    )

    assert compared[0].similarity == pytest.approx(0.8)
    assert compared[0].experimental.gray_dino_similarity == pytest.approx(0.6)
    assert compared[0].experimental.shape_similarity == pytest.approx(1.0)
    assert compared[0].experimental.fused_similarity == pytest.approx(0.86)
    assert compared[0].verdict == "review"
    assert compared[0].experimental_verdict == "consistent"


def test_build_report_keeps_baseline_contract_and_nests_experimental_result():
    reference_detection = Detection(5, "saddle", 0.93, (1, 1, 5, 5))
    actual_detection = Detection(5, "saddle", 0.79, (2, 2, 6, 6))
    part = MatchedPart(
        part_name="saddle",
        reference_detection=reference_detection,
        actual_detection=actual_detection,
        reference_crop=Image.new("RGB", (4, 4), "white"),
        actual_crop=Image.new("RGB", (4, 4), "black"),
        similarity=0.8,
        verdict="review",
        verdict_text="疑似更换，需人工复核",
        experimental=ChannelScores(
            baseline_similarity=0.8,
            gray_dino_similarity=0.6,
            shape_similarity=1.0,
            fused_similarity=0.86,
            channel_gap=0.4,
            dino_weight=0.35,
            shape_weight=0.65,
            shape_available=True,
        ),
        experimental_verdict="consistent",
        experimental_verdict_text="部件一致",
        reference_preprocessed=_build_preprocessed_crop(),
        actual_preprocessed=_build_preprocessed_crop(),
    )
    report = build_report(
        reference_path=Path("reference.jpg"),
        actual_path=Path("actual.jpg"),
        detector_path=Path("model.pt"),
        detection_threshold=0.6,
        similar_threshold=0.85,
        review_threshold=0.6,
        matched_parts=[part],
        reference_only=[],
        actual_only=[],
    )

    part_payload = report["matched_parts"][0]
    assert part_payload["similarity"] == pytest.approx(0.8)
    assert part_payload["verdict"] == "review"
    assert part_payload["experimental"]["fused_similarity"] == pytest.approx(0.86)
    assert part_payload["experimental"]["weights"] == {"dino": 0.35, "shape": 0.65}
    assert report["verdict"] == report["baseline_verdict"] == "review"
    assert report["experimental_verdict"] == "consistent"


def test_save_visualizations_writes_preprocessing_evidence(tmp_path):
    reference_detection = Detection(5, "saddle", 0.93, (1, 1, 5, 5))
    actual_detection = Detection(5, "saddle", 0.79, (2, 2, 6, 6))
    part = build_fixed_matched_part(reference_detection, actual_detection)
    save_visualizations(
        reference_image=Image.new("RGB", (8, 8), "white"),
        actual_image=Image.new("RGB", (8, 8), "black"),
        reference_detections={"saddle": reference_detection},
        actual_detections={"saddle": actual_detection},
        matched_parts=[part],
        output_dir=tmp_path,
    )
    part_dir = tmp_path / "crops" / "saddle"
    assert (part_dir / "reference_preprocessed.jpg").is_file()
    assert (part_dir / "actual_preprocessed.jpg").is_file()
    assert (part_dir / "reference_mask.png").is_file()
    assert (part_dir / "actual_mask.png").is_file()
    assert (part_dir / "reference_edges.png").is_file()
    assert (part_dir / "actual_edges.png").is_file()
```

- [ ] **Step 2: 运行主脚本测试并确认新增断言失败**

Run:

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest \
    validate/part_feature_compare/test_ebike_part_feature_compare.py -v
```

Expected: FAIL，提示实验字段和证据文件不存在；与用户已有文件名修改相关的旧测试也应明确显示预期差异。

- [ ] **Step 3: 扩展匹配、报告和可视化，删除被替代的单路编排**

`MatchedPart` 增加 `experimental: ChannelScores`、`experimental_verdict`、`experimental_verdict_text`、`reference_preprocessed` 和 `actual_preprocessed`。

`compare_matched_parts()` 一次调用 `extract()`，基线继续写入 `.similarity`，实验融合分数独立分档。`build_report()` 使用同一套最差分档规则分别计算基线和实验订单结果，并输出：

```json
{
    "verdict": "review",
    "baseline_verdict": "review",
    "experimental_verdict": "consistent",
    "matched_parts": [
        {
            "similarity": 0.71,
            "verdict": "review",
            "experimental": {
                "gray_dino_similarity": 0.82,
                "shape_similarity": 0.77,
                "fused_similarity": 0.79,
                "channel_gap": 0.05,
                "shape_available": true,
                "weights": {"dino": 0.35, "shape": 0.65},
                "verdict": "review"
            }
        }
    ]
}
```

可视化保存六个预处理文件，逐部件图显示四个分数，汇总标签缩写为 `base=` 和 `exp=`。`resolve_inputs()` 测试改为创建并断言 `actual_bike_weight.jpg`，同步当前已有修改。

- [ ] **Step 4: 运行两组单元测试并确认通过**

Run:

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest \
    validate/part_feature_compare/test_feature_channels.py \
    validate/part_feature_compare/test_ebike_part_feature_compare.py -v
```

Expected: all tests PASS。

- [ ] **Step 5: 提交流水线集成**

```bash
git add validate/part_feature_compare/ebike_part_feature_compare.py \
    validate/part_feature_compare/test_ebike_part_feature_compare.py
git commit -m "feat: report parallel part comparison scores"
```

### Task 5: 使用说明和真实订单验收

**Files:**

- Modify: `validate/part_feature_compare/README.md`
- Generated but not committed: `validate/part_feature_compare/runs/<订单号>/<运行目录>/`

**Interfaces:**

- Consumes: 完整 CLI 和 JSON 输出。
- Produces: 可执行中文说明、静态验证结果和至少一组真实模型实验结果。

- [ ] **Step 1: 更新 README 的算法、字段、文件和实验边界**

README 必须说明：

- 默认读取 `reference_3c.jpg` 和 `actual_bike_weight.jpg`；显式图片模式仍可选择其他视角。
- 原始 `similarity` 是 RGB CLS 基线，`experimental` 是未标定的灰度局部语义与形状融合。
- 权重是附件方案的实验初值，不是生产参数。
- 如何查看掩码、灰度图和边缘图，判断分数变化是否由正确前景产生。
- 灰度化不能消除视角不可观测性，正样本升分不证明真实改装负样本不会被放过。

- [ ] **Step 2: 运行静态验证**

Run:

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest \
    validate/part_feature_compare/test_feature_channels.py \
    validate/part_feature_compare/test_ebike_part_feature_compare.py -v
/Users/mark/Workspace/ultralytics/.venv/bin/python -m py_compile \
    validate/part_feature_compare/feature_channels.py \
    validate/part_feature_compare/ebike_part_feature_compare.py
/Users/mark/Workspace/ultralytics/.venv/bin/python \
    validate/part_feature_compare/ebike_part_feature_compare.py --help
ruff check validate/part_feature_compare/feature_channels.py \
    validate/part_feature_compare/ebike_part_feature_compare.py \
    validate/part_feature_compare/test_feature_channels.py \
    validate/part_feature_compare/test_ebike_part_feature_compare.py
git diff --check
```

Expected: 所有命令退出码为 0。

- [ ] **Step 3: 运行真实 YOLO 和官方 DINOv2**

Run:

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python \
    validate/part_feature_compare/ebike_part_feature_compare.py \
    --order-dir /Users/mark/Workspace/ultralytics/validate/orders/2071947915234308097_348822502067873 \
    --detector /Users/mark/Workspace/ultralytics/yunst_ai/validate_best_model/yolo11l_12000_best.pt \
    --device cpu \
    --output-dir validate/part_feature_compare/runs/2071947915234308097_348822502067873/v2_validation_20260803
```

Expected: 生成包含 `backrest`、`ebike_full`、`front_basket`、`saddle`、`seat` 中实际共同检出部件的基线和实验分数；具体类别由本次 YOLO 结果决定，不伪造缺失部件。

- [ ] **Step 4: 验证 JSON 和证据文件一致**

Run:

```bash
jq '.matched_parts[] | {part, similarity, verdict, experimental}' \
    validate/part_feature_compare/runs/2071947915234308097_348822502067873/v2_validation_20260803/comparison.json
find validate/part_feature_compare/runs/2071947915234308097_348822502067873/v2_validation_20260803/crops \
    -type f | sort
```

Expected: 每个匹配部件包含基线和实验节点，每个部件目录包含原始裁剪、对照图及六个预处理证据文件。人工打开汇总图和至少两个低分部件目录，确认不是空白图、掩码没有明显吞掉主体、分数文字未被裁断。

- [ ] **Step 5: 提交说明与最终修正**

```bash
git add validate/part_feature_compare/README.md \
    validate/part_feature_compare/feature_channels.py \
    validate/part_feature_compare/ebike_part_feature_compare.py \
    validate/part_feature_compare/test_feature_channels.py \
    validate/part_feature_compare/test_ebike_part_feature_compare.py
git commit -m "docs: explain parallel feature comparison demo"
```

真实运行产物默认不提交；最终报告给出实际分数、测试数量、静态检查结果和未执行的生产级负样本验收边界。
