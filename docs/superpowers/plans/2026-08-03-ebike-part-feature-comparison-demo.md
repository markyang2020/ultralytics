# 电动自行车部件外观特征比对 Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个使用指定 YOLO 权重和 DINOv2 ViT-B/14 对单个订单的备案图、实拍图进行部件外观特征比对的可运行 Demo。

**Architecture:** 代码放在独立的 `validate/part_feature_compare/` 目录，由单个脚本提供纯比对函数、模型适配、结果落盘、可视化和 CLI。测试通过依赖注入和伪检测结果覆盖纯逻辑，真实验收再加载 YOLO 与官方 DINOv2 权重运行指定订单。

**Tech Stack:** Python 3.8+、Ultralytics YOLO、PyTorch、torchvision、DINOv2 ViT-B/14、Pillow、pytest。

## Global Constraints

- 不修改 `ultralytics/` 核心包，也不复制 YOLO 或 DINOv2 的已有实现。
- 默认检测权重为 `/Users/mark/Workspace/ultralytics/yunst_ai/validate_best_model/yolo11l_12000_best.pt`。
- 默认订单为 `/Users/mark/Workspace/ultralytics/validate/orders/2071943118060388353_214522621506984`。
- 默认检测阈值为 `0.60`，相似度分档阈值为 `0.80` 和 `0.60`。
- 仅比较双方均检测到的同类别最高置信度框；单侧检测结果不得解释为已拆除或已加装。
- DINOv2 使用 ViT-B/14 的 768 维 CLS token，并在余弦相似度前显式执行 L2 归一化。
- 文档、日志、异常信息和必要代码注释使用中文；技术标识和第三方 API 名称保留英文。
- 输出不得覆盖输入图片；实际阈值、框、置信度和模型名称必须写入 JSON。

---

### Task 1: 纯部件配对与相似度逻辑

**Files:**

- Create: `validate/part_feature_compare/ebike_part_feature_compare.py`
- Create: `validate/part_feature_compare/test_ebike_part_feature_compare.py`

**Interfaces:**

- Produces: `Detection`、`MatchedPart`、`clip_box()`、`select_best_detections()`、`pair_detections()`、`cosine_similarity()`、`classify_similarity()`。

- [ ] **Step 1: 写相似度边界、最高置信框和越界裁剪的失败测试**

```python
def test_classify_similarity_uses_closed_threshold_boundaries():
    assert classify_similarity(0.80, 0.80, 0.60)[0] == "consistent"
    assert classify_similarity(0.60, 0.80, 0.60)[0] == "review"
    assert classify_similarity(0.5999, 0.80, 0.60)[0] == "inconsistent"


def test_select_best_detections_keeps_highest_confidence_per_class():
    detections = [
        Detection(0, "ebike_full", 0.70, (1, 1, 8, 8)),
        Detection(0, "ebike_full", 0.95, (2, 2, 9, 9)),
    ]
    assert select_best_detections(detections)["ebike_full"].confidence == 0.95


def test_clip_box_clamps_coordinates_and_rejects_empty_box():
    assert clip_box((-2.2, 1.1, 12.8, 9.2), width=10, height=8) == (0, 1, 10, 8)
    with pytest.raises(ValueError, match="无效检测框"):
        clip_box((4, 4, 4, 7), width=10, height=8)
```

- [ ] **Step 2: 运行测试并确认因接口不存在而失败**

Run: `/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest validate/part_feature_compare/test_ebike_part_feature_compare.py -v`

Expected: FAIL，提示无法导入 `Detection` 或目标函数。

- [ ] **Step 3: 实现最小纯逻辑**

```python
@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]


def classify_similarity(score: float, similar_threshold: float, review_threshold: float) -> tuple[str, str]:
    if score >= similar_threshold:
        return "consistent", "部件一致"
    if score >= review_threshold:
        return "review", "疑似更换，需人工复核"
    return "inconsistent", "部件外观不一致"
```

实现 `clip_box()` 的向下/向上取整和边界限制，`select_best_detections()` 的逐类别最高置信度选择，`pair_detections()` 的公共类别及单侧类别拆分，以及对两个一维张量进行 L2 归一化后的 `cosine_similarity()`。

- [ ] **Step 4: 运行测试并确认通过**

Run: `/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest validate/part_feature_compare/test_ebike_part_feature_compare.py -v`

Expected: PASS。

- [ ] **Step 5: 提交纯逻辑**

```bash
git add validate/part_feature_compare/ebike_part_feature_compare.py \
    validate/part_feature_compare/test_ebike_part_feature_compare.py
git commit -m "feat: add e-bike part comparison core"
```

### Task 2: YOLO 检测与 DINOv2 特征流水线

**Files:**

- Modify: `validate/part_feature_compare/ebike_part_feature_compare.py`
- Modify: `validate/part_feature_compare/test_ebike_part_feature_compare.py`

**Interfaces:**

- Consumes: Task 1 的 `Detection`、`clip_box()`、`pair_detections()`、`cosine_similarity()`、`classify_similarity()`。
- Produces: `parse_yolo_result()`、`crop_detection()`、`DinoV2FeatureExtractor`、`compare_matched_parts()`、`build_report()`。

- [ ] **Step 1: 写 YOLO 结果转换、部件裁剪、归一化特征和报告字段的失败测试**

```python
def test_crop_detection_uses_clipped_box():
    image = Image.new("RGB", (10, 8), "white")
    crop = crop_detection(image, Detection(0, "part", 0.9, (-2, 1, 12, 7)))
    assert crop.size == (10, 6)


def test_cosine_similarity_normalizes_inputs():
    assert cosine_similarity(torch.tensor([2.0, 0.0]), torch.tensor([4.0, 0.0])) == pytest.approx(1.0)


def test_build_report_records_models_thresholds_and_unmatched_parts():
    report = build_report(
        reference_path=Path("reference.jpg"),
        actual_path=Path("actual.jpg"),
        detector_path=Path("model.pt"),
        detection_threshold=0.6,
        similar_threshold=0.8,
        review_threshold=0.6,
        matched_parts=[],
        reference_only=[Detection(1, "rack", 0.9, (1, 1, 4, 4))],
        actual_only=[],
    )
    assert report["feature_model"] == "dinov2_vitb14"
    assert report["thresholds"]["detection"] == 0.6
    assert report["unmatched_parts"][0]["side"] == "reference_only"
```

- [ ] **Step 2: 运行新增测试并确认失败**

Run: `/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest validate/part_feature_compare/test_ebike_part_feature_compare.py -v`

Expected: FAIL，提示流水线接口不存在。

- [ ] **Step 3: 实现模型适配和部件比较**

`parse_yolo_result()` 从 `Results.boxes` 读取 `cls`、`conf`、`xyxy`，先转成 `Detection` 再调用 `select_best_detections()`。`DinoV2FeatureExtractor` 使用：

```python
self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
self.transform = transforms.Compose(
    [
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
```

`extract()` 批量调用 `model.forward_features(batch)["x_norm_clstoken"]`，校验最后一维为 768，并用 `torch.nn.functional.normalize()` 返回 CPU 特征。`compare_matched_parts()` 只处理公共类别，保存裁剪图引用、双方检测信息、相似度和分档。

- [ ] **Step 4: 运行测试并确认通过**

Run: `/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest validate/part_feature_compare/test_ebike_part_feature_compare.py -v`

Expected: PASS，测试不得下载真实模型。

- [ ] **Step 5: 提交流水线实现**

```bash
git add validate/part_feature_compare/ebike_part_feature_compare.py \
    validate/part_feature_compare/test_ebike_part_feature_compare.py
git commit -m "feat: compare detected parts with DINOv2"
```

### Task 3: CLI、JSON 和可视化证据

**Files:**

- Modify: `validate/part_feature_compare/ebike_part_feature_compare.py`
- Modify: `validate/part_feature_compare/test_ebike_part_feature_compare.py`

**Interfaces:**

- Consumes: Task 2 的检测、特征、报告接口。
- Produces: `parse_args()`、`resolve_inputs()`、`resolve_device()`、`save_report()`、`save_part_visualizations()`、`save_summary_visualization()`、`run_comparison()`、`main()`。

- [ ] **Step 1: 写输入模式、阈值校验、JSON 和输出文件的失败测试**

```python
def test_resolve_inputs_reads_fixed_order_filenames(tmp_path):
    (tmp_path / "reference_3c.jpg").touch()
    (tmp_path / "actual_left_front_45.jpg").touch()
    reference, actual = resolve_inputs(tmp_path, None, None)
    assert reference.name == "reference_3c.jpg"
    assert actual.name == "actual_left_front_45.jpg"


def test_validate_thresholds_rejects_inverted_similarity_bands():
    with pytest.raises(ValueError, match="一致阈值"):
        validate_thresholds(0.6, similar_threshold=0.5, review_threshold=0.6)


def test_save_report_writes_utf8_json(tmp_path):
    output = save_report({"verdict": "需人工复核"}, tmp_path)
    assert json.loads(output.read_text(encoding="utf-8"))["verdict"] == "需人工复核"
```

- [ ] **Step 2: 运行新增测试并确认失败**

Run: `/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest validate/part_feature_compare/test_ebike_part_feature_compare.py -v`

Expected: FAIL，提示 CLI 或输出接口不存在。

- [ ] **Step 3: 实现命令行编排和可视化**

`run_comparison()` 按下列顺序直线执行并在异常点早返回或抛出中文异常：校验输入 → 加载两图 → YOLO 批量检测 → 同类配对 → 无公共部件时写 JSON 并返回 → 加载 DINOv2 → 批量特征比较 → 写部件裁剪及左右对照图 → 写汇总图 → 写 JSON → 打印中文表格。

`main()` 返回退出码，模块入口使用：

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

可视化统一从实际报告数据生成，避免 JSON 与图片分别计算导致数值漂移。中文字体按 macOS PingFang、Linux Noto Sans CJK 和 Pillow 默认字体依次回退；默认字体无法绘制中文时使用英文结果代码，不能因字体缺失中断比对。

- [ ] **Step 4: 运行测试、语法和帮助命令检查**

Run:

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest validate/part_feature_compare/test_ebike_part_feature_compare.py -v
/Users/mark/Workspace/ultralytics/.venv/bin/python -m py_compile validate/part_feature_compare/ebike_part_feature_compare.py
/Users/mark/Workspace/ultralytics/.venv/bin/python validate/part_feature_compare/ebike_part_feature_compare.py --help
```

Expected: 全部成功，帮助文本包含订单、权重、检测阈值和相似度阈值参数。

- [ ] **Step 5: 提交 CLI 与输出**

```bash
git add validate/part_feature_compare/ebike_part_feature_compare.py \
    validate/part_feature_compare/test_ebike_part_feature_compare.py
git commit -m "feat: add comparison CLI and evidence outputs"
```

### Task 4: 中文使用说明与真实订单验收

**Files:**

- Create: `validate/part_feature_compare/README.md`
- Modify: `validate/part_feature_compare/ebike_part_feature_compare.py`
- Modify: `validate/part_feature_compare/test_ebike_part_feature_compare.py`

**Interfaces:**

- Consumes: Task 3 的完整 CLI。
- Produces: 可复制的安装/运行命令、实际 `comparison.json` 和 JPG 证据文件。

- [ ] **Step 1: 编写中文 README**

README 必须包括：算法流程、默认文件、首次 DINOv2 下载说明、最简命令、显式图片命令、参数表、输出目录结构、JSON 字段、如何解释分数，以及当前样例视角不一致和阈值未标定的边界。

- [ ] **Step 2: 按仓库规则格式化与静态检查**

Run:

```bash
npx prettier@3.6.2 --print-width 120 --write validate/part_feature_compare/README.md
ruff format validate/part_feature_compare/ebike_part_feature_compare.py \
    validate/part_feature_compare/test_ebike_part_feature_compare.py
ruff check validate/part_feature_compare/ebike_part_feature_compare.py \
    validate/part_feature_compare/test_ebike_part_feature_compare.py
git diff --check
```

Expected: 全部成功且没有自动修复后残留问题。

- [ ] **Step 3: 运行完整单元测试**

Run: `/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest validate/part_feature_compare/test_ebike_part_feature_compare.py -v`

Expected: 全部 PASS。

- [ ] **Step 4: 使用默认订单进行真实端到端运行**

Run:

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python \
    validate/part_feature_compare/ebike_part_feature_compare.py \
    --order-dir /Users/mark/Workspace/ultralytics/validate/orders/2071943118060388353_214522621506984 \
    --detector /Users/mark/Workspace/ultralytics/yunst_ai/validate_best_model/yolo11l_12000_best.pt \
    --device cpu
```

Expected: 加载真实 YOLO 和 DINOv2，对 `ebike_full`、`saddle` 等公共部件生成真实相似度，并输出 `comparison.json`、`comparison_summary.jpg` 及部件裁剪对照图。

- [ ] **Step 5: 回读并核对结果**

检查 JSON 中的输入路径、模型、阈值、公共部件、置信度、相似度和判定；逐张打开汇总图和部件对照图，确认无文字重叠、裁剪错误、空白图或数值不一致。

- [ ] **Step 6: 提交说明和验收后的必要修正**

```bash
git add validate/part_feature_compare/README.md \
    validate/part_feature_compare/ebike_part_feature_compare.py \
    validate/part_feature_compare/test_ebike_part_feature_compare.py
git commit -m "docs: explain e-bike feature comparison demo"
```

### Task 5: 最终复核

**Files:**

- Verify: `validate/part_feature_compare/`
- Verify: `docs/superpowers/specs/2026-08-03-ebike-part-feature-comparison-demo-design.md`
- Verify: `docs/superpowers/plans/2026-08-03-ebike-part-feature-comparison-demo.md`

**Interfaces:**

- Consumes: Tasks 1-4 的全部产物。
- Produces: 可交付的分支状态和验证记录。

- [ ] **Step 1: 对照设计逐项核查功能覆盖**

确认没有增加批量订单、Level 3、FAISS、API 服务或自动阈值标定，且没有修改 `ultralytics/` 核心代码。

- [ ] **Step 2: 运行最终验证**

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest validate/part_feature_compare/test_ebike_part_feature_compare.py -v
ruff check validate/part_feature_compare/
git diff --check HEAD~4..HEAD
git status --short --branch
```

Expected: 测试和 Ruff 通过，diff 无空白错误，worktree 干净。

- [ ] **Step 3: 审查最小性和触发归属**

回答仓库 review gate：没有可删除的重复实现；所有新增逻辑都属于独立 Demo；没有新增条件去掩盖 YOLO 漏检、视角差异或阈值未标定问题。
