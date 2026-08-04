# DINOv2 训练数据集准备实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 200 个查验订单中严格筛选 196 个备案图有效订单，为每个订单选择一张最合适的左侧实拍图并生成可训练的部件正样本对。

**Architecture:** 不修改现有训练代码。先逐单人工复核三张实拍候选图并生成 `inspection_records.json`，再直接调用现有 `build_training_pairs()` 完成 YOLO 部件裁剪，最后执行结构、图片解码和部件覆盖统计验证。

**Tech Stack:** Python 3.12、JSON、OpenCV、Ultralytics YOLO、ImageMagick。

## Global Constraints

- `cert_image` 只能使用 `reference_3c.jpg`，缺失时严格排除，不使用 `certificate.jpg`。
- `actual_images` 每个订单只能包含一张图片。
- 实拍图默认使用 `actual_bike_weight.jpg`，只有明显不接近左正侧面时才从两张左 45° 图中选更正的一张。
- 不修改或复制 `orders` 下的原始图片。
- 使用绝对路径，确保可从任意工作目录运行。

---

### Task 1: 逐单确定实拍图并生成清单

**Files:**
- Create: `validate/part_feature_train/inspection_records.json`
- Read: `validate/part_feature_train/orders/*/{reference_3c.jpg,actual_bike_weight.jpg,actual_left_front_45.jpg,actual_left_back_45.jpg}`

**Interfaces:**
- Consumes: 订单目录和三张现场候选图。
- Produces: `list[dict]` JSON；每个字典包含 `vehicle_id: str`、`cert_image: str`、`actual_images: list[str]`。

- [ ] **Step 1: 复核 196 个有效订单**

按设计文档的筛选规则查看逐单三图对照表。默认保留称重图，仅记录需要替换为 `actual_left_front_45.jpg` 或 `actual_left_back_45.jpg` 的订单。

- [ ] **Step 2: 生成结构化 JSON**

按 `vehicle_id` 排序生成清单，严格排除以下订单：

```text
2071943738191798274_354322652062700
2071949659475935234_354322690060194
2071960547805233153_354322652061797
2071967145484808194_354322690060232
```

- [ ] **Step 3: 验证 JSON 结构和文件路径**

运行一个只读校验脚本，断言记录数为 196、订单号唯一、每条只有一张实拍图、备案图和实拍图均存在且能被 OpenCV 解码、两张图片属于同一订单目录。

Expected: 输出 `records=196, unique=196, unreadable=0, invalid=0`。

### Task 2: 生成部件训练对

**Files:**
- Create: `validate/part_feature_train/data/pairs/<vehicle_id>/{reference,actual,summary.json}`
- Read: `validate/part_feature_train/prepare_data.py`
- Read: `/Users/mark/Workspace/ultralytics/yunst_ai/validate_best_model/yolo11l_12000_best.pt`

**Interfaces:**
- Consumes: Task 1 的 `inspection_records.json` 和 YOLO 部件检测权重。
- Produces: 每个有效订单的备案部件裁剪、实拍部件裁剪及检测摘要。

- [ ] **Step 1: 调用现有构建函数**

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python -c "from validate.part_feature_train.prepare_data import build_training_pairs; build_training_pairs('validate/part_feature_train/inspection_records.json', '/Users/mark/Workspace/ultralytics/yunst_ai/validate_best_model/yolo11l_12000_best.pt', 'validate/part_feature_train/data/pairs', 2)"
```

- [ ] **Step 2: 检查运行结果**

确认控制台给出 `total=196`，记录 `ok` 和 `skipped`；任何图片读取失败都必须回到 Task 1 修正，不能静默保留。

### Task 3: 验证最终数据集

**Files:**
- Read: `validate/part_feature_train/data/pairs/**`
- Read: `validate/part_feature_train/prepare_data.py`

**Interfaces:**
- Consumes: Task 2 的部件裁剪目录。
- Produces: 有效车辆数和各部件正样本对覆盖统计。

- [ ] **Step 1: 运行现有数据集验证函数**

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python -c "from validate.part_feature_train.prepare_data import verify_dataset; verify_dataset('validate/part_feature_train/data/pairs')"
```

- [ ] **Step 2: 核对目录和摘要一致性**

逐目录断言 `summary.json` 可解析，`reference` 与 `actual` 至少存在两个同名部件裁剪，所有裁剪图都能被 OpenCV 解码。

Expected: 不存在空裁剪、损坏图片或清单外订单；统计结果明确列出有效车辆数和每个部件的配对数。
