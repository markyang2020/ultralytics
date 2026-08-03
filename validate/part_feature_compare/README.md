# 电动自行车部件外观特征比对 Demo

本 Demo 用于验证《电动自行车备案图与实拍图一致性比对技术方案》中的 Level 2 部件外观特征比对流程。它会分别检测 3C 合格证官方图和现场实拍图，只对双方共同检测到的同类别部件计算 DINOv2 余弦相似度，并保存可回溯的图片和 JSON 证据。

## 实现流程

1. 使用指定 YOLO 权重检测两张图片中的电动自行车部件。
2. 每个类别保留置信度最高的检测框。
3. 只配对双方都检测到的同类别部件；单侧检测部件不进入外观比对。
4. 按 YOLO 检测框紧裁剪部件，resize 到 `224 x 224`。
5. 使用 DINOv2 ViT-B/14 提取 768 维 CLS token，并执行 L2 归一化。
6. 计算备案部件和实拍部件的余弦相似度，并按初始阈值分档。

初始相似度分档：

| 相似度                | 输出           | 含义                 |
| --------------------- | -------------- | -------------------- |
| `>= 0.80`             | `consistent`   | 部件一致             |
| `>= 0.60` 且 `< 0.80` | `review`       | 疑似更换，需人工复核 |
| `< 0.60`              | `inconsistent` | 部件外观不一致       |

这些是技术方案中的实验初始值，不是已经标定完成的生产阈值。

## 默认实验数据

脚本已配置本次确认的文件：

- YOLO 权重：`/Users/mark/Workspace/ultralytics/yunst_ai/validate_best_model/yolo11l_12000_best.pt`
- 订单目录：`/Users/mark/Workspace/ultralytics/validate/orders/2071943118060388353_214522621506984`
- 备案图：订单目录下的 `reference_3c.jpg`
- 实拍图：订单目录下的 `actual_left_front_45.jpg`

默认 YOLO 检测阈值是 `0.60`。在这组图片中，达到该阈值且双方共同检测到的部件预计包括 `ebike_full` 和 `saddle`。使用 `--det-conf 0.12` 可以查看更多低置信候选框，但低置信框不应直接进入正式判定。

## 环境要求

当前 Ultralytics 项目的 `.venv` 已包含运行所需的 `torch`、`torchvision`、`Pillow` 和 `ultralytics`。先进入实现所在 worktree：

```bash
cd /Users/mark/Workspace/ultralytics-ebike-part-feature-demo
```

首次运行时，`torch.hub` 会从 Meta FAIR 官方 `facebookresearch/dinov2` 仓库下载代码和 DINOv2 ViT-B/14 权重，并缓存到 `~/.cache/torch/hub/`。首次运行耗时取决于网络速度；后续运行会复用缓存。

如果控制台提示 `xFormers is not available`，DINOv2 会自动使用 PyTorch 原生实现继续推理，不影响本 Demo 的数值正确性，只可能影响推理速度。

## 最简运行命令

使用已配置的默认权重和默认订单：

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python \
    validate/part_feature_compare/ebike_part_feature_compare.py \
    --device cpu
```

使用指定订单目录：

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python \
    validate/part_feature_compare/ebike_part_feature_compare.py \
    --order-dir /Users/mark/Workspace/ultralytics/validate/orders/2071943118060388353_214522621506984 \
    --detector /Users/mark/Workspace/ultralytics/yunst_ai/validate_best_model/yolo11l_12000_best.pt \
    --device cpu
```

直接指定两张图片：

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python \
    validate/part_feature_compare/ebike_part_feature_compare.py \
    --reference /path/to/reference_3c.jpg \
    --actual /path/to/actual_left_front_45.jpg \
    --detector /path/to/yolo.pt \
    --output-dir /path/to/output \
    --device cpu
```

`--order-dir` 与 `--reference/--actual` 是两种独立输入模式，不能混用。

## 常用参数

| 参数                  | 默认值             | 说明                                 |
| --------------------- | ------------------ | ------------------------------------ |
| `--order-dir`         | 已配置的实验订单   | 从目录读取固定文件名的备案图和实拍图 |
| `--reference`         | 无                 | 显式指定备案图                       |
| `--actual`            | 无                 | 显式指定实拍图                       |
| `--detector`          | 已配置的 YOLO 权重 | 部件检测 `.pt` 文件                  |
| `--det-conf`          | `0.60`             | 进入本次流程的最低 YOLO 检测置信度   |
| `--similar-threshold` | `0.80`             | 判为部件一致的最低余弦相似度         |
| `--review-threshold`  | `0.60`             | 进入人工复核区间的最低余弦相似度     |
| `--imgsz`             | `640`              | YOLO 推理尺寸                        |
| `--device`            | `auto`             | 可指定 `cpu`、`mps` 或 CUDA 设备编号 |
| `--output-dir`        | 自动创建           | 显式指定结果目录                     |

查看完整帮助：

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python \
    validate/part_feature_compare/ebike_part_feature_compare.py --help
```

## 输出文件

未指定 `--output-dir` 时，结果按订单号和运行时间写入：

```text
validate/part_feature_compare/runs/<订单号>/<YYYYMMDD_HHMMSS>/
├── comparison.json
├── comparison_summary.jpg
└── crops/
    ├── ebike_full/
    │   ├── reference.jpg
    │   ├── actual.jpg
    │   └── comparison.jpg
    └── saddle/
        ├── reference.jpg
        ├── actual.jpg
        └── comparison.jpg
```

- `comparison.json`：记录输入路径、YOLO 权重、DINOv2 模型、实际阈值、检测框、检测置信度、部件相似度、分档和输出文件路径。
- `comparison_summary.jpg`：在备案图和实拍图上绘制检测框、检测置信度和公共部件相似度。
- `reference.jpg`、`actual.jpg`：实际送入 DINOv2 前的双方部件裁剪证据。
- `comparison.jpg`：单个部件的左右对照图，包含双方检测置信度、余弦相似度和判定。

## 如何查看结果

先打开 `comparison_summary.jpg`，确认 YOLO 检测框是否框中了同一个业务部件。再查看相似度偏低部件的 `crops/<part>/comparison.jpg`，排除下列影响：

- 拍摄视角不同；
- 部件被遮挡或只露出一部分；
- 检测框包含的背景比例不同；
- 部件在两张图中的尺度或清晰度差异大；
- YOLO 将相邻部件框入同一个检测框。

只有检测框和视角具备可比性时，相似度分档才具有“疑似更换”的业务解释价值。

`unmatched_parts` 表示该类别只在一侧达到检测阈值。Demo 不会把它直接判为拆除或加装，因为漏检、遮挡和视角盲区也会产生相同现象。

## 当前样例的限制

默认备案图接近标准左侧视图，实拍图更接近左前方近正视，两图的视角、主体尺度和背景差异明显。这组样例适合验证“YOLO 检测 -> 部件裁剪 -> DINOv2 特征 -> 余弦相似度 -> 证据输出”能否完整运行，但不能仅凭某个低分断言车辆已经改装。

生产使用前需要收集同视角的正样本对和改装负样本对，分别统计每个部件的相似度分布，再标定部件级阈值和人工复核区间。

## 运行测试

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest \
    validate/part_feature_compare/test_ebike_part_feature_compare.py -v
```

单元测试不会下载真实 DINOv2 权重；真实模型只在运行 Demo 时加载。
