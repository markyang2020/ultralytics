# 电动自行车部件外观特征比对 Demo

本 Demo 用于验证《电动自行车备案图与实拍图一致性比对技术方案》中的 Level 2 部件外观特征比对流程。它会分别检测 3C 合格证官方图和现场实拍图，只比较双方共同检测到的同类别部件，并行输出原始 RGB CLS 基线和灰度局部语义、形状融合实验结果。

V2 不覆盖原始基线。JSON、控制台和可视化会同时显示两组分数，便于使用同一批正负样本做消融验证。

## 实现流程

1. 使用指定 YOLO 权重检测两张图片中的电动自行车部件。
2. 每个类别保留置信度最高的检测框。
3. 只配对双方都检测到的同类别部件；单侧检测部件不进入外观比对。
4. 按 YOLO 检测框裁剪双方部件，原始 RGB 裁剪进入基线通道。
5. 使用 DINOv2 ViT-B/14 提取 768 维 RGB CLS token，得到 `similarity` 基线分数。
6. 双方裁剪执行同一套前景提取、灰度化、CLAHE、去噪和等比例白底补边。
7. 对灰度图提取 CLS 和前景加权 patch token，对前景形状提取 HOG 和边缘空间特征。
8. 按部件实验权重融合灰度 DINO 和形状相似度，写入独立的 `experimental` 节点。
9. 两套分数分别按当前实验阈值分档，并保存全部中间证据。

初始相似度分档：

| 相似度                | 输出           | 含义                 |
| --------------------- | -------------- | -------------------- |
| `>= 0.80`             | `consistent`   | 部件一致             |
| `>= 0.60` 且 `< 0.80` | `review`       | 疑似更换，需人工复核 |
| `< 0.60`              | `inconsistent` | 部件外观不一致       |

这些是技术方案中的实验初始值，同时应用于基线和实验融合分数，仅用于对照，不是已经标定完成的生产阈值。

## 分数字段

| 字段                                    | 含义                                                 |
| --------------------------------------- | ---------------------------------------------------- |
| `similarity`                            | 原始 RGB 裁剪的 DINOv2 CLS 余弦相似度，保持 V1 基线 |
| `experimental.gray_dino_similarity`     | 灰度 CLS 与前景加权 patch token 的融合特征相似度    |
| `experimental.shape_similarity`         | HOG 与边缘空间分布组成的形状相似度                   |
| `experimental.fused_similarity`         | 按部件实验权重融合后的分数                           |
| `experimental.channel_gap`              | 灰度 DINO 与形状分数的绝对差，不是概率置信度         |
| `experimental.preprocessing.*fallback` | 前景提取是否发生显式回退                             |

附件方案中的权重作为实验初值写入每个部件的 JSON。未知类别使用等权重；任一侧形状特征不可用时，只使用灰度 DINO 通道，不用固定零分拉低结果。

## 默认实验数据

脚本已配置本次确认的文件：

- YOLO 权重：`/Users/mark/Workspace/ultralytics/yunst_ai/validate_best_model/yolo11l_12000_best.pt`
- 订单目录：`/Users/mark/Workspace/ultralytics/validate/orders/2071943118060388353_214522621506984`
- 备案图：订单目录下的 `reference_3c.jpg`
- 实拍图：订单目录下的 `actual_bike_weight.jpg`

默认 YOLO 检测阈值是 `0.60`。在这组图片中，达到该阈值且双方共同检测到的部件预计包括 `ebike_full` 和 `saddle`。使用 `--det-conf 0.12` 可以查看更多低置信候选框，但低置信框不应直接进入正式判定。

## 环境要求

当前 Ultralytics 项目的 `.venv` 已包含运行所需的 `torch`、`torchvision`、`Pillow`、`numpy`、`opencv-python` 和 `ultralytics`。形状通道使用 OpenCV 梯度算子实现固定参数 HOG，不需要额外安装 `scikit-image`。先进入实现所在 worktree：

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
    --actual /path/to/actual_bike_weight.jpg \
    --detector /path/to/yolo.pt \
    --output-dir /path/to/output \
    --device cpu
```

`--order-dir` 与 `--reference/--actual` 是两种独立输入模式，不能混用。

## 常用参数

| 参数                  | 默认值             | 说明                                 |
| --------------------- | ------------------ | ------------------------------------ |
| `--order-dir`         | 已配置的实验订单   | 读取 `reference_3c.jpg` 和 `actual_bike_weight.jpg` |
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
    │   ├── reference_preprocessed.jpg
    │   ├── actual_preprocessed.jpg
    │   ├── reference_mask.png
    │   ├── actual_mask.png
    │   ├── reference_edges.png
    │   ├── actual_edges.png
    │   └── comparison.jpg
    └── saddle/
        └── ...
```

- `comparison.json`：记录输入、模型、阈值、检测框、基线结果、实验分项、权重、fallback 和两套订单结论。
- `comparison_summary.jpg`：在两张原图上绘制检测框、检测置信度、基线分数和实验融合分数。
- `reference.jpg`、`actual.jpg`：原始 RGB 部件裁剪，也是基线通道输入。
- `*_preprocessed.jpg`：前景归一化后的灰度实验输入。
- `*_mask.png`：用于背景抑制和 patch 加权的前景掩码。
- `*_edges.png`：进入形状通道的 Canny 边缘证据。
- `comparison.jpg`：显示基线、灰度 DINO、形状、实验融合分数和两套分档。

## 如何查看结果

先打开 `comparison_summary.jpg`，确认 YOLO 检测框是否框中了同一个业务部件。再查看分数变化明显部件的 `crops/<part>/comparison.jpg` 和六个预处理证据，按以下顺序排查：

1. 掩码是否保留了完整部件，是否把人、地面或相邻部件当成前景。
2. 灰度图是否保持了部件长宽比，是否被过度均衡或模糊。
3. 边缘图是否主要描述部件结构，而不是阴影和背景纹理。
4. 拍摄视角、遮挡和可见表面是否仍具有可比性。
5. YOLO 是否将相邻部件框入同一个检测框。

实验融合分数升高只说明当前预处理和特征对这组图片更接近，不能单独证明误放率没有增加。只有检测框、掩码和视角具备可比性，并在真实改装负样本上完成标定后，分档才具有生产判定价值。

`unmatched_parts` 表示该类别只在一侧达到检测阈值。Demo 不会把它直接判为拆除或加装，因为漏检、遮挡和视角盲区也会产生相同现象。

## 当前样例的限制

默认订单使用称重环节图片 `actual_bike_weight.jpg`。备案图通常是白底标准侧视，实拍图可能是斜侧视或俯视，两图的视角、主体尺度和背景仍可能明显不同。这组样例适合验证“基线与实验通道并行输出”能否完整运行，但灰度化、HOG和patch聚合都无法恢复备案图中没有出现的部件表面。

生产使用前需要收集同视角和跨视角的合格正样本、真实改装负样本，分别统计每个部件的两套分数分布。只有独立测试集证明实验通道降低误拒且没有不可接受地增加误放后，才能标定部件级阈值和人工复核区间。

## 运行测试

```bash
/Users/mark/Workspace/ultralytics/.venv/bin/python -m pytest \
    validate/part_feature_compare/test_feature_channels.py \
    validate/part_feature_compare/test_ebike_part_feature_compare.py -v
```

单元测试不会下载真实 DINOv2 权重；真实模型只在运行 Demo 时加载。
