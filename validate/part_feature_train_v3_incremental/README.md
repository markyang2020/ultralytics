# DINOv3 左正整车增量训练包

## 1. 本次训练数据

原始素材来自本机 `validate/datasets` 下三个工单目录，共1400个工单：

- `orders_1000`：1000个工单；
- `orders_200`：200个工单；
- `orders_200_2`：200个工单。

本次只使用以下同工单正样本：

```text
reference_3c.jpg  <->  ebikeFront.jpg
合格证左正图          E码通左正图
```

`actual_bike_weight.jpg`、`actual_left_front_45.jpg`、`actual_left_back_45.jpg` 均未进入本次训练包。
这样可以避免大角度图片把模型训练成过度忽略车辆局部结构。

两张原图都先使用现有YOLO模型检测 `ebike_full`，整车框外扩3%后保存为RGB JPEG。训练时只读取已经
裁剪好的图片，不再运行YOLO，也不执行前景分割。

## 2. 拆分结果

| 分区 | 工单数 | 用途 |
| --- | ---: | --- |
| 训练集 | 1110 | 更新DINOv3模型参数 |
| 验证集 | 139 | 每轮选择最佳模型和早停 |
| 独立验收集 | 139 | 训练结束后只做最终评估 |
| 排除 | 12 | 缺少合格证图或E码通图 |

数据不是按工单简单随机拆分。完全相同的原图，或者YOLO裁剪后完全相同的图片，会先合并为同一车型图组；
同组工单不会跨训练、验证和验收集。最终共198个独立图组。

审计结果：

```text
有效工单：1388
裁剪图片：2776
训练/验证交叉：0
训练/验收交叉：0
验证/验收交叉：0
跨分区原图哈希：0
跨分区裁剪图哈希：0
```

验证集和验收集各生成139个同车正样本、139个其他车型负样本。负样本禁止从同一重复图组抽取。

## 3. 目录结构

```text
part_feature_train_v3_incremental/
├── data_bbox/
│   ├── train/pairs/<工单号>/{reference,actual}/ebike_full.jpg
│   ├── validation/pairs/<工单号>/{reference,actual}/ebike_full.jpg
│   ├── validation/labeled_pairs.json
│   ├── acceptance/pairs/<工单号>/{reference,actual}/ebike_full.jpg
│   ├── acceptance/labeled_pairs.json
│   ├── dataset_manifest.json
│   ├── split_assignments.csv
│   └── quality_review.csv
├── train_common.py
├── train_dinov3.py
├── evaluate_acceptance.py
└── audit_dataset.py
```

## 4. 上传GPU服务器

本文继续使用原服务器目录：

```text
/root/autodl-tmp/dinov2/dinov3_repo
/root/autodl-tmp/dinov2/checkpoints/dinov3/best.pt
```

其中 `best.pt` 是上一次训练得到的模型，本次从它继续微调。不要覆盖或删除这个文件。

从本机执行：

```bash
rsync -aH --info=progress2 \
  /Users/mark/Workspace/ultralytics-ebike-dinov3-incremental-1200/validate/part_feature_train_v3_incremental/ \
  root@<GPU服务器IP>:/root/autodl-tmp/dinov3_incremental_1400/part_feature_train_v3_incremental/
```

也可以上传已经生成的 `part_feature_train_v3_incremental_gpu.tar.gz`，然后在服务器解压：

```bash
mkdir -p /root/autodl-tmp/dinov3_incremental_1400
tar -xzf part_feature_train_v3_incremental_gpu.tar.gz -C /root/autodl-tmp/dinov3_incremental_1400
```

## 5. 训练前检查

在GPU服务器激活原来已经能训练DINOv3的Conda环境：

```bash
conda env list
conda activate <原DINOv3环境名>

cd /root/autodl-tmp/dinov3_incremental_1400/part_feature_train_v3_incremental

python - <<'PY'
from pathlib import Path
import torch

required = (
    Path("data_bbox/dataset_manifest.json"),
    Path("/root/autodl-tmp/dinov2/dinov3_repo/dinov3/hub/backbones.py"),
    Path("/root/autodl-tmp/dinov2/checkpoints/dinov3/best.pt"),
)
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("缺少文件：\n" + "\n".join(missing))
if not torch.cuda.is_available():
    raise SystemExit("当前Conda环境没有可用CUDA")
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
PY

python -u audit_dataset.py
```

`audit_dataset.py` 必须输出 `"status": "passed"` 后再训练。

## 6. 开始增量训练

直接执行，不需要传命令行参数：

```bash
python -u train_dinov3.py 2>&1 | tee train_incremental_1400.log
```

当前参数已经写在 `train_dinov3.py` 的 `main()` 中：

- 初始模型：`/root/autodl-tmp/dinov2/checkpoints/dinov3/best.pt`；
- 训练轮数：20；
- batch size：8；
- 学习率：`5e-6`；
- warmup：2轮；
- 前20个Transformer Block保持冻结；
- 使用 `cuda:0` 和AMP；
- 连续8轮验证集没有提升时提前停止；
- 每轮都验证，但训练期间不读取独立验收集。

输出目录：

```text
checkpoints/dinov3_incremental_1400_left_view/
├── best.pt
├── last.pt
├── best_validation_metrics.json
├── validation_history.jsonl
└── run_config.json
```

`best.pt` 用于最终推理，`last.pt` 只用于训练意外中断后恢复。

## 7. 训练中断后继续

把 `train_dinov3.py` 中：

```python
resume_checkpoint = None
```

改为：

```python
resume_checkpoint = SCRIPT_DIR / "checkpoints/dinov3_incremental_1400_left_view/last.pt"
```

然后再次运行：

```bash
python -u train_dinov3.py 2>&1 | tee -a train_incremental_1400.log
```

这会同时恢复模型、优化器、学习率和已完成轮数。正常开始新一轮增量训练时不要配置 `resume_checkpoint`。

## 8. 最终独立验收

训练完成后执行一次：

```bash
python -u evaluate_acceptance.py
```

结果写入：

```text
checkpoints/dinov3_incremental_1400_left_view/acceptance_metrics.json
```

重点检查：

- 正样本自动通过数量；
- 负样本被错误判成一致的数量；
- 正负样本平均分差距；
- AUC和低误放率下的正样本召回情况。

没有完成独立验收前，不要用新的 `best.pt` 覆盖当前网页模型。
