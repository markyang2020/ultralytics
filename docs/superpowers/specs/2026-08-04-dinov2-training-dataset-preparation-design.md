# DINOv2 特征比对训练数据集准备设计

## 目标

从 `validate/part_feature_train/orders` 的查验订单中，为 DINOv2 域适应训练准备备案图与现场实拍图正样本对，并生成 `prepare_data.py` 可读取的 `inspection_records.json`。

## 数据纳入规则

- 仅纳入同时存在 `reference_3c.jpg` 和三张现场候选图的订单。
- `cert_image` 固定使用订单目录内的 `reference_3c.jpg`，不使用 `certificate.jpg` 兜底。
- 严格排除缺少 `reference_3c.jpg` 的 4 个订单，最终候选订单数为 196。
- `actual_images` 每个订单只保留一张现场图，避免同一部件被不同视角的后续裁剪结果覆盖。

## 现场图筛选规则

1. 默认选择 `actual_bike_weight.jpg`。
2. 逐单检查称重图是否为清晰、主体完整、接近左正侧面的车辆照片。
3. 如果称重图明显偏向车头或车尾、车辆纵向拍摄、主体严重不完整，比较 `actual_left_front_45.jpg` 和 `actual_left_back_45.jpg`。
4. 替换时选择两张候选图中更接近左正侧面、前后轮横向展开更充分、核心部件遮挡更少的一张。
5. 不因背景、查验员入镜或轻微俯拍单独替换；只有这些因素明显影响车辆主体或部件可见性时才替换。

## 输出结构

在 `validate/part_feature_train/inspection_records.json` 写入 JSON 数组。每条记录包含：

```json
{
    "vehicle_id": "订单目录名",
    "cert_image": "/绝对路径/reference_3c.jpg",
    "actual_images": [
        "/绝对路径/选定的现场图.jpg"
    ]
}
```

记录按 `vehicle_id` 排序，路径使用绝对路径，便于从任意工作目录运行数据准备流程。

## 训练对生成

使用现有部件检测权重 `/Users/mark/Workspace/ultralytics/yunst_ai/validate_best_model/yolo11l_12000_best.pt` 调用 `build_training_pairs()`，输出到 `validate/part_feature_train/data/pairs`。不修改订单原图，也不覆盖当前工作树中用户已有的脚本改动。

## 验证标准

- `inspection_records.json` 恰好包含 196 条互不重复的订单记录。
- 每条记录的备案图和选定实拍图都存在、可读取，并位于同一订单目录。
- 4 个缺少 `reference_3c.jpg` 的订单不出现在清单中。
- 所有 `actual_images` 都只有一个元素，且文件名属于三种允许的现场候选图之一。
- 生成训练对后，执行 `verify_dataset()`，记录成功、跳过和各部件有效配对数量。
