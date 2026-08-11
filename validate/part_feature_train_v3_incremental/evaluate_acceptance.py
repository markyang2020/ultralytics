# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""使用训练期间从未读取的冻结验收集评估DINOv3最佳模型。"""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import DataLoader

try:
    from .train_common import LabeledPairDataset, evaluate_model, load_trained_model, resolve_device
except ImportError:
    from train_common import LabeledPairDataset, evaluate_model, load_trained_model, resolve_device

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    """配置服务器路径后，只对best.pt执行一次最终验收。"""
    data_root = SCRIPT_DIR / "data_bbox"
    checkpoint_path = SCRIPT_DIR / "checkpoints/dinov3_incremental_1400_left_view/best.pt"
    repo_path = Path("/root/autodl-tmp/dinov2/dinov3_repo")
    output_path = SCRIPT_DIR / "checkpoints/dinov3_incremental_1400_left_view/acceptance_metrics.json"
    device = resolve_device("cuda:0")
    image_size = 256
    batch_size = 16
    workers = 4
    consistent_threshold = 0.80
    review_threshold = 0.60

    dataset = LabeledPairDataset(data_root / "acceptance", image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    model, checkpoint = load_trained_model(checkpoint_path, repo_path, device)
    report = evaluate_model(
        model,
        loader,
        device,
        amp_enabled=device.type == "cuda",
        max_false_positive_rate=0.01,
        consistent_threshold=consistent_threshold,
        review_threshold=review_threshold,
    )
    payload = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset": "acceptance",
        **report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    overall = report["overall"]
    bands = report["production_bands"]
    print(
        f"[验收] auc={overall['roc_auc']:.4f}, f1={overall['f1']:.4f}, "
        f"正样本自动通过={bands['positive']['consistent']}/{overall['positive_count']}, "
        f"负样本误放={bands['negative']['consistent']}/{overall['negative_count']}, "
        f"report={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
