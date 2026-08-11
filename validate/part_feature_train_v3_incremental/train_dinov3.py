# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""使用三批工单的合格证与E码通左正图增量训练DINOv3整车相似度模型。"""

from pathlib import Path

try:
    from .train_common import TrainingConfig, train_model
except ImportError:
    from train_common import TrainingConfig, train_model

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    """服务器训练参数集中配置在这里，确认路径后直接运行本文件。"""
    data_root = SCRIPT_DIR / "data_bbox"
    repo_path = Path("/root/autodl-tmp/dinov2/dinov3_repo")
    weights_path = None
    output_dir = SCRIPT_DIR / "checkpoints/dinov3_incremental_1400_left_view"

    # 从当前线上候选模型继续微调，只继承模型参数，不继承旧优化器和学习率进度。
    initial_checkpoint = Path("/root/autodl-tmp/dinov2/checkpoints/dinov3/best.pt")
    # 训练中断后填写本次输出的last.pt继续训练，首次训练保持None。
    resume_checkpoint = None

    config = TrainingConfig(
        train_root=data_root / "train",
        validation_root=data_root / "validation",
        output_dir=output_dir,
        repo_path=repo_path,
        weights_path=weights_path,
        initial_checkpoint=initial_checkpoint,
        resume_checkpoint=resume_checkpoint,
        image_size=256,
        epochs=20,
        batch_size=8,
        validation_batch_size=16,
        learning_rate=5e-6,
        warmup_epochs=2,
        freeze_until_layer=20,
        workers=4,
        device="cuda:0",
        amp=True,
        max_false_positive_rate=0.01,
        consistent_threshold=0.80,
        review_threshold=0.60,
        early_stopping_patience=8,
        seed=42,
    )
    train_model(config)


if __name__ == "__main__":
    main()
