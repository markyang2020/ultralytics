# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""DINOv3整车域适应训练共用的数据集、模型、指标和训练循环。"""

from __future__ import annotations

import gc
import importlib
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def build_image_transform(image_size: int) -> transforms.Compose:
    """保持训练、验证和生产推理的图像预处理完全一致。"""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def file_sha256(path: Path) -> str:
    """分块计算大文件SHA-256。"""
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WholeVehicleTripletDataset(Dataset):
    """读取同工单正样本，并从其他合格证图组在线采样负样本。"""

    def __init__(self, data_root: Path, image_size: int):
        pairs_root = data_root / "pairs"
        if not pairs_root.is_dir():
            raise FileNotFoundError(f"训练数据目录不存在：{pairs_root}")

        self.samples = []
        for vehicle_dir in sorted(path for path in pairs_root.iterdir() if path.is_dir()):
            reference_path = vehicle_dir / "reference/ebike_full.jpg"
            actual_path = vehicle_dir / "actual/ebike_full.jpg"
            if not reference_path.is_file() or not actual_path.is_file():
                raise FileNotFoundError(f"车辆目录缺少整车正样本：{vehicle_dir}")
            self.samples.append(
                {
                    "vehicle_id": vehicle_dir.name,
                    "reference": reference_path,
                    "actual": actual_path,
                    "reference_sha256": file_sha256(reference_path),
                    "actual_sha256": file_sha256(actual_path),
                }
            )
        if len(self.samples) < 2:
            raise ValueError(f"有效车辆正样本不足2组：{pairs_root}")
        self.groups = self._build_duplicate_groups()
        if len(self.groups) < 2:
            raise ValueError("训练数据去除重复图关系后不足2个独立组")
        self.group_index_by_vehicle = {
            sample["vehicle_id"]: group_index for group_index, group in enumerate(self.groups) for sample in group
        }
        self.transform = build_image_transform(image_size)
        print(f"[训练集] vehicles={len(self.samples)}, duplicate_groups={len(self.groups)}, image_size={image_size}")

    def _build_duplicate_groups(self) -> list[list[dict]]:
        """把任一侧裁剪图完全相同的车辆合组，避免同批次内产生假负样本。"""
        parent = list(range(len(self.samples)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        first_by_hash = {}
        for index, sample in enumerate(self.samples):
            for domain in ("reference", "actual"):
                key = sample[f"{domain}_sha256"]
                previous = first_by_hash.get(key)
                if previous is None:
                    first_by_hash[key] = index
                else:
                    union(previous, index)
        groups = {}
        for index, sample in enumerate(self.samples):
            groups.setdefault(find(index), []).append(sample)
        return list(groups.values())

    def __len__(self) -> int:
        # 每个工单每轮都参与训练，不能因为多个工单复用同一合格证样图就只抽取其中一张实拍图。
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        positive_group_index = self.group_index_by_vehicle[sample["vehicle_id"]]
        negative_group_index = random.randrange(len(self.groups) - 1)
        if negative_group_index >= positive_group_index:
            negative_group_index += 1
        negative = random.choice(self.groups[negative_group_index])
        return {
            "anchor": self._load(sample["reference"]),
            "positive": self._load(sample["actual"]),
            "negative": self._load(negative["actual"]),
        }

    def _load(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


class LabeledPairDataset(Dataset):
    """读取固定正负配对清单，验证和验收阶段禁止随机换负样本。"""

    def __init__(self, data_root: Path, image_size: int):
        records_path = data_root / "labeled_pairs.json"
        if not records_path.is_file():
            raise FileNotFoundError(f"固定配对清单不存在：{records_path}")
        records = json.loads(records_path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not records:
            raise ValueError(f"固定配对清单为空或格式错误：{records_path}")

        self.records = []
        for index, record in enumerate(records):
            if record.get("label") not in (0, 1):
                raise ValueError(f"第{index + 1}条label必须为0或1")
            reference = self._resolve(data_root, record.get("ref"), index, "ref")
            actual = self._resolve(data_root, record.get("actual"), index, "actual")
            self.records.append({**record, "reference": reference, "actual_path": actual})
        labels = {record["label"] for record in self.records}
        if labels != {0, 1}:
            raise ValueError("固定配对清单必须同时包含正负样本")
        self.transform = build_image_transform(image_size)
        positive = sum(record["label"] == 1 for record in self.records)
        print(f"[固定集] pairs={len(self.records)}, positive={positive}, negative={len(self.records) - positive}")

    @staticmethod
    def _resolve(data_root: Path, raw_path: object, index: int, field: str) -> Path:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"第{index + 1}条缺少{field}路径")
        path = data_root / raw_path
        if not path.is_file():
            raise FileNotFoundError(f"固定配对图片不存在：{path}")
        return path

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        return {
            "reference": self._load(record["reference"]),
            "actual": self._load(record["actual_path"]),
            "pair_id": str(record["pair_id"]),
            "label": torch.tensor(record["label"], dtype=torch.int64),
        }

    def _load(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


class DomainAdaptedEmbedder(nn.Module):
    """冻结DINOv3大部分骨干，只训练末端Block、norm和256维投影头。"""

    def __init__(self, backbone: nn.Module, feature_dim: int = 1024, freeze_until_layer: int = 20):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
        )
        self._freeze_backbone(freeze_until_layer)

    def _freeze_backbone(self, freeze_until_layer: int) -> None:
        blocks = self.backbone.blocks
        if not 0 <= freeze_until_layer < len(blocks):
            raise ValueError(f"freeze_until_layer应在0到{len(blocks) - 1}之间")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for block in blocks[freeze_until_layer:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in self.backbone.norm.parameters():
            parameter.requires_grad = True
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        print(f"[模型] trainable={trainable / 1e6:.1f}M, total={total / 1e6:.1f}M")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        cls_token = features["x_norm_clstoken"]
        patch_average = features["x_norm_patchtokens"].mean(dim=1)
        fused = 0.5 * cls_token + 0.5 * patch_average
        return F.normalize(self.projector(fused), dim=-1)


class CombinedContrastiveLoss(nn.Module):
    """组合Triplet与NT-Xent，沿用原DINOv3域适应训练目标。"""

    def __init__(self, margin: float = 0.3, temperature: float = 0.07):
        super().__init__()
        self.triplet = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda anchor, sample: 1 - (anchor * sample).sum(dim=-1), margin=margin
        )
        self.temperature = temperature

    def forward(
        self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        triplet_loss = self.triplet(anchors, positives, negatives)
        batch_size = anchors.shape[0]
        embeddings = torch.cat((anchors, positives), dim=0)
        similarities = torch.mm(embeddings, embeddings.t()) / self.temperature
        diagonal = torch.eye(2 * batch_size, dtype=torch.bool, device=anchors.device)
        similarities.masked_fill_(diagonal, float("-inf"))
        labels = torch.cat(
            (
                torch.arange(batch_size, 2 * batch_size, device=anchors.device),
                torch.arange(batch_size, device=anchors.device),
            )
        )
        nt_xent_loss = F.cross_entropy(similarities, labels)
        loss = 0.4 * triplet_loss + 0.6 * nt_xent_loss
        return loss, {"triplet": triplet_loss.item(), "nt_xent": nt_xent_loss.item()}


def calculate_binary_metrics(labels: np.ndarray, scores: np.ndarray, max_false_positive_rate: float) -> dict:
    """计算ROC-AUC、最佳F1阈值和低误放率工作点。"""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels和scores必须是一维且长度一致")
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("指标计算必须同时包含正负样本")

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    indexes = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    true_positives = np.cumsum(sorted_labels)[indexes]
    false_positives = 1 + indexes - true_positives
    true_positive_rates = np.r_[0.0, true_positives / positive_count]
    false_positive_rates = np.r_[0.0, false_positives / negative_count]
    roc_auc = float(np.trapz(true_positive_rates, false_positive_rates))

    operating_points = []
    thresholds = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)[::-1]]
    for threshold in thresholds:
        predictions = scores >= threshold
        true_positive = int(np.logical_and(predictions, labels == 1).sum())
        false_positive = int(np.logical_and(predictions, labels == 0).sum())
        false_negative = positive_count - true_positive
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / positive_count
        false_positive_rate = false_positive / negative_count
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        operating_points.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "false_positive_rate": float(false_positive_rate),
                "false_positive": false_positive,
                "false_negative": false_negative,
            }
        )
    best = max(operating_points, key=lambda point: (point["f1"], -point["false_positive_rate"]))
    strict = max(
        (point for point in operating_points if point["false_positive_rate"] <= max_false_positive_rate),
        key=lambda point: (point["recall"], point["precision"], point["threshold"]),
    )
    return {
        "pair_count": len(labels),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "roc_auc": roc_auc,
        "positive_score_mean": float(scores[labels == 1].mean()),
        "negative_score_mean": float(scores[labels == 0].mean()),
        "score_gap": float(scores[labels == 1].mean() - scores[labels == 0].mean()),
        "best_threshold": best["threshold"],
        "precision": best["precision"],
        "recall": best["recall"],
        "f1": best["f1"],
        "false_positive_rate": best["false_positive_rate"],
        "strict_threshold": strict["threshold"],
        "recall_at_max_fpr": strict["recall"],
        "precision_at_max_fpr": strict["precision"],
        "actual_false_positive_rate": strict["false_positive_rate"],
        "strict_false_positive": strict["false_positive"],
        "strict_false_negative": strict["false_negative"],
    }


def calculate_production_bands(
    labels: np.ndarray, scores: np.ndarray, consistent_threshold: float, review_threshold: float
) -> dict:
    """按用户确认的0.80/0.60双阈值统计正负样本三档结果。"""
    if not 0.0 <= review_threshold < consistent_threshold <= 1.0:
        raise ValueError("阈值必须满足0 <= review < consistent <= 1")
    consistent = scores >= consistent_threshold
    review = np.logical_and(scores > review_threshold, scores < consistent_threshold)
    inconsistent = scores <= review_threshold
    positive = labels == 1
    negative = labels == 0
    return {
        "consistent_threshold": consistent_threshold,
        "review_threshold": review_threshold,
        "positive": {
            "consistent": int(np.logical_and(positive, consistent).sum()),
            "review": int(np.logical_and(positive, review).sum()),
            "inconsistent": int(np.logical_and(positive, inconsistent).sum()),
        },
        "negative": {
            "consistent": int(np.logical_and(negative, consistent).sum()),
            "review": int(np.logical_and(negative, review).sum()),
            "inconsistent": int(np.logical_and(negative, inconsistent).sum()),
        },
        "negative_auto_accept_rate": float(np.logical_and(negative, consistent).sum() / max(1, negative.sum())),
        "positive_auto_accept_rate": float(np.logical_and(positive, consistent).sum() / max(1, positive.sum())),
    }


@torch.inference_mode()
def evaluate_model(
    model: DomainAdaptedEmbedder,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    max_false_positive_rate: float,
    consistent_threshold: float,
    review_threshold: float,
) -> dict:
    """在固定正负样本对上运行推理并返回逐对分数和整体指标。"""
    model.eval()
    labels = []
    scores = []
    pair_ids = []
    for batch in loader:
        reference = batch["reference"].to(device, non_blocking=True)
        actual = batch["actual"].to(device, non_blocking=True)
        amp_context = torch.autocast(device_type=device.type, dtype=torch.float16) if amp_enabled else nullcontext()
        with amp_context:
            similarities = (model(reference) * model(actual)).sum(dim=-1)
        labels.extend(batch["label"].cpu().numpy().tolist())
        scores.extend(similarities.float().cpu().numpy().tolist())
        pair_ids.extend(batch["pair_id"])
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    return {
        "overall": calculate_binary_metrics(labels_array, scores_array, max_false_positive_rate),
        "production_bands": calculate_production_bands(
            labels_array, scores_array, consistent_threshold, review_threshold
        ),
        "pairs": [
            {"pair_id": pair_id, "label": label, "score": round(score, 6)}
            for pair_id, label, score in zip(pair_ids, labels, scores)
        ],
    }


def validate_local_repo(repo_path: Path) -> None:
    """在分配大模型内存前确认DINOv3官方源码存在。"""
    if not (repo_path / "dinov3/hub/backbones.py").is_file():
        raise FileNotFoundError(f"DINOv3官方源码不完整：{repo_path}")


def create_dinov3_backbone(repo_path: Path, factory_name: str) -> nn.Module:
    """使用本地官方源码创建未加载权重的DINOv3骨干，全程不访问网络。"""
    validate_local_repo(repo_path)
    repo_string = str(repo_path.resolve())
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    factory = getattr(importlib.import_module("dinov3.hub.backbones"), factory_name)
    return factory(pretrained=False)


def load_dinov3_backbone(repo_path: Path, weights_path: Path, factory_name: str) -> nn.Module:
    """创建DINOv3骨干并加载官方预训练权重，用于不依赖旧checkpoint的首次训练。"""
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3官方预训练权重不存在：{weights_path}")
    backbone = create_dinov3_backbone(repo_path, factory_name)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state_dict, strict=True)
    del state_dict
    gc.collect()
    return backbone


@dataclass
class TrainingConfig:
    """保存一次DINOv3整车域适应训练的完整配置。"""

    train_root: Path
    validation_root: Path
    output_dir: Path
    repo_path: Path
    weights_path: Path | None
    initial_checkpoint: Path | None = None
    resume_checkpoint: Path | None = None
    image_size: int = 256
    epochs: int = 30
    batch_size: int = 8
    validation_batch_size: int = 16
    learning_rate: float = 3e-5
    warmup_epochs: int = 3
    freeze_until_layer: int = 20
    workers: int = 4
    device: str = "cuda:0"
    amp: bool = True
    max_false_positive_rate: float = 0.01
    consistent_threshold: float = 0.80
    review_threshold: float = 0.60
    early_stopping_patience: int = 8
    seed: int = 42


def resolve_device(requested_device: str) -> torch.device:
    """GPU不可用时立即失败，避免ViT-L误在CPU上长时间训练。"""
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA训练，但torch.cuda.is_available()为False")
    return torch.device(requested_device)


def save_checkpoint(path: Path, state: dict) -> None:
    """先写临时文件再原子替换，避免中断产生损坏权重。"""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def train_model(config: TrainingConfig, model_name: str = "dinov3_vitl16") -> DomainAdaptedEmbedder:
    """执行单GPU训练，每轮固定验证，只保留best.pt和last.pt。"""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    amp_enabled = config.amp and device.type == "cuda"
    print(f"[训练] device={device}, amp={amp_enabled}")

    train_dataset = WholeVehicleTripletDataset(config.train_root, config.image_size)
    validation_dataset = LabeledPairDataset(config.validation_root, config.image_size)
    if len(train_dataset) < config.batch_size:
        raise ValueError("训练样本数不能小于batch_size")
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.workers,
        pin_memory=True,
        persistent_workers=config.workers > 0,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.validation_batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        persistent_workers=config.workers > 0,
    )

    if config.initial_checkpoint is None:
        if config.weights_path is None:
            raise ValueError("从官方预训练模型开始训练时必须配置weights_path")
        backbone = load_dinov3_backbone(config.repo_path, config.weights_path, model_name)
    else:
        if not config.initial_checkpoint.is_file():
            raise FileNotFoundError(f"旧域适应checkpoint不存在：{config.initial_checkpoint}")
        backbone = create_dinov3_backbone(config.repo_path, model_name)
    model = DomainAdaptedEmbedder(backbone, freeze_until_layer=config.freeze_until_layer)
    if config.initial_checkpoint is not None:
        checkpoint = torch.load(config.initial_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        print(f"[初始化] 已加载域适应checkpoint：{config.initial_checkpoint}")
        del checkpoint
    model = model.to(device)

    backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": config.learning_rate * 0.1},
            {"params": model.projector.parameters(), "lr": config.learning_rate},
        ],
        weight_decay=1e-4,
    )
    total_steps = config.epochs * len(train_loader)
    warmup_steps = config.warmup_epochs * len(train_loader)

    def learning_rate_multiplier(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    criterion = CombinedContrastiveLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 0
    best_auc = float("-inf")
    best_f1 = float("-inf")
    epochs_without_improvement = 0

    if config.resume_checkpoint is not None:
        checkpoint = torch.load(config.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"])
        best_auc = float(checkpoint.get("best_validation_auc", best_auc))
        best_f1 = float(checkpoint.get("best_validation_f1", best_f1))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        print(f"[续训] checkpoint={config.resume_checkpoint}, next_epoch={start_epoch + 1}")

    metadata = {
        "model_family": "DINOv3",
        "model_name": model_name,
        "feature_fusion": "0.5 * CLS + 0.5 * mean(patch)",
        "projection_dim": 256,
        "official_weights_sha256": file_sha256(config.weights_path) if config.weights_path is not None else None,
        "initial_checkpoint_sha256": (
            file_sha256(config.initial_checkpoint) if config.initial_checkpoint is not None else None
        ),
        "training_config": {
            key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()
        },
        "train_vehicles": len(train_dataset.samples),
        "train_duplicate_groups_per_epoch": len(train_dataset),
        "validation_pairs": len(validation_dataset),
    }
    (config.output_dir / "run_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    history_path = config.output_dir / "validation_history.jsonl"
    if start_epoch == 0:
        history_path.write_text("", encoding="utf-8")

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            anchor = batch["anchor"].to(device, non_blocking=True)
            positive = batch["positive"].to(device, non_blocking=True)
            negative = batch["negative"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                loss, details = criterion(model(anchor), model(positive), model(negative))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
            if step % 10 == 0:
                print(
                    f"[批次] epoch={epoch + 1}/{config.epochs}, step={step}/{len(train_loader)}, "
                    f"loss={loss.item():.4f}, triplet={details['triplet']:.4f}, "
                    f"nt_xent={details['nt_xent']:.4f}",
                    flush=True,
                )

        average_loss = epoch_loss / len(train_loader)
        validation_report = evaluate_model(
            model,
            validation_loader,
            device,
            amp_enabled,
            config.max_false_positive_rate,
            config.consistent_threshold,
            config.review_threshold,
        )
        overall = validation_report["overall"]
        current_auc = float(overall["roc_auc"])
        current_f1 = float(overall["f1"])
        is_best = current_auc > best_auc or (math.isclose(current_auc, best_auc) and current_f1 > best_f1)
        if is_best:
            best_auc = current_auc
            best_f1 = current_f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        resume_state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "loss": average_loss,
            "best_validation_auc": best_auc,
            "best_validation_f1": best_f1,
            "epochs_without_improvement": epochs_without_improvement,
            "validation_metrics": validation_report,
            "metadata": metadata,
        }
        save_checkpoint(config.output_dir / "last.pt", resume_state)
        if is_best:
            save_checkpoint(
                config.output_dir / "best.pt",
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "loss": average_loss,
                    "best_validation_auc": best_auc,
                    "best_validation_f1": best_f1,
                    "validation_metrics": validation_report,
                    "metadata": metadata,
                },
            )
            (config.output_dir / "best_validation_metrics.json").write_text(
                json.dumps({"epoch": epoch + 1, **validation_report}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        with history_path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps({"epoch": epoch + 1, "loss": average_loss, **validation_report}, ensure_ascii=False) + "\n"
            )

        bands = validation_report["production_bands"]
        print(
            f"[轮次] epoch={epoch + 1}, loss={average_loss:.4f}, auc={current_auc:.4f}, f1={current_f1:.4f}, "
            f"正样本自动通过={bands['positive']['consistent']}, "
            f"负样本误放={bands['negative']['consistent']}, patience={epochs_without_improvement}",
            flush=True,
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"[早停] 连续{epochs_without_improvement}轮验证集未提升", flush=True)
            break

    print(f"[完成] best_auc={best_auc:.4f}, best_f1={best_f1:.4f}, output={config.output_dir}")
    return model


def load_trained_model(
    checkpoint_path: Path,
    repo_path: Path,
    device: torch.device,
    freeze_until_layer: int = 20,
) -> tuple[DomainAdaptedEmbedder, dict]:
    """恢复训练结果供冻结验收集评估使用。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    backbone = create_dinov3_backbone(repo_path, "dinov3_vitl16")
    model = DomainAdaptedEmbedder(backbone, freeze_until_layer=freeze_until_layer)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint
