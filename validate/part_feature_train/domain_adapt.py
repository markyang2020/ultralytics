"""
跨域部件比对 —— 域适应微调
核心思路：
  用对比学习让模型明白：
  "同一部件的渲染图" 和 "同一部件的真实照片" → 特征应该接近
  "不同部件的图"                              → 特征应该远离

数据要求：
  不需要人工标注！只需要知道"这两张图是同一个部件/车型"即可。
  来源：合格证图 + 查验站拍摄的对应车辆照片（即使分数低的那些也能用）

训练完成后：
  合格证灰度渲染图 和 现场彩色照片 在特征空间里会自然对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import json
from pathlib import Path


# ─────────────────────────────────────────────────────────
# 1. 数据集定义
# ─────────────────────────────────────────────────────────

class ComponentPairDataset(Dataset):
    """
    数据目录结构：
    data/
      pairs/
        vehicle_001/
          reference/   ← 合格证裁剪出的部件灰度图
            saddle.jpg
            front_basket.jpg
            backrest.jpg
            seat.jpg
          actual/      ← 对应现场车辆的部件裁剪图（同一辆车，可多张）
            saddle_01.jpg
            saddle_02.jpg   ← 同一个部件不同角度也行
            front_basket_01.jpg
            ...
        vehicle_002/
          ...

    一个 "pair" = (reference图, actual图, is_same_component)
    正样本：同一 vehicle_id + 同一 component_name
    负样本：不同 vehicle_id，或同一 vehicle_id 但不同 component_name
    """

    def __init__(self, data_root: str, mode: str = "train"):
        self.data_root = Path(data_root)
        self.mode = mode

        # 扫描所有 (vehicle_id, component_name, domain, image_path)
        self.samples = []
        pairs_root = self.data_root / "pairs"
        vehicle_dirs = sorted(path for path in pairs_root.iterdir() if path.is_dir()) if pairs_root.exists() else []
        for vehicle_dir in vehicle_dirs:
            vehicle_id = vehicle_dir.name

            for domain in ["reference", "actual"]:
                domain_dir = vehicle_dir / domain
                if not domain_dir.exists():
                    continue
                for img_path in domain_dir.glob("*.jpg"):
                    # 文件名约定：component_name_序号.jpg
                    # 例：saddle_01.jpg → component = saddle
                    stem = img_path.stem
                    # 去掉末尾的 _数字 部分
                    parts = stem.rsplit("_", 1)
                    component = parts[0] if (len(parts) == 2 and parts[1].isdigit()) else stem

                    self.samples.append({
                        "vehicle_id": vehicle_id,
                        "component":  component,
                        "domain":     domain,   # "reference" or "actual"
                        "path":       str(img_path),
                    })

        # 构建索引：(vehicle_id, component) → [samples]
        self.index = {}
        for s in self.samples:
            key = (s["vehicle_id"], s["component"])
            self.index.setdefault(key, []).append(s)

        self.index = {
            key: samples
            for key, samples in self.index.items()
            if {sample["domain"] for sample in samples} == {"reference", "actual"}
        }
        self.keys = list(self.index.keys())
        print(f"[Dataset] {len(self.samples)} images, "
              f"{len(self.keys)} (vehicle, component) groups")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # 每个 (vehicle, component) 组在一个 epoch 内只生成一个训练项。
        anchor_key = self.keys[idx]

        # ── 正样本对 ──────────────────────────────────────
        # 从同一组里各取一张 reference 和 actual
        group = self.index[anchor_key]
        refs    = [s for s in group if s["domain"] == "reference"]
        actuals = [s for s in group if s["domain"] == "actual"]

        img_a = self._load(random.choice(refs)["path"])
        img_p = self._load(random.choice(actuals)["path"])

        # ── 负样本 ────────────────────────────────────────
        # 策略：70% 跨车型同部件（难负样本），30% 不同部件（易负样本）
        if random.random() < 0.7:
            # 难负样本：不同 vehicle，同 component
            neg_key = self._sample_hard_negative(anchor_key)
        else:
            # 易负样本：同 vehicle，不同 component
            neg_key = self._sample_easy_negative(anchor_key)

        neg_group = self.index[neg_key]
        neg_sample = random.choice(neg_group)
        img_n = self._load(neg_sample["path"])

        return {
            "anchor":   self._transform(img_a),
            "positive": self._transform(img_p),
            "negative": self._transform(img_n),
        }

    def _sample_hard_negative(self, anchor_key):
        """不同vehicle_id，相同component → 难以区分，但应该相似"""
        anchor_vid, anchor_comp = anchor_key
        # 找所有同部件但不同车的组
        candidates = [k for k in self.keys
                      if k[1] == anchor_comp and k[0] != anchor_vid]
        if not candidates:
            candidates = [k for k in self.keys if k != anchor_key]
        return random.choice(candidates)

    def _sample_easy_negative(self, anchor_key):
        """相同vehicle_id，不同component → 明显不同"""
        anchor_vid, anchor_comp = anchor_key
        candidates = [k for k in self.keys
                      if k[0] == anchor_vid and k[1] != anchor_comp]
        if not candidates:
            candidates = [k for k in self.keys if k != anchor_key]
        return random.choice(candidates)

    def _load(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    # 图像预处理
    _transform_fn = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def _transform(self, img: Image.Image) -> torch.Tensor:
        return self._transform_fn(img)


# ─────────────────────────────────────────────────────────
# 2. 模型定义：DINOv2 + 轻量投影头
# ─────────────────────────────────────────────────────────

class DomainAdaptedEmbedder(nn.Module):
    """
    DINOv2 骨干（冻结大部分层）+ 可训练的投影头

    为什么冻结骨干大部分层？
    - DINOv2 已经学到了很强的视觉语义
    - 我们只需要最后几层适应"渲染图vs真实照"的域差异
    - 冻结底层可以防止用少量数据过拟合

    投影头的作用：
    - 把 1024 维骨干输出映射到 256 维
    - 在这个256维空间里做对比学习
    - 推理时可以用256维（更快）或原始1024维（更准）
    """

    def __init__(self, freeze_until_layer: int = 20):
        super().__init__()

        # 加载 DINOv2-ViT-L/14
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitl14_reg'
        )

        # 冻结前 freeze_until_layer 层（ViT-L共24层）
        # 只解冻最后4层 + 投影头
        self._freeze_backbone(freeze_until_layer)

        # 投影头：DINOv2输出1024维 → 投影到256维
        self.projector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
        )

    def _freeze_backbone(self, until_layer: int):
        """冻结 transformer 前 until_layer 层"""
        # 先全部冻结
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 解冻最后几个 transformer block
        total_blocks = len(self.backbone.blocks)
        for i, block in enumerate(self.backbone.blocks):
            if i >= until_layer:
                for param in block.parameters():
                    param.requires_grad = True

        # 解冻最终 norm 层
        for param in self.backbone.norm.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"[Model] Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M params")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: (B, 3, 224, 224)
        输出: (B, 256) L2归一化的特征向量
        """
        # 取 CLS token + patch 加权平均（融合全局+局部）
        features = self.backbone.forward_features(x)
        cls_tok   = features['x_norm_clstoken']      # (B, 1024)
        patch_tok = features['x_norm_patchtokens']   # (B, 256, 1024)

        # patch 平均池化作为局部特征
        patch_avg = patch_tok.mean(dim=1)             # (B, 1024)

        # 融合：CLS 偏重语义，patch 偏重局部结构
        fused = 0.5 * cls_tok + 0.5 * patch_avg      # (B, 1024)

        # 投影到256维
        projected = self.projector(fused)             # (B, 256)

        # L2 归一化
        return F.normalize(projected, dim=-1)


# ─────────────────────────────────────────────────────────
# 3. 损失函数：Triplet Loss + NT-Xent
# ─────────────────────────────────────────────────────────

class CombinedContrastiveLoss(nn.Module):
    """
    组合两种损失：
    A. Triplet Loss：直接约束 正样本比负样本更近
    B. NT-Xent (InfoNCE)：batch内所有正负样本互相对比，更稳定

    两者互补：
    - Triplet 给明确的几何约束（margin）
    - NT-Xent 利用整个batch的信息，梯度更稳定
    """

    def __init__(self, margin: float = 0.3, temperature: float = 0.07):
        super().__init__()
        self.triplet = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda a, b: 1 - (a * b).sum(dim=-1),
            margin=margin,
            reduction='mean'
        )
        self.temperature = temperature

    def nt_xent(self, anchors: torch.Tensor,
                positives: torch.Tensor) -> torch.Tensor:
        """
        NT-Xent loss（SimCLR 版）
        batch内：每个anchor和它的positive是正对，其余全是负样本
        """
        B = anchors.shape[0]
        # 拼接 anchor 和 positive
        z = torch.cat([anchors, positives], dim=0)  # (2B, 256)

        # 相似度矩阵
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # 对角线屏蔽（自身不作为负样本）
        mask = torch.eye(2*B, dtype=torch.bool, device=anchors.device)
        sim.masked_fill_(mask, float('-inf'))

        # 标签：anchor i 的正样本是 i+B；positive i+B 的正样本是 i
        labels = torch.cat([
            torch.arange(B, 2*B, device=anchors.device),
            torch.arange(0, B,   device=anchors.device)
        ])

        return F.cross_entropy(sim, labels)

    def forward(self, anchor, positive, negative):
        loss_triplet = self.triplet(anchor, positive, negative)
        loss_ntxent  = self.nt_xent(anchor, positive)
        # 两个损失加权组合
        return loss_triplet * 0.4 + loss_ntxent * 0.6, {
            "triplet": loss_triplet.item(),
            "ntxent":  loss_ntxent.item(),
        }


# ─────────────────────────────────────────────────────────
# 4. 训练主循环
# ─────────────────────────────────────────────────────────

def train(
    data_root:       str   = "data",
    output_dir:      str   = "checkpoints",
    epochs:          int   = 30,
    batch_size:      int   = 32,
    lr:              float = 3e-5,
    freeze_until:    int   = 20,   # 解冻最后4层（ViT-L共24层）
    warmup_epochs:   int   = 3,
    device:          str   = "cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # 数据
    dataset = ComponentPairDataset(data_root, mode="train")
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, num_workers=4,
                         pin_memory=True, drop_last=True)

    # 模型
    model = DomainAdaptedEmbedder(freeze_until_layer=freeze_until).to(device)

    # 优化器：骨干解冻层用小lr，投影头用大lr
    backbone_params  = [p for n, p in model.backbone.named_parameters()
                        if p.requires_grad]
    projector_params = list(model.projector.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params,  "lr": lr * 0.1},   # 骨干解冻层：1/10 lr
        {"params": projector_params, "lr": lr},          # 投影头：正常lr
    ], weight_decay=1e-4)

    # 学习率调度：warmup + cosine decay
    total_steps   = epochs * len(loader)
    warmup_steps  = warmup_epochs * len(loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = CombinedContrastiveLoss(margin=0.3, temperature=0.07)

    # 训练
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_triplet = 0.0
        epoch_ntxent  = 0.0

        for step, batch in enumerate(loader):
            anchor   = batch["anchor"].to(device)
            positive = batch["positive"].to(device)
            negative = batch["negative"].to(device)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss, details = criterion(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止骨干解冻层梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss    += loss.item()
            epoch_triplet += details["triplet"]
            epoch_ntxent  += details["ntxent"]

            if step % 20 == 0:
                lr_now = optimizer.param_groups[1]['lr']
                print(f"  Epoch {epoch+1}/{epochs} Step {step}/{len(loader)} "
                      f"| loss={loss.item():.4f} "
                      f"triplet={details['triplet']:.4f} "
                      f"ntxent={details['ntxent']:.4f} "
                      f"lr={lr_now:.2e}")

        avg_loss = epoch_loss / len(loader)
        print(f"[Epoch {epoch+1}] avg_loss={avg_loss:.4f} "
              f"triplet={epoch_triplet/len(loader):.4f} "
              f"ntxent={epoch_ntxent/len(loader):.4f}")

        # 保存最优checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(output_dir, "best.pt")
            torch.save({
                "epoch":      epoch + 1,
                "model_state": model.state_dict(),
                "loss":        avg_loss,
                "freeze_until": freeze_until,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (loss={avg_loss:.4f})")

        # 每5个epoch保存一次阶段性checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch":       epoch + 1,
                "model_state": model.state_dict(),
                "loss":        avg_loss,
            }, os.path.join(output_dir, f"epoch_{epoch+1}.pt"))

    print(f"\n[Done] Best loss: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    train(
        data_root    = "data",
        output_dir   = "checkpoints",
        epochs       = 30,
        batch_size   = 32,
        lr           = 3e-5,
        freeze_until = 20,
        device       = "cuda",
    )
