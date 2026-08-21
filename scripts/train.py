"""scripts/train.py — Baseline 训练入口。

职责：读取 configs/{dataset,model,train}.yaml → 构建数据 / 模型 / 损失 / 优化器 /
调度器 → 标准训练循环（forward / loss / backward / optimizer.step / validation /
checkpoint）。

用法（在项目根目录或任意位置均可）:
    python scripts/train.py

依赖：
    models.build_model          — 按 model.yaml 的 model_type 构建 BaselineDetector
    utils.create_dataloaders    — 三模态 Dataset + DataLoader（含 train→val 切分）
    configs/train.yaml          — optimizer / scheduler / checkpoint 等训练配置

说明：
    - 当前为 visible 单模态 baseline；红外 / 深度已由 Dataset 加载，但模型未使用
      （留给 fusion 阶段）。
    - 检测头为 anchor-free 解耦头（YOLOX 风格），损失与本约定严格对齐（见 DetectionLoss）。
    - 验证阶段输出 val_loss + mAP@0.5 + mAP@0.5:0.95（见 _validate 与
      utils.metrics），best 模型按 best_metric 选择的 mAP 指标保存。
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

# 保证 `python scripts/train.py` 时能导入项目内的 models / utils
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model            # noqa: E402
from utils import create_dataloaders      # noqa: E402
from utils.metrics import (               # noqa: E402
    compute_map,
    decode_boxes,
    postprocess,
)


# ============================================================
# 基础工具
# ============================================================

def _load_yaml(path):
    """读取 yaml 配置文件，返回解析后的字典。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_template(s, cfg):
    """把路径模板中的 ``${key}`` 替换为 train.yaml 顶层同名值。"""
    return re.sub(r"\$\{(\w+)\}", lambda m: str(cfg.get(m.group(1), m.group(0))), str(s))


def _set_seed(seed: int):
    """固定随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _select_device(cfg: dict) -> torch.device:
    """根据 train.yaml 选择设备；CUDA 不可用时自动回退到 CPU。"""
    requested = str(cfg.get("device", "cuda")).lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        gpu_ids = cfg.get("gpu_ids", [0])
        gpu = int(gpu_ids[0]) if gpu_ids else 0
        return torch.device(f"cuda:{gpu}")
    if requested.startswith("cuda"):
        print("[device] 配置要求 CUDA 但当前不可用，自动回退到 CPU。")
    return torch.device("cpu")


def _get_amp_tools(use_amp: bool):
    """返回 (autocast 上下文, GradScaler)。不同 torch 版本 API 差异做兼容。"""
    if not use_amp:
        return contextlib.nullcontext(), None
    try:
        autocast = torch.amp.autocast("cuda", enabled=True)
    except AttributeError:
        autocast = torch.cuda.amp.autocast(enabled=True)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    return autocast, scaler


# ============================================================
# 损失（anchor-free 解耦头，与 DecoupledHead 输出约定对齐）
# ============================================================

def _bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Complete-IoU，输入均为 [N, 4] 像素坐标 (cx, cy, w, h)。返回 [N] 的 CIoU。"""
    b1_x1 = box1[:, 0] - box1[:, 2] / 2
    b1_y1 = box1[:, 1] - box1[:, 3] / 2
    b1_x2 = box1[:, 0] + box1[:, 2] / 2
    b1_y2 = box1[:, 1] + box1[:, 3] / 2
    b2_x1 = box2[:, 0] - box2[:, 2] / 2
    b2_y1 = box2[:, 1] - box2[:, 3] / 2
    b2_x2 = box2[:, 0] + box2[:, 2] / 2
    b2_y2 = box2[:, 1] + box2[:, 3] / 2

    inter = (
        (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(min=0)
        * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(min=0)
    )
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # CIoU 附加项：中心距离惩罚 + 长宽比惩罚
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = (box1[:, 0] - box2[:, 0]) ** 2 + (box1[:, 1] - box2[:, 1]) ** 2
    v = (4 / (math.pi ** 2)) * (
        torch.atan(w2 / h2.clamp(min=eps)) - torch.atan(w1 / h1.clamp(min=eps))
    ) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + alpha * v)


def _focal_bce(pred_logits: torch.Tensor, target: torch.Tensor,
               gamma: float, alpha: float) -> torch.Tensor:
    """Focal BCE with logits。pred_logits / target 形状一致，target ∈ {0, 1}。"""
    p = torch.sigmoid(pred_logits)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    return (a_t * ((1 - p_t) ** gamma) * bce).mean()


class DetectionLoss(torch.nn.Module):
    """anchor-free 解耦头损失（简化基线版）。

    目标分配：每个 GT 框分配到其中心所在网格，三个尺度各分配一次
    （即每个 GT 得到 3 个正样本），无 simOTA / 无多 anchor 偏移。
    与 DecoupledHead 输出对齐：
        obj : focal BCE（全部网格，正样本=1，负样本=0）
        cls : focal BCE（仅正样本，one-hot）
        reg : CIoU loss（仅正样本，解码后的预测框 vs GT 框）

    这是用于跑通 baseline 的最小实现；后续可用完整 YOLOX 式 simOTA
    替换，只需保持 forward 签名 (preds, labels, num_labels) 不变。
    """

    def __init__(self, num_classes: int, strides=(8, 16, 32),
                 input_size=(640, 640), loss_weights=None,
                 focal_gamma: float = 1.5, focal_alpha: float = 0.25):
        super().__init__()
        self.num_classes = int(num_classes)
        self.strides = tuple(strides)
        self.input_size = tuple(input_size)
        self.loss_weights = loss_weights or {"obj": 1.0, "cls": 1.0, "reg": 1.0}
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def forward(self, preds, labels, num_labels):
        """计算总损失。

        Args:
            preds: Tuple[HeadPredictions, ...]，每尺度含 cls/reg/obj。
            labels: [B, max_n, 5]，YOLO 归一化坐标（已 letterbox 对齐到 input_size）。
            num_labels: [B]，每个样本真实框数（超出部分为 -1 填充）。
        """
        device = labels.device
        H, W = self.input_size

        obj_losses, cls_losses, reg_losses = [], [], []

        for scale_idx, stride in enumerate(self.strides):
            pred = preds[scale_idx]
            B, _, fh, fw = pred.reg.shape
            decoded = decode_boxes(pred.reg, stride)   # [B, fh, fw, 4]

            obj_t = torch.zeros(B, fh, fw, device=device)
            pos_b, pos_j, pos_i, pos_cls = [], [], [], []
            pos_pred, pos_gt = [], []

            for b in range(B):
                n = int(num_labels[b].item())
                for k in range(n):
                    row = labels[b, k]
                    if row[0] < 0:                      # padding 哨兵
                        continue
                    cls = int(row[0].item())
                    cx, cy, w, h = row[1].item(), row[2].item(), row[3].item(), row[4].item()
                    cx_px, cy_px = cx * W, cy * H
                    w_px, h_px = w * W, h * H
                    gi = min(int(cx_px / stride), fw - 1)
                    gj = min(int(cy_px / stride), fh - 1)

                    obj_t[b, gj, gi] = 1.0
                    pos_b.append(b)
                    pos_j.append(gj)
                    pos_i.append(gi)
                    pos_cls.append(cls)
                    pos_pred.append(decoded[b, gj, gi])
                    pos_gt.append(torch.tensor([cx_px, cy_px, w_px, h_px],
                                               device=device, dtype=decoded.dtype))

            # obj loss：全部网格
            obj_loss = _focal_bce(pred.obj.squeeze(1), obj_t,
                                  self.focal_gamma, self.focal_alpha)

            # cls / reg loss：仅正样本
            if pos_pred:
                b_idx = torch.tensor(pos_b, device=device)
                j_idx = torch.tensor(pos_j, device=device)
                i_idx = torch.tensor(pos_i, device=device)
                cls_logits = pred.cls[b_idx, :, j_idx, i_idx]          # [P, C]
                cls_t = F.one_hot(torch.tensor(pos_cls, device=device),
                                  self.num_classes).float()            # [P, C]
                cls_loss = _focal_bce(cls_logits, cls_t,
                                      self.focal_gamma, self.focal_alpha)

                iou = _bbox_ciou(torch.stack(pos_pred), torch.stack(pos_gt))
                reg_loss = (1.0 - iou).mean()
            else:
                cls_loss = torch.zeros((), device=device)
                reg_loss = torch.zeros((), device=device)

            obj_losses.append(obj_loss)
            cls_losses.append(cls_loss)
            reg_losses.append(reg_loss)

        n_scale = len(self.strides)
        obj_loss = sum(obj_losses) / n_scale
        cls_loss = sum(cls_losses) / n_scale
        reg_loss = sum(reg_losses) / n_scale

        lw = self.loss_weights
        return lw["obj"] * obj_loss + lw["cls"] * cls_loss + lw["reg"] * reg_loss


# ============================================================
# 优化器 / 调度器
# ============================================================

def _build_optimizer(model: torch.nn.Module, cfg: dict):
    """按 train.yaml 构建 optimizer，支持 SGD / Adam / AdamW + 参数分组 lr 缩放。"""
    opt_cfg = cfg.get("optimizer", {})
    opt_type = str(opt_cfg.get("type", "SGD"))
    base_lr = float(cfg.get("learning_rate", 1e-2))
    weight_decay = float(opt_cfg.get("weight_decay", 5e-4))
    scales = opt_cfg.get("param_lr_scales", {})

    # 参数分组：lr 缩放 × (weight_decay / no-decay，偏置与 BN 不做衰减)
    buckets = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        scale = 1.0
        for key, s in scales.items():
            if name.startswith(key + "."):
                scale = float(s)
                break
        b = buckets.setdefault(scale, {"decay": [], "no_decay": []})
        if p.ndim <= 1 or "bias" in name or "bn" in name:
            b["no_decay"].append(p)
        else:
            b["decay"].append(p)

    groups = []
    for scale, b in buckets.items():
        lr = base_lr * scale
        if b["decay"]:
            groups.append({"params": b["decay"], "lr": lr, "weight_decay": weight_decay})
        if b["no_decay"]:
            groups.append({"params": b["no_decay"], "lr": lr, "weight_decay": 0.0})

    if opt_type == "SGD":
        return torch.optim.SGD(groups, lr=base_lr,
                               momentum=float(opt_cfg.get("momentum", 0.937)),
                               nesterov=bool(opt_cfg.get("nesterov", True)))
    if opt_type == "Adam":
        return torch.optim.Adam(groups, lr=base_lr)
    if opt_type == "AdamW":
        return torch.optim.AdamW(groups, lr=base_lr)
    raise NotImplementedError(
        f"不支持的 optimizer 类型: {opt_type!r}. 已实现 SGD / Adam / AdamW。")


def _build_scheduler(optimizer, cfg: dict):
    """按 train.yaml 构建 scheduler（Cosine 支持 warmup 后的 T_max / eta_min）。"""
    sched_cfg = cfg.get("scheduler", {})
    sched_type = str(sched_cfg.get("type", "CosineAnnealingLR"))
    epochs = int(cfg.get("epochs", 300))
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    base_lr = float(cfg.get("learning_rate", 1e-2))
    min_lr_ratio = float(cfg.get("min_lr_ratio", 0.01))

    if sched_type == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1),
            eta_min=base_lr * min_lr_ratio)
    if sched_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(sched_cfg.get("step_size", 30)),
            gamma=float(sched_cfg.get("gamma", 0.1)))
    if sched_type == "MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(sched_cfg.get("milestones", [150, 225, 275])),
            gamma=float(sched_cfg.get("gamma", 0.1)))
    raise NotImplementedError(
        f"不支持的 scheduler 类型: {sched_type!r}. "
        f"已实现 CosineAnnealingLR / StepLR / MultiStepLR（OneCycleLR 未实现）。")


def _set_warmup_lr(optimizer, base_lrs, epoch: int, warmup_epochs: int,
                   base_lr: float, warmup_start_lr: float):
    """线性 warmup：每个参数组从 warmup_start_lr 比例升至其 base_lr。"""
    if epoch >= warmup_epochs:
        return
    frac = (epoch + 1) / max(warmup_epochs, 1)
    start_frac = warmup_start_lr / max(base_lr, 1e-12)
    scale = start_frac + (1.0 - start_frac) * frac
    for g, bl in zip(optimizer.param_groups, base_lrs):
        g["lr"] = bl * scale


# ============================================================
# EMA
# ============================================================

class ModelEMA:
    """指数移动平均权重，用于验证与 best checkpoint，得到更稳定的模型。

    注意：按 ``state_dict()`` 逐项同步（而非 ``parameters()``），这样会把
    BatchNorm 的 running_mean / running_var 等 buffer 也纳入 EMA。否则
    best.pth 中 BN 统计量会停留在初始化值（running_mean=0 / running_var=1），
    导致 model.eval() 推断得到退化指标（mAP≈0）。
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.updates += 1
        # 前期让 EMA 紧跟模型（模型早期变化快），逐步逼近 decay
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        model_sd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(model_sd[k], alpha=1.0 - d)
            else:
                # 非浮点 buffer（如 num_batches_tracked）直接复制
                v.copy_(model_sd[k])

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)


# ============================================================
# 训练 / 验证
# ============================================================

def _train_one_epoch(model, loader, loss_fn, optimizer, scaler, autocast,
                     device, epoch, grad_clip, ema, model_type):
    model.train()
    running, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Train Epoch {epoch + 1}", leave=False)
    for batch in pbar:
        images = batch["visible"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        num_labels = batch["num_labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast:
            # baseline 仅用 visible；fusion 用 visible + infrared + depth 三模态
            if model_type == "fusion":
                preds = model(
                    images,
                    batch["infrared"].to(device, non_blocking=True),
                    batch["depth"].to(device, non_blocking=True),
                )
            else:
                preds = model(images)
            loss = loss_fn(preds, labels, num_labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        running += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.3f}")
    return running / max(n, 1)


@torch.no_grad()
def _validate(model, loader, loss_fn, device, autocast, model_type,
              conf_thres: float = 0.001, nms_iou: float = 0.6, max_det: int = 300):
    """验证：返回 metrics 字典，含 val_loss 与 mAP@0.5 / mAP@0.5:0.95。

    解码 + NMS + mAP 复用 utils.metrics（与 scripts/evaluate.py 完全一致），
    预测框与 GT 框都在 640×640 letterbox 像素坐标系下比较。
    """
    model.eval()
    total, n = 0.0, 0
    predictions = []     # (img_id, cls, score, xyxy)
    ground_truths = []   # 每图 list[(cls, xyxy)]
    img_id = 0
    H, W = loss_fn.input_size

    for batch in loader:
        images = batch["visible"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        num_labels = batch["num_labels"].to(device, non_blocking=True)
        with autocast:
            # baseline 仅用 visible；fusion 用 visible + infrared + depth 三模态
            if model_type == "fusion":
                preds = model(
                    images,
                    batch["infrared"].to(device, non_blocking=True),
                    batch["depth"].to(device, non_blocking=True),
                )
            else:
                preds = model(images)
            loss = loss_fn(preds, labels, num_labels)
        total += loss.item() * images.size(0)
        n += images.size(0)

        # 解码预测框并收集
        batch_dets = postprocess(preds, model.strides, conf_thres, nms_iou, max_det)
        for b in range(images.size(0)):
            for det in batch_dets[b]:
                predictions.append((img_id + b, det[0], det[1], det[2]))

        # 收集 GT（letterbox 归一化坐标 → 像素 xyxy）
        for b in range(images.size(0)):
            gts = []
            for k in range(int(num_labels[b].item())):
                row = labels[b, k]
                if row[0] < 0:                      # padding 哨兵
                    continue
                cls = int(row[0].item())
                cx, cy, w, h = row[1].item(), row[2].item(), row[3].item(), row[4].item()
                gts.append((cls, np.array(
                    [(cx - w / 2) * W, (cy - h / 2) * H,
                     (cx + w / 2) * W, (cy + h / 2) * H], dtype=np.float64)))
            ground_truths.append(gts)
        img_id += images.size(0)

    metrics = {"val_loss": total / max(n, 1)}
    metrics.update(compute_map(predictions, ground_truths, loss_fn.num_classes))
    return metrics


def _best_score(metrics: dict, best_metric_key: str) -> float:
    """把 metrics 映射为「越大越好」的分数。

    mAP 类指标天然越大越好；val_loss 越小越好，故取负号。mAP 未实现时回退到
    -val_loss。
    """
    if best_metric_key in metrics and metrics[best_metric_key] is not None:
        return float(metrics[best_metric_key])
    return -float(metrics["val_loss"])


# ============================================================
# Checkpoint
# ============================================================

def _save_checkpoint(path, model, optimizer, scheduler, scaler, ema, epoch, best_score):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "best_score": best_score,
    }, path)


def _save_best(path, model, ema, epoch, best_score):
    """best checkpoint 同时保存原始权重与 EMA 权重（若启用）。

    不再把 EMA 权重临时 swap 进 model（避免改动 model 状态，也避免丢失
    原始权重的 BN 统计量）；评估时优先加载 EMA、回退原始权重。
    """
    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "epoch": epoch,
        "best_score": best_score,
    }, path)


def _load_checkpoint(path, model, optimizer, scheduler, scaler, ema, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and ckpt.get("ema"):
        ema.load_state_dict(ckpt["ema"])
    return int(ckpt.get("epoch", -1)) + 1, float(ckpt.get("best_score", -float("inf")))


def _prune_checkpoints(save_dir: Path, max_keep: int):
    if max_keep <= 0:
        return
    ckpts = sorted(save_dir.glob("epoch_*.pth"))
    for p in ckpts[:-max_keep]:
        p.unlink()


def _build_writer(cfg: dict):
    if not cfg.get("tensorboard", {}).get("enabled", False):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = PROJECT_ROOT / _resolve_template(
            cfg["tensorboard"].get("log_dir", "runs/${experiment_name}/tensorboard"), cfg)
        return SummaryWriter(log_dir=str(log_dir))
    except ImportError:
        print("[warn] tensorboard 未安装，跳过 tensorboard 日志。")
        return None


# ============================================================
# 主流程
# ============================================================

def _parse_args():
    """解析命令行参数，支持覆盖配置文件路径（与 evaluate.py 对齐）。

    默认不传参时仍读取 configs/{train,model,dataset}.yaml，保持 Baseline 行为不变。
    """
    parser = argparse.ArgumentParser(description="多模态目标检测训练")
    parser.add_argument("--model_config", type=str, default=None,
                        help="model.yaml 路径，默认 configs/model.yaml")
    parser.add_argument("--train_config", type=str, default=None,
                        help="train.yaml 路径，默认 configs/train.yaml")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="dataset.yaml 路径，默认 configs/dataset.yaml")
    return parser.parse_args()


def main():
    # 使 configs 中的相对路径（dataset_root / save_dir / log_dir）基于项目根解析
    os.chdir(PROJECT_ROOT)

    args = _parse_args()
    cfg_dir = PROJECT_ROOT / "configs"
    train_cfg = _load_yaml(args.train_config or cfg_dir / "train.yaml")
    model_cfg = _load_yaml(args.model_config or cfg_dir / "model.yaml")
    model_type = model_cfg.get("model_type", "baseline")
    dataset_cfg_path = str(args.dataset_config or cfg_dir / "dataset.yaml")

    # ---- 复现性 / 设备 ----
    seed = int(train_cfg.get("seed", 42))
    _set_seed(seed)
    device = _select_device(train_cfg)
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---- 数据 ----
    image_size = tuple(train_cfg.get("image_size", [640, 640]))
    loaders = create_dataloaders(
        str(dataset_cfg_path),
        image_size=image_size,
        batch_size=int(train_cfg.get("batch_size", 16)),
        num_workers=int(train_cfg.get("num_workers", 4)),
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        val_ratio=float(train_cfg.get("val_ratio", 0.2)),
        seed=seed,
    )

    # ---- 模型 ----
    model = build_model(model_cfg).to(device)
    num_classes = model.num_classes
    dataset_num_classes = _load_yaml(dataset_cfg_path).get("num_classes")
    if dataset_num_classes is not None and int(dataset_num_classes) != num_classes:
        print(f"[warn] model.yaml num_classes({num_classes}) != "
              f"dataset.yaml num_classes({dataset_num_classes})")

    # ---- 损失 ----
    loss_fn = DetectionLoss(num_classes=num_classes, strides=model.strides,
                            input_size=image_size)

    # ---- 优化器 / 调度器 / AMP / EMA ----
    optimizer = _build_optimizer(model, train_cfg)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scheduler = _build_scheduler(optimizer, train_cfg)
    autocast, scaler = _get_amp_tools(use_amp)
    ema = ModelEMA(model, decay=float(train_cfg.get("ema", {}).get("decay", 0.9999))) \
        if train_cfg.get("ema", {}).get("enabled", False) else None

    # ---- checkpoint 路径 ----
    ckpt_cfg = train_cfg.get("checkpoint", {})
    save_dir = PROJECT_ROOT / _resolve_template(
        ckpt_cfg.get("save_dir", "runs/${experiment_name}/weights"), train_cfg)
    save_dir.mkdir(parents=True, exist_ok=True)
    last_path = save_dir / "last.pth"
    best_path = save_dir / "best.pth"

    # ---- resume ----
    start_epoch, best_score = 0, -float("inf")
    resume = bool(ckpt_cfg.get("resume", False))
    resume_path = ckpt_cfg.get("resume_path", "") or str(last_path)
    if resume and Path(resume_path).exists():
        start_epoch, best_score = _load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, ema, device)
        print(f"[checkpoint] 从 {resume_path} 恢复，start_epoch={start_epoch}")

    # ---- 日志 ----
    writer = _build_writer(train_cfg)
    epochs = int(train_cfg.get("epochs", 300))
    val_interval = int(train_cfg.get("val_interval", 1))
    grad_clip = float(train_cfg.get("grad_clip_max_norm", 0.0))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    warmup_start_lr = float(train_cfg.get("warmup_start_lr", 1e-6))
    base_lr = float(train_cfg.get("learning_rate", 1e-2))
    save_interval = int(ckpt_cfg.get("save_interval", 10))
    save_last = bool(ckpt_cfg.get("save_last", True))
    save_best = bool(ckpt_cfg.get("save_best", True))
    best_metric_key = str(ckpt_cfg.get("best_metric", "mAP_0.5:0.95"))
    max_keep = int(ckpt_cfg.get("max_keep", 5))

    print(f"[train] device={device} amp={use_amp} epochs={epochs} "
          f"batch={train_cfg.get('batch_size')} "
          f"opt={train_cfg['optimizer']['type']} sched={train_cfg['scheduler']['type']}")
    print(f"[evaluate] 验证指标: val_loss + mAP@0.5 + mAP@0.5:0.95；"
          f"best 模型按 {best_metric_key} 选择（越大越好）。")

    # ---- 训练循环 ----
    for epoch in range(start_epoch, epochs):
        _set_warmup_lr(optimizer, base_lrs, epoch, warmup_epochs, base_lr, warmup_start_lr)

        train_loss = _train_one_epoch(
            model, loaders["train"], loss_fn, optimizer, scaler, autocast,
            device, epoch, grad_clip, ema, model_type)

        if epoch >= warmup_epochs and scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        val_loss = float("nan")
        map50 = float("nan")
        map5095 = float("nan")
        if (epoch + 1) % val_interval == 0:
            metrics = _validate(model, loaders["val"], loss_fn, device, autocast,
                                model_type)
            val_loss = metrics["val_loss"]
            map50 = metrics.get("mAP_0.5", float("nan"))
            map5095 = metrics.get("mAP_0.5:0.95", float("nan"))
            if save_best:
                score = _best_score(metrics, best_metric_key)
                if score > best_score:
                    best_score = score
                    _save_best(best_path, model, ema, epoch, best_score)
                    print(f"[checkpoint] 保存最佳模型 best.pth "
                          f"(epoch={epoch + 1}, val_loss={val_loss:.4f}, "
                          f"mAP@0.5={map50:.4f}, mAP@0.5:0.95={map5095:.4f})")

        if save_last:
            _save_checkpoint(last_path, model, optimizer, scheduler, scaler, ema,
                             epoch, best_score)
        if (epoch + 1) % save_interval == 0:
            _save_checkpoint(save_dir / f"epoch_{epoch + 1}.pth",
                             model, optimizer, scheduler, scaler, ema, epoch, best_score)
            _prune_checkpoints(save_dir, max_keep)

        print(f"Epoch [{epoch + 1}/{epochs}] lr={lr:.6f} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"mAP@0.5={map50:.4f} mAP@0.5:0.95={map5095:.4f}")
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("lr", lr, epoch)

    if writer is not None:
        writer.close()
    print("[train] 训练完成。")


if __name__ == "__main__":
    main()
