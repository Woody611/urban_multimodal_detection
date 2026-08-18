"""scripts/evaluate.py — 模型评估入口。

职责：加载训练好的权重（best.pth / last.pth）→ 构建与训练一致的验证集 →
在 ``torch.no_grad()`` 下推断 → 输出验证损失（val_loss）与目标检测指标
（mAP@0.5、mAP@0.5:0.95）。

用法（在项目根目录或任意位置均可）:
    python scripts/evaluate.py
    python scripts/evaluate.py --weights runs/urban_multimodal_det_v1/weights/best.pth
    python scripts/evaluate.py --dataset_root /path/to/data/raw --device cpu

依赖：
    models.build_model            — 按 model.yaml 的 model_type 构建检测器
    utils.dataset                 — 复现 create_dataloaders 的 train→val 切分
    scripts.train.DetectionLoss   — 复用训练时的损失，保证 val_loss 口径一致
    scripts.train._decode_boxes   — 复用训练时的框解码约定（cx/cy/w/h → 像素）

说明：
    - 验证集与训练时完全一致：从 train 目录按 val_ratio（默认 0.2）、seed
      （默认 42）在 stem 级别切分，避免数据泄漏与模态错配。
    - mAP 按 COCO 约定实现（101 点插值 AP，类别内取均值，无 GT 的类别忽略），
      在 640×640 的 letterbox 坐标空间中直接比较预测框与 GT 框。
    - 当前面向 visible 单模态 baseline；若 model_type=fusion 需扩展输入模态。
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

# 保证 `python scripts/evaluate.py` 时能导入项目内的 models / utils / scripts
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model                          # noqa: E402
from utils.dataset import (                             # noqa: E402
    load_dataset_config,
    MultiModalDataset,
    split_train_val,
    collate_fn,
)
# 复用训练脚本中的损失与框解码，保证评估口径与训练一致（不重复实现）
from scripts.train import DetectionLoss, _decode_boxes  # noqa: E402


# ============================================================
# 基础工具
# ============================================================

def _load_yaml(path):
    """读取 yaml 配置文件，返回解析后的字典。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_template(s, cfg):
    """把路径模板中的 ``${key}`` 替换为顶层同名值（与 train.py 一致）。"""
    return re.sub(r"\$\{(\w+)\}", lambda m: str(cfg.get(m.group(1), m.group(0))), str(s))


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _select_device(device_arg: str) -> torch.device:
    requested = str(device_arg).lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda:0")
    if requested.startswith("cuda"):
        print("[device] CUDA 不可用，自动回退到 CPU。")
    return torch.device("cpu")


def _bn_initialized(state_dict, tol: float = 1e-3) -> bool:
    """判断权重的 BatchNorm 运行统计量是否为「已训练」状态。

    训练前 BN 的 running_mean=0 / running_var=1；若全部 BN 仍停留在该初值，
    说明这些运行统计量从未被更新（常见于 EMA 只更新参数、不更新 buffer 的
    bug），此时用 model.eval() 推断会得到退化结果。返回 False 表示疑似退化。
    """
    rvs = [t for k, t in state_dict.items() if k.endswith("running_var")]
    if not rvs:
        return True  # 无 BN 层，无需检查
    all_at_init = all((torch.abs(t - 1.0) < tol).all().item() for t in rvs)
    return not all_at_init


# ============================================================
# 后处理：解码 + NMS（输出归一化像素坐标 xyxy）
# ============================================================

def _box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """向量化 IoU。boxes1 (N,4)、boxes2 (M,4)，均为 xyxy，返回 (N, M)。"""
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, np.finfo(np.float32).eps)


def _nms_keep(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """贪心 NMS。boxes (N,4) xyxy、scores (N,)，返回保留的索引（按分数降序）。"""
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = _box_iou_np(boxes[order[1:]], boxes[i:i + 1])[:, 0]
        order = order[1:][ious < iou_thr]
    return np.asarray(keep, dtype=np.int64)


def _postprocess(preds, strides, conf_thres: float, nms_iou: float,
                 max_det: int = 300):
    """把多尺度 HeadPredictions 解码为每图检测列表。

    置信度 = sigmoid(obj) * max_c(sigmoid(cls))；每类单独 NMS。

    Returns:
        list[list[tuple]]，长度为 batch size，每图为
        ``(class_id, score, xyxy)`` 元组列表（已按分数降序、截断到 max_det）。
    """
    B = preds[0].cls.shape[0]
    per_image = [[] for _ in range(B)]

    for stride, pred in zip(strides, preds):
        obj = torch.sigmoid(pred.obj)       # [B, 1, fh, fw]
        cls_p = torch.sigmoid(pred.cls)     # [B, C, fh, fw]
        score = obj * cls_p                 # [B, C, fh, fw]
        boxes = _decode_boxes(pred.reg, stride)  # [B, fh, fw, 4] (cx, cy, w, h)
        xyxy = torch.cat(
            [boxes[..., :2] - boxes[..., 2:] / 2.0,
             boxes[..., :2] + boxes[..., 2:] / 2.0], dim=-1)  # [B, fh, fw, 4]

        for b in range(B):
            # torch.max(dim) 返回 (values, indices)：values=类别最大分，indices=类别下标
            max_sc, cls_ids = score[b].max(dim=0)  # 各 [fh, fw]
            keep_mask = max_sc > conf_thres
            if not keep_mask.any():
                continue
            c = cls_ids[keep_mask].cpu().numpy()
            s = max_sc[keep_mask].cpu().numpy()
            bx = xyxy[b][keep_mask].cpu().numpy()
            for cls in np.unique(c):
                idx = np.where(c == cls)[0]
                order_idx = idx[np.argsort(-s[idx])]
                for k in _nms_keep(bx[order_idx], s[order_idx], nms_iou):
                    kk = int(order_idx[k])
                    per_image[b].append((int(cls), float(s[kk]), bx[kk]))

    out = []
    for dets in per_image:
        dets.sort(key=lambda x: -x[1])
        out.append(dets[:max_det])
    return out


# ============================================================
# mAP 计算（COCO 风格，101 点插值 AP）
# ============================================================

def _ap_class(preds_c, gt_by_img, iou_thr: float) -> float:
    """单类、单 IoU 阈值的 AP。

    Args:
        preds_c: list[(img_id, score, box)]，已按 score 降序。
        gt_by_img: dict {img_id: np.ndarray (M, 4) xyxy}，仅含该类的 GT。
    """
    npos = int(sum(len(v) for v in gt_by_img.values()))
    tp = np.zeros(len(preds_c))
    fp = np.zeros(len(preds_c))
    matched = {img: np.zeros(len(v), dtype=bool) for img, v in gt_by_img.items()}

    for i, (img, _score, box) in enumerate(preds_c):
        box = np.asarray(box, dtype=np.float64)
        if img in gt_by_img and gt_by_img[img].shape[0] > 0:
            ious = _box_iou_np(box[None, :], gt_by_img[img])[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not matched[img][j]:
                tp[i] = 1.0
                matched[img][j] = True
            else:
                fp[i] = 1.0
        else:
            fp[i] = 1.0

    if npos == 0:
        return float("nan")

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / npos
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float32).eps)

    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        m = precision[recall >= t]
        ap += float(m.max()) if m.size else 0.0
    return ap / 101.0


def _compute_ap_list(predictions, ground_truths, num_classes: int,
                     iou_thr: float):
    """按类别计算 AP，返回长度 num_classes 的列表（无 GT 的类别为 NaN）。"""
    aps = []
    for c in range(num_classes):
        preds_c = [(img, sc, bx) for (img, cls, sc, bx) in predictions if cls == c]
        preds_c.sort(key=lambda x: -x[1])
        gt_by_img = {}
        for img, gts in enumerate(ground_truths):
            boxes = [bx for (cls, bx) in gts if cls == c]
            if boxes:
                gt_by_img[img] = np.asarray(boxes, dtype=np.float64)
        aps.append(_ap_class(preds_c, gt_by_img, iou_thr) if gt_by_img else float("nan"))
    return aps


# ============================================================
# 验证集构建（复现 create_dataloaders 的 train→val 切分）
# ============================================================

def _build_val_loader(dataset_cfg_path, dataset_root, image_size, batch_size,
                      num_workers, pin_memory, val_ratio, seed):
    """构建与训练完全一致的验证集 DataLoader。

    由于 create_dataloaders 不支持 dataset_root 覆盖，这里手动复现其逻辑：
    从 train 目录构建完整数据集，按 val_ratio + seed 在 stem 级别切出验证子集。
    """
    cfg = load_dataset_config(dataset_cfg_path)
    if dataset_root:
        cfg["dataset_root"] = dataset_root

    target_size = tuple(image_size)
    full_train_ds = MultiModalDataset(cfg, split="train", target_size=target_size)
    _train_stems, val_stems = split_train_val(full_train_ds.samples, val_ratio, seed)
    val_set = set(val_stems)
    val_idx = [i for i, s in enumerate(full_train_ds.samples) if s in val_set]
    val_ds = Subset(full_train_ds, val_idx)

    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return loader, len(val_ds)


# ============================================================
# 主流程
# ============================================================

def _parse_args():
    parser = argparse.ArgumentParser(description="多模态目标检测模型评估")
    parser.add_argument("--weights", type=str, default=None,
                        help="权重文件路径；默认取 train.yaml checkpoint.save_dir/best.pth")
    parser.add_argument("--no_ema", action="store_true",
                        help="不使用 EMA 权重，改加载 checkpoint 中的原始 'model' 权重")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="覆盖 dataset.yaml 的 dataset_root（默认 data/raw）")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="dataset.yaml 路径，默认 configs/dataset.yaml")
    parser.add_argument("--model_config", type=str, default=None,
                        help="model.yaml 路径，默认 configs/model.yaml")
    parser.add_argument("--train_config", type=str, default=None,
                        help="train.yaml 路径，默认 configs/train.yaml")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda | cpu")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖 train.yaml 的 batch_size")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="覆盖 train.yaml 的 num_workers")
    parser.add_argument("--image_size", type=int, nargs=2, default=None,
                        metavar=("H", "W"), help="覆盖 train.yaml 的 image_size")
    parser.add_argument("--conf_thres", type=float, default=0.001,
                        help="置信度阈值（用于构建 PR 曲线）")
    parser.add_argument("--iou_thres", type=float, default=0.6,
                        help="NMS 的 IoU 阈值")
    parser.add_argument("--max_det", type=int, default=300,
                        help="每张图保留的最大检测数")
    return parser.parse_args()


def main():
    # 使 configs 中的相对路径（dataset_root / save_dir）基于项目根解析
    os.chdir(PROJECT_ROOT)

    args = _parse_args()
    cfg_dir = PROJECT_ROOT / "configs"
    train_cfg = _load_yaml(args.train_config or cfg_dir / "train.yaml")
    model_cfg = _load_yaml(args.model_config or cfg_dir / "model.yaml")
    dataset_cfg_path = str(args.dataset_config or cfg_dir / "dataset.yaml")

    seed = int(train_cfg.get("seed", 42))
    _set_seed(seed)
    device = _select_device(args.device)

    image_size = (tuple(args.image_size) if args.image_size
                  else tuple(train_cfg.get("image_size", [640, 640])))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 16))
    num_workers = (args.num_workers if args.num_workers is not None
                   else int(train_cfg.get("num_workers", 4)))
    val_ratio = float(train_cfg.get("val_ratio", 0.2))

    # ---- 权重路径 ----
    if args.weights is None:
        ckpt_cfg = train_cfg.get("checkpoint", {})
        save_dir = PROJECT_ROOT / _resolve_template(
            ckpt_cfg.get("save_dir", "runs/${experiment_name}/weights"), train_cfg)
        weights_path = save_dir / "best.pth"
    else:
        weights_path = Path(args.weights)

    if not weights_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    # ---- 模型 ----
    if str(model_cfg.get("model_type", "baseline")).lower() != "baseline":
        print("[warn] 当前 evaluate 面向 visible 单模态 baseline，"
              "fusion 模型需扩展输入模态处理。")
    model = build_model(model_cfg).to(device)
    num_classes = model.num_classes

    # ---- 加载权重（默认优先 EMA，可 --no_ema 强制用原始权重）----
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    sd_ema = ckpt.get("ema")
    sd_model = ckpt.get("model")

    # 选择权重来源；若 EMA 的 BN 运行统计量退化（running_var 恒为 1），
    # 说明训练期 EMA 未同步 BN buffer（train.py 的已知问题），此时回退到
    # 仍有有效 BN 统计量的 'model' 权重，否则模型推断会得到退化指标。
    state_dict = sd_ema if (sd_ema is not None and not args.no_ema) else sd_model
    source = "ema" if (state_dict is sd_ema) else "model"
    if state_dict is None:
        raise ValueError(f"权重文件 {weights_path} 中未找到 'model'/'ema' 字段。")

    if state_dict is sd_ema and not _bn_initialized(state_dict):
        if sd_model is not None and _bn_initialized(sd_model):
            print("[warn] EMA 权重的 BatchNorm 运行统计量处于初值（running_var≈1），"
                  "疑似训练期 EMA 未同步 BN buffer；已自动回退到 'model' 权重。")
            state_dict = sd_model
            source = "model"
        else:
            print("[warn] 当前权重 BatchNorm 运行统计量处于初值（running_var≈1），"
                  "推断结果可能严重退化（mAP≈0）。建议使用 "
                  "`--weights .../last.pth --no_ema`，或修复 train.py 中 "
                  "ModelEMA.update 未同步 BN buffer 的问题后重新训练。")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"权重与当前模型结构不匹配（请确认 model.yaml 与训练时一致）:\n{e}") from e
    model.eval()
    epoch = int(ckpt.get("epoch", -1)) + 1
    best_score = ckpt.get("best_score", None)

    # ---- 验证集（与训练切分一致）----
    val_loader, n_val = _build_val_loader(
        dataset_cfg_path, args.dataset_root, image_size, batch_size,
        num_workers, pin_memory=(device.type == "cuda"),
        val_ratio=val_ratio, seed=seed)

    # ---- 损失（复用训练口径）----
    loss_fn = DetectionLoss(num_classes=num_classes, strides=model.strides,
                            input_size=image_size)

    # ---- 推断 + 指标收集 ----
    predictions = []     # (img_id, cls, score, xyxy)
    ground_truths = []   # 每图 list[(cls, xyxy)]
    total_loss, n, img_id = 0.0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["visible"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            num_labels = batch["num_labels"].to(device, non_blocking=True)

            preds = model(images)
            loss = loss_fn(preds, labels, num_labels)
            total_loss += loss.item() * images.size(0)
            n += images.size(0)

            # 解码预测
            batch_dets = _postprocess(preds, model.strides,
                                      args.conf_thres, args.iou_thres, args.max_det)
            for b in range(images.size(0)):
                for det in batch_dets[b]:
                    predictions.append((img_id + b, det[0], det[1], det[2]))

            # 收集 GT（letterbox 归一化坐标 → 640×640 像素 xyxy）
            H, W = image_size
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

    val_loss = total_loss / max(n, 1)

    # ---- mAP ----
    ap_list_50 = _compute_ap_list(predictions, ground_truths, num_classes, 0.5)
    mAP50 = float(np.nanmean(ap_list_50))
    thr_list = np.arange(0.5, 0.95 + 1e-9, 0.05)
    mAP5095 = float(np.nanmean([
        _compute_ap_list(predictions, ground_truths, num_classes, t)
        for t in thr_list
    ]))

    # ---- 输出 ----
    dataset_cfg = load_dataset_config(dataset_cfg_path)
    class_names = dataset_cfg.get("class_names", {})

    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"weights      : {weights_path}")
    print(f"weight src   : {source} (ema/model)")
    print(f"epoch        : {epoch if epoch > 0 else 'unknown'}")
    if best_score is not None:
        print(f"best_score   : {best_score:.6f} (训练期代理指标, 越大越好)")
    print(f"device       : {device}")
    print(f"val samples  : {n_val}")
    print(f"num preds    : {len(predictions)}")
    print("-" * 60)
    print(f"val_loss     : {val_loss:.6f}")
    print(f"mAP@0.5      : {mAP50:.4f}")
    print(f"mAP@0.5:0.95 : {mAP5095:.4f}")
    print("-" * 60)
    print("per-class AP@0.5:")
    for c in range(num_classes):
        name = class_names.get(c, str(c))
        ap = ap_list_50[c]
        ap_str = f"{ap:.4f}" if np.isfinite(ap) else "n/a (无 GT)"
        print(f"  {c:>2} {name:<12} {ap_str}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
