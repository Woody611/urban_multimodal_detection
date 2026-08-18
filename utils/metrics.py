"""目标检测评价指标：框解码、NMS 后处理与 mAP 计算。

供 scripts/train.py 的验证阶段与 scripts/evaluate.py 复用，避免重复实现。
预测框与 GT 框都在同一输入尺寸（如 640×640）的像素坐标系下比较。

mAP 采用 COCO 约定：101 点插值 AP、类别内取均值、无 GT 的类别忽略。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

__all__ = [
    "decode_boxes",
    "box_iou_np",
    "nms_keep",
    "postprocess",
    "compute_ap_list",
    "compute_map",
]


# ============================================================
# 框解码（anchor-free，YOLOv5 风格，数值稳定）
# ============================================================

def decode_boxes(reg: torch.Tensor, stride: int) -> torch.Tensor:
    """把 reg 输出解码为像素坐标框。

    reg: [B, 4, fh, fw]，通道顺序 (cx, cy, w, h)，均为线性 logit。
    解码约定:
        cx = (sigmoid(reg0) * 2 - 0.5 + gx) * stride
        cy = (sigmoid(reg1) * 2 - 0.5 + gy) * stride
        w  = (sigmoid(reg2) * 2)^2 * stride
        h  = (sigmoid(reg3) * 2)^2 * stride
    返回: [B, fh, fw, 4] 像素坐标 (cx, cy, w, h)。
    """
    B, _, fh, fw = reg.shape
    reg = reg.permute(0, 2, 3, 1).contiguous()          # [B, fh, fw, 4]
    gy, gx = torch.meshgrid(
        torch.arange(fh, device=reg.device, dtype=reg.dtype),
        torch.arange(fw, device=reg.device, dtype=reg.dtype),
        indexing="ij",
    )
    cx = (torch.sigmoid(reg[..., 0]) * 2.0 - 0.5 + gx) * stride
    cy = (torch.sigmoid(reg[..., 1]) * 2.0 - 0.5 + gy) * stride
    w = (torch.sigmoid(reg[..., 2]) * 2.0) ** 2 * stride
    h = (torch.sigmoid(reg[..., 3]) * 2.0) ** 2 * stride
    return torch.stack([cx, cy, w, h], dim=-1)          # [B, fh, fw, 4]


# ============================================================
# NMS 后处理
# ============================================================

def box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
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


def nms_keep(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
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
        ious = box_iou_np(boxes[order[1:]], boxes[i:i + 1])[:, 0]
        order = order[1:][ious < iou_thr]
    return np.asarray(keep, dtype=np.int64)


def postprocess(preds, strides: Sequence[int], conf_thres: float,
                nms_iou: float, max_det: int = 300):
    """把多尺度 HeadPredictions 解码为每图检测列表。

    置信度 = sigmoid(obj) * max_c(sigmoid(cls))；每类单独 NMS。

    Args:
        preds: Tuple[HeadPredictions, ...]，每尺度含 cls/reg/obj。
        strides: 各尺度下采样倍率。

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
        boxes = decode_boxes(pred.reg, stride)  # [B, fh, fw, 4] (cx, cy, w, h)
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
            # 转 float32 再转 numpy：避免 AMP(fp16) 下坐标/分数精度不足影响 IoU 判断
            s = max_sc[keep_mask].float().cpu().numpy()
            bx = xyxy[b][keep_mask].float().cpu().numpy()
            for cls in np.unique(c):
                idx = np.where(c == cls)[0]
                order_idx = idx[np.argsort(-s[idx])]
                for k in nms_keep(bx[order_idx], s[order_idx], nms_iou):
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
            ious = box_iou_np(box[None, :], gt_by_img[img])[0]
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


def compute_ap_list(predictions, ground_truths, num_classes: int,
                    iou_thr: float) -> List[float]:
    """按类别计算 AP，返回长度 num_classes 的列表（无 GT 的类别为 NaN）。

    Args:
        predictions: list[(img_id, cls, score, xyxy)]。
        ground_truths: 每图 list[(cls, xyxy)]。
    """
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


def compute_map(predictions, ground_truths, num_classes: int) -> Dict[str, float]:
    """计算 mAP@0.5 与 mAP@0.5:0.95。

    Returns:
        {"mAP_0.5": float, "mAP_0.5:0.95": float}，键名与 train.yaml 的
        evaluation_metric 一致。
    """
    mAP50 = float(np.nanmean(compute_ap_list(predictions, ground_truths, num_classes, 0.5)))
    thr_list = np.arange(0.5, 0.95 + 1e-9, 0.05)
    mAP5095 = float(np.nanmean([
        compute_ap_list(predictions, ground_truths, num_classes, t)
        for t in thr_list
    ]))
    return {"mAP_0.5": mAP50, "mAP_0.5:0.95": mAP5095}
