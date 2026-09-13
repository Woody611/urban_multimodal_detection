"""scripts/official_map.py — 按赛题官方 mAP@50-95 公式计算，并与 fork 口径对照。

== 官方公式（严格照抄赛题文件文字）==
  1. T = {0.50, 0.55, ..., 0.95}                   共 10 个 IoU 阈值
  2. 按类别分别计算 AP
  3. 将该类别下所有图像的预测框按 confidence 从高到低排序
  4. 按该顺序遍历：若能匹配到「尚未被匹配」的真实框且 IoU >= t → TP，否则 FP
  5. recall(k) = TP(k) / N_gt ;  precision(k) = TP(k) / (TP(k) + FP(k))
  6. p_interp(r) = max{ precision : recall >= r }；无满足者取 0
  7. AP = (1/101) * Σ_{r ∈ {0,0.01,...,1}} p_interp(r)      ← 101 点「算术平均」
  8. mAP(t) = mean_c AP_c(t)（无 GT 的类别忽略）
  9. mAP@50-95 = mean_t mAP(t)

== fork（ultralytics）口径，用于对照 ==
  ultralytics/utils/metrics.py::compute_ap  (L1204-1214)
      mrec = [0, recall, 1.0] ; mpre = [1.0, precision, 0.0]
      mpre = flip(maximum.accumulate(flip(mpre)))          # envelope 同官方
      ap = np.trapz(np.interp(linspace(0,1,101), mrec, mpre), linspace(0,1,101))
  与官方的两处差异：
      (a) 求和方式：np.trapz 梯形积分  vs  官方 101 点算术平均
      (b) 尾部处理：超过 R_max 的 r 上，fork 从 (R_max, p_n) 线性插值到 (1.0, 0)
          （即插出一条「斜坡」），官方则取 0
  另外 fork 的匹配是「按 IoU 降序贪心」(engine/validator.py:252)，官方是按 conf 顺序。

本脚本把这两处差异都做成开关，便于逐项比对：
  --avg   {mean,trapz}   求和方式（官方=mean）
  --tail  {zero,ramp}    尾部处理（官方=zero）
  --match {conf,iou}     匹配顺序（官方=conf）
默认输出「官方口径」与「fork 口径」两列。

用法:
  python scripts/official_map.py --results submissions/phase2/predict_XXX/results
  python scripts/official_map.py --results <dir> --min_conf 0.005     # 离线 conf 扫描
  python scripts/official_map.py --results <dir> --sensitivity        # 打印口径敏感性
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NC = 12
IOUV_OFFICIAL = np.arange(0.5, 0.95 + 1e-9, 0.05)  # 官方 T，10 个阈值
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EPS = np.finfo(np.float32).eps


# ============================================================
# 基础工具
# ============================================================

def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """xyxy 成对 IoU。a:(N,4) b:(M,4) -> (N,M)。"""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, EPS)


def read_txt(path: Path, with_conf: bool) -> np.ndarray:
    """读 YOLO 格式 txt。with_conf=True -> (n,6) [cls,cx,cy,w,h,conf]；否则 (n,5)。"""
    n = 6 if with_conf else 5
    if not path.exists():
        return np.zeros((0, n), dtype=np.float64)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        p = s.split()
        if len(p) < n:
            continue
        rows.append([float(x) for x in p[:n]])
    return np.array(rows, dtype=np.float64).reshape(-1, n) if rows else np.zeros((0, n), dtype=np.float64)


def norm_xywh_to_native_xyxy(arr: np.ndarray, w: int, h: int):
    """(n,5)[cls,cx,cy,w,h] 归一化 -> native 像素 xyxy + cls。"""
    if len(arr) == 0:
        return np.zeros((0, 4)), np.zeros((0,))
    cx, cy, bw, bh = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h], axis=1)
    return xyxy, arr[:, 0]


# ============================================================
# AP 计算：官方 vs fork
# ============================================================

def ap_from_curve(recall: np.ndarray, precision: np.ndarray,
                  avg: str = "mean", tail: str = "zero") -> float:
    """由 recall/precision 曲线算 AP。

    avg : 'mean'  官方 101 点算术平均 | 'trapz' fork 梯形积分
    tail: 'zero'  官方：无满足 recall>=r 的点时取 0
          'ramp'  fork：mrec/mpre 末尾补 (1.0, 0.0) 后线性插值出斜坡
    """
    if recall.size == 0:
        return 0.0
    x = np.linspace(0.0, 1.0, 101)

    if tail == "ramp":
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        q = np.interp(x, mrec, mpre)
    else:  # 'zero'
        q = np.zeros_like(x)
        for i, t in enumerate(x):
            m = precision[recall >= t]
            if m.size:
                q[i] = m.max()

    return float(q.mean()) if avg == "mean" else float(np.trapz(q, x))


def curve_for_class(preds_c, gt_by_img, iou_thr: float, match: str = "conf"):
    """单类、单 IoU 阈值下构建 PR 曲线。

    Args:
        preds_c: list[(img_id, conf, box_xyxy)]，未排序
        gt_by_img: dict {img_id: (M,4) xyxy}
        match: 'conf' 官方按 conf 顺序贪心 | 'iou' fork 按 IoU 降序贪心

    Returns:
        (recall, precision) 两个 np.ndarray
    """
    npos = int(sum(len(v) for v in gt_by_img.values()))
    if npos == 0 or len(preds_c) == 0:
        return np.zeros(0), np.zeros(0)

    # 按 conf 降序（官方: 稳定排序；fork 的 argsort 非稳定，这里统一用稳定排序）
    order = sorted(range(len(preds_c)), key=lambda i: -preds_c[i][1])
    preds_sorted = [preds_c[i] for i in order]

    matched = {img: np.zeros(len(v), dtype=bool) for img, v in gt_by_img.items()}
    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))

    if match == "conf":
        # 官方文字: 按 conf 顺序遍历，匹配「尚未被匹配」的真实框中 IoU 最大者
        for i, (img, _c, box) in enumerate(preds_sorted):
            gts = gt_by_img.get(img)
            if gts is None or len(gts) == 0:
                fp[i] = 1.0
                continue
            ious = box_iou_np(box[None, :], gts)[0]
            cand = np.where(~matched[img])[0]
            if cand.size and ious[cand].max() >= iou_thr:
                j = cand[int(np.argmax(ious[cand]))]
                tp[i] = 1.0
                matched[img][j] = True
            else:
                fp[i] = 1.0
    else:
        # fork: engine/validator.py:252 —— 所有 (gt,det) 对按 IoU 降序贪心，保证一对一
        all_m = [(img, i) for i, (img, _c, _b) in enumerate(preds_sorted)]
        pairs = []
        for img, i in all_m:
            gts = gt_by_img.get(img)
            if gts is None or len(gts) == 0:
                continue
            ious = box_iou_np(preds_sorted[i][2][None, :], gts)[0]
            for j, v in enumerate(ious):
                if v >= iou_thr:
                    pairs.append((v, i, j, img))
        pairs.sort(key=lambda z: -z[0])
        used_det, used_gt = set(), {}
        for v, i, j, img in pairs:
            if i in used_det:
                continue
            used_gt.setdefault(img, set())
            if j in used_gt[img]:
                continue
            used_det.add(i)
            used_gt[img].add(j)
            tp[i] = 1.0
        fp = 1.0 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / npos
    precision = tp_cum / np.maximum(tp_cum + fp_cum, EPS)
    return recall, precision


# ============================================================
# 数据收集
# ============================================================

def collect(images_dir: Path, labels_dir: Path, results_dir: Path,
            min_conf: float, drop_corrupt: bool):
    """返回 (gt_by_cls, preds_by_cls, stats)。坐标一律 native 像素 xyxy。"""
    gt_by_cls = {c: {} for c in range(NC)}
    preds_by_cls = {c: [] for c in range(NC)}
    stats = dict(images=0, corrupt=0, gt=0, pred=0, no_pred=0)

    img_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    for img_id, img_path in enumerate(img_files):
        stem = img_path.stem
        stats["images"] += 1
        im = cv2.imread(str(img_path))
        if im is None:
            stats["corrupt"] += 1
            continue
        h, w = im.shape[:2]

        gt = read_txt(labels_dir / f"{stem}.txt", with_conf=False)
        if drop_corrupt and len(gt) and (gt[:, 1:].max() > 1.0 or gt[:, 1:].min() < 0.0):
            stats["corrupt"] += 1          # 与 verify_image_label 一致：越界标签整图作废
            continue
        if len(gt):
            boxes, cls = norm_xywh_to_native_xyxy(gt, w, h)
            for c in np.unique(cls):
                m = cls == c
                gt_by_cls[int(c)][img_id] = boxes[m]
                stats["gt"] += int(m.sum())

        pred = read_txt(results_dir / f"{stem}.txt", with_conf=True)
        if min_conf > 0 and len(pred):
            pred = pred[pred[:, 5] >= min_conf]     # 离线 conf 扫描
        if len(pred) == 0:
            stats["no_pred"] += 1
            continue
        boxes, cls, conf = norm_xywh_to_native_xyxy(pred[:, :5], w, h)[0], pred[:, 0], pred[:, 5]
        for k in range(len(boxes)):
            c = int(cls[k])
            if 0 <= c < NC:
                preds_by_cls[c].append((img_id, float(conf[k]), boxes[k]))
                stats["pred"] += 1
    return gt_by_cls, preds_by_cls, stats


def evaluate(gt_by_cls, preds_by_cls, avg: str, tail: str, match: str):
    """返回 dict: {'mAP50':.., 'mAP50-95':.., 'per_iou':[...]}"""
    per_iou = []
    for t in IOUV_OFFICIAL:
        aps = []
        for c in range(NC):
            rec, prec = curve_for_class(preds_by_cls[c], gt_by_cls[c], float(t), match=match)
            ap = ap_from_curve(rec, prec, avg=avg, tail=tail)
            if len(gt_by_cls[c]) == 0:
                continue          # 官方: 无 GT 的类别忽略
            aps.append(ap)
        per_iou.append(float(np.mean(aps)) if aps else 0.0)
    per_iou = np.array(per_iou)
    return {"mAP50": float(per_iou[0]), "mAP75": float(per_iou[5]),
            "mAP50-95": float(per_iou.mean()), "per_iou": per_iou}


def main():
    ap = argparse.ArgumentParser(description="官方口径 mAP@50-95（含 fork 对照）")
    ap.add_argument("--results", required=True, help="predict.py 生成的 results 目录")
    ap.add_argument("--split_root", default="data/processed/depth_split_train")
    ap.add_argument("--min_conf", type=float, default=0.0, help="离线 conf 过滤（等价更高 conf_thres）")
    ap.add_argument("--keep_corrupt", action="store_true", help="不剔除越界标签图（默认剔除）")
    ap.add_argument("--sensitivity", action="store_true", help="额外打印口径敏感性")
    args = ap.parse_args()

    split_root = (PROJECT_ROOT / args.split_root).resolve()
    images_dir = split_root / "images" / "val" / "visible"
    labels_dir = split_root / "labels" / "val" / "visible"
    results_dir = (PROJECT_ROOT / args.results).resolve()
    for p, nm in ((images_dir, "val visible"), (labels_dir, "val labels"), (results_dir, "results")):
        if not p.is_dir():
            raise FileNotFoundError(f"{nm} 目录不存在: {p}")

    gt, preds, st = collect(images_dir, labels_dir, results_dir,
                            args.min_conf, drop_corrupt=not args.keep_corrupt)

    official = evaluate(gt, preds, avg="mean", tail="zero", match="conf")
    fork = evaluate(gt, preds, avg="trapz", tail="ramp", match="iou")

    print("=" * 68)
    print("官方口径 vs fork 口径 mAP@50-95")
    print("=" * 68)
    print(f"images={st['images']}  corrupt剔除={st['corrupt']}  GT={st['gt']}  "
          f"pred={st['pred']}  min_conf={args.min_conf}")
    print("-" * 68)
    print(f"{'口径':<34}{'mAP50':>11}{'mAP75':>11}{'mAP50-95':>12}")
    print(f"{'官方 (mean + tail0 + conf匹配)':<34}{official['mAP50']:>11.5f}"
          f"{official['mAP75']:>11.5f}{official['mAP50-95']:>12.5f}")
    print(f"{'fork (trapz + ramp + IoU匹配)':<34}{fork['mAP50']:>11.5f}"
          f"{fork['mAP75']:>11.5f}{fork['mAP50-95']:>12.5f}")
    print(f"{'差值 (官方 - fork)':<34}{official['mAP50']-fork['mAP50']:>+11.5f}"
          f"{official['mAP75']-fork['mAP75']:>+11.5f}"
          f"{official['mAP50-95']-fork['mAP50-95']:>+12.5f}")

    if args.sensitivity:
        print("-" * 68)
        print("口径敏感性（拆开两处差异 + 匹配顺序）")
        print(f"{'avg':<7}{'tail':<7}{'match':<8}{'mAP50-95':>12}")
        for avg_ in ("mean", "trapz"):
            for tail in ("zero", "ramp"):
                for mt in ("conf", "iou"):
                    r = evaluate(gt, preds, avg_, tail, mt)
                    tag = " <-官方" if (avg_, tail, mt) == ("mean", "zero", "conf") else (
                          " <-fork" if (avg_, tail, mt) == ("trapz", "ramp", "iou") else "")
                    print(f"{avg_:<7}{tail:<7}{mt:<8}{r['mAP50-95']:>12.5f}{tag}")
    print("=" * 68)


if __name__ == "__main__":
    main()
