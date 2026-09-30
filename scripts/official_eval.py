"""scripts/official_eval.py — 冻结的官方口径本地评测器（本项目唯一 official local metric）。

设计原则：**Measurement before Optimization。**
本模块只做测量，不参与训练、不参与模型选择阈值搜索。

------------------------------------------------------------------
官方定义（来源：面向城市场景的视觉多模态目标检测-1.pdf）
------------------------------------------------------------------
§六.2  IoU 阈值 T = {0.50, 0.55, …, 0.95}（10 个）
      排序：该类别下所有测试图预测框按 confidence 从高到低
      匹配：贪心 one-to-one；与**尚未被匹配**的 GT 满足 IoU ≥ t → TP，
            否则 FP；未被匹配的 GT → FN
§六.3  AP = 101 点插值：r ∈ {0, 0.01, …, 1.0}，p_interp(r) = max{precision : recall ≥ r}，
            **算术平均** AP = (1/101) Σ p_interp(r)
§六.4  mAP(t) = (1/N) Σ_c AP_c(t)      ← N 的取值见下 "DENOMINATOR"
§六.5  mAP@50-95 = (1/10) Σ_t mAP(t)
§九(二) 每张图最大预测框数量 **100**，超过按 confidence 截断
         非法类别 / 非法坐标 / 置信度缺失 → 该预测无效

------------------------------------------------------------------
DENOMINATOR（D2，本项目未决项）
------------------------------------------------------------------
PDF §六.4 字面写 "其中 N 是类别总数" —— 字面即固定 12。
但 COCO 惯例（ultralytics 亦同）只对**存在 GT** 的类别求平均（跳过 AP==-1）。
PDF 未就本数据集明确写死，**标记为由主办方确认的未决项**。

  METRIC_A = 固定 12 类分母（无 GT 的类 AP 记 0 后计入）
  METRIC_B = 仅计入存在 GT 的类别（COCO 惯例）

本模块两个都算，**不预设哪一个是官方**。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ============================================================
# 冻结规格常量 —— 任何评测入口都必须引用这里，不得各自另写
# ============================================================
NC = 12
MAX_BOXES_PER_IMAGE = 100                      # §九(二) 硬上限
IOU_THRESHOLDS = np.arange(0.5, 0.95 + 1e-9, 0.05)   # 10 个
N_INTERP = 101                                 # §六.3
CLASS_ID_MIN, CLASS_ID_MAX = 0, NC - 1
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EPS = np.finfo(np.float32).eps

METRIC_A = "A"   # 固定 12 类分母
METRIC_B = "B"   # 仅有 GT 的类进入分母


# ============================================================
# 基础工具
# ============================================================
def _imsize(path: Path):
    """只读图像头取 (W, H)，避免 cv2 全解码（评测需反复跑，性能关键）。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size          # (W, H)
    except Exception:
        import cv2
        im = cv2.imread(str(path))
        return (im.shape[1], im.shape[0]) if im is not None else (0, 0)


def read_pred_txt(path: Path):
    """读预测 TXT。返回 (n_total, valid[n,6], n_invalid)。

    非法 = 字段数≠6 / 不可解析 / 非有限 / 类别越界。
    **坐标越界不计为"非法行"，但会被记录并在归一化后裁剪**（见 normalize_preds）。
    """
    if not path.exists():
        return 0, np.zeros((0, 6), np.float32), 0
    rows = []
    n_total = 0
    n_bad = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        n_total += 1
        p = s.split()
        if len(p) != 6:
            n_bad += 1
            continue
        try:
            v = [float(x) for x in p]
        except ValueError:
            n_bad += 1
            continue
        if not np.isfinite(v).all():
            n_bad += 1
            continue
        if not (CLASS_ID_MIN <= v[0] <= CLASS_ID_MAX):
            n_bad += 1
            continue
        rows.append(v)
    return n_total, (np.array(rows, np.float32) if rows else np.zeros((0, 6), np.float32)), n_bad


def read_gt_txt(path: Path):
    """读 GT TXT（cls cx cy w h，归一化）。"""
    if not path.exists():
        return np.zeros((0, 5), np.float32)
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.strip().split()
        if len(p) < 5:
            continue
        try:
            v = [float(x) for x in p[:5]]
        except ValueError:
            continue
        if np.isfinite(v).all():
            rows.append(v)
    return np.array(rows, np.float32) if rows else np.zeros((0, 5), np.float32)


def apply_max_boxes(pred: np.ndarray, max_boxes: int = MAX_BOXES_PER_IMAGE):
    """§九(二)：按 confidence 降序截断到 max_boxes。

    排序：conf 降序；tie-break 用 (-conf, cls) 保证**确定性**（np.argsort 非稳定）。
    """
    if len(pred) == 0:
        return pred
    order = np.lexsort((pred[:, 0], -pred[:, 5]))     # 主键 -conf，次键 cls
    return pred[order][:max_boxes]


def norm_xywh_to_xyxy(arr: np.ndarray, w: int, h: int):
    cls = arr[:, 0].astype(int)
    cx, cy, bw, bh = arr[:, 1] * w, arr[:, 2] * h, arr[:, 3] * w, arr[:, 4] * h
    x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    return np.stack([x1, y1, x2, y2], 1).astype(np.float32), cls


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ab = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / (aa[:, None] + ab[None, :] - inter + EPS)


def interpolate_ap(recall: np.ndarray, precision: np.ndarray,
                   n_points: int = N_INTERP, tail: str = "zero") -> float:
    """§六.3 官方 AP：101 点，p_interp(r) = max{precision : recall >= r}，**算术平均**。

    tail='zero'（官方）：无 recall >= r 的点 → p_interp = 0
    tail='ramp'（fork 口径，仅作对照）
    """
    if recall.size == 0:
        return 0.0
    x = np.linspace(0.0, 1.0, n_points)
    if tail == "ramp":
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        return float(np.interp(x, mrec, mpre).mean())
    q = np.zeros_like(x)
    for i, t in enumerate(x):
        m = precision[recall >= t]
        if m.size:
            q[i] = m.max()
    return float(q.mean())


def pr_curve_for_class(preds_c, gt_by_img, iou_thr: float):
    """单类单阈值的 PR 曲线。preds_c: [(img_id, conf, box_xyxy)]，gt_by_img: {img_id: boxes}。

    匹配 = §六.2：**confidence 降序**，贪心 one-to-one，与未匹配 GT 的 IoU >= t 记 TP。
    """
    if not preds_c:
        return np.zeros(0), np.zeros(0)
    preds = sorted(preds_c, key=lambda z: -z[1])          # confidence 降序
    npos = int(sum(len(v) for v in gt_by_img.values()))
    if npos == 0:
        return np.zeros(0), np.zeros(0)
    used = {i: np.zeros(len(v), bool) for i, v in gt_by_img.items()}
    tp = np.zeros(len(preds)); fp = np.zeros(len(preds))
    for k, (img_id, _conf, box) in enumerate(preds):
        gt = gt_by_img.get(img_id)
        if gt is None or len(gt) == 0:
            fp[k] = 1
            continue
        ious = box_iou_np(box[None, :], gt)[0]
        ious[used[img_id]] = -1                            # 已匹配的 GT 不可再用
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr:
            tp[k] = 1
            used[img_id][j] = True
        else:
            fp[k] = 1
    ctp = np.cumsum(tp); cfp = np.cumsum(fp)
    # recall = TP(k)/N，**不做 +EPS**：加了 EPS 会让 recall[-1] = 1-1e-7 < 1.0，
    # 从而在 r=1.00 的插值点取不到任何 precision，每个类白白丢 1/101 的 AP。
    # （COCO 的写法是 np.maximum(npos, eps)，npos>>eps 时同样等于精确除。）
    return ctp / npos, ctp / np.maximum(ctp + cfp, EPS)


# ============================================================
# 数据装载
# ============================================================
def load_split(images_dir: Path, labels_dir: Path, drop_corrupt: bool = True):
    """返回 (per_image, stats)。

    per_image: [(img_id, stem, w, h, gt_boxes, gt_cls)] —— **img_id 用图像在完整文件列表中的
    原始序号**（不是压缩后的序号）。跳过 corrupt 图时不会发生索引位移，
    否则预测会与图像错配（本模块 v1 曾因此把 mAP 从 0.51 算成 0.15）。
    `stem` 一并携带，调用方不再需要维护旁路列表。
    """
    per_image = []
    stats = dict(images=0, corrupt=0, gt=0)
    for img_id, ip in enumerate(sorted(p for p in Path(images_dir).iterdir()
                                       if p.suffix.lower() in IMG_EXTS)):
        stats["images"] += 1
        w, h = _imsize(ip)
        if w <= 1 or h <= 1:
            stats["corrupt"] += 1
            continue
        gt = read_gt_txt(Path(labels_dir) / f"{ip.stem}.txt")
        if drop_corrupt and len(gt) and (gt[:, 1:].max() > 1.0 or gt[:, 1:].min() < 0.0):
            stats["corrupt"] += 1          # 与 verify_image_label 一致：越界标签整图作废
            continue
        b = c = None
        if len(gt):
            b, c = norm_xywh_to_xyxy(gt, w, h)
            stats["gt"] += len(b)
        per_image.append((img_id, ip.stem, w, h, b, c))
    return per_image, stats


# ============================================================
# 主评测
# ============================================================
def evaluate(per_image, results_dir, metric: str = METRIC_A,
             max_boxes: int = MAX_BOXES_PER_IMAGE, top_k: int | None = None,
             min_conf: float = 0.0, iou_thresholds=IOU_THRESHOLDS):
    """统一评测入口。

    per_image : load_split 的输出（img_id, stem, w, h, gt_boxes, gt_cls）
    返回 dict：mAP50-95 / mAP50 / per_iou / per_class / counts
    """
    results_dir = Path(results_dir)
    preds_by_cls = {c: [] for c in range(NC)}
    gt_by_cls = {c: {} for c in range(NC)}
    st = dict(images=len(per_image), gt=0, pred_total=0, pred_invalid=0,
              pred_used=0, imgs_truncated=0, empty=0, max_boxes_seen=0)

    for (img_id, stem, w, h, gb, gc) in per_image:
        if gb is not None and len(gb):
            for c in np.unique(gc):
                m = gc == c
                gt_by_cls[int(c)][img_id] = gb[m]
            st["gt"] += len(gb)

        n_total, pred, n_bad = read_pred_txt(results_dir / f"{stem}.txt")
        st["pred_total"] += n_total
        st["pred_invalid"] += n_bad
        if len(pred) == 0:
            st["empty"] += 1
            continue
        st["max_boxes_seen"] = max(st["max_boxes_seen"], len(pred))
        if len(pred) > max_boxes:
            st["imgs_truncated"] += 1
        pred = apply_max_boxes(pred, max_boxes)                      # 官方硬上限
        if top_k is not None:
            pred = pred[:min(top_k, max_boxes)]
        if min_conf > 0:
            pred = pred[pred[:, 5] >= min_conf]
        if len(pred) == 0:
            continue
        boxes, cls = norm_xywh_to_xyxy(pred, w, h)
        conf = pred[:, 5]
        for i in range(len(boxes)):
            preds_by_cls[int(cls[i])].append((img_id, float(conf[i]), boxes[i]))
        st["pred_used"] += len(boxes)

    # ---- 逐 IoU：per-class AP ----
    per_iou = []
    per_class_ap = np.full((NC, len(iou_thresholds)), np.nan, np.float32)
    for ti, t in enumerate(iou_thresholds):
        aps_A = []
        for c in range(NC):
            rec, prec = pr_curve_for_class(preds_by_cls[c], gt_by_cls[c], float(t))
            ap = interpolate_ap(rec, prec)
            has_gt = len(gt_by_cls[c]) > 0
            if has_gt:
                per_class_ap[c, ti] = ap
            aps_A.append(ap if has_gt else (0.0 if metric == METRIC_A else None))
        if metric == METRIC_A:
            per_iou.append(float(np.mean(aps_A)))                    # 固定 12 类
        else:
            vals = [a for a in aps_A if a is not None]
            per_iou.append(float(np.mean(vals)) if vals else 0.0)    # 仅有 GT 的类
    per_iou = np.array(per_iou)
    out = {
        "metric": metric, "max_boxes": max_boxes, "top_k": top_k,
        "mAP50": float(per_iou[0]), "mAP75": float(per_iou[5]),
        "mAP50-95": float(per_iou.mean()),
        "per_iou": per_iou.tolist(),
        "iou_thresholds": [float(t) for t in iou_thresholds],
        "per_class_ap50_95": np.nanmean(per_class_ap, 1).tolist(),
        "counts": st,
    }
    return out


def make_stems(images_dir: Path):
    return [p.stem for p in sorted(p for p in Path(images_dir).iterdir()
                                   if p.suffix.lower() in IMG_EXTS)]


def _cli():
    ap = argparse.ArgumentParser(description="冻结官方口径评测器")
    ap.add_argument("--results", required=True)
    ap.add_argument("--images", default="data/processed/rgbid_split/images/val/visible")
    ap.add_argument("--labels", default="data/processed/rgbid_split/labels/val/visible")
    ap.add_argument("--metric", choices=[METRIC_A, METRIC_B], default=METRIC_A)
    ap.add_argument("--max-boxes-per-image", type=int, default=MAX_BOXES_PER_IMAGE)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    per_image, _ = load_split(Path(a.images), Path(a.labels))
    r = evaluate(per_image, Path(a.results), metric=a.metric,
                 max_boxes=a.max_boxes_per_image, top_k=a.top_k,
                 min_conf=a.min_conf)
    print(json.dumps(r, indent=2, ensure_ascii=False) if a.json else
          "metric=%s  mAP50-95=%.5f  mAP50=%.5f  mAP75=%.5f" %
          (r["metric"], r["mAP50-95"], r["mAP50"], r["mAP75"]))
    if a.json:
        Path(a.json).write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    _cli()
