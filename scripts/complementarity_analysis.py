"""scripts/complementarity_analysis.py — RGBID 与 F4 两模型的互补性审计（§9）。

不改模型、不改评价代码。只读 GT 标签 + 各模型的预测 TXT，在「原图坐标 + 逐类 IoU>=0.5 贪心匹配」
的统一口径下统计:

  GT 对象层面: RGBID-only / F4-only / both / neither（漏检）
  pred 层面  : 各模型 TP/FP、重复检测（多 pred 命中同一 GT）
  ensemble 层面: 相对 RGBID 新增 TP / 新增 FP / 消除漏检

用法:
  python scripts/complementarity_analysis.py \
      --split_root data/processed/rgbid_split \
      --rgbid diagnostic/rgbid_inference_gain/RGBID_baseline/results \
      --f4    diagnostic/rgbid_inference_gain/F4_baseline/results \
      --ensemble diagnostic/rgbid_inference_gain/RGBID_f4_ens_2to1/results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NC = 12
IOU_THR = 0.5
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_txt(path, with_conf):
    n = 6 if with_conf else 5
    if not path.exists():
        return np.zeros((0, n), dtype=np.float32)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        p = s.split()
        if len(p) < n:
            continue
        rows.append([float(x) for x in p[:n]])
    if not rows:
        return np.zeros((0, n), dtype=np.float32)
    return np.array(rows, dtype=np.float32).reshape(-1, n)


def _to_xyxy(arr, w, h):
    """(n,5)[cls,cx,cy,w,h] 归一化 -> xyxy + cls（原生像素）。"""
    if len(arr) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    cx, cy, bw, bh = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return np.stack([x1, y1, x2, y2], axis=1), arr[:, 0]


def _iou_matrix(a, b):
    """a:(N,4) b:(M,4) xyxy -> (N,M) IoU。"""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ab = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = aa[:, None] + ab[None, :] - inter
    return inter / np.maximum(union, np.finfo(np.float32).eps)


def _match(pred_boxes, pred_cls, pred_conf, gt_boxes, gt_cls):
    """逐类 conf 降序贪心匹配（IoU>=IOU_THR），返回 (matched_gt_idx_per_pred, pred_is_tp)。

    matched_gt_idx_per_pred[i] = 命中 GT 索引或 -1（FP）；pred_is_tp[i] = bool。
    """
    n = len(pred_boxes)
    matched = np.full(n, -1, dtype=int)
    is_tp = np.zeros(n, dtype=bool)
    if n == 0 or len(gt_boxes) == 0:
        return matched, is_tp
    used = np.zeros(len(gt_boxes), dtype=bool)
    order = np.argsort(-pred_conf)
    for i in order:
        c = pred_cls[i]
        cand = np.where((gt_cls == c) & ~used)[0]
        if cand.size == 0:
            continue
        ious = _iou_matrix(pred_boxes[i:i + 1], gt_boxes[cand])[0]
        j = cand[int(np.argmax(ious))]
        if ious[np.argmax(ious)] >= IOU_THR:
            matched[i] = j
            is_tp[i] = True
            used[j] = True
    return matched, is_tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_root", default="data/processed/rgbid_split")
    ap.add_argument("--rgbid", required=True)
    ap.add_argument("--f4", required=True)
    ap.add_argument("--ensemble", default=None)
    args = ap.parse_args()

    split_root = (PROJECT_ROOT / args.split_root).resolve()
    images_dir = split_root / "images" / "val" / "visible"
    labels_dir = split_root / "labels" / "val" / "visible"
    rgbid_dir = (PROJECT_ROOT / args.rgbid).resolve()
    f4_dir = (PROJECT_ROOT / args.f4).resolve()
    ens_dir = (PROJECT_ROOT / args.ensemble).resolve() if args.ensemble else None

    # 统计
    gt_total = 0
    gt_both = 0          # RGBID 与 F4 都命中
    gt_rgbid_only = 0    # 仅 RGBID 命中
    gt_f4_only = 0       # 仅 F4 命中
    gt_neither = 0       # 两者都漏
    pred = {"rgbid": {"tp": 0, "fp": 0, "dup": 0, "n": 0},
            "f4": {"tp": 0, "fp": 0, "dup": 0, "n": 0}}
    ens_added_tp = 0     # ensemble 相对 RGBID 新增 TP（GT 层面）
    ens_added_fp = 0     # ensemble 相对 RGBID 新增 FP（pred 层面）
    ens_lost_rgbid_tp = 0  # ensemble 相对 RGBID 丢失的 TP（GT 层面）
    pc_gt = np.zeros(NC, dtype=int)
    pc_r = np.zeros(NC, dtype=int)
    pc_f = np.zeros(NC, dtype=int)
    pc_e = np.zeros(NC, dtype=int)
    if ens_dir is not None:
        pred["ensemble"] = {"tp": 0, "fp": 0, "dup": 0, "n": 0}

    img_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    for img_path in img_files:
        stem = img_path.stem
        import cv2
        h, w = cv2.imread(str(img_path)).shape[:2]

        gt = _read_txt(labels_dir / f"{stem}.txt", with_conf=False)
        if len(gt) and (gt[:, 1:].max() > 1.0 or gt[:, 1:].min() < 0.0):
            continue  # corrupt 跳过（与 val 一致）
        gt_boxes, gt_cls = _to_xyxy(gt, w, h)
        ngt = len(gt_boxes)
        gt_total += ngt

        # 读预测并匹配
        def _load(dir_):
            p = _read_txt(dir_ / f"{stem}.txt", with_conf=True)
            if len(p) == 0:
                return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                        np.zeros((0,), np.float32))
            boxes, cls = _to_xyxy(p[:, :5], w, h)
            conf = p[:, 5]
            return boxes, cls, conf

        r_boxes, r_cls, r_conf = _load(rgbid_dir)
        f_boxes, f_cls, f_conf = _load(f4_dir)

        r_matched, r_tp = _match(r_boxes, r_cls, r_conf, gt_boxes, gt_cls)
        f_matched, f_tp = _match(f_boxes, f_cls, f_conf, gt_boxes, gt_cls)

        # GT 层面归属
        gt_hit_r = np.zeros(ngt, dtype=bool)
        gt_hit_f = np.zeros(ngt, dtype=bool)
        gt_hit_r[r_matched[r_tp]] = True
        gt_hit_f[f_matched[f_tp]] = True
        gt_both += int((gt_hit_r & gt_hit_f).sum())
        gt_rgbid_only += int((gt_hit_r & ~gt_hit_f).sum())
        gt_f4_only += int((~gt_hit_r & gt_hit_f).sum())
        gt_neither += int((~gt_hit_r & ~gt_hit_f).sum())

        # pred 层面统计（含重复检测：同一 GT 被多个 pred 命中）
        def _pred_stats(matched, is_tp):
            tp = int(is_tp.sum())
            fp = int((~is_tp).sum())
            dup = 0
            if tp > 0:
                hit_gt = matched[is_tp]
                _, counts = np.unique(hit_gt, return_counts=True)
                dup = int((counts - 1).sum())
            return tp, fp, dup, len(matched)

        for k, (m, t) in (("rgbid", (r_matched, r_tp)), ("f4", (f_matched, f_tp))):
            tp, fp, dup, n = _pred_stats(m, t)
            pred[k]["tp"] += tp
            pred[k]["fp"] += fp
            pred[k]["dup"] += dup
            pred[k]["n"] += n

        # ensemble 层面
        if ens_dir is not None:
            e_boxes, e_cls, e_conf = _load(ens_dir)
            e_matched, e_tp = _match(e_boxes, e_cls, e_conf, gt_boxes, gt_cls)
            tp, fp, dup, n = _pred_stats(e_matched, e_tp)
            pred["ensemble"]["tp"] += tp
            pred["ensemble"]["fp"] += fp
            pred["ensemble"]["dup"] += dup
            pred["ensemble"]["n"] += n
            # 新增 TP：ensemble 命中但 RGBID 漏掉的 GT
            gt_hit_e = np.zeros(ngt, dtype=bool)
            gt_hit_e[e_matched[e_tp]] = True
            ens_added_tp += int((gt_hit_e & ~gt_hit_r).sum())
            # 删除的 RGBID TP：RGBID 原先命中、但 ensemble 不再覆盖的 GT
            ens_lost_rgbid_tp += int((gt_hit_r & ~gt_hit_e).sum())
            # 各类别 GT 命中数（用于逐类 TP 变化）
            for c in np.unique(gt_cls) if ngt else []:
                cm = gt_cls == c
                c = int(c)
                pc_gt[c] += int(cm.sum())
                pc_r[c] += int((gt_hit_r & cm).sum())
                pc_f[c] += int((gt_hit_f & cm).sum())
                pc_e[c] += int((gt_hit_e & cm).sum())

    # 修正新增 FP：直接比较 ensemble FP 与 RGBID FP 总量差
    if ens_dir is not None:
        ens_added_fp = pred["ensemble"]["fp"] - pred["rgbid"]["fp"]

    print("=" * 60)
    print("互补性审计（IoU>=0.5 逐类贪心匹配，原图坐标）")
    print("=" * 60)
    print(f"GT 对象总数              : {gt_total}")
    print(f"  两模型都命中  (both)   : {gt_both}")
    print(f"  仅 RGBID 命中           : {gt_rgbid_only}")
    print(f"  仅 F4 命中              : {gt_f4_only}")
    print(f"  两模型都漏    (neither) : {gt_neither}")
    print("-" * 60)
    for k in ("rgbid", "f4"):
        s = pred[k]
        print(f"[{k:>8}] pred={s['n']}  TP={s['tp']}  FP={s['fp']}  重复检测={s['dup']}")
    if ens_dir is not None:
        s = pred["ensemble"]
        print(f"[ensemble] pred={s['n']}  TP={s['tp']}  FP={s['fp']}  重复检测={s['dup']}")
        print("-" * 60)
        print(f"ensemble 相对 RGBID 新增 TP : {ens_added_tp}")
        print(f"ensemble 相对 RGBID 新增 FP : {ens_added_fp}")
        print(f"ensemble 删除的 RGBID TP    : {ens_lost_rgbid_tp}")
        print(f"  (净 TP 变化 = {ens_added_tp} - {ens_lost_rgbid_tp} = "
              f"{ens_added_tp - ens_lost_rgbid_tp:+d})")
        print("-" * 60)
        names = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
                 6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}
        print("逐类 GT 命中数（IoU>=0.5）:")
        print(f"  {'class':<14}{'GT':>6}{'RGBID':>8}{'F4':>8}{'Ensemble':>10}")
        for c in range(NC):
            if pc_gt[c] == 0:
                continue
            print(f"  {names[c]:<14}{pc_gt[c]:>6}{pc_r[c]:>8}{pc_f[c]:>8}{pc_e[c]:>10}")
    print("=" * 60)


if __name__ == "__main__":
    main()
