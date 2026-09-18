"""scripts/modality_dropout_official_eval.py — 官方口径扩展评测（只读，不训练、不提交）。

在 `scripts/official_map.py` 的**完全相同的官方参数**下（avg=mean / tail=zero / match=conf），
额外输出 `official_map.py` 本身不打印的项：

  * 官方 mAP50 / mAP75 / mAP50-95
  * 官方口径下 micro-average 的 Precision / Recall（IoU=0.5 的 F1 最大化工作点）
  * 逐类 AP50 / AP50-95（12 类，用于 person / sign / bicycle / ball 等重点类分析）
  * 按目标尺寸的 recall（COCO 定义：small <32²、medium 32²–96²、large >96²）
      - R@0.5            ：IoU=0.5 官方 conf 顺序贪心匹配
      - R@0.5:0.95       ：10 个 IoU 阈值上的均值（COCO AR 风格）

**参数一致性**：本脚本不重新实现任何官方语义，`collect` / `curve_for_class` /
`ap_from_curve` / `NC` / `IOUV_OFFICIAL` 全部直接 import 自 `official_map.py`，
只在官方匹配流程之外**额外记录**「哪些 GT 被匹配上」，用于尺寸分桶。

⚠️ 仅用于离线分析，**不得用于正式比赛提交**。

用法:
  python scripts/modality_dropout_official_eval.py --results <dir>/results \\
      --split_root data/processed/rgbid_split --tag "New-Full" --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _p in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import official_map as OM  # noqa: E402

# COCO 面积阈值（原生像素）
SMALL_MAX, MEDIUM_MAX = 32 ** 2, 96 ** 2


def official_match(preds_c, gt_by_img, iou_thr):
    """复刻 official_map.curve_for_class 的官方匹配，并额外返回被匹配的 GT 索引。

    官方语义（照抄赛题文字）：按 confidence 从高到低遍历预测框，若能匹配到
    「尚未被匹配」的真实框且 IoU >= t 则 TP，否则 FP（取 IoU 最大者）。

    Returns:
        (preds_sorted, tp, matched_gt)
        preds_sorted: list[(img_id, conf, box)]
        tp: np.ndarray 与 preds_sorted 对齐的 0/1
        matched_gt: dict {img_id: set(gt_index)}  —— 被成功匹配的 GT
    """
    order = sorted(range(len(preds_c)), key=lambda i: -preds_c[i][1])
    preds_sorted = [preds_c[i] for i in order]
    matched = {img: np.zeros(len(v), dtype=bool) for img, v in gt_by_img.items()}
    tp = np.zeros(len(preds_sorted))

    for i, (img, _c, box) in enumerate(preds_sorted):
        gts = gt_by_img.get(img)
        if gts is None or len(gts) == 0:
            continue
        ious = OM.box_iou_np(box[None, :], gts)[0]
        cand = np.where(~matched[img])[0]
        if cand.size and ious[cand].max() >= iou_thr:
            j = cand[int(np.argmax(ious[cand]))]
            tp[i] = 1.0
            matched[img][j] = True

    matched_gt = {img: set(np.where(m)[0].tolist()) for img, m in matched.items()}
    return preds_sorted, tp, matched_gt


def size_bucket(boxes: np.ndarray) -> np.ndarray:
    """按 COCO 面积返回 0=small / 1=medium / 2=large。"""
    area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    b = np.full(len(area), 2, dtype=int)
    b[area < SMALL_MAX] = 0
    b[(area >= SMALL_MAX) & (area < MEDIUM_MAX)] = 1
    return b


def pooled_gt_by_img(gt_by_cls):
    """把逐类 GT 合并成 {img_id: (M,4)}（GT 全集，用于尺寸分桶的分母）。"""
    out: dict = {}
    for c in range(OM.NC):
        for img, boxes in gt_by_cls[c].items():
            out.setdefault(img, []).append(boxes)
    return {img: np.concatenate(v) for img, v in out.items() if v}


def evaluate_extended(gt_by_cls, preds_by_cls):
    """官方参数下的完整指标。"""
    per_iou, per_class_ap = [], {c: [] for c in range(OM.NC)}

    for t in OM.IOUV_OFFICIAL:
        aps = []
        for c in range(OM.NC):
            rec, prec = OM.curve_for_class(preds_by_cls[c], gt_by_cls[c], float(t), match="conf")
            ap = OM.ap_from_curve(rec, prec, avg="mean", tail="zero")
            if len(gt_by_cls[c]) == 0:
                per_class_ap[c].append(None)      # 官方：无 GT 的类别忽略
                continue
            aps.append(ap)
            per_class_ap[c].append(ap)
        per_iou.append(float(np.mean(aps)) if aps else 0.0)
    per_iou = np.array(per_iou)

    # --- 尺寸分桶 recall：逐类官方匹配 → 按 GT 面积归桶 ---
    gt_all = pooled_gt_by_img(gt_by_cls)
    bucket_of = {img: size_bucket(b) for img, b in gt_all.items()}
    n_gt = {k: 0 for k in ("small", "medium", "large")}
    names = ("small", "medium", "large")
    for img, b in bucket_of.items():
        for k in np.unique(b):
            n_gt[names[int(k)]] += int((b == k).sum())

    hit = {t: {k: 0 for k in names} for t in ("r50", "r5095")}
    for t_i, t in enumerate(OM.IOUV_OFFICIAL):
        for c in range(OM.NC):
            if len(gt_by_cls[c]) == 0:
                continue
            _ps, _tp, matched_gt = official_match(preds_by_cls[c], gt_by_cls[c], float(t))
            for img, idxs in matched_gt.items():
                b = bucket_of.get(img)
                if b is None:
                    continue
                for j in idxs:
                    hit["r5095"][names[int(b[j])]] += 1
                    if t_i == 0:
                        hit["r50"][names[int(b[j])]] += 1

    recall_size = {
        "R@0.5": {k: (hit["r50"][k] / n_gt[k] if n_gt[k] else 0.0) for k in names},
        "R@0.5:0.95": {k: (hit["r5095"][k] / (n_gt[k] * len(OM.IOUV_OFFICIAL)) if n_gt[k] else 0.0)
                       for k in names},
        "n_gt": n_gt,
    }

    # --- micro-average P/R（IoU=0.5，官方匹配；取 F1 最大化工作点）---
    confs, tps, fps = [], [], []
    for c in range(OM.NC):
        if len(gt_by_cls[c]) == 0:
            continue
        ps, tp, _m = official_match(preds_by_cls[c], gt_by_cls[c], 0.5)
        for i, (_img, cf, _b) in enumerate(ps):
            confs.append(cf)
            tps.append(tp[i])
            fps.append(1.0 - tp[i])
    if confs:
        o = np.argsort(-np.array(confs), kind="stable")
        tps = np.array(tps)[o]
        fps = np.array(fps)[o]
        npos = int(sum(len(v) for v in gt_all.values()))
        tp_c, fp_c = np.cumsum(tps), np.cumsum(fps)
        rec = tp_c / max(npos, 1)
        pre = tp_c / np.maximum(tp_c + fp_c, OM.EPS)
        f1 = 2 * pre * rec / np.maximum(pre + rec, OM.EPS)
        k = int(np.argmax(f1))
        pr = dict(P=float(pre[k]), R=float(rec[k]), F1=float(f1[k]), conf=float(confs[o[k]]))
    else:
        pr = dict(P=0.0, R=0.0, F1=0.0, conf=0.0)

    return {
        "mAP50": float(per_iou[0]),
        "mAP75": float(per_iou[5]),
        "mAP50-95": float(per_iou.mean()),
        "per_iou": per_iou.tolist(),
        "per_class_AP50": {c: per_class_ap[c][0] for c in range(OM.NC)},
        "per_class_AP50-95": {c: (float(np.mean([v for v in per_class_ap[c] if v is not None]))
                                  if any(v is not None for v in per_class_ap[c]) else None)
                              for c in range(OM.NC)},
        "recall_size": recall_size,
        "PR_micro_iou50": pr,
    }


def main():
    ap = argparse.ArgumentParser(description="官方口径扩展评测（逐类 AP + 尺寸 recall）")
    ap.add_argument("--results", required=True)
    ap.add_argument("--split_root", default="data/processed/rgbid_split")
    ap.add_argument("--tag", default="")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    split_root = (PROJECT_ROOT / args.split_root).resolve()
    images_dir = split_root / "images" / "val" / "visible"
    labels_dir = split_root / "labels" / "val" / "visible"
    results_dir = (PROJECT_ROOT / args.results).resolve()
    for p, nm in ((images_dir, "val visible"), (labels_dir, "val labels"), (results_dir, "results")):
        if not p.is_dir():
            raise FileNotFoundError(f"{nm} 目录不存在: {p}")

    dcfg = yaml.safe_load(open(split_root / "dataset.yaml", encoding="utf-8"))
    names = dcfg.get("names", {})

    gt, preds, st = OM.collect(images_dir, labels_dir, results_dir, 0.0, drop_corrupt=True)
    res = evaluate_extended(gt, preds)
    res["_stats"] = st
    res["_tag"] = args.tag
    res["_results"] = str(results_dir)

    print("=" * 74)
    print(f"官方口径扩展评测   tag={args.tag or '-'}")
    print(f"results={results_dir}")
    print(f"images={st['images']}  corrupt剔除={st['corrupt']}  GT={st['gt']}  pred={st['pred']}")
    print("=" * 74)
    print(f"  official mAP50      = {res['mAP50']:.5f}")
    print(f"  official mAP75      = {res['mAP75']:.5f}")
    print(f"  official mAP50-95   = {res['mAP50-95']:.5f}")
    pr = res["PR_micro_iou50"]
    print(f"  micro P/R @IoU0.5   = {pr['P']:.5f} / {pr['R']:.5f}   (F1={pr['F1']:.5f} @conf={pr['conf']:.4f})")
    rs = res["recall_size"]
    print(f"  n_gt  small/medium/large = {rs['n_gt']['small']} / {rs['n_gt']['medium']} / {rs['n_gt']['large']}")
    print(f"  R@0.5      S/M/L    = {rs['R@0.5']['small']:.5f} / {rs['R@0.5']['medium']:.5f} / {rs['R@0.5']['large']:.5f}")
    print(f"  R@0.5:0.95 S/M/L    = {rs['R@0.5:0.95']['small']:.5f} / {rs['R@0.5:0.95']['medium']:.5f} / {rs['R@0.5:0.95']['large']:.5f}")
    print("-" * 74)
    print(f"  {'id':<3}{'class':<14}{'AP50':>9}{'AP50-95':>11}")
    for c in range(OM.NC):
        a50 = res["per_class_AP50"][c]
        a = res["per_class_AP50-95"][c]
        a50s = f"{a50:.5f}" if a50 is not None else "  (无GT)"
        as_ = f"{a:.5f}" if a is not None else "  (无GT)"
        print(f"  {c:<3}{str(names.get(c, c)):<14}{a50s:>9}{as_:>11}")
    print("=" * 74)

    if args.json:
        out = PROJECT_ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
