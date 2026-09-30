"""_e1_vs_dprime_diag.py — D′ vs E1 官方口径逐项对比（只读，不重新推理）。

完全相同：官方 evaluator（scripts/official_eval.py 的函数**直接 import**）、
同一批 400 张 val prediction TXT、cap=100、同一 val split。
新增推理 = 0；改任何文件 = 0；训练 = 0。

四类分解：类别 / 尺度 / IoU 区间 / prediction distribution。
判据 metric 一律用 **官方口径**（cap=100）。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from official_eval import (  # noqa: E402
    IOU_THRESHOLDS, MAX_BOXES_PER_IMAGE, NC, apply_max_boxes, evaluate,
    interpolate_ap, load_split, norm_xywh_to_xyxy, pr_curve_for_class, read_pred_txt,
)

IMAGES = ROOT / "data/processed/rgbid_split_train/images/val/visible"
LABELS = ROOT / "data/processed/rgbid_split_train/labels/val/visible"
MODELS = {
    "Dp": ROOT / "diagnostic/sepstem_clahe/best_full/results",
    "E1": ROOT / "diagnostic/e1_region_gain/official_eval/results",
}
NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}
SMALL_A, MEDIUM_A = 1024.0, 9216.0


def build(per_image, results_dir):
    """复刻 official_eval.evaluate 的装载 + cap=100（**不调用 evaluate 之外的自定义口径**）。"""
    preds_by_cls = {c: [] for c in range(NC)}
    gt_by_cls = {c: {} for c in range(NC)}
    pred_rows = []                     # 供 prediction distribution 用
    for img_id, stem, w, h, gtb, gtc in per_image:
        if gtb is not None and len(gtb):
            for i in range(len(gtb)):
                gt_by_cls[int(gtc[i])].setdefault(img_id, []).append(gtb[i])
        _, pr, _ = read_pred_txt(Path(results_dir) / f"{stem}.txt")
        pr = apply_max_boxes(pr, MAX_BOXES_PER_IMAGE)
        if len(pr) == 0:
            continue
        boxes, cls = norm_xywh_to_xyxy(pr, w, h)
        conf = pr[:, 5]
        for i in range(len(boxes)):
            preds_by_cls[int(cls[i])].append((img_id, float(conf[i]), boxes[i]))
            pred_rows.append((img_id, int(cls[i]), float(conf[i]),
                              float(boxes[i][2] - boxes[i][0]),
                              float(boxes[i][3] - boxes[i][1])))
    return preds_by_cls, gt_by_cls, pred_rows


def scale_ap(preds_by_cls, gt_by_cls, lo, hi, iou_thresholds=IOU_THRESHOLDS):
    """按 native 面积桶过滤 GT，其余完全沿用 official_eval 的匹配与 AP。"""
    per_iou = []
    for t in iou_thresholds:
        aps = []
        for c in range(NC):
            g = {}
            for iid, boxes in gt_by_cls[c].items():
                keep = [b for b in boxes
                        if lo <= float((b[2] - b[0]) * (b[3] - b[1])) < hi]
                if keep:
                    g[iid] = np.stack(keep)   # official_eval 的 box_iou_np 要求 ndarray，不是 list
            rec, prec = pr_curve_for_class(preds_by_cls[c], g, float(t))
            aps.append(interpolate_ap(rec, prec) if len(g) else 0.0)
        per_iou.append(float(np.mean(aps)))          # METRIC_A：固定 12 类
    return float(np.mean(per_iou)), np.array(per_iou)


def main():
    per_image, stats = load_split(IMAGES, LABELS)
    print("=" * 104)
    print(f"D′ vs E1 —— 同一官方 evaluator / 同一 400 TXT / cap=100 / 不重新推理")
    print(f"  images={stats['images']}  corrupt={stats['corrupt']}  GT={stats['gt']}")
    print("=" * 104)

    R = {}
    for k, d in MODELS.items():
        pbc, gbc, prows = build(per_image, d)
        ev = evaluate(per_image, d, max_boxes=MAX_BOXES_PER_IMAGE)
        R[k] = dict(ev=ev, pbc=pbc, gbc=gbc, prows=prows)
        print(f"  [{k}] mAP50-95={ev['mAP50-95']:.5f}  mAP50={ev['mAP50']:.5f}  "
              f"preds={len(prows)}")
    D, E = R["Dp"], R["E1"]
    d0 = E["ev"]["mAP50-95"] - D["ev"]["mAP50-95"]
    print(f"\n  Δ(E1 − D′) 官方 mAP50-95 = {d0:+.5f}")

    # ---------- ① 类别 ----------
    print("\n" + "=" * 104)
    print("① 类别分解（官方口径 AP50-95，固定 12 类平均）")
    print("=" * 104)
    print(f"  {'class':<14}{'GT n':>6}{'Dp AP':>10}{'E1 AP':>10}{'Δ':>10}")
    pc_d = D["ev"]["per_class_ap50_95"]; pc_e = E["ev"]["per_class_ap50_95"]
    gt_n = {c: sum(len(v) for v in D["gbc"][c].values()) for c in range(NC)}
    rows = []
    for c in range(NC):
        if gt_n[c] == 0:
            continue
        rows.append((pc_e[c] - pc_d[c], c, gt_n[c], pc_d[c], pc_e[c]))
    for dd, c, n, a, b in sorted(rows):
        print(f"  {NAMES[c]:<14}{n:>6}{a:>10.4f}{b:>10.4f}{dd:>+10.4f}")
    neg = sum(min(x[0], 0) for x in rows); pos = sum(max(x[0], 0) for x in rows)
    tot = sum(x[0] for x in rows)
    print(f"\n  ΣΔ = {tot/NC:+.5f}（= mAP50-95 的 Δ）  负向合计 {neg/NC:+.5f} / 正向合计 {pos/NC:+.5f}")
    worst = sorted(rows)[:2]
    print(f"  最负两类 (+GT数): " + ", ".join(f"{NAMES[c]}(Δ{dd:+.4f}, n={n})" for dd, c, n, _, _ in worst))
    share = sum(dd for dd, _, _, _, _ in worst) / tot if tot else float("nan")
    print(f"  最负两类占 ΣΔ 的比例 = {share:.0%}  （越接近 100% 越说明集中）")

    # ---------- ② 尺度 ----------
    print("\n" + "=" * 104)
    print("② 尺度分解（native 面积桶；GT 过滤，预测不过滤；沿用 official_eval 的匹配/AP 函数）")
    print("=" * 104)
    print(f"  {'bucket':<10}{'Dp':>10}{'E1':>10}{'Δ':>10}{'GT n':>8}")
    for lab, lo, hi in (("small", 0, SMALL_A), ("medium", SMALL_A, MEDIUM_A),
                        ("large", MEDIUM_A, 1e18)):
        a, _ = scale_ap(D["pbc"], D["gbc"], lo, hi)
        b, _ = scale_ap(E["pbc"], E["gbc"], lo, hi)
        n = sum(1 for c in range(NC) for v in D["gbc"][c].values() for x in v
                if lo <= float((x[2]-x[0])*(x[3]-x[1])) < hi)
        print(f"  {lab:<10}{a:>10.4f}{b:>10.4f}{b-a:>+10.4f}{n:>8}")

    # ---------- ③ IoU 区间 ----------
    print("\n" + "=" * 104)
    print("③ IoU 区间分解（官方 11 个阈值）")
    print("=" * 104)
    print(f"  {'IoU':<8}{'Dp':>10}{'E1':>10}{'Δ':>10}")
    for t, a, b in zip(IOU_THRESHOLDS, D["ev"]["per_iou"], E["ev"]["per_iou"]):
        print(f"  {t:<8.2f}{a:>10.4f}{b:>10.4f}{b-a:>+10.4f}")
    pi_d = np.array(D["ev"]["per_iou"]); pi_e = np.array(E["ev"]["per_iou"])
    dlo = (pi_e[:3] - pi_d[:3]).mean(); dmid = (pi_e[3:6] - pi_d[3:6]).mean()
    dhi = (pi_e[6:] - pi_d[6:]).mean()
    print(f"\n  分段 Δ：低 IoU(.50–.60) {dlo:+.5f} | 中(.65–.75) {dmid:+.5f} | 高(.80–.95) {dhi:+.5f}")

    # ---------- ④ prediction distribution ----------
    print("\n" + "=" * 104)
    print("④ prediction distribution（同一 cap=100 后的落盘预测）")
    print("=" * 104)
    for k in ("Dp", "E1"):
        pr = np.array([(c, cf, w, h) for _, c, cf, w, h in R[k]["prows"]], float)
        per_img = {}
        for iid, c, cf, w, h in R[k]["prows"]:
            per_img[iid] = per_img.get(iid, 0) + 1
        pv = np.array(list(per_img.values()))
        print(f"  [{k}] 总框 {len(pr):<6} 图数 {len(per_img):<5} 框/图中位 {np.median(pv):.1f}")
        print(f"       conf  P10={np.percentile(pr[:,1],10):.4f} 中位={np.median(pr[:,1]):.4f} "
              f"P90={np.percentile(pr[:,1],90):.4f}")
        sq = np.sqrt(pr[:,2]*pr[:,3])
        print(f"       √area 中位={np.median(sq):.2f}  P10={np.percentile(sq,10):.2f} "
              f"P90={np.percentile(sq,90):.2f}")
    cnt_d = Counter(R["Dp"]["prows"]); cnt_e = Counter(R["E1"]["prows"])
    for k in ("Dp", "E1"):
        c = Counter(x[1] for x in R[k]["prows"])
        print(f"  [{k}] 每类预测数: " + ", ".join(f"{NAMES[i]}:{c[i]}" for i in range(NC) if c[i]))

    (ROOT / "diagnostic/e1_region_gain/_e1_vs_dprime_diag.json").write_text(
        __import__("json").dumps(dict(delta=d0,
                                     per_class=[[NAMES[c], gt_n[c], pc_d[c], pc_e[c], pc_e[c]-pc_d[c]]
                                                for _, c, n, _, _ in rows],
                                     per_iou=[[float(t), a, b, b-a] for t, a, b
                                              in zip(IOU_THRESHOLDS, D["ev"]["per_iou"], E["ev"]["per_iou"])],
                                     iou_seg=dict(low=dlo, mid=dmid, high=dhi)), indent=2), encoding="utf-8")


from collections import Counter  # noqa: E402

if __name__ == "__main__":
    main()
