"""diagnostic/final_bottleneck_audit/_analyze.py — 只读瓶颈审计计算（不训练、不重新预测）。

数据来源（全部为**已存在**的产物，不重新生成任何预测）：
  GT        : data/processed/rgbid_split/labels/val/visible
  预测      : diagnostic/rgbid_inference_gain/RGBID_baseline/results   （= 官方 0.50928 那一份）
  口径实现  : scripts/official_map.py（直接 import，参数完全一致）

输出：
  1) per-class x per-IoU AP 矩阵
  2) micro P/R 随 IoU 的变化
  3) 按 COCO area range 分桶的 AP（small / medium / large）
  4) TP 预测框的 best-IoU 分布（总体 / 按 GT 尺寸分桶）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import official_map as OM  # noqa: E402

NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}
SPLIT = ROOT / "data/processed/rgbid_split"
RESULTS = ROOT / "diagnostic/rgbid_inference_gain/RGBID_baseline/results"
THR = [float(t) for t in OM.IOUV_OFFICIAL]
SMALL_MAX, MEDIUM_MAX = 32 ** 2, 96 ** 2
OUT = Path(__file__).resolve().parent


def area(b):
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def bucket_of_boxes(boxes):
    a = area(boxes)
    r = np.full(len(a), 2, dtype=int)
    r[a < SMALL_MAX] = 0
    r[(a >= SMALL_MAX) & (a < MEDIUM_MAX)] = 1
    return r


def match_official(preds_c, gt_by_img, iou_thr):
    """官方 conf 顺序贪心匹配；返回 (preds_sorted, tp, matched_gt_idx)。"""
    order = sorted(range(len(preds_c)), key=lambda i: -preds_c[i][1])
    ps = [preds_c[i] for i in order]
    matched = {img: np.zeros(len(v), dtype=bool) for img, v in gt_by_img.items()}
    tp = np.zeros(len(ps))
    mg = [None] * len(ps)
    for i, (img, _c, box) in enumerate(ps):
        gts = gt_by_img.get(img)
        if gts is None or len(gts) == 0:
            continue
        ious = OM.box_iou_np(box[None, :], gts)[0]
        cand = np.where(~matched[img])[0]
        if cand.size and ious[cand].max() >= iou_thr:
            j = int(cand[int(np.argmax(ious[cand]))])
            tp[i] = 1.0
            mg[i] = j
            matched[img][j] = True
    return ps, tp, mg


def main():
    gt, preds, st = OM.collect(SPLIT / "images/val/visible", SPLIT / "labels/val/visible",
                               RESULTS, 0.0, True)
    print(f"images={st['images']} corrupt={st['corrupt']} GT={st['gt']} pred={st['pred']}")

    # ---------- 1) per-class x per-IoU AP ----------
    M = np.full((12, 10), np.nan)
    for ti, t in enumerate(THR):
        for c in range(12):
            if len(gt[c]) == 0:
                continue
            rec, pre = OM.curve_for_class(preds[c], gt[c], t, match="conf")
            M[c, ti] = OM.ap_from_curve(rec, pre, avg="mean", tail="zero")
    np.save(OUT / "_perclass_periou.npy", M)

    # ---------- 2) micro P/R 随 IoU ----------
    micro = []
    for ti, t in enumerate(THR):
        cs, tps, fps = [], [], []
        for c in range(12):
            if len(gt[c]) == 0:
                continue
            ps, tp, _ = match_official(preds[c], gt[c], t)
            for i, (_img, cf, _b) in enumerate(ps):
                cs.append(cf); tps.append(tp[i]); fps.append(1.0 - tp[i])
        if not cs:
            micro.append((0.0, 0.0, 0.0)); continue
        o = np.argsort(-np.array(cs), kind="stable")
        tps = np.array(tps)[o]; fps = np.array(fps)[o]
        npos = int(sum(len(bx) for c in range(12) for bx in gt[c].values()))
        tpc, fpc = np.cumsum(tps), np.cumsum(fps)
        rec = tpc / max(npos, 1)
        pre = tpc / np.maximum(tpc + fpc, OM.EPS)
        f1 = 2 * pre * rec / np.maximum(pre + rec, OM.EPS)
        k = int(np.argmax(f1))
        micro.append((float(pre[k]), float(rec[k]), float(f1[k])))
    np.save(OUT / "_micro_pr.npy", np.array(micro))

    # ---------- 3) COCO area-range AP ----------
    size_ap = {}
    for si, sname in enumerate(("small", "medium", "large")):
        aps = []
        for t in THR:
            vals = []
            for c in range(12):
                if len(gt[c]) == 0:
                    continue
                # GT 限制到该尺寸区间
                sub = {}
                for img, bx in gt[c].items():
                    b = bucket_of_boxes(bx)
                    m = b == si
                    if m.any():
                        sub[img] = bx[m]
                if not sub:
                    continue
                rec, pre = OM.curve_for_class(preds[c], sub, t, match="conf")
                vals.append(OM.ap_from_curve(rec, pre, avg="mean", tail="zero"))
            aps.append(float(np.mean(vals)) if vals else 0.0)
        size_ap[sname] = aps
    np.save(OUT / "_size_ap.npy", np.array([size_ap["small"], size_ap["medium"], size_ap["large"]]))

    # ---------- 4) TP 的 best-IoU 分布 ----------
    recs = []
    for c in range(12):
        if len(gt[c]) == 0:
            continue
        ps, tp, mg = match_official(preds[c], gt[c], 0.5)   # 以 IoU>=0.5 定义 TP
        for i, (img, _cf, box) in enumerate(ps):
            if tp[i] < 0.5:
                continue
            gts = gt[c][img]
            ious = OM.box_iou_np(box[None, :], gts)[0]
            j = int(np.argmax(ious))
            b = int(bucket_of_boxes(gts[j:j + 1])[0])
            recs.append((float(ious[j]), b, c))
    arr = np.array([(r[0], r[1], r[2]) for r in recs])
    edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    labels = ["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    hist_all = np.histogram(arr[:, 0], bins=edges)[0]
    hist_sz = {s: np.histogram(arr[arr[:, 1] == si][:, 0], bins=edges)[0] for si, s in
               enumerate(("small", "medium", "large"))}
    np.save(OUT / "_tp_iou.npy", arr)

    # ---------- 打印 ----------
    print("\n" + "=" * 96)
    print("1) AP 随 IoU 阈值（官方口径）  —— per-class 与 macro")
    print("=" * 96)
    print(f"{'class':<13}{'GT':>5}" + "".join(f"{t:>8.2f}" for t in THR))
    for c in range(12):
        n = sum(len(v) for v in gt[c].values())
        print(f"{NAMES[c]:<13}{n:>5}" + "".join(
            (f"{M[c,i]:>8.4f}" if not np.isnan(M[c, i]) else f"{'--':>8}") for i in range(10)))
    macro = np.nanmean(M, axis=0)
    print(f"{'MACRO':<13}{'':>5}" + "".join(f"{macro[i]:>8.5f}" for i in range(10)))
    drop = np.diff(macro)
    print(f"{'Δ(上一档)':<13}{'':>5}" + "".join(f"{drop[i]:>+8.4f}" for i in range(9)))

    print("\n" + "=" * 96)
    print("2) micro P/R/F1 随 IoU（官方 conf 顺序匹配，全类合并）")
    print("=" * 96)
    print(f"{'IoU':>6}{'P':>10}{'R':>10}{'F1':>10}")
    for i, t in enumerate(THR):
        print(f"{t:>6.2f}{micro[i][0]:>10.4f}{micro[i][1]:>10.4f}{micro[i][2]:>10.4f}")

    print("\n" + "=" * 96)
    print("3) 按 COCO area range 的 AP（small <32², medium 32²-96², large >96²）")
    print("=" * 96)
    print(f"{'size':<9}{'AP50':>10}{'AP75':>10}{'AP50-95':>11}{'n_GT':>8}")
    for si, s in enumerate(("small", "medium", "large")):
        n = sum(int((bucket_of_boxes(bx) == si).sum()) for c in range(12) for bx in gt[c].values())
        print(f"{s:<9}{size_ap[s][0]:>10.5f}{size_ap[s][5]:>10.5f}"
              f"{np.mean(size_ap[s]):>11.5f}{n:>8}")

    print("\n" + "=" * 96)
    print("4) TP 预测框的 best-IoU 分布（TP 定义为 IoU>=0.5 官方匹配成功）")
    print("=" * 96)
    tot = hist_all.sum()
    print(f"  TP 总数 = {tot}")
    print(f"{'IoU 区间':<12}{'n':>7}{'占比':>9}   按 GT 尺寸分布(small/medium/large)")
    for i, lb in enumerate(labels):
        row = " / ".join(f"{hist_sz[s][i]}" for s in ("small", "medium", "large"))
        print(f"{lb:<12}{hist_all[i]:>7}{hist_all[i]/tot*100:>8.1f}%   {row}")
    print(f"\n  中位 IoU = {np.median(arr[:,0]):.4f}   均值 = {arr[:,0].mean():.4f}")
    for si, s in enumerate(("small", "medium", "large")):
        sub = arr[arr[:, 1] == si][:, 0]
        if len(sub):
            hi = (sub >= 0.75).mean() * 100
            print(f"  {s:<7} n={len(sub):>5}  中位={np.median(sub):.4f}  均值={sub.mean():.4f}  "
                  f"达到 IoU>=0.75 的比例={hi:.1f}%")

    # 逐类 AP75
    print("\n" + "=" * 96)
    print("5) 逐类 AP50 / AP75 / AP50-95 / GT 数（官方口径）")
    print("=" * 96)
    print(f"{'class':<13}{'GT':>6}{'AP50':>9}{'AP75':>9}{'AP50-95':>10}{'AP75/AP50':>11}")
    rows = []
    for c in range(12):
        n = sum(len(v) for v in gt[c].values())
        rows.append((c, n, M[c, 0], M[c, 5], float(np.nanmean(M[c]))))
    for c, n, a50, a75, a in sorted(rows, key=lambda r: r[4]):
        print(f"{NAMES[c]:<13}{n:>6}{a50:>9.4f}{a75:>9.4f}{a:>10.4f}{a75/a50:>11.4f}")

    (OUT / "_summary.json").write_text(json.dumps({
        "macro_per_iou": macro.tolist(),
        "micro_pr": micro,
        "size_ap": size_ap,
        "tp_iou_hist": hist_all.tolist(),
        "tp_iou_hist_by_size": {k: v.tolist() for k, v in hist_sz.items()},
    }, indent=2), encoding="utf-8")
    print("\n[saved] _summary.json / _perclass_periou.npy / _micro_pr.npy / _size_ap.npy / _tp_iou.npy")


if __name__ == "__main__":
    main()
