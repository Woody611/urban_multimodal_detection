"""diagnostic/final_bottleneck_audit/_box_error_decomp.py — 只读：TP 框误差分解（偏置 vs 方差）。

回答一个决定性问题：
  TP 的定位误差是【系统性偏置】(框系统性偏大/偏小/中心偏移 -> loss 侧可干预)
  还是【零均值方差】(随机抖动 -> 受特征/信号限制，改 loss 预期无效)？

数据全部来自已存在的产物（GT + baseline 官方预测），不重新预测、不训练。
"""
from __future__ import annotations

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
SMALL_MAX, MEDIUM_MAX = 32 ** 2, 96 ** 2


def wh(b):
    return np.clip(b[..., 2] - b[..., 0], 1e-6, None), np.clip(b[..., 3] - b[..., 1], 1e-6, None)


def match_official(preds_c, gt_by_img, iou_thr):
    order = sorted(range(len(preds_c)), key=lambda i: -preds_c[i][1])
    ps = [preds_c[i] for i in order]
    matched = {img: np.zeros(len(v), dtype=bool) for img, v in gt_by_img.items()}
    tp = np.zeros(len(ps)); mg = [None] * len(ps)
    for i, (img, _c, box) in enumerate(ps):
        gts = gt_by_img.get(img)
        if gts is None or len(gts) == 0:
            continue
        ious = OM.box_iou_np(box[None, :], gts)[0]
        cand = np.where(~matched[img])[0]
        if cand.size and ious[cand].max() >= iou_thr:
            j = int(cand[int(np.argmax(ious[cand]))])
            tp[i] = 1.0; mg[i] = j; matched[img][j] = True
    return ps, tp, mg


def sizetag(b):
    a = np.clip(b[2] - b[0], 0, None) * np.clip(b[3] - b[1], 0, None)
    return 0 if a < SMALL_MAX else (1 if a < MEDIUM_MAX else 2)


def main():
    gt, preds, st = OM.collect(SPLIT / "images/val/visible", SPLIT / "labels/val/visible",
                               RESULTS, 0.0, True)
    rows = []   # iou, size, cls, scale_ratio, dx_n, dy_n, dar
    for c in range(12):
        if len(gt[c]) == 0:
            continue
        ps, tp, mg = match_official(preds[c], gt[c], 0.5)
        for i, (img, _cf, box) in enumerate(ps):
            j = mg[i]
            if tp[i] < 0.5 or j is None:
                continue
            g = gt[c][img][j]
            ious = OM.box_iou_np(box[None, :], gt[c][img])[0]
            iou = float(ious[j])
            gw, gh = wh(g)
            pw, ph = wh(box)
            ga = float(gw * gh); pa = float(pw * ph)
            s = float(np.sqrt(pa / ga))
            gcx, gcy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
            pcx, pcy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            sc = float(np.sqrt(ga))                      # 归一化尺度
            dxn = float((pcx - gcx) / sc)
            dyn = float((pcy - gcy) / sc)
            dar = float((pw / ph) / (gw / gh))
            rows.append((iou, sizetag(g), c, s, dxn, dyn, dar))
    A = np.array(rows)
    print(f"TP(IoU>=0.5) 数 = {len(A)}")
    szn = ("small", "medium", "large")

    def rep(mask, label):
        if mask.sum() == 0:
            print(f"  {label:<22} n=0"); return
        s, dx, dy, ar = A[mask, 3], A[mask, 4], A[mask, 5], A[mask, 6]
        n = len(s)
        def se(v):
            return v.std(ddof=1) / np.sqrt(n)
        print(f"  {label:<22} n={n:>5}  scale={s.mean():.4f}±{se(s):.4f}  |"
              f"  dx={dx.mean():+.4f}±{se(dx):.4f}  dy={dy.mean():+.4f}±{se(dy):.4f}  |"
              f"  AR={ar.mean():.4f}±{se(ar):.4f}  |  方差: s.std={s.std():.4f}")

    print("\n=== TP 框误差分解（scale=sqrt(pred_area/gt_area)，dx/dy 以 sqrt(gt_area) 归一化）===")
    print("  scale=1 且 dx=dy=0 表示无系统偏置；±值为标准误(SE)")
    rep(np.ones(len(A), bool), "ALL")
    for si, s in enumerate(szn):
        rep(A[:, 1] == si, s)
    print("\n  --- 逐类（按 TP 数排序）---")
    for c in sorted(range(12), key=lambda k: -(A[:, 2] == k).sum()):
        m = A[:, 2] == c
        if m.sum() >= 15:
            rep(m, NAMES[c])

    print("\n=== 显著性判断（|均值| vs 2*SE）===")
    for lbl, col in (("scale", 3), ("dx", 4), ("dy", 5), ("AR", 6)):
        v = A[:, col]; mu = v.mean(); s = v.std(ddof=1) / np.sqrt(len(v))
        dev = abs(mu - (1.0 if lbl in ("scale", "AR") else 0.0))
        print(f"  {lbl:<6} 均值={mu:+.4f}  SE={s:.4f}  |均值-无偏|={dev:.4f}  "
              f"-> {'显著偏置(>2SE)' if dev > 2 * s else '不显著(<=2SE)'}")

    print("\n=== 误差方向分布（TP 框相对 GT 的大小）===")
    s = A[:, 3]
    for lo, hi in ((0, 0.8), (0.8, 0.95), (0.95, 1.05), (1.05, 1.25), (1.25, 9)):
        m = (s >= lo) & (s < hi)
        print(f"  scale ∈ [{lo:.2f},{hi:.2f})  n={int(m.sum()):>5}  {m.mean()*100:>5.1f}%")
    print(f"\n  偏大(sc>1) 占比 = {100*(s>1).mean():.1f}%   偏小(sc<1) 占比 = {100*(s<1).mean():.1f}%")


if __name__ == "__main__":
    main()
