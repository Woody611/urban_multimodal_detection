"""diagnostic/p3attn_probe/_sml_eval.py — 只读：对若干 results 目录统一计算官方口径总指标 + Small/Medium/Large AP。

用法:
  python diagnostic/p3attn_probe/_sml_eval.py 名称1=路径1 名称2=路径2 ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import official_map as OM  # noqa: E402

SPLIT = ROOT / "data/processed/rgbid_split"
THR = [float(t) for t in OM.IOUV_OFFICIAL]
SMAX, MMAX = 32 ** 2, 96 ** 2


def bucket(b):
    a = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    r = np.full(len(a), 2, int)
    r[a < SMAX] = 0
    r[(a >= SMAX) & (a < MMAX)] = 1
    return r


def main():
    pairs = [a.split("=", 1) for a in sys.argv[1:]]
    gt, _, _ = OM.collect(SPLIT / "images/val/visible", SPLIT / "labels/val/visible",
                          ROOT / pairs[0][1], 0.0, True)
    res = {}
    for name, d in pairs:
        gt2, preds, st = OM.collect(SPLIT / "images/val/visible", SPLIT / "labels/val/visible",
                                    ROOT / d, 0.0, True)
        off = OM.evaluate(gt2, preds, "mean", "zero", "conf")
        # micro P/R @IoU0.5
        cs, tps, fps = [], [], []
        for c in range(12):
            if len(gt2[c]) == 0:
                continue
            order = sorted(range(len(preds[c])), key=lambda i: -preds[c][i][1])
            psc = [preds[c][i] for i in order]
            matched = {img: np.zeros(len(v), bool) for img, v in gt2[c].items()}
            for i, (img, _cf, box) in enumerate(psc):
                g = gt2[c].get(img)
                if g is None or len(g) == 0:
                    tps.append(0.0); fps.append(1.0); cs.append(_cf); continue
                ious = OM.box_iou_np(box[None, :], g)[0]
                cand = np.where(~matched[img])[0]
                if cand.size and ious[cand].max() >= 0.5:
                    j = int(cand[int(np.argmax(ious[cand]))])
                    matched[img][j] = True
                    tps.append(1.0); fps.append(0.0)
                else:
                    tps.append(0.0); fps.append(1.0)
                cs.append(_cf)
        o = np.argsort(-np.array(cs), kind="stable")
        tps = np.array(tps)[o]; fps = np.array(fps)[o]
        npos = int(sum(len(bx) for c in range(12) for bx in gt2[c].values()))
        tpc, fpc = np.cumsum(tps), np.cumsum(fps)
        rec = tpc / max(npos, 1); pre = tpc / np.maximum(tpc + fpc, OM.EPS)
        f1 = 2 * pre * rec / np.maximum(pre + rec, OM.EPS); k = int(np.argmax(f1))
        # 尺寸分桶 AP
        sz = {}
        for si, sn in enumerate(("small", "medium", "large")):
            aps = []
            for t in THR:
                vals = []
                for c in range(12):
                    if len(gt2[c]) == 0:
                        continue
                    sub = {}
                    for img, bx in gt2[c].items():
                        m = bucket(bx) == si
                        if m.any():
                            sub[img] = bx[m]
                    if not sub:
                        continue
                    r2, p2 = OM.curve_for_class(preds[c], sub, t, match="conf")
                    vals.append(OM.ap_from_curve(r2, p2, avg="mean", tail="zero"))
                aps.append(float(np.mean(vals)) if vals else 0.0)
            sz[sn] = float(np.mean(aps))
        res[name] = dict(m50=off["mAP50"], m75=off["mAP75"], m=off["mAP50-95"],
                         P=float(pre[k]), R=float(rec[k]), sz=sz, pred=st["pred"])

    print("=" * 104)
    print("官方口径（predict_rect --mode rect --imgsz 1280 + official_map，与 baseline 逐字同参）")
    print("=" * 104)
    print(f"  {'实验':<26}{'mAP50':>9}{'mAP75':>9}{'mAP50-95':>10}{'P':>8}{'R':>8}{'框数':>7}")
    for k, v in res.items():
        print(f"  {k:<26}{v['m50']:>9.5f}{v['m75']:>9.5f}{v['m']:>10.5f}{v['P']:>8.4f}{v['R']:>8.4f}{v['pred']:>7}")
    print()
    print(f"  {'size':<26}{'small':>11}{'medium':>11}{'large':>11}")
    for k, v in res.items():
        print(f"  {k:<26}{v['sz']['small']:>11.5f}{v['sz']['medium']:>11.5f}{v['sz']['large']:>11.5f}")
    base = pairs[0][0]
    print()
    print(f"  相对 {base} 的差值：")
    print(f"  {'实验':<26}{'ΔmAP50':>10}{'ΔmAP75':>10}{'ΔmAP50-95':>12}{'ΔSmall':>10}{'ΔMedium':>10}{'ΔLarge':>10}")
    b = res[base]
    for k, v in res.items():
        if k == base:
            continue
        print(f"  {k:<26}{v['m50']-b['m50']:>+10.5f}{v['m75']-b['m75']:>+10.5f}{v['m']-b['m']:>+12.5f}"
              f"{v['sz']['small']-b['sz']['small']:>+10.5f}{v['sz']['medium']-b['sz']['medium']:>+10.5f}"
              f"{v['sz']['large']-b['sz']['large']:>+10.5f}")
    print("=" * 104)


if __name__ == "__main__":
    main()
