"""p10_timeline.py — 把 P10 的 W/b 轨迹与已有的 logit / dead-rate / val mAP 放到同一时间轴上。

只读。无 forward / 训练 / 推理。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P10 = ROOT / "diagnostic/p10_out"
PROBE = ROOT / "diagnostic/full300_trajectory_probe"
RUN = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe"

DECAY, TAU, EPOCHS, LRF, LR0 = 0.9999, 2000, 300, 0.01, 0.005


def lf(e):
    return ((1 - math.cos(e * math.pi / EPOCHS)) / 2) * (LRF - 1) + 1


def main():
    s2 = json.load(open(P10 / "p10_stage2.json", encoding="utf-8"))
    wk = "model.30.cv3.0.2.weight"
    W = {r["epoch"]: r[wk + "|Wmean"] for r in s2}
    Wc = {r["epoch"]: r[wk + "|Wcent"] for r in s2}
    B = {r["epoch"]: r[wk + "|bmean"] for r in s2}
    Mmean = {r["epoch"]: r[wk + "|Mmean"] for r in s2}
    Mcent = {r["epoch"]: r[wk + "|Mcent"] for r in s2}
    cosMW = {r["epoch"]: r[wk + "|cosMw_Wm"] for r in s2}
    eps = sorted(W)

    # ---- GT-cell logit / dead rate from the trajectory probe -----------------
    pg = pd.read_csv(PROBE / "per_gt_trajectory.csv")
    logit = pg.groupby("epoch")["logit"].median().to_dict()
    dead = pg.assign(d=pg["sigmoid"] < 1e-3).groupby("epoch")["d"].mean().to_dict()
    cls_pos = pg.groupby("epoch")["cls_pos_max"].median().to_dict()

    # keep only epochs present in both
    eps = [e for e in eps if e in logit]
    # probe epochs are 0-based; snapshot 'epoch' field == 0-based epoch index too
    # ---- val mAP from results.csv -------------------------------------------
    res = pd.read_csv(RUN / "results.csv")
    res.columns = [c.strip() for c in res.columns]
    # results.csv row 1 == epoch 1 (1-based) -> 0-based index = row-1
    res["ep0"] = res["epoch"] - 1
    map5095 = res.set_index("ep0")["metrics/mAP50-95(B)"].to_dict()
    map50 = res.set_index("ep0")["metrics/mAP50(B)"].to_dict()

    # ---- EMA updates per epoch ----------------------------------------------
    md = json.load(open(PROBE / "trajectory_metadata.json", encoding="utf-8"))
    ups = {d["epoch"]: d["ema_updates_total"] for d in md["observation_timing_evidence"]["runtime"]}

    rows = []
    for e in eps:
        u = ups.get(e, int(round(109 + max(0, e - 3) * 25 + (0 if e >= 3 else 0))))
        d = DECAY * (1 - math.exp(-u / TAU))
        rows.append(dict(
            epoch=e, updates=u, ema_decay_per_update=d,
            ema_memory_updates=1 / (1 - d),
            lr=LR0 * lf(e),
            Wmean=W[e], Wcent=Wc[e], b_mean=B[e],
            Mmean=Mmean[e], Mcent=Mcent[e], cos_Mmean_Wmean=cosMW[e],
            gt_logit_median=logit[e], dead_rate=dead[e], cls_pos_med=cls_pos.get(e, float("nan")),
            val_mAP5095=map5095.get(e, float("nan")), val_mAP50=map50.get(e, float("nan")),
        ))

    hdr = f"{'ep':>4} {'upd':>5} {'decay':>6} {'mem':>5} {'lr':>9} {'|Wmean|':>8} {'|Wcent|':>8} " \
          f"{'b_mean':>8} {'cos(Mm,Wm)':>10} {'GTlogit':>8} {'dead%':>6} {'val50-95':>9}"
    print(hdr)
    for r in rows:
        print(f"{r['epoch']:>4} {r['updates']:>5} {r['ema_decay_per_update']:>6.3f} "
              f"{r['ema_memory_updates']:>5.1f} {r['lr']:>9.6f} {r['Wmean']:>8.4f} {r['Wcent']:>8.4f} "
              f"{r['b_mean']:>+8.4f} {r['cos_Mmean_Wmean']:>+10.4f} {r['gt_logit_median']:>8.3f} "
              f"{100*r['dead_rate']:>6.2f} {r['val_mAP5095']:>9.5f}")

    # ---- precedence: normalize each leg to its own ep0->epN span ------------
    print("\n[precedence] fraction of each variable's total ep0->ep290 change reached by epoch e")
    print(f"{'ep':>4} {'|Wmean|':>9} {'GTlogit':>9} {'dead%':>9} {'val_mAP':>9} {'lr':>9}")
    base = rows[0]
    last = rows[-1]
    for r in rows:
        def fr(cur, b, l):
            if abs(l - b) < 1e-12:
                return float("nan")
            return (cur - b) / (l - b)
        print(f"{r['epoch']:>4} {fr(r['Wmean'],base['Wmean'],last['Wmean'])*100:9.1f} "
              f"{fr(r['gt_logit_median'],base['gt_logit_median'],last['gt_logit_median'])*100:9.1f} "
              f"{fr(r['dead_rate'],base['dead_rate'],last['dead_rate'])*100:9.1f} "
              f"{fr(r['val_mAP5095'],base['val_mAP5095'],last['val_mAP5095'])*100:9.1f} "
              f"{fr(r['lr'],base['lr'],last['lr'])*100:9.1f}")

    # ---- lag correlation between W_mean leg and logit leg -------------------
    w = np.array([r["Wmean"] for r in rows])
    g = np.array([r["gt_logit_median"] for r in rows])
    print("\n[association] (NOT causality)")
    print(f"  corr(|Wmean|, GT logit) = {np.corrcoef(w, g)[0,1]:+.4f}   (n={len(w)})")
    dw = np.diff(w); dg = np.diff(g)
    print(f"  corr(d|Wmean|, d GT logit) = {np.corrcoef(dw, dg)[0,1]:+.4f}")

    out = P10 / "p10_timeline.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
