"""p11_transition_audit.py — P11 Healthy -> F0 Transition Audit.

ZERO GPU / ZERO training / ZERO forward / ZERO inference / no source or config change.
只读已有 artifact：P3 `attribution.json`（cohort）+ `from_snapshots/epoch_*.npz`（30 epoch 纵向 per-GT）。

用法:
  python -X utf8 diagnostic/p11_transition_audit.py --out diagnostic/p11_out
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT / "diagnostic/p3_disappearance_audit/attribution.json"
SNAP = ROOT / "diagnostic/full300_trajectory_probe/from_snapshots"
DEAD_THR = 1e-3
EPS_DEATH = []  # filled below


def cliff_delta(a, b):
    """Cliff's delta via rank-based formula (no O(n^2))."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    try:
        from scipy.stats import mannwhitneyu, rankdata
    except Exception:
        return float("nan"), float("nan")
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    # Cliff's delta from U
    d = 2.0 * u / (n1 * n2) - 1.0
    return float(d), float(p)


def load():
    snaps = sorted(SNAP.glob("epoch_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    eps, recs, H, HW, HB = [], [], [], [], []
    keys = cls = role = None
    for p in snaps:
        z = np.load(p, allow_pickle=True)
        e = int(z["epoch"][0])
        if keys is None:
            keys = z["vec_keys"].astype(str); cls = z["vec_cls"].astype(int)
            role = z["vec_role"].astype(str)
        eps.append(e)
        recs.append(pd.DataFrame(dict(
            epoch=e, key=z["vec_keys"].astype(str), cls=z["vec_cls"].astype(int),
            logit=z["logit"], sigmoid=z["sigmoid"], h_norm=z["h_norm"], p3_norm=z["p3_norm"],
            ciou_pos_max=z["ciou_pos_max"], ciou_pos_median=z["ciou_pos_median"],
            cls_pos_max=z["cls_pos_max"], cls_center_max=z["cls_center_max"],
            raw_align_pos_max=z["raw_align_pos_max"], n_pos=z["n_pos"],
            n_in_gts=z["n_in_gts"], pos_mean_ciou=z["pos_mean_ciou"],
        )))
        H.append(z["vec_h"].astype(np.float64)); HW.append(z["head_W"].astype(np.float64))
        HB.append(z["head_b"].astype(np.float64))
    df = pd.concat(recs, ignore_index=True)
    return df, np.array(eps), keys, cls, role, np.array(H), np.array(HW), np.array(HB)


def enrich(df, eps, keys, cls, role, H, HW, HB):
    """per (epoch, gt) derived metrics."""
    n_ep, n_gt, D = H.shape
    out = {}
    for i, e in enumerate(eps):
        h = H[i]                                   # (n_gt, D)
        W = HW[i]                                  # (12, D)
        b = HB[i]                                  # (12,)
        hn = np.linalg.norm(h, axis=1) + 1e-30
        Wn = np.linalg.norm(W, axis=1) + 1e-30
        Wm = W.mean(0); Wmn = np.linalg.norm(Wm) + 1e-30
        Wc = W - Wm
        Zh = h @ W.T + b[None, :]                  # (n_gt, 12) logits
        z_gt = Zh[np.arange(n_gt), cls]
        Zh_masked = Zh.copy(); Zh_masked[np.arange(n_gt), cls] = -np.inf
        z_other = Zh_masked.max(1)
        # class prototype mu_c (mean h over that class, this epoch)
        mu = np.zeros((12, D))
        for c in range(12):
            m = cls == c
            if m.sum() > 0:
                mu[c] = h[m].mean(0)
        cos_mu = np.einsum("ij,ij->i", h, mu[cls]) / (hn * (np.linalg.norm(mu[cls], axis=1) + 1e-30))
        cos_W = np.einsum("ij,ij->i", h, W[cls]) / (hn * Wn[cls])
        cos_Wm = (h @ Wm) / (hn * Wmn)
        out[e] = dict(
            z_gt=z_gt, sigmoid=1 / (1 + np.exp(-z_gt)),
            margin=z_gt - z_other, cos_mu=cos_mu, cos_W=cos_W, cos_Wm=cos_Wm,
            h_norm=hn, W_gt_norm=Wn[cls], W_mean_norm=Wmn,
            W_cent_norm=np.linalg.norm(Wc), logit_recon_err=np.abs(z_gt - Zh[np.arange(n_gt), cls]).max(),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="diagnostic/p11_out")
    A = ap.parse_args()
    out = (ROOT / A.out).resolve(); out.mkdir(parents=True, exist_ok=True)

    df, eps, keys, cls, role, H, HW, HB = load()
    print(f"[load] epochs={eps.min()}..{eps.max()} n={len(eps)} | GT/epoch={H.shape[1]} ukeys={len(set(keys))}")

    # ---- integrity: logit == head_W[cls].h + b[cls] -------------------------
    D = enrich(df, eps, keys, cls, role, H, HW, HB)
    err = max(D[e]["logit_recon_err"] for e in eps)
    print(f"[integrity] max |logit - (W_cls.h+b_cls)| over all epochs = {err:.3e}")

    # ---- cohort from P3 ------------------------------------------------------
    p3 = json.load(open(P3, encoding="utf-8"))
    p3df = pd.DataFrame(p3["rows"])
    p3df["key"] = p3df["stem"] + "#" + (p3df["gt_id"] + 1).astype(str)
    m = p3df.set_index("key")
    base = pd.DataFrame(dict(key=keys, cls=cls, role=role))
    base["failure"] = base["key"].map(m["failure"])
    base["size"] = base["key"].map(m["size"])
    base["area"] = base["key"].map(m["area"])
    base["cls_name"] = base["key"].map(m["cls_name"])
    base["stem"] = base["key"].str.split("#").str[0]
    print("\n[cohort] role x failure")
    print(pd.crosstab(base["role"], base["failure"], dropna=False))

    # per-GT time series
    T = {}
    for k in ("z_gt", "sigmoid", "margin", "cos_mu", "cos_W", "cos_Wm", "h_norm", "W_gt_norm"):
        T[k] = np.array([D[e][k] for e in eps])             # (n_ep, n_gt)
    T["best_IoU"] = np.array([df[(df.epoch == e)].sort_values("key")["ciou_pos_max"].values for e in eps]) \
        if False else np.array([D and np.array([0])])[0:0]
    # best_IoU / target from the df (order matches keys order)
    io = np.array([df[df.epoch == e]["ciou_pos_max"].values for e in eps])
    T["best_IoU"] = io
    order_ok = all((df[df.epoch == e]["key"].values == keys).all() for e in eps)
    print(f"[integrity] per-epoch row order matches vec_keys: {order_ok}")
    T["target"] = io

    final_i = len(eps) - 1
    final_dead = T["sigmoid"][final_i] < DEAD_THR
    base["final_dead"] = final_dead
    base["final_f0_p3"] = (base["failure"] == "F0_no_raw_candidate").values
    print(f"[cohort] final(ep{eps[final_i]}) sigmoid<1e-3 : {final_dead.sum()} | P3 F0: {base['final_f0_p3'].sum()}"
          f" | agree: {(final_dead == base['final_f0_p3']).sum()}/{len(base)}")

    # ---- t_dead -------------------------------------------------------------
    dead = T["sigmoid"] < DEAD_THR                          # (n_ep, n_gt)
    t_dead = np.full(len(keys), np.nan)
    for j in range(len(keys)):
        w = np.where(dead[:, j])[0]
        t_dead[j] = eps[w[0]] if len(w) else np.nan
    base["t_dead"] = t_dead
    base["init_dead"] = dead[0]
    base["ever_dead"] = ~np.isnan(t_dead)

    F0 = base[base["final_f0_p3"]].copy()
    MAT = base[(base["failure"] == "MATCHED@0.5")].copy()
    print(f"\n[cohort] F0 n={len(F0)}  MATCHED n={len(MAT)}")
    print(f"[cohort] F0 roles: {F0['role'].value_counts().to_dict()}")

    def bucket(r):
        if bool(r["init_dead"]):
            return "INIT_DEAD"
        if not r["ever_dead"]:
            return "NEVER_DEAD"
        t = r["t_dead"]
        return "EARLY(1-50)" if t <= 50 else ("MID(51-150)" if t <= 150 else "LATE(151-290)")
    F0["bucket"] = F0.apply(bucket, axis=1)
    tab = F0["bucket"].value_counts().reindex(
        ["INIT_DEAD", "EARLY(1-50)", "MID(51-150)", "LATE(151-290)", "NEVER_DEAD"]).fillna(0).astype(int)
    print("\n[Q1] F0 t_dead buckets")
    print(tab.to_string())
    print(f"  transition F0 (t_dead>ep0) = {(F0['bucket'].isin(['EARLY(1-50)','MID(51-150)','LATE(151-290)'])).sum()}"
          f" / {len(F0)} = {100*(F0['bucket'].isin(['EARLY(1-50)','MID(51-150)','LATE(151-290)'])).mean():.1f}%")

    tr = F0[F0["bucket"].isin(["EARLY(1-50)", "MID(51-150)", "LATE(151-290)"])]
    if len(tr):
        q = tr["t_dead"].quantile([.1, .25, .5, .75, .9])
        print(f"[Q2] transition F0 median t_dead = {tr['t_dead'].median():.0f}  "
              f"p10={q[.1]:.0f} p25={q[.25]:.0f} p75={q[.75]:.0f} p90={q[.9]:.0f}  n={len(tr)}")
    # all F0 (incl init dead as t=0)
    F0a = F0.copy(); F0a["t_all"] = F0a["t_dead"].fillna(np.inf)
    print(f"[Q2] ALL F0 median t_dead = {F0a['t_dead'].median():.0f} (NaN-excluded n={F0a['t_dead'].notna().sum()})")

    # ---- onset detection ----------------------------------------------------
    def onset(series_j, worsen="down", need=2):
        """first index i with `need` consecutive strictly worse observations."""
        x = series_j
        n = len(x)
        for i in range(n - need):
            seg = x[i:i + need + 1]
            if not np.all(np.isfinite(seg)):
                continue
            d = np.diff(seg)
            ok = np.all(d < 0) if worsen == "down" else np.all(d > 0)
            if ok:
                return i
        return -1

    SIGS = {"z_gt": "down", "margin": "down", "cos_mu": "down", "cos_W": "down",
            "h_norm": "down", "best_IoU": "down", "target": "down", "cos_Wm": "down"}
    ons = {k: np.array([onset(T[k][:, j], w) for j in range(len(keys))]) for k, w in SIGS.items()}

    # ---- Pre-death signal table (on transition-F0 only, within-GT normalized) --
    print("\n[Q3] pre-death onsets (transition F0 only; epoch index grid = " + str(list(eps)) + ")")
    idx_tr = tr.index.values
    rows = []
    for k in SIGS:
        oi = ons[k][idx_tr]
        td_i = np.array([int(np.where(eps == t)[0][0]) for t in tr["t_dead"].values])
        rel = (eps[np.clip(oi, 0, len(eps) - 1)] - tr["t_dead"].values) if True else None
        rel = np.where(oi >= 0, eps[np.clip(oi, 0, len(eps) - 1)] - tr["t_dead"].values, np.nan)
        before = np.sum((rel < 0) & np.isfinite(rel))
        meas = np.isfinite(rel).sum()
        rows.append(dict(signal=k, measurable=int(meas), pct_measurable=100 * meas / max(len(tr), 1),
                         onset_before_death=int(before),
                         pct_before=100 * before / max(meas, 1),
                         median_rel_days=float(np.nanmedian(rel)) if meas else float("nan")))
    onset_tab = pd.DataFrame(rows).sort_values("pct_before", ascending=False)
    print(onset_tab.to_string(index=False))

    # ---- within-GT normalized trajectories ---------------------------------
    def norm_traj(key):
        a = T[key]
        return a - a[0][None, :]
    print("\n[§10] within-GT normalized change (median), F0-transition vs MATCHED")
    nz = {k: T[k] - T[k][0][None, :] for k in ("z_gt", "margin", "cos_mu", "cos_W", "best_IoU", "h_norm")}
    idx_mat = MAT.index.values
    tab10 = []
    for k, arr in nz.items():
        for i, e in enumerate(eps):
            if e % 50 and e not in (0, 10, 20, 290):
                continue
            a = arr[i][idx_tr]; b = arr[i][idx_mat]
            d, p = cliff_delta(a, b)
            tab10.append(dict(metric=k, epoch=e, f0_med=float(np.median(a)), mat_med=float(np.median(b)),
                              diff=float(np.median(a) - np.median(b)), cliffs_d=d, p=p))
    tr10 = pd.DataFrame(tab10)
    piv = tr10.pivot_table(index="epoch", columns="metric", values="diff")
    print(piv.round(4).to_string())

    # ---- §14 effect sizes at final epoch ------------------------------------
    print("\n[§14] Mann-Whitney / Cliff's delta, F0 vs MATCHED, FINAL epoch values")
    eff = []
    for k in ("z_gt", "margin", "cos_mu", "cos_W", "cos_Wm", "h_norm", "W_gt_norm", "best_IoU"):
        a = T[k][final_i][idx_tr]; b = T[k][final_i][idx_mat]
        d, p = cliff_delta(a, b)
        eff.append(dict(metric=k, f0_med=float(np.median(a)), mat_med=float(np.median(b)), cliffs_d=d, p=p))
    effdf = pd.DataFrame(eff)
    print(effdf.to_string(index=False))

    # ---- §12 collapse shape -------------------------------------------------
    def shape(j):
        z = T["z_gt"][:, j]
        td = t_dead[j]
        if np.isnan(td):
            return "never"
        ti = int(np.where(eps == td)[0][0])
        if ti <= 1:
            return "init"
        pre = z[:ti]
        drop = z[ti] - z[ti - 1]
        prior_range = np.ptp(pre[-3:]) if ti >= 3 else np.ptp(pre)
        return "abrupt" if abs(drop) > 3 * (prior_range + 1e-6) else "gradual"
    F0["shape"] = [shape(j) for j in F0.index.values]
    print("\n[§9] collapse shape for transition F0")
    print(F0[F0["bucket"] != "INIT_DEAD"]["shape"].value_counts().to_string())

    # ---- §11 size / class / spatial -----------------------------------------
    print("\n[§11] t_dead by size (F0) and by class (all medium probe GT)")
    for s, g in F0.groupby("size"):
        print(f"  {s:7s} n={len(g):4d}  median t_dead={g['t_dead'].median()}  init_dead={int(g['init_dead'].sum())}")
    cc = base.groupby("cls_name").agg(n=("final_f0_p3", "size"), f0=("final_f0_p3", "sum"),
                                      f0_rate=("final_f0_p3", "mean"), t_dead=("t_dead", "median"))
    print(cc.sort_values("f0_rate", ascending=False).round(3).to_string())
    # border distance
    bx = base.copy()
    m2 = p3df.set_index("key")
    bx["dx"] = bx["key"].map(m2["best_raw_dx"]); bx["dy"] = bx["key"].map(m2["best_raw_dy"])
    # use GT box vs canvas from layers.npz
    import collections
    print("\n[§11] (spatial: GT-center to canvas border, canvas 1312x736)")
    zz = np.load(ROOT / "diagnostic/p3_disappearance_audit/layers.npz", allow_pickle=True)
    stems = zz["stem"].astype(str)
    # recompute GT boxes from the source label files is out of scope -> use best_raw_dx/dy if present
    print("  best_raw_dx/dy non-null:", bx["dx"].notna().sum())

    # ---- outputs ------------------------------------------------------------
    base.to_csv(out / "p11_gt_timeline.csv", index=False, encoding="utf-8")
    np.savez_compressed(out / "p11_series.npz", epochs=np.array(eps), keys=np.array(keys),
                        cls=cls, role=role, **{k: v for k, v in T.items()}, **{f"onset_{k}": v for k, v in ons.items()})
    json.dump(dict(epochs=[int(e) for e in eps], n_gt=int(len(keys)),
                   f0_buckets=tab.to_dict(), n_f0=int(len(F0)), n_matched=int(len(MAT)),
                   transition_f0=int(len(tr)),
                   median_t_dead_transition=float(tr["t_dead"].median()) if len(tr) else None,
                   logit_recon_err=float(err), order_ok=bool(order_ok),
                   onset_table=onset_tab.to_dict("records"), effects=effdf.to_dict("records")),
              open(out / "p11_summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
