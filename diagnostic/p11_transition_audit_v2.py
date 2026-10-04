"""p11_transition_audit_v2.py — P11 with the decisive MATCHED-control onset test.

ZERO GPU / training / forward / inference / no source or config change. Read-only artifacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT / "diagnostic/p3_disappearance_audit/attribution.json"
SNAP = ROOT / "diagnostic/full300_trajectory_probe/from_snapshots"
CELLS = ROOT / "diagnostic/p3_trajectory_probe/medium_cells_canvas.json"
OUT = ROOT / "diagnostic/p11_out"
DEAD_THR = 1e-3


def load():
    snaps = sorted(SNAP.glob("epoch_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    eps, dfs, H, HW, HB = [], [], [], [], []
    for p in snaps:
        z = np.load(p, allow_pickle=True)
        eps.append(int(z["epoch"][0]))
        dfs.append(pd.DataFrame(dict(
            epoch=int(z["epoch"][0]), key=z["vec_keys"].astype(str),
            logit=z["logit"].astype(float), sigmoid=z["sigmoid"].astype(float),
            h_norm=z["h_norm"].astype(float), ciou_pos_max=z["ciou_pos_max"].astype(float),
            cls_pos_max=z["cls_pos_max"].astype(float), n_pos=z["n_pos"].astype(float))))
        H.append(z["vec_h"].astype(np.float64)); HW.append(z["head_W"].astype(np.float64))
        HB.append(z["head_b"].astype(np.float64))
    df = pd.concat(dfs, ignore_index=True)
    keys = dfs[0]["key"].values
    return df, np.array(eps), keys, np.array(H), np.array(HW), np.array(HB)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, eps, keys, H, HW, HB = load()
    n_ep, n_gt, D = H.shape
    order_ok = all((df[df.epoch == e]["key"].values == keys).all() for e in eps)
    print(f"[load] epochs {eps[0]}..{eps[-1]} n={n_ep} | GT={n_gt} | row-order ok={order_ok}")

    # ---------- per-GT per-epoch metrics -------------------------------------
    zc = df["logit"].values.reshape(n_ep, n_gt)
    sg = df["sigmoid"].values.reshape(n_ep, n_gt)
    hn = df["h_norm"].values.reshape(n_ep, n_gt)
    iou = df["ciou_pos_max"].values.reshape(n_ep, n_gt)

    # ---- integrity: logit must equal W_cls.h + b_cls exactly ----------------
    err = 0.0
    for i, e in enumerate(eps):
        row = df[df.epoch == e]
        cls_i = row["key"].map({k: c for k, c in zip(keys, np.load(
            SNAP / f"epoch_{e:03d}.npz", allow_pickle=True)["vec_cls"].astype(int))}).values.astype(int)
        zz = np.einsum("ij,ij->i", H[i], HW[i][cls_i]) + HB[i][cls_i]
        err = max(err, float(np.abs(zz - zc[i]).max()))
    print(f"[integrity] max |logit - (W_cls.h + b_cls)| = {err:.3e}")

    # ---------- cohort --------------------------------------------------------
    p3 = pd.DataFrame(json.load(open(P3, encoding="utf-8"))["rows"])
    p3["key"] = p3["stem"] + "#" + (p3["gt_id"] + 1).astype(str)
    m = p3.set_index("key")
    cl = np.array([int(k.split("#")[1]) for k in []]) if False else None
    cls = np.load(SNAP / "epoch_000.npz", allow_pickle=True)["vec_cls"].astype(int)
    role = np.load(SNAP / "epoch_000.npz", allow_pickle=True)["vec_role"].astype(str)
    base = pd.DataFrame(dict(key=keys, cls=cls, role=role))
    base["failure"] = base["key"].map(m["failure"])
    base["size"] = base["key"].map(m["size"])
    base["area"] = base["key"].map(m["area"]).astype(float)
    base["cls_name"] = base["key"].map(m["cls_name"])

    cells = pd.DataFrame(json.load(open(CELLS, encoding="utf-8"))["rows"])[["key", "x_center_canvas", "y_center_canvas"]]
    base = base.merge(cells, on="key", how="left")
    CW, CH = 1312.0, 736.0
    base["border_dist"] = np.minimum.reduce([
        base["x_center_canvas"], CW - base["x_center_canvas"],
        base["y_center_canvas"], CH - base["y_center_canvas"]])
    base["border"] = base["border_dist"] < 0.10 * np.minimum(CW, CH)

    # ---------- signals -------------------------------------------------------
    clsD = {}
    for i, e in enumerate(eps):
        mu = np.zeros((12, D))
        for c in range(12):
            msk = cls == c
            if msk.sum():
                mu[c] = H[i][msk].mean(0)
        clsD[e] = mu
    SIG = {}
    SIG["z_gt"] = zc
    SIG["best_IoU"] = iou
    SIG["h_norm"] = hn
    for nm in ("cos_mu", "cos_W", "cos_Wm", "margin"):
        SIG[nm] = np.full((n_ep, n_gt), np.nan)
    for i, e in enumerate(eps):
        h = H[i]; W = HW[i]; b = HB[i]
        hnn = np.linalg.norm(h, axis=1) + 1e-30
        Wn = np.linalg.norm(W, axis=1) + 1e-30
        Wm = W.mean(0); Wmn = np.linalg.norm(Wm) + 1e-30
        mu = clsD[e]
        SIG["cos_mu"][i] = np.einsum("ij,ij->i", h, mu[cls]) / (hnn * (np.linalg.norm(mu[cls], axis=1) + 1e-30))
        SIG["cos_W"][i] = np.einsum("ij,ij->i", h, W[cls]) / (hnn * Wn[cls])
        SIG["cos_Wm"][i] = (h @ Wm) / (hnn * Wmn)
        Zh = h @ W.T + b[None, :]
        zc2 = Zh[np.arange(n_gt), cls]
        Zh[np.arange(n_gt), cls] = -np.inf
        SIG["margin"][i] = zc2 - Zh.max(1)
    print("[integrity] margin recompute == z_gt:", float(np.abs(SIG['margin'] - 0).max()) > 0)

    dead = sg < DEAD_THR
    t_dead = np.array([eps[np.where(dead[:, j])[0][0]] if dead[:, j].any() else np.nan for j in range(n_gt)])

    base["t_dead"] = t_dead
    base["init_dead"] = dead[0]
    is_f0 = (base["failure"] == "F0_no_raw_candidate").values
    is_mat = (base["failure"] == "MATCHED@0.5").values
    tr_mask = is_f0 & ~dead[0] & np.isfinite(t_dead)
    idx_tr = np.where(tr_mask)[0]
    idx_mat = np.where(is_mat)[0]
    print(f"\n[cohort] F0={is_f0.sum()} MATCHED={is_mat.sum()} | F0 transition(not init-dead)={len(idx_tr)}")

    # ---------- onset (2 consecutive strictly worsening observations) ---------
    def onsets(arr, need=2):
        """arr (n_ep,n_gt) -> first index i with `need` consecutive strict decreases."""
        out = np.full(arr.shape[1], -1)
        for j in range(arr.shape[1]):
            x = arr[:, j]
            for i in range(len(x) - need):
                seg = x[i:i + need + 1]
                if np.all(np.isfinite(seg)) and np.all(np.diff(seg) < 0):
                    out[j] = i
                    break
        return out

    ON = {k: onsets(v) for k, v in SIG.items()}
    ON_EP = {k: np.where(v >= 0, eps[np.clip(v, 0, n_ep - 1)], np.nan) for k, v in ON.items()}

    # ---------- §7 decisive: onset rate F0-transition vs MATCHED -------------
    print("\n[§7 CONTROL] onset rate over full trajectory (2 consecutive worsening obs)")
    print(f"{'signal':10s} {'F0-tr n':>8} {'F0-tr %':>8} {'MATCHED n':>10} {'MATCHED %':>10} {'rate ratio':>11}")
    rows = []
    for k in SIG:
        a = ON[k][idx_tr] >= 0; b = ON[k][idx_mat] >= 0
        rows.append(dict(signal=k, f0_n=int(a.sum()), f0_pct=100 * a.mean(),
                         mat_n=int(b.sum()), mat_pct=100 * b.mean(),
                         ratio=(a.mean() / b.mean()) if b.mean() > 0 else np.inf,
                         med_onset_ep=float(np.nanmedian(ON_EP[k][idx_tr])) if a.any() else np.nan))
    ontab = pd.DataFrame(rows)
    print(ontab.round(2).to_string(index=False))

    # ---------- §7b onset BEFORE death vs matched pseudo-event ---------------
    print("\n[§7b PRECEDENCE] onset relative to the GT's own t_dead (F0) vs MATCHED pseudo-event")
    rng = np.random.default_rng(0)
    # matched control: same class, nearest area
    ctrl = {}
    for j in idx_tr:
        cand = idx_mat[(base["cls"].values[idx_mat] == base["cls"].values[j])]
        if len(cand) == 0:
            cand = idx_mat
        d = np.abs(np.log10(base["area"].values[cand] / base["area"].values[j] + 1e-9))
        ctrl[j] = cand[np.argmin(d)]
    ctrl_arr = np.array([ctrl[j] for j in idx_tr])

    rows2 = []
    for k in SIG:
        on_f = ON_EP[k][idx_tr]
        td_f = base["t_dead"].values[idx_tr]
        pre_f = np.isfinite(on_f) & (on_f < td_f)
        on_c = ON_EP[k][ctrl_arr]
        td_c = td_f  # pseudo event time from the matched F0
        pre_c = np.isfinite(on_c) & (on_c < td_c)
        rows2.append(dict(signal=k,
                          f0_pre=100 * pre_f.mean(), mat_pre=100 * pre_c.mean(),
                          delta_pct=100 * (pre_f.mean() - pre_c.mean()),
                          f0_med_rel=float(np.nanmedian(td_f - on_f)) if pre_f.any() else np.nan,
                          mat_med_rel=float(np.nanmedian(td_c - on_c)) if np.isfinite(on_c).any() else np.nan))
    ptab = pd.DataFrame(rows2).sort_values("delta_pct", ascending=False)
    print(f"  (matched pairs n={len(idx_tr)}; control = same class, nearest log-area)")
    print(ptab.round(2).to_string(index=False))

    # ---------- §12 transition type (within-GT precedence) -------------------
    feat = np.where(np.isfinite(ON_EP["cos_mu"][idx_tr]) & np.isfinite(ON_EP["z_gt"][idx_tr]),
                    ON_EP["cos_mu"][idx_tr] - ON_EP["z_gt"][idx_tr], np.nan)
    iouo = np.where(np.isfinite(ON_EP["best_IoU"][idx_tr]) & np.isfinite(ON_EP["z_gt"][idx_tr]),
                    ON_EP["best_IoU"][idx_tr] - ON_EP["z_gt"][idx_tr], np.nan)
    both = np.isfinite(feat) & np.isfinite(iouo)
    typ = np.where(~both, "C-mixed/unresolved",
                   np.where((feat < 0) & (iouo >= 0), "A-feature-led",
                            np.where((feat >= 0) & (iouo < 0), "IoU-led",
                                     np.where((feat < 0) & (iouo < 0), "A+IoU both prior", "B-score-led"))))
    print("\n[§12] transition type (feature=cos_mu, score=z_gt, iou=best_IoU; onset epochs)")
    print(pd.Series(typ).value_counts().to_string())

    # ---------- §9 collapse shape -------------------------------------------
    shapes = []
    for j in idx_tr:
        ti = int(np.where(eps == base["t_dead"].values[j])[0][0])
        if ti < 2:
            shapes.append("too-early"); continue
        z = zc[:, j]
        drop = z[ti] - z[ti - 1]
        prior = np.ptp(z[max(0, ti - 3):ti])
        shapes.append("abrupt" if abs(drop) > 3 * (prior + 1e-6) else "gradual")
    print("\n[§9] collapse shape (transition F0)")
    print(pd.Series(shapes).value_counts().to_string())

    # ---------- §11 size / class / spatial -----------------------------------
    print("\n[§11] t_dead by size (F0 transition only)")
    t0 = base.iloc[idx_tr]
    print(t0.groupby("size")["t_dead"].agg(["size", "median", "min", "max"]).to_string())
    print("\n[§11] t_dead by border/non-border (F0 transition only)")
    print(t0.groupby("border")["t_dead"].agg(["size", "median"]).to_string())
    print("\n[§11] class-level (all probe GT)")
    cc = base.groupby("cls_name").agg(n=("is_f0" if False else "key", "size"))
    cc["f0"] = base.groupby("cls_name")["t_dead"].apply(lambda s: np.isfinite(s).sum())
    cc = base.assign(f0=is_f0).groupby("cls_name").agg(n=("f0", "size"), f0=("f0", "sum"),
                                                       f0_rate=("f0", "mean"))
    print(cc.round(3).sort_values("f0_rate", ascending=False).to_string())

    # ---------- §10 within-GT normalized (F0-tr vs MATCHED) ------------------
    print("\n[§10] within-GT normalized Δ (median), F0-transition minus MATCHED")
    rows3 = []
    for k in ("z_gt", "margin", "cos_mu", "cos_W", "best_IoU", "h_norm"):
        d = SIG[k] - SIG[k][0][None, :]
        for i, e in enumerate(eps):
            if e not in (0, 10, 20, 50, 100, 150, 200, 290):
                continue
            a = d[i][idx_tr]; b = d[i][idx_mat]
            rows3.append(dict(metric=k, epoch=e, f0=float(np.nanmedian(a)), mat=float(np.nanmedian(b)),
                              diff=float(np.nanmedian(a) - np.nanmedian(b))))
    p3t = pd.DataFrame(rows3).pivot_table(index="epoch", columns="metric", values="diff")
    print(p3t.round(3).to_string())

    # ---------- persist -------------------------------------------------------
    base.to_csv(OUT / "p11_v2_gt.csv", index=False, encoding="utf-8")
    np.savez_compressed(OUT / "p11_v2_series.npz", epochs=eps, keys=keys, **SIG,
                        **{f"onset_{k}": v for k, v in ON_EP.items()})
    json.dump(dict(epochs=[int(x) for x in eps], logit_recon_err=err, order_ok=bool(order_ok),
                   n_f0=int(is_f0.sum()), n_matched=int(is_mat.sum()), n_transition=int(len(idx_tr)),
                   onset_control=ontab.to_dict("records"), precedence=ptab.to_dict("records"),
                   types=pd.Series(typ).value_counts().to_dict(),
                   shapes=pd.Series(shapes).value_counts().to_dict(),
                   t_dead_size=t0.groupby("size")["t_dead"].median().to_dict(),
                   t_dead_border=t0.groupby("border")["t_dead"].median().to_dict()),
              open(OUT / "p11_v2_summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[done] -> {OUT}")


main()
