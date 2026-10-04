"""p10_stage2.py — P10 stage 2: EMA filter weights, momentum persistence, gradient-parameter alignment.

只读。torch.load 已有 epoch*.pt，做 CPU 代数。无 forward/backward/训练/推理。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DECAY, TAU = 0.9999, 2000
NC, EPOCHS, LRF = 12, 300, 0.01


def d_of(t):
    return DECAY * (1 - math.exp(-t / TAU))


def ema_weights(N):
    """weights w_k (k=0..N) of the recursion e_t = d_t e_{t-1} + (1-d_t) w_t, e_0 = w_init.
    w_0 = prod_{j=1..N} d_j ; w_k = (1-d_k) * prod_{j=k+1..N} d_j."""
    d = [d_of(j) for j in range(1, N + 1)]
    w = np.zeros(N + 1)
    w[0] = float(np.prod(d))
    acc = 1.0
    for k in range(N, 0, -1):  # build from the end
        w[k] = (1 - d[k - 1]) * acc
        acc *= d[k - 1]
    return w, d


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return float("nan")
    return float((a @ b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--out", default="diagnostic/p10_out")
    A = ap.parse_args()
    import torch

    wdir = (ROOT / A.weights_dir).resolve()
    out = (ROOT / A.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    snaps = []
    for p in sorted(wdir.glob("epoch*.pt")):
        m = re.search(r"epoch(\d+)", p.stem)
        if m:
            snaps.append((int(m.group(1)), p))
    snaps.sort()

    # ---- 1. EMA filter weights over epoch 0 --------------------------------
    up0 = 109
    w, d = ema_weights(up0)
    init_frac = float(w[0])
    late = float(w[up0 - 20:].sum())
    print(f"[ema-filter] N={up0} updates (epoch 0)")
    print(f"  weight on INIT (k=0)          = {init_frac:.4f}")
    print(f"  weight on last 20 updates     = {late:.4f}")
    print(f"  d_1={d[0]:.6f}  d_N={d[-1]:.6f}  effective memory at N = {1/(1-d[-1]):.1f} updates")

    # full-training filter: what fraction of e_290 is raw from before ep250?
    tot = 7384
    wf, _ = ema_weights(tot)
    print(f"[ema-filter] N={tot}: weight on updates older than the last 250 (10 epochs) "
          f"= {wf[:tot-250].sum():.4f}  (=> EMA memory ~ {1/(1-d_of(tot)):.1f} updates)")

    # ---- 2. simulate the exact filter on a LINEAR raw ramp ------------------
    # scalar proxy: rho(t) = ||W_mean|| with rho(t) = rho0 + slope*t  (t = update index)
    nb_per_epoch = 25.0
    ups = np.round(np.arange(0, 291, 10) * nb_per_epoch).astype(int) + 109  # approx, ep0=109
    ups[0] = 109
    # rebuild exact update counts from stage-1 json
    s1 = json.load(open(out / "p10_trajectory.json", encoding="utf-8"))
    ups = np.array([r["updates"] for r in s1])
    eps = np.array([r["epoch"] for r in s1])
    obs = np.array([r["W_mean_norm"] for r in s1])

    N = int(ups[-1])
    e = np.zeros(N + 1)          # EMA scalar
    raw = np.zeros(N + 1)        # raw scalar (=its own value for a linear ramp)
    raw[:] = np.linspace(0.0, 1.0, N + 1)
    e[0] = raw[0]
    dd = np.array([1.0] + [d_of(j) for j in range(1, N + 1)])
    for t in range(1, N + 1):
        e[t] = dd[t] * e[t - 1] + (1 - dd[t]) * raw[t]
    e_at = np.array([e[u] for u in ups])
    de = np.abs(np.diff(e_at))
    cum = np.concatenate([[0.0], np.cumsum(de)])
    frac_sim = cum / cum[-1]
    dRaw = np.abs(np.diff(raw[ups]))
    cumR = np.concatenate([[0.0], np.cumsum(dRaw)])
    frac_raw = cumR / cumR[-1]

    obsv = np.abs(np.diff(obs))
    obsv = np.concatenate([[0.0], obsv]); cumO = np.cumsum(obsv)
    obsv = np.concatenate([[0.0], obsv])
    cumO = np.cumsum(obsv)
    # fix: first element handling
    dObs = np.abs(np.diff(obs)); cumO = np.concatenate([[0.0], np.cumsum(dObs)])
    frac_obs = cumO / cumO[-1]

    print("\n[filter-simulation] cumulative fraction of movement completed by epoch e")
    print("  (LINEAR raw ramp = raw itself is by construction linear => 2.0% at ep10)")
    print("   ep   raw(linear)   EMA-of-linear   OBSERVED")
    sim = []
    for i, ep in enumerate(eps):
        sim.append(dict(epoch=int(ep), frac_raw=float(frac_raw[i]), frac_ema_of_linear=float(frac_sim[i]),
                        frac_observed=float(frac_obs[i])))
        if ep in (10, 20, 30, 50, 100, 150, 200, 290):
            print(f"  {ep:4d}   {frac_raw[i]*100:9.2f}%   {frac_sim[i]*100:11.2f}%   {frac_obs[i]*100:8.2f}%")
    (out / "p10_filter_sim.json").write_text(json.dumps(sim, indent=2), encoding="utf-8")

    # ---- 3. momentum / parameter alignment per branch ----------------------
    rows = []
    prev = None
    for ep, path in snaps:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ema_mod = ck["ema"]
        sd = ema_mod.state_dict()
        opt = ck["optimizer"]
        import torch.nn as nn
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        g = [[], [], []]
        for mn, mo in ema_mod.named_modules():
            for pn, _p in mo.named_parameters(recurse=False):
                fn = f"{mn}.{pn}" if mn else pn
                (g[2] if "bias" in fn else (g[1] if isinstance(mo, bn) else g[0])).append(fn)
        idx2name = {i: n for i, n in enumerate(g[2] + g[0] + g[1])}
        st = opt["state"]

        rec = {"epoch": ep}
        for wk in sorted([k for k in sd if re.search(r"\.cv3\.\d+\.\d+\.weight$", k)]):
            bk = wk[:-6] + "bias"
            W = sd[wk].float().numpy().astype(np.float64).reshape(sd[wk].shape[0], -1)
            b = sd[bk].float().numpy().astype(np.float64).reshape(-1)
            gi = [i for i, n in idx2name.items() if n == wk][0]
            mb = st[gi]["momentum_buffer"].float().numpy().astype(np.float64).reshape(W.shape)
            Wm, Wc = W.mean(0), W - W.mean(0)
            Mm, Mc = mb.mean(0), mb - mb.mean(0)
            rec[f"{wk}|Wmean"] = float(np.linalg.norm(Wm))
            rec[f"{wk}|Wcent"] = float(np.linalg.norm(Wc))
            rec[f"{wk}|Mmean"] = float(np.linalg.norm(Mm))
            rec[f"{wk}|Mcent"] = float(np.linalg.norm(Mc))
            rec[f"{wk}|bmean"] = float(b.mean())
            rec[f"{wk}|bstd"] = float(b.std())
            rec[f"{wk}|cosMw_Wm"] = cos(Mm, Wm)
            rec[f"{wk}|cosMc_Wc"] = cos(Mc.ravel(), Wc.ravel())
            rec[f"{wk}|ratio_common_vs_dev"] = float(
                np.linalg.norm(Mm) / (np.linalg.norm(Mc) / math.sqrt(W.shape[0]) + 1e-30))
            if prev is not None:
                rec[f"{wk}|persist_Mmean"] = cos(Mm, prev[f"{wk}|Mm_vec"])
                rec[f"{wk}|persist_Mcent"] = cos(Mc.ravel(), prev[f"{wk}|Mc_vec"])
                rec[f"{wk}|rotate_Wmean"] = cos(Wm, prev[f"{wk}|Wm_vec"])
            rec[f"{wk}|Mm_vec"] = Mm
            rec[f"{wk}|Mc_vec"] = Mc.ravel()
            rec[f"{wk}|Wm_vec"] = Wm
        rows.append(rec)
        prev = rec
        del ck, ema_mod, sd
        print(f"  loaded ep{ep}")

    keys = [k for k in rows[0] if k.endswith("|Wmean")]
    prim = keys[0].split("|")[0]
    print(f"\n[primary] {prim}")
    hdr = f"{'ep':>4} {'|Wmean|':>8} {'|Wcent|':>8} {'|Mmean|':>8} {'|Mcent|':>8} {'b_mean':>8} {'b_std':>7} " \
          f"{'cos(Mm,Wm)':>10} {'cos(Mc,Wc)':>10} {'Mcom/Mdev':>9} {'persMm':>7} {'persMc':>7} {'rotWm':>7}"
    print(hdr)
    for r in rows:
        print(f"{r['epoch']:>4} {r[prim+'|Wmean']:8.4f} {r[prim+'|Wcent']:8.4f} {r[prim+'|Mmean']:8.4f} "
              f"{r[prim+'|Mcent']:8.4f} {r[prim+'|bmean']:+8.4f} {r[prim+'|bstd']:7.4f} "
              f"{r[prim+'|cosMw_Wm']:+10.4f} {r[prim+'|cosMc_Wc']:+10.4f} "
              f"{r[prim+'|ratio_common_vs_dev']:9.3f} "
              f"{r.get(prim+'|persist_Mmean', float('nan')):+7.3f} "
              f"{r.get(prim+'|persist_Mcent', float('nan')):+7.3f} "
              f"{r.get(prim+'|rotate_Wmean', float('nan')):+7.3f}")

    print("\n[all branches] epoch, Wmean, Wcent, b_mean  (theoretical init bias below)")
    th = {8: math.log(5 / NC / (640 / 8) ** 2), 16: math.log(5 / NC / (640 / 16) ** 2),
          32: math.log(5 / NC / (640 / 32) ** 2)}
    print("  theoretical bias init:", {k: round(v, 4) for k, v in th.items()})
    for r in rows:
        for wk in keys:
            pass
    for i, wk in enumerate([k[: -len("|Wmean")] for k in keys]):
        s = int(wk.split(".cv3.")[1].split(".")[0])
        stride = 8 * 2 ** s
        vals = [r[f"{wk}|bmean"] for r in rows]
        print(f"  cv3.{s} stride{stride:>3}: b_mean ep0={vals[0]:+.4f} ep290={vals[-1]:+.4f} "
              f"| init={th[stride]:+.4f} | |Wmean| {rows[0][wk+'|Wmean']:.4f}->{rows[-1][wk+'|Wmean']:.4f} "
              f"| |Wcent| {rows[0][wk+'|Wcent']:.4f}->{rows[-1][wk+'|Wcent']:.4f}")

    clean = []
    for r in rows:
        clean.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()})
    (out / "p10_stage2.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    print(f"\n[done] -> {out}/p10_stage2.json")


if __name__ == "__main__":
    main()
