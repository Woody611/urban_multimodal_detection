"""_v11_analyze.py — layer 9→23 trajectory + FIRST STABLE DIVERGENCE。只读。"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LAYERS = list(range(9, 24))
MODNAME = {9: "C3k2(P2 bkbn,s4)", 10: "Conv(s8)", 11: "C3k2(s8)", 12: "Conv(s16)",
           13: "C3k2(s16)", 14: "Conv(s32)", 15: "C3k2(s32)", 16: "SPPF(s32)",
           17: "C2PSA(s32)", 18: "Upsample(s16)", 19: "Concat(s16)", 20: "C3k2(s16)",
           21: "Upsample(s8)", 22: "Concat(s8)", 23: "C3k2(P3 neck,s8)"}
# 结构判据：backbone = 9..17（下采样+SPPF+C2PSA），neck = 18..23（FPN top-down）
BACKBONE, NECK = list(range(10, 18)), list(range(18, 24))


def fit(X, y, lam=1.0, it=12):
    n, d = X.shape
    Xa = np.hstack([X, np.ones((n, 1))]); th = np.zeros(d + 1)
    for _ in range(it):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xa @ th, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-6)
        H = Xa.T @ (Xa * W[:, None]) + lam * np.eye(d + 1)
        try:
            s = np.linalg.solve(H, Xa.T @ (p - y) + lam * th)
        except np.linalg.LinAlgError:
            break
        th -= s
        if np.abs(s).max() < 1e-8:
            break
    return th[:d], float(th[d])


def auc(y, s):
    o = np.argsort(s); y = y[o]; n1 = int(y.sum()); n0 = len(y) - n1
    r = np.arange(1, len(y) + 1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cvev(X, y, f=5, seed=0):
    rng = np.random.default_rng(seed); P = []
    for c in (0, 1):
        ci = np.where(y == c)[0].copy(); rng.shuffle(ci); P.append(np.array_split(ci, f))
    A, U = [], []
    for k in range(f):
        te = np.r_[P[0][k], P[1][k]]; tr = np.setdiff1d(np.arange(len(y)), te)
        w, b = fit(X[tr], y[tr]); s = X[te] @ w + b; pr = (s > 0).astype(int)
        A.append(np.mean([np.mean(pr[y[te] == c] == c) for c in (0, 1)])); U.append(auc(y[te], s))
    return float(np.mean(A)), float(np.mean(U))


def cd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (sum((x > b).sum() for x in a) - sum((x < b).sum() for x in a)) / (len(a) * len(b))


def main():
    Z = np.load(OUT / "_v11_layers.npz")
    ids = json.loads((OUT / "_v11_ids.json").read_text(encoding="utf-8"))

    # ---------------- §9 INTEGRITY ----------------
    print("=" * 100)
    print("§9 INTEGRITY CHECK")
    print("=" * 100)
    v10 = {(r["domain"], r["key"]): r for r in
           json.loads((OUT / "_v10_stages.json").read_text(encoding="utf-8"))}
    ok = True
    for g, dom in (("G1", "V1"), ("G4", "V1"), ("TS", "T1")):
        X = Z[f"L23_{g}"]
        assert X.ndim == 2, f"{g} L23 不是 2D"
        nrm = np.linalg.norm(X, axis=1)
        assert nrm.shape[0] == X.shape[0], f"{g} norm 形状不符（axis bug）"
        diffs, coss, miss = [], [], 0
        for r, key in zip(X, ids[g]):
            ref = v10.get((dom, key))
            if ref is None:
                miss += 1; continue
            a = np.array(ref["P3"], np.float32)
            diffs.append(float(np.abs(r - a).max()))
            coss.append(float(r @ a / (np.linalg.norm(r) * np.linalg.norm(a) + 1e-12)))
        print(f"  {g:<4} L23 vs V10-P3 : n={len(X)} 可匹配={len(diffs)} 未匹配={miss} "
              f" max|Δ|={max(diffs) if diffs else float('nan'):.3e}  "
              f" cos={min(coss) if coss else float('nan'):.6f}")
        ok &= (max(diffs) < 1e-5) and (min(coss) > 0.99999)
    print(f"\n  P3 一致性: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  ⇒ STOP：P3 对不上，不解释 trajectory"); return
    # 抽查 3 个样本
    print("\n  抽查（G1 前 3 个）：")
    for i in range(3):
        print(f"    {ids['G1'][i]:<26} L9 shape={Z['L9_G1'][i].shape} |v|={np.linalg.norm(Z['L9_G1'][i]):.3f}"
              f" | L23 |v|={np.linalg.norm(Z['L23_G1'][i]):.3f}")

    # ---------------- §7 trajectory ----------------
    print("\n" + "=" * 100)
    print("§7 逐层 trajectory（layer 9 → 23）")
    print("=" * 100)
    hdr = (f"{'lyr':>4}{'module':<20}{'G1|v|':>9}{'G4|v|':>9}{'TS|v|':>9}"
           f"{'G1cos':>8}{'G4cos':>8}{'kNN(G1/TS)':>11}{'kNN(G4/TS)':>11}"
           f"{'AUC(G1/TS)':>12}{'AUC(G1/G4)':>12}{'Cliff(G1/TS)':>13}")
    print(hdr)
    T = []
    for li in LAYERS:
        X1, X4, XT = Z[f"L{li}_G1"], Z[f"L{li}_G4"], Z[f"L{li}_TS"]
        A = np.vstack([X1, X4, XT]).astype(np.float64)
        mu, sd = A.mean(0), A.std(0) + 1e-8
        Z1, Z4, ZT = ((X - mu) / sd for X in (X1.astype(np.float64), X4.astype(np.float64),
                                              XT.astype(np.float64)))
        mT = ZT.mean(0); mT /= np.linalg.norm(mT)
        c1 = float(np.median(Z1 @ mT / (np.linalg.norm(Z1, axis=1) + 1e-12)))
        c4 = float(np.median(Z4 @ mT / (np.linalg.norm(Z4, axis=1) + 1e-12)))
        uT = ZT / (np.linalg.norm(ZT, axis=1, keepdims=True) + 1e-12)
        kG1 = np.linalg.norm((Z1 / (np.linalg.norm(Z1, axis=1, keepdims=True) + 1e-12))[:, None, :]
                             - uT[None, :, :], axis=2)
        kG4 = np.linalg.norm((Z4 / (np.linalg.norm(Z4, axis=1, keepdims=True) + 1e-12))[:, None, :]
                             - uT[None, :, :], axis=2)
        a1, _ = cvev(np.vstack([Z1, ZT]), np.r_[np.ones(len(Z1)), np.zeros(len(ZT))])
        a4, _ = cvev(np.vstack([Z1, Z4]), np.r_[np.ones(len(Z1)), np.zeros(len(Z4))])
        w_, b_ = fit(np.vstack([Z1, ZT]), np.r_[np.ones(len(Z1)), np.zeros(len(ZT))])
        cld = cd(Z1 @ w_, ZT @ w_)
        T.append(dict(layer=li, auc_ts=a1, auc_g4=a4, cliff=cld, c1=c1,
                      n1=float(np.median(np.linalg.norm(X1, axis=1))),
                      n4=float(np.median(np.linalg.norm(X4, axis=1))),
                      nT=float(np.median(np.linalg.norm(XT, axis=1))),
                      knn1=float(np.median(kG1[:, :5].mean(1))),
                      knn4=float(np.median(kG4[:, :5].mean(1)))))
        print(f"{li:>4}{MODNAME[li]:<20}{T[-1]['n1']:>9.2f}{T[-1]['n4']:>9.2f}{T[-1]['nT']:>9.2f}"
              f"{c1:>8.3f}{c4:>8.3f}{T[-1]['knn1']:>11.4f}{T[-1]['knn4']:>11.4f}"
              f"{a1:>12.4f}{a4:>12.4f}{cld:>13.3f}")

    print("\n  AUC trajectory")
    print("   layer  " + "".join(f"{l:>8}" for l in LAYERS))
    print("   G1/TS  " + "".join(f"{t['auc_ts']:>8.4f}" for t in T))
    print("   G1/G4  " + "".join(f"{t['auc_g4']:>8.4f}" for t in T))

    # ---------------- §6/§8 FIRST STABLE DIVERGENCE ----------------
    print("\n" + "=" * 100)
    print("§6/§8 FIRST STABLE DIVERGENCE（相对 layer 9；需 AUC↑ + 独立指标同步 + 后续不回落到 baseline）")
    print("=" * 100)
    base = T[0]["auc_ts"]
    print(f"  baseline = layer 9 的 AUC(G1/TS) = {base:.4f}")
    print(f"  {'lyr':>4}{'AUC(G1/TS)':>12}{'Δ vs L9':>10}{'Δ(AUC G1/G4)':>14}{'norm 比 G1/TS':>15}"
          f"{'kNN(G1/TS)':>12}  判定")
    b4 = T[0]["auc_g4"]
    for i, t in enumerate(T):
        dA = t["auc_ts"] - base
        d4 = t["auc_g4"] - b4
        nr = t["n1"] / (t["nT"] + 1e-12)
        tag = ""
        if i == 0:
            tag = "NO_CLEAR_DIVERGENCE (baseline)"
        elif dA > 0.02 or abs(d4) > 0.05 or abs(nr - T[0]["n1"] / (T[0]["nT"] + 1e-12)) > 0.15:
            tag = "DIVERGENCE 起点候选"
        print(f"  {t['layer']:>4}{t['auc_ts']:>12.4f}{dA:>+10.4f}{d4:>+14.4f}{nr:>15.3f}"
              f"{t['knn1']:>12.4f}  {tag}")

    print("\n  Backbone(10–17) vs Neck(18–23) 的 AUC(G1/TS) 变化：")
    bb = [t["auc_ts"] for t in T if t["layer"] in BACKBONE]
    nk = [t["auc_ts"] for t in T if t["layer"] in NECK]
    print(f"    backbone 段: min={min(bb):.4f} max={max(bb):.4f} 跨度={max(bb)-min(bb):.4f}")
    print(f"    neck     段: min={min(nk):.4f} max={max(nk):.4f} 跨度={max(nk)-min(nk):.4f}")
    print(f"    backbone 段内首达 >0.80 的层: "
          f"{[t['layer'] for t in T if t['layer'] in BACKBONE and t['auc_ts']>0.80]}")
    print(f"    neck 段内首达 >0.80 的层: "
          f"{[t['layer'] for t in T if t['layer'] in NECK and t['auc_ts']>0.80]}")
    print(f"    L9 = {T[0]['auc_ts']:.4f}（== backbone 起点，stride 4 backbone P2）")

    print("\n  norm 比 (G1/TS) 轨迹：")
    print("   layer  " + "".join(f"{l:>8}" for l in LAYERS))
    print("   G1/TS  " + "".join(f"{t['n1']/(t['nT']+1e-12):>8.3f}" for t in T))

    (OUT / "_v11_traj.json").write_text(json.dumps(T, indent=2), encoding="utf-8")
    print(f"\n[saved] {OUT/'_v11_traj.json'}")


if __name__ == "__main__":
    main()
