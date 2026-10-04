#!/usr/bin/env python
"""p5_2_wt_dynamics_audit.py — P5.2: W_t 分类头动力学的上游机制归因（READ-ONLY / ZERO-GPU / ZERO-FORWARD）。

只在已通过 P5.1 验证的 30 个 snapshot（正确 head_W）上做代数分析：
  §0 artifact gate + P5.1 复现  →  §2 row-wise dynamics  →  §3 common-mode 分解
  →  §4 GT-row vs others  →  §5 SVD/low-rank  →  §6 temporal alignment
  →  §7 class-frequency（outcome-independent）  →  §8 机制判定

用法:  python -X utf8 diagnostic/p5_2_wt_dynamics_audit.py [--dir ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

NM = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
      6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def eff_rank(s, tol=1e-6):
    s = s[s > tol * s[0]]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "diagnostic/full300_trajectory_probe/from_snapshots"))
    A = ap.parse_args()
    S = Path(A.dir)
    files = sorted(S.glob("epoch_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    E = [int(p.stem.split("_")[1]) for p in files]
    print(f"[p5.2] {len(files)} snapshots: {E[0]} … {E[-1]}\n")

    # ================= §0 ARTIFACT GATE =================
    print("=== §0 Artifact Gate ===")
    Z = {}
    for p, e in zip(files, E):
        z = np.load(p, allow_pickle=True)
        for k in ("head_W", "head_b", "vec_h", "vec_keys"):
            if k not in z:
                print(f"STOP — ARTIFACT INVALID: ep{e} 缺 {k}"); return 1
        W, b, H = z["head_W"], z["head_b"], z["vec_h"]
        if W.shape != (12, 256) or b.shape != (12,) or H.shape[1] != 256:
            print(f"STOP — ARTIFACT INVALID: ep{e} shape {W.shape}/{b.shape}/{H.shape}"); return 1
        if not (np.isfinite(W).all() and np.isfinite(b).all() and np.isfinite(H).all()):
            print(f"STOP — ARTIFACT INVALID: ep{e} NaN/Inf"); return 1
        Z[e] = z
    if E != sorted(E) or E[0] != 0:
        print(f"STOP — epoch 顺序/起点异常: {E[:3]}"); return 1
    uw = len({hash(z["head_W"].tobytes()) for z in Z.values()})
    keys = list(Z[E[0]]["vec_keys"])
    if any(list(Z[e]["vec_keys"]) != keys for e in E):
        print("STOP — vec_keys 跨 epoch 不一致"); return 1
    print(f"  epoch 数={len(E)}  W hash 唯一={uw}/{len(E)}  vec_keys={len(keys)} 一致 ✓")
    if uw == 1:
        print("STOP — W 恒定（stale）"); return 1

    cid = Z[E[0]]["vec_cls"].astype(int)
    cells = ROOT / "diagnostic/p3_trajectory_probe/medium_cells_canvas.json"
    if cells.exists():
        rows = json.loads(cells.read_text(encoding="utf-8"))["rows"]
        if [r["key"] for r in rows] == keys:
            print("  与 medium_cells_canvas.json key 逐位一致 ✓")
    W0 = Z[0]["head_W"].astype(np.float64); b0 = Z[0]["head_b"].astype(np.float64)
    Wf = W0; bf = b0
    bp = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"
    if bp.exists():
        import torch
        ck = torch.load(bp, map_location="cpu", weights_only=False)
        m = ck.get("ema") or ck["model"]
        last = m.model[-1].cv3[0][-1]
        Wf = last.weight.detach().float().view(12, -1).numpy().astype(np.float64)
        bf = last.bias.detach().float().numpy().astype(np.float64)
    # P5.1 复现校验
    def gtlg(W, b, H):
        return np.einsum("ij,ij->i", H, W[cid]) + b[cid]
    eL = E[-1]; Ht = Z[eL]["vec_h"].astype(np.float64); H0 = Z[0]["vec_h"].astype(np.float64)
    Wt, bt = Z[eL]["head_W"].astype(np.float64), Z[eL]["head_b"].astype(np.float64)
    R = 0.5 * ((gtlg(W0, b0, Ht) - gtlg(W0, b0, H0)) + (gtlg(Wt, bt, Ht) - gtlg(Wt, bt, H0)))
    Hh = 0.5 * ((gtlg(Wt, bt, H0) - gtlg(W0, b0, H0)) + (gtlg(Wt, bt, Ht) - gtlg(W0, b0, Ht)))
    T = gtlg(Wt, bt, Ht) - gtlg(W0, b0, H0)
    err = float(np.abs(R + Hh - T).max())
    print(f"  P5.1 复现: max|R+H−T| = {err:.2e}  median R={np.median(R):+.3f} H={np.median(Hh):+.3f}  "
          f"{'✓' if err < 1e-6 else '✗ STOP'}")
    if err >= 1e-6:
        print("STOP — ARTIFACT INVALID: 无法复现 P5.1"); return 1

    out = S.parent / "p5_2_out"; out.mkdir(parents=True, exist_ok=True)

    # ================= §2 row-wise =================
    print("\n=== §2 W_t row-wise dynamics ===")
    rows2 = []
    for e in E:
        W = Z[e]["head_W"].astype(np.float64)
        for c in range(12):
            rows2.append(dict(epoch=e, cls=c, name=NM[c], norm=float(np.linalg.norm(W[c])),
                              dnorm_vs0=float(np.linalg.norm(W[c]) - np.linalg.norm(W0[c])),
                              cos_vs0=cos(W[c], W0[c]), cos_vsF=cos(W[c], Wf[c]),
                              l2_vs0=float(np.linalg.norm(W[c] - W0[c])),
                              l2_vsF=float(np.linalg.norm(W[c] - Wf[c]))))
    print(f"{'cls':14s} {'‖W0‖':>7} {'‖W290‖':>7} {'Δ‖·‖':>7} {'cos(W290,W0)':>13} {'‖Δ‖':>7}")
    for c in range(12):
        r0 = [x for x in rows2 if x["epoch"] == 0 and x["cls"] == c][0]
        r1 = [x for x in rows2 if x["epoch"] == eL and x["cls"] == c][0]
        print(f"{NM[c]:14s} {r0['norm']:>7.4f} {r1['norm']:>7.4f} {r1['dnorm_vs0']:>+7.4f} "
              f"{r1['cos_vs0']:>13.5f} {r1['l2_vs0']:>7.4f}")

    # ================= §3 common-mode =================
    print("\n=== §3 W common-mode ===")
    Wm = {e: Z[e]["head_W"].astype(np.float64).mean(0) for e in E}
    bm = {e: float(Z[e]["head_b"].astype(np.float64).mean()) for e in E}
    print(f"{'ep':>5} {'‖W_mean‖':>9} {'cos(vs W0)':>11} {'b_mean':>9} {'‖W_mean−W0m‖':>13}")
    for e in E:
        print(f"{e:>5} {np.linalg.norm(Wm[e]):>9.4f} {cos(Wm[e], Wm[0]):>11.5f} {bm[e]:>9.4f} "
              f"{np.linalg.norm(Wm[e]-Wm[0]):>13.4f}")
    # 12 类均值 logit 的 Shapley 分解
    def C(W, b, H):
        return H @ W.mean(0) + b.mean()
    Rc = {e: 0.5 * ((C(W0, b0, Z[e]["vec_h"].astype(np.float64)) - C(W0, b0, H0))
                    + (C(Z[e]["head_W"].astype(np.float64), Z[e]["head_b"].astype(np.float64), Z[e]["vec_h"].astype(np.float64))
                       - C(Z[e]["head_W"].astype(np.float64), Z[e]["head_b"].astype(np.float64), H0))) for e in E}
    Hc_ = {e: 0.5 * ((C(Z[e]["head_W"].astype(np.float64), Z[e]["head_b"].astype(np.float64), H0) - C(W0, b0, H0))
                     + (C(Z[e]["head_W"].astype(np.float64), Z[e]["head_b"].astype(np.float64), Z[e]["vec_h"].astype(np.float64))
                        - C(W0, b0, Z[e]["vec_h"].astype(np.float64)))) for e in E}
    Tc = {e: C(Z[e]["head_W"].astype(np.float64), Z[e]["head_b"].astype(np.float64), Z[e]["vec_h"].astype(np.float64))
          - C(W0, b0, H0) for e in E}
    print(f"\n  12类均值 logit 的 Shapley（末点）: Rc={np.median(Rc[eL]):+.3f}  "
          f"Hc={np.median(Hc_[eL]):+.3f}  Tc={np.median(Tc[eL]):+.3f}  "
          f"恒等式 err={max(float(np.abs(Rc[e]+Hc_[e]-Tc[e]).max()) for e in E):.2e}")

    # ================= §4 GT row vs others =================
    print("\n=== §4 GT-row vs other rows（末点） ===")
    Wl = Z[eL]["head_W"].astype(np.float64)
    go, oo = [], []
    for i in range(len(cid)):
        c = cid[i]
        oth = np.delete(np.arange(12), c)
        go.append(np.linalg.norm(Wl[c]) / np.mean([np.linalg.norm(Wl[j]) for j in oth]))
        oo.append(np.linalg.norm(Wl[c] - W0[c]))
    go = np.array(go); oo = np.array(oo)
    print(f"  ‖GT row‖ / mean(其他 row) : 中位={np.median(go):.4f}  p10={np.percentile(go,10):.4f} p90={np.percentile(go,90):.4f}")
    print(f"  ‖Δ GT row‖ (vs W0)        : 中位={np.median(oo):.4f}")

    # ================= §5 SVD =================
    print("\n=== §5 SVD / low-rank ===")
    print(f"{'ep':>5} {'spectral':>9} {'frobenius':>10} {'effrank':>8} {'top1%能量':>9} {'top3%能量':>9}")
    sv = {}
    for e in E:
        W = Z[e]["head_W"].astype(np.float64)
        s = np.linalg.svd(W, compute_uv=False)
        sv[e] = s
        en = s ** 2 / (s ** 2).sum()
        print(f"{e:>5} {s[0]:>9.4f} {np.linalg.norm(W):>10.4f} {eff_rank(s):>8.3f} {en[0]:>9.3%} {en[:3].sum():>9.3%}")
    chg = max(float(np.abs(sv[e] - sv[0]).max()) for e in E)
    print(f"  singular values 全程最大变化 = {chg:.4f}（若 ≪ 特征值量级 ⇒ 秩结构基本稳定）")

    # ================= §6 temporal alignment =================
    print("\n=== §6 Temporal alignment ===")
    def sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))
    tab = []
    print(f"{'ep':>5} {'‖ΔW‖':>7} {'‖ΔW_mean‖':>10} {'GT logit':>9} {'GT居中':>8} {'12类均值':>9} "
          f"{'live dead':>9} {'W0 dead':>8} {'WF dead':>8}")
    for e in E:
        W = Z[e]["head_W"].astype(np.float64); b = Z[e]["head_b"].astype(np.float64)
        L = Z[e]["vec_h"].astype(np.float64) @ W.T + b[None, :]
        g = L[np.arange(len(cid)), cid]
        dW = float(np.linalg.norm(W - W0))
        dm = float(np.linalg.norm(Wm[e] - Wm[0]))
        sL = sig(g)
        L0 = Z[e]["vec_h"].astype(np.float64) @ W0.T + b0[None, :]
        LF = Z[e]["vec_h"].astype(np.float64) @ Wf.T + bf[None, :]
        s0 = sig(L0[np.arange(len(cid)), cid]); sF = sig(LF[np.arange(len(cid)), cid])
        tab.append(dict(epoch=e, dW=dW, dWm=dm, gt_med=float(np.median(g)),
                        ctr=float(np.median(g - L.mean(1))), gm=float(np.median(L.mean(1))),
                        live_dead=float(np.mean(sL < 1e-3)), w0_dead=float(np.mean(s0 < 1e-3)),
                        wf_dead=float(np.mean(sF < 1e-3))))
        r = tab[-1]
        print(f"{e:>5} {r['dW']:>7.4f} {r['dWm']:>10.4f} {r['gt_med']:>9.3f} {r['ctr']:>8.2f} "
              f"{r['gm']:>9.2f} {r['live_dead']:>9.1%} {r['w0_dead']:>8.1%} {r['wf_dead']:>8.1%}")

    # ================= §7 class frequency（outcome-independent） =================
    print("\n=== §7 class frequency vs W row change（训练集 GT 计数，outcome-independent） ===")
    lab = ROOT / "data/processed/rgbid_split_train/labels/train/visible"
    freq = Counter()
    if lab.exists():
        for f in lab.glob("*.txt"):
            for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                ln = ln.strip()
                if ln:
                    try:
                        freq[int(float(ln.split()[0]))] += 1
                    except Exception:
                        pass
    tot = max(sum(freq.values()), 1)
    print(f"{'cls':14s} {'train GT':>9} {'占比':>7} {'Δ‖W‖':>8} {'cos(vs W0)':>11} {'H 中位':>8}")
    freq_rows = []
    for c in range(12):
        n = freq.get(c, 0)
        r0 = [x for x in rows2 if x["epoch"] == 0 and x["cls"] == c][0]
        r1 = [x for x in rows2 if x["epoch"] == eL and x["cls"] == c][0]
        hsub = Hh[cid == c]
        hm = float(np.median(hsub)) if len(hsub) else float("nan")
        freq_rows.append(dict(cls=c, name=NM[c], train_gt=n, frac=n / tot,
                              dnorm=r1["dnorm_vs0"], cos=r1["cos_vs0"], H_med=hm))
        print(f"{NM[c]:14s} {n:>9} {n/tot:>7.2%} {r1['dnorm_vs0']:>+8.4f} {r1['cos_vs0']:>11.5f} {hm:>+8.3f}")
    if freq:
        fr = np.array([x["frac"] for x in freq_rows if x["train_gt"] > 0])
        dn = np.array([x["dnorm"] for x in freq_rows if x["train_gt"] > 0])
        hm = np.array([x["H_med"] for x in freq_rows if x["train_gt"] > 0])
        if len(fr) > 3:
            print(f"  corr(freq, Δ‖W‖) = {np.corrcoef(fr, dn)[0,1]:+.3f}   corr(freq, H_med) = {np.corrcoef(fr, hm)[0,1]:+.3f}")

    # ================= 输出 =================
    for nm, rr in (("p5_2_row_dynamics.csv", rows2), ("p5_2_temporal.csv", tab),
                   ("p5_2_class_frequency.csv", freq_rows)):
        if rr:
            with open(out / nm, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rr[0].keys())); w.writeheader(); w.writerows(rr)
    print(f"\n[p5.2] CSV -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
