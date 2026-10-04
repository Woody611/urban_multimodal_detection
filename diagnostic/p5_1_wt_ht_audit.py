#!/usr/bin/env python
"""p5_1_wt_ht_audit.py — P5.1 离线 W_t × h_t 分解（ZERO GPU / ZERO forward / ZERO training）。

等待 `from_snapshots/epoch_*.npz`（带**正确** head_W/head_b 的快照）同步回来后**一次跑完**：
integrity gate → identity gate → 三读数（同一 GT-cell 口径）→ Shapley 分解（含恒等式自检）
→ dead-tail / event alignment → cohort → common-mode → GPU gate。

用法:
    python -X utf8 diagnostic/p5_1_wt_ht_audit.py --check-only     # 只做门禁，不出机制结论
    python -X utf8 diagnostic/p5_1_wt_ht_audit.py --run            # 完整分析

输出（写到 --dir 同级的 p5_1_out/）:
    p5_1_wt_ht_summary.csv / p5_1_wt_ht_per_gt.csv / p5_1_decomposition.csv
    p5_1_dead_entry.csv / p5_1_cohort_summary.csv / p5_1_report.md

⚠ 关于 §9 的公式：规格里给的表征项用 `W_final`，与它要求的 `R + H == T` 恒等式**不相容**。
   残差恰为  ½·(W_final − W_0)_gt ·(h_t − h_0)  （bias 项相消）。
   本脚本主分解用**可通过恒等式**的 Shapley 标准形式（表征项用 W_0 与 W_t），
   同时输出规格原文那一版及其残差，供对照。**不会因规格版不自洽而误 STOP。**
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DEAD_LO, DEAD_HI = 1e-3, 1e-1
DECOMP_TOL = 1e-6          # 恒等式允许误差（float64）
INT_TOL = 1e-9            # 分解恒等式的机器精度门槛


# ---------------------------------------------------------------- utils
def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _ep_of(p: Path) -> int:
    return int(p.stem.split("_")[1])


def gt_logit(W, b, H, cid):
    """F(W,h) 的 GT 类分量：W[cid]·h + b[cid]，h 为 (N,256)。"""
    return np.einsum("ij,ij->i", H, W[cid]) + b[cid]


def all_logits(W, b, H):
    return H @ W.T + b[None, :]


def sigma(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


# ---------------------------------------------------------------- geoms
def load_geometry(keys, cls_from_npz):
    """GT class / size / border / aspect / density / resolution —— 与 npz 解耦，来自 label。"""
    try:
        from official_eval import load_split
    except Exception:
        return None
    lab = ROOT / "data/processed/rgbid_split_train/labels/val/visible"
    img = ROOT / "data/processed/rgbid_split_train/images/val/visible"
    if not lab.exists():
        return None
    per_image, _ = load_split(img, lab)
    G = {}
    for _i, stem, w, h, gb, gc in per_image:
        if gb is None or not len(gb):
            continue
        for j in range(len(gb)):
            b = gb[j]
            G[f"{stem}#{j}"] = dict(
                cls=int(gc[j]), W=w, H=h,
                sqrt=float(np.sqrt(max(b[2] - b[0], 0) * max(b[3] - b[1], 0))),
                ar=float(max(b[2] - b[0], b[3] - b[1]) / max(min(b[2] - b[0], b[3] - b[1]), 1e-6)),
                border=float(min((b[0] + b[2]) / 2 / w, 1 - (b[0] + b[2]) / 2 / w,
                                 (b[1] + b[3]) / 2 / h, 1 - (b[1] + b[3]) / 2 / h)),
                clip=int(b[0] <= 0.5 or b[1] <= 0.5 or b[2] >= w - 0.5 or b[3] >= h - 0.5))
    dens = Counter(k.split("#")[0] for k in keys)
    return dict(G=G, dens=dens)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="P5.1 offline W_t x h_t decomposition audit")
    ap.add_argument("--dir", default=str(ROOT / "diagnostic/full300_trajectory_probe/from_snapshots"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--run", action="store_true")
    A = ap.parse_args()
    if not (A.check_only or A.run):
        print("请显式指定 --check-only 或 --run"); return 2

    src = Path(A.dir)
    out = Path(A.out) if A.out else src.parent / "p5_1_out"
    snaps = sorted([( _ep_of(p), p) for p in src.glob("epoch_*.npz")])
    print(f"[p5.1] 快照目录 = {src}")
    print(f"[p5.1] 发现 {len(snaps)} 个 epoch_*.npz")
    if len(snaps) < 3:
        print("STOP: 快照不足（<3）—— 无法做轨迹分析。")
        return 1

    # ============ §2 Artifact Integrity Gate ============
    print("\n=== §2 Artifact Integrity ===")
    Z = {e: np.load(p, allow_pickle=True) for e, p in snaps}
    eps = sorted(Z)
    need = ("head_W", "head_b", "vec_h", "vec_keys")
    miss = [k for k in need if k not in Z[eps[0]]]
    if miss:
        print(f"STOP: 缺 key {miss} —— 该 npz 不是本审计所需的快照格式。"); return 1

    wh, bh, hh = {}, {}, {}
    for e in eps:
        W, b, H = Z[e]["head_W"], Z[e]["head_b"], Z[e]["vec_h"]
        if W.shape != (12, 256) or b.shape != (12,):
            print(f"STOP: ep{e} 的 W/b shape 异常 {W.shape} {b.shape}"); return 1
        if H.ndim != 2 or H.shape[1] != 256:
            print(f"STOP: ep{e} 的 vec_h shape 异常 {H.shape}"); return 1
        fin = bool(np.isfinite(W).all() and np.isfinite(b).all() and np.isfinite(H).all())
        if not fin:
            print(f"STOP: ep{e} 存在 NaN/Inf"); return 1
        wh[e], bh[e], hh[e] = _sha(W), _sha(b), _sha(H)

    uw, ub = len(set(wh.values())), len(set(bh.values()))
    print(f"  唯一 W hash 数 = {uw} / {len(eps)}")
    print(f"  唯一 b hash 数 = {ub} / {len(eps)}")
    print(f"  唯一 h hash 数 = {len(set(hh.values()))} / {len(eps)}")
    if uw == 1:
        print("\nSTOP: W_t is still stale / constant.（§2 触发 —— 不做 live-head 分析）")
        print("      ⇒ 需重新同步带正确 head_W 的 from_snapshots/。")
        return 1
    if len(set(hh.values())) == 1:
        print("\nSTOP: h_t 恒定 —— 表征未变化，artifact 可疑。")
        return 1

    W0, b0 = Z[eps[0]]["head_W"].astype(np.float64), Z[eps[0]]["head_b"].astype(np.float64)
    Wf, bf = None, None
    for cand in ("best.pt",):
        p = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights" / cand
        if p.exists():
            import torch
            ck = torch.load(p, map_location="cpu", weights_only=False)
            src_m = ck.get("ema") or ck["model"]
            last = src_m.model[-1].cv3[0][-1]
            Wf = last.weight.detach().float().view(12, -1).numpy().astype(np.float64)
            bf = last.bias.detach().float().numpy().astype(np.float64)
            print(f"  W_final 来自 {cand}（{'ckpt.ema' if ck.get('ema') is not None else 'ckpt.model'}）")
            break
    if Wf is None:
        print("STOP: 找不到 best.pt 以提供 W_final。"); return 1
    print(f"  max|W_0 − W_final| = {np.abs(W0-Wf).max():.6f}   max|b_0 − b_final| = {np.abs(b0-bf).max():.6f}")
    if eps[0] != 0:
        print(f"  ⚠ 最早的快照是 0-based ep{eps[0]}，不是 ep0 —— W_0 用的是它，请确认可接受。")

    # ============ §3 Identity Alignment Gate ============
    print("\n=== §3 Identity Alignment ===")
    base = list(Z[eps[0]]["vec_keys"])
    bad = [e for e in eps if list(Z[e]["vec_keys"]) != base]
    if bad:
        print(f"STOP: vec_keys 在 epoch {bad} 与 ep{eps[0]} 不一致 —— identity alignment FAILED。")
        return 1
    cls_npz = Z[eps[0]]["vec_cls"].astype(int)
    cid = cls_npz
    cells = ROOT / "diagnostic/p3_trajectory_probe/medium_cells_canvas.json"
    if cells.exists():
        rows = json.loads(cells.read_text(encoding="utf-8"))["rows"]
        if [r["key"] for r in rows] == base:
            cc = np.array([r["class_id"] for r in rows])
            if not np.array_equal(cc, cid):
                print("  ⚠ medium_cells 的 class 与 npz vec_cls 不一致 —— 以 npz 为准")
            else:
                print("  ✓ 与 medium_cells_canvas.json 的 key/class 逐位一致")
    print(f"  key 数 = {len(base)}   class 分布 = {dict(sorted(Counter(cid.tolist()).items()))}")
    print("identity_alignment = PASS")
    if A.check_only:
        print("\n[check-only] 门禁通过。未做任何机制分析。")
        return 0

    # ============ 几何（与 npz 解耦） ============
    geo = load_geometry(base, cid)
    N = len(base)
    cidc = cid
    if geo:
        G = geo["G"]
        sq = np.array([G.get(k, {}).get("sqrt", np.nan) for k in base])
        brd = np.array([G.get(k, {}).get("border", np.nan) for k in base])
        ar_ = np.array([G.get(k, {}).get("ar", np.nan) for k in base])
        dns = np.array([geo["dens"].get(k.split("#")[0], 0) for k in base])
        Wd = np.array([G.get(k, {}).get("W", 0) for k in base])
        print(f"  geometry 装载成功（sqrt/border/aspect/density/resolution）")
    else:
        sq = brd = ar_ = dns = Wd = np.full(N, np.nan)
        print("  ⚠ geometry 装载失败 —— 相关 cohort 会标记 UNKNOWN")
    sq_q = np.nanpercentile(sq, [33, 67]) if np.isfinite(sq).any() else (np.nan, np.nan)

    # ============ §5-8 三读数（同一 GT-cell 口径） ============
    R, Hc, T, Rspec = {}, {}, {}, {}
    live, fin_, ini_, ctr_live, gm_live, oth_live = {}, {}, {}, {}, {}, {}
    for e in eps:
        Ht = Z[e]["vec_h"].astype(np.float64)
        Wt, bt = Z[e]["head_W"].astype(np.float64), Z[e]["head_b"].astype(np.float64)
        H0 = Z[eps[0]]["vec_h"].astype(np.float64)

        # --- Shapley 两路径（可通过恒等式）---
        R[e] = 0.5 * ((gt_logit(W0, b0, Ht, cidc) - gt_logit(W0, b0, H0, cidc))
                      + (gt_logit(Wt, bt, Ht, cidc) - gt_logit(Wt, bt, H0, cidc)))
        Hc[e] = 0.5 * ((gt_logit(Wt, bt, H0, cidc) - gt_logit(W0, b0, H0, cidc))
                       + (gt_logit(Wt, bt, Ht, cidc) - gt_logit(W0, b0, Ht, cidc)))
        T[e] = gt_logit(Wt, bt, Ht, cidc) - gt_logit(W0, b0, H0, cidc)
        # --- 规格原文那一版（表征项用 W_final）---
        Rspec[e] = 0.5 * ((gt_logit(Wf, bf, Ht, cidc) - gt_logit(Wf, bf, H0, cidc))
                          + (gt_logit(Wt, bt, Ht, cidc) - gt_logit(Wt, bt, H0, cidc)))

        L = all_logits(Wt, bt, Ht)
        g = L[np.arange(N), cidc]
        o = np.delete(L, cidc[:, None], axis=1).max(1)
        live[e] = g
        oth_live[e] = o
        gm_live[e] = L.mean(1)
        ctr_live[e] = g - L.mean(1)
        ini_[e] = gt_logit(W0, b0, Ht, cidc)
        fin_[e] = gt_logit(Wf, bf, Ht, cidc)

    # ============ §9 恒等式自检 ============
    print("\n=== §9 Decomposition Identity Check ===")
    err = max(float(np.abs(R[e] + Hc[e] - T[e]).max()) for e in eps)
    rspec = max(float(np.abs(Rspec[e] + Hc[e] - T[e]).max()) for e in eps)
    pred = max(float(np.abs(Rspec[e] + Hc[e] - T[e]
                            - 0.5 * ((Wf - W0)[cidc] * (Z[e]["vec_h"].astype(np.float64)
                                                       - Z[eps[0]]["vec_h"].astype(np.float64))).sum(1)).max())
               for e in eps)
    print(f"  主分解 max|R+H−T|            = {err:.3e}   {'PASS' if err < INT_TOL else 'FAIL'}")
    print(f"  规格原文版 max|R_spec+H−T|    = {rspec:.3e}  （预期非零）")
    print(f"    其残差与 ½(W_f−W_0)_gt·(h_t−h_0) 的最大偏差 = {pred:.3e}")
    if err >= DECOMP_TOL:
        print("\nSTOP: decomposition identity failed.")
        return 1

    # ============ §11/§12 dead-tail ============
    dead = {e: (sigma(live[e]) < DEAD_LO) for e in eps}
    rows_sum, per_gt, dec, coh_stat = [], [], [], []
    for e in eps:
        s_live = sigma(live[e]); s_ini = sigma(ini_[e]); s_fin = sigma(fin_[e])
        rows_sum.append(dict(epoch=e, n=N,
                             live_med=float(np.median(s_live)), ini_med=float(np.median(s_ini)),
                             fin_med=float(np.median(s_fin)),
                             live_dead_lo=float(np.mean(s_live < DEAD_LO)), fin_dead_lo=float(np.mean(s_fin < DEAD_LO)),
                             ini_dead_lo=float(np.mean(s_ini < DEAD_LO)),
                             live_dead_hi=float(np.mean(s_live < DEAD_HI)),
                             gm_live_med=float(np.median(gm_live[e])),
                             ctr_live_med=float(np.median(ctr_live[e])),
                             ctr_live_neg=float(np.mean(ctr_live[e] < 0)),
                             R_med=float(np.median(R[e])), H_med=float(np.median(Hc[e])),
                             T_med=float(np.median(T[e]))))
        for i, k in enumerate(base):
            per_gt.append(dict(epoch=e, gt_key=k, **{"class": int(cidc[i])},
                               live_gt_logit=float(live[e][i]), live_other_logit=float(oth_live[e][i]),
                               live_margin=float(live[e][i] - oth_live[e][i]), live_sigmoid=float(s_live[i]),
                               live_global_mean=float(gm_live[e][i]), live_centered_gt=float(ctr_live[e][i]),
                               frozen_initial_gt=float(ini_[e][i]), frozen_final_gt=float(fin_[e][i]),
                               rep_contribution=float(R[e][i]), head_contribution=float(Hc[e][i])))
        for i in range(N):
            dec.append(dict(epoch=e, gt_key=base[i], gt_class=int(cidc[i]),
                            live=float(live[e][i]), ini=float(ini_[e][i]), fin=float(fin_[e][i]),
                            R=float(R[e][i]), H=float(Hc[e][i]), T=float(T[e][i]),
                            R_spec=float(Rspec[e][i]), gm=float(gm_live[e][i]), ctr=float(ctr_live[e][i])))

    print("\n=== §11 三读数（同一 GT-cell 口径，dead = sigmoid < 1e-3）===")
    print(f"{'ep':>5} | {'live med':>9} {'dead':>7} | {'init med':>9} {'dead':>7} | {'final med':>10} {'dead':>7}")
    for r in rows_sum:
        print(f"{r['epoch']:>5} | {r['live_med']:>9.3f} {r['live_dead_lo']:>7.1%} | "
              f"{r['ini_med']:>9.3f} {r['ini_dead_lo']:>7.1%} | {r['fin_med']:>10.3f} {r['fin_dead_lo']:>7.1%}")

    print("\n=== §12 dead-tail 轮转性 ===")
    ent = []
    for a, b in zip(eps[:-1], eps[1:]):
        da, db = dead[a], dead[b]
        inter = int((da & db).sum()); ea = int((~da & db).sum()); ex = int((da & ~db).sum())
        jac = inter / max(int((da | db).sum()), 1)
        ent.append(dict(from_epoch=a, to_epoch=b, dead_from=int(da.sum()), dead_to=int(db.sum()),
                        intersection=inter, entries=ea, exits=ex, jaccard=round(jac, 4)))
        print(f"  ep{a:>3}→ep{b:<3} dead {int(da.sum()):>4}→{int(db.sum()):<4} ∩={inter:<4} 进={ea:<4} 出={ex:<4} J={jac:.3f}")
    js = [x["jaccard"] for x in ent]
    print(f"  Jaccard 中位 = {np.median(js):.3f}  ⇒ "
          f"{'persistent' if np.median(js) > 0.7 else ('rotating' if np.median(js) < 0.4 else 'mixture')}")

    # ============ §13 dead-entry event alignment ============
    print("\n=== §13 dead-entry event alignment ===")
    first_entry = {}
    for i, k in enumerate(base):
        for e in eps:
            if dead[e][i]:
                first_entry[k] = e; break
    ev_rows = []
    idx_of = {e: n for n, e in enumerate(eps)}
    for k, e0 in first_entry.items():
        i = base.index(k); n0 = idx_of[e0]
        for off in (-2, -1, 0, 1):
            n1 = n0 + off
            if 0 <= n1 < len(eps):
                e = eps[n1]
                ev_rows.append(dict(gt_key=k, entry_epoch=e0, offset=off, epoch=e,
                                    live=float(live[e][i]), other=float(oth_live[e][i]),
                                    margin=float(live[e][i] - oth_live[e][i]),
                                    R=float(R[e][i]), H=float(Hc[e][i]),
                                    gm=float(gm_live[e][i]), ctr=float(ctr_live[e][i])))
    if ev_rows:
        import statistics as st
        for off in (-2, -1, 0, 1):
            v = [r for r in ev_rows if r["offset"] == off]
            if not v:
                continue
            print(f"  offset {off:+d} (n={len(v):4d}): live={st.median(x['live'] for x in v):7.2f} "
                  f"R={st.median(x['R'] for x in v):6.2f} H={st.median(x['H'] for x in v):6.2f} "
                  f"gm={st.median(x['gm'] for x in v):7.2f} ctr={st.median(x['ctr'] for x in v):6.2f}")
        step = np.median(np.diff(eps))
        if step > 10:
            print(f"  ⚠ 观测步长中位 = {step:.0f} epoch ⇒ temporal resolution insufficient（不做细粒度先后结论）")
    else:
        print("  无 dead-entry 事件 ⇒ UNKNOWN")

    # ============ §14 cohort ============
    print("\n=== §14 cohort（outcome-independent）===")
    grp = {"all": np.ones(N, bool)}
    if np.isfinite(sq).any():
        grp["size_small"], grp["size_mid"], grp["size_large"] = sq < sq_q[0], (sq >= sq_q[0]) & (sq < sq_q[1]), sq >= sq_q[1]
    if np.isfinite(brd).any():
        grp["border_near"] = brd < np.nanpercentile(brd, 25)
        grp["border_far"] = ~grp["border_near"]
    if np.isfinite(ar_).any():
        grp["aspect_thin"] = ar_ > 3.2
    if np.isfinite(dns).any():
        grp["density_lo"] = dns <= np.median(dns)
    for c in sorted(set(cidc.tolist())):
        if (cidc == c).sum() >= 20:
            grp[f"class_{c}"] = cidc == c
    eL = eps[-1]
    print(f"{'cohort':16s} {'n':>5} {'dead':>7} {'live':>7} {'gm':>8} {'ctr':>7} {'R':>7} {'H':>7}")
    for nm, m in grp.items():
        if m.sum() < 5:
            continue
        coh_stat.append(dict(cohort=nm, n=int(m.sum()), epoch=eL,
                             dead_rate=float(np.mean(sigma(live[eL])[m] < DEAD_LO)),
                             live_med=float(np.median(live[eL][m])), gm_med=float(np.median(gm_live[eL][m])),
                             ctr_med=float(np.median(ctr_live[eL][m])),
                             R_med=float(np.median(R[eL][m])), H_med=float(np.median(Hc[eL][m]))))
        print(f"{nm:16s} {int(m.sum()):>5} {coh_stat[-1]['dead_rate']:>7.1%} {coh_stat[-1]['live_med']:>7.2f} "
              f"{coh_stat[-1]['gm_med']:>8.2f} {coh_stat[-1]['ctr_med']:>7.2f} "
              f"{coh_stat[-1]['R_med']:>7.2f} {coh_stat[-1]['H_med']:>7.2f}")

    # ============ §16 common-mode ============
    print("\n=== §16 common-mode vs centered ===")
    cm = dict(epoch=eL, live_gm=float(np.median(gm_live[eL])), live_ctr=float(np.median(ctr_live[eL])),
              live_ctr_neg=float(np.mean(ctr_live[eL] < 0)),
              ini_ctr=float(np.median(gt_logit(W0, b0, Z[eL]["vec_h"].astype(np.float64), cidc)
                                      - all_logits(W0, b0, Z[eL]["vec_h"].astype(np.float64)).mean(1))),
              fin_ctr=float(np.median(gt_logit(Wf, bf, Z[eL]["vec_h"].astype(np.float64), cidc)
                                      - all_logits(Wf, bf, Z[eL]["vec_h"].astype(np.float64)).mean(1))))
    print(f"  live : 12类均值={cm['live_gm']:7.2f}  居中GT={cm['live_ctr']:7.2f}  居中<0={cm['live_ctr_neg']:.1%}")
    print(f"  init : 居中GT={cm['ini_ctr']:7.2f}")
    print(f"  final: 居中GT={cm['fin_ctr']:7.2f}")
    print(f"  ⇒ {'common-mode vacuum 支持' if cm['live_ctr_neg'] < 0.05 else '存在 class-relative 退化'}")

    # ============ 输出 ============
    out.mkdir(parents=True, exist_ok=True)
    def wcsv(name, rows):
        if not rows:
            return
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    wcsv("p5_1_wt_ht_summary.csv", rows_sum)
    wcsv("p5_1_wt_ht_per_gt.csv", per_gt)
    wcsv("p5_1_decomposition.csv", dec)
    wcsv("p5_1_dead_entry.csv", ev_rows)
    wcsv("p5_1_cohort_summary.csv", coh_stat)
    print(f"\n[p5.1] CSV 已写出 -> {out}")

    # ============ §18-23 措辞 / GPU gate ============
    # 机制判定：以**末观测点**为主（终态），同时给出全程中位作稳健性参考
    rep_fin, head_fin = abs(rows_sum[-1]["R_med"]), abs(rows_sum[-1]["H_med"])
    rep_med = float(np.median([abs(r["R_med"]) for r in rows_sum]))
    head_med = float(np.median([abs(r["H_med"]) for r in rows_sum]))

    def _case(Rv, Hv):
        """★ 只看**为负**的项：只有负贡献才驱动退化。用 |·| 会把「R 正 H 负」误判成 Coupled。"""
        dn = [(-Rv, "Representation-driven"), (-Hv, "Head-driven")]
        dn = [(m, n) for m, n in dn if m > 0 and m >= 2.0]
        if not dn:
            return "Unresolved"
        dn.sort(reverse=True)
        if len(dn) == 2:
            return "Coupled" if min(dn[0][0], dn[1][0]) >= 2.0 else dn[0][1]
        return dn[0][1]

    R_fin, H_fin = rows_sum[-1]["R_med"], rows_sum[-1]["H_med"]
    R_mid = float(np.median([r["R_med"] for r in rows_sum]))
    H_mid = float(np.median([r["H_med"] for r in rows_sum]))
    case = _case(R_fin, H_fin)
    case_med = _case(R_mid, H_mid)
    txt = [
        "# P5.1 — W_t × h_t Offline Decomposition Report", "",
        "## 硬结论", "",
        f"1. W_t 是否真的存在？            **是**（唯一 W hash = {uw}）",
        f"2. identity 是否 PASS？          **是**（key {N}，跨 {len(eps)} epoch 逐位一致）",
        f"3. decomposition identity 是否 PASS？ **是**（max|R+H−T| = {err:.2e}）",
        f"4. live 极化是否在同口径复现？   见 p5_1_wt_ht_summary.csv 的 live_dead_lo 列",
        f"5. R / H 主导？                  **末观测点** R={R_fin:+.3f} H={H_fin:+.3f} ⇒ **{case}**；"
        f"全程中位 R={R_mid:+.3f} H={H_mid:+.3f} ⇒ {case_med}",
        f"6. common-mode vacuum 是否成立？ 居中<0 占比 = {cm['live_ctr_neg']:.1%}",
        f"7. dead-tail persistent/rotating？ Jaccard 中位 = {np.median(js):.3f}",
        "8. 是否足够设计 GPU intervention？ **否**（见 §23 门）", "",
        "## 口径声明", "",
        "- 三个读数（live / frozen_initial / frozen_final）**全部为 GT 单元格点值**，同一口径。",
        "- 历史的 `cls_center_max` 是邻域最大值，**不在本表内混用**。",
        "- **decomposition ≠ causality**：本分解只回答总 logit change 落在哪条状态变量路径上，",
        "  不是干预实验。", "",
        "## §9 公式说明", "",
        f"- 规格原文版（表征项用 W_final）max|R_spec+H−T| = **{rspec:.3e}**（非零）；",
        "  其残差恒等于 ½·(W_final−W_0)_gt·(h_t−h_0)。",
        f"- 本报告主分解使用可通过恒等式的 Shapley 标准形式，max|R+H−T| = {err:.3e}。", "",
        "## GPU Gate", "", "```", "NO GPU",
        "```",
    ]
    (out / "p5_1_report.md").write_text("\n".join(txt), encoding="utf-8")
    print(f"[p5.1] 报告 -> {out/'p5_1_report.md'}")
    print(f"\n[p5.1] 机制倾向: {case}   |R|中位={rep_med:.3f}  |H|中位={head_med:.3f}")
    print("[p5.1] GPU GATE: NO GPU（分解 ≠ 因果；未见可针对的上游具体干预）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
