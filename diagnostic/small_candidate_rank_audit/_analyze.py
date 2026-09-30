"""_analyze.py — Small Candidate Rank 5~16 Feasibility Audit（**v1，含两个方法学错误，已被 _analyze2.py 取代**）

⚠⚠ 保留此文件仅为过程记录。它有两个错误，**其输出不可直接引用**：

  错误 1（同义反复）：`max_CIoU@K`。
      Rank-A 按 CIoU 降序后，前 K 个的 max 恒等于 rank-1 的值 ⇒ 对 K=1..16 **完全相同**
      （实测 0.8808，Δ=0.0000，改善率 0.0%）。这不是发现，是排序定义使然。
      并且它掩盖了一个重要事实：**新增候选不可能提高 max**（当前 max 已由池内最优候选取得）。

  错误 2（跨 rank 不可比）：逐 rank 表。
      rank k 只对 n_cand>=k 的 GT 有定义，eligible n 从 244 降到 33；
      被选中的是「框内容纳更多 anchor」的一小撮 GT ⇒ 逐 rank 中位数把**群体变化**读成了
      「rank 质量上升」（实测 CIoU 从 rank1 的 0.8808「升」到 rank16 的 0.9132）。

修正版见 `_analyze2.py`（固定总体 + 正确的 headroom 口径）。本文件的分析结论**不进入裁决**。

---
原始 docstring：

_analyze.py — Small Candidate Rank 5~16 Feasibility Audit（纯离线只读）

只读 `diagnostic/candidate_density_counterfactual/_results.npz`。
不重新 forward / 不重新 augmentation / 不改 assigner / 不训练 / 不 backward / 不 optimizer.step。

═══ 前置结论（决定了本轮能问什么）═══
现有 NPZ 的 candidate 定义是
    idx = np.flatnonzero(mask_in_gts[g])        # select_candidates_in_gts == True
即**只含「anchor 中心严格落在 GT 框内」的候选**，框外 anchor 完全没有保存。

因此：
  · 「rank 5~16 是否存在于真实 grid」这个问题，NPZ **只能回答其中一半**：
      已存在且在池内的部分（mask_in_gts==True 但未被 topk 选中）
      —— 对 native-small，只有当 n_cand > 4（甚至 >10）时才存在。
  · 「增加 candidate density 会带来的**新增**候选」在 NPZ 里**根本不存在**，
      无法区分 §14 的 Situation A（grid 有、被中心约束滤掉）与 Situation B（grid 本来就没有）。
⇒ 按协议 §3/§15，主裁决为 D — UNRESOLVED / STATUS = INSUFFICIENT_DATA。

本脚本仍把**数据支持的那部分**（池内 rank 5~16）如实算出来，并逐格标注 eligible n，
作为**有界补充观察**，明确不改变主裁决。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AUDIT = Path(__file__).resolve().parent
SRC = AUDIT.parent / "candidate_density_counterfactual" / "_results.npz"
MAXR = 16
TOL = 1e-12


def q(x, p):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if x.size else float("nan")


def desc(x):
    x = np.asarray(x, dtype=np.float64); x = x[np.isfinite(x)]
    if not x.size:
        return None
    return dict(n=int(x.size), mean=float(x.mean()), sd=float(x.std(ddof=1)) if x.size > 1 else float("nan"),
                p10=q(x, 10), p25=q(x, 25), p50=q(x, 50), p75=q(x, 75), p90=q(x, 90))


def frac_ge(x, t):
    x = np.asarray(x, dtype=np.float64); x = x[np.isfinite(x)]
    return float((x >= t).mean()) if x.size else float("nan")


def main():
    print("=" * 108)
    print("SMALL CANDIDATE RANK 5~16 FEASIBILITY AUDIT（纯离线只读）")
    print("=" * 108)

    z = np.load(SRC)
    uid = z["gt_uid"]; area = z["gt_native_area"]; bw = z["gt_bw"]; bh = z["gt_bh"]
    cgt = z["cand_gt"]
    cciou = z["cand_ciou"]; ccls = z["cand_cls"]; calign = z["cand_align"]
    cstr = np.asarray(z["cand_stride"]).reshape(-1)

    # ---- uid 折叠（与 CF 审计同口径）----
    order = {}
    for i, u in enumerate(uid):
        order[int(u)] = i
    keep = np.array(sorted(order.values()))
    remap = np.full(len(uid), -1)
    for n_, o_ in enumerate(keep):
        remap[o_] = n_
    gsel = remap[cgt]; m = gsel >= 0
    cgt_c = gsel[m]; cciou_c = cciou[m]; ccls_c = ccls[m]; calign_c = calign[m]; cstr_c = cstr[m]
    area_c = area[keep]; bw_c = bw[keep]; bh_c = bh[keep]; ncand_c = z["gt_ncand"][keep]
    uid_c = uid[keep]

    A = np.array([("small" if a < 1024 else ("medium" if a < 9216 else "large")) for a in area_c])
    starts = np.searchsorted(cgt_c, np.arange(len(uid_c)))
    ends = np.searchsorted(cgt_c, np.arange(len(uid_c)), side="right")
    sm = np.flatnonzero(A == "small")
    n_sm = sm.size

    print(f"\n[§2 数据充分性]")
    print(f"  NPZ keys: {len(z.files)}  折后 native GT {len(uid_c)}  候选 {len(cgt_c)}")
    print(f"  candidate 定义 = mask_in_gts==True（**只含框内 anchor 中心**，框外 anchor 未保存）")
    print(f"  ⇒ 无法区分 §14 Situation A（grid 有、被中心约束滤掉）vs Situation B（grid 本就没有）")
    print(f"\n  native-small GT n = {n_sm}；n_cand 分布：min={ncand_c[sm].min()} "
          f"p50={np.median(ncand_c[sm]):.0f} p75={np.percentile(ncand_c[sm],75):.0f} "
          f"p90={np.percentile(ncand_c[sm],90):.0f} max={ncand_c[sm].max()}")
    elig = {}
    for thr in (4, 5, 8, 12, 16):
        k = int((ncand_c[sm] >= thr).sum()); elig[thr] = k
        print(f"    n_cand >= {thr:>2}: {k:>4} / {n_sm}  ({100*k/n_sm:5.1f}%)")

    # ---- 按 CIoU 降序排序（Rank-A，主口径）----
    CI = np.full((n_sm, MAXR), np.nan)
    CL = np.full((n_sm, MAXR), np.nan)
    AL = np.full((n_sm, MAXR), np.nan)
    ST = np.full((n_sm, MAXR), np.nan)
    for j, g in enumerate(sm):
        sl = slice(starts[g], ends[g])
        ci = cciou_c[sl]
        n = ci.size
        if n == 0:
            continue
        o = np.argsort(-ci)                      # Rank-A：CIoU 降序
        k = min(n, MAXR)
        CI[j, :k] = ci[o[:k]]
        CL[j, :k] = ccls_c[sl][o[:k]]
        AL[j, :k] = calign_c[sl][o[:k]]
        ST[j, :k] = cstr_c[sl][o[:k]]

    # ---- §5/§7 逐 rank 表 ----
    print("\n" + "=" * 108)
    print("§5/§7 RANK 1~16 逐 rank 质量（Rank-A = 按 CIoU 降序）")
    print("  ⚠ eligible n 随 rank 递减：rank k 只对 n_cand>=k 的 GT 有定义。")
    print("=" * 108)
    print(f"  {'rank':>4} {'elig_n':>7} {'CIoU med':>10} {'mean':>8} {'p10':>7} {'p25':>7} {'p75':>7} {'p90':>7} "
          f"| {'P(≥.3)':>7} {'P(≥.5)':>7} {'P(≥.7)':>7} {'P(≥.8)':>7} "
          f"| {'cls med':>8} {'P(cls≥.1)':>10} {'P(cls≥.3)':>10} | {'stride med':>10}")
    rank_tab = {}
    for k in range(1, MAXR + 1):
        ci = CI[:, k - 1]; cl = CL[:, k - 1]; st = ST[:, k - 1]
        e = int(np.isfinite(ci).sum())
        if e == 0:
            continue
        d = desc(ci)
        stv = st[np.isfinite(st)]
        rank_tab[k] = dict(elig_n=e, ciou=d, p_ge={t: frac_ge(ci, t) for t in (0.3, 0.5, 0.7, 0.8)},
                           cls_med=q(cl, 50), p_cls_ge={t: frac_ge(cl, t) for t in (0.1, 0.3, 0.5)},
                           stride_med=float(np.median(stv)) if stv.size else float("nan"),
                           stride_cnt={int(s): int((stv == s).sum()) for s in (8, 16, 32)})
        print(f"  {k:>4} {e:>7} {d['p50']:>10.4f} {d['mean']:>8.4f} {d['p10']:>7.4f} {d['p25']:>7.4f} "
              f"{d['p75']:>7.4f} {d['p90']:>7.4f} | "
              + " ".join(f"{100*frac_ge(ci,t):>6.1f}%" for t in (0.3, 0.5, 0.7, 0.8)) + " | "
              f"{q(cl,50):>8.4f} " + " ".join(f"{100*frac_ge(cl,t):>9.1f}%" for t in (0.1, 0.3)) +
              f" | {(float(np.median(stv)) if stv.size else float('nan')):>10.1f}")

    # ---- §6/§7 cumulative max@K ----
    print("\n" + "=" * 108)
    print("§6/§7 CUMULATIVE max@K（同一批 eligible GT 上）")
    print("=" * 108)
    print(f"  {'K':>4} {'elig_n':>7} {'maxCIoU med':>12} {'Δ vs K=4':>10} {'P(≥.5)':>8} {'P(≥.7)':>8} "
          f"{'P(≥.8)':>8} | {'maxCls med':>11} {'maxAlign med':>13} {'P(maxCIoU↑ | K>4)':>18}")
    cum = {}
    for K in range(1, MAXR + 1):
        ci = np.nanmax(CI[:, :K], axis=1)
        cl = np.nanmax(CL[:, :K], axis=1)
        al = np.nanmax(AL[:, :K], axis=1)
        ok = np.isfinite(ci)
        # 只在 n_cand>=K 的 GT 上比较（保证 K 与 4 的可比性）
        eligible = np.isfinite(CI[:, K - 1])
        base = np.nanmax(CI[eligible][:, :4], axis=1) if K >= 4 else None
        hi = (ci[eligible] > base + TOL).mean() if K > 4 and base is not None and base.size else float("nan")
        d = desc(ci[ok]); dcl = desc(cl[ok]); dal = desc(al[ok])
        cum[K] = dict(elig_n=int(eligible.sum()), maxciou=d["p50"],
                      delta_vs_k4=(d["p50"] - cum[4]["maxciou"]) if 4 in cum else float("nan"),
                      p_ge={t: frac_ge(ci[ok], t) for t in (0.5, 0.7, 0.8)},
                      maxcls=dcl["p50"], maxalign=dal["p50"],
                      frac_improved_over_k4=float(hi))
        row = cum[K]
        print(f"  {K:>4} {row['elig_n']:>7} {d['p50']:>12.4f} {row['delta_vs_k4']:>10.4f} "
              + " ".join(f"{100*row['p_ge'][t]:>7.1f}%" for t in (0.5, 0.7, 0.8)) + " | "
              f"{dcl['p50']:>11.4f} {dal['p50']:>13.4f} {100*hi:>17.1f}%")

    # ---- §8 stride 分解 ----
    print("\n" + "=" * 108)
    print("§8 STRIDE 分解（native-small 的候选来自哪个 stride）")
    print("=" * 108)
    usable = np.isfinite(ST)
    tot = int(usable.sum())
    print(f"  {'stride':>7} {'rank1-4 cnt':>12} {'rank5-10 cnt':>13} {'rank11-16 cnt':>14} "
          f"{'rank5-16 占比':>14} {'CIoU med(rank5-16)':>19} {'cls med(rank5-16)':>18}")
    stride_tab = {}
    for s in (8.0, 16.0, 32.0):
        c14 = int(((ST[:, :4] == s) & usable[:, :4]).sum())
        c510 = int(((ST[:, 4:10] == s) & usable[:, 4:10]).sum())
        c1116 = int(((ST[:, 10:16] == s) & usable[:, 10:16]).sum())
        mask516 = (ST[:, 4:16] == s) & usable[:, 4:16]
        ci516 = CI[:, 4:16][mask516]; cl516 = CL[:, 4:16][mask516]
        tot516 = int(((ST[:, 4:16] != 0) & usable[:, 4:16]).sum())
        stride_tab[s] = dict(r14=c14, r510=c510, r1116=c1116, share=(c510 + c1116) / max(tot516, 1),
                             ciou_med=q(ci516, 50), cls_med=q(cl516, 50))
        print(f"  {s:>7.0f} {c14:>12} {c510:>13} {c1116:>14} "
              f"{100*(c510+c1116)/max(tot516,1):>13.1f}% {q(ci516,50):>19.4f} {q(cl516,50):>18.4f}")
    print("  ⇒ 若 rank5~16 主要由 stride 16/32 贡献，则不能解释为『提高 stride-8 密度』。")

    # ---- §9 GT 尺寸分层 ----
    print("\n" + "=" * 108)
    print("§9 GT 尺寸分层（native sqrt 四分位）")
    print("=" * 108)
    sq = np.sqrt(area_c[sm])
    qs = np.percentile(sq, [25, 50, 75])
    print(f"  {'stratum':>16} {'n':>4} {'n_cand med':>11} {'maxCIoU@4':>10} {'maxCIoU@8':>10} "
          f"{'maxCIoU@16':>11} {'Δ@8':>8} {'Δ@16':>8} {'elig16':>7}")
    size_tab = []
    strata = [("<S1 (p25)", sq <= qs[0]), ("S2 (25-50%)", (sq > qs[0]) & (sq <= qs[1])),
              ("S3 (50-75%)", (sq > qs[1]) & (sq <= qs[2])), (">S4 (p75+)", sq > qs[2])]
    for nm, msk in strata:
        ii = np.flatnonzero(msk)
        if ii.size == 0:
            continue
        c4 = np.nanmax(CI[ii][:, :4], axis=1)
        c8 = np.nanmax(CI[ii][:, :8], axis=1)
        c16 = np.nanmax(CI[ii][:, :16], axis=1)
        e16 = int(np.isfinite(CI[ii][:, 15]).sum())
        size_tab.append(dict(stratum=nm, n=int(ii.size), ncand_med=float(np.median(ncand_c[sm][ii])),
                             m4=q(c4, 50), m8=q(c8, 50), m16=q(c16, 50),
                             d8=q(c8, 50) - q(c4, 50), d16=q(c16, 50) - q(c4, 50), elig16=e16))
        r = size_tab[-1]
        print(f"  {nm:>16} {r['n']:>4} {r['ncand_med']:>11.0f} {r['m4']:>10.4f} {r['m8']:>10.4f} "
              f"{r['m16']:>11.4f} {r['d8']:>8.4f} {r['d16']:>8.4f} {e16:>7}")

    # ---- §10 AR 分层 ----
    print("\n" + "=" * 108)
    print("§10 ASPECT-RATIO 分层")
    print("=" * 108)
    ar = bw_c[sm] / np.maximum(bh_c[sm], 1e-9)
    print(f"  {'AR 层':>14} {'n':>4} {'maxCIoU@4':>10} {'maxCIoU@8':>10} {'maxCIoU@16':>11} {'elig16':>7}")
    ar_tab = []
    for nm, msk in (("AR<0.5", ar < 0.5), ("0.5<=AR<1", (ar >= 0.5) & (ar < 1)),
                    ("1<=AR<2", (ar >= 1) & (ar < 2)), ("AR>=2", ar >= 2)):
        ii = np.flatnonzero(msk)
        if ii.size == 0:
            continue
        c4 = np.nanmax(CI[ii][:, :4], axis=1); c8 = np.nanmax(CI[ii][:, :8], axis=1)
        c16 = np.nanmax(CI[ii][:, :16], axis=1); e16 = int(np.isfinite(CI[ii][:, 15]).sum())
        ar_tab.append(dict(ar=nm, n=int(ii.size), m4=q(c4, 50), m8=q(c8, 50), m16=q(c16, 50), elig16=e16))
        print(f"  {nm:>14} {ii.size:>4} {q(c4,50):>10.4f} {q(c8,50):>10.4f} {q(c16,50):>11.4f} {e16:>7}")

    # ---- §11/§12 双门槛 useful_candidate ----
    print("\n" + "=" * 108)
    print("§12 双门槛 useful_candidate（rank 5~16）")
    print("=" * 108)
    defs = [("CIoU>=0.5 & cls>=0.05", 0.5, 0.05), ("CIoU>=0.5 & cls>=0.10", 0.5, 0.10),
            ("CIoU>=0.7 & cls>=0.10", 0.7, 0.10)]
    print(f"  {'定义':>24} {'rank5-16 全部候选':>18} {'rank5-10':>10} {'rank11-16':>10} "
          f"{'per-GT 至少1个(rank5-16)':>24}")
    use_tab = {}
    ci516 = CI[:, 4:16]; cl516 = CL[:, 4:16]; ok516 = np.isfinite(ci516)
    for nm, tc, tk in defs:
        good = (ci516 >= tc) & (cl516 >= tk) & ok516
        pergt = good.any(axis=1)
        g510 = good[:, :6]; g1116 = good[:, 6:]
        use_tab[nm] = dict(all=float(good.sum() / max(ok516.sum(), 1)),
                           r510=float(g510.sum() / max(ok516[:, :6].sum(), 1)),
                           r1116=float(g1116.sum() / max(ok516[:, 6:].sum(), 1)),
                           per_gt=float(pergt.mean()))
        print(f"  {nm:>24} {100*use_tab[nm]['all']:>17.1f}% {100*use_tab[nm]['r510']:>9.1f}% "
              f"{100*use_tab[nm]['r1116']:>9.1f}% {100*use_tab[nm]['per_gt']:>23.1f}%")
    print("  对照：rank1-4 上同一双门槛的 per-GT 比例 = "
          f"{100*(((CI[:,:4]>=0.5)&(CL[:,:4]>=0.10)&np.isfinite(CI[:,:4])).any(axis=1)).mean():.1f}%")

    json.dump(dict(
        status="INSUFFICIENT_DATA_FOR_THE_ASKED_QUESTION",
        verdict="D — UNRESOLVED",
        reason=("现有 artifact 只保存 select_candidates_in_gts==True 的候选（框内 anchor 中心），"
                "框外 anchor 未保存 ⇒ 无法区分 §14 Situation A（grid 有、被中心约束滤掉）"
                "与 Situation B（grid 本就没有），因此无法回答『增加 candidate density 是否可行』。"),
        eligibility=elig, n_small=int(n_sm),
        rank_table=rank_tab, cumulative=cum, stride=stride_tab,
        size_strata=size_tab, ar_strata=ar_tab, useful=use_tab,
        note="逐格 eligible n 已给出；rank k 只对 n_cand>=k 的 GT 有定义，故随 rank 递减。"),
        open(str(AUDIT / "_tables.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {AUDIT / '_tables.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
