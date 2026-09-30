"""_analyze.py — 反事实分析：candidate 数量 effect vs candidate quality effect（只读、离线）

CF1  Large -> Random-4    从真实 candidate set 无放回抽 4 个（seed 0..99 固定），配对反事实
CF2  Large -> Best-4      按 CIoU 取前 4（上界参考；注意 maxCIoU 恒等于 native max，见 §7）
CF2b Large -> 4th-best    前 4 的 CIoU 下界（CF2 真正有信息的部分）
CF3  Large -> Stride8-4   限制到 stride-8 候选、取离 GT 中心最近的 4 个（spatial/stride 匹配）

主结论只用 candidate geometry 指标（n_cand / maxCIoU / P(maxCIoU>=t)），
best_align 仅作辅助并明确标注其受 alpha/beta 与 cls 影响。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
SEEDS = list(range(100))          # 固定 RNG seed 列表，不每次随机生成
SMALL_MAX, MED_MAX = 1024.0, 9216.0


def cls_of(a):
    return "small" if a < SMALL_MAX else ("medium" if a < MED_MAX else "large")


def q(x, p):
    x = np.asarray(x, dtype=np.float64)
    return float(np.percentile(x, p)) if x.size else float("nan")


def desc(x):
    x = np.asarray(x, dtype=np.float64)
    return dict(n=int(x.size), mean=float(x.mean()) if x.size else float("nan"),
                median=q(x, 50), p10=q(x, 10), p25=q(x, 25), p75=q(x, 75), p90=q(x, 90),
                sd=float(x.std(ddof=1)) if x.size > 1 else float("nan"))


def frac_ge(x, t):
    x = np.asarray(x, dtype=np.float64)
    return float((x >= t).mean()) if x.size else float("nan")


def main():
    z = np.load(OUT / "_results.npz")
    meta = json.loads((OUT / "_meta.json").read_text(encoding="utf-8"))
    uid = z["gt_uid"]; area = z["gt_native_area"]
    bw = z["gt_bw"]; bh = z["gt_bh"]
    ncand = z["gt_ncand"]; mciou = z["gt_maxciou"]; mcls = z["gt_maxcls"]
    malign = z["gt_maxalign"]; zero = z["gt_zero"]
    cgt = z["cand_gt"]; cciou = z["cand_ciou"]; ccls = z["cand_cls"]
    calign = z["cand_align"]; cstride = np.asarray(z["cand_stride"]).reshape(-1)
    coffx = z["cand_offx"]; coffy = z["cand_offy"]

    print("=" * 104)
    print("CANDIDATE-DENSITY COUNTERFACTUAL — 分析")
    print("=" * 104)
    print(f"  真实 assigner = {meta['real_assigner']}   self-check = {meta['probe_selfcheck_pass']}")
    print(f"  原始实例：GT {len(uid)} / 候选 {len(cgt)}   采样 {meta['n_images']} 图 × {meta['epochs']} epoch")

    # ---- uid 折叠（保留最后一次 realization），与 E2 baseline 同口径 ----
    order = {}
    for i, u in enumerate(uid):
        order[int(u)] = i
    keep = np.array(sorted(order.values()), dtype=np.int64)
    remap = np.full(len(uid), -1, dtype=np.int64)
    for newi, oldi in enumerate(keep):
        remap[oldi] = newi
    uid_c = uid[keep]; area_c = area[keep]; ncand_c = ncand[keep]
    mciou_c = mciou[keep]; mcls_c = mcls[keep]; malign_c = malign[keep]; zero_c = zero[keep]
    gsel = remap[cgt]
    m = gsel >= 0
    cgt_c = gsel[m]; cciou_c = cciou[m]; ccls_c = ccls[m]
    calign_c = calign[m]; cstride_c = cstride[m]; coffx_c = coffx[m]; coffy_c = coffy[m]
    print(f"  uid 折叠后：GT {len(uid_c)} / 候选 {len(cgt_c)}")

    A = np.array([cls_of(v) for v in area_c])
    _sm = float(np.median(z["gt_maxciou"][keep][A == "small"])) if (A == "small").any() else float("nan")
    _lg = float(np.median(z["gt_maxciou"][keep][A == "large"])) if (A == "large").any() else float("nan")
    AR = np.asarray(bw[keep] / np.maximum(bh[keep], 1e-9), dtype=np.float64)
    SQ = np.sqrt(area_c)

    # ---------------- §4 BASELINE REPRODUCTION ----------------
    print("\n" + "=" * 104)
    print("§4 BASELINE REPRODUCTION（native-area 分桶；目标见协议 §3）")
    print("=" * 104)
    TARGET = {"small": (250, 4.0, 0.8805, 0.354), "medium": (934, 20.0, 0.9610, 0.703),
              "large": (736, 169.0, 0.9795, 0.826)}
    print(f"  {'class':>8} {'n':>6} {'n_cand med':>11} {'maxCIoU med':>12} {'maxCls med':>11} "
          f"{'bestAlign med':>14} {'zero-cand%':>11} | {'目标 n/ncand/CIoU/align':>28} {'ok':>4}")
    gate_ok = True
    B = {}
    for c in ("small", "medium", "large"):
        s = A == c
        d = dict(n=int(s.sum()), n_cand=float(np.median(ncand_c[s])), maxciou=float(np.median(mciou_c[s])),
                 maxcls=float(np.median(mcls_c[s])), align=float(np.median(malign_c[s])),
                 zero=float(zero_c[s].mean()))
        B[c] = d
        t = TARGET[c]
        ok = (abs(d["n_cand"] - t[1]) <= 1.5 and abs(d["maxciou"] - t[2]) <= 0.02
              and abs(d["align"] - t[3]) <= 0.05)
        gate_ok &= ok
        print(f"  {c:>8} {d['n']:>6} {d['n_cand']:>11.1f} {d['maxciou']:>12.4f} {d['maxcls']:>11.4f} "
              f"{d['align']:>14.4f} {100*d['zero']:>10.2f}% | "
              f"{t[0]:>7}/{t[1]:>5}/{t[2]:>7.4f}/{t[3]:>6.3f} {str(ok):>4}")
    print(f"\n  >>> BASELINE REPRODUCTION {'PASS' if gate_ok else 'FAIL'}")
    if not gate_ok:
        print("  >>> 按协议 §3：结构性偏离 ⇒ STOP，不进入反事实。")
        return 1

    # ---------------- §5 candidate geometry baseline ----------------
    print("\n" + "=" * 104)
    print("§5 CANDIDATE GEOMETRY BASELINE（分布，非仅中位数）")
    print("=" * 104)
    for c in ("small", "medium", "large"):
        s = A == c
        for name, arr in (("n_cand", ncand_c[s]), ("maxCIoU", mciou_c[s]), ("best_align", malign_c[s]),
                          ("maxCls", mcls_c[s])):
            d = desc(arr)
            print(f"  {c:>7} {name:>10}: n={d['n']:<4} mean={d['mean']:.4f} sd={d['sd']:.4f} "
                  f"p10={d['p10']:.4f} p25={d['p25']:.4f} **p50={d['median']:.4f}** "
                  f"p75={d['p75']:.4f} p90={d['p90']:.4f}")
        print(f"  {c:>7} {'P(CIoU>=':>10}: "
              + "  ".join(f"{t}={100*frac_ge(mciou_c[s], t):.1f}%" for t in (0.3, 0.5, 0.7, 0.8)))
        print()

    # ---------------- 反事实 ----------------
    big = np.flatnonzero(A == "large")
    small = np.flatnonzero(A == "small")
    # per-GT 候选切片
    starts = np.searchsorted(cgt_c, np.arange(len(uid_c)))
    ends = np.searchsorted(cgt_c, np.arange(len(uid_c)), side="right")

    def slice_of(g):
        return slice(starts[g], ends[g])

    def stats(idx_ciou, idx_cls, idx_align):
        return float(idx_ciou.max()), float(idx_cls.max()), float(idx_align.max())

    r4_ciou = np.full((len(big), len(SEEDS)), np.nan)
    r4_cls = np.full_like(r4_ciou, np.nan)
    r4_al = np.full_like(r4_ciou, np.nan)
    best4_ciou = np.full(len(big), np.nan); best4_cls = np.full_like(best4_ciou, np.nan)
    best4_al = np.full_like(best4_ciou, np.nan); best4_4th = np.full_like(best4_ciou, np.nan)
    s8_ciou = np.full(len(big), np.nan); s8_cls = np.full_like(s8_ciou, np.nan)
    s8_al = np.full_like(s8_ciou, np.nan); s8_used = np.zeros(len(big), dtype=bool)
    ncand_big = ncand_c[big].astype(np.int64)

    for j, g in enumerate(big):
        sl = slice_of(g)
        ci = cciou_c[sl]; cl = ccls_c[sl]; aa = calign_c[sl]
        n = ci.size
        if n == 0:
            continue
        u = int(uid_c[g])
        for k, sd in enumerate(SEEDS):
            if n <= 4:
                pick = np.arange(n)
            else:
                rng = np.random.default_rng([SEEDS[0] + sd, u])
                pick = rng.choice(n, size=4, replace=False)
            r4_ciou[j, k] = ci[pick].max(); r4_cls[j, k] = cl[pick].max(); r4_al[j, k] = aa[pick].max()
        if n >= 4:
            o = np.argsort(-ci)[:4]
            best4_ciou[j] = ci[o].max(); best4_cls[j] = cl[o].max(); best4_al[j] = aa[o].max()
            best4_4th[j] = ci[o].min()
        st8 = cstride_c[sl] == 8.0
        if st8.sum() >= 4:
            r = np.sqrt(coffx_c[sl][st8] ** 2 + coffy_c[sl][st8] ** 2)
            o8 = np.argsort(r)[:4]
            ci8, cl8, al8 = ci[st8][o8], cl[st8][o8], aa[st8][o8]
            s8_ciou[j] = ci8.max(); s8_cls[j] = cl8.max(); s8_al[j] = al8.max(); s8_used[j] = True

    def block(tag, x):
        d = desc(x[np.isfinite(x)])
        print(f"  {tag:<28} n={d['n']:<4} mean={d['mean']:.4f} sd={d['sd']:.4f} p10={d['p10']:.4f} "
              f"p25={d['p25']:.4f} **p50={d['median']:.4f}** p75={d['p75']:.4f} p90={d['p90']:.4f}  "
              + " ".join(f"P(>={t})={100*frac_ge(x[np.isfinite(x)], t):.1f}%" for t in (0.5, 0.7, 0.8)))
        return d

    print("=" * 104)
    print("§9 CORE TABLE — candidate 数量效应（maxCIoU 为主判据）")
    print("=" * 104)
    print(f"  {'regime':<28} n_cand 分布 / maxCIoU 分布")
    print(f"  {'Small native':<28} n_cand med={np.median(ncand_c[small]):.1f}")
    block("Small native", mciou_c[small])
    print(f"  {'Large native':<28} n_cand med={np.median(ncand_big):.1f}")
    block("Large native", mciou_c[big])
    # CF1: per-GT median over seeds
    r4m = np.nanmedian(r4_ciou, axis=1)
    print(f"  {'Large -> Random-4':<28} n_cand = 4 (抽自真实 set)")
    block("Large -> Random-4", r4m)
    print(f"  {'Large -> Best-4':<28} n_cand = 4 (按 CIoU 取前 4)")
    block("Large -> Best-4", best4_ciou)
    nused = int(s8_used.sum())
    print(f"  {'Large -> Stride8-4':<28} n_cand = 4 (stride-8 中离中心最近; 覆盖 {nused}/{len(big)})")
    block("Large -> Stride8-4", s8_ciou[s8_used])

    print("\n  maxCls 与 best_align（辅助指标；注意 best_align 受 alpha/beta 与 cls 影响，"
          "不可直接解读为 geometry quality）")
    print(f"  {'regime':<28} {'maxCls med':>11} {'bestAlign med':>14}")
    for tag, xc, xa in (("Small native", mcls_c[small], malign_c[small]),
                        ("Large native", mcls_c[big], malign_c[big]),
                        ("Large -> Random-4", np.nanmedian(r4_cls, axis=1), np.nanmedian(r4_al, axis=1)),
                        ("Large -> Best-4", best4_cls, best4_al),
                        ("Large -> Stride8-4", s8_cls[s8_used], s8_al[s8_used])):
        print(f"  {tag:<28} {np.nanmedian(xc):>11.4f} {np.nanmedian(xa):>14.4f}")

    print(f"\n  CF2 退化说明：Best-4 的 maxCIoU 恒等于 Large native 的 maxCIoU（取前 4 后取 max = 全局 max）——")
    print(f"    实测 {np.nanmedian(best4_ciou):.4f} vs native {np.median(mciou_c[big]):.4f} ⇒ CF2 对 maxCIoU 无信息。")
    print(f"    CF2 有信息的量是**前 4 的 CIoU 下界（第 4 高）**：中位 {np.nanmedian(best4_4th):.4f}")

    # ---------------- §5b 逐候选质量（与 count 无关的比较）----------------
    print("\n" + "=" * 104)
    print("§5b CANDIDATE-LEVEL QUALITY（逐候选，**与 candidate 数量无关**）")
    print("=" * 104)
    print("  这是分离「数量」与「质量」最干净的量：per-candidate 均值/中位不受 n_cand 影响。")
    print(f"  {'class':>8} {'n_cand_total':>13} {'CIoU mean':>10} {'CIoU med':>9} {'CIoU p10':>9} "
          f"{'CIoU p90':>9} | {'cls mean':>9} {'cls med':>9} | " +
          "  ".join(f"P(CIoU>={t})" for t in (0.3, 0.5, 0.7, 0.8)))
    cand_cls_of = np.array([cls_of(area_c[g]) for g in cgt_c])
    for c in ("small", "medium", "large"):
        sel = cand_cls_of == c
        x = cciou_c[sel]; y = ccls_c[sel]
        print(f"  {c:>8} {int(sel.sum()):>13} {x.mean():>10.4f} {np.median(x):>9.4f} "
              f"{q(x,10):>9.4f} {q(x,90):>9.4f} | {y.mean():>9.4f} {np.median(y):>9.4f} | " +
              "  ".join(f"{100*frac_ge(x,t):>9.1f}%" for t in (0.3, 0.5, 0.7, 0.8)))

    # ---------------- sampling curve：E[max CIoU over k] ----------------
    print("\n" + "=" * 104)
    print("§5c SAMPLING CURVE — Large GT 的 E[max CIoU over k 个随机候选]（数量的 order-statistics 曲线）")
    print("=" * 104)
    KS = [1, 2, 4, 8, 16, 32, 64, 128]
    curve = {}
    for k in KS:
        vals = []
        for j, g in enumerate(big):
            sl = slice_of(g)
            ci = cciou_c[sl]
            if ci.size == 0:
                continue
            if ci.size <= k:
                vals.append(float(ci.max()))
            else:
                med = np.median([
                    ci[np.random.default_rng([SEEDS[0] + sd, int(uid_c[g])]).choice(ci.size, k, replace=False)].max()
                    for sd in range(30)])
                vals.append(float(med))
        curve[k] = float(np.median(vals))
        print(f"  k={k:>4}   E[max CIoU] median = {curve[k]:.4f}")
    need = next((k for k in KS if curve[k] >= _sm), None)
    print(f"  Small native maxCIoU 中位 = {_sm:.4f}  ⇒ 用大目标的候选池要达到同一水平需 k ≈ "
          f"{need if need is not None else '>128'}")

    # ---------------- CF4 coverage-matched 4（附加诊断，明确定义）----------------
    print("\n" + "=" * 104)
    print("§5d CF4 — Large → Coverage-matched 4（**本审计额外构造**，非协议要求）")
    print("=" * 104)
    print("  定义：在归一化偏移空间 [-1,1]^2 里取 4 个象限中心 (±0.5,±0.5)，每个象限取最近候选，")
    print("        模拟『4 个候选覆盖整个目标』——这才是 small 的真实处境（4 个候选铺满小框）。")
    qc = [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)]
    cf4_ciou = np.full(len(big), np.nan); cf4_cls = np.full_like(cf4_ciou, np.nan)
    cf4_al = np.full_like(cf4_ciou, np.nan)
    for j, g in enumerate(big):
        sl = slice_of(g)
        ox = coffx_c[sl]; oy = coffy_c[sl]
        if ox.size < 4:
            continue
        picks = []
        for (qx, qy) in qc:
            d = (ox - qx) ** 2 + (oy - qy) ** 2
            picks.append(int(np.argmin(d)))
        picks = np.unique(picks)
        ci4 = cciou_c[sl][picks]; cl4 = ccls_c[sl][picks]; al4 = calign_c[sl][picks]
        cf4_ciou[j] = ci4.max(); cf4_cls[j] = cl4.max(); cf4_al[j] = al4.max()
    print(f"  {'regime':<28} {'n':>5} {'maxCIoU med':>12} {'maxCls med':>11} {'bestAlign med':>14}")
    for tag, x, y, z_ in (("Small native", mciou_c[small], mcls_c[small], malign_c[small]),
                          ("Large native", mciou_c[big], mcls_c[big], malign_c[big]),
                          ("Large Random-4", r4m, np.nanmedian(r4_cls, axis=1), np.nanmedian(r4_al, axis=1)),
                          ("Large Best-4", best4_ciou, best4_cls, best4_al),
                          ("Large Stride8-4", s8_ciou[s8_used], s8_cls[s8_used], s8_al[s8_used]),
                          ("Large Coverage-4", cf4_ciou, cf4_cls, cf4_al)):
        xx = x[np.isfinite(x)]
        print(f"  {tag:<28} {xx.size:>5} {np.median(xx):>12.4f} {np.nanmedian(y):>11.4f} "
              f"{np.nanmedian(z_):>14.4f}")
    spread = [np.nanmedian(r4m), np.nanmedian(s8_ciou[s8_used]), np.nanmedian(cf4_ciou)]
    print("\n  ⚠ **protocol sensitivity**：同为『4 个候选』，三种选法给出 "
          f"{spread[0]:.4f} / {spread[1]:.4f} / {spread[2]:.4f}（random / stride8 / coverage）")
    print(f"    跨度 {max(spread)-min(spread):.4f}  ⇒ CF 结论对 selection protocol 高度敏感（协议 §12 CASE D 的条件之一）")

    # ---------------- §11 Monte-Carlo uncertainty ----------------
    print("\n" + "=" * 104)
    print("§11 MONTE-CARLO UNCERTAINTY（CF1：seed 0..99 固定；per-GT 分布，不把抽样当独立 GT）")
    print("=" * 104)
    permean = np.nanmean(r4_ciou, axis=1)
    permed = np.nanmedian(r4_ciou, axis=1)
    print(f"  per-GT mean over 100 seeds : mean={np.nanmean(permean):.4f} sd={np.nanstd(permean, ddof=1):.4f} "
          f"p5={q(permean,5):.4f} p50={q(permean,50):.4f} p95={q(permean,95):.4f}")
    print(f"  per-GT median over 100 seeds: mean={np.nanmean(permed):.4f} sd={np.nanstd(permed, ddof=1):.4f} "
          f"p5={q(permed,5):.4f} p50={q(permed,50):.4f} p95={q(permed,95):.4f}")
    print(f"  总体（把 GT 与 seed 都当样本）: mean={np.nanmean(r4_ciou):.4f} sd={np.nanstd(r4_ciou):.4f} "
          f"p50={q(r4_ciou[~np.isnan(r4_ciou)],50):.4f}")

    # ---------------- Δ 与 CASE 判定 ----------------
    print("\n" + "=" * 104)
    print("§12 Δ 与 CASE 判定（主判据 = maxCIoU 中位）")
    print("=" * 104)
    sm = float(np.median(mciou_c[small])); lg = float(np.median(mciou_c[big]))
    r4 = float(np.nanmedian(r4m)); s8 = float(np.nanmedian(s8_ciou[s8_used]))
    print(f"  Small native      maxCIoU med = {sm:.4f}")
    print(f"  Large native      maxCIoU med = {lg:.4f}")
    print(f"  Large Random-4    maxCIoU med = {r4:.4f}   (Δ vs Large native = {r4-lg:+.4f})")
    print(f"  Large Stride8-4   maxCIoU med = {s8:.4f}   (Δ vs Large native = {s8-lg:+.4f})")
    print(f"  Small − Large_Random4 = {sm-r4:+.4f}")
    d_rand = lg - r4          # candidate-count 能解释多少
    resid = r4 - sm           # 数量效应解释后仍残留的差距
    print(f"\n  candidate-count 能解释的降幅 Δ_Random4 = {d_rand:+.4f} (占 native 的 {100*d_rand/lg:.1f}%)")
    print(f"  数量补偿后 Small 与 Large 仍相差 = {resid:+.4f}  (Small/Large_Random4 = {sm/r4:.4f})")

    tol = 0.02   # 预先声明的等价带（maxCIoU 绝对差）
    # 协议 §12 的三条 CASE 都默认 Large-Random4 落在 Small 与 Large-native **之间**。
    # 实测出现 **overshoot**（Random4 落到 Small 之下）——协议未覆盖该情形，故显式说明判定依据。
    if r4 < sm - tol:
        case = "A — STRONG SUPPORT（overshoot：Small > Large_Random4）"
        why = ("Large-Random4 落到 Small **之下** ⇒ 同等候选数下 small 的 maxCIoU 反而更高，"
               "即 small 的候选质量并不更差；large 的高 maxCIoU 来自 169 个候选的 order statistics。")
    elif abs(sm - r4) <= tol and d_rand > tol:
        case = "A — STRONG SUPPORT"
        why = "Large-Random4 落在 Small 水平，且 Large-native >> Large-Random4。"
    elif abs(r4 - lg) <= tol:
        case = "C — NOT SUPPORTED"
        why = "减少候选数几乎不改变 Large 的 maxCIoU ⇒ 数量不是解释。"
    elif resid > tol and d_rand > tol:
        case = "B — PARTIAL SUPPORT"
        why = "数量影响 large，但不能解释 small 与 large 的全部差距。"
    else:
        case = "D — UNRESOLVED"
        why = "分布重叠大 / 无法 size matching / 反事实对 protocol 敏感。"
    print(f"\n  等价带 tol = {tol}（预先声明）")
    print(f"  >>> CASE {case}")
    print(f"      理由：{why}")

    # ---------------- §9 size / aspect matching ----------------
    print("\n" + "=" * 104)
    print("§9 SIZE / ASPECT-RATIO MATCHING")
    print("=" * 104)
    print(f"  native sqrt: small [{np.nanmin(SQ[small]):.1f},{np.nanmax(SQ[small]):.1f}] "
          f"median {np.median(SQ[small]):.1f} | large [{np.nanmin(SQ[big]):.1f},{np.nanmax(SQ[big]):.1f}] "
          f"median {np.median(SQ[big]):.1f}")
    print("  ⇒ native 尺寸分层按定义不相交（small<32px、large>96px）⇒ **literal size matching 不可能**，")
    print("    不做伪造匹配。CF1/CF2/CF3 均为 **within-GT** 反事实，尺寸/位置/上下文被完全固定。")
    ars = np.asarray(bw[keep] / np.maximum(bh[keep], 1e-9), dtype=np.float64)
    qs = np.percentile(ars[small], [25, 50, 75])
    print(f"\n  可做的匹配 = aspect-ratio 分层（small 的 AR 四分位：{qs[0]:.3f}/{qs[1]:.3f}/{qs[2]:.3f}）")
    print(f"  {'AR 层':>18} {'n_small':>8} {'n_large':>8} {'Small maxCIoU':>14} {'LR4 maxCIoU':>12} "
          f"{'Large maxCIoU':>14}")
    strata = [(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)]
    for lo, hi in strata:
        ms = small[(ars[small] > lo) & (ars[small] <= hi)]
        pos = np.flatnonzero((ars[big] > lo) & (ars[big] <= hi))   # 在 big 内的**位置**
        if ms.size == 0 or pos.size == 0:
            continue
        mb = big[pos]
        print(f"  [{lo:>6.2f},{hi:>6.2f}] {ms.size:>8} {mb.size:>8} "
              f"{np.median(mciou_c[ms]):>14.4f} {np.nanmedian(r4m[pos]):>12.4f} "
              f"{np.median(mciou_c[mb]):>14.4f}")

    json.dump(dict(baseline=B, gate_pass=bool(gate_ok), case=case,
                   core=dict(small_maxciou=sm, large_maxciou=lg, large_random4=r4,
                             large_stride8_4=s8, delta_random4=d_rand, residual=resid,
                             ratio_small_over_lr4=sm / r4, tol=tol),
                   mc_uncertainty=dict(per_gt_mean=desc(permean), per_gt_median=desc(permed)),
                   cf2_degenerate_note="Best-4 的 maxCIoU 恒等于 native max",
                   cf2_4th_ciou_median=float(np.nanmedian(best4_4th)),
                   stride8_coverage=int(nused), n_large=int(len(big))),
              open(OUT / "_tables.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {OUT/'_tables.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
