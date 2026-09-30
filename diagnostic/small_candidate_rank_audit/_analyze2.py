"""_analyze2.py — 修正版分析（纯离线只读）

修掉 _analyze.py 的两个**方法学错误**（保留其日志作为过程记录）：

错误 1 — `max_CIoU@K` 是**同义反复**。
    Rank-A 按 CIoU 降序排列后，前 K 个的 max 恒等于 rank-1 的值。
    实测 max@K 对 K=1..16 **完全相同（0.8808，Δ=0.0000，改善率 0.0%）**——这不是发现，是定义使然。
    ⇒ 「max@K 是否上升」**不可能**用来判断 headroom。真正的含义应是：
       新增候选只能**增加正样本数量**，不可能提高 max（因为当前 max 已由池内最优候选取得）。

错误 2 — 逐 rank 列**跨 rank 不可比**。
    rank k 只对 n_cand>=k 的 GT 有定义，eligible n 从 244 降到 33，
    被选中的是「框内容纳更多 anchor」的那一小撮 GT ⇒ 逐 rank 中位数把「群体变化」读成了「rank 质量」。

本脚本给出**可比**的口径：
  A) 固定总体（n_cand>=16，n=33）上的逐 rank 表 —— 同一批 GT，rank 之间可比。
  B) 池内 headroom 的正确度量：**当前 topk=10 未选中的候选**（rank>10）的质量与数量。
  C) stride / 尺寸 / AR 分层。
  D) 明确记录「新增 anchor（框外/更细网格）是否存在」**不可回答**（Situation A vs B）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AUDIT = Path(__file__).resolve().parent
SRC = AUDIT.parent / "candidate_density_counterfactual" / "_results.npz"
TOPK = 10
MAXR = 16


def q(x, p):
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if x.size else float("nan")


def fg(x, t):
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    return float((x >= t).mean()) if x.size else float("nan")


def main():
    z = np.load(SRC)
    uid = z["gt_uid"]; area = z["gt_native_area"]; bw = z["gt_bw"]; bh = z["gt_bh"]
    cgt = z["cand_gt"]
    cciou = z["cand_ciou"]; ccls = z["cand_cls"]; calign = z["cand_align"]
    cstr = np.asarray(z["cand_stride"]).reshape(-1)

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

    print("=" * 108)
    print("SMALL CANDIDATE RANK AUDIT — 修正版（纯离线只读）")
    print("=" * 108)

    # ---- §2 数据充分性 ----
    print("\n[§2] 数据充分性")
    print(f"  candidate 定义 = mask_in_gts==True（只含**框内 anchor 中心**；框外 anchor 未保存）")
    print(f"  native-small n={n_sm}；n_cand: p50={np.median(ncand_c[sm]):.0f} "
          f"p75={np.percentile(ncand_c[sm],75):.0f} p90={np.percentile(ncand_c[sm],90):.0f} "
          f"max={ncand_c[sm].max()}")
    for thr in (5, 8, 12, 16):
        print(f"    n_cand>={thr:>2}: {int((ncand_c[sm]>=thr).sum()):>4}/{n_sm} "
              f"({100*(ncand_c[sm]>=thr).mean():5.1f}%)")
    print(f"    n_cand>topk({TOPK}): {int((ncand_c[sm]>TOPK).sum()):>4}/{n_sm} "
          f"({100*(ncand_c[sm]>TOPK).mean():5.1f}%)  ← 只有这些 GT 有『未被选中的池内候选』")

    # 排序（Rank-A = CIoU 降序）
    R = {}
    for j, g in enumerate(sm):
        sl = slice(starts[g], ends[g])
        ci = cciou_c[sl]; n = ci.size
        if n == 0:
            continue
        o = np.argsort(-ci)
        R[j] = dict(ci=ci[o], cl=ccls_c[sl][o], al=calign_c[sl][o], st=cstr_c[sl][o])

    # ---- A) 固定总体逐 rank ----
    fixed = [j for j in R if R[j]["ci"].size >= MAXR]
    print("\n" + "=" * 108)
    print(f"[A] 固定总体（n_cand>={MAXR}，n={len(fixed)}）逐 rank —— **同一批 GT，rank 间可比**")
    print("=" * 108)
    print(f"  {'rank':>4} {'CIoU med':>9} {'mean':>7} {'p25':>7} {'p75':>7} | {'P(≥.5)':>7} {'P(≥.7)':>7} "
          f"{'P(≥.8)':>7} | {'cls med':>8} {'P(cls≥.1)':>9} {'P(cls≥.3)':>9} | {'stride med':>10}")
    fx = {}
    for k in range(1, MAXR + 1):
        ci = np.array([R[j]["ci"][k - 1] for j in fixed])
        cl = np.array([R[j]["cl"][k - 1] for j in fixed])
        st = np.array([R[j]["st"][k - 1] for j in fixed])
        fx[k] = dict(ciou_med=q(ci, 50), mean=float(ci.mean()), p25=q(ci, 25), p75=q(ci, 75),
                     p_ge={str(t): fg(ci, t) for t in (0.5, 0.7, 0.8)},
                     cls_med=q(cl, 50),
                     p_cls_ge={str(t): fg(cl, t) for t in (0.1, 0.3)},
                     stride_med=float(np.median(st)))
        print(f"  {k:>4} {fx[k]['ciou_med']:>9.4f} {fx[k]['mean']:>7.4f} {fx[k]['p25']:>7.4f} "
              f"{fx[k]['p75']:>7.4f} | " + " ".join(f"{100*fx[k]['p_ge'][str(t)]:>6.1f}%"
                                                    for t in (0.5, 0.7, 0.8)) + " | "
              f"{fx[k]['cls_med']:>8.4f} " + " ".join(f"{100*fx[k]['p_cls_ge'][str(t)]:>8.1f}%"
                                                      for t in (0.1, 0.3)) +
              f" | {fx[k]['stride_med']:>10.1f}")

    # ---- B) 池内 headroom（正确口径）----
    print("\n" + "=" * 108)
    print("[B] 池内 HEADROOM —— 正确口径：**当前 topk=%d 未选中的池内候选**" % TOPK)
    print("=" * 108)
    print("  ⚠ 先记录一个同义反复（不是发现）：Rank-A 下 max_CIoU@K 对一切 K 恒等于 rank-1。")
    print("     ⇒ **新增候选不可能提高 max**（当前 max 已由池内最优候选取得）。")
    print("     ⇒ 候选密度若要起作用，只能通过**增加正样本数量**，不能通过抬高 max。")
    unused_all_ci, unused_all_cl, unused_all_st = [], [], []
    n_gt_with_unused = 0
    per_gt_useful = {0.5: 0, 0.7: 0}
    per_gt_unused_cnt = []
    for j in R:
        ci = R[j]["ci"]; cl = R[j]["cl"]; st = R[j]["st"]
        if ci.size <= TOPK:
            continue
        n_gt_with_unused += 1
        u_ci = ci[TOPK:]; u_cl = cl[TOPK:]; u_st = st[TOPK:]
        unused_all_ci.append(u_ci); unused_all_cl.append(u_cl); unused_all_st.append(u_st)
        per_gt_unused_cnt.append(u_ci.size)
        for thr in (0.5, 0.7):
            if ((u_ci >= thr) & (u_cl >= 0.1)).any():
                per_gt_useful[thr] += 1
    if unused_all_ci:
        U_ci = np.concatenate(unused_all_ci); U_cl = np.concatenate(unused_all_cl)
        U_st = np.concatenate(unused_all_st)
        print(f"\n  有未选中候选的 small GT：{n_gt_with_unused} / {n_sm} "
              f"({100*n_gt_with_unused/n_sm:.1f}%)；其未选中候选共 {U_ci.size} 个 "
              f"（每 GT 中位 {np.median(per_gt_unused_cnt):.0f}）")
        print(f"  未选中候选的 CIoU: med={q(U_ci,50):.4f} p25={q(U_ci,25):.4f} p75={q(U_ci,75):.4f}  "
              f"P(≥.5)={100*fg(U_ci,0.5):.1f}%  P(≥.7)={100*fg(U_ci,0.7):.1f}%")
        print(f"  未选中候选的 cls : med={q(U_cl,50):.4f}  P(≥.1)={100*fg(U_cl,0.1):.1f}%  "
              f"P(≥.3)={100*fg(U_cl,0.3):.1f}%")
        print(f"  未选中候选的双门槛 useful: CIoU≥0.5&cls≥0.1 = "
              f"{100*((U_ci>=0.5)&(U_cl>=0.1)).mean():.1f}% ;  CIoU≥0.7&cls≥0.1 = "
              f"{100*((U_ci>=0.7)&(U_cl>=0.1)).mean():.1f}%")
        print(f"  per-GT 至少 1 个 useful 未选中候选: CIoU≥0.5&cls≥0.1 → "
              f"{100*per_gt_useful[0.5]/max(n_gt_with_unused,1):.1f}% of those GTs "
              f"({per_gt_useful[0.5]}/{n_gt_with_unused}) ; ≥0.7 → "
              f"{100*per_gt_useful[0.7]/max(n_gt_with_unused,1):.1f}%")
        print(f"  未选中候选的 stride 构成: "
              + "  ".join(f"stride{int(s)}={100*(U_st==s).mean():.1f}%" for s in (8, 16, 32)))
        print(f"\n  ⇒ 但注意分母：只有 {100*n_gt_with_unused/n_sm:.1f}% 的 native-small GT 存在未选中候选；"
              f"\n     其余 {100*(n_sm-n_gt_with_unused)/n_sm:.1f}% 的 small GT 其**全部**候选都已被选中"
              f"（n_cand≤{TOPK}）⇒ 对它们，池内 headroom **恰好为 0**。")
    else:
        print("  无任何 small GT 存在未选中候选。")

    # ---- C) 分层 ----
    sq = np.sqrt(area_c[sm]); ar = bw_c[sm] / np.maximum(bh_c[sm], 1e-9)
    qs = np.percentile(sq, [25, 50, 75])
    print("\n" + "=" * 108)
    print("[C] 分层（每层给出 eligible16 与**未选中候选**的可用性）")
    print("=" * 108)
    print(f"  {'层':>16} {'n':>4} {'n_cand med':>11} {'elig16':>7} {'有未选中':>9} "
          f"{'未选中 CIoU med':>16} {'未选中 P(≥.5)':>15}")
    def layer_row(nm, mask_idx):
        ii = [j for j, g in enumerate(sm) if mask_idx[j]]
        if not ii:
            return None
        e16 = sum(1 for j in ii if R.get(j) and R[j]["ci"].size >= MAXR)
        uu = [(R[j]["ci"][TOPK:], R[j]["cl"][TOPK:]) for j in ii
              if R.get(j) and R[j]["ci"].size > TOPK]
        n_u = len(uu)
        cc = np.concatenate([a for a, _ in uu]) if n_u else np.array([])
        r = dict(layer=nm, n=len(ii), ncand_med=float(np.median(ncand_c[sm][ii])), elig16=e16,
                 n_with_unused=n_u, unused_ciou_med=q(cc, 50), unused_p_ge5=fg(cc, 0.5))
        print(f"  {nm:>16} {r['n']:>4} {r['ncand_med']:>11.0f} {e16:>7} {n_u:>9} "
              f"{r['unused_ciou_med']:>16.4f} {100*r['unused_p_ge5']:>14.1f}%")
        return r
    size_rows = []
    for nm, msk in (("<S1 (p25)", sq <= qs[0]), ("S2 (25-50%)", (sq > qs[0]) & (sq <= qs[1])),
                    ("S3 (50-75%)", (sq > qs[1]) & (sq <= qs[2])), (">S4 (p75+)", sq > qs[2])):
        r = layer_row(nm, msk)
        if r:
            size_rows.append(r)
    ar_rows = []
    for nm, msk in (("AR<0.5", ar < 0.5), ("0.5<=AR<1", (ar >= 0.5) & (ar < 1)),
                    ("1<=AR<2", (ar >= 1) & (ar < 2)), ("AR>=2", ar >= 2)):
        r = layer_row(nm, msk)
        if r:
            ar_rows.append(r)

    verdict = "D — UNRESOLVED"
    reason = ("现有 artifact 只保存 select_candidates_in_gts==True 的候选（框内 anchor 中心），"
              "框外 anchor 完全没有保存 ⇒ 无法区分 §14 Situation A（grid 中存在、被中心约束滤掉）"
              "与 Situation B（当前 grid 本就没有）。「增加 candidate density 能否带来新候选」"
              "这个问题在本 artifact 上**不可回答**。")
    json.dump(dict(verdict=verdict, reason=reason,
                   n_small=int(n_sm),
                   eligibility={str(t): int((ncand_c[sm] >= t).sum()) for t in (5, 8, 12, 16)},
                   n_with_unused=int(n_gt_with_unused), topk=TOPK,
                   fixed_population_rank=fx, n_fixed=len(fixed),
                   rule_out_max_headroom="Rank-A 下 max_CIoU@K 恒等于 rank-1 ⇒ 新增候选不能提高 max",
                   size_strata=size_rows, ar_strata=ar_rows),
              open(AUDIT / "_tables2.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {AUDIT / '_tables2.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
