"""_rank_decomp.py — 决定性判别：把「几何不合格」与「cls 压低」拆开（只读，零 GPU）。

只读 input：diagnostic/candidate_density_counterfactual/_results.npz（train split）
核心：对『CIoU>=0.50 且 cls<0.001』的候选，分别计算它在池内的
      **CIoU 排名**（纯几何）与 **align 排名**（几何+cls），以及两者的 top-10 命中率差。
      · 若 CIoU 排名 也排不进 top-10 ⇒ 几何自身就不够 ⇒ cls 不是该候选被排除的原因
      · 若 CIoU 排名 在 top-10 内 而 align 排名 掉出 ⇒ cls 项是决定性的
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
NPZ = ROOT / "diagnostic/candidate_density_counterfactual/_results.npz"
ALPHA, BETA, TOPK = 0.5, 6.0, 10
HIGH_IOU, HIGH_CLS = 0.50, 0.001
say = print
say("=" * 100); say("决定性判别：几何排名 vs align 排名"); say("=" * 100)
say(f"  input: {NPZ.relative_to(ROOT)}  sha={hashlib.sha256(NPZ.read_bytes()).hexdigest()}")

z = np.load(NPZ)
ciou = z["cand_ciou"].astype(np.float64); cls = z["cand_cls"].astype(np.float64)
al = z["cand_align"].astype(np.float64); gt = z["cand_gt"].astype(np.int64)
align_raw = cls ** ALPHA * ciou ** BETA
POS = al > 0
o = np.argsort(gt, kind="stable")
gs, co, cs, aa, ob = gt[o], ciou[o], cls[o], align_raw[o], POS[o]
b = np.searchsorted(gs, np.arange(len(z["gt_uid"]) + 1))

rk_c, rk_a, pos_c, key_c, pos_k, key_k = [], [], [], [], [], []
for i in range(len(b) - 1):
    lo, hi = b[i], b[i + 1]
    if hi <= lo:
        continue
    c_, k_, a_, o_ = co[lo:hi], cs[lo:hi], aa[lo:hi], ob[lo:hi]
    key = (c_ >= HIGH_IOU) & (k_ < HIGH_CLS)
    rc = np.empty(len(c_)); rc[np.argsort(-c_, kind="stable")] = np.arange(1, len(c_) + 1)
    ra = np.empty(len(c_)); ra[np.argsort(-a_, kind="stable")] = np.arange(1, len(c_) + 1)
    if key.any():
        rk_c.append(rc[key]); rk_a.append(ra[key]); key_c.append(c_[key]); key_k.append(k_[key])
    if o_.any():
        pos_c.append(c_[o_]); pos_k.append(k_[o_])
rc = np.concatenate(rk_c); ra = np.concatenate(rk_a)
pc = np.concatenate(pos_c); pk = np.concatenate(pos_k)
kc = np.concatenate(key_c); kk = np.concatenate(key_k)
pm = float(np.median(pc))
say(f"\n  关键组候选数 = {len(rc):,}")
say(f"  {'排名依据':<22}{'p25':>6}{'median':>8}{'p75':>6}{'<=top10 的比例':>16}")
say(f"  {'池内 CIoU 排名（纯几何）':<22}{np.percentile(rc,25):>6.0f}{np.median(rc):>8.0f}{np.percentile(rc,75):>6.0f}{(rc<=TOPK).mean():>16.2%}")
say(f"  {'池内 align 排名（几何+cls）':<22}{np.percentile(ra,25):>6.0f}{np.median(ra):>8.0f}{np.percentile(ra,75):>6.0f}{(ra<=TOPK).mean():>16.2%}")
say(f"\n  正样本 (n={len(pc):,}): CIoU median={np.median(pc):.3f}  cls median={np.median(pk):.4f}")
say(f"  关键组 (n={len(kc):,}): CIoU median={np.median(kc):.3f}  cls median={np.median(kk):.2e}")
say(f"  CIoU^6 比（正样本/关键组）= {(np.median(pc)/max(np.median(kc),1e-9))**BETA:,.1f}x")

cf = []
for lab, mode in (("cls_only_to_0.1", "cls"), ("ciou_only_to_posmedian", "ciou"), ("both", "both")):
    n_in = 0; n_tot = 0
    for i in range(len(b) - 1):
        lo, hi = b[i], b[i + 1]
        if hi <= lo:
            continue
        c_, k_, a_, o_ = co[lo:hi], cs[lo:hi], aa[lo:hi], ob[lo:hi]
        key = (c_ >= HIGH_IOU) & (k_ < HIGH_CLS)
        if not key.any():
            continue
        c2, k2 = c_.copy(), k_.copy()
        if mode in ("cls", "both"):
            k2[key] = 0.1
        if mode in ("ciou", "both"):
            c2[key] = pm
        a2 = k2 ** ALPHA * c2 ** BETA
        t = np.argsort(-a2, kind="stable")[:min(TOPK, hi - lo)]
        it = np.zeros(hi - lo, bool); it[t] = True
        n_in += int((key & it).sum()); n_tot += int(key.sum())
    hr = n_in / max(n_tot, 1)
    say(f"  反事实 {lab:<26} 进入 top-10 = {n_in:,}/{n_tot:,} = {hr:.2%}")
    cf.append(dict(mode=lab, n_cand=n_tot, n_in_topk=n_in, hit_rate=hr))

rows = [dict(cutoff=TOPK, rank_by="CIoU(geometry only)", hit_rate=float((rc <= TOPK).mean()),
             median_rank=float(np.median(rc))),
        dict(cutoff=TOPK, rank_by="align(geometry x cls)", hit_rate=float((ra <= TOPK).mean()),
             median_rank=float(np.median(ra)))]
with open(OUT / "rank_decomposition.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump(dict(n_key_cand=int(len(rc)),
               hit_rate_ranked_by_ciou=float((rc <= TOPK).mean()),
               hit_rate_ranked_by_align=float((ra <= TOPK).mean()),
               median_rank_ciou=float(np.median(rc)), median_rank_align=float(np.median(ra)),
               positive_ciou_median=float(np.median(pc)), positive_cls_median=float(np.median(pk)),
               key_ciou_median=float(np.median(kc)), key_cls_median=float(np.median(kk)),
               counterfactual=cf),
          open(OUT / "_rank_decomp.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
say("\nDONE")
