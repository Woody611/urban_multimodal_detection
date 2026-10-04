"""P3 — produce every table the brief asks for, from attribution.json + layers.npz."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
A = json.loads((OUT / "attribution.json").read_text(encoding="utf-8"))
L = np.load(OUT / "layers.npz", allow_pickle=True)
rows = A["rows"]
nA, nB = L["nA"], L["nB"]


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


print("=" * 78)
print("1. PIPELINE ACCOUNTING")
print("=" * 78)
print(f"raw    pre-NMS boxes/img : mean {nA.mean():.2f}  median {np.median(nA):.0f}  max {nA.max()}")
print(f"NMS    post-NMS    /img   : mean {nB.mean():.2f}  median {np.median(nB):.0f}  max {nB.max()}")
top100 = np.minimum(nB, 100)
print(f"top100 retained   /img    : mean {top100.mean():.2f}  median {np.median(top100):.0f}")
print(f"images with >100 post-NMS : {(nB>100).sum()}")
print(f"total pre-NMS {nA.sum()}  post-NMS {nB.sum()}  suppressed/lost {nA.sum()-nB.sum()}"
      f"  ({pct(nA.sum()-nB.sum(), nA.sum()):.2f}% of raw)")

print()
print("=" * 78)
print("2. SMALL-GT DISAPPEARANCE FUNNEL")
print("=" * 78)
for size in ["small", "medium", "large"]:
    s = [r for r in rows if r["size"] == size]
    if not s:
        continue
    fc = Counter(r["failure"] for r in s)
    n = len(s)
    m = fc.get("MATCHED@0.5", 0)
    print(f"\n[{size}] n={n}  official TP@0.5 = {m} ({pct(m,n):.1f}%)")
    for k in ["F0_no_raw_candidate", "F1_nms_disappearance", "F2_top100_disappearance",
              "F3_survives_official_fail"]:
        print(f"   {k:32s} {fc.get(k,0):5d}  {pct(fc.get(k,0),n):6.2f}%")

small = [r for r in rows if r["size"] == "small"]
nonTP = [r for r in small if r["failure"] != "MATCHED@0.5"]
print(f"\n--- small GT ONLY (n={len(small)}), non-TP@0.5 n={len(nonTP)} ---")
fc = Counter(r["failure"] for r in nonTP)
for k, v in fc.most_common():
    print(f"   {k:32s} {v:5d}  {pct(v,len(nonTP)):6.2f}% of non-TP   {pct(v,len(small)):6.2f}% of all small")

print()
print("=" * 78)
print("3. NMS CAUSAL AUDIT  (F1 only: candidate killed by NMS)")
print("=" * 78)
f1 = [r for r in rows if r["failure"] == "F1_nms_disappearance"]
print(f"F1 total (all sizes) = {len(f1)}   raw-candidate loss = {nA.sum()-nB.sum()}")
print("\n>>> CAUSAL TEST (brief §7 C2): of the F1 GTs, how many still have a USABLE box\n"
      "    (same class, IoU>=0.5) surviving NMS+top-100?  Those are NOT real NMS losses.\n")
for size in ["small", "medium", "large"]:
    f = [r for r in f1 if r["size"] == size]
    if not f:
        continue
    usable = [r for r in f if r["post_usable"]]
    lost = [r for r in f if not r["post_usable"]]
    print(f"[{size}] F1 n={len(f)}   usable box still survives: {len(usable)} "
          f"({pct(len(usable),len(f)):.1f}%)   genuinely stranded: {len(lost)} ({pct(len(lost),len(f)):.1f}%)")
    if lost:
        print(f"   stranded-GT median post-NMS best IoU = {med([r['post_max_iou_sameclass'] for r in lost]):.3f}"
              f"   median suppressor IoU/GT = {med([r['sup_iou'] for r in lost])}")

for size in ["small", "medium", "large"]:
    f = [r for r in f1 if r["size"] == size]
    if not f:
        continue
    harmful = [r for r in f if r["sup_iou"] is not None and r["sup_iou"] < r["best_raw_iou"]]
    print(f"\n[{size}] F1 n={len(f)}  of which suppressor overlaps GT LESS than candidate: {len(harmful)}")
    print(f"   median candidate IoU/GT   : {med([r['best_raw_iou'] for r in f])}")
    print(f"   median suppressor IoU/GT  : {med([r['sup_iou'] for r in f])}")
    print(f"   median candidate conf     : {med([r['best_raw_conf'] for r in f])}")
    print(f"   median suppressor conf    : {med([r['sup_conf'] for r in f])}")
    print(f"   median sup/cand area ratio: {med([r['sup_area_ratio'] for r in f])}")
    print(f"   suppressor same class     : {sum(1 for r in f if r['sup_same_class'])}/{len(f)}")
    print(f"   reasons                   : {Counter(r['drop_reason'] for r in f)}")
    print(f"   --- truly harmful (sup_iou < cand_iou), small-GT share ---")
    for r in harmful[:10]:
        cc = r["best_raw_conf"] or float("nan"); sc = r["sup_conf"] or float("nan")
        print(f"      {r['stem']}#{r['gt_id']} cls={r['cls_name']:11s} candIoU={r['best_raw_iou']:.3f} "
              f"candConf={cc:.4f} supIoU={r['sup_iou']:.3f} supConf={sc:.4f}")

print()
print("=" * 78)
print("4. TOP-100 CAUSAL AUDIT")
print("=" * 78)
f2 = [r for r in rows if r["failure"] == "F2_top100_disappearance"]
print(f"F2 total = {len(f2)}  (small {sum(1 for r in f2 if r['size']=='small')})")
by_img = Counter(r["stem"] for r in f2)
print(f"images involved = {len(by_img)}   top-3 density: {by_img.most_common(3)}")
if f2:
    print(f"median rank after NMS = {med([r['rank_after_nms'] for r in f2])}")
    print(f"median conf of F2 cands = {med([r['best_raw_conf'] for r in f2])}")
print(f"C1 counterfactual  top-100  mAP50-95 = {A['official_top100']['mAP50-95']:.6f}")
print(f"C1 counterfactual  uncapped mAP50-95 = {A['official_uncapped_C1']['mAP50-95']:.6f}"
      f"   delta = {A['official_uncapped_C1']['mAP50-95']-A['official_top100']['mAP50-95']:+.6f}")

# how many GTs are matched under no-cap but not under cap
cap_m = defaultdict(int); nocap_m = defaultdict(int)
for r in rows:
    if r["matched50"]:
        cap_m[r["size"]] += 1
# recompute uncapped matches from matched_at? matched50 is cap-only. Use pct delta via re-eval below.

print()
print("=" * 78)
print("5. 12-CLASS BREAKDOWN (small GT only)")
print("=" * 78)
print(f"{'class':12s} {'n':>4s} {'TP@.5':>7s} {'F0':>4s} {'F1':>4s} {'F2':>4s} {'F3':>4s} {'medIoU':>7s} {'medConf':>8s}")
for c in range(12):
    s = [r for r in small if r["cls"] == c]
    if not s:
        continue
    fc = Counter(r["failure"] for r in s)
    nm = fc.get("MATCHED@0.5", 0)
    print(f"{s[0]['cls_name']:12s} {len(s):4d} {pct(nm,len(s)):6.1f}% {fc.get('F0_no_raw_candidate',0):4d} "
          f"{fc.get('F1_nms_disappearance',0):4d} {fc.get('F2_top100_disappearance',0):4d} "
          f"{fc.get('F3_survives_official_fail',0):4d} {med([r['best_raw_iou'] for r in s]):7.3f} "
          f"{med([r['best_raw_conf'] for r in s]):8.4f}")

print()
print("=" * 78)
print("6. F0 PROFILE  (no pre-NMS same-class candidate with IoU>=0.10)")
print("=" * 78)
for size in ["small", "medium", "large"]:
    f0 = [r for r in rows if r["failure"] == "F0_no_raw_candidate" and r["size"] == size]
    tot = [r for r in rows if r["size"] == size]
    if not tot:
        continue
    noSame = [r for r in f0 if r["best_raw_conf"] is None]
    hasSame = [r for r in f0 if r["best_raw_conf"] is not None]
    print(f"\n[{size}] F0 n={len(f0)} ({pct(len(f0),len(tot)):.2f}% of {size})"
          f"   no same-class box at all: {len(noSame)}   same-class box exists but IoU<0.10: {len(hasSame)}")
    if hasSame:
        ar = [r["best_raw_area_ratio"] for r in hasSame]
        print(f"   of those: median bestIoU={med([r['best_raw_iou'] for r in hasSame]):.4f}"
              f"  median conf={med([r['best_raw_conf'] for r in hasSame]):.4f}"
              f"  median areaRatio={med(ar):.2f}"
              f"  pct areaRatio>1.5 = {pct(sum(1 for x in ar if x and x>1.5), len(ar)):.1f}%")
        print(f"   median |dx|={med([abs(r['best_raw_dx']) for r in hasSame]):.3f}"
              f"  median |dy|={med([abs(r['best_raw_dy']) for r in hasSame]):.3f}"
              f"  median rw={med([r['best_raw_rw'] for r in hasSame]):.2f}"
              f"  median rh={med([r['best_raw_rh'] for r in hasSame]):.2f}")
    if noSame:
        print(f"   no-same-class group: {len(noSame)} GT have no same-class box anywhere pre-NMS"
              f" (hard representation miss)")

print()
print("=" * 78)
print("7. RECONCILIATION WITH THE 2026-09-18 ATTRIBUTION NUMBERS")
print("=" * 78)


def stage_of(r):
    """which pipeline stage the 2026-09-18 'IoU==0 / no same-class' condition lives at"""
    if r["best_raw_conf"] is None:
        return "no same-class box pre-NMS"
    if r["failure"] == "F1_nms_disappearance":
        return "pre-NMS yes, killed by NMS"
    if r["failure"] == "F2_top100_disappearance":
        return "pre-NMS yes, not in top-100"
    return "pre-NMS yes, survived"


has_same = [r for r in small if r["best_raw_conf"] is not None]
zero_iou_pre = [r for r in has_same if r["best_raw_iou"] < 1e-9]
print(f"small GT with ANY same-class box pre-NMS      : {len(has_same)}/{len(small)} ({pct(len(has_same),len(small)):.1f}%)")
print(f"small GT with NO same-class box pre-NMS       : {len(small)-len(has_same)} ({pct(len(small)-len(has_same),len(small)):.1f}%)")
print(f"   ... best same-class IoU == 0 (pre-NMS)     : {len(zero_iou_pre)} ({pct(len(zero_iou_pre),len(small)):.1f}%)")
print(f"   ... 0 < bestIoU < 0.10 (pre-NMS)           : {sum(1 for r in has_same if 1e-9 < r['best_raw_iou'] < 0.10)}")
out = dict(
    pipeline=dict(raw_per_img=float(nA.mean()), nms_per_img=float(nB.mean()),
                  top100_per_img=float(top100.mean()), raw_total=int(nA.sum()),
                  nms_total=int(nB.sum()), imgs_gt100=int((nB > 100).sum())),
    funnel_small=dict(Counter(r["failure"] for r in small)),
    funnel_all_sizes={s: dict(Counter(r["failure"] for r in rows if r["size"] == s))
                      for s in ["small", "medium", "large"]},
    class_small={str(c): dict(Counter(r["failure"] for r in small if r["cls"] == c))
                 for c in range(12)},
    small_n=len(small), small_zero_iou_pre=int(len(zero_iou_pre)),
    small_no_sameclass_pre=int(len(small) - len(has_same)),
)
(OUT / "tables.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("\n[saved] tables.json")
