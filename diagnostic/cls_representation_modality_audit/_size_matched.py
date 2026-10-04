"""_size_matched.py — 尺寸匹配对照 + class-separation（只读，零 GPU）。

动机：主审计用的对照是「全部 medium 已检出 GT」，未按尺寸匹配；实测 G-86 的 native area
显著更小（MWP p=3e-3）⇒ 必须排除「尺寸混杂」才能把特征项差归因到分类表征。
只用 diagnostic/cls_representation_modality_audit/per_gt_representation_audit.csv（本目录上一步产物）。
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
R = list(csv.DictReader(open(OUT / "per_gt_representation_audit.csv", encoding="utf-8")))
f = lambda x: float(x)
NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}


def med(s, k):
    return float(np.median([f(r[k]) for r in s])) if s else float("nan")


A = [r for r in R if int(f(r["in_86"])) == 1]
B = [r for r in R if int(f(r["in_86"])) == 0 and int(f(r["detected"])) == 1]
C = [r for r in R if int(f(r["in_86"])) == 0 and int(f(r["detected"])) == 0]
ar = np.array([f(r["native_area"]) for r in A])
lo, hi = float(np.percentile(ar, 5)), float(np.percentile(ar, 95))
Bm = [r for r in B if lo <= f(r["native_area"]) <= hi]

print("=" * 96); print("尺寸匹配对照"); print("=" * 96)
print(f"  G-86 native area: median={np.median(ar):.0f}  P5-P95=[{lo:.0f},{hi:.0f}]")
print(f"  control(全)      median={np.median([f(r['native_area']) for r in B]):.0f}   n={len(B)}")
print(f"  control(匹配)    median={np.median([f(r['native_area']) for r in Bm]):.0f}   n={len(Bm)}")
print(f"\n  {'组':<26}{'n':>6}{'GT类logit':>11}{'特征项W·h':>11}{'|h|':>8}{'|p3|':>8}")
rows = []
for lab, s in (("G-86", A), ("control_full", B), ("control_size_matched", Bm), ("medium_missed_non86", C)):
    rows.append(dict(group=lab, n=len(s), logit_median=round(med(s, "cls_logit_stride8"), 4),
                     feature_term_median=round(med(s, "cls_feature_term"), 4),
                     h_norm_median=round(med(s, "h_norm"), 3), p3_norm_median=round(med(s, "p3_norm"), 3)))
    print(f"  {lab:<26}{len(s):>6}{med(s,'cls_logit_stride8'):>11.3f}{med(s,'cls_feature_term'):>11.3f}"
          f"{med(s,'h_norm'):>8.2f}{med(s,'p3_norm'):>8.2f}")
print(f"\n  ⇒ 尺寸匹配后 特征项差 = {med(A,'cls_feature_term')-med(Bm,'cls_feature_term'):+.3f} nats")
print(f"     尺寸匹配后 |h| 差    = {med(A,'h_norm')-med(Bm,'h_norm'):+.3f}（{med(A,'h_norm')/med(Bm,'h_norm')-1:+.1%}）")
print(f"     ⇒ 尺寸匹配 **不削弱** 该差异（未匹配时 {med(A,'cls_feature_term')-med(B,'cls_feature_term'):+.3f}）")

# ---- class separation（§13.2）：只对 n>=20 的类计算类均值 cos ----
print("\n" + "=" * 96); print("§13.2 class separation（h 与其类均值的 cosine，仅 n>=20 的类）"); print("=" * 96)
# 需要原始 h 向量 -> 从 _v7_vectors.json 取（只读）
V7 = Path(__file__).resolve().parents[2] / "diagnostic/p3_feature_space/_v7_vectors.json"
vec = json.loads(V7.read_text(encoding="utf-8"))["rows"]
H = {r["key"]: np.asarray(r["h"], np.float64) for r in vec if r["domain"] == "V1"}
byk = {r["key"]: r for r in R}
cls_h = defaultdict(list)
for r in R:
    k = f"{r['image_stem']}#{r['gt_id']}"
    if k in H:
        cls_h[int(f(r["gt_class"]))].append(H[k])
mu = {c: np.mean(v, axis=0) for c, v in cls_h.items() if len(v) >= 20}
cs = []
print(f"  {'class':<13}{'n':>6}{'cos(h,μ_cls) G-86':>20}{'cos 对照':>12}{'Δ':>9}")
for c, v in sorted(cls_h.items()):
    if c not in mu:
        continue
    ga, gb = [], []
    for r in R:
        if int(f(r["gt_class"])) != c:
            continue
        k = f"{r['image_stem']}#{r['gt_id']}"
        if k not in H:
            continue
        co = float(H[k] @ mu[c] / (np.linalg.norm(H[k]) * np.linalg.norm(mu[c]) + 1e-12))
        (ga if int(f(r["in_86"])) == 1 else gb).append(co)
    if ga and gb:
        cs.append(dict(cls=c, cls_name=NAMES[c], n=len(cls_h[c]), cos_86=round(float(np.median(ga)), 4),
                       cos_control=round(float(np.median(gb)), 4), delta=round(float(np.median(ga) - np.median(gb)), 4)))
        print(f"  {NAMES[c]:<13}{len(cls_h[c]):>6}{np.median(ga):>20.4f}{np.median(gb):>12.4f}{np.median(ga)-np.median(gb):>+9.4f}")
print("  ⇒ **descriptive only**；不据此做 class ranking 或因果结论")

with open(OUT / "size_matched_control.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(OUT / "class_separation.csv", "w", newline="", encoding="utf-8") as fh:
    if cs:
        w = csv.DictWriter(fh, fieldnames=list(cs[0].keys())); w.writeheader(); w.writerows(cs)
json.dump(dict(size_matched=rows,
               feature_term_gap_unmatched=float(med(A, "cls_feature_term") - med(B, "cls_feature_term")),
               feature_term_gap_matched=float(med(A, "cls_feature_term") - med(Bm, "cls_feature_term")),
               h_norm_ratio_matched=float(med(A, "h_norm") / med(Bm, "h_norm")),
               class_separation=cs),
          open(OUT / "_size_matched.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("\nDONE")
