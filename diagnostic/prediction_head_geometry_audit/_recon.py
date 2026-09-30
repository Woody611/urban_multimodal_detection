"""_recon.py — 侦察：G1/G2/G4 的冻结定义与现有 val 侧可测量（READ-ONLY）"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "diagnostic/small_object_cause_v2"
P3 = ROOT / "diagnostic/p3_feature_space"

rec = json.loads((V2 / "_v2_records.json").read_text(encoding="utf-8"))
R = {f"{r['image_id']}#{r['gt_id']}": r for r in rec["records"]}
v3 = json.loads((V2 / "_v3_analysis.json").read_text(encoding="utf-8"))
G1 = list(v3["E1"])
CTRL = {k: v[0] for k, v in v3["controls"].items()}
ids = json.loads((P3 / "_v11_ids.json").read_text(encoding="utf-8"))
G4 = list(ids["G4"])

print(f"G1={len(G1)}  G2={len(CTRL)}  G4={len(G4)}")
print(f"v2_records n={len(R)}  v3.annotated n={len(v3['annotated'])}")
print()
for nm, ks in (("G1", G1), ("G2", list(CTRL.values())), ("G4", G4)):
    ks2 = [k for k in ks if k in R]
    a = np.array([R[k]["best_any_class_iou"] for k in ks2])
    s = np.array([R[k]["best_same_class_iou"] for k in ks2])
    npt = np.array([R[k]["n_pred_total"] for k in ks2])
    print(f"{nm}: n={len(ks2)}/{len(ks)} in_records")
    print(f"   any_iou   max={a.max():.4f} med={np.median(a):.4f} #(==0)={int((a <= 1e-9).sum())}")
    print(f"   same_iou  max={s.max():.4f} med={np.median(s):.4f} #(==0)={int((s <= 1e-9).sum())}"
          f"  #(>=0.5)={int((s >= 0.5).sum())}")
    print(f"   n_pred_total med={np.median(npt):.0f} max={npt.max()} #(==100)={int((npt == 100).sum())}")
    ar = [R[k].get("best_any_class_area_ratio") for k in ks2]
    ar = np.array([x for x in ar if x is not None], float)
    sr = [R[k].get("best_same_class_area_ratio") for k in ks2]
    sr = np.array([x for x in sr if x is not None], float)
    print(f"   any_area_ratio n={len(ar)} med={np.median(ar) if len(ar) else float('nan'):.4f}")
    print(f"   same_area_ratio n={len(sr)} med={np.median(sr) if len(sr) else float('nan'):.4f}")
    ax = [R[k].get("center_dx_norm_gtw") for k in ks2]
    ax = np.array([x for x in ax if x is not None], float)
    print(f"   center_dx_norm_gtw n={len(ax)} med={np.median(ax) if len(ax) else float('nan'):.4f}")
    print()

print("G1 annotated 字段:", sorted(v3["annotated"][G1[0]].keys()))
print()
NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}
for nm, ks in (("G1", G1), ("G2", list(CTRL.values())), ("G4", G4)):
    ks2 = [k for k in ks if k in R]
    c = Counter(NAMES[R[k]["cls"]] for k in ks2)
    sq = np.array([R[k]["sqrt_area"] for k in ks2])
    asp = np.array([R[k]["aspect"] for k in ks2])
    img = {R[k]["image_id"] for k in ks2}
    print(f"{nm}: classes={dict(c.most_common())}")
    print(f"     sqrt(area) med={np.median(sq):.2f} IQR=[{np.percentile(sq,25):.2f},{np.percentile(sq,75):.2f}]"
          f"  aspect med={np.median(asp):.3f} IQR=[{np.percentile(asp,25):.3f},{np.percentile(asp,75):.3f}]"
          f"  n_img={len(img)}")
    print()
