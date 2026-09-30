"""_recon2.py — 决定性侦察：val 侧 raw-head 候选几何 vs 最终预测（READ-ONLY）"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "diagnostic/small_object_cause_v2"
P3 = ROOT / "diagnostic/p3_feature_space"
DOM = ROOT / "diagnostic/small_object_domains"

v3 = json.loads((V2 / "_v3_analysis.json").read_text(encoding="utf-8"))
ids = json.loads((P3 / "_v11_ids.json").read_text(encoding="utf-8"))
R2 = {f"{r['image_id']}#{r['gt_id']}": r for r in
      json.loads((V2 / "_v2_records.json").read_text(encoding="utf-8"))["records"]}
dom = json.loads((DOM / "_v6_domains.json").read_text(encoding="utf-8"))
V1 = {r["key"]: r for r in dom if r["domain"] == "V1"}

G1 = list(v3["E1"])
G2 = [v[0] for v in v3["controls"].values()]
G4 = list(ids["G4"])

print("=" * 100)
print("A. v6_domains V1：raw-head 候选几何（best_assign_iou = GT vs **decoded 预测框** 的 CIoU）")
print("   CIoU <= 真 IoU，故 best_assign_iou >= 0.5  =>  真 IoU >= 0.5（下界性质）")
print("=" * 100)
print(f"{'group':<5}{'n':>5}{'n_pos med':>11}{'#(n_pos==0)':>13}{'best_assign_iou med':>21}"
      f"{'#(>=0.5)':>10}{'#(>=0.3)':>10}{'center_prob med':>17}{'pos_prob med':>14}")
res = {}
for nm, ks in (("G1", G1), ("G2", G2), ("G4", G4)):
    rs = [V1[k] for k in ks if k in V1]
    npos = np.array([r["n_pos"] for r in rs])
    bi = np.array([r["best_assign_iou"] for r in rs])
    cp = np.array([r["center_prob"] for r in rs], float)
    pp = np.array([r["pos_prob"] for r in rs], float)
    res[nm] = dict(n=len(rs), npos=npos, bi=bi, cp=cp, pp=pp)
    print(f"{nm:<5}{len(rs):>5}{np.median(npos):>11.0f}{int((npos==0).sum()):>13}"
          f"{np.median(bi):>21.4f}{int((bi>=0.5).sum()):>10}{int((bi>=0.3).sum()):>10}"
          f"{np.nanmedian(cp):>17.4f}{np.nanmedian(pp):>14.4f}")

print()
print("=" * 100)
print("B. 决定性交叉表：raw-head 有几何合格候选  vs  最终输出有重叠")
print("=" * 100)
print(f"{'group':<5}{'n':>5}{'A: best_assign_iou>=0.5':>26}{'B: final any-IoU>0':>22}"
      f"{'A & not B':>12}")
for nm in ("G1", "G2", "G4"):
    ks = {"G1": G1, "G2": G2, "G4": G4}[nm]
    rs = res[nm]
    a = rs["bi"] >= 0.5
    b = np.array([R2[k]["best_any_class_iou"] > 1e-9 for k in ks if k in V1])
    print(f"{nm:<5}{len(a):>5}{int(a.sum()):>26}{int(b.sum()):>22}{int((a & ~b).sum()):>12}")

print()
print("C. G1 逐条明细（raw-head 候选 vs 最终输出）")
print(f"  {'key':<28}{'n_pos':>6}{'best_assign_iou':>16}{'center_prob':>13}{'pos_prob':>10}"
      f"{'final anyIoU':>14}{'n_pred':>8}")
for k in G1:
    if k not in V1:
        continue
    r = V1[k]
    print(f"  {k:<28}{r['n_pos']:>6}{r['best_assign_iou']:>16.4f}"
          f"{r['center_prob']:>13.4f}{r['pos_prob']:>10.4f}"
          f"{R2[k]['best_any_class_iou']:>14.4f}{R2[k]['n_pred_total']:>8}")
