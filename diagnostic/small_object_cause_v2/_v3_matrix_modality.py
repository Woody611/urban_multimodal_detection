"""_v3_matrix_modality.py — V3 §7 修正矩阵 + §9 多模态局部证据 + §10-13 coverage。

只读。复用 _v3_analysis.json / _v2_visual.json / _v2_attribute.json。
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "diagnostic/small_object_cause_v2"


def main():
    v3 = json.loads((OUT / "_v3_analysis.json").read_text(encoding="utf-8"))
    A = v3["annotated"]
    E1 = v3["E1"]
    ctrl = v3["controls"]
    vis = json.loads((OUT / "_v2_visual.json").read_text(encoding="utf-8"))
    V = {r["key"]: r for r in vis["target"]}
    C = {r["key"]: r for r in vis["control"]}
    EPS0 = 1e-9

    # ---------- §7 修正矩阵（阈值 = 1，非中位）----------
    print("=== §7 density x same-class-success 矩阵（70 个 zero-overlap）===")
    zero = [k for k, u in A.items() if u["any_iou"] <= EPS0]
    M = Counter()
    for k in zero:
        hi = A[k]["n_r1"] >= 1
        suc = A[k]["n_det_same_r3"] > 0
        M[("A" if (hi and suc) else "B" if (not hi and suc) else
           "C" if (hi and not suc) else "D")] += 1
    lab = {"A": "A 高密度 + 同类邻居成功", "B": "B 低密度 + 同类邻居成功",
           "C": "C 高密度 + 邻居全失败", "D": "D 低密度 + 邻居全失败"}
    for k in "ABCD":
        print(f"  {lab[k]:<26} {M[k]:>3}  ({M[k]/len(zero):.1%})")
    print("  注：A/B 即 V2 的 INSTANCE_SEPARATION/CLASS_CONFUSION；C/D 即 38 个 E1")

    # ---------- §9 多模态局部证据 ----------
    print("\n=== §9 RGB/IR/Depth 局部证据：38 missed vs matched controls ===")

    def ev(r):
        rgb = (r.get("visible_std", 0) > 30 and r.get("visible_gradmean", 0) > 30
               and r.get("visible_edgedens", 0) > 0.06)
        ir = (r.get("infrared_std", 0) > 14 and r.get("infrared_gradmean", 0) > 12)
        dz = r.get("depth_std", 0)
        return rgb, ir, dz

    def tag(r):
        rgb, ir, dz = ev(r)
        n = sum([rgb, ir, dz > 0])
        if n == 0:
            return "weak-all"
        if rgb and ir:
            return "RGB+IR"
        if rgb:
            return "RGB strong"
        if ir:
            return "IR strong"
        return "Depth only"

    tm = Counter(); cm = Counter()
    ndz = {"missed": [], "control": []}
    for k in E1:
        r = V.get(k)
        if r:
            tm[tag(r)] += 1
            if "depth_std" in r:
                ndz["missed"].append(1.0 if r["depth_std"] == 0 else 0.0)
        ck = ctrl.get(k, [None])[0]
        if ck and ck in C:
            cm[tag(C[ck])] += 1
            if "depth_std" in C[ck]:
                ndz["control"].append(1.0 if C[ck]["depth_std"] == 0 else 0.0)
    print(f"  {'tag':<14}{'38 missed':>11}{'占':>8}{'control':>10}{'占':>8}")
    for t in ("RGB+IR", "RGB strong", "IR strong", "Depth only", "weak-all"):
        a, b = tm.get(t, 0), cm.get(t, 0)
        print(f"  {t:<14}{a:>11}{a/max(1,sum(tm.values())):>8.1%}{b:>10}"
              f"{b/max(1,sum(cm.values())):>8.1%}")

    # ---------- §10-13 coverage ----------
    print("\n=== §10-12 primary mechanism coverage（对 70 个 zero-overlap）===")
    cov = Counter()
    for k in zero:
        u = A[k]
        if u["n_det_same_r3"] > 0:
            cov["SCENE_COMPETITION / INSTANCE_SEPARATION"] += 1
        elif u["n_det_other_r3"] > 0:
            cov["CLASS_CONFUSION"] += 1
        elif u["n_r1"] >= 1:
            cov["LOCAL_COMPETITION(邻居亦漏)"] += 1
        elif u["border_norm"] < 0.05:
            cov["BORDER_COMPOSITION"] += 1
        else:
            cov["LOW_DENSITY_ISOLATED"] += 1
    for c, v in cov.most_common():
        print(f"  {c:<40}{v:>3}  ({v/len(zero):.1%})")

    print("\n=== §12 关键：密度与 miss 的关系方向 ===")
    for r in (1, 3):
        rows = []
        for lo, hi, lab in ((0, 0, "0"), (1, 1, "1"), (2, 99, "2+")):
            g = [u for u in A.values() if lo <= u[f"n_r{r}"] <= hi]
            if g:
                rows.append(f"n_r{r}={lab}: {sum(1 for u in g if u['any_iou']<=EPS0)/len(g):.1%}(n={len(g)})")
        print(f"  {'  '.join(rows)}")

    (OUT / "_v3_matrix_modality.json").write_text(json.dumps(
        dict(matrix=dict(M), modality_missed=dict(tm), modality_control=dict(cm),
             coverage=dict(cov)), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {OUT/'_v3_matrix_modality.json'}")


if __name__ == "__main__":
    main()
