"""_v2_attribute.py — V2 阶段 10/11/12/13：对「任意类别 IoU==0」目标做互斥 primary cause 归因。

只读：只读已产出的 _v2_records.json / _v2_visual.json。不训练、不推理。
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent

# 阈值全部为 diagnostic，非官方定义
OCC_HI = 0.30        # 被其他 GT 框覆盖 >30% → 视作可能遮挡
LAP_LO = 300.0       # Laplacian 方差低 → 可能模糊
STD_LO = 18.0        # 局部对比度低
EDGE_LO = 0.03


def main():
    rec = json.loads((OUT / "_v2_records.json").read_text(encoding="utf-8"))
    vis = json.loads((OUT / "_v2_visual.json").read_text(encoding="utf-8"))
    V = {r["key"]: r for r in vis["target"]}
    C = {r["key"]: r for r in vis["control"]}
    R = {f"{r['image_id']}#{r['gt_id']}": r for r in rec["records"]}
    EPS0 = 1e-9

    zero_keys = [k for k, r in R.items() if r["best_any_class_iou"] <= EPS0]
    print(f"归因对象：「任意类别 IoU==0」的 small GT = {len(zero_keys)}")

    rows = []
    for k in zero_keys:
        r = R[k]
        v = V.get(k, {})
        vc = v.get("visible_std")
        vg = v.get("visible_gradmean")
        ve = v.get("visible_edgedens")
        lap = v.get("visible_lapvar")
        occ = v.get("occ_proxy")
        # 可检测性（可计算 proxy，不含人工判断）
        if vc is None or vg is None:
            vis_grade = "UNKNOWN"
        elif vc > 40 and vg > 40 and (ve or 0) > 0.10:
            vis_grade = "CLEAR"
        elif vc > 22 and vg > 20 and (ve or 0) > 0.04:
            vis_grade = "AMBIGUOUS"
        else:
            vis_grade = "LOW_VISIBILITY"

        # ---------- 互斥 primary cause（按优先级判定）----------
        if vis_grade == "UNKNOWN":
            cause = "UNKNOWN"
        elif occ is not None and occ >= OCC_HI:
            cause = "OCCLUSION"
        elif vis_grade == "LOW_VISIBILITY" and (lap is not None and lap < LAP_LO):
            cause = "DATA_VISIBILITY"
        elif vis_grade == "LOW_VISIBILITY" and (vc is not None and vc < STD_LO):
            cause = "LOW_CONTRAST"
        elif lap is not None and lap < LAP_LO:
            cause = "BLUR"
        elif r["near_same_class_detected"] > 0:
            cause = "INSTANCE_SEPARATION"
        elif r["near_other_class_detected"] > 0:
            cause = "CLASS_CONFUSION"
        elif r["neighbor_gt_count"] > 0:
            cause = "SCENE_DIFFICULTY"
        elif r["n_pred_same"] == 0:
            cause = "WEAK_FEATURE_EVIDENCE"
        else:
            # 图内别处有同类高置信框、本处零重叠 → 该位置未被模型"看到"
            cause = "WEAK_FEATURE_EVIDENCE"
        rows.append(dict(key=k, cause=cause, vis_grade=vis_grade, occ=occ,
                         lap=lap, std=vc, class_=r["name"], cls=r["cls"],
                         in_sqrt=r["sqrt_area"] * r["scale"]))

    cnt = Counter(r["cause"] for r in rows)
    print("\n=== 阶段 11 —— 互斥 primary cause ===")
    print(f"  {'cause':<24}{'count':>7}{'%':>9}   证据强度")
    strength = {
        "OCCLUSION": "E2（GT 框重叠代理，且 missed 的 occ 低于 detected → 该归因基本被否定）",
        "DATA_VISIBILITY": "E2（可计算 proxy + 尺寸匹配对照）",
        "LOW_CONTRAST": "E2", "BLUR": "E2",
        "INSTANCE_SEPARATION": "E2（同图同类已检出为直接观测）",
        "CLASS_CONFUSION": "E2", "SCENE_DIFFICULTY": "E1（邻域全漏，未控制难度）",
        "WEAK_FEATURE_EVIDENCE": "E1（由「别处有同类框」间接推断）", "UNKNOWN": "E0",
    }
    for c, n in cnt.most_common():
        print(f"  {c:<24}{n:>7}{n/len(rows):>9.1%}   {strength.get(c,'')}")

    print("\n=== 阶段 10 —— 可检测性分级 ===")
    print(f"  missed   : {dict(Counter(r['vis_grade'] for r in rows))}")
    print(f"  detected : {dict(Counter(C[k]['visible_std'] and ('CLEAR' if (C[k]['visible_std']>40 and C[k]['visible_gradmean']>40 and C[k]['visible_edgedens']>0.10) else 'AMBIGUOUS/LOW') for k in C))}")

    print("\n=== 按类别 ===")
    byc = {}
    for r in rows:
        byc.setdefault(r["class_"], Counter())[r["cause"]] += 1
    for c in sorted(byc, key=lambda z: -sum(byc[z].values())):
        print(f"  {c:<14} n={sum(byc[c].values()):<4} {dict(byc[c].most_common(3))}")

    print("\n=== 关键对照：missed 中 CLEAR 的比例 ===")
    ncl = sum(1 for r in rows if r["vis_grade"] == "CLEAR")
    print(f"  missed 且可检测性 CLEAR = {ncl}/{len(rows)} = {ncl/len(rows):.1%}")
    print(f"  => 这些目标按同一可计算 proxy 与「被成功检出的目标」并无差别")
    (OUT / "_v2_attribute.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    print(f"\n[saved] {OUT/'_v2_attribute.json'}")
    return rows, cnt


if __name__ == "__main__":
    main()
