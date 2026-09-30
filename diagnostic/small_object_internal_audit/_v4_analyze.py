"""_v4_analyze.py — V4 分析：assignment / feature / candidate 三层的 G1 vs 对照。

只读。复用 _v4_internal.json（真实 assigner 插桩 + 真实 forward）。
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
V2 = ROOT / "diagnostic/small_object_cause_v2"


def main():
    R = json.loads((OUT / "_v4_internal.json").read_text(encoding="utf-8"))
    rec = {f"{r['image_id']}#{r['gt_id']}": r for r in
           json.loads((V2 / "_v2_records.json").read_text(encoding="utf-8"))["records"]}
    v3 = json.loads((V2 / "_v3_analysis.json").read_text(encoding="utf-8"))
    E1 = set(v3["E1"])
    ctrl = {k: v[0] for k, v in v3["controls"].items()}
    EPS0 = 1e-9

    # 键匹配检查
    keys = {f"{r['stem']}#{r['gt']}" for r in R}
    print(f"v4 记录 {len(R)} 条；与 v2 键匹配 {len(keys & set(rec))}/{len(keys)}")
    print(f"G1(E1) 在 v4 中命中 {len([k for k in E1 if k in keys])}/{len(E1)}")
    print(f"control 在 v4 中命中 {len([k for k in ctrl.values() if k in keys])}/{len(ctrl)}")

    def sel(pred):
        return [r for r in R if pred(f"{r['stem']}#{r['gt']}")]

    G1 = sel(lambda k: k in E1)
    G2 = sel(lambda k: k in set(ctrl.values()))
    allk = set(rec)
    SMALL_A = 1024.0
    G4 = sel(lambda k: (k in allk and rec[k]["native_area"] < SMALL_A
                        and rec[k]["best_same_class_iou"] >= 0.5 and k not in E1
                        and k not in set(ctrl.values())))
    miss_c = [k for k in ctrl.values() if k not in keys]
    print(f"  control 未命中 {len(miss_c)} 个，其原因（stem 是否被处理 / 是否在 v2 中）：")
    for k in miss_c[:6]:
        st = k.split("#")[0]
        print(f"    {k}: stem 在 v4 中={'是' if any(r['stem']==st for r in R) else '否'}  在 v2 中={k in rec}")

    def stat(g, f):
        v = [r[f] for r in g if f in r and r[f] is not None]
        return (float(np.median(v)) if v else float("nan")), len(v)

    print("\n" + "=" * 78)
    print("§6 D — Assignment audit（真实 TaskAlignedAssigner，topk=10）")
    print("=" * 78)
    print(f"  |G1|={len(G1)}  |G2(same-img ctrl)|={len(G2)}  |G4(其它成功 small GT,同图)|={len(G4)}")
    hdr = f"  {'metric':<24}{'G1':>10}{'G2':>10}{'G4':>10}"
    print(hdr)
    for f in ("n_in_gts", "n_topk", "n_pos", "best_assign_iou", "mean_assign_iou",
              "best_align", "mean_align", "best_any_overlap"):
        row = f"  {f:<24}"
        for g in (G1, G2, G4):
            m, _ = stat(g, f)
            row += f"{m:>10.4f}"
        print(row)
    print("\n  --- Q1: positive_count == 0 的比例 ---")
    for nm, g in (("G1", G1), ("G2", G2), ("G4", G4)):
        z = sum(1 for r in g if r["n_pos"] == 0)
        print(f"    {nm}: {z}/{len(g)} = {z/max(1,len(g)):.1%}")
    print("  --- positive 所属 stride level 分布 ---")
    for nm, g in (("G1", G1), ("G2", G2), ("G4", G4)):
        c = Counter()
        for r in g:
            for s in r.get("pos_strides", []):
                c[s] += 1
        print(f"    {nm}: {dict(sorted(c.items()))}")

    print("\n" + "=" * 78)
    print("§7 E — Feature response（P3=layer23, backbone P2=layer9）")
    print("=" * 78)
    print(f"  {'metric':<24}{'G1':>10}{'G2':>10}{'G4':>10}")
    for f in ("L23_center_norm", "L23_global_norm", "L23_ratio",
              "L9_center_norm", "L9_global_norm", "L9_ratio"):
        row = f"  {f:<24}"
        for g in (G1, G2, G4):
            m, _ = stat(g, f)
            row += f"{m:>10.4f}"
        print(row)

    print("\n" + "=" * 78)
    print("§8 F — GT-center candidate response（raw head, pre-NMS）")
    print("=" * 78)
    print(f"  {'metric':<28}{'G1':>10}{'G2':>10}{'G4':>10}")
    for f in ("p2_max_score_near", "p2_max_score_all", "p2_best_same_cls_iou",
              "p3_max_score_near", "p3_max_score_all", "p3_best_same_cls_iou",
              "p4_max_score_near", "p4_max_score_all", "p4_best_same_cls_iou"):
        row = f"  {f:<28}"
        for g in (G1, G2, G4):
            m, _ = stat(g, f)
            row += f"{m:>10.4f}"
        print(row)

    # 三阶段归类
    print("\n" + "=" * 78)
    print("§9 G — Failure-stage attribution（对 G1 的 38 个）")
    print("=" * 78)
    cls = Counter()
    for r in G1:
        npos = r["n_pos"]
        bai = r.get("best_assign_iou", 0.0) or 0.0
        sc = max([r.get(f"p{i}_max_score_near", 0.0) or 0.0 for i in (2, 3, 4)])
        allsc = max([r.get(f"p{i}_max_score_all", 0.0) or 0.0 for i in (2, 3, 4)])
        iou = max([r.get(f"p{i}_best_same_cls_iou", 0.0) or 0.0 for i in (2, 3, 4)])
        if npos == 0 and bai == 0:
            cls["Assignment: zero positive (mask_in_gts or topk 失败)"] += 1
        elif sc < 0.05 * max(allsc, 1e-9):
            cls["Head: GT-center class response 远低于全图峰值"] += 1
        elif iou < 0.1:
            cls["Localization: GT 中心附近有响应但框 IoU<0.1"] += 1
        else:
            cls["Candidate generation / post-processing"] += 1
    for c, v in cls.most_common():
        print(f"  {c:<52}{v:>3}  ({v/len(G1):.1%})")

    print("\n  --- 关键量：GT 中心附近最高同类分 / 全图最高同类分 ---")
    for nm, g in (("G1", G1), ("G2", G2), ("G4", G4)):
        rr = []
        for r in g:
            near = max([r.get(f"p{i}_max_score_near", 0.0) or 0.0 for i in (2, 3, 4)])
            alls = max([r.get(f"p{i}_max_score_all", 0.0) or 0.0 for i in (2, 3, 4)])
            rr.append(near / max(alls, 1e-9))
        print(f"    {nm}: (中心最高分 / 全图最高分) 中位 = {np.median(rr):.4f}")

    (OUT / "_v4_analysis.json").write_text(json.dumps(
        dict(n=dict(G1=len(G1), G2=len(G2), G4=len(G4)), stage=dict(cls)),
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {OUT/'_v4_analysis.json'}")


if __name__ == "__main__":
    main()
