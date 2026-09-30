"""_v12_analyze.py — activation patching 结果分析 + 完整性校验 + paired 检验。只读。"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
R = json.loads((OUT / "_v12_patch.json").read_text(encoding="utf-8"))
INT = ["L16_zero", "L17_zero", "L16_proto", "L17_proto", "L17_proto_normmatch",
       "L17_displaced", "L17_random_donor"]
LAB = {"L16_zero": "L16 zero", "L17_zero": "L17 zero", "L16_proto": "L16 prototype",
       "L17_proto": "L17 prototype", "L17_proto_normmatch": "L17 norm-matched",
       "L17_displaced": "displaced L17", "L17_random_donor": "random donor"}


def main():
    G1 = [r for r in R if r["grp"] == "G1"]
    G4 = [r for r in R if r["grp"] == "G4"]
    print("=" * 104)
    print("§11 INTEGRITY —— intervention OFF 时 base_logit 与 V7/V10 的 logit_decomp 是否一致")
    print("=" * 104)
    v7 = {(r["domain"], r["key"]): r for r in
          json.loads((OUT / "_v7_vectors.json").read_text(encoding="utf-8"))["rows"]}
    for g, rows, dom in (("G1", G1, "V1"), ("G4", G4, "V1")):
        d = []
        for r in rows:
            ref = v7.get((dom, r["key"]))
            if ref is None:
                continue
            d.append(abs(r["base_logit"] - ref["logit_decomp"]))
        if d:
            print(f"  {g:<4} n={len(d)}  max|Δ base_logit| = {max(d):.3e}  "
                  f"中位={np.median(d):.3e}")
    print("  （base_logit 与 V7 的 logit_decomp 用**同一精确分解** logit_c = W_c·h + b_c、"
          "同一个 GT-center cell）")

    print("\n" + "=" * 104)
    print(f"§7 核心表（G1 n={len(G1)}，G4 对照 n={len(G4)}；Δ 相对各自未干预 baseline）")
    print("=" * 104)
    print(f"  {'Intervention':<20}{'G1 medΔlogit':>14}{'G1 IQR':>20}{'G1 medΔprob':>13}"
          f"{'%Δlogit>0':>11}{'G4 medΔlogit':>14}{'G4 %Δ>0':>10}")
    for k in INT:
        a = [r[f"{k}_dlogit"] for r in G1]
        b = [r[f"{k}_dlogit"] for r in G4]
        ap = [r[f"{k}_dprob"] for r in G1]
        print(f"  {LAB[k]:<20}{np.median(a):>14.3f}"
              f"{f'[{np.percentile(a,25):.2f},{np.percentile(a,75):.2f}]':>20}"
              f"{np.median(ap):>13.4f}{np.mean(np.array(a)>0):>11.1%}"
              f"{np.median(b):>14.3f}{np.mean(np.array(b)>0):>10.1%}")

    print("\n  基线（未干预）现状：")
    print(f"    G1 base_logit 中位 = {np.median([r['base_logit'] for r in G1]):.3f}"
          f"   base_prob 中位 = {np.median([r['base_prob'] for r in G1]):.6f}"
          f"   donor 候选数中位 = {np.median([r['n_donor_cand'] for r in G1]):.0f}")
    print(f"    G4 base_logit 中位 = {np.median([r['base_logit'] for r in G4]):.3f}")

    print("\n" + "=" * 104)
    print("§8 非参数 paired 检验（Wilcoxon signed-rank，H0: median Δ = 0）")
    print("=" * 104)
    from scipy.stats import wilcoxon
    print(f"  {'Intervention':<20}{'G1 medΔ':>10}{'W p':>12}{'rank-biserial':>15}"
          f"{'G4 medΔ':>10}{'G4 W p':>12}")
    for k in INT:
        a = np.array([r[f"{k}_dlogit"] for r in G1])
        b = np.array([r[f"{k}_dlogit"] for r in G4])
        try:
            p1 = wilcoxon(a).pvalue
        except Exception:
            p1 = float("nan")
        try:
            p4 = wilcoxon(b).pvalue
        except Exception:
            p4 = float("nan")
        # rank-biserial（配对）
        d = a[a != 0]
        rb = (d > 0).sum() / len(d) - (d < 0).sum() / len(d) if len(d) else float("nan")
        print(f"  {LAB[k]:<20}{np.median(a):>10.3f}{p1:>12.3e}{rb:>15.3f}"
              f"{np.median(b):>10.3f}{p4:>12.3e}")

    print("\n" + "=" * 104)
    print("§9 判据检查")
    print("=" * 104)
    key = "L17_proto"
    a = np.array([r[f"{key}_dlogit"] for r in G1])
    b = np.array([r[f"{key}_dlogit"] for r in G4])
    disp = np.array([r["L17_displaced_dlogit"] for r in G1])
    rand = np.array([r["L17_random_donor_dlogit"] for r in G1])
    print(f"  L17 prototype 对 G1:  中位Δ={np.median(a):+.3f}  %>0={np.mean(a>0):.1%}")
    print(f"  L17 prototype 对 G4:  中位Δ={np.median(b):+.3f}  %>0={np.mean(b>0):.1%}")
    print(f"  displaced L17 对 G1:  中位Δ={np.median(disp):+.3f}  %>0={np.mean(disp>0):.1%}")
    print(f"  random donor 对 G1:   中位Δ={np.median(rand):+.3f}  %>0={np.mean(rand>0):.1%}")
    print(f"  中心 patch − displaced = {np.median(a)-np.median(disp):+.3f}")
    print(f"  中心 patch − random    = {np.median(a)-np.median(rand):+.3f}")
    print(f"  L16 prototype 对 G1:  中位Δ={np.median([r['L16_proto_dlogit'] for r in G1]):+.3f}")
    print(f"  norm-matched 对 G1:   中位Δ={np.median([r['L17_proto_normmatch_dlogit'] for r in G1]):+.3f}")

    # 分类型看
    print("\n  按类别（L17 prototype，G1）：")
    from collections import defaultdict
    byc = defaultdict(list)
    for r in G1:
        byc[r["cls"]].append(r["L17_proto_dlogit"])
    for c in sorted(byc, key=lambda z: -len(byc[z])):
        v = byc[c]
        print(f"    cls{c:<3} n={len(v):<3} 中位Δ={np.median(v):+.3f}")

    (OUT / "_v12_result.json").write_text(json.dumps(
        {k: dict(g1_med=float(np.median([r[f'{k}_dlogit'] for r in G1])),
                 g4_med=float(np.median([r[f'{k}_dlogit'] for r in G4])))
         for k in INT}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
