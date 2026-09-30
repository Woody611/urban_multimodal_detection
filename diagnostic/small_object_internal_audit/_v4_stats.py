"""_v4_stats.py — V4 §14 统计要求：effect size + n + CI + raw p + Bonferroni + FDR + 方向。

只读。G1(38) vs G2(同图对照) / G4(同图其它成功 small GT)。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
V2 = ROOT / "diagnostic/small_object_cause_v2"


def cliffs_delta(a, b):
    """Cliff's delta（非参数 effect size），O(n^2) 但样本小。"""
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((ai > b).sum() for ai in a)
    lt = sum((ai < b).sum() for ai in a)
    n = len(a) * len(b)
    return (gt - lt) / n if n else float("nan")


def boot_ci(a, b, f=np.median, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    d = []
    for _ in range(n):
        d.append(f(rng.choice(a, len(a), True)) - f(rng.choice(b, len(b), True)))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    R = json.loads((OUT / "_v4_internal.json").read_text(encoding="utf-8"))
    rec = {f"{r['image_id']}#{r['gt_id']}": r for r in
           json.loads((V2 / "_v2_records.json").read_text(encoding="utf-8"))["records"]}
    v3 = json.loads((V2 / "_v3_analysis.json").read_text(encoding="utf-8"))
    E1 = set(v3["E1"]); ctrlv = set(v[0] for v in v3["controls"].values())
    keys = {f"{r['stem']}#{r['gt']}" for r in R}

    def g(pred):
        return [r for r in R if pred(f"{r['stem']}#{r['gt']}")]
    G1 = g(lambda k: k in E1)
    G2 = g(lambda k: k in ctrlv)
    G4 = g(lambda k: k in rec and k not in E1 and k not in ctrlv
           and rec[k]["native_area"] < 1024 and rec[k]["best_same_class_iou"] >= 0.5)
    print(f"|G1|={len(G1)} |G2|={len(G2)} |G4|={len(G4)}")

    def sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    METRICS = {
        "pos_score_max (prob)": ("pos_score_max", None),   # 已存的就是概率
        "pos_score_mean (prob)": ("pos_score_mean", None),
        "P2 near max logit": ("p2_max_score_near", None),
        "P3 near max logit": ("p3_max_score_near", None),
        "P4 near max logit": ("p4_max_score_near", None),
        "img max logit (all)": ("p2_max_score_all", None),
        "n_pos": ("n_pos", None),
        "best_assign_iou": ("best_assign_iou", None),
        "P3 center/global norm ratio": ("L23_ratio", None),
        "P3 center feature norm": ("L23_center_norm", None),
        "P2bk center/global norm ratio": ("L9_ratio", None),
    }
    try:
        from scipy.stats import mannwhitneyu
        HAVE = True
    except Exception:
        HAVE = False
    print(f"scipy={'有' if HAVE else '无'}\n")

    print(f"{'metric':<32}{'G1 med':>10}{'G2 med':>10}{'G4 med':>10}"
          f"{'cliffΔ(1-2)':>13}{'ΔCI95':>20}{'p(1v2)':>10}{'p(1v4)':>10}")
    raws = []
    for name, (f, tf) in METRICS.items():
        a = [r[f] for r in G1 if f in r and r[f] is not None]
        b = [r[f] for r in G2 if f in r and r[f] is not None]
        c = [r[f] for r in G4 if f in r and r[f] is not None]
        if tf:
            a, b, c = list(tf(np.array(a))), list(tf(np.array(b))), list(tf(np.array(c)))
        if not a or not b:
            continue
        d = cliffs_delta(a, b)
        lo, hi = boot_ci(a, b)
        p12 = mannwhitneyu(a, b, alternative="two-sided").pvalue if HAVE else float("nan")
        p14 = (mannwhitneyu(a, c, alternative="two-sided").pvalue
               if HAVE and c else float("nan"))
        raws.append(p12)
        print(f"{name:<32}{np.median(a):>10.4f}{np.median(b):>10.4f}{np.median(c):>10.4f}"
              f"{d:>13.3f}{f'[{lo:+.3f},{hi:+.3f}]':>20}{p12:>10.2e}{p14:>10.2e}")

    # 多重比较校正
    p = np.array([x for x in raws if x == x])
    m = len(p)
    print(f"\n多重比较校正（m={m} 项 G1-vs-G2 检验）：")
    print(f"  Bonferroni α = 0.05/{m} = {0.05/m:.5f}")
    print(f"  通过 Bonferroni 的项数 = {(p < 0.05/m).sum()}")
    order = np.argsort(p)
    fdr = p[order] * m / (np.arange(m) + 1)
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    print(f"  通过 BH-FDR(0.05) 的项数 = {(fdr < 0.05).sum()}")

    # §12 低密度孤立子集
    print("\n--- §12 低密度孤立子集（n_r1==0）内的 G1 vs G4 ---")
    A = v3["annotated"]
    def nd(r):
        return A.get(f"{r['stem']}#{r['gt']}", {}).get("n_r1", None)
    g1i = [r for r in G1 if nd(r) == 0]
    g4i = [r for r in G4 if nd(r) == 0]
    print(f"  G1(低密度) n={len(g1i)}  G4(低密度) n={len(g4i)}")
    for f, tf, nm in (("p3_max_score_near", None, "P3 near max logit"),
                      ("L23_ratio", None, "P3 center/global ratio"),
                      ("n_pos", None, "n_pos")):
        a = [tf(np.array([r[f] for r in g1i if f in r])) for _ in [0]] if tf else [r[f] for r in g1i if f in r]
        b = [r[f] for r in g4i if f in r]
        if a and b:
            pp = mannwhitneyu(a, b, alternative="two-sided").pvalue if HAVE else float("nan")
            print(f"    {nm:<28} G1={np.median(a):>9.4f}  G4={np.median(b):>9.4f}  p={pp:.2e}")

    (OUT / "_v4_stats.json").write_text(json.dumps(
        dict(n=dict(G1=len(G1), G2=len(G2), G4=len(G4)), m=m,
             bonf=float(0.05/m), n_pass_bonf=int((p < 0.05/m).sum()),
             n_pass_fdr=int((fdr < 0.05).sum())), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
