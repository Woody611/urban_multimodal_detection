"""_v6_analyze.py — T1/T2/V1 分层对比 + 统计（effect size + CI + Bonferroni/FDR）。只读。"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
V2 = ROOT / "diagnostic/small_object_cause_v2"

BINS = ["<12", "12-18", "18-24", "24-32", "32-32^2(=small)", "medium"]


def cliffs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return float("nan")
    gt = sum((ai > b).sum() for ai in a); lt = sum((ai < b).sum() for ai in a)
    return (gt - lt) / (len(a) * len(b))


def boot(a, b, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return (float("nan"), float("nan"))
    d = [np.median(rng.choice(a, len(a), True)) - np.median(rng.choice(b, len(b), True))
         for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    R = json.loads((OUT / "_v6_domains.json").read_text(encoding="utf-8"))
    v3 = json.loads((V2 / "_v3_analysis.json").read_text(encoding="utf-8"))
    E1 = set(v3["E1"]); CTRL = set(v[0] for v in v3["controls"].values())
    v2rec = {f"{r['image_id']}#{r['gt_id']}": r for r in
             json.loads((V2 / "_v2_records.json").read_text(encoding="utf-8"))["records"]}

    for r in R:
        if r["domain"] != "V1":
            r["grp"] = None
        elif r["key"] in E1:
            r["grp"] = "G1"
        elif r["key"] in CTRL:
            r["grp"] = "G2"
        elif (r["key"] in v2rec and r["key"] not in E1 and r["key"] not in CTRL
              and v2rec[r["key"]]["native_area"] < 1024
              and v2rec[r["key"]]["best_same_class_iou"] >= 0.5):
            r["grp"] = "G4"
        else:
            r["grp"] = None

    def sel(dom, b=None, grp=None):
        g = [r for r in R if r["domain"] == dom]
        if b:
            g = [r for r in g if r["bin"] == b]
        if grp:
            g = [r for r in g if r["grp"] == grp]
        return g

    def cell(g):
        if not g:
            return "     n=0"
        lg = [r["center_logit"] for r in g if r["center_logit"] == r["center_logit"]]
        pb = [r["center_prob"] for r in g if r["center_prob"] == r["center_prob"]]
        rt = [r["P3_ratio"] for r in g if r.get("P3_ratio") is not None]
        return (f"n={len(g):<5} logit={np.median(lg):>7.2f}  prob={np.median(pb):>6.3f}  "
                f"P3ratio={np.median(rt):>5.3f}")

    print("=" * 110)
    print("§2-4 三域 × size bin 对照（每格：n / median center logit / median class prob / median P3 ratio）")
    print("=" * 110)
    print(f"{'bin':<18}{'T1 train(no-aug)':<48}{'T2 train(real aug)':<48}")
    for b in BINS:
        print(f"{b:<18}{cell(sel('T1', b)):<48}{cell(sel('T2', b)):<48}")
    print()
    print(f"{'bin':<18}{'V1 G1 missed':<48}{'V1 G2/G4 success':<48}")
    for b in BINS:
        g1 = sel("V1", b, "G1")
        gs = sel("V1", b, "G2") + sel("V1", b, "G4")
        print(f"{b:<18}{cell(g1):<48}{cell(gs):<48}")
    print()
    print(f"{'bin':<18}{'V1 all(val, no-aug)':<48}")
    for b in BINS:
        print(f"{b:<18}{cell(sel('V1', b)):<48}")

    print("\n" + "=" * 110)
    print("§5 chain：feature → logit → probability → assignment（各域小目标 <32px 汇总）")
    print("=" * 110)
    def small32(dom, grp=None):
        g = [r for r in sel(dom, grp=grp) if r["sq"] < 32]
        if not g:
            return None
        return g
    print(f"{'domain/group':<22}{'n':>6}{'P3ratio':>10}{'P2b_ratio':>11}"
          f"{'center_logit':>14}{'center_prob':>13}{'n_pos':>8}{'best_align':>12}{'pos_target':>12}")
    for lab, g in (("T1 train no-aug", small32("T1")),
                   ("T2 train +aug", small32("T2")),
                   ("V1 val all", small32("V1")),
                   ("V1 G1 missed", small32("V1", "G1")),
                   ("V1 G2 ctrl", small32("V1", "G2")),
                   ("V1 G4 success", small32("V1", "G4"))):
        if not g:
            continue
        def M(k):
            v = [r[k] for r in g if r.get(k) is not None]
            return np.median(v) if v else float("nan")
        print(f"{lab:<22}{len(g):>6}{M('P3_ratio'):>10.3f}{M('P2b_ratio'):>11.3f}"
              f"{M('center_logit'):>14.2f}{M('center_prob'):>13.4f}{M('n_pos'):>8.1f}"
              f"{M('best_align'):>12.3f}{M('pos_target'):>12.3f}")

    print("\n" + "=" * 110)
    print("§6 统计（logit 与 prob），多重比较 Bonferroni + BH-FDR")
    print("=" * 110)
    try:
        from scipy.stats import mannwhitneyu
        HAVE = True
    except Exception:
        HAVE = False
    tests = []
    def cmp(lab, a, b, key):
        va = [r[key] for r in a if r.get(key) is not None and r[key] == r[key]]
        vb = [r[key] for r in b if r.get(key) is not None and r[key] == r[key]]
        if len(va) < 5 or len(vb) < 5:
            return
        p = mannwhitneyu(va, vb, alternative="two-sided").pvalue if HAVE else float("nan")
        lo, hi = boot(va, vb)
        tests.append((lab, key, len(va), len(vb), np.median(va), np.median(vb),
                      cliffs(va, vb), lo, hi, p))
    S = lambda d, g=None: [r for r in sel(d, grp=g) if r["sq"] < 32]
    cmp("T1 vs V1 (domain)", S("T1"), S("V1"), "center_logit")
    cmp("T1 vs V1 (domain)", S("T1"), S("V1"), "center_prob")
    cmp("T1 vs T2 (augment)", S("T1"), S("T2"), "center_logit")
    cmp("T1 vs T2 (augment)", S("T1"), S("T2"), "center_prob")
    cmp("V1 G1 vs G2/G4", S("V1", "G1"), S("V1", "G2") + S("V1", "G4"), "center_logit")
    cmp("V1 G1 vs G2/G4", S("V1", "G1"), S("V1", "G2") + S("V1", "G4"), "center_prob")
    cmp("T1 vs V1 G1", S("T1"), S("V1", "G1"), "center_logit")
    p = np.array([t[9] for t in tests], float)
    m = len(p)
    order = np.argsort(p)
    fdr = np.minimum.accumulate((p[order] * m / (np.arange(m) + 1))[::-1])[::-1]
    print(f"{'comparison':<22}{'metric':<13}{'nA':>5}{'nB':>5}{'medA':>9}{'medB':>9}"
          f"{'cliffΔ':>9}{'ΔCI95':>20}{'p':>11}{'bonf':>7}{'fdr':>7}")
    for i, (lab, key, na, nb, ma, mb, d, lo, hi, pv) in enumerate(tests):
        print(f"{lab:<22}{key:<13}{na:>5}{nb:>5}{ma:>9.3f}{mb:>9.3f}{d:>9.3f}"
              f"{f'[{lo:+.2f},{hi:+.2f}]':>20}{pv:>11.2e}"
              f"{'PASS' if pv < 0.05/m else '—':>7}{'PASS' if fdr[i] < 0.05 else '—':>7}")
    print(f"\n  m={m}  Bonferroni α={0.05/m:.5f}  通过 Bonferroni={int((p<0.05/m).sum())}"
          f"  通过 BH-FDR(0.05)={int((fdr<0.05).sum())}")


if __name__ == "__main__":
    main()
