"""_v7_analyze.py — P3 feature-space 判定。只读。

G1/G2/G4 = val；T-success = train small GT 且 logit_decomp>0（本轮定义，明示）。
head 分解只用实现对应的精确等式：logit_c = W_c · h + b_c（h = cv3[0][-1] 的输入）。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
V2 = ROOT / "diagnostic/small_object_cause_v2"
SMALL = 32.0
np.random.seed(0)


def cos(a, b):
    na = np.linalg.norm(a, axis=-1) + 1e-12
    nb = np.linalg.norm(b, axis=-1) + 1e-12
    return (a * b).sum(-1) / (na * nb)


def cliffs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return float("nan")
    return (sum((x > b).sum() for x in a) - sum((x < b).sum() for x in a)) / (len(a) * len(b))


def boot(a, b, n=2000):
    rng = np.random.default_rng(0)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    d = [np.median(rng.choice(a, len(a), True)) - np.median(rng.choice(b, len(b), True))
         for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    J = json.loads((OUT / "_v7_vectors.json").read_text(encoding="utf-8"))
    R = J["rows"]
    for r in R:
        r["p3"] = np.array(r["p3"], np.float32)
        r["h"] = np.array(r["h"], np.float32)
    v3 = json.loads((V2 / "_v3_analysis.json").read_text(encoding="utf-8"))
    E1 = set(v3["E1"]); CTRL = set(v[0] for v in v3["controls"].values())
    v2rec = {f"{r['image_id']}#{r['gt_id']}": r for r in
             json.loads((V2 / "_v2_records.json").read_text(encoding="utf-8"))["records"]}

    for r in R:
        if r["domain"] == "T1":
            r["grp"] = "T_success" if (r["sq"] < SMALL and r["logit_decomp"] > 0) else None
        else:
            k = r["key"]
            if k in E1:
                r["grp"] = "G1"
            elif k in CTRL:
                r["grp"] = "G2"
            elif (k in v2rec and k not in E1 and k not in CTRL
                  and v2rec[k]["native_area"] < 1024 and v2rec[k]["best_same_class_iou"] >= 0.5):
                r["grp"] = "G4"
            else:
                r["grp"] = None
    G = {g: [r for r in R if r["grp"] == g] for g in ("G1", "G2", "G4", "T_success")}
    print("组大小:", {k: len(v) for k, v in G.items()},
          " (T_success 定义: train small GT 且 logit_decomp>0)")

    W = None  # 类别权重在 extraction 已用于 logit_decomp；此处从 logit 分解关系反推不便，改用向量统计
    # ---------- 1. 向量几何 ----------
    def norm_stats(g, k):
        v = np.stack([r[k] for r in g]) if g else np.zeros((0, 1))
        return np.linalg.norm(v, axis=1) if len(v) else np.zeros(0)

    print("\n" + "=" * 100)
    print("§2/3 向量几何（P3 空间 与 H(=cv3 输入) 空间）")
    print("=" * 100)
    muP = np.stack([r["p3"] for r in G["G4"]]).mean(0)
    muH = np.stack([r["h"] for r in G["G4"]]).mean(0)
    muP /= np.linalg.norm(muP); muH /= np.linalg.norm(muH)
    print(f"{'group':<11}{'n':>5}{'|P3|':>9}{'|H|':>9}{'cos(P3,muG4)':>14}{'cos(H,muG4)':>13}"
          f"{'logit_decomp':>13}{'|W_c|·|H|cos':>13}")
    for g in ("T_success", "G4", "G2", "G1"):
        gg = G[g]
        if not gg:
            continue
        nP = norm_stats(gg, "p3"); nH = norm_stats(gg, "h")
        cP = cos(np.stack([r["p3"] for r in gg]), muP)
        cH = cos(np.stack([r["h"] for r in gg]), muH)
        lg = [r["logit_decomp"] for r in gg]
        print(f"{g:<11}{len(gg):>5}{np.median(nP):>9.3f}{np.median(nH):>9.3f}"
              f"{np.median(cP):>14.4f}{np.median(cH):>13.4f}{np.median(lg):>13.3f}"
              f"{'':>13}")

    # ---------- 2. kNN ----------
    print("\n--- kNN（在参考集内的最近邻欧氏距离，特征已 L2 归一化）---")
    print(f"{'query → ref':<26}{'k=5':>10}{'k=10':>10}")
    def refnorm(g):
        V = np.stack([r["h"] for r in g])
        return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    REF = {k: refnorm(G[k]) for k in ("T_success", "G4") if G[k]}
    for gname in ("G1", "G2", "G4", "T_success"):
        if not G[gname]:
            continue
        Q = refnorm(G[gname])
        for rk, RV in REF.items():
            if rk == gname:
                continue
            D = np.linalg.norm(Q[:, None, :] - RV[None, :, :], axis=2)
            D.sort(1)
            print(f"  {gname+' → '+rk:<24}{np.median(D[:, :5].mean(1)):>10.4f}"
                  f"{np.median(D[:, :10].mean(1)):>10.4f}")

    # ---------- 3. PCA ----------
    print("\n--- PCA（仅可视化，不作统计证据）---")
    ALL = np.stack([r["h"] for r in R if r["grp"]])
    lab = np.array([r["grp"] for r in R if r["grp"]])
    X = ALL - ALL.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    ev = (S ** 2) / (S ** 2).sum()
    Z = X @ Vt[:2].T
    print(f"  解释方差比 PC1={ev[0]:.3f}  PC2={ev[1]:.3f}  合计={ev[0]+ev[1]:.3f}")
    print(f"  {'group':<11}{'PC1 med':>10}{'PC1 IQR':>18}{'PC2 med':>10}{'PC2 IQR':>18}")
    for g in ("T_success", "G4", "G2", "G1"):
        m = lab == g
        if not m.any():
            continue
        a, b = Z[m, 0], Z[m, 1]
        print(f"  {g:<11}{np.median(a):>10.2f}{f'[{np.percentile(a,25):.1f},{np.percentile(a,75):.1f}]':>18}"
              f"{np.median(b):>10.2f}{f'[{np.percentile(b,25):.1f},{np.percentile(b,75):.1f}]':>18}")
    cen = {g: Z[lab == g].mean(0) for g in set(lab)}
    print("  质心两两距离:")
    ks = [k for k in ("T_success", "G4", "G2", "G1") if k in cen]
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            print(f"    {ks[i]:<11}–{ks[j]:<11}{np.linalg.norm(cen[ks[i]]-cen[ks[j]]):>8.3f}")

    # ---------- 4. linear probe ----------
    print("\n--- §5 linear probe：H 向量能否区分 G1 vs G4（严格 CV）---")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        A = np.stack([r["h"] for r in G["G1"]]); B = np.stack([r["h"] for r in G["G4"]])
        Xp = np.vstack([A, B]); yp = np.r_[np.ones(len(A)), np.zeros(len(B))]
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        for sc in ("accuracy", "balanced_accuracy", "roc_auc"):
            s = cross_val_score(pipe, Xp, yp, cv=cv, scoring=sc)
            print(f"  {sc:<20} {s.mean():.3f} ± {s.std():.3f}   folds={np.round(s,3)}")
        # 置换检验
        rng = np.random.default_rng(0)
        null = []
        for _ in range(200):
            yp2 = rng.permutation(yp)
            null.append(cross_val_score(pipe, Xp, yp2, cv=cv, scoring="balanced_accuracy").mean())
        obs = cross_val_score(pipe, Xp, yp, cv=cv, scoring="balanced_accuracy").mean()
        print(f"  置换检验 p = {np.mean(np.array(null) >= obs):.4f}（200 次置换）")
    except Exception as e:
        print(f"  unavailable: {e}")

    # ---------- 5. 分层 ----------
    print("\n--- §7 按 size bin 分层（G1 vs G4）---")
    print(f"  {'bin':<8}{'G1 n':>6}{'G1 logit':>10}{'G1 |H|':>9}{'G1 cos(H,muG4)':>16}"
          f"{'G4 n':>6}{'G4 logit':>10}{'G4 |H|':>9}{'G4 cos(H,muG4)':>16}")
    for b in ("<12", "12-18", "18-24", "24-32", ">=32"):
        a = [r for r in G["G1"] if r["bin"] == b]
        d = [r for r in G["G4"] if r["bin"] == b]
        if not a and not d:
            continue
        def s(g):
            if not g:
                return (0, np.nan, np.nan, np.nan)
            return (len(g), np.median([r["logit_decomp"] for r in g]),
                    np.median([np.linalg.norm(r["h"]) for r in g]),
                    np.median(cos(np.stack([r["h"] for r in g]), muH)))
        sa, sd = s(a), s(d)
        print(f"  {b:<8}{sa[0]:>6}{sa[1]:>10.3f}{sa[2]:>9.3f}{sa[3]:>16.4f}"
              f"{sd[0]:>6}{sd[1]:>10.3f}{sd[2]:>9.3f}{sd[3]:>16.4f}")

    # ---------- 6. 统计 ----------
    print("\n--- §9 关键对比（effect size + bootstrap CI）---")
    for lab_, f in (("G1 vs G4", lambda r: r["grp"] == "G4"),
                    ("G1 vs T_success", lambda r: r["grp"] == "T_success")):
        b = [r for r in R if f(r)]
        for key, nm in ((lambda r: np.linalg.norm(r["h"]), "|H|"),
                        (lambda r: float(cos(r["h"][None, :], muH)[0]), "cos(H,muG4)"),
                        (lambda r: r["logit_decomp"], "logit_decomp")):
            a_, b_ = [key(r) for r in G["G1"]], [key(r) for r in b]
            lo, hi = boot(a_, b_)
            print(f"  {lab_:<18}{nm:<16} G1={np.median(a_):>9.3f}  ref={np.median(b_):>9.3f}"
                  f"  cliffΔ={cliffs(a_,b_):>7.3f}  ΔCI95=[{lo:+.3f},{hi:+.3f}]")


if __name__ == "__main__":
    main()
