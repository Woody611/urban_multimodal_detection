"""_v7_addendum.py — 补齐两处决定性证据（只读）：

  1. **P3 空间**的 kNN 距离（主脚本只做了 H 空间）—— 判断 G1 在 P3 上是否已经离群
  2. **linear probe**（sklearn 不可用 → 自实现 L2-logistic regression + 严格 5-fold 分层 CV + 置换检验）
     分别在 **P3** 与 **H** 上做，比较「信息在哪一层就可分」

⚠️ 说明：分类 logit 本身就是 h 的线性函数（logit_c = W_c·h + b_c，精确等式），
   因此 **H 空间的线性 probe 与 head 的可分性是同一件事**，其高分不能单独当作
   「representation 有差异」的证据。真正有判别力的是 **P3 空间的 probe**：
   若 P3 已可分 → 信息在 P3 层；若 P3 不可分而 H 可分 → 分离是在 head 内部被造出来的。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
V2 = ROOT / "diagnostic/small_object_cause_v2"
SMALL = 32.0


def l2norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def fit_lr(X, y, lam=1.0, iters=800, lr=0.5):
    """L2 正则 logistic regression（numpy）。返回 (w, b)。"""
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = p - y
        gw = X.T @ g / n + lam * w / n
        gb = g.mean()
        w -= lr * gw; b -= lr * gb
    return w, b


def auc(y, s):
    o = np.argsort(s); y = y[o]
    r = np.arange(1, len(y) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cv_probe(X, y, folds=5, seed=0, lam=1.0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    # 分层折
    parts = []
    for c in (0, 1):
        ci = idx[y == c]; rng.shuffle(ci)
        parts.append(np.array_split(ci, folds))
    accs, auds = [], []
    for f in range(folds):
        te = np.r_[parts[0][f], parts[1][f]]
        tr = np.setdiff1d(idx, te)
        w, b = fit_lr(X[tr], y[tr], lam=lam)
        s = X[te] @ w + b
        pred = (s > 0).astype(int)
        # balanced accuracy
        rec = [np.mean(pred[y[te] == c] == c) for c in (0, 1)]
        accs.append(np.mean(rec))
        auds.append(auc(y[te], s))
    return np.array(accs), np.array(auds)


def main():
    J = json.loads((OUT / "_v7_vectors.json").read_text(encoding="utf-8"))
    R = J["rows"]
    for r in R:
        r["p3"] = np.array(r["p3"], np.float32); r["h"] = np.array(r["h"], np.float32)
    v3 = json.loads((V2 / "_v3_analysis.json").read_text(encoding="utf-8"))
    E1 = set(v3["E1"]); CTRL = set(v[0] for v in v3["controls"].values())
    v2rec = {f"{r['image_id']}#{r['gt_id']}": r for r in
             json.loads((V2 / "_v2_records.json").read_text(encoding="utf-8"))["records"]}
    for r in R:
        if r["domain"] == "T1":
            r["grp"] = "T_success" if (r["sq"] < SMALL and r["logit_decomp"] > 0) else None
        else:
            k = r["key"]
            r["grp"] = ("G1" if k in E1 else "G2" if k in CTRL else
                        "G4" if (k in v2rec and k not in E1 and k not in CTRL
                                  and v2rec[k]["native_area"] < 1024
                                  and v2rec[k]["best_same_class_iou"] >= 0.5) else None)
    G = {g: [r for r in R if r["grp"] == g] for g in ("G1", "G2", "G4", "T_success")}

    print("=" * 96)
    print("§补-1  P3 空间 与 H 空间的 kNN（L2 归一化后的欧氏距离，k=5/10）")
    print("=" * 96)
    print(f"{'query → ref':<26}{'P3 k=5':>10}{'P3 k=10':>10}{'H k=5':>10}{'H k=10':>10}")
    for space in ("p3", "h"):
        pass
    REF = {g: {sp: l2norm(np.stack([r[sp] for r in G[g]])) for sp in ("p3", "h")}
           for g in ("T_success", "G4") if G[g]}
    for q in ("G1", "G2", "G4", "T_success"):
        if not G[q]:
            continue
        Q = {sp: l2norm(np.stack([r[sp] for r in G[q]])) for sp in ("p3", "h")}
        for rk in REF:
            if rk == q:
                continue
            out = []
            for sp in ("p3", "h"):
                D = np.linalg.norm(Q[sp][:, None, :] - REF[rk][sp][None, :, :], axis=2)
                D.sort(1)
                out += [np.median(D[:, :5].mean(1)), np.median(D[:, :10].mean(1))]
            print(f"{q+' → '+rk:<26}{out[0]:>10.4f}{out[1]:>10.4f}{out[2]:>10.4f}{out[3]:>10.4f}")

    print("\n" + "=" * 96)
    print("§补-2  linear probe  G1 vs G4（严格 5-fold 分层 CV + 200 次置换检验）")
    print("=" * 96)
    A = G["G1"]
    for ref_name in ("G4", "T_success"):
        B = G[ref_name]
        y = np.r_[np.ones(len(A)), np.zeros(len(B))]
        for sp, nm in (("p3", "P3 空间"), ("h", "H 空间（≈head 自身）")):
            X = np.vstack([np.stack([r[sp] for r in A]), np.stack([r[sp] for r in B])]).astype(np.float64)
            X = (X - X.mean(0)) / (X.std(0) + 1e-8)      # 标准化
            acc, aud = cv_probe(X, y)
            rng = np.random.default_rng(1)
            null = []
            for _ in range(200):
                yp = rng.permutation(y)
                a2, _ = cv_probe(X, yp, folds=5, seed=7)
                null.append(a2.mean())
            p = float(np.mean(np.array(null) >= acc.mean()))
            print(f"  G1 vs {ref_name:<11}{nm:<20} balanced_acc={acc.mean():.3f}±{acc.std():.3f}"
                  f"  AUC={aud.mean():.3f}±{aud.std():.3f}  置换 p={p:.4f}")

    print("\n  参照：以下为「随机线性可分性上限」的直觉对照 —— 同组内 5 折（无信号）")
    for ref_name in ("G4",):
        B = G[ref_name]
        y = np.r_[np.ones(len(A)), np.zeros(len(B))]
        X = np.vstack([np.stack([r["p3"] for r in A]), np.stack([r["p3"] for r in B])]).astype(np.float64)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        yp = np.random.default_rng(3).permutation(y)
        a2, _ = cv_probe(X, yp, folds=5, seed=7)
        print(f"    G1 vs {ref_name}（标签打乱）  balanced_acc={a2.mean():.3f}±{a2.std():.3f}")


if __name__ == "__main__":
    main()
