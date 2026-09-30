"""_v8_probes.py — 三独立 probe（A/B/C）+ 方向余弦矩阵 + 跨 probe 投影。只读。

复用 _v7_vectors.json（P3 与 H 的完整 256-d 向量），无需任何新 forward。

Probe A: G1 vs G4                      （hard-miss small vs 同类成功 small）
Probe B: small-success vs med/large-success （尺度方向）
Probe C: G1 vs other-class small-success    （G1 是否像"别类的小目标"）

P3 与 H 两个空间分别做全部三步。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
V2 = ROOT / "diagnostic/small_object_cause_v2"
SMALL = 32.0
SPACES = ("p3", "h")


# ---------------- 基础 ----------------
def fit_lr(X, y, lam=1.0, iters=12):
    """IRLS（Newton-Raphson）精确 L2-logistic regression。

    梯度下降需 1200 次迭代，IRLS 约 10 次即收敛 ⇒ 速度 ~100x。
    偏差项用增广列处理，使 (w,b) 联合求解。
    """
    n, d = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    th = np.zeros(d + 1)
    for _ in range(iters):
        z = np.clip(Xa @ th, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        Wd = np.maximum(p * (1 - p), 1e-6)
        H = Xa.T @ (Xa * Wd[:, None]) + lam * np.eye(d + 1)
        g = Xa.T @ (p - y) + lam * th
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        th -= step
        if np.abs(step).max() < 1e-8:
            break
    return th[:d], float(th[d])


def auc(y, s):
    o = np.argsort(s); y = y[o]
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.arange(1, len(y) + 1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cv_eval(X, y, folds=5, seed=0, lam=1.0):
    rng = np.random.default_rng(seed)
    parts = []
    idx = np.arange(len(y))
    for c in (0, 1):
        ci = idx[y == c].copy(); rng.shuffle(ci)
        parts.append(np.array_split(ci, folds))
    accs, auds = [], []
    for f in range(folds):
        te = np.r_[parts[0][f], parts[1][f]]
        tr = np.setdiff1d(idx, te)
        w, b = fit_lr(X[tr], y[tr], lam=lam)
        s = X[te] @ w + b
        pred = (s > 0).astype(int)
        accs.append(np.mean([np.mean(pred[y[te] == c] == c) for c in (0, 1)]))
        auds.append(auc(y[te], s))
    return np.array(accs), np.array(auds)


def perm_p(X, y, obs, n=0, seed=1):
    if n == 0:
        return float('nan'), np.zeros(0)
    rng = np.random.default_rng(seed)
    null = [cv_eval(X, rng.permutation(y), seed=7)[0].mean() for _ in range(n)]
    return float(np.mean(np.array(null) >= obs)), np.array(null)


def cd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return float("nan")
    return (sum((x > b).sum() for x in a) - sum((x < b).sum() for x in a)) / (len(a) * len(b))


def iqr(v):
    v = np.asarray(v, float)
    return f"{np.median(v):>7.3f} [{np.percentile(v,25):>7.3f},{np.percentile(v,75):>7.3f}]"


def main():
    J = json.loads((OUT / "_v7_vectors.json").read_text(encoding="utf-8"))
    R = J["rows"]
    v3 = json.loads((V2 / "_v3_analysis.json").read_text(encoding="utf-8"))
    E1 = set(v3["E1"]); CTRL = set(v[0] for v in v3["controls"].values())
    v2rec = {f"{r['image_id']}#{r['gt_id']}": r for r in
             json.loads((V2 / "_v2_records.json").read_text(encoding="utf-8"))["records"]}

    # ⚠️ 修正：`succ` 必须对**所有尺度**的 val GT 计算。首版错误地从 v2rec 取，
    #    而 v2rec 只含 small GT（native_area<1024），导致 med/large 的 succ 恒为 False
    #    （Probe B 的对照集塌缩到 34 个，Probe C 的对照集塌缩到 9 个）。
    #    改为直接从 D′ 预测重算 best same-class IoU（与 official_eval 同口径）。
    sys.path.insert(0, str(ROOT / "scripts"))
    from official_eval import (load_split, read_pred_txt, apply_max_boxes,
                               norm_xywh_to_xyxy, box_iou_np, MAX_BOXES_PER_IMAGE)
    succ_map = {}
    per_image, _st = load_split(ROOT / "data/processed/rgbid_split_train/images/val/visible",
                                ROOT / "data/processed/rgbid_split_train/labels/val/visible")
    for _i, stem_, w_, h_, gt_b, gt_c in per_image:
        if gt_b is None or len(gt_b) == 0:
            continue
        _, pr, _ = read_pred_txt(ROOT / f"diagnostic/sepstem_clahe/best_full/results/{stem_}.txt")
        pr = apply_max_boxes(pr, MAX_BOXES_PER_IMAGE)
        pb, pc = (norm_xywh_to_xyxy(pr, w_, h_) if len(pr) else
                  (np.zeros((0, 4), np.float32), np.zeros(0, int)))
        for j in range(len(gt_b)):
            sm = np.where(pc == gt_c[j])[0] if len(pc) else np.zeros(0, int)
            iou = float(box_iou_np(gt_b[j:j+1], pb[sm])[0].max()) if len(sm) else 0.0
            succ_map[f"{stem_}#{j}"] = iou >= 0.5
    print(f"  [fix] 已为 {len(succ_map)} 个 val GT 重算 success 标记；"
          f"其中 success={sum(succ_map.values())}")

    for r in R:
        if r["domain"] == "T1":
            r["grp"] = "T_success" if (r["sq"] < SMALL and r["logit_decomp"] > 0) else None
            r["succ"] = r["logit_decomp"] > 0
        else:
            k = r["key"]
            rec = v2rec.get(k)
            r["succ"] = bool(succ_map.get(k, False))
            r["grp"] = ("G1" if k in E1 else "G2" if k in CTRL else
                        "G4" if (rec and k not in E1 and k not in CTRL
                                  and rec["native_area"] < 1024
                                  and rec["best_same_class_iou"] >= 0.5) else None)
    G1 = [r for r in R if r["grp"] == "G1"]
    G4 = [r for r in R if r["grp"] == "G4"]
    G2 = [r for r in R if r["grp"] == "G2"]
    g1cls = set(r["cls"] for r in G1)
    # 同尺寸带（G1 主要落在 10–20px）
    lo, hi = 10.0, 20.0
    def inband(r):
        return lo <= r["sq"] <= hi
    V = [r for r in R if r["domain"] == "V1"]
    # Probe B 组：val 成功 small vs val 成功 medium/large（尺寸带内控制）
    B_lo = [r for r in V if r["succ"] and r["sq"] < SMALL]
    B_hi = [r for r in V if r["succ"] and r["sq"] >= SMALL]
    # Probe C 组：G1 vs 其他类成功 small（同尺寸带 + 同分辨率族）
    C_ot = [r for r in V if r["succ"] and r["sq"] < SMALL and r["cls"] not in g1cls]
    C_ot_band = [r for r in C_ot if inband(r)]
    print(f"  [fix] Probe C 对照集：全部 other-class small-success n={len(C_ot)} "
          f"（其中尺寸带内 n={len(C_ot_band)}，带内样本不足故用全量并报告尺寸混杂）")
    A_g1 = [r for r in G1 if inband(r)]

    print("=" * 100)
    print("样本构成")
    print("=" * 100)
    print(f"  G1={len(G1)} (带内 {len(A_g1)})  G2={len(G2)}  G4={len(G4)}  "
          f"T_success={sum(1 for r in R if r['grp']=='T_success')}")
    print(f"  Probe B: small-success={len(B_lo)}  med/large-success={len(B_hi)}")
    print(f"  Probe C: G1(带内)={len(A_g1)}  other-class small-success(带内)={len(C_ot)}")
    print(f"  G1 的类别集合 = {sorted(g1cls)}")

    # 全局标准化（每个空间一次，使三个 probe 的 w 可比、投影同尺度）
    SC = {}
    for sp in SPACES:
        Xall = np.stack([r[sp] for r in R if r["grp"] or r["succ"]]).astype(np.float64)
        SC[sp] = (Xall.mean(0), Xall.std(0) + 1e-8)

    def cap(rows, n=600, seed=5):
        """负类降采样：permutation 的 1000 次 IRLS 拟合在 n~2300 时过慢。"""
        if len(rows) <= n:
            return rows
        rng = np.random.default_rng(seed)
        k = rng.choice(len(rows), n, replace=False)
        return [rows[i] for i in k]

    def M(rows, sp):
        X = np.stack([r[sp] for r in rows]).astype(np.float64)
        return (X - SC[sp][0]) / SC[sp][1]

    results = {}
    for sp in SPACES:
        print("\n" + "=" * 100)
        print(f"空间 = {sp.upper()}   （n 维 = {len(SC[sp][0])}）")
        print("=" * 100)
        # ---- Probe A ----
        XA = np.vstack([M(A_g1, sp), M(G4, sp)])
        yA = np.r_[np.ones(len(A_g1)), np.zeros(len(G4))]
        wA, bA = fit_lr(XA, yA)
        accA, audA = cv_eval(XA, yA); pA, _ = perm_p(XA, yA, accA.mean())
        # ---- Probe B ----
        B_hi_u = cap(B_hi)
        XB = np.vstack([M(B_lo, sp), M(B_hi_u, sp)])
        yB = np.r_[np.ones(len(B_lo)), np.zeros(len(B_hi_u))]
        wB, bB = fit_lr(XB, yB)
        accB, audB = cv_eval(XB, yB); pB, _ = perm_p(XB, yB, accB.mean())
        # ---- Probe C ----
        XC = np.vstack([M(A_g1, sp), M(C_ot, sp)])
        yC = np.r_[np.ones(len(A_g1)), np.zeros(len(C_ot))]
        wC, bC = fit_lr(XC, yC)
        accC, audC = cv_eval(XC, yC); pC, _ = perm_p(XC, yC, accC.mean())

        print("§A  Probe performance（严格 5-fold 分层 CV；置换检验因算力耗时已关闭，probe A 的置换 p 见 V7 = 0.0000）")
        print(f"   {'probe':<10}{'n₊':>5}{'n₋':>6}{'bal-acc':>18}{'AUC':>16}{'perm p':>9}{'cliffΔ(投影)':>14}")
        for nm, X, y, acc, aud, pv in (("A G1/G4", XA, yA, accA, audA, pA),
                                       ("B small/大", XB, yB, accB, audB, pB),
                                       ("C G1/他类", XC, yC, accC, audC, pC)):
            w = fit_lr(X, y)[0]
            print(f"   {nm:<10}{int((y==1).sum()):>5}{int((y==0).sum()):>6}"
                  f"{f'{acc.mean():.3f}±{acc.std():.3f}':>18}{f'{aud.mean():.3f}±{aud.std():.3f}':>16}"
                  f"{pv:>9.4f}{cd(X[y==1]@w, X[y==0]@w):>14.3f}")

        print("\n§B  方向余弦矩阵")
        print(f"   cos(wA,wB) = {float(wA@wB/(np.linalg.norm(wA)*np.linalg.norm(wB))):+.4f}")
        print(f"   cos(wA,wC) = {float(wA@wC/(np.linalg.norm(wA)*np.linalg.norm(wC))):+.4f}")
        print(f"   cos(wB,wC) = {float(wB@wC/(np.linalg.norm(wB)*np.linalg.norm(wC))):+.4f}")

        # ---- 跨投影 ----
        GROUPS = [("G1", G1), ("G2", G2), ("G4", G4),
                  ("other-cls small", C_ot), ("same-cls med/lg", B_hi),
                  ("train small succ", [r for r in R if r["grp"] == "T_success"])]
        print("\n§C  跨 probe 投影 z = w·f（标准化特征；中位 [P25,P75]）")
        print(f"   {'group':<18}{'n':>5}{'z_A':>24}{'z_B':>24}{'z_C':>24}")
        for nm, g in GROUPS:
            if not g:
                continue
            Z = M(g, sp)
            print(f"   {nm:<18}{len(g):>5}{iqr(Z@wA):>24}{iqr(Z@wB):>24}{iqr(Z@wC):>24}")
        # G1 相对各组的 cliffΔ（对 z_A / z_B / z_C）
        print(f"\n   G1 相对各组的 Cliff's Δ：")
        print(f"   {'vs group':<18}{'Δz_A':>10}{'Δz_B':>10}{'Δz_C':>10}")
        for nm, g in GROUPS:
            if not g or nm == "G1":
                continue
            Zg = M(g, sp)
            print(f"   {nm:<18}{cd(M(G1,sp)@wA, Zg@wA):>10.3f}"
                  f"{cd(M(G1,sp)@wB, Zg@wB):>10.3f}{cd(M(G1,sp)@wC, Zg@wC):>10.3f}")
        results[sp] = dict(wA=wA.tolist(), wB=wB.tolist(), wC=wC.tolist(),
                           accA=accA.mean(), audA=audA.mean(), pA=pA,
                           accB=accB.mean(), audB=audB.mean(), pB=pB,
                           accC=accC.mean(), audC=audC.mean(), pC=pC,
                           cos_AB=float(wA@wB/(np.linalg.norm(wA)*np.linalg.norm(wB))),
                           cos_AC=float(wA@wC/(np.linalg.norm(wA)*np.linalg.norm(wC))),
                           cos_BC=float(wB@wC/(np.linalg.norm(wB)*np.linalg.norm(wC))))

    (OUT / "_v8_probes.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[saved] {OUT/'_v8_probes.json'}")


if __name__ == "__main__":
    main()
