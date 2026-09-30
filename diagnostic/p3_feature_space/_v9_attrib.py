"""_v9_attrib.py — z_A 的可观测归因 + 冻结 w_A 的 train→val 投影。只读。

复用：_v7_vectors.json（P3/H 向量）、_v8_probes.json（**冻结的 w_A**）、
      _v3_analysis.json（val GT 的 density/border）、_v2_records.json、_v2_visual.json
本轮 0 次 forward。

⚠️ w_A **不重新拟合** —— 直接用 V8 保存的方向；标准化沿用 V8 同一构造。
⚠️ 严禁把 is_detected / best_iou / center_logit / G1-G4 标签作为解释变量（循环）。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
V2 = ROOT / "diagnostic/small_object_cause_v2"
SMALL = 32.0
FORBIDDEN = {"is_detected", "best_iou", "best_same_class_iou", "center_logit",
             "logit_decomp", "class_prob", "pos_score_max", "succ", "grp"}


def ols(X, y):
    Xa = np.hstack([np.ones((len(X), 1)), X])
    th, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    pred = Xa @ th
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    n, p = len(y), X.shape[1]
    adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else float("nan")
    return r2, adj, th, pred


def cv_r2(X, y, folds=5, seed=0, reps=5):
    """重复 K-fold CV R²（训练折拟合、测试折评估）—— 负值即不如截距。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        idx = rng.permutation(len(y))
        for f in range(folds):
            te = idx[f::folds]
            tr = np.setdiff1d(idx, te)
            if len(tr) < X.shape[1] + 5 or len(te) < 3:
                continue
            _, _, th, _ = ols(X[tr], y[tr])
            Xa = np.hstack([np.ones((len(te), 1)), X[te]])
            pred = Xa @ th
            ss_res = ((y[te] - pred) ** 2).sum()
            ss_tot = ((y[te] - y[tr].mean()) ** 2).sum()
            out.append(1 - ss_res / (ss_tot + 1e-12))
    return (float(np.mean(out)) if out else float("nan")), np.array(out)


def main():
    J = json.loads((OUT / "_v7_vectors.json").read_text(encoding="utf-8"))["rows"]
    P = json.loads((OUT / "_v8_probes.json").read_text(encoding="utf-8"))
    v3 = json.loads((V2 / "_v3_analysis.json").read_text(encoding="utf-8"))
    ANN = v3["annotated"]
    E1 = set(v3["E1"]); CTRL = set(v[0] for v in v3["controls"].values())
    v2rec = {f"{r['image_id']}#{r['gt_id']}": r for r in
             json.loads((V2 / "_v2_records.json").read_text(encoding="utf-8"))["records"]}
    vis = json.loads((V2 / "_v2_visual.json").read_text(encoding="utf-8"))
    VIS = {r["key"]: r for r in (vis["target"] + vis["control"])}

    sys.path.insert(0, str(ROOT / "scripts"))
    from official_eval import (load_split, read_pred_txt, apply_max_boxes,
                               norm_xywh_to_xyxy, box_iou_np, MAX_BOXES_PER_IMAGE)
    succ_map = {}
    per_image, _ = load_split(ROOT / "data/processed/rgbid_split_train/images/val/visible",
                              ROOT / "data/processed/rgbid_split_train/labels/val/visible")
    for _i, s_, w_, h_, gb, gc in per_image:
        if gb is None or len(gb) == 0:
            continue
        _, pr, _ = read_pred_txt(ROOT / f"diagnostic/sepstem_clahe/best_full/results/{s_}.txt")
        pr = apply_max_boxes(pr, MAX_BOXES_PER_IMAGE)
        pb, pc = (norm_xywh_to_xyxy(pr, w_, h_) if len(pr) else
                  (np.zeros((0, 4), np.float32), np.zeros(0, int)))
        for j in range(len(gb)):
            sm = np.where(pc == gc[j])[0] if len(pc) else np.zeros(0, int)
            succ_map[f"{s_}#{j}"] = (float(box_iou_np(gb[j:j+1], pb[sm])[0].max())
                                     if len(sm) else 0.0) >= 0.5

    for r in J:
        r["p3v"] = np.array(r["p3"], np.float64); r["hv"] = np.array(r["h"], np.float64)
        if r["domain"] == "T1":
            r["succ"] = r["logit_decomp"] > 0
            r["grp"] = "T_success" if (r["sq"] < SMALL and r["succ"]) else None
        else:
            r["succ"] = bool(succ_map.get(r["key"], False))
            k = r["key"]
            r["grp"] = ("G1" if k in E1 else "G2" if k in CTRL else
                        "G4" if (k in v2rec and k not in E1 and k not in CTRL
                                  and v2rec[k]["native_area"] < 1024 and r["succ"]) else None)

    # ---- 冻结 w_A + V8 同一标准化 ----
    res = {}
    for sp, key in (("p3", "p3v"), ("h", "hv")):
        rows_std = [r for r in J if r["grp"] or r["succ"]]
        Xall = np.stack([r[key] for r in rows_std])
        mu, sd = Xall.mean(0), Xall.std(0) + 1e-8
        wA = np.array(P[sp]["wA"])
        for r in J:
            r[f"zA_{sp}"] = float(((r[key] - mu) / sd) @ wA)
        res[sp] = dict(mu=mu, sd=sd, wA=wA)

    print("=" * 104)
    print("§C  冻结 w_A 的 train→val 投影（w_A 由 val G1/G4 学出后**冻结**，未重新拟合）")
    print("=" * 104)
    for sp in ("p3", "h"):
        print(f"\n  空间 = {sp.upper()}")
        print(f"   {'domain/group':<24}{'n':>6}{'median z_A':>13}{'IQR':>22}"
              f"{'P90':>10}{'P99':>10}{'max':>10}{'>G4_med 占比':>14}")
        g4med = np.median([r[f"zA_{sp}"] for r in J if r["grp"] == "G4"]) if any(
            r["grp"] == "G4" for r in J) else 0.0
        GRP = [("train small (all)", [r for r in J if r["domain"] == "T1" and r["sq"] < SMALL]),
               ("train med/lg", [r for r in J if r["domain"] == "T1" and r["sq"] >= SMALL]),
               ("train small success", [r for r in J if r["grp"] == "T_success"]),
               ("VAL G1", [r for r in J if r["grp"] == "G1"]),
               ("VAL G4", [r for r in J if r["grp"] == "G4"]),
               ("VAL other-cls small", [r for r in J if r["domain"] == "V1" and r["succ"]
                                        and r["sq"] < SMALL
                                        and r["cls"] not in {q["cls"] for q in J if q["grp"] == "G1"}]),
               ("VAL same-cls med/lg", [r for r in J if r["domain"] == "V1" and r["succ"]
                                        and r["sq"] >= SMALL])]
        for nm, g in GRP:
            if not g:
                continue
            v = np.array([r[f"zA_{sp}"] for r in g])
            print(f"   {nm:<24}{len(g):>6}{np.median(v):>13.3f}"
                  f"{f'[{np.percentile(v,25):.2f},{np.percentile(v,75):.2f}]':>22}"
                  f"{np.percentile(v,90):>10.3f}{np.percentile(v,99):>10.3f}{v.max():>10.3f}"
                  f"{np.mean(v > g4med):>14.1%}")
        # tail 对比：train small 的高 z_A 尾是否与 G1 同量级
        ts = np.array([r[f"zA_{sp}"] for r in J if r["domain"] == "T1" and r["sq"] < SMALL])
        g1 = np.array([r[f"zA_{sp}"] for r in J if r["grp"] == "G1"])
        print(f"   → train small 的 P99.5 = {np.percentile(ts,99.5):.3f}；"
              f"G1 的中位 = {np.median(g1):.3f}；"
              f"train small 中 ≥ G1 中位的比例 = {np.mean(ts >= np.median(g1)):.3%}")

    # ---------------- §A/§B/§D：val small GT 上的归因 ----------------
    VS = [r for r in J if r["domain"] == "V1" and r["key"] in v2rec]
    def cov(r):
        k = r["key"]; a = ANN.get(k, {}); rec = v2rec[k]; vv = VIS.get(k, {})
        return dict(
            native_sqrt=float(np.sqrt(rec["native_area"])), in_sqrt=r["sq"],
            border_norm=a.get("border_norm"), n_r1=a.get("n_r1"), n_r2=a.get("n_r2"),
            n_r3=a.get("n_r3"), n_r5=a.get("n_r5"), img_gt=a.get("img_gt"),
            n_same_r3=a.get("n_same_r3"), n_other_r3=a.get("n_other_r3"),
            visible_std=vv.get("visible_std"), visible_gradmean=vv.get("visible_gradmean"),
            visible_edgedens=vv.get("visible_edgedens"), visible_lapvar=vv.get("visible_lapvar"),
            infrared_std=vv.get("infrared_std"), infrared_gradmean=vv.get("infrared_gradmean"),
            depth_zero=(None if "depth_std" not in vv else float(vv.get("depth_std", 0) == 0)),
            cls=int(r["cls"]))
    C = {r["key"]: cov(r) for r in VS}

    from scipy.stats import spearmanr
    print("\n" + "=" * 104)
    print("§A  Univariate attribution（Spearman rho of z_A vs covariate；**已剔循环变量**）")
    print("=" * 104)
    for sp in ("p3", "h"):
        print(f"\n  空间 = {sp.upper()}   （n 因 visual 子集而分列）")
        print(f"   {'variable':<20}{'n_avail':>9}{'rho':>9}{'p':>11}")
        keys = ["native_sqrt", "in_sqrt", "border_norm", "n_r1", "n_r2", "n_r3", "n_r5",
                "img_gt", "n_same_r3", "n_other_r3", "visible_std", "visible_gradmean",
                "visible_edgedens", "visible_lapvar", "infrared_std", "infrared_gradmean"]
        for k in keys:
            xs, ys = [], []
            for r in VS:
                c = C[r["key"]]
                if c.get(k) is None:
                    continue
                xs.append(c[k]); ys.append(r[f"zA_{sp}"])
            if len(xs) < 20:
                continue
            rho, p = spearmanr(xs, ys)
            print(f"   {k:<20}{len(xs):>9}{rho:>+9.3f}{p:>11.1e}")

    print("\n" + "=" * 104)
    print("§B  Incremental multivariate R²（val small GT；CV R² = 5x5 重复 K-fold）")
    print("=" * 104)
    def build(rows, names, use_vis, use_cls):
        X = []
        for r in rows:
            c = C[r["key"]]; v = []
            for n in names:
                v.append(float(c.get(n) or 0.0))
            if use_vis:
                v += [float(c.get(q) or 0.0) for q in
                      ("visible_std", "visible_gradmean", "visible_edgedens",
                       "visible_lapvar", "infrared_std", "infrared_gradmean")]
            if use_cls:
                z = np.zeros(12); z[c["cls"]] = 1; v += z.tolist()
            X.append(v)
        return np.array(X)
    for sp in ("p3", "h"):
        print(f"\n  空间 = {sp.upper()}")
        # Model 1/2/3 需要完整覆盖 ⇒ 用有 visual 的子集；Model 0/1 另用全量
        rows_all = VS
        rows_v = [r for r in VS if all(C[r["key"]].get(q) is not None for q in
                                       ("visible_std", "visible_gradmean", "visible_edgedens",
                                        "visible_lapvar", "infrared_std", "infrared_gradmean"))]
        print(f"  几何模型用全量 n={len(rows_all)}；含 visual 的模型用 n={len(rows_v)}（visual 子集）")
        GEO = ["native_sqrt", "in_sqrt", "border_norm", "n_r1", "n_r3", "img_gt"]
        specs = [("Model 0 intercept", rows_all, [], False, False),
                 ("Model 1 geometry", rows_all, GEO, False, False),
                 ("Model 1' geometry(visual子集)", rows_v, GEO, False, False),
                 ("Model 2 +visual/modality", rows_v, GEO, True, False),
                 ("Model 3 +class FE", rows_v, GEO, True, True)]
        print(f"   {'model':<30}{'n':>6}{'k':>4}{'R²':>9}{'adjR²':>9}{'CV R²':>10}{'ΔR²':>9}")
        prev = None
        for nm, rows, names, uv, uc in specs:
            X = build(rows, names, uv, uc)
            y = np.array([r[f"zA_{sp}"] for r in rows])
            r2, adj, _, _ = ols(X, y)
            cv, _ = cv_r2(X, y)
            d = "" if prev is None else f"{r2-prev:+.4f}"
            print(f"   {nm:<30}{len(rows):>6}{X.shape[1]:>4}{r2:>9.4f}{adj:>9.4f}{cv:>10.4f}{d:>9}")
            prev = r2

    print("\n" + "=" * 104)
    print("§D  density / class 控制（z_A ~ …；CV R² 同法）")
    print("=" * 104)
    for sp in ("p3", "h"):
        print(f"\n  空间 = {sp.upper()}   （全量 val small n={len(VS)}）")
        print(f"   {'spec':<34}{'k':>4}{'R²':>9}{'adjR²':>9}{'CV R²':>10}")
        specs = [("density only (n_r3)",
                  lambda c: [float(c.get("n_r3") or 0)]),
                 ("density + class FE",
                  lambda c: [float(c.get("n_r3") or 0)] + (lambda z: z.tolist())(np.r_[np.zeros(c["cls"]), 1])[:0] or None),
                 ]
        # 显式构造三档
        def mk(names, use_cls, use_size):
            X = []
            for r in VS:
                c = C[r["key"]]; v = [float(c.get(n) or 0) for n in names]
                if use_size:
                    v += [float(c["native_sqrt"]), float(c["in_sqrt"])]
                if use_cls:
                    z = np.zeros(12); z[c["cls"]] = 1; v += z.tolist()
                X.append(v)
            return np.array(X)
        for nm, names, uc, us in (("density only", ["n_r3"], False, False),
                                  ("density + class FE", ["n_r3"], True, False),
                                  ("density + class FE + size", ["n_r3"], True, True),
                                  ("n_r3 + n_same_r3 + n_other_r3 + class + size",
                                   ["n_r3", "n_same_r3", "n_other_r3"], True, True)):
            X = mk(names, uc, us)
            y = np.array([r[f"zA_{sp}"] for r in VS])
            r2, adj, _, _ = ols(X, y)
            cv, _ = cv_r2(X, y)
            print(f"   {nm:<34}{X.shape[1]:>4}{r2:>9.4f}{adj:>9.4f}{cv:>10.4f}")

    (OUT / "_v9_attrib.json").write_text(json.dumps(
        {f"zA_{sp}": {r["domain"] + "|" + (r["grp"] or "-") + "|" + r["key"]: r[f"zA_{sp}"]
                      for r in J} for sp in ("p3", "h")}), encoding="utf-8")
    print(f"\n[saved] {OUT/'_v9_attrib.json'}")


if __name__ == "__main__":
    main()
