"""_analyze300.py — Full-300 轨迹的离线分析（纯 CPU，无 forward，无 GPU）。

读 `epoch_NNN.npz`（含 vec_h / vec_role / vec_class / head_W / head_b），计算：
  §10 W/h 四分解：W0·h0 / W0·ht / Wt·h0 / Wt·ht
  §11 表征漂移：||ht-h0||、cos(ht,h0)
  §12 分类器漂移：||Wt-W0||、cos(Wt,W0)、||W_gt||
  §15 分离度：Δ_logit / Δ_feature / Δ_h_norm（G86 − CONTROL）
  §16 首次分叉（分辨率为相邻观测点区间）
  §17 close_mosaic 前后（0-based 289 vs 299；close_mosaic 在 0-based 290 触发）
  §18 机制分类 A/B/C/D

★ 预注册阈值（事前固定，事后不得修改）：
    MATERIAL_NATS = 2.0      # 某一项对 logit 的贡献变化 ≥ 2 nats 视为"实质"
    DIV_THRESH   = (1.0, 2.0, 5.0)   # Δ_logit 相对 ep0 扩大的三档，用于报告分叉分辨率
    CLOSE_EPS    = 0.5       # "无实质变化"的容差（nats）
用法: python -X utf8 diagnostic/full300_trajectory_probe/_analyze300.py [--dir D]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MATERIAL_NATS = 2.0
DIV_THRESH = (1.0, 2.0, 5.0)
CLOSE_EPS = 0.5
ROLES = ("G86", "CONTROL", "MISSED_NON86")

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=None)
A = ap.parse_args()
OUT = Path(A.dir).resolve() if A.dir else HERE


def main():
    npzs = sorted(OUT.glob("epoch_*.npz"))
    if not npzs:
        print(f"[analyze300] 找不到 epoch_*.npz in {OUT} —— STOP")
        return 1
    print("=" * 96)
    print("FULL-300 TRAJECTORY ANALYSIS（离线）")
    print("=" * 96)
    data = {}
    for p in npzs:
        d = np.load(p, allow_pickle=False)
        e = int(p.stem.split("_")[1])
        data[e] = dict(vec_h=d["vec_h"].astype(np.float64), vec_role=np.array([str(x) for x in d["vec_role"]]),
                       vec_cls=d["vec_cls"].astype(int), W=d["head_W"].astype(np.float64),
                       b=d["head_b"].astype(np.float64))
    eps = sorted(data)
    print(f"  观测点 {len(eps)} 个: {eps}")
    print(f"  每点 cell 数 = {len(data[eps[0]]['vec_role'])}；roles = "
          f"{ {r: int((data[eps[0]]['vec_role'] == r).sum()) for r in ROLES} }")

    e0 = eps[0]
    h0, W0, b0 = data[e0]["vec_h"], data[e0]["W"], data[e0]["b"]
    role, cls = data[e0]["vec_role"], data[e0]["vec_cls"]
    W0g = W0[cls]                                     # (N,256) 每个 cell 取其 GT 类的 W0
    b0g = b0[cls]
    t1_base = W0g @ h0.T if False else np.einsum("ij,ij->i", W0g, h0)

    dec_rows, rep_rows, cls_rows, sep_rows = [], [], [], []
    for t in eps:
        ht, Wt, bt = data[t]["vec_h"], data[t]["W"], data[t]["b"]
        Wtg, btg = Wt[cls], bt[cls]
        T1 = np.einsum("ij,ij->i", W0g, h0)
        T2 = np.einsum("ij,ij->i", W0g, ht)
        T3 = np.einsum("ij,ij->i", Wtg, h0)
        T4 = np.einsum("ij,ij->i", Wtg, ht)
        logit = T4 + btg
        dh = np.linalg.norm(ht - h0, axis=1)
        ch = np.einsum("ij,ij->i", ht, h0) / (np.linalg.norm(ht, axis=1) * np.linalg.norm(h0, axis=1) + 1e-12)
        dW = np.linalg.norm(Wtg - W0g, axis=1)
        cW = np.einsum("ij,ij->i", Wtg, W0g) / (np.linalg.norm(Wtg, axis=1) * np.linalg.norm(W0g, axis=1) + 1e-12)
        nW = np.linalg.norm(Wtg, axis=1)
        for r in ROLES:
            m = role == r
            if not m.any():
                continue
            med = lambda x: float(np.median(x[m]))
            dec_rows.append(dict(epoch=t, role=r, n=int(m.sum()),
                                 T1_W0h0=round(med(T1), 4), T2_W0ht=round(med(T2), 4),
                                 T3_Wth0=round(med(T3), 4), T4_Wtht=round(med(T4), 4),
                                 logit=round(med(logit), 4),
                                 T2_minus_T1=round(med(T2 - T1), 4), T3_minus_T1=round(med(T3 - T1), 4),
                                 T4_minus_T1=round(med(T4 - T1), 4)))
            rep_rows.append(dict(epoch=t, role=r, n=int(m.sum()),
                                 h_norm_median=round(med(np.linalg.norm(ht, axis=1)), 4),
                                 dhh0_median=round(med(dh), 4), cos_hh0_median=round(med(ch), 5)))
            cls_rows.append(dict(epoch=t, role=r, n=int(m.sum()),
                                 W_norm_median=round(med(nW), 4),
                                 dWW0_median=round(med(dW), 4), cos_WW0_median=round(med(cW), 5)))
        g = lambda x, r: float(np.median(x[role == r]))
        sep_rows.append(dict(epoch=t,
                             d_logit=round(g(logit, "G86") - g(logit, "CONTROL"), 4),
                             d_feature=round(g(T4, "G86") - g(T4, "CONTROL"), 4),
                             d_h_norm=round(g(np.linalg.norm(ht, axis=1), "G86")
                                            - g(np.linalg.norm(ht, axis=1), "CONTROL"), 4),
                             d_cosH=round(g(ch, "G86") - g(ch, "CONTROL"), 5),
                             d_W_norm=round(g(nW, "G86") - g(nW, "CONTROL"), 4),
                             d_cosW=round(g(cW, "G86") - g(cW, "CONTROL"), 5)))
    for nm, rows in (("trajectory_decomposition.csv", dec_rows),
                     ("trajectory_representation_drift.csv", rep_rows),
                     ("trajectory_classifier_drift.csv", cls_rows),
                     ("trajectory_summary.csv", sep_rows)):
        with open(OUT / nm, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---------- 打印 ----------
    print("\n§15 分离度（G86 − CONTROL 的中位差）")
    print(f"  {'ep(csv)':>8}{'Δ_logit':>10}{'Δ_feature':>11}{'Δ_|h|':>9}{'Δ_cosH':>9}{'Δ_|W|':>9}{'Δ_cosW':>9}")
    for r in sep_rows:
        print(f"  {r['epoch']+1:>8}{r['d_logit']:>10.3f}{r['d_feature']:>11.3f}{r['d_h_norm']:>9.3f}"
              f"{r['d_cosH']:>9.4f}{r['d_W_norm']:>9.3f}{r['d_cosW']:>9.4f}")
    print("\n§10 四分解（G86 中位；T1=W0·h0 T2=W0·ht T3=Wt·h0 T4=Wt·ht）")
    print(f"  {'ep(csv)':>8}{'T1':>9}{'T2':>9}{'T3':>9}{'T4':>9}{'logit':>9}{'T2-T1':>9}{'T3-T1':>9}")
    for r in dec_rows:
        if r["role"] != "G86":
            continue
        print(f"  {r['epoch']+1:>8}{r['T1_W0h0']:>9.3f}{r['T2_W0ht']:>9.3f}{r['T3_Wth0']:>9.3f}"
              f"{r['T4_Wtht']:>9.3f}{r['logit']:>9.3f}{r['T2_minus_T1']:>+9.3f}{r['T3_minus_T1']:>+9.3f}")

    # ---------- §16 首次分叉 ----------
    d0 = sep_rows[0]["d_logit"]
    div = {}
    for th in DIV_THRESH:
        hit = next((r["epoch"] for r in sep_rows if r["d_logit"] <= d0 - th), None)
        div[f"gap_widens_by_{th}nats"] = (f"between csv ep{hit} and ep{hit+2}" if hit is not None and hit in eps
                                          else "not reached")
    print(f"\n§16 首次分叉（Δ_logit 相对 ep0 扩大）  ep0 Δ_logit = {d0:.3f}")
    for k, v in div.items():
        print(f"    {k:<26} {v}")

    # ---------- §17 close_mosaic ----------
    cm = [r for r in sep_rows if r["epoch"] in (274, 288, 298)]
    print("\n§17 close_mosaic 边界（0-based 290 触发；观测点 0-based 274/288/298 ⇒ csv 275/289/299）")
    for r in (cm if cm else sep_rows[-3:]):
        print(f"    csv ep{r['epoch']+1:>4}  Δ_logit={r['d_logit']:>8.3f}  Δ_feature={r['d_feature']:>8.3f}")

    # ---------- §18 A/B/C/D ----------
    _g86 = [r for r in dec_rows if r["role"] == "G86"]
    if not _g86:
        print("\n[analyze300] 无 G86 记录 —— 无法做 §18 机制分类（通常因观测图数过少）。")
        print("[analyze300] 已写出的 CSV 仍然有效；跳过 A/B/C/D 分类。")
        return 0
    last = _g86[-1]
    rep_eff = abs(last["T2_minus_T1"]); cls_eff = abs(last["T3_minus_T1"])
    if rep_eff < CLOSE_EPS and cls_eff < CLOSE_EPS:
        case = "D"
    elif rep_eff >= MATERIAL_NATS and cls_eff < MATERIAL_NATS:
        case = "A"
    elif cls_eff >= MATERIAL_NATS and rep_eff < MATERIAL_NATS:
        case = "B"
    elif rep_eff >= MATERIAL_NATS and cls_eff >= MATERIAL_NATS:
        case = "C"
    else:
        case = "D"
    print(f"\n§18 机制分类（末观测点，G86 中位；MATERIAL_NATS={MATERIAL_NATS}）")
    print(f"    |W0·ht − W0·h0| = {rep_eff:.3f} nats   （表征侧贡献）")
    print(f"    |Wt·h0 − W0·h0| = {cls_eff:.3f} nats   （分类器侧贡献）")
    print(f"    ⇒ CASE {case}")
    (OUT / "mechanism_report.md").write_text("\n".join([
        "# Mechanism report — Full-300 trajectory", "",
        f"```text\nCASE = {case}", f"representation_effect = {rep_eff:.3f} nats",
        f"classifier_effect     = {cls_eff:.3f} nats",
        f"MATERIAL_NATS         = {MATERIAL_NATS}\n```", "",
        "## 分离度序列（csv epoch 口径）", "",
        "| csv ep | Δ_logit | Δ_feature | Δ_h_norm | Δ_cosH | Δ_W_norm | Δ_cosW |",
        "|---:|---:|---:|---:|---:|---:|---:|"] +
        [f"| {r['epoch']+1} | {r['d_logit']} | {r['d_feature']} | {r['d_h_norm']} | "
         f"{r['d_cosH']} | {r['d_W_norm']} | {r['d_cosW']} |" for r in sep_rows] + ["",
         "## 首次分叉", ""] + [f"- {k}: {v}" for k, v in div.items()] + ["",
         "## 因果限制", "",
         "- 本分析是**单条轨迹**，无干预 ⇒ 只能给 temporal association，不能给因果。",
         "- close_mosaic 前后的差异只能报 temporal association（§17）。",
         "- 观测点分辨率为相邻两点区间，不得假装更精确（§16）。", ""]),
        encoding="utf-8")
    json.dump(dict(observe_epochs_0based=eps, observe_epochs_csv=[e + 1 for e in eps],
                   case=case, representation_effect_nats=rep_eff, classifier_effect_nats=cls_eff,
                   first_divergence=div, separation=sep_rows,
                   thresholds=dict(MATERIAL_NATS=MATERIAL_NATS, DIV_THRESH=list(DIV_THRESH), CLOSE_EPS=CLOSE_EPS),
                   limitations="single trajectory, no intervention; temporal association only"),
              open(OUT / "trajectory_decomposition.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[analyze300] 写出 trajectory_{{'summary,per_gt,classifier_drift,representation_drift,decomposition,tal'}}.csv"
          f" + mechanism_report.md + trajectory_decomposition.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
