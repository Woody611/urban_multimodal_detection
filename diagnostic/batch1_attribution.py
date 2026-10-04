"""batch1_attribution.py — Phase A：D′ vs M1 的 IoU-bin / class 归因（ZERO GPU，不重新推理）。

复用 diagnostic/e1_region_gain/_e1_vs_dprime_boot.py 的已验证内核
（build_units / eval_units / ap_fast）与 scripts/official_eval.py 的装载与常量，
以保证与冻结官方口径**逐位一致**。

用法:
  python -X utf8 diagnostic/batch1_attribution.py \
      --a diagnostic/sepstem_clahe/best_full/results \
      --b diagnostic/batch1_eval/m1_best/results \
      --name-a "D′" --name-b "M1(RGB)" \
      --out diagnostic/batch1_eval/_attrib_dprime_vs_m1.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

IMAGES = ROOT / "data/processed/rgbid_split_train/images/val/visible"
LABELS = ROOT / "data/processed/rgbid_split_train/labels/val/visible"

_spec = importlib.util.spec_from_file_location(
    "e1boot", ROOT / "diagnostic/e1_region_gain/_e1_vs_dprime_boot.py")
E1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E1)

NAMES = E1.NAMES
IOU_THRESHOLDS = E1.IOU_THRESHOLDS
B_BOOT, SEED = 400, 20260928          # 与既有 e1_vs_dprime 工具同参数 ⇒ 可与历史 CI 直接对照


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="control（D′）results 目录")
    ap.add_argument("--b", required=True, help="treatment（M1）results 目录")
    ap.add_argument("--name-a", default="D′")
    ap.add_argument("--name-b", default="M1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=B_BOOT)
    A = ap.parse_args()

    Da, Db = Path(A.a).resolve(), Path(A.b).resolve()
    per_image, stats = E1.load_split(IMAGES, LABELS)
    N = len(per_image)
    print("=" * 104)
    print(f"Phase A 归因：{A.name_a} (control) vs {A.name_b} (treatment)")
    print(f"  images={stats['images']} corrupt={stats['corrupt']} 有效图={N} GT={stats['gt']}  cap={E1.MAX_BOXES_PER_IMAGE}")
    print(f"  control   = {Da}")
    print(f"  treatment = {Db}")
    print("=" * 104)

    U = {"A": E1.build_units(per_image, Da, 0.0, 1e18),
         "B": E1.build_units(per_image, Db, 0.0, 1e18)}
    allidx = list(range(N))
    RA = E1.eval_units(U["A"], allidx)
    RB = E1.eval_units(U["B"], allidx)

    # ---- 复刻断言：快速内核必须与 official_eval.evaluate 一致 ----
    from official_eval import evaluate
    ok = True
    for k, d, R in (("A", Da, RA), ("B", Db, RB)):
        off = evaluate(per_image, d, max_boxes=E1.MAX_BOXES_PER_IMAGE)
        for nm, x, y in (("mAP50-95", off["mAP50-95"], R["mAP5095"]),
                         ("mAP50", off["mAP50"], R["mAP50"]),
                         ("mAP75", off["mAP75"], R["mAP75"])):
            if abs(x - y) > 2e-6:
                ok = False
                print(f"  ❌ 复刻断言 FAIL [{k}] {nm}: official={x:.6f} fast={y:.6f}")
    print(f"  [复刻断言] {'PASS' if ok else 'FAIL'}")
    assert ok, "快速内核与官方 evaluator 不一致 —— 停止"

    # ---- 配对 bootstrap（同一重采样索引喂两个模型）----
    rng = np.random.default_rng(SEED)
    boot_iou = np.zeros((A.boot, len(IOU_THRESHOLDS)))
    boot_cls = np.zeros((A.boot, 12))
    for b in range(A.boot):
        idx = rng.integers(0, N, N).tolist()
        ia = E1.eval_units(U["A"], idx)["per_iou"]
        ib = E1.eval_units(U["B"], idx)["per_iou"]
        boot_iou[b] = ia - ib          # Δ = control(D′) − treatment(M1)
        # per-class AP50-95
        ca = E1.eval_units(U["A"], idx)["per_class"].mean(1)
        cb = E1.eval_units(U["B"], idx)["per_class"].mean(1)
        boot_cls[b] = ca - cb

    def ci(samples, point):
        lo, hi = np.percentile(samples, [2.5, 97.5])
        return float(lo), float(hi), float((samples > 0).mean()), float(point)

    out = {"meta": dict(control=str(Da), treatment=str(Db), n_img=N, gt=stats["gt"],
                        cap=E1.MAX_BOXES_PER_IMAGE, boot=A.boot, seed=SEED,
                        name_a=A.name_a, name_b=A.name_b),
           "reproduction_assert": bool(ok)}

    # ================= A1: IoU bin =================
    print("\n" + "=" * 104)
    print(f"A1  IoU-bin 分解   Δ 定义为 {A.name_a} − {A.name_b}（正 = {A.name_a} 更好）")
    print("=" * 104)
    print(f"  {'IoU':<6}{A.name_a:>10}{A.name_b:>10}{'Δ':>10}   {'95% CI':<24}{'P(Δ>0)':>8}  {'显著':<5}")
    rows1 = []
    for t in range(len(IOU_THRESHOLDS)):
        pa, pb = float(RA["per_iou"][t]), float(RB["per_iou"][t])
        d = pa - pb
        lo, hi, p, _ = ci(boot_iou[:, t], d)
        sig = not (lo <= 0 <= hi)
        rows1.append(dict(iou=float(IOU_THRESHOLDS[t]), a=pa, b=pb, delta=d,
                          ci=[lo, hi], p_gt0=p, significant=bool(sig)))
        print(f"  {IOU_THRESHOLDS[t]:<6.2f}{pa:>10.5f}{pb:>10.5f}{d:>+10.5f}   "
              f"[{lo:+.5f},{hi:+.5f}]   {p:>8.3f}  {'✅' if sig else '—'}")
    d_all = float(RA["per_iou"].mean() - RB["per_iou"].mean())
    lo, hi, p, _ = ci(boot_iou.mean(1), d_all)
    print(f"  {'ALL':<6}{RA['mAP5095']:>10.5f}{RB['mAP5095']:>10.5f}{d_all:>+10.5f}   "
          f"[{lo:+.5f},{hi:+.5f}]   {p:>8.3f}  {'✅' if not (lo <= 0 <= hi) else '—'}")
    out["A1_iou_bins"] = rows1
    out["A1_overall"] = dict(a=RA["mAP5095"], b=RB["mAP5095"], delta=d_all, ci=[lo, hi], p_gt0=p)

    # 单调性 / 抵消诊断
    dvec = np.array([r["delta"] for r in rows1])
    neg = dvec[dvec < 0].sum()
    pos = dvec[dvec > 0].sum()
    print(f"\n  — 诊断 —")
    print(f"    逐 bin Δ: {[round(x,5) for x in dvec]}")
    print(f"    正值合计 {pos:+.5f} / 负值合计 {neg:+.5f} / 净值 {dvec.sum()/10:+.5f}（净值 = 10 bin 均值）")
    print(f"    .50–.75 净 Δ = {(dvec[:6].mean()):+.5f}   .80–.95 净 Δ = {(dvec[6:].mean()):+.5f}")
    print(f"    单调性（从 .50 起首次转负的 bin）: "
          f"{next((float(IOU_THRESHOLDS[i]) for i in range(10) if dvec[i] < 0), None)}")
    print(f"    仅 .85/.90/.95 三项的净 Δ = {(dvec[7:].mean()):+.5f}")
    out["A1_diagnostics"] = dict(pos_total=float(pos), neg_total=float(neg),
                                 net=float(dvec.sum() / 10),
                                 low_505075=float(dvec[:6].mean()),
                                 high_8095=float(dvec[6:].mean()),
                                 per_bin=[float(x) for x in dvec])

    # ================= A2: class =================
    print("\n" + "=" * 104)
    print(f"A2  类别分解（AP50-95，METRIC_A 12 类分母）   Δ = {A.name_a} − {A.name_b}")
    print("=" * 104)
    print(f"  {'class':<13}{'GT n':>7}{A.name_a:>10}{A.name_b:>10}{'Δ':>10}   {'95% CI':<24}{'P(Δ>0)':>8}  {'显著':<5}")
    rows2 = []
    for c in range(12):
        pa = float(RA["per_class"][c].mean())
        pb = float(RB["per_class"][c].mean())
        d = pa - pb
        lo, hi, p, _ = ci(boot_cls[:, c], d)
        sig = not (lo <= 0 <= hi)
        rows2.append(dict(cls=c, name=NAMES[c], gt_n=int(RA["npos"][c]), a=pa, b=pb,
                          delta=d, ci=[lo, hi], p_gt0=p, significant=bool(sig)))
        print(f"  {NAMES[c]:<13}{int(RA['npos'][c]):>7}{pa:>10.4f}{pb:>10.4f}{d:>+10.4f}   "
              f"[{lo:+.4f},{hi:+.4f}]   {p:>8.3f}  {'✅' if sig else '—'}")
    sig_c = [r["name"] for r in rows2 if r["significant"] and r["delta"] > 0]
    sig_m1 = [r["name"] for r in rows2 if r["significant"] and r["delta"] < 0]
    out["A2_classes"] = rows2
    print(f"\n  — 诊断 —")
    print(f"    D′ 显著更好的类: {sig_c}")
    print(f"    M1 显著更好的类: {sig_m1}")
    negs = sum(r["delta"] for r in rows2 if r["delta"] < 0)
    poss = sum(r["delta"] for r in rows2 if r["delta"] > 0)
    print(f"    负向合计 {negs:+.5f} / 正向合计 {poss:+.5f} / Σ/12 = {sum(r['delta'] for r in rows2)/12:+.5f}")
    print(f"    最负两类占**负向总量** = {abs(negs and sorted(r['delta'] for r in rows2)[:2] and sum(sorted(r['delta'] for r in rows2)[:2]))/abs(negs)*100:.1f}%"
          f"（注意：除以净值会被放大，不是集中度指标）")
    out["A2_diagnostics"] = dict(neg_total=float(negs), pos_total=float(poss),
                                 sum_over_12=float(sum(r["delta"] for r in rows2) / 12),
                                 sig_control_better=sig_c, sig_treat_better=sig_m1)

    # ================= A3: 与历史事实交叉检查 =================
    print("\n" + "=" * 104)
    print("A3  与已锁定历史数字的交叉检查")
    print("=" * 104)
    checks = [
        ("overall Δ ≈ −0.01577（M1−D′ 口径）", abs((-d_all) - (-0.01577)) < 5e-4, f"{d_all:+.5f}"),
        ("mAP50 Δ（D′−M1）≈ +0.00178", abs((float(RA["mAP50"]) - float(RB["mAP50"])) - 0.00178) < 5e-4,
         f"{float(RA['mAP50'])-float(RB['mAP50']):+.5f}"),
        ("mAP75 Δ（D′−M1）≈ −0.01091", abs((float(RA["mAP75"]) - float(RB["mAP75"])) - (-0.01091)) < 5e-4,
         f"{float(RA['mAP75'])-float(RB['mAP75']):+.5f}"),
    ]
    for nm, okc, val in checks:
        print(f"  {'✅' if okc else '❌'} {nm}   实测 {val}")
    out["A3_checks"] = [dict(name=n, ok=bool(o), value=v) for n, o, v in checks]

    Path(A.out).parent.mkdir(parents=True, exist_ok=True)
    Path(A.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[report] {A.out}")
    return 0 if all(o for _, o, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
