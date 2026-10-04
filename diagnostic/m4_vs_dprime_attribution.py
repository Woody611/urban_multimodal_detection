"""m4_vs_dprime_attribution.py — Phase E: M4 vs D′ GT-level attribution + 三模型 8-way phenotype。

ZERO GPU / 不重新推理。严格复用 gt_level_attribution.py 的 collect()（镜像 official_eval 的贪心匹配）。

用法:
  python -X utf8 diagnostic/m4_vs_dprime_attribution.py --out diagnostic/batch2_eval/_m4_dp_attrib.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "diagnostic"))

from official_eval import load_split  # noqa: E402
from gt_level_attribution import collect, size_of, SIZE_BINS, NAMES  # noqa: E402

IMAGES = ROOT / "data/processed/rgbid_split_train/images/val/visible"
LABELS = ROOT / "data/processed/rgbid_split_train/labels/val/visible"
DIRS = {
    "Dp": ROOT / "diagnostic/sepstem_clahe/best_full/results",
    "M4": ROOT / "diagnostic/batch2_eval/m4_best/results",
    "M1": ROOT / "diagnostic/batch1_eval/m1_best/results",
}
EXPECT = {"Dp": 0.51528, "M4": 0.50597, "M1": 0.49951}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="diagnostic/batch2_eval/_m4_dp_attrib.json")
    A = ap.parse_args()

    per_image, stats = load_split(IMAGES, LABELS)
    stems = {s for (_i, s, *_x) in per_image}

    print("=" * 104)
    print("§1 Provenance Gate")
    print("=" * 104)
    print(f"  images={stats['images']} corrupt={stats['corrupt']} 有效图={len(per_image)} GT={stats['gt']}")
    from official_eval import MAX_BOXES_PER_IMAGE, evaluate
    gates = []
    sets = {}
    for k, d in DIRS.items():
        sets[k] = {p.stem for p in d.glob('*.txt')}
        e = evaluate(per_image, d, max_boxes=MAX_BOXES_PER_IMAGE)
        gates.append((f"{k} 复现 {EXPECT[k]}", abs(e["mAP50-95"] - EXPECT[k]) < 5e-5,
                      f'{e["mAP50-95"]:.5f}'))
        gates.append((f"{k} 覆盖全部有效 GT 图", not (stems - sets[k]), f'{len(sets[k])} TXT'))
    gates.append(("D′/M4/M1 的 image 集合完全一致",
                  sets["Dp"] == sets["M4"] == sets["M1"], ""))
    for nm, ok, det in gates:
        print(f"    {'✅' if ok else '❌'} {nm}   {det}")
    ok = all(g[1] for g in gates)
    print(f"  M4_DPRIME_PROVENANCE_GATE = {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  ❌ STOP")
        return 1

    # ---- collect ----
    S = {k: collect(per_image, d, 0.50) for k, d in DIRS.items()}
    keys = sorted(S["Dp"].keys())
    area = {}
    for (i, s, w, h, gb, gc) in per_image:
        if gb is None:
            continue
        for c in np.unique(gc):
            for j, ii in enumerate(np.where(gc == c)[0]):
                b = gb[ii]; area[(i, int(c), int(j))] = float((b[2]-b[0])*(b[3]-b[1]))
    assert set(keys) == set(S["M4"].keys()) == set(S["M1"].keys()) and len(keys) == 2807

    def outcome(a, b):
        if a and b: return "BOTH_CORRECT"
        if a: return "Dp_ONLY"
        if b: return "M4_ONLY"
        return "BOTH_WRONG"

    rows = []
    for k in keys:
        d, m4, m1 = S["Dp"][k], S["M4"][k], S["M1"][k]
        rows.append(dict(key=k, cls=k[1], area=area[k], size=size_of(area[k]),
                         dp=d["matched"], m4=m4["matched"], m1=m1["matched"],
                         dp_best=d["best_iou"], m4_best=m4["best_iou"], m1_best=m1["best_iou"],
                         dp_iou=d["matched_iou"], m4_iou=m4["matched_iou"], m1_iou=m1["matched_iou"],
                         dp_conf=d["matched_conf"], m4_conf=m4["matched_conf"], m1_conf=m1["matched_conf"]))
    c = Counter(outcome(r["dp"], r["m4"]) for r in rows)
    tot = len(rows)
    print("\n" + "=" * 104)
    print("§2 GT-level outcome  D′ vs M4（IoU=0.50）")
    print("=" * 104)
    print(f"  {'outcome':<16}{'count':>7}{'% GT':>9}")
    for kk in ("BOTH_CORRECT", "Dp_ONLY", "M4_ONLY", "BOTH_WRONG"):
        print(f"  {kk:<16}{c[kk]:>7}{100*c[kk]/tot:>8.2f}%")
    print(f"  {'TOTAL':<16}{tot:>7}{100.0:>8.2f}%")
    net = c["Dp_ONLY"] - c["M4_ONLY"]
    print(f"  net = {net:+d}   gross churn = {c['Dp_ONLY']+c['M4_ONLY']}")
    assert sum(c.values()) == tot

    # ---- §3 size ----
    print("\n" + "=" * 104 + "\n§3 Size attribution（D′ vs M4）\n" + "=" * 104)
    print(f"  {'size':<9}{'GT':>6}{'D′_ONLY':>9}{'M4_ONLY':>9}{'net':>7}")
    for nm, _lo, _hi in SIZE_BINS:
        sub = [r for r in rows if r["size"] == nm]
        cc = Counter(outcome(r["dp"], r["m4"]) for r in sub)
        print(f"  {nm:<9}{len(sub):>6}{cc['Dp_ONLY']:>9}{cc['M4_ONLY']:>9}{cc['Dp_ONLY']-cc['M4_ONLY']:>+7}")

    # ---- §4 IoU sweep ----
    print("\n" + "=" * 104 + "\n§4 IoU-threshold sweep\n" + "=" * 104)
    print(f"  {'IoU':>6}{'BOTH':>8}{'D′_ONLY':>9}{'M4_ONLY':>9}{'net':>7}")
    iou_tab = {}
    for thr in (0.50, 0.65, 0.75, 0.85, 0.90):
        Sa = collect(per_image, DIRS["Dp"], thr); Sb = collect(per_image, DIRS["M4"], thr)
        cc = Counter()
        for k in keys:
            a, b = Sa[k]["matched"], Sb[k]["matched"]
            cc["B" if a and b else ("D" if a else ("M" if b else "W"))] += 1
        iou_tab[thr] = dict(both=cc["B"], d_only=cc["D"], m4_only=cc["M"], net=cc["D"]-cc["M"])
        print(f"  {thr:>6.2f}{cc['B']:>8}{cc['D']:>9}{cc['M']:>9}{cc['D']-cc['M']:>+7}")

    # ---- §5 class ----
    print("\n" + "=" * 104 + "\n§5 Class attribution（D′ vs M4）\n" + "=" * 104)
    print(f"  {'class':<13}{'GT':>6}{'D′_ONLY':>9}{'M4_ONLY':>9}{'net':>7}")
    for ci in range(12):
        sub = [r for r in rows if r["cls"] == ci]
        if not sub:
            continue
        cc = Counter(outcome(r["dp"], r["m4"]) for r in sub)
        print(f"  {NAMES[ci]:<13}{len(sub):>6}{cc['Dp_ONLY']:>9}{cc['M4_ONLY']:>9}{cc['Dp_ONLY']-cc['M4_ONLY']:>+7}")

    # ---- §6 rescue type ----
    dont = [r for r in rows if r["dp"] and not r["m4"]]
    m4on = [r for r in rows if r["m4"] and not r["dp"]]
    print("\n" + "=" * 104 + "\n§6 Rescue phenotype（D′ 相对 M4）\n" + "=" * 104)
    print(f"  D′_ONLY n={len(dont)}")
    R1 = [r for r in dont if r["m4_best"] > 0 and r["m4_best"] < 0.50]
    R2 = [r for r in dont if r["m4_best"] >= 0.50]
    R3 = [r for r in dont if r["m4_best"] == 0.0]
    for nm, g in (("R1 localization (M4 maxIoU<0.50)", R1), ("R2 ranking/assignment", R2), ("R3 no same-class candidate", R3)):
        print(f"    {nm:<36}{len(g):>5}  ({100*len(g)/max(len(dont),1):.1f}%)")
    print(f"  ⚠ R1 vs R3 无法区分「M4 把框放歪」与「那是另一个物体」⇒ NOT DISTINGUISHABLE FROM FROZEN PREDICTIONS")
    print(f"  M4_ONLY n={len(m4on)}")

    # ---- §7 confidence / IoU phenotype ----
    print("\n" + "=" * 104 + "\n§7 Confidence / IoU phenotype\n" + "=" * 104)
    def q(v, p):
        v = [x for x in v if x == x and x is not None]
        return float(np.percentile(v, p)) if v else float("nan")
    print(f"  {'outcome':<16}{'n':>6}{'area p50':>10}{'D′IoU p50':>11}{'D′conf p50':>12}{'M4IoU p50':>11}{'M4conf p50':>12}")
    for kk in ("BOTH_CORRECT", "Dp_ONLY", "M4_ONLY", "BOTH_WRONG"):
        sub = [r for r in rows if outcome(r["dp"], r["m4"]) == kk]
        print(f"  {kk:<16}{len(sub):>6}{q([r['area'] for r in sub],50):>10.0f}"
              f"{q([r['dp_iou'] for r in sub],50):>11.3f}{q([r['dp_conf'] for r in sub],50):>12.3f}"
              f"{q([r['m4_iou'] for r in sub],50):>11.3f}{q([r['m4_conf'] for r in sub],50):>12.3f}")

    # ---- §9 three-model 8-way ----
    print("\n" + "=" * 104 + "\n§9 三模型 8-way phenotype（D′ M4 M1）\n" + "=" * 104)
    eight = Counter(f"{int(r['dp'])}{int(r['m4'])}{int(r['m1'])}" for r in rows)
    print(f"  {'D′M4M1':<9}{'count':>7}{'% GT':>8}   含义")
    mean = {"111": "三者都命中", "110": "D′+M4 命中, M1 未命中 ← IR 已解释", "101": "D′+M1 命中, M4 未命中",
            "100": "只有 D′ ← D′ 特有（Depth/SepStem 候选）", "011": "M4+M1 命中, D′ 未命中",
            "010": "只有 M4 ← M4 反超 D′", "001": "只有 M1", "000": "三者都未命中"}
    for k in ("111", "110", "101", "100", "011", "010", "001", "000"):
        print(f"  {k:<9}{eight[k]:>7}{100*eight[k]/tot:>7.2f}%   {mean[k]}")
    print(f"  {'TOTAL':<9}{sum(eight.values()):>7}{100.0:>7.2f}%")
    assert sum(eight.values()) == 2807

    # ---- §10 linkage ----
    print("\n" + "=" * 104 + "\n§10 与 D′ vs M1 的 linkage\n" + "=" * 104)
    dp_only_vs_m1 = [r for r in rows if r["dp"] and not r["m1"]]
    m1_only_vs_dp = [r for r in rows if r["m1"] and not r["dp"]]
    for tag, grp in (("D′_ONLY(vs M1)", dp_only_vs_m1), ("M1_ONLY(vs D′)", m1_only_vs_dp)):
        cc = Counter(f"{int(r['dp'])}{int(r['m4'])}{int(r['m1'])}" for r in grp)
        covered = sum(v for k, v in cc.items() if k[1] == '1')
        print(f"  {tag:<18} n={len(grp):>4}  其中 M4 也命中 = {covered} ({100*covered/max(len(grp),1):.1f}%)  "
              f"| 110={cc['110']}  100={cc['100']}")
    _c = Counter(f"{int(r['dp'])}{int(r['m4'])}{int(r['m1'])}" for r in dp_only_vs_m1)
    _cov = sum(v for k, v in _c.items() if k[1] == "1")
    print(f"  ★ D′_ONLY(vs M1) 里 M4 也命中 = {100*_cov/max(len(dp_only_vs_m1),1):.1f}%  ({_cov}/{len(dp_only_vs_m1)})")
    _c2 = Counter(f"{int(r['dp'])}{int(r['m4'])}{int(r['m1'])}" for r in m1_only_vs_dp)
    _cov2 = sum(v for k, v in _c2.items() if k[1] == "0")
    print(f"  ★ M1_ONLY(vs D′) 里 M4 也未命中 = {100*_cov2/max(len(m1_only_vs_dp),1):.1f}%  ({_cov2}/{len(m1_only_vs_dp)})")

    out = dict(gate=bool(ok), n_gt=tot,
               outcome={k: c[k] for k in ("BOTH_CORRECT", "Dp_ONLY", "M4_ONLY", "BOTH_WRONG")},
               net=net, gross=c["Dp_ONLY"]+c["M4_ONLY"], iou_sweep=iou_tab,
               eight_way={k: eight[k] for k in ("111","110","101","100","011","010","001","000")},
               rescue_type=dict(R1=len(R1), R2=len(R2), R3=len(R3)))
    Path(A.out).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
