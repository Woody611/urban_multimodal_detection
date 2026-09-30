"""_v13_analyze.py — L17 dose-response 分析 + V12 复现门 + paired 检验。只读。"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
R = json.loads((OUT / "_v13_dose.json").read_text(encoding="utf-8"))
G1 = [r for r in R if r["grp"] == "G1"]
G4 = [r for r in R if r["grp"] == "G4"]
from scipy.stats import wilcoxon  # noqa: E402


def stat(v):
    v = np.asarray(v, float)
    return (np.median(v), np.percentile(v, 25), np.percentile(v, 75),
            float(np.mean(v > 0)),
            (wilcoxon(v).pvalue if len(v) and np.any(v != 0) else float("nan")))


def main():
    print("=" * 108)
    print("§2 V12 REPRODUCTION GATE —— R0 raw 必须复现 median Δlogit ≈ +5.869")
    print("=" * 108)
    a = [r["R0_raw_dlogit"] for r in G1]
    print(f"  V13 R0_raw  median Δlogit = {np.median(a):+.4f}   (V12 = +5.869)   n={len(a)}")
    print(f"  Δ(V13 − V12) = {np.median(a) - 5.869:+.4f}   "
          f"{'PASS' if abs(np.median(a) - 5.869) < 0.2 else 'FAIL → STOP'}")
    if abs(np.median(a) - 5.869) >= 0.2:
        return

    print("\n" + "=" * 108)
    print("§9 核心表：L17 spatial dose-response（G1 n=38；G4 n=194）")
    print("=" * 108)
    print(f"  {'R':<4}{'cells':>6}{'| G1 raw Δ':>12}{'IQR':>18}{'%>0':>7}{'p':>10}"
          f"{'| G1 norm Δ':>13}{'%>0':>7}{'| G4 raw Δ':>12}{'%>0':>7}"
          f"{'| rnd Δ':>10}{'%>0':>7}")
    dose_raw, dose_norm, dose_rnd = [], [], []
    for Rr in (0, 1, 2, 3):
        ra = [r[f"R{Rr}_raw_dlogit"] for r in G1]
        no = [r[f"R{Rr}_norm_dlogit"] for r in G1]
        rn = [r[f"R{Rr}_rnd_dlogit"] for r in G1]
        g4 = [r[f"R{Rr}_raw_dlogit"] for r in G4 if f"R{Rr}_raw_dlogit" in r]
        m, q1, q3, fr, p = stat(ra)
        mn, _, _, fn, _ = stat(no)
        mrn, _, _, frn, _ = stat(rn)
        mg, _, _, fg, _ = stat(g4) if g4 else (float("nan"), 0, 0, float("nan"), 0)
        ncell = int(np.median([r[f"R{Rr}_raw_ncells"] for r in G1]))
        dose_raw.append(m); dose_norm.append(mn); dose_rnd.append(mrn)
        print(f"  R{Rr:<3}{ncell:>6}{m:>12.3f}{f'[{q1:.2f},{q3:.2f}]':>18}{fr:>7.1%}{p:>10.2e}"
              f"{mn:>13.3f}{fn:>7.1%}{mg:>12.3f}{fg:>7.1%}{mrn:>10.3f}{frn:>7.1%}")

    print("\n  §7 displaced control（固定 +4 / +5 cells，不与中心邻域重叠）")
    for d in (4, 5):
        v = [r[f"disp{d}_dlogit"] for r in G1 if f"disp{d}_dlogit" in r]
        m, q1, q3, fr, p = stat(v)
        print(f"    +{d} cells: n={len(v):<3} median Δ={m:+.4f}  IQR=[{q1:.4f},{q3:.4f}]  %>0={fr:.1%}  p={p:.3f}")

    print("\n" + "=" * 108)
    print("§12 Dose-response interpretation")
    print("=" * 108)
    print(f"  G1 raw  Δlogit 轨迹  R0→R3: " + "  ".join(f"{x:+.3f}" for x in dose_raw))
    print(f"  G1 norm Δlogit 轨迹  R0→R3: " + "  ".join(f"{x:+.3f}" for x in dose_norm))
    print(f"  G1 rnd  Δlogit 轨迹  R0→R3: " + "  ".join(f"{x:+.3f}" for x in dose_rnd))
    print(f"  增量: R0→R1 {dose_raw[1]-dose_raw[0]:+.3f}  R1→R2 {dose_raw[2]-dose_raw[1]:+.3f}"
          f"  R2→R3 {dose_raw[3]-dose_raw[2]:+.3f}   合计 R0→R3 {dose_raw[3]-dose_raw[0]:+.3f}")
    print(f"  R0 占总增益比例 = {dose_raw[0]/dose_raw[3]:.1%}" if dose_raw[3] else "  n/a")

    print("\n  §10 绝对水平（**不以 logit=0 为阈值**，只作参考尺度）")
    for Rr in (0, 3):
        bl = np.median([r["base_logit"] for r in G1])
        pl = np.median([r[f"R{Rr}_raw_logit"] for r in G1])
        print(f"    R{Rr}: base logit {bl:+.3f} (p={1/(1+np.exp(-bl)):.2e}) → patched {pl:+.3f}"
              f" (p={1/(1+np.exp(-pl)):.2e})")

    print("\n  §11 L23 downstream 一致性（G1；cos 为与 TS L23 质心方向）")
    print(f"    {'R':<4}{'L23 norm Δ':>12}{'L23 cos 中位':>14}{'(base cos)':>12}")
    bc = np.median([r["L23_cos_base"] for r in G1])
    for Rr in (0, 1, 2, 3):
        dn = np.median([r[f"R{Rr}_raw_L23_norm"] - r["L23_norm_base"] for r in G1])
        cs = np.median([r[f"R{Rr}_raw_L23_cos"] for r in G1])
        print(f"    R{Rr:<3}{dn:>12.3f}{cs:>14.4f}{bc:>12.4f}")

    print("\n  G1-specificity（G1 raw vs G4 raw，各 R）")
    for Rr in (0, 1, 2, 3):
        ra = np.array([r[f"R{Rr}_raw_dlogit"] for r in G1])
        g4 = np.array([r[f"R{Rr}_raw_dlogit"] for r in G4 if f"R{Rr}_raw_dlogit" in r])
        if len(g4):
            print(f"    R{Rr}: G1 {np.median(ra):+.3f}  vs  G4 {np.median(g4):+.3f}"
                  f"   差 {np.median(ra)-np.median(g4):+.3f}")

    (OUT / "_v13_result.json").write_text(json.dumps(
        dict(dose_raw=dose_raw, dose_norm=dose_norm, dose_rnd=dose_rnd), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
