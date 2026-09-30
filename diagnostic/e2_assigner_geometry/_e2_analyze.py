"""_e2_analyze.py — E2 前置审计分析：reproduction gate / 分解 / alpha-beta-topk sweep（只读）

输入（全部为既有 diagnostic 产物 + 本目录新产物）：
  _e2_replay.npz                      本目录（本次 sweep 记录）
  ../small_gt_exposure/_sge_native_meta.json   uid -> native 尺寸（既有诊断产物，只读）

输出：_e2_tables.json + 控制台表格
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
NATIVE_META = OUT.parent / "small_gt_exposure" / "_sge_native_meta.json"

ALPHAS = (0.25, 0.50, 0.75)
BETAS = (2, 4, 6)
TOPKS = (5, 10, 20)
BASELINE = (0.50, 6, 10)


def cls_of(a):
    return "small" if a < 1024 else ("medium" if a < 9216 else "large")


def med(x):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64)
    return float(np.median(x)) if x.size else float("nan")


def main():
    meta = {int(k): v for k, v in json.loads(NATIVE_META.read_text(encoding="utf-8")).items()}
    z = np.load(OUT / "_e2_replay.npz")
    rmeta = json.loads((OUT / "_e2_replay_meta.json").read_text(encoding="utf-8"))
    settings = [tuple(s) for s in rmeta["settings"]]

    print("=" * 104)
    print("E2 PRE-FEASIBILITY — 分析")
    print("=" * 104)
    print(f"  真实 assigner = {rmeta['real_assigner']}   FULL_ASSIGNER_REPLAY = "
          f"{rmeta['full_assigner_replay']}   采样 = {rmeta['n_images']} 图 × {rmeta['epochs']} epoch")

    def ag(s):
        uid = z[f"AG__{s[0]}_{s[1]}_{s[2]}__uid"]
        val = z[f"AG__{s[0]}_{s[1]}_{s[2]}__val"]
        return {int(u): (float(v[0]), float(v[1]), float(v[2])) for u, v in zip(uid, val)}

    AG = {s: ag(s) for s in settings}

    # ---------------- §10 REPRODUCTION GATE ----------------
    base = AG[BASELINE]
    print("\n" + "=" * 104)
    print("§10 REPRODUCTION GATE — baseline alpha=.5 beta=6 topk=10 必须复现 Exposure Audit")
    print("=" * 104)
    expect = {"small": (4.0, 0.357), "medium": (10.0, 0.688), "large": (10.0, 0.826)}
    gate_ok = True
    print(f"  {'class':>8} {'n':>6} | {'n_pos(pre)':>10} {'n_pos(post)':>11} | "
          f"{'align':>7} | {'§13 n_pos':>10} {'§13 align':>10} | {'match':>6}")
    for c in ("small", "medium", "large"):
        us = [u for u in base if cls_of(meta[u]["native_area"]) == c]
        npre = med([base[u][0] for u in us]); npost = med([base[u][2] for u in us])
        al = med([base[u][1] for u in us])
        e_np, e_al = expect[c]
        ok = abs(npre - e_np) <= 1.0 and abs(al - e_al) <= 0.05
        gate_ok &= ok
        print(f"  {c:>8} {len(us):>6} | {npre:>10.1f} {npost:>11.1f} | {al:>7.3f} | "
              f"{e_np:>10.1f} {e_al:>10.3f} | {str(ok):>6}")
    print(f"\n  >>> REPRODUCTION {'PASS' if gate_ok else 'FAIL'}")
    if not gate_ok:
        print("  >>> FAIL ⇒ 按 §10 停止，不继续 sweep。")
        return 1

    # ---------------- §3 DECOMPOSITION ----------------
    cand_uid = z["CAND_uid"]; ncand = z["CAND_ncand"]; miou = z["CAND_maxiou"]
    mcls = z["CAND_maxcls"]; cnat = z["CAND_native"]
    CAND = {int(u): (int(n), float(i), float(c))
            for u, n, i, c in zip(cand_uid, ncand, miou, mcls)}
    CA = {}
    for a in ALPHAS:
        for b in BETAS:
            u_ = z[f"CA__{a}_{b}__uid"]; v_ = z[f"CA__{a}_{b}__val"]
            CA[(a, b)] = {int(x): float(y) for x, y in zip(u_, v_)}

    print("\n" + "=" * 104)
    print("§3 ALIGNMENT DECOMPOSITION（候选几何 / IoU / 分类）")
    print("=" * 104)
    print(f"  {'class':>8} {'n':>6} {'n_cand med':>11} {'maxIoU med':>11} {'maxCls med':>11} "
          f"{'maxAlign med':>13} {'align<.01%':>11}")
    dec = {}
    for c in ("small", "medium", "large"):
        us = [u for u in CAND if cls_of(meta[u]["native_area"]) == c]
        d = dict(n=len(us),
                 n_cand=med([CAND[u][0] for u in us]),
                 max_iou=med([CAND[u][1] for u in us]),
                 max_cls=med([CAND[u][2] for u in us]),
                 max_align=med([CA[(0.5, 6)][u] for u in us if u in CA[(0.5, 6)]]),
                 frac_align_lt01=float(np.mean([CA[(0.5, 6)].get(u, 0.0) < 0.01 for u in us])))
        dec[c] = d
        print(f"  {c:>8} {d['n']:>6} {d['n_cand']:>11.1f} {d['max_iou']:>11.4f} {d['max_cls']:>11.4f} "
              f"{d['max_align']:>13.5f} {100 * d['frac_align_lt01']:>10.1f}%")

    # 对数加性分解：log(align) = alpha*log(cls) + beta*log(IoU)
    print("\n  对数加性分解  Δlog(align) = alpha*Δlog(cls) + beta*Δlog(IoU)   （small 相对 large）")
    print(f"  {'alpha/beta':>11} {'Δlog(align)':>12} {'alpha*Δlog(cls)':>16} {'beta*Δlog(IoU)':>15} "
          f"{'cls 占比':>9} {'IoU 占比':>9}")
    decc = {}
    def meanlog(vals):
        v = np.asarray([x for x in vals if x > 0], dtype=np.float64)
        return float(np.mean(np.log(v))) if v.size else float("nan")
    for a in ALPHAS:
        for b in BETAS:
            ls = [math.log(CAND[u][2]) for u in CAND if cls_of(meta[u]["native_area"]) == "small" and CAND[u][2] > 0]
            ll = [math.log(CAND[u][2]) for u in CAND if cls_of(meta[u]["native_area"]) == "large" and CAND[u][2] > 0]
            is_ = [math.log(CAND[u][1]) for u in CAND if cls_of(meta[u]["native_area"]) == "small" and CAND[u][1] > 0]
            il = [math.log(CAND[u][1]) for u in CAND if cls_of(meta[u]["native_area"]) == "large" and CAND[u][1] > 0]
            dcls = float(np.mean(ls) - np.mean(ll)); diou = float(np.mean(is_) - np.mean(il))
            tot = a * dcls + b * diou
            decc[(a, b)] = dict(dlog_cls=dcls, dlog_iou=diou, a_cls=a * dcls, b_iou=b * diou, total=tot)
            print(f"  {a:>5}/{b:<5} {tot:>12.4f} {a*dcls:>16.4f} {b*diou:>15.4f} "
                  f"{100*a*dcls/max(abs(tot),1e-12):>8.1f}% {100*b*diou/max(abs(tot),1e-12):>8.1f}%")
    print("  （两者皆为负 ⇒ 都是劣势来源；占比说明谁主导）")

    # ---------------- §4 SWEEP ----------------
    print("\n" + "=" * 104)
    print("§4/§6 ALPHA / BETA / TOPK SWEEP（全 27 组；n_pos 用 pre-conflict，与 §13 同口径）")
    print("=" * 104)
    rows = []
    bsmall = [u for u in base if cls_of(meta[u]["native_area"]) == "small"]
    ref = {c: dict(n_pre=med([base[u][0] for u in base if cls_of(meta[u]["native_area"]) == c]),
                   n_post=med([base[u][2] for u in base if cls_of(meta[u]["native_area"]) == c]),
                   align=med([base[u][1] for u in base if cls_of(meta[u]["native_area"]) == c]),
                   zero=float(np.mean([base[u][0] == 0 for u in base if cls_of(meta[u]["native_area"]) == c])),
                   lt01=float(np.mean([base[u][1] < 0.01 for u in base if cls_of(meta[u]["native_area"]) == c])))
           for c in ("small", "medium", "large")}
    print(f"  {'alpha':>6} {'beta':>5} {'topk':>5} {'class':>8} {'n':>5} {'n_pos med':>10} "
          f"{'align med':>10} {'zero%':>7} {'align<.01%':>11}")
    for s in settings:
        A = AG[s]
        for c in ("small", "medium", "large"):
            us = [u for u in A if cls_of(meta[u]["native_area"]) == c]
            if not us:
                continue
            r = dict(alpha=s[0], beta=s[1], topk=s[2], cls=c, n=len(us),
                     n_pos=med([A[u][0] for u in us]),
                     n_pos_post=med([A[u][2] for u in us]),
                     align=med([A[u][1] for u in us]),
                     zero=float(np.mean([A[u][0] == 0 for u in us])),
                     lt01=float(np.mean([A[u][1] < 0.01 for u in us])))
            r["d_n_pos"] = r["n_pos"] - ref[c]["n_pre"]
            r["d_align"] = r["align"] - ref[c]["align"]
            r["d_zero"] = r["zero"] - ref[c]["zero"]
            r["d_lt01"] = r["lt01"] - ref[c]["lt01"]
            rows.append(r)
            mark = "  <- baseline" if s == BASELINE else ""
            print(f"  {s[0]:>6.2f} {s[1]:>5} {s[2]:>5} {c:>8} {len(us):>5} {r['n_pos']:>10.1f} "
                  f"{r['align']:>10.3f} {100*r['zero']:>6.1f}% {100*r['lt01']:>10.1f}%{mark}")

    # ---------------- §7 BETA ----------------
    print("\n" + "=" * 104)
    print("§7 BETA 专项（alpha=.5、topk=10 固定）")
    print("=" * 104)
    print(f"  {'beta':>5} {'class':>8} {'n_pos med':>10} {'align med':>10} {'zero%':>7} {'align<.01%':>11} "
          f"{'small/large align':>18}")
    betatab = {}
    for b in BETAS:
        vals = {}
        for c in ("small", "medium", "large"):
            r = [x for x in rows if x["alpha"] == 0.5 and x["beta"] == b and x["topk"] == 10 and x["cls"] == c][0]
            vals[c] = r
            print(f"  {b:>5} {c:>8} {r['n_pos']:>10.1f} {r['align']:>10.3f} {100*r['zero']:>6.1f}% "
                  f"{100*r['lt01']:>10.1f}%")
        sm, lg = vals["small"]["align"], vals["large"]["align"]
        md = vals["medium"]["align"]
        print(f"        -> small/large align = {sm/lg:.4f}   small/medium = {sm/md:.4f}   "
              f"n_pos small/medium/large = {vals['small']['n_pos']:.0f}/"
              f"{vals['medium']['n_pos']:.0f}/{vals['large']['n_pos']:.0f}")
        betatab[b] = dict(small=vals["small"], medium=vals["medium"], large=vals["large"],
                          ratio_small_large=sm / lg, ratio_small_medium=sm / md)

    # ---------------- §8 TOPK ----------------
    print("\n" + "=" * 104)
    print("§8 TOPK 专项（alpha=.5、beta=6 固定）")
    print("=" * 104)
    print(f"  {'topk':>5} {'class':>8} {'n_pos(pre)':>11} {'n_pos(post)':>12} {'align med':>10} "
          f"{'zero%':>7} {'align<.01%':>11}")
    for t in TOPKS:
        for c in ("small", "medium", "large"):
            r = [x for x in rows if x["alpha"] == 0.5 and x["beta"] == 6 and x["topk"] == t and x["cls"] == c][0]
            print(f"  {t:>5} {c:>8} {r['n_pos']:>11.1f} {r['n_pos_post']:>12.1f} {r['align']:>10.3f} "
                  f"{100*r['zero']:>6.1f}% {100*r['lt01']:>10.1f}%")

    (OUT / "_e2_tables.json").write_text(json.dumps(
        dict(reproduction_gate=dict(pass_=bool(gate_ok), expect=expect),
             decomposition=dec, log_decomposition={f"{a}_{b}": v for (a, b), v in decc.items()},
             sweep=rows, beta=betatab, baseline_ref=ref),
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {OUT/'_e2_tables.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
