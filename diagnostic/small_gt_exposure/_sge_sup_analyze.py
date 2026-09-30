"""_sge_sup_analyze.py — 监督质量的 native 分桶汇总（Table 2 + §10 Exposure-B）

与 §13 / V5 的差别：V5 按 **增强后** 面积分桶；本脚本按 **native** 面积分桶，
因此直接回答「native 小的 GT，在它活下来的那些实例上，拿到的监督质量如何」。

同时给出 Exposure-B 所需的 `n_pos > 0` 与 `best_align >= 0.01` 比例。
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent


def bkt(s):
    if s < 8:
        return "<8"
    if s < 12:
        return "8-12"
    if s < 18:
        return "12-18"
    if s < 24:
        return "18-24"
    if s < 32:
        return "24-32"
    if s < 96:
        return "32-96"
    return ">96"


def cls_of(a):
    return "small" if a < 1024 else ("medium" if a < 9216 else "large")


def med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


def main():
    d = json.loads((OUT / "_sge_supervision.json").read_text(encoding="utf-8"))
    rows = [r for r in d["rows"] if "err" not in r and r.get("native_area")]
    print("=" * 96)
    print("TABLE 2 — Supervision quality（按 **native** 面积分桶；真实 assigner topk10/a0.5/b6.0）")
    print("=" * 96)
    print(f"  samples(images×epochs)={d['n_images']}×{d['epochs']}  最终 GT 记录 n={len(rows)}")
    sizes = np.array([r["native_area"] for r in rows])
    print(f"  native sqrt: 中位 {math.sqrt(float(np.median(sizes))):.1f}px  "
          f"p10 {math.sqrt(float(np.percentile(sizes,10))):.1f}  p90 {math.sqrt(float(np.percentile(sizes,90))):.1f}")

    def table(keyfn, order):
        out = []
        for k in order:
            g = [r for r in rows if keyfn(r) == k]
            if not g:
                continue
            out.append(dict(
                key=k, n=len(g),
                n_pos_med=med([r["n_pos"] for r in g]),
                align_med=med([r["best_align"] for r in g]),
                tgt_med=med([r["pos_target_max"] for r in g]),
                zero_pos=float(np.mean([r["n_pos"] == 0 for r in g])),
                align_lt_01=float(np.mean([r["best_align"] < 0.01 for r in g])),
                aug_area_med=float(np.median([r["aug_area"] for r in g])),
                s8=float(np.mean([8.0 in r.get("pos_strides", []) for r in g])),
                s16=float(np.mean([16.0 in r.get("pos_strides", []) for r in g])),
                s32=float(np.mean([32.0 in r.get("pos_strides", []) for r in g])),
            ))
        return out

    print(f"\n  {'native sqrt':>12} {'n':>6} {'n_pos med':>10} {'align med':>10} {'tgt max med':>12} "
          f"{'zero-pos%':>10} {'align<.01%':>11} {'aug_area med':>13}")
    t1 = table(lambda r: bkt(math.sqrt(r["native_area"])),
               ["<8", "8-12", "12-18", "18-24", "24-32", "32-96", ">96"])
    for r in t1:
        print(f"  {r['key']:>12} {r['n']:>6} {r['n_pos_med']:>10.1f} {r['align_med']:>10.3f} "
              f"{r['tgt_med']:>12.3f} {100*r['zero_pos']:>9.1f}% {100*r['align_lt_01']:>10.1f}% "
              f"{r['aug_area_med']:>13.0f}")

    print(f"\n  {'class':>8} {'n':>6} {'n_pos med':>10} {'align med':>10} {'tgt max med':>12} "
          f"{'zero-pos%':>10} {'align<.01%':>11} {'pos@8':>7} {'pos@16':>7} {'pos@32':>7}")
    t2 = table(lambda r: cls_of(r["native_area"]), ["small", "medium", "large"])
    for r in t2:
        print(f"  {r['key']:>8} {r['n']:>6} {r['n_pos_med']:>10.1f} {r['align_med']:>10.3f} "
              f"{r['tgt_med']:>12.3f} {100*r['zero_pos']:>9.1f}% {100*r['align_lt_01']:>10.1f}% "
              f"{100*r['s8']:>6.1f}% {100*r['s16']:>6.1f}% {100*r['s32']:>6.1f}%")

    if len(t2) >= 3:
        s, m, l = t2[0], t2[1], t2[2]
        print("\n  small/medium  n_pos={:.3f} align={:.3f}   small/large  n_pos={:.3f} align={:.3f}".format(
            s["n_pos_med"] / max(m["n_pos_med"], 1e-9), s["align_med"] / max(m["align_med"], 1e-9),
            s["n_pos_med"] / max(l["n_pos_med"], 1e-9), s["align_med"] / max(l["align_med"], 1e-9)))
        print(f"  zero-pos 超额（small - large）: {100*(s['zero_pos']-l['zero_pos']):.1f} 个百分点")

    # §10 Exposure-B（诊断阈值，非训练规则）
    print("\n" + "=" * 96)
    print("§10 Exposure 定义（native 分桶，二值 + 连续）")
    print("=" * 96)
    print(f"  {'class':>8} {'n':>6} {'Exposure-B(>0)':>15} {'Exposure-B(>=.01)':>18} "
          f"{'median n_pos':>13} {'median align':>13}")
    for r in t2:
        print(f"  {r['key']:>8} {r['n']:>6} {1-r['zero_pos']:>15.4f} {1-r['align_lt_01']:>18.4f} "
              f"{r['n_pos_med']:>13.1f} {r['align_med']:>13.3f}")
    print("\n  注：Exposure-A（存在性，来自 replay）与 Exposure-B（有效监督，来自本表）"
          "是两件事，见 REPORT §5 的合成。")

    # 与 §13（V5，按增强后面积分桶）对照
    print("\n" + "=" * 96)
    print("对照 §13（V5 按 **增强后** 面积分桶，n_pos / best_align / align<.01）")
    print("=" * 96)
    v5 = json.loads((OUT.parent / "small_object_train_replay/_v5_replay.json").read_text(encoding="utf-8"))
    vrows = [r for r in v5["rows"] if "err" not in r]
    for c, ref in (("small", dict(n=1179, npos=9, align=0.568, tgt=0.933, lt=0.04)),
                   ("medium", dict(n=None, npos=10, align=0.760, tgt=None, lt=0.009)),
                   ("large", dict(n=None, npos=10, align=0.861, tgt=None, lt=0.002))):
        g = [r for r in vrows if r["bucket"] == c]
        if not g:
            continue
        print(f"  {c:>7} V5 n={len(g):<6} n_pos med={med([r['n_pos'] for r in g]):.1f} "
              f"align med={med([r['best_align'] for r in g]):.3f} "
              f"align<.01={100*np.mean([r['best_align']<0.01 for r in g]):.1f}%   "
              f"| §13 报告 n={ref['n']} n_pos={ref['npos']} align={ref['align']} align<.01={100*ref['lt']:.1f}%")

    (OUT / "_sge_sup_tables.json").write_text(json.dumps(dict(by_bucket=t1, by_class=t2),
                                                         indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {OUT / '_sge_sup_tables.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
