"""_redecide.py — 用 `role` 过滤后**重算** G-86 的 aggregate 与 A/B/C/D 判决（纯 CPU，无 forward）。

为什么需要它：
  `_probe.finalize()` 的 `epoch_summary.csv` / `decision.*` 把中位数算在**全部 1320 个 medium cell** 上，
  而不是规格 §9 要求的 G-86 子集 ⇒ 那两个产物的数值**答非所问**。
  但 `per_gt_trajectory.csv` 每行都带 `role`，且 `p3_metrics.csv` / `h_metrics.csv` 本来就按 role 分组，
  因此可以**离线精确重算**，不需要重训。

不修改任何已有文件：输出到 `recomputed_*`。

用法（本机或云端均可，只需 CSV）：
  python -X utf8 diagnostic/p3_trajectory_probe/_redecide.py
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diagnostic/p3_trajectory_probe"))
from _probe import P3TrajProbe  # noqa: E402

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--dir", default=None, help="轨迹所在目录（默认本脚本目录；便于用合成 CSV 自测）")
_ARGS = _ap.parse_known_args()[0]
OUT = Path(_ARGS.dir).resolve() if _ARGS.dir else Path(__file__).resolve().parent
TRAJ = OUT / "per_gt_trajectory.csv"
FIELDS = ("logit", "cls_pos_max", "cls_center_max", "ciou_pos_max",
          "raw_align_pos_max", "p3_norm", "h_norm")


def f(x):
    return float(x)


def main():
    if not TRAJ.exists():
        print(f"[redecide] 缺 {TRAJ} —— 训练尚未产出轨迹，STOP")
        return 1
    rows = list(csv.DictReader(open(TRAJ, encoding="utf-8")))
    roles = sorted({r["role"] for r in rows})
    eps = sorted({int(f(r["epoch"])) for r in rows})
    print("=" * 92)
    print("REDECIDE —— 按 role 过滤后重算 G-86 aggregate 与判决")
    print("=" * 92)
    print(f"  per_gt_trajectory.csv: {len(rows)} 行 / {len(eps)} 个 epoch / roles={roles}")
    g86 = [r for r in rows if r["role"] == "G86"]
    print(f"  G-86 行数 = {len(g86)}（期望 {86*len(eps)}）  每 epoch G-86 = "
          f"{sorted({int(f(r['epoch'])):0 for r in g86}) and len(g86)//max(len(eps),1)}")

    # ---- §9 G-86 aggregate（median / mean / p25 / p75），并同时给 CONTROL 供对照 ----
    ser = []
    for e in eps:
        for role in roles:
            s = [r for r in rows if int(f(r["epoch"])) == e and r["role"] == role]
            if not s:
                continue
            for k in FIELDS:
                v = np.array([f(x[k]) for x in s if x[k] not in ("", "nan")])
                if not len(v):
                    continue
                ser.append(dict(epoch=e, role=role, field=k, n=len(v),
                                median=float(np.median(v)), mean=float(v.mean()),
                                p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75))))
            for k in ("mask_pos_membership", "topk_membership"):
                ser.append(dict(epoch=e, role=role, field=k + "_rate", n=len(s),
                                median=float(np.mean([int(f(x[k])) for x in s])),
                                mean=float(np.mean([int(f(x[k])) for x in s])),
                                p25=float("nan"), p75=float("nan")))
    with open(OUT / "recomputed_epoch_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["epoch", "role", "field", "n", "median", "mean", "p25", "p75"])
        w.writeheader(); w.writerows(ser)

    print("\n  --- G-86（role=='G86'）逐 epoch ---")
    print(f"  {'epoch':>6}{'logit':>10}{'cls_pos':>10}{'ciou_pos':>10}{'raw_align':>12}{'mask_pos率':>12}{'|p3|':>8}{'|h|':>8}")
    for e in eps:
        g = lambda k: next((x["median"] for x in ser if x["epoch"] == e and x["role"] == "G86" and x["field"] == k), float("nan"))
        print(f"  {e:>6}{g('logit'):>10.3f}{g('cls_pos_max'):>10.4f}{g('ciou_pos_max'):>10.4f}"
              f"{g('raw_align_pos_max'):>12.5f}{g('mask_pos_membership_rate'):>12.3f}"
              f"{g('p3_norm'):>8.2f}{g('h_norm'):>8.2f}")
    print("\n  --- CONTROL（对照，同一时刻）---")
    print(f"  {'epoch':>6}{'logit':>10}{'mask_pos率':>12}")
    for e in eps:
        g = lambda k: next((x["median"] for x in ser if x["epoch"] == e and x["role"] == "CONTROL" and x["field"] == k), float("nan"))
        print(f"  {e:>6}{g('logit'):>10.3f}{g('mask_pos_membership_rate'):>12.3f}")

    # ---- 判决：用 G-86 的 series 重跑 _decide ----
    p3f, hf = OUT / "p3_metrics.csv", OUT / "h_metrics.csv"
    if not (p3f.exists() and hf.exists()):
        print(f"\n[redecide] 缺 {p3f.name} 或 {hf.name} —— 无法判决，STOP")
        return 1
    p3 = list(csv.DictReader(open(p3f, encoding="utf-8")))
    hh = list(csv.DictReader(open(hf, encoding="utf-8")))
    f2i = lambda R: [{k: (float(v) if k not in ("feature",) else v) for k, v in r.items()} for r in R]
    _mc = OUT / "medium_cells.json"
    if not _mc.exists():
        _mc = Path(__file__).resolve().parent / "medium_cells.json"
    dummy = P3TrajProbe(_mc, OUT)
    dec = dummy._decide(eps, f2i(p3), f2i(hh), [x for x in ser if x["role"] == "G86"])
    dec["NOTE"] = ("由 _redecide.py 用 role=='G86' 过滤后重算；_probe.finalize() 原产物因把中位数算在"
                   "全部 1320 个 cell 上而答非所问，故以本文件为准。")
    dec["source"] = "recomputed from per_gt_trajectory.csv (role=='G86') + p3_metrics.csv + h_metrics.csv"
    (OUT / "recomputed_decision.json").write_text(json.dumps(dec, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "recomputed_decision.md").write_text(dummy._render(dec), encoding="utf-8")

    print("\n  --- RECOMPUTED DECISION ---")
    for k in ("CASE", "REPRESENTATION_INITIAL_WEAKNESS", "TRAINING_DEGRADATION",
              "TAL_SELECTION_FEEDBACK", "P3_WEAK_H_RECOVERY", "SYNCHRONIZED_MECHANISM", "CAUSALITY_LEVEL"):
        print(f"    {k:<28}= {dec[k]}")
    print(f"    first P3 / cls / align / mask_pos = ({dec['first_P3_degradation']}, "
          f"{dec['first_cls_degradation']}, {dec['first_alignment_degradation']}, {dec['first_mask_pos_loss']})")
    print("\n[redecide] 写出 recomputed_epoch_summary.csv / recomputed_decision.json / recomputed_decision.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
