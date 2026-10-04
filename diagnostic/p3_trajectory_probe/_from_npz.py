"""_from_npz.py — 从 `epoch_NNN.npz` 重建轨迹并出判决（纯 CPU，无 forward，无 GPU）。

为什么需要它：
  云端 run 未在 ep30 停机（根因待 `_run.log` 确认），因此**可能被手动中止**。
  手动中止会让 `on_train_end` 不触发 ⇒ `finalize()` 不跑 ⇒ `per_gt_trajectory.csv` /
  `epoch_summary.csv` / `p3_metrics.csv` / `h_metrics.csv` / `decision.*` 全部缺失。
  但每个观测点的 `epoch_NNN.npz` **已经落盘**，且其中包含：
    vec_keys/vec_cls/vec_role/vec_p3/vec_h   （1320 个 medium cell 的向量与角色）
    rec_num_keys/rec_str_keys + 各字段数组 （86 条逐 GT 记录，数值与字符串都在）
  ⇒ 用它可以**逐位重建** `self.traj` 与 `self.vecs`，再调用**同一个** `finalize()`，
    产出与正常跑完完全相同的产物。

用法：
  python -X utf8 diagnostic/p3_trajectory_probe/_from_npz.py [--dir <轨迹所在目录>]
默认 --dir 为本脚本目录。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diagnostic/p3_trajectory_probe"))
from _probe import P3TrajProbe  # noqa: E402

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=None)
A = ap.parse_args()
OUT = Path(A.dir).resolve() if A.dir else HERE


def main():
    npzs = sorted(OUT.glob("epoch_*.npz"))
    if not npzs:
        print(f"[from_npz] 在 {OUT} 找不到 epoch_*.npz —— STOP")
        return 1
    mc = OUT / "medium_cells.json"
    if not mc.exists():
        mc = HERE / "medium_cells.json"
    p = P3TrajProbe(mc, OUT)
    print("=" * 92)
    print("FROM_NPZ —— 从 epoch_*.npz 重建轨迹并出判决")
    print("=" * 92)
    print(f"  目录: {OUT}")
    print(f"  找到 {len(npzs)} 个观测点: {[x.stem for x in npzs]}")
    for z in npzs:
        d = np.load(z, allow_pickle=False)
        nk = [str(x) for x in d["rec_num_keys"]]
        sk = [str(x) for x in d["rec_str_keys"]]
        n = len(d[nk[0]])
        recs = []
        for i in range(n):
            r = {k: float(d[k][i]) for k in nk}
            r.update({k: str(d[k][i]) for k in sk})
            recs.append(r)
        p.traj.extend(recs)
        p.vecs[int(recs[0]["epoch"])] = dict(
            vec_keys=[str(x) for x in d["vec_keys"]], vec_cls=d["vec_cls"].tolist(),
            vec_role=[str(x) for x in d["vec_role"]], vec_p3=d["vec_p3"], vec_h=d["vec_h"])
        print(f"    {z.stem}: {n} 条记录（G-86={sum(1 for r in recs if r['role'] == 'G86')}），"
              f"{len(p.vecs[int(recs[0]['epoch'])]['vec_keys'])} 个 cell 向量")
    dec = p.finalize()
    print("\n  --- DECISION（由 _probe.finalize 依预注册阈值给出）---")
    for k in ("CASE", "REPRESENTATION_INITIAL_WEAKNESS", "TRAINING_DEGRADATION",
              "TAL_SELECTION_FEEDBACK", "P3_WEAK_H_RECOVERY", "SYNCHRONIZED_MECHANISM", "CAUSALITY_LEVEL"):
        print(f"    {k:<28}= {dec[k]}")
    print(f"    first P3 / cls / align / mask_pos = ({dec['first_P3_degradation']}, "
          f"{dec['first_cls_degradation']}, {dec['first_alignment_degradation']}, {dec['first_mask_pos_loss']})")
    print(f"    P3 acc(G86) 逐观测点 = {dec['p3_accuracy_g86']}")
    print(f"    P3 acc(CTRL) 逐观测点 = {dec['p3_accuracy_control']}")
    print(f"    h  acc(G86) 逐观测点 = {dec['h_accuracy_g86']}")
    print(f"    random baseline      = {dec['random_baseline_majority_prior']}")
    print(f"\n[from_npz] 写出 per_gt_trajectory.csv / epoch_summary.csv / p3_metrics.csv / "
          f"h_metrics.csv / tal_metrics.csv / assignment_metrics.csv / decision.json / decision.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
