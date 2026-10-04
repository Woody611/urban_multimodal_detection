"""_traj_from_snapshots.py — 用**已存在的 `epoch*.pt` 快照**重建 EMA 轨迹（**零训练**）。

背景：D′ 的 `save_period: 10`，`Trainer.save_model()`（trainer.py:544-545）会在
`self.epoch % save_period == 0` 时写出 `weights/epoch{N}.pt`。每个快照内含
`{"model": None, "ema": deepcopy(ema.ema).half(), ...}` —— 即**该 epoch 的 EMA 本体**。

⇒ 只要这些文件还在，就能**纯推理**重建一条 10-epoch 分辨率的轨迹，且对象是**真正的 D′**
（不是复现品）。成本：29 个快照 × 281 图 ≈ 8000 次前向，GPU 上分钟级。

产出与 `_probe300_ema.py` **同格式**的 `epoch_NNN.npz`，因此 `_analyze300.py` 可直接消费。

用法：
  python -X utf8 diagnostic/full300_trajectory_probe/_traj_from_snapshots.py \
      --weights-dir runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights \
      --out diagnostic/full300_trajectory_probe/from_snapshots --device 0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from _probe300_ema import CELLS, IDENTITY_TOL_REL, EmaProbe  # noqa: E402


def _parse_args():
    ap = argparse.ArgumentParser(description="从 epoch*.pt 快照重建 EMA 轨迹（零训练）")
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | 0 | cuda。auto = 有 GPU 就用 GPU。")
    ap.add_argument("--pattern", default="epoch*.pt")
    ap.add_argument("--max-images", type=int, default=0, help=">0 时只扫前 N 张图（自测用）")
    return ap.parse_args()


def main():
    A = _parse_args()
    wdir = (ROOT / A.weights_dir).resolve() if not Path(A.weights_dir).is_absolute() \
        else Path(A.weights_dir)
    out = (ROOT / A.out).resolve() if not Path(A.out).is_absolute() else Path(A.out)
    out.mkdir(parents=True, exist_ok=True)

    snaps = []
    for p in sorted(wdir.glob(A.pattern)):
        m = re.search(r"epoch(\d+)", p.stem)
        if m:
            snaps.append((int(m.group(1)), p))
    snaps.sort()
    if len(snaps) < 2:
        print(f"[snap] {wdir} 下匹配 {A.pattern} 的快照只有 {len(snaps)} 个 —— STOP")
        return 1
    print(f"[snap] 快照 {len(snaps)} 个，0-based epoch 范围 {snaps[0][0]} … {snaps[-1][0]}")
    print(f"[snap] 首个: {snaps[0][1].name}  ({snaps[0][1].stat().st_size/1e6:.1f} MB)")

    # ★ 不再静默降级：显式要 GPU 却拿不到就直接报错退出。
    #   2026-10-03：原三元表达式在 CUDA 不可见时会**默默**退回 CPU，
    #   30 快照 × 281 图从 ~10 分钟变成 ~3 小时，而日志只有一行 device=cpu 提示。
    if A.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif A.device == "cpu":
        dev = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise SystemExit(
                f"--device {A.device} 要求 GPU，但 torch.cuda.is_available()=False。\n"
                f"  拒绝静默回落到 CPU（30 快照全量会从 ~10 分钟变成数小时）。\n"
                f"  排查：python -X utf8 -c \"import torch;print(torch.__version__, torch.cuda.is_available())\"\n"
                f"  若确实要跑 CPU，请显式传 --device cpu。")
        dev = torch.device("cuda")
    if dev.type == "cuda":
        print(f"[snap] GPU = {torch.cuda.get_device_name(0)}")
    else:
        print("[snap] ⚠ 在 CPU 上跑 —— 30 快照全量约需数小时。若机器有 GPU，请传 --device 0")

    probe = EmaProbe(CELLS, out, observe_epochs=[])
    probe.build()
    if A.max_images:
        probe._sel = probe._sel[: A.max_images]
    print(f"[snap] 观测图数 = {len(probe._sel)}  device={dev}")

    ok = True
    for ep, path in snaps:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        src = ck.get("ema") or ck.get("model")
        if src is None:
            print(f"  [skip] ep{ep}: 快照内既无 ema 也无 model"); ok = False; continue
        # ★ 这就是「追踪 EMA」：快照里的 ema 才是 val/best/部署所用的对象
        m = src.float().to(dev).eval()
        probe.attach_model(m)                    # 重置 _nl/_loss/_assigner 并记 W/b
        recs = probe.observe(m, ep)
        probe.dump_epoch(ep, recs)
        il = probe.identity_log[-1]
        flag = "OK " if il["max_rel_resid"] < IDENTITY_TOL_REL else "FAIL"
        print(f"  [{flag}] ep{ep:3d} records={len(recs):4d} "
              f"identity rel={il['max_rel_resid']:.2e} abs={il['max_abs_resid']:.2e}")
        ok &= il["max_rel_resid"] < IDENTITY_TOL_REL
        del m, ck

    probe.finalize()
    for src, dst in (("per_gt_trajectory.csv", "trajectory_per_gt.csv"),
                     ("epoch_summary.csv", "trajectory_summary.csv"),
                     ("tal_metrics.csv", "trajectory_tal.csv")):
        f = out / src
        if f.exists():
            (out / dst).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    # 标注口径（与正式 probe 区分：这是从快照重建，不是训练中观测）
    mp = out / "trajectory_metadata.json"
    md = json.loads(mp.read_text(encoding="utf-8"))
    md.update(source="rebuilt_from_epoch_snapshots", weights_dir=str(wdir),
              snapshot_pattern=A.pattern, n_snapshots=len(snaps),
              snapshot_epochs=[e for e, _ in snaps],
              note="从 save_period 快照重建的 10-epoch 分辨率轨迹；非训练中逐点观测。")
    mp.write_text(json.dumps(md, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[snap] identity 全部通过 = {ok}")
    print(f"[snap] 产物 -> {out}")
    print(f"[snap] 下一步: python -X utf8 {HERE/'_analyze300.py'} --dir {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
