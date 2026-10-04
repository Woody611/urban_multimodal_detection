"""_dryrun_ema.py — EMA probe 仪器的本机 dry-run（CPU，无训练）。

把 D′ 部署 EMA（= `ck["model"]`, strip_optimizer 路径）当作 `trainer.ema.ema` 的替身，
走一遍 EmaProbe 的完整链路：attach_model → self_check(P4) → observe → dump_epoch → finalize，
验证仪器在**部署口径**下可产出 §17 要求的全部文件，且状态可还原。

注意：这**不是**训练轨迹，只是 instrument smoke test。产物写到 _dryrun_ema/。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from _probe300_ema import CELLS, EmaProbe  # noqa: E402

OUT = HERE / "_dryrun_ema"
N_IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ck = torch.load(ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt",
                    map_location="cpu", weights_only=False)
    ema = ck["model"].float()          # 扮演 trainer.ema.ema
    # ★ 复现 validate() 之后的真实状态：BaseValidator.__call__ 被 @smart_inference_mode() 包裹
    #   （validator.py:108），且内部做 half()/float() 往返（119/208）⇒ EMA 的 param/buffer
    #   全变成 inference tensor。不模拟这一步，dry-run 就覆盖不到云端那次崩溃。
    with torch.inference_mode():
        ema.half()
        ema.float()
    print("[dryrun] 已模拟 validator 的 inference_mode 内 dtype 往返（EMA 现为 inference tensor）")

    probe = EmaProbe(CELLS, OUT, observe_epochs=(0, 1))
    probe.attach_model(ema)
    print("[dryrun] attached (P3 / H / RAW / CLS_OUT hooks)")

    # 只扫 N 张图，但要**优先包含 G86**，否则分析器缺 G86 会在渲染阶段报错
    probe.build()
    g86_stems = {r["image_stem"] for r in probe.cells if r["role"] == "G86"}
    sel = [i for i in probe._sel if Path(probe._ds.im_files[i]).stem in g86_stems]
    others = [i for i in probe._sel if i not in set(sel)]
    probe._sel = (sel + others)[:N_IMG]
    print(f"[dryrun] images = {len(probe._sel)}")

    sc = probe.self_check(ema, n_img=2)
    print("[dryrun] P4 self_check:")
    for k, v in sc.items():
        print(f"    {k:<24} = {v}")

    for ep in (0, 1):
        recs = probe.observe(ema, ep)
        probe.dump_epoch(ep, recs)
        print(f"[dryrun] observe(ep{ep}): records={len(recs)}  "
              f"identity={probe.identity_log[-1]}  restore={probe.restore_log[-1]}")
    roles = {}
    for r in recs:
        roles[r["role"]] = roles.get(r["role"], 0) + 1
    print(f"[dryrun] roles = {roles}")

    probe.finalize()
    # §17 的三个 drift 文件由后处理分析器产出（读 epoch_*.npz）
    import subprocess
    r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "_analyze300.py"), "--dir", str(OUT)],
                       capture_output=True, text=True)
    print("[dryrun] _analyze300.py rc=", r.returncode)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
    produced = sorted(p.name for p in OUT.iterdir() if p.is_file())
    print(f"[dryrun] produced: {produced}")

    need = ["trajectory_summary.csv", "trajectory_per_gt.csv", "trajectory_classifier_drift.csv",
            "trajectory_representation_drift.csv", "trajectory_decomposition.csv",
            "trajectory_tal.csv", "trajectory_metadata.json"]
    missing = [n for n in need if n not in produced]
    md = json.loads((OUT / "trajectory_metadata.json").read_text(encoding="utf-8"))
    print(f"[dryrun] metadata keys: {sorted(md)}")
    print(f"[dryrun] probe_target={md['probe_target']} probe_mode={md['probe_mode']} "
          f"coord={md['coordinate_source']}")
    ok = (not missing and sc["PASS"] and md["probe_target"] == "trainer.ema.ema"
          and md["probe_mode"] == "eval"
          and md["coordinate_source"] == "dataset_lab_bboxes_canvas_normalized")
    print(f"\n[dryrun] {'PASS' if ok else 'FAIL'}   missing={missing}  self_check_PASS={sc['PASS']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
