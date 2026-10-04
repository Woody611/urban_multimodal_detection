"""_e2e_smoke_ema.py — EMA probe 的**端到端冒烟**：在本机 CPU 上跑一个**真实 ultralytics Trainer**
（极小规模：2 epoch / imgsz 320 / 小 fraction），走完整生命周期。

为什么需要它：此前三次云端崩溃**全部**发生在 probe 与框架训练循环的交互处，而不是 probe 内部：

  · `save_model()` 里的 `torch.save(deepcopy(ema.ema))`  ← hook 生命周期 bug 在这暴露
  · `validate()` 内 `@smart_inference_mode()` + half/float 往返 ← inference tensor bug 在这暴露
  · `final_eval()` 在 trainer.py:691 二次触发 `on_fit_epoch_end` ← §6 门假阳性在这暴露

单次 `observe()` 的单元测试**结构上够不到**这三处。本脚本用真实 Trainer 覆盖：

    on_pretrain_routine_end → attach + P4 gate
    → 训练步（ema.update）
    → validate()            （inference_mode + 潜在 dtype 往返）
    → save_model()          （deepcopy + torch.save → pickle 门）
    → on_fit_epoch_end      （观测）
    → final_eval()          （strip_optimizer + 二次 on_fit_epoch_end → 幂等门）
    → on_train_end

产物写入 `_e2e_smoke/` 与 `runs/E2E_SMOKE_ema_probe/`，**不碰**真实 run 目录与正式产物。
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from ultralytics import YOLO  # noqa: E402
from _probe300_ema import CELLS, EmaProbe  # noqa: E402

RUN_NAME = "E2E_SMOKE_ema_probe"
SMOKE_OUT = HERE / "_e2e_smoke"
N_OBS_IMG = 8


class SmokeProbe(EmaProbe):
    """与正式 probe 逐字相同，只把观测图数截到 N_OBS_IMG 以控制 CPU 时间。"""

    def build(self):
        super().build()
        self._sel = self._sel[:N_OBS_IMG]


def _parse_args():
    ap = argparse.ArgumentParser(description="EMA probe 端到端冒烟（真实 Trainer，极小规模）")
    ap.add_argument("--device", default="cpu",
                    help="cpu | 0 | cuda。**云端请用 0** —— 只有 GPU + amp 才会让 validate() 走 "
                         "model.half()/float() 往返，从而真实复现 inference tensor 路径。")
    ap.add_argument("--amp", type=int, default=-1, help="-1 = 按 device 自动（非 cpu 则开）")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--fraction", type=float, default=0.02)
    return ap.parse_args()


def main():
    A = _parse_args()
    amp = A.amp if A.amp >= 0 else (0 if A.device == "cpu" else 1)
    if SMOKE_OUT.exists():
        shutil.rmtree(SMOKE_OUT)
    SMOKE_OUT.mkdir(parents=True, exist_ok=True)
    tgt = ROOT / "runs" / RUN_NAME
    if tgt.exists():
        shutil.rmtree(tgt)

    spec = importlib.util.spec_from_file_location("tr", ROOT / "scripts/train.py")
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)

    train_cfg = tr._load_yaml(ROOT / "configs/train_rgbid_sepstem_clahe.yaml")
    dataset_cfg = tr._load_yaml(ROOT / "configs/dataset.yaml")
    data_path = tr._split_train_val_rgbid(dataset_cfg, float(train_cfg.get("val_ratio", 0.2)),
                                          int(train_cfg.get("seed", 42)))
    kwargs = tr._build_train_kwargs(train_cfg, data_path)

    # ---- 缩到能跑完的规模（只改运行规模，不改 probe 相关路径）----
    kwargs.update(epochs=A.epochs, batch=2, imgsz=A.imgsz, workers=0, device=A.device,
                  amp=bool(amp), val=True, save=True, plots=False, close_mosaic=0, patience=0,
                  fraction=A.fraction, resume=False, exist_ok=False, name=RUN_NAME,
                  project="runs", cache=False, deterministic=True)
    kwargs.pop("pretrained", None)          # 不下载预训练，避免联网
    kwargs["pretrained"] = None

    probe = SmokeProbe(CELLS, SMOKE_OUT, observe_epochs=(0, 1, 2))
    model = YOLO(str(ROOT / "configs/yolo11m_sepstem.yaml"))
    state = {"gate_done": False, "final_eval_seen": 0}

    def _attach(trainer):
        from ultralytics.utils.torch_utils import de_parallel
        assert getattr(trainer, "ema", None) is not None, "ema 未创建"
        probe.attach_model(de_parallel(trainer.ema.ema))
        res = probe.self_check(de_parallel(trainer.ema.ema), n_img=1)
        print(f"[e2e] P4 self_check PASS={res['PASS']} hooks_left={res['hooks_left_on_model']} "
              f"picklable={res['picklable_like_save_model']} rel={res['identity_max_rel_resid']:.2e}",
              flush=True)
        if not res["PASS"]:
            raise RuntimeError(f"P4 gate FAILED: {res}")
        state["gate_done"] = True

    _orig = probe.on_fit_epoch_end

    def _on_fit_epoch_end(trainer):
        state["final_eval_seen"] += 1
        if state["final_eval_seen"] == 1:
            # ★ 直接判定「inference tensor 路径是否真的被走到」：
            #   validate() 在 GPU+amp 下做 half()/float() 往返，会把 EMA 的 param/buffer
            #   变成 inference tensor；CPU 下 args.half=False ⇒ 不会有。
            from ultralytics.utils.torch_utils import de_parallel
            m = de_parallel(trainer.ema.ema)
            nb = [v for _, v in m.named_buffers() if v.dtype.is_floating_point]
            npar = list(m.parameters())
            state["infer_buf"] = sum(int(v.is_inference()) for v in nb)
            state["infer_par"] = sum(int(v.is_inference()) for v in npar)
            print(f"[e2e] 首个 on_fit_epoch_end: float buffer 中 inference tensor "
                  f"{state['infer_buf']}/{len(nb)}，param 中 {state['infer_par']}/{len(npar)} "
                  f"（>0 表示 GPU/half 路径已真实走到）", flush=True)
        _orig(trainer)

    model.add_callback("on_pretrain_routine_end", _attach)
    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    model.add_callback("on_train_end", probe.on_train_end)

    print(f"[e2e] 开始真实训练（{A.epochs} epoch / imgsz{A.imgsz} / device={A.device} / amp={bool(amp)}）…", flush=True)
    model.train(**kwargs)
    print("[e2e] model.train() 正常返回（未抛异常）", flush=True)

    # ---- 结果判定 ----
    ok = True
    print(f"\n[e2e] P4 gate 执行: {state['gate_done']}")
    print(f"[e2e] on_fit_epoch_end 被调用次数: {state['final_eval_seen']}（含 final_eval 的二次触发）")

    w = ROOT / "runs" / RUN_NAME / "weights"
    for n in ("best.pt", "last.pt"):
        f = w / n
        if not f.exists():
            print(f"  [FAIL] {n} 不存在"); ok = False; continue
        import torch
        ck = torch.load(f, map_location="cpu", weights_only=False)
        stripped = ck.get("ema") is None and ck.get("epoch") == -1
        print(f"  [{'OK' if stripped else 'FAIL'}] {n} stripped={stripped} "
              f"{f.stat().st_size/1e6:.1f}MB")
        ok &= stripped

    npz = sorted(SMOKE_OUT.glob("epoch_*.npz"))
    got = {int(p.stem.split("_")[1]) for p in npz}
    want = set(range(min(3, kwargs["epochs"])))       # 只跑到实际 epoch 数
    print(f"[e2e] 观测 npz = {sorted(got)}  期望 ⊇ {sorted(want)}")
    if not want <= got:
        print(f"  [FAIL] 缺观测点 {sorted(want - got)}"); ok = False

    # logit 恒等式（修好后应全程 ~1e-6 量级）
    import json, io
    md = json.load(io.open(SMOKE_OUT / "trajectory_metadata.json", encoding="utf-8"))
    rel = [x["max_rel_resid"] for x in md["logit_identity"]]
    print(f"[e2e] logit identity rel = {[f'{r:.2e}' for r in rel]}")
    ok &= all(r < 1e-3 for r in rel)

    sr = md["state_restoration"]
    print(f"[e2e] params_untouched = {[x['params_untouched'] for x in sr]}")
    ok &= all(x["params_untouched"] for x in sr)

    print(f"\n[e2e] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
