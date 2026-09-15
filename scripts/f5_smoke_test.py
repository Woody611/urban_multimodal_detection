"""scripts/f5_smoke_test.py — F5 (YOLO11l @ imgsz=1280) 冒烟测试。

验证除显存外的全部要素，并输出参数/GFLOPS 供显存估算：
  1. scale 由文件名解析为 'l'（不是 'm'）
  2. RGBD 数据读取 / 4ch / 1280 letterbox
  3. Mid Fusion 结构完整（SilenceChannel×2 + ADD×3）
  4. 顶层模块数 = 45（与 m 一致 → _transfer_rgb_pretrained 硬编码 index pairs 依然有效）
  5. forward / backward / loss 正常（batch=2 @1280，depth 分支 stem 收到梯度）
  6. GFLOPs @640 与 @1280（thop，供显存/时间估算）

用法:
  python scripts/f5_smoke_test.py                # 默认 batch=2
  python scripts/f5_smoke_test.py --batch 1      # 内存更紧张时
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

from ultralytics.cfg import get_cfg                                    # noqa: E402
from ultralytics.data.build import build_yolo_dataset, build_dataloader  # noqa: E402
from ultralytics.nn.tasks import DetectionModel, guess_model_scale     # noqa: E402
from ultralytics.utils.ops import non_max_suppression                  # noqa: E402
from ultralytics.utils.torch_utils import get_flops                    # noqa: E402

MODEL_YAML = "configs/yolo11l_midfusion_rgbd_concat_res.yaml"
DATA_YAML = "data/processed/depth_split_train/dataset.yaml"
TRAIN_YAML = "configs/train_f5_l1280.yaml"

_ap = argparse.ArgumentParser()
_ap.add_argument("--batch", type=int, default=2, help="冒烟测试 batch（CPU 内存不足时调小）")
_ARGS, _ = _ap.parse_known_args()

IMGSZ, BATCH, NC = 1280, _ARGS.batch, 12
HYPS = yaml.safe_load(open(TRAIN_YAML, encoding="utf-8"))

ok, fail = [], []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


print("=" * 74)
print("F5 冒烟测试  YOLO11l  imgsz=1280  RGBD 4ch  mid-fusion")
print("=" * 74)

# ---------- 1. scale 由文件名解析为 l ----------
print("\n[1] scale 解析 + 模型构建")
scale = guess_model_scale(MODEL_YAML)
check("scale 由文件名解析为 l", scale == "l", f"guess_model_scale={scale!r}")
model = DetectionModel(MODEL_YAML, ch=4, nc=NC, verbose=False)
n_p = sum(p.numel() for p in model.parameters())
mods = [type(m).__name__ for m in model.model]
det_head = model.model[-1]
check("类别数 nc=12", int(det_head.nc) == NC, f"Detect.nc={int(det_head.nc)}")
check("参数为 l 量级 (>35M)", n_p > 35e6, f"params={n_p/1e6:.2f}M, 顶层模块数={len(model.model)}")
check("融合结构完整 (SilenceChannel×2 + ADD×3)",
      mods.count("SilenceChannel") == 2 and mods.count("ADD") == 3,
      f"SilenceChannel={mods.count('SilenceChannel')} ADD={mods.count('ADD')}")
# 顶层模块数 = 45（与 m 一致）是 _transfer_rgb_pretrained 硬编码 index pairs 成立的前提
check("顶层模块数=45（与 m 一致，预训练重映射 index 仍有效）", len(model.model) == 45,
      f"len(model.model)={len(model.model)}")
check("stride 正常", int(model.stride.max()) == 32, f"stride={int(model.stride.max())}")

# ---------- 2. GFLOPs @640 / @1280 ----------
print("\n[2] GFLOPs（thop）")
for sz in (640, 1280):
    try:
        g = get_flops(model, imgsz=sz)
        print(f"  GFLOPs @{sz} = {g:.2f}")
    except Exception as e:
        print(f"  GFLOPs @{sz} FAILED: {type(e).__name__}: {e}")

# ---------- 3. 数据读取 + 4ch + 1280 ----------
print("\n[3] RGBD 数据读取 / 4ch / 1280 letterbox")
overrides = {
    "imgsz": IMGSZ, "batch": BATCH, "epochs": 300, "seed": 42, "amp": True,
    "optimizer": "SGD", "lr0": HYPS["learning_rate"], "momentum": HYPS["optimizer"]["momentum"],
    "weight_decay": HYPS["optimizer"]["weight_decay"], "cos_lr": True,
    "warmup_epochs": HYPS["warmup_epochs"], "channels": HYPS["channels"],
    "use_simotm": HYPS["use_simotm"], "pairs_rgb_ir": HYPS["pairs_rgb_ir"],
    "pairs_rgb_depth": HYPS["pairs_rgb_depth"], "device": "cpu", "workers": 0,
    "data": DATA_YAML, "task": "detect", "patience": HYPS["patience"],
}
args = get_cfg(overrides=overrides)
data = yaml.safe_load(open(DATA_YAML, encoding="utf-8"))
train_path = str(ROOT / data["path"] / data["train"])
ds = build_yolo_dataset(args, train_path, BATCH, data, mode="train", stride=32,
                        use_simotm=args.use_simotm, pairs_rgb_ir=args.pairs_rgb_ir,
                        pairs_rgb_depth=args.pairs_rgb_depth)
dl = build_dataloader(ds, BATCH, 0, shuffle=False, rank=-1)
batch = next(iter(dl))
im = batch["img"]
check("RGBD 数据读取正常", im.shape[0] == BATCH, f"batch size={im.shape[0]}")
check("4-channel 输入正常", im.shape[1] == 4, f"img.shape={tuple(im.shape)}")
check("1280 letterbox 正常", im.shape[2] == IMGSZ and im.shape[3] == IMGSZ,
      f"HxW={im.shape[2]}x{im.shape[3]}")
d = im[:, 3]
check("depth 通道有实际信号", d.float().std() > 1e-3,
      f"mean={d.float().mean():.1f} std={d.float().std():.1f}")

# ---------- 4. forward / backward / loss ----------
print("\n[4] forward / backward / loss @1280")
model.train()
xb = im.float() / 255.0
tr_batch = {k: (v.float() / 255.0 if k == "img" else v) for k, v in batch.items()}
try:
    preds = model(xb)
    check("forward 正常（P3/P4/P5）", isinstance(preds, (list, tuple)) and len(preds) == 3,
          f"输出 {[tuple(p.shape) for p in preds]}")
    check("特征图 160/80/40 @1280",
          [tuple(p.shape[-2:]) for p in preds] == [(160, 160), (80, 80), (40, 40)],
          f"{[tuple(p.shape[-2:]) for p in preds]}")
    model.args = args
    loss_sum, loss_items = model.loss(tr_batch, preds)
    lv = float(loss_sum)
    check("loss 正常（有限且为正）", torch.isfinite(loss_sum) and lv > 0,
          f"total={lv:.3f}  [box,cls,dfl]={[round(float(x),3) for x in loss_items]}")
    loss_sum.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    finite = all(bool(torch.isfinite(g).all()) for g in grads)
    check("backward 正常（梯度有限）", len(grads) > 0 and finite, f"{len(grads)} 张量有梯度")
    stem_d = model.model[8].conv.weight.grad
    check("depth 分支 stem 收到梯度", stem_d is not None and bool(torch.isfinite(stem_d).all())
          and bool((stem_d != 0).any()),
          f"|grad|={float(stem_d.abs().mean()):.2e}" if stem_d is not None else "无梯度")
except Exception as e:
    check("forward/backward/loss", False, f"{type(e).__name__}: {e}")

print("\n" + "=" * 74)
print(f"结果: {len(ok)} PASS / {len(fail)} FAIL")
if fail:
    print("失败项: " + ", ".join(fail))
print("=" * 74)
sys.exit(1 if fail else 0)
