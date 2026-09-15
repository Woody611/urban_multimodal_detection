"""scripts/f4_smoke_test.py — F4 (imgsz=1280) 冒烟测试：验证除显存外的全部要素。

在 CPU 上跑（本机无 CUDA），逐条对应正式训练前的 10 项检查：
  1. RGBD 数据读取正常        2. 4-channel 输入正常      3. 1280 resize/letterbox 正常
  4. Mid Fusion 正常          5. forward 正常            6. backward 正常
  7. loss 正常                8. validation 正常         9. 显存不 OOM ← 无法本地验证，见报告
  10. 无 tensor shape mismatch

用法:
  python scripts/f4_smoke_test.py                # A 段：数据/模型/前后向/NMS，batch=8（正式值）
  python scripts/f4_smoke_test.py --batch 2      # 小 batch：CPU 内存放不下 batch=8@1280 时用
  python scripts/f4_smoke_test.py --batch 2 --val  # 追加 B 段：真实 validator val@1280

注意：本机为 CPU（无 CUDA）。1280×batch8 的 CPU 内存需求超 20GB，会 segfault；
用 --batch 2 完成前后向验证。**GPU 显存必须在云端单独验证**，本地结果不能外推。
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
from ultralytics.nn.tasks import DetectionModel                        # noqa: E402
from ultralytics.utils.ops import non_max_suppression                  # noqa: E402
from ultralytics.nn.tasks import guess_model_scale                    # noqa: E402

MODEL_YAML = "configs/yolo11m_midfusion_rgbd_concat_res.yaml"
DATA_YAML = "data/processed/depth_split_train/dataset.yaml"

_ap = argparse.ArgumentParser()
_ap.add_argument("--batch", type=int, default=8, help="冒烟测试 batch（正式训练为 8；CPU 内存不足时调小）")
_ap.add_argument("--val", action="store_true", help="追加真实 validator 全量 val 检查")
_ARGS, _ = _ap.parse_known_args()

IMGSZ, BATCH, NC = 1280, _ARGS.batch, 12
HYPS = yaml.safe_load(open("configs/train_f4_1280.yaml", encoding="utf-8"))

ok, fail = [], []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


print("=" * 74)
print("F4 冒烟测试  imgsz=1280  batch=8  RGBD 4ch  YOLO11m mid-fusion")
print("=" * 74)

# ---------- 0. 构建与训练完全一致的 args（含全部默认值） ----------
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
print(f"\nargs: imgsz={args.imgsz} batch={args.batch} nbs={args.nbs} "
      f"lr0={args.lr0} lrf={args.lrf} close_mosaic={args.close_mosaic} "
      f"deterministic={args.deterministic}")
print(f"      mosaic={args.mosaic} hsv_h={args.hsv_h} fliplr={args.fliplr} "
      f"erasing={args.erasing} box={args.box} cls={args.cls} dfl={args.dfl}")

# ---------- 1./2./3. 数据读取 + 4ch + 1280 letterbox ----------
print("\n[1-3] RGBD 数据读取 / 4-channel / 1280 letterbox")
data = yaml.safe_load(open(DATA_YAML, encoding="utf-8"))
train_path = str(ROOT / data["path"] / data["train"])

t0 = time.time()
ds = build_yolo_dataset(args, train_path, BATCH, data, mode="train", stride=32,
                        use_simotm=args.use_simotm, pairs_rgb_ir=args.pairs_rgb_ir,
                        pairs_rgb_depth=args.pairs_rgb_depth)
dl = build_dataloader(ds, BATCH, 0, shuffle=False, rank=-1)
print(f"  dataset={len(ds)} 图, 构建耗时 {time.time()-t0:.1f}s")
batch = next(iter(dl))
im = batch["img"]
check("RGBD 数据读取正常", im.shape[0] == BATCH, f"batch size={im.shape[0]}")
check("4-channel 输入正常", im.shape[1] == 4, f"img.shape={tuple(im.shape)}")
check("1280 letterbox 正常", im.shape[2] == IMGSZ and im.shape[3] == IMGSZ,
      f"HxW={im.shape[2]}x{im.shape[3]}")
check("dtype=uint8（0~255 未归一）", im.dtype == torch.uint8, f"dtype={im.dtype}")
# depth 通道（index 3）实际有值：非全 114(letterbox pad) 亦非全 0
d = im[:, 3]
check("depth 通道有实际信号", d.float().std() > 1e-3,
      f"mean={d.float().mean():.1f} std={d.float().std():.1f} min={d.min()} max={d.max()}")
# 标签
lb = batch["cls"]
check("标签类别范围合法", bool((lb >= 0).all() and (lb < NC).all()), f"n_boxes={len(lb)}")
check("标签 bbox 在归一化范围内", bool((batch["bboxes"] >= 0).all() and (batch["bboxes"] <= 1).all()))

# ---------- 4. Mid Fusion / 模型构建 ----------
print("\n[4] Mid Fusion 模型构建")
scale = guess_model_scale(MODEL_YAML)
check("scale 由文件名解析为 m", scale == "m", f"guess_model_scale={scale!r}")
model = DetectionModel(MODEL_YAML, ch=4, nc=NC, verbose=False)
n_p = sum(p.numel() for p in model.parameters())
n_l = len(model.model)
check("模型通道 ch=4", model.yaml.get("ch") == 4, f"ch={model.yaml.get('ch')}")
det_head = model.model[-1]  # Detect 头持有 nc（DetectionModel 本身无 .nc）
check("类别数 nc=12", int(det_head.nc) == NC, f"Detect.nc={int(det_head.nc)}")
check("参数量为 m 量级 (20~40M)", 2e7 < n_p < 4e7, f"params={n_p/1e6:.2f}M, layers={n_l}")
# 融合层存在性：SilenceChannel x2 + ADD x3
mods = [type(m).__name__ for m in model.model]
check("融合结构完整 (SilenceChannel×2 + ADD×3)",
      mods.count("SilenceChannel") == 2 and mods.count("ADD") == 3,
      f"SilenceChannel={mods.count('SilenceChannel')} ADD={mods.count('ADD')}")
stride = int(model.stride.max())
check("stride 正常", stride == 32, f"stride={stride}")

# ---------- 5./6./7. forward / backward / loss ----------
print("\n[5-7] forward / backward / loss")
model.train()
xb = (im.float() / 255.0)
# 训练态 forward 返回 3 层原始特征图（本 fork 不在 forward 内算 loss），
# 真实 loss 走 BaseModel.loss(batch, preds) —— 这正是 trainer 的调用路径。
tr_batch = {k: (v.float() / 255.0 if k == "img" else v) for k, v in batch.items()}
t0 = time.time()
try:
    preds = model(xb)
    fwd_s = time.time() - t0
    check("forward 正常（无 shape mismatch）", isinstance(preds, (list, tuple)) and len(preds) == 3,
          f"P3/P4/P5 输出 {[tuple(p.shape) for p in preds]}  用时 {fwd_s:.1f}s")
    check("特征图层级对应 stride 8/16/32 @1280",
          [tuple(p.shape[-2:]) for p in preds] == [(160, 160), (80, 80), (40, 40)],
          f"{[tuple(p.shape[-2:]) for p in preds]}")
    # v8DetectionLoss 从 model.args 读 box/cls/dfl 等超参（loss.py:412），
    # trainer 在 _setup_train 里也会先赋值 —— 此处对齐，保证消融值就是训练用的值。
    model.args = args
    t0 = time.time()
    loss_sum, loss_items = model.loss(tr_batch, preds)
    loss_s = time.time() - t0
    lv = float(loss_sum)
    check("loss 正常（有限且为正）", torch.isfinite(loss_sum) and lv > 0,
          f"total={lv:.3f}  [box,cls,dfl]={[round(float(x),3) for x in loss_items]}  用时 {loss_s:.1f}s")
    t0 = time.time()
    loss_sum.backward()
    bwd_s = time.time() - t0
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    finite = all(bool(torch.isfinite(g).all()) for g in grads)
    nonzero = sum(1 for g in grads if bool((g != 0).any()))
    check("backward 正常（梯度有限）", len(grads) > 0 and finite,
          f"{len(grads)} 个参数张量有梯度（{nonzero} 个非零）, 用时 {bwd_s:.1f}s")
    # 融合分支梯度是否真的回传到 depth 分支 stem（index 8 = depth 分支第一个 Conv）
    stem_d = model.model[8].conv.weight.grad
    check("depth 分支 stem 收到梯度（融合真的在训）",
          stem_d is not None and bool(torch.isfinite(stem_d).all()) and bool((stem_d != 0).any()),
          f"|grad|={float(stem_d.abs().mean()):.2e}" if stem_d is not None else "无梯度")
except Exception as e:
    check("forward/backward/loss", False, f"{type(e).__name__}: {e}")
model.zero_grad(set_to_none=True)

# ---------- 10. 推理态 + NMS postprocess @1280 ----------
print("\n[10] 推理态 + NMS postprocess @1280")
model.eval()
with torch.no_grad():
    preds_i = model(xb)
    if isinstance(preds_i, (list, tuple)):  # 推理态返回 (concat_pred, extras)
        preds_i = preds_i[0]
    det = non_max_suppression(preds_i, conf_thres=0.001, iou_thres=0.7, nc=NC,
                              max_det=300, multi_label=True)
check("推理输出可被 NMS 正常解析", len(det) == BATCH and all(d_.shape[1] == 6 for d_ in det),
      f"每图框数={[int(d_.shape[0]) for d_ in det]}（随机权重，数值无意义）")

# ---------- 9. 显存（本地无法测） ----------
print("\n[9] 显存")
print("  [SKIP] 本机 torch=2.4.1+cpu，无 CUDA，无法测量 VRAM — 必须在云端 GPU 上验证")
print(f"         参考：参数量 {n_p/1e6:.2f}M；F1(imgsz=1024,batch=8) 云端可跑")

print("\n" + "=" * 74)
print(f"结果: {len(ok)} PASS / {len(fail)} FAIL")
if fail:
    print("失败项: " + ", ".join(fail))
print("=" * 74)

# ---------- B 段：真实 validator ----------
if _ARGS.val:
    print("\n[B] 真实 validator 全量 val @1280（未训练权重，mAP 数值无意义，只验证流程可跑通）")
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics import YOLO
    m = YOLO(MODEL_YAML)
    t0 = time.time()
    r = m.val(data=DATA_YAML, imgsz=IMGSZ, conf=0.001, iou=0.7, max_det=300, batch=BATCH,
              device="cpu", plots=False, save_json=False, use_simotm="RGBD", channels=4,
              pairs_rgb_ir=["visible", "depth"], pairs_rgb_depth=["visible", "depth"],
              project="runs/audit", name="f4_smoke_val", exist_ok=True)
    print(f"  validator 跑通，用时 {time.time()-t0:.1f}s  mAP50-95={float(r.box.map):.5f}（未训练，仅证明流程）")
    check("validation 正常（validator 全流程跑通）", True)
    print(f"\n最终: {len(ok)} PASS / {len(fail)} FAIL" + (f" 失败: {fail}" if fail else ""))
sys.exit(1 if fail else 0)
