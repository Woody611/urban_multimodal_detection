"""scripts/modality_dropout_final_check.py — 正式训练前最终验证（A–E）。

A. 配置实际被读取（打印实际生效值）
B. 找到确实发生 IR dropout 的样本，逐像素确认 IR 通道被整体替换为 125
C. 找到确实发生 Depth dropout 的样本，逐像素确认 Depth 通道被整体替换为 0
D. 输入 shape 恒为 [B,5,H,W]
E. VAL（augment=False）完全不执行 dropout

⚠️ 关键：get_image_and_label() 每次调用都会**重新随机抽取** dropout 状态，
   因此必须在「命中当次」立即捕获并校验，不能事后再取一次（那会是另一次抽样）。

只读脚本，不训练、不修改任何冻结产物。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.cfg import get_cfg                       # noqa: E402
from ultralytics.data.dataset import YOLODataset           # noqa: E402
import ultralytics.data.base as B                          # noqa: E402

CFG_YAML = PROJECT_ROOT / "configs/train_rgbid_modality_dropout.yaml"
CROP = yaml.safe_load(open(CFG_YAML, encoding="utf-8"))["modality_dropout"]
FAILS = []

def _find_dataset_yaml(root):
    """自动探测训练数据 yaml：云端为 rgbid_split_train，本地旧命名为 rgbid_split。"""
    cands = [
        root / "data/processed/rgbid_split_train/dataset.yaml",
        root / "data/processed/rgbid_split/dataset.yaml",
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(
        "未找到 RGBID dataset.yaml，已尝试：" + " | ".join(str(c) for c in cands) +
        " 请先在云端跑一次训练切分（train.py 会自动生成 rgbid_split_train/）。")
def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


print("=" * 76)
print("A. 配置实际被读取（值来自 configs/train_rgbid_modality_dropout.yaml）")
print("=" * 76)
for k in ("enabled", "ir_drop_prob", "depth_drop_prob", "ir_fill", "depth_fill",
          "mutually_exclusive", "apply_train_only"):
    print(f"    {k:<20} = {CROP.get(k)}")
check("A1 enabled = True", CROP.get("enabled") is True)
check("A2 ir_drop_prob = 0.075", abs(CROP.get("ir_drop_prob", 0) - 0.075) < 1e-9)
check("A3 depth_drop_prob = 0.075", abs(CROP.get("depth_drop_prob", 0) - 0.075) < 1e-9)
check("A4 ir_fill = 125", CROP.get("ir_fill") == 125)
check("A5 depth_fill = 0", CROP.get("depth_fill") == 0)

DATA_YAML = _find_dataset_yaml(PROJECT_ROOT)
print("  数据 yaml:", DATA_YAML)
dcfg = yaml.safe_load(open(DATA_YAML, encoding="utf-8"))
root = Path(dcfg["path"])


def make_hyp():
    h = get_cfg()
    h.imgsz, h.cache, h.rect = 640, False, False
    h.mosaic = h.mixup = h.copy_paste = 0.0
    h.degrees = h.translate = h.scale = h.shear = h.perspective = 0.0
    h.flipud = h.fliplr = 0.0
    h.hsv_h = h.hsv_s = h.hsv_v = h.erasing = 0.0
    h.brightness = 0.0
    h.close_mosaic = 0
    h.channels, h.use_simotm = 5, "RGBID"
    h.modality_dropout = dict(CROP)
    return h


def build(mode):
    return YOLODataset(
        img_path=str(root / ("images/train/visible" if mode == "train" else "images/val/visible")),
        imgsz=640, batch_size=8, augment=(mode == "train"), hyp=make_hyp(),
        rect=False, cache=False, single_cls=False, stride=32,
        pad=0.0 if mode == "train" else 0.5, prefix=f"{mode}: ",
        task="detect", data=dcfg, classes=None, fraction=1.0,
        use_simotm="RGBID", pairs_rgb_ir=["visible", "infrared"],
        pairs_rgb_depth=["visible", "depth"],
    )


print()
print("=" * 76)
print("B/C. 逐像素验证 dropout 的实际填充值（命中当次捕获）")
print("=" * 76)

ds_tr = build("train")
ir_hits, dep_hits = [], []
ir_ok = dep_ok = None
scanned = 0
for i in range(len(ds_tr)):
    scanned += 1
    img = ds_tr.get_image_and_label(i)["img"]      # (H,W,5) uint8, [B,G,R,IR,D]
    assert img.ndim == 3 and img.shape[2] == 5, f"shape 异常 {img.shape}"
    c3, c4 = img[:, :, 3].copy(), img[:, :, 4].copy()
    ir_filled = bool(np.all(c3 == 125))
    dep_filled = bool(np.all(c4 == 0))
    if ir_filled and not dep_filled:
        ir_hits.append(i)
        if ir_ok is None:
            ir_ok = dict(idx=i, shape=tuple(img.shape),
                         ir_unique=np.unique(c3).tolist()[:5],
                         dep_nuniq=int(len(np.unique(c4))),
                         rgb_nuniq=int(len(np.unique(img[:, :, :3]))))
    elif dep_filled and not ir_filled:
        dep_hits.append(i)
        if dep_ok is None:
            dep_ok = dict(idx=i, shape=tuple(img.shape),
                          dep_unique=np.unique(c4).tolist()[:5],
                          ir_nuniq=int(len(np.unique(c3))))
    if len(ir_hits) >= 5 and len(dep_hits) >= 5:
        break

print(f"  扫描样本数: {scanned}")
print(f"  命中 IR-drop    idx: {ir_hits}  (共 {len(ir_hits)})")
print(f"  命中 Depth-drop idx: {dep_hits}  (共 {len(dep_hits)})")
check("B1 存在 IR dropout 样本", len(ir_hits) > 0)
check("C1 存在 Depth dropout 样本", len(dep_hits) > 0)

if ir_ok:
    print(f"\n  【IR-drop】idx={ir_ok['idx']}  shape={ir_ok['shape']}  （命中当次捕获）")
    print(f"      IR  通道唯一值      : {ir_ok['ir_unique']}      <- 应恒为 [125]")
    print(f"      Depth 通道唯一值个数: {ir_ok['dep_nuniq']}      <- 应 >1（未被改动）")
    print(f"      RGB  通道唯一值个数 : {ir_ok['rgb_nuniq']}      <- 应 >1（未被改动）")
    check("B2 IR 通道被整体替换为 125", ir_ok["ir_unique"] == [125], f"{ir_ok['ir_unique']}")
    check("B3 IR-drop 未误改 Depth", ir_ok["dep_nuniq"] > 1)
    check("B4 IR-drop 未误改 RGB", ir_ok["rgb_nuniq"] > 1)

if dep_ok:
    print(f"\n  【Depth-drop】idx={dep_ok['idx']}  shape={dep_ok['shape']}  （命中当次捕获）")
    print(f"      Depth 通道唯一值    : {dep_ok['dep_unique']}      <- 应恒为 [0]")
    print(f"      IR    通道唯一值个数: {dep_ok['ir_nuniq']}      <- 应 >1（未被改动）")
    check("C2 Depth 通道被整体替换为 0", dep_ok["dep_unique"] == [0], f"{dep_ok['dep_unique']}")
    check("C3 Depth-drop 未误改 IR", dep_ok["ir_nuniq"] > 1)

print()
print("=" * 76)
print("D. 输入 shape")
print("=" * 76)
from torch.utils.data import DataLoader  # noqa: E402
B.MODALITY_DROPOUT_STATS.update({"full": 0, "ir_drop": 0, "depth_drop": 0, "both_drop": 0})
dl = DataLoader(ds_tr, batch_size=4, shuffle=False, collate_fn=YOLODataset.collate_fn, num_workers=0)
shapes = []
for bi, batch in enumerate(dl):
    shapes.append(tuple(batch["img"].shape))
    if bi >= 3:
        break
print(f"  前 {len(shapes)} 个 batch img shape: {shapes}")
check("D1 所有 batch 均为 [B,5,H,W]", all(len(s) == 4 and s[1] == 5 for s in shapes))
st = B.MODALITY_DROPOUT_STATS
tot = sum(st[k] for k in ("full", "ir_drop", "depth_drop", "both_drop"))
print("  训练分布 (n=%d): " % tot +
      "  ".join(f"{k}={st[k]/tot*100:.2f}%" for k in ("full", "ir_drop", "depth_drop", "both_drop")))
check("D2 both_drop == 0（互斥）", st["both_drop"] == 0)

print()
print("=" * 76)
print("E. VAL：augment=False 完全不执行 dropout")
print("=" * 76)
B.MODALITY_DROPOUT_STATS.update({"full": 0, "ir_drop": 0, "depth_drop": 0, "both_drop": 0})
ds_va = build("val")
n_ir_fill = n_dep_fill = 0
for i in range(len(ds_va)):
    img = ds_va.get_image_and_label(i)["img"]
    assert img.shape[2] == 5
    if np.all(img[:, :, 3] == 125):
        n_ir_fill += 1
    if np.all(img[:, :, 4] == 0):
        n_dep_fill += 1
stv = B.MODALITY_DROPOUT_STATS
print(f"  val 样本数: {len(ds_va)}")
print(f"  整通道==125 的样本数: {n_ir_fill}   整通道==0 的样本数: {n_dep_fill}")
print(f"  val dropout 计数器  : {dict(stv)}")
check("E1 val 模式 dropout 计数器零增量（函数未执行）",
      stv["ir_drop"] == 0 and stv["depth_drop"] == 0 and stv["both_drop"] == 0)
check("E2 val 的 augment 标志为 False", ds_va.augment is False)

print()
print("=" * 76)
print(f"最终验证结果: {'全部 PASS —— 可以启动训练' if not FAILS else 'FAIL -> ' + str(FAILS)}")
print("=" * 76)
sys.exit(1 if FAILS else 0)
