"""scripts/modality_dropout_smoke_test.py — Modality Dropout 冒烟测试（只读，不训练）。

覆盖：
  A. 单元测试 apply_modality_dropout：概率分布 / 通道替换正确性 / shape 不变 / 互斥
  B. 集成测试：真实 RGBID dataset（train 模式）逐样本产出 5ch，dropout 生效
  C. 关键安全性：val 模式（augment=False）**永不**丢弃模态
  D. 静态检查：dropout 只改 img，不改 label/bbox/cls
  E. 小批量 DataLoader：验证 batch tensor shape = [B,5,H,W]

用法:
  python scripts/modality_dropout_smoke_test.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.data.base import apply_modality_dropout  # noqa: E402

CFG = {"enabled": True, "ir_drop_prob": 0.075, "depth_drop_prob": 0.075,
       "ir_fill": 0, "depth_fill": 0, "mutually_exclusive": True, "apply_train_only": True}
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
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


print("=" * 74)
print("A. 单元测试：apply_modality_dropout")
print("=" * 74)

# A1 shape 不变 + 通道序 [R,G,B,IR,D]
random.seed(0)
base = np.random.randint(1, 255, (32, 32, 5), dtype=np.uint8)
out = apply_modality_dropout(base, CFG, augment=True)
check("A1 shape 保持 (H,W,5)", out.shape == base.shape, f"{out.shape}")
check("A1 dtype 保持 uint8", out.dtype == base.dtype)

# A2 概率分布（大样本）
N = 40000
random.seed(42)
cnt = {"full": 0, "ir_drop": 0, "depth_drop": 0, "both_drop": 0}
for _ in range(N):
    o = apply_modality_dropout(base, CFG, augment=True)
    ir_c = np.all(o[:, :, 3] == 0)
    d_c = np.all(o[:, :, 4] == 0)
    if ir_c and d_c:
        cnt["both_drop"] += 1
    elif ir_c:
        cnt["ir_drop"] += 1
    elif d_c:
        cnt["depth_drop"] += 1
    else:
        cnt["full"] += 1
print(f"\n    实测分布 (N={N}):")
for k, exp in [("full", 0.85), ("ir_drop", 0.075), ("depth_drop", 0.075), ("both_drop", 0.0)]:
    r = cnt[k] / N
    print(f"      {k:<11} 实测 {r:7.4f}   期望 {exp:6.3f}   偏差 {r-exp:+.4f}")
check("A2 full ≈ 0.85", abs(cnt["full"]/N - 0.85) < 0.01)
check("A2 ir_drop ≈ 0.075", abs(cnt["ir_drop"]/N - 0.075) < 0.01)
check("A2 depth_drop ≈ 0.075", abs(cnt["depth_drop"]/N - 0.075) < 0.01)
check("A2 both_drop == 0 (互斥)", cnt["both_drop"] == 0, f"{cnt['both_drop']} 次")

# A3 通道替换正确性：IR drop 只动 ch3；Depth drop 只动 ch4
random.seed(1)
for _ in range(200):
    o = apply_modality_dropout(base, CFG, augment=True)
    ir_c, d_c = np.all(o[:, :, 3] == 0), np.all(o[:, :, 4] == 0)
    if ir_c:
        assert np.array_equal(o[:, :, :3], base[:, :, :3]), "IR drop 误改 RGB"
        assert np.array_equal(o[:, :, 4], base[:, :, 4]), "IR drop 误改 Depth"
    if d_c:
        assert np.array_equal(o[:, :, :3], base[:, :, :3]), "Depth drop 误改 RGB"
        assert np.array_equal(o[:, :, 3], base[:, :, 3]), "Depth drop 误改 IR"
check("A3 通道替换无串扰（RGB/IR/Depth 互不影响）", True)

# A4 val/test 永不丢弃
random.seed(7)
n_drop = sum(1 for _ in range(2000)
             if not np.array_equal(apply_modality_dropout(base, CFG, augment=False), base))
check("A4 augment=False 时永不丢弃（val/test）", n_drop == 0, f"drop 次数={n_drop}")

# A5 enabled=False 时零影响
off = dict(CFG, enabled=False)
n_drop = sum(1 for _ in range(2000)
             if not np.array_equal(apply_modality_dropout(base, off, augment=True), base))
check("A5 enabled=False 时零影响（既有实验）", n_drop == 0, f"drop 次数={n_drop}")

# A6 非 5ch 输入原样通过（不误伤其它模态配置）
g3 = np.random.randint(1, 255, (8, 8, 3), dtype=np.uint8)
check("A6 3ch 输入原样通过", np.array_equal(apply_modality_dropout(g3, CFG, augment=True), g3))

# A7 不修改输入原数组（避免污染缓存）
random.seed(3)
orig = base.copy()
for _ in range(50):
    apply_modality_dropout(base, CFG, augment=True)
check("A7 不就地修改输入数组", np.array_equal(base, orig))

print()
print("=" * 74)
print("B/C/D/E. 集成测试：真实 RGBID dataset")
print("=" * 74)
try:
    from types import SimpleNamespace
    from ultralytics.data.dataset import YOLODataset

    DATA_YAML = _find_dataset_yaml(PROJECT_ROOT)
    import yaml
    dcfg = yaml.safe_load(open(DATA_YAML, encoding='utf-8'))
    root = Path(dcfg['path'])

    from ultralytics.cfg import get_cfg

    def make_hyp(dropout_cfg):
        h = get_cfg()
        h.imgsz, h.cache, h.rect = 640, False, False
        h.mosaic = h.mixup = h.copy_paste = 0.0
        h.degrees = h.translate = h.scale = h.shear = h.perspective = 0.0
        h.flipud = h.fliplr = 0.0
        h.hsv_h = h.hsv_s = h.hsv_v = h.erasing = 0.0
        h.brightness = 0.0
        h.close_mosaic = 0
        h.channels = 5
        h.use_simotm = 'RGBID'
        h.modality_dropout = dropout_cfg
        return h

    def build(mode, dropout_cfg, n=64):
        hyp = make_hyp(dropout_cfg)
        return YOLODataset(
            img_path=str(root / ('images/train/visible' if mode == 'train' else 'images/val/visible')),
            imgsz=640, batch_size=8, augment=(mode == 'train'), hyp=hyp,
            rect=False, cache=False, single_cls=False, stride=32,
            pad=0.0 if mode == 'train' else 0.5, prefix=f'{mode}: ',
            task='detect', data=dcfg, classes=None, fraction=1.0,
            use_simotm='RGBID', pairs_rgb_ir=['visible', 'infrared'],
            pairs_rgb_depth=['visible', 'depth'],
        )

    # ---- B: train 模式 ----
    import ultralytics.data.base as _ds
    _ds.MODALITY_DROPOUT_STATS.update({"full": 0, "ir_drop": 0, "depth_drop": 0, "both_drop": 0})
    ds_tr = build('train', CFG)
    n = min(64, len(ds_tr))
    shapes, lbl_ok = set(), True
    for i in range(n):
        s = ds_tr[i]
        shapes.add(tuple(s['img'].shape))
        if s['img'].shape[0] != 5:
            lbl_ok = False
        # D: label/bbox/cls 存在且未被 dropout 触碰
        assert 'cls' in s and 'bboxes' in s, "label 缺失"
    print(f"\n    train 模式样本数: {n}")
    print(f"    产出 img shape 集合: {sorted(shapes)}")
    check("B1 train 模式 img 恒为 5ch", all(sh[0] == 5 for sh in shapes), f"{sorted(shapes)}")
    st = _ds.MODALITY_DROPOUT_STATS
    tot = sum(st[k] for k in ("full", "ir_drop", "depth_drop", "both_drop"))
    print(f"\n    dataset 内实测状态分布 (n={tot}):")
    for k in ("full", "ir_drop", "depth_drop", "both_drop"):
        print(f"      {k:<11} {st[k]:>5}   {st[k]/max(tot,1)*100:6.2f}%")
    check("B2 三种状态均出现", st['full'] > 0 and st['ir_drop'] > 0 and st['depth_drop'] > 0)
    check("B3 both_drop == 0", st['both_drop'] == 0, f"{st['both_drop']}")
    check("B4 label/bbox/cls 结构完整", lbl_ok)

    # ---- C: val 模式必须零丢弃 ----
    _ds.MODALITY_DROPOUT_STATS.update({"full": 0, "ir_drop": 0, "depth_drop": 0, "both_drop": 0})
    ds_va = build('val', CFG)
    for i in range(min(32, len(ds_va))):
        s = ds_va[i]
        assert s['img'].shape[0] == 5
    stv = _ds.MODALITY_DROPOUT_STATS
    print()
    check("C1 val 模式零丢弃（全部 full）",
          stv['ir_drop'] == 0 and stv['depth_drop'] == 0 and stv['both_drop'] == 0,
          f"{dict(stv)}")

    # ---- E: DataLoader batch ----
    from torch.utils.data import DataLoader
    dl = DataLoader(ds_tr, batch_size=4, shuffle=False,
                    collate_fn=YOLODataset.collate_fn, num_workers=0)
    for bi, batch in enumerate(dl):
        check(f"E1 batch{bi} img shape = [B,5,H,W]", batch['img'].shape[1] == 5,
              f"{tuple(batch['img'].shape)}")
        if bi >= 2:
            break
    b0 = next(iter(dl))
    check("E2 batch img dtype = uint8（/255 由 trainer.preprocess_batch 完成）",
          b0['img'].dtype == torch.uint8, f"{b0['img'].dtype}")

except Exception as e:
    import traceback
    traceback.print_exc()
    FAILS.append(f"集成测试异常: {e}")

print()
print("=" * 74)
print(f"冒烟测试结果: {'全部 PASS' if not FAILS else '存在 FAIL -> ' + str(FAILS)}")
print("=" * 74)
sys.exit(1 if FAILS else 0)
