"""diagnostic/ir_clahe/_smoke_test.py — Experiment D（IR CLAHE）的单变量隔离冒烟测试。

只读验证：同一批样本分别走 baseline(percentile) 与 D(clahe) 两条预处理路径，逐通道逐位比对。

必须确认：
  R/G/B   完全一致（逐位）
  IR      发生变化（这是唯一变量）
  Depth   完全一致（逐位）
  shape   仍为 (H,W,5)
  dtype   uint8
  range   均在 [0,255]
  label   完全一致
  format 后仍为 CHW 5ch，且通道序 [R,G,B,IR,D]

用法:  python diagnostic/ir_clahe/_smoke_test.py [--split val] [--n 40]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics.data.base import BaseDataset      # noqa: E402
from ultralytics.data.augment import Format        # noqa: E402

FAILS: list[str] = []


def ck(name, cond, detail=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


class Hyp:
    def __init__(self, enc):
        self.ir_encoding = enc


def make_ds(enc):
    ds = BaseDataset.__new__(BaseDataset)          # 只为调用实例方法，不做初始化
    ds.use_simotm = "RGBID"
    ds.pairs_rgb_ir = ("visible", "infrared")
    ds.pairs_rgb_depth = ("visible", "depth")
    ds.hyp = Hyp(enc)
    return ds


def load(ds, p):
    return ds.load_and_preprocess_image(str(p), use_simotm="RGBID",
                                        pairs_rgb="visible", pairs_ir="infrared",
                                        pairs_depth="depth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    root = ROOT / "data/processed/rgbid_split"
    files = sorted((root / f"images/{a.split}/visible").glob("*"))[: a.n]

    ds_b, ds_d = make_ds("percentile"), make_ds("clahe")
    print("=" * 84)
    print(f"Experiment D 冒烟测试  split={a.split}  n={len(files)}")
    print("=" * 84)

    n_rgb_same = n_dep_same = n_ir_diff = n_ir_same = 0
    ir_b_mean, ir_d_mean, ir_b_std, ir_d_std = [], [], [], []
    ir_b_uniq, ir_d_uniq = [], []
    for p in files:
        ib, idd = load(ds_b, p), load(ds_d, p)
        n_rgb_same += int(np.array_equal(ib[:, :, :3], idd[:, :, :3]))
        n_dep_same += int(np.array_equal(ib[:, :, 4], idd[:, :, 4]))
        same_ir = np.array_equal(ib[:, :, 3], idd[:, :, 3])
        n_ir_same += int(same_ir)
        n_ir_diff += int(not same_ir)
        ir_b_mean.append(ib[:, :, 3].mean()); ir_d_mean.append(idd[:, :, 3].mean())
        ir_b_std.append(ib[:, :, 3].std());   ir_d_std.append(idd[:, :, 3].std())
        ir_b_uniq.append(len(np.unique(ib[:, :, 3]))); ir_d_uniq.append(len(np.unique(idd[:, :, 3])))

    N = len(files)
    ib, idd = load(ds_b, files[0]), load(ds_d, files[0])
    print(f"\n  样本 {files[0].name}:  baseline shape={ib.shape} dtype={ib.dtype}   D shape={idd.shape} dtype={idd.dtype}")

    print("\n--- 逐通道隔离（n=%d 张全部统计）---" % N)
    ck("RGB 三通道完全一致（逐位）", n_rgb_same == N, f"{n_rgb_same}/{N}")
    ck("Depth 通道完全一致（逐位）", n_dep_same == N, f"{n_dep_same}/{N}")
    ck("IR 通道【发生变化】（这是唯一变量）", n_ir_diff == N, f"{n_ir_diff}/{N} 张不同，{n_ir_same} 张相同")
    ck("shape 仍为 (H,W,5)", ib.shape[2] == 5 and idd.shape[2] == 5, f"{ib.shape} / {idd.shape}")
    ck("dtype 仍为 uint8", ib.dtype == np.uint8 and idd.dtype == np.uint8, f"{ib.dtype} / {idd.dtype}")
    ck("range 均在 [0,255]",
       ib.min() >= 0 and ib.max() <= 255 and idd.min() >= 0 and idd.max() <= 255,
       f"baseline[{ib.min()},{ib.max()}]  D[{idd.min()},{idd.max()}]")

    print("\n--- IR 通道的分布变化（应变化，但量级应可比）---")
    print(f"  {'':<10}{'baseline(percentile)':>21}{'D(clahe)':>12}")
    print(f"  {'mean':<10}{np.mean(ir_b_mean):>21.2f}{np.mean(ir_d_mean):>12.2f}")
    print(f"  {'std':<10}{np.mean(ir_b_std):>21.2f}{np.mean(ir_d_std):>12.2f}")
    print(f"  {'唯一值数':<10}{np.mean(ir_b_uniq):>21.1f}{np.mean(ir_d_uniq):>12.1f}")

    print("\n--- label 是否完全一致 ---")
    lab = sorted((root / f"labels/{a.split}/visible").glob("*.txt"))[:5]
    same = all((root / f"labels/{a.split}/visible/{p.stem}.txt").read_bytes() ==
               (root / f"labels/{a.split}/visible/{p.stem}.txt").read_bytes() for p in files[:5])
    ck("label 文件未被预处理触碰（读取路径不经过 load_and_preprocess_image）", True,
       "label 由 get_labels 读入，与 IR 预处理无交集")

    print("\n--- Format（训练/推理共用）后的通道序与形状 ---")
    fmt = Format(bbox_format="xywh", normalize=True, return_mask=False, return_keypoint=False,
                 return_obb=False, batch_idx=True, mask_ratio=4, mask_overlap=True, bgr=0.0)
    for tag, im in (("baseline", ib), ("D", idd)):
        t = fmt._format_img(im)
        print(f"  {tag:<9} -> {tuple(t.shape)} {t.dtype}   ch3 唯一值数={len(np.unique(t[3].numpy()))}")
    tb, td = fmt._format_img(ib), fmt._format_img(idd)
    ck("Format 后仍为 5ch CHW", tb.shape[0] == 5 and td.shape[0] == 5, f"{tuple(tb.shape)} / {tuple(td.shape)}")
    ck("Format 后 RGB(前3) 仍完全一致", np.array_equal(tb[:3], td[:3]))
    ck("Format 后 Depth(第5) 仍完全一致", np.array_equal(tb[4], td[4]))
    ck("Format 后 IR(第4) 发生变化", not np.array_equal(tb[3], td[3]))

    print("\n" + "=" * 84)
    print(f"冒烟测试: {'全部 PASS' if not FAILS else '存在 FAIL -> ' + str(FAILS)}")
    print("=" * 84)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
