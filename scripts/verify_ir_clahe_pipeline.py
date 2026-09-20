"""scripts/verify_ir_clahe_pipeline.py — 证明 RGBID 训练集实际喂的是 CLAHE 而非 percentile（只读）。

背景：本仓库的 IR 对比度处理有 **两个互斥分支**（base.py:334 / loaders.py:656）:

    if ir_encoding == "clahe":  CLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(im_infrared)
    else:                       percentile 1%/99% 线性拉伸

它们**不是叠加关系**。把 `percentile -> CLAHE` 串起来算会得到一个既不是 CLAHE 也不是
percentile 的第三种图像，从而误判为 "matches CLAHE: False"（2026-09 已两次踩坑）。
本脚本用**互斥假设集**逐一比对数据集真实产出的 IR 通道，并附 visible 通道作对照组，
把「数据集到底走的哪条分支」变成可判定的事实。

用法:
    python scripts/verify_ir_clahe_pipeline.py                    # 默认 train split, 20 个样本
    python scripts/verify_ir_clahe_pipeline.py --n 40 --split val
退出码: 0 = 数据集产出与 CLAHE 假设逐位一致且与 percentile 假设不一致；1 = 否则。
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPLIT_ROOT = PROJECT_ROOT / "data" / "processed" / "rgbid_split_train"
IMGSZ = 1280


def _2d(a):
    """把 (h,w,1) 与 (h,w) 归一到 (h,w)。

    注意：一旦 import ultralytics，`cv2.imread` 会被替换成 fork 版
    （ultralytics/utils/patches.py:39 `im[..., None] if im.ndim == 2 else im`，强制 3 维），
    因此 IR 原图到此是 (h,w,1)，而 CLAHE/percentile 的输出是 (h,w)。
    两侧都必须按同一语义归一化，否则会得到纯形状层面的假 FAIL。
    """
    return a[..., 0] if (a.ndim == 3 and a.shape[-1] == 1) else a


def _to_uint8_gray(im):
    """复刻 base.py:320-326 的 IR dtype 统一（在对比度处理之前）。"""
    if im.dtype == np.uint16:
        return (im.astype(np.float32) * (255.0 / 65535.0)).astype(np.uint8)
    if im.dtype != np.uint8:
        f = im.astype(np.float32)
        lo, hi = f.min(), f.max()
        return ((f - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(f)).astype(np.uint8)
    return im


def _clahe(im):
    return _2d(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(_2d(im)))


def _percentile(im):
    im = _2d(im)
    lo, hi = np.percentile(im, (1.0, 99.0))
    if hi - lo > 1:
        return np.clip((im.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return im


def _resize(im, target_hw, augment):
    """复刻 base.py:_resize_images_3 的单路 resize（r = imgsz / max(h,w)，INTER_LINEAR 优先）。"""
    im = _2d(im)
    h, w = im.shape[:2]
    th, tw = target_hw
    if (h, w) == (th, tw):
        return im
    r = IMGSZ / max(h, w)
    if r == 1:
        return im
    interp = cv2.INTER_LINEAR if (augment or r > 1) else cv2.INTER_AREA
    return cv2.resize(im, (min(int(np.ceil(w * r)), IMGSZ), min(int(np.ceil(h * r)), IMGSZ)),
                      interpolation=interp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="要检验的样本数")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    args = ap.parse_args()

    augment = args.split == "train"  # build_yolo_dataset: rect=(mode=='val') -> 训练侧走 INTER_LINEAR
    vis_dir = SPLIT_ROOT / "images" / args.split / "visible"
    files = sorted(p for p in vis_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not files:
        print(f"no images under {vis_dir}")
        return 1

    # 用真实 dataset 对象拿 ground truth（走 base.py 中真正含该分支的
    # `load_and_preprocess_image`），而不是把预处理复刻一遍
    from ultralytics.data.base import BaseDataset

    ds = BaseDataset.__new__(BaseDataset)
    ds.imgsz = IMGSZ
    ds.augment = augment
    ds.use_simotm = "RGBID"
    ds.pairs_rgb_ir = ["visible", "infrared"]
    ds.pairs_rgb_depth = ["visible", "depth"]

    class _Hyp:
        ir_encoding = "clahe"

    ds.hyp = _Hyp()

    def produce(f):
        return ds.load_and_preprocess_image(f, use_simotm="RGBID", pairs_rgb="visible",
                                            pairs_ir="infrared", pairs_depth="depth")

    ds.im_files = [str(p) for p in files[: args.n]]

    HUES = {
        "H6 CLAHE@native -> resize LIN": lambda ir, hw: _resize(_clahe(_to_uint8_gray(ir)), hw, augment),
        "H2 percentile->CLAHE -> resize": lambda ir, hw: _resize(_clahe(_percentile(_to_uint8_gray(ir))), hw, augment),
        "H1 percentile -> resize": lambda ir, hw: _resize(_percentile(_to_uint8_gray(ir)), hw, augment),
        "H5 resize only": lambda ir, hw: _resize(_to_uint8_gray(ir), hw, augment),
    }
    stats = {k: [] for k in HUES}
    stats["CTRL visible -> resize LIN"] = []
    raw_shapes, out_shapes = set(), set()

    for f in ds.im_files:
        im = produce(f)
        got_ir = im[..., 3]
        got_vis = im[..., :3]
        hw = got_ir.shape[:2]
        raw_shapes.add(tuple(cv2.imread(f, cv2.IMREAD_GRAYSCALE).shape))
        out_shapes.add(tuple(im.shape))

        ir_raw = cv2.imread(f.replace("visible", "infrared"), cv2.IMREAD_GRAYSCALE)
        for name, fn in HUES.items():
            cand = fn(ir_raw.copy(), hw)
            stats[name].append(float(np.abs(got_ir.astype(np.int16) - cand.astype(np.int16)).mean()))

        vis_raw = cv2.imread(f, cv2.IMREAD_COLOR)
        ctrl = _resize(vis_raw, hw, augment)
        stats["CTRL visible -> resize LIN"].append(
            float(np.abs(got_vis.astype(np.int16) - ctrl.astype(np.int16)).mean()))

    print(f"split={args.split}  augment={augment}  imgsz={IMGSZ}  samples={len(ds.im_files)}")
    print(f"raw IR shapes   : {sorted(raw_shapes)}")
    print(f"dataset outputs : {sorted(out_shapes)}")
    print(f"\n{'hypothesis':38s} {'mean':>10s} {'min':>10s} {'max':>10s}")
    print("-" * 72)
    for name, vals in stats.items():
        print(f"{name:38s} {np.mean(vals):10.4f} {np.min(vals):10.4f} {np.max(vals):10.4f}")

    h6 = np.max(stats["H6 CLAHE@native -> resize LIN"])
    hctrl = np.max(stats["CTRL visible -> resize LIN"])
    others = max(np.min(v) for k, v in stats.items()
                 if k not in ("H6 CLAHE@native -> resize LIN", "CTRL visible -> resize LIN"))
    print("-" * 72)
    ok = h6 < 1e-6 and hctrl < 1e-6 and others > 1e-6
    print(f"H6 max|Δ| == 0            : {h6 < 1e-6}  (max {h6:.6f})")
    print(f"visible control max|Δ|==0 : {hctrl < 1e-6}  (max {hctrl:.6f})  <- probe itself is correct")
    print(f"every other hypothesis > 0: {others > 1e-6}  (min {others:.6f})")
    print("\nIR_CLAHE_PIPELINE_CONFIRMED" if ok else "\nIR_CLAHE_PIPELINE_NOT_CONFIRMED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
