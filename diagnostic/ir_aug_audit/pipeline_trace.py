# -*- coding: utf-8 -*-
"""P0 §3/§4 — 用**真实代码路径**追踪 D' 的 IR 数据流（CPU only，不训练、不碰 CUDA）。

不重写、不猜测：直接调用
    ultralytics.data.base.BaseDataset.load_and_preprocess_image  (RGBID 分支)
以及它内部的 imread / apply_ir_encoding / _merge_channels_rgbid，
在真实样本上逐步打印 shape / dtype / range，并把"样本数组"逐位抓下来供后续审计复用。

输出:
  - stdout: 每阶段 shape/dtype/min/max
  - diagnostic/ir_aug_audit/pipeline_trace.json
"""
import os
import sys
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ultralytics.data.base as B          # noqa: E402
from ultralytics.data.base import apply_ir_encoding  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_trace.json")
IR_DIR = "data/processed/rgbid_split_train/images/train/infrared"
DEPTH_DIR = "data/processed/rgbid_split_train/images/train/depth"
VIS_DIR = "data/processed/rgbid_split_train/images/train/visible"
SAMPLE = "00000004.jpg"


def d(a):
    a = np.asarray(a)
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "min": int(a.min()), "max": int(a.max()),
            "mean": round(float(a.mean()), 4)}


def main():
    res = {"sample": SAMPLE, "note": "真实代码路径；imread = ultralytics.utils.patches.imread"}

    # ---- stage 0: 原始文件 ----
    vis = B.imread(os.path.join(VIS_DIR, SAMPLE))
    ir = B.imread(os.path.join(IR_DIR, SAMPLE), cv2.IMREAD_GRAYSCALE)
    dep = B.imread(os.path.join(DEPTH_DIR, SAMPLE), cv2.IMREAD_UNCHANGED)
    res["stage0_imread"] = {"visible": d(vis), "infrared_GRAYSCALE": d(ir), "depth_UNCHANGED": d(dep)}

    # ---- stage 1: IR 对比度处理（代码中的真实调用） ----
    ir_clahe = apply_ir_encoding(ir, "clahe")
    res["stage1_apply_ir_encoding"] = {"in": d(ir), "out_clahe": d(ir_clahe)}

    # ---- stage 2: depth -> uint8（代码的真实分支） ----
    dep2 = dep
    if dep2.ndim == 3:
        dep2 = dep2[..., 0]
    if dep2.dtype != np.uint8:
        dep2 = dep2.astype(np.float32)
        dep2[dep2 < 300] = 0.0
        dep2 = np.clip(dep2 / 19999.0 * 255.0, 0, 255).astype(np.uint8)
    res["stage2_depth_to_uint8"] = d(dep2)

    # ---- stage 3: merge（真实 _merge_channels_rgbid） ----
    try:
        merged = B.BaseDataset._merge_channels_rgbid(None, vis, ir_clahe, dep2)
        res["stage3_merge"] = {"ok": True, **d(merged)}
    except Exception as e:  # 报错也要如实记录
        res["stage3_merge"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        merged = None

    # ---- stage 4: 若 merge 失败，尝试 depth 也保持 (H,W,1) ----
    if merged is None:
        try:
            m2 = np.concatenate([vis, ir_clahe, dep2[..., None]], axis=2)
            res["stage3b_concat_fallback"] = {"ok": True, **d(m2)}
        except Exception as e:
            res["stage3b_concat_fallback"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # ---- stage 5: 真实入口 load_and_preprocess_image ----
    ds = object.__new__(B.BaseDataset)
    ds.use_simotm = "RGBID"
    ds.hyp = type("H", (), {"ir_encoding": "clahe"})()
    ds.imgsz = 1280
    ds.augment = True
    f = os.path.join(VIS_DIR, SAMPLE)
    try:
        im = ds.load_and_preprocess_image(f, use_simotm="RGBID",
                                          pairs_rgb="visible", pairs_ir="infrared", pairs_depth="depth")
        res["stage5_load_and_preprocess_image"] = {"ok": True, **d(im)}
        # 抓样本像素供后续审计
        np.save(os.path.join(os.path.dirname(OUT), "sample_5ch_ir_clahe.npy"), im)
    except Exception as e:
        res["stage5_load_and_preprocess_image"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

    for k, v in res.items():
        if k in ("sample", "note"):
            continue
        print(f"[{k}]")
        if isinstance(v, dict) and "shape" not in v:
            for kk, vv in v.items():
                print(f"   {kk}: {vv}")
        else:
            print(f"   {v}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
