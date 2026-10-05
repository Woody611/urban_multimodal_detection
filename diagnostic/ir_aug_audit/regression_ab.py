# -*- coding: utf-8 -*-
"""P0 §19 — 回归护栏：证明「加了 IR augmentation 之后，**未启用**该增强的 D′ 数据流逐位不变」。

方法（真 A/B，不是推理）：
  1. 把改动前的 `base.py`（备份在 _before/）作为**独立模块**载入；
  2. 用它的 `load_and_preprocess_image` 在真实样本上生成 5ch 数组（= 改动前的 D′ 行为）；
  3. 用**当前** base.py 的同一方法、配 D′ 配置（IR aug 键全为默认恒等）再生成一次；
  4. 端到端再跑一次：把 dataset 的该方法临时换成"改动前版本"，与当前版本在同一 RNG
     种子下取同一批样本，逐位比较最终 5ch tensor。

全部 CPU-only。
"""
import os
import sys
import json
import random
import importlib.util
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import ultralytics.data.base as B  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.cfg import get_cfg, DEFAULT_CFG_DICT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BEFORE = os.path.join(HERE, "_before", "ultralytics_data_base.py")
DATA_DIR = os.path.abspath("data/processed/rgbid_split_train")
TRAIN_VIS = os.path.join(DATA_DIR, "images", "train", "visible")


def load_old_module():
    spec = importlib.util.spec_from_file_location("_base_before", BEFORE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_base_before"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_cfg(**over):
    b = {"task": "detect", "imgsz": 1280, "rect": False, "cache": False, "single_cls": False,
         "classes": None, "fraction": 1.0, "seed": 42, "use_simotm": "RGBID", "channels": 5,
         "ir_encoding": "clahe", "pairs_rgb_ir": ["visible", "infrared"],
         "pairs_rgb_depth": ["visible", "depth"], "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0,
         "fliplr": 0.5, "flipud": 0.0, "mask_ratio": 4, "overlap_mask": True, "bgr": 0.0}
    b.update(over)
    return get_cfg(DEFAULT_CFG_DICT, overrides=b)


def data_dict():
    return {"path": DATA_DIR, "train": "images/train/visible", "val": "images/val/visible",
            "nc": 12, "names": {i: str(i) for i in range(12)}}


def main():
    old = load_old_module()
    RES = {}
    cfg = make_cfg()  # = D′，新键取默认恒等

    # ---------- A: load_and_preprocess_image 级 ----------
    files = sorted(os.listdir(TRAIN_VIS))[:20]
    old_ds = object.__new__(old.BaseDataset)
    old_ds.use_simotm, old_ds.imgsz, old_ds.augment = "RGBID", 1280, True
    old_ds.hyp = cfg
    new_ds = object.__new__(B.BaseDataset)
    new_ds.use_simotm, new_ds.imgsz, new_ds.augment = "RGBID", 1280, True
    new_ds.hyp = cfg
    maxd, ndiff = 0, 0
    for f in files:
        p = os.path.join(TRAIN_VIS, f)
        kw = dict(use_simotm="RGBID", pairs_rgb="visible", pairs_ir="infrared", pairs_depth="depth")
        a = np.asarray(old_ds.load_and_preprocess_image(p, **kw), np.int16)
        b = np.asarray(new_ds.load_and_preprocess_image(p, **kw), np.int16)
        d = int(np.abs(a - b).max())
        maxd = max(maxd, d)
        ndiff += int(d != 0)
    RES["levelA_load_and_preprocess_image"] = {"n_images": len(files), "max_abs_diff": maxd,
                                               "n_images_differing": ndiff}
    print(f"[A] load_and_preprocess_image  old-vs-new  n={len(files)}  max_abs_diff={maxd}  differing={ndiff}")

    # ---------- B: 端到端 dataset 级（train, mosaic 开启） ----------
    ds = build_yolo_dataset(cfg, TRAIN_VIS, batch=8, data=data_dict(), mode="train", rect=False,
                            stride=32, use_simotm="RGBID",
                            pairs_rgb_ir=["visible", "infrared"], pairs_rgb_depth=["visible", "depth"])
    cur = type(ds).__mro__
    cur_fn = B.BaseDataset.load_and_preprocess_image
    old_fn = old.BaseDataset.load_and_preprocess_image

    def run(fn, idx, seed):
        d = build_yolo_dataset(cfg, TRAIN_VIS, batch=8, data=data_dict(), mode="train", rect=False,
                               stride=32, use_simotm="RGBID",
                               pairs_rgb_ir=["visible", "infrared"], pairs_rgb_depth=["visible", "depth"])
        B.BaseDataset.load_and_preprocess_image = fn
        try:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            return d[idx]["img"].numpy().astype(np.int16)
        finally:
            B.BaseDataset.load_and_preprocess_image = cur_fn

    emax, nend = 0, 0
    for i in (0, 1, 2, 5, 9):
        A = run(old_fn, i, 777)
        Bv = run(cur_fn, i, 777)
        d = int(np.abs(A - Bv).max())
        emax = max(emax, d); nend += int(d != 0)
    RES["levelB_end_to_end_train"] = {"n_samples": 5, "max_abs_diff": emax, "n_samples_differing": nend}
    print(f"[B] end-to-end train tensor   old-vs-new  n=5  max_abs_diff={emax}  differing={nend}")

    RES["verdict"] = "IDENTICAL" if (RES["levelA_load_and_preprocess_image"]["max_abs_diff"] == 0
                                     and RES["levelB_end_to_end_train"]["max_abs_diff"] == 0) else "DIFFERS"
    print(f"\nVERDICT: {RES['verdict']}")
    with open(os.path.join(HERE, "regression_ab.json"), "w", encoding="utf-8") as fh:
        json.dump(RES, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
