# -*- coding: utf-8 -*-
"""P0 §17/§18/§19 — IR augmentation 的 CPU-only dry-run 门（G1..G13）。

不训练、不起 DataLoader worker、不碰 CUDA。全部在真实 Dataset/Transform 代码路径上跑。

用法:
    YOLO_OFFLINE=True python diagnostic/ir_aug_audit/dryrun_gates.py
"""
import os
import sys
import json
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ultralytics.data.base as B  # noqa: E402
from ultralytics.data.base import IR_AUG_STATS  # noqa: E402
from ultralytics.data.augment import Format  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.cfg import get_cfg, DEFAULT_CFG_DICT  # noqa: E402

DATA_DIR = os.path.abspath("data/processed/rgbid_split_train")
TRAIN_VIS = os.path.join(DATA_DIR, "images", "train", "visible")
VAL_VIS = os.path.join(DATA_DIR, "images", "val", "visible")
NC, NAMES = 12, {i: n for i, n in enumerate(
    ["person", "boat", "animal", "seat", "sign", "bicycle", "car", "ball", "light", "garbage_can", "uav", "tricycle"])}

# D′ 的 IR aug 关闭态（= 现在线上的 D′）
IR_AUG_OFF = {"ir_gamma": [1.0, 1.0], "ir_gamma_probability": 0.0,
              "ir_noise_std": 0.0, "ir_noise_probability": 0.0}
# 候选实验的开启态
IR_AUG_ON = {"ir_gamma": [0.75, 1.35], "ir_gamma_probability": 0.35,
             "ir_noise_std": 0.015, "ir_noise_probability": 0.20}

RES = {"gates": {}}


def cfg_of(ir_aug=None, **extra):
    base = {"task": "detect", "imgsz": 1280, "rect": False, "cache": False, "single_cls": False,
            "classes": None, "fraction": 1.0, "seed": 42, "deterministic": True,
            "use_simotm": "RGBID", "channels": 5, "ir_encoding": "clahe",
            "pairs_rgb_ir": ["visible", "infrared"], "pairs_rgb_depth": ["visible", "depth"],
            "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0, "close_mosaic": 10,
            "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "flipud": 0.0, "fliplr": 0.5,
            "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0, "perspective": 0.0,
            "mask_ratio": 4, "overlap_mask": True, "bgr": 0.0}
    if ir_aug:
        base.update(ir_aug)
    base.update(extra)
    return get_cfg(DEFAULT_CFG_DICT, overrides=base)


def data_dict():
    return {"path": DATA_DIR, "train": "images/train/visible", "val": "images/val/visible",
            "nc": NC, "names": NAMES}


def build(cfg, mode, vis=TRAIN_VIS):
    return build_yolo_dataset(cfg, vis, batch=8, data=data_dict(), mode=mode, rect=False, stride=32,
                              multi_modal=False, use_simotm="RGBID",
                              pairs_rgb_ir=["visible", "infrared"], pairs_rgb_depth=["visible", "depth"])


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def reset_stats():
    for k in IR_AUG_STATS:
        IR_AUG_STATS[k] = [] if isinstance(IR_AUG_STATS[k], list) else 0
    global _CALL_SAMPLES
    _CALL_SAMPLES = []


_CALL_SAMPLES = []
_ORIG = B.apply_ir_augmentation


def instrumented(*a, **kw):
    _CALL_SAMPLES.append((a[0].shape, tuple(a[0].ravel()[:4].tolist())))
    return _ORIG(*a, **kw)


B.apply_ir_augmentation = instrumented  # 只用于统计"每样本调用次数"


def g(name, ok, **info):
    RES["gates"][name] = {"pass": bool(ok), **info}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {info}")


def main():
    print("== G1: config parses ==")
    try:
        c_off, c_on = cfg_of(IR_AUG_OFF), cfg_of(IR_AUG_ON)
        g("G1_config_parse", c_on.ir_gamma == [0.75, 1.35] and c_on.ir_noise_std == 0.015,
          ir_gamma=c_on.ir_gamma, ir_gamma_probability=c_on.ir_gamma_probability,
          ir_noise_std=c_on.ir_noise_std, ir_noise_probability=c_on.ir_noise_probability)
    except Exception as e:
        g("G1_config_parse", False, error=repr(e)); return

    print("== G2/G3: dataset loads + train transform executes ==")
    try:
        ds_on = build(c_on, "train")
        s = ds_on[0]
        img = s["img"]
        g("G2_dataset_load", len(ds_on) > 0, n=len(ds_on))
        g("G3_train_transform", img.ndim == 3 and img.shape[0] == 5,
          tensor_shape=list(img.shape), dtype=str(img.dtype), n_boxes=int(len(s["cls"])))
    except Exception as e:
        import traceback; traceback.print_exc()
        g("G2_dataset_load", False, error=repr(e)); return

    print("== G4: IR augmentation actually executes + measured rate (train) ==")
    N = 120
    reset_stats()
    seed_all()
    for i in range(N):
        _ = ds_on[i]
    per_sample = len(_CALL_SAMPLES) / float(N)
    g("G4_ir_aug_executes", IR_AUG_STATS["calls"] > 0,
      calls=IR_AUG_STATS["calls"], gamma_applied=IR_AUG_STATS["gamma_applied"],
      noise_applied=IR_AUG_STATS["noise_applied"], skipped_nonaugment=IR_AUG_STATS["skipped_nonaugment"])
    RES["load_calls_per_sample"] = per_sample
    gd = np.array(IR_AUG_STATS["gamma_draws"]) if IR_AUG_STATS["gamma_draws"] else np.array([1.0])
    RES["measured"] = {
        "n_samples": N, "preprocess_calls_per_sample": per_sample,
        "gamma_rate_measured": IR_AUG_STATS["gamma_applied"] / float(N),
        "gamma_rate_nominal": c_on.ir_gamma_probability,
        "noise_rate_measured": IR_AUG_STATS["noise_applied"] / float(N),
        "noise_rate_nominal": c_on.ir_noise_probability,
        "gamma_draw_mean": float(gd.mean()), "gamma_draw_min": float(gd.min()), "gamma_draw_max": float(gd.max()),
    }
    print(f"      preprocess calls/sample = {per_sample:.3f} (Mosaic buffer+ims 缓存 ⇒ 期望 1.0)")
    print(f"      gamma rate  measured={RES['measured']['gamma_rate_measured']:.3f} nominal={RES['measured']['gamma_rate_nominal']}")
    print(f"      noise rate  measured={RES['measured']['noise_rate_measured']:.3f} nominal={RES['measured']['noise_rate_nominal']}")
    print(f"      gamma draws mean={RES['measured']['gamma_draw_mean']:.3f} "
          f"[{RES['measured']['gamma_draw_min']:.3f}, {RES['measured']['gamma_draw_max']:.3f}]")

    print("== G5/G6/G12: RGB / Depth / shape unchanged under ACTIVE aug ==")
    ds = object.__new__(B.BaseDataset)
    ds.use_simotm, ds.imgsz = "RGBID", 1280
    f_vis = os.path.join(TRAIN_VIS, sorted(os.listdir(TRAIN_VIS))[0])
    outs = {}
    for tag, c in (("off", c_off), ("on", c_on)):
        ds.augment, ds.hyp = True, c
        seed_all(123)
        outs[tag] = ds.load_and_preprocess_image(f_vis, use_simotm="RGBID",
                                                 pairs_rgb="visible", pairs_ir="infrared", pairs_depth="depth")
    o, n = outs["off"], outs["on"]
    rgb_same = all(np.array_equal(o[:, :, i], n[:, :, i]) for i in (0, 1, 2))
    dep_same = np.array_equal(o[:, :, 4], n[:, :, 4])
    ir_same = np.array_equal(o[:, :, 3], n[:, :, 3])
    g("G5_rgb_unchanged", rgb_same)
    g("G6_depth_unchanged", dep_same)
    g("G12_shape_unchanged", o.shape == n.shape == (o.shape[0], o.shape[1], 5), shape=list(o.shape), dtype=str(o.dtype))
    RES["ir_channel_changed_under_active_aug"] = not ir_same
    print(f"      IR channel differs under active aug: {not ir_same} (expected True)")

    print("== G7: val transform unchanged ==")
    v_off, v_on = build(c_off, "val", VAL_VIS), build(c_on, "val", VAL_VIS)
    seed_all(7); a = v_off[0]["img"].numpy().copy()
    seed_all(7); b = v_on[0]["img"].numpy().copy()
    identical = np.array_equal(a, b)
    g("G7_val_unchanged", identical, max_abs_diff=float(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()))
    g("G7b_val_augment_flag", v_on.augment is False, val_augment=v_on.augment, val_transforms=str(type(v_on.transforms).__name__))

    print("== G8: inference path untouched ==")
    src = open("ultralytics/data/loaders.py", encoding="utf-8").read()
    n_ref = src.count("apply_ir_augmentation")
    g("G8_inference_no_ref", n_ref == 0, references_in_loaders_py=n_ref)

    print("== G9: determinism (same seed → identical) ==")
    # 必须用**全新** dataset：Dataset 自身带 ims/buffer 缓存，复用实例时缓存状态会
    # 跨调用残留，只重设 random/np/torch 种子并不足以复位（那是测试假象）。
    det = []
    for _ in range(2):
        d = build(c_on, "train")
        seed_all(99)
        det.append(d[1]["img"].numpy().copy())
    g("G9_determinism", np.array_equal(det[0], det[1]), identical=bool(np.array_equal(det[0], det[1])))
    # 再单独验证 IR augmentation 本身在固定种子下逐位可复现
    ir0 = np.asarray(np.random.RandomState(0).randint(0, 256, (64, 64)), np.uint8)
    c_always = cfg_of({"ir_gamma": [0.75, 1.35], "ir_gamma_probability": 1.0,
                       "ir_noise_std": 0.015, "ir_noise_probability": 1.0})
    outs = []
    for _ in range(2):
        random.seed(5); np.random.seed(5)
        outs.append(B.apply_ir_augmentation(ir0.copy(), c_always, True))
    g("G9b_ir_aug_determinism", np.array_equal(outs[0], outs[1]),
      identical=bool(np.array_equal(outs[0], outs[1])), changed=bool(not np.array_equal(outs[0], ir0)))

    print("== G10/G18: identity test (aug disabled vs enabled gamma=1,noise=0) ==")
    ds_off_notag = build(cfg_of(None), "train")          # 完全不带新键（= 历史 D′）
    n_samp, diffs_max, diffs_mean = 0, [], []
    ds_hyp = build(cfg_of(IR_AUG_OFF), "train")          # 显式恒等 + p=0
    for i in (0, 1, 2, 3):
        seed_all(4242); A = ds_off_notag[i]["img"].numpy().astype(np.int16)
        seed_all(4242); Bb = ds_hyp[i]["img"].numpy().astype(np.int16)
        d = np.abs(A - Bb); n_samp += 1
        diffs_max.append(int(d.max())); diffs_mean.append(float(d.mean()))
    g("G10_identity_exact", max(diffs_max) == 0,
      n_samples=n_samp, max_abs_diff=max(diffs_max), mean_abs_diff=max(diffs_mean))
    RES["identity"] = {"n_samples": n_samp, "max_abs_diff": max(diffs_max), "mean_abs_diff": max(diffs_mean)}

    print("== G11: no NaN / Inf ==")
    seed_all(5)
    ok = True
    for i in range(8):
        t = ds_on[i]["img"]
        if not torch.isfinite(t.float()).all():
            ok = False
    g("G11_no_nan_inf", ok)

    print("== G13: final 5-channel tensor ordering ==")
    probe = np.zeros((8, 8, 5), np.uint8)
    probe[..., 0], probe[..., 1], probe[..., 2], probe[..., 3], probe[..., 4] = 10, 20, 30, 40, 50
    t = Format(bbox_format="xywh", normalize=True, return_mask=False, return_keypoint=False,
               return_obb=False, batch_idx=True)(
        {"img": probe, "cls": np.zeros(0), "instances": _empty_instances(8, 8)})
    vals = [int(t["img"][c].max()) for c in range(5)]
    order = {10: "B", 20: "G", 30: "R", 40: "IR", 50: "D"}
    measured = [order.get(v, f"?{v}") for v in vals]
    g("G13_channel_order", measured[3] == "IR" and measured[4] == "D",
      measured_order=measured, ir_index=measured.index("IR"), depth_index=measured.index("D"),
      note="tensor 名 (D′=改动前同一函数) —— config 注释里的 [B,G,R,IR,D] 与实现不一致，见报告")

    RES["IR_AUG_STATS_after"] = {k: v for k, v in IR_AUG_STATS.items() if not isinstance(v, list)}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dryrun_gates.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(RES, fh, indent=2, ensure_ascii=False)
    n_pass = sum(1 for v in RES["gates"].values() if v["pass"])
    print(f"\n==== {n_pass}/{len(RES['gates'])} gates PASS ====\nwrote {out}")


def _empty_instances(h, w):
    from ultralytics.utils.instance import Instances
    return Instances(bboxes=np.zeros((0, 4), np.float32), segments=np.zeros((0, 0, 2), np.float32))


if __name__ == "__main__":
    main()
