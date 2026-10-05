# -*- coding: utf-8 -*-
"""FINAL LAUNCH GATE — §6..§12 机器验证（CPU only，不训练、不起 CUDA）。

§6  Exact IR insertion path      —— 用 AST + 运行时钩子证明 gamma/noise 在 CLAHE 之后、merge 之前
§7  Channel identity             —— 逐通道对回原始 B/G/R、IR、Depth
§8  No double augmentation       —— 枚举全部调用点 + Albumentations 状态 + 调用率
§9  Train-only isolation         —— train / val / predict 三条路径
§10 Identity / determinism       —— Test A/B/C/D
§11 Numerical safety             —— NaN/Inf/dtype/shape + noise 单位锁定
§12 Probability runtime          —— 实测 firing rates

用法: YOLO_OFFLINE=True python diagnostic/ir_aug_audit/launch_gate.py
"""
import os
import re
import sys
import ast
import copy
import json
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ultralytics.data.base as B  # noqa: E402
from ultralytics.data.base import apply_ir_encoding, apply_ir_augmentation, IR_AUG_STATS  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.data.augment import Format  # noqa: E402
from ultralytics.data.loaders import LoadImagesAndVideos  # noqa: E402
from ultralytics.utils.instance import Instances  # noqa: E402
from ultralytics.utils.patches import imread  # noqa: E402
from ultralytics.cfg import get_cfg, DEFAULT_CFG_DICT  # noqa: E402
from scripts.train import _build_train_kwargs, resolve_data_path  # noqa: E402

import yaml  # noqa: E402
import cv2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath("data/processed/rgbid_split_train")
TRAIN_VIS = os.path.join(DATA_DIR, "images", "train", "visible")
VAL_VIS = os.path.join(DATA_DIR, "images", "val", "visible")
BASE_CFG = "configs/train_rgbid_sepstem_clahe.yaml"
IRAU_CFG = "configs/train_rgbid_sepstem_clahe_iraug.yaml"
R = {"gates": {}}


def resolve_cfg(path):
    tc = yaml.safe_load(open(path, encoding="utf-8")) or {}
    return get_cfg(DEFAULT_CFG_DICT, overrides=_build_train_kwargs(tc, resolve_data_path(tc, "configs/dataset.yaml")))


def gate(name, ok, **info):
    R["gates"][name] = {"pass": bool(ok), **info}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {info}" if info else ""))
    return ok


def data_dict():
    return {"path": DATA_DIR, "train": "images/train/visible", "val": "images/val/visible",
            "nc": 12, "names": {i: str(i) for i in range(12)}}


def build(cfg, mode, vis=None):
    return build_yolo_dataset(cfg, vis or (TRAIN_VIS if mode == "train" else VAL_VIS), batch=8,
                              data=data_dict(), mode=mode, rect=False, stride=32, use_simotm="RGBID",
                              pairs_rgb_ir=["visible", "infrared"], pairs_rgb_depth=["visible", "depth"])


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ============================================================ §6
def gate6(cfg_on):
    print("\n== §6 Exact IR insertion path ==")
    src = open("ultralytics/data/base.py", encoding="utf-8").read()
    tree = ast.parse(src)
    lines = src.splitlines()

    # (a) AST：在 RGBID 分支内，调用顺序必须是 apply_ir_encoding → apply_ir_augmentation → _merge_channels_rgbid
    order, branch_line = [], None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            t = ast.dump(node.test)
            if "RGBID" not in t:
                continue
            # 只看该分支的 **body**：elif 链嵌在 orelse 里，全量 walk 会把后面
            # RGBD/IRD 等分支的调用也算进来。
            calls = [n for b in node.body for n in ast.walk(b) if isinstance(n, ast.Call)]
            names = {(c.func.id if isinstance(c.func, ast.Name) else getattr(c.func, "attr", None)) for c in calls}
            # 只认「就是那个 RGBID 分支」——它必须同时含 apply_ir_augmentation 与 _merge_channels_rgbid
            if "apply_ir_augmentation" not in names or "_merge_channels_rgbid" not in names:
                continue
            branch_line = node.lineno
            for c in sorted(calls, key=lambda x: x.lineno):
                fn = c.func
                nm = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
                if nm in ("apply_ir_encoding", "apply_ir_augmentation", "_merge_channels_rgbid",
                          "_resize_images_3"):
                    order.append((c.lineno, nm))
            break
    seq = [n for _, n in order]
    ast_ok = seq == ["apply_ir_encoding", "apply_ir_augmentation", "_resize_images_3", "_merge_channels_rgbid"]
    R["gate6_ast_order"] = {"branch_line": branch_line, "call_order": order}

    # (b) 运行时：抓 apply_ir_augmentation 的输入/输出，与 _merge_channels_rgbid 收到的 IR 对齐
    d = build(cfg_on, "train")
    sample = sorted(os.listdir(TRAIN_VIS))[0]
    f = os.path.join(TRAIN_VIS, sample)
    rec = {}
    orig_aug = B.apply_ir_augmentation
    orig_merge = B.BaseDataset._merge_channels_rgbid

    def aug_hook(im, cfg, augment, stats=None):
        rec.setdefault("aug_in", im.copy())
        out = orig_aug(im, cfg, augment, stats)
        rec["aug_out"] = out.copy()
        return out

    def merge_hook(self, vis, ir, dep):
        rec["merge_ir"] = ir.copy()
        return orig_merge(self, vis, ir, dep)

    B.apply_ir_augmentation = aug_hook
    B.BaseDataset._merge_channels_rgbid = merge_hook
    try:
        ds = object.__new__(B.BaseDataset)
        ds.use_simotm, ds.imgsz, ds.augment, ds.hyp = "RGBID", 1280, True, cfg_on
        seed_all(3)
        ds.load_and_preprocess_image(f, use_simotm="RGBID", pairs_rgb="visible",
                                     pairs_ir="infrared", pairs_depth="depth")
    finally:
        B.apply_ir_augmentation = orig_aug
        B.BaseDataset._merge_channels_rgbid = orig_merge

    raw_ir = imread(f.replace("visible", "infrared"), cv2.IMREAD_GRAYSCALE)
    expected_clahe = apply_ir_encoding(raw_ir, "clahe")
    in_is_clahe = np.array_equal(rec["aug_in"], expected_clahe)
    out_is_merged = np.array_equal(rec["aug_out"], rec["merge_ir"])
    in_is_raw = np.array_equal(rec["aug_in"], raw_ir)
    R["gate6_runtime"] = {"aug_input_equals_CLAHE_output": bool(in_is_clahe),
                          "aug_input_equals_RAW_ir": bool(in_is_raw),
                          "aug_output_reaches_merge": bool(out_is_merged)}
    print(f"     AST call order in RGBID branch: {seq}")
    print(f"     runtime: aug input == CLAHE(raw) : {in_is_clahe}   (== raw IR: {in_is_raw})")
    print(f"     runtime: aug output == merged IR channel : {out_is_merged}")
    return gate("§6_exact_insertion_path", ast_ok and in_is_clahe and out_is_merged
                and not in_is_raw, ast_order=seq)


# ============================================================ §7
def gate7(cfg_on):
    print("\n== §7 Channel identity ==")
    files = sorted(os.listdir(TRAIN_VIS))[:12]
    ds = object.__new__(B.BaseDataset)
    ds.use_simotm, ds.imgsz, ds.augment, ds.hyp = "RGBID", 1280, True, cfg_on
    ok_ch = {"vis_B": [], "vis_G": [], "vis_R": [], "IR": [], "Depth": []}
    prop = {"rgb_unchanged_active_aug": [], "depth_unchanged_active_aug": [], "ir_differs": []}

    cfg_off = resolve_cfg(BASE_CFG)
    # A/B 必须在**必然触发**的条件下做，否则 p=0.35 的抽样miss会被误读成"没有变化"
    cfg_always = get_cfg(DEFAULT_CFG_DICT, overrides={
        "use_simotm": "RGBID", "channels": 5, "ir_encoding": "clahe",
        "ir_gamma": [0.75, 1.35], "ir_gamma_probability": 1.0,
        "ir_noise_std": 0.015, "ir_noise_probability": 1.0})
    for fname in files:
        f = os.path.join(TRAIN_VIS, fname)
        vis = imread(f)                                   # BGR (H,W,3)
        raw_ir = imread(f.replace("visible", "infrared"), cv2.IMREAD_GRAYSCALE)
        raw_dep = imread(f.replace("visible", "depth"), cv2.IMREAD_UNCHANGED)
        dep = raw_dep[..., 0] if raw_dep.ndim == 3 else raw_dep
        if dep.dtype != np.uint8:
            dep = dep.astype(np.float32); dep[dep < 300] = 0.0
            dep = np.clip(dep / 19999.0 * 255.0, 0, 255).astype(np.uint8)
        exp_ir = apply_ir_encoding(raw_ir, "clahe")

        kw = dict(use_simotm="RGBID", pairs_rgb="visible", pairs_ir="infrared", pairs_depth="depth")
        ds.hyp = cfg_always
        seed_all(11); m_on = ds.load_and_preprocess_image(f, **kw)     # aug 必然触发
        ds.hyp = cfg_off
        seed_all(11); m_off = ds.load_and_preprocess_image(f, **kw)    # D′（无 aug）
        ds.hyp = cfg_on

        ok_ch["vis_B"].append(np.array_equal(m_on[:, :, 0], vis[:, :, 0]))
        ok_ch["vis_G"].append(np.array_equal(m_on[:, :, 1], vis[:, :, 1]))
        ok_ch["vis_R"].append(np.array_equal(m_on[:, :, 2], vis[:, :, 2]))
        ok_ch["IR"].append(np.array_equal(m_off[:, :, 3], exp_ir))
        ok_ch["Depth"].append(np.array_equal(m_on[:, :, 4], dep))
        prop["rgb_unchanged_active_aug"].append(all(np.array_equal(m_on[:, :, i], m_off[:, :, i]) for i in (0, 1, 2)))
        prop["depth_unchanged_active_aug"].append(np.array_equal(m_on[:, :, 4], m_off[:, :, 4]))
        prop["ir_differs"].append(not np.array_equal(m_on[:, :, 3], m_off[:, :, 3]))

    ident = {k: bool(all(v)) for k, v in ok_ch.items()}
    props = {k: bool(all(v)) for k, v in prop.items()}
    print(f"     merged 5ch   ch0==visible B : {ident['vis_B']}   ch1==G: {ident['vis_G']}   "
          f"ch2==R: {ident['vis_R']}   ch3==CLAHE(IR): {ident['IR']}   ch4==Depth: {ident['Depth']}")
    print(f"     A/B runtime: RGB unchanged={props['rgb_unchanged_active_aug']} "
          f"Depth unchanged={props['depth_unchanged_active_aug']} IR differs={props['ir_differs']}")

    # Format 之后的张量顺序（合成探针，直接读通道值）
    probe = np.zeros((8, 8, 5), np.uint8)
    probe[..., 0], probe[..., 1], probe[..., 2], probe[..., 3], probe[..., 4] = 10, 20, 30, 40, 50
    t = Format(bbox_format="xywh", normalize=True, return_mask=False, return_keypoint=False,
               return_obb=False, batch_idx=True)(
        {"img": probe, "cls": np.zeros(0),
         "instances": Instances(bboxes=np.zeros((0, 4), np.float32), segments=np.zeros((0, 0, 2), np.float32))})
    vals = [int(t["img"][c].max()) for c in range(5)]
    names = {10: "B", 20: "G", 30: "R", 40: "IR", 50: "D"}
    order = [names.get(v, f"?{v}") for v in vals]
    print(f"     final 5ch TENSOR order (Format) : {order}   (IR idx={order.index('IR')}, D idx={order.index('D')})")
    R["gate7_tensor_order"] = order
    R["gate7_identity"] = ident
    R["gate7_ab"] = props
    return gate("§7_channel_identity", all(ident.values()) and all(props.values())
                and order[3] == "IR" and order[4] == "D"
                and sorted(order[:3]) == ["B", "G", "R"],
                identity=ident, tensor_order=order)


# ============================================================ §8
def gate8(cfg_on):
    print("\n== §8 No double augmentation ==")
    call_sites, hits = [], {}
    for root, _, fs in os.walk("."):
        if any(p in root for p in (".git", "__pycache__", "_before", "runs", "logs")):
            continue
        for fn in fs:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            try:
                s = open(p, encoding="utf-8").read()
            except Exception:
                continue
            for m in re.finditer(r"^\s*[^#\n]*\b(apply_ir_augmentation|ir_gamma|ir_noise_std)\b", s, re.M):
                hits.setdefault(p, []).append(m.group(0).strip())
            # 排除注释行
    exec_sites = {}
    for p, h in hits.items():
        s = open(p, encoding="utf-8").read()
        # 只统计"真的会执行"的行：函数定义 / 调用 / kwargs 赋值
        real = [ln for ln in s.splitlines()
                if re.search(r"(apply_ir_augmentation\s*\(|def apply_ir_augmentation|kwargs\[.ir_(gamma|noise)|ir_gamma\s*:|ir_noise_std\s*:)", ln)
                and not ln.strip().startswith("#")]
        if real:
            exec_sites[p] = real
    print("     executable IR-aug sites:")
    for p, v in exec_sites.items():
        print(f"       {p}: {len(v)}")
        for ln in v:
            print(f"          {ln.strip()}")

    n_base_calls = sum(1 for ln in exec_sites.get(os.path.join(".", "ultralytics", "data", "base.py").replace("\\", "/"), [])
                       if "apply_ir_augmentation(" in ln and "def " not in ln)
    base_key = [k for k in exec_sites if k.replace("\\", "/").endswith("ultralytics/data/base.py")]
    n_base_calls = sum(1 for ln in (exec_sites[base_key[0]] if base_key else []) if "apply_ir_augmentation(" in ln and "def" not in ln)
    loaders_hit = any("loaders.py" in k.replace("\\", "/") for k in exec_sites)

    # Albumentations 在 5ch 下的状态
    d = build(cfg_on, "train")
    alb = [t for t in d.transforms.transforms if type(t).__name__.startswith("Albumentations")]
    alb_state = [{"p": getattr(a, "p", None), "transform_is_none": getattr(a, "transform", "?") is None} for a in alb]

    # 调用率：每张源图恰好一次
    cnt = {"calls": 0}
    orig = B.apply_ir_augmentation

    def hook(*a, **k):
        cnt["calls"] += 1
        return orig(*a, **k)

    B.apply_ir_augmentation = hook
    try:
        d2 = build(cfg_on, "train")
        seed_all(1)
        for i in range(60):
            _ = d2[i]
    finally:
        B.apply_ir_augmentation = orig
    ratio = cnt["calls"] / 60.0

    print(f"     executable call sites for apply_ir_augmentation: {n_base_calls} (must be 1)")
    print(f"     loaders.py (推理) 含 IR-aug 可执行点: {loaders_hit} (must be False)")
    print(f"     Albumentations in 5ch compose: {alb_state}")
    print(f"     apply_ir_augmentation calls / training sample = {ratio:.3f} (must be <= 1.0)")
    R["gate8"] = {"n_call_sites": n_base_calls, "loaders_hit": loaders_hit,
                  "albumentations": alb_state, "calls_per_sample": ratio}
    return gate("§8_no_double_augmentation", n_base_calls == 1 and not loaders_hit
                and all(s["p"] == 0 for s in alb_state) and ratio <= 1.0 + 1e-9,
                call_sites=n_base_calls, albumentations=alb_state, calls_per_sample=round(ratio, 4))


# ============================================================ §9
def gate9(cfg_on, cfg_off):
    print("\n== §9 Train-only isolation ==")
    # VAL
    v_on, v_off = build(cfg_on, "val"), build(cfg_off, "val")
    seed_all(21); a = v_on[0]["img"].numpy().copy()
    seed_all(21); b = v_off[0]["img"].numpy().copy()
    val_ident = np.array_equal(a, b)
    # PREDICT (LoadImagesAndVideos = 提交/推理链)
    f = os.path.join(TRAIN_VIS, sorted(os.listdir(TRAIN_VIS))[0])
    preds = {}
    for tag, c in (("on", cfg_on), ("off", cfg_off)):
        ld = LoadImagesAndVideos(f, batch=1, use_simotm="RGBID", imgsz=1280,
                                 pairs_rgb_ir=["visible", "infrared"],
                                 pairs_rgb_depth=["visible", "depth"], ir_encoding=str(c.ir_encoding))
        _paths, imgs, _info = next(iter(ld))
        preds[tag] = np.asarray(imgs[0] if isinstance(imgs, (list, tuple)) else imgs).copy()
    pred_ident = np.array_equal(preds["on"], preds["off"])
    # TRAIN 必须**会**变（否则 augmentation 没生效）
    ds_on = build(cfg_on, "train")
    IR_AUG_STATS["calls"] = 0
    seed_all(31)
    for i in range(20):
        _ = ds_on[i]
    train_fires = IR_AUG_STATS["calls"] > 0

    print(f"     VAL      on == off : {val_ident}  (max_abs_diff={int(np.abs(a.astype(int)-b.astype(int)).max())})")
    print(f"     PREDICT  on == off : {pred_ident}  shape={preds['on'].shape} dtype={preds['on'].dtype}")
    print(f"     TRAIN    augmentation fires : {train_fires} (calls={IR_AUG_STATS['calls']})")
    R["gate9"] = {"val_identical": bool(val_ident), "predict_identical": bool(pred_ident),
                  "train_fires": bool(train_fires)}
    return gate("§9_train_only_isolation", val_ident and pred_ident and train_fires,
                val_identical=bool(val_ident), predict_identical=bool(pred_ident), train_fires=bool(train_fires))


# ============================================================ §10 + §11
def gate10_11(cfg_on, cfg_off):
    print("\n== §10 identity / determinism ==")
    ds = object.__new__(B.BaseDataset)
    ds.use_simotm, ds.imgsz, ds.augment = "RGBID", 1280, True
    f = os.path.join(TRAIN_VIS, sorted(os.listdir(TRAIN_VIS))[0])
    kw = dict(use_simotm="RGBID", pairs_rgb="visible", pairs_ir="infrared", pairs_depth="depth")

    ds.hyp = cfg_off
    seed_all(5); off_a = ds.load_and_preprocess_image(f, **kw)
    # ⚠ 必须用**副本**，否则会把共享的 cfg_on 改成别的实验配置，污染后续 gate
    c_ab = copy.deepcopy(cfg_on)
    ds.hyp = c_ab
    c_ab.ir_gamma_probability = 0.0; c_ab.ir_noise_probability = 0.0
    seed_all(5); A_on_p0 = ds.load_and_preprocess_image(f, **kw)
    testA = np.array_equal(off_a, A_on_p0)

    c_ab.ir_gamma_probability = 1.0; c_ab.ir_noise_probability = 0.0
    c_ab.ir_gamma = [1.0, 1.0]
    seed_all(5); B_on_id = ds.load_and_preprocess_image(f, **kw)
    testB = np.array_equal(off_a, B_on_id)

    ds.hyp = cfg_on
    seed_all(7); c1 = ds.load_and_preprocess_image(f, **kw)
    seed_all(7); c2 = ds.load_and_preprocess_image(f, **kw)
    testC = np.array_equal(c1, c2)

    c_d = copy.deepcopy(cfg_on)
    c_d.ir_gamma_probability = 1.0; c_d.ir_noise_probability = 1.0
    ds.hyp = c_d
    seed_all(7); d1 = ds.load_and_preprocess_image(f, **kw)
    seed_all(8); d2 = ds.load_and_preprocess_image(f, **kw)
    ds.hyp = cfg_on
    testD_changed = not np.array_equal(d1, d2)
    testD_valid = (d1.shape == (360, 640, 5) and d2.shape == (360, 640, 5)
                   and d1.dtype == np.uint8 and d2.dtype == np.uint8
                   and np.isfinite(d1).all() and np.isfinite(d2).all())
    print(f"     Test A (p_gamma=0,p_noise=0 → 与 D′ 逐位相同): {testA}  "
          f"max_abs_diff={int(np.abs(off_a.astype(int)-A_on_p0.astype(int)).max())}")
    print(f"     Test B (gamma=1,sigma=0 → 恒等)              : {testB}  "
          f"max_abs_diff={int(np.abs(off_a.astype(int)-B_on_id.astype(int)).max())}")
    print(f"     Test C (同 seed → 同输出)                    : {testC}")
    print(f"     Test D (异 seed → 应变化, 且 shape/dtype/finite 合法): changed={testD_changed} valid={testD_valid}")

    print("\n== §11 numerical safety ==")
    # 全量数值扫描：真实 IR × gamma/noise 网格
    bad = {"nan": 0, "inf": 0, "dtype": 0, "shape": 0}
    n_checked = 0
    files = sorted(os.listdir(TRAIN_VIS))[:15]
    for fname in files:
        base = apply_ir_encoding(imread(os.path.join(TRAIN_VIS, fname).replace("visible", "infrared"),
                                        cv2.IMREAD_GRAYSCALE), "clahe")
        for g in (0.75, 1.0, 1.35):
            for s in (0.0, 0.015):
                cfg = get_cfg(DEFAULT_CFG_DICT, overrides={
                    "ir_gamma": [g, g], "ir_gamma_probability": 1.0,
                    "ir_noise_std": s, "ir_noise_probability": 1.0})
                seed_all(13)
                out = apply_ir_augmentation(base, cfg, True)
                n_checked += 1
                if not np.isfinite(out).all():
                    bad["nan"] += int(np.isnan(out).any()); bad["inf"] += int(np.isinf(out).any())
                if out.dtype != np.uint8:
                    bad["dtype"] += 1
                if out.shape != base.shape:
                    bad["shape"] += 1
    # noise 单位锁定：σ=0.015 必须等价于 3.825 灰度级，而不是 0.015 灰度级
    probe = np.full((64, 64), 128, np.uint8)
    cfg_s = get_cfg(DEFAULT_CFG_DICT, overrides={"ir_gamma": [1.0, 1.0], "ir_gamma_probability": 0.0,
                                                 "ir_noise_std": 0.015, "ir_noise_probability": 1.0})
    seed_all(17)
    out_s = apply_ir_augmentation(probe, cfg_s, True).astype(np.float64)
    sigma_meas = float(out_s.std())
    unit_ok = abs(sigma_meas - 3.825) < 0.2     # 3.825 = 0.015*255（[0,1] 域）
    unit_not_255 = abs(sigma_meas - 0.015) > 0.1
    print(f"     grid {n_checked} cases  NaN={bad['nan']} Inf={bad['inf']} bad_dtype={bad['dtype']} shape_mismatch={bad['shape']}")
    print(f"     noise unit check: measured sigma = {sigma_meas:.4f} grays "
          f"(expect ~3.825 = 0.015*255 ⇒ [0,1] 域；若为 0.015 则说明被当成 [0,255] 域)")

    ok10 = testA and testB and testC and testD_changed and testD_valid
    ok11 = all(v == 0 for v in bad.values()) and unit_ok and unit_not_255
    gate("§10_identity_determinism", ok10, A=testA, B=testB, C=testC,
         D_changed=testD_changed, D_valid=testD_valid)
    gate("§11_numerical_safety", ok11, **bad, sigma_grays=round(sigma_meas, 4),
         noise_domain="normalized [0,1]")
    return ok10 and ok11


# ============================================================ §12
def gate12(cfg_on):
    print("\n== §12 Probability runtime ==")
    N = 400
    IR_AUG_STATS.update({"calls": 0, "gamma_applied": 0, "noise_applied": 0,
                         "skipped_disabled": 0, "skipped_nonaugment": 0, "gamma_draws": [], "noise_std_used": []})
    d = build(cfg_on, "train")
    # 逐调用观察 delta（both_fired 只能这样测；不改被审计的源码）
    both = {"n": 0}
    orig = B.apply_ir_augmentation

    def hook(im, cfg, augment, stats=None):
        g0, n0 = IR_AUG_STATS["gamma_applied"], IR_AUG_STATS["noise_applied"]
        out = orig(im, cfg, augment, stats)
        if (IR_AUG_STATS["gamma_applied"] - g0) and (IR_AUG_STATS["noise_applied"] - n0):
            both["n"] += 1
        return out

    B.apply_ir_augmentation = hook
    try:
        seed_all(2026)
        for i in range(N):
            _ = d[i]
    finally:
        B.apply_ir_augmentation = orig
    g, n = IR_AUG_STATS["gamma_applied"], IR_AUG_STATS["noise_applied"]
    calls = IR_AUG_STATS["calls"]
    gr, nr = g / calls, n / calls
    # 二项 99.9% 容差
    tol_g = 3.3 * (0.35 * 0.65 / calls) ** 0.5
    tol_n = 3.3 * (0.20 * 0.80 / calls) ** 0.5
    ok = abs(gr - 0.35) <= max(tol_g, 0.05) and abs(nr - 0.20) <= max(tol_n, 0.05)
    exp_both = calls * 0.35 * 0.20
    print(f"     N(samples)={N}  preprocess_calls={calls}  gamma_fired={g} ({gr:.4f}, nominal 0.35, tol ±{tol_g:.3f})")
    print(f"                          noise_fired={n} ({nr:.4f}, nominal 0.20, tol ±{tol_n:.3f})")
    print(f"                          both_fired={both['n']} (independence ⇒ expect ≈ {exp_both:.1f})")
    gd = np.array(IR_AUG_STATS["gamma_draws"])
    print(f"     gamma draws: mean={gd.mean():.4f} min={gd.min():.4f} max={gd.max():.4f} (must ⊂ [0.75,1.35])")
    R["gate12"] = {"N": N, "preprocess_calls": calls, "gamma_fired": g, "gamma_rate": gr,
                   "noise_fired": n, "noise_rate": nr, "both_fired": both["n"],
                   "both_expected_under_independence": round(exp_both, 2),
                   "gamma_draw_min": float(gd.min()), "gamma_draw_max": float(gd.max())}
    return gate("§12_probability_runtime", ok and gd.min() >= 0.75 and gd.max() <= 1.35
                and abs(calls / N - 1.0) < 1e-9, **R["gate12"])


def main():
    cfg_on, cfg_off = resolve_cfg(IRAU_CFG), resolve_cfg(BASE_CFG)
    ok = True
    ok &= gate6(cfg_on)
    ok &= gate7(cfg_on)
    ok &= gate8(cfg_on)
    ok &= gate9(cfg_on, cfg_off)
    ok &= gate10_11(cfg_on, cfg_off)
    ok &= gate12(cfg_on)
    npass = sum(1 for v in R["gates"].values() if v["pass"])
    print(f"\n==== §6..§12: {npass}/{len(R['gates'])} PASS ====")
    with open(os.path.join(HERE, "launch_gate.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
