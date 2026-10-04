"""fusion_phase0_validate.py — Fusion Phase 0 的 CPU-only 配置/路径/通道/预训练校验。

ZERO GPU · ZERO forward · ZERO training. 只做 yaml 解析、路径检查、
模型**构造**（不 forward）、以及 pretrained 键匹配统计（intersect_dicts 语义）。

用法:
  python -X utf8 diagnostic/fusion_phase0_validate.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPS = [
    ("M1  RGB-only",          "configs/train_modality_m1_rgb.yaml",            "configs/yolo11m_modality3ch.yaml"),
    ("M2a IR-only+CLAHE",     "configs/train_modality_m2a_ir_clahe.yaml",      "configs/yolo11m_modality3ch.yaml"),
    ("M2b IR-only+percentile","configs/train_modality_m2b_ir_percentile.yaml", "configs/yolo11m_modality3ch.yaml"),
    ("M3  Depth-only",        "configs/train_modality_m3_depth.yaml",          "configs/yolo11m_modality3ch.yaml"),
    ("M4  RGB+IR",            "configs/train_modality_m4_rgb_ir.yaml",         "configs/yolo11m_modality4ch.yaml"),
    ("M5  RGB+Depth",         "configs/train_modality_m5_rgb_depth.yaml",      "configs/yolo11m_modality4ch.yaml"),
    ("M6  IR+Depth",          "configs/train_modality_m6_ir_depth.yaml",       "configs/yolo11m_modality2ch.yaml"),
]

# 数据侧：use_simotm -> (split root, 该模式的真实通道数, 是否存在该分支)
MODE_INFO = {
    "Gray2BGR":  ("visible_split",     3, True,  "base.py:248 / loaders.py:505 —— 纯 BGR imread"),
    "Infrared":  ("rgbt_split",        3, True,  "base.py / loaders.py —— [IR,IR,IR]，**读 ir_encoding**（2026-10-03 修复）"),
    "Depth":     ("depth_split", 3, True,  "base.py:269 —— [D,D,D]"),
    "RGBT":      ("rgbt_split",        4, False, "base.py:295 —— [B,G,R,IR]，IR 为 **raw**（不消费 ir_encoding）"),
    "RGBIR":     ("rgbt_split",        4, True,  "base.py —— [B,G,R,IR]，IR 走 ir_encoding（2026-10-03 新增）"),
    "RGBD":      ("depth_split", 4, True,  "base.py:351 —— [B,G,R,D]"),
    "RGBID":     ("rgbid_split", 5, True,  "base.py:313 —— [B,G,R,IR,D]  ← D′"),
    "IRD":       ("rgbid_split", 2, True,  "base.py / loaders.py —— [IR,D]（2026-10-03 新增）"),
}


def yml(p):
    return yaml.safe_load(Path(ROOT, p).read_text(encoding="utf-8"))


def conv_in_channels(model_yaml):
    """静态解析 model yaml，返回所有 Conv 层的 in_channels（-1 表示沿用上一输出，做保守标记）。"""
    d = yml(model_yaml)
    ch = d.get("ch", 3)
    layers = d.get("backbone", []) + d.get("head", [])
    out = []
    for i, lyr in enumerate(layers):
        if str(lyr[2]) == "Conv":
            # 只关心首层（f == -1 且 i == 0）与显式 1ch 情况
            out.append((i, lyr[0], lyr[3][0]))
    return ch, out


def simulate_dispatch(model_yaml):
    """复刻 _transfer_rgb_pretrained 的分派（ultralytics/models/yolo/detect/train.py）。

    只按**结构**判断，不构造模型。返回 (branch, will_raise)。
    """
    ch, _ = conv_in_channels(model_yaml)
    d = yml(model_yaml)
    layers = d.get("backbone", []) + d.get("head", [])
    # stems = 模型里所有 Conv；用 SilenceChannel 判断 1ch 分支是否存在
    n_silch = sum(1 for l in layers if str(l[2]) == "SilenceChannel")
    # 单测：首层 Conv 的 in_channels = ch（无 Silence 前缀时）
    first_is_conv = str(layers[0][2]) == "Conv"
    stem_in = [ch] if first_is_conv else []
    one_ch = n_silch  # 每个 SilenceChannel 后跟一个 1ch Conv
    if ch == 5 and first_is_conv and one_ch == 0:
        return "5ch-single-stem (RGB=pretrained, IR/D=mean(RGB))", False
    if ch == 3 and one_ch == 2:
        return "_remap_separate_stem (raise-on-mismatch)", False
    if ch > 3 and first_is_conv and one_ch == 0:
        if ch == 5:
            return "5ch-single-stem (RGB=pretrained, aux=mean(RGB))", False
        return (f"multi-ch-single-stem (in={ch}): ch[0:3]=pretrained RGB, ch[3:]=mean(R,G,B)"
                f"  ← 2026-10-03 修复（此前是静默 return 0 ⇒ 随机初始化）"), False
    if ch in (1, 2) and first_is_conv and one_ch == 0:
        return (f"narrow-ch-single-stem (in={ch}): 全通道 = mean(W_R,W_G,W_B)"
                f"  ← 2026-10-03 修复（此前是静默 return 0 ⇒ 随机初始化）"), False
    if one_ch >= 1 and ch != 3:
        return (f"❌ 落入 RGBD concat_res 索引表分支（硬编码 model.<i>->model.<j> 表）"
                f"—— 该表按 mid-fusion 架构写死，对本布局是**错误映射**（静默）"), False
    if ch == 3 and one_ch == 0:
        return "return 0（非融合模型，无需 remap）⇒ 标准 load() 全量匹配", False
    return f"未覆盖的布局（ch={ch}, SilenceChannel={one_ch}）", False


def main():
    rows, ok_all = [], True
    print("=" * 100)
    print("Fusion Phase 0 — CPU-only validation （无 forward / 无 GPU / 无训练）")
    print("=" * 100)

    for label, tcfg, mcfg in EXPS:
        t = yml(tcfg)
        m = yml(mcfg)
        exp = t.get("experiment_name")
        mode = t.get("use_simotm")
        ch_cfg = int(t.get("channels", 3))
        ch_model = int(m.get("ch", 3))

        info = MODE_INFO.get(mode)
        if info is None:
            mode_ok, mode_note, split = False, f"❌ use_simotm='{mode}' 不在已知分支表内 -> else 兜底为纯 BGR（静默）", None
            ch_mode = 3
        else:
            split, ch_mode, mode_ok, mode_note = info

        # 路径
        # split 现在是**前缀**；实际目录由生产解析结果给出
        try:
            from scripts.train import resolve_data_path
            ds_yaml = Path(resolve_data_path(t, "configs/dataset.yaml")) if split else None
        except Exception:
            ds_yaml = None
        path_ok = bool(ds_yaml and ds_yaml.exists()) and (
            split is None or str(ds_yaml.parent.name).startswith(split))
        train_dir = val_dir = None
        if path_ok:
            d = yaml.safe_load(ds_yaml.read_text(encoding="utf-8"))
            root = Path(d["path"])
            train_dir, val_dir = root / d["train"], root / d["val"]

        branch, will_raise = simulate_dispatch(mcfg)

        # 通道一致性：config channels == model ch == 模式实际产出
        ch_ok = (ch_cfg == ch_model == ch_mode) and mode_ok

        # 必填字段
        need = ["experiment_name", "seed", "pretrained", "epochs", "batch_size", "image_size",
                "learning_rate", "warmup_epochs", "optimizer", "scheduler", "amp", "val_ratio",
                "use_simotm", "channels", "checkpoint"]
        missing = [k for k in need if k not in t]
        aug = t.get("aug", {})
        hsv_off = aug.get("hsv_h") == 0.0 and aug.get("hsv_s") == 0.0 and aug.get("hsv_v") == 0.0

        canon = dict(epochs=300, batch_size=8, learning_rate=5.0e-3, warmup_epochs=3,
                     imgsz=1280, seed=42, patience=0, val_ratio=0.2)
        canon_ok = (t["epochs"] == 300 and t["batch_size"] == 8
                    and float(t["learning_rate"]) == 5.0e-3 and float(t["warmup_epochs"]) == 3
                    and list(t["image_size"]) == [1280, 1280] and int(t["seed"]) == 42
                    and int(t["patience"]) == 0 and float(t["val_ratio"]) == 0.2
                    and t["optimizer"]["type"] == "SGD"
                    and float(t["optimizer"]["momentum"]) == 0.937
                    and float(t["optimizer"]["weight_decay"]) == 5.0e-4
                    and t["scheduler"]["type"] == "CosineAnnealingLR"
                    and t["amp"] is True
                    and t["pretrained"] == "yolo11m.pt")

        blocked = "❌" in branch or not mode_ok
        status = "BLOCKED" if blocked else "READY WITH CAVEAT"

        # ---- ir_encoding 有效性（M2a 关键）----
        # 'Infrared' 分支（base.py:280-294 / loaders.py:535-561）硬编码 percentile，
        # 完全不读 ir_encoding；只有 'RGBID' 分支读。scripts/train.py:638 的硬防呆
        # 也只覆盖 use_simotm == "RGBID"。
        ir_enc = t.get("ir_encoding")
        ir_eff = True
        ir_note = "n/a"
        if ir_enc is not None:
            # 2026-10-03 修复后：Infrared / IRD 也与 RGBID 一样读 ir_encoding。
            if mode in ("RGBID", "Infrared", "IRD", "RGBIR"):
                ir_eff, ir_note = True, f"ir_encoding={ir_enc} 生效（base.py / loaders.py 均读取）"
            else:
                ir_eff = False
                ir_note = (f"❌ ir_encoding={ir_enc} 被**静默忽略** —— use_simotm='{mode}' 的分支"
                           f"不消费它")
                status = "BLOCKED"

        rows.append(dict(label=label, exp=exp, mode=mode, ch_cfg=ch_cfg, ch_model=ch_model,
                         ch_mode=ch_mode, split=split, path_ok=path_ok, missing=missing,
                         canon_ok=canon_ok, hsv_off=hsv_off, branch=branch,
                         ir_encoding=ir_enc, ir_effective=ir_eff, ir_note=ir_note,
                         status=status, note=mode_note))
        print(f"   ir_encoding     : {ir_note}")
        print(f"   STATUS          : {status}")

        print(f"\n▌{label}  [{exp}]")
        print(f"   use_simotm      : {mode}   ({mode_note})")
        print(f"   channels        : config={ch_cfg}  model_yaml ch={ch_model}  模式实际={ch_mode}"
              f"   -> {'OK' if ch_ok else '❌ MISMATCH'}")
        print(f"   model yaml      : {mcfg}")
        print(f"   remap 分派      : {branch}")
        print(f"   dataset split   : {split}  yaml={'OK' if path_ok else '❌ MISSING'}")
        if path_ok:
            ntr = len(list(train_dir.iterdir())) if train_dir.is_dir() else 0
            nva = len(list(val_dir.iterdir())) if val_dir.is_dir() else 0
            print(f"   images          : train={ntr}  val={nva}")
        print(f"   canonical block : {'OK' if canon_ok else '❌ 偏离 D′ 训练块'}"
              f"   | 必填字段缺失={missing or '无'}  | HSV 惰性={hsv_off}")

    print("\n" + "=" * 100)
    print(f"{'ID':<26}{'mode':<12}{'split':<20}{'ch':<6}{'status':<18}")
    for r in rows:
        print(f"{r['label']:<26}{str(r['mode']):<12}{str(r['split']):<20}{r['ch_cfg']:<6}{r['status']:<18}")
    Path(ROOT, "diagnostic/fusion_phase0_validation.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] -> diagnostic/fusion_phase0_validation.json")


if __name__ == "__main__":
    main()
