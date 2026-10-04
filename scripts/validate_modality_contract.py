"""scripts/validate_modality_contract.py — Modality 实验契约门（Experiment Contract Gate）。

目的：把「这个 modality 配置能不能与 D′ 公平比较」变成一个**会 raise 的断言**，
而不是靠人看日志。任何关键字段不一致 ⇒ `AssertionError`，不是 warning。

D′（`configs/train_rgbid_sepstem_clahe.yaml` + `configs/yolo11m_sepstem.yaml`）是唯一 reference。

用法：
    python -X utf8 scripts/validate_modality_contract.py --all
    python -X utf8 scripts/validate_modality_contract.py --id M1
    python -X utf8 scripts/validate_modality_contract.py --id M1 --selftest-fault channels

ZERO GPU · ZERO forward · ZERO training（只做 yaml 解析 + 模型**构造**）。
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ============================================================
# Contract —— 全部来自 D′（reference），逐项硬编码，不得"就近取默认值"
# ============================================================
REFERENCE = {
    "name": "D′ (SepStem + IR-CLAHE)",
    "train_config": "configs/train_rgbid_sepstem_clahe.yaml",
    "model_config": "configs/yolo11m_sepstem.yaml",
    "epochs": 300,
    "imgsz": 1280,
    "batch_size": 8,
    "seed": 42,
    "patience": 0,
    "val_ratio": 0.2,
    "optimizer": "SGD",
    "lr0": 5.0e-3,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 5.0e-4,
    "warmup_epochs": 3,
    "pretrained": "yolo11m.pt",
    "scale": "m",
    "amp": True,
    "scheduler": "CosineAnnealingLR",
    "evaluator": "official",       # scripts/predict_rect.py -> scripts/official_eval.py
}

# use_simotm -> (产出通道数, 通道序, split 根, 该模式是否消费 ir_encoding)
MODE_SPEC = {
    # split 字段是**目录名前缀**（生产写 *_split_{src_key}，如 rgbt_split_train），
    # 不再写死完整名 —— 2026-10-03：此前写 "rgbt_split" 指向一个陈旧目录。
    "Gray2BGR":  (3, "[B,G,R]",          "visible_split",      False),
    "Infrared":  (3, "[IR,IR,IR]",       "rgbt_split",         True),
    "Depth":     (3, "[D,D,D]",          "depth_split",        False),
    "RGBT":      (4, "[B,G,R,IR]",       "rgbt_split",         False),  # 既有模式：IR = raw，不消费 ir_encoding
    "RGBIR":     (4, "[B,G,R,IR]",       "rgbt_split",         True),   # 2026-10-03 新增：IR 走 ir_encoding
    "RGBD":      (4, "[B,G,R,D]",        "depth_split",        False),
    "IRD":       (2, "[IR,D]",           "rgbid_split",        True),
    "RGBID":     (5, "[B,G,R,IR,D]",     "rgbid_split",        True),   # D′
}

# 实验矩阵（M2a 用 Infrared + clahe —— 修复后必须真正生效）
MATRIX = {
    "M1":  ("configs/train_modality_m1_rgb.yaml",            "configs/yolo11m_modality3ch.yaml"),
    "M2a": ("configs/train_modality_m2a_ir_clahe.yaml",      "configs/yolo11m_modality3ch.yaml"),
    "M2b": ("configs/train_modality_m2b_ir_percentile.yaml", "configs/yolo11m_modality3ch.yaml"),
    "M3":  ("configs/train_modality_m3_depth.yaml",          "configs/yolo11m_modality3ch.yaml"),
    "M4":  ("configs/train_modality_m4_rgb_ir.yaml",         "configs/yolo11m_modality4ch.yaml"),
    "M5":  ("configs/train_modality_m5_rgb_depth.yaml",      "configs/yolo11m_modality4ch.yaml"),
    "M6":  ("configs/train_modality_m6_ir_depth.yaml",       "configs/yolo11m_modality2ch.yaml"),
    "M7":  ("configs/train_rgbid_sepstem_clahe.yaml",        "configs/yolo11m_sepstem.yaml"),
}

FAILURES: list[str] = []

# Hard floor for pretrained-transfer coverage (see scripts/pretrained_coverage.py).
# Reference measurements: D 1.0000, D′ 1.0000, M4 1.0000, M1 1.0000.
# The P1 incident (candC routed to the multi_ch branch) measured 0.0019.
PRETRAINED_COVERAGE_MIN = 0.80


def need(cond, field, got, expect, extra=""):
    if not cond:
        msg = f"contract violation: {field}: got={got!r} expected={expect!r} {extra}".strip()
        FAILURES.append(msg)
        raise AssertionError(msg)


def check(mid, train_cfg_path, model_cfg_path, fault=None):
    t = yaml.safe_load((ROOT / train_cfg_path).read_text(encoding="utf-8"))
    m = yaml.safe_load((ROOT / model_cfg_path).read_text(encoding="utf-8"))
    ctx = {"id": mid, "fault": fault}

    # ---------------- 1. canonical training block（逐项 == D′） ----------------
    chk = lambda c, f, g, e: need(c and not (fault == f), f, g, e)
    chk(t["epochs"] == REFERENCE["epochs"], "epochs", t["epochs"], 300)
    chk(t["batch_size"] == REFERENCE["batch_size"], "batch_size", t["batch_size"], 8)
    chk(int(t["seed"]) == REFERENCE["seed"], "seed", t["seed"], 42)
    chk(int(t["patience"]) == REFERENCE["patience"], "patience", t["patience"], 0)
    chk(float(t["val_ratio"]) == REFERENCE["val_ratio"], "val_ratio", t["val_ratio"], 0.2)
    chk(list(t["image_size"]) == [1280, 1280], "imgsz", t["image_size"], [1280, 1280])
    chk(float(t["learning_rate"]) == REFERENCE["lr0"], "lr0", t["learning_rate"], 5.0e-3)
    chk(float(t["warmup_epochs"]) == REFERENCE["warmup_epochs"], "warmup_epochs", t["warmup_epochs"], 3.0)
    chk(t["optimizer"]["type"] == REFERENCE["optimizer"], "optimizer", t["optimizer"]["type"], "SGD")
    chk(float(t["optimizer"]["momentum"]) == REFERENCE["momentum"], "momentum", t["optimizer"]["momentum"], 0.937)
    chk(float(t["optimizer"]["weight_decay"]) == REFERENCE["weight_decay"],
        "weight_decay", t["optimizer"]["weight_decay"], 5.0e-4)
    chk(t["scheduler"]["type"] == REFERENCE["scheduler"], "scheduler", t["scheduler"]["type"], "CosineAnnealingLR")
    chk(t["amp"] is True, "amp", t["amp"], True)
    chk(str(t["pretrained"]) == REFERENCE["pretrained"], "pretrained", t["pretrained"], "yolo11m.pt")

    # ---------------- 2. 模态 / 通道 ----------------
    mode = str(t["use_simotm"])
    need(mode in MODE_SPEC, "use_simotm", mode, f"one of {sorted(MODE_SPEC)}")
    n_mode, order, split_root, consumes_ir = MODE_SPEC[mode]
    ch_cfg = int(t.get("channels", 3))
    ch_model = int(m.get("ch", 3))
    chk(ch_cfg == n_mode, "channels(config) vs mode产出", ch_cfg, n_mode)
    chk(ch_model == n_mode, "ch vs mode产出", ch_model, n_mode)
    chk(ch_cfg == ch_model, "channels(config) vs ch(yaml)", ch_cfg, ch_model)
    ctx.update(mode=mode, order=order, channels=n_mode, split=split_root)

    # ---------------- 3. model scale 显式（不得回退 nano） ----------------
    stem_name = Path(model_cfg_path).stem
    import re
    msc = re.search(r"yolo[v]?\d+([nslmx])", stem_name)
    need(msc is not None, "model yaml 文件名含 scale 字符", stem_name, "yolo11m*（否则 guess_model_scale 返回 '' ⇒ 静默回退 nano）")
    chk(msc.group(1) == REFERENCE["scale"], "scale", msc.group(1), "m")
    need("scales" in m and "m" in m["scales"], "yaml 含 scales.m", sorted(m.get("scales", {})), "n/s/m/l/x")
    need(list(m["scales"]["m"]) == [0.50, 1.00, 512], "scales.m", m["scales"]["m"], [0.50, 1.00, 512])

    # ---------------- 4. augmentation 契约 ----------------
    aug = t.get("aug", {}) or {}
    # RandomHSV 只在 3ch（RandomHSV）与 4ch（RandomHSV4C）真正生效；1/2/5ch 下
    # `RandomHSV.__call__` 因 img.shape[-1] != 3 直接 return ⇒ 无需显式置 0。
    if n_mode in (3, 4):
        chk(aug.get("hsv_h") == 0.0 and aug.get("hsv_s") == 0.0 and aug.get("hsv_v") == 0.0,
            "aug.hsv_* == 0（D′ 为 5ch ⇒ RandomHSV 惰性，需对齐）",
            (aug.get("hsv_h"), aug.get("hsv_s"), aug.get("hsv_v")), (0.0, 0.0, 0.0))
        chk(aug.get("albumentations_p") == 0.0,
            "aug.albumentations_p（D′ 5ch ⇒ Albumentations(p=0)，需对齐）",
            aug.get("albumentations_p"), 0.0)
    else:
        need(int(aug.get("albumentations_p", 0.0) or 0.0) == 0.0,
             "aug.albumentations_p 若声明则必须为 0", aug.get("albumentations_p"), 0.0)

    # ---------------- 5. ir_encoding 必须真正生效 ----------------
    ir_enc = t.get("ir_encoding")
    # 消费 ir_encoding 的模式**必须显式声明**：缺省时 base.py/loaders.py 会静默用 "percentile"，
    # 那正是 2026-09-20 事故的形状（配置名写着 _clahe、键却不在）。合同里不允许它靠默认值。
    if consumes_ir:
        need(ir_enc is not None,
             "ir_encoding 必须显式声明（该模式消费它）",
             ir_enc, "'clahe' | 'percentile' | 'raw'",
             "缺省会静默回落 percentile")
    if ir_enc is not None:
        chk(consumes_ir, "ir_encoding 是否被该模式消费",
            f"mode={mode} declares ir_encoding={ir_enc} but the mode's branch ignores it",
            "mode ∈ {RGBID, Infrared, IRD} 才读 ir_encoding")
        need(str(ir_enc).lower() in ("clahe", "percentile", "raw"), "ir_encoding 取值", ir_enc,
             "clahe|percentile|raw")

    # ---------------- 6. split 存在 且 与 D′ 同划分 ----------------
    # ★ 用**生产同一条**路径解析（必要时生成），而不是拼一个目录名
    from scripts.train import resolve_data_path
    resolved = Path(resolve_data_path(t, "configs/dataset.yaml"))
    need(resolved.exists(), "生产的 data_path 存在", str(resolved), "存在")
    need(resolved.parent.name.startswith(split_root),
         "split 目录符合该 mode 的分组", resolved.parent.name, f"前缀 {split_root}*")
    ds = resolved
    d = yaml.safe_load(ds.read_text(encoding="utf-8"))
    train_dir = Path(d["path"]) / d["train"]
    val_dir = Path(d["path"]) / d["val"]
    need(train_dir.is_dir() and val_dir.is_dir(), "split 目录", f"{train_dir} | {val_dir}", "存在")
    n_tr = len(list(train_dir.iterdir())); n_va = len(list(val_dir.iterdir()))
    chk((n_tr, n_va) == (1600, 400), "split 规模", (n_tr, n_va), (1600, 400))
    ctx["n_train"], ctx["n_val"] = n_tr, n_va

    # ---------------- 7. 推理 loader 必须支持该模式（image 路径） ----------------
    import inspect
    from ultralytics.data import loaders as _l
    lines = inspect.getsource(_l).splitlines()
    i_img = next(i for i, l in enumerate(lines) if l.strip() == 'self.mode = "image"')
    img_block = "\n".join(lines[i_img:])
    need(f"== '{mode}'" in img_block or mode == "Gray2BGR",
         "loaders.py image 路径支持该 use_simotm",
         mode, "存在对应分支（否则 fallthrough 到 else 读 visible 原图）")

    # ---------------- 8. evaluator 契约 ----------------
    need((ROOT / "scripts/official_eval.py").exists(), "official evaluator", "scripts/official_eval.py", "存在")
    need((ROOT / "scripts/predict_rect.py").exists(), "predict_rect", "scripts/predict_rect.py", "存在")
    oe = (ROOT / "scripts/official_eval.py").read_text(encoding="utf-8")
    pe = (ROOT / "scripts/predict_rect.py").read_text(encoding="utf-8")
    # 评测链必须用官方 §九(二) 的每图 100 框上限；两侧都要声明 100。
    need("MAX_BOXES_PER_IMAGE = 100" in oe, "official_eval 每图框上限", "未读到 MAX_BOXES_PER_IMAGE = 100", 100)
    need("--max_boxes" in pe and "default=100" in pe, "predict_rect 每图框上限", "未读到 --max_boxes default=100", 100)

    # ---------------- 9. pretrained transfer：fail-closed coverage gate ----------------
    # 2026-10-04 (P1): 原判据 `n_ret > 0 and stem_changed` 可被"只写了 stem 就 return"
    # 的 multi_ch 分支满足，而 body 仍全随机（实测 9.7%）。更糟的是本步此前**没有调用
    # `model.load()`** —— 而生产路径是 load() + remap 两步，单臂测量会低估覆盖率
    # （3ch 模型在 remap 后单独测量只有 0.19%）。现在按生产路径完整复现，并以
    # **逐张量 exact-match 覆盖率** 作为唯一判据。
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import _transfer_rgb_pretrained
    from ultralytics.nn.tasks import attempt_load_one_weight
    from ultralytics.utils.torch_utils import intersect_dicts

    sys.path.insert(0, str(ROOT / "scripts"))
    from pretrained_coverage import measure as _coverage_measure

    _stock, _ = attempt_load_one_weight(str(ROOT / "yolo11m.pt"))  # 与 trainer 同路径（优先 EMA）
    src = _stock.float().state_dict()
    net = YOLO(model_cfg_path).model
    sd = net.state_dict()
    inter = intersect_dicts(src, sd)
    stem = next((mm for mm in net.model if type(mm).__name__ == "Conv"), None)
    stem_before = stem.conv.weight.detach().clone()
    net.load(_stock)  # ★ 生产路径的第一步，此前被漏掉
    n_ret = _transfer_rgb_pretrained(net, _stock)
    stem_changed = not torch.equal(stem_before, stem.conv.weight.detach())
    stem_in = int(stem.conv.in_channels)
    # 单 stem 布局：首个 Conv 就是输入 stem，其 in_channels 必须等于模态通道数。
    # sepstem 布局（D′）：输入由 SilenceChannel 切片给出，首个 Conv 是 3ch 的 RGB stem，
    # 此时用 yaml 的 ch 校验（下一步做的正是这件事）。
    _n_silch = sum(1 for l in (m.get("backbone", []) + m.get("head", [])) if str(l[2]) == "SilenceChannel")
    if _n_silch == 0:
        need(stem_in == n_mode, "stem in_channels", stem_in, n_mode)
    else:
        need(int(m["ch"]) == n_mode, "ch (sepstem 输入通道)", m["ch"], n_mode)
    if n_mode == 3:
        need(n_ret == 0 and len(inter) == len(sd) - 6, "3ch 预训练路径",
             f"remap={n_ret} intersect={len(inter)}/{len(sd)}", "remap=0 且 intersect=len-6")
    need(n_ret > 0 or n_mode == 3, "非 3ch 必须走 remap",
         f"remap={n_ret}", ">0（3ch 走标准 load()）")

    # ---- fail-closed：逐张量 exact-match 覆盖率 ----
    cov = _coverage_measure(net, src, PRETRAINED_COVERAGE_MIN)
    need(
        cov["ok"],
        "pretrained fail-closed coverage gate",
        f"family={cov['family']} coverage_tensor={cov['coverage_tensors']:.4f} "
        f"coverage_param={cov['coverage_params']:.4f} failed={[k for k, v in cov['checks'].items() if not v]} "
        f"problems={ {k: v for k, v in cov['problems'].items() if v} }",
        f"覆盖率 >= {PRETRAINED_COVERAGE_MIN} 且 stem 映射正确 / 无 shape mismatch / 无未解析映射 / 无意外映射",
        "低于阈值=大面积随机初始化（P1 的 candC 事故：n_ret=5, stem_changed=True, 真实覆盖率 9.7%）",
    )
    ctx["newly_initialised"] = bool(stem_changed)
    ctx["transferred"] = int(n_ret)
    ctx["intersect"] = f"{len(inter)}/{len(sd)}"
    ctx["stem_in"] = stem_in
    ctx["cov_family"] = cov["family"]
    ctx["cov_tensor"] = round(float(cov["coverage_tensors"]), 4)
    ctx["cov_param"] = round(float(cov["coverage_params"]), 4)
    del net, sd, src, _stock
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--selftest-fault", default=None,
                    help="故意破坏一个字段以证明 gate 不是装饰（如 channels/epochs/ir_encoding/scale）")
    A = ap.parse_args()
    ids = list(MATRIX) if (A.all or not A.id) else [A.id]
    print("=" * 92)
    print(f"Modality Experiment Contract Gate   (reference = {REFERENCE['name']}; {REFERENCE['train_config']})")
    print("=" * 92)
    rows = []
    for mid in ids:
        tcfg, mcfg = MATRIX[mid]
        fault = A.selftest_fault if (A.selftest_fault and mid == ("M1" if A.selftest_fault else mid)) else None
        try:
            ctx = check(mid, tcfg, mcfg, fault=fault)
            rows.append((mid, "PASS", ctx))
            print(f"  ✅ {mid:<4} {ctx['mode']:<9} ch={ctx['channels']} {ctx['order']:<14} "
                  f"split={ctx['split']:<18} stem_in={ctx['stem_in']} remap={ctx['transferred']} "
                  f"intersect={ctx['intersect']} cov={ctx['cov_tensor']:.4f}/{ctx['cov_param']:.4f} "
                  f"({ctx['cov_family']})")
        except AssertionError as e:
            rows.append((mid, "FAIL", {"err": str(e)}))
            print(f"  ❌ {mid:<4} {e}")
    print("=" * 92)
    npass = sum(1 for _, s, _ in rows if s == "PASS")
    print(f"CONTRACT {npass}/{len(rows)} PASS")
    return 0 if npass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
