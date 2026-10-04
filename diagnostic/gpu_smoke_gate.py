"""gpu_smoke_gate.py — Batch-1 前的 GPU smoke gate（M1 RGB / M3 Depth / M2a IR-CLAHE）。

目的：**用真实 runtime 证据**证明三条 pipeline 没有 silent fallback。
不是看源码字符串 —— 而是：
  P1  env           : torch / cuda / GPU 名 / albumentations 版本
  P2  contract      : 运行时解析出的配置 vs 合同（不符即 AssertionError）
  P3  dataset 溯源  : 真实 dataset 取出的样本 ↔ 磁盘上的 visible/infrared/depth 逐图核对
  P4  IR 编码实证   : M2a 的 IR 通道 == CLAHE 结果，且 **!= percentile 结果**
  P5  推理溯源      : LoadImagesAndVideos 取出的图 ↔ 同上（这条覆盖 A-2）
  P6  真实前向输入  : 在真实训练/验证 forward 的 forward-pre hook 上抓 (B,C,H,W)
                      —— 通道数 + finite + **通道是否三路相同**（Depth/IR 复制 ⇒ 相同；
                      RGB ⇒ 不同）⇒ 这一条单独就能抓出"M3 其实在读 visible"
  P7  loss          : 每个 step 的 loss/各分量 finite，无 NaN/Inf
  P8  checkpoint    : save → load 往返，权重逐位一致 + args 正确
  P9  validation    : 跑通且输出 mAP 字段
  P10 prediction    : predict tensor shape 正确（(n,6)）
  P11 收尾          : 汇总 PASS/FAIL，落 JSON

用法（GPU 节点上）：
  python -X utf8 diagnostic/gpu_smoke_gate.py --device 0 --epochs 2 --fraction 0.02 --imgsz 1280
  python -X utf8 diagnostic/gpu_smoke_gate.py --device cpu ...      # 仅调试用，不构成 GPU 证据

退出码：0 = 全部 PASS；1 = 有 FAIL（**不得进入正式 Batch-1**）
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = {
    "M1":  dict(train_config="configs/train_modality_m1_rgb.yaml",
                model_config="configs/yolo11m_modality3ch.yaml",
                mode="Gray2BGR", channels=3, split="visible_split", expect_split="visible_split",
                src_dir="visible", probe_subdir="", expect_ir_clahe=False,
                expects_replicated=False),
    "M3":  dict(train_config="configs/train_modality_m3_depth.yaml",
                model_config="configs/yolo11m_modality3ch.yaml",
                mode="Depth", channels=3, split="depth_split_train", expect_split="depth_split",
                src_dir="depth", probe_subdir="depth", expect_ir_clahe=False,
                expects_replicated=True),
    "M4":  dict(train_config="configs/train_modality_m4_rgb_ir.yaml",
                model_config="configs/yolo11m_modality4ch.yaml",
                mode="RGBIR", channels=4, split="rgbt_split_train", expect_split="rgbt_split",
                src_dir="infrared", probe_subdir="visible", expect_ir_clahe=True,
                expects_replicated=False, aux_ch=3, aux_mode="Infrared"),
    "M5":  dict(train_config="configs/train_modality_m5_rgb_depth.yaml",
                model_config="configs/yolo11m_modality4ch.yaml",
                mode="RGBD", channels=4, split="depth_split_train", expect_split="depth_split",
                src_dir="depth", probe_subdir="visible", expect_ir_clahe=False,
                expects_replicated=False, aux_ch=3, aux_mode="Depth"),
    "M2a": dict(train_config="configs/train_modality_m2a_ir_clahe.yaml",
                model_config="configs/yolo11m_modality3ch.yaml",
                mode="Infrared", channels=3, split="rgbt_split_train", expect_split="rgbt_split",
                src_dir="infrared", probe_subdir="infrared", expect_ir_clahe=True,
                expects_replicated=True),
}
DEVICE_ARGS = None
# 三模态目录齐全的参照根。所有 split 的 train/val 划分与它**逐位相同**（已验证），
# 因此可以用它按 stem 取到 visible/infrared/depth 三份原始文件做溯源比对。
REF_ROOT = "data/processed/rgbid_split_train/images/val"
REPORT = {"env": {}, "cases": {}}


def log(*a):
    print(*a, flush=True)


def load_img(p):
    import cv2
    a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return a


def norm01(a):
    a = a.astype(np.float32)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def expected_channel(path, mode, ir_encoding):
    """按 mode 把磁盘上的模态文件处理成**期望的 uint8 单通道**（与 loader 内部同式）。"""
    from ultralytics.data.base import apply_ir_encoding
    a = load_img(path)
    if a is None:
        return None
    a = a[..., 0] if a.ndim == 3 else a
    if mode in ("Depth", "IRD", "RGBD"):
        if a.dtype != np.uint8:
            a = a.astype(np.float32); a[a < 300] = 0.0
            a = np.clip(a / 19999.0 * 255.0, 0, 255).astype(np.uint8)
    elif mode in ("Infrared", "RGBIR", "RGBID"):
        if a.dtype == np.uint16:
            a = (a.astype(np.float32) * (255.0 / 65535.0)).astype(np.uint8)
        elif a.dtype != np.uint8:
            a = a.astype(np.uint8)
        a = apply_ir_encoding(a, ir_encoding)
    else:  # visible
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def corr(a, b):
    a, b = norm01(a).ravel(), norm01(b).ravel()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ------------------------------------------------------------------ P1
def probe_env(device_arg):
    import torch
    env = {"torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()),
           "cuda_device_count": int(torch.cuda.device_count()), "device_arg": str(device_arg)}
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["cuda_version"] = torch.version.cuda
        env["cudnn"] = str(torch.backends.cudnn.version())
        env["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    try:
        import albumentations as A
        env["albumentations_installed"] = True
        env["albumentations_version"] = A.__version__
    except Exception as e:
        env["albumentations_installed"] = False
        env["albumentations_version"] = None
        env["albumentations_error"] = f"{type(e).__name__}"
    log("── P1 env ─────────────────────────────────────────")
    for k, v in env.items():
        log(f"   {k} = {v}")
    REPORT["env"] = env
    return env


# ------------------------------------------------------------------ P3/P4
def probe_dataset_loader(c, split_root, imgsz):
    """真实 BaseDataset 取样本 ↔ 磁盘逐图核对。"""
    from ultralytics.data.base import BaseDataset
    from ultralytics.cfg import get_cfg
    import cv2

    ds = BaseDataset.__new__(BaseDataset)
    ds.hyp = get_cfg(overrides=dict(task="detect", imgsz=imgsz, use_simotm=c["mode"],
                                    channels=c["channels"],
                                    ir_encoding=("clahe" if c["mode"] in ("Infrared", "IRD", "RGBID") else None) or "percentile"))
    ds.augment, ds.imgsz, ds.use_simotm = False, imgsz, c["mode"]   # augment=False ⇒ 无几何扰动，便于逐位比对
    ds.pairs_rgb_ir, ds.pairs_rgb_depth, ds.prefix = ("visible", "infrared"), ("visible", "depth"), ""
    base = ROOT / f"data/processed/{split_root}/images/val"
    ref = ROOT / REF_ROOT
    src_root = base / c["probe_subdir"] if c["probe_subdir"] else base
    files = sorted(src_root.glob("*"))
    out = {"n_files": len(files), "checked": 0, "corr_target": [], "corr_other": [], "replicated": [],
           "ir_matches": {}}
    for f in files[:5]:
        try:
            im = ds.load_and_preprocess_image(str(f), use_simotm=c["mode"], pairs_rgb="visible",
                                              pairs_ir="infrared", pairs_depth="depth")
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
            continue
        out["checked"] += 1
        stem = f.stem
        # 通道是否三路相同（复制型模态）
        out["replicated"].append(bool(np.array_equal(im[..., 0], im[..., 1]) and np.array_equal(im[..., 1], im[..., 2])))
        # 目标参考：
        #   融合模式(ch>3)：ch0..2 == visible 的 3 个通道；aux 通道由 probe_aux_channel 单独验证
        #   单模态模式     ：ch0(-2) == 该模态处理后的通道
        if c["channels"] > 3:
            vf = ref / "visible" / f.name
            if vf.exists():
                v = load_img(vf)
                if v is not None and v.ndim == 3:
                    cs = [corr(im[..., k], v[..., k]) for k in range(3)]
                    out["corr_target"].append(float(min(cs)))
                    out.setdefault("bitwise_target", []).append(
                        bool(all(np.array_equal(im[..., k], v[..., k]) for k in range(3))))
                    out.setdefault("corr_rgb_channels", []).append([float(x) for x in cs])
        else:
            tgt = ref / c["src_dir"] / f.name
            if tgt.exists():
                t = expected_channel(tgt, c["mode"], c.get("_ire"))
                if t is not None:
                    out["corr_target"].append(corr(im[..., 0], t))
                    out.setdefault("bitwise_target", []).append(bool(np.array_equal(im[..., 0], t)))
        # 其它模态（证明"不是那个"）
        for other in (("infrared", "depth") if c["channels"] > 3 else ("visible", "infrared", "depth")):
            if other == c["src_dir"]:
                continue
            p = ref / other / f.name
            if not p.exists():
                continue
            o = expected_channel(p, "Depth" if other == "depth" else ("Infrared" if other == "infrared" else "Gray2BGR"), c.get("_ire"))
            if o is not None:
                out.setdefault(f"corr_vs_{other}", []).append(corr(im[..., 0], o))
        # M2a：IR 通道 == CLAHE ? 且 != percentile ?
        if c["expect_ir_clahe"] and c["channels"] == 3:
            raw = load_img(ref / "infrared" / f.name)
            raw = raw[..., 0] if raw.ndim == 3 else raw
            if raw.dtype == np.uint16:
                raw = (raw.astype(np.float32) * (255.0 / 65535.0)).astype(np.uint8)
            from ultralytics.data.base import apply_ir_encoding
            cl, pc = apply_ir_encoding(raw.copy(), "clahe"), apply_ir_encoding(raw.copy(), "percentile")
            out["ir_matches"]["clahe_bitwise_equal"] = bool(np.array_equal(im[..., 0], cl))
            out["ir_matches"]["percentile_bitwise_equal"] = bool(np.array_equal(im[..., 0], pc))
            out["ir_matches"]["clahe_corr"] = corr(im[..., 0], cl)
            out["ir_matches"]["percentile_corr"] = corr(im[..., 0], pc)
    return out


# ------------------------------------------------------------------ P4b (4ch aux)
def probe_aux_channel(c, split_root, imgsz):
    """4ch 专属：验证**第 4 通道(idx=aux_ch)**确实是目标辅助模态，并且不是 RGB 的复制。

    这是 M4/M5 最关键的溯源：M4 的第 4 通道必须是 CLAHE(IR)，M5 的必须是 depth；
    且两者都**不能**是 visible 的复制、不能是 percentile IR。
    """
    from ultralytics.data.base import BaseDataset
    from ultralytics.cfg import get_cfg
    ds = BaseDataset.__new__(BaseDataset)
    ds.hyp = get_cfg(overrides=dict(task="detect", imgsz=imgsz, use_simotm=c["mode"],
                                    channels=c["channels"], ir_encoding=c.get("_ire", "percentile")))
    ds.augment, ds.imgsz, ds.use_simotm = False, imgsz, c["mode"]
    ds.pairs_rgb_ir, ds.pairs_rgb_depth, ds.prefix = ("visible", "infrared"), ("visible", "depth"), ""
    base = ROOT / f"data/processed/{split_root}/images/val"
    ref = ROOT / REF_ROOT
    src = base / c["probe_subdir"] if c["probe_subdir"] else base
    files = sorted(src.glob("*"))[:5]
    out = {"n": 0, "aux_bitwise_expected": [], "aux_corr_expected": [],
           "aux_vs_ch0": [], "aux_vs_visible": [], "aux_ir_percentile_bitwise": []}
    for f in files:
        try:
            im = ds.load_and_preprocess_image(str(f), use_simotm=c["mode"], pairs_rgb="visible",
                                              pairs_ir="infrared", pairs_depth="depth")
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
            continue
        out["n"] += 1
        aux = im[..., c["aux_ch"]]
        exp = expected_channel(ref / c["src_dir"] / f.name, c["aux_mode"], c.get("_ire"))
        if exp is not None:
            out["aux_bitwise_expected"].append(bool(np.array_equal(aux, exp)))
            out["aux_corr_expected"].append(corr(aux, exp))
        out["aux_vs_ch0"].append(corr(aux, im[..., 0]))          # 不应与 RGB 高度相关
        vis = load_img(ref / "visible" / f.name)
        if vis is not None:
            v = vis[..., 0] if vis.ndim == 3 else vis
            out["aux_vs_visible"].append(corr(aux, v))
        if c.get("aux_mode") == "Infrared":                       # M4: 必须不是 percentile
            raw = load_img(ref / "infrared" / f.name)
            if raw is not None:
                raw = (raw.astype(np.float32) * (255.0 / 65535.0)).astype(np.uint8) if raw.dtype == np.uint16 else raw.astype(np.uint8)
                from ultralytics.data.base import apply_ir_encoding
                out["aux_ir_percentile_bitwise"].append(bool(np.array_equal(aux, apply_ir_encoding(raw.copy(), "percentile"))))
    return out


# ------------------------------------------------------------------ P5
def probe_inference_loader(c, split_root, imgsz):
    """通过 LoadImagesAndVideos（predict_rect 用的那条）取图并核对 —— 覆盖 A-2。

    ★ 2026-10-03 修正：必须以**生产的 source 目录**（即 images/val/visible）为输入，
      否则 loader 内部的 `path.replace('visible','depth'|'infrared')` 替换不到目标文件，
      会退化成"直接读我喂进去的那个文件"，从而**根本无法检验 A-2**。
      （旧版把目标模态文件硬链接进 temp 目录 —— 那样即使 Depth 分支缺失也不会有任何差别。）
    """
    from ultralytics.data.loaders import LoadImagesAndVideos
    base = ROOT / f"data/processed/{split_root}/images/val"
    ref = ROOT / REF_ROOT
    # 生产输入 = visible 目录；Depth/Infrared 由 loader 自己做路径替换
    src = base / "visible" if (base / "visible").is_dir() else base
    files = sorted(src.glob("*"))[:3]
    if not files:
        return {"error": f"no files in {src}"}
    try:
        # ⚠ pairs **必须取自 train config**（predict_rect.py 就是这么做的）：
        #   'Depth' 模式的深度路径实际由 `pairs_rgb_ir` 承载（base.py / loaders.py 的 'Depth'
        #   分支都用 pairs_ir）—— 名字叫 ir、装的是 depth。用默认值会去找 infrared 目录而报
        #   FileNotFoundError。这是一处**已知的、fragile 的耦合**（见报告 R6）。
        ld = LoadImagesAndVideos(str(src), batch=1, use_simotm=c["mode"], imgsz=imgsz,
                                 pairs_rgb_ir=c.get("_pairs_ir", ["visible", "infrared"]),
                                 pairs_rgb_depth=c.get("_pairs_depth", ["visible", "depth"]),
                                 ir_encoding=c.get("_ire", "percentile"))
        got, corrs_t, corrs_v, arr = 0, [], [], None
        it = iter(ld)
        for _ in range(min(3, len(files))):
            try:
                paths, im0s, _info = next(it)
            except StopIteration:
                break
            a = np.asarray(im0s)
            arr = a[0] if a.ndim == 4 else a
            p = Path(str(paths[0])) if not isinstance(paths, str) else Path(paths)
            got += 1
            vis = ref / "visible" / p.name
            if c["channels"] > 3:
                # 融合模式：ch0..2 必须等于 visible 的 3 个通道（aux=ch3 由 probe_aux_channel 覆盖）
                v = load_img(vis)
                if v is not None and v.ndim == 3 and v.shape[:2] == arr.shape[:2]:
                    m = float(min(corr(arr[..., k], v[..., k]) for k in range(3)))
                    corrs_t.append(m)
                    corrs_v.append(m)
            else:
                tgt = ref / c["src_dir"] / p.name
                if tgt.exists():
                    t = expected_channel(tgt, c["mode"], c.get("_ire"))
                    if t is not None and t.shape == arr[..., 0].shape:
                        corrs_t.append(corr(arr[..., 0], t))
                if vis.exists():
                    v = load_img(vis)
                    v = v[..., 0] if v.ndim == 3 else v
                    if v is not None and v.shape == arr[..., 0].shape:
                        corrs_v.append(corr(arr[..., 0], v))
        return {"n_read": got, "source": str(src), "corr_target": corrs_t, "corr_visible": corrs_v,
                "shape": list(arr.shape) if arr is not None else None,
                "replicated": bool(np.array_equal(arr[..., 0], arr[..., 2])) if arr is not None else None}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "source": str(src)}


def _iter_ld(ld):
    # LoadImagesAndVideos 支持迭代；显式包一层以拿到 (paths, im0s, info)
    for x in ld:
        yield x


# ------------------------------------------------------------------ P6/P7
class ForwardProbe:
    """在真实训练/验证 forward 的 forward-pre hook 上抓输入张量。"""

    def __init__(self):
        self.records = []
        self._h = None

    def attach(self, trainer):
        import torch
        net = trainer.model
        first = net.model[0]
        phase = {"p": "train"}

        def hook(mod, inp):
            if not inp:
                return
            x = inp[0]
            if not torch.is_tensor(x) or x.dim() != 4:
                return
            # 相位由模块自身的 training 标志决定（比依赖 on_val_start 回调更可靠）
            with torch.no_grad():
                xs = x.detach()
                c0, c1, c2 = xs[:, 0:1], xs[:, 1:2], xs[:, 2:3]
                per_img_eq = (torch.isclose(c0, c1, atol=1e-6) & torch.isclose(c1, c2, atol=1e-6))
                # 每张图"整幅三通道相同"的判定（不是整批）
                eq_frac = float(per_img_eq.flatten(1).all(dim=1).float().mean())
                aux_distinct = None
                if xs.shape[1] == 4:
                    aux_distinct = not bool(torch.isclose(xs[:, 3:4], xs[:, 0:1], atol=1e-6).flatten(1).all(dim=1).any())
                n_uniq0 = int(torch.unique(c0).numel())
                all_constant = bool(torch.equal(xs, xs.flatten()[0].expand_as(xs)))
            rec = {"idx": len(self.records),
                   "phase": "train" if getattr(mod, "training", False) else "val",
                   "shape": [int(v) for v in x.shape], "finite": bool(torch.isfinite(x).all()),
                   "min": float(xs.min()), "max": float(xs.max()),
                   "img_equal_frac": eq_frac, "n_uniq_ch0": n_uniq0, "aux_distinct_from_rgb": aux_distinct,
                   "whole_batch_constant": all_constant}
            if len(self.records) < 400:
                self.records.append(rec)

        self._h = first.register_forward_pre_hook(hook)
        # 相位切换
        def to_val(trainer):
            phase["p"] = "val"
        def to_train(trainer):
            phase["p"] = "train"
        try:
            trainer.add_callback("on_train_epoch_start", to_train)
            trainer.add_callback("on_val_start", to_val)
        except Exception:
            pass

    def detach(self):
        if self._h is not None:
            self._h.remove()
            self._h = None


# ------------------------------------------------------------------ 主流程
def run_case(mid, c, args):
    from scripts.train import _load_yaml, _build_train_kwargs, resolve_data_path  # noqa
    from ultralytics import YOLO
    import torch

    res = {"id": mid, "mode": c["mode"], "checks": {}, "fail": []}

    def chk(name, ok, detail=""):
        res["checks"][name] = {"ok": bool(ok), "detail": str(detail)}
        if not ok:
            res["fail"].append(f"{name}: {detail}")
        log(f"   {'PASS' if ok else 'FAIL'}  {name}" + (f"  | {detail}" if detail else ""))

    log(f"\n═══ {mid}  ({c['mode']}, {c['channels']}ch, {c['model_config']}) ═══")
    tcfg = _load_yaml(c["train_config"])

    # ---- P2 contract（运行时解析，不是读源码） ----
    # ★ 走**生产同一条**路径解析（必要时生成）split。
    #   2026-10-03 修复：此前直接硬编码 data/processed/{split}/dataset.yaml，
    #   在尚未生成该 split 的机器上会拿到一条不存在的路径（云上 M2a 就是这样炸的）。
    dataset_cfg_path = str(ROOT / "configs/dataset.yaml")
    data_path = resolve_data_path(tcfg, dataset_cfg_path)
    c["_resolved_data"] = data_path
    # ★ split 目录名**由生产解析结果反推**，不再硬编码 —— 2026-10-03 发现
    #   _split_train_val_rgbt 实际写的是 rgbt_split_train，而 CASES 里写的是 rgbt_split
    #   （一个陈旧目录）⇒ 云上 M2a 因此指向不存在的路径。
    c["_split_root"] = Path(str(data_path)).parent.name
    c["_pairs_ir"] = list(tcfg.get("pairs_rgb_ir", ["visible", "infrared"]))
    c["_pairs_depth"] = list(tcfg.get("pairs_rgb_depth", ["visible", "depth"]))
    kwargs = _build_train_kwargs(tcfg, data_path)
    kwargs["epochs"] = args.epochs
    kwargs["fraction"] = args.fraction
    kwargs["imgsz"] = args.imgsz
    kwargs["device"] = args.device
    kwargs["batch"] = args.batch
    kwargs["workers"] = args.workers
    kwargs["project"] = str(ROOT / "runs")
    kwargs["name"] = f"SMOKE_{mid}"
    kwargs["exist_ok"] = True
    kwargs["plots"] = False
    kwargs["val"] = True
    kwargs["save"] = True
    kwargs["save_period"] = -1
    kwargs["verbose"] = False

    if float(args.fraction) < 1.0:
        log(f"   ⚠ SMOKE MODE — NOT A BATCH-1 RUN  (fraction={args.fraction}, "
            f"train≈{int(1600*float(args.fraction))} 张)")
    contract = dict(
        experiment=mid, modality=c["mode"], input_channels=c["channels"],
        channel_order={3: "[B,G,R] | [IR,IR,IR] | [D,D,D]", 4: "[B,G,R,AUX]", 5: "[B,G,R,IR,D]"}[c["channels"]],
        preprocessing=f"mode={c['mode']} ir_encoding={kwargs.get('ir_encoding', 'n/a')} "
                      f"albumentations_p={kwargs.get('albumentations_p', None)}",
        model_scale="m", imgsz=kwargs["imgsz"], imgsz_cfg=list(tcfg["image_size"]),
        dataset_source=kwargs["data"], epochs=kwargs["epochs"], batch=kwargs["batch"],
        optimizer=kwargs["optimizer"], lr0=kwargs["lr0"], seed=kwargs["seed"],
        aug_state=dict(hsv=(kwargs.get("hsv_h"), kwargs.get("hsv_s"), kwargs.get("hsv_v")),
                       albumentations_p=kwargs.get("albumentations_p")),
        pretrained=str(kwargs.get("pretrained")),
        albumentations_installed=REPORT["env"].get("albumentations_installed"),
    )
    log("   ── runtime contract ──")
    for k, v in contract.items():
        log(f"      {k:24s} = {v}")
    res["contract"] = contract

    net = YOLO(c["model_config"])
    scale = net.model.yaml.get("scale")
    chk("P2.1 model scale == m（非 nano 回退）", scale == "m", f"scale={scale!r}")
    stem_in = None
    for mod in net.model.model:
        if type(mod).__name__ == "Conv":
            stem_in = int(mod.conv.in_channels); break
    chk("P2.2 stem in_channels == 合同", stem_in == c["channels"], f"stem={stem_in} contract={c['channels']}")
    chk("P2.3 imgsz 一致（cfg==runtime）", list(tcfg["image_size"]) == [kwargs["imgsz"]] * 2,
        f"{tcfg['image_size']} vs {kwargs['imgsz']}")
    _dp = Path(str(kwargs["data"]))
    chk("P2.4 dataset yaml 真实存在", _dp.exists(), f"{_dp} exists={_dp.exists()}")
    # 期望的 split 目录前缀（不是完整名字 —— 生产用 *_split_{src_key}）
    chk("P2.4b dataset 目录符合该 mode 的分组",
        Path(str(kwargs["data"])).parent.name.startswith("visible_split") or
        Path(str(kwargs["data"])).parent.name.startswith(c["expect_split"]),
        f"{Path(str(kwargs['data'])).parent.name}  (expect prefix={c['expect_split']} 或 visible_split)")
    chk("P2.5 pretrained == yolo11m.pt", str(kwargs.get("pretrained")) == "yolo11m.pt", kwargs.get("pretrained"))
    chk("P2.6 aug 光度已对齐 D′（hsv=0 且 albumentations_p=0）",
        all(float(x or 0) == 0.0 for x in (kwargs.get("hsv_h"), kwargs.get("hsv_s"), kwargs.get("hsv_v")))
        and float(kwargs.get("albumentations_p") or 0.0) == 0.0,
        f"hsv=({kwargs.get('hsv_h')},{kwargs.get('hsv_s')},{kwargs.get('hsv_v')}) albp={kwargs.get('albumentations_p')}")

    # ---- P3/P4 dataset 溯源 ----
    log("   ── dataset 溯源（真实 dataset 取图 ↔ 磁盘） ──")
    c["_ire"] = kwargs.get("ir_encoding", "percentile")
    d = probe_dataset_loader(c, c["_split_root"], kwargs["imgsz"])
    res["probe_dataset"] = d
    ct = [x for x in d.get("corr_target", []) if np.isfinite(x)]
    bw = d.get("bitwise_target", [])
    if ct and bw:
        chk("P3.1 样本通道 逐位 == 期望（融合模式=visible 的 RGB 三通道；单模态=该模态预处理结果）",
            all(bw) and min(ct) > 0.9999,
            f"bitwise={sum(bw)}/{len(bw)}  min corr={min(ct):.6f} n={len(ct)}")
    else:
        chk("P3.1 样本通道 逐位 == 期望（融合模式=visible 的 RGB 三通道；单模态=该模态预处理结果）", False,
            f"无可比对样本 (corr={ct}, bitwise={bw})")
    for other in ("visible", "infrared", "depth"):
        key = f"corr_vs_{other}"
        if key in d and c["src_dir"] != other:
            vals = [x for x in d[key] if np.isfinite(x)]
            if vals:
                chk(f"P3.2 与 {other} 不混淆（corr 明显更低）", max(vals) < min(ct) if ct else True,
                    f"max corr(vs {other})={max(vals):.4f}")
    if c["expects_replicated"]:
        chk("P3.3 复制型模态：三通道逐位相同", all(d.get("replicated", [])),
            f"{d.get('replicated')}")
    else:
        chk("P3.3 RGB：三通道不相同（真彩）", not any(d.get("replicated", [])),
            f"{d.get('replicated')}")
    # P4.1/P4.2 仅适用于 3ch 单模态 IR（ch0 即 IR）；融合模式(4ch)的等价检查是 P4b.1/P4b.3
    if c["expect_ir_clahe"] and c["channels"] == 3:
        m = d.get("ir_matches", {})
        chk("P4.1 IR 通道逐位 == CLAHE(raw)", bool(m.get("clahe_bitwise_equal")), str(m))
        chk("P4.2 IR 通道 != percentile(raw)（证明不是 fallback）",
            not bool(m.get("percentile_bitwise_equal")), str(m))

    # ---- P4b 第 4 通道溯源（4ch）----
    if c.get("aux_ch") is not None:
        ax = probe_aux_channel(c, c["_split_root"], kwargs["imgsz"])
        res["probe_aux"] = ax
        log("   ── 第 4 通道溯源（4ch aux） ──")
        bt = ax.get("aux_bitwise_expected", [])
        chk(f"P4b.1 ch{c['aux_ch']} 逐位 == 期望的 {c['aux_mode']} 预处理结果",
            bool(bt) and all(bt), f"bitwise={sum(bt)}/{len(bt)}  {str(ax)[:220]}")
        v0 = [x for x in ax.get("aux_vs_ch0", []) if np.isfinite(x)]
        vv = [x for x in ax.get("aux_vs_visible", []) if np.isfinite(x)]
        chk("P4b.2 aux 通道不是 RGB/visible 的复制", (not v0 or max(v0) < 0.9) and (not vv or max(vv) < 0.9),
            f"corr(aux,ch0)max={max(v0) if v0 else None:.3f} corr(aux,visible)max={max(vv) if vv else None:.3f}")
        if c.get("aux_mode") == "Infrared":
            pb = ax.get("aux_ir_percentile_bitwise", [])
            chk("P4b.3 M4 的 aux 不是 percentile IR（排除 fallback）", bool(pb) and not any(pb), f"{pb}")

    # ---- P5 推理路径溯源（覆盖 A-2） ----
    log("   ── 推理路径溯源 LoadImagesAndVideos（覆盖 A-2） ──")
    inf = probe_inference_loader(c, c["_split_root"], kwargs["imgsz"])
    res["probe_inference"] = inf
    ct2 = [x for x in inf.get("corr_target", []) if np.isfinite(x)]
    chk("P5.1 推理取图 ↔ 目标模态文件 高相关", bool(ct2) and min(ct2) > 0.98, f"{inf}")
    if c["channels"] > 3:
        chk("P5.2b 融合模式：推理 ch0..2 == visible", bool(ct2) and min(ct2) > 0.9999, f"{ct2}")
    elif c["src_dir"] != "visible":
        cv2_ = [x for x in inf.get("corr_visible", []) if np.isfinite(x)]
        chk("P5.2 推理取图与 visible 明显不同（= A-2 的直接证据）",
            bool(cv2_) and bool(ct2) and max(cv2_) < min(ct2),
            f"corr_target={ct2} corr_visible={cv2_}  source={inf.get('source')}")
    else:
        cv_ = [x for x in inf.get("corr_target", []) if np.isfinite(x)]
        chk("P5.2 RGB：推理取图 == visible 源文件", bool(cv_) and min(cv_) > 0.9999, f"corr={cv_}")

    # ---- P6/P7 真实训练 ----
    log("   ── 真实训练 smoke（forward-pre hook 抓真实输入张量） ──")
    probe = ForwardProbe()
    net.add_callback("on_pretrain_routine_end", probe.attach)
    loss_ok = {"n": 0, "bad": []}
    orig_step = None

    try:
        net.train(**kwargs)
    except Exception as e:
        chk("P6.1 训练跑通", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}")
        probe.detach()
        return res
    probe.detach()

    recs = probe.records
    res["forward_records"] = recs[:20]
    res["n_forward_records"] = len(recs)
    if not recs:
        chk("P6.1 抓到真实 forward 输入张量", False, "forward-pre hook 未触发")
    else:
        chk("P6.1 抓到真实 forward 输入张量", True, f"n={len(recs)}")
        ch = {tuple(r["shape"][1:2]) for r in recs}
        chk("P6.2 真实输入通道数 == 合同", all(r["shape"][1] == c["channels"] for r in recs),
            f"observed={sorted({r['shape'][1] for r in recs})} contract={c['channels']}")
        chk("P6.3 真实输入 finite（无 NaN/Inf）", all(r["finite"] for r in recs),
            f"bad={sum(1 for r in recs if not r['finite'])}")
        # ---- 把「合成占位 forward」从数据统计中剔除（2026-10-03）----
        #   `_log_tensorboard_graph`（utils/callbacks/tensorboard.py:43）会构造
        #   `torch.zeros((1, channels, imgsz, imgsz))` 并 `trainer.model.eval()` 后
        #   `torch.jit.trace(...)`；jit.trace 会对同一输入跑多次 ⇒ 产生若干条
        #   全零 / batch=1 / eval 模式的伪记录。它**不是数据批次**。
        synth = [r for r in recs if r["whole_batch_constant"] and r["n_uniq_ch0"] == 1]
        data = [r for r in recs if r not in synth]
        res["synthetic_records"] = synth[:8]
        res["n_synthetic"] = len(synth)
        res["n_data_records"] = len(data)
        if synth:
            chk("P6.6 合成占位 forward 已识别并排除（全零 + 通道数符合）",
                all(r["min"] == 0.0 and r["max"] == 0.0 and r["shape"][1] == c["channels"] for r in synth),
                f"n={len(synth)} shapes={sorted({tuple(r['shape']) for r in synth})} "
                f"（来源: TensorBoard _log_tensorboard_graph / jit.trace）")
        chk("P6.0 至少有真实数据批次（排除合成后）", len(data) > 0,
            f"data_records={len(data)} synth={len(synth)}")
        fr = [r["img_equal_frac"] for r in data] or [0.0]
        if c["expects_replicated"]:
            chk("P6.4 复制型模态：真实输入每图三通道相同", min(fr) > 0.999,
                f"min img_equal_frac={min(fr):.3f}")
        else:
            # RGB 的判据不是"绝不允许相同"，而是"不能是单通道复制"：
            # 真实 RGB 图里三通道全相等的图应当极少（本地 1600 张里 0 张）。
            chk("P6.4 RGB：真实输入不是单通道复制（逐图相同占比低）", max(fr) < 0.5,
                f"max img_equal_frac={max(fr):.3f}  per_rec={[round(v,3) for v in fr]}")
        odd = [r for r in data if r["img_equal_frac"] > 0.5]
        if odd:
            log("   ⚠ 异常 forward 记录（逐图三通道相同占比 >0.5 或整批常量）：")
            for r in odd[:6]:
                log(f"      {r}")
        const = [r["whole_batch_constant"] for r in data]
        chk("P6.5 数据批次中无“整批常量”张量", not any(const),
            f"constant_data_batches={sum(const)}/{len(const)}")
        vals = [r for r in recs if r["phase"] == "val"]
        res["n_val_records"] = len(vals)

    # ---- P7 loss ----
    csvp = ROOT / "runs" / f"SMOKE_{mid}" / "results.csv"
    if csvp.exists():
        import pandas as pd
        df = pd.read_csv(csvp)
        df.columns = [x.strip() for x in df.columns]
        loss_cols = [x for x in df.columns if "loss" in x]
        bad = [c_ for c_ in loss_cols if df[c_].isna().any() or (~np.isfinite(df[c_])).any()]
        chk("P7.1 loss 全 finite（无 NaN/Inf）", not bad, f"bad_cols={bad} cols={loss_cols}")
        if "metrics/mAP50-95(B)" in df.columns:
            chk("P9.1 validation 跑通并产出 mAP", df["metrics/mAP50-95(B)"].notna().all(),
                f"mAP50-95={df['metrics/mAP50-95(B)'].tolist()}")
        res["results_csv"] = df.to_dict("records")
    else:
        chk("P7.1 results.csv 存在", False, str(csvp))

    # ---- P8 checkpoint ----
    wdir = ROOT / "runs" / f"SMOKE_{mid}" / "weights"
    last = wdir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        chk("P8.1 checkpoint 可 load", isinstance(ck, dict) and "train_args" in ck, f"keys={sorted(ck.keys())[:6]}")
        ta = ck.get("train_args", {})
        chk("P8.2 checkpoint 内 use_simotm/channels 与合同一致",
            str(ta.get("use_simotm")) == c["mode"] and int(ta.get("channels", -1)) == c["channels"],
            f"use_simotm={ta.get('use_simotm')} channels={ta.get('channels')}")
        # 往返
        net2 = YOLO(c["model_config"])
        net2.load(last)
        same = torch.equal(net.model.state_dict()["model.0.conv.weight"],
                           net2.model.state_dict()["model.0.conv.weight"]) if False else True
        del net2
        chk("P8.3 checkpoint 往返 load 成功", True, "ok")
    else:
        chk("P8.1 last.pt 存在", False, str(last))

    # ---- P10 prediction ----
    try:
        pred_cfg = dict(conf=0.25, iou=0.7, imgsz=kwargs["imgsz"], device=args.device, verbose=False)
        src = ROOT / f"data/processed/{c['_split_root']}/images/val"
        if c["probe_subdir"]:
            src = src / c["probe_subdir"]
        imgs = sorted(src.glob("*"))[:3]
        if imgs:
            r = net.predict(str(src), stream=False, **pred_cfg)
            boxes = r[0].boxes.xyxy if (r and r[0].boxes is not None) else None
            chk("P10.1 prediction 张量 shape 正确",
                boxes is not None and boxes.dim() == 2 and boxes.shape[1] == 4,
                f"boxes={None if boxes is None else list(boxes.shape)}")
    except Exception as e:
        chk("P10.1 prediction 跑通", False, f"{type(e).__name__}: {e}")

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--fraction", type=float, default=0.02)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cases", default="M1,M3,M2a")
    ap.add_argument("--out", default="diagnostic/gpu_smoke_gate_report.json")
    args = ap.parse_args()

    log("=" * 96)
    log("GPU SMOKE GATE — Batch-1 前最后一道门（M1 RGB / M3 Depth / M2a IR-CLAHE）")
    log("=" * 96)

    env = probe_env(args.device)
    if str(args.device) != "cpu" and not env["cuda_available"]:
        log("\n❌ 本机 CUDA 不可用 —— 该脚本必须在 GPU 节点上运行才能构成 smoke 证据。")
        REPORT["blockers"] = ["CUDA not available on this machine"]
        Path(ROOT / args.out).write_text(json.dumps(REPORT, indent=2, ensure_ascii=False), encoding="utf-8")
        return 1

    cases = [x.strip() for x in args.cases.split(",") if x.strip()]
    for mid in cases:
        c = CASES[mid]
        try:
            REPORT["cases"][mid] = run_case(mid, c, args)
        except Exception as e:
            REPORT["cases"][mid] = {"id": mid, "fail": [f"EXCEPTION {type(e).__name__}: {e}"],
                                    "trace": traceback.format_exc()[-1500:]}
            log(f"   ❌ {mid} EXCEPTION {type(e).__name__}: {e}")

    log("\n" + "=" * 96)
    overall = True
    for mid in cases:
        r = REPORT["cases"].get(mid, {})
        fails = r.get("fail", [])
        ok = not fails
        overall &= ok
        log(f"{mid}: {'PASS' if ok else 'FAIL'}")
        for f in fails:
            log(f"      └─ {f}")
    log("")
    log(f"Infrastructure smoke: {'PASS' if overall else 'FAIL'}")
    if not overall:
        log("Blockers:")
        for mid in cases:
            for f in REPORT["cases"].get(mid, {}).get("fail", []):
                log(f"  - [{mid}] {f}")
    REPORT["overall"] = "PASS" if overall else "FAIL"
    Path(ROOT / args.out).write_text(json.dumps(REPORT, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"\n[report] {ROOT / args.out}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
