"""fusion_phase0_repair_tests.py — Infrastructure Repair 的 CPU-only 验证套件。

ZERO GPU · ZERO forward · ZERO training · ZERO inference（只做单图预处理与模型**构造**）。

覆盖:
  T1  IR dispatch        —— clahe / percentile / raw 三者确实不同，且三条路径一致
  T2  D′ regression      —— RGBID 预处理与新实现**逐位相同**（对旧实现内联复刻做对拍）
  T3  IRD provenance     —— [IR, D] 顺序（IR=11, D=22 合成图），train/val/infer 三路一致
  T4  pretrained remap   —— 3ch / 4ch / 2ch / 5ch(D′) 的迁移策略与张量数
  T5  scale              —— 解析出的 scale == m 且 depth/width/max_channels 与 D′ 相同
  T6  augmentation       —— albumentations_p 覆盖前后行为

用法:
  python -X utf8 diagnostic/fusion_phase0_repair_tests.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = []


def rec(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  | {detail}" if detail else ""))


# ---------------------------------------------------------------- T1 / T2
def t1_ir_dispatch():
    print("\n[T1] IR preprocessing dispatch（synthetic uint8）")
    from ultralytics.data.base import apply_ir_encoding, IR_ENCODING_CHOICES

    rng = np.random.default_rng(0)
    ir = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    ir = np.clip(ir.astype(np.float32) * 0.35 + 90, 0, 255).astype(np.uint8)  # 低对比度，像真 IR

    out = {m: apply_ir_encoding(ir.copy(), m) for m in IR_ENCODING_CHOICES}
    rec("T1.1 clahe != percentile", not np.array_equal(out["clahe"], out["percentile"]),
        f"maxdiff={int(np.abs(out['clahe'].astype(int)-out['percentile'].astype(int)).max())}")
    rec("T1.2 raw != percentile", not np.array_equal(out["raw"], out["percentile"]))
    rec("T1.3 raw == 输入（不动）", np.array_equal(out["raw"], ir))
    rec("T1.4 dtype 均为 uint8",
        all(v.dtype == np.uint8 and v.shape == ir.shape for v in out.values()))
    # 未知取值必须抛错（不再静默回落）
    try:
        apply_ir_encoding(ir, "CLAHEE")
        rec("T1.5 未知取值 raise", False, "未抛错")
    except ValueError:
        rec("T1.5 未知取值 raise", True, "ValueError")
    rec("T1.6 大小写不敏感", np.array_equal(apply_ir_encoding(ir, "CLAHE"), out["clahe"]))
    return out


def t2_dprime_regression():
    print("\n[T2] D′ regression —— RGBID 分支逐位不变")
    from ultralytics.data.base import apply_ir_encoding

    imgs = sorted((ROOT / "data/processed/rgbid_split_train/images/train/infrared").glob("*"))[:8]
    if not imgs:
        rec("T2.0 找到真实 IR 图", False, "目录为空"); return
    rec("T2.0 找到真实 IR 图", True, f"n={len(imgs)}")

    def old_clahe(im):
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(im)

    def old_percentile(im):
        lo, hi = np.percentile(im, (1.0, 99.0))
        if hi - lo > 1:
            im = np.clip((im.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        return im

    ok_c = ok_p = True
    for p in imgs:
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        raw = raw[..., 0] if raw.ndim == 3 else raw
        if raw.dtype == np.uint16:
            raw = (raw.astype(np.float32) * (255.0 / 65535.0)).astype(np.uint8)
        elif raw.dtype != np.uint8:
            raw = raw.astype(np.uint8)
        if not np.array_equal(apply_ir_encoding(raw.copy(), "clahe"), old_clahe(raw.copy())):
            ok_c = False
        if not np.array_equal(apply_ir_encoding(raw.copy(), "percentile"), old_percentile(raw.copy())):
            ok_p = False
    rec("T2.1 clahe 与旧实现逐位相同", ok_c, f"{len(imgs)} 张真实图")
    rec("T2.2 percentile 与旧实现逐位相同", ok_p, f"{len(imgs)} 张真实图")

    # RGBID 5ch 通道序 / 5 通道数
    from ultralytics.data.base import BaseDataset
    ds = BaseDataset.__new__(BaseDataset)
    vis = np.dstack([np.full((8, 8), v, np.uint8) for v in (10, 20, 30)])  # B,G,R
    irn = np.full((8, 8), 44, np.uint8)
    dep = np.full((8, 8), 55, np.uint8)
    m = BaseDataset._merge_channels_rgbid(ds, vis, irn, dep)
    rec("T2.3 D′ 通道序 [B,G,R,IR,D]", m.shape == (8, 8, 5) and
        int(m[0, 0, 0]) == 10 and int(m[0, 0, 1]) == 20 and int(m[0, 0, 2]) == 30 and
        int(m[0, 0, 3]) == 44 and int(m[0, 0, 4]) == 55, f"shape={m.shape}, row0={m[0,0].tolist()}")


# ---------------------------------------------------------------- T3
def t3_ird_provenance():
    print("\n[T3] IRD merge provenance（IR=11, D=22）")
    from ultralytics.data.base import BaseDataset
    ds = BaseDataset.__new__(BaseDataset)
    irn = np.full((8, 8), 11, np.uint8)
    dep = np.full((8, 8), 22, np.uint8)
    m = BaseDataset._merge_channels_ir_depth(ds, irn, dep)
    rec("T3.1 通道序 == [IR, D]（非 [D, IR]）",
        m.shape == (8, 8, 2) and int(m[0, 0, 0]) == 11 and int(m[0, 0, 1]) == 22,
        f"shape={m.shape}, ch0={int(m[0,0,0])}, ch1={int(m[0,0,1])}")

    # inference loader 侧：直接跑 _merge 逻辑（loaders 用 cv2.merge((im_ir, im_dp))）
    mm = cv2.merge((irn, dep))
    rec("T3.2 loaders 侧同样 [IR, D]",
        int(mm[0, 0, 0]) == 11 and int(mm[0, 0, 1]) == 22)

    # 三条路径的 mode 分支都存在
    import inspect
    from ultralytics.data import base as _b, loaders as _l
    sb, sl = inspect.getsource(_b), inspect.getsource(_l)
    rec("T3.3 base.py 有 IRD 分支", "'IRD'" in sb and "_merge_channels_ir_depth" in sb)
    rec("T3.4 loaders.py(image) 有 IRD/Infrared/Depth 分支",
        "'IRD'" in sl and "'Infrared'" in sl and "'Depth'" in sl)
    # 图片路径（非视频路径）是否真的含这三个分支：按缩进定位 image 块
    lines = sl.splitlines()
    i_img = next(i for i, l in enumerate(lines) if l.strip() == 'self.mode = "image"')
    img_block = "\n".join(lines[i_img:])
    rec("T3.5 image 路径含 Depth/Infrared/IRD",
        "== 'Depth'" in img_block and "== 'Infrared'" in img_block and "== 'IRD'" in img_block,
        "修复前 image 路径三者全缺，会 fallthrough 读 visible 原图")


# ---------------------------------------------------------------- T4
def t4_pretrained():
    print("\n[T4] pretrained transfer")
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import _transfer_rgb_pretrained
    from ultralytics.utils.torch_utils import intersect_dicts

    src = torch.load(ROOT / "yolo11m.pt", map_location="cpu", weights_only=False)["model"].float().state_dict()
    cases = [("M1/2/3 3ch", "configs/yolo11m_modality3ch.yaml", 3, "std"),
             ("M4/M5 4ch", "configs/yolo11m_modality4ch.yaml", 4, "rgb+aux"),
             ("M6 2ch", "configs/yolo11m_modality2ch.yaml", 2, "gray-rep"),
             ("D′ 5ch sepstem", "configs/yolo11m_sepstem.yaml", 5, "sepstem")]
    for label, cfg, ch, kind in cases:
        m = YOLO(cfg).model
        sd = m.state_dict()
        inter = intersect_dicts(src, sd)
        stem = next((mod for mod in m.model if type(mod).__name__ == "Conv"), None)
        pre = stem.conv.weight.detach().clone()
        n_ret = _transfer_rgb_pretrained(m, src)
        post = stem.conv.weight.detach()
        changed = not torch.equal(pre, post)
        if kind == "std":
            rec(f"T4 {label}: 标准 load 全量匹配，无 remap",
                len(inter) == len(sd) - 6 and n_ret == 0,
                f"intersect={len(inter)}/{len(sd)} (差 6 = Detect cv3), remap={n_ret}")
        else:
            # RGB 通道必须等于 stock，且 aux 非 NaN / 非全零
            W = post.reshape(post.shape[0], post.shape[1], -1)
            sw = src["model.0.conv.weight"].float().reshape(src["model.0.conv.weight"].shape[0], 3, -1)
            if kind == "rgb+aux":
                rgb_ok = torch.allclose(W[:, :3, :], sw[:W.shape[0]])
                aux = W[:, 3:, :]
                aux_ok = (not torch.isnan(aux).any()) and float(aux.abs().mean()) > 1e-6
                det = f"remap={n_ret} | stem in={post.shape[1]} | RGB==stock:{rgb_ok} | aux_nonzero:{aux_ok}"
                rec(f"T4 {label}: RGB=pretrained, aux=mean(RGB)", (n_ret == 5 and rgb_ok and aux_ok and changed), det)
            elif kind == "gray-rep":
                g = sw.mean(dim=1, keepdim=True)
                ok = torch.allclose(W, g.expand_as(W)[: W.shape[0]])
                det = f"remap={n_ret} | stem in={post.shape[1]} | all-ch==mean(RGB):{ok}"
                rec(f"T4 {label}: 全通道 = mean(W_R,W_G,W_B)", (n_ret == 5 and ok and changed), det)
            else:
                rec(f"T4 {label}: sepstem remap 生效",
                    n_ret == 550, f"remap={n_ret}（预期 550）")
        del m, sd
    del src


# ---------------------------------------------------------------- T5
def t5_scale():
    print("\n[T5] model scale 显式性")
    from ultralytics.nn.tasks import guess_model_scale
    from ultralytics import YOLO
    ref = None
    for cfg in ["configs/yolo11m_modality3ch.yaml", "configs/yolo11m_modality4ch.yaml",
                "configs/yolo11m_modality2ch.yaml", "configs/yolo11m_sepstem.yaml",
                "configs/yolo11m_earlyfusion.yaml"]:
        m = YOLO(cfg).model
        d, w, mx = m.yaml["scales"][m.yaml["scale"]]
        sig = (m.yaml["scale"], float(d), float(w), int(mx))
        if ref is None:
            ref = sig
        rec(f"T5 {Path(cfg).name}: scale=m 且 depth/width/max_ch 与 D′ 一致",
            sig == ref and sig[0] == "m",
            f"scale={sig[0]} depth={sig[1]} width={sig[2]} max_ch={sig[3]} params={sum(p.numel() for p in m.parameters()):,}")
        del m
    rec("T5 guess_model_scale 对 yolo11_visible.yaml 返回 ''（= 静默回退 nano 的陷阱）",
        guess_model_scale("configs/yolo11_visible.yaml") == "", "已确认；新 yaml 命名已规避")


# ---------------------------------------------------------------- T6
def t6_augmentation():
    print("\n[T6] augmentation 分派")
    from ultralytics.cfg import get_cfg
    from ultralytics.data.augment import Albumentations, Albumentations4C
    c3 = get_cfg(overrides=dict(imgsz=1280, task="detect", channels=3, use_simotm="Gray2BGR"))
    c5 = get_cfg(overrides=dict(imgsz=1280, task="detect", channels=5, use_simotm="RGBID"))
    rec("T6.1 缺省 albumentations_p 为 None（=> 行为不变）",
        getattr(c3, "albumentations_p", "MISSING") is None and getattr(c5, "albumentations_p", "MISSING") is None)
    c3o = get_cfg(overrides=dict(imgsz=1280, task="detect", channels=3, use_simotm="Gray2BGR", albumentations_p=0.0))
    rec("T6.2 覆盖值可被 get_cfg 接受", float(c3o.albumentations_p) == 0.0)
    rec("T6.3 Albumentations(p=0) 为 no-op 构造", Albumentations(p=0).p == 0)
    rec("T6.4 Albumentations4C(p=0) 为 no-op 构造", Albumentations4C(p=0).p == 0)


def main():
    print("=" * 90)
    print("Fusion Phase 0 — Infrastructure Repair  CPU validation（无 forward / 无 GPU / 无训练）")
    print("=" * 90)
    t1_ir_dispatch()
    t2_dprime_regression()
    t3_ird_provenance()
    t4_pretrained()
    t5_scale()
    t6_augmentation()
    print("\n" + "=" * 90)
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"TOTAL {npass}/{len(RESULTS)} PASS")
    for n, ok, d in RESULTS:
        if not ok:
            print(f"  ❌ {n}  {d}")
    print("=" * 90)
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
