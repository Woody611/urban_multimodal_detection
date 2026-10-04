"""_preflight_local.py — EMA FULL-300 probe 的**本机（CPU-only）preflight 实证**。

本机无 GPU ⇒ 无法启动训练；但 P1 / P3 / P5 中**不依赖训练进程**的部分可以在本机用
D′ 的部署权重（= EMA，见 P1 证据）在真实 val 数据上跑出来。

覆盖：
  · P1-EMPIRICAL  部署 best.pt 的 model 对象 == EMA（strip_optimizer 的 x["model"]=x["ema"]）
  · P3 LOGIT IDENTITY  CLS_OUT[gt_c,y,x] == W_gt·h + b_gt（真实数据、真实权重）
  · P5 H-COLLECTION SANITY  量化「eval backbone + head.train()」（旧口径）与「纯 eval」
     （正常 validation 口径）在**同一份 EMA 权重、同一批数据**上的 h 范数差异

只读：不写任何权重/配置；只在本目录产出 json 报告。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "p3_trajectory_probe"))

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.utils import yaml_load  # noqa: E402

DPRIME = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"
CELLS = HERE.parent / "p3_trajectory_probe" / "medium_cells.json"
N_IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def load_ema_model():
    ck = torch.load(DPRIME, map_location="cpu", weights_only=False)
    return ck


def build_val_ds(cells):
    data = yaml_load(str(ROOT / "data/processed/rgbid_split_train/dataset.yaml"))
    ov = dict(imgsz=1280, task="detect", rect=False, cache=False, single_cls=False, classes=None,
              fraction=1.0, channels=5, use_simotm="RGBID", mosaic=0.0, copy_paste=0.0, mixup=0.0,
              object_scale_aug=False, augment=False)
    ds = build_yolo_dataset(get_cfg(overrides=ov),
                            str((Path(data["path"]) / data["val"]).resolve()), 8, data,
                            mode="val", use_simotm="RGBID",
                            pairs_rgb_ir=["visible", "infrared"],
                            pairs_rgb_depth=["visible", "depth"])
    want = {r["image_stem"] for r in cells}
    sel = [i for i in range(len(ds)) if Path(ds.im_files[i]).stem in want]
    by_stem = {}
    for r in cells:
        by_stem.setdefault(r["image_stem"], []).append(r)
    return ds, sel, by_stem


def sample(t, cx, cy):
    ch, hh, ww = t.shape[1:]
    sc = 1280.0 / ww
    return t[0, :, int(min(max(cy / sc, 0), hh - 1)), int(min(max(cx / sc, 0), ww - 1))]


def run(nn, ds, sel, by_stem, full_eval: bool):
    """full_eval=True → 纯 eval（= 正常 validation 口径）；False → 旧口径 eval backbone + head.train()。"""
    P3hook, Hhook, OUTHook = {}, {}, {}
    nn.model[23].register_forward_hook(lambda m, i, o: P3hook.__setitem__("v", o.detach().float().clone()))
    last = nn.model[-1].cv3[0][-1]
    last.register_forward_pre_hook(lambda m, i: Hhook.__setitem__("v", i[0].detach().float().clone()))
    last.register_forward_hook(lambda m, i, o: OUTHook.__setitem__("v", o.detach().float().clone()))
    W = last.weight.detach().float().view(last.weight.shape[0], -1).numpy().astype(np.float64)
    B = last.bias.detach().float().numpy().astype(np.float64)

    nn.eval()
    if not full_eval:
        nn.model[-1].train()
    rows, resid = [], []
    with torch.no_grad():
        for k, i in enumerate(sel):
            if k >= N_IMG:
                break
            lab = ds[i]
            if lab.get("bboxes") is None or not len(lab["bboxes"]):
                continue
            stem = Path(ds.im_files[i]).stem
            img = lab["img"].float().div(255.0)
            img = (img.unsqueeze(0) if img.ndim == 3 else img)
            P3hook.clear(); Hhook.clear(); OUTHook.clear()
            nn(img)
            P3, H, OUT = P3hook.get("v"), Hhook.get("v"), OUTHook.get("v")
            if P3 is None or H is None or OUT is None:
                continue
            for r in by_stem.get(stem, []):
                c = r["class_id"]
                y = int(min(max(r["y_center_canvas"] / (1280.0 / H.shape[-1]), 0), H.shape[2] - 1))
                x = int(min(max(r["x_center_canvas"] / (1280.0 / H.shape[-1]), 0), H.shape[3] - 1))
                hv = H[0, :, y, x].numpy().astype(np.float64)
                recon = float(W[c] @ hv + B[c])
                if full_eval and OUT.shape[-1] == H.shape[-1]:
                    actual = float(OUT[0, c, y, x])
                    resid.append(abs(actual - recon))
                rows.append(dict(key=r["key"], role=r["role"], class_id=c,
                                 h_norm=float(np.linalg.norm(hv)),
                                 p3_norm=float(np.linalg.norm(P3[0, :, y, x].numpy().astype(np.float64))),
                                 logit=recon))
    return rows, resid


def main():
    ck = load_ema_model()
    print(f"[P1] ckpt keys ema={ck['ema']!r} model_is_none={ck['model'] is None}")
    print(f"[P1] ckpt epoch={ck['epoch']} best_fitness={ck['best_fitness']} "
          f"-> strip_optimizer 已执行 ⇒ model 对象来自 ema")
    cells = json.loads(CELLS.read_text(encoding="utf-8"))["rows"]
    ds, sel, by_stem = build_val_ds(cells)
    print(f"[data] val={len(ds)} imgs, matched to medium cells: {len(sel)} (using first {N_IMG})")

    nn = ck["model"].float().eval()

    rows_ev, resid = run(nn, ds, sel, by_stem, full_eval=True)
    print(f"\n[P3] logit identity on REAL weights+data: n={len(resid)}")
    if resid:
        print(f"[P3] max_residual = {max(resid):.3e}   median = {float(np.median(resid)):.3e}")

    rows_tr, _ = run(nn, ds, sel, by_stem, full_eval=False)

    def stat(rows, role=None):
        v = np.array([r["h_norm"] for r in rows if role is None or r["role"] == role])
        return (float(np.median(v)), float(v.mean()), len(v)) if len(v) else (float("nan"),) * 3

    def fmt(m):
        return f"median={m[0]:8.3f} mean={m[1]:8.3f} n={int(m[2]):4d}"

    print("\n[P5] h 范数：纯 eval（正常 validation 口径） vs 旧口径 eval+head.train()")
    for role in ("CONTROL", "G86", "MISSED_NON86", None):
        m1 = stat(rows_ev, role); m2 = stat(rows_tr, role)
        nm = role or "ALL"
        if m1[2] == 0 and m2[2] == 0:
            print(f"  {nm:14s} (no records)")
            continue
        print(f"  {nm:14s} full_eval: {fmt(m1)} | old: {fmt(m2)}")

    out = dict(probe_target="deployed D' best.pt (= EMA, strip_optimizer 路径)",
               weights_sha256="1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda",
               n_images=N_IMG, n_records=len(rows_ev),
               P3_logit_identity=dict(n=len(resid), max_residual=(max(resid) if resid else None),
                                      median_residual=(float(np.median(resid)) if resid else None)),
               P5_h_norm=dict(full_eval_all=stat(rows_ev), old_all=stat(rows_tr),
                              full_eval_G86=stat(rows_ev, "G86"), old_G86=stat(rows_tr, "G86"),
                              full_eval_CONTROL=stat(rows_ev, "CONTROL"), old_CONTROL=stat(rows_tr, "CONTROL")))
    (HERE / "_preflight_local.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {HERE / '_preflight_local.json'}")


if __name__ == "__main__":
    main()
