"""_run.py — Train-side Native-small GT-class Response Audit（**单次冻结只读 forward**）

授权范围内的唯一新增模型计算：一次 frozen D′ best.pt 在 **train 原图、augmentation OFF** 上的 forward，
用于测量每个 train GT 的 **GT-class raw logit / sigmoid**。

严格禁止（本轮）：model.train() / backward() / autograd.grad / optimizer / scheduler / 任何训练 /
fine-tuning / 修改或保存 checkpoint / 修改 loss·TAL·head·backbone·dataset·labels·prediction TXT·
evaluator·configs·tracked source / 修改已有 diagnostic artifact / submission / 官方线上推理。

数据口径：train 原图 + augmentation **全 OFF**（Mosaic / MixUp / RandomPerspective / OASA / crop /
modality dropout 均关）。`ir_encoding` 与既有 audit 链保持一致（= 不传该键 ⇒ base.py:333 回落
**percentile**），以便与既有 val 侧 artifact 可比；该口径分歧已在 REPORT 中显著声明。

**不使用 `model.train()`**：为拿到 raw class logit，改为 hook `Detect.cv3[li][-1]`（末层 1×1 conv）
的**输出** —— 它在 eval 模式下同样执行，且等于 `W_c·h + b_c`（与 `_v7_vectors` 的定义同量）。
BN 全程保持 eval（running stats），即部署口径。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, yaml_model_load  # noqa: E402
from ultralytics.utils import yaml_load  # noqa: E402

SEPSTEM_YAML = ROOT / "configs/yolo11m_sepstem.yaml"
CKPT = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"
DS_YAML = ROOT / "data/processed/rgbid_split_train/dataset.yaml"
TR_IMG = ROOT / "data/processed/rgbid_split_train/images/train/visible"
TR_LAB = ROOT / "data/processed/rgbid_split_train/labels/train/visible"

# aug 全关；不动 ir_encoding（⇒ base.py:333 回落 percentile，与既有 audit 链一致）
OV = dict(imgsz=1280, task="detect", rect=False, cache=False, single_cls=False, classes=None,
          fraction=1.0, channels=5, use_simotm="RGBID", object_scale_aug=False,
          mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0, translate=0.1, scale=0.5,
          shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, augment=False)


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def native_sizes():
    """每个 train 图像的 native (W,H)，只读 header。"""
    from ultralytics.utils.patches import imread  # noqa: F401  (仅为保持 import 路径一致)
    from PIL import Image
    out = {}
    for ip in sorted(p for p in TR_IMG.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}):
        try:
            with Image.open(ip) as im:
                out[ip.stem] = (int(im.width), int(im.height))
        except Exception:  # noqa: BLE001
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="仅用于吞吐试跑；0 = 全量")
    args = ap.parse_args()

    print("=" * 104)
    print("TRAIN-SIDE NATIVE-SMALL GT-CLASS RESPONSE —— 单次冻结只读 forward")
    print("=" * 104)
    ck_sha = sha256(CKPT)
    print(f"  checkpoint      = {CKPT}")
    print(f"  checkpoint SHA  = {ck_sha}")
    print(f"  model yaml      = {SEPSTEM_YAML}  sha={sha256(SEPSTEM_YAML)[:16]}")
    print(f"  dataset yaml    = {DS_YAML}  sha={sha256(DS_YAML)[:16]}")

    cfg = yaml_model_load(str(SEPSTEM_YAML))
    model = DetectionModel(cfg, nc=12, verbose=False)
    w, _ = attempt_load_one_weight(str(CKPT))
    model.load(w)
    model.model[-1].stride = model.stride
    model.args = SimpleNamespace(cls_pw=None)
    model.eval()                      # ← 全程 eval；**不调用 model.train() 或任何 .train()**
    for p_ in model.parameters():
        p_.requires_grad_(False)

    sd = model.state_dict()
    nc = int(model.model[-1].nc)
    reg_max = int(model.model[-1].reg_max)
    print(f"  nc={nc} reg_max={reg_max} no={int(model.model[-1].no)} stride={model.stride.tolist()}")
    print(f"  cv3 末层形状: " + ", ".join(
        f"L{i}={tuple(sd[f'model.30.cv3.{i}.2.weight'].shape)}" for i in range(3)))

    # ---- hook cv3[li][-1] 的输出 = raw class logits（每层） ----
    feat: dict[int, torch.Tensor] = {}

    def mk(li):
        def h(m, i_, o):
            feat[li] = o.detach()
        return h

    for li in range(3):
        model.model[-1].cv3[li][-1].register_forward_hook(mk(li))

    data = yaml_load(str(DS_YAML))
    ds = build_yolo_dataset(get_cfg(overrides=OV), str((Path(data["path"]) / data["train"]).resolve()),
                            8, data, mode="val", use_simotm="RGBID",
                            pairs_rgb_ir=["visible", "infrared"],
                            pairs_rgb_depth=["visible", "depth"])
    n_all = len(ds)
    idxs = list(range(n_all)) if args.limit <= 0 else list(range(min(args.limit, n_all)))
    print(f"  dataset n={n_all}（mode=val, augment=False, mosaic=0）  本次处理 {len(idxs)} 张")
    ns = native_sizes()
    print(f"  训练图像 native 尺寸表 n={len(ns)}")

    rows, drop = [], []
    n_img_ok = 0
    for c, i in enumerate(idxs):
        try:
            lab = ds[i]
        except Exception as e:  # noqa: BLE001
            drop.append(dict(index=i, err=f"ds_load:{type(e).__name__}"))
            continue
        bb, cl = lab.get("bboxes"), lab.get("cls")
        if bb is None or len(bb) == 0:
            continue
        stem = Path(ds.im_files[i]).stem
        img = lab["img"].float().div(255.0)
        img = img.unsqueeze(0) if img.ndim == 3 else img
        feat.clear()
        with torch.no_grad():                       # ← 只读推理
            model(img)
        if len(feat) != 3:
            drop.append(dict(index=i, err="hook_missing"))
            continue
        n_img_ok += 1
        Wn, Hn = ns.get(stem, (0, 0))
        # 原生 label 行（用于 native 面积 + 交叉校验归一化坐标）
        lrows = [ln.split() for ln in (TR_LAB / f"{stem}.txt").read_text(
            encoding="utf-8").splitlines() if ln.strip()]
        for j in range(len(bb)):
            cid = int(cl[j])
            cx, cy = float(bb[j][0]) * 1280, float(bb[j][1]) * 1280
            r = dict(stem=stem, gt=j, cls=cid)
            # 与 _v7_extract.sample 完全相同的坐标规则：sc = 1280 / ww；int() 截断 + clamp
            ok = True
            for li in range(3):
                t = feat[li]
                _, _, hh, ww = t.shape
                sc = 1280.0 / ww
                x = int(min(max(cx / sc, 0), ww - 1))
                y = int(min(max(cy / sc, 0), hh - 1))
                v = float(t[0, cid, y, x])
                r[f"logit_L{li}"] = v
                r[f"prob_L{li}"] = float(1.0 / (1.0 + np.exp(-v)))
                r[f"xy_L{li}"] = [x, y]
                if not np.isfinite(v):
                    ok = False
            # native 尺寸：label 行（归一化）× native 像素
            if Wn and Hn and j < len(lrows):
                nw = float(lrows[j][3]) * Wn
                nh = float(lrows[j][4]) * Hn
                r["native_w"], r["native_h"] = nw, nh
                r["native_area"] = nw * nh
                r["native_sqrt"] = float(np.sqrt(max(nw * nh, 0.0)))
                # 交叉校验：归一化坐标应与 dataset 的 bb 一致
                dc = max(abs(float(lrows[j][1]) - float(bb[j][0])),
                         abs(float(lrows[j][2]) - float(bb[j][1])))
                r["coord_resid"] = float(dc)
                if dc > 1e-3:
                    r["coord_warn"] = True
            else:
                r["native_area"] = None
                r["drop_reason"] = "no_native_size_or_label_row"
            r["finite"] = bool(ok)
            rows.append(r)
        if (c + 1) % 100 == 0:
            print(f"    {c+1}/{len(idxs)}  images  GTs={len(rows)}  drops={len(drop)}", flush=True)

    print(f"\n  成功处理图像 = {n_img_ok} / {len(idxs)}；GT 行 = {len(rows)}；丢弃 = {len(drop)}")
    (OUT / "_rows.json").write_text(json.dumps(
        dict(checkpoint=str(CKPT), checkpoint_sha256=ck_sha, n_dataset=n_all,
             n_processed=len(idxs), n_images_ok=n_img_ok, n_gt=len(rows), drops=drop,
             overrides=OV, len_fixed=None, rows=rows),
        ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {OUT/'_rows.json'}")
    print(f"  checkpoint SHA after = {sha256(CKPT)}   {'OK 未变' if sha256(CKPT)==ck_sha else 'CHANGED ⚠'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
