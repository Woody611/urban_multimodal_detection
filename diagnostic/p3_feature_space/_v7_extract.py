"""_v7_extract.py — P3 feature-space diagnostic：抽取 GT-center 的**完整特征向量**。

只读。只碰 train/val。0 训练 / 0 改模型-head-loss-assigner-增强 / 0 碰 test-submission-online。

抽取两个层级的完整向量（同一 layer、同一坐标映射、同一采样）：
  P3vec  : layer 23 输出（P3 颈特征，256ch @160×160 for 1280 输入）
  Hvec   : Detect.cv3[0][-1]（1×1 Conv2d 12×256）的**输入** —— 分类 logit 的直接前置特征

为什么用 Hvec 做 head 几何：YOLO11 的 cv3 是
  Sequential( DWConv+Conv , DWConv+Conv , Conv2d(256,nc,1) )
因此 **class logit = W_c · h + b_c 是精确等式**（h = cv3[0][-1] 的输入），
而 **不是** W·(P3 特征) —— 后者会漏掉两层 DWConv/Conv，是伪公式。本脚本只用前者。
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "diagnostic/small_object_cause_v2"))

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, yaml_model_load  # noqa: E402
from ultralytics.utils import yaml_load  # noqa: E402

OUT = Path(__file__).resolve().parent
CKPT = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"
V2 = ROOT / "diagnostic/small_object_cause_v2"
SEED = 20260928
N_TRAIN_IMG = 300
BINS = ((0, 12, "<12"), (12, 18, "12-18"), (18, 24, "18-24"), (24, 32, "24-32"))


def binof(s):
    for lo, hi, lab in BINS:
        if lo <= s < hi:
            return lab
    return ">=32"


class Feat:
    def __init__(self, model):
        self.m = model
        self.cur = {}
        model.model[23].register_forward_hook(self._out("P3"))
        last = model.model[-1].cv3[0][-1]
        last.register_forward_pre_hook(self._pre("H"))
        self.W = last.weight.detach().clone()      # (nc,256,1,1)
        self.b = last.bias.detach().clone()        # (nc,)

    def _out(self, k):
        def h(m, i, o):
            if torch.is_tensor(o):
                self.cur[k] = o.detach()
        return h

    def _pre(self, k):
        def h(m, inp):
            self.cur[k] = inp[0].detach()
        return h

    def sample(self, t, cx, cy):
        """按输入空间像素坐标取该 cell 的完整向量 -> (C,)"""
        ch, hh, ww = t.shape[1:]
        sc = 1280.0 / ww
        x = int(min(max(cx / sc, 0), ww - 1))
        y = int(min(max(cy / sc, 0), hh - 1))
        return t[0, :, y, x].numpy().astype(np.float32)


def build_ds(train_images: bool, augment: bool):
    data = yaml_load(str(ROOT / "data/processed/rgbid_split_train/dataset.yaml"))
    ov = dict(imgsz=1280, task="detect", rect=False, cache=False, single_cls=False, classes=None,
              fraction=1.0, channels=5, use_simotm="RGBID", object_scale_aug=False,
              mosaic=1.0 if augment else 0.0, mixup=0.0, copy_paste=0.0,
              degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
              flipud=0.0, fliplr=0.5)
    if not augment:
        ov["augment"] = False
    cfg = get_cfg(overrides=ov)
    sub = data["train"] if train_images else data["val"]
    return build_yolo_dataset(cfg, str((Path(data["path"]) / sub).resolve()), 8, data,
                              mode="train" if augment else "val", use_simotm="RGBID",
                              pairs_rgb_ir=["visible", "infrared"],
                              pairs_rgb_depth=["visible", "depth"]), data


def main():
    print("=" * 78)
    print("P3 feature-space 抽取（只读）")
    print("=" * 78)
    cfg = yaml_model_load(str(ROOT / "configs/yolo11m_sepstem.yaml"))
    model = DetectionModel(cfg, nc=12, verbose=False)
    w, _ = attempt_load_one_weight(str(CKPT))
    model.load(w); model.model[-1].stride = model.stride
    model.eval(); model.model[-1].train()
    F = Feat(model)
    print(f"  W shape={tuple(F.W.shape)}  b shape={tuple(F.b.shape)}  seed={SEED}")

    rows = []
    for dom, tr, aug in (("T1", True, False), ("V1", False, False)):
        ds, data = build_ds(tr, aug)
        idxs = (list(range(0, len(ds), max(1, len(ds) // N_TRAIN_IMG)))[:N_TRAIN_IMG]
                if dom == "T1" else range(len(ds)))
        n0 = len(rows)
        for i in idxs:
            random.seed(SEED + i); np.random.seed(SEED + i)
            lab = ds[i]
            bb, cl = lab.get("bboxes"), lab.get("cls")
            if bb is None or len(bb) == 0:
                continue
            img = lab["img"].float().div(255.0)
            img = img.unsqueeze(0) if img.ndim == 3 else img
            F.cur.clear()
            with torch.no_grad():
                model(img)
            P3 = F.cur.get("P3"); H = F.cur.get("H")
            if P3 is None or H is None:
                continue
            stem = Path(ds.im_files[i]).stem
            for j in range(len(bb)):
                cxw, cyw, ww_, hh_ = [float(v) for v in bb[j]]
                cx, cy = cxw * 1280, cyw * 1280
                wpx, hpx = ww_ * 1280, hh_ * 1280
                sq = float(np.sqrt(max(wpx * hpx, 0)))
                cid = int(cl[j])
                p3v = F.sample(P3, cx, cy)
                hv = F.sample(H, cx, cy)
                logit = float(F.W[cid].view(-1) @ torch.from_numpy(hv) + F.b[cid])
                rows.append(dict(
                    domain=dom, stem=stem, gt=j, key=f"{stem}#{j}", cls=cid, sq=sq, bin=binof(sq),
                    center=([round(cx, 2), round(cy, 2)]),
                    logit_decomp=logit,
                    p3=p3v.tolist(), h=hv.tolist()))
        print(f"  [{dom}] {len(rows)-n0} GT")

    (OUT / "_v7_vectors.json").write_text(json.dumps(dict(rows=rows, seed=SEED),
                                                     ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {OUT/'_v7_vectors.json'}  ({len(rows)} GT)")


if __name__ == "__main__":
    main()
