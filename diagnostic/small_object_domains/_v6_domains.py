"""_v6_domains.py — T1 / T2 / V1 三域对照（同一 frozen D′ checkpoint，同一度量函数）。

只读。只碰 train/val。0 训练 / 0 改模型-assigner-loss-增强 / 0 碰 test-submission-online。

三域：
  T1  train 原图，augmentation OFF      build_yolo_dataset(mode="val") over **train** images
  T2  train 真实训练 pipeline（mosaic ON）  build_yolo_dataset(mode="train")  —— 与 V5 同一条
  V1  val  原图，augmentation OFF        build_yolo_dataset(mode="val") over val images

T1 与 V1 用**完全相同的预处理**（LetterBox+Format，无增强），只换 split ⇒ 分离「domain」效应
T1 与 T2 用**同一个 split**，只换 augmentation ⇒ 分离「augmentation」效应

所有域共用同一个 `measure()`，保证指标口径逐字一致。
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
from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors  # noqa: E402

OUT = Path(__file__).resolve().parent
SEPSTEM_YAML = ROOT / "configs/yolo11m_sepstem.yaml"
CKPT = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"
V2 = ROOT / "diagnostic/small_object_cause_v2"
SEED = 20260927
N_TRAIN_IMG = 250


class InstAssigner(TaskAlignedAssigner):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.last = {}

    def forward(self, *a, **kw):
        r = super().forward(*a, **kw)
        self.last.update(target_scores=r[2])
        return r

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, eps=1e-9):
        r = super().select_candidates_in_gts(xy_centers, gt_bboxes, eps)
        self.last["mask_in_gts"] = r
        return r

    def select_topk_candidates(self, m, largest=True, topk_mask=None):
        r = super().select_topk_candidates(m, largest=largest, topk_mask=topk_mask)
        self.last["mask_topk"] = r
        return r

    def get_pos_mask(self, *a, **kw):
        mp, am, ov = super().get_pos_mask(*a, **kw)
        self.last.update(mask_pos=mp, align_metric=am, overlaps=ov)
        return mp, am, ov


def build_model():
    cfg = yaml_model_load(str(SEPSTEM_YAML))
    m = DetectionModel(cfg, nc=12, verbose=False)
    w, _ = attempt_load_one_weight(str(CKPT))
    m.load(w)
    m.model[-1].stride = m.stride
    from types import SimpleNamespace
    m.args = SimpleNamespace(cls_pw=None)
    return m


BINS = ((0, 12, "<12"), (12, 18, "12-18"), (18, 24, "18-24"),
        (24, 32, "24-32"), (32, 1024, "32-32^2(=small)"), (1024, 9216, "medium"))


def binof(s):
    for lo, hi, lab in BINS:
        if lo <= s < hi:
            return lab
    return None


class Measurer:
    def __init__(self, model):
        self.model = model
        self.loss = v8DetectionLoss(model)
        self.inst = InstAssigner(topk=10, num_classes=self.loss.nc, alpha=0.5, beta=6.0)
        self.stride = model.stride
        self.feat = {}
        for li in (23, 9):
            model.model[li].register_forward_hook(self._hook(li))

    def _hook(self, li):
        def h(m, i, o):
            if torch.is_tensor(o):
                self.feat[li] = o.detach()
        return h

    def measure(self, img_np_uint8, bboxes, cls):
        """img_np_uint8: (C,H,W) uint8 tensor from dataset; bboxes: normalized xywh."""
        img = img_np_uint8.float() / 255.0
        img = img.unsqueeze(0) if img.ndim == 3 else img
        self.feat.clear()
        with torch.no_grad():
            preds = self.model(img)
        feats = (preds[1] if isinstance(preds, tuple) else preds)[: self.stride.size(0)]
        pd_d, pd_s = torch.cat([x.view(1, self.loss.no, -1) for x in feats], 2).split(
            (self.loss.reg_max * 4, self.loss.nc), 1)
        pd_s = pd_s.permute(0, 2, 1).contiguous()
        pd_d = pd_d.permute(0, 2, 1).contiguous()
        imgsz = torch.tensor(feats[0].shape[2:], dtype=pd_s.dtype) * self.stride[0]
        ap, st = make_anchors(feats, self.stride, 0.5)
        if len(bboxes) == 0:
            return []
        tg = torch.cat((torch.zeros(len(bboxes), 1), cls.view(-1, 1), bboxes), 1)
        t = self.loss.preprocess(tg, 1, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_l, gt_b = t.split((1, 4), 2)
        m_gt = gt_b.sum(2, keepdim=True).gt_(0)
        pbox = self.loss.bbox_decode(ap, pd_d)
        with torch.no_grad():
            self.inst(pd_s.detach().sigmoid(), (pbox.detach() * st).type(gt_b.dtype),
                      ap * st, gt_l, gt_b, m_gt)
        L = self.inst.last
        sc_all = pd_s[0].sigmoid()                     # (na, nc)
        apx = ap * st
        recs = []
        n = int(m_gt[0].sum())
        for g in range(n):
            cid = int(gt_l[0, g, 0])
            pos = L["mask_pos"][0, g].bool()
            am = L["align_metric"][0, g]; ov = L["overlaps"][0, g]
            tsc = L["target_scores"][0, :, cid]
            gb = gt_b[0, g]
            w = float(gb[2] - gb[0]); h = float(gb[3] - gb[1])
            sq = float(np.sqrt(max(w * h, 0)))
            c = (gb[:2] + gb[2:]) / 2
            # GT-center same-class 响应：半径 1.5*stride8 = 12px 内最高同类 logit
            d = torch.linalg.norm(apx - c[None, :], dim=1)
            near = d <= 12.0
            logit = float(pd_s[0, near, cid].max()) if near.any() else float("nan")
            prob = float(sc_all[near, cid].max()) if near.any() else float("nan")
            r = dict(cls=cid, sq=sq, bin=binof(sq),
                     center_logit=logit, center_prob=prob,
                     n_pos=int(pos.sum()),
                     best_align=float(am[pos].max()) if pos.any() else 0.0,
                     best_assign_iou=float(ov[pos].max()) if pos.any() else 0.0,
                     pos_prob=float(sc_all[pos, cid].max()) if pos.any() else 0.0,
                     pos_target=float(tsc[pos].max()) if pos.any() else 0.0)
            for li, key in ((23, "P3"), (9, "P2b")):
                f = self.feat.get(li)
                if f is None:
                    continue
                ff = f[0]
                fh, fw = ff.shape[1:]
                fx, fy = float(c[0]) / (1280 / fw), float(c[1]) / (1280 / fh)
                x0, x1 = max(0, int(fx) - 1), min(fw, int(fx) + 2)
                y0, y1 = max(0, int(fy) - 1), min(fh, int(fy) + 2)
                pat = ff[:, y0:y1, x0:x1]
                if pat.numel():
                    cn = float(pat.norm(dim=0).mean()); gn = float(ff.norm(dim=0).mean())
                    r[f"{key}_center"] = cn; r[f"{key}_ratio"] = cn / (gn + 1e-9)
            recs.append(r)
        return recs


def build_ds(train_images: bool, augment: bool, img_path_override=None):
    data = yaml_load(str(ROOT / "data/processed/rgbid_split_train/dataset.yaml"))
    if augment:
        ov = dict(imgsz=1280, task="detect", rect=False, cache=False, single_cls=False,
                  classes=None, fraction=1.0, channels=5, use_simotm="RGBID",
                  object_scale_aug=False, mosaic=1.0, mixup=0.0, copy_paste=0.0,
                  degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
                  flipud=0.0, fliplr=0.5)
        mode = "train"
    else:
        ov = dict(imgsz=1280, task="detect", rect=False, cache=False, single_cls=False,
                  classes=None, fraction=1.0, channels=5, use_simotm="RGBID",
                  mosaic=0.0, copy_paste=0.0, mixup=0.0, object_scale_aug=False, augment=False)
        mode = "val"
    cfg = get_cfg(overrides=ov)
    sub = data["train"] if train_images else data["val"]
    p = (img_path_override or Path(data["path"]) / sub).resolve()
    return build_yolo_dataset(cfg, str(p), 8, data, mode=mode, use_simotm="RGBID",
                              pairs_rgb_ir=["visible", "infrared"],
                              pairs_rgb_depth=["visible", "depth"]), data


def run_domain(name, ds, M, idxs=None, keys_of=None):
    if idxs is None:
        idxs = range(len(ds))
    rows = []
    for i in idxs:
        random.seed(SEED + i); np.random.seed(SEED + i)
        lab = ds[i]
        bb = lab.get("bboxes"); cl = lab.get("cls")
        if bb is None or len(bb) == 0:
            continue
        recs = M.measure(lab["img"], bb, cl)
        stem = Path(ds.im_files[i]).stem
        for j, r in enumerate(recs):
            r["domain"] = name
            r["stem"] = stem
            r["gt"] = j
            r["key"] = f"{stem}#{j}"
            rows.append(r)
    print(f"  [{name}] {len(rows)} GT 记录")
    return rows


def main():
    print("=" * 78)
    print("T1 / T2 / V1 三域对照（同一 frozen D′，同一 measure()）")
    print("=" * 78)
    model = build_model(); model.eval(); model.model[-1].train()
    M = Measurer(model)
    print(f"  seed={SEED}  topk={M.inst.topk}  stride={[float(s) for s in M.stride]}")

    ds_tr_noaug, data = build_ds(train_images=True, augment=False)
    ds_tr_aug, _ = build_ds(train_images=True, augment=True)
    ds_va, _ = build_ds(train_images=False, augment=False)
    print(f"  T1 dataset n={len(ds_tr_noaug)} (train 图, 无增强)")
    print(f"  T2 dataset n={len(ds_tr_aug)} (train 图, 真实 pipeline)")
    print(f"  V1 dataset n={len(ds_va)} (val 图, 无增强)")

    step = max(1, len(ds_tr_noaug) // N_TRAIN_IMG)
    t_idx = list(range(0, len(ds_tr_noaug), step))[:N_TRAIN_IMG]

    allrows = []
    allrows += run_domain("T1", ds_tr_noaug, M, idxs=t_idx)
    allrows += run_domain("T2", ds_tr_aug, M, idxs=t_idx)
    allrows += run_domain("V1", ds_va, M)

    (OUT / "_v6_domains.json").write_text(json.dumps(allrows, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(f"[saved] {OUT/'_v6_domains.json'}  ({len(allrows)} 条)")


if __name__ == "__main__":
    main()
