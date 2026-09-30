"""_sge_supervision.py — native GT → 真实 assigner 监督质量（只读）

把 `_sge_instrument` 抓到的 **final GT 身份 (uid)** 与 V5 已建立的真实
`TaskAlignedAssigner` 插桩（topk=10, alpha=0.5, beta=6.0，**不改 assigner**）对接：
    final GT 顺序 === 传进 assigner 的 bbox/cls 顺序 === uid 顺序
（final 快照在 `Format.__call__` **之前**抓，此时 instances.bboxes 已是绝对 xyxy，
 `Format` 只做 xywh 转换与 denormalize（no-op），**不改变顺序、不丢弃任何 GT**。）

于是每个最终参与 loss 的 GT 都能回溯到它的 **native** 身份 ⇒ 可以按 native 尺寸分桶，
回答「§13 那些 healthy 的监督数字，是不是只对『活下来的』GT 成立」。

模型/权重与 V5 完全一致（同 checkpoint、同 nc、同 assigner 参数），保证与 §13 可比。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _sge_instrument as INSTR  # noqa: E402

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, yaml_model_load  # noqa: E402
from ultralytics.utils import yaml_load  # noqa: E402
from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors  # noqa: E402

OUT = Path(__file__).resolve().parent
SEPSTEM_YAML = ROOT / "configs/yolo11m_sepstem.yaml"
CKPT = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"
DS_YAML = ROOT / "data/processed/rgbid_split_train/dataset.yaml"
BASE_SEED = 20260928

D_HYP = dict(imgsz=1280, task="detect", rect=False, cache=False, single_cls=False,
             classes=None, fraction=1.0, channels=5, use_simotm="RGBID",
             object_scale_aug=False, mosaic=1.0, mixup=0.0, copy_paste=0.0,
             degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
             flipud=0.0, fliplr=0.5)


class InstAssigner(TaskAlignedAssigner):
    """插桩 `forward`：stash 真实返回值（含 target_scores）。不改任何计算。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.last = {}

    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        r = super().forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        tl, tb, ts, fg, tgi = r
        self.last.update(target_labels=tl, target_bboxes=tb, target_scores=ts,
                         fg_mask=fg, target_gt_idx=tgi)
        return r

    def get_pos_mask(self, *a, **kw):
        mp, am, ov = super().get_pos_mask(*a, **kw)
        self.last.update(mask_pos=mp, align_metric=am, overlaps=ov)
        return mp, am, ov


def build_model():
    cfg = yaml_model_load(str(SEPSTEM_YAML))
    model = DetectionModel(cfg, nc=12, verbose=False)
    w, _ = attempt_load_one_weight(str(CKPT))
    model.load(w)
    model.model[-1].stride = model.stride
    model.args = SimpleNamespace(cls_pw=None)
    return model


def assign_replay(model, loss, inst, img, bboxes, cls, stride):
    with torch.no_grad():
        preds = model(img)
    if isinstance(preds, tuple):
        preds = preds[1]
    feats = preds[: stride.size(0)]
    pd_d, pd_s = torch.cat([xi.view(1, loss.no, -1) for xi in feats], 2).split(
        (loss.reg_max * 4, loss.nc), 1)
    pd_s = pd_s.permute(0, 2, 1).contiguous()
    pd_d = pd_d.permute(0, 2, 1).contiguous()
    imgsz = torch.tensor(feats[0].shape[2:], dtype=pd_s.dtype) * stride[0]
    ap, st = make_anchors(feats, stride, 0.5)
    if len(bboxes) == 0:
        return []
    tg = torch.cat((torch.zeros(len(bboxes), 1), cls.view(-1, 1), bboxes), 1)
    t = loss.preprocess(tg, 1, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_l, gt_b = t.split((1, 4), 2)
    m_gt = gt_b.sum(2, keepdim=True).gt_(0)
    pbox = loss.bbox_decode(ap, pd_d)
    with torch.no_grad():
        inst(pd_s.detach().sigmoid(), (pbox.detach() * st).type(gt_b.dtype),
             ap * st, gt_l, gt_b, m_gt)
    L = inst.last
    n = int(m_gt[0].sum())
    out = []
    for g in range(n):
        cid = int(gt_l[0, g, 0])
        pos = L["mask_pos"][0, g].bool()
        am = L["align_metric"][0, g]
        ov = L["overlaps"][0, g]
        try:
            ts_g = L["target_scores"][0, :, cid]
        except Exception:  # noqa: BLE001
            ts_g = torch.zeros_like(am)
        r = dict(cls=cid, n_pos=int(pos.sum()),
                 best_align=float(am[pos].max()) if pos.any() else 0.0,
                 pos_target_max=float(ts_g[pos].max()) if pos.any() else 0.0,
                 pos_strides=sorted({round(float(s), 1) for s in st.flatten()[pos]}) if pos.any() else [])
        gb = gt_b[0, g]
        r["aug_w"] = float(gb[2] - gb[0])
        r["aug_h"] = float(gb[3] - gb[1])
        r["aug_area"] = r["aug_w"] * r["aug_h"]
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    print("=" * 78)
    print("SGE SUPERVISION — native GT → 真实 assigner 监督（只读）")
    print("=" * 78)
    INSTR.install()
    # 注意：这里**必须用真实图像**（模型需要像素）；只做 subsample

    model = build_model(); model.eval(); model.model[-1].train()
    loss = v8DetectionLoss(model)
    inst = InstAssigner(topk=10, num_classes=loss.nc, alpha=0.5, beta=6.0)
    stride = model.stride
    print(f"  ckpt={CKPT.name}  nc={loss.nc} topk=10 alpha=0.5 beta=6.0  stride={[float(s) for s in stride]}")

    data = yaml_load(str(DS_YAML))
    cfg = get_cfg(overrides=D_HYP)
    img_path = (Path(data["path"]) / data["train"])
    ds = build_yolo_dataset(cfg, str(img_path), 8, data, mode="train", use_simotm="RGBID",
                            pairs_rgb_ir=["visible", "infrared"], pairs_rgb_depth=["visible", "depth"])
    n = len(ds)
    idxs = list(range(0, n, max(1, n // args.n_images)))[: args.n_images]
    print(f"  数据集 n={n}；采样 {len(idxs)} 图 × {args.epochs} epoch")

    # native 元信息（用于分桶）
    nmeta = {}
    for i in range(n):
        lab = ds.labels[i]
        sh = lab.get("shape")
        if not sh:
            continue
        h0, w0 = float(sh[0]), float(sh[1])
        bb = np.asarray(lab.get("bboxes", np.zeros((0, 4))), dtype=np.float64)
        for k in range(len(lab.get("cls", []))):
            if k >= len(bb):
                break
            nmeta[INSTR.encode(i, k)] = dict(native_area=float(bb[k, 2] * w0 * bb[k, 3] * h0),
                                             layer=("HR" if h0 >= 1000 else "LR"))

    cap = {}
    def hook(name, labels, uid):
        if name == "final":
            cap["uid"] = np.asarray(uid).copy()
            cap["cls"] = np.asarray(labels["cls"], dtype=np.int64).copy()

    INSTR.set_on_stage(hook)
    rows = []
    for ep in range(args.epochs):
        print(f"  --- epoch {ep + 1}/{args.epochs} ---")
        for c, i in enumerate(idxs):
            cap.clear()
            random.seed(BASE_SEED + 104729 * ep + i)
            np.random.seed(BASE_SEED + 104729 * ep + i)
            try:
                lab = ds[i]
            except Exception as e:  # noqa: BLE001
                rows.append(dict(index=int(i), err=str(e)))
                continue
            uid = cap.get("uid")
            if uid is None or len(uid) != len(lab["cls"]):
                rows.append(dict(index=int(i), err="uid_missing_or_misaligned"))
                continue
            img = lab["img"].float() / 255.0
            if img.ndim == 3:
                img = img.unsqueeze(0)
            res = assign_replay(model, loss, inst, img, lab["bboxes"], lab["cls"], stride)
            if len(res) != len(uid):
                rows.append(dict(index=int(i), err=f"res_len {len(res)} != uid {len(uid)}"))
                continue
            for j, r in enumerate(res):
                u = int(uid[j])
                m = nmeta.get(u)
                r.update(index=int(i), epoch=int(ep), uid=u,
                         native_area=(None if m is None else m["native_area"]),
                         layer=(None if m is None else m["layer"]))
                rows.append(r)
            if (c + 1) % 50 == 0:
                print(f"    {c + 1}/{len(idxs)}  rows={len(rows)}")

    bad = [r for r in rows if "err" in r]
    good = [r for r in rows if "err" not in r]
    print(f"\n  记录 {len(good)} 条最终 GT；失败/跳过 {len(bad)}")
    (OUT / "_sge_supervision.json").write_text(json.dumps(dict(rows=rows, n_images=len(idxs),
                                                               epochs=args.epochs, base_seed=BASE_SEED),
                                                          indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {OUT / '_sge_supervision.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
