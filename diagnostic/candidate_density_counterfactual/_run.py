"""_run.py — Candidate-Density Counterfactual Audit：真实 pipeline 的**逐候选**抽取（只读）

只回答一个问题：
    native-small 的低 maxIoU / low alignment，主要是 candidate **数量不足**造成的
    sampling effect，还是 small 自身 candidate geometry / quality 较差？

READ-ONLY：不训练 / 不 backward / 不 optimizer.step / 不改任何既有源码/配置/checkpoint /
不 inference 提交。只做 `torch.no_grad()` 的只读 forward + 真实 assigner 的**只读**前向，
并用 runtime monkey-patch 抓中间量（不改磁盘上任何文件）。

复用的真实组件（全部为运行中的真实对象）：
    D′ checkpoint  runs/..._sepstem_clahe/weights/best.pt
    真实 dataset   data/processed/rgbid_split_train（经 build_yolo_dataset，与训练同路径）
    真实 assigner  TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)，未修改
    真实 CIoU      TaskAlignedAssigner.iou_calculation → bbox_iou(..., CIoU=True).clamp_(0)

保存**逐候选**原始量，使任何后续反事实都可离线复核：
    每个 candidate: CIoU / cls score / align(=cls^0.5*CIoU^6) / stride / 相对 GT 中心的归一化偏移
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diagnostic/small_gt_exposure"))  # 只读复用 lineage 插桩

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


class ProbeAssigner(TaskAlignedAssigner):
    """插桩：stash 全流程中间量。**不改变任何计算**（get_box_metrics 为父类逐行副本）。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.last = {}

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self.select_topk_candidates(
            align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        self.last.update(mask_in_gts=mask_in_gts, mask_topk=mask_topk, mask_pos=mask_pos,
                         align_metric=align_metric, overlaps=overlaps)
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """父类逐行副本 + 留下 bbox_scores。⚠ `mask_gt = mask_gt.bool()` 不能删。"""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)
        self.last["bbox_scores"] = bbox_scores
        return bbox_scores.pow(self.alpha) * overlaps.pow(self.beta), overlaps


def build_model():
    cfg = yaml_model_load(str(SEPSTEM_YAML))
    model = DetectionModel(cfg, nc=12, verbose=False)
    w, _ = attempt_load_one_weight(str(CKPT))
    model.load(w)
    model.model[-1].stride = model.stride
    model.args = SimpleNamespace(cls_pw=None)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    print("=" * 100)
    print("CANDIDATE-DENSITY COUNTERFACTUAL — 逐候选抽取（只读）")
    print("=" * 100)

    prov = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ("ultralytics/utils/tal.py", "ultralytics/utils/loss.py", "ultralytics/utils/instance.py",
             "ultralytics/data/augment.py", "ultralytics/data/base.py", "ultralytics/data/build.py",
             "ultralytics/data/dataset.py", "configs/yolo11m_sepstem.yaml",
             "configs/train_rgbid_sepstem_clahe.yaml", str(CKPT.relative_to(ROOT)))}
    print("\n[provenance]")
    for k, v in prov.items():
        print(f"  {v[:16]}  {k}")

    model = build_model(); model.eval(); model.model[-1].train()
    loss = v8DetectionLoss(model)
    a = loss.assigner
    print(f"\n[real assigner] topk={a.topk} alpha={a.alpha} beta={a.beta} nc={a.num_classes}")
    sig = str(inspect.signature(TaskAlignedAssigner.__init__))
    print(f"  __init__ {sig}")
    assert (a.topk, a.alpha, a.beta) == (10, 0.5, 6.0)
    inst = ProbeAssigner(topk=a.topk, num_classes=a.num_classes, alpha=a.alpha, beta=a.beta)
    ref = TaskAlignedAssigner(topk=a.topk, num_classes=a.num_classes, alpha=a.alpha, beta=a.beta)
    stride = model.stride

    data = yaml_load(str(DS_YAML))
    ds = build_yolo_dataset(get_cfg(overrides=D_HYP), str(Path(data["path"]) / data["train"]),
                            8, data, mode="train", use_simotm="RGBID",
                            pairs_rgb_ir=["visible", "infrared"], pairs_rgb_depth=["visible", "depth"])
    n = len(ds)
    idxs = list(range(0, n, max(1, n // args.n_images)))[: args.n_images]
    print(f"  dataset n={n}；采样 {len(idxs)} 图 × {args.epochs} epoch（与 E1/E2 同 seed 协议）")

    nmeta = {}
    for i in range(n):
        lab = ds.labels[i]; sh = lab.get("shape")
        if not sh:
            continue
        h0, w0 = float(sh[0]), float(sh[1])
        bb = np.asarray(lab.get("bboxes", np.zeros((0, 4))), dtype=np.float64)
        for k in range(len(lab.get("cls", []))):
            if k >= len(bb):
                break
            nmeta[INSTR.encode(i, k)] = (float(bb[k, 2] * w0 * bb[k, 3] * h0), float(bb[k, 2] * w0), float(bb[k, 3] * h0))

    INSTR.install()
    cap = {}
    def hook(name, labels, uid):
        if name == "final":
            cap["uid"] = np.asarray(uid).copy()
    INSTR.set_on_stage(hook)

    # 逐 GT / 逐候选累加器（扁平存储，避免 per-uid 小对象造成 GC 压力）
    G = {k: [] for k in ("uid", "native_area", "bw", "bh", "ncand", "maxciou", "maxcls", "maxalign", "zero")}
    C = {k: [] for k in ("gt", "ciou", "cls", "align", "stride", "offx", "offy")}
    n_samples = 0
    probe_ok = None

    for ep in range(args.epochs):
        print(f"  --- epoch {ep + 1}/{args.epochs} ---")
        for c, i in enumerate(idxs):
            cap.clear()
            random.seed(BASE_SEED + 104729 * ep + i)
            np.random.seed(BASE_SEED + 104729 * ep + i)
            try:
                lab = ds[i]
            except Exception:  # noqa: BLE001
                continue
            uid = cap.get("uid")
            if uid is None or len(uid) != len(lab["cls"]) or len(lab["bboxes"]) == 0:
                continue
            img = lab["img"].float().div(255)
            if img.ndim == 3:
                img = img.unsqueeze(0)
            with torch.no_grad():
                preds = model(img)
            if isinstance(preds, tuple):
                preds = preds[1]
            feats = preds[: stride.size(0)]
            pd_d, pd_s = torch.cat([x.view(1, loss.no, -1) for x in feats], 2).split(
                (loss.reg_max * 4, loss.nc), 1)
            pd_s = pd_s.permute(0, 2, 1).contiguous()
            pd_d = pd_d.permute(0, 2, 1).contiguous()
            imgsz = torch.tensor(feats[0].shape[2:], dtype=pd_s.dtype) * stride[0]
            ap, st = make_anchors(feats, stride, 0.5)
            tg = torch.cat((torch.zeros(len(lab["bboxes"]), 1), lab["cls"].view(-1, 1), lab["bboxes"]), 1)
            t = loss.preprocess(tg, 1, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_l, gt_b = t.split((1, 4), 2)
            m_gt = gt_b.sum(2, keepdim=True).gt_(0)
            pbox = loss.bbox_decode(ap, pd_d)
            in_s = pd_s.detach().sigmoid()
            in_b = (pbox.detach() * st).type(gt_b.dtype)
            in_p = ap * st

            with torch.no_grad():
                if probe_ok is None:
                    r_ref = ref.forward(in_s, in_b, in_p, gt_l, gt_b, m_gt)
                    r_prb = inst.forward(in_s, in_b, in_p, gt_l, gt_b, m_gt)
                    probe_ok = bool(all(torch.equal(x.float(), y.float()) for x, y in zip(r_ref, r_prb)))
                    print(f"  [SELF-CHECK] ProbeAssigner vs 父类逐位一致: {probe_ok}")
                    assert probe_ok, "插桩与父类不一致 —— STOP"
                else:
                    inst.forward(in_s, in_b, in_p, gt_l, gt_b, m_gt)

            L = inst.last
            mig = L["mask_in_gts"][0].cpu().numpy()          # (n_gt, 8400) bool
            ov = L["overlaps"][0].cpu().numpy()
            bs_ = L["bbox_scores"][0].cpu().numpy()
            al = L["align_metric"][0].cpu().numpy()
            stn = st.cpu().numpy().reshape(-1)   # make_anchors 返回列向量 (N,1)，必须展平
            pn = in_p.cpu().numpy()
            gb = gt_b[0].cpu().numpy()
            ngt = int(m_gt[0].sum())
            for g in range(ngt):
                u = int(uid[g])
                meta = nmeta.get(u)
                if meta is None:
                    continue
                area, bw, bh = meta
                idx = np.flatnonzero(mig[g])
                gi = len(G["uid"])
                G["uid"].append(u); G["native_area"].append(area)
                G["bw"].append(bw); G["bh"].append(bh)
                G["ncand"].append(int(idx.size))
                if idx.size == 0:
                    G["maxciou"].append(0.0); G["maxcls"].append(0.0)
                    G["maxalign"].append(0.0); G["zero"].append(True)
                    continue
                ci = ov[g][idx]; cl = bs_[g][idx]; aa = al[g][idx]
                G["maxciou"].append(float(ci.max())); G["maxcls"].append(float(cl.max()))
                G["maxalign"].append(float(aa.max())); G["zero"].append(False)
                cx, cy = (gb[g][0] + gb[g][2]) / 2.0, (gb[g][1] + gb[g][3]) / 2.0
                hw = max((gb[g][2] - gb[g][0]) / 2.0, 1e-6)
                hh = max((gb[g][3] - gb[g][1]) / 2.0, 1e-6)
                C["gt"].extend([gi] * idx.size)
                C["ciou"].extend(ci.tolist()); C["cls"].extend(cl.tolist())
                C["align"].extend(aa.tolist()); C["stride"].extend(stn[idx].tolist())
                C["offx"].extend(((pn[idx, 0] - cx) / hw).tolist())
                C["offy"].extend(((pn[idx, 1] - cy) / hh).tolist())
            n_samples += 1
            if (c + 1) % 50 == 0:
                print(f"    {c + 1}/{len(idxs)}  GTs={len(G['uid'])}  cands={len(C['gt'])}")

    print(f"\n  样本 {n_samples}；GT {len(G['uid'])}；候选 {len(C['gt'])}")
    np.savez_compressed(
        OUT / "_results.npz",
        **{f"gt_{k}": np.asarray(v, dtype=np.float64 if k in
                                ("native_area", "bw", "bh", "maxciou", "maxcls", "maxalign") else np.int64)
           for k, v in G.items()},
        **{f"cand_{k}": np.asarray(v, dtype=np.float64 if k not in ("gt",) else np.int64)
           for k, v in C.items()},
    )
    (OUT / "_meta.json").write_text(json.dumps(dict(
        provenance=prov, real_assigner=dict(topk=a.topk, alpha=a.alpha, beta=a.beta, nc=a.num_classes),
        probe_selfcheck_pass=probe_ok, n_images=len(idxs), epochs=args.epochs, base_seed=BASE_SEED,
        n_samples=n_samples, n_gt=len(G["uid"]), n_cand=len(C["gt"]),
        io="真实 pipeline：D′ ckpt + 真实 dataset + 真实 TaskAlignedAssigner + 真实 CIoU；"
           "candidate = anchor 中心严格落在 GT 框内（mask_in_gts）",
        note="逐候选保存 CIoU/cls/align/stride/归一化偏移，使任何反事实可离线复核；"
             "未修改任何既有文件。"),
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {OUT/'_results.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
