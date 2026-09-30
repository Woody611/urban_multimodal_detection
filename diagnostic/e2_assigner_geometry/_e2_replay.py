"""_e2_replay.py — E2 前置可行性审计：assigner 几何的**精确**离线 replay（只读）

禁止：训练 / optimizer.step / backward / 改任何既有源码或配置 / 改 checkpoint /
      test inference / submission。本脚本只做 `torch.no_grad()` 的只读 forward 与
      assigner 的离线重算，且**不修改 assigner 源码**。

FULL_ASSIGNER_REPLAY = TRUE
    做法：`topk / alpha / beta` 是 `TaskAlignedAssigner` 的**实例属性**，在 `get_pos_mask`
    与 `select_topk_candidates` 内部被读取。因此在**完全相同的输入张量**上修改这三个属性并
    重新调用 `_forward()`，等价于「用该组超参跑一次完整 assigner」——
    包含 candidate selection（anchor 中心是否落在 GT 内）→ top-k selection → 冲突消解
    （`select_highest_overlaps`）→ target_scores 归一化的**全流程**。
    不是「只重算 alignment 公式」。

真实参数（从运行代码读出，不采信记忆）：
    loss.py:438  TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
    v8DetectionLoss.__init__(model, tal_topk=10)  ⇒  topk=10, alpha=0.5, beta=6.0
    align_metric = bbox_scores ** alpha * overlaps ** beta
    iou_calculation = bbox_iou(..., CIoU=True).clamp_(0)   # ← 是 CIoU，不是纯 IoU
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import sys
from collections import defaultdict
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

ALPHAS = (0.25, 0.50, 0.75)
BETAS = (2, 4, 6)
TOPKS = (5, 10, 20)
BASELINE = (0.50, 6, 10)

D_HYP = dict(imgsz=1280, task="detect", rect=False, cache=False, single_cls=False,
             classes=None, fraction=1.0, channels=5, use_simotm="RGBID",
             object_scale_aug=False, mosaic=1.0, mixup=0.0, copy_paste=0.0,
             degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
             flipud=0.0, fliplr=0.5)


class ProbeAssigner(TaskAlignedAssigner):
    """插桩：stash 全流程中间量。**不改变任何计算**。"""

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
        # 必须用 update 而非重新赋值：get_box_metrics 已在 self.last 里放了 bbox_scores，
        # 重新赋值会把它抹掉（首版即因此 KeyError: 'bbox_scores'）。
        self.last.update(mask_in_gts=mask_in_gts, mask_topk=mask_topk, mask_pos=mask_pos,
                         align_metric=align_metric, overlaps=overlaps)
        return mask_pos, align_metric, overlaps

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """包一层以留下**冲突消解之后**的 fg_mask / target_gt_idx（真实进 loss 的正样本）。"""
        r = super()._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        self.last["fg_mask"] = r[3]
        self.last["target_gt_idx"] = r[4]
        return r

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """**逐行**复刻父类实现，额外把 bbox_scores 留下（父类不返回它）。

        ⚠ 必须保留父类的第一行 `mask_gt = mask_gt.bool()`。首版因早前用
        `grep -vE` 读取源码时把这一行过滤掉了，导致 mask_gt 以 float32 传入并被用作
        索引 → `IndexError: tensors used as indices must be long, int, byte or bool`。
        这是**读源码时的过滤失误**，不是 assigner 的问题。
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w      <-- 父类原有，勿删
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


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    print("=" * 100)
    print("E2 PRE-FEASIBILITY — assigner 几何精确 replay（只读；不训练/不改 assigner/不改配置）")
    print("=" * 100)

    # ---------------- §11 provenance ----------------
    prov = {p: sha(p) for p in
            ("ultralytics/utils/tal.py", "ultralytics/utils/loss.py", "ultralytics/utils/instance.py",
             "ultralytics/data/augment.py", "ultralytics/data/base.py", "ultralytics/data/build.py",
             "ultralytics/data/dataset.py", "configs/yolo11m_sepstem.yaml",
             "configs/train_rgbid_sepstem_clahe.yaml", str(CKPT.relative_to(ROOT)))}
    print("\n[provenance]")
    for k, v in prov.items():
        print(f"  {v[:16]}  {k}")

    model = build_model(); model.eval(); model.model[-1].train()
    loss = v8DetectionLoss(model)

    # ---------------- §2 从运行代码读出真实 assigner 参数 ----------------
    a = loss.assigner
    print("\n[real assigner — read from the running object, not from memory]")
    print(f"  type      = {type(a).__name__}")
    print(f"  topk      = {a.topk}")
    print(f"  alpha     = {a.alpha}")
    print(f"  beta      = {a.beta}")
    print(f"  num_classes = {a.num_classes}   eps = {a.eps}")
    sig = inspect.signature(TaskAlignedAssigner.__init__)
    print(f"  __init__ signature = {sig}")
    print(f"  loss.py instantiation = {inspect.getsource(v8DetectionLoss.__init__).splitlines()[0].strip()}")
    assert (a.topk, a.alpha, a.beta) == (10, 0.5, 6.0), "assigner 参数与预期不符，STOP"
    print("  -> (topk, alpha, beta) == (10, 0.5, 6.0)  与 §2 已知一致 ✓")

    # 真实公式与 IoU 实现（打印源码行以证）
    tal_src = Path(ROOT / "ultralytics/utils/tal.py").read_text(encoding="utf-8")
    f_align = [l.strip() for l in tal_src.splitlines() if "align_metric = bbox_scores.pow" in l]
    f_iou = [l.strip() for l in tal_src.splitlines() if "def iou_calculation" in l or "CIoU=True" in l]
    print(f"  align metric (源码) = {f_align}")
    print(f"  iou (源码)         = {f_iou}")

    inst = ProbeAssigner(topk=a.topk, num_classes=a.num_classes, alpha=a.alpha, beta=a.beta)
    ref = TaskAlignedAssigner(topk=a.topk, num_classes=a.num_classes, alpha=a.alpha, beta=a.beta)
    _probe_checked = {"done": False}
    stride = model.stride

    # 安装 native→final 身份插桩（只读复用 small_gt_exposure 的实现）。
    # 缺这一步时 on_stage 永远不会被调用 ⇒ 全部样本因 uid 缺失被跳过（首版即因此得到 0 条）。
    INSTR.install()
    print(f"  lineage 插桩已安装: {INSTR.STATS}")

    data = yaml_load(str(DS_YAML))
    ds = build_yolo_dataset(get_cfg(overrides=D_HYP), str(Path(data["path"]) / data["train"]),
                            8, data, mode="train", use_simotm="RGBID",
                            pairs_rgb_ir=["visible", "infrared"], pairs_rgb_depth=["visible", "depth"])
    n = len(ds)
    idxs = list(range(0, n, max(1, n // args.n_images)))[: args.n_images]
    print(f"\n  dataset n={n}；采样 {len(idxs)} 图 × {args.epochs} epoch（与 Exposure Audit 同协议同 seed）")

    # native 元信息
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
            nmeta[INSTR.encode(i, k)] = float(bb[k, 2] * w0 * bb[k, 3] * h0)

    cap = {}
    def hook(name, labels, uid):
        if name == "final":
            cap["uid"] = np.asarray(uid).copy()
    INSTR.set_on_stage(hook)

    settings = [(al, be, tk) for al in ALPHAS for be in BETAS for tk in TOPKS]
    # AG[setting][uid] = (n_pos, best_align, zero_pos)
    AG = {s: {} for s in settings}
    # 分解量（与设置无关的候选几何）；CAND[uid] = dict(n_cand,max_iou,max_cls, native_area)
    CAND = {}
    # 候选级 max align 随 (alpha,beta) 变化（topk 无关）
    CAND_ALIGN = {(al, be): {} for al in ALPHAS for be in BETAS}

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
            if uid is None or len(uid) != len(lab["cls"]):
                continue
            img = lab["img"].float() / 255.0
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
            if len(lab["bboxes"]) == 0:
                continue
            tg = torch.cat((torch.zeros(len(lab["bboxes"]), 1), lab["cls"].view(-1, 1), lab["bboxes"]), 1)
            t = loss.preprocess(tg, 1, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_l, gt_b = t.split((1, 4), 2)
            m_gt = gt_b.sum(2, keepdim=True).gt_(0)
            pbox = loss.bbox_decode(ap, pd_d)
            in_s = pd_s.detach().sigmoid()
            in_b = (pbox.detach() * st).type(gt_b.dtype)
            in_p = ap * st
            # ---- 自检：ProbeAssigner 必须与**父类**在 baseline 参数下逐位一致 ----
            if not _probe_checked["done"]:
                with torch.no_grad():
                    r_ref = ref.forward(in_s, in_b, in_p, gt_l, gt_b, m_gt)
                r_prb = inst.forward(in_s, in_b, in_p, gt_l, gt_b, m_gt)
                same = all(torch.equal(x.float(), y.float()) for x, y in zip(r_ref, r_prb))
                print(f"  [SELF-CHECK] ProbeAssigner vs 父类逐位一致: {same}  "
                      f"(n_gt={int(m_gt[0].sum())}, fg={int(r_ref[3].sum())})")
                assert same, "ProbeAssigner 与父类不一致 —— 复刻有误，STOP"
                _probe_checked["done"] = True

            with torch.no_grad():
                for (al, be, tk) in settings:
                    inst.topk, inst.alpha, inst.beta = tk, al, be
                    inst.forward(in_s, in_b, in_p, gt_l, gt_b, m_gt)
                    L = inst.last
                    align = L["align_metric"]; mpos = L["mask_pos"]
                    for g in range(int(m_gt[0].sum())):
                        u = int(uid[g])
                        pos = mpos[0, g].bool()
                        np_pre = int(pos.sum())          # 与 V5/§13 同口径（冲突消解前）
                        ba = float(align[0, g][pos].max()) if np_pre else 0.0
                        # 冲突消解后：真正进 loss 的正样本
                        np_post = int(((L["target_gt_idx"][0] == g) & L["fg_mask"][0]).sum())
                        AG[(al, be, tk)][u] = (np_pre, ba, np_post)
                    if (al, be, tk) == BASELINE:
                        cig = L["mask_in_gts"][0].bool()
                        bs_ = L["bbox_scores"][0]; ov_ = L["overlaps"][0]
                        for g in range(int(m_gt[0].sum())):
                            u = int(uid[g])
                            m = cig[g]
                            if not bool(m.any()):
                                continue
                            CAND[u] = dict(n_cand=int(m.sum()),
                                           max_iou=float(ov_[g][m].max()),
                                           max_cls=float(bs_[g][m].max()),
                                           native_area=nmeta.get(u, float("nan")))
                        for (a2, b2) in CAND_ALIGN:
                            for g in range(int(m_gt[0].sum())):
                                u = int(uid[g])
                                m = cig[g]
                                if not bool(m.any()):
                                    continue
                                v = (bs_[g][m].pow(a2) * ov_[g][m].pow(b2)).max()
                                CAND_ALIGN[(a2, b2)][u] = float(v)
            if (c + 1) % 50 == 0:
                print(f"    {c + 1}/{len(idxs)}  AG_records={len(AG[BASELINE])}")

    # ---------------- 落盘 ----------------
    np.savez_compressed(
        OUT / "_e2_replay.npz",
        **{f"AG__{al}_{be}_{tk}__uid": np.array(sorted(AG[(al, be, tk)]), dtype=np.int64) for (al, be, tk) in settings},
        **{f"AG__{al}_{be}_{tk}__val": np.array([AG[(al, be, tk)][u] for u in sorted(AG[(al, be, tk)])],
                                                dtype=np.float64) for (al, be, tk) in settings},
        **{f"CA__{a2}_{b2}__uid": np.array(sorted(CAND_ALIGN[(a2, b2)]), dtype=np.int64)
           for (a2, b2) in CAND_ALIGN},
        **{f"CA__{a2}_{b2}__val": np.array([CAND_ALIGN[(a2, b2)][u] for u in sorted(CAND_ALIGN[(a2, b2)])],
                                            dtype=np.float64) for (a2, b2) in CAND_ALIGN},
        CAND_uid=np.array(sorted(CAND), dtype=np.int64),
        CAND_ncand=np.array([CAND[u]["n_cand"] for u in sorted(CAND)], dtype=np.int64),
        CAND_maxiou=np.array([CAND[u]["max_iou"] for u in sorted(CAND)], dtype=np.float64),
        CAND_maxcls=np.array([CAND[u]["max_cls"] for u in sorted(CAND)], dtype=np.float64),
        CAND_native=np.array([CAND[u]["native_area"] for u in sorted(CAND)], dtype=np.float64),
    )
    (OUT / "_e2_replay_meta.json").write_text(json.dumps(dict(
        provenance=prov, settings=settings, baseline=BASELINE, n_images=len(idxs),
        epochs=args.epochs, base_seed=BASE_SEED,
        real_assigner=dict(topk=a.topk, alpha=a.alpha, beta=a.beta, num_classes=a.num_classes),
        full_assigner_replay=True,
        note="alpha/beta/topk 是实例属性，改属性后重调 _forward 等价于完整 assigner 重跑"
             "（含 candidate/top-k/冲突消解）。输入张量与 augmentation realization 完全不变。"),
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  记录：baseline GT 数 = {len(AG[BASELINE])}；候选几何 GT 数 = {len(CAND)}")
    print(f"[saved] {OUT/'_e2_replay.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
