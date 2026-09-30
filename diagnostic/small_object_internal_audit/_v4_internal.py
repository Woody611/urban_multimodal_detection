"""_v4_internal.py — SMALL_OBJECT_MODEL_INTERNAL_AUDIT_V1

只读、只碰 train/val。不训练、不改核心代码、不生成新 prediction、不碰 test/submission/online。

内容：
  §5  D  TaskAlignedAssigner 复盘（in-process 子类插桩，不修改 ultralytics/）
  §7  E  feature response（P3=layer23 / backbone P2=layer9 的 GT 中心池化）
  §8  F  GT-center candidate response（raw head 输出，pre-NMS）

⚠️ 输入管线说明（限制，必须同读）：
  使用 **val 管线**（build_yolo_dataset mode="val" → LetterBox+Format），
  不是训练时的 mosaic 增强输入。理由：确定性、可复现、且与 val 预测同源。
  因此本轮的 assignment 是「在该确定性输入下真实 assigner 的行为」，
  不是「训练第 N 个 epoch 的增强样本上的行为」。
  模型：backbone/neck 为 eval（BN 不更新），仅 Detect 头置 train 以取得 raw per-level 输出。
"""
import json
import sys
from collections import Counter, defaultdict
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
DS_YAML = ROOT / "data/processed/rgbid_split_train/dataset.yaml"


class InstrumentedAssigner(TaskAlignedAssigner):
    """仅插桩：记录 select_candidates_in_gts / select_topk_candidates 的返回。

    不改变任何计算 —— 所有方法都调用 super() 后原样返回。
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.last = {}

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, eps=1e-9):
        r = super().select_candidates_in_gts(xy_centers, gt_bboxes, eps)
        self.last["mask_in_gts"] = r
        return r

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        r = super().select_topk_candidates(metrics, largest=largest, topk_mask=topk_mask)
        self.last["mask_topk"] = r
        return r

    def get_pos_mask(self, *a, **kw):
        mask_pos, align_metric, overlaps = super().get_pos_mask(*a, **kw)
        self.last.update(mask_pos=mask_pos, align_metric=align_metric, overlaps=overlaps)
        return mask_pos, align_metric, overlaps


def build():
    cfg = yaml_model_load(str(SEPSTEM_YAML))
    assert cfg.get("scale") == "m"
    model = DetectionModel(cfg, nc=12, verbose=False)
    w, _ = attempt_load_one_weight(str(CKPT))
    model.load(w)
    model.model[-1].stride = model.stride
    # v8DetectionLoss 只从 model.args 读 cls_pw；实际训练 kwargs 里 cls_pw=[] → 不生效
    from types import SimpleNamespace
    model.args = SimpleNamespace(cls_pw=None)
    return model


def main():
    print("=" * 78)
    print("SMALL_OBJECT_MODEL_INTERNAL_AUDIT_V1 —— D′ 内部机制审计（只读）")
    print("=" * 78)
    model = build()
    model.eval()
    model.model[-1].train()          # 仅 Detect 头：取得 raw per-level 输出；BN 不更新
    loss = v8DetectionLoss(model)
    inst = InstrumentedAssigner(topk=10, num_classes=loss.nc, alpha=0.5, beta=6.0)
    stride = model.stride
    print(f"  nc={loss.nc} no={loss.no} reg_max={loss.reg_max} topk={inst.topk}")
    print(f"  stride={[float(s) for s in stride]}")

    # 特征 hook
    feat = {}
    def mk(i):
        def h(m, inp, out):
            if torch.is_tensor(out):
                feat[i] = out.detach()
        return h
    model.model[23].register_forward_hook(mk(23))   # P3 颈输出
    model.model[9].register_forward_hook(mk(9))     # backbone P2 (stride 4)
    # Detect(layer 30) 返回 list，不挂钩 —— raw 输出已从 model() 返回值取得

    data = yaml_load(str(DS_YAML))
    cfgv = get_cfg(overrides=dict(
        imgsz=1280, task="detect", rect=False, cache=False, single_cls=False, classes=None,
        fraction=1.0, channels=5, use_simotm="RGBID", mosaic=0.0, copy_paste=0.0, mixup=0.0,
        object_scale_aug=False, augment=False))
    img_path = (Path(data["path"]) / data["val"]).resolve()
    ds = build_yolo_dataset(cfgv, str(img_path), 8, data, mode="val", use_simotm="RGBID",
                            pairs_rgb_ir=["visible", "infrared"],
                            pairs_rgb_depth=["visible", "depth"])
    print(f"  val 数据集 n={len(ds)}")
    stems = [Path(p).stem for p in ds.im_files]

    # 本轮目标集合
    v3 = json.loads((ROOT / "diagnostic/small_object_cause_v2/_v3_analysis.json").read_text(encoding="utf-8"))
    E1 = set(v3["E1"])
    ctrl = v3["controls"]

    rows = []
    for i in range(len(ds)):
        stem = stems[i]
        need_g1 = any(k.startswith(stem + "#") for k in E1)
        # 注意：ctrl 是 {G1_key: (control_key, type)} —— 必须检查 **control_key** 的 stem，
        # 否则跨图对照所在图不会被处理（首版即此 bug，导致 11/38 对照缺失）
        need_c = any(v[0].startswith(stem + "#") for v in ctrl.values())
        if not (need_g1 or need_c):
            continue
        lab = ds[i]
        img = lab["img"]
        # 复刻 BaseTrainer.preprocess_batch：float / 255（Format 输出为 uint8）
        img = img.float() / 255.0
        if img.ndim == 3:
            img = img.unsqueeze(0)
        else:
            continue
        with torch.no_grad():
            preds = model(img)
        if isinstance(preds, tuple):
            preds = preds[1]
        feats = preds[: stride.size(0)]
        pred_distri, pred_scores = torch.cat(
            [xi.view(1, loss.no, -1) for xi in feats], 2).split((loss.reg_max * 4, loss.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        imgsz = torch.tensor(feats[0].shape[2:], dtype=pred_scores.dtype) * stride[0]
        anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)
        bb = lab["bboxes"]
        if len(bb) == 0:
            continue
        targets = torch.cat((torch.zeros(len(bb), 1), lab["cls"].view(-1, 1), bb), 1)
        t = loss.preprocess(targets, 1, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = t.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        pred_bboxes = loss.bbox_decode(anchor_points, pred_distri)
        with torch.no_grad():
            tl, tb, ts, fg_mask, _ = inst(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        L = inst.last
        n_gt = int(mask_gt[0].sum())
        gtxyxy = gt_bboxes[0, :n_gt]
        gtc = (gtxyxy[:, :2] + gtxyxy[:, 2:]) / 2
        # 每 GT 统计
        for g in range(n_gt):
            cls_id = int(gt_labels[0, g, 0])
            in_gts = L["mask_in_gts"][0, g].bool()
            topk = L["mask_topk"][0, g].bool()
            pos = L["mask_pos"][0, g].bool()      # mask_pos 是 float（乘过 float 的 mask_gt）
            am = L["align_metric"][0, g]
            ov = L["overlaps"][0, g]
            n_pos = int(pos.sum())
            r = dict(stem=stem, gt=g, cls=cls_id,
                     gt_w=float(gtxyxy[g, 2] - gtxyxy[g, 0]), gt_h=float(gtxyxy[g, 3] - gtxyxy[g, 1]),
                     n_in_gts=int(in_gts.sum()), n_topk=int(topk.sum()), n_pos=n_pos,
                     n_anchor=int(pos.numel()))
            # 直接量：assigner 选中的 positive anchor 上，GT 类别的分类分（sigmoid 后）
            sc_all = pred_scores[0].sigmoid()[:, cls_id]
            r["pos_score_mean"] = float(sc_all[pos].mean()) if n_pos else 0.0
            r["pos_score_max"] = float(sc_all[pos].max()) if n_pos else 0.0
            if n_pos:
                r["best_assign_iou"] = float(ov[pos].max())
                r["mean_assign_iou"] = float(ov[pos].mean())
                r["best_align"] = float(am[pos].max())
                r["mean_align"] = float(am[pos].mean())
                # positive 所属 stride level = 该 anchor 的 stride
                st_flat = stride_tensor.flatten()   # shape (n,1) -> (n,)
                r["pos_strides"] = sorted(set(round(float(s), 1) for s in st_flat[pos]))
            else:
                r["best_assign_iou"] = r["mean_assign_iou"] = 0.0
                r["best_align"] = r["mean_align"] = 0.0
                r["pos_strides"] = []
            r["best_in_gts_align"] = float(am[in_gts].max()) if in_gts.any() else 0.0
            r["best_any_overlap"] = float(ov.max())
            # ---- F: GT 中心附近的 raw candidate ----
            c = gtc[g]
            for si, s in enumerate([8., 16., 32.]):
                pts = (anchor_points * stride_tensor[0, :, None]) if False else anchor_points
                # anchor_points 是 cell 坐标；乘 stride 得像素
                apx = anchor_points * stride_tensor            # (n,2) * (n,1) -> 像素坐标
                d = torch.linalg.norm(apx - c[None, :], dim=1)
                near = d <= 1.5 * s
                if int(near.sum()) == 0:
                    continue
                sc = pred_scores[0, :, cls_id]
                r[f"p{si+2}_n_near"] = int(near.sum())
                r[f"p{si+2}_max_score_near"] = float(sc[near].max())
                r[f"p{si+2}_max_score_all"] = float(sc.max())
                # 该 level 内同类最高分 anchor 的解码框与 GT 的 IoU
                lvl = (stride_tensor[0].flatten() == s)
                idx_lvl = torch.where(lvl)[0]
                if len(idx_lvl):
                    k = idx_lvl[torch.argmax(sc[idx_lvl])]
                    pb = pred_bboxes[0, k] * s
                    gb = gtxyxy[g]
                    ix1, iy1 = max(float(pb[0]), float(gb[0])), max(float(pb[1]), float(gb[1]))
                    ix2, iy2 = min(float(pb[2]), float(gb[2])), min(float(pb[3]), float(gb[3]))
                    inter = max(0., ix2 - ix1) * max(0., iy2 - iy1)
                    ua = ((pb[2]-pb[0])*(pb[3]-pb[1]) + (gb[2]-gb[0])*(gb[3]-gb[1]) - inter).clamp(min=1e-9)
                    r[f"p{si+2}_best_same_cls_iou"] = float(inter / ua)
            # ---- E: feature response ----
            for li in (23, 9):
                if li not in feat:
                    continue
                f = feat[li][0]                      # (C,h,w)
                st = float(stride[0]) if li == 23 else 4.0
                if li == 9:
                    st = 4.0
                fh, fw = f.shape[1:]
                fx, fy = float(c[0]) / (1280 / fw), float(c[1]) / (1280 / fh)
                x0, x1 = max(0, int(fx) - 1), min(fw, int(fx) + 2)
                y0, y1 = max(0, int(fy) - 1), min(fh, int(fy) + 2)
                pat = f[:, y0:y1, x0:x1]
                if pat.numel():
                    r[f"L{li}_center_norm"] = float(pat.norm(dim=0).mean())
                    r[f"L{li}_global_norm"] = float(f.norm(dim=0).mean())
                    r[f"L{li}_ratio"] = (r[f"L{li}_center_norm"] /
                                         (r[f"L{li}_global_norm"] + 1e-9))
            rows.append(r)
        feat.clear()
        if len(rows) % 40 == 0:
            print(f"    ...{len(rows)} GT 记录")

    (OUT / "_v4_internal.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"\n[saved] {OUT/'_v4_internal.json'}  ({len(rows)} GT)")
    return rows


if __name__ == "__main__":
    main()
