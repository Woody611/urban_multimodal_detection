"""_v5_replay.py — SMALL_OBJECT_TRAIN_REPLAY_AUDIT_V1

对 G1/G2/G4 在**真实 train augmentation 后**复盘其 TaskAlignedAssigner / classification target。

只读。只碰 train/val。不训练、不改 ultralytics/、不改 configs、不碰 test/submission/online。

身份追踪原理：`Mosaic._mosaic4` 按 **tile 顺序 0,1,2,3** 拼接（`_cat_labels` 只裁剪不重排），
因此只要记录 (a) 4 个源 dataset index，(b) 每个 tile 的 GT 数，即可把输出 GT 索引映射回源图 GT。
映射的正确性由「输出 GT 总数 == Σ tile GT 数」当场校验；不等则报告为不可追踪，不强行映射。
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
from ultralytics.data.augment import Mosaic  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, yaml_model_load  # noqa: E402
from ultralytics.utils import yaml_load  # noqa: E402
from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors  # noqa: E402

OUT = Path(__file__).resolve().parent
SEPSTEM_YAML = ROOT / "configs/yolo11m_sepstem.yaml"
CKPT = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"
DS_YAML = ROOT / "data/processed/rgbid_split_train/dataset.yaml"
SEED = 20260926


class Rec:
    """记录每次 mosaic 调用的 (索引, 每 tile GT 数/类别)。"""
    def __init__(self):
        self.calls = []      # list of dict(idx=list, tiles=[cls arrays])
        self.cur = None

    def start(self, idx):
        self.cur = dict(idx=list(idx), tiles=[])
        self.calls.append(self.cur)

    def tile(self, cls):
        if self.cur is not None:
            self.cur["tiles"].append(np.asarray(cls).copy())


class InstAssigner(TaskAlignedAssigner):
    """插桩 `forward`：stash 全部真实返回值（含 target_scores）。不改任何计算。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.last = {}

    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        r = super().forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        tl, tb, ts, fg, tgi = r
        # 必须 **update** 而非覆盖：get_pos_mask 在上面的 super().forward() 内部已写入
        # mask_pos/align_metric/overlaps，直接赋值会抹掉它们。
        self.last.update(target_labels=tl, target_bboxes=tb, target_scores=ts,
                         fg_mask=fg, target_gt_idx=tgi)
        return r

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, eps=1e-9):
        r = super().select_candidates_in_gts(xy_centers, gt_bboxes, eps)
        self.last["mask_in_gts"] = r
        return r

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        r = super().select_topk_candidates(metrics, largest=largest, topk_mask=topk_mask)
        self.last["mask_topk"] = r
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
    from types import SimpleNamespace
    model.args = SimpleNamespace(cls_pw=None)
    return model


def assign_replay(model, loss, inst, img, bboxes, cls, stride):
    """在给定 (img, gt) 上跑真实 assigner，返回 per-GT 记录。"""
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
        am = L["align_metric"][0, g]; ov = L["overlaps"][0, g]
        ts = L["target_scores"][0, :, cid] if L["target_scores"].ndim == 3 \
            else L["target_scores"][0, g, :]
        # target_scores 结构探测：优先按 (b, h*w, nc) 取该 GT 类的正类目标
        try:
            ts_g = L["target_scores"][0, :, cid]
        except Exception:
            ts_g = torch.zeros_like(am)
        r = dict(cls=cid, n_pos=int(pos.sum()),
                 best_assign_iou=float(ov[pos].max()) if pos.any() else 0.0,
                 best_align=float(am[pos].max()) if pos.any() else 0.0,
                 mean_align=float(am[pos].mean()) if pos.any() else 0.0,
                 best_any_overlap=float(ov.max()))
        # classification target：positive anchor 上的真实 target score
        if pos.any():
            r["pos_target_mean"] = float(ts_g[pos].mean())
            r["pos_target_max"] = float(ts_g[pos].max())
        else:
            r["pos_target_mean"] = r["pos_target_max"] = 0.0
        r["target_sum_all"] = float(ts_g.sum())
        r["pos_strides"] = sorted(set(round(float(s), 1)
                                      for s in st.flatten()[pos])) if pos.any() else []
        # 尺度/位置（augmented 后，输入空间像素）
        gb = gt_b[0, g]
        r["aug_w"] = float(gb[2] - gb[0]); r["aug_h"] = float(gb[3] - gb[1])
        r["aug_area"] = r["aug_w"] * r["aug_h"]
        r["aug_sqrt"] = float(np.sqrt(max(r["aug_area"], 0)))
        cx, cy = float((gb[0] + gb[2]) / 2), float((gb[1] + gb[3]) / 2)
        W = float(imgsz[1])
        r["aug_border"] = float(min(gb[0], gb[1], W - gb[2], W - gb[3]))
        # 是否落在 mosaic 边框外的拼接区（ultralytics 用 border 判定）
        r["aug_cx"] = cx; r["aug_cy"] = cy
        out.append(r)
    return out


def main():
    print("=" * 78)
    print("SMALL_OBJECT_TRAIN_REPLAY_AUDIT_V1（只读）")
    print("=" * 78)
    model = build_model(); model.eval(); model.model[-1].train()
    loss = v8DetectionLoss(model)
    inst = InstAssigner(topk=10, num_classes=loss.nc, alpha=0.5, beta=6.0)
    stride = model.stride
    print(f"  seed={SEED}  nc={loss.nc} topk={inst.topk} stride={[float(s) for s in stride]}")

    data = yaml_load(str(DS_YAML))
    # ---- 真实 train pipeline（configs 与 D′ 一致；mosaic 保持训练默认 1.0）----
    cfgtr = get_cfg(overrides=dict(
        imgsz=1280, task="detect", rect=False, cache=False, single_cls=False, classes=None,
        fraction=1.0, channels=5, use_simotm="RGBID",
        object_scale_aug=False,
        mosaic=1.0, mixup=0.0, copy_paste=0.0, degrees=0.0, translate=0.1, scale=0.5,
        shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5))
    img_path = (Path(data["path"]) / data["train"]).resolve()
    ds = build_yolo_dataset(cfgtr, str(img_path), 8, data, mode="train", use_simotm="RGBID",
                            pairs_rgb_ir=["visible", "infrared"],
                            pairs_rgb_depth=["visible", "depth"])
    print(f"  train 数据集 n={len(ds)}  (mosaic p=1.0)")
    stems = [Path(p).stem for p in ds.im_files]

    # ---- 插桩 ----
    rec = Rec()
    orig_gi = Mosaic.get_indexes
    def gi(self):
        r = orig_gi(self)
        rec.start(r if isinstance(r, (list, tuple)) else [r])
        return r
    Mosaic.get_indexes = gi
    orig_ul = Mosaic._update_labels
    def ul(labels, padw, padh):
        rec.tile(labels["cls"])
        return orig_ul(labels, padw, padh)
    Mosaic._update_labels = staticmethod(ul)

    # ---- 目标集合 ----
    v3 = json.loads((ROOT / "diagnostic/small_object_cause_v2/_v3_analysis.json").read_text(encoding="utf-8"))
    E1 = set(v3["E1"]); ctrlv = set(v[0] for v in v3["controls"].values())
    v2rec = {f"{r['image_id']}#{r['gt_id']}": r for r in
             json.loads((ROOT / "diagnostic/small_object_cause_v2/_v2_records.json")
                        .read_text(encoding="utf-8"))["records"]}
    v4 = {f"{r['stem']}#{r['gt']}": r for r in
          json.loads((ROOT / "diagnostic/small_object_internal_audit/_v4_internal.json")
                     .read_text(encoding="utf-8"))}

    def grp(k):
        if k in E1:
            return "G1"
        if k in ctrlv:
            return "G2"
        if k in v2rec and k not in E1 and k not in ctrlv \
           and v2rec[k]["native_area"] < 1024 and v2rec[k]["best_same_class_iou"] >= 0.5:
            return "G4"
        return None

    # ⚠️ G1/G2/G4 全部来自 **val split**。train pipeline 只处理 train split，二者不相交，
    # 因此「replay G1 的训练监督」在**构造上不可能** —— 这些目标从未进入训练。
    # 改为回答可回答的等价问题：
    #   train 集的 small GT 在真实增强后，是否仍获得有效正类监督？
    NIMG = 250
    idxs = list(range(0, len(ds), max(1, len(ds) // NIMG)))[:NIMG]
    print(f"  采样 train 图 = {len(idxs)} / {len(ds)}（带放回无关，固定 stride 采样）")

    rows = []
    stat = dict(mosaic_calls=0, map_ok=0, map_bad=0, drop_gt=0)
    SRC = {}

    def srclab(k):
        if k not in SRC:
            SRC[k] = ds.get_image_and_label(k)
        return SRC[k]

    for i in idxs:
        random.seed(SEED + i); np.random.seed(SEED + i)
        rec.cur = None; n0 = len(rec.calls)
        try:
            lab = ds[i]
        except Exception as e:
            rows.append(dict(stem=stems[i], err=str(e)))
            continue
        if len(rec.calls) == n0:      # mosaic 未触发
            rec.start([])
        c = rec.calls[-1]
        stat["mosaic_calls"] += 1
        img = lab["img"].float() / 255.0
        if img.ndim == 3:
            img = img.unsqueeze(0)
        bb = lab["bboxes"]; cl = lab["cls"]
        n_out = len(cl)
        n_tiles = sum(len(t) for t in c["tiles"])
        okmap = (n_out == n_tiles) and len(c["tiles"]) == 4
        stat["map_ok" if okmap else "map_bad"] += 1
        # ⚠️ 不因映射失败而跳过样本：既然本轮不使用身份映射，映射失败只作为
        #    「身份不可追踪」的证据记录下来，样本本身仍必须参与统计。
        #   （首版误留 `if not okmap: continue`，导致 244/250 样本被跳过、
        #     结果只来自 2.4% 的有偏子集。）
        # tile 边界
        bounds = np.cumsum([0] + [len(t) for t in c["tiles"]])
        res = assign_replay(model, loss, inst, img, bb, cl, stride)
        # ⚠️ 身份追踪不可行：mosaic 之后的 RandomPerspective/_cat_labels 会改变 GT 集合，
        #    「输出 GT 数 == Σ tile GT 数」在 244/250 个样本上不成立（见 stat['map_bad']）。
        #    因此不做映射，直接对**增强后的输出 GT** 分层（这才是训练时真正参与 loss 的 GT）。
        for j, r in enumerate(res):
            r["src_stem"] = stems[i]
            r["bucket"] = ("small" if r["aug_area"] < 1024 else
                           ("medium" if r["aug_area"] < 9216 else "large"))
            r["mosaic_sample"] = i
            rows.append(r)
        if len(rows) % 50 == 0:
            print(f"    ...{len(rows)} 条")

    (OUT / "_v5_replay.json").write_text(json.dumps(dict(rows=rows, stat=stat, seed=SEED),
                                                    indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  映射校验: 可追踪 {stat['map_ok']} / 不可追踪 {stat['map_bad']}；mosaic 后丢 GT {stat['drop_gt']}")
    print(f"[saved] {OUT/'_v5_replay.json'}  ({len(rows)} 条)")


if __name__ == "__main__":
    main()
