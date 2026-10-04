"""_probe.py — Probe-P3-Trajectory 的**只读插针**（唯一允许的改动：增加 read-only instrumentation）。

不可变性保证（`_integrity_gate.py` 实测 8 项全 PASS）：
  G1 forward 逐位不变 / G6 loss 逐位不变 / G2 RawTAL≡父类 TaskAlignedAssigner /
  G3 raw_align 未被 in-place 掩码 / G4 random·numpy·torch RNG 状态不变 /
  G5 无梯度 + 337 个参数逐位未变 / **G8 BN buffer 副作用已由快照+还原消除** /
  G7 特征语义与 `_v7_extract.py` 同口径。

关键语义（沿用 `_v7_extract.py`，保证与上一轮 audit 的特征可比）：
  P3 = model.model[23] 的 forward 输出；h = Detect.cv3[0][-1] 的 forward **前置输入**；
  logit = W_gt·h + b_gt（精确等式）；
  数据集 = build_yolo_dataset(mode="val", rect=False, imgsz=1280)，无增强；
  前向 = backbone eval（BN 用 running stats）+ head train（返回原始张量）；img.float()/255，fp32，autocast off。

§6 的语义陷阱（上一轮实测：cand_align 的 recorded==0 占 95.29%）：`tal.py:110` 的
`align_metric *= mask_pos` 是 **in-place** ⇒ 本插针在 `get_box_metrics` 的**返回处 clone** 原始张量。
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.utils import yaml_load
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors

ROOT = Path(__file__).resolve().parents[2]

# ---- 预注册阈值（§11-§14 判决用；写在代码里，事后不得修改）----
TH_NEAR_RANDOM = 0.05      # acc <= random_baseline + 0.05 视作"≈随机"
TH_DECLINE = 0.10          # max(acc) - acc(ep_last) >= 0.10 视作"显著下降"
TH_CLS_NATS = 2.0          # 中位 GT 类 logit 相对历史最大下降 >= 2 nats
TH_ALIGN_FACTOR = 0.5      # 中位 raw_alignment 相对历史最大降至 <= 50%


class RawTAL(TaskAlignedAssigner):
    """只读包装：在**返回处 clone** 原始（未掩码）align/overlaps，并暴露各 mask。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.last = {}

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        r = super().get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)
        self.last["raw_align"] = r[0].detach().clone()      # ★ 必须 clone：tal.py:110 会 in-place 破坏
        self.last["overlaps"] = r[1].detach().clone()
        return r

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, eps=1e-9):
        r = super().select_candidates_in_gts(xy_centers, gt_bboxes, eps)
        self.last["mask_in_gts"] = r.detach()
        return r

    def select_topk_candidates(self, m, largest=True, topk_mask=None):
        r = super().select_topk_candidates(m, largest=largest, topk_mask=topk_mask)
        self.last["mask_topk"] = r.detach()
        return r

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes):
        r = super().select_highest_overlaps(mask_pos, overlaps, n_max_boxes)
        self.last["mask_pos"] = r[2].detach()
        self.last["fg_mask"] = r[1].detach()
        return r

    def get_targets(self, *a, **kw):
        r = super().get_targets(*a, **kw)
        self.last["target_scores_raw"] = r[2].detach().clone()
        return r


class P3TrajProbe:
    def __init__(self, cells_json: Path, out_dir: Path,
                 observe_epochs=(0, 1, 2, 5, 10, 15, 20, 25, 30), stop_after_epoch: int = 30):
        g = json.loads(Path(cells_json).read_text(encoding="utf-8"))
        self.cells = g["rows"]                                  # 1320 medium val GT
        self.g86 = [r for r in self.cells if r["role"] == "G86"]
        assert len(self.g86) == 86, f"G-86 = {len(self.g86)}"
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.obs = set(observe_epochs); self.stop_after = stop_after_epoch
        self.traj = []            # 86 × epoch 的标量记录
        self.vecs = {}            # epoch -> dict(vec_keys, vec_cls, vec_p3, vec_h, vec_role)
        self._ds = None; self._feat = None

    # ---------- 只读钩子（挂在既有层上；不新增层 ⇒ 不改图/参数/RNG）----------
    def attach_model(self, nn_model):
        nn_model.model[23].register_forward_hook(self._mk_out("P3"))
        last = nn_model.model[-1].cv3[0][-1]
        last.register_forward_pre_hook(self._mk_pre("H"))
        self._feat = {}
        # ★ 用 **numpy** 保存分类头权重：logit = W_gt·h + b_gt 里的 h 来自 _sample()（numpy/CPU），
        #   若 W/b 留在 GPU 会触发 "Expected all tensors to be on the same device"（云端实测踩到）。
        #   纯 numpy 计算从构造上排除设备混用，且与本文件 G7 的 float64 口径一致。
        self._Wn = last.weight.detach().cpu().view(last.weight.shape[0], -1).numpy().astype(np.float64)
        self._bn = last.bias.detach().cpu().numpy().astype(np.float64)
        self._loss = v8DetectionLoss(nn_model)
        self._assigner = RawTAL(topk=10, num_classes=self._loss.nc, alpha=0.5, beta=6.0)

    def _mk_out(self, k):
        def h(m, i, o):
            if torch.is_tensor(o):
                self._feat[k] = o.detach().clone()
        return h

    def _mk_pre(self, k):
        def h(m, inp):
            self._feat[k] = inp[0].detach().clone()
        return h

    # ---------- 数据集（口径固定，无增强）----------
    def build(self):
        data = yaml_load(str(ROOT / "data/processed/rgbid_split_train/dataset.yaml"))
        ov = dict(imgsz=1280, task="detect", rect=False, cache=False, single_cls=False, classes=None,
                  fraction=1.0, channels=5, use_simotm="RGBID", mosaic=0.0, copy_paste=0.0, mixup=0.0,
                  object_scale_aug=False, augment=False)
        self._ds = build_yolo_dataset(get_cfg(overrides=ov),
                                      str((Path(data["path"]) / data["val"]).resolve()), 8, data,
                                      mode="val", use_simotm="RGBID",
                                      pairs_rgb_ir=["visible", "infrared"],
                                      pairs_rgb_depth=["visible", "depth"])
        want = {r["image_stem"] for r in self.cells}
        self._sel = [i for i in range(len(self._ds)) if Path(self._ds.im_files[i]).stem in want]
        self._cell_by_stem = {}
        for r in self.cells:
            self._cell_by_stem.setdefault(r["image_stem"], []).append(r)
        assert len(self._sel) == len(self._cell_by_stem)

    def _sample(self, t, cx, cy):
        ch, hh, ww = t.shape[1:]
        sc = 1280.0 / ww
        return t[0, :, int(min(max(cy / sc, 0), hh - 1)), int(min(max(cx / sc, 0), ww - 1))
                 ].detach().cpu().numpy().astype(np.float32)

    # ---------- 单次观测 ----------
    def observe(self, nn_model, epoch):
        st_r, st_n, st_t = random.getstate(), np.random.get_state(), torch.get_rng_state()
        was_tr = nn_model.training
        buf0 = {k: v.detach().clone() for k, v in nn_model.named_buffers()}   # ★ BN buffer 快照
        dev = next(nn_model.parameters()).device
        nn_model.eval(); nn_model.model[-1].train()
        recs, vk, vc, vp, vh, vr = [], [], [], [], [], []
        try:
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                for i in self._sel:
                    lab = self._ds[i]; bb, cl = lab.get("bboxes"), lab.get("cls")
                    if bb is None or not len(bb):
                        continue
                    stem = Path(self._ds.im_files[i]).stem
                    img = lab["img"].float().div(255.0)
                    img = (img.unsqueeze(0) if img.ndim == 3 else img).to(dev)
                    self._feat.clear()
                    preds = nn_model(img)
                    feats = (preds[1] if isinstance(preds, tuple) else preds)[: self._loss.stride.size(0)]
                    P3, Hh = self._feat.get("P3"), self._feat.get("H")
                    if P3 is None or Hh is None:
                        continue
                    # --- TAL（只读）---
                    pd_d, pd_s = torch.cat([x.view(1, self._loss.no, -1) for x in feats], 2).split(
                        (self._loss.reg_max * 4, self._loss.nc), 1)
                    pd_s = pd_s.permute(0, 2, 1).contiguous(); pd_d = pd_d.permute(0, 2, 1).contiguous()
                    _ap, st = make_anchors(feats, self._loss.stride, 0.5)
                    imgsz = torch.tensor(feats[0].shape[2:], dtype=pd_s.dtype, device=pd_s.device) * self._loss.stride[0]
                    tg = torch.cat((torch.zeros(len(bb), 1, device=bb.device), cl.view(-1, 1), bb), 1)
                    t = self._loss.preprocess(tg, 1, scale_tensor=imgsz[[1, 0, 1, 0]])
                    gt_l, gt_b = t.split((1, 4), 2)
                    self._assigner.forward(pd_s.detach().sigmoid(),
                                           (self._loss.bbox_decode(_ap, pd_d).detach() * st).type(gt_b.dtype),
                                           _ap * st, gt_l, gt_b, gt_b.sum(2, keepdim=True).gt_(0))
                    L = self._assigner.last; apx = _ap * st; sc_all = pd_s[0].sigmoid()
                    for r in self._cell_by_stem[stem]:
                        j = r["gt_id"]; cid = r["class_id"]
                        # 采样与记录在**同一循环**内完成（避免两循环间的对齐 bug）
                        p3v = self._sample(P3, r["x_center_canvas"], r["y_center_canvas"])
                        hv = self._sample(Hh, r["x_center_canvas"], r["y_center_canvas"])
                        vk.append(r["key"]); vc.append(cid); vr.append(r["role"])
                        vp.append(p3v); vh.append(hv)
                        # ★ 快速失败断言：本机 CPU-only 无法复现设备混用（本地 CPU tensor 恰好能算），
                        #   故在真实路径上显式断言两侧都是 numpy，避免云端出现晦涩的 CUDA device 错误。
                        assert isinstance(hv, np.ndarray) and isinstance(p3v, np.ndarray) and isinstance(self._Wn, np.ndarray), "probe 设备混用：W/b 与 h 必须同为 numpy"
                        lg = float(self._Wn[cid] @ hv + self._bn[cid])   # 纯 numpy，无设备问题
                        pos = L["mask_pos"][0, j].bool(); topk = L["mask_topk"][0, j].bool()
                        ov = L["overlaps"][0, j]; ra = L["raw_align"][0, j]
                        gb_ = gt_b[0, j]; c = (gb_[:2] + gb_[2:]) / 2
                        near = torch.linalg.norm(apx - c[None, :], dim=1) <= 12.0
                        recs.append(dict(
                            epoch=epoch, key=r["key"], class_id=cid, role=r["role"],
                            p3_cell_x=r["p3_cell_x"], p3_cell_y=r["p3_cell_y"],
                            logit=lg, sigmoid=1.0 / (1.0 + math.exp(-max(min(lg, 60), -60))),
                            p3_norm=float(np.linalg.norm(p3v)), h_norm=float(np.linalg.norm(hv)),
                            ciou_pos_median=float(ov[pos].median()) if pos.any() else float("nan"),
                            ciou_pos_max=float(ov[pos].max()) if pos.any() else float("nan"),
                            ciou_pool_max=float(ov.max()),
                            raw_align_pos_max=float(ra[pos].max()) if pos.any() else float("nan"),
                            raw_align_pool_max=float(ra.max()),
                            cls_pos_max=float(sc_all[pos, cid].max()) if pos.any() else float("nan"),
                            cls_pool_max=float(sc_all[:, cid].max()),
                            cls_center_max=float(sc_all[near, cid].max()) if near.any() else float("nan"),
                            n_pos=int(pos.sum()), n_topk=int(topk.sum()),
                            mask_pos_membership=int(bool(pos.any())), topk_membership=int(bool(topk.any())),
                            n_in_gts=int(L["mask_in_gts"][0, j].sum()),
                            pos_mean_ciou=float(ov[pos].mean()) if pos.any() else float("nan"),
                            raw_align_pos_mean=float(ra[pos].mean()) if pos.any() else float("nan"),
                        ))
        finally:
            with torch.no_grad():
                for k, v in nn_model.named_buffers():
                    if k in buf0:
                        v.copy_(buf0[k])                 # ★ BN buffer 还原
            random.setstate(st_r); np.random.set_state(st_n); torch.set_rng_state(st_t)
            nn_model.train(was_tr)
        self.vecs[epoch] = dict(vec_keys=vk, vec_cls=vc, vec_role=vr,
                                vec_p3=np.asarray(vp, np.float32), vec_h=np.asarray(vh, np.float32))
        return recs

    # ---------- 落盘 ----------
    def dump_epoch(self, epoch, recs):
        v = self.vecs[epoch]
        # ★ 只对数值字段 cast；key / role 是字符串，强行转 float64 会在第一个观测 epoch 崩
        num_k = [k for k, x in recs[0].items() if isinstance(x, (int, float))]
        str_k = [k for k in recs[0] if k not in num_k]
        np.savez_compressed(self.out / f"epoch_{int(epoch):03d}.npz",
                            vec_keys=np.array(v["vec_keys"]), vec_cls=np.array(v["vec_cls"]),
                            vec_role=np.array(v["vec_role"]), vec_p3=v["vec_p3"], vec_h=v["vec_h"],
                            rec_num_keys=np.array(num_k), rec_str_keys=np.array(str_k),
                            **{k: np.array([r[k] for r in recs], np.float64) for k in num_k},
                            **{k: np.array([str(r[k]) for r in recs]) for k in str_k})
        self.traj.extend(recs)

    # ---------- 指标（§7，定义与上一轮 audit 完全相同）----------
    @staticmethod
    def _nearest_centroid(keys, cls, role, F):
        cen = {}
        for c in sorted(set(cls[role == "CONTROL"])):
            m = (role == "CONTROL") & (cls == c)
            if m.sum() >= 10:
                cen[c] = F[m].mean(0)
        if not cen:
            return {}
        cs = sorted(cen); M = np.stack([cen[c] / (np.linalg.norm(cen[c]) + 1e-12) for c in cs])
        out = {}
        for r in ("G86", "CONTROL", "MISSED_NON86"):
            sel = role == r
            if not sel.any():
                continue
            V = F[sel] / (np.linalg.norm(F[sel], axis=1, keepdims=True) + 1e-12)
            pred = np.array(cs)[np.argmax(V @ M.T, axis=1)]
            out[r] = float((pred == cls[sel]).mean())
        maj = max(float(np.mean(cls[role == "CONTROL"] == c)) for c in set(cls.tolist()))
        out["majority_prior"] = float(maj)
        return out

    def metrics(self):
        rows_p3, rows_h = [], []
        for e in sorted(self.vecs):
            v = self.vecs[e]
            cls = np.asarray(v["vec_cls"]); role = np.asarray(v["vec_role"])
            for name, F, sink in (("p3", v["vec_p3"], rows_p3), ("h", v["vec_h"], rows_h)):
                m = self._nearest_centroid(v["vec_keys"], cls, role, F)
                sink.append(dict(epoch=e, feature=name, **{k: round(x, 4) for k, x in m.items()}))
        with open(self.out / "p3_metrics.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_p3[0].keys())); w.writeheader(); w.writerows(rows_p3)
        with open(self.out / "h_metrics.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_h[0].keys())); w.writeheader(); w.writerows(rows_h)
        return rows_p3, rows_h

    def finalize(self):
        eps = sorted({r["epoch"] for r in self.traj})
        # ---- TAL / assignment 表 ----
        with open(self.out / "tal_metrics.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["epoch", "key", "raw_align_pos_max", "raw_align_pool_max",
                                               "cls_pos_max", "cls_center_max", "ciou_pos_max", "logit"])
            w.writeheader(); w.writerows([{k: r[k] for k in w.fieldnames} for r in self.traj])
        with open(self.out / "assignment_metrics.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["epoch", "key", "mask_pos_membership", "topk_membership",
                                               "n_pos", "n_topk", "n_in_gts"])
            w.writeheader(); w.writerows([{k: r[k] for k in w.fieldnames} for r in self.traj])
        # ---- epoch summary ----
        ser = []
        # ★ 必须按 role 分组：self.traj 覆盖全部 1320 个 medium cell（G86+CONTROL+MISSED）。
        #   规格 §9 要的是 **G-86 aggregate**；若在全体上取中位数会答非所问
        #   （全 medium 的 median logit 会被 1070 个已检出 GT 拉高，而 G-86 实测中位是 −16.7 量级）。
        for e in eps:
            for role in sorted({r["role"] for r in self.traj}):
                s = [r for r in self.traj if r["epoch"] == e and r["role"] == role]
                if not s:
                    continue
                for f in ("logit", "cls_pos_max", "cls_center_max", "ciou_pos_max",
                          "raw_align_pos_max", "p3_norm", "h_norm"):
                    v = np.array([x[f] for x in s if not math.isnan(x[f])])
                    if not len(v):
                        continue
                    ser.append(dict(epoch=e, role=role, field=f, n=len(v), median=float(np.median(v)),
                                    mean=float(v.mean()), p25=float(np.percentile(v, 25)),
                                    p75=float(np.percentile(v, 75))))
                for f in ("mask_pos_membership", "topk_membership"):
                    ser.append(dict(epoch=e, role=role, field=f + "_rate", n=len(s),
                                    median=float(np.mean([x[f] for x in s])),
                                    mean=float(np.mean([x[f] for x in s])), p25=float("nan"), p75=float("nan")))
        with open(self.out / "epoch_summary.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["epoch", "role", "field", "n", "median", "mean", "p25", "p75"])
            w.writeheader(); w.writerows(ser)
        with open(self.out / "per_gt_trajectory.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.traj[0].keys())); w.writeheader(); w.writerows(self.traj)
        # ---- 判决（预注册阈值）----
        p3, hh = self.metrics()
        dec = self._decide(eps, p3, hh, ser)
        (self.out / "decision.json").write_text(json.dumps(dec, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.out / "decision.md").write_text(self._render(dec), encoding="utf-8")
        return dec

    def _decide(self, eps, p3, hh, ser):
        # 加固：某角色在某 epoch 缺失时返回 nan 而非 KeyError（真实数据三角色恒存在）
        g = lambda rs, r="G86": [next((x.get(r, float("nan")) for x in rs if x["epoch"] == e), float("nan")) for e in eps]
        p3g, p3c = g(p3), g(p3, "CONTROL")
        hg, hc = g(hh, "G86"), g(hh, "CONTROL")
        rb = next((x["majority_prior"] for x in p3 if "majority_prior" in x), float("nan"))
        G86 = lambda e_, k: next((x["median"] for x in ser if x["epoch"] == e_ and x.get("role") == "G86" and x["field"] == k), float("nan"))
        mp = [G86(e, "mask_pos_membership_rate") for e in eps]
        lg = [G86(e, "logit") for e in eps]
        al = [G86(e, "raw_align_pos_max") for e in eps]
        first = eps[0]
        maxp3, lastp3 = max(p3g), p3g[-1]
        decline = maxp3 - lastp3
        early = p3g[1] if len(p3g) > 1 else p3g[0]
        def first_degrade(v, th):
            mx = -1e9
            for e, x in zip(eps, v):
                mx = max(mx, x)
                if mx - x >= th:
                    return e
            return None
        d_p3 = first_degrade(p3g, TH_DECLINE)
        d_cls = first_degrade(lg, TH_CLS_NATS)
        d_al = first_degrade([math.log(max(x, 1e-30)) for x in al], math.log(1 / TH_ALIGN_FACTOR))
        d_mp = first_degrade(mp, TH_DECLINE)
        near_rand_early = early <= rb + TH_NEAR_RANDOM
        h_decline = (max(hg) - hg[-1]) if not any(math.isnan(x) for x in hg) else float("nan")
        h_recovers = (not math.isnan(h_decline)) and (hg[-1] > rb + TH_NEAR_RANDOM) and (h_decline < TH_DECLINE)
        case = "INCONCLUSIVE"
        if not any(math.isnan(x) for x in p3g):
            if near_rand_early and decline < TH_DECLINE:
                # ★ C 是 A 的子情形：P3 全程 ≈ 随机，但 h 明显 > 随机且稳定 ⇒ 先判 C
                case = "C" if h_recovers else "A"
            elif (not near_rand_early) and decline >= TH_DECLINE:
                seq = [x for x in (d_p3, d_cls, d_al, d_mp) if x is not None]
                case = "B" if (len(seq) >= 4 and seq == sorted(seq)) else "D"
        return dict(
            epoch_points=list(eps), random_baseline_majority_prior=round(rb, 4),
            p3_accuracy_g86=[round(x, 4) for x in p3g], p3_accuracy_control=[round(x, 4) for x in p3c],
            h_accuracy_g86=[round(x, 4) for x in hg], h_accuracy_control=[round(x, 4) for x in hc],
            mask_pos_rate=mp, cls_logit_median=lg, raw_align_median=al,
            first_P3_degradation=d_p3, first_cls_degradation=d_cls,
            first_alignment_degradation=d_al, first_mask_pos_loss=d_mp,
            p3_decline_total=round(decline, 4), p3_early=round(early, 4),
            near_random_early=bool(near_rand_early),
            h_decline_total=round(float(h_decline), 4) if not math.isnan(h_decline) else None,
            h_recovers=bool(h_recovers),
            thresholds=dict(TH_NEAR_RANDOM=TH_NEAR_RANDOM, TH_DECLINE=TH_DECLINE,
                            TH_CLS_NATS=TH_CLS_NATS, TH_ALIGN_FACTOR=TH_ALIGN_FACTOR),
            CASE=case,
            REPRESENTATION_INITIAL_WEAKNESS=("SUPPORTED" if case == "A" else "NOT_SUPPORTED"),
            TRAINING_DEGRADATION=("SUPPORTED" if case == "B" else "NOT_SUPPORTED"),
            TAL_SELECTION_FEEDBACK=("SUPPORTED_CANDIDATE" if case == "B" else "NOT_SUPPORTED"),
            P3_WEAK_H_RECOVERY=("SUPPORTED" if case == "C" else "NOT_SUPPORTED"),
            SYNCHRONIZED_MECHANISM=("NOT_SUPPORTED" if case == "D" else "SUPPORTED"),
            CAUSALITY_LEVEL=("mechanistically supported (single trajectory, no intervention)"
                             if case in ("A", "B", "C") else "INCONCLUSIVE"),
            NOTE="本判决由 _probe.py::_decide 依预注册阈值自动生成；阈值写在代码内，事前固定。")

    @staticmethod
    def _render(d):
        L = ["# DECISION — Probe-P3-Trajectory", "", "```text", f"CASE = {d['CASE']}", ""]
        for k in ("REPRESENTATION_INITIAL_WEAKNESS", "TRAINING_DEGRADATION", "TAL_SELECTION_FEEDBACK",
                  "P3_WEAK_H_RECOVERY", "SYNCHRONIZED_MECHANISM", "CAUSALITY_LEVEL"):
            L.append(f"{k:<28}= {d[k]}")
        L += ["```", "", "## 时间序列", "", "| epoch | P3 acc(G86) | P3 acc(CTRL) | h acc(G86) | mask_pos率 | cls logit | raw_align |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
        for i, e in enumerate(d["epoch_points"]):
            L.append(f"| {e} | {d['p3_accuracy_g86'][i]} | {d['p3_accuracy_control'][i]} | "
                     f"{d['h_accuracy_g86'][i]} | {d['mask_pos_rate'][i]:.3f} | "
                     f"{d['cls_logit_median'][i]:.3f} | {d['raw_align_median'][i]:.5f} |")
        L += ["", f"random baseline（control 的多数类先验）= {d['random_baseline_majority_prior']}", "",
              "## 时间顺序", "",
              f"- 首次 P3 退化：{d['first_P3_degradation']}",
              f"- 首次 cls 退化：{d['first_cls_degradation']}",
              f"- 首次 alignment 退化：{d['first_alignment_degradation']}",
              f"- 首次 mask_pos 丢失：{d['first_mask_pos_loss']}", "",
              f"> {d['NOTE']}", ""]
        return "\n".join(L)

    # ---------- ultralytics 回调 ----------
    def on_fit_epoch_end(self, trainer):
        from ultralytics.utils.torch_utils import de_parallel
        ep = int(trainer.epoch)
        if self._ds is None:
            self.build()
        if ep in self.obs:
            recs = self.observe(de_parallel(trainer.model), ep)
            self.dump_epoch(ep, recs)
            ng = sum(1 for r in recs if r["role"] == "G86")
            print(f"[probe] epoch {ep}: G-86={ng} / medium-cell records={len(recs)}, "
                  f"cell vectors={len(self.vecs[ep]['vec_keys'])}", flush=True)
        # ★ 保险：在**最后一个观测点就地 finalize**，产物不再依赖 on_train_end 是否触发。
        #   云端首跑实测 trainer.stop=True 未使循环 break（根因待 _run.log 确认），
        #   若产物只在 on_train_end 写，一旦未停机/被手动中止就会全部缺失。
        if not getattr(self, "_finalized", False) and ep >= max(self.obs):
            self.finalize()
            self._finalized = True
            print(f"[probe] finalized at last observation epoch {ep} -> 产物已落盘", flush=True)
        if ep >= self.stop_after:
            trainer.stop = True
            print(f"[probe] stop_after_epoch={self.stop_after} reached -> trainer.stop=True", flush=True)

    def on_train_end(self, trainer):
        if not getattr(self, "_finalized", False):
            self.finalize()
            self._finalized = True
        print("[probe] finalized: " + str(sorted(self.vecs)), flush=True)
