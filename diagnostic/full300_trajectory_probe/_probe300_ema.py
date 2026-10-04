"""_probe300_ema.py — Full-300 EMA trajectory probe（**部署口径**）。

与旧 `_probe300.py` 的三处实质修正（旧版已判 INVALID）：

  ① **probe target**：`trainer.ema.ema`（不是 `trainer.model`）。
     D′ 的 val / best.pt / 部署都走 EMA：`validator.py:118`、`trainer.py:524-525`、
     `torch_utils.py:610-611`。旧版追 raw model ⇒ 全部 trajectory 作废。

  ② **cell 坐标**：用 `medium_cells_canvas.json` 的 **frozen `p3_cell_x/p3_cell_y`**
     （dataset `lab["bboxes"]` × 1280，含 letterbox pad）。
     旧版用「标签文件归一化中心 × 1280」，漏 pad ⇒ |Δy| 中位 9.6 cell、最大 34.4 cell。

  ③ **mode**：纯 `model.eval()`（= 正常 validation/inference 语义）。
     旧版 `eval backbone + head.train()` 是历史 diagnostic 口径，**不作为正式机制口径**。
     为在 eval 下拿到 raw 三尺度特征，改为 hook `model[-1]` 的 **pre-hook**（Detect 的输入
     就是 feats 列表），从而彻底不需要把 head 切回 train。

§8 P4 语义：EMA 会被正常 `ema.update()` 持续改写 —— 本项**不要求** EMA 参数全程不变，
只要求 **probe observation 本身不额外修改**。因此每次 observe 都做
snapshot(flags/buffers/params/RNG) → eval → 只读观测 → restore → verify。
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "p3_trajectory_probe"))
from _probe import RawTAL, P3TrajProbe, v8DetectionLoss  # noqa: E402

# §9 观测点（0-based）。results.csv 写 self.epoch + 1（trainer.py:666）⇒ 0-based e ↔ csv e+1
OBSERVE_0B = (0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 289, 299)

CELLS = HERE.parent / "p3_trajectory_probe" / "medium_cells_canvas.json"

# logit 恒等式的判据：**相对**残差上限（对 logit 量级归一）。
# 实测参考：CPU(FP32) ≈ 4e-06；云端 GPU(cuDNN + TF32) ≈ 1.0e-3 绝对（logit ~10 ⇒ 相对 ~1e-4）。
# 恒等式真被破坏（hook 错层 / cell 错位）时相对残差为 O(1)，因此 1e-2 仍足够严格。
IDENTITY_TOL_REL = 1e-2


class EmaProbe(P3TrajProbe):
    """部署口径 EMA probe。"""

    def __init__(self, cells_json=CELLS, out_dir=HERE, observe_epochs=OBSERVE_0B):
        super().__init__(cells_json, out_dir, observe_epochs=observe_epochs,
                         stop_after_epoch=10 ** 9)   # §3 不早停
        self.restore_log = []
        self.identity_log = []
        self.timing_log = []          # §6 运行期时机证据
        self._updates_prev = 0
        self._last_seen_epoch = -1    # §6 门幂等：final_eval() 会再触发一次同名回调
        self._cell = {r["key"]: (int(r["p3_cell_x"]), int(r["p3_cell_y"])) for r in self.cells}

    # ---------- hooks ----------
    # ★ 关键：attach_model **不注册任何 hook**，只保存引用（W/b/loss/assigner）。
    #   hook 改在每次观测期间临时注册、finally 里注销（_register/_unregister）。
    #
    #   原因（2026-10-02 云端实测崩在 epoch 0 末）：
    #     trainer.save_model() 执行 torch.save(deepcopy(self.ema.ema))（trainer.py:520-525）。
    #     既然 probe 现在挂的是 **EMA 本体**，注册在它上面的 forward hook 会被一起 pickle；
    #     而 hook 是 `_mk_out.<locals>.h` 这样的局部闭包 ⇒
    #       AttributeError: Can't pickle local object 'P3TrajProbe._mk_out.<locals>.h'
    #     旧 probe 侥幸没崩，只是因为它挂的是 raw model，被 pickle 的是**无 hook 的 EMA**。
    #   副作用（正向）：模型在观测之外始终是干净的，P4 的「非侵入」因此更强。
    def attach_model(self, nn_model):
        last = nn_model.model[-1].cv3[0][-1]
        self._nn_ref = nn_model
        self._nl = len(nn_model.model[-1].cv2)
        self._feat = {}
        self._Wn = last.weight.detach().cpu().view(last.weight.shape[0], -1).numpy().astype(np.float64)
        self._bn = last.bias.detach().cpu().numpy().astype(np.float64)
        self._loss = v8DetectionLoss(nn_model)
        self._assigner = RawTAL(topk=10, num_classes=self._loss.nc, alpha=0.5, beta=6.0)
        self._refresh_head(nn_model)

    def _refresh_head(self, nn_model):
        """★ 必须在**每个观测点**重新读取分类头权重。
        2026-10-03 事故：W/b 只在 attach 时取一次 ⇒ head_W 在 19 个 epoch 上逐位相同
        （max|W−W_ep0| = 0.0），npz 的 logit 列变成 W_init·h_t 而**不是模型真实 logit**，
        logit 恒等式从 ep1 起 O(1) 失败（rel 0.63→0.96），W_t 轨迹 / T3 / T4 / 分类器漂移全部作废。
        旧 _probe300.py:57-58 是在 observe() 内逐 epoch 取的，重写时漏掉了这一步。"""
        last = nn_model.model[-1].cv3[0][-1]
        self._Wn = last.weight.detach().cpu().view(last.weight.shape[0], -1).numpy().astype(np.float64)
        self._bn = last.bias.detach().cpu().numpy().astype(np.float64)

    def _register(self, nn_model):
        """观测期间临时挂 hook。返回 handles，必须在 finally 里 _unregister。"""
        last = nn_model.model[-1].cv3[0][-1]
        det = nn_model.model[-1]
        hs = [nn_model.model[23].register_forward_hook(self._mk_out("P3")),
              last.register_forward_pre_hook(self._mk_pre("H"))]
        # eval 下拿 raw feats：Train 时 Detect 返回 cat(cv2[i], cv3[i])，eval 时走 _inference
        # 不再返回该张量。直接 hook 两个分支再自行 cat，即可在纯 eval 下复现同一张量。
        for i in range(self._nl):
            hs.append(det.cv2[i].register_forward_hook(self._mk_out(f"CV2_{i}")))
            hs.append(det.cv3[i].register_forward_hook(self._mk_out(f"CV3_{i}")))
        hs.append(last.register_forward_hook(self._mk_out("CLS_OUT")))
        return hs

    @staticmethod
    def _unregister(handles):
        for h in handles:
            h.remove()

    @staticmethod
    def _count_hooks(m):
        """残留 hook 计数 —— 收尾自检用（必须恒为 0）。"""
        n = 0
        for mod in m.modules():
            n += (len(getattr(mod, "_forward_hooks", {}))
                  + len(getattr(mod, "_forward_pre_hooks", {}))
                  + len(getattr(mod, "_backward_hooks", {})))
        return n

    def _raw_feats(self):
        """(B, 4*reg_max+nc, H, W) 三尺度，等价 train-mode Detect 的输出。"""
        out = []
        for i in range(self._nl):
            a, b = self._feat.get(f"CV2_{i}"), self._feat.get(f"CV3_{i}")
            if a is None or b is None:
                return None
            out.append(torch.cat((a, b), 1))
        return out

    @staticmethod
    def _checksum(m):
        with torch.no_grad():
            return (round(float(sum(float(p.detach().float().sum()) for p in m.parameters())), 3),
                    round(float(max(float(p.detach().abs().max()) for p in m.parameters())), 6))

    def _snap(self, m):
        return dict(flags=[x.training for x in m.modules()],
                    bufs={k: v.detach().clone() for k, v in m.named_buffers()},
                    paramsum=self._checksum(m),      # ★ 观测**前**的 checksum
                    rng=(random.getstate(), np.random.get_state(), torch.get_rng_state()))

    def _restore(self, m, s):
        for x, f in zip(m.modules(), s["flags"]):
            x.training = f
        # ★ 缓冲区还原用**重绑定**，不用 in-place `copy_`。
        #   根因（2026-10-02 云端第二次崩溃）：
        #     BaseValidator.__call__ 整体被 @smart_inference_mode() 包裹（validator.py:108），
        #     且在其内部执行 model.half() / model.float()（validator.py:119 / 208，GPU+AMP 时
        #     args.half=True）。nn.Module._apply 在 inference_mode **内**新建并替换
        #     buffer/param.data ⇒ validate() 之后 EMA 的 buffer 全是 inference tensor。
        #     对它做 `v.copy_(...)` 即抛
        #       "Inplace update to inference tensor outside InferenceMode is not allowed"。
        #   重绑定不动 inference tensor，并把它换回普通 tensor，两全。
        #   另：纯 eval 观测本就不该改 buffer，故**先比较、相同就跳过**（这也是 P4 的证据）。
        with torch.no_grad():
            for k, v in list(m.named_buffers()):
                ref = s["bufs"].get(k)
                if ref is None or torch.equal(v.detach(), ref):
                    continue
                mod, name = m, k
                if "." in k:
                    path, name = k.rsplit(".", 1)
                    for part in path.split("."):
                        mod = getattr(mod, part)
                mod._buffers[name] = ref.clone()
        random.setstate(s["rng"][0]); np.random.set_state(s["rng"][1])
        torch.set_rng_state(s["rng"][2])

    def _verify(self, m, s):
        return dict(mode_flags_restored=all(x.training == f for x, f in zip(m.modules(), s["flags"])),
                    bn_buffers_restored=all(torch.equal(v.detach(), s["bufs"][k])
                                            for k, v in m.named_buffers()),
                    # ★ 与**观测前**快照比（不是与 attach 时的冻结值比 —— EMA 本就被逐 epoch 改写）
                    params_untouched=(self._checksum(m) == s["paramsum"]),
                    rng_restored=(random.getstate() == s["rng"][0]
                                  and torch.equal(torch.get_rng_state(), s["rng"][2])))

    # ---------- 单次观测（部署口径） ----------
    def observe(self, nn_model, epoch):
        self._refresh_head(nn_model)           # ★ W_t = 本 epoch 的分类头权重
        s = self._snap(nn_model)
        hs = self._register(nn_model)          # ★ hook 仅存活于本次观测
        recs, vk, vc, vr, vp, vh = [], [], [], [], [], []
        resid, rel = 0.0, 0.0
        dev = next(nn_model.parameters()).device
        try:
            nn_model.eval()                              # ★ 纯 eval（§5/§6）
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                for i in self._sel:
                    lab = self._ds[i]
                    bb, cl = lab.get("bboxes"), lab.get("cls")
                    if bb is None or not len(bb):
                        continue
                    stem = Path(self._ds.im_files[i]).stem
                    img = lab["img"].float().div(255.0)
                    img = (img.unsqueeze(0) if img.ndim == 3 else img).to(dev)
                    self._feat.clear()
                    nn_model(img)
                    feats = self._raw_feats()
                    P3, Hh, OUT = self._feat.get("P3"), self._feat.get("H"), self._feat.get("CLS_OUT")
                    if feats is None or P3 is None or Hh is None or OUT is None:
                        continue
                    # ---- TAL（只读；raw_align 在 mask 之前 clone）----
                    pd_d, pd_s = torch.cat([x.view(1, self._loss.no, -1) for x in feats], 2).split(
                        (self._loss.reg_max * 4, self._loss.nc), 1)
                    pd_s = pd_s.permute(0, 2, 1).contiguous(); pd_d = pd_d.permute(0, 2, 1).contiguous()
                    _ap, st = make_anchors(feats, self._loss.stride, 0.5)
                    imgsz = torch.tensor(feats[0].shape[2:], dtype=pd_s.dtype, device=pd_s.device) \
                        * self._loss.stride[0]
                    tg = torch.cat((torch.zeros(len(bb), 1, device=bb.device), cl.view(-1, 1), bb), 1)
                    t = self._loss.preprocess(tg, 1, scale_tensor=imgsz[[1, 0, 1, 0]])
                    gt_l, gt_b = t.split((1, 4), 2)
                    self._assigner.forward(pd_s.detach().sigmoid(),
                                           (self._loss.bbox_decode(_ap, pd_d).detach() * st).type(gt_b.dtype),
                                           _ap * st, gt_l, gt_b, gt_b.sum(2, keepdim=True).gt_(0))
                    L = self._assigner.last; apx = _ap * st; sc_all = pd_s[0].sigmoid()
                    for r in self._cell_by_stem[stem]:
                        j = r["gt_id"]; cid = r["class_id"]
                        x, y = self._cell[r["key"]]           # ★ frozen cell 下标
                        p3v = P3[0, :, y, x].detach().cpu().numpy().astype(np.float32)
                        hv = Hh[0, :, y, x].detach().cpu().numpy().astype(np.float32)
                        vk.append(r["key"]); vc.append(cid); vr.append(r["role"])
                        vp.append(p3v); vh.append(hv)
                        assert isinstance(hv, np.ndarray) and isinstance(self._Wn, np.ndarray)
                        lg = float(self._Wn[cid] @ hv.astype(np.float64) + self._bn[cid])
                        _a = float(OUT[0, cid, y, x])
                        _d = abs(_a - lg)
                        resid = max(resid, _d)
                        rel = max(rel, _d / (abs(_a) + abs(lg) + 1.0))
                        pos = L["mask_pos"][0, j].bool(); topk = L["mask_topk"][0, j].bool()
                        ov = L["overlaps"][0, j]; ra = L["raw_align"][0, j]
                        gb_ = gt_b[0, j]; c = (gb_[:2] + gb_[2:]) / 2
                        near = torch.linalg.norm(apx - c[None, :], dim=1) <= 12.0
                        recs.append(dict(
                            epoch=epoch, key=r["key"], class_id=cid, role=r["role"],
                            p3_cell_x=x, p3_cell_y=y,
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
            self._unregister(hs)               # ★ 必须在 save_model() 之前清干净
            self._restore(nn_model, s)
        self.vecs[epoch] = dict(vec_keys=vk, vec_cls=vc, vec_role=vr,
                                vec_p3=np.asarray(vp, np.float32), vec_h=np.asarray(vh, np.float32))
        self.restore_log.append(dict(epoch=epoch, **self._verify(nn_model, s)))
        self.identity_log.append(dict(epoch=epoch, n=len(recs), max_abs_resid=resid,
                                      max_rel_resid=rel, tol_rel=IDENTITY_TOL_REL))
        return recs

    # ---------- §8/§12 in-process P4 gate（训练启动时先跑，FAIL 即中止） ----------
    def self_check(self, nn_model, n_img: int = 2):
        """在**真实 trainer.ema.ema** 上验证：probe observation 非侵入 + logit 恒等式 + 观测可用。"""
        if self._ds is None:
            self.build()
        # 判据说明：`OUT` 由 cuDNN 卷积产生，GPU 上默认允许 TF32（10-bit mantissa ⇒ ~1e-3 绝对误差，
        # logit 量级 ~10 时）。本机 CPU 实测 ~4e-06，云端 GPU 实测 1.0e-3 —— 都是同一恒等式。
        # 因此门限用**相对**残差（对 logit 量级归一），绝对残差仅记录。
        # 恒等式若真被破坏（hook 错层 / cell 错位 / 索引错），相对残差会是 O(1)，仍会被拦住。
        out = dict(probe_target="trainer.ema.ema", n_img=0,
                   identity_max_abs_resid=0.0, identity_max_rel_resid=0.0)
        self._refresh_head(nn_model)
        s = self._snap(nn_model)
        hs = self._register(nn_model)
        dev = next(nn_model.parameters()).device
        n = 0
        try:
            nn_model.eval()
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                for i in self._sel:
                    if n >= n_img:
                        break
                    lab = self._ds[i]
                    if lab.get("bboxes") is None or not len(lab["bboxes"]):
                        continue
                    self._feat.clear()
                    img = lab["img"].float().div(255.0)
                    img = (img.unsqueeze(0) if img.ndim == 3 else img).to(dev)
                    nn_model(img)
                    Hh, OUT = self._feat.get("H"), self._feat.get("CLS_OUT")
                    if Hh is None or OUT is None:
                        continue
                    n += 1
                    stem = Path(self._ds.im_files[i]).stem
                    for r in self._cell_by_stem[stem]:
                        cid = r["class_id"]; x, y = self._cell[r["key"]]
                        hv = Hh[0, :, y, x].detach().cpu().numpy().astype(np.float64)
                        actual = float(OUT[0, cid, y, x])
                        recon = float(self._Wn[cid] @ hv + self._bn[cid])
                        d = abs(actual - recon)
                        out["identity_max_abs_resid"] = max(out["identity_max_abs_resid"], d)
                        out["identity_max_rel_resid"] = max(
                            out["identity_max_rel_resid"], d / (abs(actual) + abs(recon) + 1.0))
        finally:
            self._unregister(hs)
            self._restore(nn_model, s)
        out["n_img"] = n
        out["hooks_left_on_model"] = self._count_hooks(nn_model)   # 必须为 0
        # ★ 复现 trainer.save_model() 的真实序列（trainer.py:520-525）：
        #   torch.save({... "ema": deepcopy(self.ema.ema).half() ...})
        #   hook 是局部闭包 ⇒ 若观测后未清干净，这里会复现
        #   "AttributeError: Can't pickle local object '..._mk_out.<locals>.h'"。
        import copy as _copy
        import io as _io
        try:
            _buf = _io.BytesIO()
            torch.save({"model": None, "ema": _copy.deepcopy(nn_model).half(),
                        "updates": 0, "optimizer": None, "train_args": {}}, _buf)
            out["picklable_like_save_model"] = True
        except Exception as e:  # noqa: BLE001
            out["picklable_like_save_model"] = False
            out["picklable_error"] = f"{type(e).__name__}: {e}"
        out.update(self._verify(nn_model, s))
        out["identity_tol_rel"] = IDENTITY_TOL_REL
        out["PASS"] = bool(n >= 1 and out["mode_flags_restored"] and out["bn_buffers_restored"]
                           and out["params_untouched"] and out["rng_restored"]
                           and out["identity_max_rel_resid"] < IDENTITY_TOL_REL
                           and out["hooks_left_on_model"] == 0
                           and out["picklable_like_save_model"])
        return out

    # ---------- 落盘：加 W_t / b_t（§11 四分解用） ----------
    def dump_epoch(self, epoch, recs):
        super().dump_epoch(epoch, recs)
        p = self.out / f"epoch_{int(epoch):03d}.npz"
        d = dict(np.load(p, allow_pickle=False))
        d["head_W"] = self._Wn
        d["head_b"] = self._bn
        np.savez_compressed(p, **d)

    def finalize(self):
        dec = super().finalize()
        for src, dst in (("per_gt_trajectory.csv", "trajectory_per_gt.csv"),
                         ("epoch_summary.csv", "trajectory_summary.csv"),
                         ("tal_metrics.csv", "trajectory_tal.csv")):
            f = self.out / src
            if f.exists():
                (self.out / dst).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
        (self.out / "trajectory_metadata.json").write_text(json.dumps(dict(
            probe_target="trainer.ema.ema",
            probe_mode="eval",
            coordinate_source="dataset_lab_bboxes_canvas_normalized",
            cells_file=str(CELLS.name),
            epochs=300,
            observation_epochs=list(self.obs),
            indexing="0-based（csv = 0-based + 1，trainer.py:666）",
            sample_definition="G86 (86) / CONTROL (1070 medium detected) / MISSED_NON86 (164)；共 1320",
            close_mosaic=10,
            observation_timing="after_normal_ema_update",
            observation_timing_evidence=dict(
                code_path=("trainer.py:344 self.epoch=epoch → ~380 batch loop "
                           "optimizer_step() →587-595 ema.update() →454 on_fit_epoch_end"),
                runtime=self.timing_log),
            early_stop="NONE",
            state_restoration=self.restore_log,
            logit_identity=self.identity_log,
        ), indent=2, ensure_ascii=False), encoding="utf-8")
        return dec

    # ---------- 回调 ----------
    def on_fit_epoch_end(self, trainer):
        ep = int(trainer.epoch)
        # ★ 幂等：trainer.final_eval() 在 :691 会**再次**触发 on_fit_epoch_end，
        #   此时 self.epoch 仍是 299、ema.updates 已停 ⇒ 原 §6 检查会假阳性 raise
        #   （2026-10-03 实测：300 epoch 全部跑完后在收尾阶段崩掉）。
        if ep == self._last_seen_epoch:
            return
        self._last_seen_epoch = ep
        if self._ds is None:
            self.build()
        # ---------- §6 观测时机：runtime 证据 ----------
        # 代码路径（ultralytics/engine/trainer.py）：
        #   344  self.epoch = epoch
        #   348  self.scheduler.step()
        #   ~380 for batch: ... optimizer_step()  →  587-595 scaler.step(); scaler.update();
        #                                              zero_grad(); self.ema.update(self.model)
        #   454  self.run_callbacks("on_fit_epoch_end")     ★ 本回调
        # ⇒ on_fit_epoch_end 必然晚于本 epoch 全部 optimizer step 与对应 ema.update()。
        # ModelEMA.update() 每次 +1（torch_utils.py:554）⇒ 用 updates 增量做运行期证明。
        got = int(trainer.ema.updates)
        delta = got - self._updates_prev
        self._updates_prev = got
        self.timing_log.append(dict(
            epoch=ep, ema_updates_total=got, ema_updates_this_epoch=delta,
            batches=len(trainer.train_loader),
            after_normal_ema_update=bool(delta > 0)))
        if delta <= 0:
            raise RuntimeError(
                f"§6 违反：0-based ep{ep} 结束时 trainer.ema.updates 未增加"
                f"（total={got}）⇒ EMA 未在本 epoch 更新过 ⇒ STATUS=INVALID")
        if ep in self.obs:
            recs = self.observe(_ema_of(trainer), ep)
            self.dump_epoch(ep, recs)
            print(f"[ema-probe] 0-based ep{ep} (csv ep{ep+1}): records={len(recs)}  "
                  f"ema_updates=+{delta} (total {got})", flush=True)
            # ---------- ★ FAIL FAST ----------
            # 2026-10-03 事故：W/b 未随 epoch 刷新 ⇒ npz 的 logit 列作废，
            # 但脚本一路跑到 ep299（12 h）才被发现。实测该 bug 在 **ep0 就已经** 让
            # rel = 0.44。这里把恒等式升级为硬门：**第一个观测点就停**，代价从 12 h 压到 ~2 min。
            # 阈值余量：修好后实测 rel ~1e-6（CPU）/ ~6e-5（GPU·TF32），TOL=1e-2 ⇒ 100× 余量。
            # 先把 npz 落盘再抛，保证失败现场有证据。
            il = self.identity_log[-1]
            if il["max_rel_resid"] >= IDENTITY_TOL_REL:
                raise RuntimeError(
                    f"logit 恒等式在 0-based ep{ep} 失败：rel={il['max_rel_resid']:.3e} "
                    f">= TOL={IDENTITY_TOL_REL}（abs={il['max_abs_resid']:.3e}）。"
                    f"典型成因：分类头 W/b 未随观测刷新（W_t 轨迹作废，机制分类不可得）。"
                    f"已保留 ep{ep} 的 npz 作为现场证据。STATUS=INVALID。")
        if not getattr(self, "_finalized", False) and ep >= max(self.obs):
            self.finalize(); self._finalized = True
            print(f"[ema-probe] finalized at 0-based ep{ep}", flush=True)

    def on_train_end(self, trainer):
        if not getattr(self, "_finalized", False):
            self.finalize(); self._finalized = True


def _ema_of(trainer):
    """★ 唯一允许的 probe target。"""
    from ultralytics.utils.torch_utils import de_parallel
    assert getattr(trainer, "ema", None) is not None, "trainer.ema 不存在 —— 拒绝 probe"
    return de_parallel(trainer.ema.ema)


# make_anchors 需要从 _probe 的命名空间导入
from ultralytics.utils.tal import make_anchors  # noqa: E402
