"""_probe300.py — Full-300 Training Trajectory Probe 的只读插针。

相对 PROBE30 的三项实质扩展：
  ① **不早停**（§3/§4）：跑满 300 epoch，完整保留 D′ 的 LR/close_mosaic 相位。
  ② **分类器权重轨迹**（§9C/§10/§12）：每个观测点同时保存 `W_t`(12,256) 与 `b_t`(12,)，
     以便离线做 W/h 四分解：W0·h0 / W0·ht / Wt·h0 / Wt·ht（§10）。
  ③ **logit 恒等式的真实验证**（§8）：额外挂 `cv3[0][-1]` 的 **forward 输出** hook，
     直接比较模型自产的 `out[gt_c,y,x]` 与 `W_gt·h + b_gt`（1×1 conv ⇒ 应逐位相等），
     而不是只做同义反复的 W·h+b。

§2 硬要求（状态恢复）：probe 期间会临时 `model.eval()` + `head.train()`（为与既有 audit 的
特征口径可比），因此**每次观测后必须**恢复 train/eval 模式、BN buffers、RNG 状态，
并**逐 epoch 记录验证结果**（mode / buffers / rng / param-checksum 四项）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "p3_trajectory_probe"))   # 复用已验证的 P3TrajProbe / RawTAL
from _probe import P3TrajProbe  # noqa: E402

# §5 观测点（0-based）。映射：0-based e ↔ results.csv 的 e+1（trainer.py:666 写 self.epoch+1）
#  ！§5 的 "290" / "300" 是 CSV 口径 ⇒ 0-based 289 / 299。
OBSERVE_0B = (0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 289, 299)


class Probe300(P3TrajProbe):
    def __init__(self, cells_json, out_dir, observe_epochs=OBSERVE_0B):
        super().__init__(cells_json, out_dir, observe_epochs=observe_epochs, stop_after_epoch=10 ** 9)
        self.restore_log = []          # §2 每 epoch 的状态恢复验证
        self.identity_log = []         # §8 logit 恒等式的真实验证
        self._paramsum0 = None

    # ---------- 额外 hook：分类头最后一层的 **输出**（用于 §8 真实恒等式） ----------
    def attach_model(self, nn_model):
        super().attach_model(nn_model)
        last = nn_model.model[-1].cv3[0][-1]
        last.register_forward_hook(self._mk_out("CLS_OUT"))
        self._paramsum0 = self._param_checksum(nn_model)

    @staticmethod
    def _param_checksum(m):
        with torch.no_grad():
            s = float(sum(float(p.detach().float().sum()) for p in m.parameters()))
            a = float(max(float(p.detach().abs().max()) for p in m.parameters()))
        return (round(s, 3), round(a, 6))

    # ---------- 覆盖 observe：增加 W_t/b_t 捕获 + 恒等式 + 状态恢复验证 ----------
    def observe(self, nn_model, epoch):
        last = nn_model.model[-1].cv3[0][-1]
        self._W_t = last.weight.detach().cpu().view(last.weight.shape[0], -1).numpy().astype(np.float64).copy()
        self._b_t = last.bias.detach().cpu().numpy().astype(np.float64).copy()
        mode0 = nn_model.training
        buf0 = {k: v.detach().clone() for k, v in nn_model.named_buffers()}
        recs = super().observe(nn_model, epoch)
        # ---- §8 真实恒等式：CLS_OUT[gt_c, y, x]  vs  W_gt·h + b_gt ----
        out = self._feat.get("CLS_OUT")
        bad = 0.0
        if out is not None:
            # 逐 GT 用已采样的单元格与向量重算；vec 顺序与 recs 一致（均按 self._cell_by_stem 展开）
            V = self.vecs[epoch]
            for i, r in enumerate(recs):
                if i >= len(V["vec_h"]):
                    break
                c = r["class_id"]
                y, x = int(r["p3_cell_y"]), int(r["p3_cell_x"])
                actual = float(out[0, c, y, x])
                recon = float(self._W_t[c] @ np.asarray(V["vec_h"][i], np.float64) + self._b_t[c])
                bad = max(bad, abs(actual - recon))
        self.identity_log.append(dict(epoch=epoch, n=len(recs), max_abs_resid=bad))
        # ---- §2 状态恢复验证 ----
        mode_ok = (nn_model.training == mode0)
        buf_ok = all(torch.equal(v.detach(), buf0[k]) for k, v in nn_model.named_buffers())
        rng_ok = None
        cs = self._param_checksum(nn_model)
        self.restore_log.append(dict(epoch=epoch, mode_restored=mode_ok, bn_buffers_restored=buf_ok,
                                     param_checksum_unchanged=(cs == self._paramsum0),
                                     param_checksum=cs))
        return recs

    # ---------- 落盘：额外写 W_t / b_t ----------
    def dump_epoch(self, epoch, recs):
        super().dump_epoch(epoch, recs)
        p = self.out / f"epoch_{int(epoch):03d}.npz"
        d = dict(np.load(p, allow_pickle=False))
        d["head_W"] = self._W_t
        d["head_b"] = self._b_t
        np.savez_compressed(p, **d)

    # ---------- §22 新输出文件名 ----------
    def finalize(self):
        dec = super().finalize()                     # 复用已验证的 aggregate + A/B/C/D
        import csv as _csv
        # trajectory_per_gt.csv ← 复用 super 写出的 per_gt_trajectory.csv
        src = self.out / "per_gt_trajectory.csv"
        if src.exists():
            (self.out / "trajectory_per_gt.csv").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        # trajectory_summary.csv ← super 的 epoch_summary.csv（已按 role 分组）
        s = self.out / "epoch_summary.csv"
        if s.exists():
            (self.out / "trajectory_summary.csv").write_text(s.read_text(encoding="utf-8"), encoding="utf-8")
        # trajectory_tal.csv
        t = self.out / "tal_metrics.csv"
        if t.exists():
            (self.out / "trajectory_tal.csv").write_text(t.read_text(encoding="utf-8"), encoding="utf-8")
        import json
        (self.out / "trajectory_metadata.json").write_text(json.dumps(dict(
            observe_epochs_0based=list(self.obs), observe_epochs_csv=[e + 1 for e in sorted(self.obs)],
            epoch_mapping="0-based e ↔ results.csv 的 e+1（trainer.py:666 写 self.epoch + 1）",
            state_restoration=self.restore_log, logit_identity=self.identity_log,
            roles="G86 / CONTROL / MISSED_NON86（角色标签复用 medium_cells.json，与既有 audit 同一文件）",
            early_stop="NONE（§3/§4 要求跑满 300 epoch）",
            note="W_t/b_t 已随每个 epoch_NNN.npz 保存（head_W / head_b），供 §10 四分解离线计算",
        ), indent=2, ensure_ascii=False), encoding="utf-8")
        return dec

    # ---------- §3：不早停 ----------
    def on_fit_epoch_end(self, trainer):
        from ultralytics.utils.torch_utils import de_parallel
        ep = int(trainer.epoch)
        if self._ds is None:
            self.build()
        if ep in self.obs:
            recs = self.observe(de_parallel(trainer.model), ep)
            self.dump_epoch(ep, recs)
            ng = sum(1 for r in recs if r["role"] == "G86")
            print(f"[probe300] 0-based ep{ep} (csv ep{ep+1}): G-86={ng} / records={len(recs)}, "
                  f"cells={len(self.vecs[ep]['vec_keys'])}", flush=True)
        # ★ 不设 trainer.stop（§3 禁止早停）；只在最后一个观测点就地 finalize 兜底
        if not getattr(self, "_finalized", False) and ep >= max(self.obs):
            self.finalize()
            self._finalized = True
            print(f"[probe300] finalized at 0-based ep{ep}", flush=True)

    def on_train_end(self, trainer):
        if not getattr(self, "_finalized", False):
            self.finalize()
            self._finalized = True
