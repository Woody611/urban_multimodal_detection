# FULL-300 EMA PROBE — RUN REPORT（2026-10-03）

```text
FULL-300 EMA PROBE STATUS: INVALID
```

**FAIL：**
- §16 `logit identity passes` —— **FAIL**（相对残差 0.44 → 0.96，O(1)，系统性增长）
- §16 `no unexpected exception` —— **FAIL**（收尾阶段 `RuntimeError`，我的 §6 门假阳性）
- `P4 / params_untouched` —— 每 epoch 均为 False（判据实现错误，非真实污染）

**ROOT CAUSE（三处，全部在我的 probe 里，数据链本身没坏）：**

1. **`head_W`/`head_b` 只在 attach 时取一次，从未刷新**（`_probe300_ema.py:77-78`）。
   实测 `head_W` 在 19 个 epoch 上**逐位相同**（`max|W−W_ep0| = 0.0`）。
   ⇒ npz 的 `logit` 列 = `W_init · h_t + b_init`，**不是模型真实 logit**；
   logit 恒等式从 ep1 起 O(1) 失败（abs 1.8 → 25.7）。
   *旧 `_probe300.py:57-58` 是在 `observe()` 内逐 epoch 取的 —— 我重写成 EMA 版时漏了这一步。*
2. **`_paramsum0` 冻结于 attach 时刻**（`:81`，在 `:164` 比较）。EMA 本就被逐 epoch 改写 ⇒ 恒为 False。
   这是**判据实现错误**：P4 应比较「观测前后」，不是「与 attach 时比」。
3. **§6 门不幂等**。`trainer.final_eval()` 在 `trainer.py:691` **再次**触发 `on_fit_epoch_end`，
   此时 `trainer.epoch` 仍是 299、`ema.updates` 已停 ⇒ `delta=0` ⇒ 假阳性 raise。

---

## §16 完整性逐项

| # | 检查项 | 结果 |
|---|---|---|
| 1 | 300 training epochs completed | ✅ `results.csv` 300 行（csv 1..300） |
| 2 | final 0-based epoch = 299 | ✅ |
| 3 | final CSV epoch = 300 | ✅ |
| 4 | final EMA observation exists | ✅ `epoch_299.npz` |
| 5 | P4 runtime gate PASS | ✅ 见下（但 per-epoch `params_untouched` 字段无效） |
| 6 | all observation epochs exist | ✅ 19/19 |
| 7 | coordinate source correct | ✅ `dataset_lab_bboxes_canvas_normalized` |
| 8 | **logit identity passes** | ❌ **FAIL** |
| 9 | no NaN/Inf | ⚠ 仅 180/12,991,440 = 0.7%，且全部是 `*_pos_max` 的**设计内哨兵**（该 GT 该 epoch 无正样本 anchor）；模型输出无 NaN |
| 10 | no unexpected exception | ❌ 收尾 `RuntimeError`（缺陷 3） |
| 11 | D′ training configuration unchanged | ✅ 19 项逐项一致（epochs/batch/imgsz/seed/box/cls/dfl/ir_encoding=clahe/…） |

**§21 三条 INVALID 触发器（epoch 299 完成 / EMA observation 在正常 update 之后 / P4 gate 失败）均未触发** ——
数据链是干净的；但 §16 第 8 项失败，按 §19 不进入机制分析。

---

## 有效的部分（数据链证据）

- **probe target** = `trainer.ema.ema` ✅（metadata + 日志 `attached to trainer.ema.ema`）
- **probe mode** = `eval` ✅
- **§6 时机**：**300/300** 个 epoch 的 `ema_updates` 增量 > 0（`[109,46,29,25,25,…]`），
  即每次观测都发生在该 epoch 的 `ema.update()` 之后 —— 运行期证明，非源码推断
- **P4 runtime gate**（训练前，真实 ema.ema 上）：
  `identity_rel=6.30e-05`、`hooks_left_on_model=0`、`picklable_like_save_model=True`、
  `mode_flags/bn_buffers/params/rng` 四项全 True ⇒ **PASS**
- **两个 checkpoint 均已正常 strip**（`best.pt`/`last.pt`：`ema=None, epoch=-1, optimizer=None`，40.6 MB）——
  即 `final_eval()` 的 strip 与最终 val **都已完成**，崩溃发生在其后的回调里
- **仪器活性**（仅确认非退化，**非**机制结论）：

| ep | ‖h‖ mean G86 | ‖h‖ mean CTRL | cos(h_t, h_0) G86 |
|---:|---:|---:|---:|
| 0 | 50.04 | 66.51 | +1.000 |
| 20 | 34.77 | 43.46 | +0.848 |
| 100 | 36.82 | 44.31 | +0.846 |
| 200 | 40.36 | 49.85 | +0.780 |
| 299 | 40.63 | 53.93 | +0.752 |

## 作废的部分

| 量 | 状态 | 原因 |
|---|---|---|
| npz `logit` 列 | ❌ 作废 | = `W_init·h_t`，非模型真实 logit |
| `head_W` / `head_b` | ❌ 作废 | 19 个 epoch 恒为 W_init |
| §12 classifier 轨迹 | ❌ 作废 | 同上 |
| §13 `T3 = W_t·h_0`、`T4 = W_t·h_t` | ❌ 作废 | 需 W_t |
| §14 `Δ_|W|`、`Δ_cosW` | ❌ 作废 | 同上 |
| §10 classifier drift | ❌ 作废 | 同上 |
| **§15 机制分类 A/B/C/D** | ⛔ **无法给出** | 分离 h 漂移与 W 漂移是本问题的核心，W 侧全废 |
| `h` / `P3` / TAL / 分组身份 | ✅ 有效 | 取自 live EMA 的真实 forward |

---

## 已实施的修复（已在本机验证）

| # | 修复 | 验证 |
|---|---|---|
| 1 | `_refresh_head()`：每个观测点重读 W/b | 模拟 W 变化后 `head_W` 逐 epoch 差异 = 7.13e-02（修前 0.0）；logit 恒等式 ep0/ep1 = 7.4e-07 / 2.2e-06（修前 ep1 ≈ 0.6） |
| 2 | `params_untouched` 改为与**观测前**快照比 | ep0/ep1 均 True（修前恒 False） |
| 3 | §6 门幂等（`ep == self._last_seen_epoch` 直接 return） | 同 epoch 二次触发不再 raise |

另：`_analyze300.py` **尚未**在新 npz 上跑过；本目录此前混入的旧（作废的 raw probe）
分析产物已归档到 `_old_raw_probe/`。

---

## 因果限制

- 单条轨迹、无干预 ⇒ 只能给 temporal association。
- close_mosaic（0-based 290）前后只能报时序关联。
- 观测点分辨率 = 相邻两点区间，不得假装更精确。

## NEXT

**BEFORE ANY ANALYSIS**：修好的 probe 需重跑一次完整 300 epoch（约 12–13 h）。
必须重传：`diagnostic/full300_trajectory_probe/_probe300_ema.py`。
重跑后先复核 §16 第 8 项（`logit_identity` 的 `max_rel_resid` 应全程 ~1e-6）与第 10 项（收尾无异常），
两者 PASS 后才进入 §15 机制分类。
