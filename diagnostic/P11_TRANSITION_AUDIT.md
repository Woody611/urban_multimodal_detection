# P11 — Healthy → F0 Transition Audit

**日期** 2026-10-03 · **性质** ZERO GPU / ZERO training / ZERO forward / ZERO inference / 无 source·config 改动
**数据** `full300_trajectory_probe/from_snapshots/epoch_{0,10,…,290}.npz`（30 个 EMA 快照 × 1320 纵向 GT）+ `p3_disappearance_audit/attribution.json`（cohort）+ `p3_trajectory_probe/medium_cells_canvas.json`（空间坐标）
**脚本** `diagnostic/p11_transition_audit.py`、`p11_transition_audit_v2.py`　**产出** `diagnostic/p11_out/`

---

## Executive Verdict

P11 的问题是「最终 F0 的 GT 在什么时候、通过什么可观测量从 healthy 变成 F0」。答案是一个**否定性但结构清晰的结论**：
**不存在一个连贯的 "healthy → F0 transition" 群体，也不存在任何领先于 score 崩塌的可观测量。**
116 个最终 F0 中 23.3% 从一开始就 dead、33.6% 在 30 个快照里从未越过 dead 阈值、43.1% 属于"越阈"——
但**这 43.1% 里只有 20% 在死前曾达到过 sigmoid ≥ 0.5（死前峰值中位数只有 0.168）**，绝大多数只是在一个**本就不可见**的置信度区间里来回抖动、偶然穿过 `1e-3`。
更关键的是：在事件窗口 `[t−20, t]` 内做前后半段 lead-lag 检验后，**所有 feature 侧候选先兆（`cos(h,μ_GT)`、`cos(h,W_GT)`、`cos(h,W_mean)`）在**前半段都不下降**（+0.007 / +0.004 / −0.028），它们的下降全部集中在后半段——与 `z_GT` 崩塌同一区间；
而同一检验用在 **MATCHED 对照上，"feature 先兆"出现率反而更高**（cos_mu 15.0% vs F0 7.5%；cos_W 10.0% vs 2.5%）。
**⇒ 唯一稳定领先的是 `z_GT` / `margin` 本身，即结果变量，属循环论证。P11 未找到任何可用的时间先兆。**

---

## 1. Data / Artifact Integrity

| 项 | 值 |
|---|---|
| snapshots | **30**（ep0…ep290，步长 10；`save_period=10`，epoch210 缺失已确认不存在） |
| longitudinal GT | **1320**（281 张图；G86 86 / CONTROL 1070 / MISSED_NON86 164 —— **outcome-selected cohort，必须声明**） |
| cohort join | `stem#N` ↔ P3 `gt_id = N−1`；**0 重复 key**，1320/1320 命中 |
| F0 | **116**（P3 `failure == "F0_no_raw_candidate"`）；其中 **77** 在 ep290 也满足 `sigmoid<1e-3`，**39 从未越阈** |
| MATCHED | **984**（P3 `MATCHED@0.5`） |
| **定义冲突（须知）** | ep290 `sigmoid<1e-3` 的 GT 共 **462/1320**，而 P3-F0 只有 **116** ⇒ 两个定义**不等价**；且 P3 是在**最终模型（best.pt @ ep246）**上判的，快照只到 **ep290** —— 存在 checkpoint 口径差 |
| 完整性 | `logit ≡ W_cls·h + b_cls`，30 个 epoch 上 max 误差 **3.55e-15** ⇒ logit 列是真实分类器读出，可完整重建 |
| GPU | **0** |
| training / forward / inference | **0 / 0 / 0** |
| missing metrics | raw W（`"model": None`，见 P10）· per-step gradient · per-epoch pre-NMS candidate boxes · per-epoch official matching · optimizer state（存在但本轮未用） |
| 未做 | 任何 intervention；任何 cohort 重新发明；任何 forward |

---

## 2. t_dead Distribution

`t_dead = 首个 sigmoid(z_GT) < 1e-3 的观测 epoch`

| bucket | n | 占比 | median t_dead | p10 | p25 | p75 | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **INIT_DEAD** (ep0 已 dead) | **27** | 23.3% | 0 | — | — | — | — |
| **EARLY** (1–50) | 28 | 24.1% | 30 | 10 | 20 | 40 | 50 |
| **MID** (51–150) | 15 | 12.9% | 100 | 60 | 70 | 120 | 140 |
| **LATE** (151–290) | 7 | 6.0% | 180 | 160 | 170 | 220 | 230 |
| **NEVER_DEAD**（30 快照内未越阈） | **39** | 33.6% | — | — | — | — | — |
| **transition 合计** (EARLY+MID+LATE) | **50** | **43.1%** | **45** | 10 | 20 | 118 | 171 |
| 全部 F0（含 INIT_DEAD 记 t=0） | 116 | 100% | 20 | — | — | — | — |

**★ 但"越阈"≠"曾经健康"：**

| 死前峰值 sigmoid | ≥0.5 | ≥0.25 | ≥0.1 | ≥0.05 | ≥0.01 |
|---|---:|---:|---:|---:|---:|
| transition F0 (n=50) | **10 (20.0%)** | 21 (42.0%) | 32 (64.0%) | 36 (72.0%) | 40 (80.0%) |

死前峰值 sigmoid 中位数 = **0.1678**（p25 0.0236 / p75 0.4248 / max 0.8631）。
MATCHED 参照：全程峰值中位 **0.8098**，76.2% 曾 ≥0.5。

**⇒ 80% 的"transition F0"死前从未超过 0.5 置信度；中位峰值 0.168。它们不是"从可检出变漏检"，而是在低置信区穿阈值。**

---

## 3. Transition Timing

事件窗口 `[t−20, t]`（快照间隔 10 epoch ⇒ 恰好 3 点 t−20 / t−10 / t），**within-GT Δ**：

| signal | F0-transition 中位 [IQR] | MATCHED 对照（pseudo-event） | 差 |
|---|---:|---:|---:|
| `z_GT` | **−4.878** [−6.50, −3.84] | +0.189 [−0.39, +1.05] | −5.067 |
| `margin` | −2.236 [−4.21, −0.29] | +0.765 [−0.34, +2.11] | −3.001 |
| `cos(h, μ_GT)` | **−0.1385** [−0.191, −0.051] | +0.0093 [−0.007, +0.036] | **−0.1478** |
| `cos(h, W_GT)` | **−0.1606** [−0.237, −0.104] | +0.0057 [−0.019, +0.067] | **−0.1663** |
| `cos(h, W_mean)` | −0.1749 [−0.281, −0.116] | −0.0179 [−0.072, +0.022] | −0.1571 |
| `h_norm` | −1.586 [−8.73, +0.88] | +0.443 [−6.50, +2.79] | −2.029 |
| `best_IoU` (= TAL target) | −0.0172 [−0.136, +0.016] | +0.0131 [−0.016, +0.048] | −0.0303 |

**⚠ 这张表看起来像"feature 先兆存在"—— 它是 P11 最大的陷阱。§7c 会否定它。**

---

## 4. Pre-Death Signals（含 MATCHED 对照）

### 4a. 无条件 onset 率（2 连续观测恶化，全轨迹）

| signal | F0-tr | MATCHED | ratio |
|---|---:|---:|---:|
| `z_GT` | 100.0% | 96.3% | 1.04 |
| `margin` | 100.0% | 95.2% | 1.05 |
| `best_IoU` | 100.0% | 99.5% | 1.01 |
| `cos(h,μ_GT)` | 100.0% | 99.9% | 1.00 |
| `cos(h,W_GT)` | 100.0% | 99.9% | 1.00 |
| `cos(h,W_mean)` | 100.0% | 100.0% | 1.00 |
| `h_norm` | 98.0% | 98.4% | 1.00 |

**⇒ 该规则在 30×10-epoch 网格上**饱和**：几乎每个 GT 在某个地方都有一段 ≥20 epoch 的连降。此规则**无效**，已弃用。**

### 4b. 匹配 pseudo-event 下的"死前 onset"率（控制 = 同类 + 最近 log-area）

| signal | F0 死前 onset | MATCHED 对照 | 差 (pp) | F0 相对 t_dead 中位 |
|---|---:|---:|---:|---:|
| `best_IoU` | 68.0% | 48.0% | **+20.0** | −10 |
| `z_GT` | 66.0% | 56.0% | +10.0 | −20 |
| `margin` | 56.0% | 46.0% | +10.0 | −15 |
| `cos(h,W_GT)` | 56.0% | 48.0% | +8.0 | −20 |
| `cos(h,W_mean)` | 72.0% | 74.0% | **−2.0** | −20 |
| `h_norm` | 86.0% | 90.0% | **−4.0** | −40 |
| `cos(h,μ_GT)` | 52.0% | 56.0% | **−4.0** | −10 |

**⇒ 只有 score 侧（含 `best_IoU`）有微弱分离；所有 feature 侧分离为负。且最大分离 (+20pp) 与结果变量自身 (+10pp) 同量级。**

### 4c. ★ 决定性：事件窗口内的前后半段 lead-lag

窗口 `[t−20, t]` 切两半：`d1 = 前半段变化`，`d2 = 后半段变化`。"先兆"的定义 = **`d1 < 0` 且 `z_GT` 的 `d1 ≥ 0`**（feature 先动、score 后动）。

| signal | F0 `d1` 中位 | F0 `d2` 中位 | F0 先兆率 | **MATCHED 先兆率** |
|---|---:|---:|---:|---:|
| `cos(h,μ_GT)` | **+0.0073** | −0.1306 | **7.5%** | **15.0%** ← 对照更高 |
| `cos(h,W_GT)` | **+0.0041** | −0.1686 | **2.5%** | **10.0%** ← 对照更高 |
| `cos(h,W_mean)` | −0.0278 | −0.1533 | 12.5% | **22.5%** ← 对照更高 |
| `best_IoU` | +0.0042 | −0.0333 | 20.0% | 20.0% ← 相同 |
| `z_GT` | −0.2389 | −4.2436 | 55.0% 已在降 | 0.0%（SE） |

**⇒ feature 侧信号在窗口前半段【完全不降】，其下降 100% 集中在与 `z_GT` 崩塌同一的后半段。**
**§3 表里的 "feature 窗口效应" 是【结果】不是【前兆】。**
40-epoch 窗口稳健性复核同向：F0 vs MATCHED 的前半段 `cos(μ)` 变化 −0.0124 vs −0.0065（仅 1.9×），而 `z_GT` 是 −0.398 vs +0.155。

---

## 5. F0 vs MATCHED Longitudinal Trajectories（median，F0-transition / MATCHED）

| ep | `z_GT` | `cos(h,μ_GT)` | `best_IoU` |
|---:|---:|---:|---:|
| 0 | −3.65 / −2.10 | +0.86 / +0.89 | +0.84 / +0.84 |
| 10 | −3.87 / −0.89 | +0.89 / +0.96 | +0.82 / +0.84 |
| 20 | −4.12 / −0.50 | +0.88 / +0.96 | +0.80 / +0.83 |
| 50 | −4.34 / −0.02 | +0.81 / +0.95 | +0.78 / +0.83 |
| 100 | −6.18 / +0.20 | +0.82 / +0.96 | +0.74 / +0.82 |
| 150 | −8.51 / +0.52 | +0.82 / +0.95 | +0.74 / +0.81 |
| 200 | −10.50 / +0.72 | +0.81 / +0.95 | +0.70 / +0.81 |
| 290 | −12.71 / +0.24 | +0.76 / +0.94 | +0.72 / +0.81 |

**⇒ `z_GT` 的分歧是【持续、早期即存在、单调扩大】的，不是"某时刻转折"。**
`cos(h,μ_GT)` 只有到 ep290 才勉强分开（0.76 vs 0.94），`best_IoU` 全程只差 0.1 左右。

---

## 6. Within-GT Normalized Changes

`Δ = X(t) − X(ep0)` 的中位数差（F0-transition − MATCHED）：

| ep | `z_GT` | `margin` | `cos(h,W_GT)` | `cos(h,μ_GT)` | `h_norm` | `best_IoU` |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | **−1.782** | −2.354 | −0.056 | −0.018 | −0.176 | −0.009 |
| 20 | −1.768 | −2.424 | −0.047 | −0.034 | −0.747 | −0.037 |
| 50 | −3.190 | −3.297 | −0.094 | −0.047 | −1.008 | −0.046 |
| 100 | −3.878 | −3.244 | −0.148 | −0.055 | −1.515 | −0.078 |
| 150 | −7.290 | −2.908 | −0.217 | −0.059 | −1.025 | −0.078 |
| 200 | −7.107 | −4.468 | −0.229 | −0.051 | −0.963 | −0.089 |
| 290 | −9.728 | −5.582 | −0.266 | −0.062 | −1.265 | −0.082 |

**⇒ 到 ep10，F0-transition 相对 MATCHED 已经低了 1.78 nats。分歧在最早可观测点就存在并可预测最终结局。**
**这不构成时间机制，只说明"最终 F0"在 ep10 就可被认出。**

---

## 7. Matched-Control Analysis

- 匹配：**same class + 最近 log(area)**（1:1 nearest neighbour），n=50 对。
- **限制：无法匹配 spatial position**（`cells` artifact 只有 canvas 中心，无完整 bbox），且匹配集本身是 outcome-selected（CONTROL = 最终检出）。
- 结果：见 §4b（最大 +20pp）与 §4c（**feature 侧对照先兆率更高**）。
- **MATCHED CONTROL: LIMITATION** —— 对照来自同一 outcome-selected 池，非随机抽样。

---

## 8. Size / Class / Spatial Effects

**t_dead by size（transition F0, n=50）**

| size | n | median t_dead | min | max |
|---|---:|---:|---:|---:|
| small | 14 | 75 | 10 | 230 |
| medium | 30 | 35 | 10 | 220 |
| **large** | **6** | **70** | 10 | 120 |

**⇒ 与 size 无单调关系（large 中位甚至晚于 medium）；n=6 的 large 无统计意义。**

**border / non-border（canvas 1312×736，border = 中心距边 <73.6px）**

| border | n | median t_dead |
|---|---:|---:|
| False | 30 | 55 |
| True | 20 | 35 |

⇒ 边界目标略早，但 n 小、无对照修正。

**class（全部 1320）**

| class | n | F0 | F0 rate | (F0 中位数 t_dead) |
|---|---:|---:|---:|---:|
| ball | 8 | 3 | 0.375 | 5 |
| seat | 20 | 5 | 0.250 | 5 |
| garbage_can | 17 | 4 | 0.235 | 25 |
| sign | 71 | 16 | 0.225 | 0 |
| bicycle | 42 | 9 | 0.214 | 30 |
| light | 84 | 16 | 0.190 | 0 |
| uav | 6 | 1 | 0.167 | 0 |
| person | 470 | 38 | 0.081 | 60 |
| animal | 318 | 18 | 0.057 | 45 |
| car | 147 | 6 | 0.041 | 40 |
| boat | 11 | 0 | 0.000 | — |

⇒ 类别跨度 0–37.5%，但 n 从 6 到 470；与 P8.5 一致（**与频率无单调关系**）。

---

## 9. F0 Transition Types

**崩塌形状（transition F0）**：gradual **28** / abrupt **12** / too-early 10 ⇒ **70% 渐变 / 30% 突变**。

**先兆类型（窗口内 half-split 判定）**

| type | 判定 | n |
|---|---|---:|
| **A — feature-led** | `cos(μ)` 或 `cos(W)` 前半段先降 | **4（且控制组更高）** |
| **B — score-led** | `z_GT`/`margin` 先降 | **21** |
| IoU-led | `best_IoU` 先降 | 17 |
| A+IoU both prior | | 8 |

**⇒ 但 §4c 表明 A 型的"先兆"在对照中更常见 ⇒ A 型不成立，B/IoU 的排序在 ~100% 基率下接近噪声。**
**正确的描述是：④ 型分布接近均匀 ⇒ 不支持任何单一机制 ⇒ 属于 C(mixed)。**

---

## 10. Temporal Causality Assessment

| candidate | temporal precedence | consistency | matched-control separation | status |
|---|---|---|---|---|
| `cos(h, μ_GT)`（feature） | ❌ `d1 = +0.0073`（前半段不降） | 7.5% | ❌ 对照 **15.0%**（更高） | **NOT SUPPORTED** |
| `cos(h, W_GT)` | ❌ `d1 = +0.0041` | 2.5% | ❌ 对照 **10.0%** | **NOT SUPPORTED** |
| `cos(h, W_mean)` | ❌ `d1 = −0.028`，与对照 −0.027 同幅 | 12.5% | ❌ 对照 **22.5%** | **NOT SUPPORTED** |
| `h_norm` | ❌ 对照先兆率 90% > F0 86% | — | ❌ 无分离 | **NOT SUPPORTED** |
| `best_IoU` / TAL target | ⚠ `d1 = +0.004`（不领先） | 20% | ❌ 对照 = 20%（相同） | **NOT SUPPORTED** |
| `margin` | ⚠ 与 `z_GT` 同时 | 55% | 0%（SE，循环） | **CIRCULAR** |
| **`z_GT`（结果变量本身）** | ✅ 55% 在前半段已在降 | 55% | — | **CIRCULAR —— 它就是结局** |

**没有一条满足 `within-GT temporal precedence + matched control + effect consistency`。**

---

## 11. GPU Gate

| Gate | 要求 | 结果 |
|---|---|---|
| **1** | 明确 temporal precedence（X 退化 → t_dead） | ❌ **FAIL** —— 唯一领先的是 `z_GT` 自身；所有 feature 侧 `d1 ≈ 0` 或为正 |
| **2** | effect 在 F0 内部一致（≥60% transition F0） | ❌ **FAIL** —— 最好的非循环候选 `best_IoU` 20%，`cos(μ)` 7.5%；`z_GT` 55% 但循环 |
| **3** | MATCHED 对照中不存在同等退化 | ❌ **FAIL（反向）** —— `cos(μ)` 15.0% vs 7.5%，`cos(W)` 10.0% vs 2.5%，`cos(W_mean)` 22.5% vs 12.5%，`h_norm` 90% vs 86% |
| **4** | X 与已关闭机制不同 | N/A（无候选） |
| **5** | 单一变量最小干预 | ❌ 无法提出 |
| **6** | 可写出证伪条件 | ❌ 无候选可证伪 |

```text
GPU = NO-GO
```

---

## FINAL VERDICT

```text
D — HETEROGENEOUS / MULTI-MECHANISM → GPU NO-GO
```

**理由**：最终 F0 不是一个人群——
23.3% 从未活过（INIT_DEAD）、33.6% 在 30 个快照里从未越阈、43.1% 越阈但其中 **80% 死前从未超过 0.5 置信度**（死前峰值中位 0.168）；崩塌形状渐变/突变 70/30；先兆类型分布接近均匀。
**在这个本身就不连贯的群体里，没有任何可观测量领先于 score 崩塌**，而唯一的领先者就是 score 自己（循环）。

**⇒ 明确建议关闭路线：**

```text
CLOSE:  "F0 score-vacuum → generic classification-head optimization"
```

（含 feature-loss / BCE 重加权 / cosine·norm 归一化 / logit 重标定 等其所有下游变体 —— P6 已从单调性上排除，P11 从时间上排除。）

---

## Decision Memo（≤10 行）

```text
What actually happens before F0?
  Nothing observable leads. z_GT/margin decline, and everything else
  (feature cosine, IoU, target) declines in the SAME 10-epoch window or later.
What does not happen?
  No feature-manifold precursor; no IoU precursor; no target precursor.
  Matched controls show the "precursors" at equal or higher rates.
  There is also no coherent "healthy -> F0" population to explain:
  80% of threshold-crossers never exceeded 0.5 confidence before dying.
What is still unknown?
  Whether the 20% of transition-F0 that WERE confidently detected (n=10)
  share a mechanism — too few to test at 10-epoch resolution.
What experiment, if any, is justified?
  None. GPU NO-GO. Close the F0 score-vacuum / classification-head route.
```

---

## 附：本轮对既有路线的定位（避免重复）

| 已有结论 | P11 的增量 |
|---|---|
| P5/P5.1 score vacuum、W_t×h_t 分解 | 不变；P11 用同一 artifact |
| P5.2 / P10 W_mean ↑、方向冻结、时间反向 | 不变；P11 不重复 head 级分析 |
| P6 norm/cosine/单调重标定被数学排除 | **P11 从时间维度再加一层排除** |
| P7 F0 heterogeneous、mismatch 非普遍 | **P11 量化了异质性来源（80% 死前从未可信）** |
| P8.5 TAL target / BCE 正梯度被排除 | 不变；P11 独立确认 `best_IoU` 不是先兆 |
| P9 background budget 未立 | 不变 |
| **P11 唯一新增价值** | **把"最终 F0 静态现象"转成"单 GT 时间事件"后，先兆检测失败且对照反向 ⇒ 该群体本身不可解释为单一 transition** |
