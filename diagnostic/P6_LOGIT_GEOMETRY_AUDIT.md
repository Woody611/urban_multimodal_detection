# P6-A / P6-B Zero-GPU Audit — Logit Distribution + Classifier Geometry

**日期** 2026-10-03 · **性质** ZERO-GPU / ZERO-TRAINING / ZERO-FORWARD / READ-ONLY
数据：30 个 EMA snapshot（`from_snapshots/`）+ P3 attribution + 训练集 label 统计。未补任何数据。

## 1. Executive verdict

```text
P6-A Logit Distribution : OPEN-WEAK
P6-B W×h×cosθ           : OPEN-WEAK
GPU candidate           : NO
```

**两者都不是"论文上有效"，而是"在 D′ 上被证明不可干预"。** 见 §4 / §13-gate-8。

---

## 2. P6-A evidence table（Logit Distribution）

口径：GT-cell，GT 类，1320 个 medium GT。

| Observable | Early (ep0) | Late (ep290) | Δ | Evidence for anomaly |
|---|---:|---:|---:|---|
| GT logit mean | −3.014 | −4.827 | **−1.813** | 均值下移，但幅度有限 |
| GT logit median | −2.282 | −0.907 | **+1.375** | **中位数反而上升** |
| **GT logit variance** | **9.511** | **62.139** | **×6.53** | ★ **最强的分布异常** |
| p10 | −7.84 | **−17.24** | −9.40 | 下尾拉长 |
| p90 | +0.35 | **+2.22** | +1.87 | **上端被钉住** |
| p90 − p10 | 8.19 | **19.46** | ×2.38 | 展宽 |
| BG logit mean | — | — | — | **NOT MEASURABLE FROM EXISTING ARTIFACTS** |
| BG logit variance | — | — | — | 同上 |
| FG−BG separation | — | — | — | 同上（无背景/全图 cell 的 h 落盘） |

**⇒ 不是整体平移，是【向下不对称展开】**：p90 只从 +0.35 涨到 +2.22（+1.87），p10 从 −7.84 掉到 −17.24（−9.40）。
上端被钉住、下尾拉长 —— 与 P5 的"极化"完全一致，但这里给出了**方差量级（×6.5）**。

**背景/前景分离：不可测**（artifact 里只有 GT-cell 的 `vec_h`）。**不为此补跑 forward。**

---

## 3. P6-B evidence table（W × h × cosθ）

`z_c = ‖W_c‖·‖h‖·cosθ_c + b_c` —— **恒等式精确成立**（max 残差 `3.6e-15`）。

| Quantity | Early (ep0) | Late (ep290) | Δ | Relation to dead-rate |
|---|---:|---:|---:|---|
| ‖W_c‖（GT 类，中位） | 0.5890 | 0.7520 | **+27.7%** | 弱（见下：dead/alive 几乎相同） |
| **‖W_mean‖** | 0.1696 | **0.4889** | **+188%** | 与 live dead 14.8%→35.0% **同向** |
| **‖W_c − W_mean‖（中位）** | **0.5605** | **0.5395** | **−3.7%（持平）** | ★ 判别部分**没有增长** |
| ‖h‖（中位） | 64.98 | 50.90 | −21.7% | 弱 |
| cosθ（GT 类，中位） | +0.1901 | +0.2267 | +19.3%（先升后降，峰 +0.344@ep20） | **中位改善** |
| b（中位） | −9.703 | −9.758 | −0.055（**冻结**） | 无 |
| centered W / ‖W_c‖ | 0.9513 | **0.7423** | −22.0% | common-mode 占比上升 |

### ★ dead vs alive（同一个 epoch，ep290）

| 组 | n | **cosθ** | **‖W_c‖** | ‖h‖ | b | z |
|---|---:|---:|---:|---:|---:|---:|
| dead | 462 | **−0.1275** | 0.7520 | 47.20 | −9.719 | **−14.87** |
| alive | 858 | **+0.2633** | 0.7646 | 51.88 | −9.758 | **+1.62** |

**⇒ 判别量是【角度】**：‖W_c‖ 差 1.7%，‖h‖ 差 9.0%，b 几乎相同，
而 cosθ 差 **0.391**，乘以 ‖W‖‖h‖≈36 ⇒ **Δz ≈ 14 nats，足以完全解释 16.5 nats 的差距**。

### ★ 负 cosθ 是 W 旋转造成的（决定性）

同一个 `h_290`，只换 head：

| 组合 | dead 中位 cosθ | alive 中位 cosθ |
|---|---:|---:|
| `W_0 × h_290` | **+0.1956** | +0.4342 |
| **`W_290 × h_290`（live）** | **−0.1275** | +0.2633 |
| `W_0 × h_0` | +0.1457 | +0.2051 |
| `W_290 × h_0` | −0.0253 | +0.0968 |

**h 漂移单独作用反而改善** dead 组（+0.146 → +0.196）；是 **W 的旋转**把它翻负的（+0.196 → −0.128）。

---

## 4. Mechanism decomposition

```text
absolute logit decline (dead 组的 z: −2.28 → −14.87):
    weight norm contribution  = ~0    （‖W_c‖ 在 dead/alive 间差 1.7%）
    feature norm contribution = 小    （‖h‖ 差 9.0%，且方向有利）
    angular contribution      = 主导  （cosθ 差 0.391 × ‖W‖‖h‖ ≈ 14 nats）
    bias contribution         = ~0    （b 冻结，差 0.04）
```

**但注意同时存在一个真实的结构异常：**

```text
‖W_mean‖        : 0.1696 → 0.4889   (+188%)
‖W_c − W_mean‖  : 0.5605 → 0.5395   (−3.7%, 持平)
centered/‖W_c‖  : 0.9513 → 0.7423   (−22%)
```

⇒ **classifier 把增长几乎全部投在 common-mode 方向，判别部分没有增长。**
**这恰好是 brief §10 提出的假设，现由 D′-specific 数据确认。**

---

## 5. Relation to P5.2

| # | 问题 | 回答 |
|---|---|---|
| 1 | P5.2 的 `W_mean` growth 是否只是正常 classifier learning？ | **不是**。正常 learning 应增长判别部分；这里 `‖W_c−W_mean‖` **持平（−3.7%）**，增长全在 common-mode。**这是本轮相对 P5.2 的新增** |
| 2 | top-singular concentration 是否与 dead-rate 有关？ | **时间上同向**，但**不是判别的** —— dead/alive 之间 ‖W_c‖ 几乎相同，判别靠 cosθ |
| 3 | common-mode growth 是否影响 absolute confidence 而不影响 discrimination？ | **是**。centered GT logit **6.28 → 12.85（改善）**，而绝对 logit/方差恶化 |
| 4 | 当前 evidence 是否足以支持 classifier normalization intervention？ | **否 —— 见 §6 的单调性论证** |

---

## 6. Literature mechanism mapping（§12）

### L1 — Logit Normalization（LogitNorm 类）
**不对应。** LogitNorm 处理的是"logit 幅度无界增长导致 over-confidence"；D′ 的现象是**方差向下不对称展开**，
而上端被钉住（p90≈+2.2）。且它作用于**训练期的 logit 尺度**，与 dead 组的**负 cosθ**无关。

### L2 — Weight Normalization / Cosine Classifier
**表面像，实质不对应 —— 且有单调性反证。**
这类方法通过**范数归一化**改变 logit;但 D′ 上：

```
‖W_c‖ : dead∧F0 = 0.7520   dead∧MATCHED = 0.7520   alive = 0.7646
```

**范数在"会失败"与"仍被检出"两组之间完全相同** ⇒ **范数不携带判别信息** ⇒ 归一化范数**不改变任何排序**。

更强的一般性结论（**可证明**）：
dead 组的 `cosθ = −0.19 < 0`。**任何对 `z` 的单调重标定**（范数归一化、cosine classifier 的 `z=τ·cosθ+b`、温度缩放）
都保号不保序 —— 负 cosθ 的样本**不可能**在正 cosθ 样本仍高于阈值的同时被抬到阈值之上。
**要救它们只能改变 cosθ 的符号，即改变 W 或 h 的【方向】** —— 那正是 P5.2 判为
`mechanism NOT IDENTIFIED` 的那个未识别上游机制。

### L3 — Margin-based classifier
**无 D′ 证据。** angular separation **确实**是 dead/alive 的判别量，
但它是**结果**而非瓶颈：dead∧F0 与 dead∧MATCHED 的 cosθ 只差 **0.081**（CI [−0.111, −0.055]），
alive +0.264 —— 是一个**连续谱**，不是"margin 不足"的二元结构。
且 rare-class 起因已被官方 per-class AP 否证（`corr(频率, AP) = −0.204`）。

> 本轮严格遵守：**literature relevance ≠ D′ mechanism evidence**。

---

## 7. ★ Gate #8 —— 决定性的零 GPU 检验

brief §13 的第 8 条要求：**能说明为什么它可能改变 official AP50-95，而不仅是 local confidence distribution。**

**（a）GT-cell dead ≠ 检不出来。** ep290 的 462 个 dead：

| 真实结局 | n | % |
|---|---:|---:|
| **MATCHED@0.5（其实检出来了）** | **256** | **55.4%** |
| F0 无候选 | 148 | 32.0% |
| F1 NMS 消失 | 43 | 9.3% |

**（b）但 dead 是 F0 的必要条件**：medium F0 的 **95%（148/155）**都是 dead。

**（c）关键：dead 内部能否被几何区分？**

| 组 | n | **cosθ** | **‖W_c‖** | ‖h‖ | z |
|---|---:|---:|---:|---:|---:|
| dead ∧ F0 | 148 | **−0.1933** | **0.7520** | 41.84 | −16.03 |
| dead ∧ MATCHED | 256 | −0.1125 | **0.7520** | 54.68 | −14.81 |
| alive ∧ MATCHED | 808 | +0.2640 | 0.7646 | 52.32 | +1.71 |

`Δcosθ = −0.0807`，bootstrap 95% CI `[−0.1115, −0.0552]`（不含 0）—— 几何**能**部分区分。

**但（d）‖W_c‖ 在两组间【完全相同】**（0.7520），判别完全靠角度 ⇒ 由 §6 的单调性论证，
**范数/温度类的重标定不可能修好它们**。

```text
⇒ Gate #8: FAIL
   不是因为"缺乏动机"，而是因为唯一与 D′ 几何对应的干预族
   （weight norm / cosine / temperature）在【单调性上】不可能改变 dead 组的排序。
```

---

## 8. GPU Gate（§13 逐条）

| # | 条件 | 结果 |
|---|---|---|
| 1 | 明确的 classifier-geometry / logit-distribution anomaly | ✅（方差 ×6.5；`‖W_mean‖` +188% vs `‖W_c−W_mean‖` 持平） |
| 2 | anomaly 与 dead-score phenotype 稳定关联 | ✅（同 epoch dead/alive 是纯 cosθ 分割） |
| 3 | 不是已有 P5/P5.1/P5.2 已解释的现象 | ⚠ 部分（common-mode 增长 P5.2 已述；**方差 ×6.5 与 centered-W 持平是本轮新增**） |
| 4 | 不是单纯 long-tail premise | ✅（rare class AP 更高） |
| 5 | 存在明确 intervention | ⚠ 形式上有（weight norm / cosine），但见 #8 |
| 6 | 有明确 falsification criterion | ✅ |
| 7 | 不只是改 inference confidence threshold | ✅（训练侧改动） |
| **8** | **能说明为何可能改变 official AP50-95** | ❌ **FAIL（§7）** |

```text
GPU NO-GO
```

---

## 9. 最终结论（§14 格式）

```text
P6-A Logit Distribution:
VERDICT = OPEN-WEAK
  （方差 ×6.5 是真实、新的 D′-specific 异常；
    但 BG/FG 分离 NOT MEASURABLE，且无 metric 链接）

P6-B W×h×cosθ:
VERDICT = OPEN-WEAK
  （brief §10 的假设被 D′ 数据【确认】：common-mode +188% / 判别部分持平；
    dead/alive 纯 cosθ 分割、且由 W 旋转造成。
    但对应的干预族在单调性上不可能改变结果 ⇒ Gate #8 FAIL）

GPU candidate: NO

If NO:
  precise closure reason =
    (1) dead 组的 cosθ < 0，任何对 z 的单调重标定（范数归一化 / cosine classifier / 温度）
        都保号不保序 —— 负 cosθ 样本无法被抬过阈值而不破坏正 cosθ 样本的排序；
    (2) ‖W_c‖ 在 dead∧F0 与 dead∧MATCHED 间【完全相同】(0.7520) ⇒ 范数不携带判别信息；
    (3) 要救它们必须改变 cosθ 的【符号】，即改变 W 或 h 的方向 ——
        那正是 P5.2 判为 mechanism NOT IDENTIFIED 的未识别上游机制，本审计未新增该环节的证据。
```

**本轮没有把 P5.2 的 classifier geometry 异常转化为可证伪干预；相反，它证明了最自然的那一族干预在数学上无效。**

---

## 附：本轮新做的核查

1. `z = ‖W_c‖‖h‖cosθ + b` 恒等式在 16 个快照上精确成立（残差 ≤3.6e-15）。
2. **方差 ×6.53**（9.511 → 62.139）、p90 被钉住而 p10 下坠 —— P5 未量化的分布异常。
3. **`‖W_c − W_mean‖` 持平（−3.7%）而 `‖W_mean‖` +188%** —— 确认 brief §10 假设。
4. **同一 `h_290` 下 `W_0` 给 +0.196、`W_290` 给 −0.128** —— 负 cosθ 归因于 W 旋转。
5. **dead∧F0 vs dead∧MATCHED 的 `‖W_c‖` 完全相同** —— Gate #8 的闭合依据。
6. 分类头源码确认**无任何 norm / cosine / 温度**，且全库**从未试过** weight normalization。
