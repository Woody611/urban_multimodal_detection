# P5.2 — W_t Classification-Head Dynamics / Upstream Mechanism Audit

**日期** 2026-10-03 · **性质** READ-ONLY / ZERO-GPU / ZERO-TRAINING / ZERO-FORWARD
脚本 `diagnostic/p5_2_wt_dynamics_audit.py` · 日志 `diagnostic/P5_2_run.log` · CSV `diagnostic/full300_trajectory_probe/p5_2_out/`

## Verdict

```text
MECHANISM NOT IDENTIFIED
GPU: NO-GO
```

**§0 Artifact Gate：PASS**（30 snapshots、W hash 30/30 唯一、1320 keys 逐位一致、P5.1 复现 `max|R+H−T|=3.55e-15`、
`median R=+3.753 / H=−4.002`）。

**已确立**：`W_t` 发生了什么（几何上）**已经查清**。
**未确立**：**训练中是什么让它这样变** —— 因此按 Gate C 判定 NO-GO。

---

## 1. W_t geometry

**[FACT] 12 个类 row 全部变大、全部旋转（没有 row collapse、没有少数类例外）：**

| cls | ‖W₀‖ | ‖W₂₉₀‖ | Δ‖·‖ | cos(W₂₉₀, W₀) |
|---|---:|---:|---:|---:|
| person | 0.6044 | 0.8623 | **+0.2580** | 0.764 |
| animal | 0.5646 | 0.7520 | +0.1874 | 0.754 |
| car | 0.5815 | 0.7646 | +0.1831 | 0.743 |
| light | 0.5856 | 0.7438 | +0.1582 | 0.783 |
| tricycle | 0.5523 | 0.6116 | **+0.0593** | 0.813 |

均值：`E‖W_c‖` 0.5813 → **0.7289（+25.4%）**。

**[FACT] 低秩/共同方向塌缩（§5 SVD）：**

| ep | spectral | frobenius | effrank | **top-1 能量** | top-3 能量 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.6995 | 2.0147 | 11.914 | **12.06%** | 33.15% |
| 50 | 1.3851 | 2.3430 | 11.378 | 34.95% | 50.09% |
| 150 | 1.6499 | 2.5114 | 11.073 | 43.16% | 56.78% |
| 290 | 1.7030 | 2.5326 | 10.985 | **45.22%** | 58.60% |

谱范数 +143%，Frobenius 仅 +26%，**top-1 能量从 12% 涨到 45%**，effrank 基本稳定。

**[FACT] 那个 top-1 方向就是共同模**：`cos(W_mean, top-1 singular vector) = 0.9993`（末点）。
`‖W_mean‖` 0.1696 → **0.4889（+188%）**。

## 2. Common-mode

**[FACT] score vacuum 的电平下降几乎 100% 是 head 造成的**（12 类均值 logit 的 Shapley，末点）：

```
Rc(表征) = +0.154     Hc(head) = −6.617     Tc(总) = −5.948      恒等式 err = 3.55e-15
```

**[FACT] `b_mean` 几乎冻结**：−9.7331 → −9.7949（Δ = 0.062）。
且 `max|b_t − b_0| = 0.078125` —— 恰为 fp16 在 |b|≈9.8 处的 **10 ULP**（快照 EMA 以 `.half()` 存储）
⇒ **bias 在整个训练中位移 ≤0.078，实质冻结。**

**[FACT] 每个类 row 相对数据平均表征 `h_mean` 的对齐符号全部翻转：**

| cls | cos(W₀, h̄₀) | cos(W₂₉₀, h̄₂₉₀) | W·h̄ : W₀ → W₂₉₀ |
|---|---:|---:|---:|
| person | **+0.144** | **−0.106** | +3.99 → −3.27 |
| car | +0.056 | −0.204 | +1.50 → −5.58 |
| tricycle | −0.039 | −0.374 | −0.99 → −8.18 |

**[FACT] 这个翻转不是表征造成的**：`cos(h_mean(t), h_mean(0))` 全程 **0.85**（ep30 后稳定），
而 `cos(W_c(t), W_c(0))` = 0.74–0.85 ⇒ **h 稳定、W 旋转**。

**[FACT] 判别力反而在提升**：类居中 GT logit 6.28 → **+12.85**（末点居中<0 仅 2.3%）。

⇒ **score-level degradation 与 class-separation degradation 是两回事**：
崩溃的只是**共同模电平**（`W·h̄ + b̄`），类间判别全程改善。

## 3. Temporal alignment

| ep | ‖ΔW‖ | ‖ΔW_mean‖ | GT logit | GT居中 | 12类均值 | live dead | W₀ dead | W_F dead |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | −2.28 | 6.28 | −8.67 | 14.8% | 14.8% | 41.1% |
| 10 | 0.699 | 0.191 | −1.11 | 8.52 | −9.73 | 11.8% | 4.8% | 14.7% |
| 50 | 1.167 | 0.313 | −0.21 | 10.11 | −10.72 | 17.3% | 4.6% | 20.0% |
| 100 | 1.383 | 0.368 | −0.11 | 11.23 | −11.87 | 22.4% | 5.5% | 23.9% |
| 200 | 1.531 | 0.406 | +0.19 | 12.74 | −13.51 | 28.1% | 6.8% | 28.4% |
| 290 | 1.549 | 0.411 | −0.91 | 12.85 | **−14.46** | **35.0%** | **8.9%** | 34.9% |

**时间关系（只报关联，不报因果）：**
- `‖ΔW‖` 与 `‖ΔW_mean‖` 在 **ep0→10 完成约 45%/46% 的总位移**，其后减速
- 12 类均值 **单调下降**贯穿全程；live dead **单调上升**贯穿全程 —— 二者 **temporally aligned**
- GT 居中 logit **同向上升** —— 与 dead rate **反向**，进一步印证"掉的只是电平"
- `W₀` 冻结读数的 dead 率 **反向下降到 8.9%** —— 与 live 35.0% 形成对照

⇒ **W drift 与 score vacuum temporally aligned；但本分析不能区分"先 W 后 score"**（观测步长 10 epoch，
且两者在 ep0→10 同时起步）。

## 4. Specific upstream mechanism

**没有找到满足 §8 标准的机制。已排除与已证伪的候选如下：**

| 候选 | 判定 | 依据 |
|---|---|---|
| **M1 class-frequency / negative-class accumulation** | **部分（不足以解释 score）** | `corr(训练集类频率, Δ‖W_c‖) = **+0.900**`（强）；但 **`corr(频率, Δ(W_c·h̄)) = −0.021`（无）** ⇒ 频率解释了 row **范数**增长，**完全不解释** logit 下降。**不能作为 score vacuum 的上游** |
| **M2 common-mode gradient drift** | **现象重述** | "共同模在漂"就是 §2 的观察本身，未指出驱动量 |
| **M3 late-stage classifier weight competition** | **不支持** | 无晚期转折：谱量、`‖ΔW‖`、12类均值全为"早期快、后平台"的单调形 |
| **M4 EMA / optimizer dynamics** | **无法判定** | artifact 只有 EMA 快照，无 raw 权重 / 无 optimizer state / 无梯度 ⇒ UNKNOWN |
| **M5 loss weighting / target construction** | **无法判定** | 同上，无训练侧中间量 |
| **M6 bias–weight decoupling（新候选）** | **证据支持"冻结"，但成因不可验** | bias 位移 ≤0.078（10 ULP）而 `W·h̄` 翻了 −8 nats；bias 未做补偿。**可测**。但"为什么 bias 不补偿"（如 sigmoid 饱和 ⇒ 梯度消失）**需要梯度，artifact 里没有** ⇒ 停在候选层 |

**关键负面结果**：所有"单侧"解释都被否 ——
- 表征侧：`h_mean` 方向稳定（cos 0.85）、类居中 logit 持续改善
- 频率侧：对 logit 变化相关性 ≈ 0
- 判别侧：全程改善

⇒ 剩下的唯一结构性事实是 **"W 整体相对数据旋转 + bias 冻结"**，但**驱动这两者的训练动力学不在现有 artifact 中**。

## 5. Falsifiable intervention

```text
No intervention justified yet.
```

Gate 逐条：

| Gate | 满足？ | 说明 |
|---|---|---|
| A 明确可重复的 W_t 结构变化 | ✅ | top-1 能量 12%→45%；12/12 row 旋转；对齐符号全翻转 |
| B 与 score vacuum 有明确时间关系 | ✅ | 单调共变，`Hc=−6.62` vs `Tc=−5.95` |
| **C 具体 upstream mechanism（非"分类器问题"）** | ❌ | M1 只解释范数；M4/M5 无 artifact；M6 停在候选 |
| D 单变量最小干预 | ❌ | 机制未定时任何干预都是猜 |
| E 能从现有 artifact 写出明确预测 | ❌ | — |

⇒ **NO-GO**。

## 6. What remains unknown（阻塞下一步的最多 3 条）

1. **驱动 W 整体旋转的量是什么？** 需要训练侧中间量（raw 权重轨迹 / 每类梯度范数 / optimizer state）。
   现有 artifact 只有 EMA 快照 ⇒ **无法从磁盘回答**。
2. **bias 为什么冻结？** 位移 ≤0.078（10 ULP）是事实，但"饱和导致梯度消失"只是 HYPOTHESIS，
   需要梯度或 raw 权重才可验。
3. **row 范数增长 ∝ 频率（corr +0.900）是否为因果？** 目前是 12 个点的横截面相关，
   且它对 logit 变化无解释力（corr −0.021）。

---

## 附：本轮**没有**做的事

- 未使用 G86 / final miss / TP / FP / matched50 定义任何 cohort
- 未 forward、未训练、未重跑、未改任何模型或数据
- 未重新讨论 P2 / P3 / OASA / TTA / 分辨率 / 容量等已关闭方向
- 未把 `temporally aligned` 写成 `caused by`
