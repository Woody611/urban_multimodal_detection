# P8 — 1536 Optimization Compatibility Audit

**日期** 2026-10-03 · **性质** ZERO GPU / ZERO training / ZERO forward / ZERO inference —— 只读配置、args、results.csv、报告、源码。

## 1. Artifact availability

| Artifact | 状态 |
|---|---|
| 1536 run #1 | `runs/urban_multimodal_det_yolo11_rgbird_1536_finetune/`（25 ep，args+results.csv+png） |
| 1536 run #2 | `runs/urban_multimodal_det_yolo11_rgbird_1536_finetune_lr5e4/`（25 ep） |
| 实验报告 | `reports/rgbid_1536_finetune_report.md`（含 4 个官方评测点） |
| 纯推理对照 | 同报告：`1536 纯推理` = 0.50398（1280 训练的模型在 1536 推理） |
| D′ 基线 | `runs/..._sepstem_clahe/`（300 ep，官方 0.515281） |
| **D′ lineage 的 1536 run** | **不存在** —— 见 §2 |
| 1536 的官方 evaluator 输出 | 报告中已记录（4 点），**无需补跑** |

---

## 2. Exact 1280 vs 1536 diff —— **这不是一个 1536 实验**

**关键发现：历史上那个"1536 失败"的实验根本不是对 D′ 的 1536 测试。它同时改变了 5 个变量。**

| Variable | D′@1280（incumbent） | "1536" 实验 | Same? | Potential scale interaction |
|---|---|---|---|---|
| **model** | **`yolo11m_sepstem.yaml`** | **`yolo11m_earlyfusion.yaml`** | ❌ | **去掉了 D′ 的核心贡献（SepStem）** |
| **ir_encoding** | **clahe（显式）** | **—（args.yaml 无此键）** | ❌ | **静默回落 percentile** |
| **epochs** | **300** | **25** | ❌ | 训练预算只有 1/12 |
| **pretrained** | `yolo11m.pt`（从头） | `ir_quicktest/best.pt`（**热启动微调**） | ❌ | 起点已收敛 |
| patience | 0 | 80 | ❌ | — |
| imgsz | 1280 | **1536** | ✅ 目标变量 | — |
| batch | 8 | 8 | ✅ | — |
| optimizer | SGD | SGD | ✅ | — |
| lr0 | 0.005 | 0.005（probe1）/ **0.0005**（probe2） | ⚠ | 已试过 0.1× LR |
| momentum / wd | 0.937 / 5e-4 | 同 | ✅ | — |
| warmup_epochs | 3.0 | 3.0 | ✅（但分母不同） | 见 §5 |
| box / cls / dfl | 7.5 / 0.5 / 1.5 | 同 | ✅ | — |
| mosaic / mixup / scale / fliplr | 1.0 / 0 / 0.5 / 0.5 | 同 | ✅ | — |
| close_mosaic | 10 | 10 | ✅ | 但 25ep 下 close_mosaic 占比不同 |
| rect / channels / use_simotm | False / 5 / RGBID | 同 | ✅ | — |
| AMP / EMA | True / 默认 | 同 | ✅ | — |

**⇒ 结论：「1536 失败」这个前提本身不成立 —— 我们没有 1536 对 D′ 的证据，无论正负。**
（D′ 于 9/22 才出现；1536 实验是 9/18，早于 SepStem 与 IR-CLAHE 合并。）

**并且**：这个实验里 **LR 已经试过一次 0.1×**（probe2 lr0=5e-4），结果 **0.50673 < 1280 基线 0.50928**。

---

## 3. Object-scale analysis（实测 label geometry，非理论）

| cohort | n | native √area 中位 | **@1280** | **@1536** |
|---|---:|---:|---:|---:|
| **small (<1024)** | 371 | 24.2 px | **17.1 px** | **20.6 px** |
| medium (1024–9216) | 1320 | 57.8 px | 39.6 px | 47.5 px |
| large (≥9216) | 1116 | 166.7 px | 111.1 px | 133.4 px |

理论线性放大 **1.2000×**、面积 **1.4400×**。与 brief 给的 17.7→21.2 一致（口径略有差异）。

**P3 stride-8 覆盖**：small 在 1280 下 ≈2.1 cell、在 1536 下 ≈2.6 cell —— **仍在 stride-8 的最小可分辨量级附近**。

---

## 4. Optimization scaling analysis

| | D′@1280 | 1536 probe1 | 1536 probe2 |
|---|---:|---:|---:|
| pixels/batch ∝ imgsz²·b | 1.31e7 | **1.89e7（×1.44）** | 1.89e7 |
| effective batch | 8 | 8 | 8 |
| LR / eff-batch | 6.25e-4 | 6.25e-4 | 6.25e-5 |
| LR × batch | 0.040 | 0.040 | 0.004 |
| optimizer steps/epoch | ≈192 | ≈192 | ≈192（**imgsz 不改变步数**） |

**⇒ 计算量 ×1.44，但 optimizer step 数完全相同（batch 不变）。** 因此"每步的像素数"涨了 1.44×，
而 LR 未按像素归一化 —— 这是一个**真实存在、可识别的 optimization scaling mismatch**。
（probe2 用 0.1× LR 做了部分补偿。）

---

## 5. Warmup / scheduler analysis

| | D′@1280 | 1536 |
|---|---:|---:|
| warmup epochs | 3.0 | 3.0 |
| total epochs | 300 | **25** |
| **warmup 占比** | **1.0%** | **12.0%（12×）** |

**⇒ 1536 的 warmup 吃掉了 12% 的预算**，而它是**从已收敛模型热启动**的 —— 组合起来产出的
曲线形态见 §8。**这是明确的 mismatch（Gate 1 意义上），但它是"实验设计"的 mismatch，不是"1536 本身"的证据。**

---

## 6. Augmentation geometry analysis

`mosaic=1.0`、`scale=0.5`、`close_mosaic=10`、`rect=False` —— **两个 1536 run 与 D′ 逐字相同**。

- Mosaic canvas = 2×imgsz ⇒ 1280:2560 / 1536:3072（比例不变）
- `scale=0.5` 的对数均匀缩放**与 imgsz 无关** ⇒ **随机缩放的相对幅度不变**

**⇒ 从配置层面看，1536 的 20% 线性放大【不会】被 augmentation 结构性地抵消。**
（精确的目标尺寸分布需 dataloader 执行 ⇒ **NOT MEASURABLE**，本轮不做。）

**但**：25-epoch 的 run 里 `close_mosaic=10` 意味着 **40% 的训练在 mosaic 关闭后**，
而 D′ 的 300ep 里只有 3.3% —— 这是**又一个**不可比项。

---

## 7. Loss / localization analysis

`box=7.5 / cls=0.5 / dfl=1.5` 与 TAL（α=0.5, β=6.0）**完全一致**。

**关于 localization**：D′ 的官方 per-IoU 曲线 `[0.770,0.754,0.723,0.687,0.645,0.501,0.403,0.310,0.215,0.053]`，
AP75 = 0.510963。而 1536 probe2 的 mAP75 = **0.54516 vs 1280 基线 0.56386 —— 更低**。
（虽是 confounded 值，但**没有任何 AP75 的正面证据**。）

**关于 classification**：P5/P5.1/P5.2/P6/P7 已闭环，**本轮不重开 cls / cosine / margin 方向**。

---

## 8. Historical 1536 learning curves

| probe | lr0 | 训练内 val mAP50-95 峰值 | 末值 | 官方 |
|---|---:|---:|---:|---:|
| Probe 1 | 5e-3 | **0.54244 @ep1** | 0.5397 | 0.49401（ep1） |
| Probe 2 | 5e-4 | **0.55294 @ep2** | 0.5433 | **0.50673（ep2）** |
| 1280 基线（同期） | — | — | — | **0.50928** |
| 1536 纯推理（1280 模型） | — | — | — | 0.50398 |

**⇒ 两个 probe 都在 ep1–2 见顶，随后下滑。** 这是**热启动微调的瞬态**（warmup 占 12% + 已收敛起点），
**不是收敛曲线** ⇒ **无法从中读出"1536 收敛更慢/更早平台"的任何结论**。

**§12 统计意义**：每个 LR 只有 **1 个 run**，且变量不隔离 ⇒
`single-run negative result ≠ proof of intrinsic scale failure`（brief 原文）。**反向同样不成立。**

---

## 9. Failure mechanism classification

| 类别 | 判定 |
|---|---|
| A — Intrinsic scale saturation | ❌ **不成立**（没有有效的 1536 实验来支持它） |
| B — Optimization mismatch | ⚠ **存在**（LR/像素 1.44× 未归一化、warmup 占 12%、热启动 25ep）——但这是**实验设计的 mismatch**，不是"1536 本身"的 |
| C — Augmentation/geometry mismatch | ❌ 配置层面两 run 与 D′ 逐字相同，无结构性抵消（唯一不可比项是 close_mosaic 占比） |
| D — Capacity/optimization interaction | OPEN-WEAK（无法确认） |
| **E — No identifiable mismatch（不足以支持 GPU）** | ← **本轮落点**，但理由不是"1536 已证明无效" |

**本轮的真实落点是：既不能判 A（没有有效实验），也不能判 B（mismatch 全在实验设计上，且 LR 已试过 0.1×），
而 Gate 3/4/5 全部不满足 ⇒ NO-GO。**

---

## 10. Relation to D′ bottleneck

**这是本轮的决定性一节。**

| D′ 的失败结构 | 占比 |
|---|---:|
| **F0 无 raw candidate** | **25.61%** |
| F1 NMS 消失 | 8.09% |
| F2 top-100 消失 | 2.96% |
| **F3 存活但官方匹配失败（= 定位/匹配失败）** | **3.50%** |

**⇒ D′ 的弱项不是 localization（F3 仅 3.5%），是 F0。**
而 P3→P4→P5 已把 F0 的机制定位为 **GT 类 sigmoid < conf 阈值（类分数问题）**，
P7 进一步显示其 feature 落在一个**健康流形的下尾**（全局 M_proto>0 占比 91.7%，dead∧F0 仅 51%）。

**⇒ 「1536 → 目标变大 → 小目标定位更好」这条链条，指向的不是 D′ 已被识别的瓶颈。**
Gate 4（预期改善必须针对 D′ 当前弱项 small/medium **localization**）**不成立**。

---

## 11. Minimal GPU matrix

```text
N/A —— 未进入 stage design（见 §12）。
```

（若强行设计，按 brief §14 的 sequential screening，Stage 1 最多 3 runs；但本轮未通过 Gate，不提供。）

## 12. GPU gate

| Gate | 要求 | 结果 |
|---|---|---|
| 1 | 历史 1536 失败存在可识别的 optimization/geometry mismatch | ✅ **有**（LR 未按 1.44× 像素归一化；warmup 占 12%；热启动 25ep）—— **但这是实验设计的 mismatch** |
| 2 | 该 mismatch 与 imgsz 变化有因果合理关系 | ✅ 部分（像素/步 ×1.44 确实来自 imgsz） |
| **3** | 能明确预测 AP75/AP90（非仅 loss） | ❌ **无任何正面 AP75 证据**；confounded 数据反而显示 1536 的 mAP75 更低（0.54516 vs 0.56386） |
| **4** | 预期改善针对 D′ 当前弱项 **small/medium localization** | ❌ **D′ 的弱项是 F0（类分数），不是 localization（F3 仅 3.50%）** |
| **5** | 有机会超过 0.515281 且不依赖噪声 | ❌ 零证据（唯一同族对照 1536=0.50673 vs 1280=0.50928，且早已试过 0.1× LR） |
| 6 | 第一阶段 ≤3 runs | — |

```text
Gate 3 / 4 / 5 任一不成立 ⇒ NO GPU
```

## 13. Final verdict

```text
FINAL VERDICT: NO-GO
```

### Q1–Q7 逐条

**Q1 — 之前 1536 为什么输？**
**它没有"输"在一个有效的 1536 实验上。** 那个实验同时改了 **架构（SepStem→earlyfusion）、IR 编码（clahe→percentile）、
训练预算（300→25ep）、起点（从头→热启动微调）**，且两个 probe 都在 ep1–2 见顶后下滑 —— 是热启动瞬态，
不是收敛曲线。**它对我们想知道的事零信息量。**

**Q2 — intrinsic / optimization / augmentation？**
**三者都不是可判定的。** A 无法判定（无有效实验）；B 存在但**全在实验设计上**；C 在配置层面不成立。
⇒ 落入 **E 的邻近 —— 但理由不是"1536 已被证明无效"，而是"没有证据、且目标与瓶颈不符"。**

**Q3 — 1536 理论上能否改善 small-object bottleneck？**
**不能针对 D′ 的 bottleneck。** D′ 的小目标瓶颈是 **F0 = 类分数低于阈值**（不是分辨率不足）；
F3（定位失败）仅 3.5%。**目标变大 ≠ 类分数变高。**

**Q4 — 1536 是否会损害 D′ 已有的 medium/large localization 收益？**
**confounded 数据弱支持"会"**（同族 1280 基线 mAP75 0.56386 > 1536 probe2 0.54516）。
但单 run、多变量 ⇒ **不足以作为结论**，只作为风险提示。

**Q5 — 最该改哪个变量？**
```text
1st priority : 无（未进入 tuning 阶段）
2nd priority : 无
```
理由：**不是"该改哪个变量"的问题，而是"目标变量与瓶颈不符"的问题。**

**Q6 — 最小 GPU 矩阵？** `N/A`（未通过 Gate）。

**Q7 — 最终 verdict：** 见上。

---

```text
FINAL VERDICT: NO-GO
```
