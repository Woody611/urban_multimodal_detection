# Paper-Derived Mechanism Audit — Re-open GPU Candidate Space

**日期** 2026-10-03 · **性质** READ-ONLY · 全部结论由 vendored 源码 / 已有 artifact / 官方结果得出，
未训练、未 forward、未改任何文件。

## Verdict

```text
OPEN-STRONG = NONE
GPU = NO-GO
```

五个论文机制里，**一个已经在当前代码中实现**（L1），**一个与前者本质重复**（L5），
**一个被官方数据直接否证**（L3），剩下两个**只有"理论上可能"、无任何 D′ 证据**（L2/L4）。

---

## Final table

| Candidate | Paper mechanism | Existing coverage | Independent D′ evidence | Contradiction | Implementation | **GPU status** |
|---|---|---|---|---|---|---|
| **L1** GFL / VFNet | cls target/score 融入定位质量 | **已实现**：`tal.py:112-116,150-155` —— `target_scores = one_hot × norm_align_metric`，其中 `norm_align_metric ∝ (cls^0.5 · CIoU^6) × CIoU` | 无 | 「train-side cls supervision」终裁 **RED**；TAL 已把 GT 类 target 峰值钉死为 maxCIoU（98.6% 精确） | 残余只有 **loss 形式**（`VarifocalLoss_YOLO`/`QualityfocalLoss_YOLO` 已在 repo 但**被注释**，`loss.py:415-420`） | **CLOSED** |
| **L2** TOOD | 任务对齐 = 对齐分配 + 对齐 target + 特征对齐 | **TAL 已实现**（同 L1）；**未覆盖**的只有 **TAP**（head 内的特征级任务对齐） | 无 | — | 需**新增模块**（架构级），非单变量 loss 改动 | **OPEN-WEAK** |
| **L3** EFL / EQL / Seesaw | 类别级负梯度抑制 → 稀有类被压 | 无 | 模式暗示（见下） | **官方 per-class AP 直接否证** | — | **CLOSED** |
| **L4** CAGrad | cls / loc 梯度冲突 | 无任何测量 | 无 | P5.1 的 R/H Shapley **不能替代**梯度冲突测量 | 需 instrumented run | **OPEN-WEAK** |
| **L5** Integral/multi-IoU | cls target 区分 IoU quality | **⊂ L1**（target 已含 `CIoU^6`） | AP75/AP50 = 0.657（常规区间） | 前提「当前是 binary target」在本仓库**为假** | 与 L1 合并（不单列） | **CLOSED** |

---

## L1 — Quality-aware Classification

**逐项回答 brief 的 6 问：**

| # | 问题 | 答案 |
|---|---|---|
| 1 | YOLO11 cls target 是否已显式含 localization quality？ | **是**。`tal.py:150` `align_metric = bbox_scores^α · overlaps^β`，`overlaps = CIoU`（`:155`，`beta=6.0`）；`:112-116` 把 `align_metric × pos_overlaps` 归一化后乘进 `target_scores` |
| 2 | assignment quality 是否只用于选 positive？ | **否**。它**同时**进入 positive 选择**与**最终 cls target |
| 3 | 若两者分离，是否构成未测机制？ | **前提不成立**（未分离）⇒ 不构成 |
| 4 | 是否与 confidence calibration / conf=0.0001 属不同机制？ | **是不同机制**（那两个是推理侧阈值），但**不因此成为候选** —— 它的机制面已被 TAL 覆盖 |
| 5 | 官方 AP50:95 是否提供额外动机？ | **否**。per-IoU 曲线 `[0.770,0.754,0.723,0.687,0.645,0.501,0.403,0.310,0.215,0.053]` 是常规单调衰减；且 target 已用 `CIoU^6` 极强加权高 IoU |
| 6 | 能否只改 cls target/loss，其余不动？ | **技术上可以**（`loss.py:415-420` 有现成的 `VarifocalLoss_YOLO` / `QualityfocalLoss_YOLO`，仅需取消注释） |

**⇒ CLOSED。** 残余的只是一个 **loss 形式变体**（BCE-on-soft-target → Varifocal/QFL），
而：① 机制本体已实现；② 该轴（cls 监督）已被 「train-side cls supervision」审计判 **RED**
（结论是监督信号**不是**瓶颈）；③ 无任何独立证据表明当前形式次优。

**为什么不给 loss 形式变体开 GPU**：它满足"技术上单变量"，但**不满足 §4-Q3/Q4**
（未被现有 negative 覆盖 + 有独立证据支持）。它是**同一机制的参数化变体**，不是新机制。

---

## L2 — Classification / Localization Task Alignment

| # | 问题 | 答案 |
|---|---|---|
| 1 | TOOD 核心机制是否已覆盖？ | **大部分是**。TOOD = TAL（对齐分配 + 对齐 target）+ TAP（特征对齐）。**TAL 已实现** |
| 2 | assignment quality 是否已用于 cls target？ | **是**（同 L1） |
| 3 | TOOD 里仍真正缺失的？ | **只有 TAP** —— head 内学习一个任务对齐的特征注意力 |
| 4 | 不能把"重做 TAL"包装成新实验 | 同意；TAL 侧无候选 |

**⇒ OPEN-WEAK。** 未覆盖部分（TAP）是**架构级新模块**，而非单变量改动；
**没有任何 D′ 侧证据**表明需要特征级任务对齐；且其形态落在已关闭的 "attention" 家族近邻。

```text
not sufficient for GPU
```

---

## L3 — Class-wise Negative Gradient Suppression

**brief 明确提醒不要把 `corr(freq, Δ(W_c·h_m)) = −0.021` 当成对该机制的证伪。我不这么做。**
下面给出的是**另一条、更直接的**否证。

**模式确实存在（与 L3 同向）** —— 稀有类的 head 损失更大：

| cls | train GT | P5.2 H 中位 |
|---|---:|---:|
| person | 4426 | −2.973 |
| animal | 2328 | −3.343 |
| car | 1283 | −3.234 |
| light | 1211 | **−7.392** |
| uav | 174 | **−8.231** |

**但官方 per-class AP 与之相反**（D′ lineage，`Dp_last/_official.json`）：

| | train GT | **官方 AP50-95** |
|---|---:|---:|
| person | 4426（最频繁） | **0.443** |
| tricycle | 23（最稀有） | **0.702（最高）** |
| seat | 514 | 0.638 |
| boat | 107 | 0.526 |

```
corr(类频率, 官方 AP50-95) = −0.204      ← 稀有类 AP 更高，不是更低
corr(官方 AP50-95, P5.2 H中位) = −0.276   ← head 损失更大的类，AP 并未更低
```

**⇒ CLOSED（有反证）。** EFL/EQL/Seesaw 的**前提**是"稀有类因负梯度被抑制而表现更差"；
D′ 上**最稀有的 tricycle 拥有最高的 AP**，**最频繁的 person 反倒最低之一**。
机制quantity（实际负梯度压力）在现有 artifact 中**不存在**，且**模式本身已被 metric 层否证**。

**按 brief 的规则**："如果没有机制证据，直接 CLOSED" ⇒ **CLOSED**。

---

## L4 — Classification / Localization Gradient Conflict

| # | 问题 | 答案 |
|---|---|---|
| 1 | cls/box/dfl 是否共享主要 representation？ | **是**（共享 backbone 与 neck；只有 head 末端的 cv2/cv3 分支分开） |
| 2 | 是否已有实验测过 `cos(g_cls, g_box)`？ | **没有**。全库无任何梯度冲突测量 artifact |
| 3 | P5.1 的 R/H Shapley 能否替代？ | **不能**。那是**前向**的状态分解（`W·h` 的两条路径），与**反向梯度**无关；两者数学对象不同 |
| 4 | 是否存在极小的 instrumented diagnostic？ | **不存在 zero-GPU 版本** —— 需要一次带梯度记录的训练 |
| 5 | 是否与 W drift / AP dynamics 有时间对应？ | **未知**（从未测量） |

**⇒ OPEN-WEAK。** brief 明确禁止"仅因 multi-task conflict 是常见问题就开 GPU"。
当前**零测量**，不满足 Gate C（有独立于"希望它有效"的证据）。

```text
not sufficient for GPU
```

---

## L5 — IoU-threshold-aware Classification

| # | 问题 | 答案 |
|---|---|---|
| 1 | 当前 cls target 是否区分 IoU quality？ | **是** —— target 含 `CIoU^6`（`beta=6.0`），对高 IoU 极端加权 |
| 2 | 是否存在 AP50 正常但 AP75/AP90 不足的 evidence？ | **否**。AP75/AP50 = 0.657、per-IoU 单调平滑，属常规 |

```
corr(...) 无需；per_iou = [0.770,0.754,0.723,0.687,0.645,0.501,0.403,0.310,0.215,0.053]
```

| 3 | 是否与 L1 本质重复？ | **是（⊂ L1）** |
| 4 | 合并 | 已合并，**不单列 GPU experiment** |
| 5 | — | — |

**⇒ CLOSED。** 该方向的前提（"普通 binary classification target 对 IoU quality 区分不足"）
在本仓库**为假**：target 不仅不是 binary，而且用 `CIoU^6` 把定位质量作为**主要**权重。

---

## 结论

```text
OPEN-STRONG = NONE
GPU = NO-GO
```

**每个非 OPEN-STRONG 的关闭原因（一句话）：**

- **L1 CLOSED** —— 机制已在 `tal.py` 实现（cls target 已含 `CIoU^6`）；残余仅为 loss 形式变体，且 cls 监督轴已 RED。
- **L2 CLOSED 部分 / OPEN-WEAK 部分** —— TOOD 的 TAL 已覆盖；仅剩 TAP，属架构级新模块且无 D′ 证据。
- **L3 CLOSED** —— 稀有类 AP **更高**（tricycle 0.702 最高），与机制前提相反；且无负梯度压力 artifact。
- **L4 OPEN-WEAK** —— 零测量；前向 Shapley 不能替代反向梯度冲突。
- **L5 CLOSED** —— ⊂ L1；"binary target" 前提为假。

**链条完整性自查**：本轮**没有**使用 "P5 发现 head 在变 → 所以改 head" 这类推理。
五个候选全部走了 `paper mechanism → D′-specific observable → existing evidence → (intervention) → falsification`
的完整链条；四个在此链条上断于 **existing evidence** 一环。

---

## 附：本轮新做的核查（非复述）

1. **读源码确认 L1/L5 的机制已实现**：`tal.py:112-116 / 150 / 155`（CIoU^β 进入 target）、
   `loss.py:415-420`（GFL/VFNet/QFL 实现存在但被注释，D′ 用普通 BCE，`cls_pw: []`）。
2. **读官方 per-class AP 否证 L3**：`corr(类频率, 官方AP50-95) = −0.204`。
3. **确认 L4 无测量**：全库不存在 `cos(g_cls, g_box)` 或任何梯度冲突 artifact。
4. **确认 L5 ⊂ L1**：同一 `tal.py` 代码路径。
