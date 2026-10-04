# L2/L4 Diagnostic Run — Feasibility & Information-Value Audit

**日期** 2026-10-03 · **性质** READ-ONLY。未训练、未 forward、未改任何文件。

## Verdict

```text
L2 diagnostic = FAIL
L4 diagnostic = FAIL

OPEN-STRONG = NONE
GPU = NO-GO
```

**失败原因不是成本**（两者的成本都 **<1%** full training，远低于 25% 门槛）——
**是信息量**：L2 的前提被已有 D′-specific 证据直接否证；L4 即使测出阳性也判定不了 phenotype（Rule 2）、
且指向的干预没有预期 metric 收益（Rule 3）。

---

## 0. 顺带更正：checkpoint 官方证据比我上轮说的多

上一轮我写 "NO EXISTING OFFICIAL RESULT（除 best/last 外）" —— **过强，更正**。

`diagnostic/checkpoint_selection_audit/eval_ep*.json` 有 **8 个 D′ checkpoint 的官方评测**
（Sep 18，`predict_rect --mode rect`，同一 `rgbid_split` val split / 同一几何；cap 300 vs 100 不构成差异，
因每图仅 ~12 框）：

| 0-based | official mAP50-95 | vs best.pt |
|---:|---:|---:|
| 200 | 0.497143 | −0.01814 |
| 230 | 0.503690 | −0.01159 |
| 240 | 0.500973 | −0.01431 |
| 250 | 0.498583 | −0.01670 |
| 260 | 0.501458 | −0.01382 |
| 270 | 0.499782 | −0.01550 |
| 280 | 0.503960 | −0.01132 |
| **290** | **0.507417** | **−0.00786** |

**⇒ 全部低于 best.pt（0.515281），最好者仍差 0.00786。**
这**强化** checkpoint-selection 的 `NOT JUSTIFIED`：现在是**实测官方证据**而非仅 val 外推。
附带：8 点实测官方 sd ≈ **0.0032**（与既有噪声带估计一致），best.pt 高出最优点 **2.5 sd** ⇒ 不是噪声。

⚠ 同时注意一个反例：val 在 0-based **280** 最高（0.56151 > 290 的 0.55930），
而官方在 **290** 最高（0.50742 > 280 的 0.50396）—— **两者在 280/290 上排序相反**，再次印证 val→official 非严格单调。

---

## L4 — Gradient Conflict Diagnostic

### A. Measurement（可行性）

**可以分离。** `v8DetectionLoss.__call__` 返回**三项独立 loss**（`loss[0]=box`、`loss[1]=cls`、`loss[2]=dfl`），
因此标准做法成立：对同一 batch 做**两次 backward**（一次只回传 `loss[1]`，一次回传 `loss[0]+loss[2]`），
在 shared 参数上分别取 `grad`，逐参数拼接后算 cosine。

可分层测量：SepStem 后第一 shared stage / neck / head 末端 cv2·cv3 **之前的**共享卷积。

**不能用 `W·h`、Shapley、loss 相关性替代** —— 同意，那是前向量。

### B. Temporal sampling

**不需要重训**：early/middle/late 直接用**已有的 30 个 `epoch*.pt`**（例如 0 / 150 / 290）。
每阶段 20 batch ⇒ 60 forward + 120 backward。

### C–E.

| 项 | 结论 |
|---|---|
| Phenotype linkage | **不可建立**（见下） |
| Directionality | 技术上可做（沿 g_cls 方向扰动测 ΔL_box），但见下 |
| Control | **有**（同一 pipeline 跑多个 checkpoint） |

### F. Cost gate

```
full training ≈ 300 ep × 192 batch = 57,600 step（forward+backward）
diagnostic   ≈ 60 forward + 120 backward ≈ 300 forward-等价
⇒ ≈ 0.17% full training        【PASS（<10%）】
```

### L4 判定

| Item | Required | L4 |
|---|---|---|
| New measurable variable | yes | **yes**（cos(g_cls,g_box) 逐层） |
| Existing D′ evidence | yes | **no** |
| Can measure without full retrain | yes | **yes** |
| Control available | yes | **yes** |
| **Phenotype linkage measurable** | yes | **NO** |
| Directionality measurable | yes | yes（技术上） |
| GPU cost | <25% | **0.17%** ✅ |
| Information gain | — | **LOW** |
| **Diagnostic status** | — | **FAIL** |

**FAIL 的三条理由：**

1. **Rule 2**：诊断能给出的是一组 cosine 值 + 与 phenotype 的**3 点相关**。
   这**判定不了** conflict 是否*导致* phenotype —— 需要干预而非相关。
2. **Rule 1**：多任务检测器在 shared 层上 cls/box 梯度余弦**普遍为负** ⇒ **"存在 conflict"是预期内的必然结果**，
   测出来几乎不含信息。
3. **Rule 3 + 决定性的一条**：它要挂靠的 phenotype（score vacuum / W drift）
   **已被 Checkpoint audit §7 证明与 metric 同向**（`corr(val, live_dead)=+0.792`）。
   ⇒ 即使测出 conflict 并据此做 gradient balancing，**预期 metric 收益 ≤0**。
   **诊断即使阳性，也产生不了有意义的下一步干预。**

```text
L4 = FAIL
```

---

## L2 — Task/Feature Alignment Diagnostic

### 先回答 brief 要求的前置问题

> TOOD/TAL 的核心机制是否已被当前代码覆盖？

**是。** 见上一轮 `PAPER_DERIVED_MECHANISM_AUDIT.md` L1/L2：
`tal.py:112-116,150,155` —— 对齐分配（`align_metric = cls^α·CIoU^β` + topk + in_gts）与对齐 target 都已实现。
**未覆盖的只有 TAP**（head 内特征级任务对齐）。

### 决定性：D′ 上 **没有** spatial misalignment，且有反证

brief 要测的是"classification strongest 与 localization strongest 是否存在系统性 spatial misalignment"。
**这个量在 D′ 上已经被测过，结论是【对齐】。**

`diagnostic/prediction_head_geometry_audit/REPORT.md`（G1 失败阶段审计）：

> **STAGE 4 — classification / score**（**同一批 anchor 上 box 分支已给出 IoU≥0.5 的框，而 GT 类的 sigmoid 概率中位 3e-6**）
> head 的框回归输出**已经**为 63% 的 G1 给出 IoU≥0.5 的框（CIoU 中位 0.6278），
> 但这些框在最终输出里一个都没有出现；与此同时，**同一批 anchor 上 GT 类的 sigmoid 概率中位是 3×10⁻⁶**。

**⇒ 在同一个 anchor 上：框是对的（loc-strong），分数是死的（cls-dead）。**
这正是"cls 与 loc 在空间上**已经对齐**，失败发生在分类分支的输出"。

**TAP 式的特征对齐解决不了"框对、分数死"** —— 特征在空间上是好的，坏的是分类读出。
这与 P5/P5.1/P5.2 完全一致（head-driven、判别力反而在提升、真空是共同模）。

### L2 判定

| Item | Required | L2 |
|---|---|---|
| New measurable variable | yes | yes（top-k 空间重叠、cosine、距离） |
| **Existing D′ evidence** | yes | **有 —— 且是反证** |
| Can measure without full retrain | yes | yes（forward only） |
| Control available | yes | yes |
| Phenotype linkage measurable | — | n/a（**前提已否证**） |
| Directionality measurable | — | n/a |
| GPU cost | <25% | <0.5% ✅ |
| Information gain | — | **LOW** |
| **Diagnostic status** | — | **FAIL** |

**FAIL（有反证）**：诊断的前提（系统性 spatial misalignment）在 D′ 上**不成立**；
已有 anchor 级证据显示 cls 与 loc 对齐、失败在分类读出。**再测一遍只是重复一个已知为否的命题。**

```text
L2 = FAIL
```

---

## Final

```text
L2 diagnostic = FAIL   （前提被 D′ 证据否证）
L4 diagnostic = FAIL   （Rule 2/3：阳性也判定不了 phenotype，且无有意义的下一步干预）

OPEN-STRONG = NONE
GPU = NO-GO
```

**两者都不是"太贵"而是"测了没用"** —— 成本分别仅 0.17% 与 <0.5% full training，
但按 Rule 1/2/3 均不通过。

**因此不进入 diagnostic，也不进入 optimization。** D′ = 0.515281 保持。

---

## 附：本轮新做的核查

1. `v8DetectionLoss` 返回三项独立 loss（L4-A 可行性成立）。
2. 找到 `diagnostic/prediction_head_geometry_audit/REPORT.md` 的 anchor 级证据 ⇒ **L2 前提被否证**。
3. 找到 `diagnostic/checkpoint_selection_audit/` 的 **8 个官方评测** ⇒ 更正上轮的 "NO EXISTING OFFICIAL RESULT"，
   并强化 checkpoint-selection 的 NOT JUSTIFIED（全部低于 best.pt，最好者差 0.00786）。
