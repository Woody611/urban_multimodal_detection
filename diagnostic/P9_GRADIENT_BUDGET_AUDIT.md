# P9 — Classification Gradient Budget / Common-Mode Attribution Audit

**日期** 2026-10-03 · **性质** ZERO GPU / 只读源码 + 已有 artifact CPU 重统计 · 无 forward/backward/训练

## 1. Executive verdict

```text
MECHANISM STILL OPEN（PARTIAL）
GPU NO-GO
```

**本轮得到一条精确的代数结构，但拿不到它的量级；量级拿不到就无法通过 Gate 1/3，因此不申请 GPU。**

---

## 2. D′ classification loss implementation（源码证据）

| 项 | 实测 |
|---|---|
| loss 类 | `ultralytics/utils/loss.py:415` `self.bce = nn.BCEWithLogitsLoss(reduction="none")` |
| 其余实现 | `VarifocalLoss_YOLO` / `QualityfocalLoss_YOLO` / `SlideLoss` / `FocalLoss_YOLO` **全部被注释**（`:416-420`） |
| class weighting | `cls_pw: []` ⇒ **无** |
| 分类 logits shape | `(b, h*w, 12)`，三尺度拼接（`loss.py` 里 `pd_s`） |
| target shape | `(b, h*w, 12)`，**只在 assigned positive 的 GT 类上非零**（one-hot 基底 × `norm_align_metric`） |
| reduction | `loss[1] = loss_cls.sum() / target_scores_sum`，`target_scores_sum = max(target_scores.sum(), 1)` |
| 归一化含义 | **分母 = 正样本 target 之和**（不是 anchor 数、不是 fg 数） |
| objectness | YOLO11 头 **无独立 objectness 分支** ⇒ 不存在 objectness-independent 的额外分类惩罚 |
| 每个 background anchor | **对 12 个类都有 negative BCE 项**（y=0） |

## 3. Exact gradient formulation

```
∂L/∂z_{a,c} = (sigmoid(z_{a,c}) − y_{a,c}) / target_scores_sum
y_{a,c} = 0            ∀ c≠GT  或  ∀ background anchor a
y_{a,c} = CIoU_target  only at TAL-assigned positive anchors, c = GT class
```

**∂L/∂W_c = Σ_a (∂L/∂z_{a,c}) · h_a**

---

## 4. Gradient budget

| source | count /图 | abs budget /图 | 依据 | 可信度 |
|---|---:|---:|---|---|
| **GT positive** | ≈70（7 GT × n_pos 10） | **≤24** | 1320 medium GT 的 `\|σ−y\|`（P8.5）外推到全尺寸，**上界** | proxy |
| **positive non-GT**（同一 anchor 上 c≠GT） | 同类 anchor × 11 | **UNMEASURABLE** | 需 per-anchor 12 类向量，**未落盘** | — |
| **background / unassigned**（y=0） | **≥ 12.9**（框外）＋ 框内未分配部分 | **≥2.59**（框外 Σconf） | `layerA` 框外行 Σconf；框内未分配部分**无法区分** | proxy，**下界** |

**术语项计数（EXACT）**：

```
anchors/图 = 160²+80²+40² = 33600
(anchor,class) 项/图 = 33600×12 = 403,200
positive-target 项/图 ≈ 7×10 = 70
⇒ positive 项占比 = 0.01740%
```

**⚠ 但项数 ≠ 预算**：403,200 个 y=0 项里绝大多数 σ(z) ≪ 0.001（D′ 全图仅 ~82 个 (anchor,class) 对超过 conf 0.001），
每项梯度 ~1e-7 量级 ⇒ **y=0 预算由那少数高 conf 行主导**，不是由项数主导。

**⇒ Gate 1（background >> positive）无法成立**：可测的下界 2.59 vs positive 上界 24 —— **方向相反**；
而真正未知的是"框内未分配 anchor"（y=0 但 σ 高）有多大，那部分**恰好无法从 artifact 区分**。

---

## 5. Temporal audit

```text
UNMEASURABLE
```

需要 **per-anchor、per-epoch** 的 logit/assignment；现有 artifact 只有 **per-GT** 的 `cls_pos_max` /
`cls_center_max` / `ciou_pos_max`（30 epoch 的 1320 medium GT）与 **单快照**的 `layerA`（val、final 模型）。
**无法把 budget 按 epoch 拆开。** 不伪造。

---

## 6. Common-mode attribution —— 本轮唯一的精确结果

**代数（EXACT，与数据无关）**：

对任一 **y=0** 的 (anchor, class) 项，`∂L/∂W_c = σ(z_c) · h_a`。同一 anchor 的 `h_a` **对 12 个类完全相同**，故：

```
ΔW_c^{y=0} = −λ · σ(z_c) · h_a        （c = 1..12，同一 h_a）
```

若各 σ(z_c) 相近（background 上 12 类分数都低时成立），则

```
ΔW_c^{y=0} ≈ −λ σ̄ h_a   ∀c   ⇒   Δ(W_c − W_mean) = 0，ΔW_mean = −λ σ̄ h_a
```

**⇒ 所有 y=0 项对 classifier 的更新【100% 落在 common-mode】，对 discriminative 分量的贡献【恒等于 0】。**

对 y>0 的 positive 项：`ΔW_c = +λ(σ−y)h`，**只作用于 W_GT** ⇒ 有 discriminative 分量。

**⇒ 「谁在推动 W_mean？」的答案：y=0 项【必然】100% 推 common-mode；positive 项只有 1/12 的份额落到 common-mode。**

**这一条精确解释了 P5.2 的形态**（`‖W_mean‖` +188% 而 `‖W_c−W_mean‖` −3.7% 基本持平）：
**只要 y=0 项的预算不为零，它就只加在 common-mode 上，从不加在 discriminative 上。**

**⚠ 但这是 D′-independent 的普遍性质**（任何用 y=0 negative + 共享 h 的 BCE 稠密检测器都成立），
**且其量级未测** ⇒ 不构成 D′-specific 证据，也不能外推到"这就是 W_mean +188% 的原因"。

---

## 7. Positive-gradient paradox（§9）

P8.5 已证：F0 的 positive 梯度**强**（0.55–0.72），却仍进入更深负 logit。逐条检查 brief 给的 7 个候选：

| # | 候选 | 本轮结论 |
|---|---|---|
| 1 | positive 在总预算中小 | **UNMEASURABLE**（只能给出上界 24/图；无 per-anchor 分解） |
| 2 | positive 梯度方向主要进 common-mode | **否（代数）**：positive 项对 W_mean 的贡献只有总量的 1/12 |
| 3 | background 数量级远大于 positive | **未支持**：可测下界 2.59 vs 上界 24，方向相反 |
| 4 | non-GT 抵消 positive | **UNMEASURABLE** |
| 5 | h 的方向/norm 使 g·h 更新很弱 | **UNMEASURABLE**（无 per-anchor h） |
| 6 | EMA 掩盖瞬时 raw-weight dynamics | **UNMEASURABLE**（只有 EMA 快照，无 raw 权重轨迹） |
| 7 | optimizer / LR / momentum | **UNMEASURABLE**（无 optimizer state） |

**⇒ 悖论未被解决。** 5 项中 2 项判否/未支持、4 项不可测。

## 8. Causal chain（§12 逐边标注）

```text
classification targets                         [SUPPORTED]  P8.5 + 源码
        ↓
BCE gradient sources                           [SUPPORTED]  §3 精确
        ↓
GT / non-GT / background budget                [NOT SUPPORTED]  §4 —— 无法确立 background >> positive
        ↓
common-mode vs discriminative update           [PARTIALLY SUPPORTED]  §6 代数：y=0 ⇒ 纯 common-mode
        ↓
W_mean ↑ / W_centered ≈ flat                   [SUPPORTED]  P5.2 实测（+188% / −3.7%）
        ↓
GT-cell absolute logit ↓                       [SUPPORTED]  P5.2 实测（−8.67 → −14.46）
        ↓
F0 ↑                                           [SUPPORTED]  P3 实测
```

**⇒ 链条在【budget】这一环断开。** 后续各环本身都成立，但把它们串起来的因果连接未建立。

## 9. Falsification test（只描述逻辑，不开 GPU）

若机制成立，最小证伪即：

```text
把 y=0（background / unassigned-anchor）分类项的贡献下调一个数量级
（例如对非 assigned anchor 的 cls 项做按 anchor 数归一或硬性降权），
其余一切不变。

若机制正确 ⇒ W_mean 增长应显著变缓，且 GT-cell 绝对 logit 不塌；
若 W_mean 轨迹不变 ⇒ 该机制被证伪。
```

**但本轮 Gate 1 未通过 ⇒ 这个干预【现在不应该跑】**。

## 10. Final gate（§13）

| Gate | 要求 | 结果 |
|---|---|---|
| 1 | background/non-GT budget **>>** GT positive budget | ❌ **未通过**（可测下界 2.59 vs 上界 24，方向相反） |
| 2 | 该 imbalance 明确投影到 common-mode | ⚠ **PARTIAL**（y=0 项代数上 100% common-mode，但量级未测） |
| 3 | temporal precedence / 强方向性 | ❌ **UNMEASURABLE** |
| 4 | 可用最小干预直接 falsify | ✅ 逻辑可写（§9），但无证据支持先跑 |
| 5 | 能同时解释 W_mean +188% / centered flat / mean logit −8.67→−14.46 | ⚠ 解释前两个，第三个需要量级 |

| 问 | 答 |
|---|---|
| Mechanism identified? | **PARTIAL** |
| D′-specific? | **NO**（§6 的代数对任何 BCE 稠密检测器成立；D′ 特异性只在"形态吻合"，不在机制本身） |
| Temporal precedence? | **UNKNOWN** |
| Common-mode attribution? | **PARTIAL** |
| Minimal intervention identifiable? | **NO**（预算未定，无法定干预） |
| **GPU candidate?** | **NO** |

---

## 附：本轮新做的核查

1. 源码确认 `loss.py:415` 普通 BCE、`cls_pw: []`、归一化分母 = Σ positive targets、无 objectness 分支。
2. **精确项计数**：positive 项占 (anchor,class) 项的 **0.01740%**（33600 anchors × 12 类）。
3. **精确代数**：**y=0 项对 (`W_c − W_mean`) 的贡献恒等于 0**，100% 落在 common-mode —— 这精确解释 P5.2 的形态。
4. **layerA 按几何切分**：框外（必为 y=0）Σconf **2.59/图**；框内 42.84/图 —— 但框内**无法区分** assigned positive 与 unassigned-inside（后者 y 也是 0）。
5. 确认 **temporal / per-anchor / optimizer 类证据全部 UNMEASURABLE**，未做任何模拟或理论值顶替。

**结论：本轮没有把"监督预算"这一环建立起来 —— 不是被证伪，是现有 artifact 无法测。因此不申请 GPU。**
