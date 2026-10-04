# Mechanism analysis — TAL cls feedback

严格按任务的四个 Level 分层，**不得越级**。

---

## Level 1 — 数学结构（`PROVEN`）

`tal.py:150` `align_metric = bbox_scores.pow(alpha) * overlaps.pow(beta)`，cls 取自 **GT 类那一列** 的
sigmoid 概率（`tal.py:147` + `loss.py:522`），几何用 CIoU（`tal.py:155`）。
有效 `alpha=0.5, beta=6.0, topk=10`（`loss.py:438` 实例化）。

⇒ **cls 参与 TAL 排序**，且其指数为 0.5。**秩 1 结论：`PROVEN`。**

### 由结构直接导出的两个数值事实（解析，非经验）

```
(1) cls^0.5 的上界约束：cls < 0.001  ⇒  align <= 0.001^0.5 * 1^6 = 0.0316   （无论 CIoU 多好）
(2) 等 alignment 边界：cls_B / cls_A = (iou_A / iou_B)^(beta/alpha) = ^12
```
`(2)` 的实例（`equal_alignment_boundary.csv`）：

| iou_A / iou_B | cls_B / cls_A | 含义 |
|---|---:|---|
| 0.5 / 0.8 | 3.55e-03 | CIoU 0.8 的候选只需 0.5 那个候选 **0.36%** 的 cls 即可打平 |
| 0.5 / 0.9 | 2.77e-04 | 只需 **0.028%** |
| 0.7 / 0.9 | 3.54e-02 | 只需 3.5% |

⇒ **几何项（CIoU^6）在数值上压倒性强于 cls 项（cls^0.5）**：CIoU 0.5→0.8 的几何优势
等价于 **282× 的 cls 优势**。
**但同时**：cls 项有一个**乘性地板** —— 一旦 cls 极低，`cls^0.5` 会把 align 整体压到地板附近，
几何优势再多也换不回来（见下）。

---

## Level 2 — assignment 后果（`PROVEN`，条件性）

给定冻结的 cls 值，`cls^0.5` 的乘性地板在**真实池**里是**可判定的**：
对 cls = 1.64e-07（该组中位），要追平 cls = 0.8697 的正样本需要

```
(ciou_A / ciou_B)^6 > (0.8697 / 1.64e-07)^0.5 = 2304   ⇒  ciou_A / ciou_B > 2304^(1/6) = 3.63
```
而 CIoU ≤ 1、正样本 CIoU 中位 0.953 ⇒ 需要 `ciou_A > 3.46` ⇒ **在合法 CIoU 范围内不可能**。
⇒ **在冻结的分数分布下，这一档候选在数学上无法进入 top-10。`PROVEN`（条件于当前分数）。**

---

## Level 3 — 观测数据（`SUPPORTED`，且已完成几何/cls 的分解）

1,905,266 个真实候选（train split，10,105 GT；top-k 为 **OBSERVED**：`cand_align > 0` 即 `mask_pos`）
—— 且重建校验通过：重算排序 top-10 **100.00% 覆盖**观测正样本（89,735/89,735）。

### 关键组：`CIoU >= 0.50 且 cls < 0.001` → **217,025 个候选**（11.39% of all），涉及 7,668 个 GT

**决定性分解**（`rank_decomposition.csv`）—— 同一批候选，只换排名依据：

| 排名依据 | 池内 median rank | ≤ top-10 的比例 |
|---|---:|---:|
| **CIoU（纯几何）** | 29 | **11.09%** |
| **align（几何 × cls）** | 30 | **0.14%** |

⇒ 两个**互相独立**的事实：
1. **几何本身就不够**：88.9% 的该组候选连池内**几何** top-10 都进不去（median rank 29 / 池约 41）
   ⇒ 对多数候选，cls **不是**它们被排除的原因。
2. **cls 项造成额外排除**：在**几何合格**的那 11.09% 中，align 排名把它们压到只剩 **0.14%**
   ⇒ 相对几何基线 **79× 的额外排除**，且 **97.7% 的 GT 一个该组候选都没进正样本**。

**反事实**（`candidate_counterfactual.csv`，几何/cls 单变量）：

| 干预 | 进入 top-10 |
|---|---:|
| 原始值（观测） | 0.14% |
| 只把 cls → 0.1（几何不变） | 0.83% |
| 只把 CIoU → 正样本中位 0.953（cls 不变） | 0.21% |
| 两者同时 | 1.69% |

⇒ 单改 cls（10^6× 提升）只回收 0.83%；单改几何只回收 0.21%；**两者都改也只有 1.69%**。
⇒ 说明该组的低分**同时**由"几何略差"与"cls 极低"两侧贡献，**没有单一决定性变量**。

---

## Level 4 — 训练因果（`NOT_OBSERVABLE`）★ 必须与上面严格分开

**这是本轮最重要的限制，且它同时削弱和限制上面的解释：**

- 冻结的 `cand_cls` 是 **训练后** 的分数。正样本的 cls 中位 **0.8697**，而
  `target = maxCIoU`（上一轮：中位 0.72）**正是 BCE 把正样本的 cls 往上推**的产物。
- 非正样本**从未获得监督**（它们不在 `mask_pos` 中），因此它们的 cls **从未被训练提高**。
- ⇒ 「低 cls 的候选被排除」与「被排除的候选所以 cls 低」**是同一份冻结数据的两种读法**，
  **冻结快照无法分离方向**。
- ⇒ 要分离，需要 **per-epoch 的 candidate-level trajectory**
  （epoch t 的 score → TAL assignment → target → epoch t+1 的 score），
  这**在冻结产物中不存在**，且需要带插桩的训练。

```
TRAINING_CAUSALITY = NOT_OBSERVABLE
```

---

## §13 反证检查（必须做）

**问**：如果 cls 极低，为什么这些候选仍能拿到 median 0.72 的 peak target？

**答（代码级，`PROVEN`）**：因为 `target_peak = maxCIoU over positives`，**cls 不进入 target 的幅度**：
```
tal.py:110-116   target = 1.0 * (align/max_align) * maxCIoU_over_positives
```
在 align 最大的那个正样本上 `align/max_align = 1` ⇒ `target = maxCIoU`，**与 cls 无关**。

⇒ **代码里确实存在一种机制，使「低 cls 但高 CIoU」的候选一旦入选就拿强 target**：
`pos_overlaps = (overlaps * mask_pos).amax(-1)` 只用 **CIoU**，不用 cls。

**这在两个方向上都有含义，必须同时写明：**
- **削弱**「cls feedback → supervision collapse」：一旦入选，监督强度只看几何，不看 cls。
- **但**它同时意味着：**入选与不入选之间的差异完全由 cls 决定**（因为 target 的幅度不受 cls 影响，
  而入选与否受 cls 影响）⇒ 反馈**只能通过"入选/不入选"这一道闸门**起作用，不能通过 target 幅度起作用。

---

## 逻辑错误检查（§15）

**已避免**：没有因为 `cls ≈ 1e-6` 就直接断言"所以它没进 top-k"。
本轮用的是 **同池内其他候选的真实 align**（`rank_decomposition.csv`），即相对排序，不是绝对阈值。
并且为低分档单独做了 **几何排名** 的对照（median 29），从而把"几何不够"与"cls 压低"分开。
