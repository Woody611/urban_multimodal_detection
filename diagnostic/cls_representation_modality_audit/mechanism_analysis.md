# Mechanism analysis — 分层与措辞纪律

本文件只做**分层**与**措辞审计**；数值见 `summary.md` / `per_gt_representation_audit.csv`。

---

## Level 1 — 输出层解离（`OBSERVED`）

```
GEOMETRIC_OUTPUT_CAPABILITY = OBSERVED
   86/86 在冻结 pre-final head 产物中有 CIoU >= 0.50 的候选（上一轮）
CLS_OUTPUT_COLLAPSE = OBSERVED
   86/86 在最终 post-NMS 输出中无 best_any >= 0.50；GT 类 logit 中位 −16.719
```
**这只是 output-level dissociation**（§12 明令）。它**不自动**证明 classifier representation 坏，
也**不自动**证明 IR/Depth 有害。

## Level 2 — 结构层（`PROVEN`，仅"允许"不"导致"）

- `Detect.forward`（`head.py:74`）：`cv2[i]` 与 `cv3[i]` 读**同一张量** `x[i]` ⇒ 输入共享。
- 两条 tower 参数完全独立，宽度 64 vs 256。
- 三模态在 `yaml:53` **concat** 融合；此后无 modality-specific 分支。

⇒ `MODALITY_INTERACTION_EXISTS = TRUE`（代码可证）；
`ARCHITECTURE_PERMITS_...` 可以说，**`ARCHITECTURE_CAUSES_...` 不可以说**（§21）。

## Level 3 — 表征层（`OBSERVED`，本轮新增，这是本轮的实质进展）

用**冻结**的 `p3` / `h` / `logit_decomp`（语义已逐行验证）：

| 读数 | G-86 | 对照（已检出，尺寸匹配） |
|---|---:|---:|
| 特征项 `W·h` 中位 | −6.965 | +10.950 |
| `h` 的 nearest-centroid 类别准确率 | 0.4651 | 0.9065 |
| `p3` 的 nearest-centroid 类别准确率 | **0.2558** | 0.6645 |
| cos(h, 类均值) | 0.62–0.70（7/7 类） | 0.83–0.97 |
| 共享 bias 项 | −9.758 | −9.758（**相同**） |

⇒ ① 位置相关的差异**全在特征项**（bias 是共享常数，结构上无法产生位置差异）。
⇒ ② **共享的 P3** 在这些 cell 上类别可分性 **≤ 随机**（0.256 vs 0.291），
   而 box tower 用同一份 P3 产出 **CIoU ≥ 0.50** 的几何。
⇒ ③ 这也回答了 §20：**存在** frozen evidence 表明 shared feature 支持几何却几乎不承载类别信息。
⇒ ④ 但见 Level 4 的反向可能。

## Level 4 — 因果方向（`NOT_OBSERVABLE`）★ 与上面严格分开

冻结的 `h`/`p3` 是**训练后**产物。86 从未进入 `mask_pos` ⇒ 它们所在 cell 的分类通路
**从未获得过该类的监督**（上一轮已证 target 只给正样本）。因此：

```
「该 cell 的共享特征类别不可分」
   与
「该 cell 从未被分类监督塑形，所以不可分」
```
是**同一份单快照数据的两种读法**，**方向不可分离**。分离需要 **per-epoch trajectory**。

```
TRAINING_CAUSALITY = NOT_OBSERVABLE        （与上一轮结论一致）
```

## §19 三个替代解释的逐一标记

| 解释 | 标记 | 依据 |
|---|---|---|
| **H1** classification representation 本身无法区分该 GT class | **PARTIALLY_SUPPORTED** | 有直接表征证据（准确率 0.465 vs 0.907、cos 差 −0.15~−0.34、特征项 −17.9 nats，且尺寸匹配后不减）；但 0.465 **仍高于随机** ⇒ "无法区分"过强；且方向不可分离 |
| **H2** multimodal fusion 后 class-discriminative 信息被削弱 | **NOT_OBSERVABLE** | 无 per-modality feature / logit / 梯度 artifact；`modality_dropout` 是训练期实验 |
| **H3** representation 尚可但 TAL / optimization dynamics 导致 collapse | **NOT_OBSERVABLE** | 上一轮已证 TAL **静态** assignment bias（79× 额外排除）；但 `TAL assignment bias ≠ TAL causes collapse`，无 trajectory |

## §21 禁止的伪 attribution（本轮全部**未**输出）

以下均**无直接 artifact 支持**，本审计**未**写成结论：
`IR hurts classification` / `Depth harms` / `RGB dominates` / `IR CLAHE causes class collapse` /
`SepStem causes class collapse` / `fusion destroys semantics` / `backbone loses class information` /
`classifier is underpowered` / `classifier is overfit`。

**已改用的措辞**：`architecture permits …` / `observed` / `consistent with` / `structurally plausible` /
`NOT_OBSERVABLE`。

## §15/§16/§17 的严格结论

- 融合前后通道 64→64 ⇒ `STRUCTURAL_COMPRESSION_PRESENT = FALSE`（**不是** information bottleneck）。
- 三模态合并前**同为 uint8 [0,255]** ⇒ `RANGE_DIFFERENCE = NOT OBSERVED`（**不是** range mismatch bug）。
- 分类头 bias 初值 −9.6395、训练后 −9.7949（位移 −0.15）；imgsz=1280 的等效先验为 −11.03。
  bias 是**全位置共享常数** ⇒ **不能**解释同类内"有的检出、有的塌陷"。
  初始化只影响早期，**不足以解释最终 epoch 的 1e-6**。
- **不存在**独立的 classification normalization / dropout / attention / gating / prior-bias 机制。
