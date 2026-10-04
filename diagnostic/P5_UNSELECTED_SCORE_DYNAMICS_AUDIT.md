# P5 — Unselected Score-Dynamics Audit

**日期** 2026-10-03 · **性质** 只读 / 零 GPU / 零训练 / 零 forward · **incumbent** D′ = SepStem + IR-CLAHE (0.515281)

> 结论分三层标注：**[FACT]** artifact 直接支持 ／ **[INFERENCE]** 由多个 FACT 推出 ／ **[HYPOTHESIS]** 无直接证据。

---

## 1. Executive Verdict

```text
VERDICT B — Mechanism partially supported
NO GPU. 需要一个额外 zero-GPU audit（把已存在的 30 快照 npz 同步回来）。
```

**本轮最重要的净结果**：
- **Hypothesis A（Assignment → Score）被否证** —— assignment 全程饱和且恒定。
- **score death 不是误分类，是"真空"** —— 整个 12 类向量一起被压低，GT 类通常仍是 argmax。
- **score death 是"极化"**：训练让中位分数单调升（0.375→0.866），同时死尾单调增（1.9%→13.0%）。
- **size 故事的符号取决于读数**（live head vs 固定最终头方向相反）⇒ 不能下 size 结论。

**但因果归因仍缺一个关键 artifact**：per-epoch 的**正确 W_t** 不存在于本地 artifact 中
（那轮 FULL300 的 `head_W` 陈旧；正确的 30 快照 npz 在云端未同步）。该 artifact **已存在**，
所以下一步是**同步 + 离线复算**，不是训练。

---

## 2. Artifact Inventory

| Artifact | Geometry | Epoch | Checkpoint | Contains | Does not contain | Valid for P5? |
|---|---|---|---|---|---|---|
| `runs/..._sepstem_clahe/weights/best.pt` | — | ~245 | sha `1cae45f7…` | 完整 W(12×256), b(12) | — | **是**（分类头真值） |
| `diagnostic/full300_trajectory_probe/epoch_{0,1,2,5,…,299}.npz`（19 个） | **rect** | 19 点 | D′ EMA（追踪中） | `vec_h`,`vec_p3`,`cls_pos_max`,`cls_center_max`,`cls_pool_max`,`ciou_*`,`n_pos`,`mask_pos_membership`,`h_norm` | **`head_W` 陈旧（恒定）⇒ `logit` 列无效** | **部分**（见下） |
| `diagnostic/p3_feature_space/_v7_vectors.json`（V1 2807） | **square** | 单快照 | D′ best.pt | `h`(256) per GT、`logit_decomp`、`bin`、`sq` | 无时间维 | 是（单快照） |
| `diagnostic/cls_representation_modality_audit/per_gt_representation_audit.csv` | square | 单快照 | D′ best.pt | `gt_class`,`native_area`,`cls_logit_stride8`,`h_norm` | 时间维 | 是（单快照） |
| `diagnostic/p3_disappearance_audit/layers.npz` + `attribution.json` | **rect** | 单快照 | D′ best.pt | pre-NMS 候选、failure 桶 | **无 decoded box 张量；只存获胜类** | 部分 |
| `runs/..._sepstem_clahe/results.csv` | — | 300 | — | **聚合** loss/mAP | **无 per-class / per-GT** | 仅作背景 |
| `runs/..._FULL300_EMA/weights/epoch*.pt` | — | 30 | EMA | **正确 W_t** | 在**云端**，本地缺失 | **缺（关键）** |

**标记**：
- **train/val**：轨迹 npz 与 representation audit 均为 **val**（无增强）。
- **pre/post NMS**：`layers.npz` 的 layerA 是 **post-conf-gate**、layerB 是 post-NMS。
- **score-gated vs raw**：layerA 已过 `amax(cls) > 0.001`；**raw decoded 张量从未落盘**。
- **outcome-selected**：`attribution.json` 的 `failure` / `matched50` **是**；轨迹 npz 的 `role`(G86/CONTROL) **是**；本轮全部**不使用**它们定义 cohort。

**本轮实际使用的数据**：19 个 rect 轨迹 npz 的 **`vec_h` / `cls_*` / `ciou_*` / `n_pos` / `mask_pos_membership`**（这四类量都来自**真实 forward**，不受陈旧 `head_W` 影响）＋ `best.pt` 的 W/b ＋ GT 几何（由 label 文件直接算）。

---

## 3. Cohort Definition（全部 outcome-independent）

- **Class**：12 类，其中 n≥20 的 7 类可做统计（person 482 / animal 359 / car 167 / light 102 / sign 84 / bicycle 50 / garbage_can 24）。
- **Size**：用户口径 small<1024 / medium 1024–9216 / large≥9216。**本轨迹只覆盖 medium（1320 全部落在 1024–9216）** ⇒ 大小维度用 **medium 内部三分位**（<49.3 / 49.3–69.0 / ≥69.0 px 线性尺寸）。
- **Class × Size**：对 n≥20 的类 × 三分位（样本 <20 不做强结论）。
- **Geometry 派生 cohort**（本轮新增，纯 GT/图像定义）：距最近图像边归一化距离、长宽比、图内 GT 密度、图像分辨率层（HR 1920×1080 / LR 640×360）。

**全部不使用**：G86 / final miss / TP / FP / matched50 / best_same_iou 阈值 / role 字段。

---

## 4. Assignment → Score Analysis

**[FACT]** 在**所有** cohort、**所有** 19 个观测点上：

```
mask_pos_membership = 1.000        （每个 GT 恒有正样本 anchor）
n_pos               = 10.0         （top-k=10 恒定打满）
ciou_pos_max        = 0.80 – 0.86  （平坦，无趋势）
```

分组示例（完整表见附录）：

| cohort | ciou_pos_max ep0 → ep299 |
|---|---|
| 近边 (border<0.126) | 0.837 → 0.802 |
| 远边对照 | 0.834 → 0.800 |
| 小 (sqrt<45) | 0.809 → 0.757 |
| 大 (sqrt≥75) | 0.857 → 0.833 |

**[INFERENCE]** ⇒ **Hypothesis A（Assignment → Score）不受支持**。
assignment 从未"先恶化"，它**从来没有恶化**。score 的变化不可能由 assignment 驱动 ——
至少在 TAL 正样本分配这一层没有可用的自由度（top-k 饱和、CIoU 平坦）。

**[HYPOTHESIS]** assignment 的**质量**（而非数量）是否在下游（如 target score 的构造）有别的作用，本轮无法判定。

---

## 5. Representation → Score Analysis

### 5.1 h_norm 轨迹（per-GT，outcome-independent）

| cohort | ep0 | ep10 | ep20 | ep100 | ep299 |
|---|---:|---:|---:|---:|---:|
| 近边 | 63.3 | 45.2 | 41.3 | 42.0 | 49.5 |
| 远边 | 65.5 | 45.7 | 42.5 | 42.5 | 51.7 |
| 小 | 63.6 | 45.5 | 41.7 | 41.3 | 50.2 |
| 大 | 62.3 | 44.9 | 41.3 | 42.6 | 52.2 |

**[FACT]** `h_norm` 在 **ep0→ep20 急降 ~35%**，此后**单调回升**至 ep299。所有 cohort 形状相同。

**[INFERENCE]** 这是一个**warmup 重标定瞬态**，不是退化；且各 cohort 之间没有可见的 h_norm 分化
（近边 vs 远边、小 vs 大 在任一 epoch 的差都 ≲3）。

### 5.2 固定最终分类头读取（counterfactual readout）

用 `W_final`（best.pt）读每 epoch 的 `h_t`，得到"若分类头冻结在最终态、仅表征演化"的 GT 类 logit 中位：

| ep | 全体 | 小 | 中 | **大** |
|---|---:|---:|---:|---:|
| 0 | −5.56 | −4.74 | −4.61 | **−8.86** |
| 5 | +0.15 | +0.34 | +0.61 | −1.32 |
| 50 | +0.04 | −0.16 | +0.54 | −1.06 |
| 150 | −0.10 | +0.40 | +0.90 | −4.54 |
| 250 | −0.40 | +0.70 | +1.46 | −7.41 |
| 299 | −0.80 | **+0.68** | **+1.53** | **−8.76** |

**[FACT]** 在固定最终读数下：**小/中的表征持续靠近最终类方向，大目标的表征持续远离**。

**[INFERENCE]** 表征侧**存在** size-相关的漂移，而且 **大小排序与 live-head 相反**（见 §9）。

---

## 6. Multimodal Interaction Analysis

**[FACT]** 本轮**没有**任何可用的 per-cohort × per-modality artifact：
`_v7_vectors` / `per_gt_representation_audit` 只有 RGBID 融合后的 `p3`/`h`，无 IR/Depth 分解；
modality ablation 的历史产物是**逐目标 TP 存活率**，属 outcome-selected。

**[HYPOTHESIS]** 三模态融合是否对特定 class×size 产生不对称影响 —— **UNKNOWN**。
要判定需要 forward（本轮禁止）。

⇒ **Hypothesis C：UNKNOWN。**

---

## 7. Class × Size Analysis

### 7.1 末 epoch per-GT 分数分布（`cls_center_max`，无 assignment 依赖）

| class | n | p25 | p50 | **dead(<1e-3)** |
|---|---:|---:|---:|---:|
| car | 167 | 0.781 | 0.893 | **6.6%** |
| animal | 359 | 0.722 | 0.883 | 9.5% |
| person | 482 | 0.635 | 0.847 | 10.6% |
| light | 102 | 0.017 | 0.878 | 20.6% |
| sign | 84 | 0.017 | 0.872 | 23.8% |
| garbage_can | 24 | 0.0015 | 0.614 | 25.0% |
| bicycle | 50 | 0.0002 | 0.683 | **28.0%** |

**[FACT]** 每类都是**双峰**：中位 0.61–0.89，同时 6.6%–28% 的成员低于 1e-3。class 调制 ~4×。

### 7.2 Size

基率 13.0%（1320 全体，ep299）：

| 分层 | n | dead 率 |
|---|---:|---:|
| sqrt < 49.3 px | 440 | **18.3%** |
| 49.3–69.0 | 440 | 10.0% |
| ≥ 69.0 | 440 | 10.8% |

**[FACT]** size 调制 ~1.8×，且**极化在三个 size 桶里都发生**（1.8%/1.6%/2.3% → 18.3%/10.0%/10.8%）。

### 7.3 Geometry / image 派生（本轮新增）

| 分层 | n | dead 率 |
|---|---:|---:|
| **距图像边 < 0.126** | 330 | **19.4%** |
| 距图像边 ≥ 0.126 | 990 | 10.9% |
| **长宽比 > 3.20（细长）** | 132 | **8.3%** |
| 长宽比 ≤ 1.64（方形） | 660 | 14.8% |
| 图内 GT ≤ 7 | 688 | 16.1% |
| 图内 GT > 7 | 632 | 9.7% |
| HR (1920×1080) | 1281 | 13.3% |
| LR (640×360) | 39 | 5.1% |

**[FACT]** 长宽比方向**与"细长目标更易死"相反**（8.3% < 14.8%）。

---

## 8. Temporal Precedence

### 8.1 score death 是**纵向极化**，不是横截面少数

| ep | dead(<1e-3) | dead(<1e-1) | 分数中位 |
|---:|---:|---:|---:|
| 0 | 5.5% | 27.0% | 0.375 |
| 2 | **1.9%** | 15.0% | 0.587 |
| 10 | 3.7% | 14.7% | 0.516 |
| 50 | 5.0% | 13.6% | 0.667 |
| 100 | 6.9% | 14.7% | 0.740 |
| 200 | 10.9% | 18.6% | 0.839 |
| 299 | **13.0%** | 18.9% | **0.866** |

**[FACT]** 中位分数**单调上升**，同时死尾**单调扩大** ⇒ **极化（polarization）**，不是塌缩。

**[FACT]** 死掉的多是**新面孔**：ep10∩ep299 Jaccard **0.242**；ep20∩ep299 **0.261**；ep100∩ep299 0.503。

### 8.2 三类量的时间排序

- **assignment**：全程平坦（无变化可排序）
- **representation (h_norm)**：ep0→ep20 急降，ep20 后平稳，无 cohort 分化
- **score**：ep2 起极化，持续到 ep299

**[INFERENCE]** 没有观察到 "assignment → score" 或 "representation → score" 的**单调时间先后**：
representation 的显著变化（warmup 瞬态）**早于**极化，但**在极化开始时已经结束**。
因此不能声称 representation-first。

**[HYPOTHESIS]** 极化是一个**与 warmup 解耦的、贯穿全程的分类侧过程**。

---

## 9. Negative Evidence / Counterexamples

1. **细长目标假设被否**：长宽比 >3.2 的 dead 率 **8.3%**，低于方形的 14.8%。G86 的"细长类"组成**不能**用长宽比解释。
2. **"小目标是问题"读数依赖**：live head 下小目标 dead 率最高（18.3% vs 大 10.8%）；固定最终头下**大目标**最差（54.1% vs 小 29.8%）。**符号相反** ⇒ 不能下"小目标机制"结论。
3. **边界效应不是标签裁剪造成的**：近边组 23.0% 的框被图像边界裁剪，远边组 0.0%；但近边组内 **未裁剪 20.0% > 裁剪 17.1% > 远边 10.9%**。⇒ 边界效应**在裁剪控制后仍在**。
4. **assignment 饱和**：`n_pos` 恒 10、`mask_pos` 恒 1.0 —— 这一层没有自由度，任何"assignment 退化"叙事都不成立。
5. **class 只是调制不是决定**：dead 率 6.6%–28%，但**每一类都有 dead 成员**，没有一类是"全死"。

---

## 10. Mechanism Matrix

| Mechanism | Evidence available | Outcome-independent? | Temporal support? | Geometry valid? | Strength | Status |
|---|---|---|---|---|---|---|
| **Assignment → Score** | `n_pos`/`mask_pos`/`ciou_pos_max` 全程平坦饱和 | 是 | **无变化可排序** | rect 单口径 | **Unsupported** | 否证 |
| **Representation → Score** | h_norm 瞬态 + 固定头 counterfactual | 是 | 瞬态**早于**极化但已结束 | rect（h 来自 live forward） | **Weak** | 不足以判定 |
| **Multimodal interaction** | 无 per-cohort×modality artifact | — | — | — | **UNKNOWN** | 无法判定 |
| **Class/size imbalance** | class 4× / size 1.8× / 边 1.8× / 密度 1.7× 调制；极化在所有桶都发生 | 是 | 只在 score 侧 | rect | **Moderate** | 是调制器，非机制 |
| **Pure box-regression failure** | 未持久化 decoded box 张量 | — | — | — | **UNKNOWN** | 无法判定 |

**补充 FACT（本条不属于 A–D 任何一项）**：**score death 是"真空"而非误分类** ——
对 451 个 dead GT（square 口径，`_v7_vectors`×best.pt）算完整 12 类 logit：
GT 类中位 **−14.57**、max-other 中位 **−17.93**、margin 中位 **+3.58**、max-other sigmoid ≥1e-3 仅 **2.2%**。
⇒ 整个向量一起被压低，GT 类通常仍是 argmax。**不是"被别的类抢走"。**

---

## 11. What Survives From Previous G86 Analysis

- G86 ⊂ rect 漏检集、几何不敏感（上一轮 hygiene audit 结论）—— 不受本轮影响。
- "训练放大既有 gap"这一描述 —— 在本轮以**更强、无选择**的形式重现为**极化**（中位升 + 死尾增）。
- score death 与 conf 阈值的因果链 —— 仍成立。

## 12. What Is Retired

- 「**训练使某群逐渐死掉**」的叙事 —— 无选择数据显示：死尾增长是**极化**，且**成员大多是新面孔**（Jaccard 0.24）。
- 「**small-object-only mechanism**」 —— 符号依赖读数（§9.2）。
- 「**assignment 是上游**」 —— 该层无自由度（§4）。
- 「**细长目标是机制**」 —— 方向相反（§9.1）。
- 一切把 CASE C 的 rep/cls 比例当作因果的说法（沿用上一轮降级）。

---

## 13. Minimum GPU Intervention

```text
无。
```

按 §Q5 的五项条件逐条：

| 条件 | 满足？ |
|---|---|
| cohort outcome-independent | ✅ |
| artifact chain 自洽 | ✅ |
| temporal evidence 支持机制 | ❌（§8.2） |
| geometry 无混淆 | ⚠（轨迹 rect，分类头/真空分析 square —— 已分离标注） |
| intervention 可直接针对机制 | ❌ |

⇒ **NO GPU — mechanism unresolved.**

**但有一个正当的、非训练的下一步（zero-GPU）**：
把云端已存在的 **30 个快照 npz**（`from_snapshots/epoch_*.npz`，约 78 MB）同步回来 ——
它们带**正确的 `head_W`/`head_b`**（那轮修复已生效），可以给出 **per-epoch 的 W_t × h_t**，
从而在**无选择**的条件下判定：极化的死尾是"表征漂移撞上固定分类器"还是"分类器在追表征"。
**这正是本轮唯一缺失的变量，而它已经存在于磁盘上。**

---

## 14. Final Verdict

```text
VERDICT B — Mechanism partially supported
```

- **有方向性证据**：A 被否证；真空（非误分类）；极化；三到四个 outcome-independent 调制器。
- **不足以值得 GPU**：没有任何一条满足 Q5 五项。
- **下一步是同步 artifact，不是训练**。

**关于 §18 的硬性要求**：本轮**没有**把"classification stage 已关闭"当作前提 ——
正是因此才发现"真空"与"极化"这两个新的、无选择的表层事实；也正因此才敢说
assignment 层无自由度。但**上游机制仍未找到**：能找的都找了，缺的变量**已经存在于磁盘**。

---

## 附录：完整 cohort 轨迹表

（`cls_center_max` / `ciou_pos_max` / `h_norm` / `mask_pos` / `n_pos` × 4 cohort × 19 epoch，
见会话中输出的完整表；`mask_pos` 恒 1.000、`n_pos` 恒 10.0 已在上文说明。）
