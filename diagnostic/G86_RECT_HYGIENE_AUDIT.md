# G86 — RECT PIPELINE DATA HYGIENE AUDIT

**日期** 2026-10-03 · **性质** 只读 / 零 GPU / 零训练 / 零 full inference · **incumbent** D′ = SepStem + IR-CLAHE (0.515281)

---

## 0. Executive Verdict

```text
STATUS B — PARTIALLY VALID
```

**结论一句话**：G86 在 rect 口径下**不被否证**（86/86 全部落在 rect 的末输出漏检集内，且该集合对 square→rect 的几何切换**几乎不敏感**，Jaccard 0.977），
但 G86 的**定义**（`certified_prefinal_loss`，含头部认证）与它最关键的那个读数（GT 类 sigmoid 5.5e-08）**都只存在于 square 口径**，
在 rect 流水线中**没有可复现的对应 artifact**。

⇒ **部分证据保留，部分必须降级。不做任何优化结论。**

---

## 1. Artifact Manifest

| Artifact | Date | Geometry | Checkpoint | Input size | Coordinate system | Source |
|---|---|---|---|---|---|---|
| `diagnostic/small_object_domains/_v6_domains.json` | Sep 26 | **square** (`rect=False`) | D′ best.pt | 1280 | square-canvas | `_v6_domains.py:177,184` |
| `diagnostic/medium_93_47_attribution/per_gt_attribution.csv` | Oct 1 | square（继承 v6） | D′ best.pt | 1280 | — | `input_manifest.txt` |
| `diagnostic/medium_93_47_attribution/_run.log` / `consistency_checks.txt` | Oct 1 | square | D′ best.pt | — | — | — |
| `diagnostic/p3_feature_space/_v7_vectors.json` | Sep 27 | **square** (`rect=False`) | D′ best.pt | 1280 | square-canvas | `_v7_extract.py:79-83` |
| `diagnostic/cls_representation_modality_audit/per_gt_representation_audit.csv` | Oct 1 | square（源自 v7） | D′ best.pt | 1280 | square-canvas | `_audit.py` |
| `diagnostic/p3_disappearance_audit/layers.npz` | Oct 3 | **rect** | D′ best.pt | 1280 | **原图像素** | `meta_layers.json` |
| `diagnostic/p3_disappearance_audit/attribution.json` | Oct 3 | rect | D′ best.pt | 1280 | 原图像素 | 同上 |
| `diagnostic/sepstem_clahe/best_full/results`（生产预测 400 txt） | Sep 22 | rect | D′ best.pt | 1280 | 原图归一化 | 生产产物 |
| `runs/..._sepstem_clahe/weights/epoch{0..290 step10}.pt` | Oct 3 | —（权重） | EMA 快照 | — | — | `save_period=10` |
| `runs/..._sepstem_clahe/weights/best.pt` | Sep 22 | —（权重） | sha `1cae45f7…fae4fda` | — | — | — |
| 数据集 split | — | — | — | — | `data/processed/rgbid_split_train/{images,labels}/val/visible` 400 图 / 2807 GT | — |

**统一项**：所有 artifact 用的是**同一个 checkpoint**（sha `1cae45f7…`，除 EMA 快照外）、**同一个 val split**、**同一个 imgsz=1280**。
**不统一项**：几何（square vs rect）；坐标系（square-canvas vs 原图像素）。

**未定位项**：`_v6_domains.json` 内的 `pos_prob` 是否与 rect 有任何对应物 —— **UNKNOWN**（rect 侧无任何 TAL-assign artifact）。

---

## 2. Canonical G86 Identity

- **key = `f"{image_stem}#{gt_id}"`**，`gt_id` = **标签文件中的行序**（非数组下标、非预测下标、非行号）。
- 86 个 key 由 `diagnostic/p3_trajectory_probe/_g86.py` 从 `per_gt_attribution.csv` 的 `certified_prefinal_loss == 1` 导出。
- **Tolerance**：下游 CSV 存的 canvas 坐标是 `round(x, 4)` ⇒ 坐标等价容差 **1e-4 px**（实测最大偏差 0.0001 px，见 §3）。
- 稳健性：`medium_cells_canvas.json` 与 label 文件、`_v7_vectors.json`、`attribution.json` 的 key **全部一一对应**（1320/1320、2807/2807 匹配，无缺）。

---

## 3. Geometry Equivalence

对 86 个 G86 GT 各自的 (cx, cy) 做 **原始 → canvas → 逆变换 → 原始** 往返：

| 变换 | 最大绝对误差 |
|---|---:|
| square（`gain=min(1280/w,1280/h)`，居中 pad） | **1.137e-13 px** |
| rect（`ceil(shape·1280/32+0.5)·32` 画布） | **2.274e-13 px** |

存储端：`medium_cells_canvas.json` 的 canvas 坐标 vs 按 square 重算 = **max|Δ| 0.0001 px**（= `round(,4)` 精度）。

⇒ **坐标变换精确、可逆、无漂移**。两个几何各自内部自洽。

**但**：`medium_cells_canvas.json` 的坐标是 **square-canvas**（其生成脚本 `_medium_cells.py` 用 `rect=False`），
而 `layers.npz` 的候选是 **原图像素**（rect）。二者不可直接相减 —— 本轮按各自口径分别映射后才比较。

---

## 4. G86 Membership Reproduction

**目标**：不信任 G86 文件，从 rect pipeline 的原始证据重建。

**能重建的部分**（`certified_prefinal_loss` 的**必要**条件 = "末输出无 ≥0.5 同类框"）：

| 集合 | 定义 | n |
|---|---|---:|
| G86（历史，square 定义） | `certified_prefinal_loss==1` | **86** |
| `rect_miss` | P3 rect `matched50 == False` | 256 |
| `sq_miss` | square `frozen_best_same_iou < 0.5` | 250 |
| `sq_zero` | square `frozen_best_same_iou == 0` | 145 |

| 比对 | ∩ | recall | precision | Jaccard | 仅历史 | 仅对方 |
|---|---:|---:|---:|---:|---:|---:|
| G86 vs `rect_miss` | **86** | **100.0%** | 33.6% | 0.336 | **0** | 170 |
| G86 vs `sq_miss` | **86** | **100.0%** | 34.4% | 0.344 | **0** | 164 |
| G86 vs `sq_zero` | **86** | **100.0%** | 59.3% | 0.593 | **0** | 59 |

**并且**：`rect_miss` 与 `sq_miss` 的 **Jaccard = 0.977**（250/256 重合）。

⇒ 两个结论：
1. **G86 ⊂ rect_miss，无一例外**。G86 不构成对 rect 流水线的矛盾。
2. **"末输出漏检"这个集合对 square→rect 的几何切换几乎不敏感**（Jaccard 0.977）。所以几何不是 G86 成员的判别因子。

**不能重建的部分**（`certified_prefinal_loss` 的**充分**条件 = "头部 square TAL 正样本 CIoU≥0.5"）：
rect 侧**不存在任何 TAL-assign artifact**（`layers.npz` 只有 post-conf 的 layerA 与 post-NMS 的 layerB）。
⇒ **G86 无法在 rect pipeline 完整重建**；能重建的只有它 100% 包含于 `rect_miss`。

---

## 5. Definition Equivalence

| Definition | square（`_v6_domains`） | Oct-1 attribution | rect P3 | Equivalent? |
|---|---|---|---|---|
| "certified" 的载体 | TAL 正样本 anchor 的 **CIoU** | 继承 v6（同列） | **不存在** | ✗ 无对应物 |
| CIoU/IoU 阈值 | ≥ 0.50 | ≥ 0.50 | — | — |
| 几何 | square（`rect=False`） | square | **rect** | ✗ |
| 正样本选择 | `select_candidates_in_gts` + topk=10 | 同 | — | — |
| "pre-final" 定义 | pre-NMS 头部张量 | pre-NMS 头部张量 | — | — |
| "final" 定义 | `best_full/results`（生产预测，post-NMS+top100） | 同 | `matched50`（官方 IoU≥0.5） | ≈ 同 |
| 置信阈值 | — | — | **0.001**（layerA 入口） | 新增门槛 |
| 坐标帧 | square-canvas | square-canvas | 原图像素 | ✗ |

**判定：NOT EQUIVALENT。** 差异不是"基本一样"：
- 「certified」这一半**只在 square 存在**；
- rect 侧多了一道 **conf>0.001** 的门（layerA 定义），square 侧的"认证"没有这道门。

这两条差异**恰好落在 P4 结论所依赖的变量上**。

---

## 6. Independent GT-class Score Reproduction

**P4 的声明**：`G86 GT-class sigmoid 中位 5.5e-08，100% < 1e-3`。

**该数字的来源链**：`per_gt_representation_audit.csv::cls_sigmoid_stride8` ← `_v7_vectors.json::logit_decomp` ← `_v7_extract.py`（**`rect=False`，square**）。

| 量 | square | rect |
|---|---|---|
| GT 类 sigmoid 中位 | 5.5e-08 | **UNKNOWN** |
| <1e-3 占比 | 100% | **UNKNOWN** |

**rect 侧不可独立复现**：`layers.npz` 只持久化**获胜类**（报告原文："only the winning class was persisted"），
GT 类的分数从未落盘；重算需要一次前向，而本轮**禁止 inference**。

⇒ **按 §7 要求如实标注：rect 口径下 UNKNOWN，不强行构造。**

**关于 control**：本轮**未**构造 control。可用 artifact 中，唯一能给出 GT 类分数的 `per_gt_representation_audit.csv`
其"已检出"对照组本身就是按**最终检测结果**定义的 ⇒ 必然产生 selection/circularity（见 §8）。
**不依赖最终结果的 control：无法构造 ⇒ UNKNOWN。**

---

## 7. Raw Candidate / F0 Dependency Audit

### 7.1 Case 判定

**结论：Case A** —— 原始回归候选存在，被**类分数阈值**滤除。

**依据（架构 + 代码路径，不是倒推）**：
- YOLO11 头是 **anchor-free 稠密**预测：**每个 cell 每个尺度都产生一个框**。所以"该位置没有框"在架构上不可能。
- `layerA` 的记录入口是 `non_max_suppression` 的 `xc = prediction[:, 4:mi].amax(1) > conf_thres`（conf=0.001），
  再按 `multi_label=True` 逐类展开。⇒ **一个 12 类分数全都 <0.001 的框会被整个丢弃、不落盘。**
- 实测佐证：G86 的 **any-class** raw IoU 中位也是 **0.000**（≥0.5 仅 1.2%）——不只是 GT 类没有，是**没有任何类**在那附近留下候选。

**诚实限度**：**存储证据无法直接分离 Case A 与 Case B**，因为 `layers.npz` **未持久化 decoded box 张量**。
上面的判定依赖"稠密头必然产框"这一架构事实；若未来要硬证据，需要重新落盘 pre-score 的原始张量（本轮禁止）。

### 7.2 F0 的依赖性（§9）

P3 的 **F0 定义为「没有 usable same-class raw candidate（IoU ≥ 0.10）」**，而 usable candidate 的入口**本身就依赖 class-score 阈值**。

⇒ **F0 与 GT-class score death 不是独立观察**（在 Case A 下，F0 基本由后者蕴含）。

量化（square 侧可得）：small F0 桶 95 个中 **95.8% 的 GT 类 sigmoid < 1e-3**；
剩下 4.2% 才可能是真正的几何性 F0。

---

## 8. Selection / Circularity Audit

**Selection variables（定义 G86 用的）**：

```
G86 membership  =  head_certified(square)  ∧  final_miss
CONTROL membership = ¬in_86  ∧  detected=1        ← 同样按最终结果定义
```

**Outcome variables（后来用来"证明"G86 特征的）**：
`cls_logit_stride8` / `cls_sigmoid_stride8` / 轨迹的 `rep_sym`/`cls_sym` / raw IoU。

```
                true latent difficulty
                  /                \
         representation         classifier (W)
                  \                /
                   \              /
                      score ─────────────┐
                        │                │
                   final miss ───────────┤
                        │                │
                   G86 selection ────────┘   ← 用 outcome 选组，再用 outcome 论证
```

| 方向 | 现有数据能支持什么 |
|---|---|
| selection → outcome | **定义蕴含**（G86 ⊂ final miss ⊂ score<阈值），不可作为证据 |
| score 随 epoch 演化 | **可描述**（轨迹确实测了 score），但组本身是 outcome-selected ⇒ 组间差异只能**描述**，不能归因 |
| score → final miss | **可支持**（阈值机制，Case A） |
| representation/classifier → score | 对称分解可**描述**贡献比例，但**不可**从 outcome-selected 组外推到全体 |

**特别指出**：`CONTROL` 定义为 `detected=1` —— **它也是 outcome-selected**。
所以 P4 §15 那张"class+size 匹配后 Δlogit = −17.611"的对照**仍然循环**，我已在 P4 报告中自我标注为循环，此处再次确认。

---

## 9. CASE C Evidence Impact

| P4 结论 | 审计后的状态 |
|---|---|
| G86 在 rect 口径不被否证（86/86 ⊂ rect_miss） | **保留** |
| square→rect 几何切换不改变漏检成员（Jaccard 0.977） | **保留**（本轮新增，强化了 rect 与历史链的兼容性） |
| 坐标变换精确可逆 | **保留** |
| F0 由类分数（而非框生成）驱动 | **保留**（Case A） |
| "G86 的 GT 类 sigmoid = 5.5e-08 / 100% <1e-3" | **降级**为 **square-only 读数**；rect 侧 UNKNOWN |
| "G86 vs CONTROL 的轨迹差异是因果机制" | **降级**为**描述**（两组均为 outcome-selected） |
| "raw candidate absent 证明生成能力不足" | **退役**（是 score 阈值效应） |
| 对称分解 CASE C 的**数值** | 保留为**描述性分解**；不因本轮审计而失效，但也不因此获得因果地位 |

---

## 10. Final Status

```text
STATUS B — PARTIALLY VALID
```

逐条对应 §11 的判据：

| STATUS C 的触发条件 | 是否触发 |
|---|---|
| G86 无法在 rect pipeline 重建 | 部分 —— **必要半**（final miss）100% 重建；**充分半**（头部认证）无 rect artifact |
| membership 高度依赖跨 geometry artifact | ✗（漏检集对几何不敏感，Jaccard 0.977） |
| 核心变量定义发生改变 | **是** —— "certified" 只在 square 存在；rect 多一道 conf 门 |
| trajectory decomposition 建立在不可复现的 group 上 | 否（group 可复现且不被否证），但**是 outcome-selected** |

⇒ 不到 INVALID（G86 未被否证），也到不了 CLEAN（定义不等价、rect 侧关键读数 UNKNOWN）。

---

## 11. Evidence Retained

1. **G86 ⊂ rect 末输出漏检集，86/86 无例外**（`rect_miss` n=256）。
2. **漏检成员对 square→rect 几何不敏感**：`rect_miss ∩ sq_miss` Jaccard **0.977**。
3. **坐标变换精确**（往返误差 ~1e-13 px；存储精度 1e-4 px）。
4. **F0 是类分数阈值效应**（Case A，架构依据）⇒ P4 的"F0 不是框生成失败"这一**重新归因**成立。
5. G86 的组成（class / size / 图像分布）在 square 口径内自洽可复现。

## 12. Evidence Retired / Downgraded

1. **"GT 类 sigmoid 5.5e-08"** 不得再作为 **rect/D′ 现行 pipeline** 的读数引用 —— 它是 square 测量，rect 侧 UNKNOWN。
2. **G86 的 `certified_prefinal_loss` 定义**仅在 square 链条内有效，跨到 rect 时必须重新声明为"末输出漏检子集"。
3. **`raw candidate absent` 不得再作为"生成能力不足"的证据**（是 conf 阈值效应）。
4. **G86 vs CONTROL 的任何对比不得表述为因果**（两组都是 outcome-selected）；`CONTROL = detected=1` 这一点必须随结论一起声明。

## 13. What This Audit Does NOT Establish

- 不建立任何优化方向，也不排除任何优化方向。
- 不证明 G86 是一个"真群体"（它同时是 outcome-selected 与 cross-geometry 的产物，只是**未被否证**）。
- 不给出 rect 口径下的 GT 类分数量（**UNKNOWN**）。
- 不区分 Case A/C（`layers.npz` 未存 decoded box，**UNKNOWN**）。
- 不为 P4 的 CASE C 背书因果地位 —— 只确认其**数值描述**未被本轮审计推翻。
- 结论不适用于 small-GT 的 371 群体（本轮只审 G86 与 1320 medium）。
