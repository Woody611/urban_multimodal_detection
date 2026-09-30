# Training-side Classification Supervision Audit

**日期**：2026-09-29
**性质**：**READ-ONLY / NO-TRAINING / NO-FORWARD / NO-BACKWARD**。
0 training run / 0 `model(...)` / 0 `loss.forward` / 0 `backward()` / 0 `autograd.grad` / 0 hook+forward /
0 validation inference / 0 新 prediction TXT / 0 新 submission / 0 修改任何既有文件 / 0 重定义 G1-G2-G4 /
0 重跑已关闭的实验。
**脚本全文未 `import torch`**（只用 Python + numpy 读已有 JSON/JSON-labels，纯数学）。

---

## 1. EXECUTIVE VERDICT

```text
Primary supported mechanism:    A — supervision QUANTITY deficit（作为**已被测量的事实**）
                                  native-small n_pos 中位 4 vs medium/large 10（MWU p≈0，n=10105）
Secondary mechanism:            B — target STRENGTH deficit（**真实但幅度有限**）
                                  pos_target_max 0.8747 vs 0.9767（−10.4%）；
                                  同图等分基准 share 0.856 vs 1.109（−23%）
Not supported:                  C — background-anchor gradient dilution（**源码层面不成立**）
Still unresolved:               D — class imbalance（描述性，无一致关系）
                                E — spatial / feature support（A–D 未能闭合，但本轮无正面证据）

Training recommendation:        RED
```

```text
NO TRAINING GREEN LIGHT
```

**没有任何机制达到 A 且机制链完整。** 只有 A 达到 A（作为事实），但它**不构成**完整因果链
（为什么会少？因为小框内本来就只有 ~4 个 stride-8 anchor 中心 —— 这是 anchor 网格几何的**后果**，
不是任何监督设计的后果）。B 有真实但**不足以解释观测落差**的幅度。C 在源码层面被排除。
D/E 无判定证据。

---

## 2. HARD GATES

| Gate | Result | Evidence |
|---|---|---|
| no training | **PASS** | 脚本无 `train`/`optimizer`/`scheduler`/`loss.backward` 调用；无新 run 目录 |
| no forward | **PASS** | 脚本**未 `import torch`**，无任何模型/预测器/predictor 构造；无 hook 注册 |
| no backward | **PASS** | 无 `backward()` / `autograd` |
| no optimizer step | **PASS** | 无优化器构造 |
| no existing file modification | **PASS** | 以本轮开始时刻 `2026-09-29 16:16:01` 为界，全仓库扫描：**0** 个既有文件被写（`_mtime_scan.py`） |
| provenance intact | **PASS** | `git status` 前后逐行相同；`best.pt` `sha256sum -c` OK；8 个关键 artifact SHA256[:16] 前后一致（见 §18） |

```text
existing files changed   = 0
training runs launched   = 0
forward passes launched  = 0
backward passes launched = 0
optimizer steps          = 0
prediction TXT regenerated = 0
```

---

## 3. ACTUAL CODE PATH

```
GT label (cls 整数, 0-based)
 │
 ├─ `loss.py:508-509`  targets = cat((batch_idx, batch['cls'], batch['bboxes']), 1) → preprocess()
 │                     gt_labels, gt_bboxes = targets.split((1,4), 2)
 │                     ⇒ gt_labels = dataset 的原始类整数，无 ±1
 ▼
TAL  TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)   ← `loss.py:438`，**源码字面量**
 │
 ├─ `tal.py:120`      mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
 ├─ `tal.py:139-152`  get_box_metrics:
 │        ind[1] = gt_labels.squeeze(-1)
 │        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]   ← sigmoid 后的**同类**分
 │        overlaps[mask_gt]    = bbox_iou(gt, pd, xywh=False, CIoU=True).clamp_(0)
 │        align_metric = bbox_scores.pow(alpha) · overlaps.pow(beta)      ← cls^0.5 · CIoU^6
 ├─ `tal.py:123`      mask_topk = select_topk_candidates(align_metric, topk_mask=…)
 ├─ `tal.py:124`      mask_pos  = mask_topk · mask_in_gts · mask_gt
 ▼
target_scores  `tal.py:226-241` get_targets
 │        target_scores = zeros((b, h*w, nc))
 │        target_scores.scatter_(2, target_labels, 1)          ← **raw target = one-hot 1.0**
 │        target_scores = where(fg_mask>0, target_scores, 0)   ← 只留正样本
 ▼
归一化  `tal.py:111-116` _forward（**本轮核心**）
 │        align_metric *= mask_pos
 │        pos_align_metrics = align_metric.amax(-1, keepdim=True)        ← 每 GT 的 max align
 │        pos_overlaps      = (overlaps*mask_pos).amax(-1, keepdim=True) ← 每 GT 的 max CIoU
 │        norm_align_metric = (align_metric · pos_overlaps / (pos_align_metrics + eps)).amax(-2)
 │        target_scores = target_scores · norm_align_metric
 ▼
cls loss  `loss.py:523-535`
          target_scores_sum = max(target_scores.sum(), 1)     ← **全 batch × 全 anchor × 全类**
          self.bce = nn.BCEWithLogitsLoss(reduction='none')   ← `loss.py:415`
          loss_cls = self.bce(pred_scores, target_scores.to(dtype))
          if self.cls_pw is not None: …                       ← `loss.py:430`；D′ 无 `cls_pw` ⇒ **None**
          loss[1] = loss_cls.sum() / target_scores_sum
```

**由源码推出的两条恒等式**（均在 §5 数值验证）：

```text
(I1) target[a] = 1.0 × align[a, g_a] · maxCIoU[g_a] / maxAlign[g_a]        （g_a = a 的分配 GT）
(I2) ⇒ **每个 GT 的峰值分类 target = 它最好的正样本框与它的 CIoU** = maxCIoU[g]
```

`_v5_replay` n=3544 实测 `|pos_target_max − best_assign_iou|`：**median = 0.000e+00**，
**3496/3544（98.6%）在 1e-5 内精确**。
48 条例外（1.4%）源于插桩取的是 `get_pos_mask` 的 **mask_pos（冲突消解前）**，
被 `select_highest_overlaps` 改判走的 anchor 上 `ts_g[pos]` 反映的是**同图同类另一个 GT** 的目标
—— 是**插桩侧测量瑕疵**，不是 loss 公式性质。

---

## 4. SUPERVISION TABLE

### 4.1 主表 —— **native 分桶**（`_sge_supervision.json` 的 `native_area`，n=10105）

| scale | n GT | **median n_pos** | median best_align | **median pos_target_max** | layer HR% |
|---|---:|---:|---:|---:|---:|
| **small** | 1325 | **4** | 0.3570 | **0.8747** | 73.9% |
| **medium** | 4647 | **10** | 0.6879 | **0.9552** | 93.8% |
| **large** | 4133 | **10** | 0.8263 | **0.9767** | 100.0% |

`n_pos`：small vs medium / large 的 MWU **p ≈ 0**（n=1325 vs 4647 / 4133）。

### 4.2 补充表 —— **aug 分桶**（`_v5_replay.json`，⚠ 混入增强，见 §4.3）

| aug-bucket | n GT | median n_pos | mean_align | sum_align | pos_target_mean | **target_mass** |
|---|---:|---:|---:|---:|---:|---:|
| small | 1179 | 9 | 0.4466 | 3.5801 | 0.7546 | **6.1893** |
| medium | 1736 | 10 | 0.6772 | 6.7718 | 0.8749 | 8.7490 |
| large | 629 | 10 | 0.8114 | 8.1141 | 0.9315 | 9.3148 |

（`sum_align = n_pos × mean_align` 为导出量；`target_mass = n_pos × pos_target_mean`，
见 §4.4 为何**不能**用 `target_sum_all`。）

### 4.3 ⚠ aug-bucket 与 native-bucket 的**分歧必须点明**

```text
native 分桶：small n_pos 中位 = 4   （vs large 10）  ⇒ 数量赤字明显
aug    分桶：small n_pos 中位 = 9   （vs large 10）  ⇒ 数量赤字几乎消失
```

真因：**mosaic / RandomPerspective 的放大** —— native-small 目标增强后进入 aug-medium/large。
⇒ **任何用 aug 尺度做的 small/large 比较都会系统性低估数量赤字。**
本报告的主表（§4.1）一律用 **native**。

### 4.4 ⚠ 一处 artifact 语义陷阱（协议 §2 警告的那类）

`_v5_replay.py:133,145` 的 `target_sum_all = target_scores[0, :, cid].sum()`
**不是 per-GT 量**，而是 **`cid` 类通道在全部 anchor 上的和** —— 同图同类 GT 会互相污染。

```text
实测反例：n_pos=10, pos_target_mean=0.8911, target_sum_all=277.8157
  若它是 per-GT 量应 ≈ n_pos×mean = 8.9112；实测 277.8157 ⇒ 不是 per-GT 量
```

⇒ 本报告**不使用** `target_sum_all` 作为 per-GT 质量；改用 `n_pos × pos_target_mean`
（= `ts_g[pos]` 的和，`pos` 是该 GT 自己的正样本）✓。
但它**可以**用于重建 batch 分母（§6）。

### 4.5 协议 §4 的 8 项可得性

| # | 量 | 来源 | 状态 |
|---:|---|---|---|
| 1 | positive anchor 数 `n_pos` | `_sge_supervision` / `_v5_replay` | AVAILABLE |
| 2 | raw TAL alignment（逐 anchor） | 只落盘 max/mean，逐 anchor 未落盘 | **PARTIAL** |
| 3 | max alignment | `best_align` | AVAILABLE |
| 4 | sum alignment | `n_pos × mean_align`（导出） | DERIVED |
| 5 | normalized target score | `pos_target_max` / `pos_target_mean`（归一化后 ✓） | AVAILABLE |
| 6 | target score sum（per-GT） | `target_sum_all` **不是 per-GT**；改用 `n_pos×pos_target_mean` | DERIVED |
| 7 | target score mean（per-GT） | `pos_target_mean` | AVAILABLE |
| 8 | target score max（per-GT） | `pos_target_max`（98.6% 精确 = `best_assign_iou`） | AVAILABLE |

**无一项需要新 forward。** 唯一 PARTIAL 的 #2 也不影响本轮任一结论。

---

## 5. NORMALIZATION ANALYSIS

```text
raw alignment  →  normalization  →  final target
   align[a]         × maxCIoU[g]/maxAlign[g]        target[a]
   （cls^0.5·CIoU^6）                                （初值 1.0）
```

**⇒ small 是否因为 normalization 获得了额外的 supervision deficit？**

**否 —— normalization 本身不是 small 特有的惩罚。** 它把 raw one-hot `1.0` 统一拉到
**该 GT 自己的 CIoU 水平**（恒等式 I2）。它对所有尺度**同一规则**作用，不区分大小。

**但**它有一个真实的副作用：**把分类监督强度与几何质量绑死**。
因此 §4.1 的 `pos_target_max` 差异（0.8747 / 0.9552 / 0.9767）本质上是
**CIoU 差异的投影**，而不是归一化对 small 的额外惩罚。

幅度核算：

```text
pos_target_max 相对赤字（small vs large） = (0.9767 − 0.8747)/0.9767 = 10.4%
per-GT target_mass 相对赤字（aug 桶）      = (9.3148 − 6.1893)/9.3148 = 33.6%
同图等分基准 share（见 §6）                = 0.856 vs 1.109 = 22.8%
```

⇒ **B 的赤字存在、可测，但幅度在 10%–34% 量级**；而 §8 显示实际响应的落差是
**12.6 个 logit（约 5 个数量级概率）**。**幅度对不上。**

---

## 6. BACKGROUND DILUTION ANALYSIS（+ batch 分母重建 / per-GT share）

### 6.1 源码判定

```text
loss[1] = Σ_{(b,a,c)} BCE(logit, target) / Σ_{(b,a,c)} target      （loss.py:535 / :523）
```

- 分母**只含 target 非零项**（背景项 target=0 ⇒ 不进分母）；
- 每个正样本项的梯度权重是**共享的 1/D**，**与同图 anchor 总数无关**；
- ⇒ **「背景 anchor 越多 ⇒ 某 GT 的梯度被按比例稀释」在源码层面不成立。**

背景项的真实作用是**叠加一个把全类 logit 往下压的梯度**（balance 问题），**不是除法稀释**。

结构性对照（源码计数）：单图 anchor = 160²+80²+40² = **33600**；class-channel 项 = **403200**；
`_v5_replay` 每图正样本数中位 = **107** ⇒ **非零 target 项 : 全部通道项 ≈ 1 : 3768**。
（每个正样本贡献**恰好 1** 个非零 target；其余 11 个同类通道项 target=0。）

### 6.2 batch 分母重建与 share

按 `(mosaic_sample, cls)` 去重后求和 = `target_scores_sum`（可重建样本 249 个，D 中位 **86.17**）。
per-GT `share = target_mass / D`（**分母归一化后的 target 质量占比**）：

| aug-bucket | n | share 中位 | share p90 | mass 中位 |
|---|---:|---:|---:|---:|
| small | 1179 | 0.02964 | 0.07646 | 6.1893 |
| medium | 1736 | 0.05952 | 0.15016 | 8.7490 |
| large | 629 | 0.07125 | 0.22498 | 9.3148 |

同图**等分基准**归一（`share × n_GT_in_image`，1.0 = 恰好拿到等分份额）：

| aug-bucket | n | share×n_GT 中位 | p10 | p90 |
|---|---:|---:|---:|---:|
| small | 1179 | **0.8564** | 0.2387 | 1.2432 |
| medium | 1736 | 1.0809 | 0.9298 | 1.3405 |
| large | 629 | **1.1086** | 0.9793 | 1.3612 |

⇒ small 拿到等分份额的 **0.86×**，large 1.11× ⇒ **相对赤字约 23%**。
⚠ 这是**target 质量份额**，**不是梯度测量**。
⚠ 且它本质上是 §5 的 B 的另一种呈现（share ∝ mass），**不是独立的第三条机制**。

### 6.3 梯度层面的结论

```text
BACKGROUND_ANCHOR_GRADIENT_DILUTION = NOT_PROVEN
```

**没有 gradient artifact ⇒ 不能声称 small 收到的梯度更弱。**
本轮结论严格限于「**target 数值**层面」。

---

## 7. CLASS IMBALANCE ANALYSIS

train 直接统计（1600 图，GT 11,570，native-small 1,392 = **12.0%**）：

| class | total GT | small GT | small ratio | small imgs | total share |
|---|---:|---:|---:|---:|---:|
| person | 4426 | 703 | 15.9% | 193 | 38.3% |
| boat | 107 | 17 | 15.9% | 7 | 0.9% |
| animal | 2328 | 107 | 4.6% | 40 | 20.1% |
| seat | 514 | 30 | 5.8% | 30 | 4.4% |
| sign | 659 | 124 | 18.8% | 97 | 5.7% |
| bicycle | 550 | 79 | 14.4% | 25 | 4.8% |
| car | 1283 | 97 | 7.6% | 65 | 11.1% |
| ball | 70 | 22 | 31.4% | 17 | 0.6% |
| light | 1211 | 56 | 4.6% | 25 | 10.5% |
| garbage_can | 225 | 39 | 17.3% | 25 | 1.9% |
| uav | 174 | 118 | **67.8%** | 97 | 1.5% |
| tricycle | 23 | 0 | 0.0% | 0 | 0.2% |

**结论（克制）**：
- `cls_pw = None`（`configs/train_rgbid_sepstem_clahe.yaml` 无该键，`loss.py:430`）⇒ **类别加权未启用**；
- 因此在标准 BCE 下，稀有类**正样本项更少**而**负样本项数与其它类相同** ⇒ 结构上更受负项支配 —— 这是**结构性事实**；
- **但**两条关系检验都不支持「频率 → 监督 → 响应」的链条：
  - Spearman(train 频率, train-side small 的 per-GT mass) = **−0.0545, p=0.881, n=10 类**
  - Spearman(train 频率, val GT 类 logit 中位) = **−0.2455, p=0.467, n=11 类**
- ⇒ **不做 12 类的显著性解释**（协议 §12）。**D = 判定不足。**

---

## 8. G1 / G2 / G4 INTERPRETATION

### 8.1 硬边界（协议 §10）

```text
G1 / G2 / G4 是 **validation** GT（any-class IoU==0 / matched control / small success）。
train-side artifact（`_sge_supervision`、`_v5_replay`）是 **train** GT，其中**没有** G1/G2/G4 的身份。
```

⇒ **本轮不写、也不能写「G1 在训练时收到了 X」。** 下面一切均为 **train-side population analysis**。

### 8.2 并置对照（population 级，非配对）

| population | n | median | p10 | p90 |
|---|---:|---:|---:|---:|
| **train** native-small `pos_target_max` | 1325 | **0.8747** | 0.5549 | 0.9616 |
| val G1 `best_assign_iou`（= target_max 的预测值，恒等式 I2） | 38 | 0.6278 | 0.2393 | 0.8172 |
| val G1 实测 GT-class sigmoid（`pos_prob`） | 38 | **0.000003** | 1.02e-07 | 2.89e-04 |

⇒ **train 侧监督被「下达」了（target 峰值中位 0.875），val G1 的响应却是 3e-6。**
但这是**两个不同 population 的并置**，**不构成**「G1 训练时监督不足」的证据。

### 8.3 ★ 判别性检查（§13b）

用 val 全部 2847 GT（自建 `native_area` + 冻结 TXT 的纯 IoU 检出状态）的
`logit_decomp`（GT 中心 cell 的 `W_c·h + b_c`），按 native × 检出状态分层：

| native | detected | n | GT 类 logit 中位 | sigmoid | train 侧 pos_target_max 中位 |
|---|---|---:|---:|---:|---:|
| **small** | True | 223 | **+0.4873** | 0.6195 | 0.8747 |
| **small** | False | 148 | **−12.1473** | 0.0000 | 0.8747 |
| **medium** | True | 1070 | **+0.8828** | 0.7074 | 0.9552 |
| **medium** | False | 250 | **−14.1531** | 0.0000 | 0.9552 |
| large | True | 996 | −17.4932 | 0.0000 | 0.9767 |
| large | False | 97 | −16.9436 | 0.0000 | 0.9767 |

```text
检出−未检出 落差：small 12.63 logit   medium 15.04 logit
尺度间落差（同一状态内）：检出的 small→medium 仅 0.40 logit；未检出的仅 2.01 logit
```

**⚠ 必须声明的测量层级范围（本轮由实测异常反查得到）**：

```text
`_v7_extract.py:52`  last = model.model[-1].cv3[0][-1]
⇒ `logit_decomp` 是 **stride-8（P3）那一条 class 分支**的 logit，不是「模型对任意尺度的类别响应」。
实测：输入 √area ≥ 96 的 670 个 GT，logit 中位 −17.22、p90 = −14.64、**无一 > 0**；
而这些目标显然被检出（大目标 AP@0.50 ≈ 0.79）。抽检 3 例 v6 与 v7 一致（≈ −15…−17）。
⇒ 真因：**大目标由 stride-16/32 分支负责**，P3 本来就不为大目标输出。**不是 bug，是层级范围。**
⇒ **large 一行在本表不可解释**，只作「该测量不适用于 large」的证据保留。
⇒ **small 是本测量的正确层级**（stride-8 就是小目标的分支）⇒ small 的结论有效。
```

### 8.4 分布形状（§13c）与未检出率（§13d）

native-small（剔除全幅框）n=371 的 logit 直方图：

```text
[−40,−8) 31.0%   [−8,−4) 11.1%   [−4,−2) 7.0%   [−2,0) 12.9%   [0,1) 15.1%   [1,3) 22.9%
```

**⚠ 我上一版把这个形状写成「两态、中间为空」，是过度断言，已更正。**
实测中间区间 (−8, 0) 占 **31.0%** ⇒ **分布是宽的、两峰但没有空档**；
`+0.49` 与 `−12.15` 是**分组中位数**，**不是**总体上的两态分离。
**不得据此声称「存在一个离散的坏态」。** 可支持的弱陈述：small 的响应分布比 medium **更左移且更宽**。

未检出率：small **39.6%** / medium **18.8%** / large **8.8%**（单调下降）。

### 8.5 由此得到的**两半分开**的陈述

```text
(a) 尺度影响「**有多少个**落在低响应区」        → 未检出率 39.6% / 18.8% / 8.8%（单调）
(b) 尺度几乎不影响「**低响应区本身有多低**」    → 未检出态 small −12.15 vs medium −14.15（≈2 logit）
```

⇒ **尺度渐变的监督赤字（A/B）与 (b) 对不上**（幅度 10–34% vs 12.6 logit），
但**可以**解释 (a)。**这两半必须分开陈述，不能合并成一句「监督不足导致漏检」。**

---

## 9. MECHANISM MATRIX

| Mechanism | Evidence | Verdict |
|---|---|---|
| **A** supervision quantity | native-small `n_pos` 中位 **4** vs medium/large **10**；两个独立 artifact（`_sge_supervision` n=10105 / `_v5_replay` n=3544）同向；MWU **p≈0**。**但** 该数字是 **anchor 网格几何的后果**（小框内本来只有 ~4 个 stride-8 中心），不是监督设计的后果；其**因果后果**未被本轮证明 | **A — Strongly supported**（作为**事实**）<br>⚠ 因果链**不完整** |
| **B** target strength | raw target 恒为 **1.0**（源码）；归一化后 `pos_target_max` **0.8747 / 0.9552 / 0.9767**（−10.4%）；per-GT mass −33.6%；同图等分 share **0.856 vs 1.109**（−22.8%）。**赤字真实存在**，但幅度**远小于**实际响应落差（12.6 logit / ~5 个数量级） | **B — Partially supported** |
| **C** gradient dilution | 源码：分母只含 target 非零项，每项权重是共享 1/D，**不按 anchor 数除**；背景项的作用是叠加压制梯度而非除法稀释。**梯度幅度层面无 artifact ⇒ NOT PROVEN** | **C — Not supported**（所述机制）<br>梯度幅度 = **NOT PROVEN** |
| **D** class imbalance | 频率表已给（§7）；`cls_pw=None` ⇒ 无类别加权（结构性事实已记录）；但 ρ(频率, small mass)=−0.05 (p=0.88, n=10)、ρ(频率, val 响应)=−0.25 (p=0.47, n=11)，**无一支持链条** | **D — Insufficient evidence** |
| **E** spatial / feature support | A–D 组合无法闭合 12.6 logit 的落差（见 §8.5）；但本轮**没有**任何正面证据指向某个具体的 E 机制（本轮不 forward）。**按协议 §1：不因为 A–D 不成立就宣布 E 为真** | **D — Insufficient evidence** |

---

## 10. WHAT THIS AUDIT RULES OUT

1. **「归一化对 small 有额外惩罚」** —— **排除**。归一化对所有尺度同一规则，把 target 统一拉到该 GT 的 CIoU 水平（恒等式 I2，98.6% 数值精确）。
2. **「背景 anchor 数量按比例稀释某 GT 的梯度」** —— **源码层面排除**。`loss.py:523/535` 的分母只含非零 target，不含 anchor 数。
3. **「small 的分类监督被稀释到近乎为零」** —— **排除**。native-small 的 `pos_target_max` 中位 **0.8747**，per-GT mass 中位 **6.19**，同图等分 share **0.856** —— 都是**正常量级**，不是近零。
4. **「类别加权缺失导致某类被系统性压制」** —— **未获支持**（`cls_pw=None` 是事实，两条关系检验都无一致关系）。
5. **「`target_sum_all` 是 per-GT 目标质量」** —— **排除**（语义陷阱，实测反例 277.8 vs 8.91）。
6. **「`logit_decomp` 可代表任意尺度的类别响应」** —— **排除**（它是 stride-8 分支；对大目标系统性无效）。

---

## 11. WHAT REMAINS UNKNOWN

1. **训练期 vs 验证期的响应落差从何而来** —— train-side native-small 的 target 峰值中位 **0.875**，
   而 val G1 的实测响应 **3e-6**。监督数值正常，响应不形成。**这是本轮留下的核心落差，未解释。**
2. **尺度如何决定「有多少个掉进低响应区」**（39.6% → 8.8%）。已知监督侧只有 10–34% 的赤字，
   不足以解释 39.6% vs 8.8% 的**检出率**差异（见 §8.5 的两半拆分）。
3. **D 的因果性**：稀有类在 BCE 下受负项支配是结构事实，但它是否转化为 val 响应差异，**n=11 类无法判定**。
4. **E 的任何具体机制**：本轮**没有**正面证据（既不能确认也不能否认）。
5. **梯度层面**：small 是否真的收到更弱的分类梯度 —— **NOT PROVEN**（无 gradient artifact）。

---

## 12. NEXT EXPERIMENT DECISION

```text
RED
```

**理由**：没有任何机制达到 A 且机制链完整（见 §9）。协议 §12 允许 GREEN 的条件
（"已有充分机制证据，可以设计单变量训练实验"）**不满足**。

```text
不允许训练；继续机制审查。
```

### 12.1 唯一能改变结论的测量（**本轮被协议禁止，故未做**）

```text
最小下一次 audit（如果将来被授权）：
   在 **train 侧** 对 native-small GT 测「训练过程中 GT 类响应的实际分布」，
   与 **val 侧** native-small（G1 / G4）的对应分布对齐。

需要的唯一新增 artifact：
   一次 train-split forward + 按 native 尺度分桶的 GT 类 logit dump
   （即把 `_v6_domains` 的 V1 域做法**原样**用在 T1 域上，并保留 native 尺度）。

为什么它能区分两个 competing hypotheses：
   H1（监督不足）：train 侧 native-small 的 GT 类响应也应当**随尺度**降低，
                  且其降低幅度应与 A/B 的赤字（10–34%）同量级。
   H2（特征支撑不足）：train 侧 native-small 的响应应当**接近 medium**
                  （因为 train 上模型拟合良好，监督数值也正常），
                  而落差只出现在 val —— 那么落差只能来自**泛化/表征**，不是监督。
   ⇒ 只要拿到 train 侧 native-small 的 GT 类 logit 分布，H1 与 H2 就是**可分辨的**。
```

⚠ 但该测量需要 **forward**，本轮协议明令禁止 ⇒ **未执行**，且**不因结论需要就违反**。
**也不建议**把它作为下一步的理由去开训练 —— 即使 H1 成立，A/B 的幅度（10–34%）也解释不了
12.6 logit 的落差（§8.5）。

---

## 13. 本轮我自己的更正（如实记录）

| # | 我写成 | 实测 | 更正 |
|---|---|---|---|
| 1 | §13c「分布 = 两态、中间为空」 | 中间区间 (−8,0) 占 **31.0%** | **过度断言**，改为「宽分布、两峰无空档；+0.49/−12.15 只是分组中位数」 |
| 2 | §7「每图正样本数中位 ≈ 0」 | `mosaic_sample` 不连续（0…1494 步长 6）⇒ 我用 `range(30)` 分组错了 | 改为按**实际出现的** `mosaic_sample` 分组，得中位 **107** |
| 3 | `target_sum_all` 一度被当作 per-GT 质量 | 它是**类通道**总和 | 见 §4.4，改用 `n_pos × pos_target_mean` |
| 4 | 把 native-large 的低 logit 当成模型性质 | 实为 `logit_decomp` 只在 **stride-8 分支** | 见 §8.3，并把 large 行标为不可解释 |
| 5 | 首版 val GT 表只覆盖 native-small（误用 `_v2_records`） | 需全部 GT | 改为从 val label + 冻结 TXT 纯 IoU 自建（n=2847） |

**没有任何一项更正改变了 A/B/C/D/E 的判定方向**（A 仍为事实级支持，C 仍被源码排除，B 仍为幅度有限）。

---

## 14. Provenance

```text
TRAINING        = NO
FORWARD         = NO
BACKWARD        = NO
OPTIMIZER_STEP  = NO

EXISTING_FILES_CHANGED         = 0
EXISTING_CHECKPOINTS_CHANGED   = 0
EXISTING_PREDICTIONS_CHANGED   = 0
EXISTING_DIAGNOSTICS_CHANGED   = 0

GIT_STATUS_BEFORE = 73 行（72 个 status 条目 + 1 行 HEAD）
GIT_STATUS_AFTER  = 73 行，逐行相同
GIT_HEAD_BEFORE   = f659609c28f6b4f500212197d5229550a36d12d8
GIT_HEAD_AFTER    = f659609c28f6b4f500212197d5229550a36d12d8
```

**「无既有文件被修改」的验证口径（可复核）**：
以本轮开始时刻 `2026-09-29 16:16:01`（`_git_before.txt` 的 mtime）为界，
全仓库（排除 `.git` / `__pycache__` / 本轮自身目录）扫描 mtime 晚于该时刻的文件 ⇒
**命中 0 个**（`_mtime_scan.py`）。

### 读取对象 SHA256[:16]

| 对象 | SHA256[:16] |
|---|---|
| `runs/…_sepstem_clahe/weights/best.pt` | `1cae45f75693f541` |
| `diagnostic/small_gt_exposure/_sge_supervision.json` | `7dc2e3c69a5f368b` |
| `diagnostic/small_object_train_replay/_v5_replay.json` | `da5dfbd6dcdb287d` |
| `diagnostic/small_gt_exposure/_sge_native_meta.json` | `29e3df9526466ec1` |
| `diagnostic/small_object_domains/_v6_domains.json` | `3505d9160c47aae8` |
| `diagnostic/small_object_cause_v2/_v2_records.json` | `da015742973d1efa` |
| `diagnostic/p3_feature_space/_v7_vectors.json` | `977593ae00126100` |
| `ultralytics/utils/tal.py` | `aae7e8ac438f00cd` |
| `ultralytics/utils/loss.py` | `0f092cf22a372f6d` |
| `ultralytics/nn/modules/head.py` | `2fa868a4464ce88b` |
| `configs/train_rgbid_sepstem_clahe.yaml` | `a4e329cfc3d22020` |

（全部与本轮开始时打印值一致；`sha256sum -c` 对 `best.pt` 通过。）

### NEW_FILES（全部位于 `diagnostic/train_side_cls_supervision/`）

```text
REPORT.md           本报告
_analyze.py         审计脚本（未 import torch；只写本目录）
AUDIT.log           完整运行日志
_tables.json        结构化结果
_mtime_scan.py      全仓库 mtime 扫描（用于「0 既有文件被改」的可复核验证）
_mtime_scan.txt     上述扫描的输出（命中 0）
_ckpt_before.txt    checkpoint 的 sha256sum 基线
_git_before.txt     git status + HEAD（before）
_git_after.txt      git status + HEAD（after）
_run.txt            运行输出副本
```

---

**本轮结束，停止。不训练、不设计干预、不通过 forward 补证据、不自行启动下一轮。**
