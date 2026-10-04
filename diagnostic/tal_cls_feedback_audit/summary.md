STATUS = PASS

# TAL Classification-Score Feedback — Zero-GPU Audit

只读。0 GPU / 0 training / 0 inference / 0 forward / 0 backward / 0 evaluator /
0 源码·YAML·config·checkpoint 修改 / 0 覆盖既有 diagnostic 目录。CHECK C1–C7 **全 PASS**。

```
PROVENANCE_STATUS = PASS
```
`tal.py aae7e8ac…` / `loss.py 0f092cf2…` / `train cfg a4e329cf…` / `best.pt 1cae45f7…` 四项与 provenance 记录逐位吻合。

---

## ★ 一个必须首先记录的字段语义陷阱（本轮 C1 发现，决定了全部分析的重做）

`tal.py:110` 的 `align_metric *= mask_pos` 是 **in-place**，探针持有同一 tensor
⇒ dump 出的 **`cand_align` 是 post-mask_pos 版本**（非正样本被置 0），**不能用于排序**。

```
recorded cand_align == 0 的比例 = 95.29%
recorded==0 而 cls^0.5·CIoU^6 > 0 的比例 = 58.38%   ← 原始 alignment>0 但非正样本 ⇒ 被置 0
recorded > 0 的比例 = 4.71%                        ← 正是正样本集（mask_pos）
```
⇒ 排序一律改用重算的 `align_raw = cls^0.5 · CIoU^6`；
**而 `cand_align > 0` 恰好给出观测到的正样本集 `mask_pos`** ⇒ **top-k 是直读，不是重建**。
**重建校验**：重算排序 top-10 **100.00% 覆盖**观测正样本（89,735 / 89,735）；每 GT 正样本数 median 10 / max 10（= topk 上限）。

---

## Q1 — TAL 是否实际使用 GT-class cls score？

**是（`PROVEN`）**。`tal.py:147` 取 **GT 类那一列**；`loss.py:522` 传入 `pred_scores.detach().sigmoid()`
（sigmoid 概率、已 detach）；`tal.py:150` `align_metric = bbox_scores^0.5 · overlaps^6`；几何用 CIoU（`tal.py:155`）。

## Q2 — 在当前 alpha=0.5, beta=6 下，cls 对 alignment 的实际敏感度

```
align = cls^0.5 · CIoU^6        ⇒  cls 的指数只有 0.5，CIoU 的指数是 6
等 alignment 边界：cls_B/cls_A = (iou_A/iou_B)^12
  iou 0.5 → 0.8 :  B 只需 A 的 0.36% 的 cls 即可打平   （= 282× 的 cls 优势）
  iou 0.5 → 0.9 :  只需 0.028%
```
⇒ **几何主导**；但 cls 有一个**乘性地板**：`cls < 0.001 ⇒ align ≤ 0.0316`（无论 CIoU 多好）。
在真实池里该地板是**决定性**的：cls = 1.64e-07 的候选要追平 cls = 0.87 的正样本，
需要 `ciou_A/ciou_B > 3.63`，而 CIoU ≤ 1 ⇒ **在合法范围内不可能**。

## Q3 — 真实 low-cls + high-IoU candidates 是否在 top-k 中系统性吃亏？

**是（`SUPPORTED`）**，1,905,266 个真实候选 / 10,105 GT，关键组 `CIoU≥0.50 ∧ cls<0.001` = **217,025 个**（11.39%），
涉及 **7,668 个 GT**。**决定性分解**（同批候选，只换排名依据）：

| 排名依据 | 池内 median rank | ≤ top-10 比例 |
|---|---:|---:|
| **CIoU（纯几何）** | 29 | **11.09%** |
| **align（几何 × cls）** | 30 | **0.14%** |

⇒ **两个独立事实**：① 88.9% 的该组候选**连几何 top-10 都进不去** ⇒ 对多数候选 cls **不是**原因；
② 在**几何合格**的 11.09% 中，cls 项把它们压到 **0.14%** ⇒ **相对几何基线 79× 的额外排除**，97.7% 的 GT 一个都没进。

## Q4 — 把 cls 从 1e-6 提升到 1e-3 / 1e-2，assignment 是否改变？

**改变很小（`SUPPORTED`，反事实为条件性算术）**：

| 干预 | 进入 top-10 |
|---|---:|
| 原始值（观测） | 0.14% |
| 只把 cls → 1e-3 | 0.18% |
| 只把 cls → 1e-2 | 0.33% |
| 只把 cls → 0.1 | **0.83%** |
| 只把 CIoU → 正样本中位 0.953（cls 不变） | 0.21% |
| **两者同时**（cls→0.1 且 CIoU→0.953） | **1.69%** |

⇒ 单改 cls（10^6× 提升）只回收 0.83%；**即便 cls 与几何同时改到正样本水平也只有 1.69%**
⇒ 该组的低分**同时**由"几何略差"（正样本 CIoU 中位 0.953 vs 该组 0.808）与"cls 极低"两侧贡献，
**没有单一决定性变量**。

## Q5 — 是否有证据支持「低 cls → TAL disadvantage → 更弱 supervision」的反馈链？

**分三段，只有前两段有证据：**

| 环节 | 状态 | 依据 |
|---|---|---|
| ① 低 cls 在冻结快照下**机械地**被 TAL 排除 | `PROVEN`（条件于当前分数） | 11.09% → 0.14%；cls 地板 0.0316 |
| ② 被排除 = 拿不到 target（不进 `mask_pos` 就没有分类 target） | `PROVEN`（代码） | `tal.py:128` / `loss.py:527` |
| ③ **这个回路在训练中实际造成了 collapse** | **`NOT_OBSERVABLE`** | 见下 |

**③ 为什么不可观测（本轮最重要的限制）**：冻结的 `cand_cls` 是**训练后**的分数；正样本 cls 中位 **0.8697**
而 `target = maxCIoU`（中位 0.72）**正是把正样本 cls 往上推的产物**；非正样本从未获得监督，
cls 从未被提高。⇒「低 cls 被排除」与「被排除所以低 cls」**是同一份冻结数据的两种读法，方向不可分离**。
需要 **per-epoch candidate-level trajectory**（score → assignment → target → score），冻结产物中不存在。

```
TRAINING_CAUSALITY = NOT_OBSERVABLE
```

### §13 反证检查（必做，且结果双向）

`target_peak = maxCIoU over positives`，**cls 不进入 target 幅度**（`tal.py:110-116`；
`pos_overlaps = (overlaps*mask_pos).amax(-1)` 只用 CIoU）。
- 这**削弱**「cls feedback → supervision collapse」：一旦入选，监督强度**只看几何、不看 cls**。
- 但同时也意味着：**入选与不入选之间的差异完全由 cls 决定** ⇒ 反馈**只能通过「入选/不入选」这道闸门**起作用，
  **不能**通过 target 幅度起作用。

## Q6 — 这个机制是否足以成为下一轮训练实验的依据？

**不足以（见 FINAL MECHANISM VERDICT 的 `NEXT_EXPERIMENT_JUSTIFIED = NO`）**：
反馈的**方向性**不可观测，任何针对 TAL 的训练改动在当前测量集下**不可证伪**
（要么做一个带 per-epoch candidate 插桩的诊断性训练，要么先解决方向性问题 —— 两者都不是"改 gain"能回答的）。
本轮**不启动任何训练**。

---

## ==================================================
## FINAL MECHANISM VERDICT
## ==================================================

```
ASSIGNMENT_BIAS:
    SUPPORTED
    （Level 1 数学耦合 PROVEN；Level 2 地板效应 PROVEN；Level 3 真实候选 top-k 后果已量化：
      217,025 个候选 / 7,668 个 GT 上，几何基线 11.09% → 几何×cls 0.14% = 79× 额外排除）

TRAINING_CAUSALITY:
    NOT_OBSERVABLE
    （冻结 cls 是训练后产物；正样本 cls 0.87 本身由 target≈maxCIoU 训练而来 ⇒ 方向不可分离；
      需 per-epoch candidate trajectory，冻结产物中不存在）

NEXT_EXPERIMENT_JUSTIFIED:
    NO

REASON:
    1. 上一轮已证明分类监督**存在且强**（target 中位 0.723、|grad| 0.723、86/86 全部进入 loss）
       ⇒ 不是"监督缺失/微弱的 TAL 问题"。
    2. 本轮证明 TAL 的 cls 耦合在**冻结快照**下确实造成 79× 的额外排除（Level 3 已量化），
       但**方向不可观测**：正样本的高 cls 本身就是被 target 训练出来的。
    3. 该组的低分是「几何略差 + cls 极低」两侧共同贡献，反事实下即使两者都改到正样本水平也只有 1.69% 进入 top-k
       ⇒ 不存在"改一个变量即可验证"的单变量干预。
    4. 因此任何针对 TAL/assignment 的训练改动都**不可证伪**，不构成合格的下一轮实验依据。

GPU USED:
    0

TRAINING RUNS:
    0

INFERENCE RUNS:
    0

SOURCE MODIFICATIONS:
    0
==================================================
```

---

## 生成文件（新目录，未覆盖既有 diagnostic 目录）

```
diagnostic/tal_cls_feedback_audit/
    input_manifest.txt           输入 SHA + npz 字段语义（含 post-mask_pos 说明）
    provenance.md                PROVENANCE_STATUS = PASS
    tal_equation_audit.md        alpha/beta/topk/变量名/行号 + 字段语义陷阱
    cls_iou_sensitivity.csv      §4 解析敏感度表
    equal_alignment_boundary.csv §5 等 alignment 边界（^12）
    candidate_quadrants.csv      §7 四象限（逐 GT）
    candidate_counterfactual.csv §9/§10 反事实
    rank_decomposition.csv       ★ 决定性分解：CIoU 排名 vs align 排名
    per_gt_feedback.csv          §11 86 条（val GT，仅条件性算术）
    mechanism_analysis.md        Level 1–4 分层 + §13 反证 + §15 逻辑错误检查
    summary.md                   本文件
    consistency_checks.txt       C1–C7
    summary.json / _run.log / _audit.py / _rank_decomp.py / _rank_decomp.json
```

## 异常与修正（不省略）

1. **脚本连写三轮出现 shell/f-string 引号问题**（`<<'PYEOF'` heredoc 两次被 shell 报
   `unexpected EOF while looking for matching '`，以及一次 f-string 花括号未转义）。
   §17 明确点名过这两类，我仍然重犯了；最终改为**用 Write 直接落盘脚本文件再执行**，问题消失。
2. **cand_align 字段语义错误在第一次运行中被 C1 抓到**：我最初把 `cand_align` 当作原始 alignment 做排序，
   得到 `top-10 命中率 0.23%`、`pool maxCIoU 进入 top-10 49.9%` 等**看似支持 assignment bias 的结论**。
   该结论**全部作废并已用重算值重做**。**若没有 C1，本轮会给出一个建立在校验和上的错误结论。**
3. **单变量反事实已按 §原值保留**：npz 只读，全部运算在副本上进行；C4 已验证。
4. **86 条是 val GT**，本轮**未**把它们当作 training candidate（§11）；`per_gt_feedback.csv` 中
   `sample_scope = VALIDATION_GT_NOT_TRAINING_CANDIDATE`，其中 alignment 倍数仅为条件性算术。
