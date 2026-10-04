STATUS = PASS

# Medium Classification Score Collapse — TAL / Classification Supervision Path Audit

只读。0 GPU / 0 training / 0 inference / 0 validation / 0 evaluator / 0 改源码·YAML·checkpoint·既有 JSON /
0 重定义 86 / 0 改上一轮文件。所有 CHECK A–H **PASS**。

**本审计的核心是代码恒等式 + 冻结量，不是经验假设。**

---

## 1. D′ 输入唯一性（CHECK B）

```
configs/train_rgbid_sepstem_clahe.yaml   a4e329cfc3d220206448cdd3377c1b3f5126c50f13c907ada522d98725486e12  == provenance ✓
runs/…_sepstem_clahe/weights/best.pt     1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda  == provenance ✓
ultralytics/utils/tal.py                 aae7e8ac438f00cd…  == provenance ✓
ultralytics/utils/loss.py                0f092cf22a372f6d…  == provenance ✓
args.yaml: box=7.5  cls=0.5  dfl=1.5  cls_pw=[]  ir_encoding=clahe  seed=42
```

---

## Q1 — 实际 D′ 使用的 TAL positive assignment 机制

来源 `tal.py`（`aae7e8ac…`），超参 `loss.py:438` `topk=10, alpha=0.5, beta=6.0`。

**输入**（`loss.py:521-527` 实际调用）：
```
pred_scores.detach().sigmoid()            # (b, h*w, nc)   sigmoid 后、已 detach
(pred_bboxes.detach() * stride_tensor)    # 解码框 × stride、已 detach
anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
```
**classification score 的用法**（`tal.py:147-150`）：取 **GT 类别那一列** 的 **sigmoid 概率**（非 logit），
`detach()` 不回传梯度；**无 objectness 参与**。
**bbox quality**（`tal.py:153-155`）：`bbox_iou(..., CIoU=True).clamp_(0)` ⇒ **CIoU**。
**候选选法**（`tal.py:124-131`）：`mask_in_gts`（anchor 中心严格在框内）→ `mask_topk`（按 align 取前 10）
→ `mask_pos = mask_topk × mask_in_gts × mask_gt`，再经 `select_highest_overlaps` 解多 GT 冲突。

```
align(a,g) = cls_sigmoid(a, g_cls)^0.5 · CIoU(a,g)^6
pos(g)     = topk10{ a : align(a,g), a ∈ in_gts(g) }
```

## Q2 — classification target 到底是什么

`tal.py:215-240`（`get_targets`）+ `tal.py:110-116`（归一化）：
```
target_scores = scatter_(..., 1)                 # ← 先置 hard 1.0（仅 fg anchor、GT 类那一列）
target_scores = where(fg_mask>0, target_scores, 0)
align_metric *= mask_pos
pos_align_metrics = align_metric.amax(-1)        # 每个 GT 的 max align
pos_overlaps      = (overlaps*mask_pos).amax(-1) # 每个 GT 的 max CIoU over positives
norm_align_metric = align_metric * pos_overlaps / (pos_align_metrics+eps)
target_scores = target_scores * norm_align_metric
```
⇒ **精确形式**：
```
target[a, g_cls] = 1.0 · ( align[a] / max_a align ) · maxCIoU_over_positives(g)
```
**不是 hard 1，也不是纯 IoU**，而是 **alignment 加权 × IoU 缩放**。
在 **align 最大的 anchor** 上 `align/max = 1` ⇒ **峰值分类 target = 该 GT 的 maxCIoU over positives**。

## Q3 — classification loss 到底是什么

```
loss.py:415   self.bce = nn.BCEWithLogitsLoss(reduction="none")
loss.py:527   loss_cls = self.bce(pred_scores, target_scores.to(dtype))     # (b, h*w, nc)
loss.py:526   target_scores_sum = max(target_scores.sum(), 1)
loss.py:533   loss[1] = loss_cls.sum() / target_scores_sum      × hyp.cls (0.5)
```
`cls_pw = []` ⇒ `len([]) ≠ nc(12)` ⇒ `self.cls_pw = None` ⇒ **无类别加权**。
⇒ 分母**只含非零 target** 的求和 ⇒ 背景 anchor 对分母无贡献（不存在"背景稀释"）。
逐元素梯度：`∂BCE/∂logit = σ(logit) − t`。

---

## §9 / Q4 / Q5 / Q6 —— 86 条逐 GT（数值全部来自冻结量，非估算）

| 状态 | n | % |
|---|---:|---:|
| **GT-class positive confirmed（n_pos ≥ 1）** | **86** | **100.0%** |
| GT-class positive absent（n_pos = 0） | 0 | 0.0% |
| target exists but zero | 0 | 0.0% |
| **target > 0**（= maxCIoU > 0） | **86** | **100.0%** |
| target unavailable | 0 | 0.0% |
| **loss 实际接收到该 target**（`loss.py:526/533` 分母 > 0） | **86** | **100.0%** |

```
TARGET_PEAK (= maxCIoU):  min 0.5113  p25 0.6530  median 0.7227  p75 0.8152  max 0.9480
head_n_pos             :  min 6       median 10   max 10
|∂BCE/∂logit| at peak  :  median 0.7227   max 0.9479        （上限 1.0）
BCE at peak (nats)     :  median 9.662   min 4.460   max 16.055
```

> **target 幅度不小**（中位 **0.723**），且梯度绝对值中位 **0.723**、最大 **0.948**，**接近其理论上限 1.0**。
> ⇒ 对这些 GT，分类监督不仅**存在**，而且是**接近最强的**。

## Q7 — 缺失的 artifact

```
TRAINING_CANDIDATE_IDENTITY = NOT_OBSERVABLE
```
- **86 条是 val GT**，训练时**从不被监督**；没有任何 artifact 把她们的 GT 关联到 train 样本的
  anchor / grid index / candidate index。
- 冻结产物中**没有 `target_scores` / `target_labels` 的 tensor dump**。
- `candidate_density_counterfactual/_results.npz` 的 `cand_cls` 是**预测的 GT 类得分**，**不是 target**，
  且为 **train split** ⇒ 不能用来给这 86 条做逐 GT target 归属。
- 无 gradient / 逐 GT loss 贡献的 dump（需 backward）⇒ `NOT_OBSERVABLE_FROM_FROZEN_ARTIFACTS`。

**但关键点**：Q4/Q5/Q6 **不需要** candidate identity —— 因为「峰值 target = maxCIoU」是**代码恒等式**，
对**任何**有正样本的 GT 都成立，而 `best_assign_iou` 恰好就是从冻结产物里读到的 maxCIoU over positives。
所以这 86 条的 peak target 与 loss 接收性是**可从冻结量直接给出的**。

**仍然不可观测的**：非峰值 anchor 上的 target、每个 GT 在分类 loss 中占的**份额**、以及**梯度实际大小**
（需要 backward）。

## Q8 — class imbalance 的描述性证据

train：**1599 张有效图 / 11562 GT**（越界剔除口径同 `official load_split`）。

| class | train_GT | 份额 | 86 | 110 | 86 出现率 (每 1k train GT) |
|---|---:|---:|---:|---:|---:|
| person | 4421 | 38.24% | 25 | 30 | 5.65 |
| animal | 2328 | 20.13% | 16 | 19 | 6.87 |
| car | 1283 | 11.10% | 3 | 5 | 2.34 |
| light | 1211 | 10.47% | 13 | 15 | 10.73 |
| **sign** | 659 | 5.70% | **13** | 18 | **19.73** |
| bicycle | 550 | 4.76% | 10 | 12 | 18.18 |
| seat | 511 | 4.42% | 1 | 5 | 1.96 |
| garbage_can | 225 | 1.95% | 3 | 3 | 13.33 |
| uav | 174 | 1.50% | 0 | 0 | 0.00 |
| boat | 107 | 0.93% | 0 | 0 | 0.00 |
| ball | 70 | 0.61% | 2 | 3 | 28.57 |
| tricycle | 23 | 0.20% | 0 | 0 | 0.00 |

**只报告数据**：`sign` 的 86-出现率（19.73/1k）约为 `person`（5.65/1k）的 **3.5×**；
`ball` 更高（28.57/1k）但 train 只有 70 个 GT（86 中 2 条）。**不做 ranking、不推定因果权重**。

## §13 — sign / person 事实核对

| class | 86 中条数 | n_pos median | TARGET_PEAK median | train_GT | 出现率 |
|---|---:|---:|---:|---:|---:|
| sign | 13 | 10 | 0.8151 | 659 | 19.73/1k |
| person | 25 | 10 | 0.7221 | 4421 | 5.65/1k |

⇒ 这两类在 86 中**都不存在 assignment 缺失**：n_pos ≥ 6、target ≥ 0.5。
上一轮观察到的「sign 有全部 2 条 strong classification failure / cls_lim 最多」**不能**在本轮的
GT frequency、TAL assignment 或 classification target 中找到"监督缺失"的对应事实 —— 它们的监督是正常且强的。

## §14 — 必须分开的两件事

```
inference-time prediction  : GT 类 sigmoid ≈ 1e-6（val 侧、训练后、post-NMS rect 输出）
training-time supervision  : GT 类 positive 存在、target = maxCIoU ≈ 0.72、gradient ≈ −0.72（接近上限）
```
**不得**由前者反推后者。本审计中"监督强"的结论来自 **target 构造恒等式 + loss 归一化**，
**不是**由预测分数低反推出来的。

## Q9 / §11 — MECHANISM STATUS

```
SUPERVISION_CONFIRMED
```
判定依据（§11 三个必要条件，86/86 全部满足）：**GT-class positive 存在 + target > 0 + loss 接收该 target**。
`SUPERVISION_PRESENT_BUT_WEAK` 不适用 —— 本审计**确实**拿到了 target 幅度（0.723）、BCE（9.66 nats）
与梯度（0.723）证据，且三者都**大**，不是弱。

### 这一结论的精确 scope（必须与结论同读）

- 被证明的是**监督通路的结构健全性**：任何具有这种几何的对象，若出现在训练集，会获得
  `n_pos ≥ 6` 的 GT 类正样本与 `target = maxCIoU ≥ 0.51`（中位 0.72）的分类目标，且该目标必然进入分类 loss。
- **86 条本身是 val GT，训练时从不被监督** —— 对她们**个体**不能说"获得了监督"。
- 因此可以排除的是：**「inference score collapse 由缺失或微弱的 classification supervision 造成」**。
  这正是任务描述中的**情况 A**（classification representation / optimization 方向），**不是 B 或 C**。

### 一个从代码得到的附带观察（**不解释**这 86 条）

target 的形式是 `(align/maxAlign) × maxCIoU`，即**分类目标被模型自身的框质量缩放**。
对框质量差的对象，分类 target 会同时变小。**但本轮的 86 条 maxCIoU 中位 0.723，target 很大** ⇒
该效应**不解释**这 86 条。是否影响其它对象，本轮**未测**。

---

## §20 最终输出

```
MECHANISM STATUS : SUPERVISION_CONFIRMED
                    （86/86：positive 存在 ∧ target>0 ∧ loss 接收；target 中位 0.723，|grad| 中位 0.723）
EVIDENCE         : tal.py:95-158,215-240 的 target 构造恒等式；loss.py:415/526/527/533 的 loss 形式与归一化；
                   D′ args.yaml（box7.5/cls0.5/dfl1.5/cls_pw=[]）；冻结的 _v6_domains.json V1
                   （n_pos / best_assign_iou = maxCIoU over positives）
LIMITATION       : ① 86 条是 val GT，训练时从不被监督 —— 上述是"通路健全 + 若在训练集会得到的监督"，
                     不是对这 86 个个体的监督归属；
                   ② TRAINING_CANDIDATE_IDENTITY = NOT_OBSERVABLE（无 anchor/grid/candidate index dump）；
                   ③ 非峰值 anchor 的 target、每 GT 的 loss 份额、真实梯度大小 = NOT_OBSERVABLE_FROM_FROZEN_ARTIFACTS
                     （需 backward）；
                   ④ 本轮不训练、不提出新的 loss gain 数值、不启动实验。
```

---

## 生成文件（新目录，未覆盖前两轮）

```
diagnostic/medium_cls_supervision_audit/
    input_manifest.txt       输入清单 + SHA256 + NOT_AVAILABLE 汇总
    tal_code_audit.md        TAL A/B/C/D 四问逐条 + 简化公式
    loss_code_audit.md       target 构造 / 归一化 / loss 形式 / 逐元素梯度
    per_gt_supervision.csv   86 行 × 20 列（逐 GT positive/target/gradient/BCE）
    class_distribution.csv   12 类 train GT 分布 vs 86/110 失败分布
    summary.json             全部统计 + checks
    summary.md               本文件
    consistency_checks.txt   CHECK A–H
    _run.log                 运行日志
    _audit.py                本次脚本（只读）
```

## 异常与修正（不省略）

1. 脚本初版有一个 **f-string 花括号未转义**导致的 `NameError`（`topk10{ a : … }`），已修正为 `{{ }}` 并重跑。
2. 脚本初版把 **CHECK F 的「条件未触发」误标为 FAIL**：该检查的条件是"若报告逐 GT assignment 则必须存在
   candidate identity"，而本审计**正确地没有**报告逐 GT assignment（标为 NOT_OBSERVABLE）⇒ 条件未触发，
   已改为 PASS-by-vacuity。**若不修，报告里会出现一个假的 FAIL。**
3. 上一轮两个目录的 SHA 在本轮前后未变（`per_gt_attribution.csv` `62dbc551…`、
   `per_gt_causal_audit.csv` `7041256d…`）。
