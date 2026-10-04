STATUS = PASS

# Medium 86 条 Head→Final Candidate Disappearance Causal Audit

只读审计。0 训练 / 0 GPU / 0 重推理 / 0 重跑 evaluator / 0 改源码·YAML·checkpoint·既有 JSON /
0 重定义 86 / 0 自造 NMS / 0 改阈值。**结论等级词表：`PROVEN` / `SUPPORTED` / `CONSISTENT_WITH` /
`NOT_OBSERVABLE` / `UNRESOLVED`。**

---

## 1. Frozen inputs

| 输入 | SHA256 |
|---|---|
| `diagnostic/medium_93_47_attribution/per_gt_attribution.csv` | `62dbc551f4703bb3162baed16266f793b6e9cba4f21509236afee186bfc0c605` |
| `diagnostic/small_object_domains/_v6_domains.json`（head 侧） | `3505d9160c47aae813bef8774ea81cdcd3be51df8f5705c4f8058e17e232f355` |
| `diagnostic/sepstem_clahe/best_full/results`（final 侧） | 400 txt（上一轮已内容级确认） |
| `diagnostic/box2_infer_val/_medium_pair.log`（145/98/47 参照） | `a9a36cace44793892c17c058cd1f1658dff3f7dbd49e084924f30fbbeec45a65` |

冻结代码（阈值/构成/界的唯一来源）：`metrics.py` `d2c8e4bf…`、`ops.py` `1c674a2705…`、
`tal.py` `aae7e8ac…`、`predict_rect.py` `0f247f54…`。
`--conf 0.001` / `--iou 0.7` / `--max_det 300` / `--max_boxes 100` / **`multi_label=True`**。

## 2. Sample-set validation

```
N_86 = 86   ✓（与上一轮一致；逐条 (image_stem, gt_id) 相同）
  in_collapse    = 72
  in_wrong_class = 31
  failure_group  = {collapse 55, collapse+wrong_class 17, wrong_class 14}
CHECK 3: 86 条全部 prefinal certified ∧ final best_any < .50  ✓
```

## 3. Head Candidate Record —— 字段可用性（不估算）

| 目标字段 | 状态 |
|---|---|
| image_id / gt_id / gt_class | ✅ `key = stem#gt` |
| **head CIoU** | ✅ `best_assign_iou`（TAL 正样本 anchor 的最大 **CIoU**） |
| **head class score** | ✅ `pos_prob`（该 GT **全部 TAL 正样本 anchor** 上 GT 类 sigmoid 的**最大值**） |
| head center logit / prob | ✅ `center_logit` / `center_prob`（GT 类，GT 中心 12px 内，stride-8） |
| head best_align / pos_target / n_pos | ✅ |
| candidate bbox / candidate class / candidate rank | ❌ **NOT_AVAILABLE** |
| objectness / confidence | ❌ **NOT_AVAILABLE**（且模型**没有** objectness 分支，见 §6） |
| 真实 IoU | ❌ **NOT_AVAILABLE**（只有 CIoU） |
| pre-NMS / post-NMS 同一 candidate 对照 | ❌ **NOT_AVAILABLE** |

## 4. CIoU 与 IoU 的关系

**`CIoU <= IoU` = `PROVEN`（代码级，非数据断言）**
```
metrics.py:305   return iou - (rho2 / c2 + v * alpha)     两项均 >= 0
tal.py:155       bbox_iou(..., CIoU=True).squeeze(-1).clamp_(0)
```
⇒ `best_assign_iou >= 0.50` **认证**存在 IoU ≥ .50 的 head 候选。
**无任何冻结产物提供真实 IoU** ⇒ 本审计一律记
```
prefinal_geometry = CERTIFIED_BY_CIOU_LOWER_BOUND
```
**不写**「prefinal IoU = …」。

## 5. §6 confidence 的构成 = `PROVEN`

```
ops.py:250   xc = prediction[:, 4:mi].amax(1) > conf_thres
ops.py:247   nc = nc or (prediction.shape[1] - 4)
```
⇒ `confidence = 全部类别 sigmoid 的最大值`，**无独立 objectness 通道**（YOLO11 anchor-free，`no = nc + 4*reg_max`）。
⇒ `CONFIDENCE_COMPOSITION = CLASS_PROBABILITY_MAX`（**已确认，不是 UNKNOWN**）。
**关键推论**：`confidence >= GT 类得分` ⇒ `pos_prob < 0.001` **不等于** confidence < 0.001。

## 6. §5 class-score distribution

| score bucket | 86 全体 | 93-collapse 部分(72) | 47-wrong_class 部分(31) |
|---|---:|---:|---:|
| <1e-6 | **35 (40.7%)** | 35 (48.6%) | 10 (32.3%) |
| 1e-6–1e-5 | **37 (43.0%)** | 37 (51.4%) | 7 (22.6%) |
| 1e-5–1e-4 | 10 (11.6%) | 0 | 10 (32.3%) |
| 1e-4–1e-3 | 1 (1.2%) | 0 | 1 (3.2%) |
| >=1e-3 | **3 (3.5%)** | **0** | **3 (9.7%)** |
| unavailable | 0 | 0 | 0 |

```
86      : median 1.624e-06  mean 5.845e-05  min 2.100e-09  max 1.928e-03
collapse: median 1.099e-06  max 9.781e-06      ← 全部 <= 1e-5
wrongcls: median 6.736e-06  max 1.928e-03
```

## 7. §7 confidence-filtering evidence — 两个层次必须分开

### 7a. **GT 类通道在认证 anchor 上被阈值滤除 = `SUPPORTED`（83/86）**
推导链（全部来自冻结数据/代码）：
```
pos_prob = 该 GT 全部 TAL 正样本 anchor 上 GT 类 sigmoid 的最大值
argmax-CIoU 的 anchor ∈ TAL 正样本集合            → 其 GT 类得分 <= pos_prob
multi_label=True (predict_rect.py:262)            → 每一类独立成候选
ops.py:250 候选需 max-over-classes > conf         → 该 anchor 的 GT 类候选需 GT 类得分 > 0.001
若 pos_prob < 0.001                               → 该 anchor 的 GT 类候选不可能被发出
⇒ 83 条满足 pos_prob < 0.001
```
**scope 必须写明**：这是 **GT 类通道**的分数过滤，不是「整个框被滤掉」。

### 7b. **框（any-class）的消失 = `CONSISTENT_WITH_SCORE_FILTERING`（55/86）**
`confidence = max-over-classes`，该值**在任何冻结产物中都不存在** ⇒ 无法证明框本身被 conf 滤掉。
`final best_any_iou < .50` 的 86 条中：
```
55 条 final 里【没有任何框】与该 GT 重叠   → CONSISTENT_WITH_SCORE_FILTERING
31 条 final 里有框重叠但类别错            → BOX_SURVIVED_AS_WRONG_CLASS
```
31 条属于后者 ⇒ 对它们而言**框并没有消失**，失败发生在**类别竞争**（该位置的胜出类不是 GT 类）。

## 8. §Table A —— 86 条最终机制

| Mechanism | n | % |
|---|---:|---:|
| **SUPPORTED_SCORE_FILTERING**（严格：`pos_prob < conf 0.001`） | **83** | **96.5%** |
| CONSISTENT_WITH_SCORE_FILTERING | **0** | 0.0% |
| SUPPORTED_NMS | 0 | 0.0% |
| SUPPORTED_GEOMETRY_MISMATCH | 0 | 0.0% |
| SUPPORTED_TOP100 | 0 | 0.0% |
| MULTI_FACTOR | 0 | 0.0% |
| **UNRESOLVED**（`pos_prob >= 0.001`，推导不成立） | **3** | **3.5%** |

（辅助，非上表类别）`BOX_SURVIVED_AS_WRONG_CLASS` 31 / `CONSISTENT_WITH_SCORE_FILTERING`(any-class) 55。

**3 条 UNRESOLVED 明细**（GT 类得分**不低于**阈值 ⇒ 候选本应被发出却缺席）：
```
001035#9                        sign    pos_prob=1.166e-03  head_ciou=0.8784  final_any_iou=0.0069  pred_cls=6  raw_preds=20  trunc=0
002987#12                       person  pos_prob=1.190e-03  head_ciou=0.7405  final_any_iou=0.1958  pred_cls=5  raw_preds=23  trunc=0
hehe_230_000002_043_00000061#13 person  pos_prob=1.928e-03  head_ciou=0.7133  final_any_iou=0.3239  pred_cls=5  raw_preds=19  trunc=0
```
这 3 条的缺席只能归因于**类内 NMS / 截断 / 几何变化**，三者均 `NOT_OBSERVABLE` ⇒ 记 `UNRESOLVED`，**不猜**。

## 9. §Table B / C

| head CIoU | class score state | n |
|---|---|---:|
| >=.50 | COLLAPSED (<=1e-5) | **72** |
| >=.50 | LOW (1e-5–1e-3) | 11 |
| >=.50 | NORMAL (>=1e-3) | 3 |
| >=.75 | COLLAPSED | **26** |
| >=.75 | LOW | 7 |
| >=.75 | NORMAL | 1 |

## 10. §12 高几何 × 极低 class score（本轮最关键 subgroup）

| 条件 | n | % of 86 | median score | max score |
|---|---:|---:|---:|---:|
| `head CIoU >= .50` ∧ `score <= 1e-5` ∧ `final best_any < .50` | **72** | **83.7%** | 1.099e-06 | 9.781e-06 |
| `head CIoU >= .75` ∧ `score <= 1e-5` ∧ `final best_any < .50` | **26** | **30.2%** | 8.655e-07 | 9.383e-06 |

> ⚠ 措辞纪律（§13）：不得写成「86 个 GT 的 IoU 都 >= .50」。正确表述是：
> 「86 个 GT 在冻结 pre-final/head 产物中**存在 CIoU >= .50 的候选**，因而获得 **IoU >= .50 的下界认证**；
> 但这些 GT 在冻结 final post-NMS rect 输出中**均没有 best-any IoU >= .50**。」

## 11. §8 NMS / §9 geometry / §10 top-100

| 机制 | 状态 | 依据 |
|---|---|---|
| **NMS** | **`NOT_OBSERVABLE`（86/86）** | 任何冻结产物都无 candidate identity、也无 NMS 前/后同一 candidate 对照 ⇒ 不得用 IoU 猜测对应关系，不得自造 NMS |
| **geometry / rect mismatch** | **D′ 上 `NOT_OBSERVABLE`（86/86）** | head 产物是 **square** pre-NMS 张量，final 是 **rect** post-NMS；无中间 transform metadata |
| ↳ 异模型定量上界 | `CONTROL_ONLY_DIFFERENT_MODEL` | F4(rgbd) medium n=1316：square→rect 的 `best_same` IoU Δ 中位 **+0.0000**，`|Δ|` p95 **0.0990**，跨过 .50 门槛 **升 11 / 降 19**（净 −8，占 0.6%）。⇒ 推理几何能挪动个别框跨过 .50，但**不足以解释 86 条（占 medium 6.5%）整体消失**。**异模型，非 D′ 直接证据。** |
| **top-100** | `POSSIBLE` 2 / `NOT_OBSERVABLE` 84 | 全 val 只有 **4 张**图原始框数 >100；86 条中落在这些图里的仅 **2 条**。缺 candidate rank ⇒ 只能 `POSSIBLE`。**「图触顶」≠「该 GT 的候选因 top-100 消失」（§10 硬要求）** |

2 条 `POSSIBLE` 明细：`003987#55`(bicycle, raw_preds=145) / `shuming_340_00000020#1`(sign, raw_preds=105)。

## 12. §15 Counterfactual

```
COUNTERFACTUAL = NOT_IDENTIFIABLE
```
理由：要判断「只看 pre-final 的 class score，这些候选是否有机会进入 final」，需要
**该 anchor 处 max-over-classes 的置信度**与完整的后处理输入；冻结产物里**没有**这个量。
`pos_prob` 只是 GT 类通道的得分，是 confidence 的**下界**，下界低于阈值**不能**推出 confidence 低于阈值。
—— **数学上显然很低 ≠ 已证明该 pipeline 的 filtering 规则导致消失。** 本审计不跨越这一步。

## 13. §16 Consistency checks

```
[PASS] CHECK 1  86 条与上一轮逐条一致（(stem, gt_id) 集合相同）        n=86
[PASS] CHECK 2  每条含 image_id / gt_id / gt_class
[PASS] CHECK 3  全部 prefinal certified ∧ final best_any<.50
[PASS] CHECK 4  class score 全为 numeric（统一 float；防 '1' vs 1 类型 bug）  non-numeric=0
[PASS] CHECK 5  统计可由逐 GT CSV 重新聚合
[PASS] CHECK 6  未覆盖上一轮文件（新目录 medium_head_to_final_audit）
[PASS] CHECK 7  CIoU<=IoU 的代码级界成立（metrics.py:305 两项非负 + tal.py:155 clamp）
[PASS] CHECK 8  无任何 artifact 提供真实 IoU / candidate identity / 前后 NMS 对照
```

## 14. §19 六个问题的答案

1. **86 条中有多少 class score <= 1e-5？** → **72 (83.7%)**（<=1e-4 为 82；>=1e-3 为 3）
2. **有多少同时满足 `head CIoU >= .50 + score <= 1e-5 + final best-any < .50`？** → **72 (83.7%)**（把 CIoU 提到 >=.75 则 **26, 30.2%**）
3. **有多少可以真正标记 `SUPPORTED_SCORE_FILTERING`？** → **83 (96.5%)**（严格条件 `pos_prob < conf=0.001`，且推导链完整、composition 与阈值均已由冻结代码确认）。**scope = GT 类通道在认证 anchor 上。**
4. **有多少只能标记 `CONSISTENT_WITH_SCORE_FILTERING`？** → **GT 类通道上 0 条**（因为 composition/threshold 已确认，`CONSISTENT_WITH` 的前提"无法确认"不成立）。**但在 any-class 框的消失上：55 条只能记 `CONSISTENT_WITH_SCORE_FILTERING`**（缺 max-over-classes）；另 31 条记为 `BOX_SURVIVED_AS_WRONG_CLASS`。
5. **NMS / geometry / top-100 有没有获得直接证据？** → **都没有直接证据**。NMS `NOT_OBSERVABLE`×86；geometry 在 D′ 上 `NOT_OBSERVABLE`×86（仅有异模型上界：推理几何量级 p95≈0.099、净 −8/1316，不足以解释整体消失）；top-100 仅 `POSSIBLE`×2。
6. **最终应写成哪一个？** →
   - 对「**GT 类通道在具有几何下界认证的 anchor 上被 conf 阈值滤除**」：**`SUPPORTED`**（83/86）
   - 对「**这些候选为何从最终输出消失**」这一**整体因果问题**：**`UNRESOLVED`**

## 15. 最终结论（按 §20 克制表述）

冻结 pre-final/head 产物显示，86 条 medium 失败 GT 中的 **83 条**在具有 `CIoU >= .50` 的几何质量下界认证的同时，
其 GT 类 score 低于推理阈值 `conf=0.001`（中位 `1.624e-06`），并在最终 post-NMS rect 输出中消失。
这对「**GT 类别的 score collapse 与该类候选无法进入最终输出相关**」提供了 **`SUPPORTED`** 级别的支持。

**不可写成**：classification 已证明是唯一瓶颈、或已证明是根本原因。

**必须同时写明**：当前冻结产物**不足以完成因果分离** —— score filtering 是**唯一**获得直接证据的通道；
NMS、几何变换、top-100 三者**全部 `NOT_OBSERVABLE`**，因此「为何整体消失」的完整归因仍为 **`UNRESOLVED`**。

---

## 16. 生成文件（新目录，未覆盖上一轮）

```
diagnostic/medium_head_to_final_audit/
    input_manifest.txt          输入清单 + SHA256 + 字段可用性
    per_gt_causal_audit.csv     86 行 × 43 列（逐 GT causal state）
    summary.json                全部统计 + checks + control
    summary.md                  本文件
    consistency_checks.txt      CHECK 1–8
    _run.log                    完整运行日志
    _audit.py                   本次脚本（只读）
```

## 17. 异常与修正（不省略）

1. **本审计脚本初版有一个过度断言**：`PRIMARY_INTERPRETATION` 被硬编码为 86 条全 `SUPPORTED_SCORE_FILTERING`，
   但 Table B 显示有 **3 条** `pos_prob >= 0.001`，其推导**不成立**。已改为条件赋值，
   最终 83 SUPPORTED / 3 UNRESOLVED。**若未发现，本审计就会把一个未经支持的分裂结论报成 100%。**
2. **`CONSISTENT_WITH_SCORE_FILTERING` 在 Table A 中为 0 是刻意的**，不是遗漏：
   用户规则中该等级的触发条件是「composition/threshold 无法从冻结数据确认」，
   而本审计**已从冻结代码确认**了 composition（`ops.py:250`）与阈值（`predict_rect.py:186`），
   故该前提不成立；相应地，未达 `SUPPORTED` 严格条件的 3 条被记为 `UNRESOLVED` 而不是降级塞进 `CONSISTENT_WITH`。
3. **§9 geometry 的量化上界来自 F4（rgbd）而非 D′**，只作对照，未混入主统计（上一轮禁止 3 的延续）。
4. head 侧与 final 侧来自**两个不同几何**（square pre-NMS vs rect post-NMS），
   本审计**没有**假设二者可逐框对应；所有结论均限定在各层自身可观测的量上。
