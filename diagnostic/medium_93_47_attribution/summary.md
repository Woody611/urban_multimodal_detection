STATUS = PASS

# Medium 失败 GT 的 93+47 零 GPU attribution

只读诊断。0 训练 / 0 推理 / 0 源码修改 / 0 config 修改 / 0 checkpoint 修改 / 0 预测重生成 /
0 evaluator 重跑 / 0 阈值改动。未重新定义 93 与 47，只按原规则从冻结输入**恢复**。

---

## 1. Frozen inputs

D′ 冻结预测的识别**基于内容而非文件名**：medium(1320) 上「同类零重叠=145 / 任意类零重叠=98」
与冻结参照 `_medium_pair.log` 逐位吻合；作为对照，box2 得 148/99、D′ last 得 146/100，
也各自与自己的参照吻合 ⇒ 唯一确定。

| 输入 | 路径 | 规模 | SHA256 |
|---|---|---|---|
| GT labels | `data/processed/rgbid_split_train/labels/val/visible` | 400 txt | (dir) |
| GT label cache | `…/labels/val/visible.cache` | — | `2fb020240b2841d2bf574ff0754b831789acfc217a8b90c758e7818b80e1738e` |
| images | `data/processed/rgbid_split_train/images/val/visible` | 400 files | (dir) |
| **D′ 冻结预测** | `diagnostic/sepstem_clahe/best_full/results` | 400 txt | `1919f7cc430436c16369a3bb75a0d5698f1956c05b0f39823e7ec67c1c4a7849` |
| pos_prob 来源（V1） | `diagnostic/small_object_domains/_v6_domains.json` | 7926 recs | `3505d9160c47aae813bef8774ea81cdcd3be51df8f5705c4f8058e17e232f355` |
| 145/98/47 冻结参照 | `diagnostic/box2_infer_val/_medium_pair.log` | — | `a9a36cace44793892c17c058cd1f1658dff3f7dbd49e084924f30fbbeec45a65` |

## 2. Sample-set validation

```
zero_same_class (medium, best_same_iou <= 1e-9)  = 145   == 冻结参照 145  ✓
wrong_class     (zero_same ∧ best_any > 1e-9)    =  47   == 冻结参照  47  ✓
collapse        (zero_same ∧ pos_prob <= 1e-5)   =  93   == 冻结参照  93  ✓

collapse ∩ wrong_class = 30
union                  = 110        ← ⚠ 不是 140
```

**⚠ 集合事实（必须记录）**：任务的标称 `union = 140` 不成立。93+47=140 只在两集合互斥时等于并集，
而 `wrong_class` 是 `zero_same` 的**子集条件**（`best_any>0 ∧ best_same=0`），与 `collapse` 同受
`zero_same` 约束 ⇒ **实际重叠 30，并集 110**。按 CHECK B 不去重、如实报告。

## 3. Attribution summary

### A. 93 classification-collapse

| | n | % |
|---|---:|---:|
| `best_any_iou >= .50` | 6 | 6.5% |
| `best_any_iou >= .75` | 1 | 1.1% |
| `best_any_iou >= .90` | 1 | 1.1% |
| `best_same_iou >= .50 / .75 / .90` | 0 | 0.0% |
| LOCALIZATION_LIMITED | 87 | 93.5% |
| CLASSIFICATION_LIMITED | 6 | 6.5% |
| STRONG_CLASSIFICATION_FAILURE | 1 | 1.1% |
| POSSIBLE_RANKING_LIMITED | 0 | 结构性不可达（见 §6 H） |
| TOP100_LIMITED | NOT OBSERVABLE FROM FROZEN OUTPUT | |
| MIXED / UNRESOLVED | 0 / 0 | |

### B. 47 wrong_class

| | n | % |
|---|---:|---:|
| `best_any_iou >= .50` | 9 | 19.1% |
| `best_any_iou >= .75` | 2 | 4.3% |
| `best_any_iou >= .90` | 1 | 2.1% |
| `best_same_iou >= .50 / .75 / .90` | 0 | 0.0% |
| LOCALIZATION_LIMITED | 38 | 80.9% |
| CLASSIFICATION_LIMITED | 9 | 19.1% |
| STRONG_CLASSIFICATION_FAILURE | 2 | 4.3% |
| POSSIBLE_RANKING_LIMITED | 0 | 结构性不可达 |
| MIXED / UNRESOLVED | 0 / 0 | |

### C. union 110

| | n | % |
|---|---:|---:|
| `best_any_iou >= .50` | 9 | 8.2% |
| `best_any_iou >= .75` | 2 | 1.8% |
| `best_any_iou >= .90` | 1 | 0.9% |
| `best_same_iou >= .50` | 0 | 0.0% |
| **LOCALIZATION_LIMITED** | **101** | **91.8%** |
| **CLASSIFICATION_LIMITED** | **9** | **8.2%** |
| STRONG_CLASSIFICATION_FAILURE | 2 | 1.8% |
| POSSIBLE_RANKING_LIMITED | 0 | 结构性不可达 |
| MIXED / UNRESOLVED | 0 / 0 | |

### C′. 加入**第二个冻结产物（头部侧）**的三分法 — auxiliary，非重定义

`head_best_assign_iou` = `_v6_domains.json` V1 里 TAL 正样本 anchor 的最大 CIoU。
因 CIoU ≤ IoU，`CIoU >= .50` 即**认证**头部存在 IoU ≥ .50 的框（pre-NMS、square 几何）。
该列已写入 `per_gt_attribution.csv`（`head_best_assign_iou` / `head_pos_prob` / `head_n_pos` /
`certified_prefinal_loss`），**用户规则的 `attribution` 列未被改动**。

| 组 | n | final 有 ≥.50 | **头部认证但 final 无** | 两者皆无 |
|---|---:|---:|---:|---:|
| collapse | 93 | 6 (6.5%) | **72 (77.4%)** | 15 (16.1%) |
| wrong_class | 47 | 9 (19.1%) | **31 (66.0%)** | 7 (14.9%) |
| **union** | **110** | **9 (8.2%)** | **86 (78.2%)** | **15 (13.6%)** |

头部侧分布（collapse 93）：CIoU ≥.50 **78/93 (83.9%)**；≥.60 62 (66.7%)；≥.70 44 (47.3%)；
≥.75 **29 (31.2%)**；≥.90 4 (4.3%)。GT 类 sigmoid（`pos_prob`）：p50 **1.16e-06**，p90 6.40e-06，max 9.78e-06。

## 4. best-any vs best-same table

| best_any IoU | best_same IoU | interpretation | n |
|---|---|---|---:|
| <0.50 | <0.50 | localization-limited | **101** |
| >=0.50 | <0.50 | classification-limited | **9** |
| >=0.75 | <0.50 | strong classification failure | **2** |
| >=0.75 | >=0.50 | geometry exists; inspect ranking | **0** |
| >=0.90 | >=0.75 | very strong geometric candidate | **0** |

## 5. Class breakdown

| class | collapse n | wrong_class n | any_iou mean | any_iou med | same_iou mean | same_iou med | cls_lim | loc_lim | strong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| person | 27 | 9 | 0.0416 | 0.0000 | 0.0000 | 0.0000 | 1 | 29 | 0 |
| animal | 19 | 1 | 0.0363 | 0.0000 | 0.0000 | 0.0000 | 1 | 18 | 0 |
| light | 14 | 4 | 0.0192 | 0.0000 | 0.0000 | 0.0000 | 0 | 15 | 0 |
| sign | 14 | 12 | 0.2379 | 0.0431 | 0.0000 | 0.0000 | **4** | 14 | **2** |
| bicycle | 8 | 8 | 0.2004 | 0.1721 | 0.0000 | 0.0000 | 1 | 11 | 0 |
| car | 4 | 5 | 0.2774 | 0.2477 | 0.0000 | 0.0000 | 1 | 4 | 0 |
| seat | 4 | 3 | 0.1250 | 0.0442 | 0.0000 | 0.0000 | 1 | 4 | 0 |
| ball | 2 | 2 | 0.0957 | 0.0418 | 0.0000 | 0.0000 | 0 | 3 | 0 |
| garbage_can | 1 | 3 | 0.1149 | 0.0481 | 0.0000 | 0.0000 | 0 | 3 | 0 |

（其余 3 类 — boat / uav / tricycle — 在本集合中 0 条，未列出。不做主观排名。）

**两条纯数据观察**：
1. 全 12 类中只有 **sign** 出现 `strong_classification_failure`（2 条，占全部 2 条），且它的
   `cls_lim` 最多（4）。`light` / `ball` / `garbage_can` 的 `cls_lim` 为 0。
2. 9 条 final 有 ≥.50 的候选中，**5 条**的 `best_any_pred_class = 0 (person)`
   （另有 1 条 pred=8 light、1 条 pred=9 garbage_can、1 条 pred=5 bicycle、1 条 pred=11 tricycle）。
   其中 3 条的候选框 IoU ≥ 0.5 且置信度很高（0.888 / 0.859 / 0.504），仍被判给 person。

## 6. Consistency checks

```
[PASS] CHECK A  collapse==93 / wrong_class==47                            collapse=93 wrong_class=47
[PASS] CHECK B  union rows==110 (标称 93+47; 实际重叠 30)                  rows=110 union=110 intersection=30
[PASS] CHECK C  best_any_iou >= best_same_iou for every GT                tol=1e-06
[PASS] CHECK D  0 <= IoU <= 1
[PASS] CHECK E  best_same 的类别 == GT 类别                                by construction (same-class filtered)
[PASS] CHECK F  best_any 类==gt 类 ⇒ best_any_iou==best_same_iou           violations=0 tol=1e-06
[PASS] CHECK G  aggregates 可由 per-GT rows 重新聚合                      recount == summary
[PASS] CHECK H  POSSIBLE_RANKING_LIMITED 在本样本结构性不可达             all 110 have best_same==0
```

## 7. Interpretation

### classification evidence
- 在**最终预测**层：union 110 中只有 **9 (8.2%)** 存在任意类 IoU ≥ .50 的候选 ⇒ 按用户规则
  CLASSIFICATION_LIMITED = 9；再叠 ≥.75 条件得 STRONG = 2。
- 在**头部侧冻结产物**层（第二个 artifact）：**86/110 (78.2%)** 被认证头部存在 IoU ≥ .50 的框，
  而最终输出里没有；同时它们的 GT 类 sigmoid 中位 **1.16e-06**、最大 9.78e-06 ⇒ 头部有几何上成立的
  候选，但**该类别的得分塌陷到近乎零**，因此被过滤。
- 两条合起来：**「好框 + 类别得分塌陷 + 未进入最终输出」这一链路，对 86/110 有直接证据。**

### localization evidence
- 按用户规则的冻结输出口径，**101/110 (91.8%)** 是 LOCALIZATION_LIMITED。
- 但其中 **86 条**在头部侧被认证存在 ≥.50 的框 ⇒ 这 101 条**不能**被读成「模型无法定位这些目标」。
- 真正「头部与最终都没有 ≥.50 候选」的只有 **15/110 (13.6%)**（person 4 / seat 3 / light 2 /
  animal 2 / bicycle 1 / sign 1 / ball 1 / car 1）。

### ranking evidence
- **NOT OBSERVABLE FROM FROZEN OUTPUT**：本样本按 `best_same == 0` 预选，`POSSIBLE_RANKING_LIMITED`
  要求的 `best_same >= .50` 在 110 条中**一条都不存在** ⇒ ranking 类在本样本内结构性不可达。
- `TOP100_LIMITED`：冻结产物无法证明任一候选因 100 框上限被截断 ⇒ NOT OBSERVABLE（不因某图框数 <100 推断）。

### unresolved evidence
- **「类别」是直接死因、还是「后处理/几何」是直接死因，本证据不判定。** `_v6_domains.json` 是
  square 几何 + pre-NMS；最终预测是 rect + NMS + conf + top-100。两者之间的落差同时包含
  得分过滤、NMS 抑制与几何改变，**三者的相对贡献本轮无法分离**。
- 15 条「两者皆无」的具体成因（小/遮挡/标注）本轮**未测**，标 UNRESOLVED。

## 8. IMPLICATION

> 在本组 110 条 medium 失败 GT 上，获得最多直接证据的机制是
> **「头部存在几何成立的候选（86/110 认证 ≥.50），但该 GT 类别的得分塌陷到 ~1e-6，
> 候选未进入最终输出」** —— 即一个**pre-final / 分数塌陷**机制。
> 「分类是直接死因」与「后处理是直接死因」的区分**未被本证据确定**。
> 纯 localization 无法定位的只占 15/110 (13.6%)。

---

## 生成的诊断文件（均为新建，未覆盖任何既有文件）

```
diagnostic/medium_93_47_attribution/
    input_manifest.txt          全部输入的路径 / 规模 / SHA256 + 内容级识别依据
    per_gt_attribution.csv      110 行 × 32 列（含头部侧 auxiliary 4 列）
    summary.json                93 / 47 / union 三组统计 + 集合关系 + checks
    summary.md                  本文件
    consistency_checks.txt      CHECK A–H
    _run.log                    完整运行日志
    _attribution.py             本次分析脚本（只读；置于本诊断目录内，未改动任何既有源码）
```

## 异常与 ambiguity（不省略）

1. **标称 union = 140 不成立**：实际 `collapse ∩ wrong_class = 30`，并集 **110**。已按 CHECK B 如实报告，未去重。
2. **最终输出口径 与 头部侧口径 结论方向相反**（LOCALIZATION_LIMITED 101 vs 头部认证 86）。
   原因是两者来自**不同的冻结产物**（post-NMS 最终预测 vs pre-NMS 头部张量），
   本轮**未**判定哪个更"真"；两个数字都已列出，未取其一。
3. **`POSSIBLE_RANKING_LIMITED` 在本样本内恒为 0，且这是预选造成的，不是"没有 ranking 问题"**。
   要评估 ranking，必须另取一个**不按 best_same==0 预选**的样本。
4. 本分析初期出现过一次 `certified_prefinal_loss` 的**字符串/整数比较 bug**（`=='1'` 对 int 恒假），
   已定位并修正；`summary.md` 与 `per_gt_attribution.csv` 中给出的是修正后的数字。
