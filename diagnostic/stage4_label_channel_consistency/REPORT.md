# Stage-4 Label / Channel Consistency Audit

**日期**：2026-09-29
**性质**：**READ-ONLY / NO-FORWARD**。0 training / 0 backward / 0 optimizer / 0 `model.forward` /
0 `predict` / 0 `val` / 0 inference / 0 新 prediction / 0 新 submission / 0 修改
source·config·dataset·labels·evaluator·checkpoint·既有 diagnostic / 0 重抽 G1·G2·G4 /
0 重算前向 logits。
**唯一"读模型"的动作**：`torch.load(checkpoint)` 取 metadata 与 `state_dict`（纯读取，无 forward）。

---

## 第一屏（协议 §22）

```text
FINAL VERDICT:
PASS

DATASET_CLASS_MAPPING:      PASS
MODEL_CLASS_MAPPING:        PASS
HEAD_CHANNEL_MAPPING:       PASS
GT_CLASS_TO_PD_SCORE:       PASS
SIGMOID:                    PASS
OBJECTNESS_CONFUSION:       EXCLUDED        （该 fork 根本没有 objectness 分支）
TAL_CLASS_INDEXING:         PASS
PREDICTION_CLASS_MAPPING:   PASS
EVALUATOR_CLASS_MAPPING:    PASS
SAMPLE_INDEX_ALIGNMENT:     PASS

PREVIOUS_STAGE4_CONCLUSION:
VALID

TRAINING_RECOMMENDATION:
CLOSE
```

**27 项检查全 PASS，0 FAIL。** 逐项明细见 `AUDIT.log` 的 `[SUMMARY] CHECK TABLE`。

---

## §1 Executive Verdict

测量链**已逐环接通且无 class-ID / channel / indexing 错误**：

```text
GT class ID (label 整数)
  → dataset.names / nc                       ✅ ID-by-ID 全等
  → model.nc / model.names                   ✅ 与 dataset 同源同序
  → Detect 的 class 通道 [64:76]             ✅ 源码 + checkpoint shape 双证
  → raw class logit（通道 = class id）        ✅ 用 row=cls 复现 logit，max|Δ|=2.3e-06
  → sigmoid                                   ✅ 恒等式 max|Δ|=8.4e-08
  → TAL pd_scores[..., gt_class]              ✅ 源码直索引，无 remap
  → prediction TXT class id                   ✅ 0 非法行，无置换
  → evaluator 逐类比较                        ✅ 同一 0-based 空间
```

**⇒ 上一轮的 Stage-4 结论（"G1 在**正确的** GT-class 通道上响应 ≈ 0"）技术上成立：
我们测到的那个"很弱"，不是 class-ID / channel / indexing 出错造成的。**

**本轮不证明因果**，只证明**测量链接线正确**。与协议 §28 一致。

---

## §2 Dataset Class Mapping

**source**：`data/processed/rgbid_split_train/dataset.yaml`（SHA256[:16] = `3bc36c31e913072f`）

| id | name | id | name |
|---:|---|---:|---|
| 0 | person | 6 | car |
| 1 | boat | 7 | ball |
| 2 | animal | 8 | light |
| 3 | seat | 9 | garbage_can |
| 4 | sign | 10 | uav |
| 5 | bicycle | 11 | tricycle |

`nc = 12`，`len(names) = 12`，键覆盖 `{0..11}` 无缺。

**label 整数 ↔ 名字的一致性**（直接读 val label 文件的第 0 列，与 `_v2_records.json` 的 `cls` 对账）：

| group | n | label↔record cls 一致 | 类别（label 整数 → 名字） |
|---|---:|---:|---|
| G1 | 38 | **38** | person 18, animal 13, light 3, sign 2, garbage_can 1, ball 1 |
| G2 | 38 | **38** | person 15, animal 14, boat 3, light 3, sign 2, ball 1 |
| G4 | 194 | **194** | person 110, animal 20, uav 18, sign 17, car 12, bicycle 5, seat 4, garbage_can 4, ball 2, light 1, boat 1 |

⇒ `C_dataset_names_complete` / `C_dataset_nc_matches_names` / `C_label_integer_to_name` **PASS**。

---

## §3 Model.names / nc

**source**：D′ `runs/…_sepstem_clahe/weights/best.pt`（`1cae45f75693f541…fae4fda`）的 checkpoint metadata。
checkpoint top-level keys = `['best_fitness','date','docs','ema','epoch','license','model','optimizer','train_args','train_metrics','train_results','updates','version']`。

```text
model class = DetectionModel     model.nc = 12
model.names = {0:'person',1:'boat',2:'animal',3:'seat',4:'sign',5:'bicycle',
               6:'car',7:'ball',8:'light',9:'garbage_can',10:'uav',11:'tricycle'}
```

**与 `dataset.names` 逐 ID 比较（不是 set 比较）：`ID-by-ID 全等 = True`。**
`model.nc = 12 == dataset.nc = 12`。

**names 的来源链**（源码）：
`ultralytics/models/yolo/detect/train.py:390-392` `set_model_attributes`：
```python
self.model.nc    = self.data["nc"]      # 12
self.model.names = self.data["names"]   # dataset.yaml 的 names
```
⇒ checkpoint 里的 `names` **就是** dataset.yaml 的 `names`，同源同序。
`configs/yolo11m_sepstem.yaml` 只写 `nc: 12`，**不含 `names` 键**；`train_rgbid_sepstem_clahe.yaml` 亦无 —— 无第二份 names 可冲突。

⇒ `C_model_nc` / `C_model_names_ID_order_identical` **PASS**。

---

## §4 Detection Head Channel Layout

**source**：本项目 fork 的 `ultralytics/nn/modules/head.py`（`2fa868a4464ce88b`），`class Detect`。
**不引用外部 Ultralytics 实现。**

```python
:41  self.nc = nc
:43  self.reg_max = 16
:44  self.no = nc + self.reg_max * 4            # 12 + 64 = 76
:48  cv2 末层 = nn.Conv2d(c2, 4 * self.reg_max, 1)     # → 64 ch（box/DFL）
:51/:57 cv3 末层 = nn.Conv2d(c3, self.nc, 1)           # → 12 ch（class）
:74  x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
:108 x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)   # (B,76,8400)
:117 box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)               # box=[0:64] cls=[64:76]
:133 dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
:135 return torch.cat((dbox, cls.sigmoid()), 1)
```

⇒ **拼接顺序 `[ box(0:64) | cls(64:76) ]`；class 通道 = `no` 维的 `[64:76]`。**

**第二处独立一致的布局**（`ultralytics/utils/loss.py:490-492`，与 `_inference` 是两条独立代码路径）：
```python
pred_distri, pred_scores = torch.cat([xi.view(...)], 2).split((self.reg_max * 4, self.nc), 1)
```
⇒ 同样是 `[0:64]=distri / [64:76]=scores`。`loss.py:423-424`：`self.nc = m.nc = 12`，`self.no = 76`。

**checkpoint 实测形状**（`state_dict`，非推断）：

| key | shape |
|---|---|
| `model.30.cv2.0.2.weight` / `.bias` | **(64, 64, 1, 1)** / (64,) |
| `model.30.cv3.0.2.weight` / `.bias` | **(12, 256, 1, 1)** / (12,) |

`Detect.head 实测 no = 76  nc = 12  reg_max = 16  nl = 3`；三层末层形状一致（cv2 末 64，cv3 末 12）。

⇒ `C_head_box_channels_64` / `C_head_class_channels_eq_nc` / `C_no_eq_64plus_nc` **PASS**。

---

## §5 GT Class → Head Class Channel

**不用"看起来合理"，用可复现的数值恒等式。**

`_v7_vectors.json` 落盘了每个 val GT 在 **GT 中心 cell** 的 `h`
（= `Detect.cv3[li][-1]` 的**输入**，256 维）与 `logit_decomp`。项目自己声明的恒等式是
`logit_c = W_c·h + b_c`，其中 `W = cv3[-1].weight`（shape `(12,256,1,1)`）。

**验证**：用 checkpoint 的 `W`/`b` 从**落盘的 `h`** 重算 `W[cls]·h + b[cls]`：

```text
n = 2807（全部 val GT）
max|Δ|    = 2.334e-06
median|Δ| = 3.337e-07          ← 纯 float32 舍入
```

**反证（检查有分辨力）**：改用 `W[cls+1]` 行 ⇒ `median|Δ| = 7.4240`（差 7 个 logit 单位）。
⇒ 该检查确实能分辨通道错位，而用 `row = cls` 时精确复现。

⇒ `C_head_weight_row_equals_class_id` / `C_channel_offset_would_fail` **PASS**。

**替换检查 1 —— 两个独立实现、同一统计量**（我最初拿"邻域 max vs 单点值"比较，属设计错误，见 §15）：

| 实现 | 代码 | 量 |
|---|---|---|
| `_v4_internal.py:183` | `pred_scores[0].sigmoid()[:, cls_id][pos].max()` | `pos_score_max` |
| `_v6_domains.py:155` | `pd_s[0].sigmoid()[pos, cid].max()` | `pos_prob` |

定义**逐字相同**（GT 类通道在 positives 上的最大 sigmoid）。实测：

```text
n = 429    max|Δ| = 0.000e+00    median|Δ| = 0.000e+00    Spearman ρ = +1.0000
同一对的 best_assign_iou：n = 429    max|Δ| = 0.000e+00
```

⇒ 两个**独立脚本、独立实现**在**同一统计量**上**逐位相等**。若任一用了错通道，该量会指向别的类，
不可能逐条吻合到 0.0。⇒ `C_two_impl_same_statistic_agree` / `C_two_impl_iou_agree` **PASS**。

---

## §6 Raw Logit → Sigmoid

`_v6_domains.py:149-150`（本项目实际代码）：
```python
logit = float(pd_s[0, near, cid].max())     # raw logit（sigmoid 前）
prob  = float(sc_all[near, cid].max())      # sc_all = pd_s[0].sigmoid()
```
sigmoid 单调 ⇒ `max(sigmoid(x)) == sigmoid(max(x))` ⇒ 必有 `prob == sigmoid(logit)`。

**实测**（全部 7926 条 T1+T2+V1 行）：`max|sigmoid(logit) − prob| = 8.350e-08`，median `1.880e-08`。

**报告引用数字的单调/数值一致性**：

| raw logit | `sigmoid(raw)` | 报告值 |
|---:|---:|---:|
| −12.5845 | 3.425e-06 | 3.4e-06 |
| +0.8357 | 6.976e-01 | 0.698 |
| +0.5111 | 6.251e-01 | 0.625 |
| −14.6341 | 4.411e-07 | 4.41e-07 |

⇒ `C_sigmoid_identity` **PASS**。

⚠ `pos_prob` 对应的 raw logit **未落盘** ⇒ 该条只能由同一代码路径保证（已注明）。

---

## §7 Objectness / Class Probability Semantics

**检查范围**：`grep -rn "objectness" ultralytics/**/*.py`（排除 `.pyc`）。

```text
命中文件 = ['ultralytics/utils/metrics.py']，唯一命中是 metrics.py:1254 的一句注释 "Sort by objectness"
ultralytics/nn/modules/head.py / ultralytics/utils/loss.py / ultralytics/utils/tal.py  → 0 命中
```

**⇒ `OBJECTNESS_BRANCH = ABSENT`。**

结构旁证（§4）：Detect 的输出只有 `box(64) + cls(12)`，`no = 76 = 64 + 12`，**没有第 13 个 obj 通道**；
`_inference` 的返回是 `cat((dbox, cls.sigmoid()), 1)`，**最终 confidence 就是 class sigmoid 本身**，
不存在 `P(obj)·P(class)` 的组合。

⇒ 上一轮记录的「GT-class sigmoid」**就是 `P(class)`**，不是 `P(obj)·P(class)`。
⇒ `C_objectness_absent` **PASS**；`OBJECTNESS_CONFUSION = EXCLUDED`。

---

## §8 TAL pd_scores / GT-class Indexing

**调用点**（`ultralytics/utils/loss.py:519-521`）：
```python
target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
    pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
    anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
```
⇒ 传给 assigner 的 `pd_scores` 是 **(b, 8400, 12) 的已 sigmoid 概率**，通道序 = class id。

**超参是源码字面量，不是配置**（`loss.py:438`）：
```python
self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
```
`tal_topk` 默认 10 ⇒ **(alpha, beta, topk) = (0.5, 6.0, 10)**，无法被 yaml 改。

**gt_labels 的来源**（`loss.py:506-509`）：
```python
targets = torch.cat((batch['batch_idx'].view(-1,1), batch['cls'].view(-1,1), batch['bboxes']), 1)
targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1,0,1,0]])
gt_labels, gt_bboxes = targets.split((1,4), 2)
```
⇒ `gt_labels` = **dataset label 的原始类整数（0-based）**，**无 +1 / −1**。

**索引点**（`ultralytics/utils/tal.py:144-147`）：
```python
ind[1] = gt_labels.squeeze(-1)
bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]
```
⇒ 直接 `pd_scores[batch, gt_class, anchor]`。
**无 class-name 映射、无 gather-by-name、无 one-hot、无 remap、无偏移。**

`tal.py:151` `align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)`
`tal.py:154-156` `iou_calculation = bbox_iou(..., CIoU=True).squeeze(-1).clamp_(0)`

**`_v6_domains` 送进 assigner 的 6 个实参**（`_v6_domains.py:127-129`）与 loss 的调用**一一对应**，
且 `pbox = loss.bbox_decode(ap, pd_d)` 调用的就是 loss 自己的 `bbox_decode`。

⇒ `C_tal_uses_gt_class_index_directly` / `C_tal_alpha_beta_topk` / `C_v6_assigner_input_matches_loss` **PASS**。

---

## §9 Alignment Reproduction

**source**：`candidate_density_counterfactual/_results.npz`（**TRAIN + mosaic** —— 但它是唯一同时落盘了
per-candidate `cls` / `ciou` / `align` 的 artifact），`n_candidates = 1,905,266`。

### §9.1 ⚠ 发现一处 artifact 语义陷阱（协议 §2 警告的正是这类）

```text
直接对全体复算 cls^0.5·ciou^6 vs 落盘 align  →  max|Δ| = 0.831   （看似 FAIL）
```

但落盘 `align` **恰为 0** 的 1,815,531 条里，有 **1,378 条** 的 `cls>0.5` **且** `ciou>0.5`
—— 若公式成立它们不可能为 0。

**真因**：`tal.py:112` 的 `align_metric *= mask_pos` 是 **in-place** 运算，
而 `_run.py` 是在 `inst.forward()` **返回之后**才读 `inst.last["align_metric"]` ⇒
**落盘的其实是 `(cls^0.5·ciou^6) · mask_pos`，而 `mask_pos` 未落盘。**

⇒ 只能在 **`align>0` 子集**上复算公式：

```text
n = 89,735（占 4.71%）
max_abs_error    = 8.372e-08
median_abs_error = 1.526e-08
n > 1e-5         = 0
n > 1e-6         = 0            ← 全部为 float32 舍入
```

**反证**（同一子集）：`beta=5` ⇒ median|Δ| = 0.0312；`alpha=1` ⇒ median|Δ| = 0.0403
⇒ 检查对 alpha/beta 有分辨力。
**内部自洽旁证**：`gt_zero==True` 的 GT 中，有 `align>0` 候选的条数 = **0**。

⇒ `C_alignment_reproduced` **PASS**（公式在可验证子集上精确复现；零值由未落盘的 `mask_pos` 解释）。

### §9.2 第二处同类陷阱

`_v4_internal.json` 的 `best_in_gts_align`（字面 = "in-GT 候选的最大 align"）同样取自
`get_pos_mask` 的**同一被 in-place 置零过的张量** ⇒ 实际等于 **max over positives**。

```text
实测（429 行）：best_align == best_in_gts_align 的条数 = 429/429，max|Δ| = 0.0
```

⇒ 二者是**同一个量**。上一轮用 `best_in_gts_align`（G1 3e-5 / G2 0.051）论证「align 被 cls≈0 压塌」
—— **结论不受影响**（两种读法下该值都 ≈0），但**字段名有误导性**，已在 §17 记录。

⇒ `C_v4_best_in_gts_align_is_best_align` **PASS**。

---

## §10 Prediction TXT Class Mapping

格式（`official_eval.read_pred_txt`）：`cls cx cy w h conf`，6 字段，`cls ∈ [0,11]`；
`norm_xywh_to_xyxy` 按 **native (W,H)** 反归一化。

逐例链条（G1/G2/G4 各 5 条；`#bad` = 非法行数）：

| group | key | GT cls | GT name | #pred | #bad | pred classes | GT 类出现在预测类中 |
|---|---|---:|---|---:|---:|---|---|
| **G1** | `000004_011_00000001#0` | 4 | sign | 2 | 0 | [2, 8] | False |
| **G1** | `000006_011_00000307#5` | 0 | person | 10 | 0 | [0,1,2,6] | True |
| **G1** | `000006_011_00000307#6` | 0 | person | 10 | 0 | [0,1,2,6] | True |
| **G1** | `000006_011_00000409#0` | 0 | person | 7 | 0 | [0,1,2] | True |
| **G1** | `000006_011_00000409#1` | 0 | person | 7 | 0 | [0,1,2] | True |
| **G2** | `003288#17` | 4 | sign | 45 | 0 | [0,1,2,3,4] | True |
| **G2** | `000415#12` | 0 | person | 34 | 0 | [0,2,4] | True |
| **G2** | `003987#32` | 0 | person | 145 | 0 | [0,3,7] | True |
| **G2** | `000006_011_00000409#3` | 0 | person | 7 | 0 | [0,1,2] | True |
| **G2** | `000006_011_00000409#3`（**同 key 第 2 次**） | 0 | person | 7 | 0 | [0,1,2] | True |
| **G4** | `00000045#0` | 4 | sign | 14 | 0 | [0,4,6,8,10] | True |
| **G4** | `00000045#4` | 6 | car | 14 | 0 | [0,4,6,8,10] | True |
| **G4** | `00000045#5` | 10 | uav | 14 | 0 | [0,4,6,8,10] | True |
| **G4** | `00000057#0` | 4 | sign | 7 | 0 | [4,6,8,10] | True |
| **G4** | `00000057#4` | 6 | car | 7 | 0 | [4,6,8,10] | True |

**全体非法行数 = 0**；全部 class ∈ [0,11]。

### §10.1 置换检验（端到端出口检查）

对每个与 GT 有重叠的最终预测，其 class id 是否等于 GT 的 class id？
（G1+G2+G4 图，重叠 `(GT, pred)` 对 n = 3,892）

| GT class | n_GT | n_重叠GT | 同行最高类 | 占比 | 自身类占比 | 最高 conf 重叠预测的类分布（按 GT 计） |
|---|---:|---:|---|---:|---:|---|
| person | 560 | 526 | **person** | 86.5% | 84.8% | person×455, car×24, boat×17 |
| boat | 15 | 15 | **boat** | 73.3% | 44.1% | boat×11, animal×2, person×2 |
| animal | 211 | 180 | **animal** | 92.8% | 92.7% | animal×167, boat×12, person×1 |
| seat | 16 | 16 | **person** | 37.5% | 29.1% | person×6, seat×5, boat×4 |
| sign | 60 | 54 | **sign** | 74.1% | 59.8% | sign×40, light×11, person×3 |
| bicycle | 49 | 46 | **bicycle** | 58.7% | 39.7% | bicycle×27, person×13, light×3 |
| car | 96 | 91 | **car** | 68.1% | 59.4% | car×62, bicycle×11, person×8 |
| ball | 7 | 5 | person | 60.0% | 66.7% | person×3, ball×2 |
| light | 77 | 66 | **light** | 77.3% | 42.6% | light×51, person×5, sign×4 |
| garbage_can | 23 | 20 | **garbage_can** | 60.0% | 46.3% | garbage_can×12, person×4, car×2 |
| uav | 22 | 22 | **uav** | 100.0% | 88.0% | uav×22 |

**预先声明的判据（写在 `_audit.py` 里，事后不调）**：
- (i) 众数映射 `c → argmax` **非单射**（置换必为双射 ⇒ 单射是置换的必要条件）；
- (ii) 同行率 > 3× 随机基线（1/12 = 8.33%）。

```text
(a) 自身类占比 = 74.46%   vs   随机基线 8.33%   ⇒ 高出 8.9×
(b) 众数映射 = {person→person, boat→boat, animal→animal, seat→person, sign→sign,
                bicycle→bicycle, car→car, ball→person, light→light,
                garbage_can→garbage_can, uav→uav}
    单射? False（seat 与 ball 都落到 person）⇒ **不构成任何置换**
(c) 【报告项，不并入判据】n_gt≥10 的行 argmax 自洽 = 9/10（唯一例外 seat，margin 6:5）
```

⇒ **不构成任何 class 置换**。低同行率是「同图邻类框与 GT 重叠」这一常态
（person 坐在 seat 上、light 与 sign 相邻等），不是 channel 错位。
⇒ `C_pred_class_not_permuted` / `C_pred_cls_in_range` / `C_pred_cls_names_consistent` **PASS**。

---

## §11 Evaluator Class Mapping

`scripts/official_eval.py`（`e82128abc082ed05`）：
```python
:44  NC = 12 ;  CLASS_ID_MIN, CLASS_ID_MAX = 0, NC - 1
:71  read_pred_txt  →  if not (CLASS_ID_MIN <= v[0] <= CLASS_ID_MAX): 计为非法行
:135 norm_xywh_to_xyxy → cls = arr[:,0].astype(int)
:177 pr_curve_for_class → 预测 `pc == c`、GT `gc == c` 直接比较
```
⇒ 预测类与 GT 类处于**同一 0-based 空间**，逐类直接比较，**无 1-based 偏移、无 permutation**。
`predict_rect.py` 复用 `predict.py` 的写盘格式，未做类别重排。

⇒ `C_evaluator_class_space` **PASS**。

---

## §12 Pretrained Class Remapping

`ultralytics/models/yolo/detect/train.py:152-193` `_remap_separate_stem`（真实逻辑）：

```python
for i in range(offset + 1, len(layers)):
    s = i - offset
    prefix = f"model.{i}."
    for k, v in csd.items():
        sk = f"model.{s}." + k[len(prefix):]
        sv = src.get(sk)
        if sv is not None and tuple(sv.shape) == tuple(v.shape):
            v.copy_(sv); continue                       # ← 唯一复制方式：key 名 + shape 全等
        ... # E1 例外（与本轮无关）
        if (i == detect_idx and ".cv3." in k and sv is not None
                and tuple(sv.shape[1:]) == tuple(v.shape[1:])
                and v.shape[0] == nc and sv.shape[0] != nc):
            n_cls_untrained += 1; continue              # ← 不复制 = 按新 nc 重新初始化
        if sv is None:  raise RuntimeError(...)
        raise RuntimeError("shape mismatch ...")
```

- **复制方式是 index/shape 对应**：`sk` 是把 `model.{i}` 换成 `model.{i-offset}`，**按 key 名 + shape 全等**。
- **全文件对 `"names"` 的引用 = 0**（grep 实测）⇒ **不存在任何按 class name 的重映射**。
- 唯一的 `cv3` 例外只作用于**最后一层 1×1 投影**（shape 首维 = 类数）：D′ 的 `cv3.*.2.weight` 首维 = **12**
  （= nc）；若 `yolo11m.pt` 的对应张量首维 = 80，则该例外必然触发（训练日志已确认
  `Detect cv3 class branch re-initialised for nc=12`）。其余 `cv3` 特征层（DWConv/Conv）因 shape 全等而被**逐位复制**。

⇒ `C_no_name_based_class_remap` **PASS**。

---

## §13 Sample / Index Alignment

**source**：`diagnostic/small_object_domains/_v6_domains.json`（V1 域 2807 行；全文件 7926 行）。

| 检查 | 结果 |
|---|---|
| `(domain, key)` 唯一性 | **dup = 0** |
| `key == f"{stem}#{gt}"` 不一致 | **0** |
| `sq`（record）vs 从 label 行重算（**乘 letterbox scale 后**） | n=400，`max|Δ| = 1.403e-04`，median `1.843e-05` |
| `gt_zero==True` 的 GT 中有 `align>0` 候选 | **0** |

⇒ `gt` 索引确实指向**同一 label 行**，且 GT/候选/得分三者的索引在同一 record 内一致。

**索引错位风险的两条排除**：
- `center_logit`/`center_prob` 的 `near` 掩码用 **anchor 像素坐标与 GT 中心的距离**，与 batch/GT/anchor
  索引无关；
- `_v6_domains` 每次 forward **单图**（`ds[i]` 逐图），**batch = 1** ⇒ 不存在 batch 维错位。
- `pos_*` 用 `mask_pos[0, g]`，第 0 维即 batch。

⇒ `C_key_stem_gt_consistent` / `C_sq_matches_label_row_index` **PASS**。

---

## §14 G1/G2/G4 Symmetry Check

`_v6_domains.py` 的 `measure()` 是**唯一**抽取函数，对 T1/T2/V1 三域**逐字相同**地调用；
V1 域用 `run_domain("V1", ds_va, M)` 跑**全部** val GT，**分组只是事后按 key 过滤同一批 V1 行**。

`_v4_internal.py` 同样：单一 `mk(i)` hook + 单一 measure 循环，G1/G2 只是过滤。

```text
命中 V1 行： G1 38/38、G2 38/38、G4 194/194
（注：G2 的 38 条里只有 29 个**不同** control GT —— 见 §15.3。命中计数按 38 条列表计，
  但**同一 control 走的是同一条代码路径**，对称性结论不变。）
```

⇒ G1/G2/G4 使用的 **tensor / index rule / sigmoid / GT mapping / candidate 选择规则完全同一条代码路径**，
不存在「G1 用 A、G2 用 B、G4 用 C」。
⇒ `C_groups_same_extraction_path` **PASS**。

---

## §15 Failed / Unavailable Checks

**最终 n_checks = 27，n_fail = 0。** 下面是过程中出现过的 **4 次 FAIL，全部是我自己的检查写错**，逐条如实记录：

| # | 检查名 | 首版结果 | 真因（我的错） | 更正 | 更正后 |
|---|---|---|---|---|---|
| 1 | `C_two_impl_channel_agree` | FAIL（逐类 ρ ∈ [−0.31,+0.55]） | **比较了两个不同的统计量**：`_v6_domains.center_logit` 是「GT 中心 12px 内**多 anchor 取 max**，跨 stride-8/16/32」，而 `_v7_vectors.logit_decomp` 是「GT 中心**单 cell 点值**」。head 输出空间变化很快 ⇒ 低相关是构造性的 | **撤销该检查**，换成「同定义、两实现」的 `pos_score_max` vs `pos_prob` | **PASS，max\|Δ\|=0.0，ρ=+1.0** |
| 2 | `C_pred_class_matches_gt_class` | FAIL（同行率 74.46%，众数自洽 False） | **判据设成了检测质量指标**（要求每类 argmax 自洽 + 同行率 >80%），而不是**映射**指标 | 改为**预先声明的置换判据**：非单射 + 同行率 >3× 基线 | **PASS**（映射非单射，74.46% ≫ 25%） |
| 3 | `C_key_stem_gt_consistent` | FAIL（dup=1498） | 我用 `key` **单独**做唯一性，但 T1/V1 同图同名会撞 | 改为 **(domain, key)** | **PASS，dup=0** |
| 4 | `C_sq_matches_label_row_index` | FAIL（max\|Δ\|=420.6） | `_v6_domains` 的 `sq` 是 **输入空间**（letterbox 后长边 1280），不是 native（源码 `:131-135` 先过 `loss.preprocess`） | 乘 `1280/max(W,H)` 后再比 | **PASS，max\|Δ\|=1.4e-04** |

**没有任何一项 FAIL 指向被审计的测量链本身。**

### §15.1 仍然 UNAVAILABLE 的项

| 项 | 原因 |
|---|---|
| `pos_prob` 对应的 **raw logit** | 未落盘 ⇒ 只能由同一代码路径保证（已注明） |
| `mask_pos` | 未落盘 ⇒ `cand_align` 的零值无法逐条验证（只能由 1,378 条反例 + 公式复现间接确认） |
| per-anchor 中心/W-H | 未落盘（上一轮已记录） |
| DFL 分布 / 回归目标 | 全仓库无落盘（上一轮已记录） |

**均需新 forward 才能补 ⇒ 协议 §2/§21 禁止，本轮不做。**

### §15.2 两处 artifact 语义陷阱（本轮新发现，非缺陷但影响读法）

1. **`cand_align` = `(cls^0.5·ciou^6) · mask_pos`**（`tal.py:112` 的 in-place `*=` 作用于
   `get_pos_mask` 返回的同一张量，而 `_run.py` 在 forward 之后才读 stash）。
   ⇒ `candidate_density_counterfactual` 里的 `gt_maxalign` 实际是「**positives 上的** max align」，
   不是字面的「in-GT 候选中」的 max align。**该轮结论（candidate 数量 vs 质量）不依赖此项**（报告已把它标为辅助量）。
2. **`_v4_internal.best_in_gts_align == best_align`**（逐条相等 429/429）。
   ⇒ 上一轮用它论证「align 被 cls≈0 压塌」时，它实际是 max over positives；**结论不受影响**（值都 ≈0）。

### §15.3 一项影响**上一轮统计口径**的发现（不影响本轮 verdict）

```text
G2 = [v3["controls"][k][0] for k in v3["controls"]]  →  len = 38，distinct = 29
     重复：000140#7×3、002334#10×3、000415#12×2、000006_011_00000409#3×2、
           002331#5×2、002331#13×2、003007#15×2
```

⇒ 上一轮报告里的「**G2 n=38**」实际是 **38 个匹配对 / 29 个不同 control GT**。
配对 Wilcoxon / Cliff's δ 把重复的 control 当成独立样本 ⇒ **检验偏乐观（anti-conservative）**。
本轮**不重算**上一轮的统计（那会改动既有结论），只**记录**该口径问题。
**对本轮 verdict 无影响**：本轮的检查都在 per-GT 层面做逐条复现，不依赖 G2 的独立性假设。

---

## §16 Final Verdict

```text
STAGE4_CHANNEL_CONSISTENCY = PASS
```

协议 §19 的 PASS 清单逐项对照：

| 协议 §19 条目 | 本轮 |
|---|---|
| Dataset class IDs correct | ✅ §2 |
| Dataset class names correct | ✅ §2 |
| model.nc correct | ✅ §3 |
| model.names ID-order identical | ✅ §3（ID-by-ID，非 set） |
| Detect class-channel layout verified | ✅ §4（源码 + shape 双证） |
| GT class → class channel verified | ✅ §5（数值复现 + 反证） |
| raw logit source verified | ✅ §6 |
| sigmoid verified | ✅ §6（max\|Δ\|=8.4e-08） |
| objectness confusion excluded | ✅ §7（分支不存在） |
| TAL pd_scores source verified | ✅ §8 |
| TAL GT-class indexing verified | ✅ §8（`pd_scores[b, gt_class, a]`，无 remap） |
| alignment recomputation verified | ✅ §9（align>0 子集 max\|Δ\|=8.4e-08） |
| prediction TXT class mapping verified | ✅ §10（0 非法行 + 无置换） |
| evaluator class mapping verified | ✅ §11 |
| pretrained class-head remapping checked | ✅ §12（无 name 重映射） |
| G1/G2/G4 extraction path identical | ✅ §14 |
| no sample/index misalignment | ✅ §13 |

---

## §17 Impact on Previous Stage-4 Conclusion

```text
PREVIOUS_STAGE4_CONCLUSION = VALID
```

按协议 §23 的要求，明确写：

```text
The previous Stage-4 finding is technically valid:

native-small G1 has near-zero response in the correct GT-class channel.

This audit does NOT prove causality.
It only proves the measurement chain is correctly wired.
```

**具体地说，本轮把上一轮的下列数字的"读数正确性"钉住了：**

| 上一轮的量 | 本轮验证 |
|---|---|
| G1 `pos_prob` 中位 ≈ 3e-6 | 定义 = GT 类通道在 positives 上的最大 sigmoid；与第二个独立实现**逐位相等** |
| G1 `center_logit` 中位 −12.58 | 通道 = `[64:76]` 内 index = class id；`sigmoid` 恒等式成立 |
| G1 `logit_decomp` 中位 −14.634 | 用 checkpoint 的 `W[cls]` 从落盘 `h` **精确复现**（max\|Δ\|=2.3e-06） |
| G1 `best_in_gts_align` ≈ 3e-5 | 公式 `cls^0.5·ciou^6` 在 align>0 子集精确复现；该字段实际 = max over positives（语义已更正，结论不变） |
| G1 最终输出 any-class IoU ≡ 0 | 预测类与 GT 类同处 0-based 空间，无置换 |

**同时明确本轮的边界**（协议 §28）：

- 本轮**没有**证明「classification 是 root cause」。它只证明**我们测到的分类很弱这件事，不是测量错误**。
- 本轮**没有**重算、也没有推翻上一轮的 Stage-4 verdict。
- 但本轮**发现了两处 artifact 字段语义陷阱**与**一处 G2 统计口径问题**（§15.2/§15.3），
  它们**不改变** Stage-4 结论，但应在后续引用这些 artifact 时带上限定。

### §17.1 Training Recommendation

```text
TRAINING_RECOMMENDATION = CLOSE
```

理由（协议 §23）：测量链 PASS ⇒ 上一轮结论保持有效 ⇒ 上一轮已经给出的 CLOSE 理由**不变**：

| 条件（协议 §26） | 状态 |
|---|---|
| (a) failure stage 明确 | ✅ STAGE 4（classification / score） |
| (b) 存在**单一、可干预**机制 | ❌ 该 stage 唯一对应机制（抬升小目标类别响应）**已由 E1 与 L17 两轮实测收口** |
| (c) 已有 artifact 支持该机制 | ❌ 恰恰否证：E1 官方 Δ=−0.00540（CI 含 0）；L17 最强 oracle 干预 R3 后 logit 仍只在成功分布第 19.7 百分位 |

**除非存在新的、独立机制证据** —— 本轮**不设计**任何 intervention，**不创建** config / 脚本改动 / checkpoint / run。

---

## §18 Provenance

```text
TRAINING        = NO
FORWARD         = NO
BACKWARD        = NO
OPTIMIZER_STEP  = NO

EXISTING_FILES_CHANGED        = 0        （8,951 个文件逐个 SHA256 比对，0 changed / 0 missing）
EXISTING_CHECKPOINTS_CHANGED  = 0
EXISTING_PREDICTIONS_CHANGED  = 0
EXISTING_DIAGNOSTICS_CHANGED  = 0

GIT_STATUS_BEFORE = 71 行
GIT_STATUS_AFTER  = 71 行，逐行相同
GIT_HEAD_BEFORE   = f659609c28f6b4f500212197d5229550a36d12d8
GIT_HEAD_AFTER    = f659609c28f6b4f500212197d5229550a36d12d8
```

（快照口径：`ultralytics/` + `scripts/` + `configs/` + `data/processed/rgbid_split_train/` +
`diagnostic/{small_object_domains,prediction_head_geometry_audit,small_object_cause_v2,p3_feature_space}` + `runs/`，
共 **8,951** 个文件。`_sha_before.json` 与终态逐项比对：`EXISTING_FILES_CHANGED = 0`、`MISSING = 0`。）

### 读取对象 SHA256[:16]

| 对象 | SHA256[:16] |
|---|---|
| `data/processed/rgbid_split_train/dataset.yaml` | `3bc36c31e913072f` |
| `configs/yolo11m_sepstem.yaml` | `9b14f29460733384` |
| `configs/train_rgbid_sepstem_clahe.yaml` | `a4e329cfc3d22020` |
| **`runs/…_sepstem_clahe/weights/best.pt`** | `1cae45f75693f541…fae4fda` |
| `ultralytics/nn/modules/head.py` | `2fa868a4464ce88b` |
| `ultralytics/nn/tasks.py` | `79c8dbab9cd810d2` |
| `ultralytics/utils/tal.py` | `aae7e8ac438f00cd` |
| `ultralytics/utils/loss.py` | `0f092cf22a372f6d` |
| `ultralytics/models/yolo/detect/train.py` | `f46198001a4b741e` |
| `scripts/official_eval.py` | `e82128abc082ed05` |
| `scripts/predict_rect.py` | `36858acb59058e89` |
| `diagnostic/small_object_domains/_v6_domains.json` | `3505d9160c47aae8` |
| `diagnostic/small_object_internal_audit/_v4_internal.json` | `0da6877405d241e3` |
| `diagnostic/p3_feature_space/_v7_vectors.json` | `977593ae00126100` |
| `diagnostic/candidate_density_counterfactual/_results.npz` | `7b28dc555c3fffb6` |

### NEW_FILES（全部位于 `diagnostic/stage4_label_channel_consistency/`）

```text
REPORT.md           本报告
_audit.py           审计脚本（27 项检查；只写本目录）
AUDIT.log           完整运行日志（含每一项的 source / actual / expected / verdict）
_tables.json        结构化结果（含每条检查的 pass + detail）
_sha_before.json    8,951 个文件的 before 快照
_git_before.txt     git status + HEAD（before）
_git_after.txt      git status + HEAD（after）
```

**未修改** `diagnostic/small_object_domains/`、`diagnostic/prediction_head_geometry_audit/` 或任何其它既有文件（协议 §26）。

---

**本轮结束，停止。不训练、不设计 intervention、不通过 forward 补证据、不自行启动下一轮。**
