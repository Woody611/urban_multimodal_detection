# P8.5 — F0 Score-Vacuum Upstream Audit

**日期** 2026-10-03 · **性质** ZERO GPU / 只读代码 + 已有 artifact 重统计 · 无 forward、无训练、无新 checkpoint。

## 1. Artifact availability

| 需要的量 | 是否存在 | 来源 |
|---|---|---|
| `target_scores` / `norm_align_metric` 的**直接**落盘 | ❌ | 从未落盘 |
| **`best_assign_iou`**（= TAL 正样本最大 CIoU，**就是 target 的等价量**，见 §2） | ✅ | `small_object_domains/_v6_domains.json`（V1 全 2807 val GT，含 small） |
| `best_align` / `n_pos` / `pos_target` / `center_logit` | ✅ | 同上 |
| 逐 epoch 的 `ciou_pos_max` / `cls_pos_max` / `raw_align_pos_max`（1320 medium） | ✅ | `full300_trajectory_probe/from_snapshots/*.npz` × 30 epoch |
| failure 桶（F0/F1/F2/F3/MATCHED） | ✅ | `p3_disappearance_audit/attribution.json` |
| **背景/全图 anchor 的 target** | ❌ | **NOT MEASURABLE FROM EXISTING ARTIFACTS** |

**几何 caveat**：`_v6_domains` 是 **square**（Sep 26），`attribution.json` 是 **rect**（Oct 3）⇒ §3/§8/§9 的 join 是**跨几何**的；
§11 的时间分析用的是 rect 轨迹（同几何）。

---

## 2. Exact D′ classification-target pipeline（从当前 repo 源码导出，非记忆）

`ultralytics/utils/tal.py`（D′ 未改动该文件）：

```python
# get_box_metrics (tal.py:150,155)
overlaps     = bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).clamp_(0)   # ★ CIoU，非 IoU
align_metric = bbox_scores.pow(alpha) * overlaps.pow(beta)                      # α=0.5, β=6.0

# forward (tal.py:112-116)
align_metric *= mask_pos
pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)      # (b, n_max_obj, 1)
pos_overlaps      = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + eps)).amax(-2)
target_scores     = target_scores * norm_align_metric.unsqueeze(-1)            # ★
```

**关键代数（本轮推导）**：在 `align` 取到该 GT 最大值的那个 anchor 上 `align = pos_align_metrics`，故

```
norm_align_metric  ==  pos_overlaps  ==  该 GT 正样本 anchor 上的最大 CIoU
```

**⇒ 一个 GT 在其正样本上的最大 classification target ≡ 它的最大正样本 CIoU。**

- 非 GT 类 target = 0（one-hot 基底）
- 背景 anchor target = 0（`fg_mask` 外）
- α=0.5 只进入 `align`（影响 top-k 选择），**不进入 target 的数值**
- `loss.py:415`：`self.bce = nn.BCEWithLogitsLoss(reduction="none")`，`cls_pw: []` ⇒ 普通 BCE、无类权重
- 归一化：`loss[1].sum() / target_scores_sum`

---

## 3–6. Target-score distribution / 分解 / β=6 / target→logit

**核心（V1，按 size × failure，target = `best_assign_iou`）：**

| size | bucket | n | **target 中位** | p10 | p90 |
|---|---|---:|---:|---:|---:|
| **small** | **F0** | 95 | **0.5497** | 0.1611 | 0.8252 |
| small | MATCHED | 222 | **0.7667** | 0.5822 | 0.8892 |
| medium | F0 | 155 | 0.7233 | 0.5093 | 0.8708 |
| medium | MATCHED | 1064 | 0.8203 | 0.6642 | 0.9330 |
| large | F0 | 51 | 0.8174 | 0.6577 | 0.9585 |
| large | MATCHED | 1009 | 0.9060 | 0.7320 | 0.9732 |

**small 内三档**：F0 **0.5497** / F1 0.3587 / F3 0.3781 / MATCHED **0.7667**。

**§8 β=6 的精确作用域**（本轮的关键澄清）：

```
target  = CIoU            （线性，β 不参与）
align   ∝ cls^0.5 · CIoU^6 （β=6 只在这里）
```

| IoU | IoU¹ (target) | IoU⁶ (align) |
|---|---|---|
| 0.50 | 0.5000 | **0.01562** |
| 0.70 | 0.7000 | 0.11765 |
| 0.81 | 0.8100 | 0.28243 |

**⇒ β=6 把"定位差异"放大的是【正样本选择】，不是【监督目标】。**
而 P5 已证明选择层**饱和**：`mask_pos_membership ≡ 1.000`、`n_pos ≡ 10.0` 对所有 cohort/epoch 成立。

**⚠ 几何 caveat**：`ciou_pos_max` 是 target 的**下界**（anchor 级 `amax(-2)` 可能被同 anchor 上更强的 GT 抬高）；
但三个 cohort 走同一算法，比较仍然有效。

---

## 7. BCE gradient analysis（§11）—— **决定性**

`∂L/∂z = sigmoid(z) − y`，`z = center_logit`，`y = best_assign_iou`：

| bucket | n | z 中位 | σ(z) | **y (target)** | **\|grad\| 中位** |
|---|---:|---:|---:|---:|---:|
| **small / F0** | 95 | −12.229 | 0.0000 | 0.5497 | **0.5495** |
| small / MATCHED | 222 | +1.102 | 0.7507 | 0.7667 | **0.1269** |
| **medium / F0** | 155 | −12.735 | 0.0000 | 0.7233 | **0.7178** |
| medium / MATCHED | 1064 | +1.898 | 0.8697 | 0.8203 | **0.0872** |
| **large / F0** | 51 | −14.065 | 0.0000 | 0.8174 | **0.8109** |
| large / MATCHED | 1009 | +2.296 | 0.9085 | 0.9060 | **0.0803** |

**⇒ F0 组拿到的 BCE 梯度是 MATCHED 组的 4–10 倍。**
"低 target ⇒ 弱监督"这个链条的**最后一环方向相反**：F0 的监督**远强于**健康组（因为预测离 target 很远）。

---

## 8. Small / medium / large

见 §3 表。**target 差距随 size 增大而缩小**（small −0.217 / medium −0.097 / large −0.089），
**但即便是 small，target 0.55 仍是一个「强正样本」目标**，不是被压到 0。

---

## 9. Class-level analysis

F0 率按类（V1）：

| class | n | F0 率 |
|---|---:|---:|
| ball | 19 | 31.6% |
| sign | 150 | 26.0% |
| bicycle | 106 | 20.8% |
| garbage_can | 57 | 19.3% |
| light | 244 | 16.0% |
| seat | 107 | 14.0% |
| animal | 731 | 8.9% |
| person | 1045 | 8.4% |
| car | 288 | 5.2% |
| uav | 30 | 3.3% |
| boat | 28 | 0.0% |

**⇒ F0 率与 class frequency 无单调关系**（person 4426 训练 GT 却只有 8.4%；uav 174 只有 3.3%；boat 107 为 0%）。
**class frequency 不是 F0 的 trigger**（与 P5.2 的 `corr(freq, Δ(W·h)) = −0.021` 一致）。

## 10. F0 vs F3

F3（存活但匹配失败）的 target **0.3781**，**比 F0 的 0.5497 更低**。
⇒ F3 才是"定位差"的那一批，F0 的相对定位质量**更好**。**不能用 F3 解释 F0。**

---

## 11. Temporal precedence（§13/§14 硬门槛）

同一批 GT（ep290 的 C1=dead∧F0 与 C3=healthy）逐 epoch：

| ep | **C1 target** | **C1 logit** | C3 target | C3 logit |
|---:|---:|---:|---:|---:|
| 0 | **0.7793** | −6.451 | 0.8394 | −1.325 |
| 10 | 0.7745 | −5.701 | 0.8385 | −0.309 |
| 100 | 0.7367 | −10.463 | 0.8253 | +0.857 |
| 200 | 0.7310 | −13.973 | 0.8185 | +1.507 |
| 290 | **0.7252** | **−16.032** | 0.8108 | +1.705 |

**⇒ C1 的 target 全程只降 0.054（几乎平），而 logit 塌了 −9.58 nats（相差 ~180×）。**

```
TEMPORAL PRECEDENCE NOT ESTABLISHED
```

**target 既没有「先降」，幅度也差两个数量级 ⇒ 它不可能是 logit 塌缩的上游。**
（C1 vs C3 的 target 差在 **ep0 就已存在**：0.779 vs 0.839 —— 是**既有属性**，不是训练中产生的因果上游。）

---

## 12. Candidate mechanism

```text
MECHANISM NOT SUPPORTED
```

被否证的链条：

```
poor localization → low CIoU → CIoU^6 → low target → weak cls supervision → low logit → F0
                                              ▲
                              ✗ 这一步在两个方向上都不成立：
                                 (a) target 并没有被压到"弱"
                                 (b) 实际梯度反而更强
```

## 13. Falsification tests

| 规则 | 是否触发 |
|---|---|
| **F2**（overlap ≈ healthy 且 target 差异小） | **部分**（medium/large 差异仅 0.09；small 0.22） |
| **F4**（logit 下降明显早于/快于 target 下降） | ✅ **触发** —— target Δ=−0.054 vs logit Δ=−9.58 |
| F3（target 与 logit 无稳定关系） | ✅ 时间上不共变 |
| F6（F0 由与 TAL 无关的因素解释） | ✅ 梯度方向相反 |

**⇒ 规则 F4 单独即足以关闭。**

---

## 14. Relation to P5/P6/P7

- **P5**：assignment 饱和（`n_pos≡10`）—— 本轮补充：**饱和在 β=6 加剧之前就成立**，因为 β 只作用于选择层。
- **P6**：classifier norm/cosine/温度被**单调性**关闭 —— 与本轮无关（本轮在 target 侧）。
- **P7**：dead∧F0 的 feature 在健康流形下尾（51% 正确）—— 与本轮一致：**监督不缺、目标不低，但特征/分类输出仍低**。
- 三线汇合于同一结论：**F0 不是"上游监督被削弱"造成的，而是下游输出本身的问题**，且其上游成因 P5.2 判为 `NOT IDENTIFIED`。

---

## 15. Minimal intervention

```text
N/A —— 机制未被支持，不提出任何 intervention。
```

（brief §19：**必须先把机制与干预严格分开**。本轮机制未成立。）

**并且 §22/§23 的两个陷阱也已检查**：
- 提高 target 不会带来"免费提升" —— 因为 F0 组的梯度**本来就已经是健康组的 4–10 倍**，再加只会把更多低质量框推高 confidence、增加 FP 与 top-100 污染（官方 evaluator 按 confidence 降序贪心匹配）。
- 也不存在"降低 β"的依据 —— β 只作用于**已饱和**的选择层。

---

## 16. GPU Gate

| Gate | 要求 | 结果 |
|---|---|---|
| 1 | 存在 overlap→target→supervision→logit→F0 链条 | ❌ **链条在 supervision 环节断裂** |
| 2 | 至少一个环节有明显 cohort separation | ⚠ 只有 target 有（small −0.22），但方向被梯度证据推翻 |
| 3 | 存在 temporal precedence | ❌ **F4 触发（方向相反）** |
| 4 | 机制针对 F0 而非 F3 | ⚠ F3 的 target(0.378) 比 F0(0.550) 更低 ⇒ 若真有此机制，应先在 F3 显形 |
| 5 | BCE+TAL 有可改参数 | ✅（β、target floor 等） |
| 6 | 存在最小可证伪 intervention | ❌（机制未成立） |
| 7 | 影响 F0 rate 而非仅 training loss | ❌ |

```text
GPU NO-GO
```

## 17. Final Verdict

```text
NO-GO

PRECISE CLOSURE:
  杀死机制的是 §7 的 BCE 梯度核算 —— 方向完全相反。
  F0 组的分类监督【不缺】：|∂L/∂z| 中位 = 0.55(small)/0.72(medium)/0.81(large)，
  而 MATCHED 组只有 0.13/0.09/0.08 —— F0 组拿到的是 4–10 倍【更强】的梯度。
  辅证：§11 temporal —— target 全程只降 0.054，logit 塌了 −9.58 nats（差 ~180×），
  且 C1 与 C3 的 target 差在 ep0 就已存在 ⇒ 是既有属性、不是训练中的因果上游（F4 触发）。
  补充：β=6 的放大只作用于【已饱和的选择层】(n_pos≡10)，不作用于 target 数值；
  F3 的 target(0.378) 比 F0(0.550) 更低 ⇒ 定位差的那批是 F3 不是 F0。
```

**⇒ TAL 把 localization quality 转换成 classification supervision strength，这条通路存在，
但它不是 D′ score-vacuum / F0 的上游成因。该路线关闭。**
