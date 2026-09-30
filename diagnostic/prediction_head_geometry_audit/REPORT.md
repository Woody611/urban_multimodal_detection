# Native-small G1 vs G2/G4 Prediction-Head Geometry Audit

**日期**：2026-09-29
**性质**：**READ-ONLY / NO-TRAINING / NO-FORWARD**。0 training / 0 backward / 0 optimizer.step /
0 新 checkpoint / 0 修改既有 source·config·dataset·label·evaluator·diagnostic·prediction TXT·checkpoint /
0 重新生成 submission / **0 重新 forward** / 0 重抽 G1·G2·G4 / 0 随机替换 control。
只读已有 TXT / GT / 既有 artifact，做纯数学分析；只写 `diagnostic/prediction_head_geometry_audit/`。

---

```text
FINAL VERDICT:
C

PRIMARY FAILURE STAGE:
STAGE 4 — classification / score
（同一批 anchor 上 box 分支已给出 IoU≥0.5 的框，而 GT 类的 sigmoid 概率中位 3e-6）

REGRESSION_GEOMETRY:
NOT_SUPPORTED

RANKING:
NOT_SUPPORTED

CANDIDATE_GEOMETRY:
NOT_SUPPORTED

TRAINING_RECOMMENDATION:
CLOSE
```

**⚠ C 的读法**：`CLASSIFICATION` 成立、**`RANKING` 不成立**（最终输出里没有任何候选，无从被排序）。
若只问"几何有没有失败"，答案同时是 **F — NO_CLEAR_GEOMETRIC_FAILURE**；本报告选 C，因为它额外指明了失败发生在哪一支。

**§24 的十个问题**（先答，再给证据）：

| # | 问题 | 答案 |
|---|---|---|
| Q1 | G1/G2/G4 candidate **数量**是否不同？ | **否**。`n_pos` 中位 5 / 5 / 7；G1 **无一条为 0**（§4） |
| Q2 | candidate **center** 误差是否不同？ | **UNAVAILABLE**（val 侧无 per-anchor 中心表）。但从最终预测侧：G1 最近的框中心距 GT 中心中位 **7.01 个 GT 对角线**（§5） |
| Q3 | candidate **width/height** 相对 GT 是否不同？ | **否**。框内 decoded 框与 GT 的 CIoU 中位 **0.6278**（G2 0.7156 / G4 0.7694）。CIoU≤IoU ⇒ 真 IoU 下界（§6） |
| Q4 | candidate **aspect ratio** 是否不同？ | **UNAVAILABLE**（无 per-anchor w/h 落盘）。G1 与 G2 的 **GT** AR 分布无差异（0.855 vs 0.838，p=0.775）（§12） |
| Q5 | candidate → final prediction 是否**系统性漂移**？ | **不是"漂移"，是"筛掉"**。24 条 G1 的 head 输出已含 IoU≥0.5 的框，**24 条在最终输出全部零重叠**（§7/§13） |
| Q6 | **width / height / center** 哪个贡献最大？ | 在 raw head 层**三者都不显著**（框已达标）。在最终输出层：G1 没有"最近的框"，`dcenter` 中位 7 个对角线（§8） |
| Q7 | 是否存在 **good candidate 但 final 选错**？ | **否**——最终输出里根本没有候选可被选中 ⇒ **RANKING 不成立**（§9） |
| Q8 | 是否存在 **geometry 合理但 class score/ranking 不足**？ | **是，这是主机制**。GT 类概率中位 **3e-6**，near-GT 同类 logit **−12.6**（§9） |
| Q9 | **failure stage** 落在哪里？ | **STAGE 4 — classification / score**（§14） |
| Q10 | 是否值得设计**新的训练干预**？ | **否 ⇒ CLOSE**。该 stage 唯一对应的可干预机制（抬升小目标类别响应）已由 E1 实测为无增益并收口（§15） |

---

## §1 Executive Verdict

**一句话**：**G1 的失败不在几何。** head 的框回归输出**已经**为 63% 的 G1 给出 IoU≥0.5 的框（CIoU 中位 0.6278，
保守下界），但这些框在最终输出里**一个都没有出现**；与此同时，**同一批 anchor 上 GT 类的 sigmoid 概率中位是 3×10⁻⁶**。
⇒ 失败发生在**分类分支的输出**（协议 STAGE 4），而不是回归、候选生成或排序。

本轮**推翻**了协议 §0 的历史前提（"存在 candidate 但预测框几何与 GT 严重不匹配 ⇒ 怀疑 regression bottleneck"）：

```text
协议 §0 前提：G1 的预测框几何 ≈ GT 的 3×
实测：        G1 与 top-100 的任意类别任意框的 IoU **恒等于 0**（38/38）
              ⇒ "best same-class predicted box ≈ 3×" 属于**更宽的『定位失败(0<IoU<0.5)』群体**，
                 不是冻结的 G1。G1 的 v2_records 中 area_ratio / center_dx 字段**全部为 None**（38/38）。
```

**本轮与协议 §23 的禁令逐条对齐**：没有从"框大 / IoU 低 / 中心偏移"推出 regression bottleneck —— 反而证实了
**这些现象在 G1 上根本不存在**，真正的失败点由数据定在别处。

---

## §2 Frozen G1/G2/G4 Definition

**直接复用既有定义，未重新抽样。**

| 组 | 定义来源 | 定义 | n | n_img |
|---|---|---|---:|---:|
| **G1** | `_v3_analysis.json.E1` | native-small ∧ `cause ∈ (SCENE_DIFFICULTY, WEAK_FEATURE_EVIDENCE)`，**等价于「任意类别 IoU == 0」** | **38** | 21 |
| **G2** | `_v3_analysis.json.controls[k][0]` | G1 的 matched control（**跨图**，按 input-sqrt 匹配） | **38** | 19 |
| **G4** | `_v11_ids.json.G4` | native-small ∧ `best_same_class_iou ≥ 0.5` ∧ ∉G1/G2 | **194** | 72 |

分组 key 在 val GT 表中缺失数 = **0**。

| 组 | 类别分布 | native √area 中位 [IQR] | aspect 中位 [IQR] | 分辨率 |
|---|---|---|---|---|
| G1 | person 18, animal 13, light 3, sign 2, garbage_can 1, ball 1 | 22.86 [19.92, 25.69] | 0.855 [0.626, 1.000] | 1920×1080 ×38 |
| G2 | person 15, animal 14, boat 3, light 3, sign 2, ball 1 | 22.91 [19.75, 27.81] | 0.838 [0.577, 1.000] | 1920×1080 ×38 |
| G4 | person 110, animal 20, uav 18, sign 17, car 12, bicycle 5, seat 4, garbage_can 4, ball 2, light 1, boat 1 | 25.28 [19.46, 28.49] | 0.746 [0.486, 1.256] | 1920×1080 ×155, 640×360 ×39 |

**G1/G2 尺寸可比**（MWU p=0.930），**AR 分布无差异**（p=0.775，Cliff δ=+0.039）⇒ §18 的 AR 分层不是必需。

### §2.1 复现门（必须先过）

从 D′ 预测 TXT + val GT，用**冻结评测器** `scripts/official_eval.py` 的口径重算 `_v2_records.json` 的两个核心量：

```text
对照 n = 371 条 native-small GT
max|Δ best_any_class_iou|  = 1.553e-07
max|Δ best_same_class_iou| = 1.553e-07     （纯 float32 舍入）
PASS ✅ 重算口径与既有记录一致
```

---

## §3 Coordinate / Candidate Data Chain

**代码链 FULLY_TRACED（全部引用实际 source）**：

```
GT (归一化 xywh, val label)
 │   official_eval.read_gt_txt  →  norm_xywh_to_xyxy(gt, W_native, H_native)
 ▼
letterbox 预处理
 │   load_image：长边 → imgsz=1280
 │   LetterBox：rect=False → 方形 1280×1280；rect=True → 逐图矩形画布
 ▼
backbone/neck  →  Detect head (ultralytics/nn/modules/head.py:25 `class Detect`)
 │   _inference() head.py:104-135
 │     x_cat = cat([xi.view(1, no, -1)], 2)                    # no = reg_max*4 + nc = 64 + 12 = 76
 │     box, cls = x_cat.split((reg_max*4, nc), 1)              # reg_max = 16
 │     dbox = decode_bboxes(dfl(box), anchors) * strides
 │       DFL：16-bin softmax @ arange(16)  →  期望距离 (l,t,r,b)   [conv.py `class DFL`]
 │       decode_bboxes → dist2bbox(..., xywh=True)             # tal.py:348
 │     return cat((dbox, cls.sigmoid()), 1)                    # (1, 8400, 4+12)，xywh 输入像素
 ▼
NMS / postprocess
 │   predict_rect.py → 排序 + NMS(iou=0.7) + conf≥0.001 + max_det=100
 ▼
预测 TXT：`cls cx cy w h conf`（**按 native 图像尺寸归一化**）
 │   official_eval.read_pred_txt  →  apply_max_boxes(..., 100)  →  norm_xywh_to_xyxy(pred, W, H)
 ▼
最终预测（native xyxy）
```

**训练期 assigner 链（val 侧 artifact 用的是同一条）**：
`TaskAlignedAssigner(topk=10, α=0.5, β=6.0)`（`ultralytics/utils/tal.py`）
→ `select_candidates_in_gts`（anchor 中心严格在 GT 框内）
→ `get_box_metrics`：`overlaps = iou_calculation(gt, pd_boxes)`，
`iou_calculation = bbox_iou(..., CIoU=True).clamp_(0)`（**CIoU，不是 IoU**）
→ `align = cls^α · CIoU^β` → topk → `mask_pos = topk ∧ in_gts ∧ mask_gt`

**数据链（本轮更正后的状态）**：

| 环节 | val 侧可用性 |
|---|---|
| GT → feature cell / candidate **数量** | ✅ `n_in_gts` / `n_topk` / `n_pos` |
| candidate **box geometry** | ✅ `best_assign_iou` / `best_any_overlap`（**decoded 框**与 GT 的 CIoU，下界） |
| candidate **center 偏移** | ❌ 无 per-anchor 表 ⇒ UNAVAILABLE |
| regression **target / DFL 分布** | ❌ 全仓库无任何落盘 ⇒ UNAVAILABLE |
| candidate → final **追踪** | ❌ final 是 post-NMS 标量列表 ⇒ UNAVAILABLE |
| final **prediction** | ✅ 400 个 TXT |

---

## §4 Candidate Count（协议 §5）

来源：`small_object_domains/_v6_domains.json` **V1 域**（val 原图、aug OFF、**全部 2807 val GT**）
与 `small_object_internal_audit/_v4_internal.json`（val，30 图，429 GT）。
两者都由 `model.eval(); model.model[-1].train()` 前向 + **真实 TaskAlignedAssigner** 产出。

| 组 | n | `n_in_gts` 中位 | `n_topk` 中位 | **`n_pos` 中位** | **#(n_pos==0)** |
|---|---:|---:|---:|---:|---:|
| **G1** | 38 | 5 | 10 | **5** | **0** |
| G2 | 38 | 5 | 10 | 5 | 0 |
| G4 | 194 | 6 | 10 | 7 | 1 |

**⇒ 三组 candidate 数量没有量级差异，且 G1 无一条为 0。**
⇒ 协议 §1 的 A（"candidate center 本身就很远"）与"candidate 不存在"**均不成立**。

（§12 的 scale sanity：native-small 全体的 `n_cand` 中位 4，medium 21，large 169 —— 与
`candidate_density_counterfactual` 的 TRAIN 侧计数同量级，说明候选生成机制一致。）

---

## §5 Candidate Center Geometry（协议 §6）

**UNAVAILABLE —— 没有 per-anchor 中心偏移表落盘。**
（v6 只落了『GT 中心 12px 内最高同类 logit』，不是中心坐标；补测需新 forward，协议 §2 禁止。）

**可测的替代量（最终预测侧）**：G1 图内**离 GT 中心最近的那个框**，其中心距中位 = **7.0112 个 GT 对角线**
（IQR [4.27, 10.67]，p10 2.99，p90 14.22）。

⇒ 最终输出里**根本不存在"临近的框"**。这本身不证明"candidate 中心错"（因为 candidate 层未落盘），
但它排除了"candidate 离得很近、只是被 NMS 挪走"这一类解释。

---

## §6 Candidate Box Geometry（协议 §7/§12 —— **本轮最关键的更正**）

`best_assign_iou` / `best_any_overlap` = **GT 框内 anchor 上 decoded 预测框与 GT 的 CIoU 最大值**。
`tal.py` 的 `iou_calculation = bbox_iou(..., CIoU=True).clamp_(0)`，而 **CIoU ≤ 真 IoU**
⇒ `≥0.5` **蕴含真 IoU ≥ 0.5**。**这是下界性质，所有结论因此保守。**

| 组 | n | **best CIoU 中位** | IQR | p10 | p90 | **#(≥0.5)** | #(≥0.3) |
|---|---:|---:|---|---:|---:|---:|---:|
| **G1** | 38 | **0.6278** | [0.430, 0.732] | 0.2393 | 0.8172 | **24** | **33** |
| G2 | 38 | 0.7156 | [0.638, 0.811] | 0.5655 | 0.9115 | 36 | 38 |
| G4 | 194 | 0.7694 | [0.679, 0.841] | 0.5944 | 0.8872 | 189 | 191 |

G1 vs G2：MWU p=0.0054，Cliff δ=−0.3712；G1 vs G4：MWU p=8.5e-07，Cliff δ=−0.5054。

**⇒ G1 的候选几何略低于 G2/G4，但绝对水平良好：
24/38 = 63.2% 的 G1 在 raw head 里已有真 IoU≥0.5 的框，86.8% 已有 IoU≥0.3。**
**这不足以解释最终输出零重叠。**

### §6.1 ❌ 协议 §12 要求的「3× box」分解对 G1 不适用

`_v2_records.json` 里 G1 的 `best_same_class_area_ratio` / `center_dx_norm_gtw` **全部为 None**
（38/38，实测证实）—— 因为 G1 定义就是 any-class IoU == 0，**不存在被匹配的框**。
**"3× box"属于更宽的『定位失败（0<IoU<0.5）』群体，不是冻结的 G1。本轮不张冠李戴。**

对**有**匹配框的 G2/G4，几何干净得很（见 §8 表）：area_ratio 中位 **1.22 / 1.04**，
w_ratio **1.09 / 1.01**，h_ratio **1.06 / 1.03**，aspect_ratio **0.979 / 0.997**。

---

## §7 / §13 Candidate → Final Geometry 分解

```text
GT
 │
 ├─ candidate 生成：      n_pos 中位 5，无 0                    ✅ 正常
 ├─ candidate box 几何：  decoded 框 CIoU 中位 0.6278（24/38 ≥0.5）  ✅ 正常
 ├─ regression / decode： 同上（CIoU 就是对 decode 后的框算的）      ✅ 正常
 ├─ ★ classification：    同一批 positive anchor 上 GT 类 sigmoid **中位 3e-6**  ❌ 失败
 ├─ ranking / NMS：       最终输出零重叠 ⇒ 无候选可被排序            ⚪ n/a
 ▼
最终预测：G1 与 any-class 任意框 IoU ≡ 0（38/38）              ❌ 无输出
```

**三层 IoU 对比**（协议 §14 要求）：

| 层 | 量 | G1 | G2 | G4 |
|---|---|---:|---:|---:|
| IoU(GT, candidate) | 框内 decoded 框 CIoU（**下界**） | **0.6278** | 0.7156 | 0.7694 |
| IoU(GT, final) | best any-class IoU | **0.0000** | 0.6864 | 0.7464 |
| **Δ** | | **−0.6278** | −0.0292 | −0.0230 |

⇒ 中间层**没有缺失**（candidate 层就是 `best_assign_iou`）。链条完整，且恶化点唯一：
**几何在 head 输出处已经达标，在最终输出处消失。**

---

## §8 Regression Error

协议 §10 要求的 `|log(pred_w/GT_w)|` 等只能对**最终预测**算；而对 G1，最终预测不存在 ⇒ **UNAVAILABLE**。

**替代分解（对 G1 图中「最近的框」做 center/size 隔离，三种选择规则全报）**：

| 规则 | `dcenter`（GT 对角） | `w_ratio` | `h_ratio` | `area_ratio` | `center_fixed ≥0.5` | `size_fixed ≥0.5` |
|---|---:|---:|---:|---:|---:|---:|
| nearest_center | **7.01** | 2.06 | 1.94 | 6.37 | 28.9% | **0.0%** |
| highest_conf | 18.18 | 7.14 | 10.59 | 67.84 | 5.3% | 0.0% |
| largest_area | 20.98 | 10.14 | 13.13 | 122.74 | 5.3% | 0.0% |

**读法限定**：这些分解回答的是"如果只修一个因素，**最近的**那个框能否变成 IoU≥0.5"，
**不是**"head 的回归输出错了多少"——后者由 §6 直接回答（**没错**）。
`size_fixed` 恒为 0 是因为框中心本就在天边；这一节的净信息是 **`dcenter` 中位 7 个对角线**。

**对照组（G2/G4 的 best same-class 预测，真实几何）**：

| 组 | `raw` IoU | `center_fixed` | `size_fixed` | `dcenter` | w_ratio | h_ratio | area_ratio | aspect_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G2 | 0.6864 | 0.7666 | 0.7986 | 0.0656 | 1.094 | 1.063 | 1.2175 | 0.9789 |
| G4 | 0.7464 | 0.8103 | 0.8006 | 0.0558 | 1.0136 | 1.0261 | 1.0423 | 0.9972 |

⇒ 在**有**匹配框的组上，单因素隔离的提升都很小（+0.08 / +0.06），
**说明这些组本身没有明显的单一几何误差源** —— 与"几何不是瓶颈"一致。

---

## §9 Classification / Ranking

### §9.1 ★ 分支解离（本轮的判定性证据）

| 组 | n | **best CIoU 中位**（box 分支） | **`pos_prob` 中位**（cls 分支，同一批 anchor） | `center_prob` 中位（中心 12px 内同类） | `_v4_internal` `pos_score_max` 中位 |
|---|---:|---:|---:|---:|---:|
| **G1** | 38 | **0.6278** | **0.000003** | **0.000003** | **0.000003**（max 0.0030） |
| G2 | 38 | 0.7156 | **0.680127** | 0.697562 | 0.680127 |
| G4 | 194 | 0.7694 | **0.727966** | 0.761614 | 0.666663 |

**⇒ 这是同一次前向、同一批 anchor 上的两支输出：框的 CIoU 是 0.63，类别的 sigmoid 概率是 3×10⁻⁶。**
⇒ **不是"几何错"，是"分类分支在该处没有输出"。**

**near-GT 同类 raw logit**（`_v4_internal`，GT 中心 1.5×stride 内，sigmoid 前）：

| 组 | stride-8 | stride-16 | stride-32 |
|---|---:|---:|---:|
| **G1** | **−12.58** | **−11.36** | **−9.64** |
| G2 | +0.836 | +1.012 | +1.278 |
| G4 | +0.882 | +1.262 | +1.351 |

⇒ G1 的同类响应比 G2/G4 低约 **13 个 logit**。**不是"附近有响应但类别错"，是"该处同类响应缺失"。**

**`best_in_gts_align`**（= `cls^0.5 · CIoU^6`）：G1 **0.000030** vs G2 0.0506 / G4 0.1030。
⇒ **因为 `cls ≈ 0`，align 被 cls 项压到 0** —— 这正是 TAL 无法选出这些 anchor 的直接原因。

### §9.2 第三个独立 val 前向的三角验证

`p3_feature_space/_v7_vectors.json` V1 域（2807 val GT）的 `logit_decomp` = GT 中心 cell 的 `W_c·h + b_c`
（与 assigner 无关的独立读数）：

| 组 | n | 同类 logit 中位 | 对应 sigmoid |
|---|---:|---:|---:|
| **G1** | 38 | **−14.6341** | **4.41e-07** |
| G2 | 38 | −0.0896 | 4.78e-01 |
| G4 | 194 | +0.5111 | 6.25e-01 |

⇒ **三个独立 val 前向（`_v6_domains` / `_v4_internal` / `_v7_vectors`）一致。**

### §9.3 ★ 交叉表（协议 §21 的反事实）

| 组 | n | **A: raw head 已有 IoU≥0.5 的框** | B: 最终输出有重叠 | **A ∧ ¬B** |
|---|---:|---:|---:|---:|
| **G1** | 38 | **24** | **0** | **24** |
| G2 | 38 | 36 | 38 | 0 |
| G4 | 194 | 189 | 194 | 0 |

⇒ **G1：24 条"head 已给出 IoU≥0.5 的框"，其中 24 条在最终输出里零重叠。G2/G4：0 条。**
⇒ 协议 §21 的判据在此**明确触发**："大量存在 candidate 本身已经 good，但 final prediction bad"。

### §9.4 Ranking

G1 的 best any-class IoU ≡ 0 ⇒ **最终输出里不存在任何几何合格（IoU≥0.5）的候选**，
因此"存在更好的候选但被排到后面"**在定义上不成立** ⇒ **RANKING = NOT_SUPPORTED**。

补充（排除"图像整体死掉"）：G1 图内**同类预测依然存在**（31/38），同类最高 conf 中位 **0.9264**，
图内全部预测最高 conf 中位 **0.9556**（G2 0.9556 / G4 0.9506）。
⇒ 这些图像**不是没有高置信框**，只是不落在 G1 目标上。

### §9.5 §15 的反事实 null（排除"这些图本来就稀疏"）

| null | 定义 | 结果 |
|---|---|---|
| **N1** | 同图内、**保持 G1 的 GT 尺寸**、位置均匀随机撒（每 GT 400 次），与**同一预测集合**算 any-class IoU | null P(IoU==0) 中位 **0.9213**；**观测 G1 = 1.0000** |
| **N2** | 同一批图内**其它 native-small 真实 GT**（非 G1/G2/G4） | n=32，P(零重叠)=**34.4%**，中位 IoU 0.0497，**P(≥0.5)=0.0%** |

⇒ 两条同时成立：**(i)** G1 的零重叠比同图随机位置更极端；(ii) **G1 所在的图像里 native-small 整体都检不出来**
（图内预测数中位仅 12）—— 这是一个**图像级困难条件**。**不能只归因于目标本身。**
（N2 的 `P(≥0.5)=0.0%` 我最初写成"其它 small GT 被检出"，已在 `AUDIT.log` 中更正。）

---

## §10 Failure Taxonomy

协议 Type C(center) / S(size) / A(aspect) / R(ranking) / U(unresolved) 需要 **candidate 中心与 w/h**，
这些量 val 侧未落盘。分类只能基于两件**已测**的事：框内 decoded 框 CIoU 下界、GT 处 GT 类响应。

| stratum | n | 占比 |
|---|---:|---:|
| **STAGE 4（框 CIoU≥0.5 且 cls≈0）← 主群** | **24** | **63.2%** |
| STAGE 4-borderline（0.3 ≤ 框 CIoU < 0.5 且 cls≈0） | 9 | 23.7% |
| 框也差（CIoU<0.3 且 cls≈0）⇒ 混合 | 5 | 13.2% |
| cls 正常（≥0.01） | 0 | 0.0% |

⇒ **33/38 = 86.8% 属于"框已经够好/接近够好，但 GT 类响应 ≈ 0"。**

| 协议 Type | 判定 |
|---|---|
| **Type S（尺寸）** | **NOT SUPPORTED** —— 框 CIoU 中位 0.63（§6） |
| **Type C（中心）** | **NOT SUPPORTED**（candidate 层）—— n_pos 正常、框已达标 |
| **Type R（排序）** | **不成立** —— 没有任何候选进入最终输出，无从被排到后面 |
| **Type A（长宽比）** | **UNAVAILABLE** —— 无 per-anchor w/h 落盘 |
| **Type U** | 仅 5/38（13.2%）属"框也差"的混合情形 |

---

## §11 Class Stratification（协议 §16）

| class | n_G1 | 框 CIoU 中位 | cls prob 中位 | final any-IoU 中位 | n_pred 中位 |
|---|---:|---:|---:|---:|---:|
| **person** | **18** | 0.6750 | 0.000002 | 0.0000 | 10 |
| **animal** | **13** | 0.4967 | 0.000006 | 0.0000 | 19 |
| light | 3 | 0.7707 | 0.000080 | 0.0000 | 3 | 
| sign | 2 | 0.6674 | 0.000000 | 0.0000 | 51 |
| garbage_can | 1 | 0.6943 | 0.000000 | 0.0000 | 5 |
| ball | 1 | 0.4324 | 0.000000 | 0.0000 | 19 |

**⇒ 两个可统计的类（person 18 + animal 13 = 31/38 = 81.6%）形态完全一致**：
框 CIoU ≈ 0.6（animal 0.497 偏低但仍接近）、GT 类概率 ≈ 0、最终命中 0。
**不是某一类特有的现象**（协议 §23："one class shows effect → general mechanism" 的禁令在此不构成问题，因为**所有**类都同向）。

---

## §12 Scale / Aspect-Ratio Controls（协议 §17/§18）

**分桶一律用 `native_area`，绝不用 `aug_area`**（small <1024，medium 1024–9216，large >9216）。

| bucket | n | best any-IoU 中位 | #(IoU==0) | n_pred 中位 |
|---|---:|---:|---:|---:|
| small | 371 | 0.6086 | 70 | 27 |
| medium | 1320 | 0.7622 | 98 | 17 |
| large | 1116 | 0.8849 | 26 | 13 |

（medium/large 只作机制 sanity check，主比较是 native-small 内部的 G1 vs G2/G4。）

**AR 控制**：G1 vs G2 的 GT AR：0.855 [0.626,1.000] vs 0.838 [0.577,1.000]，
MWU p=0.775、Cliff δ=+0.039 ⇒ **分布无差异，分层不是必需**。
G1 vs G2 的 native √area：MWU p=0.930 ⇒ G2（按 in_sqrt 匹配的 control）**尺寸可比**。

---

## §14 Failure Stage Verdict（协议 §19）

| STAGE | 判定 | 依据 |
|---|---|---|
| 1 candidate generation / **center** | **NOT_SUPPORTED** | `n_pos` 中位 5、无一条为 0（§4） |
| 2 candidate **box geometry** | **NOT_SUPPORTED** | 框内 decoded 框 CIoU 中位 **0.6278**，24/38 ≥0.5（§6） |
| 3 **regression / decoding** | **NOT_SUPPORTED** | 同上（CIoU 就是 decode 后的框算出来的） |
| **4 classification / score** | **★ SUPPORTED** | 同一 anchor 上 GT 类 sigmoid 中位 **3e-6**；near-GT logit **−12.6**（§9.1） |
| 5 ranking / postprocess | **NOT_SUPPORTED** | 最终输出零重叠，无候选可被排序（§9.4） |
| 6 unresolved | — | — |

```text
PRIMARY FAILURE STAGE = STAGE 4 (classification / score)
FINAL VERDICT = C   （CLASSIFICATION 成立；RANKING 不成立）
```

**协议 §20 的 CASE 逐条核对**：

| CASE | 条件 | 本轮 |
|---|---|---|
| A regression geometry | candidate center 对 ∧ candidate geometry 对 ∧ decoded/final 系统性背离 ∧ G1 > G2/G4 | **前半满足、后半不满足** —— 几何**没有**背离，而是被筛掉 ⇒ 不判 A |
| B candidate geometry | candidate 本身错位/过大 | **不成立**（§6） |
| C classification/ranking | good candidate 存在但 final 不是它 | **部分成立**：good candidate 存在，final **一个都没有**。协议 CASE C 的字面条件（"systematically selects another candidate"）**不满足**；但 §19 的 STAGE 4 定义精确命中 ⇒ 采用 C 并**明示这一偏离** |
| D mixed | 多 stage 显著失败 | 13.2% 属混合，但 86.8% 单归 STAGE 4 ⇒ 不判 D |
| E unresolved | 缺 candidate → regression → final 链路 | **不成立**——三层 IoU 齐全（§7） |

**若只问"几何有没有失败"** ⇒ **F — NO_CLEAR_GEOMETRIC_FAILURE**，字面成立。
**本报告选 C**，因为它额外指出失败发生在分类分支，这更可行动。

---

## §15 Training Recommendation（协议 §26）

```text
TRAINING_RECOMMENDATION:
CLOSE
```

协议 §26 要求三个条件**同时**成立才能给 FEASIBILITY_AUDIT：

| 条件 | 本轮 |
|---|---|
| (a) failure stage **明确** | ✅ **STAGE 4**，三层 IoU + 三个独立前向一致 |
| (b) 存在**单一、可干预**机制 | ❌ **该 stage 唯一对应的机制是「抬升 native-small 在低响应处的类别响应」。** 这个机制**已经做过并已收口**：<br>• **E1 RegionResponseGain**（L13/15/17 全局学习增益）：官方 0.50988 vs D′ 0.51528（Δ=−0.00540），配对 bootstrap 95% CI [−0.0193, +0.0054] **含 0** ⇒ NO MEASURABLE GAIN<br>• **L17 donor intervention**：oracle 位置 + oracle 向量替换的最强干预（R3，49 cells）也只把中位 logit 从 −14.63 推到 **−6.07**，仍位于 val 成功小目标分布的**第 19.7 百分位**（`l17_selectivity_audit`，终裁 **C / CLOSE_L17**） |
| (c) 已有 artifact **支持**该机制 | ❌ 上述 artifact 恰恰**否证**它：抬升响应既不能把 logit 推过检出线，也不能在训练期产生可测增益 |

⇒ **(b)(c) 均不成立 ⇒ CLOSE。**

**特别说明（为什么这不与 Q8 的"positive finding"矛盾）**：
本轮**定位**了失败 stage（新信息，且与 E1/L17 完全一致），但**没有提出新的机制** ——
它把 E1 与 L17 的负结果**解释**通了：既然框是对的、缺的只是类别响应，那么任何"增益"路线都必须让响应
跨越检出线才有效，而 E1/L17 的实测说它跨不过去。**这是一个收敛，不是新开口。**

**不创建**任何 config / 脚本改动 / checkpoint / run / submission。

---

## §16 Limitations / Unavailable Artifacts

1. **§6 candidate/anchor 中心偏移表** —— val 侧无落盘 ⇒ **UNAVAILABLE**（§5）。
2. **§9 回归目标 / DFL 分布**（tx/ty/tw/th、4×16 softmax）—— 全仓库无任何落盘，任何 split 都没有 ⇒ **UNAVAILABLE**。
   （已由 Explore 子代理独立穷举确认：`dist2bbox` 只出现在 `ultralytics/` 内，从无序列化。）
3. **§8 per-anchor candidate→final 追踪** —— final 是 post-NMS 标量列表 ⇒ **UNAVAILABLE**。
4. **Aspect-ratio 分支（Type A）** —— 需 per-anchor w/h ⇒ **UNAVAILABLE**。
5. **⚠ letterbox 口径差异（本轮新声明，非新引入）**：所有 val 侧 raw-head artifact
   （`_v6_domains` V1 / `_v4_internal` / `_v7_vectors` / `_v11`–`_v14`）都建在 **`rect=False` 方形 letterbox**；
   冻结预测 TXT 来自 `predict_rect.py`，其 `--mode` **默认 `rect`**（逐图矩形画布）。
   ⇒ **不影响 §9.1 的解离**（box 与 cls 出自**同一次前向、同一批 anchor**），也不影响 §7/§12（native 坐标）；
   只让"raw-head ↔ 冻结预测"的桥接略弱一档。**本轮不跑任何前向来消除它**（协议 §2）。
6. **G1 全部为 1920×1080**，G4 含 39 个 640×360 ⇒ 跨组分辨率构成不同。§11/§12 的分层未发现该差异贡献信号
   （G1 vs G2 同为 HR，形态仍一致）。
7. **`best_assign_iou` 的 topk 依赖**：positives 由 `align = cls^0.5·CIoU^6` 的 top-10 选出，
   而 G1 的 cls≈0 ⇒ align≈0，topk 在此处**近乎退化**。因此该量应读作
   **"GT 框内 anchor 上 decoded 框 CIoU 的最大值"**（≈ `candidate_density` 的 `maxciou`），
   而**不是**"TAL 会选中的那个候选的质量"。
   这**不影响结论方向**（我们要问的正是"框内有没有好框"），但**必须如此读**。
   旁证：`best_in_gts_align` 中位 G1 = 3e-5 vs G2 = 0.0506 ⇒ G1 的 align 确实近乎退化。
8. **train 侧参考不可迁移**：`candidate_density_counterfactual`（TRAIN + mosaic，native-small
   `maxCIoU` 中位 0.8775、P(≥0.5)=91.8%）**与 G1 不可配对**（不同 split / 不同 augmentation）。
   本轮**只把它当机制刻度**，不作 G1 证据（G1 的证据用 val 侧的 §6）。
9. **单一 checkpoint**：全部结论基于 D′ `best.pt`；换权重后方向是否保持未验证。
10. **非配对比较**：G1 vs G4 不是配对（不同目标/图像）；G1 vs G2 是既有 matched control（in_sqrt 匹配，MWU p=0.93）。
11. **未能排除的两个残余解释**：
    (a) NMS 抑制 —— 若能有一个框与"好候选"的 IoU>0.7 却与 GT 完全不相交，则它可解释部分样本。
    该几何情形在 GT 中位 23px 的尺度下**极不可能但不是不可能**；本轮无法直接检验（无 per-anchor 数据）。
    (b) cap-100 截断 —— **6/38** G1 所在图像的预测数恰为 100（`003987#0..#4`、`003262#31`），
    对这 6 条不能排除截断；其余 **32/38** 未触顶，截断不适用。

---

## §17 Provenance

```text
TRAINING              = NO
BACKWARD              = NO
OPTIMIZER_STEP        = NO
MODEL_FORWARD         = NO

EXISTING_FILES_CHANGED        = 0
EXISTING_CHECKPOINTS_CHANGED  = 0
EXISTING_PREDICTIONS_CHANGED  = 0
EXISTING_DIAGNOSTICS_CHANGED  = 0
EXISTING_CONFIGS_CHANGED      = 0

GIT_STATUS_BEFORE = 70 行（与本会话 L17 轮结束时逐行相同）
GIT_STATUS_AFTER  = 71 行，唯一新增 = `?? diagnostic/prediction_head_geometry_audit/`
GIT_HEAD          = f659609c28f6b4f500212197d5229550a36d12d8（未动）
```

**预测 TXT 未被触碰**：`diagnostic/sepstem_clahe/best_full/results/` 400 个文件的
聚合 SHA256 = `7d6745110546fec0`，mtime 最早 2026-09-22 07:58:57 / 最晚 2026-09-22 08:20:20（**均为历史时间**）。

| 读取对象 | SHA256[:16] |
|---|---|
| `diagnostic/sepstem_clahe/best_full/results/`（D′ 400 TXT） | `7d6745110546fec0` |
| `small_object_cause_v2/_v2_records.json` | `da015742973d1efa` |
| `small_object_cause_v2/_v3_analysis.json` | `41d753f5fe24f372` |
| `p3_feature_space/_v11_ids.json` | `e761e673e239fa2a` |
| `p3_feature_space/_v7_vectors.json` | `977593ae00126100` |
| **`small_object_domains/_v6_domains.json`**（关键 val 侧候选数据） | `3505d9160c47aae8` |
| **`small_object_internal_audit/_v4_internal.json`**（第二独立读数） | `0da6877405d241e3` |
| `candidate_density_counterfactual/_results.npz`（TRAIN 参考） | `7b28dc555c3fffb6` |
| `e2_assigner_geometry/_e2_replay.npz`（TRAIN） | `f9277d66037dcc23` |
| `scripts/official_eval.py`（冻结评测口径） | `e82128abc082ed05` |
| `scripts/predict_rect.py` | `36858acb59058e89` |
| `ultralytics/nn/modules/head.py` | `2fa868a4464ce88b` |
| `ultralytics/utils/tal.py` | `aae7e8ac438f00cd` |
| `ultralytics/utils/loss.py` | `0f092cf22a372f6d` |
| `runs/…_sepstem_clahe/weights/best.pt` | `1cae45f75693f541…fae4fda` |

**NEW_FILES**（全部位于 `diagnostic/prediction_head_geometry_audit/`）：
`REPORT.md` / `_analyze.py` / `AUDIT.log` / `_tables.json` / `_results.npz` /
`_recon.py` / `_recon2.py`（侦察过程记录）/ `_git_after.txt`。

**本轮自己的两处更正（如实记录）**：
1. 我最初据 `candidate_density` + `e2_assigner_geometry` 判定"val 侧无 candidate 级数据 ⇒ §5–§9 全部 UNAVAILABLE"。
   **该判定不完整**：另有 `_v6_domains.json`（V1 域，全部 2807 val GT）与 `_v4_internal.json`（val，429 GT）。
   已在 §3/§6 明示更正，并把"原始误判 → 发现 → 更正"留在 `AUDIT.log` 与本节。
2. §9.5 的 N2 结论句首版写反（把"0.0% 达到 IoU≥0.5"写成"被检出"），已修正。

---

**本轮结束，停止。不训练、不设计下一架构、不自行启动下一轮。**
