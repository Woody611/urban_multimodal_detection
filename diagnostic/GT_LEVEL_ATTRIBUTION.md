# D′ vs M1 — GT-level Attribution（ZERO GPU / 不重新推理）

**日期** 2026-10-04 · **脚本** `diagnostic/gt_level_attribution.py`（严格镜像 `scripts/official_eval.py::pr_curve_for_class` 的贪心匹配）
**数据** D′ = `diagnostic/sepstem_clahe/best_full/results`（复现 0.51528 ✅）· M1 = `diagnostic/batch1_eval/m1_best/results`（复现 0.49951，云端为 0.49944，差 7e-5 CPU/GPU 浮点）
**GT** = 同一 `load_split`（400 图 / 2 corrupt / 有效 398 / GT 2807）

---

## Step 0 — 硬门

| 检查 | 结果 |
|---|---|
| D′ 与 M1 的 image 集合完全一致 | ✅ 400 == 400 |
| 全部有效 GT 图（398）都被两侧覆盖 | ✅ 未覆盖集合 = ∅ |
| 两侧多出的 stem 相同（= 2 张被 skip 的 corrupt 图） | ✅ `000050`, `003817` |
| D′ / M1 各自复现官方口径 | ✅ 0.51528 / 0.49951 |
| class id / 坐标约定 | ✅ 两侧走同一个 `load_split`+`norm_xywh_to_xyxy`，构造上一致 |

> **修正记录**：初版 gate 写成「preds 集合 == GT 集合」⇒ 因 corrupt 图的 TXT 存在而误报 FAIL。
> 正确谓词是「有效 GT ⊆ preds」+「两侧多出的 stem 相同」。**gate 触发是对的，是我的谓词写错了。**

---

## Table 1 — GT outcome（IoU = 0.50，官方贪心匹配）

| outcome | count | % GT |
|---|---:|---:|
| BOTH_CORRECT | **2166** | 77.16% |
| **D′_ONLY（rescue）** | **129** | **4.60%** |
| **M1_ONLY（regression）** | **101** | **3.60%** |
| BOTH_WRONG | 411 | 14.64% |
| **TOTAL** | **2807** | 100.00% |

- **net rescue = +28（+1.00% of GT）**，与 Δ(D′−M1)=+0.01577 **同向 ✅**
- **★ 关键特征：gross churn 230 GT，net 只有 28。** 两个模型是**互相交换** GT，不是 D′ 单向更好。
- ⚠ rescue count 不是 mAP decomposition，**不可换算**。

---

## Table 2 — Size（Δ 与 net rescue 同轴：正 = D′ 更好）

| size | GT | D′_ONLY | M1_ONLY | net rescue | rescue rate |
|---|---:|---:|---:|---:|---:|
| small | 371 | 26 | 20 | **+6** | +1.62% |
| **medium** | 1320 | 86 | 53 | **+33** | **+2.50%** |
| **large** | 1116 | 17 | 28 | **−11** | **−0.99%** |

**但 net rescue 随 IoU 阈值强烈变化 —— 这是本轮最重要的发现之一：**

| thr | BOTH_OK | D′_ONLY | M1_ONLY | net |
|---:|---:|---:|---:|---:|
| 0.50 | 2166 | 129 | 101 | **+28** |
| 0.65 | 1884 | 164 | 122 | +42 |
| **0.75** | 1448 | 224 | 166 | **+58** |
| 0.85 | 864 | 184 | 142 | +42 |
| 0.90 | 522 | 148 | 125 | +23 |

按 size 在 thr=0.85：small **−1** / medium **+27** / **large +16**。

> **⇒ large 从 IoU=0.50 的 −11 翻转为 IoU=0.85 的 +16。**
> **D′ 不是"救回更多大目标"，而是"把同一批大目标定位得更准"。**

---

## Table 3 — Class（12 类全出，IoU=0.50）

| class | GT | D′_ONLY | M1_ONLY | net | rescue rate |
|---|---:|---:|---:|---:|---:|
| person | 1045 | 55 | 42 | **+13** | +1.24% |
| boat | 28 | 0 | 0 | 0 | 0.00% |
| animal | 731 | 42 | 24 | **+18** | **+2.46%** |
| seat | 107 | 4 | 1 | +3 | +2.80% |
| **sign** | 150 | 8 | 2 | **+6** | **+4.00%** |
| bicycle | 106 | 6 | 5 | +1 | +0.94% |
| **car** | 288 | 7 | 10 | **−3** | **−1.04%** |
| ball | 19 | 0 | 4 | −4 | −21.05% |
| **light** | 244 | 5 | 8 | **−3** | **−1.23%** |
| garbage_can | 57 | 2 | 4 | −2 | −3.51% |
| uav | 30 | 0 | 1 | −1 | −3.33% |
| tricycle | 2 | 0 | 0 | 0 | 0.00% |

D′_ONLY 最多的类：**person 55 / animal 42 / sign 8 / car 7 / bicycle 6**。

> **⚠ 与 A2 的冲突（必须记录）**：A2 的类 AP 显示 D′ 显著更好的是 **light(+0.0624) / sign(+0.0576) / car(+0.0383) / animal(+0.0226)**。
> 但 GT-level 的 **light = −3、car = −3（净为负）**；只有 **animal(+18) / sign(+6)** 两个方向一致。
> **⇒ light / car 那两类的 AP 增益不是"命中率"增益，而是排序增益**（PR 曲线重排，match 状态没变）。详见 §"两个分量"。

---

## Table 4 — D′ rescue 的类型（严格由预测内容判定）

| rescue type | 判定 | count | % D′_ONLY |
|---|---|---:|---:|
| **R1 localization** | M1 有同类候选但 maxIoU < 0.50 | **107** | **82.9%** |
| R2 ranking / assignment | M1 有同类候选且 maxIoU ≥ 0.50 | 11 | 8.5% |
| R3 no same-class candidate | M1 该图无同类候选 | 11 | 8.5% |

**⚠ R1 的局限**：它把"M1 把**这个** GT 的框放歪了"与"M1 那个候选其实是**另一个**物体"混在一起，现有 prediction 无法区分。**不猜。**

## Table 5 — D′_ONLY × size × D′ matched IoU

| size | [.50,.60) | [.60,.70) | [.70,.80) | [.80,.90) | [.90,1.00] | total |
|---|---:|---:|---:|---:|---:|---:|
| small | 13 | 4 | 4 | 4 | 1 | 26 |
| **medium** | 19 | 18 | **31** | 14 | 4 | **86** |
| large | 8 | 1 | 3 | 3 | 2 | 17 |

⇒ rescue 的主力是 **medium × IoU .70–.80（31 个）** 与 **medium × .50–.60（19 个）**。

---

## Table 6 — Rescue phenotype（四组分布）

| outcome | n | area p50 | D′ IoU p50 | **D′ conf p50** | M1 IoU p50 | **M1 conf p50** |
|---|---:|---:|---:|---:|---:|---:|
| BOTH_CORRECT | 2166 | 7901 | 0.843 | **0.912** | 0.835 | **0.917** |
| **D′_ONLY** | 129 | 2436 | 0.715 | **0.245** | — | — |
| **M1_ONLY** | 101 | 3344 | — | — | 0.677 | **0.047** |
| BOTH_WRONG | 411 | 2268 | — | — | — | — |

**D′_ONLY 的 M1 侧 best same-class：IoU p50 = 0.293，但 conf p50 = 0.9505。**

> **★★ 这是机制的核心**：对 D′ 救回的那些 GT，M1 当时**给了一个 ≥0.95 高置信度、但 IoU 只有 0.29 的同类框**；
> 而 D′ 给了一个**置信度只有 0.245、但 IoU 0.715 正确**的框。
> 且两个方向的 rescue 都发生在**低置信区**（D′ 0.245 vs M1 0.047，**差 5×**）—— BOTH_CORRECT 则在 0.91。

---

## Step 9 — Modality phenotype（**结论：无信号，两条都撤回**）

未分层时看似有信号：`depth_valid_ratio` D′_ONLY **0.385** vs BOTH_CORRECT **0.779**；`rgb_edge` 1599 vs 1187。

**做 size-matched 对照后：**

| size | `dvalid` D′_ONLY | `dvalid` BOTH_CORRECT | `edge` D′_ONLY | `edge` BOTH_CORRECT |
|---|---:|---:|---:|---:|
| small | 0.000 (26) | 0.000 (157) | 3256 | 2717 |
| **medium** | **0.587** (86) | **0.459** (940) | 1537 | 1390 |
| large | 0.839 (17) | 0.866 (992) | **805** | **929** |

- **`dvalid` 在 medium 桶内方向反转**（0.587 > 0.459）⇒ 未分层的 0.385 vs 0.779 **是 size 混杂**（D′_ONLY 集中在 medium/area 2436）。
- **`edge` 不单调**（small/medium 微正、large 反向）⇒ 无一致信号。

> **⇒ 没有任何简单的图像统计特征能刻画 rescue phenotype。这是 null result，如实报告。**
> （深度 PNG 有部分读取失败，`dvalid` 覆盖 157/196 (small)、940/978 (medium)、992/992 (large)；exploratory only，未做配对检验。）

---

## Step 10 — Consistency checks

| Check | 结果 |
|---|---|
| **A** D′_ONLY 的 size 分布 | small 20.2% / medium 66.7% / large 13.2% ⇒ **medium 主导，small 不是主力** ✅ |
| **B** D′_ONLY 的类分布 | person 55 / animal 42 居首；**light(−3)、car(−3) 净为负** ⚠ **与 A2 不一致，见下** |
| **C** net rescue 与 Δ 同向 | ✅ +28 与 +0.01577 同向（但不可换算） |

### ★ 两个分量的分解（本轮推导，hypothesis-level）

D′ 的优势至少含**两个方向相反可分量的成分**：

1. **命中率分量（match-rate）**：net **+28 GT**，**localization 型**（R1 82.9%），**medium 主导**（+33），且在 0.50 时 **large 为负（−11）**、到 0.85 才转正（+16）。
2. **排序分量（ranking）**：**light / car 类 AP 明显上升（+0.062 / +0.038）但 GT 命中率净为负** ⇒ 这类增益**不是命中率增益**，是 PR 曲线重排（同置信区间内 TP 排到 FP 前面）。

这与 per-size / per-IoU 的 mAP 分解**不矛盾**：mAP 在 IoU 0.5–0.95 上积分 PR 曲线，**"同一批 GT 被定位得更准"和"TP 排序更好"都会抬高 AP，但不改变 IoU=0.50 的命中位图**。

> **⚠ 同时更正上一份报告 `FUSION_ATTRIBUTION_PHASE_ABC.md` §1 的符号错误**：
> 该表把 E1 工具（`Δ = M1 − D′`）的 size/段 数值放进了标着 `Δ = D′ − M1` 的表里。
> 用本脚本以统一 Δ=D′−M1 重算（已核）：
> **large = +0.02142**（非 −0.02142）· **.80–.95 = +0.02328** · **.50–.60 = +0.01365** · **.65–.75 = +0.00787** · **small = −0.00180**。
> **A1/A2 的结论方向不变**（D′ 在 large 与高 IoU 更好，small ≈ 0），只是表中符号写反了。

---

## 五个问题的回答

### Q1 — D′ 到底救回多少 GT？
**净 +28 个（129 救回 vs 101 丢失），占 2807 个 GT 的 +1.00%。**
但 **gross churn 是 230 个 GT** —— 两者是**互相交换**，不是 D′ 单向覆盖 M1。

### Q2 — 是否主要来自 medium/large？
**在 IoU=0.50 下：medium 主导（+33），large 反而是负的（−11），small ≈ 0（+6）。**
**但在 IoU=0.85 下 large 翻正为 +16、medium +27、small −1。**
⇒ 准确表述：**rescue 是 medium-centric 的；large 的收益不是"救回"而是"重新定位更准"。**

### Q3 — 是否与 P3/P11 的 small-object failure 重合？
**不重合。明确写：multimodal rescue phenotype 与 P3/P11 small-object failure 不重合。**
- small 桶 net 仅 +6（+1.62%），D′_ONLY 里 small 只占 20.2%；
- P3/P11 的困难类在这里的表现是 **light −3 / car −3（净为负）**、ball −4、garbage_can −2、uav −1；
- 只有 **animal +18 / sign +6 / seat +3 / person +13** 为正。

### Q4 — 更像哪一类 rescue？
**以 localization rescue 为主（R1 82.9%），但不是干净的单一型。**
- R1 107 / R2 11 / R3 11 ⇒ **不是 candidate-generation rescue**（R3 只 8.5%）；
- 但 R1 的判定无法区分"M1 的框歪了"与"那是另一个物体"；
- 且伴随一个**明确的置信度不对称**：D′ 救回时 conf p50 **0.245**，M1 救回时只有 **0.047**（5×）。

### Q5 — 下一阶段优化 RGB detector，还是 multimodal fusion？

**从 GT-level 证据看：都不强烈支持，而 fusion 这一侧的靶点不成立。**

支持"不再投入复杂 fusion"的证据：
1. **净效果极小**：+28 GT / 2800 = **+1.00%**；gross churn 230 说明两模型高度互换；
2. **没有稳定的 phenotype**：Step 9 在 size-matched 后**两条图像统计信号全部消失**；类分布**部分为负**（light/car/ball/garbage_can/uav）；size 效应在 0.50 下对 large 是**负的**；
3. **单模态上限很低**：IR 0.176 / Depth 0.261（Batch-1）；
4. **R3（candidate-generation rescue）只有 8.5%** ⇒ **不需要更早的 modality fusion / backbone fusion**（排除 Case D）；
5. **很小的 small-object 重合度** ⇒ **不满足 Case E**（不重新打开 small-object × multimodal）。

唯一微弱的正面依据：**R1（localization）82.9%** —— 若一定要走，只有"**feature-level / mid-level fusion 针对定位**"（Case C）这一条有依据。**但它没有图像统计层面的靶点**，因此**无法写出一个可证伪的最小干预**。

> **⇒ 判决：Case A 主导（D′_ONLY 净量小 + 无稳定 phenotype）⇒ 不支持继续复杂 fusion。**
> Case C 只作为弱次要项保留，且因为缺少 phenotype 靶点，**当前不足以支撑一个新的 fusion architecture 实验**。

---

## 附：本轮的 harness bug（第 5、6 次同类）

1. **Step 0 gate 谓词**：写成「preds 集合 == GT 集合」，未考虑 corrupt 图的 TXT 仍存在 ⇒ 误报 FAIL。已改为「有效 GT ⊆ preds」+「两侧多出的 stem 相同」。
2. **Step 9 坐标**：`norm_xywh_to_xyxy` 返回的**已是像素**坐标，我又乘了一次 `W/H` ⇒ 框全跑到角落，`n=1/2`。已修。

**教训（与前 4 次同源）：探针的坐标/通道/集合假设必须显式声明并断言，不能靠默认。**

---

## STOP

**不启动任何 GPU 实验。** 等待下一步决策：
1. GT-level attribution ✅
2. rescue phenotype ✅（无稳定 phenotype）
3. 与 P3/P11 的关系 ✅（不重合）
4. 与 A1/A2 的关系 ✅（含一次符号更正 + 两个分量的分解）
5. RGB vs Fusion 的推荐 ✅（Case A 主导）
