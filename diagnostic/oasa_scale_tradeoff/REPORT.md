# OASA SCALE TRADE-OFF SIMULATION

**日期**：2026-09-23 · **性质**：offline geometry / supervision-retention 分析
**产出**：`diagnostic/oasa_scale_tradeoff/`（`_scale_tradeoff.py` / `_probe.py` / `_results.json` / 本报告）
**未做**：0 次训练、0 次 1-epoch smoke、0 次 submission、0 次 online 评测；未修改 `ObjectScaleAug` 正式实现；未修改任何正式配置。

---

## 0. 方法声明（与本次最重要的一个前提）

### 0.1 STEP 2.5 的 harness **已不存在**

穷尽检索确认：仓库内没有 STEP 2.5 的 simulation 脚本（特征串 `visible_fraction` / `anchor_vis` / `1080p` / `ROI area` 只命中 `verify_oasa_sync.py`、`augment.py`、`loaders.py` 三个**实现**文件；`diagnostic/oasa_smoke/` 只有 4 张 PNG）。
STEP 2.5 是进程内 monkey-patch，**跑完未落盘**。

因此无法"复用"该 harness。本轮的替代做法是**比复用更严格**的：

> **不重新实现任何几何算法** —— 直接驱动 `ultralytics/data/augment.py` 里**真实的 `ObjectScaleAug` 算子**。
> 本脚本只做四件事：① 用真实链路 `build_yolo_dataset` 建 dataset；② 用 `get_image_and_label` 取**训练时 OASA 收到的同一个 labels dict**；③ 把算子的 `scale` 换成 1.4 / 2.0（其余参数逐项固定）；④ 统计该算子自己的 `keep` 决策。
> ⇒ retention 由**正式实现自己**算出，不是本脚本的近似。

**class 身份的传递**：把 `labels["cls"]` 临时替换为 `arange(n)` 索引标记。算子只把 `cls` 当不透明数组做 `cls[keep]`，从不读取其数值 ⇒ 幸存者索引可精确还原（真实类别保存在旁路数组）。**这不改变任何几何 / keep 判定。**

**paired 设计**：每图固定 seed（`SEED0 + i`）。算子内第一次随机调用是 prob 门、第二次是 anchor 抽取，且 `small` 集合与 `scale` 无关 ⇒ 同图在 1.4 / 2.0 下**门控与 anchor 完全一致**。实测：两条件下 `eligible` / `applied` 图像集合逐图相同（gain = 0）。

**native 口径**：small / medium / large 一律用 **native 像素面积**判定（small < 32² = 1024，medium < 96² = 9216，large ≥ 9216），与 STEP 2.5 / STEP 4B 一致。**未使用 input-space 面积。**

### 0.2 两个必须区分的口径

| 口径 | 定义 | 用途 |
|---|---|---|
| **dataset 层面** | 该桶丢失数 ÷ 该桶全量（1599 图） | 监督总量的真实损失 |
| **affected-image 内** | 该桶丢失数 ÷ 被 OASA 处理过的图内的该桶数 | **STEP 2.5 的 40.5% / 58.7% 是这个口径** |

本报告**两套都给出并分别标注**。混用两套正是本轮要避免的错误。

---

## Dataset

- train images：**1599**（split 为 1600，其中 1 张因 label 坐标越界被 ultralytics 整图丢弃 —— 与云端 smoke 的 `1600 images, 0 backgrounds, 1 corrupt` 逐字一致）
- total GT：**11560**
- native small GT：**1392**（占全部 GT 的 **12.0%**）
- native small-containing images：**401 / 1599 = 25.1%**
- native 分辨率构成：**1080p (1080×1920) 1477 张**、**360p (360×640) 122 张**

> 上述四项与 STEP 2.5 / STEP 4B 记录**逐项一致**（401 图 / 25.1% / 12.0%），是 harness 有效性的第一层证据。

**算子参数**（两条件完全相同，仅 `scale` 不同）：`prob=0.5`、`small_area=1024`、`min_visible=0.8`、`mosaic_runtime=1.0`（Mosaic ON）。

---

## Geometry

| Metric | baseline | scale=1.4 | scale=2.0 |
|---|---:|---:|---:|
| ROI width | 1.0 | **0.714286** | 0.500000 |
| ROI height | 1.0 | **0.714286** | 0.500000 |
| ROI area | 1.0 | **0.510204** | 0.250000 |
| cropped-out area | 0 | 0.489796 | 0.750000 |
| √area multiplier | 1.0× | **1.401×** | **2.000×** |

**实测 anchor 几何**（非理论推算，取自算子 `stats`）：

| Metric | baseline | scale=1.4 | scale=2.0 |
|---|---:|---:|---:|
| anchor n（applied 图数） | — | 209 | 209 |
| anchor √area before（input 空间） | 19.49 | 19.49 | 19.49 |
| anchor √area after（input 空间） | 19.49 | **27.30** | **38.99** |
| **after / before** | 1.000 | **1.401** | **2.000** |
| anchor √area 最终输入空间（×0.5 Mosaic） | 9.75 | **13.65** | **19.49** |
| anchor visibility 中位 | — | **1.000** | **1.000** |
| anchor visibility ≥ 0.8 比例 | — | **100%** | **100%** |
| anchor kept rate | — | **209/209 = 100%** | **209/209 = 100%** |

**分分辨率 anchor √area（input 空间）**：

| 分辨率 | n@1.4 / n@2.0 | before | after @1.4 | after @2.0 | mult |
|---|---:|---:|---:|---:|---:|
| 1080p | 144 / 144 | 17.35 | 24.29 | 34.69 | 1.401 / 2.000 |
| 360p | 65 / 65 | 46.48 | 65.09 | 92.95 | 1.401 / 2.000 |

⇒ **两条件都精确兑现配置的倍数**（1.401 / 2.000），anchor **从未被 ROI 裁剪误删**（kept = 100%，`skip_visibility = 0`）。

**Gating（两条件相同，paired 的直接后果）**：`seen=1599`、`eligible=401 (25.1%)`、`applied=209 (13.1%)`、`skip_prob=192`、`skip_nosmall=1198`、`skip_nomosaic=0`、`skip_visibility=0`。
⇒ `scale` **不改变哪些图被处理**，只改变被处理时毁掉多少监督。

---

## GT Retention

### A. dataset 层面（1599 图全量）

| Metric | baseline | scale=1.4 | scale=2.0 |
|---|---:|---:|---:|
| all GT retention | 100% | **96.17%** | **92.32%** |
| small retention | 100% | **92.74%** | **85.06%** |
| medium retention | 100% | **96.12%** | **91.83%** |
| large retention | 100% | **97.26%** | **95.08%** |

### B. affected-image 内口径（209 图，**STEP 2.5 的可比口径**）

| Metric | baseline | scale=1.4 | scale=2.0 |
|---|---:|---:|---:|
| GT before → after | — | 2530 → 2087 | 2530 → 1642 |
| all GT retention | 100% | **82.49%** | **64.90%** |
| small retention | 100% | **87.20%** | **73.64%** |
| medium retention | 100% | **82.88%** | **64.01%** |
| large retention | 100% | **73.81%** | **52.99%** |
| 1080p drop | 0 | 17.26% | 34.47% |
| 360p drop | 0 | 19.84% | 40.89% |

### C. small GT retention（native `area < 1024`，单独列出）

| 口径 | baseline | scale=1.4 | scale=2.0 |
|---|---:|---:|---:|
| small GT before（dataset） | 1392 | 1392 | 1392 |
| small GT after（dataset） | 1392 | **1291** | **1184** |
| **small dataset-level retention** | 100% | **92.74%** | **85.06%** |
| **small dataset-level loss** | 0 | **7.26%** | **14.94%** |
| small before（affected 图内） | — | 789 | 789 |
| small after（affected 图内） | — | **688** | **581** |
| **small affected-image retention** | 100% | **87.20%** | **73.64%** |
| **small affected-image loss** | 0 | **12.80%** | **26.36%** |

### D. 分层诊断：small drop 完全由「该图 small 数量」决定

| 该图 small 数 | 图数 | small before | drop @1.4 | drop @2.0 |
|---|---:|---:|---:|---:|
| 1 | 84 | 84 | **0.0%** | **0.0%** |
| 2 | 43 | 86 | 10.5% | 18.6% |
| 3 | 27 | 81 | 14.8% | 33.3% |
| 4–5 | 30 | 131 | 18.3% | 29.0% |

⇒ **只有 1 个 small 的图，drop 结构性为 0**（anchor 就是那个 small，ROI 必然把它完整包住）。small 密度越高、drop 越大。

---

## Class Retention

### A. dataset 层面（全量 11560 GT）

| class | GT before | after @1.4 | ret @1.4 | after @2.0 | ret @2.0 |
|---|---:|---:|---:|---:|---:|
| person | 4420 | 4146 | 93.8% | 3895 | 88.1% |
| boat | 107 | 102 | 95.3% | 98 | 91.6% |
| animal | 2327 | 2305 | 99.1% | 2264 | 97.3% |
| seat | 511 | 501 | 98.0% | 496 | 97.1% |
| sign | 659 | 628 | 95.3% | 596 | 90.4% |
| bicycle | 550 | 523 | 95.1% | 499 | 90.7% |
| car | 1283 | 1254 | 97.7% | 1230 | 95.9% |
| ball | 70 | 68 | 97.1% | 65 | 92.9% |
| light | 1211 | 1191 | 98.3% | 1160 | 95.8% |
| garbage_can | 225 | 215 | 95.6% | 207 | 92.0% |
| **uav** | 174 | 162 | **93.1%** | 140 | **80.5%** |
| tricycle | 23 | 22 | 95.7% | 22 | 95.7% |

### B. affected-image 内口径（209 图）

| class | before | after @1.4 | drop @1.4 | after @2.0 | drop @2.0 |
|---|---:|---:|---:|---:|---:|
| person | 1404 | 1130 | 19.5% | 879 | 37.4% |
| boat | 26 | 21 | 19.2% | 17 | 34.6% |
| animal | 156 | 134 | 14.1% | 93 | 40.4% |
| seat | 34 | 24 | 29.4% | 19 | 44.1% |
| sign | 180 | 149 | 17.2% | 117 | 35.0% |
| bicycle | 175 | 148 | 15.4% | 124 | 29.1% |
| car | 194 | 165 | 14.9% | 141 | 27.3% |
| ball | 15 | 13 | 13.3% | 10 | 33.3% |
| light | 206 | 186 | 9.7% | 155 | 24.8% |
| **garbage_can** | 44 | 34 | **22.7%** | 26 | **40.9%** |
| **uav** | 94 | 82 | **12.8%** | 60 | **36.2%** |
| tricycle | 2 | 1 | 50.0% | 1 | 50.0% |

**最受伤的类**：affected 口径下 `garbage_can`(40.9%)、`seat`(44.1%)、`animal`(40.4%)、`uav`(36.2%)。dataset 口径下 **`uav` 最差（80.5% @2.0）** —— 因为 uav 高度集中在少数受影响图中。`tricycle` 只有 2 个样本，不可解读。

---

## STEP 2.5 Cross-check

| 量 | 本轮模拟 | STEP 2.5 记录 | 判定 |
|---|---|---|---|
| eligible 率 | **25.1%** | 25.1%（native） | ✅ **复现** |
| small-containing 图 | **401 / 25.1%** | 401 / 25.1% | ✅ **复现** |
| small / 总 GT | **1392 / 12.0%** | 1392 / 12.0% | ✅ **复现** |
| anchor 1080p √area before（input） | **17.35** | **17.18** | ✅ **复现（Δ1%）** |
| anchor 放大倍数 | **2.000** | 2.000 | ✅ **复现** |
| anchor visibility 中位 | **1.000** | 1.000（≥0.95 = 100%） | ✅ **复现** |
| anchor kept | **209/209 = 100%** | skip 0 | ✅ **复现** |
| affected-image GT drop（整体） | 35.10% | 40.5%（1080p, n=80） | ⚠️ Δ5.4pp |
| affected-image drop 1080p | 34.47% | 40.5% | ⚠️ Δ6.0pp |
| affected-image drop 360p | 40.89% | 58.7%（n=40） | ⚠️ Δ17.8pp |
| **affected-image small drop** | **26.36%** | **12.4%** | ❌ **未复现** |
| anchor 360p √area before | 46.48 | 22.18 | ❌ **未复现** |
| dataset-level small loss | **14.94%** | **1.6%** | ❌ **未复现（见下）** |
| dataset-level medium loss | 8.17% | 5.3% | ⚠️ |
| dataset-level large loss | 4.92% | 7.4% | ⚠️ |

### 第三层证据（最强）：与 300ep run 的算子自报对比

300ep OASA v1 run 的日志里，算子每次 log 都会自报 `gt N->M`（同一个 `stats["gt_before"]/["gt_after"]` 计数）。8 行的 **affected-image 内 drop** 为：

```
40.95% / 33.62% / 35.51% / 39.88% / 35.99% / 32.82% / 33.03% / 33.68%
均值 = 35.69%
```

**本轮模拟 = 35.10%。Δ = 0.6pp。**

这是同一份代码、同一份数据、同一口径（算子自身计数）下的对照，且样本量 800 vs 1599。**⇒ harness 的头号指标与真实实现吻合到 0.6pp。**

### 未复现项的原因（逐条）

1. **`affected-image small drop` 26.36% vs 12.4% → 样本构成，非口径。**
   §D 的分层表证明 small drop 是 small 密度的函数。STEP 2.5 的 80 图共 193 small（**2.4 small/图**），本轮 209 图共 789 small（**3.8 small/图**）。按 §D 的分层权重，均值 2.4 的样本自然落在 ~12–15% 区间。
   **两条都正确，只是样本不同。**

2. **`anchor 360p` 46.48 vs 22.18 → 小样本。**
   1080p（占 92.4%）的 anchor 复现到 Δ1%，360p 未复现。本轮 360p applied = 65 图，STEP 2.5 = 40 图，且其抽样子集未记录、无法还原。

3. **`dataset-level loss` 1.6% vs 14.94% → STEP 2.5 的换算乘数是错的**（见下节，这是本轮独立发现的问题，不是 harness 差异）。

---

## ⚠️ 独立发现：STEP 2.5 的 dataset 层面换算无效

STEP 2.5 §6 的 dataset 层面数字是用**图像占比**换算的：

```
small: 12.4%（affected 图内）× 12.5%（受影响图像占比）= 1.6%
medium: 42.2% × 12.5% = 5.3%      large: 59.4% × 12.5% = 7.4%
```

**这个乘数是错的。** dataset 层面丢失应当乘的是「**该桶目标落在受影响图中的比例**」，而不是「图像占比」。而受影响图在 small 上是**极度富集**的：

| 桶 | 受影响图占该桶目标的真实比例 | STEP 2.5 用的乘数 | 偏差 |
|---|---:|---:|---:|
| small | **56.68%** | 12.5% | **偏低 4.5×** |
| medium | **22.69%** | 12.5% | 偏低 1.8× |
| large | **10.47%** | 12.5% | 偏高 1.2× |

正确换算（本轮，两条路径精确闭合）：

| 桶 | affected 内 drop | × 正确乘数 | = dataset-level | 直接实测 |
|---|---:|---:|---:|---:|
| small | 26.36% | 56.68% | 14.94% | **14.94%** ✅ |
| medium | 35.99% | 22.69% | 8.17% | **8.17%** ✅ |
| large | 47.01% | 10.47% | 4.92% | **4.92%** ✅ |

**影响**：OASA v1 获批时所依据的「small dataset 层面只损失 1.6%」，本轮实测为 **14.94%（9.3×）**。
其中**仅修正乘数**一项（完全沿用 STEP 2.5 自己的 within-image 12.4%）：`12.4% × 56.68% = 7.03%` → 已使 1.6% 低估 **4.4×**；其余差异来自 within-image 本身的样本构成（§Cross-check 未复现项 1）。
⇒ v1 的监督代价此前被**系统性低估**。这与 v1 在 ep201–290 出现的提前饱和（有效数据多样性下降）在方向上是自洽的。

---

## Trade-off

| Metric | baseline | scale=1.4 | scale=2.0 |
|---|---:|---:|---:|
| ROI area | 100% | 51.02% | 25.00% |
| √area multiplier | 1.0× | **1.4×** | **2.0×** |
| affected images | 0 | 209 (13.1%) | 209 (13.1%) |
| **all-GT retention（affected 内）** | 100% | **82.49%** | **64.90%** |
| **small retention（affected 内）** | 100% | **87.20%** | **73.64%** |
| **medium retention（affected 内）** | 100% | **82.88%** | **64.01%** |
| **large retention（affected 内）** | 100% | **73.81%** | **52.99%** |
| **small retention（dataset）** | 100% | **92.74%** | **85.06%** |

**1.4 相对 2.0 的 retention gain**：

```
all-GT retention gain      = 82.49% - 64.90% = +17.59 pp
small retention gain       = 87.20% - 73.64% = +13.56 pp
medium retention gain      = 82.88% - 64.01% = +18.87 pp
large retention gain       = 73.81% - 52.99% = +20.82 pp
affected images gain       = 209 - 209        =   0   （paired 设计的必然结果）
dataset-level small loss   = 7.26% vs 14.94%  → 损失减半（−51.4%）
```

> ⚠️ **ROI ≠ retention。** scale=1.4 的 ROI area 是 51.02%，但 **GT retention 不是 51%**（实测 82.49% / small 87.20%）。retention 由上表 1599 张图、11560 个 bbox 的真实几何模拟得出，非 ROI 面积换算。

---

## Decision

```text
SCALE_1.4_WORTH_TESTING = YES
```

**依据（仅事实性判断）：**

1. **情况 A 的两项条件均满足**：retention **明显高于 2.0**（small +13.56pp、all +17.59pp，dataset 层面 small 损失减半）；small-object scale **仍精确保持 1.4×**（实测 multiplier = 1.401，anchor kept = 100%）。
2. **保留的代价被明确量化**：scale=1.4 仍损失 **7.26% 的 dataset 层面 small 监督**、affected 图内 **12.80%**。这不是零代价。

**明确不声称的事：**

- 本模拟**不预测 AP**。retention 改善 → AP 改善**不是**推论。scale=1.4 同时把**放大收益从 2.0× 降到 1.4×**，两个方向相反的作用都变弱，净效果必须由训练回答。
- 若你的「情况 C」阈值比 ~7% dataset-level small loss 更严，则应读作 `UNCERTAIN` 而非 `YES`。**阈值由你定，数字已在上面。**

**未复现项的处理**：§Cross-check 的三项未复现均有可核查的解释（样本构成 / 小样本 / 换算乘数错误），且**头号指标（affected-image GT drop）与真实 300ep run 的算子自报吻合到 0.6pp**。因此不触发"先 STOP"——但若你认为 STEP 2.5 的 `small 12.4%` 是必须逐字复现的基准，则本轮的 `small` 相关结论应降级为待定，需先解决该口径分歧。

---

## 建议的下一次正式实验（**不自行执行**）

单变量：`object_scale_aug_scale: 2.0 → 1.4`，其余（`prob=0.5`、`small_area=1024`、`min_visible=0.8`、`close_mosaic`、`ir_encoding`、`lr0`…）逐项不动。
预期先验：**监督损失减半，但放大倍数也减半** —— 若 v1 的负向主要来自监督毁损/提前饱和，1.4 应改善；若主要来自放大本身无益，1.4 应仍不改善。这是一个**能区分两种机制的实验**，而不只是"参数小一点再试一次"。

---

## Training

```text
NOT RUN
```

本轮训练次数 = 0；1-epoch smoke = 0；submission = 0；online 评测 = 0；修改正式配置 = 0；修改 ObjectScaleAug 实现 = 0 行。

**复现命令**：

```bash
python diagnostic/oasa_scale_tradeoff/_scale_tradeoff.py   # → _results.json
```
