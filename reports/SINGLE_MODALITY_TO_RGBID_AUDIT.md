# 单模态能力 → RGBID 融合收益 · 只读诊断审计

**日期**：2026-09-19
**性质**：**READ-ONLY**。未训练、未修改任何配置 / 模型 / 数据集 / 增广 / trainer / evaluator；未生成 submission。
**新增文件**：仅本报告 `reports/SINGLE_MODALITY_TO_RGBID_AUDIT.md`
**新计算**：§6 的 object-level 模态互补统计（由磁盘上已存在的预测 TXT 现算，未落盘任何中间文件）

---

## 1. Executive Summary

> **当前是否值得先优化单模态、再把优化后的单模态接回 RGBID 融合？**
>
> **不值得（`NO_GO / INSUFFICIENT_EVIDENCE`）。**

三条独立理由，任一单独即足以拦住这次训练：

1. **不存在可信的单模态 baseline。** 仓库里 RGB / IR / Depth 的单模态数字**全部是 training-val（fork 口径）**，来自不同 split、不同 imgsz、不同训练配方，**从未被官方口径评估过**，也没有保存任何预测。拿这些数字当基线去规划实验，前提不成立。
2. **这个策略在本项目已经被直接测过一次，并已关闭。** **IR CLAHE（Exp D）就是"优化单模态表示 → 接回 RGBID 融合"**：它改的是 IR 的预处理，测的是融合模型的官方 mAP。结果是 `D_REJECTED`（官方 +0.00148，配对 bootstrap 四指标 CI 全含 0，mAP75 −0.062）。
3. **object-level 证据显示模态的独立贡献处于噪声量级，且不落在小目标上。** 对冻结的正式模型做逐目标集合差：**97.2% 的 TP 在 IR 与 Depth 同时移除后仍然存在**（即 RGB 单独就足够）；IR 独有贡献 34/2325 = **1.46%**，Depth 独有 46/2325 = **1.98%**；同时反向流失 32 / 26 个 —— **净增益接近抵消**。而 **89.8% 的漏检（469/522）在三种模态配置下全部漏掉**，这是与模态无关的失败模式。

**因此：瓶颈不在"缺哪个模态的信息"，而在"所有模态都给不出信息"的那批目标上。** 单模态优化最多触及约 2% 的检出，且其历史先例已返回零效应。

---

## 2. Frozen Official Baseline

```text
RGBID official mAP50-95 = 0.53180
```

| 层级 | 值 | 口径 | 来源 |
|---|---|---|---|
| **A. 线上榜单（冻结基线）** | **0.53180** | 比赛平台返回，用户转录 | `reports/final_competition_freeze_audit.md:13` 等 9 处 |
| B. 本地官方口径复现 | 0.50928 | `predict_rect.py` + `official_map.py`，400 图 val | `diagnostic/modality_dropout/eval_baseline.json` |
| C. 训练内 val（fork） | 0.55800 | `results.csv` ep237 | `runs/.../rgbird_ir_quicktest/results.csv` |
| 本地官方 → 线上偏移 | +0.02252 | — | `reports/conf_0001_online_submission_audit.md:99` |

**层级必须分开。** 本审计全文凡引用"官方"数值均显式标注 A / B / C。
> ⚠️ 自我更正：本会话早前生成的 `reports/small_object_crop_final_report.md` §4 与 `reports/small_object_crop_phase0_freeze.md` 把 0.53180 标为「官方 mAP50-95」而未区分 A/B 层，属表述不精确——正确写法是「线上 0.53180 / 本地官方口径 0.50928」。本报告采用后者口径。

冻结链完好性：`FREEZE_MANIFEST.sha256` 复核 **16/17 通过**，唯一不符项 `scripts/predict_rect.py` 系本会话之前既有漂移（`656aa58`→`6baa736`），非本次审计造成。

---

## 3. Current Modality Status

### 3.1 单模态 —— 全部为 TRAINING_ONLY

| 模态 | checkpoint | data | P | R | mAP50 | mAP50-95 | 最佳 ep | 状态 |
|---|---|---|---:|---:|---:|---:|---:|---|
| RGB | `runs/..._e0_yolo11n_rgb` | visible_split | 0.73312 | 0.49761 | 0.55653 | **0.32286** | 236 | TRAINING_ONLY |
| RGB | `runs/..._e1_yolo11s_rgb` | visible_split | 0.79847 | 0.64233 | 0.69085 | **0.44069** | 288 | TRAINING_ONLY |
| IR | `runs/..._yolo11_infrared_v2` | — | 0.54426 | 0.25450 | 0.27381 | **0.12162** | 149 | TRAINING_ONLY |
| Depth | `runs/..._yolo11_depth_v2` | — | 0.44509 | 0.27060 | 0.26330 | **0.14538** | 197 | TRAINING_ONLY |
| Depth | `runs/..._yolo11s_depth` | — | 0.75251 | 0.35231 | 0.38419 | **0.21728** | 145 | TRAINING_ONLY |
| Depth | `runs/..._yolo11_depth2` | — | 0.68601 | 0.38391 | 0.39820 | **0.22363** | 112 | TRAINING_ONLY |
| **RGBID** | `runs/..._rgbird_ir_quicktest/weights/best.pt` | rgbid_split | 0.83887 | 0.74868 | 0.77601 | **0.50928** | 237 | **OFFICIAL (B)** |

**为什么这些单模态数字不能当基线（逐条）**：

- **无一是 OFFICIAL 或 REPRODUCIBLE(B)** —— 全部只有 `results.csv` 里的训练内 val。
- **split 不同，且彼此也不同**：RGB 用 `visible_split`；`yolo11_depth_v2` / `yolo11s_depth` 用 `depth_split`；`yolo11_depth2` 用 **`depth_split_train_bal`**（类别过采样版，而该增广已被证伪为有害）；`yolo11_infrared_v2` 用 **`rgbt_split`**。**四个互不相同的 data 配置，无一为 `rgbid_split`。** 不同 split = 不同 GT 集合，横竖不可比。
- **imgsz 不同**：单模态均为 640；RGBID 为 1280。
- **世代不同**：单模态跑分 epochs 一律 200（实际 early-stop 在 112–197），RGBID 为 300；优化器/增广/patience 各异。
- **架构上并非真正的"单模态模型"**：仓库中**不存在 IR-only 或 Depth-only 的 model yaml**。这两个 run 复用 `yolo11_visible.yaml`（3ch），靠 `use_simotm=Depth|Infrared` + `pairs_rgb_ir` 让 loader 把**第二模态复制成 3 通道**再喂给 3ch 主干。所以它们是"3 通道 RGB 主干 + 单模态灰度复制"，不是为该模态设计的架构。
- **无预测落盘**：这些 run 目录下只有 `results.csv / results.png / tfevents / weights/`，**从未保存过 val 预测**，故无法用官方口径回算。
- **历史上被点名的 0.312 depth-only 与其 checkpoint 已不在磁盘**（`docs/model_design.md:92` 记录 0.31209 / 0.30267 / 0.12162；但 `runs/` 下已无 `urban_multimodal_det_yolo11_depth`、无 `..._rgb_baseline`）。且该 0.312 属后期被推翻的「假天花板」世代（见 [[rgbd-fusion-findings]]）。

> **解析陷阱（复核用）**：8 个 pre-E 系列 run 的 `results.csv` 是 **15 列布局、无 mAP75 列**，mAP50-95 在第 9 列而非第 10 列。按 16 列解析会把 `val/box_loss` 当成 mAP50-95，得到 ~4.0 的垃圾值。本表数值已按各 run 实际列数核对。

- **另有 1 个 run 目录无权重**：`runs/urban_multimodal_det_yolo11m_rgbd/` 仅有 `submission.zip`，无 `weights/`。

→ **结论：`REPRODUCIBLE` 级别的单模态基线在本仓库中不存在。**

### 3.2 模态消融 —— 唯一可信的「模态贡献」证据

对**冻结的正式模型**逐通道置常数（IR→125，Depth→0），同一模型 / 同一 split / 同一推理参数，官方口径（B）：

| 条件 | mAP50 | mAP75 | **mAP50-95** | P | R | 框数 |
|---|---:|---:|---:|---:|---:|---:|
| Full（RGB+IR+Depth） | 0.77601 | 0.56386 | **0.50928** | 0.8578 | 0.7029 | 4824 |
| **IR 移除** | 0.77385 | 0.55128 | **0.49990** | 0.8294 | 0.7154 | 4698 |
| **Depth 移除** | 0.77081 | 0.55179 | **0.49753** | 0.8209 | 0.7154 | 4809 |
| 边际贡献 | — | — | IR **+0.00937** / Depth **+0.01175** | | | |

**关键旁注**：移除 IR 会**提高 recall**（0.7029→0.7154）却**降低 mAP50-95**——即该模态带来的框整体更"值钱"但更少，是典型的 precision/recall 换手，而非信息量的大幅变化。

来源：`diagnostic/modality_dropout/eval_*.json`、`reports/rgbid_final_bottleneck_audit.md:298-303`

---

## 4. Per-class Comparison

### 4.1 正式模型逐类（官方口径 B）

| class | AP50-95 | AP50 | AP75 | 该类对 macro 的贡献 |
|---|---:|---:|---:|---:|
| person | 0.4394 | 0.7437 | 0.4327 | 0.0366 |
| boat | 0.4863 | 0.7918 | 0.5256 | 0.0405 |
| animal | 0.4921 | 0.7859 | 0.5111 | 0.0410 |
| seat | 0.6059 | 0.7552 | 0.6783 | 0.0505 |
| sign | 0.3784 | 0.6229 | 0.3537 | 0.0315 |
| bicycle | 0.3806 | 0.6143 | 0.4071 | 0.0317 |
| car | 0.5107 | 0.8234 | 0.5093 | 0.0426 |
| ball | 0.4139 | 0.6644 | 0.5175 | 0.0345 |
| light | 0.4950 | 0.7121 | 0.5504 | 0.0413 |
| garbage_can | 0.5338 | 0.8007 | 0.5482 | 0.0445 |
| uav | 0.5731 | 0.9978 | 0.7323 | 0.0478 |
| tricycle | 0.8020 | 1.0000 | 1.0000 | 0.0668 |

来源：`diagnostic/rgbid_inference_gain/summary_full.txt`、`reports/rgbid_final_bottleneck_audit.md:89-100`

### 4.2 逐类 · 逐模态 —— 仓库中不存在

> **`NOT AVAILABLE`**
>
> 按模态拆分的逐类 AP **在本仓库中不存在**。逐类 AP 只有三份，且**全部是 RGBD（无 IR）**：
> `reports/f1_class_metrics.csv`（F1 = RGBD 4ch 中期融合 @1024）、
> `reports/f4_class_metrics.csv`（F4 = RGBD 4ch @1280）、
> `reports/f5_class_metrics.csv`（F5 = YOLO11l RGBD @1280）。
>
> 无 IR-only / Depth-only / RGB-only 的逐类 AP。**故 brief §5 要求的「某模态明显擅长而 RGB 弱的类别」无法回答——不是没找到，是没有。**

唯一可用的**逐模态、逐类**证据是本研究新算的 §6 object-level 集合差（检出个数，非 AP）。

---

## 5. Scale Comparison

### 5.1 正式模型逐尺度（官方口径 B，n_gt = 371/1320/1116）

| 量 | small | medium | large |
|---|---:|---:|---:|
| AP50-95 | **0.05625** | **0.17984** | **0.61711** |
| AP50 / AP75 | 0.10286 / 0.06400 | 0.34098 / 0.18122 | 0.79204 / 0.69824 |
| R@0.5 | 0.67116 | 0.81364 | 0.86470 |
| R@0.5:0.95 | 0.38248 | 0.50871 | 0.65134 |
| TP 中位 IoU | 0.7375 | 0.7846 | 0.8940 |
| TP@IoU0.75 占比 | 45.8% | 60.1% | 84.2% |

来源：`diagnostic/final_bottleneck_audit/_summary.json`、`reports/rgbid_modality_dropout.md:133`、`reports/rgbid_final_bottleneck_audit.md:124-145`

**瓶颈在 small（AP50-95 = 0.056）且失败形态是定位**（small 的 TP 中位 IoU 0.7375，仅 45.8% 能过 0.75）。

### 5.2 IR / Depth 是否在 RGB 最弱的尺度上提供额外信息？—— **否**

本研究新算（§6 方法），逐尺度模态依赖度：

| 尺度 | Full TP | **IR 独有贡献** | **Depth 独有贡献** |
|---|---:|---:|---:|
| small | 218 | 4 (1.83%) | 2 (**0.92%**) |
| medium | 1074 | 21 (1.96%) | 28 (2.61%) |
| large | 1033 | 9 (0.87%) | 16 (1.55%) |

> **IR 不偏向小目标**（1.83% vs 1.96%，与 medium 基本持平）；**Depth 在小目标上贡献最低**（0.92%，低于 medium 的 2.61%）。
>
> 即：**两个模态的独立贡献都不是集中在 RGB 最薄弱的 small 尺度上**，而是集中在 medium。这与「靠 IR/Depth 补小目标」的假设方向相反。

### 5.3 三份 RGBD 逐类表（供参照，均无 IR）

F1/F4/F5 的逐类 AP50-95 见 `reports/f{1,4,5}_class_metrics.csv`（本报告不重复罗列，避免与 `reports/f4_experiment_report.md`、`f5_experiment_report.md` 重复）。

---

## 6. Error Complementarity

brief §7 要求的 "RGB FN → IR/Depth 是否能发现" 类指标，仓库中**从未计算过**。本审计用磁盘上**已存在**的三组 400 图预测 TXT 现算：

- `diagnostic/rgbid_inference_gain/RGBID_baseline/results/`（Full）
- `diagnostic/modality_dropout/RGBID_base_ir_dropped/results/`（RGB+Depth）
- `diagnostic/modality_dropout/RGBID_base_depth_dropped/results/`（RGB+IR）

**方法**：native 归一化 xywh → xyxy；逐类贪心匹配 IoU ≥ 0.5；400 图全部命中，GT **2847** 框（注意：仓库其他报告记 2807，差额 40 来自 2 张损坏图的排除；本表未排除）。

### 6.1 总量

| 指标 | 值 | 占 Full TP / Full 漏检 |
|---|---:|---:|
| GT 总数 | 2847 | — |
| Full TP | 2325 | — |
| **IR 与 Depth 同时移除后仍命中** | **2259** | **97.15% of TP** |
| **IR 独有贡献**（Full 中、移除 IR 即丢） | **34** | **1.46% of TP** |
| **Depth 独有贡献** | **46** | **1.98% of TP** |
| Full 漏检总数 | 522 | — |
| **其中三种配置全部漏掉** | **469** | **89.8% of FN** |
| Full 漏、但移除 IR 反而命中 | 32 | 6.1% of FN |
| Full 漏、但移除 Depth 反而命中 | 26 | 5.0% of FN |
| Full 漏、但两配置都命中 | 5 | 1.0% of FN |

### 6.2 集合关系（brief §7 要求的交集）

| 交集 | 值 |
|---|---:|
| RGB 单独充分（IR、Depth 皆可移除仍命中） | 2259 |
| IR 独有 TP（= Full∩¬IRdrop） | 34 |
| Depth 独有 TP（= Full∩¬Ddepthdrop） | 46 |
| Full 漏 ∩ IRdrop 漏 ∩ Ddepthdrop 漏 | **469** |

> brief §7 所列 `RGB_FN ∩ IR_FN` 等**模态对模态**交集 **`NOT AVAILABLE`** —— 需要 RGB-only / IR-only / Depth-only 三个独立模型在同一 split 上的预测，而这些预测从未生成（§3.1）。上表是**同模型通道消融**下的等价物。

### 6.3 逐类

| class | GT | Full | IRdrop | Ddrop |
|---|---:|---:|---:|---:|
| person | 1073 | 873 | 873 | 870 |
| boat | 28 | 26 | 26 | 26 |
| animal | 733 | 603 | 604 | 593 |
| seat | 107 | 84 | 85 | 84 |
| sign | 150 | 109 | 109 | 111 |
| bicycle | 116 | 85 | 86 | 78 |
| car | 288 | 254 | 253 | 251 |
| ball | 19 | 13 | 14 | 14 |
| light | 244 | 198 | 193 | 200 |
| garbage_can | 57 | 48 | 49 | 46 |
| uav | 30 | 30 | 29 | 30 |
| tricycle | 2 | 2 | 2 | 2 |

逐类差异全部在**个位数**：IR 最集中在 `light`（198→193，5 个）；Depth 最集中在 `bicycle`（85→78，7 个）。

### 6.4 ⚠️ 本分析的方法学上限（必须与数字同读）

消融是把通道**置常数**（IR→125、Depth→0）。这**不是模型训练时见过的输入分布**，属 OOD。因此：

- 移除条件下的预测劣化**混入了分布外伪影**，不完全等于"信息移除"；
- 该偏差方向不确定，可能**低估**（伪影掩盖了真实贡献）也可能**高估**（伪影本身造成误检）；
- 因此上表应读作**量级参考**，不是精确归因。

**即便打上这个折扣，结论方向不变**：净变化（IR +34/−32、Depth +46/−26）在 2325 个 TP 的量级上接近抵消，与 §3.2 的聚合边际贡献（+0.9% / +1.2% mAP）互相印证。

---

## 7. Historical Evidence

### 7.1 已关闭的方向（不得重新提议）

| 方向 | 结果 | 数值 | 来源 |
|---|---|---|---|
| **IR CLAHE（Exp D）** | **`D_REJECTED`** | 官方(B) 0.51076 vs 0.50928（+0.00148），**bootstrap 四指标 CI 全含 0**，P(Δ≤0)=0.407，mAP75 −0.06225 | `reports/rgbid_ir_clahe_result.md` |
| **Small Object Crop** | **`CROP_NO_GO`**，从未实现 | crop 放大 1.16–1.58× < mosaic 缩小 2.00× | `reports/rgbid_small_object_crop_audit.md` |
| **P2 检测层** | **`P2_REJECTED`** | 0.50232 vs 0.50928；small +0.00236 / large −0.01399 | `reports/rgbid_p2_probe.md` |
| **1536 分辨率** | 证伪（两探针） | 0.49401 / 0.50673，均低于基线；small AP 仅 +0.0019 | `reports/rgbid_1536_finetune_report.md` |
| **Modality Dropout** | **`MODALITY_DROPOUT_REJECTED`** | 0.49029 vs 0.50928（−0.01899） | `reports/rgbid_modality_dropout.md` |
| **P3 注意力探针** | 证伪 | 0.50251 vs 0.50928 | `reports/rgbid_p3attn_probe_report.md` |
| **conf = 0.0001** | **线上证伪** | 线上 0.50451 vs 0.53180（−2.729 分） | `reports/conf_0001_online_submission_audit.md` |
| **Ensemble（1:1/2:1/3:1）** | **`FORBIDDEN_BY_RULES`** | 官方明令禁止多模型集成 | `submissions/rgbid_f4_ensemble_3to1_candidate/FORBIDDEN_BY_RULES.md` |
| **F5 YOLO11l 容量** | 证伪 | 0.52619 vs F4 0.53273 | `reports/f5_experiment_report.md` |
| **Depth 表征杠杆**（log 归一化 / 离线几何增强 / 类别过采样 / 融合算子 9 轮） | 全部证伪 | 0.112 / 0.215 / 0.215 / ≤0.312 | [[rgbd-fusion-findings]] |

**RGBT**：`experiments/urban_multimodal_det_yolo11_rgbt/metrics.json` 存在（0.25745），但为早期小分辨率实验，非当前管线，且无官方评估。

### 7.2 决定性一条

> **"先优化单模态、再融合"这个策略本身，在本项目已经以 IR CLAHE 的形式被完整执行并关闭。**
>
> IR CLAHE 改进的是 **IR 单模态的表示**（对比度/动态范围），检验的是**融合模型的官方 mAP**——这正是本 brief 设想的策略。结果是点估计 +0.00148、配对 bootstrap 四指标 CI 全含 0。

唯一未被该先例覆盖的变体是「**Depth** 单模态优化」与「**RGB** 单模态优化」。前者见 §7.1 末行（Depth 表征杠杆已全线证伪）；后者见 §9。

---

## 8. Single-Modality → Fusion Risk

### 8.1 为什么「单模态提升 ≠ 必然带来 RGBID 提升」

结合本项目已有证据，至少四条机制会截断传递：

1. **融合已经吃掉了该模态的信息。** §6.1：97.15% 的 TP 在移除 IR 与 Depth 后仍存在，说明 RGB 路径已承载主要信息；单模态 mAP 提高多半是在**放大融合已经拥有的信号**，而非注入新信号。
2. **提升可能只是 calibration。** §3.2 的直接观察：移除 IR 会**提高 recall**（0.7029→0.7154）却降低 mAP50-95。若单模态优化主要改变置信度分布而非定位精度，融合端只会看到 P/R 换手。
3. **提升可能落在与融合误差不重叠的区域。** §6.1：89.8% 的漏检是三种模态配置**共同**漏掉的。单模态变强，改善的是那 2% 的可归因目标，而非占 90% 的共同失败。
4. **本项目已有同类机制的负向实证。** CLAHE（§7.2）与 modality dropout 都是"针对模态本身做的干预"，两者均未迁移到融合指标。

### 8.2 直接先例：单模态指标上升，而融合指标几乎不动（本项目已实测）

brief §10 要求专门寻找「单模态指标提升 → 融合指标没有提升」的实例。**仓库里有一个，而且是同一策略的实例**：

**IR CLAHE（Exp D）= 优化 IR 单模态表示 → 接回 RGBID 融合**

| 口径 | baseline | IR CLAHE | Δ |
|---|---:|---:|---:|
| **训练内 val（fork，C 层）** | 0.55800 @ep237 | **0.56806 @ep263** | **+0.01006** |
| **官方口径（B 层）** | 0.50928 | 0.51076 | **+0.00148** |
| 传递率 | — | — | **≈ 14.7%** |
| 官方 mAP75 | 0.56386 | 0.50161 | **−0.06225** |
| 配对 bootstrap（400 次） | — | — | Δ CI **[−0.00653, +0.00920] 含 0**，P(Δ≤0)=0.407 |

> **IR CLAHE 是全仓库训练内 val 最高的单模态侧改动**（0.56806，高于正式基线 0.55800，也高于 P2 探针 0.55856）。
> **但它的官方口径只动了 +0.00148，且 bootstrap CI 跨零；同时 mAP75 掉 0.062。**
>
> 这就是「单模态/内部指标变好，融合指标没有变好」的**本项目实测版本**，而非理论推演。
> 它同时给出传递率的经验上界：**内部 +0.01006 → 官方 +0.00148，约 15%。**

### 8.3 反方向风险（单模态优化反而有害）

- **Modality Dropout** 是明确先例：让模型对模态扰动更鲁棒，融合 mAP 反降 0.01899。
- **1536 微调** 两探针均低于基线，且训练内 val 一度更高（§7.1）——**单模态/内部指标向好而官方指标下降**在本项目已发生过。
- **E9 depth padding / E8 nearest**：depth 预处理改动被证伪，说明动单模态预处理会静默改变融合行为。

### 8.4 本审计的诚实保留

**支持**单模态路线的证据也存在，不应隐去：

- IR 与 visible 的像素级相关为**弱负**（mean −0.067 / median −0.041；52.2% 为负；32.5% 的 |corr| < 0.3）——IR **不是** visible 的复制，确实携带独立信息。这与「IR 无互补」的直觉相反。
- 消融显示 IR / Depth 的边际贡献**为正**（+0.00937 / +0.01175），方向正确。

**但这些证据只支持"模态有信息"，不支持"优化该模态能带来融合增益"** —— 前者已被 §3.2 证实，后者才是本 brief 要回答的问题，而它被 §7.2 的先例与 §6 的量级否证。

### 8.5 附带发现（数据层，非本轮任务但影响判断）

- **visible 本身高度去色**：mean saturation 0.118、通道间平均绝对差 ~6.7/255、corr(B,G) 均值 0.961 / 中位 0.975。可见光分支的三通道**彼此高度冗余**。
- **IR 为 8-bit 灰度复制**：实测红外图 `corr(B,G) = 0.9999`（中位 1.0000）——三通道是同一灰度的复制品，**1 个有效通道被存成 3 个**。热像原生 12/14-bit 已损失——见 [[ir-modality-findings]]。这限制了 IR 单模态的可优化上限。
- 采样 400 图中 5 张的 IR 或 depth 为**常数图**（无方差）。

---

## 9. Candidate Decision

| 候选模态 | 当前能力 | 明显弱点 | 对其他模态的独立互补证据 | 历史优化证据 | 融合传递潜力 |
|---|---|---|---|---|---|
| **RGB** | 最强单模态（E1 RGB-only 0.44069，TRAINING_ONLY）；承载 97.15% 的融合 TP | 去色（S≈0.12）；三通道冗余（corr 0.96–0.98） | 不适用（它是被补的一方） | 无独立的"RGB 单模态优化"实验；E0→E1 的 n→s 增益属主干线本身 | **INSUFFICIENT_EVIDENCE** |
| **IR** | 0.12162（TRAINING_ONLY，旧世代，8-bit 数据定损） | 单模态天花板极低；数据已量化；移除 IR 反而提高 recall | **弱正**：独有 TP 34/2325 = 1.46%；与 visible 弱负相关（携带独立信息） | **IR CLAHE 已测已否**（`D_REJECTED`，CI 全含 0）；IR 预处理 3 份拷贝、改一处不通 | **CONTRADICTED_BY_HISTORY** |
| **Depth** | 最强单模态旁证（depth-only 0.21728 / 0.22363，TRAINING_ONLY） | 小目标贡献最低（0.92%） | **弱正**：独有 TP 46/2325 = 1.98%（三模态中最高） | **表征杠杆全线证伪**：log 归一化 0.112、离线几何增强 0.215、类别过采样 0.215、融合算子 9 轮 ≤0.312 | **CONTRADICTED_BY_HISTORY** |

> 按 brief 要求，**不使用**任何排序 / 评分 / "最强""最优"表述。上表为分类标签。

---

## 10. Final Next Step

```text
NO_GO / INSUFFICIENT_EVIDENCE
```

**门槛逐条核对（brief §11）**：

| # | 条件 | 判定 | 依据 |
|---|---|---|---|
| 1 | 有可信的单模态 baseline | ❌ | §3.1：全部 TRAINING_ONLY，split/imgsz/世代不一致，无预测落盘 |
| 2 | 有明确、未解决的性能瓶颈 | ✅ | §5.1：small AP50-95 = 0.056；定位不足 |
| 3 | 有证据说明该模态有独立互补信息 | ❌ | §6：1.46% / 1.98%，且**不落在 small 尺度**；净增益接近抵消 |
| 4 | 该机制未被历史实验否定 | ❌ | §7.2：IR CLAHE = 本策略，已 `D_REJECTED` |
| 5 | 不与 Crop / CLAHE 已关闭方向重复 | ❌ | CLAHE 正是本策略的直接先例 |
| 6 | 不需大规模改 fusion 架构 | ✅ | — |
| 7 | 可用现有官方 RGBID evaluation | ✅ | `predict_rect.py` + `official_map.py` 链完好 |
| 8 | 不会破坏 0.53180 | ✅ | 冻结链 16/17 完好 |
| 9 | 成本与预期收益匹配 | ❌ | 11h GPU 换取对约 2% 检出量的机制，其先例 ΔCI 含 0 |

**9 条中 5 条不通过。**

### 为什么不给 `PROCEED_WITH_DIAGNOSTIC_ONLY`

诊断路径（例如把 §6 的互补性分析补成完整的三模态独立模型对照）确有独立价值，但它**不会回答"是否值得训练"这个问题**——因为拦住训练的是 §7.2 的先例（策略已被测过并被否）与 §3.1 的基线缺失（要补基线本身就要训练）。诊断会消耗时间而不改变结论方向。

### 若用户不接受本判定，唯一的解锁条件

需由用户提供**新的、可推翻 §7.2 的信息**，例如：

- 证明 IR CLAHE 的失败是**实现问题**而非机制问题（可检验：检查 3 份 IR 预处理拷贝是否一致，见 [[ir-preprocessing-three-copies]]）；或
- 指出一个**机制上不同于"提高单模态表观质量"**的单模态优化方向（即不是再改对比度/分辨率/增强）。

在缺少上述证据前，**不应启动任何单模态优化训练**。

---

## 11. 合规状态

| 项 | 状态 |
|---|---|
| 训练 | ❌ 未执行 |
| 模型结构 / dataset / dataloader / augmentation / trainer / evaluator | ❌ 一字未改 |
| 正式 RGBID 配置 | ❌ 未修改 |
| `best.pt` / `last.pt` / `submission.zip` | ✅ 逐字节未变（§2 冻结复核 16/17） |
| submission | ❌ 未生成 |
| 新增文件 | 仅本报告 |
| 未提交修改（只读记录，未修复） | `reports/rgbid_ir_clahe_result.md`（+46 行）、`submissions/rgbid_d_ir_clahe_candidate_infer.log`（4 行）——均系本会话之前的既有改动，**非本次审计产生** |

```text
OFFICIAL_MODEL_REMAINS_0.53180
```
