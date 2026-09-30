# F1 — Native-small Synchronized CopyPaste · 训练结果分析

**日期**：2026-09-30
**run**：`runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_f1/`（300 ep 完成，无中断）
**性质**：只读分析。训练已由你完成；本轮**未训练**、**只做 val 推理一次**（冻结官方 pipeline）用于 §30 的官方判决。

---

## 1. EXECUTIVE VERDICT

```text
F1 official mAP50-95:   0.51865      （D′ = 0.51528）
Δ vs D′:                +0.00336      （mAP50 +0.01222，mAP75 +0.01169）
Online score:           N/A — 未提交（fork 口径给出**反号**信号，见 §5）
Verdict:                YELLOW
```

**§31 阈值判定**：GREEN 需 >0.52228 → **未达到**；YELLOW 带 [0.50828, 0.52228] → **在带内**；RED <0.50828 → 否。
**§32 配对 bootstrap**：Δ = +0.00336，**95% CI [−0.00706, +0.01821] 含 0**，**P(Δ>0) = 0.780**。

```text
§39 FINAL DECISION:  CLOSE
```

理由（§39 的 KEEP 需同时满足「官方达到预设 improvement threshold」**且**「不是由其他尺度牺牲换来」）：
**官方未达到 0.52228 阈值** ⇒ 第一条即不满足。（第二条**反而满足**：F1 的 small 是正的、medium/large 是负的，
即它是**用 medium/large 的小幅回吐换 small 的小幅上升**，净额才 +0.0034。）

---

## 2. IMPLEMENTATION INTEGRITY

`runs/..._f1/args.yaml` vs `runs/..._sepstem_clahe/args.yaml`（D′）逐项对照，**40 个关键项全部相同**：

```text
model unchanged:      ✅ configs/yolo11m_sepstem.yaml（同一文件）
loss unchanged:       ✅ box=7.5 / cls=0.5 / dfl=1.5 / cls_pw=[] 全同
TAL unchanged:        ✅ 同一份 tal.py + loss.py（本轮未改动这两文件）
optimizer unchanged:  ✅ SGD / lr0=0.005 / lrf=0.01 / momentum=0.937 / wd=0.0005 / warmup=3.0 / cos_lr=True
dataset unchanged:    ✅ 同一 dataset.yaml
validation unchanged: ✅ 同一 val split；val 预处理未动
imgsz/epochs/batch:   ✅ 1280 / 300 / 8
seed:                 ✅ 42（deterministic=True）
增强:                  ✅ mosaic=1.0 / close_mosaic=10 / translate=0.1 / scale=0.5 / degrees=shear=perspective=0
                         fliplr=0.5 / flipud=0 / mixup=0 / **copy_paste=0**（原生 copy_paste 未启用）
ir_encoding:          ✅ clahe
resume:               ✅ False；pretrained=yolo11m.pt
only F1 replay changed:✅ F1 独有键 = 6 个 OASA 默认（全 false/惰性）+ **native_small_replay=1.0**
                         + **native_small_replay_max_new_instances=2** + 4 个冻结常数
```

**训练变量差异集合 = 空**（`name` 是 experiment_name 派生，只决定 run 目录）。

---

## 3. PRETRAIN / CHECKPOINT

```text
initial checkpoint:        yolo11m.pt
SHA256:                    d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95
final best.pt SHA256:      9c6f524e04eb373a1acc0dd25c6242336136ee0ca844a69d7ae574ff37194c75
final last.pt SHA256:      3993ac6eb1633a365199afcd756f3a163fd491b97f8321934022c4d5b7f04d58
D′ best.pt（未变）:         1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda
best epoch（fork-val）:     224 / 300
```

§30 的**基线复现门 PASS**：用同一冻结 evaluator 重算 D′ 的官方 mAP50-95 = **0.5152807421** vs 记录 0.51528（Δ=1e-06）。

---

## 4. EXPOSURE CHANGE（§27/§28）

```text
D′  small exposure:   1392 native-small instance / epoch   （进入 augmentation pipeline 之前）
F1  small exposure:   1392 + 401×2 = 2194 / epoch
multiplier:           **1.576×  (+57.6%)**       （preflight 已预注册，见 _exposure.json）
```

**实际 replay 率**：

| 口径 | 值 | 来源 |
|---|---|---|
| eligible 条件下 | **100%**（每个 eligible 样本都拿到满额 2 个，`skip_attempts=0`） | preflight 实测 169 样本 |
| 每样本 | 25.06% | = eligible 率，因 p=1.0 |
| 每样本新增实例数 | 1.183（前 20 样本口径 2.0） | preflight 实测 |

**⚠ 训练期没有插桩 replay 计数器** —— §27/§28 要求的「训练过程中实测 replay 率」**无法从落盘 artifact 直接读取**。
可用的**间接证据**：

1. `args.yaml` 里 `native_small_replay=1.0`、`max_new_instances=2` 确实生效（不是静默回落）；
2. **epoch 1 的 train loss 就与 D′ 分叉**（box 1.46066→1.49996、cls 2.40433→2.42935、dfl 1.27032→1.29095）
   ⇒ **数据管线确实改变了**，replay 变换在训练中被执行（若未执行，同 seed 同配置应逐位相同）；
3. preflight 在同一份代码上实测 100% 满额。

**⇒ exposure 增幅 = +57.6%（按预注册规则的第一性重建）；"训练时实际跑到了这个量"属于间接推断，不是直接测量。**

---

## 5. OFFICIAL METRICS

| metric | D′ | F1 | Δ |
|---|---:|---:|---:|
| **mAP50-95** | 0.51528 | **0.51865** | **+0.00336** |
| mAP50 | 0.77826 | **0.79048** | +0.01222 |
| mAP75 | 0.51096 | **0.52265** | +0.01169 |
| pred 总数 | 4482 | 4793 | +311 |
| TP@IoU.50 | 2295 | 2305 | +10 |
| 最大召回 | 0.8176 | 0.8212 | +0.0036 |

`counts`：images 398 / GT 2807 / pred_invalid 0 / imgs_truncated 0 / max_boxes_seen 100。
**回归测试**：predictor 与 evaluator 均为冻结管线（cap=100、无非法行、无越界截断异常）；D′ 复现门 PASS。

### 5.1 ⚠ 口径分歧（必须同读）

| 口径 | D′ | F1 | Δ |
|---|---:|---:|---:|
| **官方 evaluator（§30 判决口径）** | 0.51528 | **0.51865** | **+0.00336** |
| fork（训练内 val，best） | 0.56690 (ep246) | 0.56485 (ep224) | **−0.00205** |
| fork（末 epoch） | 0.55522 | 0.54582 | −0.00940 |
| fork mAP50 | 0.82652 | **0.83700** | +0.01048 |
| fork mAP75 | 0.58889 | **0.60446** | +0.01557 |

⇒ **两个口径给出相反符号。** 按 [[metric-instrument-fork-over-official]]（**同架构族内 fork 排序=线上**），
F1 的**线上 delta 存在为负的风险**。§30 规定官方口径为判决依据 ⇒ 本报告按官方判 YELLOW；
但**上线前应把这条反号信号计入**。

---

## 6. BOOTSTRAP（§32，B=400，配对 image-level，两模型同一重采样索引）

| 量 | 点估 Δ | bootstrap 均值 Δ | 95% CI | P(Δ>0) |
|---|---:|---:|---|---:|
| **Δ mAP50-95** | **+0.00336** | +0.00510 | **[−0.00706, +0.01821] 含 0** | **0.780** |
| Δ mAP50 | +0.01222 | +0.01297 | [−0.00884, +0.03261] 含 0 | 0.887 |
| Δ mAP75 | +0.01169 | +0.01084 | [−0.01862, +0.04409] 含 0 | 0.775 |
| Δ IoU .50–.60 段 | +0.01158 | +0.01232 | [−0.00748, +0.03162] 含 0 | 0.880 |
| Δ IoU .65–.75 段 | +0.00967 | +0.00962 | [−0.00695, +0.02907] 含 0 | 0.843 |
| Δ IoU .80–.95 段 | −0.00753 | −0.00371 | [−0.01630, +0.00996] 含 0 | 0.290 |

⇒ **主指标 Δ 与 0 不可区分**（P(Δ>0)=0.78，CI 含 0）。§31 的 YELLOW 条款因此适用：
**「improvement 只是 noise ⇒ 不继续扩大 F1 sweep」**。

**IoU 区间分解（官方 10 阈值）**：

```text
0.50 +0.0122   0.55 +0.0130   0.60 +0.0095   0.65 +0.0120   0.70 +0.0053
0.75 +0.0117   0.80 +0.0037   0.85 +0.0076   0.90 −0.0018   0.95 −0.0395
```

⇒ **增益集中在 IoU 0.50–0.85，在最严阈值 0.95 上大幅回吐（−0.0395）**。
这正是 fork（mAP50/mAP75 升、mAP50-95 反而略降）与官方（净升但很小）符号差异的来源。

---

## 7. SCALE DECOMPOSITION（§33，native 面积桶）

| bucket | GT n | D′ | F1 | Δ | Δ重归一化 | 含 GT 类数 |
|---|---:|---:|---:|---:|---:|---:|
| all | 2807 | 0.5153 | 0.5186 | **+0.0034** | +0.0034 | 12 |
| **small** | 371 | 0.0477 | 0.0545 | **+0.0068** | **+0.0074** | 11 |
| medium | 1320 | 0.1730 | 0.1706 | **−0.0024** | −0.0026 | 11 |
| large | 1116 | 0.5790 | 0.5750 | **−0.0040** | −0.0043 | 11 |

⇒ **这是本实验最值得记录的一条：预注册的机制方向命中了 —— small ↑、medium/large ↓。**
净额 +0.0034 = (+0.0068 small) + (−0.0024 medium) + (−0.0040 large)。
⇒ 但 **small 的 +0.0068 本身也含 0**（bootstrap [−0.00453, +0.01586]，P=0.882），**不能称为确证的 small 改善**。

---

## 8. CLASS DECOMPOSITION（§34，全部 12 类）

| class | GT n | D′ | F1 | Δ | 95% CI | P(Δ>0) |
|---|---:|---:|---:|---:|---|---:|
| tricycle | 2 | 0.7525 | 0.7020 | **−0.0505** | [−0.0505, +0.0000] | 0.000 |
| sign | 150 | 0.4014 | 0.3738 | −0.0276 | [−0.0567, +0.0006] | 0.030 |
| car | 288 | 0.5420 | 0.5244 | −0.0175 | [−0.0329, +0.0055] | 0.072 |
| boat | 28 | 0.5340 | 0.5339 | −0.0001 | [−0.0788, +0.0706] | 0.465 |
| uav | 30 | 0.5344 | 0.5372 | +0.0028 | [−0.0370, +0.0408] | 0.545 |
| animal | 731 | 0.5066 | 0.5095 | +0.0029 | [−0.0111, +0.0180] | 0.660 |
| seat | 107 | 0.6306 | 0.6337 | +0.0031 | [−0.0189, +0.0245] | 0.600 |
| ball | 19 | 0.4044 | 0.4080 | +0.0036 | [−0.0951, +0.0793] | 0.590 |
| person | 1045 | 0.4517 | 0.4571 | +0.0054 | [−0.0041, +0.0155] | 0.838 |
| **light** | 244 | 0.5421 | 0.5759 | **+0.0338** | **[+0.0115, +0.0533]** | 1.000 |
| bicycle | 106 | 0.3931 | 0.4290 | +0.0359 | [−0.0025, +0.0907] | 0.968 |
| **garbage_can** | 57 | 0.4907 | 0.5392 | **+0.0485** | **[+0.0067, +0.1024]** | 0.993 |

⇒ **CI 不含 0 的只有 3/12**：tricycle（n=2，忽略）、**light +0.0338**、**garbage_can +0.0485**。
bicycle +0.0359 边缘（P=0.968）。
⇒ 增益集中在 **light / garbage_can / bicycle**，而 **sign / car 为负**（car n=288，Δ=−0.0175，P(Δ>0)=0.072）。
**不挑好看的类别：12 类全列**。

---

## 9. SMALL-OBJECT SPECIFIC RESULT（§35 的唯一问题）

> **F1 是否真正改善 native-small detection？**

**检出率（official IoU≥0.5，同类匹配）**：

| bucket | GT n | D′ detected | F1 detected | Δ rate | Δ 个数 |
|---|---:|---:|---:|---:|---:|
| **small** | 371 | 223 (0.6011) | **224 (0.6038)** | **+0.0027** | **+1** |
| medium | 1320 | 1070 (0.8106) | 1071 (0.8114) | +0.0008 | +1 |
| large | 1116 | 1019 (0.9131) | 1040 (0.9319) | **+0.0188** | **+21** |

⇒ **答案：没有实质改善。** native-small 只多检出 **1 个目标**（223→224）。
**最大的检出率提升在 large（+21 个目标，+0.0188）** —— 与「加小目标曝光」的意图相反。
small 的 mAP +0.0068 来自**已检出目标的排序/定位改善**，不是新检出。

**§35 的明令因此适用：`不要把结果解释为 small-object solution`。**

### 9.1 §36 G1 follow-up（冻结定义未改）

| subset | n | D′ #IoU==0 | F1 #IoU==0 | D′ 检出 | F1 检出 |
|---|---:|---:|---:|---:|---:|
| **G1**（any-class IoU==0） | 38 | **38** | **33** | 0 | **2** |
| non-G1 native-small | 333 | 32 | 34 | 223 | 222 |

⇒ **F1 撬开了 38 个 G1 中的 5 个（2 个真正检出、3 个变成部分重叠）**；但 non-G1 小目标净 −1，
所以 small 总体只 +1。**这是「少数硬样本被撬动」而不是「小目标面改善」。**

---

## 10. FINAL DECISION

```text
CLOSE
```

**理由**：
1. 官方 mAP50-95 = 0.51865，**未达 §31 的 GREEN 阈值 0.52228**；
2. 配对 bootstrap **Δ 与 0 不可区分**（CI 含 0，P(Δ>0)=0.78）⇒ 落在 §31 的「improvement 只是 noise」分支
   ⇒ **不继续扩大 F1 sweep**（禁止 p=0.2/0.4/0.5/0.8）；
3. **small detection rate 未改善**（+1/371），而检出率增益主要落在 **large** ⇒ 不是 small-object solution；
4. fork 口径给出**反号**信号（−0.00205），按同族 fork-线上对应关系，线上 delta 有负风险。

**但必须同时记录本实验的正面信息**（供下一轮决策，不作翻案）：

```text
机制方向命中：small Δ = +0.0068 / medium −0.0024 / large −0.0040
             —— 预注册的「加小目标曝光 → 小目标受益」方向被数据支持，只是幅度落在噪声带内。
G1 被撬动：  38/38 零重叠 → 33/38（2 个检出）—— 与「G1 是特征/表征失败」的既有结论一致：
             连 +57.6% 的小目标曝光也只能撬动 38 个中的 2 个。
```

⇒ **不推翻 §37 的表述纪律**：结论只能是「**当前 replay protocol 未产生达到阈值的官方增益**」，
**不能**说「small-object exposure 没用」。

---

## 11. Provenance

```text
TRAINING（本轮）        = 0
BACKWARD / OPTIMIZER    = 0
本轮新增推理            = 1 次 val 推理（400 图，冻结 predict_rect.py = rect/conf0.001/iou0.7/imgsz1280/max_boxes100）
                        —— 仅用于 §30 官方判决；未提交、未生成 submission
D′ checkpoint 未变      = 1cae45f75693f541…fae4fda
D′ 冻结预测未变          = diagnostic/sepstem_clahe/best_full/results（未写入）
既有文件修改            = 0（本轮只新增 diagnostic/f1_native_small_replay/ 下的分析脚本与产物）
```

**本轮新增产物**：

```text
diagnostic/f1_native_small_replay/
    RESULT.md                      ← 本报告
    infer_val/results/*.txt        ← F1 best.pt 的 val 预测（400 个，供复核）
    _f1_official.json              ← F1 官方 evaluator 结果
    _dprime_official.json          ← D′ 官方重算（复现门）
    _f1_vs_dprime_boot.py/.json    ← 配对 bootstrap + 尺度/类别/IoU 分解（改编自 E1 的同一脚本）
    _boot.log                      ← 其完整输出
    _small_object_analysis.py/.json/_small.log  ← §35 检出率 + §36 G1 follow-up
    _infer.log / _exposure.json / _hashes_after.txt
```

---

**本轮结束，停止。不训练、不提交、不扩大 sweep。等你决定是否走线上验证。**
