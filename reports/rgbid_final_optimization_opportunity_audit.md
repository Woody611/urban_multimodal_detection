# RGBID 最终优化机会审计（READ-ONLY）

**日期**：2026-09-18
**性质**：**READ-ONLY**。未训练、未创建/修改任何 YAML、未修改源码、未修改 dataset / checkpoint / submission / 正式 runs、未重新预测、未上传线上。
**目标**：判断「在不重新训练或极低风险的前提下，是否还有符合规则、能提升 official mAP50-95 的机会」。

---

## 0. Executive Summary

> # `NO_SAFE_OPPORTUNITY`
>
> 逐项核验后，**没有任何**同时满足「不需要训练 + 规则允许 + 有正面证据」的机会。
>
> 最接近的两个都不成立：
> - **TTA multi-scale**：rule **UNKNOWN**，本地 official **+0.00432**，但
>   **对 small-object AP 的增益实测为 0（0.05600 vs 0.05625）** —— 收益全部来自大目标；
>   且退化框从 65 增至 188、覆盖文件从 7.5% 升至 12.3%，正是 conf=0.0001 **线上 −2.729** 的同款失效模式。
> - **其余 8 个 checkpoint**：官方口径**全部低于 0.50928**（最高 ep290 = 0.50742）。
>
> **正式保底 `0.53180` 应保持不变。**

> ⚠️ **必须先披露一件事**：上一轮我用来评估 checkpoint 的后台链**未被真正终止**（`TaskStop` 只杀掉了包装进程，bash 子进程继续运行），
> 因此在用户指示「只需要做 240/260/280/290」之后，它**又额外评估了 200 / 230 / 250 / 270**。
> 本次审计发现后已彻底终止该进程（含其 python 子进程），并清理了被中断的 `ep210` 临时目录。
> 额外产出的 4 组结果**完整且有效**（与既有 4 组同口径、同脚本），故在本报告中作为证据使用；
> 但这属于**超出授权范围的计算**，在此明确记录。
> 另：`ep240` / `ep260` 由两条链**并发重复推理**，两次独立运行得到**完全相同的官方值**（0.50097 / 0.50146），
> 反而构成一次推理确定性的验证。

---

## 1. 比赛规则逐项核对（§1）

**规则来源**：赛题原文 PDF `面向城市场景的视觉多模态目标检测-1.pdf`
（SHA256 `37d204800ccbde5a9f748647b08326d99f5d0301367f574a695f16528217a832`，8 页）。

> ⚠️ **该 PDF 当前已不在仓库中**（`find -iname "*.pdf"` 无命中，此前 git status 曾显示为未跟踪文件）。
> 因此本轮**无法重新提取原文**，以下引用的是
> `reports/p0_rule_and_candidate_audit.md`（2026-09-17）中**逐字记录**的原文条款，未做任何转述或推断。

### 1.1 已确认的原文条款

| 出处 | 原文 |
|---|---|
| 九(二) | 「每张测试图必须提交一个同名的预测 TXT 文件……**若某张图未检测到目标，也须提交一个空 TXT 预测文件，不允许缺失**。此外，**每张图最大预测框数量为 100**，超过数量会按置信度截断；**出现非法类别、非法坐标或置信度缺失，将导致该预测无效**，不参与结果运算。」 |
| 九(三) | 「依据性能评估指标 **mAP@50-95** 进行打分。**注：若参赛者提交结果低于赛事方公布的基线成绩，赛事方有权将其认定为无效成绩。**」 |
| 十一(一) | 「禁止调用任何在线服务或 API……禁止采用手工标注测试集目标位置的方式生成结果，所有预测结果必须由算法模型自主推理输出，严禁任何形式的人工干预测试结果生成过程。**禁止将多个不同结构或训练阶段的模型进行简单集成，直接采用投票法、平均法等方式决策。**」 |
| 十一(二) | 「仅可使用赛事官方提供的训练数据集，严禁私自扩充外部数据。允许使用 ImageNet、COCO、Objects365 等公开预训练权重。……」 |
| 二 | 「confidence 范围为 **[0,1]**」 |

### 1.2 逐项判定（严格按「原文无条款 = UNKNOWN，不解释为允许」）

| # | 检查项 | 原文是否有条款 | **判定** |
|---|---|---|---|
| 1 | inference image size restrictions | ❌ 无 | **UNKNOWN** |
| 2 | TTA | ❌ 无 | **UNKNOWN** |
| 3 | multi-scale inference | ❌ 无 | **UNKNOWN** |
| 4 | horizontal flip | ❌ 无 | **UNKNOWN** |
| 5 | multiple predictions（每图多框） | ✅ 有（≤100 框/图） | **ALLOWED（上限 100）** |
| 6 | multiple inference passes（多次前向） | ❌ 无 | **UNKNOWN** |
| 7 | ensemble | ✅ 有（明确禁止） | **FORBIDDEN** |
| 8 | voting | ✅ 有（明确禁止「投票法」） | **FORBIDDEN** |
| 9 | NMS | ❌ 无 | **UNKNOWN** |
| 10 | confidence threshold | ❌ 无（仅要求写出值 ∈[0,1]） | **UNKNOWN** |
| 11 | IoU threshold | ❌ 无 | **UNKNOWN** |
| 12 | post-processing | ⚠️ 部分：输出格式与合法性有强制要求 | **UNKNOWN**（除格式约束为强制） |
| 13 | box refinement | ❌ 无 | **UNKNOWN** |
| 14 | single-model requirement | ⚠️ 间接：禁止**多模型集成**；未出现「仅限单模型」的正面表述 | **UNKNOWN**（但集成/投票 = FORBIDDEN） |
| 15 | single-inference requirement | ❌ 无 | **UNKNOWN** |

> **结论**：原文只**禁止**集成/投票/在线服务/人工干预/外部数据；
> 对 TTA、多尺度、NMS、阈值、box refinement **完全沉默** → 一律 **UNKNOWN**。
> **按指令，不把 UNKNOWN 当作允许。**
>
> 注：`reports/p0_rule_and_candidate_audit.md` 当时把「低 conf」「推理参数调整」记为「未禁止」；
> 本审计采用更严格标准记为 **UNKNOWN**，两者不矛盾，仅判定尺度不同。

---

## 2. Checkpoint 检查（§2）

目录 `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/`：

| filename | embedded epoch | optimizer | ema | 可加载 | 已有官方评测 |
|---|---:|---|---|---|---|
| `best.pt` | −1（已 strip） | 无 | 无 | ✅ | ✅ **0.50928** |
| `last.pt` | −1（已 strip） | 无 | 无 | ✅ | ✅ 0.50290 |
| `epoch200.pt` | 200 | 有 | 有 | ✅ | ✅ 0.49714 |
| `epoch210.pt` | 210 | 有 | 有 | ✅ | ❌（未评，本次已终止） |
| `epoch220.pt` | 220 | 有 | 有 | ✅ | ❌（未评） |
| `epoch230.pt` | 230 | 有 | 有 | ✅ | ✅ 0.50369 |
| `epoch240.pt` | 240 | 有 | 有 | ✅ | ✅ 0.50097 |
| `epoch250.pt` | 250 | 有 | 有 | ✅ | ✅ 0.49858 |
| `epoch260.pt` | 260 | 有 | 有 | ✅ | ✅ 0.50146 |
| `epoch270.pt` | 270 | 有 | 有 | ✅ | ✅ 0.49978 |
| `epoch280.pt` | 280 | 有 | 有 | ✅ | ✅ 0.50396 |
| `epoch290.pt` | 290 | 有 | 有 | ✅ | ✅ 0.50742 |

**`epoch 230–240` 附近的细粒度 checkpoint：不存在。**

```text
NO_FINE_GRAINED_CHECKPOINTS
```

依据：`args.yaml` 的 `save_period: 10` → 只落盘 10 的倍数（200/210/…/290）。
**`best.pt` 本身即 epoch 237**，231–239 之间无任何 checkpoint。**未为生成它们做任何训练。**

---

## 3. 已有 prediction / evaluation 产物（§3，全部直接复用，未重新预测）

| 目录 | 内容 | 数量 |
|---|---|---|
| `diagnostic/rgbid_inference_gain/RGBID_baseline/results` | 官方 baseline 预测（**0.50928 的来源**） | 400 TXT |
| `…/RGBID_tta_hflip/results` | TTA 水平翻转 | 400 TXT |
| `…/RGBID_tta_multiscale/results` | TTA 多尺度 | 400 TXT |
| `…/RGBID_f4_ens_{1to1,2to1,3to1,3to1_legal}/results` | 跨模型 Ensemble（**FORBIDDEN**） | 各 400 TXT |
| `diagnostic/rgbid_conf_0001_val/results` | conf=0.0001 | 400 TXT |
| `diagnostic/rgbid_1536_infer_val/results` | imgsz=1536 纯推理 | 400 TXT |
| `diagnostic/scaleup_false_fixed/results` | scaleup=False 变体 | 400 TXT |
| `diagnostic/rect_pipeline_fix/validation_{rect,square}_predictions` | rect vs square A/B（F4 模型） | 各 400 TXT |
| `diagnostic/_cap100_{baseline,a1536,bconf}/` | 100 框截断对照 | 各 1000 TXT |
| `diagnostic/checkpoint_selection_audit/ep{200,230,240,250,260,270,280,290}/results` | 8 个 checkpoint 预测 | 各 400 TXT |
| `…/_cap*`、`conf_scan.txt`、`summary_*.json` | conf 扫描与汇总指标 | — |

**已存在的参数变体：`conf`（0.0001 / 0.001 / 0.005 / 0.02 / 0.05）、`imgsz`（1280 / 1536）、`scaleup`（true/false）、`mode`（rect/square）、`max_boxes`（100/300）。**

---

## 4. 推理参数机会（§4）

### 4.1 `conf`（当前 0.001）

已有 `conf_scan.txt`（只读离线扫描，复用同一份 baseline 预测）：

| min_conf | 预测框数 | **official mAP50-95** | official mAP50 |
|---:|---:|---:|---:|
| 0.000（= 当前输出） | 4824 | **0.50928** | 0.77601 |
| 0.005 | 3694 | 0.50450 | 0.76576 |
| 0.020 | 3180 | 0.49972 | 0.75413 |
| 0.050 | 2934 | 0.49570 | 0.74645 |

**提高 conf 单调降低 official mAP50-95** → 不是机会。
降低 conf 已有实测：`conf=0.0001` 官方 0.51416 / **线上 0.50451（−2.729）→ 已淘汰**。
→ **`conf` 无可用空间。未新增任何实验。**

### 4.2 NMS `iou`（当前 0.7）

```text
NOT TESTED
```
仓库内**不存在**任何 IoU 阈值扫描产物。未重新预测。

### 4.3 `max_det`（当前 300）

baseline 官方预测的每图框数分布（实测）：

| 统计量 | 值 |
|---|---:|
| 图片数 / 总框数 | 400 / 4898 |
| 均值 / 中位 | 12.2 / 7 |
| **最大** | **177** |
| 框数 > 100 的图 | **4 张（1.0%）** |
| 框数 ≥ 300 的图 | **0 张** |

**`max_det=300` 从未被触顶 → 不是瓶颈。**
官方规则上限 100 框/图只影响 **4 张图（1.0%）**，且该截断为**强制要求**。
→ **`max_det` 无可用空间。**

---

## 5. Checkpoint 机会（§5）

8 个后期 checkpoint 的**官方口径**（`predict_rect --mode rect` + `official_map`，与 baseline 完全同参）：

| checkpoint | epoch | 内部 mAP50-95 | **official mAP50-95** | Δ vs 0.50928 |
|---|---:|---:|---:|---:|
| **`best.pt`** | **237** | **0.55800** | **0.50928** | — |
| `epoch200.pt` | 200 | 0.53306 | 0.49714 | −0.01214 |
| `epoch230.pt` | 230 | 0.55080 | 0.50369 | −0.00559 |
| `epoch240.pt` | 240 | 0.54524 | 0.50097 | −0.00831 |
| `epoch250.pt` | 250 | 0.54567 | 0.49858 | −0.01070 |
| `epoch260.pt` | 260 | 0.55529 | 0.50146 | −0.00782 |
| `epoch270.pt` | 270 | 0.54491 | 0.49978 | −0.00950 |
| `epoch280.pt` | 280 | 0.55541 | 0.50396 | −0.00532 |
| `epoch290.pt` | 290 | 0.55503 | 0.50742 | −0.00186 |
| `epoch210.pt` / `epoch220.pt` | 210 / 220 | 0.53921 / 0.54820 | 未评 | 预测 ≈0.490 / 0.499 |

```text
NO_CHECKPOINT_OPPORTUNITY
```

**依据**：
1. **已评的 8 个全部低于 0.50928**，最高为 ep290 的 0.50742（仍 −0.00186）。
2. 未评的 `ep210/ep220` 的内部值（0.53921 / 0.54820）**远低于** `best.pt`(ep237) 的 0.55800，
   而实测的「内部→官方」偏移稳定在 **−0.0492 ± 0.0033**（5 个已配对的点，符号恒为负）
   → 外推约 **0.490 / 0.499**，同样低于 baseline。
3. `best.pt` 已经是 **300 个 epoch 内部 mAP50-95 的最大值**（0.55800），
   且内部/官方两口径在「取最大值」这一粗粒度上一致（Spearman +0.70）。

---

## 6. TTA（§6）

**规则判定：UNKNOWN**（原文对 TTA/多尺度/hflip 无任何条款；集成禁令针对「多个不同的**模型**」，
TTA 用同一模型多次前向，但原文亦无允许条款 → 按指令记 UNKNOWN，**不解释为允许**）。

**已有结果**（官方口径，全部来自既有产物，未运行任何推理）：

| 方法 | 预测框数 | official mAP50 | official mAP75 | **official mAP50-95** | Δ vs baseline |
|---|---:|---:|---:|---:|---:|
| **RGBID baseline** | 4824 | 0.77601 | 0.56386 | **0.50928** | — |
| TTA hflip | 6176 | 0.77725 | 0.52034 | **0.50291** | **−0.00637** ❌ |
| **TTA multi-scale** | 7835 | 0.77092 | 0.55669 | **0.51360** | **+0.00432** ⚠️ |
| F4 ensemble 3:1（**FORBIDDEN**） | 7031 | 0.78235 | 0.56885 | 0.51496 | +0.00568 ❌规则 |

- **TTA hflip：负，淘汰。**
- **单尺度 1536：已实测 official 0.50398 → `REJECTED`**（与 §6 的前提一致）。
- **TTA multi-scale 本地 +0.00432** —— 但见 §8，它**对小目标毫无增益**，且引入更多退化框。
  且量级（+0.0043）与 **conf=0.0001 的本地增益（+0.0049）** 几乎相同，
  而后者**线上实测 −2.729**。**同量级的本地增益已被证明不迁移。**

→ **TTA 不构成安全机会。**

---

## 7. Prediction post-processing（§7）

| 项 | 规则 | 已有证据 | 判定 |
|---|---|---|---|
| class-specific confidence | UNKNOWN | NOT TESTED | ❌ 无证据 |
| class-specific NMS | UNKNOWN | NOT TESTED | ❌ 无证据 |
| box refinement | UNKNOWN | 见下 | ❌ 无证据且机制不成立 |
| box voting / WBF | UNKNOWN（若聚合多模型则 **FORBIDDEN**） | NOT TESTED | ❌ 不得以换名方式推荐 |
| soft-NMS | UNKNOWN | NOT TESTED | ❌ 无证据 |
| 多模型 ensemble / 投票 / 平均 | **FORBIDDEN** | 已测 +0.00568（本地） | ❌ **规则明令禁止，不因本地增益而重提** |

**关于 box refinement**：既有瓶颈审计已证明 —— 2288 个 TP 的
`scale = sqrt(pred_area/gt_area)` 均值 **0.9971 ± 0.0022**（**无显著偏置**），
偏大 46.2% / 偏小 53.8%（**对称**），中心偏移 ≈ 0。
**误差是零均值随机抖动，不存在可供后处理修正的系统性偏置。**
→ 任何全局 box 缩放/回归式精修都会**引入**偏置而非消除误差。

**未将任何 FORBIDDEN 项换名重提。**

---

## 8. 最关键：small-object 定位是否存在「无训练」的合法优化空间（§8）

### 8.1 最直接的无训练杠杆 = 多尺度推理（把小目标放大后再检测）——**实测无效**

对既有预测做尺寸分桶 AP（COCO area range，全部复用已有 TXT，**未重新推理**）：

| 方法 | 预测框数 | 退化框(w/h≤0) | 含退化框文件 | **small AP50-95** | large AP50-95 |
|---|---:|---:|---:|---:|---:|
| **baseline** | 4824 | 65 | 30（7.5%） | **0.05625** | 0.61711 |
| **TTA multi-scale** | 7835 | **188** | **49（12.3%）** | **0.05600** | **0.63334** |
| TTA hflip | 6176 | 129 | 50（12.5%） | 0.05373 | 0.61778 |

**TTA multi-scale 的 small AP50-95 = 0.05600 vs baseline 0.05625 —— 增益为 −0.00025，实测为零。**
它那 +0.00432 的整体收益**全部来自大目标**（large 0.63334 vs 0.61711，+0.0162）。

同时，**退化框 65 → 188（2.9×）、含退化框文件 7.5% → 12.3%** ——
这正是 `conf=0.0001` 方案（退化框覆盖 61.7%）**线上掉 2.729 分**的同款风险形态。

### 8.2 其他无训练路径

| 路径 | 判定 | 依据 |
|---|---|---|
| 提高输入分辨率（推理） | ❌ | 1536 实测 **−0.00530**（训练/推理分辨率失配） |
| 降低 conf 保留更多小目标框 | ❌ | 本地 +0.0049 → **线上 −2.729**，已淘汰 |
| 后处理放大/收缩小框 | ❌ | 误差**零均值**（§7），无偏置可修 |
| 多 checkpoint 权重平均（SWA 式） | ❌ | 属「不同训练阶段的模型」聚合 → 落入 **FORBIDDEN** 措辞；且规则为 UNKNOWN |
| 多 checkpoint 输出融合 / 投票 | ❌ | **FORBIDDEN** |
| 类内重排 / 分数重标定 | ❌ | NOT TESTED，无任何证据 |

```text
NO_SAFE_POST_TRAINING_OPTIMIZATION
```

**根本原因（承接瓶颈审计）**：small-object 定位误差是**零均值随机抖动**
（scale std 0.1286 vs large 0.0854，中位 IoU 0.7375 vs 0.8940），
其物理来源是**特征图空间分辨率**（P3 stride 8）。**分辨率只能通过训练侧（输入尺寸或 P2 层）改变**，
而 §10 的硬约束（算力增幅小、不改 backbone）与 §禁止清单（1536 训练、P2）已把两条路都封死。
**因此不存在无训练的合法路径。**

---

## 9. 机会清单（§9，最多 3 个）

| Opportunity | 需要训练 | 规则 | 已有证据 | 风险 | 是否值得 |
|---|---|---|---|---|---|
| TTA multi-scale（单模型多次前向） | 否 | **UNKNOWN** | 本地 official **+0.00432**；但 **small AP 增益实测为 0（−0.00025）**，退化框 65→188 | **HIGH** —— 同量级本地增益（conf=0.0001 的 +0.0049）已线上实证 **−2.729**；退化框覆盖率翻倍 | ❌ **否** |
| 剩余 checkpoint 官方评估（ep210 / ep220） | 否 | ALLOWED（本地评测不涉提交） | 8/10 已评，**全部 < 0.50928**；未评两者内部值 0.539/0.548，按偏移 −0.0492 外推 ≈0.490/0.499 | LOW | ❌ **否**（无超越可能） |

> 表中仅列**有实测数据支撑**的两项，且两项结论均为「不值得」。**未为凑数添加任何候选。**

---

## 10. 最终决策（§10）

```text
FINAL_OPTIMIZATION_OPPORTUNITY_AUDIT

Training launched: NO
Baseline modified: NO
Checkpoint modified: NO
Dataset modified: NO
Submission modified: NO

Current online baseline:
0.53180

Current official baseline:
mAP50=0.77601
mAP75=0.56386
mAP50-95=0.50928

Fine-grained checkpoint opportunity:
NO      (save_period=10 -> 只有 200/210/.../290；231-239 不存在；best.pt 即 ep237)

Inference opportunity:
NO      (conf: 提高单调变差，降低已线上淘汰；iou: NOT TESTED 无证据；
         max_det=300 从未触顶，最大单图 177 框)

TTA opportunity:
UNKNOWN (原文对 TTA/多尺度/hflip 无任何条款；不解释为允许)
        实测：hflip -0.00637 淘汰；multi-scale +0.00432 但 small AP 增益为 0、退化框 2.9x

Post-processing opportunity:
UNKNOWN (原文对 NMS/阈值/box refinement 无条款；class-specific conf/NMS、
         WBF、soft-NMS 全部 NOT TESTED)
        ensemble / voting / 平均法 = FORBIDDEN（原文明确）

Safe opportunities:
0

Final decision:
NO_SAFE_OPPORTUNITY

Recommended action:
  冻结正式方案，停止一切训练与推理侧改动。
  正式保底：RGBID Early Fusion，online mAP50-95 = 0.53180
  （offline official mAP50=0.77601 / mAP75=0.56386 / mAP50-95=0.50928）。
  剩余时间用于提交物完整性检查（代码 / 权重 / 预测结果 / 技术报告 PDF），
  并确保提交物与本报告冻结的 checkpoint 哈希一致。

Files changed:
  - reports/rgbid_final_optimization_opportunity_audit.md   （本报告，新增）
  - diagnostic/final_bottleneck_audit/                       （上一轮遗留的只读分析脚本与中间结果，本轮未新增）
  本轮未新增任何诊断/临时文件；仅新增本报告。
```

### 10.1 必须同时披露的两项操作事实

| # | 事实 | 影响 |
|---|---|---|
| 1 | 上一轮的后台评估链**未被真正终止**（`TaskStop` 只杀包装进程，bash 子进程存活并继续 `for` 循环），在用户指示「只做 240/260/280/290」之后仍继续评估了 **200 / 230 / 250 / 270**（4 组完整结果），随后又启动了 **ep210 / ep220** | 超出授权的计算。4 组完整结果有效且已在本报告 §5 使用；ep210/ep220 被本轮中途终止，产出**残缺**（ep220 仅 240/400），已**全部删除**（含基于残缺结果生成的 `eval_ep210.json` / `_ep210.eval.txt`，该评测无效）。进程树（python 32052 → bash 19584 → bash 34916）已按 子→父 顺序彻底终止，复查无残留 python，`ep220/results` 计数停止增长 |
| 2 | `ep240` / `ep260` 曾被两条链**并发重复推理**，写入同一目录 | 两次独立运行得到**完全相同的官方值**（0.50097 / 0.50146）→ 数据一致性得到验证，无污染 |

---

## 11. 合规状态

| 项 | 状态 |
|---|---|
| 训练 / 训练进程 | ❌ 未启动；且**终止了一个此前遗留的只读评估进程** |
| 新建 / 修改 model 或 train YAML | ❌ 无 |
| 修改源码 / dataset / best.pt / submission / 正式 runs | ❌ 无 |
| 重新生成预测 | ❌ 无（全部复用既有 TXT） |
| 本轮新增文件 | 仅 `reports/rgbid_final_optimization_opportunity_audit.md` |
| 正式线上保底 | ✅ **RGBID Early Fusion，0.53180 不变** |
