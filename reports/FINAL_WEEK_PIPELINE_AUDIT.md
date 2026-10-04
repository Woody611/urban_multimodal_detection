# FINAL WEEK — 全项目实验链路与线上分数差距总审计（READ-ONLY）

**日期**：2026-09-30 · **性质**：纯只读。0 训练 / 0 forward / 0 backward / 0 修改已有文件 / 0 生成 submission。
固定 baseline：**D′ = YOLO11m RGBID 5ch + SepStem + IR-CLAHE @1280**，`best.pt` sha256 `1cae45f7…fae4fda`。

> 本报告为**新增**文件，未修改仓库内任何既有文件。所有只读验证脚本写在系统 temp 目录。

---

## 0. 三条最重要的结论

1. **没有发现 RED 级 pipeline bug。** D′ 的 train/val/predict/submission 四路预处理在 CLAHE、通道序、归一化、几何上**一致**；checkpoint 归属正确；评测器与官方规则一致。
2. **真正的瓶颈是"量尺分辨率"，不是链路正确性。** 同一次训练内部、仅换 checkpoint（无任何实验变量），官方口径散布 sd≈**0.0030–0.0042**、极差 **0.0104–0.0151**。而项目半个月来裁决的每一个 Δ 都落在 **0.002–0.008**。**所有"实验无效"的结论，其 Δ 都在噪声带内。**
3. **`online ≈ fork × 85.9` 是对的，但只对"同架构族 + 训练内 val"成立**（sepstem 族 3 点：85.85 / 85.92 / 85.93，离散 0.09%）。跨架构立刻失效（D = 84.47）。这条规则**没有被证伪**，反而被本轮数据加强。

---

## A. Pipeline health

```text
YELLOW
```

无致命缺陷，但存在 **3 个潜伏陷阱 + 2 个可追溯性缺口 + 1 个命名歧义**（见 §E）。

---

## B. Baseline D′ health

```text
TRUSTED
```

| 检查项 | 结果 | 证据 |
|---|---|---|
| `config_diff` | **EMPTY** | `args.yaml` 36 个键全部由 config 或 ultralytics 默认解释 |
| `ir_encoding` | **`clahe`（三处一致）** | `configs/train_rgbid_sepstem_clahe.yaml:61` / `args.yaml` / `ckpt.train_args` |
| checkpoint | `best.pt` = **ep246**，fitness 0.59270 | sha256 `1cae45f75693f541…fae4fda`，与 V11/V13/E1 契约记录一致 |
| `last.pt` | **不同文件**（ep300, 0.55522） | sha256 `b8398b87…`；`cmp` 首异字节 31 |
| `ch / nc / params` | 5 / 12 / 20,061,972 | embedded `model.yaml`；SepStem 结构（3× `SilenceChannel`） |
| 推理用的 checkpoint | **best.pt**（非 last.pt） | 数值佐证：报告 0.56690 贴近 best.pt 0.5669，远离 last.pt 0.55522 |
| 可复现性 | **否（代码已漂移）** | `predict_rect.py` `c55f7744…`→`abc22e8a…`；`default.yaml` `6f91a277…`→`04f0d20a…` |

**唯一缺口**：`diagnostic/sepstem_clahe/best_full/results/`（400 TXT）**没有推理日志**。归属靠文献链（`reports/RGBID_SEPSTEM_CLAHE_RESULT.md:129,142`、`diagnostic/depth_reliability_small_miss/REPORT.md:266-268`），不是自证。对照：D 有 `diagnostic/ir_clahe/_infer_best_full.log`。

---

## C. Historical experiment reliability

| Experiment | Intended change | Runtime active? | Checkpoint correct? | Evaluator | Trustworthy | A–G |
|---|---|---|---|---|---|---|
| **D′** | SepStem + IR-CLAHE | YES | YES | official cap=100 → 0.51528；线上 48.712 | **TRUSTED** | **A** |
| **D** (IR-CLAHE) | percentile→CLAHE | YES | YES | 0.50996（线上 47.985） | **TRUSTED** | **A** |
| **SepStem percentile** | CLAHE（对照） | **NO** | **NO** | — | **UNTRUSTED** | **C** |
| **E1** | RegionResponseGain@[13,15,17] | YES（20,077,332 params，`e1_enabled=True` 在 best 与 resumed last 均在） | YES | official −0.00540，**CI [−0.0193,+0.0054] 含 0** | PARTIALLY TRUSTED | **B** |
| **F1** | native-small replay | YES（代码/配置级；**无运行期证人**） | YES | official +0.00337，fork **−0.00205（反号）**，CI 含 0 | PARTIALLY TRUSTED | **B** |
| **OASA 1.4 / 2.0** | 小目标 scale aug 1.4×/2.0× | YES（1.4 有日志；2.0 只有 1-ep smoke） | YES | 线上 48.134 / 47.610 < D′ 48.712 | **TRUSTED** | **A** |
| **P2** | 加 P2 头 | YES（`nl=4`, stride [4,8,16,32], 20,804,608） | YES | 0.49674 | **TRUSTED** | **A** |
| **P3 attention** | layer-16 identity attention + freeze | YES 但**协议偏离**（`freeze()` 不冻 BN，199/212 BN 漂移，confound 主导 Δ） | YES | 0.50071 / 0.51084 | PARTIALLY TRUSTED | **D** |
| **1536 finetune** | imgsz 1536 | YES | YES，但 **best = ep1/ep2 of 25** | baseline 复用 1280 历史记录（**不同 operating point**） | PARTIALLY TRUSTED | **E** |
| **modality dropout** | 整模态 dropout 7.5%/7.5% | YES（逐像素证人：IR≡125 / Depth≡0） | YES | 0.49022 | **TRUSTED** | **A** |
| **TTA** (hflip/multiscale) | 推理期 TTA | YES（预测数 4824→6176/7835） | N/A | hflip 两口径同负；multiscale fork −0.01445 | **TRUSTED** | **A** |

**分类统计**：A×7、B×2、C×1、D×1、E×1。

### 唯一的确认级"变量未生效"
`runs/urban_multimodal_det_yolo11_rgbid_sepstem_percentile/`：`args.yaml` **无 `ir_encoding` 键**，ckpt `train_args` 亦无 → `ultralytics/data/base.py:333` 静默回落 `percentile`。该目录 `name/save_dir` 仍写着 `..._sepstem_clahe`（命名污染）。**已在 `reports/RGBID_SEPSTEM_CLAHE_RUN_DIAGNOSIS.md` 记录并从 D vs D′ 比较中剔除。**
➡ 这是历史事件，**不影响当前 incumbent**。

---

## D. Online gap diagnosis（完整串联）

### D.1 四层数字的精确身份

| 数字 | 身份 | 口径 | cap |
|---|---|---|---|
| **0.56690** | D′ **训练内 val**（`model.val()`，rect=**False** 方形，FP16） | fork（trapz+ramp+IoU匹配，metric **B**） | max_det 300 |
| **0.56596 / 0.56609** | D′ **fork-on-predictions**（`official_map.py` fork 行，同一 400 TXT） | 同上 | 无 / **100** |
| **0.51528** | D′ **官方口径**（`official_eval.py`，同一 400 TXT） | 官方（mean+tail0+conf匹配，metric **A**） | **100** |
| **48.712** | 榜单（新测试集，1000 图） | 官方定义（见 D.3） | 100 |

> ⚠️ **纠正常见误传**：`0.56806 @ep263` 是 **D**（early fusion）的训练内 val，**不是 D′**。D′ = `0.56690 @ep246`。二者在 `reports/RGBID_SEPSTEM_CLAHE_RESULT.md:33` 并排。

### D.2 fork ↔ official 的 0.0508 差距 —— 逐项实测，完全解释

同一批 400 TXT、同一 GT、同 10 IoU 阈值、同 101 点、同贪心一对一匹配语义：

| # | 差异 | 官方 | fork | 实测 Δ |
|---|---|---|---|---|
| 1 | AP 尾部 | `zero`（无 recall≥r → 0） | `ramp`（哨兵斜坡到 (1,0)） | **+0.05170**（98.3%） |
| 2 | 积分 | 101 点**算术平均** | `np.trapz` | +0.00093（1.8%） |
| 3 | 匹配顺序 | 置信度降序 | IoU 降序 | −0.00182（−3.6%） |
| 4 | 类分母 | A（固定 12） | B（仅有 GT 的类） | **0.00000**（本 val 12 类皆有 GT） |
| 5 | 100 框上限 | 100 | 100 | 同 regime 内 0 |
| | **合计** | | | **= 0.05081 ✓ 与实测差完全相同** |

### D.3 官方 PDF 已找回 —— 官方口径 = 榜单口径

规则 PDF 在 **git 历史**中（HEAD `5f1a8d5` 删除，父提交 `3b4f10a` 存在），**可机读**（sha256 `37d20480…`，与项目记忆一致）：

- 12 类 id 0–11；IoU 阈值 10 个 0.50–0.95
- 匹配：**按置信度降序**，贪心一对一
- AP：101 点，`p_interp(r)=max{precision: recall≥r}`，**算术平均**
- mAP(t)：**「其中 N 是类别总数」→ 固定 12 类分母（METRIC_A）** ⟹ **项目悬置项 D2 依原文判定为 A**
- **每图最多 100 框，按置信度截断**；非法类别/坐标/缺 conf → 该预测无效

➡ `scripts/official_eval.py` 与官方规则**逐条一致**。**ultralytics fork 才是异类**（trapz + ramp 均非 COCO 亦非官方）。

### D.4 `online ≈ fork × 85.9` 的适用边界（本轮实测）

| run | 训练内 val | 线上（新集） | ratio |
|---|---:|---:|---:|
| **D′** sepstem_clahe | 0.56690 | 48.712 | **85.93** |
| **OASA 1.4** | 0.56070 | 48.134 | **85.85** |
| **OASA 2.0** | 0.55409 | 47.610 | **85.92** |
| D ir_clahe（**跨架构**） | 0.56806 | 47.985 | **84.47** ← 失效 |

**sepstem 族内 3 点离散仅 0.09%** ⟹ 项目既有裁定「同架构族内 fork 排序 = 线上、线上 ≈ fork × 85.9」**被本轮数据加强**。
**跨架构失效**（D 偏 1.7%，且**排序反转**：内部 val 判 D(0.56806) > D′(0.56690)，线上判 D′ > D）。

**排序正确性对照**：

| 对比 | fork/内部 val 排序 | official 排序 | 线上真实排序 | 谁对 |
|---|---|---|---|---|
| D′ vs OASA1.4 vs OASA2.0（**同族**） | 0.56690 > 0.56070 > 0.55409 ✓ | 0.51528 > **0.51122 > 0.50739** ✗ | 48.712 > 48.134 > 47.610 | **fork 全对，official 反转 1.4/2.0** |
| D′ vs D（**跨架构**） | 0.56690 < 0.56806 ✗ | 0.51528 > 0.50996 ✓ | 48.712 > 47.985 | **official 对，fork 反转** |

➡ **两把尺子各在一半的场景里正确**，而其适用边界恰好就是项目已经写下的那条规则。

### D.5 分辨率的生死线（本审计最重要的量化）

同一 run、同一 val、**只换 checkpoint、无任何实验变量**：

| 来源 | 指标 | sd | 极差 |
|---|---|---:|---:|
| baseline `rgbird_ir_quicktest` ep200–290（官方口径，cap=100） | 0.00330 | **0.01035** |
| 同 run 训练内 val 末 50 epoch | 0.00422 | 0.01506 |
| D′ 训练内 val 末 50 epoch | 0.00298 | — |

换算到线上尺度（×85.9）：**1 sd ≈ 0.26–0.36 分；极差 ≈ 0.89–1.29 分。**

而当前目标是从 48.712 提到 **≥50**，即需要 **+1.29 线上分 ≈ +0.015 训练内 val** —— **恰好等于仪器自身的极差**。

> **结论：要在最后一周靠本地实验"确证"一个足以达标的改动，在统计上要求该改动的真实效应 ≥ 仪器极差。**
> 所有已裁决实验的 Δ（0.002–0.008）都在极差之内 —— 它们的"无效"结论**不是方法无效的证据，而是仪器分辨不出的证据**。

**附带偏差**：`best.pt` = 300 个 epoch 在同一 val 上取 argmax ⟹ headline 数字含**选择偏差**（胜者诅咒）。

### D.6 残差归因（本地 → 线上）

| 环节 | 量 | 性质 |
|---|---:|---|
| 训练内 val 0.56690 → fork-on-dump 0.56596 | −0.00094 | 方形 vs 矩形几何 + FP16/FP32（可忽略） |
| fork → official（同批预测） | **−0.0508** | **纯记账口径**，已逐项完全解释 |
| official 0.51528 → 线上 48.712 | — | **不同集合**（400 val vs 1000 test），不可直接相减 |

**残差的真实来源（有证据）**：
- **构成偏移（composition shift）**：族 `N` 占 test 73.0% 但 train 52.6%；`shuming_*` 占 test 5.4% 但 train 18.5%。**且 split_val 自身即对 test 有偏**（shuming 20.25% vs 5.4%）⟹ val 上选出的任何阈值都不可迁移。
- **帧级近重复**：test 中 14.5% 在 train 有 >0.99 邻居（族 N 的 LR 子集高达 96.6%，HR 子集仅 0.3%）；val 为 6.8%。
- **分层内无分布偏移**（visible |d|≤0.077、depth ≤0.118、IR ≤0.258），**padding/宽高比偏移恒为 0**（全部 16:9 ⟹ 画布恒 1280×736）。

⚠️ **"复赛换集导致整体降 5–7 分"是推断，不是实测**：仓库内没有该结论的任何测量支撑（旧集/新集不可相减，且两集难度无 GT 可比）。**标记为 HYPOTHESIS。**

---

## E. Confirmed bugs / defects（仅列有代码或 artifact 证据者）

> **无 RED 级缺陷。** 以下均为 YELLOW/P2。

| ID | 缺陷 | 位置 | 影响 | 严重度 |
|---|---|---|---|---|
| **E1** | 流经冻结链的 val dump 用 `max_boxes=300` 落盘（最大 184 行/图），而生产 submission 用 `max_boxes=100`；100 框规则**只在 evaluator 侧**强制 | `diagnostic/*/best_full/_infer_best_full.log:2`；`submissions/*_infer.log:2` | **实测 cap 影响 = 0.00129**（仅 4/400 图被截断，丢 6 个 TP@IoU.5 = 0.21% GT）⟹ 不构成评测错误 | P2 |
| **E2** | `scripts/predict.py` 构造 `LoadImagesAndVideos` 时**不传 `ir_encoding`** → 静默 `percentile`；且为方形 1280² + `scale_boxes` 少传 `ratio_pad` | `predict.py:366-373`、`:356-357`、`:399` | **当前为潜伏陷阱**：其默认指向 F4 RGBD 4ch 模型，不在 RGBID 链上；但若有人传 CLAHE 配置给它，即重现 2026-09-20 静默回落事故 | **P1（治理）** |
| **E3** | `ultralytics/data/*` 被 `.gitignore:11` 吞掉 ⟹ 训练/推理预处理代码**无哈希可校验**；`FREEZE_MANIFEST.sha256`（2026-09-17）不含 `base.py / loaders.py / dataset.py` | `.gitignore`、`reports/FREEZE_MANIFEST.sha256` | D′ 的预处理路径**无法从 manifest 自证**（当前以文件哈希 + 逐行同构替代） | **P1（治理）** |
| **E4** | IR 预处理有 **3 份拷贝**；`dataset.py:563-684` 是**死代码**（属 `ClassificationDataset`，只支持 percentile，且会因缺 `self.use_simotm` 崩溃） | `base.py:313-350` / `loaders.py:643-679` / `dataset.py:563-684` | 当前两条活路径**逐行同构**（已 diff 确认）；死代码是**未来陷阱** | P2 |
| **E5** | `official_eval.py:75-76` docstring 声称越界坐标"会被裁剪（见 `normalize_preds`）"，但 **`normalize_preds` 不存在**，`norm_xywh_to_xyxy` **不裁剪** | `official_eval.py:75-76,135-139` | 评估器不裁剪 ⟹ 若生产端停止裁剪，坐标会静默偏移 | P3 |
| **E6** | **"fork" 一词在本项目指两个不同的计算**：(a) `model.val()` 训练内 val；(b) `official_map.py` fork 行（对 TXT 重算）。二者在 D′ 上差 0.0001–0.0009，但来自不同代码路径 | `metrics.py:1204-1214` vs `official_map.py:116-128` | **记账歧义**：任何"fork Δ"必须先声明是哪一个 | **P1（记账）** |
| **E7** | 默认 split root 在三个入口不一致（`predict_rect.py:170` = `depth_split_train`；`official_eval.py:326` = `rgbid_split`；`official_map.py:279` = `depth_split_train`） | 同上 | 当前三者 stem/GT 完全相同（抽查 25 文件 0 差异）⟹ **潜伏** | P3 |
| **E8** | 4 行 GT 标签越界（3 个文件：`train/003107`、`val/000050`×2、`val/003817`）→ `load_split(drop_corrupt=True)` **整图作废** | `official_eval.py:231-234` | 所谓"400 图 val"**实际是 398 图**；对两臂对称，不产生偏差 | 记录项 |
| **E9** | legacy P2 第一轮 run 目录被覆盖（`reports/rgbid_p2_probe.md` §8） | — | 不影响 P2 裁定 | 记录项 |
| **E10** | OASA 2.0 的 300-epoch run **是否用修复后的代码无从证实**（只有 1-ep smoke 日志，而 smoke 用的是**有缺陷的 input-space 判据**，eligible 38% vs 预注册 25.1%） | `reports/OASA_STEP4_A100_SMOKE.md`、`OASA_STEP4B_NATIVE_AREA_FIX.md` | 间接时序证据倾向"用了修复版"（修复 mtime 09-22 22:24 早于 2.0 run 起跑 ≈09-22 23:03）⟹ **UNKNOWN，非确认缺陷** | 记录项 |

### STOP-IF-FOUND 清单逐条核对

| # | 条件 | 结果 |
|---|---|---|
| 1 | D′ train/val/test 预处理不一致 | **未触发**（`predict.py` 不在链上） |
| 2 | CLAHE train/val/test 不一致 | **未触发**（四路皆 CLAHE，`predict.py` 例外且不在链上） |
| 3 | RGBID 通道序不一致 | **未触发**（`[B,G,R,IR,D]`→`[R,G,B,IR,D]` 四路一致） |
| 4 | checkpoint mismatch | **未触发**（`best.pt` sha 与记录一致） |
| 5 | 推理用错 checkpoint | **未触发**（= best.pt/ep246，数值佐证） |
| 6 | 类别映射错 | **未触发**（14,417 行标签 id ∈ [0,11]） |
| 7 | evaluator mismatch | **未触发**（官方评测器 == 规则 PDF 逐条） |
| 8 | submission 坐标错 | **未触发** |
| 9 | 文件名/图像序错位 | **未触发**（**按 stem 映射**，400/400 与 1000/1000 集合完全相等） |
| 10 | max_det/NMS/conf 不一致 | **部分触发（不致命）** → 见 E1（cap 跨阶段差，实测 0.0013）、E2（`predict.py` 方形+percentile 潜伏） |
| 11 | 实验变量未真正生效 | **触发 1 例** → `sepstem_percentile`（历史，已剔除；见 §C） |
| 12 | 增强配置了但没执行 | **触发（已知且无害）** → 5ch 下 `RandomHSV` 直接 return、alb 被显式 `p=0` |
| 13 | test 分布解释无证据却被当成事实 | **触发** → "复赛换集降 5–7 分"为推断（见 D.6） |

---

## F. Hypotheses（与 E 严格分开）

| ID | 假设 | 为何未证实 | 如何证实 |
|---|---|---|---|
| **H1** | 当前 48.712→50 的缺口，主要不是"方法不行"，而是**构成偏移 + val 选择偏差**使本地无法预筛 | val 与 test 族构成不同（shuming 20.25% vs 5.4%），但缺 test GT | 需要带 GT 的公开验证集或主办方反馈 |
| **H2** | 4 张超 100 框的图像（person 密集场景）是 cap 影响的全部来源；若 test 更密集，损失会放大 | 本 val 只损失 6 TP | 需 test GT |
| **H3** | `official_eval.py` 的 `best.pt` 选择偏差约 +0.008~0.012（300-epoch argmax） | 需固定 epoch 的对照 | 用固定 ep250 与 best 的官方差（本审计测得 ep250 0.49867 vs best 0.51528） |
| **H4** | `predict.py` 的 percentile+方形路径若被误用于 CLAHE 配置，会产生与 09-20 同级的事故 | 无实例 | 静态已足够；建议加断言 |
| **H5** | 新测试集的 5–7 分下移由"难度/口径变化"造成 | **纯推断**，仓库内无测量 | 无（本地不可验证） |

---

## G. Highest-information next experiment（至多 2 个）

> 依据 D.5：**任何 Δ < 0.010（≈0.9 线上分）在本 val 上都不可裁决。** 因此"下一步"必须选择**信息价值**而非"再训一个模型"。

### 候选 1（强烈推荐，**GPU 成本 = 0**）

- **Hypothesis**：F1（native-small replay）的本地两口径**反号**（official +0.00337 / fork −0.00205）是仪器在噪声带内的分裂；其真实效应只有线上能裁决。
- **Exact single variable**：`native_small_replay`（已训练完成，`best.pt` sha `9c6f524e…`，包已在 `submissions/rgbid_sepstem_clahe_f1_candidate/`）。
- **Expected effect**：若机制成立 +0.5~+1.5 线上分；若为噪声，48.7 ± 0.4。
- **Failure criterion**：线上 ≤ 48.712（不优于 D′）。
- **Stop criterion**：得分低于 D′ 即永久关闭该方向，不再回看。
- **Estimated GPU cost**：**0**（仅推理 1000 图，本机 CPU ≈26–30 min 或云端 GPU 数分钟）。
- **Why higher information than previous**：这是**唯一能把"两口径反号"这一仪器级矛盾落到实处的动作**；后续任何决策（是否信 fork、是否继续在小目标方向投入）都依赖它。且它是**已训练、已合规打包**的零成本资产。

### 候选 2（仅在候选 1 结果为正、且仍有 ≥4 天时）

- **Hypothesis**：当前所有 Δ 判定的分母（run 级方差）从未被测过；只用 checkpoint 级噪声会低估。
- **Exact single variable**：**seed 42 → 43**，其余逐字不变（D′ recipe）。
- **Expected effect**：**0（构造上）**。
- **Failure criterion**：不适用（这是标定实验，不是增益实验）。
- **Stop criterion**：得到 run 级 sd 后立即停止，不训练第三个 seed。
- **Estimated GPU cost**：1 × 300ep ≈ 11 h V100。
- **Why higher information**：它给出"Δ 必须多大才算真"的**唯一正确分母**。没有它，最后一周的任何训练都是在噪声上下注。

> **明确不建议**：新 attention / 新 loss / 新增强 / 新分辨率 / 新 P2-P5 / 新 depth 处理 / 新 TTA / 新 replay / 超参扫描。理由见 D.5 —— 这些改动的期望效应普遍 < 0.005，**低于仪器极差 2–3 倍**。

---

## H. FINAL DECISION

```text
3. PIPELINE HEALTHY — no further training justified
```

**理由**：

1. **Pipeline 健康**（B: TRUSTED，无 RED 级缺陷；四路预处理一致、checkpoint 归属正确、评测器与官方规则逐条一致）。
2. **但仪器分辨率为 ±0.9~1.3 线上分，而达标所需为 +1.29 线上分** —— **训练无法在本地被验证为有效**。在最后一周，一次 11 h 的训练若产生 +0.005 的真实效应，本地看到的将是噪声；若产生 −0.005，同样看到噪声。
3. **唯一零成本、高信息量的可用资产**是**已经训练并已合规打包的 F1 候选**（候选 1）。它不是新训练，是把一次已有的、本地无法裁决的实验交给线上裁决。
4. 若不做候选 1，则**应当接受 D′ 为最终方案并停止**——因为在现有仪器下，任何进一步训练都等价于**在噪声上下注**，其期望增益为负（含 GPU 与时间成本）。

**未采纳的选项 1（PIPELINE BUG — fix before training）**：不成立。E1/E2 类问题均为 P1/P2 治理项或潜伏陷阱，**没有一个会污染当前 D′ 的评测链**；且本轮明令不修改任何文件。

---

## 附：本审计新增的只读产物

- 系统 temp：`_calib.py`、`_floor.py`、`rules.pdf` / `rules.txt`（从 git 历史提取）及各 sub-audit 脚本
- 本仓库：**仅本文件**（`reports/FINAL_WEEK_PIPELINE_AUDIT.md`，新增，未修改任何既有文件）
