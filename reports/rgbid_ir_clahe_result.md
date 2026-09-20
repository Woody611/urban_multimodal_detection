# Experiment D（IR CLAHE）结果报告与裁决

**日期**：2026-09-19
**实验定义**：`RGBID@1280 baseline` vs `RGBID@1280 + IR CLAHE`（唯一变量 = IR 对比度处理）
**训练产出**：`runs/urban_multimodal_det_yolo11_rgbid_ir_clahe/`（300/300 epoch，无早停，无异常）
**判定**：# ⚠️ **`D_REJECTED` 已被线上实测推翻 → 改判 `D_WIN`**

> **勘误（2026-09-19 晚）**：本文档 §4 的 `D_REJECTED` 判定**错误**。
> 线上实测 **D = 0.5553** vs 基线 **0.53180** → **Δ_online = +0.02350**。
> 错因见 §4.6。**D 为当前最优已验证模型，线上 0.5553。**

---

## 0. 结论速览

| 项 | Baseline（线上 0.53180） | D（IR CLAHE） | Δ |
|---|---:|---:|---:|
| **线上（真实评分器）** | **0.53180** | **0.5553** | **+0.02350** ✅ |
| 本地官方 mAP50-95 | 0.50928 | 0.51076 | +0.00148 |
| 本地 fork mAP50-95 | 0.55415 | 0.56607 | **+0.01192** |
| 本地官方 mAP50 | 0.77601 | 0.77119 | −0.00482 |
| 本地官方 mAP75 | 0.56386 | 0.50161 | **−0.06225** |
| Precision | 0.8578 | 0.8561 | −0.0017 |
| Recall | 0.7029 | 0.7207 | +0.0178 |
| 预测框数(val) | 4824 | 4341 | **−483（−10.0%）** |
| 预测框数(test) | 7370 | 6388 | −982（−13.3%） |
| Small AP | 0.05625 | 0.05473 | −0.00152 |
| Medium AP | 0.17984 | 0.18737 | +0.00753 |
| Large AP | 0.61711 | 0.62155 | +0.00444 |

**提交包**：`submissions/rgbid_d_ir_clahe_candidate/submission.zip`
SHA256 `968810029125f7d2b4862e6521a7765b75fea76972e5dc55fd5f0d16a0bafe89`（270,626 B，1000 条目）

**配对图像级 bootstrap（400 iter, seed 42）—— 仅描述本地官方口径**：

| metric | point Δ | 2.5% | 97.5% | P(Δ≤0) | σ |
|---|---:|---:|---:|---:|---:|
| mAP50-95 | +0.00149 | −0.00653 | +0.00920 | 0.407 | 0.00417 |
| small | −0.00152 | −0.00621 | +0.00635 | 0.598 | 0.00332 |
| medium | +0.00753 | −0.00616 | +0.01469 | 0.215 | 0.00546 |
| large | +0.00444 | −0.00728 | +0.01471 | 0.258 | 0.00590 |

> ⚠️ **该 bootstrap 只证明「在本地官方口径下 Δ 不可区分于 0」，不证明「D 无增益」。**
> 它检验的是**错误的代理指标**。线上实测 +0.02350 说明 D 的增益真实存在。

---

## 1. ⚠️ 必须先记录的阻塞项：推理路径不应用 CLAHE（已修复）

### 1.1 问题

IR 对比度处理在仓库内有 **3 份独立拷贝**，Experiment D 只改了其中 1 份：

| 位置 | 承载 | 用途 | 改前是否支持 `ir_encoding` |
|---|---|---|---|
| `ultralytics/data/base.py:313` | `BaseDataset` | **训练 + `model.val()`** | ✅ 已改（D 报告的改动点） |
| `ultralytics/data/loaders.py:638` | `LoadImagesAndVideos` | **`predict_rect.py` / `predict.py`（官方评测 + 提交）** | ❌ 硬编码 percentile |
| `ultralytics/data/dataset.py:632` | `ClassificationDataset` | 未使用（潜在隐患，本次未改） | ❌ 硬编码 percentile |

**后果**：`rgbid_ir_clahe_audit.md` §12 预注册的评测命令会**用 percentile 预处理喂一个 CLAHE 训练的模型**——
静默的 train/infer 预处理错配。官方 mAP50-95 拿不到，**提交也不成立**（`predict.py` 走同一个 loader，
test 集同样会拿到 percentile IR）。

### 1.2 逐位实证（n=5，改前）

```
img              Base(clahe)==CLAHE  Base(pctl)==PCTL  Loader==CLAHE  Loader==PCTL
00000045.jpg                   True              True          False          True
...
  BaseDataset(ir_encoding=clahe) IR == CLAHE : 5/5
  BaseDataset(ir_encoding=pctl)  IR == PCTL  : 5/5   [baseline 可复现性 ✓]
  Loader (predict_rect.py)       IR == CLAHE : 0/5   <-- 关键
  Loader (predict_rect.py)       IR == PCTL  : 5/5   <-- 关键
  RGB+Depth 两路径逐位一致: True
```

**根因**：`predict_rect.py` 从 `--train_config` 只读 `use_simotm / pairs_rgb_ir / pairs_rgb_depth / channels`
四项，**不读 `ir_encoding`**。冒烟测试 `diagnostic/ir_clahe/_smoke_test.py` 只构造了 `BaseDataset`，
从未触碰 `LoadImagesAndVideos`——所以「全部 PASS」也没检出这个洞。

### 1.3 修复（本次实施）

| 文件 | 改动 | 改前 SHA256 | 改后 SHA256 |
|---|---|---|---|
| `ultralytics/data/loaders.py` | `LoadImagesAndVideos.__init__` 增 `ir_encoding="percentile"`；RGBID 分支镜像 base.py 的 if/else | （未跟踪/未记录） | `f84bdcf29081e6f95b58fbf19c255d1d…` |
| `scripts/predict_rect.py` | 从 `train_cfg` 读 `ir_encoding` 并透传；日志行打印 `ir_encoding=` | `656aa58789a467e0…` **（冻结项）** | `6baa7367da9376a94555428834fb33e4…` |

默认值 `percentile` 保证既有行为不变。

### 1.4 修复兼容性证明

| 检查 | 结果 |
|---|---|
| 默认路径（不传 `ir_encoding`）IR 像素 vs 改前实测 | ✅ 3/3 逐项一致（mean/std/uniq 全等） |
| 显式 `ir_encoding='percentile'` vs 默认 | ✅ 逐位一致 |
| `ir_encoding='clahe'` vs 参考 CLAHE 实现 | ✅ 逐位一致 |
| clahe 路径下 RGB/Depth | ✅ 逐位不变 |
| 冻结链（baseline 权重+配置）重跑 24 图 vs 冻结参考输出 | 22/24 逐位相同；2 文件仅**置信度末位 1 ULP**（`…360783`→`…360782`；`…851385`→`…851384`），框坐标完全一致 |
| 上述 2 文件的差异归因 | ✅ **批次组成效应**，非补丁：同一补丁代码连跑 3 次结果**互相逐位相同（24/24）**；改 `--batch 16→4` 后差异扩大到 7 处 1-ULP + 3 处其他 → 该管线对批次分组敏感。参考是 400 图全量跑（末批 16），子集是 24 图（末批 8），末批组成不同。 |

---

## 2. 0.53180 正式链的完整复现（对照基线口径）

| 环节 | 值 | 校验 |
|---|---|---|
| 权重 | `runs/.../rgbird_ir_quicktest/weights/best.pt` | SHA256 `f9dddbfa…d83e55` ✅ 匹配 `FREEZE_MANIFEST.sha256` |
| 脚本 | `scripts/predict_rect.py` | 改前 `656aa587…528be8` ✅ 匹配冻结清单 |
| 评测复现 | `official_map.py`（本次实跑） | `0.77601 / 0.56386 / 0.50928` ✅ 与记录逐位一致 |

**本地官方口径评测命令**（判据用）：

```bash
python scripts/predict_rect.py \
  --weights runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt \
  --train_config configs/train_rgbird_ir_quicktest.yaml \
  --source data/processed/rgbid_split/images/val/visible \
  --mode rect --imgsz 1280 --conf 0.001 --iou 0.7 --max_det 300 --max_boxes 300 \
  --batch 16 --output diagnostic/rgbid_inference_gain/RGBID_baseline

python scripts/official_map.py \
  --results diagnostic/rgbid_inference_gain/RGBID_baseline/results \
  --split_root data/processed/rgbid_split
```

**D 的对应命令（逐字同构，仅 weights / train_config / output 三处路径不同）**：

```bash
python scripts/predict_rect.py \
  --weights runs/urban_multimodal_det_yolo11_rgbid_ir_clahe/weights/best.pt \
  --train_config configs/train_rgbid_ir_clahe.yaml \
  --source data/processed/rgbid_split/images/val/visible \
  --mode rect --imgsz 1280 --conf 0.001 --iou 0.7 --max_det 300 --max_boxes 300 \
  --batch 16 --output diagnostic/ir_clahe/best_full

python scripts/official_map.py \
  --results diagnostic/ir_clahe/best_full/results \
  --split_root data/processed/rgbid_split
```

运行日志自证预处理已生效：

```
[predict_rect] use_simotm=RGBID channels=5 ir_encoding=clahe
[predict_rect] 完成 400 张图 → diagnostic/ir_clahe/best_full/results（400 个 TXT）
```

> 注：`--train_config` 必须一并替换——它决定 `use_simotm/channels/pairs_*`，不换连通道数都不对。
> baseline 之所以能「只换 weights」，是因为它的训练与推理**碰巧用了同一份 percentile 实现**；
> D 是第一个把这两条路径拆开的实验。修复后二者恢复同构。

---

## 3. 训练期 val 指标（results.csv，ultralytics 口径）

| | D (clahe) | B (baseline) | Δ |
|---|---:|---:|---:|
| best epoch | 263 | 237 | — |
| mAP50 | 0.81879 | 0.81513 | +0.00366 |
| mAP75 | 0.57624 | 0.61859 | −0.04235 |
| mAP50-95 | 0.56806 | 0.55800 | +0.01006 |

逐 epoch：mAP50-95 在 203/300 个 epoch 上 D 更高，mAP50 在 204/300，但 **mAP75 只有 96/300**。
末 50 epoch 均值同样是「mAP50↑ / mAP50-95↑ / mAP75↓」。
→ 训练期与官方口径给出**同一形状的系统性改变**，不是噪声抖动。

### 3.1 训练配置差异更正

`rgbid_ir_clahe_audit.md` §4.4/§14.2 声称配置差异「恰好 2 处」且「忠于 baseline 保留 `patience: 80`」。
**落盘实际为 `patience: 0`**，故真实差异是 **3 处**：`ir_encoding` / `patience` / `name`。
实际影响为零（baseline best 在 ep237，237+80=317>300，baseline 本来就是被 epoch 上限终止），
但记录必须更正。

---

## 4. 裁决

### 4.1 预注册判据

`rgbid_ir_clahe_audit.md` §12：「官方 mAP50-95 **> 0.50928** → 候选；**≤ 0.50928** → 关闭该方向。」

D = 0.51076 **> 0.50928** → **点估计通过筛**。

### 4.2 但增益在噪声底内，不具备裁决力

| 参考噪声 | 量级 | D 的 Δ 相当于 |
|---|---:|---:|
| baseline 自身 8 个 checkpoint 的官方 mAP50-95 波动 | range 0.0103 / **std ≈ 0.0033** | **0.45σ** |
| 配对 bootstrap 的 Δ 分布 σ | 0.00417 | 0.36σ |
| 配对 bootstrap 95% CI | **[−0.00653, +0.00920]** | **含 0**，P(Δ≤0)=0.407 |

四个指标的 CI **全部含 0**——与 P2 探针（0.50232 vs 0.50928，四指标 CI 全含 0）**同一结论模式**。

### 4.3 形状解读（与 F5 同型）

- **mAP75 −0.06225**、框数 −10%：CLAHE 让模型置信度分布整体移动，产生**更少但更"自信"的框**；
- **Recall +0.0178 / Precision −0.0017**、mAP50 −0.00482：低阈值召回上升，但紧定位显著退化；
- 增益仅来自 medium(+0.00753, P=0.215) / large(+0.00444, P=0.258)，**small 为负(−0.00152, P=0.598)**。

即：IR CLAHE **没有改善小目标**（这是 Experiment D 的理论动机），反而在紧定位上付出代价，
总分靠中/大目标的中段阈值勉强抹平。

### 4.4 迁移风险

本项目已记录的「本地官方 Δ → 线上 Δ」迁移记录很差：

| 方案 | 本地官方 Δ | 线上 Δ | 结果 |
|---|---:|---:|---|
| conf=0.0001 | **+0.00488** | **−0.02729** | 方向相反，淘汰 |
| Ensemble 3:1 | +0.00568 | 未提交 | 两口径符号相反 |

D 的本地官方 Δ（+0.00148）**小于** conf=0.0001 当初那个已经迁移失败的 Δ（+0.00488）。
在无统计显著性的前提下提交，期望收益为负。

### 4.5 判定

# ~~`D_REJECTED`~~ → 改判 **`D_WIN`**（线上实测推翻）

**线上实测**：D = **0.5553** vs 基线 **0.53180** → **Δ_online = +0.02350**。
上述 5 条理由**全部只对本地官方口径成立**，对线上评分为假。原判定作废。

### 4.6 错因分析（必读，避免重犯）

**错误**：把本地官方口径（`official_map.py`）当作决策工具，尽管本项目**在两天前就已经证伪过它**。

**证据链**：

| 方案 | Δ 本地官方 | Δ 本地 fork | Δ 线上 | 官方符号 | fork 符号 |
|---|---:|---:|---:|:---:|:---:|
| conf=0.0001（2026-09-17） | +0.00488 | **−0.03248** | **−0.02729** | ✗ | **✓** |
| **D（IR CLAHE）** | +0.00148 | **+0.01192** | **+0.02350** | ✓ | **✓** |
| | | | | **1/2** | **2/2** |

- 2026-09-17 的 `conf_0001_online_submission_audit.md` §139-145 **已经写明**：
  「线上实际落在 fork 所预言的方向上」「**官方口径单独不足以预测线上**」「建议后续任何推理侧决策同时参考两种口径」。
- 记忆条目 `conf-0001-online-rejected` 记的是同一句：「**fork 口径方向才正确**」。
- 本轮我在 §0 表格里**自己列出了 fork Δ = +0.01192**，却仍以官方口径的 +0.00148 作裁决，
  并把 bootstrap CI 当作否决依据——而那个 CI 检验的正是那个已被证伪的代理指标。
- 结论：**预注册判据本身（§12「官方 mAP50-95 > 0.50928」）选错了量具。**
  忠实执行一个错量具，不等于结论正确。

**改正后的量具规则**（建议写入后续所有推理侧决策）：
1. **方向看 fork 口径**（2/2 与线上同号）；官方口径不单独作为否决依据。
2. **幅度不可用任何本地口径外推**（conf=0.0001：fork −0.0325 → 线上 −0.0273，比值 0.84；
   D：fork +0.0119 → 线上 +0.0235，比值 1.97。比值不稳定）。
3. 在不能提交实测的前提下，**只有 fork Δ 明确为负才可关闭一个方向**；
   fork Δ 为正时应视为「未决」，而不是「已证伪」。

### 4.7 用改正后的量具回看历史否决（是否还有漏网的赢家）

| 方案 | Δ 本地官方 | Δ 本地 fork | 结论 |
|---|---:|---:|---|
| **D (IR CLAHE)** | +0.00148 | **+0.01192** | ✅ 唯一 fork 为正 → 线上 +0.02350，**已证实** |
| TTA hflip | −0.00637 | −0.01703 | fork 负 → 否决正确 |
| TTA multiscale | +0.00432 | −0.01445 | fork 负 → 否决正确 |
| Ensemble 1:1 | +0.00944 | −0.00918 | fork 负 → 否决正确（且规则禁止集成） |
| Ensemble 2:1 | +0.00846 | −0.00995 | 同上 |
| Ensemble 3:1 | +0.00901 | −0.00942 | 同上 |
| conf=0.0001 | +0.00488 | −0.03248 | fork 负 → 线上 −0.02729，**已证实否决正确** |
| P2 检测尺度 | −0.00695 | −0.01223 | fork 负 → 否决正确 |
| 1536 微调 | −0.00255 | （未测） | 官方已负，非候选 |

→ **推理侧候选里 fork 为正的只有 D 一个**，历史的否决在改正后的量具下**全部站得住**。
即：本项目并没有因为量具问题丢弃其他真实增益；**D 是唯一的漏网者**。

---

## 5. 冻结清单影响（需你裁决）

`scripts/predict_rect.py` 是 `FREEZE_MANIFEST.sha256` 记录的 17 项之一：

| | 值 |
|---|---|
| 冻结记录 | `656aa58789a467e003517a3f6ca1f05ab25dc065cf5861a8df66468f90528be8` |
| 现状 | `6baa7367da9376a94555428834fb33e4…` |

**这是本次修复引入的、有意的改动**，不是意外篡改。默认路径（`percentile`）已证明数值等价（§1.4）。
但**冻结清单目前与实际不符**，日后按清单校验会报不一致。建议二选一：

- **(a)** 在 `FREEZE_MANIFEST.sha256` 追加一条 `FROZEN_CODE_AMENDED` 记录，注明改动原因与等价性证明；
- **(b)** 把改动移出冻结脚本（新建 `scripts/predict_rect_clahe.py`），恢复 `predict_rect.py` 原哈希。

未在未授权情况下修改该清单。

---

## 6. 产物清单

| 路径 | 内容 |
|---|---|
| `runs/urban_multimodal_det_yolo11_rgbid_ir_clahe/` | 训练产出（best.pt / last.pt / results.csv / args.yaml） |
| `diagnostic/ir_clahe/best_full/results/` | D 的 val 400 图官方口径推理结果（本次新建） |
| `diagnostic/ir_clahe/_infer_best_full.log` | 推理日志（自证 `ir_encoding=clahe`） |
| `diagnostic/ir_clahe/_bootstrap_delta.py` | D vs baseline 的配对 bootstrap（`_bootstrap_d.log`） |
| `diagnostic/ir_clahe/_compat*/` | 补丁兼容性对照（24 图 × 4 组） |
| `diagnostic/ir_clahe/_smoke_test.py` | 原冒烟测试（只覆盖 BaseDataset，未覆盖 loaders） |
| `reports/rgbid_ir_clahe_result.md` | 本报告 |

**未修改**：任何 checkpoint、`configs/*`、`ultralytics/data/base.py`（仍为 `8bcf8349…`）、
正式 submission.zip、`scripts/predict.py`、`official_map.py`。

**已修改（有意）**：`ultralytics/data/loaders.py`、`scripts/predict_rect.py`（见 §1.3/§5）。

---

## 6.1 测试集提交候选包（2026-09-19 生成，待用户上传）

用户指示「以 D 为最优模型跑测试集看线上分数」，故生成候选包。**此动作与 §4.5 的 `D_REJECTED` 判定无关**——
判定未被推翻，只是用户决定实测线上。

**链**：与 0.53180 完全同构，仅 weights / train_config / output 三处路径不同。

```bash
python scripts/predict_rect.py \
  --weights runs/urban_multimodal_det_yolo11_rgbid_ir_clahe/weights/best.pt \
  --train_config configs/train_rgbid_ir_clahe.yaml \
  --source data/raw/test/visible \
  --mode rect --imgsz 1280 --conf 0.001 --iou 0.7 --max_det 300 --max_boxes 100 \
  --batch 16 --output submissions/rgbid_d_ir_clahe_candidate
python scripts/check_submission.py \
  --results submissions/rgbid_d_ir_clahe_candidate/results \
  --zip submissions/rgbid_d_ir_clahe_candidate/submission.zip --expect 1000
```

日志自证：`use_simotm=RGBID channels=5 ir_encoding=clahe`。

**产物**：`submissions/rgbid_d_ir_clahe_candidate/submission.zip`（270,626 B）
**未触碰**：`submissions/rgbird_ir_quicktest/`（0.53180 正式包）

**完整性审计（对拍正式包）**：

| 项 | 正式 0.53180 | D 候选 | 判定 |
|---|---:|---:|---|
| TXT 数 / 与 test 图 1:1 | 1000 / True | 1000 / True | ✅ |
| 框总数 | 7370 | 6388（**−982，−13.3%**） | — |
| 字段数≠6 | 0 | 0 | ✅ |
| 类别 id | 0–11 | 0–11，越界 0 | ✅ |
| 坐标越界 [0,1] / 框体超边界 | 0 / 0 | 0 / 0 | ✅ |
| conf 越界 / NaN·Inf | 0 / 0 | 0 / 0 | ✅ |
| conf 范围 | [0.001000, 0.981424] | [0.001002, 0.984406] | ✅ |
| 退化框 w=0\|h=0 | 155（2.10%） | 98（1.53%） | ✅ 更少 |
| zip 条目 / 目录条目 / 非 `results/*.txt` / CRC | 1000 / 0 / 0 / OK | 1000 / 0 / 0 / OK | ✅ |
| `check_submission.py` 10 项 | — | ALL PASS（单图最多 54 框，0 张触发 100 框截断） | ✅ |

**线上实测结果**：**0.5553**（用户上传后返回）

| | 本地官方 | 本地 fork | **线上** |
|---|---:|---:|---:|
| Baseline | 0.50928 | 0.55415 | 0.53180 |
| D | 0.51076 | 0.56607 | **0.5553** |
| Δ | +0.00148 | +0.01192 | **+0.02350** |

- 本地→线上**水平**落差：基线 0.55415→0.53180（−0.02235，fork）/ 0.50928→0.53180（+0.02252，官方）；
  D 0.56607→0.5553（**−0.01077**，fork）/ 0.51076→0.5553（+0.04454，官方）。
- 两个本地口径**夹住**了线上值：线上落在官方之上、fork 之下；官方低估、fork 高估。
- ⚠️ 提交前的预期（「围绕 0.53180 持平到小幅波动」）**被实测推翻**，实际是 +0.02350。
  当时的推理依据是「本项目两次记录 Δ 迁移失败」——但那两次的 fork Δ 都是**负的**，
  不能用来预测一个 fork Δ 为正的方案。**判据用错了对象。**

**D 自此为当前最优已验证模型（线上 0.5553），正式方案由 `rgbird_ir_quicktest` 迁移至
`rgbid_ir_clahe`。** 距目标 0.570 还差 **0.0147**。

---

## 7. 遗留隐患

`ultralytics/data/dataset.py:632`（`ClassificationDataset`）仍有**第 3 份**硬编码 percentile 的
RGBID IR 预处理。当前 5ch 检测路径不走它，故不影响本实验；但若将来有人用它加载 RGBID 数据，
会静默拿到 percentile。建议后续统一为同一个 helper。**本次未改**（超出授权范围）。
