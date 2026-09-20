# Small Object Crop 实验 · 最终报告

**日期**：2026-09-19
**最终状态**：**`SMALL_OBJECT_CROP_CLOSED_NOT_RUN`**（未训练结案）
**正式模型**：未变更

---

## 1. 最终判决

```
SMALL_OBJECT_CROP_REJECTED  (未训练，基于既有量化审计结案)
OFFICIAL_MODEL_REMAINS_0.53180
```

**本次未运行任何训练。** 依据用户 2026-09-19 裁决：接受既有 `CROP_NO_GO` 结论，结案。

---

## 2. 为什么未运行（两项独立事实）

### 2.1 实现从未存在

任务书假定 `ultralytics/data/base.py` 中存在「新增的 `apply_small_object_crop()`」，并已配有
`scripts/small_object_crop_smoke_test.py` 与 `configs/train_rgbid_small_object_crop.yaml`。

**三者均不存在** —— 本地仓库、git 历史、6 个同步 zip（`ultralytics/ scripts/ configs/ experiments/ diagnostic/ reports/`）全部检索无命中。

全仓库仅 3 处提及 `small_object_crop`，且均为文档/只读脚本：

| 路径 | 性质 |
|---|---|
| `reports/next_optimization_review.md` §6 | 设计提案，原文标注「**本轮不实现**」 |
| `reports/rgbid_small_object_crop_audit.md` | 只读可行性审计，结论 `CROP_NO_GO`，原文「**不提出 Probe**」 |
| `diagnostic/small_object_crop_audit/_crop_audit.py` | 只读模拟脚本，不接入任何训练入口 |

当前 `get_image_and_label`（`base.py:586-605`）实际调用序列为
`load_image → apply_modality_dropout → ratio_pad → update_labels_info`，无 crop 环节。

### 2.2 既有审计已给出量化 `CROP_NO_GO`

`reports/rgbid_small_object_crop_audit.md`（2026-09-18，只读，全 1600 train 图实测模拟）两条决定性依据：

1. **与 mosaic 功能冗余。** crop 放大倍数恒为 `1/√r` ∈ **1.16×–1.58×**（r=0.75→0.40），
   而 mosaic 自身缩小 **2.00×**。即使最激进 r=0.40，crop+mosaic 下 small 中位 √area = **14.0 px**，
   仍**低于 baseline 自身无 mosaic 窗口内已有的 17.7 px（−21%）**。
   ⇒ crop 未创造新尺度区间，只是**部分抵消** mosaic。
2. **同族机制已两次实测为中性。**
   ① 1536 训练 = 全局 **1.44×** 放大（**强于** crop 的 1.16–1.58×），两个 LR 探针官方 0.49401 / 0.50673，**small AP 仅 +0.0019**；
   ② P2 检测层 **Δ = +0.00013**。

同审计诚实记录的两条**有利**发现（不足以改变判定）：部分截断标签噪声 ≤1.4%（传统 crop 最大顾虑不成立）；多模态同步为构造性保证（「先合并再裁」与「先分别裁再合并」实测逐位相同）。

**变量隔离前提**：crop 若要真正超过 baseline 自身尺度，必须在被 crop 图上**同时关闭 mosaic** —— 那是第二个变量，已被排除。这一点单独即足以否证「严格单变量」形态下的本方案。

---

## 3. 任务书要求的 15 项内容对照

| # | 项 | 状态 |
|---|---|---|
| 1 | 实验假设 | 已定义（小目标有效训练分辨率不足） |
| 2 | 唯一变量 | 已定义（`small_object_crop` OFF→ON） |
| 3 | 实际配置 | ⛔ 不存在（`configs/train_rgbid_small_object_crop.yaml` 未创建） |
| 4 | smoke test | ⛔ 未执行（脚本不存在） |
| 5 | GPU integration test | ⛔ 未执行（无实现可测） |
| 6 | training provenance | ⛔ 未训练 |
| 7 | official evaluation | ⛔ 未评测（无新 checkpoint） |
| 8 | baseline vs experiment | 见 §4 |
| 9 | ΔmAP50-95 | **N/A**（Experiment 侧不存在） |
| 10 | bootstrap / 波动分析 | N/A |
| 11 | small/medium/large | N/A |
| 12 | 框数量变化 | N/A |
| 13 | 最终判决 | `SMALL_OBJECT_CROP_REJECTED`（未训练结案） |
| 14 | 是否改变正式模型 | **否** |
| 15 | 当前正式模型 SHA256 | 见 §4 |

---

## 4. Baseline vs Experiment

| 项目 | RGBID Baseline | Small Object Crop |
|---|---|---|
| 权重 | `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt` | **未产生** |
| 线上榜单 mAP50-95 | **0.53180** | — |
| 本地官方口径 mAP50-95 | 0.50928 | — |
| 变更 | — | 无 |

> **口径说明**（2026-09-19 更正）：0.53180 是**比赛平台线上分数**；用 `predict_rect.py` + `official_map.py` 在 400 图 val 上的本地复现为 **0.50928**。二者相差 +0.02252（本地官方 → 线上偏移）。此前本节与 §1 横幅把 0.53180 标为「官方 mAP50-95」而未区分口径，属表述不精确，已更正。

**ΔmAP50-95 = N/A**（未训练，无 Experiment 侧数值可比）

### 正式模型冻结校验（2026-09-19 复核）

```
f9dddbfaca4cb08bde31d082eb0a54d499b8441eb1f97c6134a028caa6d83e55  40635237
runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt
```

与 `reports/FREEZE_MANIFEST.sha256` 记录**逐字节一致** ✅

正式资产链复核：`best.pt` / `submission.zip` / `configs/train_rgbird_ir_quicktest.yaml` /
`configs/yolo11m_earlyfusion.yaml` / `data/processed/rgbid_split/dataset.yaml` / `args.yaml` /
`results.csv` **全部完好**。

---

## 5. 遗留风险（本次未修，独立于 crop，建议单独立项）

### 5.1 `.gitignore` 无锚定 `data/` 规则吞噬 `ultralytics/data/`

```
$ git check-ignore -v ultralytics/data/base.py
.gitignore:4:data/    ultralytics/data/base.py
$ git ls-files ultralytics/data/ | wc -l
0
```

`.gitignore` 第 4 行 `data/` **无锚定**，同时匹配项目根 `data/` 与 `ultralytics/data/`。
后果：RGBID 的 **5 通道加载 / IR 预处理 / 几何变换插入点**所在目录
**不在 git、不在 FREEZE_MANIFEST，修改不可 diff、不可回溯、不可校验**。

| 文件 | SHA256(前32) | mtime | 在 manifest |
|---|---|---|---|
| `ultralytics/data/base.py` | `8bcf834930155bd58a99ccfc5e871133` | **09-18 22:29** | ❌ |
| `ultralytics/data/loaders.py` | `f84bdcf29081e6f95b58fbf19c255d1d` | **09-19 12:00** | ❌ |
| `ultralytics/data/dataset.py` | `0aba2bc14793ab85ac477d0eab0cf408` | 09-17 19:19 | ❌ |
| `ultralytics/data/augment.py` | `1e0ca17c665db9ebf9a9bb6d019ddb5d` | 09-13 07:12 | ❌ |
| `ultralytics/nn/tasks.py` | `8228854e9db090b45cbd62590c7c5e5b` | 09-06 13:39 | ❌ |
| `ultralytics/engine/trainer.py` | `de94f366c8218916c88e001b2ef22a08` | 08-25 09:27 | ❌ |
| `scripts/train.py` | `80f6cdec1e52d41258ca7726e4ff79c7` | 09-18 22:29 | ❌ |
| `scripts/official_map.py` | `7b4aa24d2856391c3fbeb237c3453d7f` | 09-13 12:20 | ❌ |

前两项 mtime **晚于** manifest 生成时刻（09-17 13:55），即官方评测路径代码在冻结后被改动且无哈希可证中性。

**建议修法**：`.gitignore` 改为 `/data/`（锚定根目录）；将上表文件 SHA256 补入 FREEZE_MANIFEST。

### 5.2 `scripts/predict_rect.py` 相对冻结版漂移

manifest 冻结版 `656aa58…`（11286 B，= commit `41abf7b`）→ 磁盘 `6baa736…`（11724 B）。
两次冻结后修改：`aec32aa` 新增 `ir_encoding` 透传、`2cc8961` 新增 1 行 print。

静态分析判定对正式 RGBID 配置**行为中性**（该配置无 `ir_encoding` 键 → `"percentile"` 默认路径；
`loaders.py:656` 的 CLAHE 分支需 `== "clahe"` 才进入），但**未做输出逐位比对**。
若后续还需线上评测，建议补做冻结版 vs 当前版在同 split 上的 submission TXT 逐位一致性实测。

---

## 6. 合规状态

| 项 | 状态 |
|---|---|
| 训练 | ❌ 未执行 |
| 正式 best.pt / last.pt / results.csv / submission.zip | ✅ 逐字节未变 |
| 正式配置 / trainer / dataset / dataloader / augmentation | ❌ 一字未改 |
| 验证集 | ✅ 未触碰 |
| 新 submission | ❌ 未生成 |
| 新增文件 | `reports/small_object_crop_phase0_freeze.md`、`reports/small_object_crop_final_report.md`（本文件） |

---

```text
SMALL_OBJECT_CROP_REJECTED
OFFICIAL_MODEL_REMAINS_0.53180
```
