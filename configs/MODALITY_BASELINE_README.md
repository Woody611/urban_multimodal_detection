# Fusion Phase 0 — Modality Baseline & Fusion Configuration

> ## ★ Infrastructure Repair 已完成（2026-10-03）
> 本 README 下面 §1/§2 的 **Status 列已被本轮修复取代**。现行状态：
>
> | ID | Modalities | use_simotm | channels | Status |
> |---|---|---|---:|---|
> | M1 | RGB | `Gray2BGR` | 3 | **READY** |
> | M2a | IR + CLAHE | `Infrared` | 3 | **READY**（'Infrared' 分支已读 `ir_encoding`） |
> | M2b | IR + percentile | `Infrared` | 3 | **READY** |
> | M3 | Depth | `Depth` | 3 | **READY**（推理侧 image 路径已补 Depth 分支） |
> | M4 | RGB+IR | **`RGBIR`** | 4 | **READY**（新分支，IR 走 `ir_encoding`） |
> | M5 | RGB+Depth | `RGBD` | 4 | **READY** |
> | M6 | IR+Depth | **`IRD`** | 2 | **READY**（新模式） |
> | M7 | D′ | `RGBID` | 5 | **incumbent（未改动）** |
>
> 三条原 BLOCKER 全部解除；三条新修复见 `diagnostic/FUSION_PHASE0_REPAIR_REPORT.md`。
> 强制门：`python -X utf8 scripts/validate_modality_contract.py --all`（8/8 PASS）。
>
> ⚠ **`ultralytics/data/base.py` 与 `ultralytics/data/loaders.py` 被 `.gitignore:11` 忽略**，
> 改动**不进 git**；只能用 SHA-256 追踪（见修复报告 §G）。
>
> ⚠ `use_simotm='RGBT'` **不再用于本矩阵**：其 IR 分支不做任何对比度处理（raw），
> 无法表达 D′ 的 IR-CLAHE。M4 改用新增的 `RGBIR`。既有 `RGBT` 行为**未被改动**。


**状态** 2026-10-03 · 仅配置与审计，**未训练、未 forward、未用 GPU**
**Incumbent** D′ = SepStem + IR-CLAHE, RGB+IR+Depth early fusion, imgsz 1280, 300 ep, YOLO11m
→ 本地 **official mAP50-95 = 0.515281**，线上 **48.712**

---

## 0. 本阶段要回答的问题

> D′ 的 0.515281，是受限于**单模态本身的能力**，还是 **early fusion 在融合中损失了模态信息**？

要回答它，必须先有不被污染的**单模态 / 双模态基线**。本目录就是那批基线。

---

## 1. 实验矩阵

| ID | Modalities | Config | Model YAML | channels | Fusion | Status |
|---|---|---|---|---|---|---|
| **M7** | RGB+IR+Depth | `train_rgbid_sepstem_clahe.yaml` | `yolo11m_sepstem.yaml` | 5 | D′ | **incumbent** |
| **M1** | RGB | `train_modality_m1_rgb.yaml` | `yolo11m_modality3ch.yaml` | 3 | single | **READY WITH CAVEAT** |
| **M2a** | IR + CLAHE | `train_modality_m2a_ir_clahe.yaml` | `yolo11m_modality3ch.yaml` | 3 | single | **BLOCKED** |
| **M2b** | IR + percentile | `train_modality_m2b_ir_percentile.yaml` | `yolo11m_modality3ch.yaml` | 3 | single | **READY WITH CAVEAT** |
| **M3** | Depth | `train_modality_m3_depth.yaml` | `yolo11m_modality3ch.yaml` | 3 | single | **READY WITH CAVEAT** |
| **M4** | RGB+IR | `train_modality_m4_rgb_ir.yaml` | `yolo11m_modality4ch.yaml` | 4 | early | **BLOCKED** |
| **M5** | RGB+Depth | `train_modality_m5_rgb_depth.yaml` | `yolo11m_modality4ch.yaml` | 4 | early | **BLOCKED** |
| **M6** | IR+Depth | `train_modality_m6_ir_depth.yaml` | `yolo11m_modality2ch.yaml` | 2 | early | **BLOCKED ×2** |

---

## 2. 逐个配置

### M1 — RGB-only `train_modality_m1_rgb.yaml`

| 项 | 值 |
|---|---|
| purpose | 真彩 3 通道单模态上界 |
| input channels | 3（`use_simotm: "Gray2BGR"` ⇒ `imread(path)` 纯 BGR，**无灰度化**） |
| preprocessing | 无（原图 8-bit BGR，letterbox 后 `/255`） |
| model yaml | `configs/yolo11m_modality3ch.yaml`（`ch: 3`，scale=m 由文件名判定） |
| pretrained | `yolo11m.pt` — `intersect_dicts` **643/649 = 99.1%**，`_transfer_rgb_pretrained` 返回 0（无需 remap） |
| expected evaluator | `scripts/predict_rect.py` → `scripts/official_eval.py`（official 口径，每图 ≤100 框） |
| caveat | ① `Gray2BGR` 分支在 configs/ 下**从未被使用过**，属首次启用路径；② 3ch 下 `RandomHSV`/`Albumentations` 与 D′（5ch）不同 ⇒ 已用 `aug.hsv_*=0` 压掉 HSV，残余见 §4 |

> **⚠ 历史陷阱**：`configs/train.yaml` / `configs/train_e0.yaml` 把 `use_simotm: "SimOTMBBS"` 标为 "(RGB baseline)"，
> 但 `SimOTMBBS` 的分支体是 `imread(..., IMREAD_GRAYSCALE)` + `cv2.merge([gray, blur, blur])`
> （`base.py:675-678`）——**灰度+3×3 模糊，不是 RGB**。M1 必须用 `Gray2BGR`。

### M2a — IR-only + CLAHE `train_modality_m2a_ir_clahe.yaml`

| 项 | 值 |
|---|---|
| purpose | **与 D′ 的 IR 编码完全一致**的 IR 单模态上界 |
| input channels | 3（`[IR,IR,IR]`） |
| preprocessing | 需要 CLAHE，但**当前代码做不到** → 见 caveat |
| model yaml | `configs/yolo11m_modality3ch.yaml` |
| pretrained | 同 M1（99.1%） |
| **Status** | **BLOCKED — 不要训练** |
| caveat | `use_simotm='Infrared'` 的分支（`base.py:280-294`、`loaders.py:535-561`）把 IR 对比度处理**硬编码为 1%/99% percentile**，**从不读 `ir_encoding`**；`scripts/train.py:638` 的硬防呆**只覆盖 `use_simotm=="RGBID"`**。⇒ 本配置会静默按 percentile 跑，与 2026-09-20 的 ir_encoding 静默回落事故同类 |

### M2b — IR-only + percentile `train_modality_m2b_ir_percentile.yaml`

| 项 | 值 |
|---|---|
| purpose | 可运行的 IR 单模态基线；也是 D′ 的 **non-CLAHE 对照** |
| input channels | 3（`[IR,IR,IR]`） |
| preprocessing | 16-bit→uint8（如有）→ **1%/99% percentile 拉伸** |
| model yaml | `configs/yolo11m_modality3ch.yaml` |
| caveat | D′ 用 **CLAHE**，M2b 用 **percentile** ⇒ 比 IR 单模态横向比较时必须声明这一项。**真正"不处理"的 raw IR 在当前代码下不可配置**（需额外源码开关，本轮不申请） |

### M3 — Depth-only `train_modality_m3_depth.yaml`

| 项 | 值 |
|---|---|
| purpose | Depth 单模态上界 |
| input channels | 3（`[D,D,D]`） |
| preprocessing | 16-bit PNG(mm) → `<300mm` 置 0 → `/19999×255` → uint8，复制为 3 通道。**与 D′ 的 depth 通道公式逐字相同** |
| model yaml | `configs/yolo11m_modality3ch.yaml` |
| caveat | D′ 的 depth 分支是 **1ch stem**，初值 `mean(W_R,W_G,W_B)`；M3 是**完整 RGB stem 吸收三份相同深度**。这是单模态唯一自然选择，但**不是 D′ aux-stem 初始化方式的复刻**，需在结论里声明 |

### M4 — RGB+IR `train_modality_m4_rgb_ir.yaml`

| 项 | 值 |
|---|---|
| purpose | 去掉 Depth 后 early fusion 的表现 |
| input channels | 4（`use_simotm='RGBT'` ⇒ `[B,G,R,IR]`，数据侧**已可用**） |
| model yaml | `configs/yolo11m_modality4ch.yaml` |
| **Status** | **BLOCKED — 模型侧** |
| caveat | `_transfer_rgb_pretrained` 的 5ch 分支写死 `in_channels == 5`；4ch 不匹配，且**无 1ch stem** ⇒ `return 0`（**无日志、无 raise**）。实测：`intersect_dicts` 642/649，**恰好只有 `model.0.conv.weight` 被丢弃** ⇒ 首个 4ch Conv 静默保持随机初始化，其余层全预训练 ⇒ **与 D′ 不可比** |

### M5 — RGB+Depth `train_modality_m5_rgb_depth.yaml`

| 项 | 值 |
|---|---|
| purpose | 去掉 IR 后 early fusion 的表现 |
| input channels | 4（`use_simotm='RGBD'` ⇒ `[B,G,R,D]`） |
| model yaml | `configs/yolo11m_modality4ch.yaml` |
| **Status** | **BLOCKED — 模型侧（同 M4）** |
| caveat | ⚠ `RGBD` 模式下 `pairs_rgb_ir` 承载的是 **depth** 映射（沿既有 `train_multimodal_rgbd.yaml` 写法） |

### M6 — IR+Depth `train_modality_m6_ir_depth.yaml`

| 项 | 值 |
|---|---|
| **Status** | **BLOCKED ×2 — config only，不训练** |
| caveat | ① 数据侧无此模式：`base.py`/`loaders.py` 的 `_merge_channels*` 只有 (RGB+IR)/(RGB+IR+RGB)/(RGB+IR+D)/(RGB+D)，**没有 (IR+D)**；② 模型侧 2ch remap 缺失（同 M4） |

---

## 3. 训练命令（**仅在 Status=READY\* 时**）

```bash
# M1
python scripts/train.py --model_config configs/yolo11m_modality3ch.yaml \
                        --train_config configs/train_modality_m1_rgb.yaml
# M2b
python scripts/train.py --model_config configs/yolo11m_modality3ch.yaml \
                        --train_config configs/train_modality_m2b_ir_percentile.yaml
# M3
python scripts/train.py --model_config configs/yolo11m_modality3ch.yaml \
                        --train_config configs/train_modality_m3_depth.yaml
```

评测（**必须用 official 口径**，不得用 ultralytics val mAP 作为结论）：

```bash
python scripts/predict_rect.py --weights runs/<exp>/weights/best.pt \
                               --train_config configs/train_modality_m1_rgb.yaml \
                               --split val --pack
python scripts/official_eval.py --results <predictions.txt> \
        --images data/processed/visible_split/images/val \
        --labels data/processed/visible_split/labels/val --metric METRIC_A
```

> `predict_rect.py` 从 `--train_config` 读 `use_simotm / channels / pairs_* / ir_encoding`，
> 因此**权重与配置必须成对**，否则输入分布会被静默改变。

---

## 4. 已声明的差异与残余不可消除项

| # | 差异 | 影响范围 | 处置 |
|---|---|---|---|
| D1 | **无 `rect_late_start_epoch`** | M1/M2a/M2b/M3/M4/M5/M6 | 原版 D′ 配置本身也没有该键（仅 rectlate10 变体有），且切换点 ep291 > D′ best@ep246，对 0.515281 无贡献。为让所有被比较 checkpoint 同处 **square 几何**，不启用 |
| D2 | **`aug.hsv_* = 0.0`** | 全部 | D′ 为 5ch ⇒ `RandomHSV` 在 `augment.py:3497` 直接 return（"光度增强惰性"）。3ch/4ch 下 HSV 会生效，故显式置 0 对齐 |
| D3 | **`Albumentations(p=1.0)` vs `Albumentations(p=0)`** | M1/M2a/M2b/M3（3ch）/ M4/M5（4ch 走 `Albumentations4C`） | **不可由 config 消除**。差异内容：`A.Blur(p=.01) / MedianBlur(p=.01) / ToGray(p=.01) / CLAHE(p=.01) / RandomBrightnessContrast(p=0) / RandomGamma(p=0)`。**仅当云环境装了 `albumentations` 包时生效**（本机未安装 ⇒ 本机为 no-op）。**Batch-1 前必须实测云环境是否安装**；若已安装，需评估是否接受该 1% 级差异（或做一次 1 行源码改动把 5ch 的门限放宽，本轮不申请） |
| D4 | IR 编码 | M2b | percentile ≠ D′ 的 CLAHE（M2a 被 block，无法消除） |
| D5 | aux stem 初始化语义 | M3 | M3 用完整 3ch RGB stem；D′ 的 depth 分支是 1ch + `mean(W_R,W_G,W_B)` |
| D6 | 预训练来源路径不同但等价 | 全部 | D′ 靠 `_remap_separate_stem` 传 **550** 张量（标准 `load()` 只匹配 61/661 = 9.2%，因为 Silence 前缀导致 index 位移）；3ch 基线靠标准 `load()` 传 **643/649 = 99.1%**。两者同源 `yolo11m.pt`，**Detect 分类头在两边都被重新初始化**（D′ 6 张量、3ch 6 张量）⇒ 公平 |

---

## 5. Validation（CPU-only，已执行）

```bash
python -X utf8 diagnostic/fusion_phase0_validate.py
```

覆盖：yaml 解析 · 路径存在性 · `channels`（config）== `ch`（model yaml）== 模式实际产出 ·
canonical training block 逐项比对 · `ir_encoding` 是否真的生效 · `_transfer_rgb_pretrained` 分派预演。

**开训前仍需人工确认的两项**（本机无法判定）：
1. 云环境是否安装 `albumentations`（决定 D3 是否生效）。
2. `use_simotm="Gray2BGR"` 在云上首次运行时的实读通道数 = 3（本机已静态确认分支体）。

---

## 6. 不要做的事

- 不要用 `SimOTMBBS` 当 RGB baseline（它是灰度）。
- 不要在 M2a 修好之前训练它（会静默退化成 M2b）。
- 不要在 M4/M5/M6 修好 remap 之前训练它们（stem 会是随机初始化）。
- 不要用 ultralytics val mAP 作为跨配置结论（本项目的判据是 **official 口径**）。
