# Fusion Phase 0 — Audit Report

**日期** 2026-10-03 · **性质** ZERO GPU / ZERO training / ZERO forward / 无 source·config 语义改动 / 无 commit
**产出** `configs/train_modality_m1..m6*.yaml`、`configs/yolo11m_modality{2,3,4}ch.yaml`、`configs/MODALITY_BASELINE_README.md`、`diagnostic/fusion_phase0_validate.py` + `fusion_phase0_validation.json`

---

## Executive Verdict

**单模态基线可以公平地做（M1/M2b/M3 已就绪），但"与 D′ 的 IR 编码完全一致"的 IR 基线（M2a）和全部双模态早期融合（M4/M5/M6）被**同一类共享源码缺口**挡住**：`ultralytics/data/*.py` 的 `Infrared` 分支不读 `ir_encoding`，`ultralytics/models/yolo/detect/train.py` 的 `_transfer_rgb_pretrained` 只为 5ch / (3+1+1)ch 两种布局写了预训练迁移。两者都是**静默降级**——不会报错，只会让实验悄悄变成另一个实验（IR+CLAHE 变 IR+percentile；4ch stem 保持随机初始化）。好消息是：**数据划分、标签、评测器、通道序、训练预算这四块经实测完全可比**，D′ 未被触碰（SHA 不变），且 M1/M2b/M3 可以在**不改任何共享源码**的前提下立刻开跑。

**FINAL VERDICT: `B — READY WITH CAVEATS`**（Batch-1 的 3 个实验全部可跑；Batch-2 需要一次最小源码修复）

---

## 1. D′ Reproducibility Fingerprint

| 项 | 值 |
|---|---|
| config path | `configs/train_rgbid_sepstem_clahe.yaml` · **sha256 `a4e329cfc3d220206448cdd3377c1b3f5126c50f13c907ada522d98725486e12`** |
| model yaml | `configs/yolo11m_sepstem.yaml` · **sha256 `9b14f2946073338414b4441784b6df87a868b982be32ac8d50b6d1c4a17f3dd8`** |
| dataset | `data/processed/rgbid_split_train/dataset.yaml`（train `images/train/visible` = **1600**，val = **400**） |
| train script | `scripts/train.py` + `--model_config` / `--train_config` / `--dataset_config`（默认 `configs/dataset.yaml`） |
| IR preprocessing | **CLAHE** `clipLimit=2.0, tileGridSize=(8,8)`（`base.py:334-336`），**互斥于** percentile |
| input channels | **5** = `[B, G, R, IR, D]`（`_merge_channels_rgbid`，`base.py:430-433`） |
| fusion implementation | **SepStem**：`Silence → SilenceChannel[0,3]→Conv(3→48) ‖ [3,4]→Conv(1→8) ‖ [4,5]→Conv(1→8) → Concat(64) → stock YOLO11m` |
| pretrained | `yolo11m.pt`；标准 `load()` 只匹配 **61/661 = 9.2%**（Silence 前缀导致 index 位移）⇒ 真正迁移靠 `_remap_separate_stem`，**实测 550 张量**（含 3 个 stem + layers 8..30），Detect cv3 分类分支按 nc=12 **重新初始化 6 张量** |
| optimizer | **SGD + nesterov**, lr0 5.0e-3, lrf 0.01, momentum 0.937, weight_decay 5.0e-4 |
| warmup | 3.0 epoch（`warmup_bias_lr=0.1`, `warmup_momentum=0.8`） |
| augmentation | 全部 `ultralytics/cfg/default.yaml` 默认；**5ch ⇒ `Albumentations(p=0)` 且 `RandomHSV` 直接 return**（光度增强惰性） |
| close_mosaic | 10 |
| batch / imgsz / epochs / seed | 8 / 1280 / 300 / 42（`patience: 0`，`val_ratio: 0.2`） |
| loss | 默认 `v8DetectionLoss`，普通 BCE，`cls_pw: []` |
| evaluator | `scripts/predict_rect.py` → `scripts/official_eval.py`（official 口径，每图 ≤100 框，METRIC_A/B 双算） |
| weights | `runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt`（40,641,641 B, 2026-09-22 07:33，未被本轮触碰） |
| 结果 | local official **0.515281** · online **48.712** |

---

## 2. Modality Data Path Audit

```
configs/dataset.yaml  (path=…/data/raw, train=train/visible, val=…, nc=12)
        │
        ├─ scripts/train.py::main  —— val_ratio=0.2 且 base yaml 的 val 无标注 ⇒ 按 use_simotm 分派
        │      RGBT|Infrared        -> rgbt_split
        │      Depth|RGBD           -> depth_split_train
        │      RGBID                -> rgbid_split_train          ← D′
        │      (else)               -> visible_split              ← M1 (Gray2BGR)
        │   ⚠ 出目录已存在 ⇒ `if out_yaml.exists(): return` 幂等复用，不会重切
        │
        ▼
BaseDataset.load_image → load_and_preprocess_image(file_path, use_simotm, pairs_rgb, pairs_ir, pairs_depth)
        │   ★ 第二/第三模态靠**路径字符串替换**：
        │        file_path.replace("visible", "infrared")   /   ("visible", "depth")
        │
        ├─ RGB      : 'Gray2BGR' | (else)  -> imread(path)                 -> HxWx3  BGR
        ├─ IR       : 'Infrared'           -> imread(…ir…, UNCHANGED)[...,0]
        │                                     → uint8 → **percentile 1/99**（硬编码）→ np.stack×3
        ├─ Depth    : 'Depth'              -> imread(…d…, UNCHANGED)[...,0]
        │                                     → <300mm 置 0 → /19999×255 → uint8 → np.stack×3
        ├─ RGB+IR   : 'RGBT'    -> _merge_channels(vis, ir)        -> HxWx4 [B,G,R,IR]
        ├─ RGB+RGB  : 'RGBRGB6C'-> _merge_channels_rgb(vis, ir)    -> HxWx6
        ├─ RGB+D    : 'RGBD'    -> _merge_channels(vis, d)         -> HxWx4 [B,G,R,D]
        └─ RGB+IR+D : 'RGBID'   -> _merge_channels_rgbid(vis, ir, d) -> HxWx5 [B,G,R,IR,D]
        │
        ▼  下游：v8_transforms（Mosaic / RandomPerspective / LetterBox）→ trainer `.float()/255`
```

**回答 brief §2 的三个问题：**

* **RGB-only** — 输入就是 **3 通道原彩 BGR**，**不存在 dummy 通道**。但必须走 `Gray2BGR` 分支（`base.py:248`：`im = imread(file_path)  # BGR`）；任何**未识别**的 `use_simotm` 也会落到同一 `else`，那是静默兜底，不可依赖。
* **IR-only** — `16-bit → uint8`（若有）→ **percentile 1%/99% 拉伸** → `np.stack×3`，dtype `uint8`。**★ D′ 的 IR-CLAHE 只对 D′ 的 IR 生效，且 IR-only 路径根本读不到 `ir_encoding`** —— 这正是 M2a 的 blocker，不是"意外路径"而是"缺路径"。
* **Depth-only** — `16-bit I;16` → `<300` 置 0 → `/19999×255` → `uint8` → `stack×3`。**表达式与 RGBID 分支的 depth 处理逐字相同**（`base.py:269-279` vs `341-348`）⇒ 语义明确，**NOT BLOCKED**。

**★ 关键公平性事实（实测）**：
`visible_split` / `rgbt_split` / `depth_split_train` / `rgbid_split` / `rgbid_split_train` 的 val 划分**逐位相同**（各 400），train 也相同（1600 → 200 batch × 8 = 与 D′ checkpoint `updates=25/epoch` 一致）；`rgbid_split_train` 内 `visible == infrared == depth` 的 stem 集合完全相同。

---

## 3. Input / Channel Audit

| ID | 通道序 | 来源 |
|---|---|---|
| M1 RGB | `[B, G, R]` | `imread()` 原生 BGR |
| M2a/M2b IR | `[IR, IR, IR]` | `np.stack([im]*3)` |
| M3 Depth | `[D, D, D]` | 同上 |
| M4 RGB+IR | `[B, G, R, IR]` | `_merge_channels` |
| M5 RGB+Depth | `[B, G, R, D]` | `_merge_channels` |
| M6 IR+Depth | 不存在 | 无分支 |
| **M7 = D′** | **`[B, G, R, IR, D]`** | `_merge_channels_rgbid`（`base.py:430-433`，从代码确认，非变量名推测） |

**三处通道数必须一致且已校验**：
`train_config.channels` → `args.channels` → `build_transforms(channels=…)`（`dataset.py:465`，控制 LetterBox padding / RandomHSV 门限）；
`model_yaml.ch` → `DetectionModel`（`tasks.py:347`，**只有 yaml 的 `ch` 被读，`channels` 参数默认 3 被忽略**）。
`augment.py:2881-2887` 在两者不符时 `raise ValueError`（**响亮失败**，非静默）。

> ⚠ **`configs/yolo11_visible.yaml` 只写 `channels: 3`、没有 `ch:`** —— 它是靠 `DetectionModel(ch, ch=3)` 的**参数默认值**侥幸正确。新 yaml 已显式写 `ch:`。

---

## 4. Pretrained Initialization Audit

**统一源**：`yolo11m.pt`（仓库根，40,684,120 B）。**无任何模态使用随机初始化**。

| ID | 迁移路径 | 实测张量 | 匹配率 |
|---|---|---|---|
| M1/M2a/M2b/M3 | 标准 `model.load()`（`_transfer_rgb_pretrained` 返回 **0**，无 remap） | **643 / 649** | **99.1%** |
| **D′ (M7)** | `_remap_separate_stem`（标准 `load()` 仅 61/661 = 9.2%，因 Silence 前缀 index 位移） | **550** | — |
| M4/M5/M6 | **`return 0`，无日志无 raise** | 642/649，**唯一被丢弃的键 = `model.0.conv.weight`** | ❌ |

**⇒ 公平性结论**：D′ 与 3ch 基线的预训练**同为全量、同源**；差异只在**路径**（显式 remap vs 标准 load），而两条路径的**未迁移键都恰好是 Detect cv3 分类分支的 6 个张量**（D′ 侧日志原文："Detect cv3 class branch re-initialised for nc=12: 6 tensors not transferred"）⇒ **分类头在两边都是重新初始化的，不存在"一边预训练一边随机"的不公平。**

**IR 复制的语义声明**：M2a/M2b 是 `IR → [IR,IR,IR]` 喂进 **RGB 预训练 stem**。这是"把 IR 当作 3 通道图像"，**不是**原生 IR 表示；brief §5 要求显式声明，此处声明。

---

## 5. Augmentation / Geometry Audit

| 组件 | D′ (5ch) | M1/M2a/M2b/M3 (3ch) | M4/M5 (4ch) | 处置 |
|---|---|---|---|---|
| Mosaic / mixup / copy_paste / fliplr / translate / scale / degrees / shear / perspective / flipud | 全部 `default.yaml` 默认 | 同 | 同 | ✅ 完全一致 |
| `close_mosaic` | 10 | 10 | 10 | ✅ |
| LetterBox（几何） | 对**每个** modality 用同一 `ratio_pad`？ | — | — | ⚠ 见下 |
| **RandomHSV** | **直接 return**（`augment.py:3497` `if img.shape[-1] != 3`） | **生效**（hsv_h .015 / hsv_s .7 / hsv_v .4） | `RandomHSV4C` 生效 | **已置 0 对齐**（`aug.hsv_*=0`） |
| **Albumentations** | `Albumentations(p=0)` | `Albumentations(p=1.0)` | `Albumentations4C(p=1.0)` | **不可由 config 消除**；内容为 `Blur/MedianBlur/ToGray/CLAHE` 各 p=0.01；**仅当云环境装了 `albumentations` 包时生效**（本机 `ModuleNotFoundError` ⇒ 本机 no-op） |
| RandomErasing (`erasing: 0.4`) | — | — | — | **inert**：`erasing` 只被 `classify_augmentations` 消费（`augment.py:2859`），detect 任务不用 |

**geometry 同步**：RGBID 模式下，`_resize_images_3` 对三路分别按 `imgsz/max(h,w)` 缩放后再 merge，随后**单一 5ch 张量**走 Mosaic/RandomPerspective/LetterBox ⇒ 三模态几何**天然同步**，bbox 也只有一份。3ch/4ch 同理（单一张量）。**不存在模态间几何错位。**

**⇒ brief §7 的要求（单模态不得偷改 augmentation）已通过 `aug.hsv_*=0` 处理；残余 D3 已声明。**

---

## 6. IR-CLAHE Audit

`ir_encoding` 只有**一个**消费点：`base.py:333`（RGBID 分支）。

| 路径 | 是否读 `ir_encoding` |
|---|---|
| `base.py` **RGBID**（训练） | ✅ `base.py:333` |
| `base.py` **Infrared**（训练，单模态） | ❌ **硬编码 `np.percentile(im, (1.0, 99.0))`**（`base.py:293-294`） |
| `loaders.py` **RGBID**（评测/提交） | ✅ `loaders.py:656`（读 `self.ir_encoding`，该字段 `loaders.py:344` 已定义） |
| `loaders.py` **Infrared** | ❌ **硬编码 percentile**（`loaders.py:557-559`），尽管 `self.ir_encoding` **就在同一个类里可用** |
| `dataset.py` | 该文件整条 IR/深度逻辑**未被使用**（记忆：三份拷贝中的"未用"那份） |

**`scripts/train.py:638` 的硬防呆**：`if kwargs.get("use_simotm") == "RGBID" and "ir_encoding" not in kwargs: raise SystemExit` —— **只覆盖 RGBID**。

```text
SHARED-SOURCE BLOCKER #1  (M2a)
  文件: ultralytics/data/base.py            'Infrared' 分支 (~293-294)
        ultralytics/data/loaders.py          'Infrared' 分支 (~557-559)
  症状: `ir_encoding: clahe` 被静默忽略 ⇒ M2a 退化成 M2b，而目录名写着 clahe
  最小修复: 把这两处的硬编码 percentile 换成与 RGBID 分支相同的
            `ir_encoding = getattr(getattr(self, "hyp", None), "ir_encoding", "percentile")`
            二选一分派（loaders 侧改用已有的 `self.ir_encoding`）
  本轮处置: **不修改**，M2a 标 BLOCKED，不进入 Batch-1
```

---

## 7. Early-Fusion Architecture Audit

`_transfer_rgb_pretrained`（`ultralytics/models/yolo/detect/train.py:203-325`）的分派是**穷举式**的：

```
1) 目标==源 全键匹配        -> return 0（resume / 微调）
2) 存在 in_channels==5 的 stem 且无 1ch stem -> 5ch early fusion：RGB=pretrained, 其余=mean(R,G,B)
3) 恰好 1 个 3ch stem + 恰好 2 个 1ch stem   -> _remap_separate_stem（raise-on-mismatch）  ← D′
4) 无 1ch stem                              -> return 0  ← ★ 4ch/2ch 落在这里
5) 否则                                     -> RGBD concat_res 硬编码索引表 remap
```

**实测证据**（模型构造，无 forward）：

| model yaml | ch | intersect_dicts | 被丢弃的键 | `_transfer_rgb_pretrained` |
|---|---:|---:|---|---:|
| `yolo11m_modality3ch.yaml` | 3 | **643/649** | Detect cv3 ×6 | **0**（标准 load 已全量，正确） |
| `yolo11m_modality4ch.yaml` | 4 | 642/649 | **`model.0.conv.weight`** + cv3 ×6 | **0** ← ❌ |
| `yolo11m_modality2ch.yaml` | 2 | 642/649 | 同上 | **0** ← ❌ |
| `yolo11m_sepstem.yaml` (D′) | 5 | 61/661 | 大量 | **550**（remap 生效） |

```text
SHARED-SOURCE BLOCKER #2  (M4 / M5 / M6)
  文件: ultralytics/models/yolo/detect/train.py  _transfer_rgb_pretrained (~256-271)
  症状: 4ch / 2ch 单 stem 不匹配任何分支 ⇒ `if not any(in_channels == 1): return 0`
        **无日志、无 raise**；而 `model.load()` 已因 shape 不符丢掉 `model.0.conv.weight`
        ⇒ 首个 Conv **保持 kaiming 随机初始化**，其余 642 个张量全预训练。
        （只有 1 个键被丢 ⇒ 在日志里几乎不可能被发现。）
  最小修复（推荐）: 把 5ch 分支的条件从 `in_channels == 5` 放宽为 `in_channels > 3`
        （或改名为 multi_ch），n_aux = in_channels - 3，仍用
        `five_ch.conv.weight[:, 3:].copy_(w.mean(dim=1, keepdim=True).repeat(1, n_aux, 1, 1))`。
        ⚠ 该分支同时要求 `not any(in_channels == 1)`，对单 stem 4ch 模型成立。
  备选（更贴近 D′ 的"每模态一个 stem"哲学、但改动更大）: 让 sepstem 分派接受
        `len(_three)==1 and len(_ones) in (1,)`，否则**现在的代码会把它误送进
        RGBD concat_res 硬编码索引表**（比随机初始化更糟：看起来像成功了）。
  本轮处置: **不修改**，M4/M5/M6 标 BLOCKED，仅留配置
```

**M6 额外 blocker**：数据侧根本没有 `(IR+D)` 的 merge 分支（`_merge_channels*` 只有 4 种），需要新增一个 `IRD` 模式。

---

## 8. Official Evaluator Audit

`scripts/official_eval.py`（PDF §六.2/3/4/5 + §九(二)）是**模态无关**的：它只吃
`--results`（预测 txt）、`--images`（**仅用于枚举 stem**）、`--labels`、`--max-boxes-per-image 100`、`--metric {METRIC_A,METRIC_B}`。
⇒ **全部 M1–M6 可复用同一评测器，无需新建 metric。**

`scripts/predict_rect.py` 从 `--train_config` 读取 `use_simotm / channels / pairs_rgb_ir / pairs_rgb_depth / ir_encoding`
并传给 `LoadImagesAndVideos`（`loaders.py`），后者对每个模式都有对应分支 ⇒ 推理链同样模态泛化。

> ⚠ 与训练侧**同样**的缺口：`loaders.py` 的 `Infrared` 分支也硬编码 percentile ⇒ 即使 M2a 在训练侧被修好，**推理侧不修则输入分布不一致**。两处必须一起改。

> ⚠ `official_eval.py` 默认 `--images data/processed/rgbid_split/images/val/visible`（注意是 `rgbid_split` 而非 `rgbid_split_train`）。实测两者划分完全相同，但**显式传参**以免误用。

> ⚠ 结论一律用 official 口径。本地 **ultralytics val mAP 不作为跨配置判据**（记忆：官方口径 4 次错符号、fork 4/4 对符号）。

---

## 9. Apples-to-Apples Fairness Table

| Item | D′ (M7) | M1 RGB | M2a IR-CLAHE | M2b IR-pct | M3 Depth | M4 RGB+IR | M5 RGB+D | Status |
|---|---|---|---|---|---|---|---|---|
| dataset base | `configs/dataset.yaml` | 同 | 同 | 同 | 同 | 同 | 同 | ✅ |
| split root | `rgbid_split_train` | `visible_split` | `rgbt_split` | 同 M2a | `depth_split_train` | `rgbt_split` | `depth_split_train` | ⚠ 目录不同但**划分逐位相同**（实测） |
| labels | `labels/{train,val}/visible` | 同（visible_split 内 flat） | 同 | 同 | 同 | 同 | 同 | ✅ 标签 modality-independent |
| imgsz | 1280 | 1280 | 1280 | 1280 | 1280 | 1280 | 1280 | ✅ |
| epochs / patience | 300 / 0 | 同 | 同 | 同 | 同 | 同 | 同 | ✅ |
| batch / accum / AMP | 8 / 8 / true | 同 | 同 | 同 | 同 | 同 | 同 | ✅ |
| optimizer / lr0 / lrf / mom / wd | SGD / 5e-3 / 0.01 / .937 / 5e-4 | 同 | 同 | 同 | 同 | 同 | 同 | ✅ |
| warmup | 3 ep（bias_lr .1, mom .8） | 同 | 同 | 同 | 同 | 同 | 同 | ✅ |
| augmentation（几何） | default | 同 | 同 | 同 | 同 | 同 | 同 | ✅ |
| augmentation（光度） | **inert** | `hsv_*=0` 已对齐 | 同 | 同 | 同 | 同 | 同 | ⚠ **D3 残留**：`Albumentations(p=1.0)` vs `(p=0)` |
| preprocessing | IR=CLAHE, D=/19999, RGB=BGR | 纯 BGR | CLAHE（**未生效→BLOCKED**） | percentile | =D′ 的 depth 公式 | RGB 原彩 + IR pct | RGB 原彩 + D | ⚠ **D4**：M2b 的 IR 编码 ≠ D′ |
| pretrained | `yolo11m.pt`（remap 550） | `yolo11m.pt`（load 643） | 同 M1 | 同 | 同 | **BLOCKED** | **BLOCKED** | ✅ 除被 block 者 |
| input channels | 5 | 3 | 3 | 3 | 3 | 4 | 4 | ✅ 已三方校验 |
| rect_late | **无**（原版 D′） | 无 | 无 | 无 | 无 | 无 | 无 | ✅ D1 |
| evaluator | `official_eval.py` METRIC_A/B | 同 | 同 | 同 | 同 | 同 | 同 | ✅ |

---

## 10. Configuration Files Created

| ID | Config | Model YAML | Status | Caveat |
|---|---|---|---|---|
| M1 | `configs/train_modality_m1_rgb.yaml` | `configs/yolo11m_modality3ch.yaml` | **READY WITH CAVEAT** | `Gray2BGR` 分支首次启用；D3 残留 |
| M2a | `configs/train_modality_m2a_ir_clahe.yaml` | 同 | **BLOCKED** | `ir_encoding` 被静默忽略（Blocker #1） |
| M2b | `configs/train_modality_m2b_ir_percentile.yaml` | 同 | **READY WITH CAVEAT** | IR 编码 ≠ D′ 的 CLAHE（D4）；D3 残留 |
| M3 | `configs/train_modality_m3_depth.yaml` | 同 | **READY WITH CAVEAT** | aux-stem 初始化语义与 D′ 不同（D5）；D3 残留 |
| M4 | `configs/train_modality_m4_rgb_ir.yaml` | `configs/yolo11m_modality4ch.yaml` | **BLOCKED** | 4ch stem 静默随机初始化（Blocker #2） |
| M5 | `configs/train_modality_m5_rgb_depth.yaml` | 同 | **BLOCKED** | 同上 |
| M6 | `configs/train_modality_m6_ir_depth.yaml` | `configs/yolo11m_modality2ch.yaml` | **BLOCKED ×2** | 无 `IRD` 数据模式 + Blocker #2 |
| — | `configs/MODALITY_BASELINE_README.md` | — | — | 使用说明与差异清单 |

---

## 11. CPU Validation

`diagnostic/fusion_phase0_validate.py`（无 forward / 无 GPU）：

| 检查 | 结果 |
|---|---|
| 7 个 train config YAML 解析 | ✅ 全部通过 |
| 必填字段完整性 | ✅ 0 缺失 |
| canonical training block 逐项 == D′ | ✅ 7/7 OK（epochs 300 / batch 8 / lr 5e-3 / warmup 3 / imgsz 1280 / seed 42 / patience 0 / val_ratio 0.2 / SGD .937 / wd 5e-4 / cosine / amp / yolo11m.pt） |
| 数据集路径存在 | ✅ 全部存在（1600 train / 400 val） |
| `channels`(config) == `ch`(model yaml) == 模式实际产出 | ✅ 除 M6（模式不存在，故意暴露） |
| `_transfer_rgb_pretrained` 分派预演 | ✅ 5ch→remap / 3ch→0（正确）/ 4ch,2ch→**0（❌ 静默）** |
| 模型**构造**（无 forward） | ✅ 3ch = 20,062,260 params；4ch/2ch stem `(0, 4→64)` / `(0, 2→64)` |
| pretrained 键匹配 | ✅ 3ch 643/649；4ch 642/649 且**恰丢 `model.0.conv.weight`**；D′ remap 550 |
| scale 由文件名判定 | ✅ `yolo11m_modality*.yaml` → scale **m**（对比：`yolo11_visible.yaml` → None ⇒ 静默回退 nano） |

**本机无法判定、开训前需人工确认（2 项）**：
1. 云环境是否安装 `albumentations`（决定 D3 是否真的生效）。
2. 云上首次运行 `Gray2BGR` 时的实读通道数（本机已静态确认分支体为纯 BGR）。

---

## 12. Git / D′ Integrity

| 检查 | 结果 |
|---|---|
| `configs/train_rgbid_sepstem_clahe.yaml` sha256 | **`a4e329cf…86e12` 未变** |
| `configs/yolo11m_sepstem.yaml` sha256 | **`9b14f294…3fdd8` 未变** |
| `git status -- ultralytics data` | **空** ⇒ 共享源码与数据集**零改动** |
| D′ weights | `best.pt` 40,641,641 B / 2026-09-22 07:33，**未被打开写入** |
| 本轮新增 | 11 个 config + `diagnostic/fusion_phase0_{validate.py,validation.json}` + 本报告 |
| commit | **未提交**（brief 要求） |
| 恢复既有实验 | 无删除 / 无覆盖 |

```text
SHARED-SOURCE BLOCKER（汇总，均**未动手**）
  #1  base.py / loaders.py 的 'Infrared' 分支不读 ir_encoding          → 阻塞 M2a
  #2  _transfer_rgb_pretrained 无 4ch/2ch 分支（静默 return 0）         → 阻塞 M4 / M5 / M6
  #3  base.py 无 (IR+Depth) 的 merge 分支                              → 阻塞 M6
```

---

## 13. GPU Batch-1 Recommendation

**最多 3 个，且只推荐已 READY 的：**

| # | 实验 | 配置 | 为什么值得跑 |
|---|---|---|---|
| 1 | **M1 RGB-only** | `yolo11m_modality3ch.yaml` + `train_modality_m1_rgb.yaml` | 目前**没有任何可信的 RGB 单模态数字**——历史 "RGB baseline" 全是 `SimOTMBBS`（灰度+模糊）或从零 RGB 分支。这是"融合到底有没有增益"的绝对上界。 |
| 2 | **M3 Depth-only** | 同上 + `train_modality_m3_depth.yaml` | 主线最强的互补模态；且 depth 预处理与 D′ **逐字相同**，是三者中与 D′ 可比性最高的。 |
| 3 | **M2b IR-only (percentile)** | 同上 + `train_modality_m2b_ir_percentile.yaml` | IR 是 CLAHE 增益的归属对象。⚠ 因 Blocker #1 暂时只能用 percentile，结论必须带上 D4 声明。 |

**不推荐训练 M2a / M4 / M5 / M6**（BLOCKED）。

**开跑前必须做**：① 确认云上 `albumentations` 是否安装；② M1 用 1–2 个 epoch 的 smoke run 确认 `Gray2BGR` 读到的是 3 通道原彩（而非某条兜底路径）。

---

## 14. GPU Batch-2 Conditional Plan

**只有在 Batch-1 产出 official 数字之后才决定。** 决策树：

```
令 R = official(M1 RGB), D = official(M3 Depth), I = official(M2b IR)

情形 A: R 已经 ≥ 0.515281·0.97        → 融合没带来增益 ⇒ 直接进入 Fusion Architecture Phase
                                          （问题变成"为什么 early fusion 吃掉了单模态的能力"）
情形 B: max(R,D,I) 明显低于 R 且 R 本身远低于预期
                                       → 单模态能力不足是主因 ⇒ 优先补 M2a（修 Blocker #1），
                                          并把 CLAHE 的归属问题查清，而不是改融合结构
情形 C: R 与 D′ 接近、D 或 I 中有一个意外强
                                       → 才值得解 Blocker #2 去跑 M4/M5，做"去一模态"的边际分析

⚠ 无论哪种情形：
   · 不推荐 mid/late fusion（brief §9 明令；且 D′ 已证 mid-fusion 系列未超 D′）
   · 不推荐在这一阶段引入 attention / gated / transformer / new neck
   · M4/M5 需要先把 Blocker #2 修掉 —— 而"修不修"取决于 Batch-1 的结论，不是现在
```

---

## FINAL VERDICT

```text
B — READY WITH CAVEATS
```

* 单模态比较的**数据划分 / 标签 / 评测器 / 通道序 / 训练预算**四块经实测**完全可比**；
* **M1 / M2b / M3 可以立刻开跑**，不需改任何共享源码；
* **M2a**（IR+CLAHE）与 **M4 / M5 / M6**（双模态早期融合）被 **3 个静默降级型共享源码缺口**阻塞，已给出最小修复方案，**本轮未动手**；
* D′ 零改动、零污染，无 commit。

---

## Decision Memo

```text
1. 能否公平测 RGB-only?   能。M1 READY WITH CAVEAT（Gray2BGR 首次启用 + D3 残留）
2. 能否公平测 IR-only?    部分能。M2b(percentile) READY；M2a(CLAHE) BLOCKED —— 这是最重要的缺口，
                          因为它正是 D′ 实际使用的 IR 编码
3. 能否公平测 Depth-only? 能。M3 READY WITH CAVEAT（aux-stem 初始化语义需声明）
4. 能否公平测 RGB+IR / RGB+Depth? 数据侧能，模型侧不能 —— 4ch stem 会静默随机初始化
5. 第一批跑哪几个?        M1 RGB / M3 Depth / M2b IR-percentile（且先做 albumentations + 通道数 smoke）
6. 什么会触发 mid/late fusion? 只有 Batch-1 显示"单模态本身有明确能力，而 D′ 的 early fusion
                          没能把它兑现"时 —— 即情形 A/C。在此之前不进入融合架构阶段。
```

### 最重要的一句

**这轮最值钱的发现不是"配置写好了"，而是两条静默降级**：
`Infrared` 分支不读 `ir_encoding`（会让 IR-CLAHE 悄悄变成 IR-percentile），
`_transfer_rgb_pretrained` 没有 4ch/2ch 分支（会让 4ch stem 悄悄留在随机初始化）。
**两条都不会报错、都不会进日志。如果在修好之前开跑，得到的将是一个看起来很正常的、但完全不可比的数字。**
