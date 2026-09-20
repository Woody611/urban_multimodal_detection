# D' = IR-CLAHE + Separate-Stem — TRAIN ENTRY DRY-RUN（Phase 1）

**日期**：2026-09-19 · **性质**：dry-run（1 个真实 batch 后立即停止；**未启动 300 epoch 训练**）

## 判决

```text
TRAIN_ENTRY_READY
```

---

## 0. 实验身份与命令

```text
D  (基线) = IR-CLAHE + Early Fusion        线上 0.5553
D' (候选) = IR-CLAHE + Separate-Stem       ← 本次
```

```bash
python scripts/train.py \
  --model_config configs/yolo11m_sepstem.yaml \
  --train_config configs/train_rgbid_sepstem_clahe.yaml
```

**关于 `--train_config` 的一处必要偏离（请确认）**：
brief §16 字面写 `configs/train_rgbid_sepstem.yaml`，但该文件是 **percentile** IR（我为上一轮融合实验建的）。
brief §3 硬性要求候选必须建立在 **D 的 IR-CLAHE** 上，且明确「禁止训练 Separate-Stem + 原始 IR」。
两条冲突时 §3 更具体，故本轮新建 **`configs/train_rgbid_sepstem_clahe.yaml`** ——
它是 **D 的 `configs/train_rgbid_ir_clahe.yaml` 的逐行副本，唯一改动是 `experiment_name`**，
符合 §16 第 4 条「只替换 Separate-Stem model config / experiment name」。

---

## A. 配置（在真实 trainer 内打印）

| 项 | 值 | 判定 |
|---|---|---|
| scale | **`m`** | ✅ 非 nano |
| ch / nc | **5 / 12** | ✅ |
| params | **20,061,972**（≈20.06M） | ✅ |
| epochs / patience | 300 / **0** | ✅ 与 D 一致（D 禁用早停） |
| batch / imgsz | 8 / 1280 | ✅ |
| pretrained | `yolo11m.pt` | ✅ |
| lr0 / momentum / wd | 0.005 / 0.937 / 5.0e-4 | ✅ |
| cos_lr / warmup / seed / deterministic | True / 3.0 / 42 / True | ✅ |
| close_mosaic / box / cls / dfl / nbs | 10 / 7.5 / 0.5 / 1.5 / 64 | ✅ |
| aug（hsv_h/s/v, mosaic, mixup, copy_paste, flipud, fliplr, erasing） | 0.015/0.7/0.4, 1.0, 0.0, 0.0, 0.0, 0.5, 0.4 | ✅ 全部 ultralytics 默认，D/D' 逐项相同 |
| **ir_encoding** | **`clahe`** | ✅ |
| device | `cpu` | ⚠️ 见 §J 声明的运行时覆盖 |

**独立审计（workflow，4 个审计员 + 对抗性复核）**：对照 D 的落盘 `args.yaml` 逐 key 核验，
**76 项检查，7 项被标 UNEXPECTED，对抗性复核后确认 0 项**。
D vs D' 训练配置**只有 `experiment_name` 一个 key 不同，其余 28/28 逐字相同**。

被标但被推翻的 7 项（全部与 D' 无关或预先存在）：D 头部注释陈旧（`patience=80` 与 `patience: 0` 自相矛盾）、
D 头部注释里过时的命令示例、4 处**未触发**的 silent-fallback 隐患、
`dataset.py` 里第三条 IR 拷贝（percentile-only，**不在 D/D' 路径上**）、
以及既有的 `predict_rect.py` 冻结漂移。

---

## B. IR-CLAHE pipeline —— 逐像素证明已生效

| 证据 | 结果 |
|---|---|
| trainer args 中的 `ir_encoding` | **`clahe`** |
| 训练数据集对象的 `ds.hyp.ir_encoding` | **`'clahe'`** |
| 代码路径 | `base.py:333` `if ir_encoding == "clahe": cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(...)` **else** percentile —— **互斥，非叠加** |
| 评测路径 | `loaders.py:657` 同一调用、同一参数、同一顺序、同一通道序 |

**决定性逐像素判定**（取真实训练样本，把数据集的 IR 通道与 8 种候选管线假设逐一比对）：

| 假设 | idx0 mean\|Δ\| | idx3 mean\|Δ\| |
|---|---:|---:|
| **H6 `CLAHE@原分辨率 → resize LIN`** | **0.0000** | **0.0000** |
| H2 `percentile→CLAHE → resize` | 15.02 | 15.66 |
| H1 `percentile → resize` | 15.56 | 14.92 |
| H5 `仅 resize，无对比度处理` | 13.78 | 13.76 |

**对照组**：visible 三通道 vs `resize INTER_LINEAR` → mean\|Δ\| = **0.000**（证明我的复刻本身正确）。

> **结论：CLACHE 确实生效，且与 D 使用完全相同的实现（逐位相同）。**
> 上一轮 dry-run 里那条"matches CLAHE: False"是我**把 percentile 与 CLAHE 串起来算**导致的
> 构造错误 —— 二者是**互斥**的。已用 8 假设对照 + visible 对照组定位并改正。

---

## C. 真实 forward graph

| 分支 | 切片 | Conv | 输出 shape（实测） |
|---|---|---|---|
| RGB | ch `[0:3]` | 3→48 | `(1,48,64,64)` |
| IR | ch `[3:4]` | 1→8 | `(1,8,64,64)` |
| Depth | ch `[4:5]` | 1→8 | `(1,8,64,64)` |
| **Concat** | — | — | `(1,64,64,64)` |

`concat == 64 channels` ✅ ｜ `下一层 conv.in_channels == 64` ✅

IR / Depth 的身份由**喂给该 stem 的 `SilenceChannel.c_start`** 判定（3=IR, 4=Depth），
非模块位置猜测。

---

## D. Pretrained remap 审计

| 项 | 结果 |
|---|---|
| backbone exact-match | **245 / 245** |
| head exact-match | **287 / 287** |
| 合计 | **532 / 538 = 98.9%**（与 D 完全相同） |
| 合法例外 | Detect `.cv3.` 分类分支 **6 个** tensor（源 COCO `nc=80` → 本数据集 `nc=12`，必然重建） |
| remap 日志 | `550 tensors transferred ... Detect cv3 class branch re-initialised for nc=12: 6 tensors not transferred` |

> `load()` 打印的 `Transferred 61/661` **不作为判据**（只反映按名匹配）。判据是上表的 structure-corresponding 比对。

---

## E. 三个 stem 初始化（逐位验证）

| stem | 规则 | `torch.equal` |
|---|---|---|
| RGB（3→48） | `stock_stem.weight[:48]` + BN `[:48]` | **True**（max\|Δ\|=0.000e+00） |
| IR（1→8） | `stock[:8].float().mean(dim=1, keepdim=True)` | **True**（max\|Δ\|=0.000e+00） |
| Depth（1→8） | 同 IR，独立执行 | **True**（max\|Δ\|=0.000e+00） |

---

## F/G. Runtime + gradient dependency

| 检查 | RGB | IR | Depth |
|---|---|---|---|
| 合成扰动 `max|Δoutput|` | 1.163e+02 | 8.470e+01 | 1.026e+02 |
| **真实 batch 梯度** `|g|sum` | **3877.65** | **721.21** | **485.04** |
| 梯度非 None / 有限 / 非 0 | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ | ✅ / ✅ / ✅ |

（合成扰动取自 `scripts/verify_sepstem_pretrained.py`，本轮复跑 **35/35 checks passed**；
真实 batch 梯度取自 `scripts/train.py` 正式入口的 backward hook。）

---

## H/I. 真实训练入口 + 一个真实 batch

**走的是 `scripts/train.py::main()`**，不是 helper：
真实参数解析 → 真实 dataset 切分 → 真实 `YOLO(model_config)` → 真实 `model.train()` →
真实 `DetectionTrainer` → 真实 `get_model()` → 真实 `_transfer_rgb_pretrained` → 真实 DataLoader。

| 项 | 值 |
|---|---|
| **真实 batch 输入 shape** | **`(8, 5, 1280, 1280)`** —— 真实 5 通道 RGBID 样本 |
| 数据 | `rgbid_split_train`（1600 train / 400 val；`_split_train_val_rgbid` 幂等命中，**未重写数据**） |
| forward + backward | ✅ 1 次 |
| **`optimizer.step` 次数** | **1** |
| 该步 per-group lr | `[0.1, 0.0, 0.0]` |
| momentum buffer \|sum\| | rgb 55.52 / ir 10.33 / depth 6.95（**三个均非零**） |

**三个 stem 是否实际发生参数变化**：

```
rgb    bn.bias  changed=True  max|delta|=4.043579e-04
ir     bn.bias  changed=True  max|delta|=4.198030e-04
depth  bn.bias  changed=True  max|delta|=2.057552e-04
any param of each stem changed : {'rgb': True, 'ir': True, 'depth': True}
```

> `conv.weight` / `bn.weight` 在第 1 步未变，是 ultralytics warmup 的**既定设计**：
> `trainer.py:373` 的 `warmup_bias_lr if j == 0 else 0.0` 只给 **param_groups[0]（偏置组）**
> 非零 lr，卷积累与 BN weight 组该步 lr=0.0（实测 `[0.1,0,0]`）。这对**任何模型**（含 D）都一样。
> 证明"stem 真正参与优化"的正确证据 = 偏置组实际更新 + momentum buffer 非零，两者均已满足。

**是否超过 1 个 batch / 1 次 optimizer.step**：**否 / 否**（batches=1, steps=1）。

---

## J. 无污染 + 声明的运行时覆盖

| 项 | 结果 |
|---|---|
| `runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/` | **不存在**（无冲突，可安全开训） |
| D 的资产 | **未触碰**：`best.pt` sha256 `3c2f942c…ffa3`、`last.pt` `842ba1b1…54c0`、`results.csv` `95a2f2bf…ebd6` |
| `OFFICIAL_FALLBACK` best.pt | sha256 `f9dddbfa…3e55` **与要求完全一致，未变** |
| `FREEZE_MANIFEST` | **16/17 通过**（唯一不符项为**既有**的 `predict_rect.py` 漂移，非本次造成） |
| dry-run 输出位置 | `%TEMP%\rgbid_sepstem_clahe_dryrun\run\` |

**仅在 `%TEMP%` 脚本内做的两处运行时覆盖（不涉及任何项目文件）**：
1. `device → "cpu"`：本机无 GPU；训练配置本身仍是 `device: cuda / gpu_ids: [0]`，云端执行无需此覆盖。
2. `project/name → %TEMP%`：满足 brief §14「不得污染正式 runs」。

---

## D 的训练内 val 参照（供训练后对照，非判据）

```text
D (IR-CLAHE) best ep=263  mAP50-95=0.56806  mAP50=0.81879  mAP75=0.57624  P=0.87429  R=0.72746
D final ep=300            mAP50-95=0.55612
```

---

## 结论

```text
TRAIN_ENTRY_READY
```

A–J 全部通过。D vs D' 的差异经独立审计确认为**恰好两项**：`model_config`（fusion architecture）
与 `experiment_name`。IR-CLAHE 预处理经逐像素证明与 D **逐位相同**。

**到此停止。等待你的指令后才启动正式训练。**

```text
OFFICIAL_FALLBACK = 0.53180（未变更）
D  online = 0.5553（未变更）
DECISION: TRAIN_ENTRY_READY
```
