# RGBID Separate-Stem Fusion — Pretrained Remap Fix + Verification

**日期**：2026-09-19
**范围**：修复 Separate-Stem 候选的 pretrained 迁移，并做只读验证。**未训练、未创建 run、未改 baseline。**
**验证脚本**：`scripts/verify_sepstem_pretrained.py`（只读；退出码 0 = 全部通过）
**最终判决**：**`REMAP_READY_FOR_TRAINING`**（35/35 checks passed，exit 0）

---

## 1. 上一轮的问题（已修复）

`ultralytics/models/yolo/detect/train.py::_transfer_rgb_pretrained` 原有**两条硬编码分支**，
按 stem 的 `in_channels` 分布判定：

| 布局 | 原判定 | 原处理 |
|---|---|---|
| `[5]` RGBID 早期融合 | 有 `in=5` 且无 `in=1` | 只补首层 Conv |
| `[3,1]` RGBD 中期融合 | 有 `in=1` | 硬编码 `pairs` 索引表 + 首个 1ch stem 均值初始化 |
| **`[3,1,1]` 三-stem（本候选）** | 无 `in=5` → 跳过分支1；有 `in=1` → **误入分支2** | `pairs` 表完全不对应；`next(in==1)` 只取第一个 1ch stem |

**实测后果（上一轮）**：候选的 backbone/head 只有 **1/538 = 0.2%** 真正等于预训练，
即"传了 `pretrained: yolo11m.pt` 却实际从零训练"，且**不报错**。

---

## 2. 变更清单（严格限定在允许范围内）

| # | 文件 | 状态 | 目的 |
|---|---|---|---|
| 1 | `ultralytics/models/yolo/detect/train.py` | **修改** | 新增 `_remap_separate_stem()`；在 `_transfer_rgb_pretrained` 内新增**第三条 dispatch 分支** |
| 2 | `configs/yolo11m_sepstem.yaml` | 新建 | 候选模型结构（3 stem + Concat + 原 YOLO11m） |
| 3 | `configs/train_rgbid_sepstem.yaml` | 新建 | 官方 RGBID train config 的逐行副本，仅改 `experiment_name` |
| 4 | `scripts/verify_sepstem_pretrained.py` | 新建 | 只读验证（本报告全部运行期证据的来源） |

**未触碰**：`base.py` / `loaders.py` / dataset / augmentation / loss / optimizer /
official evaluation / `predict_rect.py` / baseline 配置 / 任何 checkpoint / `OFFICIAL_FALLBACK`。
**未训练**：无 `model.train()` / `trainer.train()` / `scripts/train.py` 调用；`runs/` 下无新目录。

---

## 3. 修复 model scale

```text
guess_model_scale("configs/yolo11m_sepstem.yaml") == 'm'   ✅
params == 20,061,972（≈ 20.06M，YOLO11m）                   ✅（非 2.59M nano）
```

上一轮的教训（用内存 dict 构建 → `Assuming scale='n'` → 静默 nano）已通过**落地真实 yaml 文件**
解决。文件名 `yolo11m_sepstem.yaml` 命中正则 `yolo[v]?\d+([nslmx])`。

---

## 4. 修复 Pretrained Remap — 显式、可验证、无静默 fallback

### 4.1 dispatch 顺序（关键）

新分支必须**先于** RGBD 分支判定，否则会继续被后者吞掉：

```python
# ---- Separate-Stem fusion (exactly one 3ch stem + exactly two 1ch stems) ----
_three = [m for m in stems if m.conv.in_channels == 3]
_ones  = [m for m in stems if m.conv.in_channels == 1]
if len(_three) == 1 and len(_ones) == 2:
    return _remap_separate_stem(model, src, _three[0], _ones)
```

### 4.2 三个 stem 的确定性初始化规则（写死在代码里，逐 tensor 验证）

| stem | in | out | 规则 |
|---|---:|---:|---|
| **RGB** | 3 | 48 | `stock_stem.conv.weight[:48]`；BN 同取 `[:48]`（**输出通道前缀**） |
| **IR** | 1 | 8 | `stock_stem.conv.weight[:8].float().mean(dim=1, keepdim=True)` → 灰度归约后再取输出前缀；BN 取 `[:8]` |
| **Depth** | 1 | 8 | 与 IR **同规则、独立执行** |

**IR / Depth 的身份判定不靠模块位置**：读取**喂给该 stem 的 `SilenceChannel` 的 `c_start`**
（3 → infrared，4 → depth），从 model yaml 的 `from` 字段解析。取不到就 `raise`。

**灰度归约强制在 float32 计算**再转回参数 dtype：`yolo11m.pt` 存的是 fp16，
若直接在其原生 dtype 上求均值，结果会带 ~4e-5 的 fp16 舍入误差、不可复现。
（这是本轮修出来的第二个问题，见 §6.3。）

### 4.3 backbone / head 映射

```text
offset = 融合 Concat 在 yaml 中的层号（本模型 = 7，从 yaml 推导，非硬编码）
candidate layer i (i > offset)  <->  stock layer (i - offset)
```

对候选尾部**每一个 ndim>0 的 tensor**，逐项要求 stock 侧存在**同名同形状**的对应张量；
任何缺失或形状不符 → `raise RuntimeError`。

### 4.4 唯一的合法例外（显式声明，非静默跳过）

| 例外 | 原因 | 约束 |
|---|---|---|
| Detect 的 **`.cv3.`（分类分支）** | 源 ckpt 是 COCO `nc=80`，本数据集 `nc=12` → 分类头必然重建（与 stock ultralytics 对任何新 nc 微调的行为一致） | 仅当 `i == detect_idx` 且 `.cv3.` 且在 k 且 `sv.shape[1:] == v.shape[1:]` 且 `v.shape[0] == nc` 且 `sv.shape[0] != nc` 才放行；**每一项都记入日志** |

实测被跳过的正是 **6 个** tensor（3 个尺度 × weight/bias），并打印：

```text
Separate-stem pretrained remap: 550 tensors transferred
(3 stems + layers 8..30 <- stock 1..23);
Detect cv3 class branch re-initialised for nc=12: 6 tensors not transferred
```

> **无 silent fallback**：除上述声明过的分类头外，任何找不到 / 形状不符都会 `raise`，
> 不会 `logger.info` 后 `continue`。本轮开发中该 `raise` **确实触发了**（见 §6.3），证明其有效。

---

## 5. 验收清单（brief §14 的 24 项）

| # | 条件 | 结果 | 证据 |
|---|---|---|---|
| 1 | scale == `m` | ✅ | `got 'm'` |
| 2 | params ≈ 20.06M | ✅ | `20,061,972` |
| 3 | input ch == 5 | ✅ | `ch=5` |
| 4 | nc == 12 | ✅ | `Detect nc=12` |
| 5 | RGB stem init 显式验证 | ✅ | `== stock[:48]`，`max\|Δ\|=0.000e+00` |
| 6 | IR stem init 显式验证 | ✅ | `== mean(RGB)[:8]`，`max\|Δ\|=0.000e+00` |
| 7 | Depth stem init 显式验证 | ✅ | `== mean(RGB)[:8]`，`max\|Δ\|=0.000e+00` |
| 8 | backbone exact-match ≥ 98% | ✅ | **245/245 = 100.0%** |
| 9 | head exact-match ≥ 98% | ✅ | **287/287 = 100.0%** |
| 10 | 无 silent fallback | ✅ | raise-on-mismatch 已在开发中实际触发并拦截 |
| 11 | DetectionTrainer 真实加载路径已验证 | ✅ | 走 `DetectionTrainer.get_model()`，非手工调用 |
| 12 | RGB runtime dependency > 0 | ✅ | `max\|Δ\|=1.163e+02` |
| 13 | IR runtime dependency > 0 | ✅ | `max\|Δ\|=8.470e+01` |
| 14 | Depth runtime dependency > 0 | ✅ | `max\|Δ\|=1.026e+02` |
| 15 | RGB gradient > 0 | ✅ | `\|g\|sum=5.344e+06` |
| 16 | IR gradient > 0 | ✅ | `\|g\|sum=8.765e+05` |
| 17 | Depth gradient > 0 | ✅ | `\|g\|sum=1.091e+06` |
| 18 | concat == 64 channels | ✅ | 输出 `(1,64,64,64)`；下一层 `conv.in_channels==64` |
| 19 | save/reload 保持 Separate-Stem | ✅ | reload 后 31 层、`ch=5`、layer0 Identity、layer1 SilenceChannel、3 stems、params 相同 |
| 20 | 未修改 baseline | ✅ | 见 §7 回归测试 |
| 21 | 未修改 dataset | ✅ | `git status` 无 `data/` 变更 |
| 22 | 未修改 augmentation | ✅ | `augment.py` 未触碰 |
| 23 | 未修改 evaluation | ✅ | `predict_rect.py` / `official_map.py` 未触碰 |
| 24 | 未执行训练 | ✅ | 无 train 调用；`runs/` 无新目录 |

```text
RESULT: 35/35 checks passed      （24 项验收 + 11 项过程检查）
REMAP_READY_FOR_TRAINING
```

---

## 6. 运行期证据

### 6.1 事前 / 事后（同一 metric）

| 模型 | `load()` 报告 | remap 返回 | **结构对应张量真正等于预训练** |
|---|---:|---:|---:|
| Baseline（现官方 `ch:5`） | 642/649 | 5 | **532/538 = 98.9%** |
| Candidate **修复前** | 61/661 | 9 | **1/538 = 0.2%** ❌ |
| Candidate **修复后** | 61/661 | **550** | **532/538 = 98.9%** ✅ |

> ⚠️ **`load()` 自己报的 "Transferred 61/661" 在修复后仍然是 61/661** —— 因为它只反映
> `intersect_dicts` 的按名匹配，不包含我们的 remap。**该数字不能用作验收指标**，
> 唯一可信的是"结构对应张量是否真的等于预训练"。这一点已写入记忆。

逐层明细（修复后）：stock L1–L23 → candidate L8–L30，**每一有参数的层都是满额**
（5/5、45/45、45/45、5/5 … 97/97）。`0/0` 的层是 SilenceChannel / Concat / Upsample 等**无参数层**，
本就无物可迁。

### 6.2 候选模型结构（实测）

```text
scale            : m
input channels   : 5
nc               : 12
layers           : 31（420 sublayers）
params           : 20,061,972
GFLOPs           : 68.17 @640  /  272.68 @1280
baseline 对照    : 20,063,412  /  273.85 @1280   → Δparams −1,440 (−0.0072%)
```

### 6.3 本轮开发中 `raise` 真实触发的两次（证明"无静默 fallback"不是空话）

1. **Detect 分类头**：首次运行时 raise
   `shape mismatch for 'model.30.cv3.0.2.weight': candidate (12,256,1,1) vs stock (80,256,1,1)`。
   这暴露了"分类头因 nc 不同必须重建"这一**合法例外**——遂改为**显式声明 + 形状断言 + 日志**，
   而不是 `continue` 掉。
2. **fp16 均值**：IR/Depth stem 的 `torch.equal` 比对出现 `max\|Δ\|=4.07e-05`。
   根因是灰度均值在 ckpt 原生 **fp16** 下计算。已改为 **float32 计算后再转回**，
   现在 `max\|Δ\| = 0.000e+00`（逐位相等）。

---

## 7. Baseline 回归测试（防止共享代码被改坏）

`_transfer_rgb_pretrained` 是**共享函数**，新增分支不得影响既有两条路径：

| 既有布局 | 检查 | 结果 |
|---|---|---|
| `[5]` RGBID 早期融合（现官方） | 仍返回 5 | ✅ `returned 5` |
| 同上 | body 结构对应 exact-match | ✅ **532/532 = 100.0%** |
| `[3,1]` RGBD 中期融合 | 新分支判定为 **False**（`three=1 ones=1`） | ✅ 证明其代码路径逐行未变 |
| 同上 | 仍正常 remap | ✅ `returned 537` |

> 第三行是关键：新分支的前提条件 `len(three)==1 and len(ones)==2` 对 RGBD 布局为**假**，
> 因此该布局执行的仍是**修改前的那几行代码**，行为不可能改变。

---

## 8. 结论

```text
REMAP_READY_FOR_TRAINING
```

候选模型经**正式训练的加载路径**构建后，backbone/head 的预训练保留度与现官方 baseline **完全一致**
（同为 532/538 = 98.9%），三个 stem 按确定性规则初始化且逐位可验证，
唯一的非迁移项（Detect 分类头）已被显式声明并受形状断言约束。

**开训前的唯一剩余动作**（不在本阶段范围内）：
```bash
python scripts/train.py --model_config configs/yolo11m_sepstem.yaml \
    --train_config configs/train_rgbid_sepstem.yaml
```

**本阶段到此停止，未启动训练。**

```text
OFFICIAL_FALLBACK = 0.53180（未变更）
DECISION: REMAP_READY_FOR_TRAINING
```
