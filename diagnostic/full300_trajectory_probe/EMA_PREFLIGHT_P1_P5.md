# EMA FULL-300 PROBE — PREFLIGHT P1–P5

```text
EMA FULL-300 PROBE STATUS: INVALID
```

> 判定依据 §三：**P5 = FAIL** ⇒ `DO NOT START GPU RUN`。
> 另：本机 `torch 2.4.1+cpu`、`cuda_available=False`、无 `nvidia-smi` ⇒ 物理上也无法启动 GPU 训练。

---

## 0. 结论摘要（先说最重要的）

1. **P1 PASS** —— `trainer.ema.ema` 就是 val / best.pt / 部署所用的对象，代码路径 + 实证双证。
2. **P3 PASS** —— `CLS_OUT[c,y,x] == W_c·h + b_c`，真实权重 + 真实数据上 `max_residual = 8.26e-06`（FP16 模型，纯浮点舍入）。
3. **P5 FAIL** —— 旧采集路径存在**空间采样伪影**，且**已定位、已量化、已复现**：
   `medium_cells.json` 的 cell 坐标用 `label归一化中心 × 1280` 计算，**漏掉 letterbox 纵向 padding**
   （16:9 图进 1280² 方形画布时 pad=(0, 280)）。旧 probe 因此**一直采在错误的网格 cell 上**。
4. 用户给出的 h 量级矛盾（raw probe ≈27/29 vs EMA audit ≈40/51）**根因不是 raw-vs-EMA，也不是
   mode 伪影，就是这个采样坐标 bug**（详见 §5）。
5. 附带发现：上一轮 FULL300 **从未跑到 epoch 300**（csv 1..297），且 `final_eval()` 未执行 ⇒ §6 独立违规。

---

## 1. P1 — EMA object identity

```text
P1 EMA IDENTITY: PASS
```

### 代码路径证据

| 环节 | 位置 | 内容 |
|---|---|---|
| EMA 创建 | `ultralytics/engine/trainer.py:296` | `self.ema = ModelEMA(self.model)` |
| EMA 更新 | `ultralytics/engine/trainer.py:594-595` | `if self.ema: self.ema.update(self.model)` |
| **validation 用 EMA** | `ultralytics/engine/validator.py:118` | `model = trainer.ema.ema or trainer.model` |
| fitness 决定 best | `ultralytics/engine/trainer.py:609-610` | `best_fitness` 由 `self.validator(self)` 的 fitness 更新 |
| **checkpoint 存 EMA** | `ultralytics/engine/trainer.py:524-525` | `"model": None,  # resume and final checkpoints derive from EMA` ／ `"ema": deepcopy(self.ema.ema).half()` |
| best.pt 写出 | `ultralytics/engine/trainer.py:542-543` | `if self.best_fitness == self.fitness: self.best.write_bytes(...)` |
| **部署路径** | `ultralytics/utils/torch_utils.py:610-611` | `if x.get("ema"): x["model"] = x["ema"]  # replace model with EMA` |

⇒ val / best / 部署三条路都落在 `trainer.ema.ema` 上。

### 本机实证（D′ 部署权重）

```text
ckpt = runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt
  sha256        = 1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda
  ck["ema"]     = None          ← strip_optimizer 已把它置 None
  ck["model"]   = DetectionModel(nc=12)，全部 tensor dtype = float16（int64 除外）
  ck["epoch"]   = -1  ck["best_fitness"] = None   ← strip_optimizer 签名
```

`strip_optimizer` 的顺序是 `x["model"] = x["ema"]` → 然后 `x["ema"] = None`。
⇒ 文件里 `ema is None` 且 `model` 为 FP16，**只可能**意味着 `model` 对象来自 `ema`。

**P1 PASS。**

---

## 2. P2 — EMA forward semantics

```text
P2 EMA FORWARD SEMANTICS: PASS（本机可验部分）
```

本机无训练进程 ⇒ 无法在 `trainer` 上下文里比对。改为**可证伪的代理检验**：
用部署权重（= EMA）在真实 val 数据上前向，看能否**逐位复现**冻结参考（`_v7_vectors.json`）。

结果（用 dump 自己的坐标采样，5 个 key）：

| key | dump_y | ||h||@dump_cell | ||h||_dump | max\|diff\| | cos |
|---|---:|---:|---:|---:|---:|
| 00000045#1 | 604.0 | 25.7273 | 25.7273 | **0.00000** | **1.0000** |
| 00000045#2 | 637.0 | 55.1262 | 55.1262 | **0.00000** | **1.0000** |
| 00000045#3 | 772.0 | 48.8579 | 48.8579 | **0.00000** | **1.0000** |
| 00000045#6 | 386.0 | 45.1224 | 45.1224 | **0.00000** | **1.0000** |
| 00000045#7 | 687.0 | 42.7806 | 42.7806 | **0.00000** | **1.0000** |

⇒ 部署 EMA 前向在数值上**完全确定**，且与既有参考口径一致。

⚠ 未决项：真正意义上的 P2（**在训练进程内**证明 `trainer.ema.ema` 与 `BaseValidator` 所用模型同源）
必须在云端 in-process harness 里跑，本机给不了。云端 harness 已列入 §6 待办。

---

## 3. P3 — logit reconstruction

```text
P3 LOGIT IDENTITY: PASS
max_residual = 8.26e-06
```

本机真实数据（D′ 部署权重 + val 集，60 图 198 条 GT）：

```text
n = 198      max_residual = 8.259988e-06      median_residual = 2.019e-06
```

恒等式 `CLS_OUT[gt_c, y, x] == W_gt · h + b_gt`（`cv3[0][-1]` 是 1×1 conv ⇒ 应逐位相等）。
8.3e-06 的残差与 **FP16 权重 + FP32 累加**的舍入量级一致（相对误差 ~1e-6）。
旁证：`cls_representation_modality_audit` 的 C6 在 4960 行上给出 `max|resid| = 2.3e-06`。

⇒ 不是恒等式不成立，是数值精度。**PASS。**
（阈值：若残差 ≳1e-3 才判 FAIL；实测低两个数量级。）

---

## 4. P4 — probe 非侵入性

```text
P4 NON-INTRUSIVE PROBE: NOT RUN（本机无训练进程，不可伪造）
```

已有的 `_integrity_gate300.py` 覆盖 G1(forward 逐位)/G4(RNG)/G5(无梯度+参数)/G8(BN buffer 还原)，
**但它是为旧 `trainer.model` 钩子写的**，对新的 `trainer.ema.ema` 钩子**不构成证据**：

- hook 目标变了 ⇒ 需重新证明 forward 不变、参数不变；
- EMA 每步都在被 `ema.update()` 改写 ⇒ 「参数未变」必须在 **EMA 参数**上重新断言；
- 旧 probe 在 `observe()` 里做 `eval() + head.train()` 且快照/还原 BN ⇒ 新路径若改用纯 `eval()`
  则必须重新证明「forward output 未被观测改变」。

⇒ 云端 in-process harness 必做，未完成前 P4 = NOT RUN。

---

## 5. P5 — h magnitude collection sanity

```text
P5 H-COLLECTION SANITY: FAIL
```

### 5.1 现象

| 来源 | 口径 | h 范数 |
|---|---|---|
| 旧 FULL300 probe（raw 模型） | `medium_cells.json` 坐标 | ≈ **27 / 29** |
| `cls_representation_modality_audit`（EMA） | `_v7_vectors.json` 坐标 | G86 **40.38** / CONTROL **51.51** |

### 5.2 根因：采样坐标漏掉 letterbox padding

两边坐标来源不同：

- `_v7_extract.py:128-129`：`cxw,cyw,_,_ = lab["bboxes"][j]; cx,cy = cxw*1280, cyw*1280`
  —— `lab["bboxes"]` 是 **ultralytics dataset 输出**，**已经归一化到 letterbox 后的画布** ⇒ ×1280 正确。
- `_medium_cells.py:40-41`：`cxw, cyw = gtn[k]; cx, cy = cxw*1280.0, cyw*1280.0`
  —— `gtn` 是**标签文件里的归一化中心**，相对**原图** ⇒ ×1280 **漏掉了 pad**。

而 `_medium_cells.py:59` 自己写着 `cell_mapping="_v7_extract.py 口径：canvas = 归一化中心 × 1280"`
—— 意图是「对齐 `_v7_extract` 口径」，但两者读的 bbox 来源不同，**这个注释是错的**。

实测该图 letterbox：

```text
00000045.jpg 640x360 → gain=2.0 → 1280x720 → ratio_pad=((2.0,2.0),(0,280))
标签 gt#1 cy=0.450
  正确 canvas y = 0.450*360*2 + 280 = 604.0     ← _v7_extract dump 记录的 center
  medium_cells y = 0.450*1280      = 576.0     ← 旧 probe 实际采样的位置
```

### 5.3 量化（1320 个 medium cell 全量比对）

```text
matched keys                     : 1320 / 1320
|Δx|  median = 0.00   max = 0.01      ← x 方向无 pad，恰好正确
|Δy|  median = 76.74  max = 274.82    ← 单位：canvas px（stride-8 ⇒ 中位 9.6 cell，最大 34.4 cell）
坐标完全相同的 key              : 1 / 1320
```

### 5.4 决定性复现

用 **dump 自己的坐标**采样，即可**逐位**复现冻结参考（§2 表，`max|diff| = 0.00000`，`cos = 1.0000`）。
⇒ 权重、mode、预处理全部一致，**唯一差异就是采样坐标**。

### 5.5 该差异足以解释全部 h 量级矛盾

```text
_v7_vectors.json (V1, 正确 cell, EMA)：
  G86      median = 40.382   mean = 40.895   n = 86
  CONTROL  median = 51.510   mean = 52.170   n = 1070
  MISSED   median = 43.954   mean = 44.323   n = 164
  ALL      median = 45.945   mean = 44.847   n = 2807
```

**G86=40.4 / CONTROL=51.5 正是用户所说的「EMA audit h ≈ 40 / 51」。**

同一份 EMA 权重、同样用 `medium_cells` 的（错误）坐标采样时：

```text
full_eval  : ALL median 29.054 | G86 26.676 | CONTROL 29.313
eval+head.train : ALL median 27.866 | G86 25.869 | CONTROL 28.096
```

⇒ 错误坐标下 h≈27–29，与旧 probe 的「27/29」吻合。

**结论：h 量级矛盾由采样坐标 bug 解释，不需要引入 raw-vs-EMA 假设。**
附带量化：`head.train()` vs 纯 `eval()` 的 mode 效应只有 median **−4.1%**（29.05→27.87），
量级远小于坐标 bug，**不是**主因。

### 5.6 后果

旧 FULL300 / PROBE30 的全部空间采样量（`vec_p3`、`vec_h`、`logit`、以及所有基于
`p3_cell_x/p3_cell_y` 的 TAL 读数）**采在了错误的 cell 上**。旧 CASE B 因此**无效**——
与「hook 错对象」是**两个独立**的失效原因。

---

## 6. 启动 GPU run 前必须先完成的修正

1. **probe target** → `trainer.ema.ema`（`_run_probe300.py` 的 `attach_model(de_parallel(trainer.model))`
   与新 `on_fit_epoch_end` 里的 `de_parallel(trainer.model)` 都要改）。
2. **采样坐标** → 改用 **dataset bbox**（已 canvas-归一化），或显式 `× gain + pad`；
   并以 `_v7_vectors.json` 的 V1 行做**逐位回归测试**（本报告 §2 表即该测试）。
3. **mode** → §5 要求「与正常 validation/inference 相同的 model mode」。
   ⚠ 但既有参考口径（`_v7_vectors.json`）用的是 `eval backbone + head.train()`。
   改纯 `eval()` 后新轨迹**无法与旧参考逐位对比**（mode 效应 ~4%）。此为**必须显式裁决的冲突**，
   不能默默选一个。
4. **P4** → 为 EMA 钩子重写非侵入性 gate。
5. **§6 跑到 300** → 上一轮只到 csv 297；且 `final_eval()` 未执行（weights 80.9MB 未 strip）。
   新 run 需加「epoch × 观测点覆盖」完整性门。

---

## 7. Artifacts

```text
diagnostic/full300_trajectory_probe/_preflight_local.py    ← 本机 P1/P3/P5 实证脚本
diagnostic/full300_trajectory_probe/_preflight_local.json  ← 其输出
diagnostic/full300_trajectory_probe/EMA_PREFLIGHT_P1_P5.md ← 本报告
```

本报告**未修改**任何既有源码 / YAML / config / checkpoint / 权重；未启动训练；未做任何 forward 之外的写操作。

---

## 8. ONE NEXT STEP

```text
修 _medium_cells.py 的坐标口径（dataset bbox 而非 label 归一化 ×1280），
并用 _v7_vectors.json V1 行做逐位回归（断言 max|diff| == 0）；
然后才谈 hook 目标切换与 P4 gate。
```

在采样坐标修正并通过回归前，**不得**启动 GPU run。
