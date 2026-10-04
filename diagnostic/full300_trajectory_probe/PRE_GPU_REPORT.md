# EMA FULL-300 PROBE — PRE-GPU REPORT

```text
PRE-GPU EMA PROBE STATUS: READY
```

> 所有 7 项门槛 PASS。**未启动 GPU**（且本机 `torch 2.4.1+cpu`、`cuda_available=False`、无 `nvidia-smi`）。
> 本报告只建立「正确、可复现、部署一致的 measurement instrument」，**不含任何机制判断**（§14）。

---

## 0. 门槛汇总

| # | Gate | 状态 | 关键读数 |
|---|---|---|---|
| 1 | coordinate repair | **PASS** | 1320/1320 与冻结 V1 逐位一致 |
| 2 | legacy V1 regression | **PASS** | `max|ΔP3| = 0.0`、`max|Δh| = 0.0`、`max|Δlogit| = 1.96e-06` |
| 3 | deployment baseline | **PASS** | n=1320，纯 `eval()` |
| 4 | EMA identity | **PASS** | `ck["ema"]=None` + `model` 为 FP16（strip_optimizer 路径） |
| 5 | logit identity | **PASS** | `max_residual = 9.31e-06`（n=1320） |
| 6 | non-intrusive probe | **PASS** | flags/buffers/params/RNG 四项全还原 |
| 7 | h collection sanity | **PASS** | G86 ratio 0.9999 / CONTROL 1.0007 |

---

## 1. Coordinate repair

**问题**：`_medium_cells.py` 用「**标签文件**归一化中心 × 1280」。标签中心相对**原图**，而 16:9 图进
1280² 方形画布要 letterbox（`ratio_pad=((gain,gain),(0,pad_y))`，pad_y=280）⇒ 采到错误 cell。
`_v7_extract.py` 用的是 **dataset** `lab["bboxes"]`（已 canvas 归一化）⇒ ×1280 才正确。
两者 bbox 来源不同，旧注释「`_v7_extract.py` 口径」是**错的**。

**修复**：`diagnostic/p3_trajectory_probe/_medium_cells.py` 改从 `build_yolo_dataset(mode="val")` 的
`lab["bboxes"]` 取坐标；输出到**新文件** `medium_cells_canvas.json`。

```text
coordinate_source = dataset_lab_bboxes_canvas_normalized
旧文件 medium_cells.json 保持不动：sha256 = 5d0ba324616bd48d8c7b9a01300eb867f4bc410ed26b78d247f62f2de5223c6b（与 5+ 个既有 diagnostic 记录一致）
新文件 medium_cells_canvas.json  sha256 = 1f6517c82bd735bf2889bdddb9aa7aefda51f1e3020cf1c6ebd01f648ac59aef
```

回归（内建于生成器，FAIL 即不产出）：

```text
与 _v7_vectors.json V1 比对：1320/1320 坐标逐位等价（max|Δx| = max|Δy| = 0.005）
```

`.005` 不是误差 —— 冻结参考里的 `center` 是 `round(x, 2)`（`_v7_extract.py:138`），**0.005 就是它的存储精度**。

角色组成不变：`G86 = 86`、`CONTROL = 1070`、`MISSED_NON86 = 164`（共 1320，覆盖 281 张 val 图）。

---

## 2. Legacy V1 regression（口径 A · 仅历史回归）

```text
mode             = eval backbone + head.train()
cells            = medium_cells_canvas.json（frozen p3_cell_x/p3_cell_y）
weights          = D′ best.pt（部署 EMA）sha 1cae45f7…fae4fda
n_keys           = 1320
max|ΔP3|         = 0.0
max|Δh|          = 0.0
max|Δlogit|      = 1.960e-06      (tol 1e-3)
STATUS           = PASS
```

**1320/1320 的 P3 与 h 逐位为 0**，logit 只差 FP 舍入。⇒ 坐标修正后，历史 `_v7_vectors.json V1`
可以**完全复现**；旧 27/29 vs 40/51 的矛盾确认由坐标 bug 造成（§7）。

---

## 3. Deployment-mode baseline（口径 B · 正式口径）

```text
probe_target     = trainer.ema.ema
probe_mode       = model.eval()            ← 正常 validation / inference 语义
coordinate_source= dataset_lab_bboxes_canvas_normalized
weights          = D′ best.pt sha 1cae45f7…fae4fda（未重训）
n_records        = 1320
STATUS           = PASS
```

采集实现要点：`eval()` 下 `Detect` 走 `_inference` 不再返回 train-mode 的
`cat(cv2[i], cv3[i])` 张量。为**不把 head 切回 train**，改为直接 hook `cv2[i]` / `cv3[i]`
再自行 concat —— 在纯 eval 下复现同一张量。无 batch=1 特殊化、无手动 mode 切换、无 BN/dropout/
preprocessing/geometry/dtype 改动。

---

## 4. EMA identity

```text
ck["ema"]        = None
ck["model"]      = DetectionModel(nc=12)，dtype ∈ {float16, int64}
ck["epoch"]      = -1        ck["best_fitness"] = None
```

`strip_optimizer`（`torch_utils.py:610-611`）先 `x["model"] = x["ema"]` 再 `x["ema"] = None`；
`validator.py:118` `model = trainer.ema.ema or trainer.model`；
`trainer.py:524-525` 存 `"model": None` / `"ema": deepcopy(self.ema.ema)`。
⇒ 部署 `best.pt` 的 `model` 对象**就是 EMA**。**PASS**。

---

## 5. P3 logit identity

```text
identity         = CLS_OUT[gt_c, y, x] == W_gt · h + b_gt
n                = 1320
max_residual     = 9.315e-06        (CPU / FP32)
median_residual  = 1.727e-06
STATUS           = PASS
```

### 5b. 判据修正：绝对 → 相对（2026-10-02，首次云端运行后）

首次云端 P4 gate 实测 `identity_max_abs_resid = 1.0025e-03`，被原阈值 `1e-3` 拦下。
**这不是恒等式被破坏**（四个还原标志全 True），是**绝对阈值不适用于 GPU**：

| 环境 | 绝对残差 | 原因 |
|---|---:|---|
| 本机 CPU / FP32 | ~4e-06 | FP32 累加 |
| 云端 GPU | ~1.0e-03 | cuDNN 默认启用 **TF32**（10-bit mantissa ⇒ logit ~10 时约 1e-3） |

（`_dryrun_ema.py` 未调用 `.to(cuda)`，因此它在云端实际跑在 CPU 上、给出 3.8e-06 —— 这个混淆曾掩盖差异。）

**现判据**：`rel = |OUT − (W·h+b)| / (|OUT| + |W·h+b| + 1.0) < 1e-2`。
分母加 `1.0` floor 是必要的：CONTROL 的 logit 中位仅 +0.88，近零 logit 会让纯相对残差虚高。

**判别力实测**（负例对照）：

| 情形 | abs | rel | 判定 |
|---|---:|---:|---|
| 真实路径（CPU） | 4.7e-06 | 1.7e-07 | PASS |
| 真实路径（GPU 实测值） | 1.0e-03 | ~3e-05…4e-04 | PASS |
| 负例 A：W 错位一个类 | 6.98 | 0.189 | **FAIL** ✓ |
| 负例 B：cell 错位 5 格 | 6.12 | ~0.46 | **FAIL** ✓ |

⇒ 真实数值与真实破坏之间相隔 **3 个数量级**，阈值有充分余量。

---

## 6. P4 non-intrusive probe

```text
scope = probe observation 本身不得额外修改 EMA/raw/optimizer/scheduler/RNG；
        正常 ema.update() 造成的变化**不在**本项范围内（EMA 本就被持续改写）
方法  = flags 快照 → 切 eval → 只读观测 → restore → 校验（**未使用 deepcopy**）

legacy pass     : mode_flags_restored=True  bn_buffers_unchanged=True  params_unchanged=True  rng_restored=True
deployment pass : mode_flags_restored=True  bn_buffers_unchanged=True  params_unchanged=True  rng_restored=True
STATUS          = PASS
```

⚠ **执行边界（不夸大）**：本机无训练进程，以上是在**部署 EMA 权重**上对 `observe()` 本身的验证。
真正的「in-training」P4 由 `_run_probe300_ema.py` 的 **in-process gate** 在云端训练启动时强制执行：
`on_pretrain_routine_end` → `probe.self_check(trainer.ema.ema)` → 任一 False 即 `raise` 中止训练
（产物写 `p4_gate_ema.json`），因此**训练不可能在 gate FAIL 的情况下继续**。

---

## 7. H collection sanity

同一份权重、同一批 cell，只差 head 的 train/eval：

| 角色 | n | Legacy V1（eval+head.train） | Deployment（eval） | ratio |
|---|---:|---:|---:|---:|
| G86 | 86 | **40.382** | **40.380** | 0.9999 |
| CONTROL | 1070 | **51.510** | **51.545** | 1.0007 |
| ALL | 1320 | **49.859** | **50.091** | 1.0046 |

**Legacy 列的 40.382 / 51.510 与 `cls_representation_modality_audit` 的 h_norm 中位数完全一致**
（独立复算，见 `diag/cls_representation_modality_audit/per_gt_representation_audit.csv`）——
这正是用户所称的「EMA audit h ≈ 40 / 51」。

⇒ **27/29 vs 40/51 的差异已被坐标修复解释**：旧 probe 采在错误 cell 上（|Δy| 中位 9.6 cell、
最大 34.4 cell），而非 raw-vs-EMA。修复后 **deployment mode 与 legacy 只差 0.05%–0.46%**，
h 数值稳定。**STATUS = PASS**。

---

## 8–10. Final observation contract

```json
{
  "probe_target":      "trainer.ema.ema",
  "probe_mode":        "eval",
  "coordinate_source": "dataset_lab_bboxes_canvas_normalized",
  "cells_file":        "diagnostic/p3_trajectory_probe/medium_cells_canvas.json",
  "epochs":            300,
  "observation_epochs": [0,1,2,5,10,20,30,50,75,100,125,150,175,200,225,250,275,289,299],
  "indexing":          "0-based（csv = 0-based + 1，trainer.py:666）",
  "sample_definition": "G86=86 / CONTROL=1070 / MISSED_NON86=164（共 1320）",
  "close_mosaic":      10
}
```

---

## 11. Instrument dry-run（本机 CPU，非训练轨迹）

`_dryrun_ema.py`：把 D′ 部署 EMA 当作 `trainer.ema.ema` 替身，走完整链路。

```text
attach_model → self_check(P4) → observe×2 → dump_epoch → finalize → _analyze300.py
images=8  records=41/epoch  roles={G86:13, CONTROL:25, MISSED_NON86:3}
identity_max_resid = 8.74e-06        restore: 四项全 True
_analyze300.py rc = 0
§17 七个文件全部产出：trajectory_summary.csv / trajectory_per_gt.csv /
  trajectory_classifier_drift.csv / trajectory_representation_drift.csv /
  trajectory_decomposition.csv / trajectory_tal.csv / trajectory_metadata.json
metadata: probe_target=trainer.ema.ema  probe_mode=eval  coordinate_source=dataset_lab_bboxes_canvas_normalized
STATUS = PASS
```

---

## 12. 与上一轮 INVALID 的差异对照

| | 旧 PROBE30 / FULL300（INVALID） | 本轮 |
|---|---|---|
| probe target | `de_parallel(trainer.model)`（raw） | `de_parallel(trainer.ema.ema)` |
| cell 坐标 | 标签归一化 × 1280（漏 letterbox pad） | dataset `lab["bboxes"]` × 1280 |
| mode | `eval backbone + head.train()` | 纯 `model.eval()` |
| 跑到 300 | **否**（csv 止于 297，`final_eval` 未执行） | 由 runner 断言 + 完整性门（见 §13） |

---

## 13. 启动前仍需注意（非阻塞，但需人工确认）

1. **§6 完整性门**：新 run 必须真跑到 0-based 299（csv 300）且 `final_eval()` 执行
   （判据：`weights/best.pt` 被 strip 到 ~40 MB、`epoch=-1`）。上一轮止于 csv 297。
2. **观测成本**：19 个观测点 × 281 图 = 5339 次 CPU/GPU 前向（云端 GPU 下可忽略）。
3. **P4 gate 是硬阻塞**：若云端 `self_check` FAIL，训练会直接 `raise` 中止 —— 这是设计行为，不是崩溃。

---

## 13b. §6 观测时机 —— 源码实核 + 运行期证明

**源码路径**（`ultralytics/engine/trainer.py`，本轮逐行核对，非推测）：

```text
344   self.epoch = epoch                       ← 0-based
348   self.scheduler.step()
~380  for i, batch in enumerate(self.train_loader):
          ... optimizer_step(batch, i)
587-595   def optimizer_step():
              scaler.unscale_(); clip_grad_norm_(); scaler.step(optimizer); scaler.update()
              optimizer.zero_grad()
              if self.ema: self.ema.update(self.model)      ← EMA 更新
      ... on_train_epoch_end
      ... ema.update_attr(...) ; validate()（用 EMA）; save_model()（写 EMA）
454   self.run_callbacks("on_fit_epoch_end")   ★ probe 在这里
```

⇒ **`on_fit_epoch_end` 必然晚于本 epoch 的全部 optimizer step 及对应 `ema.update()`。**
`observation_timing = "after_normal_ema_update"` **成立**。

**运行期证明**（不依赖对源码的信任）：`ModelEMA.update()` 每次 `self.updates += 1`
（`torch_utils.py:554`）。probe 在每个 `on_fit_epoch_end` 记录 `ema.updates` 增量：

```python
delta = trainer.ema.updates - prev
if delta <= 0:
    raise RuntimeError("§6 违反：… 未在本 epoch 更新过 ⇒ STATUS=INVALID")
```

该门已用 stub trainer 实测：正常路径记录
`[{"epoch":0,"ema_updates_this_epoch":7,...},{"epoch":1,"ema_updates_this_epoch":14,...}]`；
人为冻结 `updates` 时**正确 raise**。证据逐 epoch 写入
`trajectory_metadata.json::observation_timing_evidence.runtime`。

---

## 13c. hook 生命周期修复（2026-10-02，第二次云端运行后）

**现象**：云端跑到 epoch 0 末崩溃：

```text
File "ultralytics/engine/trainer.py", line 441, in _do_train
    self.save_model()
File ".../ultralytics/engine/trainer.py", line 520, in save_model
    torch.save(
AttributeError: Can't pickle local object 'P3TrajProbe._mk_out.<locals>.h'
```

**根因**：`save_model()` 执行 `torch.save(deepcopy(self.ema.ema))`（`trainer.py:520-525`）。
probe 现在挂的是 **EMA 本体**，注册在它上面的 forward hook 会被一并 pickle；而 hook 是
`_mk_out.<locals>.h` 这种**局部闭包** ⇒ 不可 pickle。
旧 probe 侥幸没崩，只因为它挂的是 raw model，被 pickle 的是**无 hook 的 EMA** ——
这是「改成追踪 EMA」带来的**新**失效面。

**修复**：`attach_model()` 不再注册任何 hook，只保存引用；hook 改由 `_register()` 在
**每次观测期间临时注册**、`finally` 里 `_unregister()` 注销。模型在观测之外始终干净。

**新增回归门**（`self_check` 内，直接复现 `save_model()` 的序列）：

```python
torch.save({"model": None, "ema": copy.deepcopy(nn_model).half(), ...}, BytesIO())
```

| | hooks_left_on_model | picklable_like_save_model | PASS |
|---|---:|---|---|
| 正例（新实现） | 0 | True | **True** |
| 负例（旧式常驻 hook） | 9 | **False** | **False** |

负例复现出的错误信息与云端崩溃**逐字相同**。⇒ 该失效模式已被门捕获，不会再消耗一次 12 小时运行。

---

## 13d. inference-tensor 缓冲区还原修复（2026-10-02，第三次云端运行后）

**现象**：epoch 0 的 `on_fit_epoch_end` → `observe` → `finally: _restore()` 抛：

```text
File ".../_probe300_ema.py", line 138, in _restore
    v.copy_(s["bufs"][k])
RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
```

**根因**：`BaseValidator.__call__` 整体被 `@smart_inference_mode()` 包裹（`validator.py:108`），
且**在其内部**执行 dtype 往返：

```python
self.args.half = self.device.type != "cpu" and trainer.amp   # GPU+AMP ⇒ True
model = trainer.ema.ema or trainer.model
model = model.half() if self.args.half else model.float()    # :119
...
if self.training:
    model.float()                                            # :208
```

`nn.Module._apply` 在 `inference_mode` **内**新建并替换 param/buffer ⇒ **每次 validate() 之后，
EMA 的 param 与 float buffer 全部变成 inference tensor**。对它们做 `v.copy_(...)`
（in-place，且在 inference_mode 之外）即抛错。

**为什么 D′ 自己没事**：`ModelEMA.update()` 走 `self.ema.state_dict()`，而
`nn.Module.state_dict()` 默认 `keep_vars=False` ⇒ 返回 `param.detach()`，
**detach 后的张量不受该限制**（已实测）。我的 `_restore` 用的是 `named_buffers()`，
它返回**原始（被标记的）**张量 ⇒ 撞上限制。

**修复**：`_restore` 改**重绑定**而非 in-place：

```python
if ref is None or torch.equal(v.detach(), ref):
    continue                      # 纯 eval 观测本就不该改 buffer
mod._buffers[name] = ref.clone()  # 重绑定：不动 inference tensor，且把它换回普通 tensor
```

**回归覆盖**：`_dryrun_ema.py` 现在先做
`with torch.inference_mode(): ema.half(); ema.float()`，**忠实复现 validate() 之后的状态**，
再跑完整观测链。不模拟这一步，dry-run 就覆盖不到该崩溃。

| | 修复前 | 修复后 |
|---|---|---|
| `_restore`（inference tensor 上） | `RuntimeError`（云端已复现） | OK，`_verify` 四项全 True |
| dry-run（含 inference_mode 往返） | — | **PASS** |

---

## 14. Artifacts

```text
diagnostic/p3_trajectory_probe/_medium_cells.py            ← 坐标口径已修（输出改到 canvas 文件）
diagnostic/p3_trajectory_probe/medium_cells_canvas.json    ← 新 cells（1320，含 frozen cell 下标）
diagnostic/full300_trajectory_probe/_preflight_deploy.py   ← 全量 preflight（7 门）
diagnostic/full300_trajectory_probe/_preflight_deploy.json ← 其读数
diagnostic/full300_trajectory_probe/_preflight_deploy.log
diagnostic/full300_trajectory_probe/_probe300_ema.py       ← 部署口径 EMA probe
diagnostic/full300_trajectory_probe/_run_probe300_ema.py   ← 云端 runner（含 P4 in-process gate）
diagnostic/full300_trajectory_probe/_dryrun_ema.py         ← 仪器 dry-run
diagnostic/full300_trajectory_probe/_dryrun_ema/           ← dry-run 产物（非训练轨迹）
diagnostic/full300_trajectory_probe/PRE_GPU_REPORT.md      ← 本报告
```

**未修改**：任何源码 / 训练配置 / 权重 / checkpoint；`medium_cells.json`、`_v7_vectors.json`、
`per_gt_representation_audit.csv` 三个冻结件的 SHA 均未变。**未启动训练**。
