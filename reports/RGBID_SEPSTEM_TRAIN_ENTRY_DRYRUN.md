# RGBID Separate-Stem Fusion — Train Entry Dry-Run

**日期**：2026-09-19
**范围**：用**正式训练入口**跑通 1 个真实 batch（1 forward + 1 backward + 1 optimizer.step）后停止。
**未训练完整 epoch、未创建正式 run、未修改 baseline / dataset / augmentation / loss / optimizer / evaluation。**
**最终判决**：**`TRAIN_ENTRY_READY`**

---

## 1. 正式训练命令

```bash
python scripts/train.py \
  --model_config configs/yolo11m_sepstem.yaml \
  --train_config configs/train_rgbid_sepstem.yaml
```

**走的是真实入口**：`scripts/train.py::main()` —— 真实参数解析 → 真实 dataset 切分 → 真实
`YOLO(model_path)` → 真实 `model.train(**kwargs)` → 真实 `DetectionTrainer` → 真实
`get_model()` → 真实 `_transfer_rgb_pretrained` → 真实 DataLoader → 真实 forward/backward/step。
**不是**手工实例化模型，**没有**绕过 `scripts/train.py`。

---

## 2. 训练前配置确认（在真实 trainer 内打印）

| 项 | 值 | 判定 |
|---|---|---|
| model_config | `configs/yolo11m_sepstem.yaml` | ✅ |
| train_config | `configs/train_rgbid_sepstem.yaml` | ✅ |
| **scale** | **`m`** | ✅ 非 `n`（非 2.59M nano） |
| **ch** | **5** | ✅ |
| **nc** | **12** | ✅ |
| **params** | **20,061,972**（≈20.06M） | ✅ |
| **pretrained** | **`yolo11m.pt`** | ✅ |
| epochs / batch / imgsz | 300 / 8 / 1280 | ✅ 与官方 RGBID 配置逐项相同 |
| device | `cpu` | ⚠️ 见 §9 声明的运行时覆盖 |

---

## 3. Pretrained Remap 在正式入口下的结果

在 `on_train_start` 时刻（remap 已执行完毕）逐 tensor 复算，与 `yolo11m.pt` 比对：

| stem | 期望初始化 | `torch.equal` | `max\|Δ\|` |
|---|---|---|---|
| **RGB**（in=3, out=48） | `stock[:48]` | **True** | **0.000e+00** |
| **IR**（in=1, out=8） | `mean(stock[:,0:3])[:8]` | **True** | **0.000e+00** |
| **Depth**（in=1, out=8） | `mean(stock[:,0:3])[:8]` | **True** | **0.000e+00** |

**与上一轮 remap audit 脚本结果完全一致**（backbone/head 结构对应 532/538 = 98.9%，stem 逐位相等）。
正式入口与 audit 脚本**无分歧**。

> 再次强调：`load()` 打印的 `Transferred 61/661` **不是**成功指标（它只反映按名匹配）；
> 唯一可信的是上述 tensor-level 比对。

---

## 4. 真实 dataset 与真实 batch

| 项 | 值 |
|---|---|
| 数据 | `data/processed/rgbid_split_train`（RGBID 三模态切分，1600 train / 400 val） |
| 入口内的切分函数 | `_split_train_val_rgbid`（**幂等**：`dataset.yaml` 已存在 → 直接返回，**未重写任何数据**） |
| 每 epoch batch 数 | 200（1600 / batch 8） |
| **真实 batch 输入 shape** | **`(8, 5, 1280, 1280)`** |

✅ 真实 5 通道 RGBID 样本（B,G,R,IR,Depth 合并），**非 synthetic tensor**。
✅ RGB / IR / Depth pairing 正常（能加载出 5 通道即证明三模态路径全部命中）。

---

## 5. Forward + Backward

| stem | grad 非 None | grad 有限 | `|grad|` sum |
|---|---|---:|---:|
| **RGB** | ✅ True | ✅ True | **5557.64** |
| **IR** | ✅ True | ✅ True | **1184.35** |
| **Depth** | ✅ True | ✅ True | **869.304** |

梯度在 `backward` 的 tensor hook 中捕获（因为 `optimizer_step()` 内部会先 `zero_grad()`，
到 `on_train_batch_end` 时梯度已被清零）。

**三个 stem 的梯度全部非 None、有限、非 0。** ✅

---

## 6. Optimizer Step

| 项 | 值 |
|---|---|
| `optimizer.step` 执行次数 | **1**（wrapper 计数） |
| optimizer | `SGD(lr=0.005, momentum=0.937)`，3 个参数组 |
| **该步实际 per-group lr** | **`[0.1, 0.0, 0.0]`** |
| SGD momentum buffer \|sum\| | rgb **49.56** / ir **10.56** / depth **7.75**（三个 stem 全部非零） |

momentum buffer 非零 ⇒ **三个 stem 的梯度确实进入了 optimizer.state**。

### 6.1 三个 stem 是否实际发生参数变化

| stem | 参数 | 变化 | `max\|Δ\|` |
|---|---|---|---|
| **rgb** | **bn.bias** | **✅ True** | **8.337498e-04** |
| rgb | bn.weight | False | 0.000000e+00 |
| rgb | conv.weight | False | 0.000000e+00 |
| **ir** | **bn.bias** | **✅ True** | **7.216185e-04** |
| ir | bn.weight | False | 0.000000e+00 |
| ir | conv.weight | False | 0.000000e+00 |
| **depth** | **bn.bias** | **✅ True** | **5.722046e-04** |
| depth | bn.weight | False | 0.000000e+00 |
| depth | conv.weight | False | 0.000000e+00 |

**三个 stem 都真实发生了参数更新** ⇒ 三个 stem 都真正参与优化。

### 6.2 为什么 conv.weight / bn.weight 在第 1 步没变（**是设计，不是缺陷**）

`ultralytics/engine/trainer.py:373` 的 warmup 调度：

```python
ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
```

`j == 0` 指 **param_groups[0]**。而 `build_optimizer`（`trainer.py`）的构造顺序是：

```python
optimizer = optim.SGD(g[2], ...)              # param_groups[0] = 偏置（含 BN bias）
optimizer.add_param_group({"params": g[0]})   # param_groups[1] = 卷积权重
optimizer.add_param_group({"params": g[1]})   # param_groups[2] = BN weight
```

因此**第 0 次迭代只有偏置组拿到非零 lr（`warmup_bias_lr=0.1`），卷积权重与 BN weight 的 lr 就是 0.0** ——
实测 `[0.1, 0.0, 0.0]` 与之吻合。

这是 ultralytics 对**任何模型**（包括现官方 baseline）的既定行为，**不是本候选的缺陷**。
在一个 batch 的预算内，能证明"stem 参与优化"的正确证据就是偏置组的变化 + momentum buffer 非零，
两者都已给出。

---

## 7. 模型规模复核

```text
params = 20,061,972      （baseline 20,063,412，Δ = −1,440 = −0.0072%）
scale  = m               （非 n / 非 2.59M）
GFLOPs ≈ 272.68 @1280    （baseline 273.85）
```

---

## 8. Run 目录

| 项 | 值 |
|---|---|
| 实际写入位置 | `%TEMP%\rgbid_sepstem_dryrun\run\`（仅 `args.yaml` + `weights/`） |
| `runs/` 下新建目录 | **0 个**（`ls runs/ \| grep sepstem` 为空） |
| 是否覆盖 `runs/..._rgbird_ir_quicktest/` | **否** |
| 是否覆盖任何 baseline checkpoint | **否** |
| `FREEZE_MANIFEST` | **16/17 完好**（唯一不符项为既有的 `predict_rect.py` 漂移） |

---

## 9. 声明的运行时覆盖（仅 2 项，均为环境约束，不涉及项目文件）

| # | 覆盖 | 原因 | 是否影响实验有效性 |
|---|---|---|---|
| 1 | `device` → `"cpu"` | **本机无 GPU**（见既有事实：本地 torch 为 CPU 版，训练必走云端）。训练配置仍是 `device: cuda / gpu_ids: [0]`，云端执行时不需要此覆盖 | 否 —— 只影响算力，不影响架构/数据/超参 |
| 2 | `project/name` → `%TEMP%` | 避免 dry-run 污染 `runs/`（brief §9 要求） | 否 —— 只影响输出落盘位置 |

覆盖通过**运行时注入**实现（temp 脚本内 monkeypatch `scripts.train._resolve_device` 与
`_build_train_kwargs`），**未修改任何项目文件**。

---

## 10. 停止纪律

| 项 | 结果 |
|---|---|
| 是否执行超过 1 个 batch | **否** —— `batches executed = 1` |
| 是否执行超过 1 次 optimizer.step | **否** —— `optimizer.step calls = 1` |
| 是否进入 epoch 2 | **否** |
| 停止方式 | `on_train_batch_end` 回调抛 `DryRunDone` 哨兵，`scripts/train.py::main` 退出后捕获 |

---

## 11. 最终判决

```text
TRAIN_ENTRY_READY
```

全部检查通过：正式入口 → 真实 config → 真实 5ch batch → forward → backward → 1×optimizer.step，
三个 stem 梯度非零、momentum buffer 非零、`bn.bias` 实际更新；
scale/ch/nc/params 全部正确；pretrained remap 与 audit 脚本结论一致（三个 stem 逐位相等）；
无 run 污染；未超过 1 batch / 1 step。

**到此停止。不自动开始正式训练。**

```text
OFFICIAL_FALLBACK = 0.53180（未变更）
DECISION: TRAIN_ENTRY_READY
```
