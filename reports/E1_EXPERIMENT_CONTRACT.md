# E1_EXPERIMENT_CONTRACT — L12–L17 Region Response Gain

**实验名**：E1 = D′ + RegionResponseGain(L13, L15, L17)
**日期**：2026-09-27
**状态**：见文末 `EXPERIMENT READY / NOT READY`（由 `diagnostic/e1_region_gain/_e1_gates.py` 的退出码决定）

> **审计结论：21/21 门 PASS，EXPERIMENT READY。**
> 审计过程中捕获并修复了一个会**静默破坏单变量对照**的实现缺陷（见 §Audit finding）。

---

## Frozen（全部冻结，不得改动）

| 项 | 值 |
|---|---|
| dataset | `data/processed/rgbid_split_train/dataset.yaml`（`scripts/train.py::_split_train_val_rgbid` 生成，幂等复用） |
| split | `val_ratio 0.2`，`seed 42`，按**原图 stem 分组**切分（杜绝近重复泄漏） |
| input size | **1280** |
| model baseline | **D′ = SepStem + CLAHE**（`configs/yolo11m_sepstem.yaml`，31 层，**20,061,972** 参数） |
| pretrained checkpoint | **`yolo11m.pt`**（sha256 `d5ffc1a674953a08…`），走既有 `_remap_separate_stem` |
| optimizer | SGD，momentum 0.937，weight_decay 5e-4 |
| scheduler | CosineAnnealingLR（`cos_lr: true`） |
| epochs | **300**，`patience: 0`（禁用早停，与 D′ 同等预算） |
| batch | 8 |
| augmentation | `configs/train_rgbid_sepstem_clahe_e1.yaml` 与 D′ 逐字相同 |
| CLAHE | `ir_encoding: clahe`（clipLimit 2.0, tileGridSize 8×8） |
| evaluator | 冻结 `scripts/official_map.py`（metric A，cap 100，val 400 图 / 2807 GT） |
| inference | 冻结 `scripts/predict_rect.py --mode rect --imgsz 1280 --conf 0.001 --iou 0.7` |
| threshold / NMS / 输出格式 | 冻结不变 |
| seed | 42 |
| 起始权重 | **`yolo11m.pt`**，与 D′ 完全同一起点；**不从任何实验 checkpoint warm-start** |

---

## Changed（唯一变量）

**只有 E1 机制本身。**

`RegionResponseGain` 挂在 SepStem graph 的**既有层内部**（不新增 top-level 层，故层索引与预训练迁移口径均不变）：

```python
y = x * (1 + tanh(dw3x3_depthwise(x)))      # dw 零初始化 => t=0 时 y == x 逐位
```

| 位点 | 层 | 类 | shape@1280 | 角色 |
|---:|---|---|---|---|
| 13 | C3k2 | stride 16 | (1,512,80,80) | P4 backbone 特征 → 直连 `L19 = Concat(up(L17), L13)` 的 **local skip** |
| 15 | C3k2 | stride 32 | (1,512,40,40) | P5 backbone 特征 |
| 17 | C2PSA | stride 32 | (1,512,40,40) | **V13 的因果位点** → 经 L18 上采样直连 L19 的 **top-down** 支路 |

参数代价：3 × (512·3² + 512) = **15,360**（20,077,332 vs 20,061,972，**+0.0766%**）。FLOPs 增量可忽略。

---

## Hypothesis

V11 表明 hard-miss 小目标（G1）与 train-success 的判别性差异**首次稳定出现在 L12（stride 16）**，且 backbone 承载其中 85% 的分离度；L13→L16 的 G1/TS **响应范数比单调衰减**（0.872 → 0.772），即深层对该类目标的响应被系统性削弱。

V13 进一步给出**因果**证据：把 L17 目标位置的响应替换为「典型成功小目标」的向量，G1 class logit median **+5.869**（p=3.1e-08），且**空间特异性极强**（位移 +4 cells 只有 −0.0038）、G1 特异（G4 仅 −0.098）。其中**约 70% 的 rescue 来自向量幅值而非方向**。

⇒ 因此在该区域引入一个**恒等初始化、空间局部、可学习的响应幅值增益**，有可能把被削弱的响应拉回可检测水平，减少 small-object hard miss（D′ 的 small 失败结构：完全无框 19.1%、定位失败 18.6%、**分类失败仅 0.8%**）。

**已知反面事实（不得淡化）**：V13 同时证明 **L17 不充分** —— 扩到 49 个 cell（7×7）也只能把 logit 从 −14.63 推到 −6.07，且 R0 单 cell 已占 76.5% 增益（early saturation）。故本实验属于「低风险、机制有据、但**不保证充分**」的干预。

---

## Primary metric

**official local mAP50-95**（冻结评测器，metric A，cap 100）。

- D′ baseline = **0.51528**
- 评测器重评差 = **0.00e+00**（确定性，见 `reports/OFFICIAL_EVAL_FREEZE_AND_HISTORICAL_RETEST.md`）

## Secondary diagnostics

| 指标 | D′ baseline |
|---|---|
| small AP75 | **0.0439** |
| small AP50 | 0.1006 |
| small recall @K=100 | **60.1%** |
| medium AP50 | 0.3482 |
| large AP90 | 0.4272 |
| class-level AP | 见冻结评测报告 |
| small 失败结构 | 无框 19.1% / 定位 18.6% / 分类 0.8% |

全部用**同一个**冻结 evaluator，且与 D′ 同参数。

---

## Success rule（**预先定死，事后不得修改**）

**噪声标尺**：评测器噪声 = 0；决定判定的是**单次训练内的 checkpoint 选择噪声 ≈ 0.007**
（证据：P2 探针 = baseline 自身 9 个 checkpoint 的均值 0.50232 vs best 0.50928；`base last ep300` 0.50073 vs best 0.50924）。

| 判定 | 条件 |
|---|---|
| **SIGNAL** | mAP50-95 ≥ **0.52228**（= D′ + 0.007）**且** small AP75 ≥ **0.0509**（= 0.0439 + 0.007）**且** fork 口径 Δ 与 official Δ 同号 |
| **INCONCLUSIVE / NO_MEANINGFUL_GAIN** | 落在 **[0.50828, 0.52228)** —— 记录在案，不提交，**不计为 reject** |
| **REJECT** | < **0.50828**（= D′ − 0.007） |
| **瓶颈未解决** | 总分达 SIGNAL 但 small AP75 未同步上升 ⇒ 即便微升也判为「未解决瓶颈」，不得据此声称修复 |

**跨口径纪律**（项目既有裁定）：官方口径**不可单独作否决依据**；同架构族内 **fork 排序 = 线上**且线上 ≈ fork × 85.9；最终以**线上分数**为准。两个口径符号相反时，方向信 fork。

---

## Provenance

| 项 | 值 |
|---|---|
| git commit | `f659609`（branch `yyy`）+ 工作区改动 |
| 冻结基线口径 | **当前工作区**：`augment.py 5cb9a407…` / `default.yaml 991a89b3…` / `scripts/train.py 9f55b09f…` / `detect/train.py b021354f…` —— 与 V11/V13 记录**逐一吻合** |
| baseline ckpt | `runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt`，sha256 `1cae45f75693f541…`（= V11/V13 的 `1cae45f7…`，同一模型） |
| E1 config | `configs/yolo11m_sepstem_e1.yaml` + `configs/train_rgbid_sepstem_clahe_e1.yaml`（SHA 见 `_e1_gates.json`） |
| modified source | `ultralytics/nn/modules/block.py`、`ultralytics/nn/tasks.py`、`ultralytics/models/yolo/detect/train.py`、`scripts/preflight_check_train_config.py`（E1 硬性断言；+ 2 个新 config） |
| train command | `python scripts/train.py --model_config configs/yolo11m_sepstem_e1.yaml --train_config configs/train_rgbid_sepstem_clahe_e1.yaml` |
| full config | 见上两份 yaml（逐字，无 CLI 覆盖） |
| seed / GPU | 42 / 云端单卡 |
| framework | ultralytics (fork) + torch；版本随训练日志记录 |
| dataset manifest | `data/processed/rgbid_split_train/dataset.yaml`（SHA 记录在案） |

> ⚠ `detect/train.py` 是 V11/V13 的**冻结 hash 文件**；本实验对其做了**唯一一处**改动：为 E1 新增模块加入 declared + shape-asserted 的迁移例外。该 delta 必须与本文件一同归档。

---

## STOP gates

任一失败即 STOP，**不自行修补**，报告 failure / root cause / proposed fix / affected files：

1. Gate A（E1=false identity）任一 FAIL
2. Gate B（E1=true 生效）任一 FAIL
3. **G16 FAIL** —— E1=true 扰动了共享权重初始化（单变量对照失效）
4. **G17 FAIL** —— 预训练迁移对非 E1 key 不再逐位一致
5. SL1 / SL2 / SL3 任一 FAIL
6. `parse_model` 的「实际覆盖 == 请求层」断言被触发
7. `scripts/preflight_check_train_config.py` 退出码非 0（含 `--expect-e1 true` 的 E1 硬性断言失败）
8. 任何需要额外修改 evaluator / inference / loader / augmentation / loss / optimizer / scheduler / NMS 才能跑通的情况

---

## 训练顺序

1. E1=false baseline identity test → Gate A
2. E1=true forward/backward activation test → Gate B + C
3. E1=true short pilot（云上 ~3 ep，仅排工程异常）
4. 正式训练（云上 300 ep，~11 h），**从 `yolo11m.pt` 起始**

---

## Audit finding —— 审计抓到的一个真实缺陷（必须记录）

**G16 首次运行 FAIL**：`first divergent shared key = model.14.conv.weight`。

**Root cause**：`nn.Conv2d.__init__` 会调用 `reset_parameters()`，**从全局 RNG 流取数**。
E1 模块虽然随后被 `zeros_` 覆盖成恒等，但那一次取数已经发生 —— 于是 L13 之后构建的**每一层**
都拿到了与 baseline 不同的随机初值。若不发现，`D′ vs E1` 就变成
「机制 + 半个网络的随机初始化不同」的**双变量实验**。

**Fix**（`ultralytics/nn/modules/block.py`）：在构造 `dw` 前后 `get_rng_state` / `set_rng_state`，
使该模块**零熵消耗**。修复后 G16 报 `661 个共享 key 逐位相同；新增 e1 key = 6`。

> 这正是本审计存在的意义。该缺陷在任何「只看 loss 有没有下降」的流程里都不会暴露。

**G12 / SL1 的两次 FAIL 是审计脚本自身的期望写错**（fp16 ckpt 用 fp32 输入；把级联增益
误当成各位点独立），已修正判据 —— 实现代码未因此改动。

---

## Verdict

```text
EXPERIMENT READY
```

**依据**：`python -X utf8 -u diagnostic/e1_region_gain/_e1_gates.py` → **21 门，FAIL 0**，退出码 0。
详见 `diagnostic/e1_region_gain/_e1_gates.json` 与 `_gates_run.txt`。

关键门读数：
- G5/G6/G7（E1=false）：forward / loss / gradient **max|d| = 0.000e+00**
- G16（E1=true）：**661 个共享 key 逐位相同**，仅新增 6 个 e1 key ⇒ 初始化零扰动
- SL2/G6b（E1=true，t=0）：forward 与 loss **逐位等于 D′** ⇒ 只加容量、不加扰动
- SL1：级联增益 measured == expected（0.099668 / 0.209270 / 0.329795）
- SL3：增益支撑集恰为 3×3 邻域 ⇒ 局部机制，非全局池化
- G12b：`YOLO(ckpt).predict` 在 1280×736 5ch rect 输入下结构一致（输出 162 框）

### 开关状态：已翻至 `e1_enabled: true`（✅ 已执行）

`configs/yolo11m_sepstem_e1.yaml` 现为 **`e1_enabled: true`**（E1 实验的执行状态）。
`_e1_gates.py` 已改为**两臂显式强制**（不依赖文件出厂值），故 Gate A 的有效性与该开关无关。

### 云上开训前必须跑的最后一道硬闸

```bash
python scripts/preflight_check_train_config.py \
    --model_config configs/yolo11m_sepstem_e1.yaml \
    --train_config configs/train_rgbid_sepstem_clahe_e1.yaml \
    --expect-ir-encoding clahe --expect-e1 true
```

`--expect-e1 true` 会硬性断言 7 项（任一不满足即 `E1 PRE-FLIGHT FAIL` + 退出码 1 + 禁止开训）：

| # | 断言 | 实测 |
|---|---|---|
| 1 | `e1_enabled` **严格为 True**（bool，非真值字符串） | `True (bool)` ✅ |
| 2 | `e1_layers` 严格等于 `[13, 15, 17]` | `[13, 15, 17]` ✅ |
| 3 | 实际构建参数量 == **20,077,332** | `20,077,332` ✅ |
| 4 | E1 module 数量 == **3** | `3` @ `[13, 15, 17]` ✅ |
| 5 | E1 新增张量 == **6** | `6` ✅ |
| 6 | graph 含 `RegionResponseGain` + 源码/真实 remap 日志出现 `E1 region response gain` | 实测 remap 日志 `6 new tensors` ✅ |
| 7 | 任一不满足 → **PRE-FLIGHT FAIL**，禁止启动训练 | 已实测：开关为 false 时 7 项 FAIL，退出码 1 ✅ |

不给 `--expect-e1` 时该段**完全不执行** —— 已用 diff 证明 baseline 调用路径输出**逐字节不变**。

若忘记翻开关，会静默训出第二遍 D′（11 小时白跑）；这道闸就是为它设的。

### 本次审计的最终 provenance（SHA256）

| 文件 | SHA256（前 24） |
|---|---|
| `configs/yolo11m_sepstem.yaml`（D′ baseline，**未改**） | `9b14f2946073338414b44417` |
| `configs/yolo11m_sepstem_e1.yaml`（新增；`e1_enabled: true`） | `b2cf656027e2235696d6a124` |
| `configs/train_rgbid_sepstem_clahe.yaml`（D′，**未改**） | `a4e329cfc3d220206448cdd3` |
| `configs/train_rgbid_sepstem_clahe_e1.yaml`（新增） | `99cfc5793b2df348a41ad5b4` |
| `ultralytics/nn/modules/block.py`（改：RNG 中性 + E1 分支） | `28b47f10079a0fc0454fcd91` |
| `ultralytics/nn/tasks.py`（改：e1_* 读取 + 覆盖断言） | `79c8dbab9cd810d2609e7ff1` |
| `ultralytics/models/yolo/detect/train.py`（改：remap 例外） | `f46198001a4b741ec4238e3a` |
| `scripts/preflight_check_train_config.py`（改：E1 硬性断言） | `fda7296594a2ac47c188a0c3` |
| `diagnostic/e1_region_gain/_e1_gates.py`（审计脚本） | `7ad17c0a40db76cab58d64a2` |
| `yolo11m.pt`（预训练源） | `d5ffc1a674953a08e11a8d21` |
| **baseline ckpt** `best.pt`（**未触碰**） | `1cae45f75693f54146e35c5f` |

### 未决事项

训练（第 3/4 步：云上 pilot + 300ep 正式）**尚未执行**，需你确认后才启动。
本机无 GPU（torch 2.4.1+cpu），训练必须走云端。
