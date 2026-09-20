# D' 训练结果诊断 —— **IR 预处理未生效，实验判定 INVALID**

**日期**：2026-09-20 · **性质**：只读诊断（未训练、未修改任何文件、未提交）
**结论**：**`SEPSTEM_RUN_INVALID_IR_ENCODING`** —— 该 checkpoint 不能回答"D vs D'"这个问题。

---

## 0. 结论速览

| 项 | 结果 |
|---|---|
| fusion architecture | ✅ Separate-Stem 正确落地 |
| pretrained remap | ✅ 已在真实训练中生效 |
| 训练超参 | ✅ 与 D 逐项一致 |
| **IR 预处理** | ❌ **未使用 CLAHE，实际是 percentile** |
| **D vs D' 的变量数** | ❌ **2 个**（fusion ＋ IR 预处理），非 1 个 |
| 预注册问题"D vs D'" | **无法回答（实验污染）** |

---

## 1. 决定性证据：`ir_encoding` 从未被传入

**两处独立记录一致**：

### 1.1 落盘 `args.yaml`

```
D  runs/urban_multimodal_det_yolo11_rgbid_ir_clahe/args.yaml      → ir_encoding: clahe
D' runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/args.yaml → 无 ir_encoding 键
```

### 1.2 checkpoint 内嵌 `train_args`（独立第二来源）

```
D   model=configs/yolo11m_earlyfusion.yaml   ir_encoding='clahe'
D'  model=configs/yolo11m_sepstem.yaml       ir_encoding='<ABSENT>'
```

**为什么"缺席"就等于 percentile**：

- `scripts/train.py:476` `if "ir_encoding" in train_cfg: kwargs["ir_encoding"] = ...` —— 键不在配置里就不传。
- `ultralytics/data/base.py:333` `getattr(getattr(self, "hyp", None), "ir_encoding", "percentile")` —— 属性不存在 → **回落到 `percentile`**。
- `args.yaml` 会 dump **完整 namespace（含 defaults）**——证据：D' 的 args.yaml 里 `modality_dropout` 明明没在配置里设置，却照样出现。
  因此 `ir_encoding` **整条缺席**只能说明它既不在配置、也不在 default.yaml → 数据集侧走的是 percentile 默认分支。

> 即：本次训练喂给模型的 IR 是 **1%/99% 分位拉伸**，不是 **CLAHE(clipLimit=2.0, tileGridSize=(8,8))**。

---

## 2. 与本地配置文件的矛盾（说明云上用的是第三个文件）

| 本地文件 | `experiment_name` | `ir_encoding` |
|---|---|---|
| `configs/train_rgbid_sepstem.yaml` | `..._rgbid_sepstem` | **缺席** |
| `configs/train_rgbid_sepstem_clahe.yaml` | `..._rgbid_sepstem_clahe` | `clahe` |
| `configs/train_rgbid_ir_clahe.yaml`（D 的） | `..._rgbid_ir_clahe` | `clahe` |

D' 的 run 名 = `..._rgbid_sepstem_clahe`（**只有 `_clahe` 那个文件才有这个名字**），
但 args 里 **没有** `ir_encoding`（**只有非 clahe 那个文件才没有这个键**）。

⇒ 云上使用的训练配置**不是本地任何一个文件**，而是"`..._sepstem_clahe` 的 experiment_name ＋ 无 ir_encoding"的组合
—— 最可能是在 `train_rgbid_sepstem.yaml`（percentile 版）基础上只改了 `experiment_name`。

**这是本轮实验的根因，且属于流程问题而非代码问题。**

---

## 3. 该跑分里**正确**的部分（已逐项验证）

| 检查 | 证据 |
|---|---|
| 架构 = Separate-Stem | `layer0=Identity`(Silence)、`layer1=SilenceChannel`、31 层、`ch=5`、`nc=12`、params **20,061,972** |
| model yaml 正确 | ckpt `train_args.model = configs/yolo11m_sepstem.yaml` |
| **pretrained remap 已生效** | ep1 mAP50-95 = **0.25607**（from-scratch 应≈0.001；D 的 ep1 = 0.22871，同量级） |
| backbone/head 结构与 stock 对应 | 形状兼容张量 **532** 个 |
| 超参 = D | epochs 300 / patience 0 / batch 8 / imgsz 1280 / SGD m0.937 wd5e-4 / lr0 0.005 / cos_lr / warmup 3 / seed 42 / close_mosaic 10 / 全部 aug 默认值 —— **逐项相同** |
| 训练完整 | results.csv 300 行；`best.pt` / `last.pt` 各 40,641,577 B |

**唯一的变量污染就是 IR 预处理。**

---

## 4. 数值（仅作诊断，**非判据**）

| 模型 | fusion | IR | 训练内 val best | best ep |
|---|---|---|---|---:|
| 官方 baseline | early | percentile | 0.55800 | 237 |
| **D** | early | **CLAHE** | **0.56806** | 263 |
| **D'（本次）** | **sepstem** | **percentile** | **0.56157** | 256 |

- 对 D 的差：**−0.00649**（但两者差两个变量，**不可比**）
- 对**官方 baseline** 的差：**+0.00357**（这两者只差 fusion 一个变量 → **可比**）

曲线：ep1 0.25607 → ep256 0.56157 → ep300 0.54113（无早停，与 D 同预算）。

---

## 5. 该怎么理解这个 checkpoint

**它不能回答**：`D' (IR-CLAHE + Separate-Stem) > D (0.5553) ?`

**它能回答（且是干净的）**：`Separate-Stem + percentile  vs  官方 baseline(early + percentile, 线上 0.53180)` ——
只差 fusion 一个变量，直接对照官方 baseline 的 **0.53180**。

即：这份权重可以**按另一组身份**使用，但不能冒充 brief §0 定义的 D'。

---

## 6. 建议（三选一，需你裁决）

| 方案 | 含义 | 代价 |
|---|---|---|
| **A. 重跑** | 用真正的 `configs/train_rgbid_sepstem_clahe.yaml` 重训，回答预注册问题 | 再一次 300ep |
| **B. 改判身份** | 承认这是一次 **fusion-only** 实验，提交后与**官方 0.53180** 比，而非 0.5553 | 0（但答案不是原问题） |
| **C. 弃用** | 不提交 | 0 |

**开训前的防呆**（无论选哪个）：启动后立即检查 `args.yaml` 里 **`ir_encoding` 是否存在且等于 `clahe`**；
不存在就立刻停机。本轮 dry-run 之所以没拦住，是因为 dry-run 用的是**本地**配置（有 clahe），
而云上实际用的配置不同 —— **dry-run 无法覆盖"云上配置与本地不一致"这一类风险**。

---

```text
SEPSTEM_RUN_INVALID_IR_ENCODING
D  online            = 0.5553   （未变更）
OFFICIAL_FALLBACK    = 0.53180  （未变更）
本轮修改的既有文件    = 0
本轮训练次数          = 0
```
