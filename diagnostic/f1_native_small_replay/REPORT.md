# F1 — Native-small Synchronized CopyPaste

**日期**：2026-09-29
**性质**：**single-variable training experiment 的 preflight**。
本轮**未训练**（本地无 GPU；训练必须走云端）、**未启动 train.py**、**未 forward / backward / optimizer**。
所有 gate 均为 **dataset 层**验证。

---

## 0. VERDICT

```text
EXPERIMENT READY              （17 / 17 preflight gates PASS）
```

```text
F1 唯一自变量:  native-small instance exposure ↑   （+57.6% / epoch）
模型 · loss · TAL · optimizer · scheduler · resolution · D′ 增强 · seed 全部不变
```

**⚠ 一处必须你知道的取舍（本轮已由你裁决）**：协议 §9 冻结的是 `p=0.30`（eligible 条件下）。
preflight 实测**只有 401/1600 = 25.06% 的训练图含 ≥1 个 native-small GT**，因此

| 设置 | 每图实际 replay 率 | **总 exposure 乘数** |
|---|---:|---:|
| `p=0.30`, ≤1 个（协议原文） | 7.5% | **1.086× (+8.6%)** |
| **`p=1.0`, ≤2 个（本轮采用）** | 25.06% | **1.576× (+57.6%)** |
| `p=1.0`, ≤1 个（协议规则的硬上限） | 25.06% | 1.288× (+28.8%) |

⇒ 按协议原文的 `p=0.30` 只做一个 **+8.6%** 的操纵，功效过低；
采用 **p=1.0 + 每图 ≤2 个** 才让本实验真正能回答「增加 exposure 是否有用」。**开训后不得再改（§38）。**

**训练位置**：本机为 CPU-only torch（`Local has no GPU`），300 ep ／ 1280 ／ yolo11m 必须云端执行。
下面的命令是云端形态。

---

## 1. 唯一自变量与冻结语义（实现时定死）

| 项 | 值 | 来源 |
|---|---|---|
| native 面积阈值 | **1024**（native COCO small < 32²） | `native_small_replay_small_area` |
| **触发概率 p** | **1.0** | `native_small_replay`（F1 config） |
| **每图最多新增** | **2** | `native_small_replay_max_new_instances` |
| instance scale | **1.0**（仅平移） | 代码：crop/paste 1:1 |
| rotation / shear / perspective | **0 / 0 / 0** | 代码：无任何几何变换 |
| crop context margin | **0.10**（bbox 宽/高，固定） | `native_small_replay_margin` |
| 与已有 GT 的最大 IoU | **0.10** | `native_small_replay_max_dest_iou` |
| 与 source 框的最大 IoU | **0.10** | `native_small_replay_max_src_dst_iou` |
| destination 重采样上限 | **20**（用尽则停止后续轮，不强行粘贴） | `native_small_replay_max_attempts` |
| source 选择 | **均匀随机**，且只用**原始** native-small GT | 代码 |
| destination | **本图内**均匀随机 top-left | 代码 |
| 类别 | **保持 source 类别** | 代码 |
| seed | **42**（与 D′ 同） | F1 config |
| imgsz / epochs | **1280 / 300** | F1 config |

**native 面积口径（§2 的硬要求）**：此刻 bbox 位于 `load_image` 之后的 input space，
代码用 `ori_shape/resized_shape` 换算回 native 再比阈值 —— 与仓库既有 OASA 同一处理
（`augment.py` 注释里记录了不换算会把 1080p 图的阈值等效放宽到 2304 native px²）。

---

## 2. 实现（本轮改动 4 处，全部非模型）

| # | 文件 | 改动 | SHA256 |
|---|---|---|---|
| 1 | `ultralytics/data/augment.py` | 新增 `class NativeSmallReplay` + `_f1_iou_one_vs_many`；`v8_transforms` 的 Compose **首元素**插入 | `53c75148fc168a53…` |
| 2 | `ultralytics/cfg/default.yaml` | 新增 6 个 `native_small_replay*` 键（默认值全为惰性：`0.0` / `1024` / `0.10` / `0.10` / `0.10` / `20` / `1`） | `04f0d20a8a55cc4b…` |
| 3 | `scripts/train.py` | 按既有 `ir_encoding` / `object_scale_aug` 范式转发这些键 | `db4371e360ae0e38…` |
| 4 | `configs/train_rgbid_sepstem_clahe_f1.yaml` | **新增**：D′ config 的逐行副本 + `experiment_name` + 2 个 F1 键 | `534ae8c4500b33dd…` |

**未改动**：任何模型代码（backbone / neck / head / SepStem / 初始化）、`ultralytics/utils/loss.py`、
`tal.py`、evaluator、dataset、labels、val 预处理、inference、`best.pt`。

**`configs/train_rgbid_sepstem_clahe_f1.yaml` 相对 D′ 的逐行 diff（实测）** = 仅
`experiment_name`（1 行）+ F1 注释块与 2 个键（27 行新增）。**有效键值 diff**：

```text
仅 F1 有: native_small_replay = 1.0
          native_small_replay_max_new_instances = 2
仅 D′ 有: （空）
其它键值差异: 无（name 是 experiment_name 的派生，只决定 run 目录）
```

**关键实现语义**：

- `p <= 0.0` 时 `__call__` **立即 return，不消耗任何 RNG** ⇒ 不写该键的配置（含 D′）数据流与 RNG 序列**逐位不变**（§19 已由 Gate 10 实测）。
- **多模态同步性由构造保证**：RGB/IR/Depth 已在 `base.py:load_image` 合并为单张 5ch 数组，
  **一次 crop/paste 即等价于三模态同 native 矩形**。代码里没有任何 modality-specific 几何操作。
- **不触碰 depth registration**（§5）：不做 warp / alignment / interpolation / calibration。
- source 池 = **进入 `__call__` 时的原始 GT**（`small0` 在 paste 之前算好），因此第 2 轮不会把
  第 1 轮粘贴出来的实例再当 source。

---

## 3. REAL TRANSFORM ORDER（Gate 8）

```text
before:  load_image → [ObjectScaleAug] → [Mosaic → CopyPaste → RandomPerspective]
                       → MixUp → Albumentations → RandomHSV → RandomFlip(v) → RandomFlip(h) → Format

after  :    load_image → ★NativeSmallReplay★ → [ObjectScaleAug] → [Mosaic → CopyPaste → RandomPerspective]
                       → MixUp → Albumentations → RandomHSV → RandomFlip(v) → RandomFlip(h) → Format
```

实测（`transform_order.txt`）：

```text
Compose 顺序 = ['NativeSmallReplay','ObjectScaleAug','Compose','MixUp','Albumentations',
               'RandomHSV','RandomFlip','RandomFlip','Format']
pre_transform 内部 = ['Mosaic','CopyPaste','RandomPerspective']
```

⇒ **插入点 = Compose 的第一个元素**，即 `load_image` 之后、**Mosaic 之前** —— 与协议 §17 的目标顺序一致。
（`ObjectScaleAug` 在 F1 中 disabled（`object_scale_aug` 缺省 false）⇒ 零开销 passthrough。）

---

## 4. PREFLIGHT GATES — 17/17 PASS

| Gate | 结果 | 实测证据 |
|---|---|---|
| **G1** source 只用 native-small | **PASS** | 200 条记录中越界 **0**；`src_area_native < 1024` 恒成立 |
| **G2** scale = 1.0 | **PASS** | `max|Δw|, max|Δh| = 0.000e+00`（阈值 1e-6） |
| **G3** RGB/IR/Depth 同步 | **PASS** | 200 条记录的 **5 个通道逐通道和 paste 前后完全一致**（0 例不符） |
| **G4** bbox 尺寸保持 | **PASS** | `max|Δ| = 0.000e+00` |
| **G5** class 保持 | **PASS** | 新标签类 ≠ source 类：**0** 例 |
| **G6** 遮挡约束 | **PASS** | 与已有 GT 的最大 IoU = **0.0118**（阈值 0.10）；越界 **0** 对 |
| **G7** 标签完整性 | **PASS** | 原标签被保留 + **恰好新增 1 条/记录**：**0** 违例 |
| **G7b** 每图上限 | **PASS** | 实测 `max_per_call = 2`（= `max_new_instances`）；`multi(≥2)` = 100 |
| **G8** transform order | **PASS** | 见 §3 |
| **G9** D′ 等价 | **PASS** | 训练变量差异集合 **恰好** = {`native_small_replay`=1.0, `max_new_instances`=2} |
| **G9b** 继承 D′ args.yaml | **PASS** | 28 个关键项逐项 OK（ir_encoding=clahe / RGBID / 5ch / 1280 / 300 / SGD / lr0 / lrf / momentum / wd / warmup / seed=42 / mosaic=1.0 / close_mosaic=10 / translate / scale / degrees / shear / perspective / fliplr / flipud / mixup=0 / copy_paste=0 / rect=False / yolo11m.pt / patience=0 / amp） |
| **G10a** 构建不改 RNG | **PASS** | 构建含 F1 的 transforms 前后 `random` / `numpy` / `torch` 三个 RNG state **逐位相同** |
| **G10b** p=0 不耗 RNG | **PASS** | 连续 10 次调用，RNG state 不变 |
| **SMOKE_no_error / _5ch / _no_nan** | **PASS** | 20 样本 0 异常；`(5,1280,1280)` CHW；无 NaN/Inf |
| **PARAM** 参数量 | **PASS** | `--model_config` 与 D′ **同一文件**（`configs/yolo11m_sepstem.yaml` sha `9b14f294…`）；F1 不接触模型代码 ⇒ **0 参数变化** |

**Gate 9 的一处判据说明**：`name`（= `experiment_name`）必然不同（只决定 run 目录，协议要求独立目录），
已列入**预期差异白名单**并校验其值 = F1 的 experiment_name；它不是训练变量。

---

## 5. EXPOSURE CHANGE（§27/§28 —— 开训前先算清）

```text
train 图 = 1600        （dataset 载入 1599，1 张坐标越界被自动剔除）
GT      = 11567
native-small GT = 1392
含 >=1 个 native-small GT 的图 = 401   (eligible 率 = 0.2506)

F1: p=1.0, max_new_instances=2
   ⇒ 每个 eligible 样本稳定新增 2 个（preflight: skip_attempts = 0）
   ⇒ 每 epoch 新增 = 401 × 2 = 802 个 native-small instance

   D′  native-small exposure = 1392 / epoch
   F1  native-small exposure = 1392 + 802 = 2194 / epoch
   **multiplier = 1.5761×   (+57.6%)**
```

⚠ 这是**进入 augmentation pipeline 之前的 native-small instance 数**的比值；mosaic 之后每个源图可能
被复制进多个 tile，但**该比值对两侧同等放大**，因此作为「exposure 的相对增量」是良定义的。

**§28 要求的训练后复核**（训练完成后必须做，用同一脚本重算并比对实际 replay 率）：
`diagnostic/f1_native_small_replay/_exposure.json` 已写入本轮基线值，训练后重跑同一统计即可。

---

## 6. SMOKE TEST（§25，20 samples，无模型）

```text
样本 20：异常 0
图像 shape 集合 = [(5, 1280, 1280)]      （Format 后为 CHW；5 = RGBID 通道数）
label 数（变换后） = [26,9,29,9,15,7,28,5,12,23,7,19,27,15,6,17,19,25,8,8]
replay 统计 = seen 20 / eligible 20 / replayed 20 / instances_added 40 / max_per_call 2 /
              skip_attempts 0 / skip_prob 0 / skip_nosmall 0
⇒ 无 NaN、无异常、无形状/通道异常
⇒ 这 20 张恰好全部 eligible（前 20 个样本较富），不代表全体（全体 eligible 率 25.06%）
```

另做了 400→169 样本的取证扫描（Gate 1–7 的证据来源）：

```text
扫描 169 张样本；捕获 replay 记录 200 条
stats = {seen:169, eligible:100, replayed:100, instances_added:200, multi:100,
         max_per_call:2, skip_prob:0, skip_nosmall:69, skip_attempts:0}
```

⇒ **eligible 样本 100% 拿到满额 2 个**，destination 重采样**从未用尽**（`skip_attempts=0`）。

---

## 7. PROVENANCE

```text
TRAINING_RUNS_STARTED = 0
FORWARD / BACKWARD / OPTIMIZER_STEP = 0
CHECKPOINT_MODIFIED = 0
```

| 文件 | before（D′ 侧） | after |
|---|---|---|
| `configs/train_rgbid_sepstem_clahe.yaml`（D′） | `a4e329cfc3d22020…` | **未变** `a4e329cfc3d22020…` |
| `configs/yolo11m_sepstem.yaml` | `9b14f29460733384…` | **未变** `9b14f29460733384…` |
| `runs/…_sepstem_clahe/weights/best.pt` | `1cae45f75693f541…fae4fda` | **未变** |
| `yolo11m.pt`（§22 的初始化权重） | `d5ffc1a674953a08…` | **未变**（与 `reports/E1_EXPERIMENT_CONTRACT.md` 记录一致） |

**本轮修改的 4 个文件**（§2 表）+ 新增 diagnostic 目录 `diagnostic/f1_native_small_replay/`。

`.gitignore` **未改动**；`ultralytics/data/augment.py` 在既有 exception 下**可被 git 追踪**
（`git check-ignore` 返回未被忽略）⇒ 该代码改动**可哈希核验**。

**产物**（§26 manifest）：

```text
diagnostic/f1_native_small_replay/
    preflight.json          ← manifest（frozen 常数 + 全部 gate + 实际 replay 率）
    config_diff.json        ← 有效键值逐项 diff + 28 项继承对照
    transform_order.txt     ← 真实 before/after 顺序
    smoke_test.json         ← 20 样本 smoke
    provenance_before.txt   ← git HEAD/status + best.pt sha
    _exposure.json          ← exposure 乘数基线（训练后重算比对）
    _preflight.py / _run.txt / preflight.log / _hashes_after.txt
```

**git HEAD**：`f659609c28f6b4f500212197d5229550a36d12d8`。

---

## 8. 云端训练命令（**需你确认后才执行**）

```bash
# ① 训练前 preflight（既有脚本，硬门禁）
python scripts/preflight_check_train_config.py \
    --model_config configs/yolo11m_sepstem.yaml \
    --train_config configs/train_rgbid_sepstem_clahe_f1.yaml \
    --expect-ir-encoding clahe

# ② 正式训练（从 yolo11m.pt 重新开始，**不 resume**，不 warm-start 自任何实验 checkpoint）
python scripts/train.py \
    --model_config configs/yolo11m_sepstem.yaml \
    --train_config configs/train_rgbid_sepstem_clahe_f1.yaml
```

训练后**必须**（协议 §27–§35，本轮不做）：exposure 复核、官方 evaluator 判决（`MAX_BOXES_PER_IMAGE=100`）、
paired bootstrap（B=400）、scale/class decomposition、small detection rate、checkpoint SHA。

---

## 9. 本轮我自己的错误（如实记录）

| # | 现象 | 真因（我的） | 更正 |
|---|---|---|---|
| 1 | `get_cfg` 报 `'native_small_replay' is not a valid YOLO argument` | 自定义键必须先在 `default.yaml` 注册（OASA 即如此，`default.yaml:146-151`），我漏了 | 在 `default.yaml` 注册 7 个键（默认全惰性） |
| 2 | `Gate 9` FAIL | 我把 `name`（experiment_name 派生，只决定 run 目录）当成训练变量 | 白名单化并校验其值 |
| 3 | `SMOKE_5ch` FAIL | `Format` 之后图像是 **CHW**，我按 HWC 读了 `shape[2]` | 改读 `shape[0]` |
| 4 | `np.delete(ious, k)` 单框时 `zero-size array to reduction` | 只有 1 个 GT 时删除自身得到空数组 | `others.size` 守卫 |
| 5 | `G2/G4` FAIL，`max|Δ| = 6.1e-05` | 记录层把 destination 建成了 **float32**，在量级 1280 上 `(dx1+w)−dx1` 有 ~6e-5 舍入 | 记录用 float64，写标签时再降 dtype |
| 6 | F1 config 行尾变 CRLF | Windows `write_text` 默认换行转换 | 强制 LF；逐行 diff 现在**只**剩预期差异 |

**没有一项更正改变判定方向；最终 17/17 PASS。**

---

**本轮结束，停止。未训练、未 forward、不自行启动训练。等你确认后再走 §8 的云端命令。**
