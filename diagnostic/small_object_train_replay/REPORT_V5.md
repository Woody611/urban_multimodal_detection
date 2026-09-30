# SMALL_OBJECT_TRAIN_REPLAY_AUDIT_V1

**性质**：只读诊断。0 训练 / 0 resume / 0 改 checkpoint / 0 改 `ultralytics/` / 0 改 configs / 0 改 train script / 0 改 model / 0 改 evaluator。

---

## 0. Executive：本轮的**第一个结论推翻了实验前提**

> **G1/G2/G4 全部取自 val split。train pipeline 只处理 train split，二者不相交。**
> 因此「replay G1 在训练时的 supervision」**在构造上不可能** —— 这些目标从未进入训练。

证据：首版脚本按 G1/G2/G4 选图后，`需 replay 的图 = 0`（0 个目标 GT 落在 train split）。
split 由 `_split_train_val_rgbid(val_ratio=0.2)` 生成，train 1600 / val 400，互斥。

**⇒ 问题被改写为可回答的等价形式**：*train 集的 small GT，在真实增强后是否仍获得有效正类监督？*
这决定了 Hypothesis A（训练期监督饥饿）是否成立 —— 若训练侧对 small GT 的监督本身健康，则 G1 的 val 失效只能是泛化/决策边界问题。

---

## A. Data integrity

```
train/val only                 : 是
test / submission / online     : 未读取（脚本路径全部为 data/processed/rgbid_split_train/）
no training / no resume        : 是
augment.py                     : 5cb9a407625921ce3f12ffc1  （与本轮开始时一致）
cfg/default.yaml               : 991a89b32de769d1d444cf34  （一致）
scripts/train.py               : 9f55b09f9e1229b61fefd496  （一致）
models/yolo/detect/train.py    : b021354f541bbb3c7770eb34  （一致 —— 该改动来自更早的 resume-guard 轮次）
checkpoint (D′ best.pt)        : 1cae45f75693f54146e35c5f… （一致，未改动）
本轮修改的既有文件              : 无
```

---

## B. Replay integrity

| 项 | 值 |
|---|---|
| seed | **20260926**（另有 per-image `SEED+i`，保证同图可重复） |
| dataset | `build_yolo_dataset(mode="train")`，真实 train split，n=1599 |
| augmentation pipeline | **项目真实 pipeline**：`Mosaic(p=1.0)` → `CopyPaste(p=0)` → `RandomPerspective(translate=0.1, scale=0.5, degrees/shear/perspective=0)` → `RandomFlip(fliplr=0.5)` → `LetterBox(1280)` → `Format`。**未手写任何替代增强** |
| 采样 | 250 图（固定 stride 采样），**249 个 mosaic 样本成功** |
| 记录 GT 数 | **3544** |

### ⚠️ Reproducibility 限制（不假装确定性）

首版尝试用「`Mosaic._mosaic4` 按 tile 顺序拼接」把输出 GT 映射回源图 GT。
实测 **映射校验：可追踪 6 / 不可追踪 244** —— mosaic 之后的 `RandomPerspective` / `_cat_labels`
会改变 GT 集合，「输出 GT 数 == Σ tile GT 数」不成立。

**⇒ 因此本轮不报告 per-GT 的 retention / 身份追踪**（见 §E），改为对**增强后的输出 GT** 直接分层
—— 这些正是训练时真正参与 loss 的 GT，无需身份。

---

## C. Train-time assignment（真实增强样本上）

按**增强后**尺度分层（输入空间 1280 像素）：

| bucket | n | aug_sqrt 中位 | n_pos 中位 | best_align 中位 | pos_target_max 中位 | align<0.01 | target<0.01 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **small** (<1024) | **1179** | 21.5 | 9.0 | **0.568** | **0.933** | 4.0% | 0.8% |
| medium | 1736 | 51.2 | 10.0 | 0.760 | 0.967 | 0.9% | 0.0% |
| large | 629 | 149.8 | 10.0 | 0.861 | 0.982 | 0.2% | 0.0% |

### 更细分（small 桶内部）—— **存在单调的尺度依赖，但不是「饥饿」**

| aug_sqrt | n | best_align 中位 | pos_target_max 中位 | n_pos 中位 |
|---|---:|---:|---:|---:|
| **<12px** | 115 | **0.167** | **0.780** | **2.0** |
| 12–18px | 261 | 0.392 | 0.887 | 4.0 |
| 18–24px | 336 | 0.583 | 0.936 | 8.5 |
| 24–32px | 467 | 0.666 | 0.954 | 10.0 |

**关键读数**：即使最小的 <12px 桶，`best_align = 0.167`（**非 0**）、`pos_target_max = 0.780`（**非 0**）、`n_pos = 2`（**非 0**）。

---

## D. Train-time classification supervision

`target_scores` 直接取自真实 `TaskAlignedAssigner.forward` 的第 3 个返回值（**未自定义替代量**）：

- **small 桶**：`pos_target_max` 中位 **0.933**，`target<0.01` 的仅 **0.8%**
- 即使 <12px：`pos_target_max` 中位 **0.780**，`target<0.01` 占比仍低

**⇒ 训练时该 GT 确实产生了 positive class target，且目标分接近 1。**
**⇒「`alignment ≈ 0 → target_score ≈ 0`」在训练侧**没有**系统性发生。**

---

## E. Augmentation exposure

| 项 | 结果 |
|---|---|
| Mosaic involvement | **100%**（p=1.0，249/249 触发；`mosaic_calls=250`） |
| RandomPerspective / scale / translate | 全部样本均经过（真实 pipeline 默认启用） |
| fliplr | 按 p=0.5 随机 |
| **per-GT retention（mosaic 后是否存活）** | **无法计算** —— 身份映射在 244/250 样本上不可追踪（§B） |
| augmented 对象尺度 | 见 §C：small 桶 aug_sqrt 中位 21.5px |

**⇒ 无法回答「G1 是否被训练增强 disproportionately 丢弃」。** 按你的 §9，如实报告为**不可计算**，不以近似量替代。

---

## F. Train → Val mechanism comparison

| Group | 数据来源 | train-time n_pos | train-time align | train-time cls target | val P3 logit | **val class prob** |
|---|---|---:|---:|---:|---:|---:|
| **small（train, n=1179）** | 真实增强 | 9.0 | **0.568** | **0.933** | — | — |
| **small <12px（train, n=115）** | 真实增强 | 2.0 | **0.167** | **0.780** | — | — |
| **G1（val, n=38）** | 无增强 | 5.0 | **0.0000** | 0.0000 | **−12.58** | **0.0000** |
| G2/G4（val, n=98） | 无增强 | 5.0 / 6.0 | 0.1345 | 0.7532 | +0.83 / +0.88 | **0.7532** |

```
训练侧：small GT 得到 align 0.17–0.57、target 0.78–0.93 的**有效正类监督**
                    ↓
验证侧：G1 的 align = 0.0000、类别概率 = 0.0000（G2/G4 = 0.7532）
```

**⇒ 监督在训练侧被交付；模型在**留出实例**上完全无响应。**
**⇒ Hypothesis A（训练期监督饥饿 / alignment starvation）不被支持。**

---

## G. Mechanism decision

```text
REPRESENTATION_GENERALIZATION   （主，E2）
TRAINING_SUPERVISION            （仅尺度依赖的弱化，E2，但不足以解释 G1）
UNRESOLVED                      （B 与 C 在本轮证据下不可区分）
```

| 候选 | 判定 | 证据等级 |
|---|---|---|
| **TRAINING_SUPERVISION** | **不支持为 G1 的原因**。训练侧 small GT 的 align 中位 0.568、target 0.933；即使 <12px 仍为 0.167 / 0.780；`align<0.01` 仅 4.0%。**但存在真实的尺度依赖弱化**（<12px 的 align 0.167 vs 24–32px 的 0.666），属 E2 的次要发现 | **E2** |
| **REPRESENTATION_GENERALIZATION** | **最一致**：训练侧交付监督、验证侧 align=0.0000 / prob=0.0000；V4 已证 assignment 几何正常（n_pos p=0.184）、P3 特征存在（弱 8%） | **E2** |
| **CLASSIFICATION_BOUNDARY** | 与上者在现有证据下**不可区分**（两者都预测「训练有监督、推理无响应」） | E1 |
| **AUGMENTATION_EXPOSURE** | **无法评估** —— per-GT retention 因身份不可追踪而无法计算 | E0 |
| **UNRESOLVED** | 保留 | — |

**严禁的逻辑跳跃（本轮一条都未采用）**：未提出改 topk / 改 class loss / 加 P3 attention / 直接训练。

---

## H. Coverage

| mechanism | N | /70 | % |
|---|---:|---:|---:|
| REPRESENTATION_GENERALIZATION（覆盖 G1 全部） | **38** | 38/70 | **54.3%** |
| TRAINING_SUPERVISION（尺度依赖弱化，<12px；**非 G1 的直接原因**） | — | — | 不适用 |
| AUGMENTATION_EXPOSURE | — | — | **不可计算** |

---

## I. Final Gate

```text
HOLD
```

**理由**：

1. **本轮的核心问题在原始形式下不可回答** —— G1 是 val GT，训练侧从未见过它们；「训练时监督是否饥饿」对它们无定义。
2. 改写后的可回答问题已得答案：**训练侧的 small GT 监督健康**（align 0.568 / target 0.933）⇒ Hypothesis A 被排除。
3. 剩下的 Hypothesis B 与 C 在现有证据下**不可区分**（都预测「训练有监督、推理失效」）。按 §11，不可区分时不得强行归类。
4. **AUGMENTATION_EXPOSURE 无法评估**（身份追踪失败）——这本身是 §16 意义上的缺失证据。

**还缺的证据**：能区分 B（表征泛化）与 C（决策边界）的判据。
二者都自洽于「train 有监督 / val 无响应」，需要一个新的可观测差异，例如：
**同一批小目标在 train split 上的模型输出**（train 样本上的 logit 是否也≈0）。
- 若 train 样本上的 logit 正常、val 上≈0 → **B（泛化）**
- 若 train 样本上同样≈0 → **C（决策边界/监督未能转化为输出）**
这是**只读**的（对 train split 做一次 forward），不需要训练。

---

## §17 Reproducibility

```text
training              : 0
assigner modification : 0（仅 in-process 子类插桩）
model / config / script modification : 0
new inference         : 0（仅 forward，不落盘 prediction）
online submission     : 0
test data accessed    : 0
```

**新增文件（均在 `diagnostic/small_object_train_replay/`）**
```
_v5_replay.py      真实 train pipeline replay + 真实 assigner/target_scores 插桩
_v5_replay.json    3544 条 GT 记录（含 stat）
_v5_run.log        运行日志
REPORT_V5.md       本报告
```

**复现**
```bash
python -X utf8 diagnostic/small_object_train_replay/_v5_replay.py
```

---

## 附：本轮自己的错误（记录备查）

1. **首版把目标集合设为 G1/G2/G4 → `需 replay 的图 = 0`**。这暴露了前提错误（G1 属 val），是本轮最有价值的发现，但当时是当 bug 遇到的。
2. **在移除身份映射后，误留 `if not okmap: continue`** —— 导致 244/250 样本被跳过，结果只来自 6 个样本 / 112 GT 的有偏子集（当时显示 align 0.594 / target 0.945，与最终 0.568 / 0.933 接近，但样本量差 32 倍）。已修正并重跑。
3. `InstAssigner.forward` 首版用 `self.last = dict(...)` **覆盖**了 `get_pos_mask` 在 super() 内部写入的键 → `KeyError: 'mask_pos'`。改为 `update`。
