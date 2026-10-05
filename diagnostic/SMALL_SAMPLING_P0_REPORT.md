```
========================================
SMALL-OBJECT SAMPLING P0
========================================

D′        = RGB + IR + Depth
D′ IR-Aug = RGB + IR + Depth
M4        = RGB + IR ONLY

GPU_USED = NO
TRAINING_STARTED = NO
CUDA_USED = NO

SMALL_SAMPLING_P0 = CONDITIONAL
GPU_CANDIDATE     = MAYBE
}
```

**日期** 2026-10-05 · **脚本** `diagnostic/small_sampling_p0.py` · **产出** `diagnostic/small_sampling_p0/{p0_results.json,run.log}`
**HEAD** `878cbcdeb40c92590b593e5b993885ce41735223` · branch `yyy`
`SAMPLING_CANDIDATE_MODALITY = RGB + IR + Depth` · `SAMPLING_CANDIDATE_IS_D_PRIME_DERIVATIVE = YES` · `M4 = RGB + IR ONLY`

---

## 1. 真实 sampling call graph（读源码，非猜测）

```text
BaseTrainer._setup_train
  └─ DetectionTrainer.get_dataloader(mode="train", rank=LOCAL_RANK)      [trainer.py]
       └─ build_yolo_dataset(...)                                        [build.py:96]
       └─ build_dataloader(dataset, batch=8, workers=4, shuffle=True, rank)   [build.py:142]
            sampler = None if rank == -1 else DistributedSampler(...)     [build.py:147]
            └─ InfiniteDataLoader(shuffle=True, sampler=None, batch_size=8)   [build.py:28]
                 └─ torch RandomSampler(dataset) ─ BatchSampler(bs=8, drop_last=False)
                      └─ _RepeatSampler(BatchSampler)   ← INFINITE            [build.py:71]
                 __len__ = len(BatchSampler) = ceil(1600/8) = 200              [build.py:41]
       ↓
dataset.__getitem__(i) → get_image_and_label(i) → load_image(i)             [base.py:532]
       → load_and_preprocess_image(visible / infrared / depth, 同一 stem)
       → buffer.append(i)                    ← FIFO, max = min(1600, 8*8, 1000) = 64   [base.py:679]
       → v8_transforms → Mosaic(p=1.0) → get_indexes(buffer=True)
                                          = random.choices(dataset.buffer, k=3)        [augment.py:3337]
       → 3 side images 各自再走 load_image → 再 append 进 buffer
       → _mosaic4 → RandomPerspective → Format → [B,5,H,W] = [R,G,B,IR,D]
```

| FILE | SHA-256(16) | FUNCTION | LINE |
|---|---|---|---|
| `ultralytics/data/build.py` | `f014263f2bfe5921` **(ignored)** | `build_dataloader` / `InfiniteDataLoader` / `_RepeatSampler` | 142 / 28 / 71 |
| `ultralytics/data/base.py` | `d7d2315ed4391c97` **(ignored)** | `load_image`, buffer FIFO | 676–683 |
| `ultralytics/data/augment.py` | `abfc90f53f75f7ad` | `Mosaic.get_indexes` / `_mosaic4` | 3334 / 3347 |
| `ultralytics/engine/trainer.py` | tracked | `_setup_train` → `get_dataloader(rank=LOCAL_RANK)` | — |
| `ultralytics/models/yolo/detect/train.py` | `71096a9e044652a5` | `get_dataloader` | 657 |

**关键事实**：`LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))`。本项目用 `python scripts/train.py` 启动（非 torchrun）
⇒ `rank == -1` ⇒ `sampler = None` ⇒ `shuffle=True` ⇒ **RandomSampler（无放回）**。
**每 epoch 恰好 1600 次抽取、覆盖全部 1600 图一次**；`steps/epoch = 200`；`300 epoch ⇒ 60,000 optimizer steps`。

**`default.yaml` 中不存在任何 sampler 钩子**（唯一相关的既有机制是 `cls_pw`（loss 权重，非采样）
与 `native_small_replay_*`（F1，内容干预，已关闭））。

---

## 2. Small-object definition provenance

```text
REUSE_EXISTING_DEFINITION = YES
definition source = diagnostic/gt_level_attribution.py:35
                    SIZE_BINS = (("small",0,1024), ("medium",1024,9216), ("large",9216,inf))
threshold        = native pixel area < 1024 px^2   (= COCO 32^2)
```
同一常量在 `cls_representation_modality_audit/_audit.py:42`、`e1_region_gain/_e1_vs_dprime_diag.py:31`、
`candidate_density_counterfactual/_analyze.py:21` 复用 ⇒ 项目内一致。**本轮未重新定义、未改口径。**

### ★ 任务书给定数字的 provenance 核对（必须报告）

| 任务书 | 本轮实测（同口径） | 结论 |
|---|---|---|
| train 1600 / val 400 | 1600 / 400 | ✅ |
| total GT 11570 | **11570** | ✅ |
| small GT 1392 (12.0%) | **1392 (12.03%)** | ✅ |
| images with small = 401/1600 | **401** | ✅ |
| mean/median/max small per image = 0.87 / 0 / 35 | **0.87 / 0 / 35** | ✅ |
| `person small = 703/4426` | **703 / 4426** | ✅ |
| `uav small = 118 / 174` | uav = **124 / 659**；**118/174 = `seat`** | ❌ 行标签错位 |
| `ball small = 22 / 70` | ball = **0 / 23**；**22/70 = `tricycle`** | ❌ 行标签错位 |
| median small `sqrt(area)` ≈ 17.7 px @1280 | **24.09 px native** / **23.22 px @1280** | ❌ 无法复现 |

⇒ **聚合数字全部精确复现**；**两个类的数值真实存在，但属于 `seat` 与 `tricycle`，不是 uav / ball**。
⇒ `ball` 在 train 中**没有任何 small 实例**（全文仅 23 个实例）。**17.7 px 在三种面积口径下都复现不出**
（true dims 24.09 / 假定 1920×1080 → 24.6 / 假定 1280×1280 → 25.3），本轮不采用该数字。

### ★★ 口径敏感性（任务书未提，但决定实验选谁）

| 判据 | 含 small 的图片数 | small 实例数 |
|---|---:|---:|
| **native 面积 < 1024**（canonical） | **401 / 1600 = 25.06%** | 1392 |
| **@1280 面积 < 1024**（模型实际看到的分辨率） | **611 / 1600 = 38.19%** | **2803** |

HR 图被 letterbox 缩到 ×0.667（面积 ×0.444）、LR 图被放大 ×2（面积 ×4）⇒ 两者的"small"不是同一批目标。
**"哪些图该被重复" 这个问题的答案依赖选哪把尺**，而任务书假定了 native。这是一个尚未决策的设计点。

---

## 3. Baseline exposure（没有采样、没有 Mosaic 放大）

```text
每 epoch 抽取 1600 次（无放回）⇒ 每图恰好 1 次
模型实际看到：1600 samples × 4 mosaic slots = 6,400 图次 / epoch
small-GT 曝光（slot 级）      = 3.569 个 / sample
small 图占比（抽取侧）        = 25.06%
small 图占比（mosaic slot 侧）= 24.82%
Mosaic buffer 内 small 占比   = 24.82%     ← 与抽取侧相同
```

---

## 4. 候选曝光模拟（Design B：加权、长度不变；20 epoch × 1600 = 32,000 samples）

```text
config       centre small%  slot small%  smallGT/sample  vs base
baseline            25.06%       24.82%          3.569     1.000
w=1.25              29.89%       30.06%          4.345     1.218
w=1.50              33.36%       33.32%          4.783     1.340
w=1.75              37.11%       37.26%          5.257     1.473
w=2.00              40.57%       40.26%          5.832     1.634
```

**⇒ 曝光确实增加，且是可测的。** 但见 §6b/§7 —— 代价不在曝光这一侧。

---

## 5. x1.25 / x1.5 / x1.75 / x2 全表

| w | 抽到 small 图的比例 | 过表达倍数 | 净曝光（端到端） | 每 epoch 唯一图 | 唯一样本覆盖 | 每 epoch 漏掉的 small 图 |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 (baseline) | 25.06% | 1.00× | 1.000 | **1600** | **100%** | **0%** |
| 1.25 | 29.48% | 1.18× | **1.218** | 1009 | 63.0% | **31.1%** |
| 1.50 | 33.41% | 1.33× | **1.340** | 1001 | 62.6% | **26.7%** |
| 1.75 | 36.92% | 1.47× | **1.473** | 992 | 62.0% | **22.9%** |
| 2.00 | 40.08% | 1.60× | **1.634** | 980 | 61.3% | **20.4%** |

⚠ 历史数字 "x2 ⇒ exposure +59.9%" 是**抽取侧**的解析值（`1600·2/(2·401+1199) = 1.599`）。
**端到端实测 1.634 > 1.599**（见 §6b），历史数字是**轻微低估**。

---

## 6. Mosaic interaction（§7/§15 的核心）

**(a) Mosaic 不稀释。** `Mosaic.get_indexes(buffer=True)` 取的是 `dataset.buffer`，
而 `buffer` 是 `load_image` 里 **按 sampler 顺序 append** 的 FIFO（上限 64）。因此侧面图与中心图
**同分布**：`slot small% ≈ centre small%`（24.82 vs 25.06；40.26 vs 40.57）。

```text
config   draw small%   BUFFER small%   naive draw gain   measured mosaic gain
baseline     25.06%        24.82%            1.000                1.000
w=1.25       29.48%        30.06%            1.176                1.218
w=1.50       33.41%        33.32%            1.333                1.340
w=1.75       36.92%        37.25%            1.473                1.473
w=2.00       40.08%        40.25%            1.599                1.634
```

**(b) 轻微放大机制**：侧面图被 append 回 buffer，形成 `1/4 新鲜 + 3/4 自重采样` 的反馈
⇒ buffer 的 small 占比略高于抽取占比 ⇒ 端到端增益 ≥ 解析增益。

**(c) 不能声称的部分**：`_mosaic4` 的 `random.uniform` 中心点 + `RandomPerspective` 缩放/平移会**裁掉**
部分贴片，`box_candidates` 还会过滤过小框 —— **像素级存活率本轮未模拟**。
```text
MOSAIC_EXPOSURE_CLAIM = 索引级 SUPPORTED / 像素级 NOT_SUPPORTED
```

---

## 7. Optimizer-step / LR-schedule integrity（§8/§9，硬门槛）

**Design A（在 dataset 里重复 small 图）**

| w | len(dataset) | steps/epoch | 总 steps | Δ vs baseline |
|---:|---:|---:|---:|---:|
| 1.00 | 1600 | 200 | 60,000 | — |
| 1.25 | 1700 | 213 | 63,900 | **+6.5%** |
| 1.50 | 1800 | 225 | 67,500 | **+12.5%** |
| 1.75 | 1901 | 238 | 71,400 | **+19.0%** |
| 2.00 | 2001 | 251 | 75,300 | **+25.5%** |

⇒ 同时改了 sampling 与 optimizer budget。**试图用"减少 epochs"补偿也不成立**：
ultralytics 的 `cos_lr` 是**按 epoch 索引**算的（`(1-cos(πx/epochs))/2`），
改 epochs 会改变 LR 曲线形状，且 `close_mosaic=10` / `warmup=3` 都是 epoch 单位 ⇒ 相位边界移动。
`PURE_SINGLE_VARIABLE = NO`。

**Design B（加权、长度不变）**：`steps/epoch=200`、`total=60,000`、`Δ=0` ✅ LR schedule 完全不变 ✅
**但代价见 §8 —— 覆盖率崩塌。**

### ★★★ 这是一个数学约束，不是实现瑕疵

> 从 N 个样本里抽 **恰好 N 次**（无论带不带权重）⇒ 每个样本的期望曝光恒为 1。
> **等长度 + 全覆盖 + 过采样，三者不可兼得。**

加权**置换**（Efraimidis–Spirakis 等）只能改**顺序**、不能改**多重集** ⇒ 曝光计数不变，无用。
⇒ 任何真正的过采样都**必须**增加抽取次数，因而必须在"更多 optimizer steps"与"更低 unique 覆盖"之间二选一。

---

## 8. Unique-image coverage（§12）

```text
config      uniq img/epoch   /1600   small uniq (of 401)   non-small uniq (of 1199)
baseline              1600   1.000         401   (100%)            1199   (100%)
w=1.25                1009   0.630         276   ( 68.8%)            732   ( 61.1%)
w=1.50                1001   0.626         294   ( 73.3%)            707   ( 59.0%)
w=1.75                 992   0.620         309   ( 77.1%)            683   ( 57.0%)
w=2.00                 980   0.613         319   ( 79.6%)            661   ( 55.1%)
```

**两个反直觉结果**：
1. 每 epoch 只看到 **61–63%** 的数据集（vs 100%）—— 训练从"跑遍全量一次"变成"反复跑 61% 子集"。
2. 即使把 small 图加权 2×，**每 epoch 仍有 20–31% 的 small 图一次都没被抽到**
   （基线是 0% 漏掉，因为无放回必全覆盖）。
   ⇒ 干预**增加了曝光总量却降低了曝光多样性**。
⚠ 注意"100 epoch 覆盖率"这种口径会**饱和到 1.000**、完全掩盖该现象 —— 必须用 per-epoch 尺度。

---

## 9. Class bias（§11）

**small 实例曝光 Δ%（w=2.0）**：person +66.4% / seat +74.1% / sign +94.2% / bicycle +83.9% /
boat +75.7% / light +73.0% / garbage_can +74.0% / uav +52.1% / **car +41.9%** /
**tricycle +46.2%** / **animal +22.8%** / ball n/a。极差 **71.4 pp**。

**但真正的问题在"全部实例"（含非 small）的曝光：**

| class | ALL slots baseline | w=2.0 | Δ% | small Δ% |
|---|---:|---:|---:|---:|
| person | 363,022 | 449,970 | +24.0% | +66.4% |
| **car** | 186,844 | 180,401 | **−3.4%** | +41.9% |
| **boat** | 43,071 | 38,119 | **−11.5%** | +75.7% |
| **ball** | 1,857 | 1,734 | **−6.6%** | n/a |
| **animal** | 100,346 | 99,477 | **−0.9%** | +22.8% |
| **seat** | 13,343 | 20,565 | **+54.1%** | +74.1% |
| light | 44,230 | 62,728 | +41.8% | +73.0% |
| garbage_can | 103,534 | 105,027 | +1.4% | +74.0% |

```text
CLASS_BIAS_RISK = HIGH
```
**这不是"small-object sampling"，而是"含 small 目标的图片的过采样"** ——
它按**图片**重分配曝光，因而整体类别构成被一起改变：`seat +54%` 的同时 `boat −11.5%`、`car −3.4%`。
§11 担心的"变成 uav/person 过采样"没有发生；**发生的是另一种、更隐蔽的类别再分配**。

---

## 10. DDP / reproducibility（§16/§17）

```text
DDP_SAMPLER_SAFETY = FAIL (未实现)
```
本项目当前**非 DDP**（`LOCAL_RANK=-1` ⇒ `sampler=None` ⇒ `RandomSampler`），所以本轮 run 不受影响。
但若将来用 DDP：`build_dataloader` 走 `DistributedSampler`，自定义加权 sampler 必须
① 实现 `set_epoch()`；② 保证每个 rank 拿到**不相交**的子集；③ 各 rank 的加权总量一致。
「加权 + DDP 无重复 + 无遗漏」是非平凡问题，**必须显式实现并测**，不能假设。
种子：`seed_worker` 会设置每个 worker 的 `random.seed`；自定义 sampler 需要在 `__iter__` 里用
**自己的** `random.Random(seed + epoch)`，否则会与 Mosaic 的全局 `random` 流互相干扰。

---

## 11. Implementation plan（§20，最小侵入）

风险阶梯：`既有钩子`（不存在）→ **`小的本地 sampler`（推荐）** → dataloader 重写 → dataset 重写。

```text
改动 1  ultralytics/data/build.py::build_dataloader(+1 参数 sampler=None，3 行)
        风险：低。val/test 调用不传 ⇒ 行为逐位不变。
改动 2  新增 WeightedSmallImageSampler（~40 行，放 ultralytics/data/build.py）
        必须暴露 len()==1600、__iter__ 用自身 RNG、可选 set_epoch()
改动 3  ultralytics/models/yolo/detect/train.py::get_dataloader：
        mode=="train" 且 config 开启时构造 sampler 并传入
改动 4  default.yaml + scripts/train.py：新增 1 个 opt-in 键（默认关闭且不消耗 RNG）
        —— 关闭时 D′ 数据流/RNG 逐位不变（与 ir_aug 同一模式，已有先例）
总计 ≈ 50–70 行；不触碰 base.py / augment.py / Mosaic / evaluator / validation
```
**不得**为了这个实验重构 dataloader 或 Mosaic。

---

## 12. Git / SHA provenance

```text
HEAD = 878cbcdeb40c92590b593e5b993885ce41735223   branch yyy
TRAINING_STARTED = NO        CUDA_USED = NO
D′ checkpoint / M4 checkpoint / IR-Aug config : 本轮未写入（git status 无相关条目）
本机无法观测云端进程/显存。
```
| SHA(16) | git | 文件 |
|---|---|---|
| `d7d2315ed4391c97` | **IGNORED** | `ultralytics/data/base.py` |
| `f014263f2bfe5921` | **IGNORED** | `ultralytics/data/build.py` |
| `abfc90f53f75f7ad` | tracked | `ultralytics/data/augment.py` |
| `71096a9e044652a5` | tracked | `ultralytics/models/yolo/detect/train.py` |
| `a4e329cfc3d22020` | tracked | `configs/train_rgbid_sepstem_clahe.yaml`（D′） |
| `9b14f29460733384` | tracked | `configs/yolo11m_sepstem.yaml` |
| `56c07c8520cf5b25` | **IGNORED** | `ultralytics/data/loaders.py` |

⚠ `.gitignore:11` 的 `ultralytics/data/*` ⇒ `build.py`/`base.py`/`loaders.py` **不进 git**，
本轮的 sampling 结论所依赖的**全部关键代码**都对 `git diff` 不可见，只能靠 SHA-256 取证。

---

## 13. Hard Gates

| # | Gate | 结果 | 依据 |
|---|---|:--:|---|
| 1 | Zero-GPU | **PASS** | `CUDA_VISIBLE_DEVICES=""`；只用 numpy/random/struct |
| 2 | No training | **PASS** | 只构建 dataset 索引 + Monte Carlo |
| 3 | D′ identity | **PASS** | 未写任何 checkpoint |
| 4 | Three-modality RGB+IR+Depth | **PASS** | 候选是 D′ 派生；`RGBID`/5ch/pairs 不动；只改**图片索引** |
| 5 | M4 remains RGB+IR only | **PASS** | M4 未触碰 |
| 6 | Small-object definition | **PASS（带 caveat）** | 复用 canonical；任务书 3 个数字不复现 |
| 7 | Sampling insertion point | **PASS** | `build_dataloader` sampler 参数，已被定位 |
| 8 | Pairing integrity | **PASS（结构性）** | 只换 index；三模态同一 stem 由 `replace(pairs_rgb,…)` 保证 |
| 9 | Mosaic interaction understood | **PASS** | 已实测：buffer FIFO，**不稀释**，1.634× |
| 10 | Optimizer-step integrity | **FAIL** | Design A **+6.5…+25.5%**；Design B 为 0% |
| 11 | LR-scheduler integrity | **PASS/FAIL** | Design B 不变；Design A 若补偿 epochs 则 cosine 形状改变 |
| 12 | Small-object exposure measurable | **PASS** | 1.218–1.634×，可复现 |
| 13 | Class-bias risk acceptable | **FAIL** | ALL-实例：seat +54.1% / boat −11.5% / ball −6.6% |
| 14 | Unique-image coverage | **FAIL** | Design B：100% → **61–63%**；每 epoch 漏 20–31% 的 small 图 |
| 15 | Validation isolation | **PASS** | val 走 `shuffle=False`，sampler 仅 `mode=="train"` |
| 16 | Seed reproducibility | **PASS（有条件）** | 需自定义 RNG + 不污染 Mosaic 的全局 `random` |
| 17 | DDP safety | **FAIL/UNVERIFIED** | 当前非 DDP；加权 + DDP 不相交子集未实现 |
| 18 | Minimal implementation | **PASS** | ≈50–70 行 |
| 19 | **Single-variable feasibility** | **FAIL** | **无任何一种设计同时满足 [10] 与 [14]** |
| 20 | No existing-route duplication | **FAIL** | 见 §14 |

---

## 14. ★ 与已关闭路线的重复（§20 / §26）

`ultralytics/data/augment.py:3958 class NativeSmallReplay`（**F1**）：

```text
阈值       : native area < 1024      ← 与本候选同一个 canonical 判据
配置       : native_small_replay: 1.0 , max_new_instances: 2
自述曝光   : "每图 <=2 个 ⇒ 总 exposure ≈ +57.6%"
线上结果   : 48.204  vs  D′ 48.712  =  −0.508   ⇒  路线 CLOSE
本地 fork  : −0.00205（近乎为零）
```

本候选 Design B **w=2.0 的端到端曝光 = +63.4%**，w=1.75 = +47.3%。
⇒ **本候选的整个可用区间（+21.8% … +63.4%）把已线上否决的 +57.6% 整个包住。**

**必须诚实标注这不等价**：F1 是**内容干预**（在图内复制 small 实例并新增标签），
本候选是**索引干预**（只改抽取频率，不合成任何内容）。两者机制不同，**F1 的否决不能直接推出本候选无效**。
但它把「提高 small 曝光」这一杠杆的**先验**从"未知"压到"在该量级上已试过一次、结果为负"。

叠加机制侧证据：P5/P6/P7/P8.5/P11 已判定 small-object 失败的主因是
**分类读出的 score vacuum（head / common-mode 动力学）**，而非"该目标没被训练到"。
若瓶颈在读出侧，则**提高曝光量不改变该机制** —— 与 F1 的线上结果方向一致。

---

## 15. 最终判决

```text
========================================
CASE B — CONDITIONAL
========================================
SMALL_SAMPLING_P0 = CONDITIONAL
GPU_CANDIDATE     = MAYBE
RECOMMENDED       = NOT YET (不得直接训练)
```

### blocking issues

1. **[19] 单变量不可行**：等长度 + 全覆盖 + 过采样三者数学上不可兼得（§7）。
   Design A → optimizer steps +6.5…+25.5%；Design B → 每 epoch 唯一覆盖 100%→61–63%。
2. **[13] 类别再分配**：干预按"图片"重分配曝光，`seat +54.1%` 与 `boat −11.5%` 同时发生，
   不是 small-object-specific 的干预。
3. **[20] 与已关闭的 F1 在曝光量级上重叠**（+57.6% 已被线上否决 −0.508）。
4. **[17] DDP 安全性未实现**（当前 run 不受影响，但属未验证项）。
5. **判据未决**：native（401 图）vs @1280（611 图）选出的是**不同的图片集合**，任务书假定了 native。

### required fix（若决定推进）

**Design C — 两臂匹配对照（唯一干净的设计）**

```text
Arm 1 : im_files = 1600 原图 + small 图各复制 1 份            → len 2001
Arm 2 : im_files = 1600 原图 + 随机 401 张各复制 1 份          → len 2001   ← 对照
两臂 : 300 epochs, 251 steps/epoch, 75,300 steps, 同一 cosine 曲线, 同一 mosaic/close_mosaic
唯一差异 = 【哪 401 张图获得额外曝光】
```
- 两臂**都保持 100% 唯一覆盖**（每 epoch 每个原图至少出现一次）⇒ [14] 不再是问题
- 两臂 steps 完全相同 ⇒ [10] 不再是问题
- 唯一变量 = "small 定向重复" vs "随机重复" ⇒ [19] PASS
- **代价：2 次 GPU run**（本容器 ≈ 6–11 h/次，D′ 规模）

若不愿付 2× 成本，则**不要做这个实验** —— 单臂版本无法把增益归因给 sampling。

### expected information gain（有界且不大）

若 Design C 的 Arm1 > Arm2：得到"**图片级重复**（无合成内容）有效"，
可与 F1 的**内容级**失败区分开 —— 这是唯一的新信息。
若 Arm1 ≈ Arm2：则 definitively 关闭"提高 small 曝光"这一整条杠杆（两种机制都已试过）。
⇒ **信噪比低、成本 2×、且与已否决路线同量级重叠。**

---

## 16. 最后回答

> **如果 D′ IR-Aug GPU 实验结束且结果需要选择下一项实验，D′ + Small-object Sampling 是否值得消耗下一次 GPU run？**

**不推荐作为"下一次"。** 三条可验证的理由：

1. **同一杠杆、同一量级、已经线上试过并失败**：F1 `NativeSmallReplay` 用**同一个** native-area<1024 判据、
   把 small 曝光推到 **+57.6%**，线上 **48.204 vs D′ 48.712（−0.508）**；本地 fork 也只给 −0.00205。
   本候选 Design B 的可用区间（+21.8%…+63.4%）把它整个包住。（机制不同：F1 是内容干预、本候选是索引干预 ——
   所以不是逻辑否决，是**先验被压到"该量级上已试过一次、结果为负"**。）
2. **干净版本要 2× GPU**：§7 证明了等长度+全覆盖+过采样不可兼得，所以单臂必然混杂
   （要么 steps +25.5%，要么唯一覆盖掉到 61%）。唯一单变量版本 Design C 需要对照臂 ⇒ 两次 run。
3. **机制侧不支持**：P5/P6/P7/P8.5/P11 已把 small-object 失败归因到**分类读出的 score vacuum**，
   而非"训练曝光不足"；提高曝光量不作用在该机制上。且本干预在 ALL 实例层面是类别再分配
   （seat +54.1% / boat −11.5%），连"small-object-specific"这个定性都不成立。

**什么会改变这个判断**：若能给出证据表明瓶颈**确实是 image-level exposure 而非读出侧**
（例如：D′ 在"含 small 图"上的 train loss 显著高于"不含 small 图"，且该 gap 随训练不收敛），
那么 Design C 就是值得付 2× 成本的那一次实验。**在此之前，下一次 GPU 应优先给别的假设。**

---

## HARD STOP

未训练、未推理、未开 CUDA、未改任何代码/配置/checkpoint、未触碰 M4 与 D′ IR-Aug。
未重新打开 Depth attribution / OASA / TTA / NMS / max_det / top-100 / score calibration / P2 / box×2 / D′ seed-2。
