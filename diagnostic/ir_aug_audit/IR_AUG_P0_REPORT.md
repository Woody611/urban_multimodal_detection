# D′ IR-Specific Augmentation — P0 Audit + Experiment Preparation

**日期**: 2026-10-04 · **任务性质**: CPU-only 审计 + 实验准备（**未训练、未占用 CUDA、未触碰 M4 Seed-2**）
**审计对象**: incumbent **D′ = RGB + IR + Depth（SepStem + CLAHE）**，官方 0.51528（本地官方口径）

全部证据脚本与中间产物在本目录：`ir_stats.py` / `pipeline_trace.py` / `sanity_and_matrix.py` /
`dryrun_gates.py` / `regression_ab.py` / `verify_run_args.py` + 对应 `*.json`。

---

## A. Executive Decision

```
IR_AUG_READY = PASS
GPU_TRAINING_ALLOWED = YES
GPU_LAUNCH = DO NOT LAUNCH  (本任务不启动；且当前 GPU 被 M4 Seed-2 占用 ⇒ 见 §J)
```

15/15 CPU dry-run gates PASS；回归 A/B 证明未启用该增强时 D′ 数据流**逐位不变**；身份测试
`max_abs_diff = 0`；推理/评测链路无任何引用。

---

## B. D′ Provenance（从 repo/artifacts 重新确认，非引用提示词）

```
D_PRIME_PROVENANCE
------------------
model config        : configs/yolo11m_sepstem.yaml
                      sha256 9b14f2946073338414b4441784b6df87a868b982be32ac8d50b6d1c4a17f3dd8
train config        : configs/train_rgbid_sepstem_clahe.yaml
                      sha256 a4e329cfc3d220206448cdd3377c1b3f5126c50f13c907ada522d98725486e12
experiment directory: runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/
best.pt             : runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt
best.pt SHA256      : 1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda
model/config SHA256 : 见上两行（本次审计前后均未变，已复核）
base.py SHA256      : bf380a22…76afe（改动前） → d7d2315e…4db88（改动后，见 §G）
loaders.py SHA256   : 56c07c8520cf5b255516afb30621fecbf3ebc5dc4c658a8dd9689ade537c3362（未改）
augment.py SHA256   : abfc90f53f75f7ad5a5c13e842847e6e6cbc8bcdab8b1be57e88f7cff801282b（未改）
tasks.py SHA256     : 79c8dbab9cd810d2609e7ff162d856beea8ab8e19b7ee6a933fb83ec1a617efd（未改）
training seed       : 42
imgsz               : 1280 (square 训练画布；val rect)
batch               : 8
epochs              : 300（patience=0，无早停）
IR encoding         : clahe  (CLAHE clipLimit=2.0, tileGridSize=(8,8))
Depth preprocessing : 16-bit/8-bit → uint8；<300mm 置 0；clip(d/19999*255)
RGB augmentation    : 5ch 下**全部关闭**（见 §L）
official evaluator  : scripts/official_eval.py（纯 TXT 打分器，不加载模型/数据集）
official score      : 0.51528（本地官方口径；线上 48.712，**新测试集**）
```

**训练数据**（args.yaml 指向云上 `/root/autodl-tmp/.../rgbid_split_train/dataset.yaml`；本地同构副本
`data/processed/rgbid_split_train/`，**1600 train / 400 val**，三模态各 1600/400）。provenance 闭环 ✅。

---

## C. Actual Pipeline（真实 source tracing，非 config 猜测）

`BaseDataset.load_image` → `load_and_preprocess_image` 的 **RGBID 分支**
（[base.py:488-529](ultralytics/data/base.py#L488-L529)）：

```
visible  = imread(path)                          # BGR      (H,W,3) uint8
infrared = imread(path.replace(...), GRAYSCALE)  # ⚠ (H,W,1) uint8 —— patches.imread 强制 3 维
depth    = imread(path.replace(...), UNCHANGED)

IR   : dtype 统一（uint16→×255/65535；其余 min-max→uint8）
       → apply_ir_encoding(im, "clahe")           # base.py:59，输入 (H,W,1) → **输出 (H,W)**
       → [★ 推荐插入点] apply_ir_augmentation(...)  # base.py:516（本次新增，train-only）
Depth: → uint8 (<300mm→0)
       → _resize_images_3 （本数据集三模态同尺寸 ⇒ 实际 no-op）
       → _merge_channels_rgbid → (H,W,5) uint8  [B,G,R,IR,D]

load_image      → rect resize 长边到 1280（×2, INTER_LINEAR）
__getitem__     → v8_transforms Compose → Format（uint8 保持，BGR→**RGB**）
trainer         → batch["img"].float()/255      # detect/train.py:671 ⇒ 模型输入 ∈ [0,1]
```

**关键实测（`pipeline_trace.json`）**：IR 进入 `apply_ir_encoding` 的 shape 是 **(360,640,1)**，
CLAHE 之后变成 **(360,640)**。这是 fork 的 `ultralytics.utils.patches.imread`
（`im[..., None] if im.ndim == 2`）造成的，**不是** config 能看出来的。

---

## D. CLAHE 位置 — 已用代码证明

```
CLAHE_POSITION = raw IR（未归一化、uint8、(H,W,1)）上；
                 在 **geometric augmentation 之前**、**tensor conversion 之前**、
                 **normalization 之前**、**merge 之前**。
位置 = 选项 A（raw IR 上），不是 normalization 后、不是 tensor 后。
```

调用链（全部可点开）：

| # | 位置 | 证据 |
|---|---|---|
| 1 | `apply_ir_encoding(im, mode)` | [base.py:59-86](ultralytics/data/base.py#L59-L86) — `cv2.createCLAHE(2.0,(8,8)).apply(im)` at L82 |
| 2 | RGBID 训练/val 调用点 | [base.py:515](ultralytics/data/base.py#L515) — `im_infrared = apply_ir_encoding(im_infrared, ir_encoding)` |
| 3 | 推理调用点（**独立的第二份实现**） | [loaders.py:653-689](ultralytics/data/loaders.py#L653-L689) — L666-671 **内联** CLAHE/percentile，**不调用** `apply_ir_encoding` |
| 4 | 归一化在 CLAHE **之后很远** | [detect/train.py:671](ultralytics/models/yolo/detect/train.py#L671) `/255` |
| 5 | 几何增强在 CLAHE **之后** | `v8_transforms` 的 Compose（Mosaic/RandomPerspective/RandomFlip） |

> 注：`apply_ir_encoding` 返回的 shape 对 CLAHE 是 `(H,W)`、对 percentile 是 `(H,W,1)` —— 一个既有的
> 形状不一致，`cv2.merge` 两者都能吃，故 D′ 不受影响；**本报告的实现刻意同时接受这两种形状**。

---

## E. Gamma / Noise 插入点裁定

```
RECOMMENDED_INSERTION_POINT = Candidate B
    raw IR → CLAHE → **gamma → noise** → merge → resize → 模型
    （证据：该数组就是模型实际消费的 IR 表示；见下）

NOISE_INSERTION = 与 gamma 同点（CLAHE 之后）—— **不**放在 CLAHE 之前
```

### 候选点对比

| 候选 | 位置 | 输入 dtype/range | 结论 |
|---|---|---|---|
| **A** | raw IR → **gamma/noise** → CLAHE | uint8 `(H,W,1)` [0,255] | ❌ 否决 |
| **B** | raw IR → CLAHE → **gamma/noise** → merge | uint8 `(H,W)` [0,255] | ✅ **采纳** |
| C | post-merge 5ch 数组的 ch3（Compose 内） | uint8 `(H,W,5)` | ❌ 否决（见下） |

**为什么是 B（证据）**

1. **这就是模型看到的分布**。CLAHE **之后**的数组经 merge → resize → `/255` 直接进 stem。
   gamma/noise 作用在它上面 = 直接在"模型输入邻域"里做抖动，与被增强对象严格对应。
2. **A 的强度会被 CLAHE 部分抵消且不可控**。CLAHE 是**局部**直方图均衡，会把全局单调曲线
   部分重均衡掉；A 的有效强度既依赖 CLAHE 的实现又逐块变化，不再是一个受控单变量。
3. **noise 若放在 A，会被 CLAHE 的局部增益放大**，而 CLAHE 的局部增益无上界且**空间变化**
   → 实际噪声 σ 不再是配置值。B 则让 σ 严格等于配置值（实测证实，见 §I）。
4. **B 的最小 diff**：一个函数 + 一个调用点（[base.py:516](ultralytics/data/base.py#L516)），
   与既有 IR 处理（CLAHE、modality dropout）**同一结构位置**，都在 `load_and_preprocess_image` 内。
5. **C 否决理由**：要把新模块塞进 `v8_transforms` 的 Compose；它会作用在 Mosaic 之后的画布
   （含 114 边框填充）且新增一个 RNG 消费者，属于更大的结构改动，却没有换来更强的可控性
   （实测每样本恰好 1 次调用，B 的概率语义已经干净，见 §I-4）。

**不会破坏 D′ 已有 IR 表征**：gamma 是对数域的单调色调映射，不改变通道语义、不改变
CLAHE 本身；noise 只在 1/255 粒度上扰动。**train/val distribution mismatch**：val 不加抖动
（`augment=False` 短路），这是**刻意的 augmentation gap**，不是污染 —— 符合 §11 的设计。

---

## F. Gamma 实现审计

```python
x   = im.astype(np.float32) / 255.0          # uint8 → float, [0,255] → [0,1]
x   = np.power(x, gamma)                     # gamma = uniform(lo, hi)
x   = np.clip(x, 0.0, 1.0)                   # 饱和
out = np.clip(x * 255.0, 0, 255).astype(np.uint8)   # → uint8，与输入同 shape
```

**没有**直接 `x ** gamma`（那会在 [0,255] 域上做，gamma>1 直接溢出）。

| gamma | 语义 | 实测（mid_gray 128） |
|---|---|---|
| `< 1` | 提亮中间调、抬升暗部 | 0.75 → 152 |
| `= 1` | **恒等** | 128 → 128（`all_values` 全 256 级逐位相同） |
| `> 1` | 压暗中间调、压死暗部 | 1.35 → 100 |

**合成 sanity（`sanity_and_matrix.json::synthetic`）**：black / dark_gray / mid_gray / bright_gray /
white / gradient / 全 256 级 —— γ∈{0.5,0.75,0.85,1.0,1.2,1.35,2.0} 全部
**finite=True、dtype=uint8、shape 不变、无 NaN/Inf**；`0→0`、`255→255` 恒为不动点；
`gradient` 输出**单调不减**；**γ=1.0 对全部 7 个 pattern 逐位 identity**；clip 行为正确。

---

## G. 参数推荐（每个数字都有本项目证据，未照搬外部 repo）

```
SAFE_GAMMA_RANGE = [0.75, 1.35]     （LOW=0.75 · CENTER=1.0 · HIGH=1.35）
RECOMMENDED_GAMMA = [0.75, 1.35]
RECOMMENDED_GAMMA_PROBABILITY = 0.35
RECOMMENDED_NOISE_STD = 0.015       （单位 = 归一化 [0,1] 域 ⇒ 灰度级 σ = 3.825）
RECOMMENDED_NOISE_PROBABILITY = 0.20
```

### §G-1 gamma 范围：用"自然曝光跨度"当尺子

对 120 张真实 train IR：gamma 诱导的**曝光伸缩比**（`mean(x^γ)/mean(x)`）与数据本身
**图与图之间的天然曝光跨度**比较：

| 量 | 值 |
|---|---|
| 数据天然 per-image mean | p05=0.181, p50=0.250, p95=0.488 |
| **天然曝光跨度 p95/p05** | **2.690×** |
| γ=0.75 诱导 | 1.304× |
| γ=1.35 诱导 | 0.710× |
| **γ∈[0.75,1.35] 诱导跨度 = 1.304/0.710** | **1.837×** |
| （对照）γ=0.5 → 1.735×，γ=2.0 → 0.403× ⇒ 跨度 4.30× | 超出自然流形 |

**结论**：`[0.75,1.35]` 诱导的色调伸缩跨度 **1.84× < 自然跨度 2.69×** ⇒ 落在数据自身流形**之内**，
是保守的 tone jitter。自然包络约在 `[0.55, 1.5]`；`[0.75,1.35]` 稳稳在内。**采纳外部 repo 的
范围是被本项目数据支持的，而不是因为"别人用了"**；反之 γ≤0.5 / ≥2.0 在本项目**明确拒绝**。

### §G-2 noise 数值空间（**锁死**）

```
noise_domain    = 归一化 [0,1]（即 im/255），**不是** [0,255]
noise_std_units = 同一个 [0,1] 域；ir_noise_std: 0.015  ⇒  灰度级 σ = 0.015×255 = 3.825
```
若按 [0,255] 域解释，0.015 灰度级 = 0.0059% 满量程 —— 完全不可见，显然不是该参数的本意。
**合成实测**：σ=0.015 时 10×10 mid_gray 测得的 σ = **3.7549 灰度级**（预期 3.825，差 1.8% 来自
uint8 取整）✅；σ=0 时 **逐位 identity** ✅。无 NaN/Inf、dtype/σ 不受影响、种子可复现（G9b）。

### §G-3 σ=0.015 在本数据上是否合理？—— 边界占用率证据

| arm | frac@0 | frac@255 |
|---|---|---|
| CLAHE only（D′） | 0.000009 | 0.000642 |
| gamma only | 0.024947 | 0.000642 |
| **noise only（σ=3.825）** | **0.063843** | 0.000728 |
| gamma+noise | 0.045711 | 0.000764 |
| **raw IR 本身** | **0.198994** | — |

CLAHE 输出有**硬地板**（per-image p01 中位 = 2 灰度级），所以噪声把约 6.4% 的像素压到 0。
**但这不构成伪影**：raw IR **本身就有 19.9% 的像素恰好为 0**（大面积全黑区），CLAHE 把它们
抬到 ~2-3；噪声只是把它们送回 0 —— **比原始分布自身的零质量还少 3 倍**。
顶部 headroom 充足（p99 中位 231/255），frac@255 几乎不变 ⇒ **无 contrast explosion、无
dynamic range collapse**（std 保持在 0.19–0.21，见 `sanity_and_matrix.json::interaction`）。

### §G-4 概率语义（**纠正一个常见误解**）

实测 **每训练样本恰好 1.0 次** `load_and_preprocess_image` 调用（40 samples → 40 calls；
120 samples → 119 calls），因为 Mosaic 从 `dataset.buffer` 取 index，而 `self.ims` 已缓存
预处理结果（[base.py:559-565](ultralytics/data/base.py#L559-L565)）。

> ⇒ **`ir_gamma_probability` 就是「被增强图像的比例」，不存在 Mosaic 放大。**
> 120 样本实测：gamma 率 0.342（名义 0.35）、noise 率 0.217（名义 0.20）；gamma 抽样
> mean=1.066 ∈ [0.780, 1.346] ⊂ [0.75,1.35] ✅

---

## H. 与既有 D′ augmentation 的兼容性（§12）

**实测 introspect D′ 的 5ch train Compose**：

```
NativeSmallReplay(p=0 恒等) · ObjectScaleAug(enabled=False) ·
Compose(Mosaic p=1.0, CopyPaste p=0, RandomPerspective) · MixUp(p=0)
· Albumentations  p=0  transform=None        ← 5ch 完全关闭
· RandomHSV       h=0.015 s=0.7 v=0.4        ← 但 shape[-1]!=3 时**立即 return**
· RandomFlip ×2 · Format
```

- **重复作用？不会。** 本增强在 **load 阶段、merge 之前**；Mosaic / RandomPerspective /
  RandomFlip 都作用在合并后的 5ch 数组上；Albumentations `p=0`；RandomHSV 在
  [augment.py:3497](ultralytics/data/augment.py#L3497) `if img.shape[-1] != 3: return labels` 直接跳过。
- **RGB 被误伤？不会。** 增强只作用于 `im_infrared`（merge 前），G5/G6 逐位验证 RGB/Depth 不变。
- **IR-specific 是否失效？不会。** 独立可控，且**是 D′ 5ch 唯一的训练期光度变换**。
  （`brightness: 0.2` 只在 `channels==1` 时生效，5ch 走 RandomHSV 分支 —— 也是 inert。）

---

## I. 实现（本次新增的代码）

### I-1 最小 diff（machine-readable）

```diff
# configs/train_rgbid_sepstem_clahe.yaml  →  configs/train_rgbid_sepstem_clahe_iraug.yaml
-experiment_name: "urban_multimodal_det_yolo11_rgbid_sepstem_clahe"
+experiment_name: "urban_multimodal_det_yolo11_rgbid_sepstem_clahe_iraug"
+ir_gamma: [0.75, 1.35]
+ir_gamma_probability: 0.35
+ir_noise_std: 0.015
+ir_noise_probability: 0.20
```
✅ 机器校验（`yaml.safe_load` 后按 key 比）：`added = 上述 4 键`、`removed = []`、
`changed = {experiment_name}` —— **恰好一个自变量**。model_config **完全相同**
（`configs/yolo11m_sepstem.yaml`）。

`default.yaml` 新增（**默认即恒等+关闭 ⇒ 零 RNG 开销**）：
```yaml
ir_gamma: [1.0, 1.0]
ir_gamma_probability: 0.0
ir_noise_std: 0.0
ir_noise_probability: 0.0
```

### I-2 代码改动清单（before → after SHA）

| 文件 | before SHA256 | after SHA256 | 改动 |
|---|---|---|---|
| `ultralytics/data/base.py` | `bf380a22…76afe` | `d7d2315e…4db88` | +113 行（纯新增：`apply_ir_augmentation` + `_ir_gamma_range` + 计数器），1 处调用点 |
| `scripts/train.py` | `67cb9fe0…90dd` | `d368f6c6…3212` | +19 行（仅当配置含新键时转发为 kwargs） |
| `ultralytics/cfg/default.yaml` | `cf06d3a6…3d3d4` | `f119165d…04d8` | +14 行（4 个默认键 + 注释） |

完整 patch: `diff_base_py.patch`（134 行）/ `diff_train_py.patch`（31 行）。
⚠ `ultralytics/data/base.py` **不在 git 追踪内**（`.gitignore:4` 的无锚定 `data/`；
见 `reports/OASA_PHASE1_AUDIT.md §5 R2`）⇒ 改动前副本已备份到
`diagnostic/ir_aug_audit/_before/ultralytics_data_base.py`，SHA 记录在 `SHA_BEFORE.txt`。

### I-3 不受影响的产物（回归护栏）

| 对象 | SHA（本次审计前后一致） |
|---|---|
| D′ `best.pt` | `1cae45f7…e4fda` |
| `configs/train_rgbid_sepstem_clahe.yaml` | `a4e329cf…86e12` |
| `configs/yolo11m_sepstem.yaml` | `9b14f294…f3dd8` |
| **M4 Seed-2（未触碰）** | `configs/train_modality_m4_rgb_ir_seed2.yaml` = `e7e88a89…2814`；`runs/..._m4_rgb_ir_seed2/` mtime 仍为 2026-10-04 22:28（早于本次会话） |

### I-4 旧码 vs 新码 A/B（`regression_ab.py`）

把改动前的 `base.py` 作为独立模块载入，与当前实现同种子、同样本对比：

```
[A] load_and_preprocess_image   old-vs-new  n=20  max_abs_diff = 0   differing = 0
[B] end-to-end train tensor     old-vs-new  n=5   max_abs_diff = 0   differing = 0
VERDICT: IDENTICAL
```
⇒ **未启用该增强时，D′ 的数据流逐位不变**（不只是"应该没问题"）。

---

## J. CPU dry-run 结果（§17 全 13 门 + 2 门加强）

```
==== 15/15 gates PASS ====
G1  config parse            PASS  ir_gamma=[0.75,1.35] p=0.35 std=0.015 p=0.2
G2  dataset loads           PASS  1599 train samples
G3  train transform         PASS  tensor [5,1280,1280] uint8
G4  IR aug executes         PASS  calls=119/120（gamma 41, noise 26）
G5  RGB unchanged           PASS  逐位（aug 开启条件下，ch0/1/2 相同）
G6  Depth unchanged         PASS  逐位（ch4 相同）
G7  val transform unchanged PASS  max_abs_diff = 0.0；val.augment=False
G8  inference path          PASS  loaders.py 中 apply_ir_augmentation 引用数 = 0
G9  determinism             PASS  同种子→同 tensor（全新 dataset 实例）
G9b IR aug determinism      PASS  同种子→同输出，且 ≠ 输入
G10 identity (disabled vs γ=1,σ=0)  PASS  n=4  max_abs_diff=0  mean_abs_diff=0.0
G11 no NaN/Inf              PASS  8 样本全 finite
G12 shape unchanged         PASS  (360,640,5) uint8
G13 channel ordering        PASS  实测 [R, G, B, IR, D]；IR=idx3, D=idx4
```
**实测调用率**：preprocess calls/sample = 0.992（≈1.0）；gamma rate 0.342 / nominal 0.35；
noise rate 0.217 / nominal 0.20。

### §J-1 G13 的一处**文档/实现不一致**（必须记录）

任务书写的是 5 通道顺序 `[B, G, R, IR, D]`，`configs/yolo11m_sepstem.yaml` 的注释也这么写。
但 `Format._format_img` 的 5ch 分支（[augment.py:3259-3262](ultralytics/data/augment.py#L3259-L3262)）
做的是 `transpose(2,0,1)[:3][::-1]` = **BGR → RGB**，实测张量顺序是 **`[R, G, B, IR, D]`**。

- 这是 **D′ 既有行为**，**本次改动没有触碰 `Format`**，顺序在改动前后完全一致。
- 与 `SilenceChannel[0,3] → RGB stem（预训练于 RGB）` 的语义**自洽**；yaml 注释是陈旧的。
- **对 IR augmentation 无影响**：IR 恒在 idx 3、Depth 恒在 idx 4。

---

## K. 实验定义（§14/§15/§16）

```
EXPERIMENT_NAME : D_prime_IR_Aug_gamma_noise
Baseline        : D′（configs/train_rgbid_sepstem_clahe.yaml + configs/yolo11m_sepstem.yaml）
Intervention    : IR-specific gamma(0.75–1.35, p=0.35) + noise(σ=0.015, p=0.20)   ← **仅训练**
Everything else : D′ exact
RECOMMENDED_STRATEGY = A（combined）
```

**为什么是 A 而不是 B**：noise 的数值域已锁死、σ 已实测（3.7549/3.825）、边界行为已被
"raw IR 本身 19.9% 全黑"合理化 ⇒ **noise 风险并不高于 gamma**，无需降级为 gamma-only。

**Strategy C 不成立**：所有 gate PASS。

**Seed（§16）**：`baseline D′ seed = 42` → **候选实验 seed = 42**（预注册，逐字继承）。
不依据任何结果反向挑种子；D′ 的 Seed-2 体系未启用（M4 Seed-2 是另一条并行线，与本实验无关）。

**所有冻结项**：model / loss / TAL / optimizer / LR / scheduler / epochs / batch / imgsz /
seed / Depth preprocessing / RGB augmentation / val / inference / submission —— 逐字不变。
`unexpected_changes = 0`。

---

## L. GPU 命令（`IR_AUG_READY=PASS` 才提供；**本任务不执行**）

```bash
# 0) 训练前（云上）—— 与正式训练同一条解析路径
python scripts/preflight_check_train_config.py \
    --model_config configs/yolo11m_sepstem.yaml \
    --train_config configs/train_rgbid_sepstem_clahe_iraug.yaml \
    --expect-ir-encoding clahe
# 期望 RESULT: ALL CHECKS PASSED

# 1) 训练（唯一新增变量 = IR gamma/noise）
python scripts/train.py \
    --model_config configs/yolo11m_sepstem.yaml \
    --train_config configs/train_rgbid_sepstem_clahe_iraug.yaml

# 2) 开训后**立刻**验证 args.yaml 真的带上了 IR aug（防 2026-09-20 同构事故）
python diagnostic/ir_aug_audit/verify_run_args.py \
    runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_iraug/args.yaml
# 期望 IR_AUG_ACTIVE = YES
```

> ⚠ 若停止本次训练也不影响 D′ / M4 Seed-2；它是**独立目录 + 独立配置文件**，不覆盖任何既有产物。

---

## M. 预注册成功判据（§21-I）

```
PRIMARY   : scripts/official_eval.py 的官方口径分数（本地），对照 D′ 0.51528
SECONDARY : mAP50 / mAP75 / AP50-95
            AP by size (small / medium / large)
            IoU-level attribution（按 IoU 阈值分桶的 AP 差）
            线上提交分（同集对照，禁止跨测试集相减）
判读纪律  : 本地官方口径在同架构族内曾 4 次给出错误符号（见 metric-instrument 记录），
            故**提交前用 fork 口径交叉验证**；结论以线上为准。
```

---

## N. 结论（三选一）—— CASE B：READY BUT DEFER

```
CASE B — READY BUT DEFER
理由: 技术上全部门 PASS，但 GPU 当前被并行运行的 **M4 Seed-2** 占用，
      且本任务明令禁止占用 CUDA / 启动训练。
```

```
========================================
D′ IR-SPECIFIC AUGMENTATION P0 DECISION
========================================

D_PRIME_BASELINE:
    official = 0.51528   (best.pt sha256 1cae45f7…e4fda)

PIPELINE_AUDIT:
    PASS

CLAHE_POSITION:
    raw IR (uint8, (H,W,1)) 上；geometric aug / normalization / tensor 之前、merge 之前
    训练+val 走 apply_ir_encoding (base.py:59/515)；推理走 loaders.py:653-689 的内联副本

GAMMA_INSERTION:
    B — CLAHE 之后、merge 之前（base.py:516）；作用在模型实际看到的 IR 表示上

NOISE_INSERTION:
    B — 与 gamma 同点（CLAHE 之后）；不放在 CLAHE 之前（避免 CLAHE 局部增益放大 σ）

NUMERICAL_DOMAIN:
    PASS   (gamma: x=im/255∈[0,1] → x^γ → clip; noise: σ 单位 = [0,1] 域, 0.015 ⇒ 3.825 灰度级)

TRAIN_ONLY:
    PASS   (augment=False 短路；loaders.py 零引用；默认键恒等且零 RNG)

VAL_INFERENCE_UNCHANGED:
    PASS   (val max_abs_diff = 0.0；official_eval.py 是纯 TXT 打分器)

RGB_UNCHANGED:
    PASS   (逐位，aug 开启条件下)

DEPTH_UNCHANGED:
    PASS   (逐位)

IDENTITY_TEST:
    PASS   (disabled vs γ=1,σ=0：max_abs_diff = 0, mean_abs_diff = 0, n = 4)

DETERMINISM:
    PASS   (同种子 → 同 tensor；同种子 → 同 IR aug 输出)

PROVENANCE:
    PASS   (D′ config/model/best.pt SHA 全部从 repo 复核；数据 split 1600/400 闭环)

MINIMAL_DIFF:
    PASS   (added = 4 个 IR aug 键; changed = experiment_name; removed = [])

CPU_DRY_RUN:
    PASS   (15/15 gates)

RECOMMENDED_GAMMA:
    [0.75, 1.35]  (诱导曝光跨度 1.84× < 自然跨度 2.69× ⇒ 落在数据流形内)

RECOMMENDED_GAMMA_PROBABILITY:
    0.35  (每样本恰好 1 次预处理 ⇒ 字面比例，无 Mosaic 放大)

RECOMMENDED_NOISE_STD:
    0.015  (单位 = 归一化 [0,1]；= 3.825 灰度级；raw IR 本身 19.9% 像素为 0)

RECOMMENDED_NOISE_PROBABILITY:
    0.20

IR_AUG_READY:
    PASS

GPU_TRAINING_ALLOWED:
    YES

GPU_LAUNCH:
    DEFER   (M4 Seed-2 占用 GPU；本任务禁止启动训练)

BLOCKERS:
    无技术 blocker。
    唯一非技术阻塞：并发 M4 Seed-2 正在占用 GPU。

NEXT_ACTION:
    待 M4 Seed-2 结束后，按 §L 的 3 条命令执行（preflight → train → verify_run_args）；
    训练前后均用 official_eval.py 对照 D′ 0.51528。

========================================
```

---

## O. 假设状态（§23）

```
HYPOTHESIS:
    IR-specific augmentation may improve robustness of IR representation
    and potentially improve raw-head candidate generation
    (P3 已证：small GT 25.61% 在 pre-NMS 就无可用候选 ⇒ 表征/raw-head 是真实瓶颈).

STATUS:
    UNTESTED
```
本 P0 任务**只**证明该增强被正确、单变量、可回滚地准备好；**不**主张它一定能改善
small-object 瓶颈。
