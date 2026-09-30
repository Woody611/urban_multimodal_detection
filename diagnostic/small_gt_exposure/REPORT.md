# D′ Small-GT Effective Exposure Audit

**日期**：2026-09-28
**性质**：**READ-ONLY diagnostic**。0 训练 / 0 finetune / 0 optimizer step / 0 backward /
0 改 checkpoint / 0 改模型结构 / 0 改 augment.py / base.py / train.py / default.yaml /
0 改 D′ 配置 / 0 改 evaluator / 0 test inference / 0 submission / 0 改动任何已有实验结果。

回答的唯一核心问题：

> 在 D′ 的真实训练 pipeline 中，一个 **native small GT** 从原图进入 augmentation，
> 到最终进入 detection loss，实际获得了多少次、什么质量的有效 supervision？

---

## 0. 结论摘要

```text
裁决：A — STRONG EVIDENCE（针对「native-small GT 存在明确、稳定的 effective-supervision deficit」）
对原假设「§13 的 healthy 数字是 survivorship 造成的」：PARTIALLY —— 真因是「分桶污染」，不是 survivorship。
```

三句话：

1. **augmentation 没有饿死 small GT。** Mosaic 把每个 GT 的实例化次数从 1/epoch 提到 **~4.0/epoch**，
   且该倍数对 small/medium/large 几乎**完全相同**（4.0246 / 4.0087 / 4.0034）；
   `zero-final% = 0.00%`（11,560 个 native GT 在 100 epoch 内没有一个完全无监督）。
2. **但 §13 的 healthy 数字确实是被抬高的 —— 抬高它的是分桶口径，不是存活率。**
   按增强后面积分桶时，Mosaic 的整体尺度收缩使 **66% 的 native-medium 目标掉进 "small" 桶**，
   把它们 medium 级的监督质量（n_pos 10 / align 0.688）混入 small 桶，
   于是一个真实的 (n_pos 4.0 / align 0.357) 被报成 (8.0 / 0.521)。
3. **监督劣势的所在 stage 是 assigner，不是 augmentation。**
   Mosaic 阶段 small-vs-large 差 2.2pp、RP 阶段差 3.3pp；而 assigner 阶段
   `best_align` 中位 **0.357 vs 0.826**、`n_pos` 中位 **4 vs 10**。
   其中 `n_pos` 的差距大部分是锚框几何的机械后果（`n_pos = min(topk, GT 内 anchor 数)`）。

---

## 1. FROZEN PROVENANCE

### 1.1 本轮审计用到的文件与 SHA256（前 16）

| 文件 | SHA256 | mtime | 相对 D′ best.pt |
|---|---|---|---|
| `configs/yolo11m_sepstem.yaml`（D′ model yaml） | `9b14f29460733384` | 2026-09-19 15:04 | 早于 ✅ |
| `configs/train_rgbid_sepstem_clahe.yaml`（D′ train yaml） | `a4e329cfc3d22020` | 2026-09-20 08:12 | 早于 ✅ |
| `ultralytics/data/augment.py` | `5cb9a407625921ce` | **2026-09-22 22:14** | **晚于 ⚠️** |
| `ultralytics/data/base.py` | `8bcf834930155bd5` | 2026-09-18 22:29 | 早于 ✅ |
| `ultralytics/data/loaders.py` | `f84bdcf29081e6f9` | 2026-09-19 12:00 | 早于 ✅ |
| `ultralytics/data/utils.py` | `7eda220f7c8d5e17` | 2026-08-25 11:16 | 早于 ✅ |
| `ultralytics/data/build.py` | `f014263f2bfe5921` | 2026-09-02 20:22 | 早于 ✅ |
| `ultralytics/data/dataset.py` | `0aba2bc14793ab85` | 2026-09-17 19:19 | 早于 ✅ |
| `ultralytics/utils/loss.py` | `0f092cf22a372f6d` | 2026-09-09 21:00 | 早于 ✅ |
| `ultralytics/utils/tal.py`（assigner） | `aae7e8ac438f00cd` | 2026-08-25 09:27 | 早于 ✅ |
| `ultralytics/utils/instance.py`（Instances） | `78a72d1f4354c29f` | 早于 ✅ | |
| `scripts/train.py` | `9f55b09f9e1229b6` | 早于 ✅ | |
| `ultralytics/cfg/default.yaml` | `991a89b32de769d1` | 早于 ✅ | |
| **D′ best.pt** | `1cae45f75693f54146e35c5f` | 2026-09-22 07:33 | 基准 |

**D′ best.pt = `1cae45f7…`，与 V11/V13 记录的 `1cae45f7…` 同一模型 ✅**（未被本轮或 E1 触碰）。

环境：Python 3.8.15 / torch 2.4.1+cpu（本机无 GPU，全部为 CPU 只读计算）。

### 1.2 ⚠️ 发现一个「晚于 D′ 训练」的文件改动 —— 已判定**不影响**本审计

`ultralytics/data/augment.py` 的 mtime（**2026-09-22 22:14**）**晚于** D′ best.pt（**2026-09-22 07:33**）
约 14.7 小时。按你的要求：**未擅自恢复、未覆盖**，先报告并判定是否影响本审计。

**改动内容**（以 git index 版本为对照，归一化行尾后 diff = +206 行）：

| 位置 | 改动 |
|---|---|
| `1512a1513,1703` | 新增 `ObjectScaleAug` 类（OASA，191 行） |
| `3769a3961` / `3771c3963` | `Mosaic(...)` 提升为局部变量 `_mosaic_inst`（**构造参数逐字相同**：`Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic, dtype=dtype)`），`pre_transform` 引用它 |
| `3802a3995,4007` | 在 `v8_transforms` 里构造 `_oasa` 并插入 Compose 首位 |
| `3803a4009` | Compose 列表新增 `_oasa` |

OASA 正是 2026-09-22 21:40 起的 OASA 实验（`configs/oasa_pre_mosaic.yaml` mtime 21:40）所需，时间线自洽。
⇒ **D′ 是用「无 OASA」的那一版训练的。**

**判定依据（不看注释，看代码 + 实测）**：

1. `ObjectScaleAug.__call__` 第一行即 `if not self.enabled: return labels`
   （`augment.py:1574-1576`）——**在任何 `random.*` 调用与任何数据操作之前返回**。
2. D′ 的 `args.yaml` 中**没有** `object_scale_aug` 键 ⇒ `getattr(hyp,"object_scale_aug",False)` = False
   ⇒ `enabled=False` ⇒ 零 RNG 消耗、零数据修改的 passthrough。
3. 唯一另一处改动是把内联 `Mosaic(...)` 提为变量，**构造参数与位置完全不变**。

⇒ **对 D′ 的配置而言，当前 augment.py 与 D′ 训练时的版本行为等价。**
本审计的 replay 因此成立（`_sge_verify.py` V1 进一步用「真实图像 vs 打桩」逐位验证了几何等价）。

### 1.3 D′ 真实 runtime hyp（来源 = `runs/…_sepstem_clahe/args.yaml`，**非**从 train.yaml 反推）

```
imgsz 1280   epochs 300   batch 8   seed 42   deterministic true   rect false
optimizer SGD   lr0 0.005   cos_lr true   amp true   patience 0
mosaic 1.0   close_mosaic 10   mixup 0.0   copy_paste 0.0
degrees 0.0   translate 0.1   scale 0.5   shear 0.0   perspective 0.0
flipud 0.0    fliplr 0.5      erasing 0.4 (5ch 下 alb p=0 ⇒ 失效)
hsv_h/s/v 0.015/0.7/0.4 (5ch 下 RandomHSV 直接 return ⇒ 失效)
channels 5   use_simotm RGBID   ir_encoding clahe   object_scale_aug 缺省=false
```

### 1.4 实测确认的真实 pipeline（顺序 + 每步 p）

```
load_image(长边缩放到 1280) → [0] ObjectScaleAug(off)
  → [1] pre_transform = Compose[ Mosaic(p=1.0) → CopyPaste(p=0.0) →
                                RandomPerspective(translate .1, scale .5, 其余 0,
                                                  pre_transform=LetterBox(1280)) ]
  → [2] MixUp(p=0.0) → [3] Albumentations(p=0) → [4] RandomHSV(5ch 直接 return)
  → [5] RandomFlip(vertical, p=0.0) → [6] RandomFlip(horizontal, p=0.5)
  → [7] Format
```
`close_mosaic`（最后 10/300 epoch）经 `YOLODataset.close_mosaic` 置 `hyp.mosaic=0.0` 并
**重建 transforms**（`dataset.py:197-202`，已实读确认）⇒ 该期无 Mosaic，且
`mosaic_border` 不再存在 ⇒ `RandomPerspective.pre_transform=LetterBox` 转为生效。
本审计对这两个 regime **分别 replay**，再按 290/300 + 10/300 加权。

### 1.5 两处真实 GT 丢弃点（实读）

| # | 位置 | 条件 |
|---|---|---|
| 1 | `Mosaic._cat_labels` → `instances.clip(2·imgsz)` → `remove_zero_area_boxes()` | bbox 被拼图边界裁到零面积 |
| 2 | `RandomPerspective.__call__` → `box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.10)` | `box2` 面积 < `box1` 的 **10%**、或任一边 < 2px、或长宽比 > 100 |

> `area_thr=0.10` 是对 **`scale` 归一后的** box1 而言（`instances.scale(scale)`），
> 因此它在数学上等价于「面积被压到 <10%」的过滤；这是 RandomPerspective 唯一的丢弃机制。

### 1.6 一个必须记录的机制事实：Mosaic 的 tile 来自「最近加载过的 64 张图」

`Mosaic.get_indexes()` = `random.choices(list(dataset.buffer), k=3)`
（`augment.py:3337`），而 `dataset.buffer` 是 `BaseDataset.load_image` 维护的 FIFO，
容量 `max_buffer_length = min(ni, batch_size*8, 1000)` = **min(1599, 64, 1000) = 64**
（`base.py:166`）。

⇒ **Mosaic 的 3 个 companion 不是全数据集均匀采样，而是「本 worker 最近加载过的 64 张」**。
本审计的 replay 逐行镜像了该 buffer 机制（否则 `random.choices([])` 会直接崩，且会丢掉这一真实机制）。

### 1.7 与 V5 的关系（为什么必须重做）

`diagnostic/small_object_train_replay/_v5_replay.py` 的产物（§13 引用的
small n=1179 / n_pos 9 / best_align 0.568 等）**是按「增强后」面积分桶**的。
V5 自己在日志里记录了放弃身份追踪的原因：

```
映射校验: 可追踪 6 / 不可追踪 244
```
（判据「输出 GT 数 == Σ tile GT 数」在 244/250 个样本上不成立。）

⇒ V5 回答的是「**final 已经小**的 GT 监督好不好」；
本轮要用 native→final 身份回答「**native 小**的 GT 有多少活到了 final」。两者不可互推。

---

## 2. 方法：native→final 身份追踪（`_sge_instrument.py`）

不重写 augmentation，只挂载：

| 挂载点 | 作用 |
|---|---|
| `BaseDataset.get_image_and_label` | 给每个源图（主图 + 3 个 tile）的 instances 打上 `uid = index*100000 + k` |
| `Instances.concatenate` | Mosaic 四拼时同步搬运 uid |
| `Instances.__getitem__` | 布尔掩码过滤时同步搬运 uid |
| `Instances.remove_zero_area_boxes` | Mosaic 零面积过滤时同步搬运 uid |
| `RandomPerspective.__call__` | 内部 `new_instances = Instances(...)` 是**新建对象**，用 PENDING_uid 让其继承输入 uid（长度相同，构造性成立） |
| `Format.__call__` | final 快照（Format 会 pop 掉 instances，故必须在其内部前抓） |

**所有包装都是「先调原函数，再搬运 uid」；不改变任何计算。**

### 2.1 插桩正确性验证（`_sge_verify.py`，必须先 PASS）

| 检验 | 内容 | 结果 |
|---|---|---|
| **V1** STUB-EQUIVALENCE | 用「同形状零图」替代真实像素跑同一 pipeline，同 seed 下 final `cls` / `bboxes` / 空间 shape 必须逐位相同 | **96/96 逐位相同**（mosaic-ON/5ch、mosaic-OFF/5ch、mosaic-ON/1ch、mosaic-OFF/1ch 四组） |
| **V2** UID-CORRECTNESS | final 每个 GT 的 uid 解出的 `(source_index, k)`，其 native 类别必须 == final 类别（类别来自完全独立的代码路径） | **768/768 = 100.000%** |
| **V3** NO-DESYNC | uid 长度处处等于 instances 长度 | **desync 0**（`concat_no_uid 0`、`rp_pending_miss 0`） |

**打桩的合法性**：几何只依赖 `img.shape[:2]`（Mosaic 放置、RandomPerspective 的 M、
`box_candidates` 均只用形状与 bbox），且 D′ 的像素级增强全部失效。V1 已用真实图像逐位证明。
1ch 打桩（5ch→1ch）实测 **13× 加速**（122.7→9.5 ms/sample，mosaic-ON），bbox 仍逐位相同。

### 2.2 明确声明：REPLAY 的精确性边界

| 项 | 状态 |
|---|---|
| pipeline 结构 / 每步 p / 真实 transform 类 | **EXACT**（用 `build_yolo_dataset` 构造真实 `YOLODataset`，即训练同一条代码路径） |
| GT 身份 lineage | **EXACT**（V2 100% 独立校验） |
| 两处丢弃机制 | **EXACT**（原函数自己的返回值） |
| buffer 机制 | **机制 EXACT**（逐行镜像）；**但 buffer 内容不可复现** |
| 训练时的逐样本 RNG 流 | **不可复现**（跨 dataloader worker、跨 epoch 持久） |

⇒ **本审计是 Monte-Carlo 分布估计，不是 exact count。** 所有比例均为跨
`N` 个独立 epoch 的估计，`N` 在 §3 中给出。**不把近似 replay 写成「真实训练曝光」。**

---

## 3. Replay 配置

| 项 | 值 |
|---|---|
| dataset | `data/processed/rgbid_split_train/dataset.yaml`，**n = 1599**（1600 图，1 张 `003107.png` 因 non-normalized 坐标被 ultralytics 剔除，实测） |
| regime | mosaic-ON（p=1.0）、mosaic-OFF（p=0.0，对应 close_mosaic 期） |
| epochs（Monte-Carlo） | 每 regime 100 |
| RNG 协议 | `random.seed(BASE_SEED + 7919*rep)`、`np.random.seed(同)`，`BASE_SEED=20260928`；每 epoch 重新 shuffle 索引顺序（模拟 dataloader shuffle）；buffer **跨 epoch 持久**（模拟 persistent_workers） |
| 全 300ep 混合 | 290/300 × ON + 10/300 × OFF |

---

## 4. 结果

### 4.0 样本规模

| 项 | 值 |
|---|---:|
| native GT 总数 | **11,560** |
| ├ native-small（area < 1024 px²） | **1,392**（12.0%） |
| ├ medium（1024–9216） | 5,535 |
| └ large（> 9216） | 4,633 |
| mosaic-ON：100 epoch 的 final target 出现次数 | 2,080,894 |
| mosaic-OFF：100 epoch 的 final target 出现次数 | 1,089,432 |

### Table 1 — Pipeline survival（按 native sqrt(area) 分桶）

**mosaic-ON**（290/300 epoch 处于该 regime）

| native sqrt | n | inst/ep | P(inst) | mos\|inst | fin\|inst | **ExposureA** | fin→small |
|---|---:|---:|---:|---:|---:|---:|---:|
| **<8** | **0** | — | — | — | — | — | — |
| 8–12 | 17 | 4.0265 | 1.0000 | 0.9039 | 0.4254 | 0.8459 | 1.0000 |
| 12–18 | 238 | 4.0149 | 1.0000 | 0.8980 | 0.4285 | 0.8424 | 1.0000 |
| 18–24 | 431 | 4.0312 | 1.0000 | 0.8938 | 0.4271 | 0.8380 | 1.0000 |
| 24–32 | 706 | 4.0239 | 1.0000 | 0.8923 | 0.4312 | 0.8452 | 1.0000 |
| 32–96 | 5535 | 4.0087 | 1.0000 | 0.8992 | 0.4358 | 0.8473 | 0.5291 |
| >96 | 4633 | 4.0034 | 1.0000 | 0.9156 | 0.4709 | 0.8716 | 0.0053 |

**mosaic-OFF**（最后 10/300 epoch）

| native sqrt | n | inst/ep | mos\|inst | fin\|inst | ExposureA |
|---|---:|---:|---:|---:|---:|
| 8–12 | 17 | 1.0000 | 1.0000 | 0.9206 | 0.9206 |
| 12–18 | 238 | 1.0000 | 1.0000 | 0.9413 | 0.9413 |
| 18–24 | 431 | 1.0000 | 1.0000 | 0.9502 | 0.9502 |
| 24–32 | 706 | 1.0000 | 1.0000 | 0.9402 | 0.9402 |
| 32–96 | 5535 | 1.0000 | 1.0000 | 0.9335 | 0.9335 |
| >96 | 4633 | 1.0000 | 1.0000 | 0.9528 | 0.9528 |

> **`<8` 桶为空（n=0）**：本训练集**没有** native sqrt < 8px 的 GT；最小的桶是 8–12（n=17）。
> **`zero-final% = 0.00%`（所有类）**：11,560 个 native GT 在 100 个 epoch 里**没有一个**出现过
> 「整轮无 final target」。
> `fin→small` = 到达 final 的实例中面积仍 <1024 的比例（small 桶恒为 1.000，因为再缩小也还是 small）。

### §8/§9 — Mosaic 与 RandomPerspective(+scale) 对尺寸的作用

`ratio = sqrt(final area) / sqrt(native area 在 1280 长边空间下的值)`；1.0 = 未被缩放。

| regime | class | n_gt | **mosaic 阶段 ratio 中位** | **final 阶段 ratio 中位** | p10 | p90 | final <0.5× |
|---|---|---:|---:|---:|---:|---:|---:|
| ON | small | 1392 | **1.000** | **0.795** | 0.545 | 1.287 | **0.9%** |
| ON | medium | 5535 | 1.000 | 0.789 | 0.540 | 1.275 | 1.9% |
| ON | large | 4633 | 1.000 | **0.752** | 0.518 | 1.230 | **6.3%** |
| OFF | small | 1392 | 1.000 | **0.968** | 0.591 | 1.380 | 0.0% |
| OFF | medium | 5535 | 1.000 | 0.957 | 0.589 | 1.376 | 0.1% |
| OFF | large | 4633 | 1.000 | 0.942 | 0.589 | 1.364 | 0.3% |

「变小/不变/变大」（以 0.8× / 1.25× 为界）：

| regime | class | mosaic 后 | final 后 |
|---|---|---|---|
| ON | small | 变小 0.6% / 不变 99.4% | 变小 **50.7%** / 不变 37.3% / 变大 12.1% |
| ON | large | 变小 3.4% / 不变 96.6% | 变小 **56.1%** / 不变 34.8% / 变大 9.0% |
| OFF | small | 不变 100.0% | 变小 31.9% / 不变 46.3% / 变大 21.8% |
| OFF | large | 不变 100.0% | 变小 33.8% / 不变 46.9% / 变大 19.3% |

**读法（关键）**：

1. **Mosaic 阶段 ratio 恒为 1.000**（p10 = p90 = 1.000）⇒ Mosaic **不缩放**目标，只做
   「1:1 摆放 + 裁剪」。这同时验证了本审计的 `native_work` 基准定义是对的。
2. final 阶段 ON 的额外收缩来自 `RandomPerspective`：`M = T @ R @ C`，输出取 **2560² 画布的
   中心 1280² 窗口**（无 0.5 重采样），目标尺寸 × `s ~ U(0.5, 1.5)`。
   OFF regime 无窗口裁剪、只有 s ⇒ ratio 中位 **0.968 ≈ s 的中位 1.0** ✅
   （独立的机制自洽性检验）。
   ON 的 0.795 中位低于 OFF 的 0.968，其差额来自**部分裁剪的目标**（被窗口切掉一部分但
   `box_candidates` 仍保留：裁剪后面积 ≥ 未裁剪的 10%）⇒ 存活者面积被压低。
3. **尺寸损失不是 small 专属**：`<0.5×` 占比 small **0.9%** 反而**低于** large **6.3%**；
   「final 后变小」占比 small 50.7% 也低于 large 56.1%。**large 被缩得更狠。**

### Table 4 — Augmentation contribution（ON vs OFF，按类）

| class | n | inst/ep **ON** | inst/ep OFF | mosaic 丢弃 | RP 丢弃 **ON** | RP 丢弃 OFF | ExposureA ON | ExposureA OFF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **small** | 1392 | **4.0246** | 1.0000 | **0.1061** | **0.5178** | 0.0568 | 0.8425 | 0.9432 |
| medium | 5535 | 4.0087 | 1.0000 | 0.1008 | 0.5133 | 0.0665 | 0.8473 | 0.9335 |
| large | 4633 | 4.0034 | 1.0000 | 0.0844 | 0.4845 | 0.0472 | 0.8716 | 0.9528 |

### Table 3 — Relative disadvantage（全 300 epoch 混合：290/300 ON + 10/300 OFF）

| class | n | ExposureA(mix) | fin\|inst(mix) |
|---|---:|---:|---:|
| small | 1392 | 0.8458 | 0.4465 |
| medium | 5535 | 0.8502 | 0.4524 |
| large | 4633 | 0.8743 | 0.4870 |

**比值**：

| 比较 | ExposureA 比 | fin\|inst 比 |
|---|---:|---:|
| small / medium | 0.9949 | 0.9870 |
| small / large | **0.9674** | **0.9168** |

### 分层敏感性（native 像素面积会混淆 HR/LR 两个分辨率层）

| regime | layer | class | n | native sqrt 中位 | work sqrt 中位 | ExposureA | fin\|inst | inst/ep |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ON | HR | small | 1145 | 24.9 | 16.6 | 0.8397 | 0.4284 | 4.0277 |
| ON | HR | medium | 5361 | 58.3 | 38.9 | 0.8466 | 0.4351 | 4.0077 |
| ON | HR | large | 4633 | 165.7 | 110.5 | 0.8716 | 0.4709 | 4.0034 |
| ON | LR | small | 247 | 18.8 | 37.5 | 0.8553 | 0.4341 | 4.0105 |
| ON | LR | medium | 174 | 39.8 | 79.6 | 0.8702 | 0.4584 | 4.0398 |
| OFF | HR | small | 1145 | 24.9 | 16.6 | 0.9394 | 0.9394 | 1.0000 |
| OFF | HR | large | 4633 | 165.7 | 110.5 | 0.9528 | 0.9528 | 1.0000 |

> `native sqrt` = 原图像素下的 sqrt(area)；`work sqrt` = load_image 长边缩放到 1280 之后的同一 GT。
> LR 图的 native 面积天然小（原图仅 640×360），所以「native < 1024 px²」在两层里含义不同 —— 
> 上表把两层分开列，避免该混淆。**同一层内（HR）small < medium < large 的单调性依然存在。**

---

## 5. 有效监督质量（native → 真实 assigner）

采样：250 图 × 3 epoch = **10,105** 条最终 GT 记录，**0 条 uid 失配**（又一次独立验证插桩）。
真实 `TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)`，与 V5/§13 同参数、同 checkpoint。

### Table 2 — Supervision quality（按 **native** 面积分桶）

| class | n | **n_pos 中位** | **best_align 中位** | tgt_max 中位 | zero-pos% | align<.01% | pos@8 | pos@16 | pos@32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **small** | 1325 | **4.0** | **0.357** | 0.875 | **2.9%** | **10.0%** | 92.8% | 66.5% | 15.0% |
| medium | 4647 | 10.0 | 0.688 | 0.955 | 0.3% | 2.0% | 87.8% | 93.5% | 10.3% |
| large | 4133 | 10.0 | 0.826 | 0.977 | 0.0% | 1.1% | 18.1% | 76.8% | 29.7% |

**相对劣势**：small/medium `n_pos` = **0.400**、`align` = **0.519**；small/large = **0.400** / **0.432**。

按 native sqrt 细分（`8-12` 桶 n=9，样本不足，仅供参考）：

| native sqrt | n | n_pos 中位 | align 中位 | zero-pos% | align<.01% |
|---|---:|---:|---:|---:|---:|
| 8–12 | 9 | 1.0 | 0.008 | 22.2% | 66.7% |
| 12–18 | 201 | 8.0 | 0.380 | 4.5% | 11.4% |
| 18–24 | 406 | 3.0 | 0.258 | 3.7% | 15.5% |
| 24–32 | 709 | 5.0 | 0.410 | 1.7% | 5.6% |
| 32–96 | 4647 | 10.0 | 0.688 | 0.3% | 2.0% |
| >96 | 4133 | 10.0 | 0.826 | 0.0% | 1.1% |

### ★ 决定性对照：同一批记录，**只换分桶口径**

| 分桶口径 | class | n | n_pos 中位 | align 中位 | zero-pos% | align<.01% |
|---|---|---:|---:|---:|---:|---:|
| **`aug_area`**（V5/§13 口径） | small | 3489 | **8.0** | **0.521** | 1.5% | 6.1% |
| **`native_area`**（本审计口径） | small | 1325 | **4.0** | **0.357** | 2.9% | 10.0% |

**污染量化**：`aug_area < 1024` 这个桶里，**只有 32.9% 是真正的 native-small**，
**66.1% 是 native-medium**（另 1.1% native-large）—— 即被 mosaic+RP 整体缩小约 0.8× 后
「掉进」small 桶的中等目标。

**口径自洽性**：本脚本在 **V5 自己的 JSON** 上重算得到
`small n=1179 / n_pos 9.0 / align 0.568 / align<.01 4.0%`，与 §13 报告的数字**逐项吻合**
（§13: n=1179 / 9 / 0.568 / 4.0%）；medium 10.0/0.760/0.9%、large 10.0/0.861/0.2% 亦逐项吻合。
而本审计自己的 replay 用 `aug_area` 口径得到 8.0 / 0.521 / 6.1% —— 与 §13 **同量级**
（差异为不同 seed/不同图像子样本的抽样差）。⇒ **两套 pipeline 一致；改变结论的是分桶口径本身。**

### §10 两种 Exposure 定义

| class | n | Exposure-A（存在性，replay，全 300ep 混合） | Exposure-B(>0) | Exposure-B(≥0.01) | median n_pos | median align |
|---|---:|---:|---:|---:|---:|---:|
| small | 1325 / 1392 | **0.8458** | 0.9713 | **0.9004** | **4.0** | **0.357** |
| medium | 4647 / 5535 | 0.8502 | 0.9972 | 0.9802 | 10.0 | 0.688 |
| large | 4133 / 4633 | 0.8743 | 0.9998 | 0.9889 | 10.0 | 0.826 |

> `≥0.01` 是**诊断阈值**，不是新的训练规则（沿用你 §10 的提法）。
> **连续量才是主判据**：`n_pos` 与 `best_align` 的分化远大于任何二值阈值。

### 绝对值（避免只看比值）

- 每 native-small GT 在全 300 epoch 的期望 final target 次数 ≈ `300 × 4.0246 × 0.4465` ≈ **539**
- 期望「至少产生 1 个 final target」的 epoch 数 ≈ `300 × 0.8458` ≈ **254 / 300**（large 为 262/300）
- ⇒ **不存在「native-small GT 被饿死」的情形**：`zero-final% = 0.00%`（100 epoch 内无一个 GT 完全无监督）

---

## 6. OBSERVED FACTS / STATISTICAL EVIDENCE / INTERPRETATION / IMPLICATION

### OBSERVED FACTS

1. Mosaic 把每个 GT 的**实例化次数**从 1/epoch 提到 **~4.0/epoch**（1 次主图 + ~3 次作为别人的 tile，
   tile 来自 64 图 FIFO buffer），**且这个 4.0 对 small/medium/large 几乎完全相同**（4.0246 / 4.0087 / 4.0034）。
2. Mosaic 阶段**不缩放**目标（ratio 恒为 1.000，p10=p90=1.000），只做 1:1 摆放与裁剪。
3. 丢弃只发生在两处：Mosaic 边界裁剪（small 10.6% / large 8.4%）与
   `RandomPerspective.box_candidates`（small 51.8% / large 48.5%，ON）。
4. `fin|inst`（最终存活率）small **0.4294** < medium 0.4358 < large **0.4709**（ON）。
5. `ExposureA` small **0.8425** < medium 0.8473 < large **0.8716**（ON）；
   OFF regime 该单调性**不成立**（small 0.9432 > medium 0.9335）。
6. `final/native` 尺寸比：ON 中位 **0.795**（small）/ 0.752（large）；OFF 中位 **0.968**。
   `final < 0.5×` 占比：**small 0.9%** 反而**低于** large **6.3%**。
7. 最小 native 桶是 **8–12px（n=17）**；**没有任何 native sqrt < 8px 的 GT**。
8. 按 **native** 分桶，small 的 `n_pos` 中位 **4.0** vs medium/large 的 10.0；
   `best_align` 中位 **0.357** vs 0.688/0.826；`align<0.01` **10.0%** vs 2.0%/1.1%。
9. `aug_area<1024` 桶中**只有 32.9% 是 native-small**，66.1% 是 native-medium。

### STATISTICAL EVIDENCE

- Monte-Carlo：每 regime **N=100** epoch × **1599** 图；native GT **11,560**。
  桶级比例的有效观测数 = N × n_bucket（small 桶 ≈ 1.4×10⁵），标准误 ≲ 0.3%。
- 确定性：两次独立 replay（同一 seed 协议）给出**逐位相同**的计数
  （ON 2,080,894 / OFF 1,089,432）⇒ 差异不是 MC 噪声。
- 尺寸/存活判据由 `_sge_verify.py` V1 用**真实图像 vs 打桩**逐位验证（96/96），
  且 V2 用**独立代码路径**验证 uid（768/768 = 100%）。
- **口径自洽性**：在 V5 自己的 JSON 上重算，与 §13 逐项吻合（见 §5）。
- **未做**：未做假设检验/置信区间（本审计是描述性的）。跨桶比值差异（如 n_pos 4 vs 10）
  远大于任何合理的抽样误差，但本报告**不**为其附 p 值。

### INTERPRETATION

1. **原问题的答案是「不是」**：§13 那些 healthy 数字**不是**因为「只看存活后的 GT」而被抬高的。
   存活损失确实存在（每次实例化只有 ~45% 活到 final），但它**几乎与尺寸无关**
   （small vs large 的相对劣势仅 3–9%），因此**无法解释一个 small 专属的问题**。
2. **真正的 artifact 是分桶污染**：按增强后面积分桶时，mosaic 把整体尺度压到 ~0.8×，
   使 **66% 的 native-medium 目标掉进 small 桶**，而这些目标保留着 medium 级的监督质量
   （n_pos 10 / align 0.688），从而**把 small 桶的表观质量从 (4.0, 0.357) 抬到 (8.0, 0.521)**。
   换成 native 分桶后，native-small 的监督劣势**立刻显形**。
3. **监督劣势的所在 stage 是 assigner，不是 augmentation**：
   - Mosaic 阶段 small-vs-large 差 2.2pp；RP 阶段差 3.3pp；合计 ~5.5pp
   - 而 assigner 阶段 `n_pos` 4 vs 10、`best_align` 0.357 vs 0.826 —— **差 ~50–60%**
4. **n_pos 的巨大差距有相当部分是「天花板效应」**：`n_pos = min(topk=10, GT 内 anchor 数)`。
   一个 work-space sqrt ≈ 16.6px 的 small GT 在 stride 8 下只覆盖约 2×2 = 4 个 cell
   ⇒ **n_pos 中位正好是 4.0**，这是锚框几何的机械后果，不是「监督被剥夺」。
   `best_align` **没有天花板**，它 0.357 vs 0.826 的差距才是不可用几何完全解释的部分。
5. **augmentation 的净作用**：Mosaic 对 small GT 是**净正**（实例化 ×4，代价只是 10.6% 边界丢弃），
   **不是**「伤害 small」。让目标变小的是 `RandomPerspective` 的 `s ~ U(0.5,1.5)` 随机缩放
   ＋ 输出只取中心 1280² 窗口造成的部分裁剪；两者对 small/large 的影响方向与假设相反
   （**large 被缩得更狠**：`<0.5×` 6.3% vs small 0.9%，「final 后变小」56.1% vs 50.7%）。

### EXPERIMENTAL IMPLICATION

1. **不要再沿着「augmentation 让 small GT 曝光不足」这条线做实验。** 本审计的三个独立通道
   （存在性、尺寸、实例化次数）都指向：augmentation 对 small 的 differential 影响 ≤ 9%，
   而 shared 损失对所有尺寸一视同仁。
2. **若要继续攻 small-object hard miss，矛盾点应落在 assigner / 锚框几何**：
   native-small GT 在 stride 8 下只有 ~4 个可分配 anchor，`best_align` 中位仅 0.357。
   但**本审计不据此提出任何具体改法**（例如改 topk / stride / assigner 都未经检验，
   且 `reports/next_optimization_review.md` 已记录「加检测尺度」家族 3/3 证伪）。
3. **一个方法学纠正（对未来诊断直接可用）**：任何「按 GT 面积分桶」的监督统计
   **必须按 native 面积分桶**，否则会被 augmentation 的尺度收缩系统性污染
   （本次实测污染率 66%）。`diagnostic/small_object_train_replay/REPORT_V5.md`
   及其下游引用（§13）的表观结论应以此为限。

---

## 7. 裁决

```text
A — STRONG EVIDENCE
```

**范围限定（必须同读）**：A 只适用于「**native-small GT 存在明确、稳定的
effective-supervision deficit**」这一条 —— 连续量上 small/medium `n_pos` = 0.400、
`align` = 0.519，`align<0.01` 10.0% vs 2.0%/1.1%，且经三处独立验证与两次确定性 replay 复现。

**不**适用于：
- ❌ 「augmentation 让 small GT 曝光不足」—— 该假设**证伪**（§6 INTERPRETATION 1、5）。
- ❌ 「small GT 被大量丢失/缩小」—— `zero-final% = 0.00%`，`<0.5×` 占比 small 0.9% < large 6.3%。
- ❌ 任何因果主张 —— 本审计是**描述性**的；#3/#4 的 assigner 定位是**机制推理**，不是因果实验。

**对原假设（§13）的判定**：`存活偏差导致 §13 数字虚高` —— **PARTIALLY**。
存活偏差真实存在（~55% 实例化丢失）但**尺寸中性**，故非主因；
真正的主因是**分桶污染**（66% native-medium 混入 small 桶），这是本审计新发现并已量化。

```text
HOLD — 不据此设计下一实验、不自动改配置、不自动训练、不自动提交。
```

---

## 8. 产物与复现

| 文件 | 作用 |
|---|---|
| `_sge_instrument.py` | native→final 身份插桩（7 个挂载点）+ 零 I/O 打桩 `load_image`（镜像 buffer 机制） |
| `_sge_verify.py` / `_sge_verify.json` | **前置门**：V1 stub 等价 96/96、V2 uid 正确 768/768、V3 desync 0 |
| `_sge_replay.py` / `_sge_replay.npz` / `_sge_native_meta.json` | Monte-Carlo replay（各 100 epoch） |
| `_sge_supervision.py` / `_sge_supervision.json` | 250 图 × 3 epoch 的真实 assigner 监督（10,105 条，0 失配） |
| `_sge_analyze.py` / `_sge_tables.json` | Table 1/3/4 + §8/§9 + 全 300ep 混合 + 分层敏感性 |
| `_sge_sup_analyze.py` / `_sge_sup_tables.json` | Table 2 + §10 Exposure-B + 与 §13 对照 |
| `_sge_replay_run.log` / `_sge_sup_run.log` / `_sge_analyze_out.txt` | 运行日志 |

复现顺序：

```bash
python -X utf8 -u diagnostic/small_gt_exposure/_sge_verify.py          # 必须 VERIFY PASS
python -X utf8 -u diagnostic/small_gt_exposure/_sge_replay.py --replays-on 100 --replays-off 100
python -X utf8 -u diagnostic/small_gt_exposure/_sge_supervision.py --n-images 250 --epochs 3
python -X utf8 -u diagnostic/small_gt_exposure/_sge_analyze.py
python -X utf8 -u diagnostic/small_gt_exposure/_sge_sup_analyze.py
```

## 9. 本轮合规状态

| 项 | 状态 |
|---|---|
| 训练 / finetune / optimizer step / backward | ❌ 未执行（仅 `torch.no_grad()` 的只读 forward） |
| 修改 checkpoint / 模型结构 / augment.py / base.py / train.py / default.yaml | ❌ 未改动（`git status` 中这些文件与本轮开始前逐字节相同） |
| 修改 D′ 配置 / evaluator / inference | ❌ 未改动 |
| test inference / submission / 线上提交 | ❌ 未执行 |
| 改动任何已有实验结果 | ❌ 未改动；`small_object_train_replay/` 只读 |
| 本轮新增 | 仅 `diagnostic/small_gt_exposure/` 下新文件 |

⚠ 唯一被读写的既有文件是**上一轮 E1 实验**留下的 `ultralytics/nn/{tasks,modules/block}.py`、
`ultralytics/models/yolo/detect/train.py`、`scripts/preflight_check_train_config.py`
及 `configs/*_e1*.yaml`（均为 E1 阶段的改动，**本轮未再触碰**）。
本审计的 `augment.py` 与 D′ 训练时**行为等价但不逐字节相同** —— 差异与判定见 §1.2。

