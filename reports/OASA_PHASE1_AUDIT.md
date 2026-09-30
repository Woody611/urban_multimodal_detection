# OASA — Phase 1 只读审计

**日期**：2026-09-23 · **性质**：**READ-ONLY**。未训练、未改任何代码/配置。
**目标**：定位 OASA 插入点、确认调用链、识别风险，**不实现**。

---

## 1. 当前 augmentation 顺序（实读代码，非推测）

**入口**：`ultralytics/data/base.py::BaseDataset.__getitem__` → `self.transforms(self.get_image_and_label(index))`

```
get_image_and_label(index)                    base.py:588
  ├─ load_image(index)                        base.py:435   ← RGBID 分支在此合并成 5ch，**native 分辨率**
  ├─ apply_modality_dropout(..., self.augment) base.py:594
  ├─ ratio_pad / rect_shape
  └─ update_labels_info(label)                base.py:611  → instances.bbox_format="xyxy"
        ↓
self.transforms(...)   = Compose([...])       augment.py:3803
  ├─ pre_transform = Compose([                 augment.py:3769
  │     ① Mosaic(dataset, imgsz, p=hyp.mosaic)         ← p=1.0（RGBID 无 aug 段）
  │     ② CopyPaste(p=hyp.copy_paste)                  ← p=0.0，关闭
  │     ③ RandomPerspective(degrees,translate,scale,shear,perspective,
  │                         pre_transform=LetterBox(imgsz,imgsz))   ← scale=0.5 抖动
  │   ])
  ├─ MixUp(dataset, p=hyp.mixup)               ← p=0.0，关闭
  ├─ alb = Albumentations(p=0)                 ← 5ch 显式关闭
  ├─ random_hsv                                ← RandomHSV 对 shape[-1]!=3 直接 return（5ch 无效）
  ├─ RandomFlip(vertical, p=0.0)
  └─ RandomFlip(horizontal, p=0.5)
        ↓
Format  → CHW, [R,G,B,IR,D], /255             augment.py:~3068
```

**决定性事实（由代码算术得出，非推测）**：

```text
Mosaic 画布 = 2s × 2s，s = imgsz = 1280  →  2560 × 2560      augment.py:3170-3172
标的 tile 在画布中按**接近 native 尺寸**放置（resized_shape，本数据集 load_image 不改尺寸）
RandomPerspective + LetterBox 把 2560 → 1280
⇒ 每个像素在最终输入中被压缩 2.00×
```

这与本项目既有的 crop 审计实测一致（`reports/rgbid_small_object_crop_audit.md` §8：
small √area 中位 **含 mosaic 8.8 px / 无 mosaic 17.7 px**，比值 ≈2.0）。

---

## 2. ★ OASA insertion point —— 存在一个必须先裁决的分叉

### 候选 A：`base.py::get_image_and_label`，紧跟 `apply_modality_dropout` 之后

| | |
|---|---|
| 位置 | `base.py:594` 之后，`update_labels_info` 之前 |
| 模态同步 | ✅ **构造性**（此时已是合并后的单个 5ch 数组） |
| train/val 门控 | ✅ `self.augment` 已存在，`build.py:103` `augment=mode=="train"` |
| 需要改的文件 | `base.py` + `default.yaml` + `scripts/train.py` + 新 helper |
| **致命问题** | ❌ **Mosaic 在它之后运行，会把放大整体抵消 2.00×** |

**定量**：scale=2.0 → ROI 面积 1/4 → 放大 2.00×；Mosaic 缩小 2.00× ⇒ **净比 = 1.00**。
即使把 ROI resize 回 native 尺寸，最终输入空间的 anchor 尺寸**与 baseline 相同**。
唯一残留差异是 3.3% 的 `close_mosaic` 末段 epoch 里会真的放大 —— 那 10 个 epoch 之外收益为零。

> **这正是本项目 crop 审计 `CROP_NO_GO` 的同一条论证**（crop 放大 1.16–1.58× < mosaic 2.00×）。
> 区别仅在于 OASA scale=2.0 恰好**打平**而非落后。

### 候选 B：`augment.py::v8_transforms` 的 `pre_transform` 内，**Mosaic 与 RandomPerspective 之间**

| | |
|---|---|
| 位置 | `augment.py:3772`（`CopyPaste` 之后、`RandomPerspective` 之前） |
| 模态同步 | ✅ 此时仍是合并后的 5ch 数组 |
| **能否真正放大** | ✅ **能**。在此裁剪 W×W 窗口，RandomPerspective+LetterBox 把 W→1280 ⇒ 放大 1280/W |
| 需要的放大 | 取 W = 1280（从 2560 画布裁）⇒ 相对 mosaic baseline **净放大约 2.0×** |
| 代价 | ⚠️ 被选中的样本**实际上退化为单 tile**（W=1280 只覆盖约 1/4 画布）⇒ 该样本失去 mosaic 的 4 图拼接多样性 |
| 影响面 | 25.1% 图含 small GT × prob 0.5 ≈ **12.5% 的样本**受影响 |
| 需要改的文件 | **`ultralytics/data/augment.py`** ⚠️ + `default.yaml` + `scripts/train.py` + 新 helper |

### 候选 C：整条 transforms 之后

❌ 此时图像已是 1280×1280，再 zoom = **上采样，无新信息**。排除。

### ⚠️ 与 Prompt 假设的差异（需明确）

Prompt §11 假定可以直接在 `load_image → merge → augmentation` 链上找一个"保证三模态同步"的位置。
**实际仓库结构比这更不利**：几何增广（Mosaic）**作用在合并后的 5ch 数组上**
（不是分模态各做一次），所以"三模态同步"在候选 A/B **都是构造性满足的**；
真正的约束不是模态同步，而是 **Mosaic 会把 OASA 的尺度收益抵消掉**。

**另一个必须记录的差异**：`ultralytics/data/augment.py` 被 `.gitignore:4` 吞掉
（见记忆 `augment-py-e9-residue-trap`），**不进 git、不同步云端**。
选候选 B 意味着修改一个**无版本控制**的文件 → 云端必须手动核对（这是本项目已经踩过的坑）。

---

## 3. 需要修改的文件（按候选）

| 候选 | 文件 | 性质 | 风险 |
|---|---|---|---|
| A | `ultralytics/data/base.py` | 新增 helper + 1 处调用 | 低（gitignored 但会随 ultralytics 目录同步） |
| A | `ultralytics/cfg/default.yaml` | 加 5 个键 | 低 |
| A | `scripts/train.py` | 透传 5 个键 | 低 |
| A | `configs/oasa.yaml`（新） | 训练配置 | 低 |
| B | 上述全部 **+ `ultralytics/data/augment.py`** | 在 `pre_transform` 内插一层 | **中**（gitignored + 改变 mosaic 语义） |

**不改**：模型结构、loss、optimizer、数据内容、val/test split、official evaluator。

---

## 4. 完整 config → runtime → inference 调用链（以 `modality_dropout` 为已验证先例）

| Stage | 位置 | 机制 | 现状 |
|---|---|---|---|
| config | `configs/oasa.yaml`（新） | `object_scale_aug: true` 等 5 键 | 待建 |
| default | `ultralytics/cfg/default.yaml` | `modality_dropout` 在同一文件 L137 有默认值 | **先例存在** |
| train.py | `scripts/train.py:485-486` | `if "modality_dropout" in train_cfg: kwargs["modality_dropout"]=dict(...)` | **先例存在，照抄** |
| trainer.args | `ultralytics/engine/trainer.py:102` `get_cfg` | kwargs 合并进 args namespace | 已验证（D' dry-run 看到 `modality_dropout` 出现在 args） |
| dataset | `build.py:96-114` `build_yolo_dataset(cfg=self.args,...)` → `YOLODataset(..., hyp=cfg)` | `self.hyp` = args namespace | 已验证 |
| OASA | `base.py:594` 后 `getattr(getattr(self,"hyp",None),"object_scale_aug",False)` | 与 `apply_modality_dropout` 完全同构 | 待实现 |
| validation | `build.py:103` `augment = (mode=="train")` → val 时 `self.augment=False` | OASA 用 `self.augment` 门控后**自动 OFF** | 机制已存在 |
| inference | `scripts/predict_rect.py:187` → `LoadImagesAndVideos` | **完全不经过 `BaseDataset`/`transforms`** | ✅ **结构性不可能继承 OASA** |
| evaluator | `scripts/official_eval.py`（已冻结，6/6 regression） | 独立模块 | ✅ |

> **inference 安全性（Prompt §30）**：`LoadImagesAndVideos` 与 `BaseDataset` 是两条**互不相交**的路径
> （`loaders.py` vs `base.py`），OASA 只加在 `base.py` → **推理端不可能误执行**。
> 若选候选 B（改 `augment.py`），推理更不会经过（`augment.py` 只被训练/val 的 Dataset 使用）。
> **这与 CLAHE 的情况根本不同**——CLAVE 是 preprocessing，训练/推理各有一份拷贝，所以才会漂移。

---

## 5. 可能导致 OASA "看似生效但实际无效" 的风险点

| # | 风险 | 检测方法 | 严重度 |
|---|---|---|---|
| **R1** | **Mosaic 抵消**（候选 A） | 直接测**最终输入空间**的 anchor 像素尺寸，而非中间某步 | **P0** |
| **R2** | `augment.py` 被 gitignore，云端不同步 → 候选 B 在云端可能根本没生效 | 开训前在云端 `grep OASA` + 核对 args.yaml | **P0（若选 B）** |
| **R3** | config 漂移（default=false / train=true / dataset=false） | 开训后读 `args.yaml` 的 `object_scale_aug`；OASA 每 N 个样本打印一次实际计数 | P1 |
| **R4** | **val 偷跑 OASA** | 短跑时打印 `split`；val 阶段 `applied` 计数必须为 0 | P1 |
| **R5** | ROI clamp 后 anchor 被裁到 <80% 却仍强行 augment | 统计 `skipped_anchor_visibility` 计数 | P1 |
| **R6** | 中/大目标被大量裁掉（§21） | 统计每档 GT 的保留率（small/medium/large 分开报） | P1 |
| **R7** | ROI 在 640×360 的图上尺度不同（本数据集有 122/1600 张 360p） | 分分辨率统计 applied 比例 | P2 |
| **R8** | scale 抖动（`hyp.scale=0.5`）随机抵消部分放大 | 报告的是**分布**（中位放大倍数）而非单值 | P2 |

---

## 6. 我准备如何做 unit test / visual smoke test

### 6.1 Deterministic unit test（`scripts/test_oasa.py`，纯内存，不训练）

固定构造 5ch 数组 + 已知 bbox，`rng` 定种子，断言：

```
Geometry   : RGB/IR/Depth 的 crop 坐标完全相同（同一数组切片 → 逐位相同）
Anchor     : anchor visible_fraction >= 0.8
Scale      : new_anchor_w / old_anchor_w ≈ 2.0 （容差 ±5%）
Labels     : 0<=x<=1, 0<=y<=1, 0<w<=1, 0<h<=1，无 NaN
Alignment  : 三模态最终 spatial dims 完全一致
Gate       : augment=False 时输出与输入**逐位相同**
Skip       : anchor 无法满足 0.8 可见度时返回原样本（不做任何改动）
```

### 6.2 Visual smoke test（`scripts/oasa_visual_smoke.py`）

输出 3~5 组 before/after **PNG**（RGB / IR / Depth 三联 + bbox 叠加）到 `diagnostic/oasa_smoke/`，
人工核对：anchor 是否变大、bbox 是否跟随、三模态是否对齐、是否出现黑边/拉伸、GT 丢失量。

### 6.3 最终输入尺度实测（针对 R1，**这是判据**）

对同一批样本，测量 anchor 在**最终模型输入空间**的 √area：
`native → (OASA) → Mosaic → RandomPerspective/LetterBox → final`
并与 baseline 同 pipeline 对比。**只有当最终输入空间的尺寸确实提高，实验才算成立。**

### 6.4 1~4 batch smoke test + 1~3 epoch 短跑

用现有 `%TEMP%` dry-run harness 模式（真实 `scripts/train.py`、真实 config、1 batch 后停），
再做 1–3 epoch 短跑确认无 NaN / bbox 损坏 / 模态错位。

---

## 7. 配置漂移检查（当前状态）

| 检查 | 结果 |
|---|---|
| `object_scale_aug` 是否已存在于 `default.yaml` | **不存在**（需新增，无冲突） |
| 是否有历史 OASA 残留代码 | **无**（全仓库无 `object_scale_aug` / `apply_object_aware_scale`） |
| train/val 门控机制 | ✅ 已有 `augment = mode=="train"`（`build.py:103`） |
| inference 是否会经过训练 transforms | ✅ **不会**（`LoadImagesAndVideos` 独立路径） |
| 冻结 baseline | `0.53180`（`FREEZE_MANIFEST` 16/17 完好）；**最近线上最优为 D' = 48.712（新测试集）** |
| small area 定义 | **复用既有审计定义**：native 像素，small < 32² = **1024**（`rgbid_small_object_crop_audit.md` §1） |

---

## 8. 需要你裁决的一件事

候选 A 与候选 B 都能做到"三模态同步 + train-only + 可复现"，
**唯一区别是能否真正提高最终输入空间的 anchor 尺度**：

| | 候选 A（`base.py`，pre-Mosaic） | 候选 B（`augment.py`，post-Mosaic） |
|---|---|---|
| 三模态同步 | ✅ 构造性 | ✅ 构造性 |
| train-only | ✅ | ✅ |
| **最终输入空间是否真的放大** | ❌ **被 mosaic 抵消（净 1.00×）** | ✅ **净约 2.0×** |
| 改动文件 | 3 个（+1 新建） | 4 个（**含 gitignored 的 `augment.py`**） |
| 副作用 | 无 | 被选中样本退化为单 tile，12.5% 样本受影响 |
| 预期效果 | **接近零** | 真实尺度提升 |

**我的判断**：若选 A，这个实验几乎注定是"工程正确但机制不生效"，会重演 crop 审计的结论。
但选 B 要动 gitignored 的 `augment.py`（云端同步风险），且改变了 mosaic 语义。

**在你裁决前我不会写任何代码。** 请指示：

1. 走 **候选 B**（真实尺度提升，代价是动 `augment.py` + 12.5% 样本退化为单 tile）
2. 走 **候选 A**（最干净，但预计无效——可作为"确认 mosaic 抵消"的对照）
3. 或第三种方案（例如：OASA 放在候选 A 位置**并**对该样本跳过 mosaic —— 但那是两个变量，我不建议）

```text
本轮训练次数 = 0 ；修改文件数 = 0
Phase 1 状态：审计完成，等待插入点裁决
```
