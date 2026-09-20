# Small Object Crop 实验 · Phase 0 冻结记录 + Phase 1 代码审计

**日期**：2026-09-19
**性质**：**只读**。未训练、未创建 run、未修改任何配置 / 权重 / 提交产物。
**结论**：**`PHASE_0_PARTIAL` + `PHASE_1_BLOCKED`** → 见 §3、§4。**未启动任何训练。**

---

## 0. 速览

| Phase | 项 | 结果 |
|---|---|---|
| 0 | FREEZE_MANIFEST 校验 | **16/17 OK，1 MISMATCH**（`scripts/predict_rect.py`） |
| 0 | 正式 best.pt / submission.zip / 训练配置 / 模型 yaml / dataset yaml / args / results.csv | ✅ **全部逐字节完好** |
| 0 | 官方评测链代码是否被哈希冻结 | ❌ **`ultralytics/data/` 全目录被 gitignore，不在 git、不在 manifest** |
| 1 | `apply_small_object_crop()` 实现 | ❌ **不存在**（本地、git、6 个同步 zip 内均无） |
| 1 | `scripts/small_object_crop_smoke_test.py` | ❌ **不存在** |
| 1 | `configs/train_rgbid_small_object_crop.yaml` | ❌ **不存在** |
| 2 | smoke test | ⛔ **无法执行（脚本不存在）** |

---

## 1. Phase 0 · 冻结资产链校验

校验命令：逐行读取 `reports/FREEZE_MANIFEST.sha256`（2026-09-17 13:55:44 生成），比对 SHA256 + 字节数。

### 1.1 ✅ 完好（16 项，逐字节匹配）

| 类型 | 路径 | 说明 |
|---|---|---|
| MODEL | `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt` | **RGBID 正式 best.pt，线上 0.53180**，f9dddbfa… |
| MODEL | `runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt` | F4 备用，线上 0.51051 |
| SUBMISSION | `submissions/rgbird_ir_quicktest/submission.zip` | **0.53180 对应提交包** |
| SUBMISSION | `submissions/rgbid_f4_ensemble_3to1_candidate/submission.zip` | 归档，不提交 |
| FROZEN_CODE | `scripts/predict.py` | |
| CONFIG | `configs/dataset.yaml` | |
| CONFIG | `data/processed/rgbid_split/dataset.yaml` | RGBID 切分 |
| CONFIG | `configs/train_rgbird_ir_quicktest.yaml` | **RGBID 训练配置** |
| CONFIG | `configs/yolo11m_earlyfusion.yaml` | **RGBID 模型配置** |
| RUN_META | `runs/.../rgbird_ir_quicktest/args.yaml` | |
| RUN_META | `runs/.../rgbird_ir_quicktest/results.csv` | |
| RUN_META | `runs/.../rgbd_f4_1280/args.yaml` | |
| DOC | `README.md` / `docs/experiment_log.md` / `docs/model_design.md` / `experiments/README.md` | |

### 1.2 ⚠️ MISMATCH（1 项）：`scripts/predict_rect.py`

| | SHA256 | 字节 |
|---|---|---|
| manifest 冻结版 | `656aa58789a467e003517a3f6ca1f05ab25dc065cf5861a8df66468f90528be8` | 11286 |
| 当前磁盘版 | `6baa7367da9376a94555428834fb33e4521c6d57b938bc249782b2aa995a3b5c` | 11724 |

**溯源**：manifest 冻结版 == commit `41abf7b`（`complete rgbid experiment`）内该文件，逐字节吻合。
其后被**两次**修改：

| commit | 变更 | 字节 |
|---|---|---|
| `aec32aa`（ir_clahe 实验） | 新增 `ir_encoding = train_cfg.get("ir_encoding","percentile")`，透传给 `LoadImagesAndVideos` | 11625 |
| `2cc8961`（complete check） | 仅新增 1 行 `print(...)` | 11724 |

**行为影响判定（静态分析，未执行验证）**：

- `configs/train_rgbird_ir_quicktest.yaml` **不含** `ir_encoding` 键 → 取值 `"percentile"`（代码自带默认）。
- `ultralytics/data/loaders.py:323/344`：`ir_encoding="percentile"` 默认 + `or "percentile"` 兜底；`loaders.py:656` 的 CLAHE 分支为 `== "clahe"` 时才进入。
- ⇒ **对正式 RGBID 配置，两处改动均为「新增可选参数 + 一行日志」**，代码路径与冻结版一致。变更注释亦自述「冻结提交链输出逐位不变」。

**但本项仍标记为未闭合**：这是**静态阅读**结论，未做「冻结脚本 vs 当前脚本」的输出逐位比对。
Phase 7 要求「与 0.53180 完全一致的 official 口径」，因此**开训前应实测闭合**（比对方案见 §5）。

### 1.3 ❌ 官方评测链代码未被哈希冻结（缺口）

FREEZE_MANIFEST 覆盖 `scripts/` 与 `configs/`，但 **RGBID 的 5 通道加载、IR 预处理、几何变换插入点全部位于 `ultralytics/data/`，该目录不在 manifest 中**：

```
$ git check-ignore -v ultralytics/data/base.py
.gitignore:4:data/    ultralytics/data/base.py
$ git ls-files ultralytics/data/ | wc -l
0
```

`.gitignore` 第 4 行的 `data/` 规则**无锚定**，同时匹配了项目根 `data/` 与 **`ultralytics/data/`**。
后果：该目录**既不在 git、也不在 manifest**，其修改**不可 diff、不可回溯、不可校验**。

| 文件 | SHA256(前32) | 字节 | mtime | 是否在 manifest |
|---|---|---|---|---|
| `ultralytics/data/base.py` | `8bcf834930155bd58a99ccfc5e871133` | 33155 | **2026-09-18 22:29** | ❌ |
| `ultralytics/data/loaders.py` | `f84bdcf29081e6f95b58fbf19c255d1d` | 51527 | **2026-09-19 12:00** | ❌ |
| `ultralytics/data/dataset.py` | `0aba2bc14793ab85ac477d0eab0cf408` | 35962 | 2026-09-17 19:19 | ❌ |
| `ultralytics/data/augment.py` | `1e0ca17c665db9ebf9a9bb6d019ddb5d` | 168661 | 2026-09-13 07:12 | ❌ |
| `ultralytics/nn/tasks.py` | `8228854e9db090b45cbd62590c7c5e5b` | 59180 | 2026-09-06 13:39 | ❌ |
| `ultralytics/engine/trainer.py` | `de94f366c8218916c88e001b2ef22a08` | 38238 | 2026-08-25 09:27 | ❌ |
| `scripts/train.py` | `80f6cdec1e52d41258ca7726e4ff79c7` | 26996 | 2026-09-18 22:29 | ❌ |
| `scripts/official_map.py` | `7b4aa24d2856391c3fbeb237c3453d7f` | 13580 | 2026-09-13 12:20 | ❌ |
| `ultralytics/nn/**`（模型结构） | — | — | — | ✅ 441 文件受 git 跟踪 |

**注意**：`ultralytics/data/base.py`（09-18 22:29）与 `loaders.py`（09-19 12:00）mtime **均晚于 manifest 生成时刻**（09-17 13:55）。
即：**官方评测路径所依赖的两个文件，在冻结之后都被改动过，且没有任何哈希可证明改动对 RGBID 路径中性。**
这与既有记录 `ir-preprocessing-three-copies`（IR 预处理存在 3 份拷贝）一致，属**已知但未闭合**的结构性风险。

### 1.4 输出目录隔离

- 正式产物：`runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/`（本实验**不应写入**）
- 新实验拟用：`runs/urban_multimodal_det_yolo11_rgbid_small_object_crop/`（**当前不存在，命名无冲突**）
- ✅ 隔离方案可行；best.pt / last.pt / results.csv / reports / FREEZE_MANIFEST 本次**均未被触碰**。

---

## 2. Phase 1 · 代码审计

### 2.1 审计对象不存在

任务书要求审计 `ultralytics/data/base.py` 中「新增的 `apply_small_object_crop()`」。**该函数不存在。**

```
$ grep -rn "apply_small_object_crop" ultralytics/
（无输出）

$ grep -rn "apply_modality_dropout" ultralytics/data/base.py
36:def apply_modality_dropout(im, cfg, augment):
594:        label["img"] = apply_modality_dropout(
```

全仓库检索 `small_object_crop` 仅命中 **3 处**，全部为文档/只读脚本：

| 路径 | 性质 |
|---|---|
| `reports/next_optimization_review.md` | **设计提案**（§6，明写「**本轮不实现**」） |
| `reports/rgbid_small_object_crop_audit.md` | **只读可行性审计**，结论 `CROP_NO_GO` |
| `diagnostic/small_object_crop_audit/_crop_audit.py` | 只读模拟脚本（不接入训练） |

6 个同步 zip（`ultralytics/ scripts/ configs/ experiments/ diagnostic/ reports/`）**全部检索无命中**（除上述文档）。

### 2.2 插入点现状（`get_image_and_label`，base.py:586-605）

当前实际调用序列为：

```
load_image  →  apply_modality_dropout  →  ratio_pad  →  update_labels_info
```

任务书假定的 `load_image → apply_small_object_crop → update_labels_info` **尚未存在**。
`apply_modality_dropout` 的守卫为 `not augment or not cfg or not cfg.get("enabled", False)` → **val/test 天然短路**，这与任务书对 crop 的要求同构，可作为实现时的沿用范式。

### 2.3 既有审计已对该方案给出 `CROP_NO_GO`

`reports/rgbid_small_object_crop_audit.md`（2026-09-18，只读）已完成本方案的量化审查，**结论为 NO_GO**，并明确「按用户 §15 条直接结案，**不提出 Probe**」——这正是实现不存在的原因。

其决定性依据（两条，均引用实测）：

1. **与 mosaic 功能冗余**：crop 放大倍数恒为 `1/√r` ∈ **1.16×–1.58×**，而 mosaic 自身缩小 **2.00×**。
   即使在最激进的 r=0.40 档，crop+mosaic 下 small 中位 √area = **14.0 px**，仍**低于 baseline 自己在无 mosaic 窗口内已有的 17.7 px（−21%）**。
   ⇒ crop 未创造新尺度区间，只是**部分抵消 mosaic**。
2. **同族机制已两次实测为中性**：① 1536 训练 = 全局 **1.44×** 放大（强于 crop 的 1.16–1.58×），两个 LR 探针官方 0.49401 / 0.50673，**small AP 仅 +0.0019**；② P2 检测层 **Δ = +0.00013**。

同审计也诚实记录了两条**有利于** crop 的发现（但未改变判定）：部分截断标签噪声 ≤1.4%；多模态同步为构造性保证（实测「先合并再裁」与「先分别裁再合并」逐位相同）。

---

## 3. Phase 2 · Smoke test

**无法执行。**

- `scripts/small_object_crop_smoke_test.py` **不存在**（`scripts/` 下 39 个脚本已逐一确认）。
- 待测函数 `apply_small_object_crop()` **不存在**。
- 待测配置 `configs/train_rgbid_small_object_crop.yaml` **不存在**。

按任务书纪律「如果 smoke test FAIL：**立即停止，不训练**」「不要临时修改实现后直接进入训练」——
**本轮不实现、不训练、不提交。**

---

## 4. 判定

| 状态 | 适用性 |
|---|---|
| `ACCEPTED` | — |
| `REJECTED` | — |
| `INVALID` | — |
| **`BLOCKED_PRE_IMPLEMENTATION`** | ✅ **当前状态** |

**任务书 Phase 0 的停止条件已触发**：「如果无法确认 0.53180 的正式资产链路，立即停止，不要训练。」
- 正式权重/提交包/配置链 **完好且可校验**；
- 但**官方评测链代码（`ultralytics/data/`）不在冻结体系内**，且 `predict_rect.py` 已漂移 —— 资产链**未能完整确认**。

**任务书 Phase 1/2 的前置条件亦不成立**：被审计对象从未实现。

**正式模型 `0.53180` 状态：未受影响。本轮未修改、未覆盖任何正式资产。**

**`OFFICIAL_MODEL_REMAINS_0.53180`**

---

## 5. 若要推进，需用户裁决的三条路径

> 本节仅为选项陈述，**不构成建议、不自行执行**。

**路径 A — 终止（与既有审计一致）**
`rgbid_small_object_crop_audit.md` 已判 `CROP_NO_GO` 并给出量化依据；若无新证据推翻「crop 放大 < mosaic 缩小」与「同族机制 2/2 中性」，则直接结案，节省 11h。

**路径 B — 新证据驱动重启**
需指明**哪条新证据**推翻上述两条决定性依据（例如：实测 crop+mosaic 联合下的 small AP 而非模拟推算）。否则属重复已被否证的假设。

**路径 C — 先闭合 Phase 0 缺口再谈实验**（与是否做 crop 无关，独立价值）
1. 实测 `predict_rect.py` 冻结版 vs 当前版在 RGBID 配置下的输出**逐位一致性**（用 F4 或 RGBID best.pt，同一 test split，比对 submission TXT）。
2. 修复 `.gitignore` 的 `data/` 无锚定规则（改为 `/data/`），使 `ultralytics/data/` 纳入版本控制。
3. 将 `ultralytics/data/{base,loaders,dataset,augment}.py`、`scripts/train.py`、`scripts/official_map.py`、`ultralytics/nn/tasks.py`、`ultralytics/engine/trainer.py` 的当前 SHA256 补入 FREEZE_MANIFEST。

---

## 6. 合规状态

| 项 | 状态 |
|---|---|
| 训练 | ❌ 未执行 |
| 正式 best.pt / last.pt / results.csv / submission.zip | ✅ 逐字节未变（§1.1） |
| 正式配置 / trainer / dataset / dataloader / augmentation | ❌ 一字未改 |
| 验证集 | ✅ 未触碰 |
| 新增 | 仅本文件 `reports/small_object_crop_phase0_freeze.md` |
