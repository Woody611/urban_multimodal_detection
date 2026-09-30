# STEP 3 IMPLEMENTATION AUDIT — OASA（方案 A · Pre-Mosaic）

**日期**：2026-09-23 · **0 次训练、0 次 A100、0 次提交**
**唯一变量**：`object_scale_aug`（Pre-Mosaic Object-Aware Small-object Scale Aug）

---

## 1. Files changed

| 文件 | 改动 | 追踪状态 |
|---|---|---|
| `ultralytics/data/augment.py` | 新增 `ObjectScaleAug` 类；`v8_transforms` 中保存 Mosaic 实例引用并把 OASA 插到 Compose **第一位** | 已 `git add`（`A`/`AM`） |
| `ultralytics/cfg/default.yaml` | 新增 6 个键（默认 `false`） | 已追踪 |
| `scripts/train.py` | 从 train yaml 透传 OASA 键（与 `modality_dropout` 同构） | 已追踪 |
| `configs/oasa_pre_mosaic.yaml` | **新建**（`train_rgbid_sepstem_clahe.yaml` 逐行副本 + OASA 段 + 新 experiment_name） | 新建 |

**未修改**：模型结构 / loss / optimizer / dataset / 数据内容 / val-test split / official evaluator / inference / `base.py`。

---

## 2. Gitignore changes

```gitignore
data/                          # 原规则，语义未动
!ultralytics/data/
ultralytics/data/*
!ultralytics/data/augment.py   # ← 唯一例外
```

| 检查 | 结果 |
|---|---|
| `git ls-files ultralytics/data/augment.py` | **能找到** ✅ |
| `ultralytics/data/base.py` 是否被额外放行 | **否**（仍 `.gitignore:11`）✅ 符合"最小单文件例外" |
| loaders/dataset/build/utils.py | **仍被忽略** ✅ |
| 是否改成 `!ultralytics/data/*` 或整目录放开 | **否** ✅ |

> **本轮未修改 `base.py`**（见 §3），因此**无需**为其新增例外。

---

## 3. OASA code location

**实现文件**：`ultralytics/data/augment.py::ObjectScaleAug`（新增类，约 150 行）
**插入点**：`augment.py::v8_transforms` 返回的 `Compose` 的**第一个元素**

```
load_image → 【OASA】 → Mosaic → CopyPaste → RandomPerspective → LetterBox → Flips → Format
              ▲
        Compose([_oasa, pre_transform, MixUp, alb, random_hsv, flip_v, flip_h])
```

**为什么是这个位置**：`pre_transform` 的第一个元素就是 Mosaic，所以把它放在 `pre_transform`
**之前**即等价于 `load_image → OASA → Mosaic`，满足任务书 §9 的 Pre-Mosaic 要求。

**为什么不需要改 `base.py`**：任务书假定 OASA 要插在 `load_image` 内部；
但 `v8_transforms` 返回的 Compose 是 `get_image_and_label` 的**直接下游**，
在那里插一层即得到完全相同的语义位置 —— 于是**改动只落在已放行的 `augment.py` 一个文件**，
不动 `base.py`、不扩大 gitignore 例外。

---

## 4. Config chain

| Stage | Config | Code location | Runtime verified |
|---|---|---|---|
| config | `configs/oasa_pre_mosaic.yaml` | `object_scale_aug: true` 等 6 键 | ✅ yaml 解析确认 |
| default | `ultralytics/cfg/default.yaml` | 同文件已有 `modality_dropout` 默认值先例 | ✅ |
| train.py | `scripts/train.py`（`_build_train_kwargs` 内，紧随 modality_dropout） | `kwargs["object_scale_aug"]=…` | ✅ |
| trainer.args | `engine/trainer.py:102 get_cfg` | kwargs 合并进 namespace | ✅ |
| dataset | `data/build.py:96-114 build_yolo_dataset(cfg=self.args,…)` → `YOLODataset(hyp=cfg)` | `self.hyp` = args namespace | ✅ |
| **OASA** | `augment.py::v8_transforms` → `ObjectScaleAug(mosaic_ref=_mosaic_inst)` | 读 `getattr(hyp,"object_scale_aug",False)` | ✅ 见 §6/§7 |
| validation | `dataset.py:174 build_transforms` → `else: Compose([LetterBox, Format])` | val 走另一条分支 | ✅ 见 §9 |
| inference | `scripts/predict_rect.py:187` → `LoadImagesAndVideos` | 与 Dataset/transforms **完全不相交** | ✅ 结构性不可能继承 |
| evaluator | `scripts/official_eval.py`（已冻结 6/6 regression） | 独立模块 | ✅ |

---

## 5. Runtime config verification（实际运行时的真实状态）

```
[OASA] enabled=True insertion=pre_mosaic scale=2.0 prob=1.0 small_area=1024 min_visible=0.8 version=v1
[OASA] seen=… mosaic_on=1 eligible=60 applied=60 skip_nomosaic=0 skip_prob=0 skip_nosmall=0
       skip_visibility=0 mode(center/clip/recenter)=31/29/0 anchor_vis_median=1.000 anchor_kept=60 gt 742->497
```

- `transforms[0]` 实测**就是** `ObjectScaleAug` 实例 ✅
- 三个 gating 均已实测：`enabled` / `prob` / **实际 Mosaic 实例的 `p`** ✅
- **不使用 `hyp.mosaic`** 判断 Mosaic 状态——直接读 `mosaic_ref.p`（真实 runtime state）✅

---

## 6. OASA OFF equivalence（硬门禁）

`object_scale_aug=false` 时 `__call__` **立即返回、不做任何修改**：

| 检查 | 结果 |
|---|---|
| 返回同一对象（`out is labels`） | **True** ✅ |
| `img` 未变 | **True** ✅ |
| `cls` 未变 | **True** ✅ |
| `instances.bboxes` 未变 | **True** ✅ |
| `ds[0]` 正常产出 | `img=(5,1280,1280) bboxes=(22,4) cls=(22,1)` ✅ |

⇒ **OASA OFF 与 baseline pipeline 逐位等价**（不只是"数值接近"，是恒等返回）。

---

## 7. OASA ON geometry（paired）

同一批 1080p 含 small GT 的样本：

| 量 | 值 |
|---|---|
| applied | **60/60**（prob=1.0） |
| **anchor √area（Mosaic 输入空间）** | **21.17 → 42.33 px ⇒ 2.000×**（配置 2.0） |
| **最终输入空间（×Mosaic 0.5）** | **baseline 10.58 → OASA 21.17 px ⇒ 2.000×** |
| anchor 保留 | **60/60** ✅ |
| ROI 模式 | centered 31 / clipped 29 / recentered 0 / **skip 0** ✅ |
| GT 数量 | 742 → 497（−33%，与 STEP 2.5 一致） |

> **不接受 1.3×/1.5×/1.7× 之类"有改善"的表述** —— 实测就是 **2.000×**，与配置一致。

---

## 8. close_mosaic gating

```python
ds.close_mosaic(ds.hyp)      # 真实入口：设 hyp.mosaic=0 **并重建 transforms**
```

| 检查 | 结果 |
|---|---|
| 重建后 `ObjectScaleAug.enabled` | `True`（config 未变） |
| 重建后 **`mosaic_ref.p`** | **0.0** ✅ |
| close_mosaic 期 `applied` | **0** ✅ |
| close_mosaic 期 `skip_nomosaic` | **20** ✅ |

⇒ **OASA 只在 Mosaic 真正开启时触发**，与 §3 的强制要求一致（避免 close_mosaic 期再叠 2×）。

---

## 9. Validation isolation

| 检查 | 结果 |
|---|---|
| val dataset `augment` | **False** |
| val transforms | **`['LetterBox', 'Format']`** |
| 是否含 `ObjectScaleAug` | **False** ✅ |
| val 样本产出 | `img=(5,736,1312) bboxes=(9,4)` 正常，bbox 在 [0.023,0.828] 内 ✅ |

⇒ **val/test 走的是另一条 Compose 分支，OASA 结构性不可能触发**（不是靠 flag 短路）。

---

## 10. Visual smoke

`diagnostic/oasa_smoke/oasa_smoke_0{0,1,2,3}.png`（4 组，各含 before / after 的 **visible + infrared + depth** 三联 + bbox 叠加）：

`oasa_smoke_00.png` `01` `02` `03`（约 2.8–3.1 MB each）

人工可核对：anchor 明显放大、bbox 跟随、三模态同一变换、无错位。

---

## 11. Batch smoke

真实训练入口，**3 个 batch 的 forward + loss + backward**：

```
batch 0: img=(2,5,1280,1280) cls=14 bboxes=14 loss_sum=23.5914 |grad|=8.284e+05 NaN=False
batch 1: img=(2,5,1280,1280) cls=18 bboxes=18 loss_sum=23.4340 |grad|=8.897e+05 NaN=False
batch 2: img=(2,5,1280,1280) cls=24 bboxes=24 loss_sum=25.1012 |grad|=6.616e+05 NaN=False
```

| 检查 | 结果 |
|---|---|
| crash | 无 ✅ |
| NaN / Inf | 无 ✅ |
| bbox 合法（0≤x≤1, w>0, h>0） | 通过 ✅ |
| channel 数 | **5** ✅ |
| shape | 恒定 (2,5,1280,1280) ✅ |
| object count 突变 | 无（cls/bboxes 数量一致）✅ |

---

## 12. Runtime provenance

```text
git commit        : f659609c28f6b4f500212197d5229550a36d12d8
working tree      : 有未提交改动（见下表），非 clean —— 明确记录
config path       : configs/oasa_pre_mosaic.yaml
  sha256          : 7fce03dc281dd59b1912d9a84cd33156…
ultralytics/data/augment.py     : 6cd355bf1ca98989f695fc5fe4117691…
ultralytics/cfg/default.yaml    : 991a89b32de769d1d444cf34ebb978f1…
scripts/train.py                : 9f55b09f9e1229b61fefd49622d31d30…
.gitignore                      : fc912dd9b6745dc5b8b7b09f35463396…
OASA 参数        : enabled=true prob=0.5 scale=2.0 small_area=1024.0 min_visible=0.8 log_every=200
Mosaic runtime   : 正常期 p=1.0（OASA 可触发）；close_mosaic 期 p=0.0（OASA 自动 OFF）
```

未提交改动（均为本项目先前已批准的改动 + 本轮 OASA）：
`.gitignore`、`scripts/{check_submission,official_map,predict_rect,preflight_check_train_config,train}.py`、
`ultralytics/cfg/default.yaml`、`ultralytics/data/augment.py`、新增 `configs/oasa_pre_mosaic.yaml` 与若干 reports。

---

## 13. Remaining risks

| # | 风险 | 级别 | 说明 |
|---|---|---|---|
| R1 | **`augment.py` 需手动同步云端** | **P0（运维）** | 本仓库无自动同步，云端靠上传 `ultralytics/`。现已 `git ls-files` 可见，但**开训前必须在云端 `grep ObjectScaleAug` 并核对 args.yaml** |
| R2 | anchor 抽样用全局 `random` | P2 | STEP 2.5 用 `default_rng(i)`、正式实现用全局 `random.choice` —— 两者都通过 seed 复现，但**具体 anchor 不同**；两次实测都得到 2.000× |
| R3 | 360p 未在 TEST 3 重测 | P2 | STEP 2.5 已实测 360p = 2.000×；TEST 3 只覆盖 1080p |
| R4 | GT 丢弃 −33%（受影响样本内） | 已量化 | dataset 层面 small −1.6% / medium −5.3% / large −7.4%，与既有 crop 审计吻合 |
| R5 | `prob=0.5` ⇒ 约一半合格图生效 | 设计内 | 每 epoch 的 anchor 不同（随机），deviation 由 seed=42 控制 |
| R6 | `base.py` / `loaders.py` 仍未被追踪 | 既有风险 | 与本实验无关，属独立的 provenance 缺口（已在 `OASA_PHASE2_GATE_STATUS.md` 记录） |

---

# VERDICT

```text
STEP 3 = PASS
```

全部 11 项门禁通过：语法/import ✅｜OFF 恒等等价 ✅｜ON 几何 **2.000×** ✅｜close_mosaic gating ✅｜
val 隔离 ✅｜visual smoke ✅｜batch smoke（3 batch forward+backward）✅｜config 链逐层验证 ✅｜
provenance 完整 ✅｜未扩容 gitignore ✅｜未改 `base.py` ✅

**唯一阻塞项是运维性的 R1**：`augment.py` 必须确认已同步到云端并在开训日志里看到 `[OASA]` marker。

**下一步（A100 smoke）等你指令后才执行。本轮到此停止。**

```text
本轮训练次数 = 0 ；A100 = 0 次 ；正式 submission = 0 次
```
