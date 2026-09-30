# STEP 4B — A100 SMOKE AFTER NATIVE-AREA FIX

**日期**：2026-09-23
**本轮改动**：`ObjectScaleAug` 的 small-area eligibility 从 input-space 恢复为 **native-space**（1 处，10 行含注释）
**本环境**：无 GPU、无云端访问 → **STEP C–E 需你在 A100 上执行**；STEP A/B 与全部本地可验证项已完成

---

## Code

**PASS**

### 精确 diff（只有这一处逻辑变化）

```diff
         areas = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
+        # small_area 是 **native 面积口径**（COCO: small < 32² = 1024 native px）。
+        # 但此刻 b 位于 load_image 之后的 input space（本数据集长边已被缩到 imgsz），
+        # 因此必须把 input-space 面积换算回 native 再比较。
+        # 否则对 92.4% 的 1080p 图，阈值等效被放宽到 1024×2.25=2304 native px²，
+        # 命中的锚点总体将偏离预注册定义（实测 eligible 38.2% vs 预注册 25.1%）。
+        # NOTE: `areas`（input space）仍用于 visible_fraction 的比值计算 —— 比值与空间无关。
+        _o, _r = labels.get("ori_shape"), labels.get("resized_shape")
+        if _o and _r and _r[0] * _r[1] > 0:
+            areas_native = areas * (_o[0] * _o[1]) / (_r[0] * _r[1])
+        else:
+            areas_native = areas
+        small = np.where(areas_native < self.small_area)[0]
-        small = np.where(areas < self.small_area)[0]
```

### 参数未变（逐项 grep 核对）

```
self.prob        = float(prob)         ← 0.5 未动
self.scale       = float(scale)        ← 2.0 未动
self.small_area  = float(small_area)   ← 1024.0 未动（**没有**改成 2304/2048）
self.min_visible = float(min_visible)  ← 0.8 未动
```

**未改**：Mosaic / 模型 / loss / optimizer / dataset / split / evaluator / inference / `base.py`。
`git diff --stat`：

```
 scripts/train.py             |   9 ++         （STEP 3 的透传，非本轮）
 ultralytics/cfg/default.yaml |  10 +++        （STEP 3 的默认键，非本轮）
 ultralytics/data/augment.py  | 208 +++++++++  （含 STEP 3 新增类 + 本轮 1 处修复）
```

---

## Small-area definition

**PASS** —— 本地用**全部 1600 张 train 图**、`prob=1.0` 干净计数（等价于云端会看到的 eligible）：

| 项 | 实测 | 目标 |
|---|---:|---:|
| **native small GT** | **1392 / 11560 = 12.0%** | **12.0%** ✅ |
| **native small-containing images** | **401 / 1599 = 25.1%** | **25.1%** ✅ |
| **OASA eligible（= applied @prob=1.0）** | **401 = 25.1%** | **≈25.1%** ✅ |
| *（修复前）* | *611 = 38.2%* | — |

**eligible 已从 38.2% 精确回落到 25.1%**，与预注册定义逐位吻合。

---

## Runtime

**PASS**（本地全量 1600 图；云端 per-worker 计数待 STEP C 复核）

| 项 | 结果 |
|---|---|
| accounting closure | `applied 401 + skip_prob 0 + skip_nosmall 1198 + skip_nomosaic 0 + skip_vis 0 = 1599 = seen` ✅ **闭合** |
| `anchor_kept` | **401 / 401** ✅ |
| `anchor visibility` | median **1.000**，`skip_visibility = 0` ✅ |
| Mosaic trigger | `mosaic_ref.p = 1.0`（本地构建）；云端上一轮 `mosaic_on=1` 每行皆是 ✅ |
| validation OFF | val transforms = `['LetterBox','Format']`，**不含 ObjectScaleAug** ✅ |

---

## Geometry

**PASS**

| 项 | 值 |
|---|---|
| expected scale | **2.0**（`object_scale_aug_scale`） |
| observed scale | **2.000×**（anchor √area input-space 中位 **19.18 → 38.37**） |

> 注意口径：本轮 19.18 是**修复后的锚点总体**（native small）；
> STEP 2.5 的 17.18 是 STEP 2.5 harness 用 native 面积选出的另一批样本；
> 两者**不可混为一谈**，但**都得到 2.000×**。修复未改变几何。

---

## Training

| 项 | 结果 | 来源 |
|---|---|---|
| 5ch | ✅ | 云端上一轮：`YOLO11m_sepstem ... 20,038,996 params (fused)`（未融合 20,061,972）；本地 3-batch `img=(2,5,1280,1280)` |
| loss | ✅ 有限 | 云端上一轮 box 1.587→1.492 / cls 3.266→2.441 / dfl 1.385→1.314；本地 3 batch 23.59 / 23.43 / 25.10 |
| grad | ✅ 有限 | 本地 `\|grad\| = 8.28e5 / 8.90e5 / 6.62e5` |
| NaN / Inf | ✅ 无 | 云端与本地均无 |
| bbox validity | ✅ | 本地 3 batch 断言 `0<=x<=1, w>0, h>0` 全通过 |
| CUDA error | ✅ 无 | 云端 GPU_mem 19.2G / 40G |

> 本轮修复**只改 eligibility 判据**，不触及形状/通道/数值路径 → 上述训练侧结论不受影响。
> 但按纪律，**STEP C 的云端 smoke 仍须重跑并重新确认一遍**（见下）。

---

## Checkpoint

| 项 | 结果 |
|---|---|
| loadable | ⬜ 待 STEP C（上一轮已证明可加载并完成 398 图验证） |
| SHA256 | ⬜ 待提供 |

---

## Sync verification

**PASS**（`scripts/verify_oasa_sync.py`，本地已跑通）

```
[PASS] augment.py 含 `class ObjectScaleAug`
[PASS] augment.py 含 pre_mosaic marker
[PASS] augment.py 在 v8_transforms 中实例化 OASA
[PASS] 4 个文件 SHA256 全部匹配
[PASS] train.py 解析出全部 OASA 键 ；trainer.args 六项全部 = True/0.5/2.0/1024.0/0.8/200
[PASS] dataset.transforms[0] 是 ObjectScaleAug ；enabled=True ；mosaic_ref.p=1.0
[PASS] val transforms 不含 ObjectScaleAug  — ['LetterBox','Format']
CLOUD_SYNC = PASS
```

---

## STEP C–E：请你在 A100 上执行（我无云端访问）

### 需要重新同步的文件（SHA256，供核对）

| 文件 | SHA256（完整） |
|---|---|
| **`ultralytics/data/augment.py`（必须覆盖）** | `5cb9a407625921ce3f12ffc13df2d703f0a49d9ce1ba067edaf025bced96c518` |
| `scripts/verify_oasa_sync.py`（EXPECT 已更新） | `7982efe5227f2d166936bfabb75833fd250d7013e84e5edd0252064ac4ae1dea` |
| `configs/oasa_pre_mosaic.yaml` | `7fce03dc281dd59b1912d9a84cd33156a2b4a2f020fd2c9498c9d1d98ef39640` |
| `configs/oasa_smoke_1ep.yaml` | `d29cea797d63498c152d594e13020016282f33deedd288f0ae9aeda228026634` |
| `ultralytics/cfg/default.yaml` | `991a89b32de769d1d444cf34ebb978f18d310815e364d9883aaca4a8a62db9f5` |
| `scripts/train.py` | `9f55b09f9e1229b61fefd49622d31d303a02382c051585b94e34e826c3633001` |

### 命令

```bash
cd /root/autodl-tmp/urban_multimodal_detection

# ⚠ 上一轮 smoke 目录已存在，ultralytics 会自增到 ..._SMOKE_1ep2
#   若想复用同名目录，先归档旧的（不删，保留证据）：
mv runs/urban_multimodal_det_yolo11_rgbid_oasa_SMOKE_1ep \
   runs/_archive_oasa_SMOKE_1ep_round1

# ---- STEP B（云端）：一致性自检，必须 CLOUD_SYNC = PASS ----
python scripts/verify_oasa_sync.py

# ---- STEP C：重跑 1-epoch smoke（条件与上轮完全相同）----
python scripts/train.py \
  --model_config configs/yolo11m_sepstem.yaml \
  --train_config configs/oasa_smoke_1ep.yaml

# ---- STEP E：checkpoint provenance ----
sha256sum runs/urban_multimodal_det_yolo11_rgbid_oasa_SMOKE_1ep/weights/best.pt

# ---- STEP F：git provenance ----
git status --short
git diff --stat
grep -n object_scale_aug runs/urban_multimodal_det_yolo11_rgbid_oasa_SMOKE_1ep/args.yaml
```

### STEP D 验收判据（请对照日志）

```text
[OASA] ... small_area=1024 ...                        ← 1024 未被改
eligible/(seen−skip_nomosaic) ≈ 25%（不再 ≈38%）      ← 最重要
applied + skip_prob + skip_nosmall + skip_nomosaic + skip_visibility == seen   （每行闭合）
anchor_kept == applied
anchor_vis_median ≈ 1.000 ；skip_visibility == 0
mosaic_on = 1
val 阶段无任何 [OASA] 行
```

---

## Overall

```text
本地侧（STEP A + B + D 的本地等价验证）：PASS
云端 STEP C–F：待你在 A100 上执行
```

**最终 `PASS — ready for 300-epoch official experiment` 我暂不签发** ——
它依赖 STEP C 的云端复核（尤其 `eligible ≈ 25%` 与 checkpoint SHA256）。
把那三条命令的输出贴回来，我立即出最终判决。

### 已明确区分：结构验证 vs 本轮 smoke 验证

| 结论 | 来源 | 状态 |
|---|---|---|
| close_mosaic → OASA OFF | **STEP 3 TEST 4**（真实 `close_mosaic()` 入口） | ✅ **既有结构验证**，本轮未重跑（1 epoch 触发不到，按 §D-6 不延长训练） |
| val → OASA OFF | STEP 3 TEST 5 + 本轮本地 transforms 复核 | ✅ 结构验证 |
| native 口径 eligible ≈25% | **本轮 STEP A/B（本地全量 1600 图）** | ✅ 本轮 |
| 云端 eligible ≈25% | **STEP C** | ⬜ 待执行 |

```text
本轮 300ep 正式训练 = 0 次 ；多 seed = 0 ；submission = 0 ；smoke mAP 未用于判断 OASA 有效性
```
