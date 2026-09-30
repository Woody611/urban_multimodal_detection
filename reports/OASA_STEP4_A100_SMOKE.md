# STEP 4 — A100 Smoke Run（结果与判决）

**日期**：2026-09-23 · **云端**：A100-PCIE-40GB · 1 epoch · 200 iter
**目录**：`runs/urban_multimodal_det_yolo11_rgbid_oasa_SMOKE_1ep/`

---

## STEP 4 A100 SMOKE

```text
Cloud code:
  ObjectScaleAug present = PASS
  pre_mosaic marker      = PASS      （日志 8 行 [OASA] ... insertion=pre_mosaic）

Config:
  args.yaml values       = PASS      （6 个键全部落地，见下）

Runtime:
  Mosaic OASA trigger    = PASS      （mosaic_on=1；applied=38/36/36/41/73/81/74/86 全 >0）
  close_mosaic OASA off  = PASS      （1 epoch 未覆盖；STEP 3 TEST 4 以真实 close_mosaic() 入口验证）
  validation OASA off    = PASS      （val 阶段无任何 [OASA] 行；val transforms 本地验证 = [LetterBox, Format]）

Geometry:
  expected ~2×           = PASS      （见 §几何，含口径问题说明）

Training:
  forward / backward     = PASS      （box 1.587→1.492, cls 3.266→2.441, dfl 1.385→1.314 全部有限）
  NaN / Inf              = PASS      （无）
  5ch                    = PASS      （模型 20,038,996 params(fused)，与 sepstem 架构一致）
  CUDA error             = PASS      （无；GPU_mem 19.2G / 40G）

Checkpoint:
  loadable               = PASS      （"Validating .../best.pt..." 成功加载并完成 398 图验证）
  SHA256                 = ⬜ 未提供（请补：`sha256sum runs/..._oasa_SMOKE_1ep/weights/best.pt`）

OVERALL:
  STEP 4 = PASS（运行闭环）—— 但发现 1 项 SPEC CONFORMANCE DEFECT，见 §缺陷
```

`args.yaml` 实测：

```
124:object_scale_aug: true
125:object_scale_aug_prob: 0.5
126:object_scale_aug_scale: 2.0
127:object_scale_aug_small_area: 1024.0
128:object_scale_aug_min_visible: 0.8
129:object_scale_aug_log_every: 200
```

---

## 运行期证据（每 worker 一份统计，共 4 个 worker）

```
[OASA] enabled=True insertion=pre_mosaic scale=2.0 prob=0.5 small_area=1024 min_visible=0.8 version=v1
[OASA] seen=200 mosaic_on=1 eligible=79 applied=38 skip_nomosaic=0 skip_prob=41 skip_nosmall=121
       skip_visibility=0 mode(center/clip/recenter)=17/21/0 anchor_vis_median=1.000 anchor_kept=38 gt 382->248
...
[OASA] seen=400 ... applied=86 ... skip_prob=62 skip_nosmall=252 gt 1048->680
```

**完整性核对**（逐 worker 逐 200 样本，每一项都精确闭合）：

| worker | seen | applied | skip_prob | skip_nosmall | skip_nomosaic | skip_visibility | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 38 | 41 | 121 | 0 | 0 | **200 ✅** |
| 2 | 200 | 36 | 38 | 126 | 0 | 0 | **200 ✅** |
| 3 | 200 | 36 | 48 | 116 | 0 | 0 | **200 ✅** |
| 4 | 200 | 41 | 36 | 123 | 0 | 0 | **200 ✅** |
| 1 | 400 | 73 | 83 | 244 | 0 | 0 | **400 ✅** |
| 2 | 400 | 81 | 73 | 246 | 0 | 0 | **400 ✅** |
| 3 | 400 | 74 | 80 | 246 | 0 | 0 | **400 ✅** |
| 4 | 400 | 86 | 62 | 252 | 0 | 0 | **400 ✅** |

**关键读数**：

- `mosaic_on=1` 每行皆是 → OASA gating 读的是**真实 Mosaic runtime state** ✅
- `skip_nomosaic=0` → 1 epoch 全程 mosaic 开启 ✅
- **`anchor_kept == applied`（8/8 行）→ anchor 从未被裁掉** ✅
- `anchor_vis_median = 1.000` ✅
- `skip_visibility=0` → 0.8 可见度门限从未失败 ✅
- `recentered=0`，centered/clipped 分布与 STEP 2.5 一致 ✅
- GT 保留率 382→248 / 551→338 / 484→313 / 1035→… ≈ **35–40%**，与 STEP 2.5 实测吻合 ✅
- `prob=0.5` 生效（`skip_prob` ≈ eligible 的一半）✅

> `seen` 是**per-worker 计数**（`num_workers=4`），不是全局；8 行 = 4 worker × 2 次 log。**不是 bug**，解读时注意。

---

## ⚠️ 缺陷：`small_area` 判据用的是**输入空间**，不是 native（SPEC CONFORMANCE DEFECT）

### 证据链（三步独立吻合）

| 口径 | 含 small GT 的 train 图 | small GT 占比 |
|---|---:|---:|
| **native**（`area_native < 1024`）—— 任务书 §5 与既有审计的定义 | **401 / 1600 = 25.1%** | 1392 / 11570 = **12.0%** |
| **输入空间**（`area_input < 1024`）—— 本实现的实际行为 | **611 / 1600 = 38.2%** | — |
| 云端 smoke `eligible` 实测 | **≈ 38%**（74–86 / 200） | — |

⇒ native 口径**精确复现**审计记录的 25.1% / 12.0%；
⇒ 输入空间口径**精确复现**云端实测的 38%。**两步都吻合，说明判断成立。**

### 根因

`ObjectScaleAug.__call__` 里：

```python
H, W = img.shape[:2]                       # ← 1280 输入空间（不是 native）
areas = np.clip(b[:,2]-b[:,0],0,None) * np.clip(b[:,3]-b[:,1],0,None)
small = np.where(areas < self.small_area)  # ← 1024 被拿来与**输入空间**面积比
```

`load_image` 已把长边缩放到 1280，所以对 92.4% 的 1080p 图，
`area_input = area_native / 2.25` ⇒ 实际等效于 native 阈值 **1024 × 2.25 = 2304 px²**。

**影响**：OASA 命中的是"输入空间里 <32²"的目标，对应 native 最大 **48²**；
比预注册的"native <32²"**宽 52% 的图**（611 vs 401），锚点也系统性偏大。

**几何本身不受影响**（仍精确 2.000×），**单变量性质不受影响**，但
**跑的锚点总体不是预注册的那个总体** —— 违反任务书 §5
「必须复用该定义，不得自己重新发明 threshold」。

> 也解释了 STEP 2.5 与 STEP 3 TEST 3 数字不同：
> STEP 2.5 harness 用 **native** 面积选 anchor（8.59 → 17.18 px），
> 正式实现用**输入空间**面积选（TEST 3：10.58 → 21.17 px）——**两者锚点总体不同**。

### 最小修复（1 行，不改变实验定义，只是**恢复**预注册定义）

`labels` 在 OASA 入口已带 `ori_shape`（native）与 `resized_shape`（= `img.shape`），**实测可用**：

```
ori_shape=(360,640)  resized_shape=(720,1280)  面积比 = 0.2500（与预测一致）
```

```python
# 把输入空间面积换算回 native 再比较
_o, _r = labels.get("ori_shape"), labels.get("resized_shape")
if _o and _r and _r[0] * _r[1] > 0:
    areas = areas * (_o[0] * _o[1]) / (_r[0] * _r[1])
small = np.where(areas < self.small_area)     # 现在 small_area 是 native 语义
```

修复后预期：`eligible` 从 ≈38% 回落到 ≈25%，锚点变为真正的 native small GT。

**这是"恢复预注册定义"，不是"修改实验设计"** —— 故不建议按 §18 的 `BLOCKED` 处理，
但仍**等你确认后**再改（它改变锚点总体，属设计参数）。

---

## 其它观察（非缺陷）

- `bicycle` P=0.177 / `sign` R=0.187 / `ball` P=R=0 —— **1 epoch 的训练伪影**，非质量信号。
  按 §9，**不得**用 smoke mAP（0.259）判断 OASA 是否有效。
- `No module named 'seaborn'` → ConfusionMatrix 画图失败：**云端环境缺 seaborn，与本实验无关**。
- 模型 `20,038,996 params (fused)` 与未融合的 20,061,972 一致（融合剥离 BN）✅

---

## 待你裁决 / 补齐

1. **是否应用上面那 1 行口径修复**（恢复 native 定义）—— 建议：**是**，然后重跑一次 1-epoch smoke 确认 `eligible ≈ 25%`。
2. **补 `best.pt` 的 SHA256**（`sha256sum runs/urban_multimodal_det_yolo11_rgbid_oasa_SMOKE_1ep/weights/best.pt`）。
3. 是否补跑 `verify_oasa_sync.py`（STEP 4 §2 要求的静态 SHA 比对）——运行期 marker 已间接证明代码到位，但该脚本的输出未提供。

**在这 3 项落定前，不进入 300 epoch 正式训练。**

```text
本轮 300ep 正式训练 = 0 次 ；正式 submission = 0 次
```
