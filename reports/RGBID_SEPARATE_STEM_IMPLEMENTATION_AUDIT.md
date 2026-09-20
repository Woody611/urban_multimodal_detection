# RGBID Separate-Stem Fusion Implementation Audit

**日期**：2026-09-19 · **性质**：READ-ONLY（未训练、未改项目代码/配置、未创建 run）
**候选机制**：`RGB 3ch → RGB Stem` ＋ `IR 1ch → IR Stem` ＋ `Depth 1ch → D Stem` → Concat → 原 YOLO11 backbone
**审计产物**：本文件。运行期证据由 `%TEMP%` 下的两个一次性脚本生成（**不在仓库内**，未污染项目树）。

---

## 1. Frozen Baseline

```text
Official RGBID mAP50-95 (线上榜单)        = 0.53180   ← OFFICIAL_FALLBACK，不得覆盖
本地官方口径复现 (official_map.py, 400图) = 0.50928
训练内 val (fork, results.csv ep237)      = 0.55800
```

冻结链复核 16/17 完好（唯一不符项为既有 `predict_rect.py` 漂移，非本次造成）。

---

## 2. Current Actual Architecture

官方模型 = `configs/yolo11m_earlyfusion.yaml` + `runs/..._rgbird_ir_quicktest/weights/best.pt`

```text
输入 5ch [B,G,R,IR,D]  ← base.py::_merge_channels_rgbid (cv2.merge)
   ↓
Conv2d(5, 64, k=3, s=2)   ← 唯一与标准 YOLO11m 不同的层
   ↓
标准 backbone → 标准 PAN-FPN head → Detect(nc=12) @P3/P4/P5
```

实测：`backbone / head / scales` 与 3ch 标准 `yolo11_visible.yaml` **逐项相等**，唯一差异 `ch: 5`。
**不存在 fusion 模块、gating、attention、routing。** 唯一可按模态归因的参数块是 stem（3,008 / 20,063,412）。

---

## 3. Proposed Architecture

```
[-1, 1, Silence, []]              # 0  5ch 透传
[ 0, 1, SilenceChannel, [0, 3]]   # 1  → RGB   [B,3,H,W]
[-1, 1, Conv, [48, 3, 2]]         # 2  → RGB   [B,48,H/2,W/2]
[ 0, 1, SilenceChannel, [3, 4]]   # 3  → IR    [B,1,H,W]
[-1, 1, Conv, [8, 3, 2]]          # 4  → IR    [B,8,H/2,W/2]
[ 0, 1, SilenceChannel, [4, 5]]   # 5  → Depth [B,1,H,W]
[-1, 1, Conv, [8, 3, 2]]          # 6  → Depth [B,8,H/2,W/2]
[[2, 4, 6], 1, Concat, [1]]       # 7  → 融合   [B,64,H/2,W/2]
[-1, 1, Conv, [128, 3, 2]]        # 8  ← 原 backbone L1 起（其后整体 +7 位移）
```

**关键可行性结论**：本设计**不需要任何新模块代码**。所用 `Silence` / `SilenceChannel` / `Conv` / `Concat`
**全部已注册**（`ultralytics/nn/modules/__init__.py:81,173`），且已被 `yolo11_midfusion.yaml`、
`yolo11_latefusion.yaml` 两个**已实际训练过**的架构使用。

`Silence.forward` = `return x`（build 时替换为 `nn.Identity`）；`SilenceChannel.forward` = `return x[..., c_start:c_end, :, :]`
—— **纯 torch 切片，梯度安全**；`parse_model` 经 `globals()[m]`（`tasks.py:1059`）解析。

---

## 4. Config → Code → Forward Trace

**本候选最重要的性质：它不需要新增任何 YAML 配置字段。** 整个设计只体现为 model yaml 的 module 列表，
因此**不存在"配置写了但没生效"的可能性**（没有配置项可供失效）。

| 元素 | YAML 位置 | 谁读取 | 最终影响 | 生效 |
|---|---|---|---|---|
| `Silence` | backbone[0] | `tasks.py::parse_model` | 透传 5ch | **YES**（实测 forward 执行） |
| `SilenceChannel[0,3]/[3,4]/[4,5]` | backbone[1],[3],[5] | 同上 | RGB / IR / Depth 切片 | **YES** |
| `Conv[48,3,2]` / `Conv[8,3,2]`×2 | backbone[2],[4],[6] | 同上 | 3→48ch；1→8ch ×2 | **YES** |
| `Concat[1]` | backbone[7] | 同上 | 48+8+8→64 | **YES** |
| `ch: 5` | yaml 顶层 | `DetectionModel` | 保持 5ch 输入 | **YES**（**必须保持 5**，见 §10） |

> **无 HARD FAIL 情形 A/E**：没有新增配置项，因此不可能出现"配置存在但被默认值覆盖 / trainer 没读取"。

---

## 5. Tensor Shape Trace

运行期实测（synthetic `[1,5,128,128]`，CPU）：

| 层 | 模块 | 输出 shape |
|---|---|---|
| 2 | RGB stem | `(1, 48, 64, 64)` |
| 4 | IR stem | `(1, 8, 64, 64)` |
| 6 | Depth stem | `(1, 8, 64, 64)` |
| 7 | Concat | `(1, 64, 64, 64)` |

`concat out channels == sum of stems` → **True**（48+8+8 = 64 = 原 stem 输出通道）。

切分位置**唯一且显式**：三条 `SilenceChannel` 直接在 5ch 张量上按 `[0,3]/[3,4]/[4,5]` 切片，**不存在"先过某模块再猜通道归属"**。

---

## 6. Parameter / FLOPs Budget

| 量 | Baseline | Candidate | Δ |
|---|---:|---:|---:|
| Total params | 20,063,412 | 20,061,972 | **−1,440 (−0.0072%)** |
| Stem params (实测) | 3,008 | 1,568（3 个 stem 合计） | −1,440 |
| GFLOPs @1280（实测） | 273.85 | 272.68 | **−1.17 (−0.43%)** |
| Backbone + Head params | 20,060,404 | 20,060,404 | **0（逐项未动）** |

Baseline stem 实测：`Conv2d(5, 64, 3×3, bias=False)` → conv 2,880 + BN 128 = **3,008**（与先前审计一致）。
Candidate stems：`3×48×9=1296` + `1×8×9=72` + `1×8×9=72` + BN(48+8+8)×2 = **1,568**。

> **Δparams ≈ 0 且为负**；ΔFLOPs −0.43%。**没有因"fusion"偷增模型规模**，满足 §7 的"优先目标 Δparams ≈ 0"。

---

## 7. Runtime Smoke Test

在 synthetic 输入上构建并前向（**未读取、未修改任何正式 checkpoint**）：

| # | 检查 | 结果 |
|---|---|---|
| A | RGB Stem forward 执行 | ✅ hooked，输出 `(1,48,64,64)` |
| B | IR Stem forward 执行 | ✅ hooked，输出 `(1,8,64,64)` |
| C | Depth Stem forward 执行 | ✅ hooked，输出 `(1,8,64,64)` |
| D | Concat fusion forward 执行 | ✅ hooked，输入 `[(1,48,64,64),(1,8,64,64),(1,8,64,64)]` |
| E | Concat 通道数正确 | ✅ 64 = 48+8+8 |
| F | 进入原 backbone 的通道数正确 | ✅ 64（与原 stem 输出一致） |
| — | 模型构建成功 | ✅ 31 层，`Detect(nc=12)` 正常 |

**无 HARD FAIL 情形 F**：smoke test 用的是 `DetectionModel` 本体（与 trainer 同一个类、同一 yaml），**不是 demo 模型**。

---

## 8. Modality Dependency Test

对 synthetic 输入分别单模态扰动，比较输出最大差：

| 扰动 | `max\|Δoutput\|` | 判定 |
|---|---:|---|
| 仅 RGB (ch 0:3) | 7.63e-05 | ✅ 非零 |
| 仅 IR (ch 3) | 3.81e-05 | ✅ 非零 |
| 仅 Depth (ch 4) | 4.58e-05 | ✅ 非零 |

> **无 HARD FAIL 情形 B/C**：三个模态**都真实进入计算图并影响输出**，IR/Depth 均未被旁路。
> （注：此测试只证明"进入计算图"，不证明"有用"。）

---

## 9. Gradient Dependency Test

`forward → dummy loss → backward`，检查各 stem 卷积权重梯度：

| Stem | grad shape | \|grad\| sum | 判定 |
|---|---|---:|---|
| RGB stem (model.2) | `(48,3,3,3)` | 527,417 | ✅ 非 None、有限、非零 |
| IR stem (model.4) | `(8,1,3,3)` | 27,087 | ✅ |
| Depth stem (model.6) | `(8,1,3,3)` | 31,353 | ✅ |

**三个 stem 全部收到梯度**，融合层亦在图中。

---

## 10. Train / Eval Consistency

| 项 | 训练路径 | 官方推理路径 | 一致 |
|---|---|---|---|
| IR 读取 | `base.py:315` `cv2.IMREAD_GRAYSCALE` | `loaders.py:645` `cv2.IMREAD_GRAYSCALE` | ✅ |
| IR 归一化 | 1%/99% 分位拉伸（`percentile` 默认） | 同左（`ir_encoding` 默认 `percentile`） | ✅ |
| Depth 编码 | `<300→0`，`clip(/19999*255)` | 同左 | ✅ |
| 通道顺序 | `cv2.merge((b,g,r,ir,depth))` | 同左 | ✅ |
| resize | `_resize_images_3`（本数据三模态同尺寸 → no-op） | 尺寸不等才 resize（→ no-op） | ✅ |
| 模型构建 | `DetectionModel(yaml)` | `predict_rect.py:174` `YOLO(weights)` → **从 checkpoint 读架构** | ✅ |

> **无 HARD FAIL 情形 D/H**。候选**保持 `ch: 5` 输入**、切分放在模型内部，数据管线**逐字未动**。
> `predict.py:350` 的 `nn_model.yaml.get("ch", channels)` 读的是 **checkpoint 内嵌 yaml**，候选 `ch:5` 不变 → 通过。

---

## 11. Augmentation Audit

5 通道下的实际状态（代码 + 实测确认）：

| Augmentation | Config | Code condition | 5ch 实际状态 |
|---|---|---|---|
| RandomHSV | `hsv_h/s/v=0.015/0.7/0.4` | `augment.py:3308` `if img.shape[-1]!=3: return labels` | **DISABLED** |
| Albumentations (+erasing) | `erasing=0.4` | `augment.py:3800` `if channels==5: alb=Albumentations(p=0)` | **DISABLED** |
| Mosaic | `mosaic=1.0` | 通道无关（`np.full(..., img.shape[2])`） | **ACTIVE** |
| RandomFlip (h/v) | 默认概率 | 通道无关 | **ACTIVE** |
| RandomPerspective / affine | `scale/translate/degrees` | 5ch 用 `borderValue=114`（`augment.py:1097`） | **ACTIVE** |
| LetterBox | 推理/val | 5ch 落入 `else` → `np.full(...,114)`（`augment.py:1662`） | **ACTIVE** |
| MixUp | `mixup=0.0`（默认） | 通道无关 | 配置关闭 |

> **`OUT_OF_SCOPE`**：HSV/Albumentations 在 5ch 下失效是**既有缺陷**，按 §14 **本次不修** ——
> 与 Fusion 同修即出现第二个变量。候选保持 5ch 输入，故 augmentation 状态与 baseline **逐项相同**。

---

## 12. Single-Variable Audit

| Component | Baseline | Candidate | Changed? |
|---|---|---|---|
| Dataset / Split / ImgSize | `rgbid_split` / 1280 | 同 | NO |
| Augmentation | 5ch 实际状态（§11） | 逐项相同 | NO |
| Backbone / Head | 标准 YOLO11m | 逐项相同（params 完全相同） | NO |
| `ch` / 通道顺序 | 5 / `[B,G,R,IR,D]` | 同 | NO |
| Loss / Optimizer / LR / Epochs / Pretrained 源 | 官方 | 同 | NO |
| Evaluation pipeline | `predict_rect.py` + `official_map.py` | 同 | NO |
| **Fusion** | **单 5ch stem** | **3 stem + concat** | **YES** |

**仅 Fusion 一处变化**，其余逐项冻结。

---

## 13. Hard-Fail Check

| 情形 | 判定 |
|---|---|
| A 配置写了但代码仍走旧路 | ✅ 未触发（无新增配置项，§4） |
| B 建了三分支但 forward 只用 RGB | ✅ 未触发（§7、§8 实测） |
| C IR/Depth 执行但不参与输出 | ✅ 未触发（§8、§9 实测） |
| D 训练走新 fusion、官方评测走旧 5ch | ✅ 未触发（§10，`predict_rect` 从 checkpoint 读架构） |
| E 新增参数走默认值 | ✅ 未触发（无新增配置项） |
| F smoke test 用的是 demo 模型 | ✅ 未触发（§7，用的就是 `DetectionModel`） |
| **G 加载 pretrained 时 silent fallback 到随机初始化** | ❌ **已触发 —— 运行期实测，见下** |
| H 官方评测预处理与训练不一致 | ✅ 未触发（§10） |
| I 为跑通而偷改 aug/dataset/eval | ✅ 未触发（未改任何文件） |

### G 的运行期证明（本次审计的决定性发现）

**复刻 `DetectionTrainer.get_model()` 的真实加载路径**（`ultralytics/models/yolo/detect/train.py:186-196`）：

```
model = DetectionModel(yaml)
weights, _ = attempt_load_one_weight('yolo11m.pt')
model.load(weights)                    # intersect_dicts：按「名字+形状」匹配
_transfer_rgb_pretrained(model, weights)
```

对**结构对应**的 backbone/head 张量逐层比对"是否等于预训练值"：

| 模型 | `load()` 报告 | `_transfer_rgb_pretrained` 返回 | **结构对应张量真正等于预训练的比例** |
|---|---|---:|---:|
| **Baseline**（现官方架构，`ch:5`） | 642/649 | 5 | **532/538 = 98.9%** ✅ |
| **Candidate**（3-stem） | **61/661** | **9** | **1/538 = 0.2%** ❌ |

逐层明细：stock L1–L23 → candidate L8–L30，**每一层都是 0/…**（唯一 1/97 出现在 Detect 头）。

**根因**（`detect/train.py:21-116`）：`_transfer_rgb_pretrained` 只有两条**硬编码分支**——

1. **RGBID 早期融合**（L57-58）：要求存在 `in=5` 的 stem **且无任何 `in=1` 的 stem**。候选是 `3/1/1`、**无 `in=5`** → **跳过**。
2. **RGBD 中期融合**（L76-116）：只要存在 `in=1` 的 stem 就进入，但 ① 用**硬编码 `pairs` 索引表**（L84-90），
   该表为 `concat_res` 布局而写（键 `2,3,4,5,6,16,17,23,24,30,31,34,37,38,40,41,43,44`），与候选**完全不对应**；
   ② `depth = next(...in_channels==1)`（L105）**只取第一个** 1ch stem，候选有 IR/Depth **两个** → **另一个静默保持随机**。

实测确认：`candidate IR stem initialized from stock? False`、
`candidate Depth stem initialized from stock? False`。

**后果**：若以 `pretrained: yolo11m.pt` 直接开训，实际得到的是
**近乎从零训练（body 0.2% 预训练）**，而不是设计中的"预训练微调"。
这与本项目的历史事故同类（记忆 `fusion-pretrained-remap`：融合模型 RGB 分支曾因 Silence 前缀
index 偏移而从未真预训练）。**这正是本审计要防的"训练了 11 小时却不是我们设计的那个模型"。**

---

## 14. Final Decision

```text
IMPLEMENTATION_NOT_READY
```

**架构本身可行**（§3、§5–§9 全部通过，且 Δparams ≈ 0、无需新模块、无需新配置项、单变量干净）。
**但唯一阻塞项是 pretrained 迁移路径，且它是 P0 级**：不改就会静默退化为从零训练。

### 需要的改动（最小、可验证、不触碰 baseline）

| # | 文件 | 目的 | 风险 |
|---|---|---|---|
| 1 | `ultralytics/models/yolo/detect/train.py` | 在 `_transfer_rgb_pretrained` 增加**第三条分支**：识别 `3ch + 1ch + 1ch` 三-stem 布局，按**结构对应**（而非硬编码索引表）把 stock L1–L23 映射到候选 L8–L30；三个 stem 分别初始化（RGB stem ← stock 前置 3ch；IR/Depth stem ← `mean(W_R,W_G,W_B)`） | 中：需正确处理 +7 位移与两个 1ch stem |
| 2 | `configs/yolo11m_sepstem.yaml`（**新建**） | 承载 §3 的 backbone/head；**文件名必须含 `yolo11m`** 以让 `guess_model_scale` 解析为 `scale='m'` | 低 |
| 3 | `configs/train_rgbid_sepstem.yaml`（**新建**） | 复制官方 train config，仅改 `experiment_name` + `model_config`；`channels: 5`、`use_simotm: RGBID`、`pretrained` 保持 | 低 |
| 4 | `scripts/verify_sepstem_pretrained.py`（**新建**，可选） | 复用既有 `scripts/verify_ir_pretrained_remap.py` 的模式，断言迁移率 ≈99% 而非 0.2% | 低 |

**验收断言（开训前必须通过，否则不得启动训练）**：
```text
load() 报告 transferred            ≈ 643/649（而非 61/661）
结构对应张量等于预训练的比例        ≥ 98%（而非 0.2%）
IR stem  ← mean(R,G,B)               True
Depth stem ← mean(R,G,B)             True
guess_model_scale(候选 yaml)         == 'm'（而非回退 'n'）
```

### 附带风险（已在本次实测中暴露，必须在建 yaml 时规避）

**scale 由文件名正则决定**：本次第一轮用**内存 dict** 构建候选，`detect/train.py` 打印
`WARNING ⚠️ no model scale passed. Assuming scale='n'`，模型被静默建成 **nano（2.59M）**而非 YOLO11m（20.06M）。
故**建 yaml 时必须让文件名含 `yolo11m`**，否则实验同样是"训练了另一个模型"（与记忆 `model-scale-gotcha` 同源）。

---

## 附：本次审计的只读保证

未训练、未创建 run、未生成 submission；未修改仓库内任何文件（工作树新增的只有本报告）；
未覆盖任何 checkpoint（`yolo11m.pt` 仅只读加载）。运行期脚本 `%TEMP%\audit_sepstem_{smoke,pretrained}.py`
与 `%TEMP%\yolo11m_sepstem_audit.yaml` **均在仓库之外**。
`FREEZE_MANIFEST` 复核 16/17 完好（唯一不符项为既有 `predict_rect.py` 漂移）。

```text
OFFICIAL_FALLBACK = 0.53180（未变更）
DECISION: IMPLEMENTATION_NOT_READY
```
