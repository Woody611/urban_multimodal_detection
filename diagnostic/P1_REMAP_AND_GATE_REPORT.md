# P1 — M4+1ch Residual Depth Pretrained Remap + Modality Contract Gate 修复

**日期** 2026-10-04 · **性质** CPU-only / read-patch-probe / 未训练 / 未推理 / 未接触 Seed-2
**脚本** `diagnostic/p1_remap_audit.py` · `diagnostic/fusion_phase0_contract_selftest.py`
**产出** `diagnostic/p1_remap_audit/{p1_audit.json, mapping_manifest_candC.json}` · `diagnostic/p1_run.log`

---

## 1. Status

```text
P1_STATUS    = PASS
3M_READY     = YES
CUDA_USED    = NO
SEED2_TOUCHED = NO
```

**但 `3M_READY = YES` 只表示"技术可执行"，不表示"值得执行"**：Phase F 的
`DEPTH_ATTRIBUTION = NOT_SUPPORTED` / `NEXT_GPU = NO-GO` 未被本轮改变（见 §11）。

---

## 2. Provenance

```text
HEAD (P1 开始时) = 5f1a8d5802aa49adf35e8fa6fb112a1030b3e6d3
HEAD (现在)      = 878cbcdeb40c92590b593e5b993885ce41735223   ← "complete  experiments"
                    ⚠ 该 commit 由**外部**在本次会话进行中产生，**不是我提交的**。
                    我的 P1 改动随之进入 878cbcd（`--stat`：train.py +337 / validate_modality_contract +297 /
                    pretrained_coverage +394 / p1_remap_audit +442 / contract_selftest +110）。
未提交           = diagnostic/fusion_phase0_contract_selftest.py (+73，P1 的 coverage 用例)
                   diagnostic/p1_remap_audit/p1_audit.json、diagnostic/p1_run.log（重跑产物）
```

| SHA-256 (16) | git | 文件 |
|---|---|---|
| `71096a9e044652a5` | tracked | `ultralytics/models/yolo/detect/train.py` |
| `79c8dbab9cd810d2` | tracked | `ultralytics/nn/tasks.py` |
| `bf380a22d2491f73` | ⚠ **IGNORED** | `ultralytics/data/base.py` |
| `56c07c8520cf5b25` | ⚠ **IGNORED** | `ultralytics/data/loaders.py` |
| `abfc90f53f75f7ad` | tracked | `ultralytics/data/augment.py` |
| `cf06d3a61a5cc9a9` | tracked | `ultralytics/cfg/default.yaml` |
| `67cb9fe02757fd59` | tracked | `scripts/train.py` |
| `9d704300143a5786` | tracked | `scripts/predict_rect.py` |
| `e82128abc082ed05` | tracked | `scripts/official_eval.py` |
| `e08c36340b5c45c5` | tracked | `scripts/validate_modality_contract.py` （**本 P1 修改**） |
| `e56507d66d09605c` | tracked | `scripts/pretrained_coverage.py` （**本 P1 新增**；独立复核轮补 manifest 的 `source_shape`/`exact_equal` 两字段） |
| `1ce1bc7938a2cf5e` | tracked | `diagnostic/p1_remap_audit.py` （**本 P1 新增**） |
| `ca65cc33098284d3` | tracked | `diagnostic/fusion_phase0_contract_selftest.py` （已提交版本） |
| `9b14f29460733384` | tracked | `configs/yolo11m_sepstem.yaml` |
| `edc95f4665ead847` | tracked | `configs/yolo11m_earlyfusion.yaml` |
| `58906628b858274c` | tracked | `configs/yolo11m_modality3ch.yaml` |
| `871d9d0da0122453` | tracked | `configs/yolo11m_modality4ch.yaml` |
| `6e7bfb1555a3c68d` | tracked | `configs/yolo11m_modality2ch.yaml` |
| `d5ffc1a674953a08` | ⚠ **IGNORED** | `yolo11m.pt` |

**⚠ IGNORED / NOT VISIBLE TO GIT STATUS**：`.gitignore:11`（无锚定的 `ultralytics/data/*`）
吞掉整个 `ultralytics/data/`，另加 `yolo11m.pt`。
`base.py`/`loaders.py`（Phase-0 修复的载体）**永远不出现在 `git diff` 里**，只有 SHA-256 可校验。

**本次未触碰**：`configs/train_rgbid_sepstem_clahe.yaml`（D′）、`configs/train_modality_m4_*`、
`configs/*m5*`、`runs/**`、`scripts/predict_rect.py`、`scripts/train.py`、`ultralytics/data/**`、
Depth 预处理、augmentation、optimizer/loss/epochs/imgsz —— `git status` 可证。

---

## 3. Patch

| file | lines | 内容 |
|---|---|---|
| `ultralytics/models/yolo/detect/train.py` | **+255 / −0** | 新增 `_find_residual_depth_adapter()`（strict 布局识别）+ `_remap_residual_depth_adapter()`（fail-closed remap）；在 `_transfer_rgb_pretrained` 中**先于 multi-ch 分派**该分支；新增两条 fail-closed 守卫（multi-ch 的"stem 不在 layer 0"、以及"带 Silence 前缀 + >3ch 首 Conv"的未支持布局） |
| `scripts/pretrained_coverage.py` | **新文件 394 行** | 独立重导映射 + 逐张量 manifest + 覆盖率度量 + fail-closed gate（`analyse/manifest/measure/gate`） |
| `scripts/validate_modality_contract.py` | **改 §9 一节** | ① 补上此前**漏掉的 `model.load()`**（生产路径第一步）；② 判据从 `n_ret>0 and stem_changed` 换为覆盖率门，阈值 `PRETRAINED_COVERAGE_MIN = 0.80` |
| `diagnostic/p1_remap_audit.py` | **新文件 442 行** | 本轮全部证据的可复现驱动 |
| `diagnostic/fusion_phase0_contract_selftest.py` | **+73** | 两个永久回归用例（residual-Depth 布局的对照 + 静默降级缺陷） |

### 修复的语义

```text
input 5ch [B,G,R,IR,D]
  ├─ SilenceChannel[0,4] → Conv(4→C)        ← M4 stem：ch[0:3]=预训练 RGB，ch[3]=mean(R,G,B)；BN←stock model.0.bn
  └─ SilenceChannel[4,5] → ZeroConv2d(1→C)  ← 保持精确零初始化，remap **不写**
                            ↓
                     ADD(alpha=1.0) → M4_stem + 0
                            ↓
              layer i ← stock layer (i − add_idx)，逐张量 shape 校验
```

**为什么必须新增分支**：该布局没有任何 1ch **Conv** stem，此前落入 `multi_ch` 分支
（只写第一个 Conv 就 `return`），而 `Silence` 前缀已使 `load()` 的名字匹配率从 98.9% 崩到 9.7%
⇒ 整个位移后的 body 保持随机初始化。

---

## 4. Mapping Coverage

```text
exact_match_ratio = (# 目标张量 逐位 == 其"应然来源"张量) / (# 应然预训练张量)
```

| model | expected pretrained | exact tensors | tensor coverage | parameter coverage | gate |
|---|---:|---:|---:|---:|:--:|
| **D** | 532 | 532 | **1.0000** | **1.0000** | PASS |
| **D′** | 532 | 532 | **1.0000** | **1.0000** | PASS |
| **M4** | 532 | 532 | **1.0000** | **1.0000** | PASS |
| **candC** | 532 | 532 | **1.0000** | **1.0000** | PASS |

```text
HARD GATE : >= 0.80     （P1 前的 candC 实测 0.0019）
EXPECTED  : 除新增 Depth branch 外不应出现大面积 random-init —— 实测 **1.0000**，达标。
```

> **指标口径说明（重要，避免误读历史数字）**：本轮的 1.0000 与 Readiness Audit 里
> D′ 96.2% / candC 9.7% **不是同一把尺**。旧口径是"该张量是否逐位等于**某个** stock 张量"
> （分子分母都含派生 stem 的 mean 通道，永远无法 100%）；新口径是"是否逐位等于**它应然来源**"
> —— 派生 stem（`DERIVED`）与 Detect 分类头（`NEW_RANDOM`）、ZeroConv（`ZERO_INIT`）
> 都从分母里**排除**，各自单独校验。新口径更严（0.0019 vs 旧的 9.7%）。

**分布**：`PRETRAINED 532 / DERIVED 5 / ZERO_INIT 2 / NEW_RANDOM 6 / other 0`。

---

## 5. Mapping Manifest

`diagnostic/p1_remap_audit/mapping_manifest_candC.json` — 545 条，逐张量含
`target_name / source_name / target_shape(shape) / source_shape / numel / mapping_reason / kind / status / exact_equal`
（`exact_equal`：539 = true，6 = null（`NEW_RANDOM`，无源可比））。

| group | kind | tensors | params |
|---|---|---:|---:|
| Depth ZeroConv | `ZERO_INIT` | 2 | 640 |
| backbone/neck/head | `PRETRAINED` | 532 | 20,096,976 |
| new (Detect classifier) | `NEW_RANDOM` | 6 | 9,252 |
| stem | `DERIVED` | 5 | 2,560 |
| **TOTAL** | | **545** | **20,109,428** |

分类语义：`PRETRAINED` = 必须逐位等于具名 stock 张量；`DERIVED` = 由 stock 切片构造（单独按切片校验）；
`ZERO_INIT` = 必须精确为零；`NEW_RANDOM` = 合法的全新张量（Detect 分类头，nc 改变）；
`other` = 无参数的融合管线层（Silence / SilenceChannel / ADD）。

**无 `UNEXPECTED` 行**：新/零张量中没有任何一个是 stock 的逐位副本（即没有错映射的痕迹）。

---

## 6. Critical Layer Audit

| 层 | 结论 | 证据 |
|---|---|---|
| **M4 stem** (L2, 4ch→64) | ✅ | `ch[0:3]` 逐位 == stock `model.0.conv.weight`；`ch[3]` 逐位 == `mean(W_R,W_G,W_B)`；BN 4 项逐位 == stock `model.0.bn.*[:64]` |
| **Depth ZeroConv** (L4) | ✅ | 2 张量（weight+bias），`max|W| = 0.0e+00`，`max|b| = 0.0e+00`，status 全 `ZERO_OK`；remap 只**校验**不写入 |
| **ADD** (L5) | ✅ | `type=ADD`，`learnable_params = 0`，`alpha = 1.0` ⇒ `x1 + 1.0·x2`，不引入任何可训练参数或初始化 |
| **backbone / neck** | ✅ | L6..L23 ↔ stock L1..L18，532 张量全部逐位相等 |
| **head** | ✅ | 同上（含 L28 = Detect；其 `cv3` 的**空间卷积** `cv3.0.0.0`(256,1,3,3) 正常继承，只有**分类输出层**(12 vs 80) 合法重建 6 张量） |
| **shape mismatch** | 0 | — |
| **unresolved expected mapping** | 0 | — |
| **unexpected mapping** | 0 | — |

> 旧门的一个真实缺陷在此暴露：`validate_modality_contract.py` 此前把 **任何 `.cv3.` 键**
> 都当成"重建"，因此从未发现 detector 的 `cv3.0.0.0` 其实**是**预训练迁移的。新 manifest 按
> remap 的真实条件逐键判定。

---

## 7. Identity-at-Initialization

CPU，64×64 dummy 输入，`model.eval()`：

| 比较 | max_abs_diff | mean_abs_diff | relative | verdict |
|---|---:|---:|---:|:--:|
| candC M4 stem(L2) vs M4 `model.0` | **0.000e+00** | 0.000e+00 | 0.000e+00 | **EXACT** |
| candC fused(ADD) vs M4 stem | **0.000e+00** | 0.000e+00 | 0.000e+00 | **EXACT** |
| ZeroConv2d(Depth) 输出 vs 全零 | **0.000e+00** | 0.000e+00 | 0.000e+00 | **EXACT** |

`depth input max|d| = 3.880`（非零真实输入），而 `max|z| = 0.000e+00` ⇒ 零初始化在**非零深度输入**下仍恒为 0。
**`IDENTITY_AT_INIT = PASS`**（未做任何 tolerance 放宽：逐位相等）。

---

## 8. Negative / Fail-Closed Tests

| # | 注入 | 预期 | 实测 |
|---|---|---|---|
| A1 | ADD 之后插入一层 → body 位移错位 | RAISE | ✅ `no pretrained counterpart for 'model.7.conv.weight'` |
| A2 | Depth 改由 `SilenceChannel[3,4]`（IR 切片）喂入 | RAISE | ✅ `ZeroConv2d at layer 4 is not fed by a SilenceChannel[4,5]; refusing to guess` |
| B1 | 从 src 删除 `model.0.conv.weight` | RAISE | ✅ `stock stem 'model.0.conv.weight' not present` |
| B2 | 从 src 删除中段张量 | RAISE | ✅ `no pretrained counterpart for 'model.15.cv1.conv.weight'` |
| **C** | **用 P1 之前的旧分派跑 candC** | **静默成功但 gate FAIL** | ✅ **`no raise, n_ret=5, coverage_t=0.0019, gate=FAIL`** |
| C2 | 新代码但关掉新分支 | RAISE | ✅ `[multi-ch stem remap] the 4ch stem is at layer 2, not 0 …` |

**测试 C 是本轮的核心反证**：它精确复现了原始事故 ——
旧代码**不报错**、`n_ret = 5`、`stem_changed = True`（旧门判据全部满足），
而真实覆盖率只有 **0.0019**，被新门拦下。
新代码则有两道独立防线：C2 证明**分派守卫**先 raise，C 证明即使分派失效、**覆盖率门**也会拦。

`diagnostic/fusion_phase0_contract_selftest.py` 已把这两个用例永久固化：**self-test 14/14**
（原 12 项全 CAUGHT + P1 对照 PASS + P1 缺陷 CAUGHT）。

---

## 9. Regression

对 `configs/` 下**全部 30 个** yolo11* 模型 yaml，逐一对
「P1 之前的 `_transfer_rgb_pretrained`」与「P1 之后的」做 before/after 对比
（before 由脚本按 exact-string 剥除 P1 的四处插入后重建，见 `p1_remap_audit.build_pre_p1_module`）：

```text
scanned = 30    regressions = 0
```

| 模型 | before n_ret | after n_ret | Δ coverage | 结论 |
|---|---:|---:|---:|---|
| **D** (`yolo11m_earlyfusion`) | 5 | 5 | 0.0000 | 不变 |
| **D′** (`yolo11m_sepstem`) | 550 | 550 | 0.0000 | 不变 |
| **M4** (`yolo11m_modality4ch`) | 5 | 5 | 0.0000 | 不变 |
| M1/M3/M6、sepstem_40_16_8、sepstem_e1、earlyfusion_p3attn、rgbid_p2_probe、15 个 midfusion/rgbt、3 个 visible | 一一相同 | 一一相同 | 0.0000 | 不变 |

**`expected Δ = 0`，实测 Δ = 0，未出现任何变化 ⇒ 无需 STOP。**

同时 `scripts/validate_modality_contract.py --all` 在**新判据下**：

```text
✅ M1 cov=1.0000/1.0000   ✅ M2a cov=1.0000/1.0000  ✅ M2b cov=1.0000/1.0000  ✅ M3 cov=1.0000/1.0000
✅ M4 cov=1.0000/1.0000   ✅ M5  cov=1.0000/1.0000  ✅ M6  cov=1.0000/1.0000  ✅ M7 cov=1.0000/1.0000
CONTRACT 8/8 PASS
```

---

## 10. Final Gate

```text
correct remap                        = PASS  (D/D′/M4/candC 覆盖 532/532)
exact pretrained coverage >= 80%     = PASS  (1.0000)
critical mapping PASS                = PASS  (stem 切片 + ZeroConv + ADD + body/head)
shape mismatch = 0                   = PASS
unexpected mapping = 0               = PASS
unresolved expected mapping = 0      = PASS
ZeroConv exact zero-init             = PASS  (max|W|=0, max|b|=0，非零输入下输出恒 0)
M4 identity-at-initialization PASS   = PASS  (max_abs_diff = 0.000e+00，逐位相等)
negative fail-closed tests PASS      = PASS  (A1/A2/B1/B2/C/C2)
D/D′/M4 regression PASS              = PASS  (30 configs，0 regressions)
CUDA_USED = NO                       = PASS
Seed-2 untouched                     = PASS
────────────────────────────────────────────────
P1_STATUS = PASS      3M_READY = YES
```

---

## 11. 必须并列陈述的限制

1. **`3M_READY = YES` ≠ "该做这个实验"**。Phase F 冻结的
   `DEPTH_ATTRIBUTION = NOT_SUPPORTED` / `NEXT_GPU = NO-GO` 本轮**未改变**：
   `100` vs `010` 的 Depth quality 分离 size-matched 后全不显著（p 0.50–0.97）；
   image-level `valid_fraction` 与 D′ 净增益 **rho = −0.1479, p = 0.0043（方向相反）**；
   D′ − M4 = −0.00931，CI [−0.02036, +0.00943] 含 0。
   **本轮的成果是：如果决定要跑，这个实验现在至少是有解释力的**；不是"值得跑"。
2. **candC 原型仍未进入 `configs/`**（`READY_TO_PROMOTE = YES`，但按 brief §13/§1 本轮不promote、不训练）。
   原型位置：`%TEMP%\f3m_audit\yolo11m_candC_m4stem_zeroinit_depth_residual.yaml`，文本亦内嵌于
   `diagnostic/p1_remap_audit.py::CANDC`（可从仓库独立重建）。
3. **HR/LR 双分层仍在训练集里**（train depth = 1478 PNG(uint16 1920×1080) + 122 JPEG(8bit)，7.6%），
   `base.py` 的 `im_depth.dtype != np.uint8` 门使 LR 绕过 `<300` + `/19999` ⇒ 同通道两种量纲。
   这是 D′ 的既有行为，任何 3-modality 实验都会继承，**必须在结论里声明**。
4. 云上环境是否安装 `albumentations` **仍未知**（本机未安装 ⇒ 恒 no-op）。

---

## HARD STOP

不训练、不推理、不 smoke、不启动任何 GPU 作业、不干扰正在运行的 Seed-2。

---

## 12. 独立复核（2026-10-04 第二次执行）

本文档 §1–§11 由第一次执行产生。为确认结论不依赖单次运行，**在同一 checkout 上独立重跑了全部 gate**，
未改动任何 remap 代码语义（唯一改动：`pretrained_coverage.manifest()` 增加 `source_shape` / `exact_equal`
两个**只读字段**，`measure()` 的输出与判据不变）。

```text
checkout       = 878cbcdeb40c92590b593e5b993885ce41735223  (branch yyy)
环境           = torch 2.4.1+cpu / CUDA_VISIBLE_DEVICES 未设 / torch.cuda.is_available()=False / device_count=0
```

| 复核项 | 结果 |
|---|---|
| `diagnostic/p1_remap_audit.py` | **P1_STATUS = PASS · 3M_READY = YES · CUDA_USED = NO** |
| coverage D / D′ / M4 / candC | 全部 **532/532 = 1.0000**（tensor & parameter），gate PASS |
| manifest | 545 行；`PRETRAINED 532 / DERIVED 5 / ZERO_INIT 2 / NEW_RANDOM 6` |
| ZeroConv | `max|W| = 0.0e+00`，`max|b| = 0.0e+00`，status `['ZERO_OK','ZERO_OK']` |
| ADD | `type=ADD · learnable_params=0 · alpha=1.0` |
| identity-at-init | 三项 `max_abs = 0.000e+00` **逐位相等** ⇒ `IDENTITY_AT_INIT = PASS` |
| negative tests | A1 / A2 / B1 / B2 / C / **C（旧分派 → coverage 0.0019 被门拦下）** / C2 全 PASS |
| regression | `scanned=30  regressions=0` |
| contract selftest | **14/14** |
| contract `--all` | **CONTRACT 8/8 PASS** |

**Seed-2 未被触碰（前后逐字节对照）**：
`runs/urban_multimodal_det_modality_m4_rgb_ir_seed2/` 在本轮开始（`14:57:11Z`）与结束（`15:00:39Z`）
两次快照均为 **0 个文件**（本地只有空 `weights/` 目录；Seed-2 是云端作业），前后 diff 为空
⇒ `SEED2_TOUCHED = NO`。

**复核结论与第一次完全一致**：`P1_STATUS = PASS · 3M_READY = YES`。
