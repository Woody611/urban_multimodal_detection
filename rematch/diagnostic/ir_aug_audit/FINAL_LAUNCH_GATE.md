# D′ IR-Aug — Final Launch Gate（2026-10-04/05，CPU-only）

**本文件不是新的实验设计，也不是效果讨论。** 只回答一件事：
> 在消耗 GPU 之前，D′ + IR-specific augmentation 这个实验，是否可以安全、正确地启动？

```
GPU_LAUNCH_NOW = DO NOT LAUNCH     ← M4 Seed-2 占用 GPU；本轮未启动、未占用 CUDA、未触碰 M4
```

工具：`gate_resolved_config.py` / `launch_gate.py` / `regression_ab.py` / `second_fire_gate.py` /
`verify_run_args.py` / `audit_manifest.json`（审计时刻 18 个文件的 SHA 基线）。

---

## 逐门结果

| # | Gate | 结果 | 证据 |
|---|---|---|---|
| 1 | Experiment identity | **PASS** | name = `urban_multimodal_det_yolo11_rgbid_sepstem_clahe_iraug`；非 M4/3M/Seed-2 |
| 2 | Provenance | **PASS** | 见下方 SHA 表；`base.py`/`loaders.py` 用 SHA+备份确权（gitignore 不可见） |
| 3 | D′ baseline integrity | **PASS** | best.pt `1cae45f7…e4fda` 未变；baseline config/model SHA 未变 |
| 4 | M4 Seed-2 isolation | **PASS（本机可验证部分）** | M4 run 目录 **0 个文件**，mtime 22:28 < 本次会话；M4 config SHA 记录在案 |
| 5 | Resolved-config single-variable | **PASS** | 131 个 resolved key 全量比对，仅 5 个不同（4 个 IR aug 键 + `name`），0 unexpected |
| 6 | Exact IR insertion path | **PASS** | AST 调用序 + 运行时数组同一性双重证明 |
| 7 | Channel identity | **PASS** | 逐通道对回 B/G/R/IR/Depth；A/B 下 RGB/Depth 逐位不变、IR 变 |
| 8 | No double augmentation | **PASS** | 可执行调用点 = 1；loaders.py = 0；Albumentations p=0 且 transform=None；1.0 次/样本 |
| 9 | Train-only isolation | **PASS** | VAL `max_abs_diff=0`；PREDICT 走 LoadImagesAndVideos 逐位相同；TRAIN 确实触发 |
| 10 | Identity / determinism | **PASS** | Test A/B `max_abs_diff=0`；C 同种子一致；D 异种子变化且合法 |
| 11 | Numerical safety | **PASS** | 90 组网格 NaN=0 Inf=0 dtype/shape 全 OK；σ 实测 3.8465 ≈ 3.825 ⇒ 单位确为 [0,1] 域 |
| 12 | Probability runtime | **PASS** | 见下（单种子 + 5 种子两种口径） |
| 13 | Preflight script | **PASS** | `exit code = 0`，`RESULT: ALL CHECKS PASSED` |
| 14 | Fresh-run directory | **PASS** | `runs/*iraug*` 不存在；`resume=false`、`exist_ok=false` 已确认 |
| 15 | GPU resource | **DEFER** | M4 Seed-2 占用；本轮不得启动 |

---

## §2 Provenance

```
HEAD                      878cbcdeb40c92590b593e5b993885ce41735223  (branch yyy)
D′ best.pt                1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda
D′ train config           a4e329cfc3d220206448cdd3377c1b3f5126c50f13c907ada522d98725486e12
IR-Aug train config       e40f01b7cb3b0569a3e3f78e97f06944d8b2ed9b5fde869087f76b9daad7227e
model config (sepstem)    9b14f2946073338414b4441784b6df87a868b982be32ac8d50b6d1c4a17f3dd8
base.py                   d7d2315ed4391c9724fafaf080bd7c7e8bc401de162084e0cf805e71304cdb88
loaders.py                56c07c8520cf5b255516afb30621fecbf3ebc5dc4c658a8dd9689ade537c3362
train.py                  d368f6c65cdeb206102d9f702604c680ba56d62caa074423a7c5e16fddb93212
default.yaml              f119165d27d6bdc1e40d5bf18a1891893313e422f249cafb62bf15cfd2dc04d8
predict_rect.py           9d704300143a5786f7b74450f35e6c656d55c82812ca34d0e7b830d96034f524
augment.py                abfc90f53f75f7ad5a5c13e842847e6e6cbc8bcdab8b1be57e88f7cff801282b
tasks.py                  79c8dbab9cd810d2609e7ff162d856beea8ab8e19b7ee6a933fb83ec1a617efd
```

⚠ **gitignore 陷阱（复述）**：`.gitignore:4` 的无锚定 `data/` 吞掉整个 `ultralytics/data/`：
`base.py` / `loaders.py` **不进 git**，`git status` 看不见它们的改动。因此本次全部用
SHA-256 + `_before/` 备份确权，**没有**依赖 git diff。

⚠ **工作树中的既有改动（非本轮产生）**：`diagnostic/fusion_phase0_contract_selftest.py`、
`diagnostic/p1_remap_audit/*`、`diagnostic/p1_run.log`、`scripts/pretrained_coverage.py`
在本次任务**开始前**就已是 modified 状态。本轮**未触碰** P1 remap 实现。
本轮自身改动（tracked）只有：`scripts/train.py`、`ultralytics/cfg/default.yaml`；
新增（untracked）：`configs/train_rgbid_sepstem_clahe_iraug.yaml`、`diagnostic/ir_aug_audit/`。

---

## §5 Resolved-config（不是 YAML 文本比对）

解析路径与正式训练完全相同：`scripts.train._build_train_kwargs` → `get_cfg(DEFAULT_CFG_DICT, overrides=…)`。

```
resolved keys compared: 131        （含 default.yaml 默认值 ⇒ "没写"的字段也被真正比较）
resolved data path : both = D:\gyt\AIC\urban_multimodal_detection\data\processed\rgbid_split_train\dataset.yaml
差异（全部 ALLOWED）:
   ir_gamma              [1.0,1.0] -> [0.75,1.35]
   ir_gamma_probability  0.0       -> 0.35
   ir_noise_probability  0.0       -> 0.2
   ir_noise_std          0.0       -> 0.015
   name                  …_sepstem_clahe -> …_sepstem_clahe_iraug
only-in-baseline: {}   only-in-iraug: {}   unexpected: 0
SINGLE_VARIABLE_GATE = PASS
```
`epochs/patience/batch/imgsz/seed/optimizer/lr0/lrf/momentum/weight_decay/warmup_*/box/cls/dfl/
mosaic/close_mosaic/mixup/copy_paste/hsv*/scale/translate/degrees/shear/perspective/flip*/rect/amp/
pretrained/resume/channels/use_simotm/ir_encoding` **逐项相同**（同一 131 键比对的结果，非人工核对）。

---

## §6 Exact IR insertion path（双重证明）

**(a) AST**：RGBID 分支 body 内的调用顺序 =
`apply_ir_encoding → apply_ir_augmentation → _resize_images_3 → _merge_channels_rgbid`
（`_resize_images_3` 在本数据集为 no-op，三模态同尺寸）

**(b) 运行时数组同一性**：
```
aug 的输入 == CLAHE(raw IR)          : True
aug 的输入 == raw IR                 : False      ← 排除 gamma→CLAHE
aug 的输出 == _merge_channels_rgbid 收到的 IR : True   ← 排除 gamma/noise→RGB/Depth
```

---

## §7 Channel identity

```
merged 5ch:  ch0==visible B : True   ch1==G : True   ch2==R : True
             ch3==CLAHE(IR) : True   ch4==Depth : True
A/B（aug 必然开启 p=1.0 vs D′）: RGB unchanged=True  Depth unchanged=True  IR differs=True
final 5ch TENSOR order (Format): ['R','G','B','IR','D']   IR idx=3  D idx=4
```
任务书写的 `[B,G,R,IR,D]` 与 `configs/yolo11m_sepstem.yaml` 的注释一致，但**实测张量是 `[R,G,B,IR,D]`**
（[augment.py:3259-3262](ultralytics/data/augment.py#L3259-L3262) 的 `transpose(2,0,1)[:3][::-1]` = BGR→RGB）。
这是 **D′ 既有行为，本轮未触碰 `Format`**；与 `SilenceChannel[0,3]→RGB stem（RGB 预训练）` 自洽。
IR 恒在 idx 3、Depth 恒在 idx 4 ⇒ 对 IR augmentation 无影响。

---

## §8 No double augmentation

| 检查 | 结果 |
|---|---|
| `apply_ir_augmentation` 可执行调用点（全 repo） | **1**（`base.py` RGBID 分支） |
| `loaders.py`（推理/提交）可执行点 | **0** |
| 其它模态分支（Infrared / RGBIR / RGBD / IRD） | 均**不调用** |
| Albumentations（5ch compose） | `p=0` **且** `transform=None` ⇒ `__call__` 首行短路 return |
| RandomHSV | `img.shape[-1] != 3` 立即 return（[augment.py:3497](ultralytics/data/augment.py#L3497)） |
| 实测调用率 | **1.000 次 / 训练样本** |

⇒ 一张源图**至多一次** IR gamma/noise。

⚠ **环境相关观察（非 blocker，但需在云上确认）**：`Albumentations.__call__` 的首行是
`if self.transform is None or random.random() > self.p: return labels`。
本机**未安装** albumentations ⇒ `transform is None` ⇒ 短路，**不消耗 RNG**。
若云端装了 albumentations，`transform` 非 None，则该行会求值 `random.random() > 0` 并
**消耗一次 RNG**（仍然 return，功能不变）。这是 **D′ 既有行为**，对 D′ 与 D′-IRAug **完全同构**
（单变量对照不受影响）；但意味着"同种子⇒同样本组成"只在**同一环境内**成立。
云上确认命令：`python -c "import albumentations, sys; print(albumentations.__version__)"`

---

## §9 Train-only isolation

```
VAL      on == off : True   (max_abs_diff = 0)          # val dataset augment=False
PREDICT  on == off : True   (LoadImagesAndVideos 逐位相同，shape=(360,640,5) uint8)
TRAIN    augmentation fires : True  (20/20 calls)
```
三条路径都跑了**实际代码**，不是阅读代码。`official_eval.py` 是纯 TXT 打分器，不加载模型/数据集。

---

## §10/§11 Identity, determinism, numerical safety

```
Test A  p_gamma=0,p_noise=0 → 与 D′ 逐位相同 : True   max_abs_diff = 0
Test B  gamma=1,sigma=0     → 恒等            : True   max_abs_diff = 0
Test C  同 seed → 同输出                       : True
Test D  异 seed → 变化, 且 shape/dtype/finite 合法 : True
90 组 (15 图 × γ∈{0.75,1,1.35} × σ∈{0,0.015}) : NaN=0  Inf=0  bad_dtype=0  shape_mismatch=0
noise 单位: 实测 σ = 3.8465 灰度级  (期望 0.015×255 = 3.825 ⇒ 归一化 [0,1] 域)
```

---

## §12 Probability runtime

| 口径 | N | gamma fired | rate | noise fired | rate | both |
|---|---|---|---|---|---|---|
| 单种子（gate12） | 400 样本 | 155 | 0.3875 (z=+1.57) | 80 | 0.2000 (z=0.00) | 33（独立⇒28.0） |
| **5 种子合并** | 1500 样本 | 517 | **0.3447 (z=−0.43)** | 317 | **0.2113 (z=+1.10)** | — |

容差：二项 3.3σ。两者都在容差内 ⇒ 概率与 config 一致。
gamma 抽样 min/max = 0.757 / 1.349 ⊂ [0.75, 1.35] ✅
预处理调用 400/400 = 1.000 次/样本 ⇒ **无 Mosaic 放大**。

---

## §14 Fresh-run directory

```
runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_iraug/   → 不存在（无 stale weights/args.yaml/results.csv/ckpt）
resume = False   exist_ok = False   project = runs
```
同名兄弟目录（`…_sepstem_clahe`, `_box2`, `_e1`, `_f1`, `_oasa`, `_rectlate10`, …）**均不含** `_iraug`。

⚠ **路径名不一致（任务书笔误，非阻塞）**：任务书 §14/§18 写的是
`runs/urban_multimodal_det_yolo11m_rgbid_sepstem_clahe_iraug/`（`yolo11` 后多了个 `m`），
**实际 resolved 名称无这个 `m`**（与 incumbent `urban_multimodal_det_yolo11_rgbid_sepstem_clahe` 保持同一命名族）。
直接照抄 §18 的命令会 file-not-found；请用：
```
runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_iraug/args.yaml
```
`verify_run_args.py` 的默认路径已按实际名称写死。

---

## §13 Preflight

```
$ python scripts/preflight_check_train_config.py \
    --model_config configs/yolo11m_sepstem.yaml \
    --train_config configs/train_rgbid_sepstem_clahe_iraug.yaml --expect-ir-encoding clahe
RESULT: ALL CHECKS PASSED        exit code = 0
```
（含通道/scale/`ir_encoding` 显式声明/`check_dict_alignment`/fork 完整性断言。）

---

## 未执行的门（**故意**）

- **§17 Second-Fire Gate**：必须在 **M4 Seed-2 彻底结束之后**运行。工具已备好：
  `python diagnostic/ir_aug_audit/second_fire_gate.py`
  （含 18 文件漂移检测 vs `audit_manifest.json`、resolved-config、§6–§12、回归 A/B、preflight、
  fresh-run；全 PASS 才打印 `FINAL_PREFLIGHT = PASS / GPU_LAUNCH_AFTER_M4 = READY`）
- **§18 30 秒硬门**：训练启动后才跑 `verify_run_args.py`
- **§19 首批 smoke**：训练启动后才看第一批 `[B,5,H,W]`

本轮**未执行任何训练**，也**未通过任何** subprocess/nohup/tmux/os.system 启动训练。

---

## 漂移检测（本轮结束时）

```
audited files: 18 | drifted: 0
DRIFT_CHECK = PASS (no drift)
```

---

## 输出

```
==================================================
D′ IR-AUG FINAL LAUNCH GATE
==================================================

[1] Experiment identity              PASS
[2] Provenance                       PASS
[3] D′ baseline integrity            PASS
[4] M4 Seed-2 isolation              PASS   (本机可验证部分)
[5] Resolved-config single-variable  PASS
[6] Exact IR insertion path          PASS
[7] Channel identity                 PASS
[8] No double augmentation           PASS
[9] Train-only isolation             PASS
[10] Identity/determinism            PASS
[11] Numerical safety                PASS
[12] Probability runtime             PASS
[13] Preflight script                PASS
[14] Fresh-run directory             PASS
[15] GPU resource                    DEFER

--------------------------------------------------

IR_AUG_READY            = PASS
TECHNICAL_BLOCKERS      = NONE
SCIENTIFIC_STATUS       = UNTESTED
GPU_LAUNCH_NOW          = DO NOT LAUNCH
FINAL_PREFLIGHT         = PASS
GPU_LAUNCH_AFTER_M4     = READY_PENDING_SECOND_FIRE
--------------------------------------------------
```

`READY_PENDING_SECOND_FIRE` 的含义：**第一轮 Launch Gate 全 PASS**，但 §17 规定
「只有 M4 结束、Second-Fire 全部 PASS 才置 READY」——M4 尚未确认结束，故此刻**不得**宣告 READY。

---

## 需要人工决策 / 云上确认的事项（不是本机可解）

1. **M4 Seed-2 是否真的还在跑**：本机看不到远端进程。本轮只能证明**本机没有对它做过任何写入**
   （run 目录 0 文件、mtime 22:28 < 会话开始、config SHA 未变）。请在云上确认进程已退出、显存已释放。
2. **云上是否有 albumentations**（§8 的环境观察）——不影响单变量有效性，但影响"同种子⇒同样本组成"的跨机可比性。
3. **§14/§18 的目录名笔误**——按实际名称（无 `m`）执行。

## 科学解释锁（§20）

```
IR_AUG_EFFECT = UNTESTED
```
本文件**不主张** IR augmentation 会提分、会解决小目标、或会改善 raw-head candidate generation。
唯一合法假设：D′ 已证明 IR/CLAHE 有价值；IR-specific gamma/noise **可能**改善 IR 表示鲁棒性，
因此值得做一次**单变量 falsification**。
