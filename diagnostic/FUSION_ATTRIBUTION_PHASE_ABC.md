# Fusion Attribution — Phase A / B / C 报告

**日期** 2026-10-03 · **性质** ZERO GPU（A/B 全部 CPU）+ Phase C 待执行
**问题** D′ = RGB+IR+Depth 相对 RGB-only 的 +0.0158，来自 IR、Depth 还是协同？
**前置约束** 在 M4/M5 配对实验完成前，**不得**因 IR-only/Depth-only 弱就宣布 Fusion NO-GO。

---

## 1. A1 — IoU-bin 分解（D′ vs M1，10 个 bin 逐个做配对 bootstrap）

`Δ = D′ − M1`（正 = D′ 更好）。数据：`diagnostic/sepstem_clahe/best_full/results`（D′ 冻结）vs
`diagnostic/batch1_eval/m1_best/results`（M1 best，本机 CPU 复现 0.49951 vs 云端 0.49944，差 7e-5）。
工具 `diagnostic/batch1_attribution.py`（复用已验证内核；**复刻断言 PASS**：快速内核 == `official_eval.evaluate`）。
B=400，seed=20260928（与既有 E1 工具同参数）。

| IoU | D′ | M1 | Δ(D′−M1) | 95% CI | P(Δ>0) | 显著 |
|---:|---:|---:|---:|---|---:|---|
| 0.50 | 0.77826 | 0.77647 | **+0.00178** | [−0.01787, +0.02197] | 0.585 | — |
| 0.55 | 0.76347 | 0.74002 | **+0.02345** | [−0.00247, +0.04925] | 0.953 | — |
| 0.60 | 0.72841 | 0.71270 | +0.01571 | [−0.00716, +0.03556] | 0.920 | — |
| 0.65 | 0.69575 | 0.68058 | +0.01517 | [−0.01026, +0.04156] | 0.843 | — |
| 0.70 | 0.64676 | 0.62742 | +0.01935 | [−0.00989, +0.04998] | 0.877 | — |
| **0.75** | 0.51096 | **0.52188** | **−0.01091** | [−0.04340, +0.05252] | 0.537 | — |
| 0.80 | 0.41152 | 0.39561 | +0.01590 | [−0.00698, +0.04194] | 0.900 | — |
| **0.85** | 0.31504 | 0.28830 | **+0.02674** | **[+0.00836, +0.04530]** | 0.998 | **✅** |
| 0.90 | 0.20738 | 0.19908 | +0.00830 | [−0.01288, +0.02764] | 0.760 | — |
| 0.95 | 0.09525 | 0.05305 | **+0.04220** | [−0.00562, +0.05089] | 0.830 | — |
| **ALL** | 0.51528 | 0.49951 | **+0.01577** | **[+0.00033, +0.03003]** | 0.975 | **✅** |

**诊断（brief A1 的四个问题）**

1. **gain 从哪个 IoU 开始出现**：**从 .50 之后立刻**（.55 就有 +0.0234，是本表最大单 bin 之一），
   **不是**从高 IoU 才开始。
2. **是否单调增加**：**完全不单调**。Δ 序列
   `[+0.0018, +0.0235, +0.0157, +0.0152, +0.0194, −0.0109, +0.0159, +0.0267, +0.0083, +0.0422]`。
3. **是否只有 .85/.90/.95 在贡献**：**否**。三项净 Δ = +0.02574，但 **.50–.75 段净 Δ = +0.01076**（不为零）；
   `.80–.95` 段 = +0.02328。**低 IoU 段贡献了净值的三分之一。**
4. **.50–.75 是否正负抵消**：**基本没有抵消**。负值只有 .75 一项（−0.0109）；正值合计 +0.16860、负值合计 −0.01091。

> ### ★ 对上一轮解读的正式更正
> 上一轮我基于三段汇总（`.50–.60 / .65–.75 / .80–.95`）说过"**增益集中在高 IoU 段**"。
> **逐 bin 一看不成立**：只有 **.85 单个 bin** 的 CI 不含 0，**.80 / .90 / .95 全部含 0**。
> "集中在高 IoU" 是**段内平均 + CI 宽度差异**造成的错觉，不是效应大小的集中（.55 的 Δ 比 .80 还大）。
> **正确表述：D′ 在几乎每个 IoU 档都领先（唯一反向是 .75），但只有 .85 单独越过了 CI。**

---

## 2. A2 — 类别归因（12 类，AP50-95，METRIC_A 分母）

| class | GT n | D′ | M1 | Δ | 95% CI | P(Δ>0) | 显著 |
|---|---:|---:|---:|---:|---|---:|---|
| light | 244 | 0.5421 | 0.4797 | **+0.0624** | [+0.0067, +0.1190] | 0.988 | **✅** |
| sign | 150 | 0.4014 | 0.3438 | **+0.0576** | [+0.0251, +0.0865] | 1.000 | **✅** |
| car | 288 | 0.5420 | 0.5036 | **+0.0383** | [+0.0042, +0.0705] | 0.993 | **✅** |
| animal | 731 | 0.5066 | 0.4839 | **+0.0226** | [+0.0047, +0.0431] | 0.995 | **✅** |
| seat | 107 | 0.6306 | 0.6078 | +0.0228 | [−0.0052, +0.0603] | 0.927 | — |
| boat | 28 | 0.5340 | 0.5152 | +0.0188 | [−0.0572, +0.1072] | 0.670 | — |
| bicycle | 106 | 0.3931 | 0.3782 | +0.0149 | [−0.0225, +0.0494] | 0.790 | — |
| person | 1045 | 0.4517 | 0.4444 | +0.0073 | [−0.0047, +0.0194] | 0.838 | — |
| tricycle | 2 | 0.7525 | 0.7515 | +0.0010 | [+0.0000, +0.0010] | 0.630 | — |
| garbage_can | 57 | 0.4907 | 0.5031 | −0.0124 | [−0.0657, +0.0337] | 0.328 | — |
| uav | 30 | 0.5344 | 0.5563 | −0.0219 | [−0.0738, +0.0229] | 0.203 | — |
| ball | 19 | 0.4044 | 0.4266 | −0.0222 | [−0.1038, +0.0773] | 0.328 | — |

**D′ 显著更好的类：light / sign / car / animal（4 个）；M1 显著更好的类：0 个。**
负向合计 −0.0564、正向合计 +0.2457、Σ/12 = +0.01577。
（最负两类占**负向总量** 78% —— 注意"除以净值"会被放大，不是集中度指标。）

**与 P3/P11 小目标失败类的重合检查 —— 结论是"不重合"：**

| 显著类 | small(<1024) | medium | large | small 占比 |
|---|---:|---:|---:|---:|
| light | 11 | 102 | 131 | **4.5%** |
| car | 18 | 167 | 103 | 6.3% |
| animal | 63 | 359 | 309 | 8.6% |
| sign | 30 | 84 | 36 | 20% |

（全体 val GT：small 371 / medium 1320 / large 1116。）

⇒ **这 4 个显著类全部是 medium/large 主导**，与"small Δ ≈ 0"一致，
**不是** small-object classification failure classes 的改善。**类别层面的显著性可以完全由尺度层面解释。**

**⚠ 不得解释成 modality causality。** 这里只说明"D′ 相对 M1 在哪些类上更好"，不说明原因。

---

## 3. A3 — 与已锁定历史数字的交叉检查

| 检查 | 期望 | 实测 | 结果 |
|---|---|---|---|
| overall Δ(D′−M1) | +0.01577 | **+0.01577** | ✅ |
| mAP50 Δ | +0.00178 | **+0.00178** | ✅ |
| mAP75 Δ | −0.01091 | **−0.01091** | ✅ |

**三项逐位吻合 ⇒ evaluator / results pairing 无误，可继续。**

---

## 4. A 阶段结论（候选机制，非因果）

> **D′ 相对 RGB-only 的 +0.0158 增益：**
> · **在 IoU 维度上不集中** —— 几乎每一档都为正，唯一反向是 .75；只有 **.85 单独显著**；
> · **在尺度维度上集中** —— medium/large 显著、small ≈ 0；
> · **类别上体现为 light/sign/car/animal 四类**，而这四类全部是 medium/large 主导 ⇒ 与尺度结论自洽。
>
> **这是一个"候选机制"，不是已证明的因果。** 要判断这 +0.0158 究竟来自 IR、Depth 还是协同，
> **必须做 M4/M5 配对实验**（brief 明令：不得因单模态弱就宣布 NO-GO）。

---

## 5. Phase B — M4 / M5 Contract Audit（全 CPU）

### 5.1 合同门（`scripts/validate_modality_contract.py --all`）→ **CONTRACT 8/8 PASS**

```
✅ M1   Gray2BGR  ch=3 [B,G,R]      split=visible_split       stem_in=3 remap=0   intersect=643/649
✅ M2a  Infrared  ch=3 [IR,IR,IR]   split=rgbt_split          stem_in=3 remap=0   intersect=643/649
✅ M2b  Infrared  ch=3 [IR,IR,IR]   split=rgbt_split          stem_in=3 remap=0   intersect=643/649
✅ M3   Depth     ch=3 [D,D,D]      split=depth_split         stem_in=3 remap=0   intersect=643/649
✅ M4   RGBIR     ch=4 [B,G,R,IR]   split=rgbt_split          stem_in=4 remap=5   intersect=642/649
✅ M5   RGBD      ch=4 [B,G,R,D]    split=depth_split         stem_in=4 remap=5   intersect=642/649
✅ M6   IRD       ch=2 [IR,D]       split=rgbid_split         stem_in=2 remap=5   intersect=642/649
✅ M7   RGBID     ch=5 [B,G,R,IR,D] split=rgbid_split         stem_in=3 remap=550 intersect=61/661
```

### 5.2 逐项审计表（brief 要求的 20 项）

| # | 项 | M4 (RGB+IR-CLAHE) | M5 (RGB+Depth) |
|---|---|---|---|
| 1 | YAML parse | ✅ `configs/train_modality_m4_rgb_ir.yaml` | ✅ `..._m5_rgb_depth.yaml` |
| 2 | model scale | ✅ `yolo11m_modality4ch.yaml` → **m**（含 scale 字符，非 nano 回退） | ✅ 同 |
| 3 | input channel count | ✅ config `channels: 4` == yaml `ch: 4` == 模式产出 4 | ✅ 同 |
| 4 | first stem pretrained transfer | ✅ `remap=5`；`model.0.conv.weight` **不再被丢弃** | ✅ 同 |
| 5 | RGB 三通道来自 pretrained RGB stem | ✅ `ch[0:3] = stock model.0.conv.weight`（CPU 断言 `RGB==stock:True`） | ✅ 同 |
| 6 | aux 通道初始化符合既有 convention | ✅ `ch[3] = mean(W_R,W_G,W_B)`（与 RGBD/sepstem aux 同约定） | ✅ 同 |
| 7 | Detect cv3 reinit | ✅ 6 张量未迁移（nc 80→12），与 D′/M1 同等 | ✅ 同 |
| 8 | **train loader provenance** | ✅ ch0..2 **逐位 == visible**（5/5）；ch3 **逐位 == CLAHE(IR)**（5/5） | ✅ ch0..2 逐位==visible；ch3 **逐位 == depth**（5/5） |
| 9 | **val loader provenance** | ✅ 同上（`load_and_preprocess_image`，val 同源） | ✅ 同 |
| 10 | **inference loader provenance** | ✅ `LoadImagesAndVideos` 从 `.../val/visible` 出 **shape=[360,640,4]**，ch0..2 corr=1.0 | ✅ 同 |
| 11 | **IR encoding** | ✅ `ir_encoding: clahe` 且**真的生效**：ch3 **逐位 != percentile(IR)**（5/5 False） | n/a |
| 12 | **Depth encoding** | n/a | ✅ ch3 与 D′ 的 depth 公式**逐位相同**（`<300→0`、`/19999×255`） |
| 13 | Albumentations / HSV contract | ✅ `aug.hsv_*=0` + `aug.albumentations_p=0.0`（合同门强制） | ✅ 同 |
| 14 | evaluator source | ✅ `predict_rect.py` → `official_eval.py`，cap=100 | ✅ 同 |
| 15 | scale contract | ✅ `scales.m = [0.50, 1.00, 512]` 与 D′ 相同 | ✅ 同 |
| 16 | **no silent fallback** | ✅ 三项独立反证：aux**不是** visible 复制（corr −0.46）、**不是** percentile IR、stem 被显式初始化（`remap=5` 且权重被写入） | ✅ aux 不是 visible（corr −0.45…−0.56）、不是 IR |
| 17 | config SHA | `1f70fa0f82e6bf7f` | `47410cda914c496e` |
| 18 | base.py / loaders.py SHA | `bf380a22d2491f73` / `56c07c8520cf5b25` | 同 |
| 19 | **D′ incumbent SHA 未变** | `configs/train_rgbid_sepstem_clahe.yaml` = **`a4e329cfc3d22020`**（= 轮前） · `yolo11m_sepstem.yaml` = **`9b14f29460733384`**（= 轮前） | ✅ |
| 20 | 数据文件未修改 | ✅ 未触碰 `data/`；本机仅在 `diagnostic/batch1_eval/` 写预测 | ✅ |

### 5.3 溯源矩阵（`diagnostic/gpu_smoke_gate.py` 探针，CPU 复跑）

```
M1   ch=3 pairs_ir=['visible','infrared'] | P3.1 PASS | P5 PASS
M3   ch=3 pairs_ir=['visible','depth']    | P3.1 PASS | P5 PASS
M2a  ch=3 pairs_ir=['visible','infrared'] | P3.1 PASS | P5 PASS
M4   ch=4 pairs_ir=['visible','infrared'] | P3.1 PASS | P5 PASS | ch3==Infrared 5/5
M5   ch=4 pairs_ir=['visible','depth']    | P3.1 PASS | P5 PASS | ch3==Depth     5/5
```

**Phase B：全 PASS。**

### 5.4 本轮对探针的两处修正（记录，避免误判）

1. **通用探针在 4ch 上原先把 ch0 当模态通道** —— 4ch 的 ch0 是 RGB 的 B，导致假 FAIL。
   已改为**按 `channels>3` 分流**：融合模式查 ch0..2 == visible，aux 由专用探针 `probe_aux_channel` 覆盖。
2. **`Depth` 的深度路径由 `pairs_rgb_ir` 承载（R6 脆弱耦合）** —— 探针必须像 `predict_rect.py` 一样
   **从 train config 取 pairs**；用默认 `["visible","infrared"]` 会去找不存在的 `infrared/` 目录而读到 0 张图。

---

## 6. Phase C — GPU Smoke

**2026-10-03 首次执行（A100-40GB，`--cases M4,M5 --epochs 2 --fraction 0.02 --imgsz 1280`）**

| | 结果 | 说明 |
|---|---|---|
| **M5** | **PASS**（29/29） | 全部检查通过 |
| **M4** | **FAIL 1 项（P4.1）→ 已定位为本探针 bug，非 pipeline 缺陷** | 见下 |

**M4 的那 1 项 FAIL 是探针 bug，不是缺陷。** `P4.1 IR 通道逐位 == CLAHE(raw)` 是
**M2a（3ch IR 单模态）专用**的检查，它比较 `im[...,0]`；而 **4ch 融合模型的 ch0 是 RGB 的 B**。
实测证据：`clahe_corr = −0.4145`（正是 B 通道 × CLAHE(IR) 的相关，不是 IR × CLAHE）。
把 `expect_ir_clahe` 也挂在 M4 上导致了一个对它无意义的检查被触发。

**M4 真正该看的检查全部 PASS**：

```
P4b.1 ch3 逐位 == 期望的 Infrared 预处理结果   | bitwise=5/5  corr=1.0     ✅
P4b.3 M4 的 aux 不是 percentile IR            | [False]×5                ✅
P4b.2 aux 不是 RGB/visible 的复制             | corr(aux,ch0) = −0.414   ✅
P3.1  融合模式 ch0..2 逐位 == visible          | bitwise=5/5              ✅
P5.2b 推理 ch0..2 == visible                  | [1.0, 1.0, 1.0]          ✅
P6.2  真实 forward C=4；P6.4 非单通道复制      |                          ✅
multi-ch stem init: ch[0:3]=pretrained RGB, ch[3:4]=mean(R,G,B)        ✅
Transferred 642/649（与 M1/D′ 同源、同等 cv3 reinit）                  ✅
```

**修复**：`P4.1/P4.2` 加 guard `and c["channels"] == 3`（其 4ch 等价物是 `P4b.1/P4b.3`）。
CPU 复验：M2a 仍 RUN（`clahe_bitwise_equal=True` / `percentile_bitwise_equal=False`），
M4/M5 SKIP 且 `P4b.1=True`。

**修复前 SHA `d74fb00621fd3004` → 修复后 `1bca25e659ff85f2`。**
按 brief 的硬门流程（FAIL → 定位 → 修复 → **重新跑对应 gate**），**需重跑一次 M4/M5 smoke**。

**这是同一类 harness bug 的第 4 次**（探针假设了错误的通道布局）：
① 4ch 的 ch0 被当成模态通道；② `Depth` 的 pairs 必须从 config 取；
③ 推理探针硬链接模态文件而非用生产 source 目录；④ 本条（P4.1 未按通道数 gate）。
**教训：探针的通道假设必须由 `channels` / `mode` 显式分派，不能靠默认。**

**下一条（也是唯一允许执行的）GPU 命令：**

```bash
cd ~/autodl-tmp/urban_multimodal_detection
python -X utf8 diagnostic/gpu_smoke_gate.py --device 0 --cases M4,M5 --epochs 2 --fraction 0.02 --imgsz 1280
```

⚠ **同步要求**：`diagnostic/gpu_smoke_gate.py` 本轮已扩展（新增 M4/M5 case + `probe_aux_channel` +
融合模式分流），当前 SHA 前 16 位 **`d74fb00621fd3004`**。云上若还是旧版，会退回只跑 M1/M3/M2a。
另外 `scripts/train.py`（`67cb9fe02757fd59`）与 `ultralytics/data/{base,loaders}.py`
（`bf380a22d2491f73` / `56c07c8520cf5b25`）也需一致 —— 后两者被 `.gitignore` 忽略，`git pull` 带不上。

**C 阶段必须证明（新版会自动打）**：
- C1 runtime：CUDA / forward / loss finite / backward / optimizer step / ckpt save+load
- **C2 provenance（关键）**：`P6.2 C=4`；`P6.6` 合成记录排除；`P4b.1` **ch3 逐位 == 期望的 IR-CLAHE / depth**；
  `P4b.2` aux 不是 RGB/visible 的复制；**`P4b.3` M4 的 aux 不是 percentile IR**；`P6.5` 数据批次无整量
- C3 validation / inference：val 跑通、prediction shape 正确、official evaluator 可读
- C4 checkpoint：重新 load 后 channels / modality routing / scale 正确

---

## 7. Formal-training Gate

| Gate | 状态 |
|---|---|
| A1 IoU-bin | ✅ PASS |
| A2 class | ✅ PASS |
| A3 交叉检查 | ✅ PASS（三项逐位吻合） |
| B 合同门 8/8 | ✅ PASS |
| B 溯源矩阵 5/5 | ✅ PASS |
| B D′ incumbent SHA 未变 | ✅ PASS |
| **M4 smoke** | ❌ **NOT RUN** |
| **M5 smoke** | ❌ **NOT RUN** |

```text
FORMAL-TRAINING GATE: NOT SATISFIED
⇒ 禁止启动 M4/M5 300 epoch 正式训练。
```

---

## 附：本轮所有 SHA（前 16 位）

| 文件 | SHA-256 (16) |
|---|---|
| `ultralytics/data/base.py` | `bf380a22d2491f73` |
| `ultralytics/data/loaders.py` | `56c07c8520cf5b25` |
| `ultralytics/models/yolo/detect/train.py` | `6538ab6f3a0ca4a3` |
| `ultralytics/cfg/default.yaml` | `cf06d3a61a5cc9a9` |
| `scripts/train.py` | `67cb9fe02757fd59` |
| `scripts/validate_modality_contract.py` | `283c28f006bbe834` |
| `scripts/official_eval.py` | `e82128abc082ed05` |
| `diagnostic/gpu_smoke_gate.py` | **`d74fb00621fd3004`**（本轮已改） |
| `diagnostic/batch1_attribution.py` | `2c06307b0a787d8d`（本轮新增） |
| `configs/yolo11m_modality4ch.yaml` | `871d9d0da0122453` |
| `configs/train_modality_m4_rgb_ir.yaml` | `1f70fa0f82e6bf7f` |
| `configs/train_modality_m5_rgb_depth.yaml` | `47410cda914c496e` |
| **D′ `configs/train_rgbid_sepstem_clahe.yaml`** | **`a4e329cfc3d22020`（未变）** |
| **D′ `configs/yolo11m_sepstem.yaml`** | **`9b14f29460733384`（未变）** |


---

## ⚠ 符号更正（2026-10-04）

本报告 §1 的 **size / IoU-段** 数值取自 `_e1_vs_dprime_boot.py`，其内部约定是 **Δ = E1 − Dp = M1 − D′**，
却被我放进了标着 `Δ = D′ − M1` 的表里 ⇒ **符号写反**。
用 `diagnostic/gt_level_attribution.py` 以统一 Δ=D′−M1 重算：

**large = +0.02142** · **.80–.95 = +0.02328** · **.50–.60 = +0.01365** · **.65–.75 = +0.00787** · **small = −0.00180**

**A1/A2 的结论方向不变**（D′ 在 large 与高 IoU 更好、small ≈ 0）。详见 `diagnostic/GT_LEVEL_ATTRIBUTION.md`。
