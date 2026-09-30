# L17 Spatial Selectivity / Sufficiency Audit

**日期**：2026-09-29
**性质**：**STRICT READ-ONLY**。0 training / 0 backward / 0 optimizer.step / 0 新 checkpoint /
0 修改既有 source·config·dataset·label·evaluator·diagnostic·checkpoint / 0 submission /
**0 重新 forward** / 0 重抽 G1-G4 / 0 重抽 control / 0 新 intervention。
**MODEL_FORWARD = 0**（唯一新增计算是对**已落盘向量**的只读代数 + 读取 checkpoint 权重）。

---

## §21 最终结果（协议 §21 要求的顶部块）

```text
FINAL VERDICT:
C

CAUSAL_INVOLVEMENT:
SUPPORTED

SPATIAL_SELECTIVITY:
UNRESOLVED
   （可测量的那一半 —— 因果通路的注入位置特异性 —— SUPPORTED；
     协议 §5–§8 要求的那一半 —— 表征是否更集中于目标框 —— UNAVAILABLE）

SEMANTIC_SELECTIVITY:
UNRESOLVED
   （协议 §9 需要的『干预后非目标类投影』从未落盘 ⇒ UNAVAILABLE）

SUFFICIENCY:
NOT_DIRECTLY_MEASURED
   （协议 §12 只有 Level 2 可得；Level 1/3 均 UNAVAILABLE。
     可得代理的方向 = NOT_SUPPORTED，见 §7）

E1_CONTRADICTION:
PARTIALLY_EXPLAINED

NEXT ACTION:
CLOSE_L17
```

**C 的适用范围必须同读**（见 §10、§11）：本轮选 **C** 而非 **D**，是因为**能够测量的**那些量
（sufficiency 代理、donor-identity/content 选择性、位移特异性）**全部同向**，且都不是"测不了"而是"测了是反面"。
但若你的决策规则是"只有 §5–§9 那种直接测量才算证据"，那本轮应读作 **D**。

---

## §0 一句话结论

> **L17 donor replacement 改变的是一个「幅值主导、donor 内容几乎无关、被 ±1 cell 邻域绑住、
> 且强烈依赖 target state」的局部响应，而不是一个「更像目标 / 更集中于目标 / 更具类别选择性」的
> 可训练表征。**
> 最直接的 artifact 证据：把 donor 向量换成**任意**同类别成功小目标的向量，仍拿到 **72.8%** 的效应；
> donor 在 L17 上把「与成功质心的 cos」只改变了 **+0.030**（0.308 → 0.301，等于没变）。
> 因此 **NEXT ACTION = CLOSE_L17**。

---

## §1 协议 §2「绝对禁止事项」逐条合规

| 禁止项 | 状态 | 证据 |
|---|---|---|
| training / backward / optimizer.step | **未做** | `_analyze.py` 无 `loss`/`backward`/`optimizer`/`train()` 调用 |
| 新 checkpoint | **未做** | 无写 `.pt` |
| 修改已有 source / config / dataset / label / evaluator / diagnostic | **未做** | 见 §14：476 个既有 artifact SHA256 **0 变化** |
| 修改已有 checkpoint | **未做** | `best.pt` = `1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda`（前后一致） |
| 重新生成 submission | **未做** | 无写 submissions/ |
| **重新运行模型 forward** | **未做** | 全文唯一"读取模型"的动作是 `torch.load` 取 `cv3[0][-1]` 的权重做点积 |
| 重新抽 G1/G2/G4 | **未做** | 直接读 `_v3_analysis.json.E1` / `controls` / `_v11_ids.json.G4` |
| 随机重抽 control | **未做** | 复用 V13 的 disp+4/+5 与 V12 的 `L17_displaced` |
| 自己改 donor 定义 | **未做** | 见 §3：donor 规则逐字取自 `_v12_patch.py`，并与 V12 落盘字段 **38/38 对账** |
| 用新随机样本替代既有 V13/V14 样本 | **未做** | 232 条逐 key 连接 |
| 新 gain / attention / donor / kernel / layer / training config | **未做** | 协议 §18 |

**声明一处仓库外副作用**：首次 `torch.load` 时 ultralytics 自初始化，在
`C:\Users\45548\AppData\Roaming\Ultralytics\settings.json` 写了一个 settings 文件（仓库外、用户目录）。
发现后即把 `YOLO_CONFIG_DIR` 指向系统临时目录，后续运行只写 `%TEMP%\_l17_sel_ultra_cfg\`。
**未触碰任何仓库文件**，如实记录。

---

## §2 第一核心发现：协议 §5–§8 与 §9 **在本 artifact 集合上不可测**

这是本轮的**先决事实**，不是回避。`_analyze.py` 对 `p3_feature_space/` 与 `small_object_cause_v2/`
全部 `.json`/`.npz` 做了程序化扫描：

```text
扫描文件数 = 17
含 ndim>=3 数组的 .npz 条目 = 0     ← 这是唯一能承载 (1,512,40,40) 激活图的容器
```

各 artifact 实际落盘的空间内容（人工核对源码后确认）：

| artifact | 落盘内容 | 能否支撑 §5–§9 |
|---|---|---|
| `_v11_layers.npz` | 每 (层,组) 一个 **(n, C)** 矩阵 = **GT 中心单 cell 向量** | ✗ 无邻域、无整图 |
| `_v7_vectors.json` | 每 GT：`p3`(256)、`h`(256) —— 仍是 **单 cell 向量** + 标量 logit | ✗ |
| `_v10_stages.json` | 每 GT：P3/S1/S2 各 256 维 —— 单 cell | ✗ |
| `_v9_attrib.json` | 每 GT：`zA_p3`/`zA_h` **标量** | ✗ |
| `_v8_probes.json` | 探针权重(256) + 精度**标量**（且非 class 探针） | ✗ |
| `_v12_patch.json` | donor key、`y17/x17`、`L17_norm`/`L23_norm` **标量** + 各干预 Δlogit | ✗ |
| `_v13_dose.json` | base/R0–R3/disp 的 **logit 标量** + L23 norm/cos 标量 | ✗ |
| `_v14_upstream.json` | L12–L15 的 Δlogit 标量 + L23norm 标量 | ✗ |

⇒ 因此：

| 协议项 | 需要的量 | 判定 |
|---|---|---|
| §5 GT-box activation mass（mass_inside / 1.5x / 2x / ring） | 干预前/后 L17 全图 | **UNAVAILABLE** |
| §6 inside vs outside（mean/median/p90、ratio） | 同上 | **UNAVAILABLE** |
| §7 peak location（`Δdistance_to_GT_center`） | 同上 | **UNAVAILABLE** |
| §7 Top-1%/5%/10% coverage | 同上 | **UNAVAILABLE** |
| §8 entropy / effective area | 同上 | **UNAVAILABLE** |
| §9 class selectivity（target vs max non-target，**干预前后**） | 干预后的 h 或 12 类投影 | **UNAVAILABLE**（只落了目标类**一个标量**） |

**按协议 §2「如果现有 artifact 无法回答某项指标：报告 UNAVAILABLE，不要为了补数据自行 forward」——
本报告不 forward、不重建、不插值。** 因此 §15 的 P1/P2/P3/P4 形态判别**无法执行**，这直接决定了 §9 的
pattern 归类为 **P5**。

---

## §3 §4 重建已有 causal effect（只读）

### §3.1 §3 donor / 坐标 / 量级 的可得性

| 字段 | 来源 | 状态 |
|---|---|---|
| G1 identities (38) / G4 identities (194) / G2 identities (38) | `_v3_analysis.json` / `_v11_ids.json` | AVAILABLE |
| **donor identities** | **V13 未落盘**；但 **V12 `donor` 字段**（同规则、同池、同 232 目标）有 | AVAILABLE（跨 artifact） |
| donor layer | 17（硬编码） | AVAILABLE |
| intervention spatial coords | V12 的 `y17,x17`/`y23,x23` | AVAILABLE |
| displaced-control coords | V13: `(y0, x0±{4,5})`；V12: `(y17+3, x17)` —— **两轮定义不同** | AVAILABLE（两个独立控制） |
| intervention magnitude（向量本身） | 未直接落盘，但 donor 向量可由 `L17_G4[i]` **精确复原** | AVAILABLE（复原） |
| original / intervened logits | `base_logit`、各干预 `_logit` | AVAILABLE |

**donor 身份是"复原"而不是"重抽"**：规则（`同类别 ∧ G4 池 → min|Δsq| → tie-break key`）
在 V12/V13 跑**之前**就已固定，且 **V12 把结果落盘了**。用 V12 落盘字段逐条对账：

```text
G1 donor 对账：                        match 38/38   ✅
V12 `n_donor_cand` == V13 `n_donor`：  38/38         ✅
```

⇒ **V13 的 donor 身份可无歧义复原**，没有引入任何新抽样。

### §3.2 复现门（必须先过）

```text
V13 R0_raw median Δlogit = +5.8687      （V13 报告 = +5.869）   Δ = −0.0003   n = 38   PASS
```

### §3.3 核心表（**不以 logit>0 作 success criterion**）

| setting | n | median Δlogit | IQR | mean | %>0 | paired bootstrap 95% CI | Wilcoxon p |
|---|---:|---:|---|---:|---:|---|---:|
| **R0_raw** (1 cell) | 38 | **+5.8687** | [+2.015, +9.540] | 6.0897 | 89.5% | [+4.061, +8.497] | **3.14e-08** |
| R1_raw (3×3) | 38 | +7.6327 | [+3.234, +12.518] | 7.5859 | 84.2% | [+4.820, +10.612] | 1.27e-08 |
| R2_raw (5×5) | 38 | +7.7296 | [+3.220, +12.449] | 7.5385 | 84.2% | [+5.226, +11.130] | 1.48e-08 |
| R3_raw (7×7) | 38 | +7.6682 | [+3.329, +12.453] | 7.5287 | 84.2% | [+5.194, +11.184] | 1.48e-08 |
| R0_norm（只换方向） | 38 | +1.7628 | [+0.183, +3.311] | 2.0189 | 76.3% | [+0.495, +2.781] | 1.41e-04 |
| R0_rnd（随机同类别 donor） | 38 | +4.2723 | [+1.345, +9.939] | 5.3403 | 84.2% | [+1.804, +8.606] | 8.09e-07 |
| **disp+4 cells** | 38 | **−0.0038** | [−0.008, −0.000] | −0.0056 | 23.7% | [−0.006, −0.002] | 5.49e-06 |
| **disp+5 cells** | 38 | **−0.0045** | [−0.009, −0.001] | −0.0069 | 13.2% | [−0.007, −0.003] | 5.18e-07 |
| **G4 R0_raw** | 194 | **−0.0976** | [−1.356, +0.141] | −0.7434 | 34.0% | [−0.244, −0.018] | — |
| **G4 R2_raw** | 194 | **−0.4887** | [−3.276, +0.070] | −1.8031 | 28.4% | [−1.072, −0.297] | — |

✅ **判据复现**：`G1 Δ ≫ displaced control Δ`（+5.869 vs −0.004，差 ~5.9）；
`G1 Δ ≫ G4 Δ`（+5.869 vs −0.098；R2: +7.730 vs −0.489）。

---

## §4 §10 空间特异性 —— **可测量的那一半**

| 控制 | 定义 | n | median Δlogit | Wilcoxon p | paired(R0−ctrl) 中位 | boot95 | Cliff's δ |
|---|---|---:|---:|---:|---:|---|---:|
| V13 disp+4 | 同向量、同幅值，**x 平移 4 cells** | 38 | **−0.00385** | 5.49e-06 | +5.8711 | [+4.063, +8.513] | +0.7895 |
| V13 disp+5 | 同上，**x 平移 5 cells** | 38 | **−0.00452** | 5.18e-07 | +5.8685 | [+4.066, +8.523] | +0.7895 |
| V12 `L17_displaced` | 同向量，**y 平移 3 cells**（独立第二轮） | 38 | **−0.00535** | — | — | — | +0.7895 |

⇒ **因果通路的注入位置特异性 = SUPPORTED**：把同一向量、同一幅值搬出 ±1 cell 邻域，
效应从 **+5.87** 掉到 **−0.004**（**3 个数量级**）。两个不同方向、两个不同轮次一致。

> ⚠ **这是"在哪里注入"的特异性，不是"表征是否更集中于目标"的特异性。**
> 前者直接可测（上方），后者需要全图激活（§2 = UNAVAILABLE）。二者不可互相替代。

---

## §5 代理量：L17 向量内容几何（协议 §5–§9 不可测时的最近替代）

### §5.1 方向是**非判别性**的；范数才是

| 组 | n | median `cos(v, μ_TS)` | median `‖v‖` | mean `‖v‖` |
|---|---:|---:|---:|---:|
| **G1**（hard miss） | 38 | **+0.3080** | **7.8504** | 8.1507 |
| **G4**（val small success） | 194 | **+0.3435** | **9.1668** | 9.1273 |
| **TS**（train small success） | 467 | **+0.3320** | **9.5468** | 9.7437 |

**⇒ 三组的"方向"几乎相同（0.308 / 0.344 / 0.332），唯一区分量是"幅值"（7.85 / 9.17 / 9.55）。**
（与 `_v11_traj.json` 的 `n1=7.850 / n4=9.167 / nT=9.547` 对账一致，mean 差 ≤0.30。）

### §5.2 donor 把内容改成了什么？

| 量 | 值 | 读法 |
|---|---:|---|
| `cos(ov, μ_TS)` → `cos(dv, μ_TS)` | 0.3080 → 0.3006，**Δ = +0.0303** | **几乎没有变"更像目标"** |
| 下游 L23 的对应 cos（V13 已落盘） | 0.6958 → 0.6535，**Δ = −0.0295** | **反而略降** |
| `ρ = ‖dv‖/‖ov‖` | 中位 **1.3192**，IQR [1.055, 1.421]，**78.9% > 1** | donor 系统性更大 |

⇒ **"更像目标"这一子问题在内容面上 = NOT SUPPORTED。**

### §5.3 幅值 vs 方向：**方法学敏感性必须公开**

`median(raw) − median(norm)` 与 `median(raw − norm)` 是**不同目标的中位数**，二者不相等。
三个估计量给出**不同的**"幅值占比"：

| R | med(raw) | med(norm) | paired med Δ | **paired share** | diff-of-medians | **diff share** | mean share |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | 5.8687 | 1.7628 | 3.1356 | **53.4%** | 4.1059 | **70.0%** | 66.8% |
| R1 | 7.6327 | 1.7984 | 3.9988 | 52.4% | 5.8343 | 76.4% | 64.2% |
| R2 | 7.7296 | 2.0272 | 4.0587 | 52.5% | 5.7024 | 73.8% | 63.1% |
| R3 | 7.6682 | 2.0094 | 4.0272 | 52.5% | 5.6588 | 73.8% | 63.2% |

⇒ 稳健结论：**幅值占主导（52%–74%），方向不可忽略（约 30%）**。
**本报告不挑选其中任何一个作为定论**（协议 §8「不要为了得到更好结果而选择方法」）。
V13 报告中的"约 70%" = `diff-of-medians` 口径（R3 = 73.8%），复现一致。

### §5.4 相关检验（**全部 exploratory；n=38；multiple-comparison burden 高**）

| 检验 | Spearman ρ | p |
|---|---:|---:|
| Δraw vs ρ（幅值比） | +0.2505 | 0.129 |
| **Δraw−Δnorm（幅值分量） vs ‖dv‖−‖ov‖** | **+0.4619** | **0.0035** |
| Δnorm（方向分量） vs `cos(ov,dv)` | −0.1933 | 0.245 |
| Δnorm（方向分量） vs **Δcos-to-μTS**（"更像目标"的程度） | +0.3001 | 0.067 |
| **Δraw vs Δcos-to-μTS** | **−0.0717** | **0.669** |
| Δraw vs ΔL23cos | −0.2524 | 0.126 |

⇒ **最关键的一行是加粗的最后第二行**：**效应的大小与"变得更像目标"的程度无关**（ρ=−0.07, p=0.67）。
这是"donor 只是抬高了响应、并未改善内容"的**直接相关证据**。

**读者须知**：上述 6 项属 exploratory，且 §5 的向量几何**不是** §5–§9 要求的量。它回答的是
"donor 把 L17 的**内容**改成了什么"，**不是**"激活是否更集中于目标框"。

---

## §6 §11 G4 specificity 与「G4 无效应」的机制

| 量 | G1 (n=38) | G4 (n=194) | 检验 |
|---|---:|---:|---|
| `ρ = ‖dv‖/‖ov‖` 中位 | **1.3192** | **1.0171** | MWU p=8.29e-06，Cliff δ=**+0.4577** |
| `‖ov‖` 中位 | 7.8504 | 9.1668 | — |
| Δlogit R0 中位 | **+5.8687** | **−0.0976** | Cliff δ=+0.7770 |

**部分支持（必须给出限定）**：

- **(a) SUPPORTED**：G1 的幅值赤字**真实且 G1 特有**。G4 的 donor 几乎不改变幅值（ρ≈1.02），
  G1 的则系统性抬高（ρ≈1.32）⇒ 部分解释了为何 G4 无效应。
- **(b) 但限定很强**：合并 232 例 `Δraw vs ρ` 的 Spearman 仅 **+0.2501**（p=1.18e-04，**ρ²≈0.063**）；
  **G1 内部**同样只有 **+0.2505**（p=0.129，ρ²≈0.063）。
  ⇒ **ρ 只解释 Δraw 方差的 ~6%。**
- **⇒ 结论：效应既非纯幅值、也非纯 donor 内容，而是「幅值赤字 × target state」的交互。**
  这一点对 §11 很重要：**"把 L17 整体放大"这类不区分 target state 的机制，先天只能命中很小一部分方差。**

---

## §7 §12 Sufficiency

| Level | 定义 | 判定 |
|---|---|---|
| Level 1 | candidate / anchor matching / best IoU / best score / n_pos | **UNAVAILABLE**（从未落盘；本轮不 forward） |
| Level 2 | head logits | **AVAILABLE（仅目标类一个标量）** |
| Level 3 | 完整 prediction | **UNAVAILABLE** |

**协议 §12 明确要求：Level 2 ⇒ `SUFFICIENCY = NOT DIRECTLY MEASURED`。本报告照此签发。**

### §7.1 可得代理（同一读出口、同一 checkpoint）

先做读数一致性验证：用 checkpoint 的 `cv3[0][-1]` 权重从 V7 落盘的 `h` 重算 logit，
与 V7 落盘的 `logit_decomp` 对账 ⇒ `n=2807，max|Δ| = 2.334e-06` ✅ **一致**。

| 分布（同一读出口 `W_c·h+b_c` @ GT 中心 cell） | n | p10 | median | p90 |
|---|---:|---:|---:|---:|
| G1 base | 38 | −19.193 | **−14.634** | −10.409 |
| G1 R0 patched | 38 | −16.849 | **−8.154** | +0.145 |
| **G1 R3 patched（最强干预）** | 38 | −16.766 | **−6.071** | +0.817 |
| **V1 G4（val small success，同集）** | 194 | −4.990 | **+0.511** | +1.759 |
| T1 TS（train small success） | 467 | +0.346 | +1.602 | +2.155 |

| 比较 | 结果 |
|---|---|
| G1 **R3** patched 位于 G4 分布的 | **第 19.7 百分位**（80.3% 的 val 成功小目标比它高） |
| G1 **R0** patched 位于 G4 分布的 | 第 14.0 百分位 |
| G1 R3 / R0 位于 TS 分布的 | 第 4.4 / 3.4 百分位 |

⇒ **代理方向 = NOT_SUPPORTED**：即使取**最大**干预 R3（49 cells、oracle 位置、oracle 向量），
G1 的目标类 logit 中位 −6.071（p=2.31e-03）仍**低于约 80% 的 val 成功小目标**。

> ⚠ **读法限定**：G1 与 G4/TS 是**不同目标、不同图像**（非配对），这是**参考分布比较**、不是配对检验。
> 两者用**同一读出口、同一 checkpoint**，故尺度可比，但**不可读作配对效应量**。

### §7.2 §9 的"基线侧"（**干预后仍不可测**）

| 量 | 值 |
|---|---|
| G1 base `target − max_non_target` 中位 | +2.4258（IQR [−0.312, +4.812]） |
| G1 base 目标类已是**最高分类**的比例 | 68.4% |

⇒ 与既有结论一致（分类失败仅 0.8%）：**hard miss 不是分类问题**。
但 **干预后的非目标类投影从未落盘 ⇒ 协议 §9 的 before/after = UNAVAILABLE。**

---

## §8 §15 模式归类

协议 §15 要求把 G1 intervention 归入 P1–P5。**本轮的诚实答案是 P5**：

```text
Pattern P5 — Artifact insufficient
```

**理由**：P1/P2/P3/P4 的判别式**全部依赖 §5–§9**（`inside↑/outside≈`、`target↑/non-target≈`、
peak 移动、entropy 变化）。这些量在现有 artifact 中**不存在**（§2），因此**无法判别**。

**唯一能够排除的是 P4 的否定形式**（"logit ↑ 但 spatial activation 完全不变"）—— 我们连 activation
都没落盘，所以连"完全不变"也不能声称。

---

## §9 §16 最终 verdict

```text
FINAL VERDICT: C
```

**为什么是 C 而不是 A / B / D：**

| 选项 | 判定 | 理由 |
|---|---|---|
| **A**（有 spatial/semantic selectivity，值得进入 feasibility audit） | **排除** | 三个子问题里，唯一可测的"更像目标"**实测 NOT SUPPORTED**（§5.2：Δcos = +0.030，且 Δraw vs Δcos 相关 −0.07）；另两个 UNAVAILABLE。**没有任何可测证据支持 A** |
| **B**（causal effect 存在但 intervention 是 generic/non-selective） | **不作为主判** | B 的 §7 定义（`inside↑ outside↑`、`target↑ non-target↑`）**正是本 UNAVAILABLE 的那部分**。断言 B 等于断言一个没测过的形态。**方向对，形态未证** |
| **C**（causal effect 存在，但 artifacts **未显示**有用的 selectivity/sufficiency） | **✅ 采用** | 因果效应 SUPPORTED（§3）；artifacts **未显示**有用 selectivity（§5：内容面实测为反面）+ sufficiency（§7 代理为反面）。措辞"未显示"精确 |
| **D**（artifacts 不足） | **条件性适用** | 对 §15 的 pattern 归类**确实是 D/P5**。若你的决策规则要求 §5–§9 的直接测量，应读作 D |

**C 的动作 = `close L17 optimization route unless a new measurement is justified`。**

**为什么不推荐去补那个 new measurement：** 因为**即使 §5/§9 补测后全部转正，也不会改变结论** ——
sufficiency 是**独立**变差的（§7 代理 + E1 的训练期实测，见 §11）。
补测只能回答"表征是否更集中/更类别选择"，回答不了"够不够检出"。
**所以补测的期望信息量低于其成本。** 这是 C 而非 D 的实质理由。

---

## §10 §17 为什么 V13 有强 effect，而 E1 无 measurable gain

逐条评估协议 §17 列出的 6 个机制（**不声称是哪一个，除非 artifact 支持**）：

| 机制 | 判定 | artifact 依据 |
|---|---|---|
| **M1** donor replacement 是 content-specific | **NOT SUPPORTED** | 随机同类别 donor 拿到 raw 的 **72.8%**（+4.272 vs +5.869）；`cos(v,μ_TS)` 变化仅 **+0.030** |
| **M2** E1 是 generic gain | **PARTIALLY SUPPORTED** | 效应**由幅值主导**（52%–74%）；但方向分量仍占 **~30%**（+1.76）⇒ E1"只有增益没有内容"**不是完整解释** |
| **M3** donor 改 spatial pattern，E1 只改 amplitude | **NOT SUPPORTED（作为主因）** | donor 效应的**主体本身就是 amplitude** ⇒ 该失配**不能**解释 E1 为何无效。（而 §7 形态本身**无法直接测量**，见 §2） |
| **M4** donor 影响 class-specific representation | **UNAVAILABLE** | §9 干预后类结构未落盘。唯一下游代理（L23 vs TS 质心 cos）**反而下降** −0.0295，**不支持**该机制 |
| **M5** E1 的 location/parameterization 不等价 | **SUPPORTED（结构性）** | V13 = **GT 中心单 cell 的 oracle 向量替换**、冻结模型、推理态；E1 = L13/15/17 上的**全局乘性增益**，必须由 SGD 自行发现 |
| **M6** causal effect 是局部、条件性的，不适合全局训练 | **SUPPORTED** | (i) 被 ±1 cell 邻域绑住（R0→R1 +1.76，R2/R3 增量 +0.10/−0.06，**饱和**）；(ii) 位移 +4/+5 即归零；(iii) **ρ²≈0.06** ⇒ 效应强烈依赖 target state |

### §10.1 独立于上表的、最强的一条

```text
V13 最强干预（R3, 49 cells, oracle 位置, oracle 向量）:
    中位 logit  −14.634  →  −6.071      (p = 2.31e-03)
val 成功小目标（G4）中位 logit:
                              +0.511
⇒ G1 最强干预后仍位于 G4 分布的 第 19.7 百分位
```

⇒ **一个"全知位置 + 全知向量"的 oracle 编辑，都没能把 hard-miss 推到成功区间。**
E1 是一个必须在训练中学出来的**全局**增益，其可达效果只能更小。
**V13 证明的是「L17 参与因果」，不是「增强 L17 足以检出」。** 两者不矛盾。

⚠ **逻辑边界（必须写清）**：R3 是**推理态局部编辑**，它**不能**严格界定"训练期干预"的上限
（训练会重排下游权重）。**真正回答"训练期行不行"的是 E1 本身** —— 而 E1 的回答是 no measurable gain。
所以本条是**支持性证据**，不是**决定性证明**。

### §10.2 E1_CONTRADICTION 的评级

```text
E1_CONTRADICTION: PARTIALLY_EXPLAINED
```

- **已解释**：为什么预测层"抬一下响应"不会带来 mAP 增益 —— 效应 is spatially bound、
  target-state-conditional（ρ²≈0.06）、donor-content-insensitive；且 E1 的 parameterization
  与 V13 的 oracle 编辑**结构性不等价**（M5）。
- **未解释**：为什么 **E1 的官方 Δ = −0.00540（CI 含 0）与 V13 的 +5.87 相差如此之大**。
  这需要 §5–§9 的形态测量（E1 是否把整个 feature map 一起变亮）才能定论，**本轮 UNAVAILABLE**。

---

## §11 §18 NEXT HYPOTHESIS（**仅记录，不实施**）

```text
H1  效应 = 幅值赤字 × target state 的交互（ρ²≈0.06），而非单一标量赤字。
    若将来要测，应测 "per-target 可预测性"：能否由 target 自身特征预测 Δlogit？
    —— 需要一次新的 forward dump，本轮禁止。
H2  要判别 §15 的 P1/P2/P3，唯一需要的是**一次 L17 全图 (1,512,40,40) 的干预前/后 dump**。
    本轮禁止 forward，故未做。是否值得做，由 §9 的 C 决定（本报告建议：不值得，见 §9）。
```

**不创建** 任何新 gain / attention / donor / kernel / layer / training config（协议 §18）。

---

## §12 Limitations

1. **§5–§9 UNAVAILABLE**（§2）—— 这是本轮最大的限制，也是 pattern 归类只能给 P5 的原因。
2. **donor 身份是跨 artifact 复原**（V13 未落盘，用 V12 对账 38/38）。若 V12 与 V13 的池构造有
   未察觉差异，该复原会失效；但二者规则在源码层逐字相同，且 `n_donor_cand` 也 38/38 一致。
3. **§5/§6 的向量几何是代理量**，不是协议要求的空间量。**不可读作 spatial selectivity 的证据。**
4. **非配对比较**（G1 vs G4/TS 参考分布）：同一读出口与 checkpoint，但不同目标/图像。
5. **n(G1) = 38** ⇒ 所有相关检验 exploratory，multiple-comparison burden 高，
   未做（也不主张）任何校正后的显著性声明。
6. **G4 的 `R1_raw`/`R3_raw` 未跑**（V13 按规格只跑 R0/R2）⇒ G4 只有两个半径。
7. **V14 的 G4 只累积到 26/174**（其自身限制）⇒ 本轮的 G4 结论**只用 V13 的 194 条**，不混用。
8. **不推出"L17 有害"、不推出"应当去掉 L17"、不推出"E1 方向错误"**。本轮最多只能说
   "**现有 artifact 未显示 L17 intervention 具有可训练利用的 selectivity/sufficiency**"。

---

## §13 §20 Provenance

```text
TRAINING        = NO
BACKWARD        = NO
OPTIMIZER_STEP  = NO
MODEL_FORWARD   = NO

EXISTING_FILES_CHANGED       = 0          （476 个既有 artifact 逐个 SHA256 比对）
EXISTING_CHECKPOINTS_CHANGED = 0
EXISTING_DIAGNOSTICS_CHANGED = 0
EXISTING_CONFIGS_CHANGED     = 0

GIT_STATUS_BEFORE = 70 行（10 M + 60 ??）
GIT_STATUS_AFTER  = 70 行，逐行相同
GIT_HEAD          = f659609c28f6b4f500212197d5229550a36d12d8（前后一致）

NEW_FILES = diagnostic/l17_selectivity_audit/
              REPORT.md  _analyze.py  AUDIT.log  _tables.json  _results.npz
              _sha_before.json  _git_before.txt  _git_after.txt
```

### V13/V14 artifact SHA256（**前后逐位未变**）

| artifact | SHA256[:16] |
|---|---|
| `_v13_dose.json`（V13） | `f6139cef5a153689` |
| `_v14_upstream.json`（V14） | `347c4c3de50cf089` |
| `_v12_patch.json` | `51b6cb6cc3d30aef` |
| `runs/…_sepstem_clahe/weights/best.pt` | `1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda` |

（完整 476 项前后哈希表：`_sha_before.json`；比对结果 = 0 changed / 0 missing。）

### 读取对象清单

`_v13_dose.json` / `_v14_upstream.json` / `_v12_patch.json` / `_v11_layers.npz` / `_v11_ids.json` /
`_v11_traj.json` / `_v7_vectors.json` / `_v9_attrib.json` / `_v8_probes.json` / `_v10_stages.json` /
`_v3_analysis.json` / `_v2_records.json` / `best.pt`（**仅取 `model.30.cv3.0.2.weight/bias`**）/
`_v13_analyze.py` / `_v12_patch.py` / `_v11_layers.py` / `_v7_extract.py`（用于确认干预与落盘口径）。

### 未采用的方法（明确记录，避免"选方法挑结果"）

- **熵 / effective area（§8）**：本可对**单 cell 向量**算一个"通道熵"当代理，但那是**语义上的偷换**
  （§8 要的是 spatial activation 的 probability map）。**未采用。**
- **用 `_v9_attrib.json` 的 `zA_h` 当 class 归因**：该字段是**标量**、非 class-specific。**未采用。**
- **用 `_v8_probes.json` 当 class 探针**：其 `wA/wB/wC` 是**成功/失败**方向探针，非类别探针。**未采用。**
- **三个幅值占比估计量**：全部报告，**不挑选**。

---

**本轮结束，停止。不训练、不设计下一架构、不实施 §11 的 H1/H2、不自行启动下一轮。**
