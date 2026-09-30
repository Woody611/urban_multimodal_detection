# L17 spatial dose-response（R0–R3）

**性质**：只读 causal diagnostic。0 训练 / 0 反传 / 0 改 checkpoint-模型源码-loss-assigner-augmentation-config /
0 test inference / 0 submission / 0 新架构实验。**未保存任何被改过的模型**。

**完整性**：`augment.py 5cb9a407…` / `default.yaml 991a89b3…` / `scripts/train.py 9f55b09f…` / `detect/train.py b021354f…` / `best.pt 1cae45f7…` 全部未变；仅新增 `diagnostic/p3_feature_space/` 下文件；**无新增 runs/**；未生成 submission。

---

## §1 Frozen 条件（与 V12 逐字相同）

checkpoint / preprocessing / eval mode / images / GT-center mapping / donor selection
（同类别 ∧ small ∧ 成功 ∧ G4 → `min|Δsqrt(area)|` → tie-break `(stem,gt)`；**未重新选择**）。
**G1 = 38｜G4 = 194**；共 232 target / 93 图。

几何：L17 stride 32 ⇒ **40×40@1280**。`R0=1 cell`、`R1=3×3=9`、`R2=5×5=25`、`R3=7×7=49`，全部 clamp 在边界内。
Mode A = raw donor；Mode B = **逐 cell** norm-matched（每个被替换 cell 各自匹配其原 norm）。

> ⚠️ 首版曾用 **L23(160×160)** 的 cell 坐标去 patch **L17(40×40)** ⇒ `IndexError`。两个坐标系已分开计算。

## §2 V12 REPRODUCTION GATE —— **PASS**

```
V13 R0_raw median Δlogit = +5.8687      (V12 = +5.869)      Δ = −0.0003    n = 38
```

---

## §9 核心表

| R | cells | **G1 raw Δ** | IQR | %>0 | paired p | G1 norm Δ | %>0 | G4 raw Δ | %>0 | **rnd Δ** | %>0 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **R0** | 1 | **+5.869** | [2.01, 9.54] | 89.5% | **3.1e-08** | +1.763 | 76.3% | **−0.098** | 34.0% | +4.272 | 84.2% |
| **R1** | 9 | **+7.633** | [3.23, 12.52] | 84.2% | **1.3e-08** | +1.798 | 68.4% | *(未跑)* | — | +6.424 | 73.7% |
| **R2** | 25 | **+7.730** | [3.22, 12.45] | 84.2% | **1.5e-08** | +2.027 | 71.1% | **−0.489** | 28.4% | +6.186 | 71.1% |
| **R3** | 49 | **+7.668** | [3.33, 12.45] | 84.2% | **1.5e-08** | +2.009 | 71.1% | *(未跑)* | — | +6.159 | 71.1% |

G4 只在 R0/R2 跑（按规格），故 R1/R3 的 G4 列为空。

## §7 displaced control（固定方向 +4 / +5 cells，不与中心邻域重叠）

| displacement | n | median Δ | IQR | %>0 | p |
|---|---:|---:|---|---:|---:|
| **+4 cells** | 38 | **−0.0038** | [−0.0081, −0.0005] | 23.7% | 0.000 |
| **+5 cells** | 38 | **−0.0045** | [−0.0089, −0.0014] | 13.2% | 0.000 |

**⇒ 空间特异性极强**：中心 R0 (+5.869) vs 位移 +4 (+−0.0038)，差 **+5.87**。

---

## §12 Dose-response

```
G1 raw  Δlogit   R0→R3:  +5.869   +7.633   +7.730   +7.668
G1 norm Δlogit   R0→R3:  +1.763   +1.798   +2.027   +2.009
G1 rnd  Δlogit   R0→R3:  +4.272   +6.424   +6.186   +6.159

增量:  R0→R1 +1.764    R1→R2 +0.097    R2→R3 −0.061     合计 R0→R3 +1.800
R0 占总增益比例 = 76.5%
```

**⇒ 对应你 §12 的 Pattern A（early saturation）：单 cell 已产生大部分 rescue，R1 后再扩无增益（R2/R3 反而略降）。**

## §10 绝对水平（**不以 logit=0 作阈值**，仅作参考尺度）

| | base logit (p) | patched logit (p) |
|---|---|---|
| R0（1 cell） | −14.634 (4.4e-07) | **−8.154 (2.9e-04)** |
| R3（49 cells） | −14.634 (4.4e-07) | **−6.071 (2.3e-03)** |

**⇒ 即使 patch 49 个 cell（7×7），logit 也只到 −6.07。扩大 49 倍面积只换来 +1.8 的额外 logit。**

## §11 L23 downstream 一致性

| R | L23 norm Δ | L23 cos（对 TS 质心方向） |
|---|---:|---:|
| base | — | 0.6958 |
| R0 | +0.627 | 0.6535 |
| R1 | +0.587 | 0.6685 |
| R2 | +0.574 | 0.6671 |
| R3 | +0.570 | 0.6671 |

**⚠️ 解离**：logit 随 patch 上升（+5.87），但 **L23 与 TS 质心的 cosine 反而略降**（0.696 → 0.667）。
⇒ **logit 的改善不能由「L23 向 TS 质心对齐」解释** —— 该 cosine 不是有效的 mediator 指标。
L23 norm 只 +0.63 且随 R 基本不变。

## §9 G1-specificity

| R | G1 raw | G4 raw | 差 |
|---|---:|---:|---:|
| R0 | +5.869 | −0.098 | **+5.97** |
| R2 | +7.730 | −0.489 | **+8.22** |

**⇒ G1-specificity 成立，且随半径扩大而增强**（G4 不但无 rescue，反而略被压低）。

### ⚠️ norm vs direction 的重要分离

raw (+5.869) vs norm-matched (+1.763) 在 R0 相差 **+4.1**；R3 时 +7.668 vs +2.009 相差 **+5.7**。
⇒ **rescue 有约 70% 来自 donor 向量的**幅值**，而非其方向内容。**
（这与 V12 的 random-donor 结果一致：**任何**典型成功小目标的 L17 向量都有效，donor 选择不敏感。）

---

## §15 最终 verdict

```text
L17 spatial dose-response:            SATURATING

G1-specificity:                       YES
Spatial specificity:                  YES

L17 local neighborhood involvement:   SUPPORTED
L17 sufficiency:                      NOT SUPPORTED

Next causal region:                   L12–15（上游；**本轮未做因果检验，仅为候选**）
HOLD:                                 YES
```

### 对应你 §12/§13 的解释

**Pattern A（early saturation）**：R0 已捕获 76.5% 的总增益，R2/R3 无进一步增益。
⇒ **L17 local cell 是主要可作用区域，但单 cell donor 本身不足以完全恢复；扩大空间范围也不足以。**

**并且严格遵守 §13**：本轮**不**声称「L17 是唯一 causal bottleneck」。patch 面积增加本身改变更多 activation，
故只能说 **spatial extent of the L17 intervention affects downstream response**；由于已经饱和，
甚至不能说「越大越强」。

**L17 不充分（NOT SUPPORTED）的直接证据**：49 个 cell 的 patch 只把 logit 推到 −6.07 ——
若 L17 local neighborhood 是充分瓶颈，扩大范围应持续逼近 0。它没有。

---

## 产物

```
diagnostic/p3_feature_space/_v13_dose.py      R0–R3 × {raw, norm, random, displaced} + G4
diagnostic/p3_feature_space/_v13_dose.json    232 target 完整记录
diagnostic/p3_feature_space/_v13_analyze.py   复现门 + 核心表 + paired 检验
diagnostic/p3_feature_space/_v13_result.json  汇总
diagnostic/p3_feature_space/REPORT_V13.md     本报告
```

## 附：本轮自己的一个 bug（已修）

L17（40×40）与 L23/cv3-输入（160×160）是两个坐标系；首版用 L23 坐标去 patch L17 ⇒ `IndexError: index 95 out of bounds for dimension 2 with size 40`。已分开计算 `(y0,x0)` 与 `(y23,x23)`；**在跑出任何结果之前发现**，不影响任何被报告的数字。
