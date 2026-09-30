# w_A 的因素归因 + 冻结方向的 train→val 投影

**性质**：只读。0 训练 / 0 改模型-loss-assigner-增强 / 0 核心文件改动 / 0 碰 test-submission-online。
**本轮 0 次 forward** —— 全部复用 `_v7_vectors.json` 与 `_v8_probes.json`。

**完整性**：`augment.py 5cb9a407…` / `default.yaml 991a89b3…` / `scripts/train.py 9f55b09f…` / `detect/train.py b021354f…` / `best.pt 1cae45f7…` —— 全部与本轮开始时一致。

> ⚠️ `w_A` **未重新拟合**，直接使用 V8 保存的方向；标准化沿用 V8 同一构造。
> ⚠️ 已剔除循环变量：`is_detected` / `best_iou` / `center_logit` / `logit_decomp` / `class_prob` / G1-G4 标签，**均不作解释变量**。

---

## C. 冻结 w_A 的 train→val 投影（**本轮最重要**）

### P3

| domain/group | n | median z_A | IQR | P90 | P99 | max |
|---|---:|---:|---|---:|---:|---:|
| train small (all) | 467 | **−6.156** | [−9.31, −3.59] | −1.576 | 1.514 | **4.315** |
| train med/lg | 1686 | **+2.715** | [0.10, 4.93] | 6.590 | 9.134 | 11.302 |
| train small success | 437 | −6.132 | [−9.34, −3.58] | −1.510 | 1.353 | 2.457 |
| **VAL G1** | 38 | **+4.185** | [3.42, 5.58] | 7.019 | 9.361 | 9.807 |
| VAL G4 | 194 | −5.775 | [−7.79, −4.36] | −3.258 | −1.920 | −1.552 |
| VAL other-cls small | 47 | −3.164 | [−5.47, −0.69] | 1.351 | 4.161 | 5.374 |
| VAL same-cls med/lg | 1850 | +3.147 | [0.64, 5.07] | 6.590 | 9.274 | 11.833 |

**train small 中 ≥ G1 中位的比例 = 0.214%（1/467）**

### H

| domain/group | n | median z_A | IQR | P90 | P99 | max |
|---|---:|---:|---|---:|---:|---:|
| train small (all) | 467 | **−5.466** | [−6.60, −4.62] | −3.504 | 2.690 | **4.422** |
| train med/lg | 1686 | +2.792 | [−4.29, 10.80] | 14.003 | 17.199 | 20.472 |
| **VAL G1** | 38 | **+6.425** | [4.80, 8.06] | 10.575 | 14.872 | 16.477 |
| VAL G4 | 194 | −6.061 | [−6.79, −4.38] | −3.046 | −1.879 | −1.305 |

**train small 中 ≥ G1 中位的比例 = 0.000%**

### 判读（对应你 §7 的 情况 A / B）

```
train small 的中位 z_A（−6.16 / −5.47）≈ VAL G4 的中位（−5.78 / −6.06）
                                        ⇒ 「已检出小目标」在两域**完全一致**
但 VAL G1 位于 +4.19 / +6.43 —— **train small 的最大值只有 +4.32 / +4.42**
                                        ⇒ G1 所在的方向区域在 train 小目标分布中**几乎不存在**
                                          （H 空间 0.000%；P3 0.214%）
```

⇒ **情况 B ⇒ TRAIN→VAL REPRESENTATION GAP：SUPPORTED。**

**这不是普通的 domain shift**（「成功样本是否相似」在本轮已确认**高度相似**），
而是**「val G1 特有的失败方向」在 train 表征分布中基本没有对应区域**。

---

## A. Univariate attribution（Spearman rho；已剔循环变量）

| variable | n | **P3 rho** | P3 p | **H rho** | H p |
|---|---:|---:|---:|---:|---:|
| `native_sqrt` | 371 | **−0.021** | **0.68** | +0.116 | 0.026 |
| `in_sqrt` | 371 | −0.129 | 0.013 | −0.120 | 0.021 |
| `border_norm` | 371 | **−0.207** | 6.0e-05 | −0.200 | 1.1e-04 |
| `n_r1` | 371 | −0.103 | 0.047 | −0.068 | 0.19 |
| `n_r2` | 371 | −0.192 | 2.0e-04 | −0.094 | 0.071 |
| `n_r3` | 371 | **−0.234** | 5.2e-06 | −0.120 | 0.020 |
| `n_r5` | 371 | **−0.284** | **2.5e-08** | −0.148 | 4.4e-03 |
| `img_gt` | 371 | −0.218 | 2.2e-05 | −0.137 | 8.4e-03 |
| `n_same_r3` | 371 | −0.240 | 2.9e-06 | −0.087 | 0.095 |
| `n_other_r3` | 371 | **+0.014** | **0.79** | −0.048 | 0.36 |
| `visible_std` | 236 | **+0.077** | **0.24** | +0.174 | 7.3e-03 |
| `visible_gradmean` | 236 | +0.052 | 0.43 | +0.155 | 0.017 |
| `visible_edgedens` | 236 | −0.062 | 0.35 | +0.072 | 0.27 |
| `visible_lapvar` | 236 | +0.100 | 0.12 | +0.188 | 3.8e-03 |
| `infrared_std` | 236 | **+0.058** | **0.38** | +0.088 | 0.18 |
| `infrared_gradmean` | 236 | +0.106 | 0.10 | +0.155 | 0.017 |

**读数**：
- **与尺寸无关**（`native_sqrt` rho −0.021, p=0.68）✓ 与 V8 一致
- **视觉/模态证据在 P3 上全部不显著**（p ≥ 0.10）⇒ **MODALITY/IMAGE-EVIDENCE: NOT SUPPORTED**
- **密度/稀疏类**是唯一稳定的家族：`n_r5` −0.284、`n_r3` −0.234、`n_same_r3` −0.240、`img_gt` −0.218
- `n_other_r3` ≈ 0（异类邻居无关）⇒ 是**同类稀疏**而非一般拥挤

---

## B. Incremental multivariate R²（val small GT；CV R² = 5×5 重复 K-fold）

### P3

| model | n | k | R² | adjR² | **CV R²** | ΔR² |
|---|---:|---:|---:|---:|---:|---:|
| Model 0 intercept | 371 | 0 | 0.0000 | 0.0000 | 0.0000 | — |
| Model 1 geometry | 371 | 6 | 0.1427 | 0.1285 | **0.1098** | +0.1427 |
| Model 1′ geometry（visual 子集） | 236 | 6 | 0.2051 | 0.1843 | **0.1588** | +0.0624 |
| Model 2 +visual/modality | 236 | 12 | 0.2287 | 0.1872 | **0.1451** | +0.0236 |
| Model 3 +class FE | 236 | 24 | 0.2797 | 0.1978 | **0.1074** | +0.0510 |

### H

| model | n | k | R² | adjR² | **CV R²** | ΔR² |
|---|---:|---:|---:|---:|---:|---:|
| Model 1 geometry | 371 | 6 | 0.1613 | 0.1475 | **0.1286** | +0.1613 |
| Model 1′ geometry（visual 子集） | 236 | 6 | 0.2040 | 0.1831 | 0.1549 | +0.0426 |
| Model 2 +visual/modality | 236 | 12 | 0.2173 | 0.1752 | **0.1143** | +0.0134 |
| Model 3 +class FE | 236 | 24 | 0.3350 | 0.2594 | **0.1003** | +0.1177 |

**⇒ 关键判读**：
- **加 visual / class 后名义 R² 上升，但 CV R² 反而下降**（P3：0.1588 → 0.1451 → 0.1074；H：0.1549 → 0.1143 → 0.1003）⇒ **纯过拟合，额外变量没有可泛化的解释力**
- **可泛化解释的上限 ≈ CV R² 0.11–0.16** ⇒ **约 85% 的 z_A 方差无法被任何已测可观测量解释**

---

## D. Density / class 控制

### P3（全量 val small n=371）

| spec | k | R² | adjR² | **CV R²** |
|---|---:|---:|---:|---:|
| density only (`n_r3`) | 1 | 0.0686 | 0.0661 | **0.0613** |
| density + class FE | 13 | 0.1344 | 0.1029 | **0.0654** |
| density + class FE + size | 15 | 0.1419 | 0.1056 | **0.0621** |
| n_r3+n_same+n_other + class + size | 17 | 0.1467 | 0.1056 | 0.0605 |

### H

| spec | k | R² | adjR² | **CV R²** |
|---|---:|---:|---:|---:|
| density only | 1 | 0.0541 | 0.0516 | 0.0456 |
| density + class FE | 13 | 0.2187 | 0.1902 | **0.1350** |
| density + class FE + size | 15 | 0.2289 | 0.1963 | **0.1373** |
| 三密度项 + class + size | 17 | 0.2298 | 0.1927 | 0.1296 |

**⇒ 密度在**控制 class 与 size 之后 CV R² 仍稳定在 ~0.06（P3）/ ~0.14（H）** ⇒ 不是被 class/size 混杂出来的假相关，**可以保留为弱因素**；但量级仅 **6%–14%**。

---

## E. Mechanism verdict

```text
SCALE:                        NOT SUPPORTED
MODALITY/IMAGE-EVIDENCE:      NOT SUPPORTED
DENSITY/CONTEXT:              WEAKLY SUPPORTED
CLASS/SEMANTIC:               WEAKLY SUPPORTED
TRAIN→VAL REPRESENTATION GAP: SUPPORTED
MIXED / UNRESOLVED:           SUPPORTED
```

| 候选 | 判定 | 依据 |
|---|---|---|
| SCALE | **NOT SUPPORTED** | `native_sqrt` rho −0.021 (p=0.68)；cos(w_A,w_B) = −0.18 |
| MODALITY/IMAGE-EVIDENCE | **NOT SUPPORTED** | P3 上全部 visual/IR rho 不显著（p ≥ 0.10）；加 visual 后 **CV R² 下降** |
| **DENSITY/CONTEXT** | **WEAKLY SUPPORTED** | `n_r5` rho −0.284（p=2.5e-08）；控制 class+size 后 CV R² 仍 ~0.06 / ~0.14。**但只解释 6–14%** |
| CLASS/SEMANTIC | **WEAKLY SUPPORTED** | cos(w_A,w_C)=+0.27~0.34；但 class FE 只提高名义 R²、**CV R² 反而下降** |
| **TRAIN→VAL REPRESENTATION GAP** | **SUPPORTED** | train small 与 VAL G4 的 z_A 几乎相同（−6.16 / −5.78）；VAL G1 在 +4.19，而 **train small 最大值仅 +4.32（H：无一例达到 G1 中位，0.000%）** |
| **MIXED / UNRESOLVED** | **SUPPORTED** | 最佳 CV R² ≈ 0.11–0.16 ⇒ **~85% 方差不可由已测可观测变量解释** |

---

## F. 下一步（只允许一个）

按你 §最后一条的判据：**所有 observable 的 CV R² 都很低（≈0.13–0.16，远未达「解释了很大比例」）**
⇒ `w_A` 主要编码的是**当前可观测变量无法解释的 latent representation factor**。

> **因此下一步转向：比较「train 成功 small」与「val G1」在 P3→cv3 的 layer-by-layer trajectory / channel response，定位从哪一层开始产生 divergence。**

具体（**只读**）：
- 对 `cv3[0]` 的每一层（`Sequential(DWConv, Conv)` → `Sequential(DWConv, Conv)` → `Conv2d(12,1)`）
  逐层取 GT-center 处的**完整张量**（本轮 V7 只取了首尾两层）
- 对 **train 成功 small** 与 **val G1** 分别做同一投影，报告逐层的
  `cos(v_layer, μ_layer^{train small succ})`、`|v_layer|`、以及与 `w_A` 在每层的**对应方向**
- 判据：若 divergence 在**第 1 层之后就出现** ⇒ 该层对 G1 输入的处理已异常；若在**第 2 层之后** ⇒ 更深的非线性组合造成

**不做**：cls-loss / assigner / P2 / OASA / crop / 1536 / augmentation sweep。

```text
HOLD — no training
```

---

## 产物

```
diagnostic/p3_feature_space/_v9_attrib.py    归因 + 增量 R² + 冻结投影（只读）
diagnostic/p3_feature_space/_v9_attrib.json  z_A 明细
diagnostic/p3_feature_space/REPORT_V9.md     本报告
```

---

## 附：本轮限制

1. **visual 协变量只覆盖 236/371**（V3 的 visual 子集仅含 70 missed + 166 controls，偏向 G1）⇒ Model 2/3 的子集可能与全量不同质；已同时给出 Model 1（全量）与 Model 1′（子集）以便对照。
2. **train 侧无 density/border/visual 协变量**（V5/V6 未记录）⇒ §A/B/D 只在 val 侧做；§C 只用向量投影，不受影响。
3. `_v9_attrib.py` 中两处未使用的 `specs` 列表是残留代码，不影响结果。
