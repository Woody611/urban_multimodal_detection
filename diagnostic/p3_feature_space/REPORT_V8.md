# THREE-PROBE DIRECTION ANALYSIS（P3 / H）

**性质**：只读。0 训练 / 0 改模型-head-loss-assigner-增强 / 0 核心文件改动 / 0 碰 test-submission-online。
**复用**：`_v7_vectors.json`（54MB，4960 GT × 256-d P3 × 256-d H）。**本轮 0 次 forward。**

---

## 0. Data integrity

`augment.py 5cb9a407…` / `default.yaml 991a89b3…` / `scripts/train.py 9f55b09f…` / `detect/train.py b021354f…` / `best.pt 1cae45f7…` —— **全部与本轮开始时一致**；未读 test/submission/online；未改任何既有文件。

---

## A. Probe performance（严格 5-fold 分层 CV）

| probe | 空间 | n₊ | n₋ | balanced acc | ROC-AUC |
|---|---|---:|---:|---:|---:|
| **A** G1 vs G4 | **P3** | 33 | 194 | **0.824 ± 0.043** | **0.954 ± 0.024** |
| **B** small-success vs med/lg-success | **P3** | 462 | 600 | 0.939 ± 0.012 | 0.988 ± 0.003 |
| **C** G1 vs other-class small-success | **P3** | 33 | 47 | 0.979 ± 0.026 | 0.988 ± 0.015 |
| A G1 vs G4 | H | 33 | 194 | 0.928 ± 0.054 | 0.985 ± 0.007 |
| B small vs med/lg | H | 462 | 600 | 0.899 ± 0.015 | 0.966 ± 0.006 |
| C G1 vs other-class | H | 33 | 47 | 0.990 ± 0.020 | 1.000 ± 0.000 |

置换检验**已关闭**（1000 次 IRLS 拟合在本机实测耗尽 14017s CPU 仍未完成）。probe A 的置换 p 已由 V7 给出 = **0.0000**（200 次置换）。

**修正的两处采样 bug（本轮自查）**：
1. `succ` 首版取自 `v2rec`（只含 small GT）⇒ med/large 的 success 恒为 False，Probe B 对照集塌缩到 **34**、Probe C 到 **9**。改为**从 D′ 预测对全部 2807 个 val GT 重算** best same-class IoU（success = 2312）✓
2. Probe C 的尺寸带内仅 9 个 ⇒ 改用全部 other-class small-success（n=47），并在报告中标注尺寸混杂

---

## B. Direction cosine matrix

| | P3 | H |
|---|---:|---:|
| **cos(w_A, w_B)** | **−0.1755** | **−0.1727** |
| **cos(w_A, w_C)** | **+0.3418** | **+0.2720** |
| cos(w_B, w_C) | −0.0629 | −0.0061 |

**⇒「G1-vs-G4」方向与「尺度」方向近乎**正交**（−0.17）；与「G1-vs-其他类」方向只有**中等**对齐（+0.27 ~ +0.34）。**

---

## C. 跨 probe 投影 z = w·f（标准化特征；中位 [P25,P75]）

### P3

| group | n | z_A（G1 侧为 +） | z_B（small 侧为 +） | z_C（G1 侧为 +） |
|---|---:|---|---|---|
| **G1** | 38 | **+4.19** [3.42, 5.58] | +11.25 [8.10, 16.39] | **+5.57** [4.25, 6.78] |
| G2 | 29 | −2.96 [−7.60, −0.71] | +15.90 [9.78, 25.85] | +2.62 [−0.53, 4.89] |
| G4 | 194 | **−5.78** [−7.79, −4.36] | +17.10 [7.34, 21.97] | +0.30 [−1.93, 3.09] |
| other-cls small | 47 | −3.16 [−5.47, −0.69] | +8.37 [5.12, 13.00] | **−5.65** [−8.20, −4.20] |
| **same-cls med/lg** | 1850 | **+3.15** [0.64, 5.07] | **−7.05** [−9.50, −4.07] | +1.91 [0.03, 3.09] |
| train small succ | 437 | −6.13 [−9.34, −3.58] | +14.01 [8.03, 22.04] | −1.76 [−4.54, 0.80] |

### G1 相对各组的 Cliff's Δ

| vs group | Δz_A | Δz_B | Δz_C |
|---|---:|---:|---:|
| G2 | 0.927 | −0.261 | 0.610 |
| G4 | **0.994** | −0.174 | 0.821 |
| other-cls small | 0.932 | 0.277 | **1.000** |
| **same-cls med/lg** | **0.319** | **0.974** | 0.949 |
| train small succ | 0.986 | −0.127 | 0.969 |

### H 空间（同结论，略更锐）

| group | z_A | z_B | z_C |
|---|---|---|---|
| **G1** | **+6.43** [4.80, 8.06] | +9.63 [6.01, 14.59] | +7.00 [5.56, 9.89] |
| G4 | −6.06 [−6.79, −4.38] | +7.60 [5.39, 9.71] | +0.40 [−1.55, 1.69] |
| **same-cls med/lg** | **+3.45** [−3.67, 10.76] | −4.00 [−7.46, −0.36] | +1.48 [0.07, 3.01] |
| other-cls small | −6.40 [−7.03, −4.60] | +4.68 [4.00, 6.50] | −7.91 [−10.06, −6.21] |

**Cliff's Δ(G1 vs same-cls med/lg)**：Δz_A = **0.226**、Δz_B = **0.887**。

---

## D. P3 vs H

| | Probe A | Probe B | Probe C | cos(w_A,w_C) |
|---|---:|---:|---:|---:|
| P3 | 0.954 / 0.824 | 0.988 / 0.939 | 0.988 / 0.979 | +0.342 |
| H | 0.985 / 0.928 | 0.966 / 0.899 | 1.000 / 0.990 | +0.272 |

**⇒ H 在 A/C 上更锐（与 V7 的「head 是放大器」一致），但方向余弦两空间几乎相同 ⇒ `cv3` 变换**保持**了方向几何，没有把它旋转到另一个语义轴。**

---

## E. Mechanism verdict

```text
SCALE/CONTEXT:    NOT SUPPORTED
CLASS/SEMANTIC:   WEAKLY SUPPORTED
MIXED:            SUPPORTED
```

### 依据（逐条对应你 §6 的 Pattern）

**Pattern 1（scale/context）—— NOT SUPPORTED**
- `cos(w_A, w_B) = −0.176`（要求「高度一致」）✗
- 补充证据：`z_A` 与 **native √area** 的 Spearman **rho = −0.036, p = 0.49（NS）** ⇒ w_A **与目标原生尺寸无关** ✗

**Pattern 2（class/semantic）—— WEAKLY SUPPORTED**
- `cos(w_A, w_C) = +0.34(P3) / +0.27(H)` ⇒ 只有中等对齐，达不到「高度一致」 ⚠️
- 且**组间关系不符**：G1 在 z_C 上为 **+5.57**，other-class small 为 **−5.65** ⇒ 两者在 w_C 方向**位于相反两侧** ✗
- 同类 medium/large 在 z_C 上为 +1.91，**与 G1 同侧** ⇒ 也违背「同类 medium/large 不表现出同样 separation」

**Pattern 3（mixed）—— SUPPORTED**
- `w_A` 与 `w_B` 有明确（负向）关系、与 `w_C` 有中等关系 ⇒ 「同时与两者有关」
- **最独特的观察**：在**区分 G1 与 G4** 的方向上，**同类 medium/large 落在 G1 一侧**
  （z_A：G1 **+4.19** / med-lg **+3.15** / G4 **−5.78**；Cliff's Δ(G1 vs med/lg) = **0.319**，而 Δ(G1 vs G4) = **0.994**）
  ⇒ **G1 的表征在「区别于已检出小目标」的方向上，更像 medium/large 那一群，而不是像已检出的小目标。**

### 补充：`w_A` 到底编码了什么？（描述性相关，非下一步的替代）

对全部 371 个 val small GT，`z_A` 与协变量的 Spearman：

| 协变量 | rho | p | 判读 |
|---|---:|---:|---|
| `is_detected` | −0.652 | 2.8e-46 | **循环**（w_A 就是按它拟合的），不作证据 |
| `n_r3`（r=3 邻居数） | −0.229 | 8.2e-06 | 中等：z_A 高 ↔ 邻居少 |
| `img_gt`（图内 GT 数） | −0.205 | 7.2e-05 | 中等：z_A 高 ↔ 图更稀疏 |
| `border_norm` | −0.184 | 3.8e-04 | 中等：z_A 高 ↔ 更靠近边框 |
| `in_sqrt` | −0.120 | 2.1e-02 | 弱 |
| `n_r1` | −0.089 | 8.8e-02 | NS |
| **`native_sqrt`** | **−0.036** | **0.49** | **NS ⇒ 与尺寸无关** |

**⇒ 除循环项外，所有已测几何协变量对 `z_A` 的解释力都很小（|rho| ≤ 0.23）。`w_A` 主要编码的是**未被这些几何量捕获**的东西。**

---

## F. 下一步（只允许一个）

> **给 `z_A` 做「可观测量回归」，把 `w_A` 编码的因素命名出来。**

用本轮已落盘的向量 + 已有的 GT/图像侧可观测量，对 **`z_A`（P3 与 H 各做一次）** 拟合一个**只读**的可解释模型（如带 L2 的线性回归 / 梯度提升 + 特征重要度），候选协变量：
- 几何：`native_sqrt`、`in_sqrt`、`aspect`、`border_norm`
- 场景：`n_r1/r3/r5`、`nearest_same/other_class_dist_norm`、`img_gt`
- **局部图像证据**：V3 已算的 `visible_std / gradmean / edgedens / lapvar`、`infrared_std / gradmean`、`depth` 零值比
- 模态：`infrared_std`（V3 唯一显著项）、depth 有效性

**判据**：若某个可观测量（尤其 IR 对比度或局部上下文量）能稳定解释 `z_A` 的较大方差 ⇒ **该因素被命名**，下一步才有明确靶点；若 R² 仍很低 ⇒ `w_A` 编码的是**当前在 val 侧不可观测的东西**（例如训练增强分布下的统计量），此时进入 **UNRESOLVED**，应转为度量「train 侧同方向投影的同名协变量关系」来找差异。

**在此判据落地前：**
```text
HOLD — insufficient evidence
```

**仍然禁止**：改 cls loss / 改 assigner / P2 / augmentation sweep / crop / 1536 / OASA / test-online 实验。

---

## 产物

```
diagnostic/p3_feature_space/_v8_probes.py  三 probe + 余弦矩阵 + 跨投影（IRLS，只读）
diagnostic/p3_feature_space/_v8_probes.json 结果
diagnostic/p3_feature_space/_v8_run.log     运行日志
diagnostic/p3_feature_space/REPORT_V8.md    本报告
```
```bash
python -X utf8 -u diagnostic/p3_feature_space/_v8_probes.py
```

---

## 附：本轮自己的错误与限制

1. **`succ` 采样 bug**（见 §A）—— 已修正并重跑；修正前 Probe B/C 的对照集分别只有 34 / 9，结论不可用。
2. **置换检验因算力关闭** —— 1000 次 IRLS 拟合实测耗尽 14017s CPU 未完成。已改为纯 CV，并在报告中明示；probe A 的置换 p 引用 V7 的 0.0000。
3. **Probe C 的尺寸不匹配** —— 尺寸带内 only 9 个，改用全量 47 个，已在报告标注。
4. **`is_detected` 的相关系数是循环的** —— w_A 按该标签拟合，故 rho=−0.652 不作为证据，已在表中标出。
