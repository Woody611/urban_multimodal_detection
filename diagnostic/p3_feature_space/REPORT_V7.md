# P3 FEATURE-SPACE DIAGNOSTIC

**性质**：只读。0 训练 / 0 改模型-head-loss-assigner-增强 / 0 改核心文件 / 0 碰 test-submission-online。

---

## 0. Data integrity

| 项 | 状态 |
|---|---|
| train/val | 严格分离（1600 / 400） |
| checkpoint | `1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda`（未改动） |
| `augment.py` / `default.yaml` / `train.py` / `detect/train.py` | `5cb9a407…` / `991a89b3…` / `9f55b09f…` / `b021354f…` **全部与本轮开始时一致** |
| 修改核心文件 | ❌ 无（`git status` 的 `M` 全部来自更早轮次，哈希未变） |
| 读 test / submission / online | ❌ 无 |
| 训练 | ❌ 无 |

---

## 1. 抽取方法（**精确分解，非伪公式**）

`Detect.cv3[lvl]` = `[Sequential(DWConv+Conv), Sequential(DWConv+Conv), Conv2d(256, nc, 1)]`（`legacy=False`）。

因此对 P3 层：

```text
class logit  =  W_c · h  +  b_c          ← 精确等式
h = cv3[0][-1] 的输入（256ch @160×160）
W_c = cv3[0][-1].weight[c]  (256,)
```

**⚠️ 不是 `W · (P3 颈特征)`** —— P3 与 logit 之间还有两层 DWConv/Conv，写成 `w·f` 会是漏项的伪公式。本轮**只用上面这条精确等式**。

对每个 GT，在 GT center 对应的 cell 取**完整 256 维向量**（P3）与 **256 维向量**（H）。四组同层、同坐标映射、同采样。

样本：T1（train，无增强）**2153 GT**；V1（val，无增强）**2807 GT**。

组定义：
- **G1** = val hard-miss 38｜**G2** = val 同图同类同尺寸成功 29｜**G4** = val 成功同类 small 194
- **T_success** = **train** small GT 且 `logit_decomp > 0`，**437**（本轮定义，明示）

---

## 2. 向量几何

| group | n | \|P3\| | \|H\| | **cos(P3, μ_G4)** | **cos(H, μ_G4)** | **logit_decomp** |
|---|---:|---:|---:|---:|---:|---:|
| T_success | 437 | 8.466 | 51.279 | 0.6481 | 0.8007 | **+1.655** |
| G4 | 194 | 7.395 | 47.314 | 0.7015 | 0.8189 | +0.511 |
| G2 | 29 | 7.651 | 46.951 | 0.7133 | 0.7301 | −0.047 |
| **G1** | 38 | **6.370** | **46.717** | **0.7476** | **0.6213** | **−14.634** |

**两个关键读数**：

1. **`cos(P3, μ_G4)`：G1 = 0.7476，反而**高于** G4 自身（0.7015）** ⇒ **在 P3 空间，G1 的方向并没有偏离成功簇**（cosine-to-centroid 不支持 B）
2. **`cos(H, μ_G4)`：G1 = 0.6213 vs G4 = 0.8189** ⇒ **偏离出现在 P3 → H 之间（cv3 栈内部）**
3. **`|H|` 几乎相同**（46.72 vs 47.31，cliffΔ −0.224、CI 跨 0）⇒ **不是 norm 崩塌**

---

## 3. kNN（两个空间，k=5/10；L2 归一化后欧氏距离）

| query → ref | P3 k=5 | P3 k=10 | **H k=5** | H k=10 |
|---|---:|---:|---:|---:|
| **G1 → T_success** | 0.7779 | 0.7974 | **0.8025** | 0.8196 |
| **G1 → G4** | 0.7212 | 0.7486 | **0.6788** | 0.7098 |
| G2 → T_success | 0.6945 | 0.7311 | **0.3090** | 0.3259 |
| G2 → G4 | 0.6988 | 0.7433 | **0.3163** | 0.3840 |
| G4 → T_success | 0.7024 | 0.7343 | **0.2760** | 0.3152 |
| T_success → G4 | 0.7405 | 0.7793 | **0.2475** | 0.2797 |

**决定性对比**：
- **P3 空间：所有组彼此距离都是 0.69–0.80，G1 并不突出**（G1→G4 0.7212 ≈ G2→G4 0.6988 ≈ G4→T_success 0.7024）
- **H 空间：G1 是 0.6788，而其他全部组为 0.2475–0.3163 ⇒ G1 是 2.2–2.7 倍的离群点**
- **T_success ↔ G4 极近（0.2475/0.2797）⇒ 训练域与验证域的成功样本在特征空间**高度重合**

---

## 4. PCA（**仅可视化，不作统计证据**）

解释方差比 PC1 = 0.319、PC2 = 0.154（合计 0.472，二维图有意义）。

| group | PC1 中位 | PC1 IQR | PC2 中位 | PC2 IQR |
|---|---:|---|---:|---|
| T_success | 9.65 | [−17.8, 19.6] | −0.68 | [−3.2, 3.3] |
| G4 | 11.21 | [−16.8, 17.2] | −0.26 | [−2.9, 4.5] |
| G2 | −7.70 | [−15.9, 16.1] | 2.09 | [−1.4, 12.0] |
| **G1** | −2.00 | **[−13.2, 4.2]** | 2.88 | [−0.7, 8.4] |

质心两两距离：T_success–G4 **1.365**｜T_success–G1 6.720｜G4–G1 6.085｜G2–G1 2.774

**⇒ 各组的 IQR（PC1 跨度 ~30）远大于质心间距离（1.4–6.7）⇒ PCA 上**严重重叠**，不构成可分性证据。**（唯一的信息是 G1 的 PC1 IQR 明显更窄：跨度 17.4 vs 其他 33–37。）

---

## 5. Linear probe（严格 5-fold 分层 CV + 200 次置换检验）

sklearn 不可用 → **自实现 L2-logistic regression（numpy）+ 分层 CV + 置换检验**。

| comparison | 空间 | balanced acc | ROC-AUC | 置换 p |
|---|---|---:|---:|---:|
| **G1 vs G4** | **P3** | **0.824 ± 0.031** | **0.947 ± 0.023** | **0.0000** |
| **G1 vs G4** | **H**（≈head 自身） | **0.942 ± 0.058** | **0.988 ± 0.013** | **0.0000** |
| G1 vs T_success | P3 | 0.960 ± 0.048 | 0.999 ± 0.003 | 0.0000 |
| G1 vs T_success | H | 0.998 ± 0.005 | 1.000 ± 0.000 | 0.0000 |
| **对照：G1 vs G4 标签打乱** | P3 | **0.574 ± 0.094** | — | — |

**⇒ 打乱对照 0.574（≈chance）⇒ probe 有效。
⇒ P3 单层已能以 balanced acc 0.824 / AUC 0.947 区分 G1 与 G4 ⇒ 判别信息**已经存在于 P3**。
⇒ H 层更高（0.942 / 0.988）⇒ head 是**放大器**，不是**制造者**（从 0.947 → 0.988，幅度有限）。**

**⚠️ 必须同读的限定**：由于 `logit_c = W_c·h + b_c` 是**精确等式**，H 空间的线性 probe 与 head 自身的可分性是**同一件事**，其高分不能单独作为「representation 有差异」的证据。**有判别力的是 P3 空间 probe。**

---

## 6. Head 几何

- 分解为**精确恒等式**，无未解释残差：`logit_c = W_c · h + b_c`
- G1 的 `cos(H, μ_G4)` = 0.6213 vs G4 0.8189（cliffΔ −0.578，CI [−0.255,−0.117]）
- G1 的 `|H|` 与 G4 无显著差异（cliffΔ −0.224，CI [−5.86,+1.03]）
- **⇒ 差异是**方向**差异，不是幅度差异；且该方向差异在 cv3 栈内部被放大成 −14.6 的 logit**

**⇒ head 不是「把正确表征映射错」，而是「对一个已经离簇的 h 忠实输出很低的 logit」。**

---

## 7. 尺度分层（G1 vs G4）

| bin | G1 n | G1 logit | G1 \|H\| | G1 cos(H,μ_G4) | G4 n | G4 logit | G4 \|H\| | G4 cos(H,μ_G4) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| <12 | 6 | **−12.71** | 42.18 | 0.5903 | 19 | −0.381 | 46.44 | 0.6852 |
| 12–18 | 24 | **−14.29** | 47.39 | 0.6270 | 72 | −0.088 | 45.89 | 0.8588 |
| 18–24 | 8 | **−18.21** | 42.45 | 0.6234 | 65 | +0.608 | 47.29 | 0.8720 |
| 24–32 | 0 | — | — | — | 4 | +1.591 | 64.99 | 0.5757 |
| ≥32 | 0 | — | — | — | 34 | +1.457 | 54.11 | 0.6056 |

**⇒ G1 在**每一个** bin 都崩塌（−12.7 ~ −18.2），且 `cos(H,μ_G4)` 在每个 bin 都 ~0.59–0.63（G4 为 0.69–0.87）**
**⇒ 与 V6 相互验证：不是 <12px scale floor。**

---

## 8. 统计（effect size + bootstrap CI 优先）

| 对比 | 量 | G1 | 参照 | Cliff's Δ | ΔCI95 |
|---|---|---:|---:|---:|---|
| G1 vs G4 | \|H\| | 46.717 | 47.314 | −0.224 | **[−5.86, +1.03]（跨 0）** |
| G1 vs G4 | cos(H, μ_G4) | 0.621 | 0.819 | **−0.578** | [−0.255, −0.117] |
| G1 vs G4 | logit_decomp | −14.634 | 0.511 | **−0.977** | [−16.54, −13.88] |
| G1 vs T_success | \|H\| | 46.717 | 51.279 | −0.516 | [−9.61, −2.85] |
| G1 vs T_success | cos(H, μ_G4) | 0.621 | 0.801 | −0.635 | [−0.272, −0.095] |
| G1 vs T_success | logit_decomp | −14.634 | 1.655 | **−1.000** | [−17.47, −15.13] |

**n 警告**：G1 仅 38。上表以 **Cliff's Δ 与 bootstrap CI** 为主，p 值仅作辅助；linear probe 用严格 CV + 置换检验；PCA 不作证据。

---

## 9. Evidence matrix（§8 要求）

| Evidence | Representation B | Head/Boundary C |
|---|---|---|
| P3 norm / \|P3\| | **AGAINST**（6.37 vs 7.40，轻度，无崩塌） | — |
| P3 ratio（V6 的 center/global） | **AGAINST**（轻度 8%） | — |
| **G1↔G4 cosine（P3）** | **AGAINST**（0.748 **>** G4 自身 0.702） | — |
| **G1↔G4 cosine（H）** | **SUPPORT**（0.621 vs 0.819） | NEUTRAL |
| **kNN（P3）** | **NEUTRAL**（0.721 ≈ G2 的 0.699；无分离） | — |
| **kNN（H）** | **SUPPORT**（0.679 vs 0.316，2.2×） | NEUTRAL |
| PCA separation | NEUTRAL（IQR ≫ 质心距） | NEUTRAL |
| **linear probe（P3）** | **SUPPORT**（AUC 0.947，置换 p<0.005） | — |
| linear probe（H） | SUPPORT（AUC 0.988） | **AGAINST**（head 是 h 的精确线性映射，未见额外错映射） |
| **class-weight alignment** | **SUPPORT**（logit = W·h+b 精确，无残差；差异全在 h 的方向） | — |
| **train→val shift** | **AGAINST 作为 G1 的成因**（T_success↔G4 kNN 0.2475，两域成功样本高度重合） | — |

---

## 10. 结论（§10 要求的四问）

### A. Feature-space finding
**G1 在 P3 空间**没有**明显脱离成功簇**（cosine 更高、kNN 无分离）；
但 **P3 单层已含可分信息**（linear probe AUC 0.947，失衡对照 0.574）。
⇒ **部分脱离**：不是「明显离群」，而是**存在系统性但重叠度高的差异**。

### B. Head geometry finding
**不是**「feature 接近成功却与 class direction 不匹配」。
实测：`cos(H, μ_G4)` 0.621 vs 0.819 ⇒ **h 本身已经离簇**；head 的 `W_c·h + b_c` 是对该 h 的**忠实**线性响应，无未解释项。
⇒ **head 是放大器（0.947 → 0.988），不是错映射器。**

### C. Domain finding
**不足以解释 G1。** T_success ↔ G4 在 H 空间的 kNN 距离仅 **0.2475**（成功样本在 train/val 之间高度重合），而 G1 距两者 **0.68–0.80**。
⇒ domain shift 无法产生 G1 这种量级的离群。

### D. Mechanism verdict

```text
Representation (B):          SUPPORTED
Head / decision boundary (C): NOT SUPPORTED
Scale floor:                 NOT SUPPORTED
```

**一句话**：G1 的 P3 表征与成功小目标存在**系统性但部分重叠**的差异（AUC 0.947）；
该差异在 head 的 cv3 栈内被放大，最终以近乎恒等的线性映射输出极低 logit。
**断裂点在表征侧，不在 head 的映射规则侧。**

---

## 11. Next experiment（**只允许一个**）

> **在 P3 空间，把「G1 vs G4」的判别方向 `w` 投影到其他参照集，判断 `w` 究竟是「类别判别方向」还是「尺度/上下文方向」。**

具体：用 §5 的 P3 logistic probe 取出 256 维判别方向 `w`（只读），然后计算 `w·f` 在以下集合上的分布：
1. **同类的 medium/large 成功 GT**（G4 之外的尺度）
2. **其他类的 small 成功 GT**
3. **train 域的成功 small GT（T_success）**

**判据**：
- 若 `w·f` **只**把 G1 与「同类 small 成功」分开，而在其他类/其他尺度上无区分
  ⇒ `w` 是**类别判别方向** ⇒ 下一步应追问「G1 这一实例的局部内容如何在 P3 上被编码成别的类别码」
- 若 `w·f` 同时把「small 成功 vs medium/large 成功」分开（即 `w` 也编码尺度/上下文）
  ⇒ `w` 是**尺度/上下文方向** ⇒ 下一步应追问「G1 的邻域上下文如何在 P3 上被编码」

这是**只读**的（复用本轮已落盘的 54MB 向量 + 一个线性投影），且直接回答「表征差异到底是什么」。

**在此判据落地前：**
```text
HOLD — insufficient evidence
```

**明确不做**：不改 cls loss、不增大 small-object loss、不改 assigner、不上 P2、不改 augmentation。

---

## 产物

```
diagnostic/p3_feature_space/_v7_extract.py    P3/H 完整向量抽取（精确 head 分解）
diagnostic/p3_feature_space/_v7_vectors.json  4960 GT × (256-d P3 + 256-d H)  54MB
diagnostic/p3_feature_space/_v7_analyze.py    几何 / kNN / PCA / 分层 / 统计
diagnostic/p3_feature_space/_v7_addendum.py   P3-kNN + 自实现 linear probe（sklearn 不可用）
diagnostic/p3_feature_space/_v7_run.log       运行日志
diagnostic/p3_feature_space/REPORT_V7.md      本报告
```

**复现**
```bash
python -X utf8 diagnostic/p3_feature_space/_v7_extract.py
python -X utf8 diagnostic/p3_feature_space/_v7_analyze.py
python -X utf8 diagnostic/p3_feature_space/_v7_addendum.py
```

---

## 附：本轮的方法学说明

1. **sklearn 不可用** → linear probe 自实现（L2-logistic regression, numpy, 800 迭代）+ 分层 CV + 200 次置换检验；并给出**标签打乱对照**（0.574）证明 probe 有效。未用近似指标冒充。
2. **H 空间的线性 probe 与 head 自身可分性等价**（因为 logit 就是 h 的线性函数）—— 这一点在报告中已明示，避免把它误读为独立证据。
3. **PCA 只用于可视化**，且明确报告「IQR ≫ 质心距 ⇒ 重叠严重」。
4. T_success 的定义（train small GT 且 `logit_decomp>0`）已明示；train GT 无「成功检测」的真值定义，故采用此操作性定义。
