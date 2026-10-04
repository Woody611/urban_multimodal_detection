# P7 Zero-GPU Feature-Manifold Audit

**日期** 2026-10-03 · **性质** ZERO-GPU / ZERO-TRAINING / ZERO-FORWARD / ZERO-INFERENCE · 未补任何 feature dump

## Artifact availability

```text
h available?                     YES —— 30 个 EMA snapshot 的 vec_h，(1320, 256)
                                 source = Detect.cv3[0][-1] 的 **forward 前置输入**，
                                 即分类分支末端 1×1 conv 的输入；stride 8；float32；
                                 与 P5/P5.1/P5.2 用的是**完全同一个 tensor**
prototype computable?            YES（但只有 10/12 类在健康参照组里 n≥5；tricycle/uav 样本不足）
nearest-neighbor computable?     YES（1320² pairwise，纯 CPU）
temporal feature trajectory?     YES（30 个 epoch）
background / all-cell feature?   NO —— **NOT MEASURABLE FROM EXISTING ARTIFACTS**
```

---

## 1. 第一层：prototype 几何

原型只用 **C3（alive∧MATCHED，即健康参照）** 构造，避免把 outcome 泄漏进原型；
所有 cosine 均在 **L2-normalized** 特征上计算。

| cohort | n | **cos(h, μ_GT)** | max-other | **M_proto** | NN_same | NN_other | **M_NN** | W·h (z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **dead∧F0** | 147 | **0.5349** | 0.5032 | **+0.0058** | 0.8713 | 0.8549 | **+0.0137** | −16.04 |
| dead∧MATCHED | 246 | 0.6013 | 0.3811 | **+0.2183** | 0.9663 | 0.8495 | +0.1008 | −14.66 |
| **alive∧MATCHED** | 807 | **0.9377** | 0.5184 | **+0.4156** | 0.9879 | 0.8266 | +0.1571 | +1.71 |

**feature 空间的正确率：**

| cohort | M_proto>0 | M_NN>0 | **argmax(h,μ)==GT** |
|---|---:|---:|---:|
| **dead∧F0** | **51.0%** | 60.5% | **51.0%** |
| dead∧MATCHED | 93.5% | 95.9% | 93.5% |
| alive∧MATCHED | **99.6%** | 99.0% | **99.6%** |

**⇒ `dead∧F0` 的 `cos(h, μ_GT)` 只有 0.535（健康组 0.938），且【近一半更接近别的类原型】。**

---

## 2. §11 classifier-space vs feature-space 交叉表（决定性）

第 2 列 = `feat✓ cls✗`（brief §11 定义的"最强 misalignment 证据"）。

| cohort | n | feat✓cls✓ | **feat✓cls✗** | **feat✗cls✓** | feat✗cls✗ |
|---|---:|---:|---:|---:|---:|
| **dead∧F0** | 147 | 45.6% | **5.4%** | **4.8%** | **44.2%** |
| dead∧MATCHED | 246 | 85.4% | 8.1% | 0.4% | 6.1% |
| alive∧MATCHED | 807 | 99.5% | 0.1% | 0.1% | 0.2% |

**⇒ `feat✓cls✗`（5.4%）≈ `feat✗cls✓`（4.8%）—— 对称，无系统性 disagreement。**
主导格是 **两者同时错（44.2%）** ⇒ 两个空间**一致地失败**，不是互相矛盾。

**brief §19 Gate 2（classifier-space 与 feature-space 系统性 disagreement）⇒ FAIL。**

---

## 3. §10 classifier 方向 vs feature 流形

| cls | **cos(W_c, μ_c)** | max_{j≠c} cos(W_c, μ_j) |
|---|---:|---:|
| 0 person | **+0.2374** | −0.0180 |
| 6 car | +0.2509 | −0.0415 |
| 8 light | +0.2342 | −0.0713 |
| （10 类全部） | **+0.215 ~ +0.259** | **−0.018 ~ −0.076** |

**⇒ classifier 的每一行都【正确指向】自己的 class prototype**（正余弦、且对其它类均为负）。
**⇒ "classifier 本身转离开了 class manifold" 不成立。**

---

## 4. §13 时间维度（全体 GT，无 outcome 选择）

| ep | M_proto 中位 | >0 占比 | cos(h,μ_GT) 中位 | p10 | p90 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.1760 | 85.0% | 0.7417 | −0.046 | 0.272 |
| 10 | 0.2228 | 92.4% | 0.8679 | 0.045 | 0.360 |
| 150 | **0.3228** | **93.7%** | 0.8426 | 0.070 | 0.413 |
| 250 | 0.2988 | 92.3% | 0.8126 | 0.036 | 0.413 |
| 290 | 0.2873 | 91.7% | 0.8036 | 0.036 | 0.403 |

**⇒ 全局 feature manifold 没有退化 —— 从 85.0% 改善到 92~94%，ep150 后轻微回落 11%。**
**`dead∧F0` 不是"流形塌了"，而是这个健康流形的【下尾】**（其 M_proto 中位 +0.006，落在全体 p10 附近）。

---

## 5. Mechanism verdict

```text
C = MIXED / UNIDENTIFIED
```

**构成分解（dead∧F0, n=147）：**

| 子群 | 占比 | 机制 |
|---|---:|---|
| 两个空间**同时判错** | **44.2%** | **B（feature 落在错误流形）** |
| feature 错、classifier 对 | 4.8% | B（同上，classifier 比 feature 更准） |
| **两个空间都对**（但绝对分数低） | **45.6%** | **既非 A 也非 B —— 是 P5 已刻画的 common-mode vacuum** |
| feature 对、classifier 错 | 5.4% | A（classifier–feature misalignment） |

- **A 被排除**：Gate 1（feature 明确在正确流形上）FAIL —— `dead∧F0` 的 M_proto 中位仅 +0.006、49% 更接近别的类；
  Gate 2 FAIL —— 无系统性 disagreement；§10 显示 classifier 方向本身正确。
- **B 在"失败子集"里占 9:1**（49.0% vs 5.4%），但**只覆盖一半**。
- **剩余 45.6% 不属于 A/B 任何一类**，是 P5/P6 已确立的 common-mode vacuum（centered logit 仍为正、GT 类仍是 argmax，但绝对电平低）。

**⇒ 没有单一机制覆盖 dead∧F0。** 按 brief §15 的判定：**C**。

---

## 6. §16-A1：现有监督是否已含等价的 feature-class alignment 信号？

`v8DetectionLoss` 的三项 = BCE（对 TAL 的质量加权软 target）+ CIoU/DFL（框）。
TAL target = `CIoU^6 · cls^0.5 · CIoU` 归一化 —— 那是**分数目标**，不是 **feature 流形目标**。
**没有任何 feature-space / prototype / contrastive / metric-learning 项。**
（但这不影响判定：Gate 1 已 FAIL。）

---

## 7. §17 文献映射

| 论文机制 | 与本轮结果的关系 |
|---|---|
| ① Double-Head / Rethinking Cls & Loc | 本轮**未**发现"分类与定位需要不同空间特征"的 D′ 证据；box 正确而 score 死是**已有**事实（G1），不是新证据 |
| ② TSCODE / ③ Decoupled DETR | 与"feature 侧失败占主导"**方向一致**，但**没有任何 D′ 证据**表明加任务专属分支能改变它；且属架构级改动 |
| ④ DenseCL / DetCo | D′ **已经自然形成**健康的 dense manifold（93.7% 正确），**不支持**"需要额外 dense contrastive 预训练" |
| ⑤ TOOD | 仅作 baseline，**未重开** |
| ⑥ C2AM / cosine / margin | **CLOSED**（P6 已证明单调性上不可能修好），**未重开** |

> **literature mechanism ≠ D′-specific evidence。** 上述映射无一条构成 D′ 侧证据。

---

## 8. GPU Gate（§19）

| # | 条件 | |
|---|---|---|
| **1** | **dead∧F0 的 feature 明确仍位于正确 class manifold** | ❌ **FAIL**（M_proto 中位 +0.006，49% 更接近别的类） |
| **2** | **classifier-space 与 feature-space 系统性 disagreement** | ❌ **FAIL**（5.4% ≈ 4.8%，对称） |
| 3 | 该 disagreement 与 dead/F0 phenotype 相关 | n/a（#2 不成立） |
| 4 | 现有监督未提供等价机制 | ✅ |
| 5 | 存在最小可证伪 intervention | ⚠ |
| 6 | 能说明为何改变 official AP50-95 | ❌ |

```text
GPU NO-GO
```

---

## 9. 最终输出

```text
P7 Feature-Manifold Audit
=========================

Artifact availability:
    h = YES (Detect.cv3[0][-1] 前置输入, 256d, stride 8, 与 P5 同一 tensor)
    prototype = YES (10/12 类; 原型取自健康参照 C3)
    NN = YES
    temporal = YES (30 epochs)
    background/all-cell feature = NOT MEASURABLE FROM EXISTING ARTIFACTS

Feature verdict:
    C = MIXED / UNIDENTIFIED
    （A 被明确排除；B 在失败子集占 9:1 但只覆盖 49%；余 46% 是 P5 已知的 common-mode vacuum）

Evidence:
    strongest positive evidence =
        1. dead∧F0 的 cos(h, μ_GT) = 0.535 vs 健康组 0.938；49% 更接近别的类原型；
           M_proto 中位仅 +0.006（落在全体 p10）⇒ **feature 侧确实劣化**
        2. 全局流形健康且在改善（85.0% → 93.7% @ep150 → 91.7% @ep290）⇒ dead∧F0 是**下尾**而非流形塌缩
    strongest negative evidence =
        1. **§11 交叉表对称**（feat✓cls✗ 5.4% ≈ feat✗cls✓ 4.8%）⇒ 无 classifier–feature misalignment
        2. **§10：cos(W_c, μ_c) = +0.21~+0.26 且对其它类均为负** ⇒ classifier 方向正确
        3. 45.6% 的 dead∧F0 在**两个空间都判对**，只是绝对分数低

Classifier-feature disagreement:
    = 5.4% (feat✓cls✗) vs 4.8% (feat✗cls✓) —— **对称，无系统性**

Relation to P5.2:
    = 一致且互补。P5.2 说"W 旋转 + bias 冻结，upstream 未识别"；
      本轮进一步说明：**W 的方向是对的**（指向自己的流形），
      失败集中在 feature 的下尾 + 一个不属 A/B 的 common-mode 分量。

Relation to existing TAL/BCE:
    = TAL 提供的是**分数目标**（CIoU^6·cls^0.5），不是 feature 流形目标；
      但本轮 Gate 1/2 双 FAIL，所以"缺 feature 对齐损失"**不构成可行动依据**。

Literature mechanism relevance:
    = ②③（任务专属特征）方向一致但**零 D′ 证据**；
      ④ 被 D′ 自身健康的 dense manifold 反证；
      ⑤⑥ 未重开。literature relevance ≠ D′ mechanism evidence。

GPU candidate:
    NO

If NO:
    precise closure reason =
      (1) brief §19 Gate 1 FAIL —— dead∧F0 的 feature **不**明确位于正确流形
          （M_proto 中位 +0.006、49% 更接近别的类），所以【不能】把它提升为
          "classifier–feature misalignment"；
      (2) brief §19 Gate 2 FAIL —— feature-space 与 classifier-space **无系统性 disagreement**
          （5.4% ≈ 4.8%，主导格是两者同时错 44.2%）；
      (3) §10 显示 classifier 方向本身正确（cos(W_c,μ_c) 全正、对其它类全负），
          ⇒ "classifier 转离了流形"不成立；
      (4) 余下 45.6% 属 P5/P6 已刻画的 common-mode vacuum，**不是新机制**。
```

**本轮没有把 P5.2 的几何异常升级为新的可证伪机制 —— 相反，它排除了最接近"新机制"的那一个（classifier–feature misalignment），并把失败定位到 feature 的下尾 + 一个已知的 common-mode 分量。**
