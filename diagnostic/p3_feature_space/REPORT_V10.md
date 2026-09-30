# cv3 逐层 trajectory：P3 → S1 → S2

**性质**：只读。0 训练 / 0 改模型-loss-assigner-增强 / 0 核心文件改动 / 0 碰 test-submission-online。

**完整性**：`augment.py 5cb9a407…` / `default.yaml 991a89b3…` / `scripts/train.py 9f55b09f…` / `detect/train.py b021354f…` / `best.pt 1cae45f7…` 全部与本轮开始时一致。

---

## 1. Stage 定义（**取自代码，非假设**）

```python
cv3[0][0] = Sequential(DWConv(s1), Conv(s1))   →  S1
cv3[0][1] = Sequential(DWConv(s1), Conv(s1))   →  S2   ==  H（cls Conv 的输入）
cv3[0][2] = Conv2d(256, nc=12, 1)              →  分类头 —— **不计入 representation stage**
```

**⇒ 只有 3 个可观测 stage（P3 / S1 / S2）**；你 §1 列的 4 个中，`S2` 与 `S3(=H)` 在实现上是同一个张量。

采样与 V7 完全一致（GT-center 单 cell、输入空间像素 ÷8、同一次 forward 取三层）。
**交叉验证**：本轮 P3 向量与 V7 的 4960 个向量**逐元素相同**（cos = +1.0000，范数一致）。

---

## 10. 核心表：逐 stage distribution shift（G1 vs train-success small）

| Stage | \|G1\| 中位 | \|TS\| 中位 | cos(v_G1, μ_TS) | kNN5(G1→TS) | **AUC(G1/TS)** | balAcc | **Cliff's Δ** | ΔCI95 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **P3** | 6.37 | 8.38 | **−0.4284** | 1.1696 | **0.9941** | 0.9625 | **1.000** | [+10.92, +14.23] |
| **S1** | 14.52 | 31.46 | **−0.6958** | 1.0735 | **1.0000** | 0.9936 | **1.000** | — |
| **S2 (=H)** | 46.72 | 50.91 | −0.6688 | 1.2094 | **1.0000** | 0.9936 | **1.000** | — |

n：G1 = 38、T_success = 467、G4 = 194。

---

## 7. AUC trajectory（关键对照）

```
G1 vs train-success :  P3 0.9941  →  S1 1.0000  →  S2 1.0000
G1 vs G4            :  P3 0.8706  →  S1 0.9559  →  S2 0.9734
```

**Δ(P3→S1) = +0.0059（G1/TS）｜Δ(S1→S2) = 0.0000**

**⇒ 对「G1 vs train-success」这条跨域对比，P3 已经**饱和**（0.994），cv3 只能再加 +0.006。**
**⇒ cv3 对「G1 vs G4」（val 内部对比）确实有实质放大（+0.085 @S1、+0.018 @S2），但那是放大一个**已经存在**的信号（P3 已达 0.871），不是创造。**

---

## 4. 其它指标

- **cos(v_G1, μ_train-succ)**：−0.4284 (P3) → **−0.6958 (S1)** → −0.6688 (S2)
  ⇒ cv3 确实把 G1 的方向**转得更远**（S1 处最远），但该旋转**几乎不增加可分性**（AUC 只 +0.006）
- **kNN5(G1→train-succ)**：1.1696 → 1.0735 → 1.2094 —— **基本不变**
  ⇒ G1 在**每一层**都同样远离 train 成功样本；cv3 没有把它推远多少
- **范数比 \|G1\|/\|TS\|**：0.76 (P3) → **0.46 (S1)** → 0.92 (S2)
  ⇒ S1 处范数差最大，S2 又收回；但 V7 已确认这不是崩塌（无灾难性 collapse），且 kNN/AUC 表明它不是分离的主因

---

## 11/13. Verdict

```text
FIRST DIVERGENCE:   P3 —— 已经存在（AUC 0.994, Cliff's Δ 1.000, balAcc 0.9625）
AMPLIFICATION STAGE: cv3（S1 为主）—— 但对 G1-vs-train-success 仅 +0.0059
P3 ALREADY SEPARATED: YES

cv3 role: AMPLIFIER（且对跨域对比而言是**极弱的** amplifier）
```

**对应你 §11 的 情况 A**：
```
P3 AUC ≈ 0.99  |  S1 ≈ 1.00  |  S2 ≈ 1.00
→ cv3 主要是 amplification，不是 origin
→ 此时下一步不应该继续研究 cv3 第一层
```

**明确禁止的表述**：**不**声称「S1/S2 是根因」。本轮只确定了 FIRST DIVERGENCE（P3）与 AMPLIFICATION STAGE（cv3，极弱）。

---

## Next action（只允许一个）

按你 §最后一条：**P3 已经强烈分离 ⇒ 停止继续拆 cv3**，回到 **backbone / neck 的 representation formation**。

而 V7 已经给出一条**现成的定位线索**：

| 层 | G1 的 center/global ratio | 对照 |
|---|---:|---:|
| **layer 9（backbone P2, stride 4）** | **1.4201** | G4 = **1.4204**（**完全相同**） |
| layer 23（P3 颈输出, stride 8） | **1.0492** | G4 = 1.1586（明显更低） |

> **⇒ 在 layer 9（stride 4）G1 与成功样本**完全不可分**；到 layer 23（stride 8）已经强烈分离。
> ⇒ divergence 进入的位置在 **layer 9 → layer 23 之间**（backbone 的 10–17 与 neck 的 18–23）。

**唯一的下一步**：

> **用与 V10 完全相同的方法（GT-center 单 cell、同采样、同 stage 指标），沿 layer 9 → 18 → 19 → 20 → 21 → 22 → 23 逐层做 trajectory，定位 G1 与 train-success 的 AUC 从哪一层起离开 ≈0.5。**

判据：
- 若在 **backbone 段（10–17）** 就抬升 ⇒ 与 stride 8 的 backbone 表征形成有关
- 若在 **neck 段（18–23，含 FPN upsample + Concat + C3k2）** 才抬升 ⇒ 与**自顶向下融合 + 小目标在 P3 上的上下文聚合**有关

**这仍是纯只读**（一份 forward + 逐层 hook），且直接回答「P3 为什么对 G1 形成了 train 中不存在的罕见表征」。

```text
HOLD — no training
```

**不做**：cls-loss / assigner / P2 / OASA / crop / 1536 / augmentation sweep。

---

## 附：本轮自己的一个 bug（已定位，不影响结论）

`_v10_stages.py` 打印范数列时写成 `np.linalg.norm(XG)`（**漏了 `axis=1`**），得到的是矩阵的 **Frobenius 范数**而非逐行范数 —— 首版因此显示 `|G1|=40.64 / |TS|=185.62`，而真值是 6.37 / 8.38（√38×7≈43、√467×8.5≈184 恰好吻合，可反证）。
**所有实质指标（cos / kNN / AUC / Cliff's Δ）均使用 `axis=1`，未受影响**；上表范数列已改用 V7 交叉验证过的正确值。

**产物**
```
diagnostic/p3_feature_space/_v10_stages.py   逐层抽取 + 分析（只读）
diagnostic/p3_feature_space/_v10_stages.json 4960 GT × (P3|S1|S2) 完整向量
diagnostic/p3_feature_space/_v10_run.log     运行日志
diagnostic/p3_feature_space/REPORT_V10.md    本报告
```
