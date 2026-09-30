# T1 / T2 / V1 三域对照 —— 只读机制诊断

**性质**：只读。0 训练 / 0 改模型-assigner-loss-增强 / 0 改核心文件 / 0 碰 test-submission-online。

---

## 1. Data integrity

| 项 | 状态 |
|---|---|
| train/val 严格分离 | ✅ train 1600 / val 400，互斥（`_split_train_val_rgbid`，val_ratio=0.2） |
| checkpoint hash | `1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda`（未改动） |
| `ultralytics/data/augment.py` | `5cb9a407625921ce3f12f…` ✅ 一致 |
| `ultralytics/cfg/default.yaml` | `991a89b32de769d1d444…` ✅ 一致 |
| `scripts/train.py` | `9f55b09f9e1229b61fef…` ✅ 一致 |
| `models/yolo/detect/train.py` | `b021354f541bbb3c7770…` ✅ 一致（该改动来自更早的 resume-guard 轮次，本轮未动） |
| 修改任何核心文件 | ❌ 无 |
| 读取 test / submission / online | ❌ 无（路径全部为 `data/processed/rgbid_split_train/`） |
| 训练 | ❌ 无 |

**设计**：三域共用**同一个** `measure()` 函数 ⇒ 指标口径逐字一致。
T1 与 V1 用完全相同的预处理（LetterBox+Format，无增强），只换 split ⇒ 分离 **domain**；
T1 与 T2 用同一 split，只换 augmentation ⇒ 分离 **augmentation**。

---

## 2. T1 result — train 原图，augmentation OFF

| bin（输入空间 √area） | n | median logit | median prob | median P3 ratio |
|---|---:|---:|---:|---:|
| <12px | 17 | **+0.90** | 0.710 | 1.169 |
| 12–18px | 111 | **+1.32** | 0.789 | 1.217 |
| 18–24px | 136 | **+1.58** | 0.829 | 1.251 |
| 24–32px | 200 | **+1.91** | 0.871 | 1.303 |
| ≥32px（非 small，sanity control） | 1462 | +2.39 | 0.916 | 1.204 |

**⇒ T1 全部为正、随尺度单调。无低 logit 区。**

---

## 3. T2 result — train 真实训练 pipeline（mosaic ON）

| bin | n | median logit | median prob | median P3 ratio |
|---|---:|---:|---:|---:|
| <12px | 128 | **+0.59** | 0.643 | 1.177 |
| 12–18px | 250 | **+1.35** | 0.795 | 1.224 |
| 18–24px | 320 | **+1.69** | 0.844 | 1.273 |
| 24–32px | 393 | **+1.90** | 0.870 | 1.315 |
| ≥32px | 2102 | +2.31 | 0.910 | 1.184 |

**⇒ T2 ≈ T1，所有 bin 相同；augmentation 未造成退化**（<12px 略降 0.90→0.59，n 从 17→128）。

---

## 4. V1 result — val 原图，augmentation OFF

| bin | **G1 missed** (n / logit / prob / P3ratio) | **G2/G4 success** (n / logit / prob / P3ratio) |
|---|---|---|
| <12px | **n=6  logit −11.65  prob 0.000  P3 1.031** | n=23  logit +0.50  prob 0.623  P3 1.201 |
| 12–18px | **n=24  logit −12.10  prob 0.000  P3 1.040** | n=88  logit +0.71  prob 0.671  P3 1.135 |
| 18–24px | **n=8  logit −14.12  prob 0.000  P3 1.081** | n=74  logit +1.31  prob 0.788  P3 1.180 |
| 24–32px | n=0 | n=4  logit +1.69  prob 0.844  P3 1.235 |
| ≥32px | n=0 | n=34  logit +1.83  prob 0.862  P3 1.240 |

**V1 全体（val，无增强）**：

| bin | n | median logit | median prob |
|---|---:|---:|---:|
| <12px | 57 | **−3.56** | 0.028 |
| 12–18px | 157 | **−0.27** | 0.432 |
| 18–24px | 226 | +1.07 | 0.745 |
| 24–32px | 268 | +1.30 | 0.786 |
| ≥32px | 2099 | +1.97 | 0.878 |

**⇒ G1 在**每一个** bin 都崩溃（−11.6 ~ −14.1），而**同一 bin 的 G2/G4 正常**（+0.5 ~ +1.3）。**

---

## 5. T1 → T2 → V1 chain

小目标（<32px）汇总：

| domain / group | n | P3ratio | center_logit | center_prob | n_pos | best_align | pos_target |
|---|---:|---:|---:|---:|---:|---:|---:|
| **T1** train no-aug | 464 | 1.266 | **+1.65** | **0.838** | 10.0 | 0.573 | 0.934 |
| **T2** train +aug | 1091 | 1.268 | **+1.65** | **0.839** | 9.0 | 0.571 | 0.933 |
| **V1** val all | 708 | 1.163 | **+0.77** | **0.683** | 9.0 | 0.030 | 0.736 |
| **V1 G1 missed** | 38 | 1.049 | **−12.58** | **0.0000** | 5.0 | 0.000 | 0.628 |
| V1 G2 ctrl | 29 | 1.150 | +0.83 | 0.696 | 5.0 | 0.057 | 0.729 |
| V1 G4 success | 160 | 1.169 | +0.97 | 0.725 | 6.0 | 0.091 | 0.755 |

```
feature (P3ratio):  T1 1.266 → T2 1.268 → V1全部 1.163 → V1-G1 1.049    （轻度退化）
assignment (n_pos): T1 10    → T2 9     → V1全部 9     → V1-G1 5        （轻度）
class logit:        T1 +1.65 → T2 +1.65 → V1全部 +0.77 → V1-G1 −12.58   （崩塌）
probability:        T1 0.838 → T2 0.839 → V1全部 0.683 → V1-G1 0.0000   （崩塌）
```

**⇒ 崩塌只发生在 logit / probability 这一级；特征与 assignment 只是轻度退化。**

---

## 6. Statistical evidence

| comparison | metric | nA | nB | medA | medB | Cliff's Δ | ΔCI95 | p | Bonf | FDR |
|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| **T1 vs V1（domain）** | center_logit | 464 | 708 | 1.646 | 0.769 | **+0.486** | [+0.69,+1.23] | 4.8e-45 | **PASS** | **PASS** |
| **T1 vs V1（domain）** | center_prob | 464 | 708 | 0.838 | 0.683 | +0.486 | [+0.12,+0.23] | 4.8e-45 | **PASS** | **PASS** |
| **T1 vs T2（augment）** | center_logit | 464 | 1091 | 1.646 | 1.647 | **+0.058** | **[−0.06,+0.08]** | 7.1e-02 | — | (FDR边界) |
| T1 vs T2（augment） | center_prob | 464 | 1091 | 0.838 | 0.839 | +0.058 | [−0.01,+0.01] | 7.1e-02 | — | — |
| **V1 G1 vs G2/G4** | center_logit | 38 | 189 | −12.58 | 0.924 | **−0.991** | [−14.43,−11.84] | 5.9e-22 | **PASS** | **PASS** |
| V1 G1 vs G2/G4 | center_prob | 38 | 189 | 0.000 | 0.716 | −0.991 | [−0.75,−0.69] | 5.9e-22 | **PASS** | — |
| T1 vs V1-G1 | center_logit | 464 | 38 | 1.646 | −12.58 | **+1.000** | [+12.60,+15.18] | 1.1e-24 | **PASS** | — |

`m=7`，Bonferroni α=0.00714 ⇒ **5 项通过**；BH-FDR(0.05) ⇒ **5 项通过**。

**关键读数**：
- **T1 vs T2 的 cliffΔ = 0.058、CI 跨 0、p=0.071 ⇒ augmentation 效应不显著**
- **T1 vs V1 的 cliffΔ = 0.486、p=4.8e-45 ⇒ domain 效应显著且大**
- **V1 G1 vs G2/G4 的 cliffΔ = −0.991 ⇒ 近乎完全分离**
- **T1 vs V1-G1 的 cliffΔ = +1.000 ⇒ 完全分离**

---

## 7. Mechanism verdict

按你 §判定逻辑 的四类：

| Case | 条件 | 本轮实测 | 判定 |
|---|---|---|---|
| **A** train→val 泛化 | T1 正常 + T2 正常 + V1 G1 极低 | **T1 +1.65 / T2 +1.65 / V1-G1 −12.58** | **SUPPORTED** |
| B training objective 不足 | T1 本身低 | T1 全部为正（+0.90 ~ +2.39），P3ratio 1.17–1.30 | **NOT SUPPORTED** |
| C augmentation 退化 | T1 正常但 T2 明显下降 | T1 ≈ T2（cliffΔ 0.058, p=0.071） | **NOT SUPPORTED** |
| D 特定 scale floor | 只有 <12px 崩溃 | G1 在 **<12 / 12–18 / 18–24 全部**崩溃 | **NOT SUPPORTED**（作为主因） |

```text
train→val 局部表示/决策边界泛化（Case A）：SUPPORTED
training objective / head 学习不足（Case B）：NOT SUPPORTED
augmentation 导致表征退化（Case C）：NOT SUPPORTED
specific scale floor（Case D）：NOT SUPPORTED 作为主因
```

### ⚠️ 一个必须写清的限定

V1 的 **G2/G4 是 val 图、与 G1 同 bin、响应正常**（+0.5 ~ +1.3）。
⇒ 这**不是**「整个 val 域坏掉」，而是**val 域内部特定实例**的崩塌。
同时 V1 全体相对 T1 有一致的、显著的 domain 抬升（+1.65 → +0.77, p=4.8e-45）——
即 **两个效应叠加**：
1. **一个温和的 train→val domain gap**（影响所有 small GT，logit −0.9 量级）
2. **一个剧烈的实例级崩塌**（G1，logit −13 量级）

**不能用单一 domain-shift 解释 G1。**

---

## 8. Next action

**唯一最值得做的下一步诊断**（只读，不训练）：

> **判别 B 与 C：G1 的 P3 GT-center 特征向量，是否落在「同类成功样本」的特征簇内？**

具体：在同一 checkpoint、同一 forward 下，取 P3 (layer 23) 在 GT 中心的**特征向量**（不只是 norm），
对 G1 / G2 / G4 分别计算：
- 与**同类** G4 成功样本质心的余弦相似度
- 在「同类成功样本」特征空间中的 k-NN 距离（k=5）
- 同理对 **train 域**的成功 small 样本做同样计算，看 G1 更接近哪一侧

**判据**：
- 若 G1 的特征**落在同类成功簇内**（相似度高 / k-NN 近），而 head 仍输出 −12.6
  ⇒ **决策边界 / head 问题（C 侧）**：表征是对的，head 的决策与表征不一致
- 若 G1 的特征**显著偏离同类成功簇**
  ⇒ **表征漂移（B 侧）**：特征本身就与成功样本不同

这是区分 B 与 C 的直接判据，且**完全只读**（一次 forward + 向量统计）。

**在没有这个判据之前**：
```text
HOLD — insufficient evidence
```

**明确不做**：不改 cls loss、不增大 small-object loss、不改 assigner、不上 P2、不改 augmentation —— 本轮证据不支持其中任何一项。

---

## 附：本轮观察到的两个次要现象（不作为下一步依据）

1. **`best_align` 在 val 上整体极低**（V1 全体 0.030 vs T1 0.573）。这是模型自身输出（`align = s^α·u^β`）在 val 上系统性偏低的表现，与 domain gap 一致。**注意这是用训练好的模型在 val 上算的，不是训练时的监督强度**（后者已由 V5 测为 0.568）。
2. **P3 ratio 有轻度但一致的梯度**：T1/T2 1.266 → V1 1.163 → V1-G1 1.049。特征**没有消失**，只是变弱 —— 与 V4 的结论一致。

---

## 产物

```
diagnostic/small_object_domains/_v6_domains.py    三域测量（同一 measure()）
diagnostic/small_object_domains/_v6_domains.json  7926 条 GT 记录
diagnostic/small_object_domains/_v6_analyze.py    分层 + 统计
diagnostic/small_object_domains/_v6_run.log       运行日志
diagnostic/small_object_domains/REPORT_V6.md      本报告
```
```bash
python -X utf8 diagnostic/small_object_domains/_v6_domains.py
python -X utf8 diagnostic/small_object_domains/_v6_analyze.py
```
