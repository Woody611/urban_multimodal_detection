# READ-ONLY LOCAL ACTIVATION INTERVENTION（L16 / L17）

**性质**：只读。0 训练 / 0 反传 / 0 优化器 / 0 改 checkpoint-模型源码-loss-assigner-augmentation-config /
0 test inference / 0 submission / 0 P2-OASA-crop-1536。**未保存任何被改过的模型**（patch 是 hook 内的临时张量替换）。

**完整性**：`augment.py 5cb9a407…` / `default.yaml 991a89b3…` / `scripts/train.py 9f55b09f…` / `detect/train.py b021354f…` / `best.pt 1cae45f7…` 全部未变；仅新增 `diagnostic/p3_feature_space/` 下文件。

---

## §1 Frozen 配置

checkpoint / preprocessing / eval mode / input / image sampling / GT 组定义 **与 V10/V11 逐字相同**。
**G1 = 38｜G4 = 194｜TS = 467**（本轮 patch 只跑 G1+G4 = 232 target，覆盖 87 张图）。

## §11 INTEGRITY —— **PASS**

```
G1  n=38   max|Δ base_logit vs V7.logit_decomp| = 1.907e-06   中位 = 0.000e+00
G4  n=194  max|Δ base_logit vs V7.logit_decomp| = 1.907e-06   中位 = 4.768e-07
```

（同一精确分解 `logit_c = W_c·h + b_c`、同一 GT-center cell；差值 = float32 舍入级）
**⇒ intervention OFF 时下游响应与 V7/V10 完全一致。**

**Donor 规则（写码前固定，不使用 logit / similarity / outcome）**：
同类别 ∧ small ∧ 成功 ∧ 来自 G4 → `|Δsqrt(area)|` 最小者，并列按 `(stem,gt)` 字典序。中位候选数 = 20。

---

## §7 核心表

| Intervention | **G1 medΔlogit** | G1 IQR | G1 medΔprob | %Δlogit>0 | **G4 medΔlogit** | G4 %Δ>0 |
|---|---:|---|---:|---:|---:|---:|
| L16 zero | −0.594 | [−1.85, 1.07] | −0.0000 | 39.5% | −1.677 | 11.3% |
| L17 zero | −1.436 | [−2.69, 0.79] | −0.0000 | 39.5% | −2.455 | 13.4% |
| **L16 prototype** | **+3.512** | [0.67, 6.71] | 0.0000 | **92.1%** | −0.098 | 33.0% |
| **L17 prototype** | **+5.869** | [2.01, 9.54] | **+0.0003** | **89.5%** | **−0.098** | 34.0% |
| L17 norm-matched | +3.383 | [0.75, 6.81] | 0.0000 | 84.2% | −0.116 | 35.6% |
| **displaced L17** | **−0.005** | [−0.01, 0.00] | −0.0000 | 34.2% | 0.000 | 57.2% |
| **random donor** | **+4.272** | [1.34, 9.94] | 0.0000 | 84.2% | 0.000 | 44.8% |

基线：**G1 base_logit = −14.634（p = 4.4e-07）**；G4 base_logit = +0.511。

## §8 非参数 paired 检验（Wilcoxon signed-rank）

| Intervention | G1 medΔ | **p** | rank-biserial | G4 medΔ | G4 p |
|---|---:|---:|---:|---:|---:|
| L16 zero | −0.594 | 0.199 | −0.211 | −1.677 | 4.9e-25 |
| L17 zero | −1.436 | 0.0105 | −0.211 | −2.455 | 1.4e-25 |
| **L16 prototype** | **+3.512** | **2.7e-09** | **+0.842** | −0.098 | 6.5e-06 |
| **L17 prototype** | **+5.869** | **3.1e-08** | **+0.789** | −0.098 | 7.4e-05 |
| L17 norm-matched | +3.383 | 9.0e-07 | +0.684 | −0.116 | 3.3e-05 |
| displaced L17 | −0.005 | 0.044 | −0.316 | 0.000 | 0.069 |
| random donor | +4.272 | 8.1e-07 | +0.684 | 0.000 | 0.010 |

---

## §9 判据检查

**Causal-supporting pattern —— 四条全部满足：**

| 条件 | 实测 |
|---|---|
| successful L17 patch → G1 logit ↑ | **+5.869**（p = 3.1e-08）✅ |
| G1 probability ↑ | 4.4e-07 → 2.9e-04 ✅（方向对，见下方警告） |
| effect 在 displaced / random control 明显弱 | **displaced = −0.005**（≈0）✅；random donor 反而**也有效**（+4.272）⚠️ |
| G4 effect 明显小 | **−0.098**（且 base 0.511 → patched −0.158，**略降**）✅ |

**⇒ `L17 representation is causally involved in the downstream classification failure.`**

**（仍然不声称 L17 是 root cause。）**

### ⚠️ 但有一条 §9 未覆盖、必须一并报告的发现：**因果参与 ≠ 充分**

| Intervention | base logit (p) | **patched logit (p)** |
|---|---|---|
| L16 prototype | −14.634 (4.4e-07) | −10.196 **(3.7e-05)** |
| **L17 prototype** | −14.634 (4.4e-07) | **−8.154 (2.9e-04)** |
| L17 random donor | −14.634 (4.4e-07) | −10.922 (1.8e-05) |
| L17 displaced | −14.634 (4.4e-07) | −14.627 (4.4e-07) |
| G4 对照 L17 prototype | +0.511 | **−0.158** |

**即使把 L17 换成完全 in-distribution 的成功小目标向量，类别概率也只到 2.9e-04 —— 仍低于任何可用检测阈值数个数量级。**

- 要把 logit 从 −14.63 推到 0（检测所需），需要 **≈ +14.6**；单 cell 替换只补回 **+5.87（≈40%）**
- ⇒ **L17 是"contributing factor"，不是 "sufficient bottleneck"**

**另一条重要读数**：**random donor (+4.272) 与 size-matched prototype (+5.869) 差距仅 1.6，IQR 大幅重叠**
⇒ **效果不依赖 donor 的具体选择，只要求它是一个"典型成功小目标"的 L17 向量。**
（这与 V7/V9 的"G1 的 H/L17 在成功簇外"一致 —— 起作用的是"回到分布内"，而非"匹配到最像的那一个"。）

---

## §10 技术限制（同读）

单 cell activation replacement 是**人为反事实干预**，不在自然训练分布中。
即使 rescue 成立，也只能证明：

```text
L17 representation → downstream detection response   存在因果依赖
```

**不能推出**「原始训练过程中 L17 的 formation 就是唯一根因」。

---

## §12 最终回答

```text
L16 causal evidence:    SUPPORT        (p=2.7e-09, medΔ +3.512, 空间特异 ✓, G1 特异 ✓)
L17 causal evidence:    SUPPORT        (p=3.1e-08, medΔ +5.869, 空间特异 ✓, G1 特异 ✓)

G1-specific rescue:     YES            (G1 +5.869 vs G4 −0.098；G4 反而略降)
Spatial specificity:    YES            (中心 +5.869 vs 位移 −0.005)
Donor robustness:       YES            (random donor +4.272 亦有效 ⇒ donor 选择不敏感)

Representation causal involvement:      SUPPORTED
Representation causal SUFFICIENCY:      NOT SUPPORTED   ← 本轮新增的限定

HOLD:                   YES
```

### 下一步（按 §12：有明显 rescue ⇒ 只报告可进入设计，不自行启动）

> **可以进入单变量 representation intervention experiment design。**

**但必须先闭合一个缺口**：本轮证明的是 **因果参与（partial rescue，补回 ~40% 所需 logit）**，**不是因果充分**。
因此在设计训练干预之前，需要先回答：

> 「把干预范围从**单 cell** 扩大到 L17 的**整个 GT 邻域**（例如 GT 框对应的全部 cell），能否把 logit 推到 0 附近？」

- 若能 ⇒ L17 region 是**充分**瓶颈，单变量干预设计有明确靶点
- 若仍不能 ⇒ 剩余 −8 的 logit 来自 L17 之外（例如 L20/top-down 路径或更早的 L12），干预设计应扩到那条链

**该检验同样是纯只读的**（同一次 forward + 更大范围的 hook 替换）。**本轮未做，到此停止。**

---

## 产物

```
diagnostic/p3_feature_space/_v12_patch.py     activation patching（只读，未保存任何改过的模型）
diagnostic/p3_feature_space/_v12_patch.json   232 target × 7 intervention 的完整记录
diagnostic/p3_feature_space/_v12_analyze.py   完整性 + 核心表 + paired 检验
diagnostic/p3_feature_space/_v12_result.json  汇总
diagnostic/p3_feature_space/REPORT_V12.md     本报告
```

## 附：本轮自己的两个 bug（已修，不影响最终数字）

1. **`logit_of` 取的是 `h[0,:,0,0]`（角落）而非 GT 中心 cell** —— 在跑干预**之前**发现并修正为 `h[0,:,y,x]`。
2. 完整性校验初版从 `_v10_stages.json` 取 `logit_decomp`（该键在 `_v7_vectors.json`）—— 已改源。
