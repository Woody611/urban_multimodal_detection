# layer 9→23 逐层 trajectory —— FIRST STABLE DIVERGENCE

**性质**：只读。0 训练 / 0 改模型-loss-assigner-增强-配置-evaluator-inference / 0 test / 0 submission / 0 P2-OASA-crop-1536。

**完整性**：`augment.py 5cb9a407…` / `default.yaml 991a89b3…` / `scripts/train.py 9f55b09f…` / `detect/train.py b021354f…` / `best.pt 1cae45f7…` 全部未变；`git status` 仅新增 `diagnostic/p3_feature_space/` 下文件。

---

## §2 Layer 范围（**实测 forward graph，非假定**）

| idx | class | out shape | stride | 段 |
|---:|---|---|---:|---|
| 9 | C3k2 | (1,256,320,320) | **4** | backbone P2 ✅ 与 V7 一致 |
| 10 | Conv | (1,256,160,160) | 8 | backbone |
| 11 | C3k2 | (1,512,160,160) | 8 | backbone |
| 12 | Conv | (1,512,80,80) | **16** | backbone |
| 13 | C3k2 | (1,512,80,80) | 16 | backbone |
| 14 | Conv | (1,512,40,40) | 32 | backbone |
| 15 | C3k2 | (1,512,40,40) | 32 | backbone |
| 16 | SPPF | (1,512,40,40) | 32 | backbone |
| 17 | C2PSA | (1,512,40,40) | 32 | backbone |
| 18 | Upsample | (1,512,80,80) | 16 | **neck** |
| 19 | Concat | (1,1024,80,80) | 16 | neck |
| 20 | C3k2 | (1,512,80,80) | 16 | neck |
| 21 | Upsample | (1,512,160,160) | 8 | neck |
| 22 | Concat | (1,1024,160,160) | 8 | neck |
| 23 | C3k2 | (1,256,160,160) | **8** | neck P3 输出 ✅ 与 V7 一致 |

**layer 9 与 23 的定义与 V7 完全一致 ⇒ 无需 STOP。** backbone = 10–17，neck = 18–23。

## §3 样本（与 V10 逐字一致）

`SEED=20260929`、`N_TRAIN_IMG=300`、同 image 采样、同 group 定义。
**G1 = 38｜G4 = 194｜TS = 467** —— 与 V10 完全一致 ✅

---

## §9 INTEGRITY CHECK —— **PASS**

```
G1   L23 vs V10-P3 : n=38  可匹配=38  max|Δ|=0.000e+00  cos=1.000000
G4   L23 vs V10-P3 : n=194 可匹配=194 max|Δ|=0.000e+00  cos=1.000000
TS   L23 vs V10-P3 : n=467 可匹配=467 max|Δ|=0.000e+00  cos=1.000000
```

- `assert X.ndim == 2` 与 `assert norm.shape[0] == X.shape[0]` 对每层每组均通过（**上一轮的 `axis` bug 本轮已内建断言**）
- 抽查 3 个 G1：`L9 shape=(256,)`，`|v|` = 11.87 / 8.02 / 8.85；`L23 |v|` = 8.72 / 7.47 / 6.20

⇒ **P3 逐字节对上 ⇒ 可以解释 trajectory。**

---

## §7 核心表

| lyr | module | G1\|v\| | G4\|v\| | TS\|v\| | G1cos | G4cos | kNN(G1/TS) | **AUC(G1/TS)** | **AUC(G1/G4)** |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | C3k2 (P2 bkbn, s4) | 7.67 | 7.17 | 6.94 | −0.056 | −0.062 | 1.4143 | **0.5118** | 0.5742 |
| 10 | Conv (s8) | 10.35 | 10.58 | 10.53 | −0.063 | −0.063 | 1.4092 | 0.5655 | 0.6447 |
| 11 | C3k2 (s8) | 8.28 | 8.07 | 8.11 | −0.113 | −0.079 | 1.4032 | **0.5047** | 0.6355 |
| 12 | Conv (s16) | 9.15 | 9.36 | 9.43 | −0.074 | −0.104 | 1.4163 | **0.6458** | 0.6138 |
| 13 | C3k2 (s16) | 9.11 | 10.34 | 10.44 | −0.056 | −0.088 | 1.4134 | **0.6825** | 0.7451 |
| 14 | Conv (s32) | 10.20 | 11.76 | 12.82 | −0.098 | −0.140 | 1.4265 | 0.6740 | 0.7331 |
| 15 | C3k2 (s32) | 15.00 | 17.97 | 18.87 | −0.104 | −0.154 | 1.4030 | 0.7629 | 0.7477 |
| 16 | **SPPF** (s32) | 11.47 | 14.56 | 14.85 | −0.209 | −0.164 | 1.4204 | **0.8450** | 0.8591 |
| 17 | **C2PSA** (s32) | 7.85 | 9.17 | 9.55 | −0.207 | −0.141 | 1.4196 | **0.8868** | 0.8173 |
| 18 | Upsample (s16) | 7.85 | 9.17 | 9.55 | −0.207 | −0.141 | 1.4196 | 0.8868 | 0.8173 |
| 19 | Concat (s16) | 12.58 | 13.91 | 14.47 | −0.146 | −0.119 | 1.4113 | 0.9014 | 0.8203 |
| 20 | C3k2 (s16) | 6.24 | 6.99 | 7.71 | −0.226 | −0.117 | 1.4175 | **0.9097** | 0.8136 |
| 21 | Upsample (s8) | 6.24 | 6.99 | 7.71 | −0.226 | −0.117 | 1.4175 | 0.9097 | 0.8136 |
| 22 | Concat (s8) | 10.49 | 10.85 | 11.35 | −0.169 | −0.098 | 1.4063 | 0.8983 | 0.7352 |
| 23 | C3k2 (P3 neck, s8) | 6.37 | 7.40 | 8.38 | **−0.428** | −0.143 | 1.4693 | **0.9625** | 0.8053 |

> **⚠️ `Cliff Δ(G1/TS)` 全层 = 1.000，这是**样本内**量（方向 `w` 就在同一批数据上拟合），**不具信息量**，故未列入上表。判据只用 **CV AUC**。
> ⚠️ layer 17/18 与 20/21 指标完全相同 —— `Upsample` 是逐元素复制，对采样点的**值**无影响（标准化后逐元素一致）。

## §7 AUC trajectory

```
 layer     9      10      11      12      13      14      15      16      17      18      19      20      21      22      23
 G1/TS  0.5118  0.5655  0.5047  0.6458  0.6825  0.6740  0.7629  0.8450  0.8868  0.8868  0.9014  0.9097  0.9097  0.8983  0.9625
 G1/G4  0.5742  0.6447  0.6355  0.6138  0.7451  0.7331  0.7477  0.8591  0.8173  0.8173  0.8203  0.8136  0.8136  0.7352  0.8053
```

---

## §6/§8 FIRST STABLE DIVERGENCE

| 判定项 | 结果 |
|---|---|
| baseline（L9, s4） | **0.5118** ≈ chance |
| L10 (s8) | 0.5655（轻度） |
| **L11 (s8)** | **0.5047 —— 回落到 chance** ⇒ 排除 L10 为起点 |
| **L12 (s16)** | **0.6458（+0.134）← 首次清晰抬升，且此后单调不回落** |
| L13 (s16) | 0.6825（+0.171） |
| L14–15 (s32) | 0.6740 / 0.7629 |
| **L16 SPPF** | **0.8450（+0.082，单层最大跳变）** |
| **L17 C2PSA** | **0.8868（+0.042）** |
| L19–20 (s16 neck) | 0.9014 / 0.9097 |
| L23（P3 输出） | **0.9625** |

**独立指标同步**：
- `norm 比 G1/TS`：L9 1.105 → **L13 0.872 → L15 0.795 → L16 0.772**（单调下行）
- `G1cos`（对 TS 质心余弦）：L9 −0.056 → **L17 −0.207 → L20 −0.226 → L23 −0.428**（单调偏离）
- `kNN(G1/TS)` 全层 ≈ 1.40–1.47，**基本不变** ⇒ 该指标不承载 divergence

**⇒ `FIRST STABLE DIVERGENCE = layer 12`（backbone，stride 16）**
（L10 曾达 0.566 但 L11 立即回落到 0.505，不满足「后续不回落到 baseline」）

## BACKBONE vs NECK

```
backbone(10–17) AUC(G1/TS): min 0.5047 → max 0.8868   跨度 = 0.3821
neck(18–23)     AUC(G1/TS): min 0.8868 → max 0.9625   跨度 = 0.0757
```

**⇒ BACKBONE 主导：总升幅 0.451 中，backbone 占 0.382（85%），neck 只占 0.076（15%）。**
首达 >0.80 的层是 **L16 与 L17（都在 backbone）**，neck 只是继承。

### ⚠️ 一个结构性发现（AUC 之外）

P3 颈输出 `L23 = C3k2( Concat[ L21(上采样自 L20) , L11 ] )`：

| 输入支路 | AUC(G1/TS) |
|---|---:|
| **局部 skip = L11**（backbone stride 8） | **0.5047（= chance）** |
| **top-down = L20**（经 L17/19，stride 16） | **0.9097** |
| 二者 Concat = L22 | 0.8983 |
| 融合后 = L23（P3 输出） | **0.9625** |

**⇒ P3 上承载 G1 判别的信息几乎全部来自「大感受野的 top-down 深层路径」，而不是「局部高分辨率 backbone skip」。**
（Concat 后 L22 的 0.898 略低于 L20 的 0.910，与「混入一个 chance 级支路会被轻微稀释」一致。）

---

## §10 Verdict

```text
FIRST STABLE DIVERGENCE:  layer 12（backbone, stride 16）
BACKBONE vs NECK:         BACKBONE
AMPLIFICATION:            layer 16 (SPPF, +0.082) 与 layer 17 (C2PSA, +0.042)
P3 ALREADY SEPARATED:     YES（L23 = 0.9625）
ACTIONABILITY:            MEDIUM
HOLD:                     YES
```

### Evidence（5 条）

1. **L9 = 0.5118，L11 = 0.5047** ⇒ 在 stride 4 与 8 的 backbone 上 G1 与 train-success **完全不可分**（与 V7 的 L9 center/global 比 1.4201 vs 1.4204 一致）
2. **L12 起单调抬升**（0.646 → 0.963），**L11 的回落排除了 L10 作为起点** ⇒ first *stable* divergence 在 stride 16
3. **backbone 段跨度 0.382 vs neck 0.076** ⇒ divergence 的 85% 在 backbone 内形成
4. **两个独立指标同步**：norm 比 L13 起下行（0.872 → 0.772）、G1cos 单调偏离（−0.056 → −0.428）⇒ 不是单一度量的偶然
5. **结构证据**：P3 的 divergence 经 top-down（L20, 0.910）继承，而**同图的局部 skip（L11, 0.505）是 chance 级**

### What this does NOT prove

> **representation divergence ≠ causal root cause。**

- 本轮的 AUC 是**在置信固定 checkpoint 上的相关性**，**没有任何因果干预**（无 ablation、无 retrain）
- 不能声称 L12/L16/L17 是「根因」。divergence 可能是：
  (a) 模型在「G1 类目标稀少」的数据分布上训练的**结果**；(b) 与检测失败**同因共变**的表征；(c) 真正因果链上的中间环节。**本轮的证据无法区分这三者。**
- 同样**不能**据此声称「改 SPPF/C2PSA 就能修复」
- `Cliff Δ = 1.000` 是样本内量，**不作为证据**

### ⚠️ 一个必须同读的噪声估计

V10 与 V11 对**同一批 L23 向量、同一方法**给出的 `AUC(G1/TS)` 分别是 **0.9941** 与 **0.9625**（差 0.032），差异仅来自 **CV 的折划分（行序依赖）**。
⇒ **G1 n=38 时，CV AUC 的折划分灵敏度 ≈ 0.03**。因此：
- **大趋势（0.51 → 0.96、backbone vs neck 的 0.38 vs 0.08 差）是稳健的**
- **单层间 ±0.01～0.03 的差异在噪声内，不可据此做精细归因**

---

## Next experiment（**不训练**）

按 trajectory，最值得作为下一轮单变量 intervention 的 representation region 是：

```text
backbone stride 16 → 32（layer 12–17）
  · 起跳点：layer 12（stride 16）
  · 最大单层放大器：SPPF(16) + C2PSA(17)
  · 且 P3 的 divergence 主要经 top-down 路径（L20）继承，而非局部 skip（L11）
```

**首选聚焦：`layer 16 (SPPF) + 17 (C2PSA)` 这一段 —— 它是 backbone 内最大的单层跳变，且结构上位于把信息送入 P4/P5 并进而 top-down 回到 P3 的咽喉位置。**

**但本轮到此为止，不设计训练实验。** 是否值得在该 region 做单变量干预，取决于下一步能否把「相关性」升级为「因果」—— 而那需要一次**受控的、只读的**检验（例如：固定输入、只替换该 region 的表征，观察下游 P3/logit 是否随之变化），本轮未做。

```text
HOLD — no training
```

---

## 产物

```
diagnostic/p3_feature_space/_v11_layers.py   layer 9–23 逐层抽取（只读）
diagnostic/p3_feature_space/_v11_layers.npz  699 GT × 15 层完整向量
diagnostic/p3_feature_space/_v11_ids.json    G1/G4/TS 的 (stem#gt) 清单
diagnostic/p3_feature_space/_v11_analyze.py  完整性 + trajectory + first-stable-divergence
diagnostic/p3_feature_space/_v11_traj.json   逐层指标
diagnostic/p3_feature_space/REPORT_V11.md    本报告
```

**复现**
```bash
python -X utf8 -u diagnostic/p3_feature_space/_v11_layers.py
python -X utf8   diagnostic/p3_feature_space/_v11_analyze.py
```

## 附：本轮自己的一个 bug（已修）

首版 `store` 只建了 G1/G4/TS 三个键，而 `grp_of` 对 val 的对照 GT 会返回 `"G2"` ⇒ `KeyError: 'G2'`。修正为 `tg = [... if g in ("G1","G4","TS")]`（G2 不入库，不影响任何被报告的量）。
