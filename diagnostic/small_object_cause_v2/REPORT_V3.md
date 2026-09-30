# SMALL_OBJECT_CAUSE_ATTRIBUTION_V3

**性质**：只读。0 训练 / 0 新推理 / 0 核心代码修改 / 0 evaluator 修改 / 0 提交。
**新文件**：`diagnostic/small_object_cause_v2/{_v3_analysis.py,_v3_matrix_modality.py,_v3_analysis.json,_v3_matrix_modality.json,REPORT_V3.md}`
**修改的既有文件**：**无**

---

## 1. Frozen population

70 个「任意类别 IoU==0」的 small GT 中，V2 的 E1 家族 = **38**：`SCENE_DIFFICULTY` 21 + `WEAK_FEATURE_EVIDENCE` 17。

### ⚠️ 先更正 V2 的标签（这一点改变了全部后续解读）

`_v3_analysis.py` 实测这 38 个的邻域结构：

| V2 标签 | 实测 | 真实含义 |
|---|---|---|
| `SCENE_DIFFICULTY` (21) | `neighbor_gt_count` 1–5（中位 2），**邻居同类中位 2、异类 0** | **拥挤的同类邻域，邻居也全漏** → 你矩阵的 **Case C** |
| `WEAK_FEATURE_EVIDENCE` (17) | **`neighbor_gt_count` 全为 0** | **真正孤立，邻域内无任何 GT** → 你矩阵的 **Case D** |

**⇒ 38 = 21「拥挤」+ 17「孤立」。** V2 的 `WEAK_FEATURE_EVIDENCE` 命名有误导（它其实是 isolated），本轮起改用 C/D 表述。

类别：person 18 / animal 13 / light 3 / sign 2 / garbage_can 1 / ball 1；输入空间 √area 中位 15.3px。

---

## 2. Matched controls

| 控制类型 | 数量 |
|---|---:|
| Control A：同图 + 同类 + 相似尺寸（ratio 0.67–1.5）+ 已检出 | **19** |
| Control A′：同图 + 异类 + 相似尺寸 + 已检出 | **5** |
| Control B：跨图 + 同类 + 相似尺寸 + 同分辨率 | **14** |
| **无可靠 control** | **0** |

**⇒ 38/38 都有可用对照，配对分析成立。**（19+5=24 同图，14 跨图）

---

## 3. Global density

| | 38 missed 中位 | matched control 中位 |
|---|---:|---:|
| 该图 GT 总数 | 由 §4 局部密度刻画（见下） | — |

（全局密度未显示与 missed 的关联；决定性证据在 §4 的局部密度与 §5 的方向检验。）

---

## 4. Local density —— **结果与竞争假说相反**

多半径（r × GT 对角）邻域：

| 量 | 38 missed 中位 | matched control 中位 |
|---|---:|---:|
| `n_r1`（r=1 邻居数） | **0.0** | **0.5** |
| `n_r2` | **0.0** | **1.5** |
| `n_r3` | **1.0** | **2.0** |
| `n_r5` | **1.0** | **2.0** |
| `n_same_r3` | 0.0 | 0.5 |
| `n_other_r5` | 0.0 | 1.0 |

**⇒ missed 目标的局部 GT 密度系统性**低于**其配对对照，而不是更高。**

### §5 方向检验（全部 371 small GT）

| r=1 邻居数 | n | any-zero-overlap 率 |
|---|---:|---:|
| 0 | 169 | **22.5%** |
| 1 | 108 | 19.4% |
| 2+ | 94 | **11.7%** |

| r=3 邻居数 | n | 率 |
|---|---:|---:|
| 0 | 81 | 24.7% |
| 1 | 63 | 27.0% |
| 2+ | 227 | **14.5%** |

**⇒ 单调，但方向是「密度越高、越不容易 miss」。**

**⇒ 「局部实例竞争导致 miss」这一假设被本数据否定（不是无证据，是方向相反）。**

可能的解释（**未验证**）：密度是被检出的场景/目标典型性的代理 —— 密集街景中的目标大、近、典型；孤立目标更可能远、小、非常规。也可能存在检出-密度反向因果（模型在密集中更常触发）。

---

## 5. Competition analysis

对 38 个 E1（**按构造它们不含「邻居已检出」的情形**）：

| 模式 | Count | % |
|---|---:|---:|
| Competition-C：高密度 + 邻居全部失败 | **21** | 55.3% |
| Competition-D：孤立（无邻域） | **17** | 44.7% |
| Competition-A（同类邻居已检出） | **0** | — 按构造排除 |
| Competition-B（异类邻居已检出） | **0** | — 按构造排除 |

A/B 类的实例在 V2 被归入 `INSTANCE_SEPARATION`(11) 与 `CLASS_CONFUSION`(6)，**不在本轮的 38 内**。

---

## 6. Border analysis

`border_margin = min(x1, y1, W−x2, H−y2)`；`border_norm = margin / min(W,H)`

| border_norm | 38 missed | 占 38 | control | 占 ctrl | **全体 small 的 zero-overlap 率** |
|---|---:|---:|---:|---:|---:|
| **< 0.05** | **7** | **18.4%** | 3 | 7.9% | **32.0%** |
| 0.05–0.10 | 1 | 2.6% | 3 | 7.9% | 19.0% |
| 0.10–0.20 | 9 | 23.7% | 12 | 31.6% | 22.5% |
| > 0.20 | 21 | 55.3% | 20 | 52.6% | **16.7%** |

- 38 个在最贴边档的富集 = **18.4% vs 对照 7.9% ⇒ 2.3×**
- 全体 small 的 zero-overlap 率在最贴边档 **32.0%**，最远档 **16.7%** ⇒ **边界确实与 miss 相关**
- **但**：被截断（margin ≤ 1px）的只有 **2/38**；且最贴边档只覆盖 **7/38 = 18.4%**

**⇒ 边界是一条真实但覆盖面小的线索。**

**不把「靠近 border」直接等同于「border problem」** —— 可能是构图/上下文机制，也可能是远处的目标同时更小、更贴边。

---

## 7. Density × competition matrix

阈值 `n_r1 >= 1` = 高密度（**注：初版用中位数作阈值，中位为 0 导致矩阵退化，已修正**）：

| | 同类邻居成功 | 邻居全失败 |
|---|---:|---:|
| **高密度** (n_r1≥1) | **A: 8 (11.4%)** | **C: 24 (34.3%)** |
| **低密度** (n_r1=0) | **B: 4 (5.7%)** | **D: 34 (48.6%)** |

**最大格 = D（低密度 + 无邻近成功）= 34，占 48.6%。**

按你的解释框架：
- **D 是最大格** → 指向「class-specific representation / appearance blind spot」而非竞争
- C（24, 34.3%）= 在模型通常表现良好的密集场景里出现的不寻常失败
- A+B（12, 17.1%）= 真正的实例/类别竞争

---

## 8. Animal vs Person（叠加密度与边界控制）

| 控制层 | animal n | animal miss | person n | person miss |
|---|---:|---:|---:|---:|
| **密度 r=1 = 0** | 42 | **28.6%** | 51 | **33.3%** |
| 密度 r=1 = 1 | 17 | **58.8%** | 65 | **9.2%** |
| 密度 r=1 ≥ 2 | 4 | 50.0% | 79 | 8.9% |
| border_norm < 0.10 | 13 | 53.8% | 21 | 4.8% |

### ⚠️ 这削弱了 V2 的「纯类别效应」结论

- **在零密度层，animal 与 person 的差距消失（28.6% vs 33.3%，person 甚至更差）**
- 差距**只在密度 ≥1 的层出现**（57.1% vs 9.0% 合并）
- 但 animal 在密度≥1 的样本只有 **21 个**，且这是**事后分层**

**⇒ V2 的 `CLASS_EFFECT`（E2–E3）应下调**：animal/person 差异**与密度交互**，不是无条件的类别效应。
更准确的说法：**person 会随密度/场景变典型而改善，animal 不改善** —— 但这是 E1（分层后样本小）。

---

## 9. RGB / IR / Depth local evidence

### 分类标签（38 vs matched controls，n=38 / 34）

| tag | 38 missed | 占 | control | 占 |
|---|---:|---:|---:|---:|
| **RGB+IR** | 11 | 28.9% | **23** | **67.6%** |
| **RGB strong（仅 RGB）** | **19** | **50.0%** | 6 | 17.6% |
| IR strong（仅 IR） | 5 | 13.2% | 4 | 11.8% |
| Depth only | 0 | 0.0% | 0 | 0.0% |
| **weak-all** | **3** | **7.9%** | 1 | 2.9% |

**⇒ 92.1% 的 38 个目标在至少一个模态里有可辨识的局部证据。**

### 连续量 + 显著性检验（Mann-Whitney，n=38 vs 34）

| metric | missed 中位 | ctrl 中位 | 比值 | p |
|---|---:|---:|---:|---:|
| **infrared_std** | **12.29** | **19.79** | **0.62** | **0.0054** |
| **infrared_edgedens** | 0.000 | 0.000 | — | **0.0129** |
| visible_lapvar | 1464.4 | 909.2 | 1.61 | 0.4199 |
| visible_gradmean | 85.15 | 82.73 | 1.03 | 0.7308 |
| visible_std | 40.74 | 39.47 | 1.03 | 0.7055 |
| visible_edgedens | 0.158 | 0.140 | 1.13 | 0.7650 |
| infrared_gradmean | 20.48 | 17.61 | 1.16 | 0.4006 |
| depth_std / depth_gradmean | 0.000 | 0.000 | — | 0.4625 / 0.5816 |

**唯一显著的是 IR 通道的局部对比度**：missed 比配对对照低 **38%**。
**RGB 全部无差异；Depth 全部无差异**（非零率 34.2% vs 38.2%）。

⚠️ **必须同读的两条限制**：
1. 9 项检验，Bonferroni α = 0.0056；`infrared_std` 的 p = **0.0054 勉强守住**，属边界显著，**不应单独作为强结论**
2. `weak-all` 只有 7.9% ⇒ **不支持「内在低可见度」**；但「RGB 证据充足而模型不响应」也**没有指向任何具体模块**

---

## 10. Primary mechanism candidates

| 候选 | 支持 | 反对 |
|---|---|---|
| **SCENE_COMPETITION**（局部竞争） | — | **方向相反**（§4/§5：密度是保护性的）→ **否定** |
| **INSTANCE_SEPARATION** | §7 的 A+B 共 12（17.1%） | 覆盖面 <20% |
| **CLASS_SPECIFIC_BLINDSPOT** | §9 的 IR 对比度差异（p=0.0054）；§8 animal 在密度≥1 不改善 | §8 零密度层差距消失；IR 通道本征贡献仅 1.46% |
| **BORDER_COMPOSITION** | §6 富集 2.3×；zero-overlap 率 32.0% vs 16.7% | 只覆盖 7/38（18.4%）；截断仅 2/38 |
| **LOW_VISIBILITY / intrinsic data** | — | **92.1% 在 ≥1 模态有局部证据** → **基本否定** |
| **UNKNOWN** | 最大格 D（48.6%）无明确机制 | — |

---

## 11. Evidence level

| 结论 | 等级 |
|---|---|
| **局部密度与 miss 反向（密度是保护性的）** | **E2**（全 371 small GT 单调 + 配对对照同向） |
| **不是内在低可见度（92.1% 有至少一个模态的局部证据）** | **E2** |
| **RGB 局部证据无差异** | **E2**（配对 + 显著性检验） |
| **IR 局部对比度低 38%（p=0.0054）** | **E2，边界**（Bonferroni α=0.0056） |
| **Depth 局部证据无差异** | **E2** |
| **边界富集 2.3×** | **E1–E2**（富集明显但覆盖小） |
| **最大格 D = 低密度孤立 miss（48.6%）** | **E1**（机制未定） |
| **animal 类别效应与密度交互** | **E1**（分层后 animal n=21） |

---

## 12. Coverage —— 每个机制能解释多少

对 **70 个 zero-overlap** small GT：

| 机制 | Count | % |
|---|---:|---:|
| **LOW_DENSITY_ISOLATED** | 25 | 35.7% |
| **LOCAL_COMPETITION（邻居亦漏）** | 18 | 25.7% |
| SCENE_COMPETITION / INSTANCE_SEPARATION | 12 | 17.1% |
| CLASS_CONFUSION | 10 | 14.3% |
| BORDER_COMPOSITION | 5 | 7.1% |

**⇒ 没有任何单一机制覆盖超过 36%。**

---

## 13. Model-actionable fraction

- **可用模型干预的**：INSTANCE_SEPARATION + CLASS_CONFUSION = **22（31.4%）**
- **机制不明、无干预点**：LOW_DENSITY_ISOLATED + LOCAL_COMPETITION = **43（61.4%）** ← 且**这两个恰恰是密度保护性的反常子集**
- **数据侧 / 构图侧**：BORDER = 5（7.1%）

**⇒ 61.4% 的目标没有任何已确立的机制，更谈不上干预点。**

---

## 14. Training Gate

```text
HOLD
```

**四条理由：**

1. **最大机制仍只有 E1** —— 48.6% 落在矩阵 D 格（低密度孤立），无机制；61.4% 无干预点（§13）
2. **最强的候选被本轮数据否定** —— SCENE_COMPETITION 的方向相反（§4/§5，E2）
3. **auto 类别效应被削弱** —— animal/person 差距在零密度层消失（§8），V2 的 E2–E3 应下调为 E1
4. **唯一 E2 的新线索（IR 对比度低 38%）没有模型干预点** —— 「IR 局部对比低」不能直接翻译成某个模块的改动；且 IR 本征贡献仅 1.46%（既有模态消融）

按你的 §12 判据，`GO-TO-MODEL` 需要「机制达 E2/E3 + 覆盖面足够 + 有明确模型干预点 + 可设计单变量实验」。**当前四者无法同时满足**（E2 的线索无干预点；有干预点的覆盖 <20%）。

---

## 15. If GO（本轮不适用）

不输出。

---

## 16. If HOLD —— 还缺什么证据

**唯一值得补的一项**：把「48.6% 的低密度孤立 miss」与「25.7% 的邻居亦漏」合并后的 **61.4% 无机制群体**做一次可判别性检验。

**判别设计（只读，不需训练）**：
对每个目标，取其 GT 框内像素，比较三组：
```
G1 = 38 个 E1 missed
G2 = 它们的 matched detected controls
G3 = 同图同尺寸的**非 small** GT（作为"模型显然能处理"的参照）
```
用**同一套**局部描述子（已有 `_v2_visual.py` 的 proxy 即可），检验 G1 与 G2 是否存在**任何**稳定的分布差异 —— 本轮只发现了 IR 对比度一项。

**若再加入描述子后仍然只有 IR 这一项** → 说明该群体在像素层与成功目标**不可分**，那问题更可能在**assignment/训练目标**层面，而非输入表征；届时应转向检查 `TaskAlignedAssigner` 对孤立小目标的 topk 行为（这才是可设计单变量实验的干预点）。

**若出现新的稳定判据** → 按 §13 的 `Pattern 1–6` 重新归因，再决定 Gate。

**不建议**：在拿到上面这项证据前开训。

---

## 17. Reproducibility

```text
training                 : 0
core code modification    : 0
evaluator modification   : 0
new inference            : 0
online submission        : 0
```

**新增文件（均在 `diagnostic/small_object_cause_v2/`）**
```
_v3_analysis.py            §2 对照 / §3 全局密度 / §4 局部密度 / §5 方向检验 / §6 边界 / §7 矩阵 / §8 class
_v3_matrix_modality.py     §7 修正矩阵 / §9 多模态证据 / §10-12 coverage
_v3_analysis.json          annotated 371 small GT + 38 的 controls
_v3_matrix_modality.json   矩阵 / 模态标签 / coverage
REPORT_V3.md               本报告
```

**读取的既有资产（未改动）**
```
diagnostic/small_object_cause_v2/_v2_{records,visual,attribute}.json   V2 产出
diagnostic/sepstem_clahe/best_full/results/                            D′ val 预测
data/processed/rgbid_split_train/{images,labels}/val/                  val GT + RGB/IR/Depth
scripts/official_eval.py                                               直接 import
ultralytics/utils/patches.py:imread                                    pipeline 同一 reader
```

**复现命令**
```bash
python -X utf8 diagnostic/small_object_cause_v2/_v3_analysis.py
python -X utf8 diagnostic/small_object_cause_v2/_v3_matrix_modality.py
```

---

## 附：本轮发现的自身错误（记录备查）

1. **§7 矩阵初版用 `median(n_r1)` 作 dense 阈值，而中位为 0** → 矩阵退化为「全部高密度」。已改为阈值 1 重算（`_v3_matrix_modality.py`）。
2. **V2 标签 `WEAK_FEATURE_EVIDENCE` 名不符实**：其 17 个成员的 `neighbor_gt_count` 全为 0，实为「孤立」而非「弱特征」。本轮改用 C/D 表述，并据此重新解读。
