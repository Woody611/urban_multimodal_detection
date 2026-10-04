# P4 — COUPLED MECHANISM AUDIT

**日期** 2026-10-03 · **性质** 只读 / 零 GPU / 零训练 · **incumbent** D′ = SepStem + IR-CLAHE (0.515281)

---

## 0. Executive Verdict

```text
CASE 1 — NO ACTIONABLE MECHANISM  ⇒  STOP optimization experiments
```

三个待统一的现象（epoch-0 gap / training amplification / raw-generation F0）**确实统一** ——
但统一它们的是一个**阶段**而非新机制：**GT 类分数低于 conf 阈值**。

这个阶段**已经被历史实验关闭**（G1 终裁 STAGE 4 classification/score、train-side cls supervision RED、
P2 score calibration）。本轮新增的只是**把 F0 从"生成失败"重新归因为"评分失败"** —— 这条重新归因
有实际价值（它解释了为什么 generation 类干预注定无效），但它**不开启新入口**。

---

## 1. Evidence Ledger

| # | Observation | Evidence | What it proves | What it does NOT prove | Closed mechanisms |
|---|---|---|---|---|---|
| 1 | epoch-0 G86/CONTROL gap | `epoch0.pt`(pre-training EMA) → G86 −7.03 vs CONTROL −1.81 = **−5.223 nats** | 劣势在训练前已存在 | **不证明**是训练造成的；也**不能**排除选择效应（G86 是按最终结果选的） | — |
| 2 | training amplification | 30 快照重建轨迹，−5.223 → −17.843，**+12.620** | 训练显著放大既有 gap | 不证明"训练创造塌缩" | — |
| 3 | symmetric decomposition | `rep_sym+cls_sym+Δb==Δlogit` 精确(误差0)；G86 −2.563/−7.063；CONTROL +5.394/−2.507 | 水平侧分类器 2.8×；**分离侧表征 63%/分类器 36%** ⇒ CASE C | 不给出"哪个可干预" | 表征侧杠杆(attention/capacity/resolution/modality)已闭环 |
| 4 | raw-generation F0 dominance | P3: small F0=95(25.6%) ≫ NMS 2 / top-100 5 | 失败在候选产生阶段 | **不证明**是"框生成"问题（见 #5） | NMS / top-k / conf / max_det |
| 5 | **F0 的真实机制** | G86 GT类 sigmoid 中位 **5.5e-08**、**100% <1e-3**；small F0 桶 **95.8% <1e-3** | F0 = 类分数低于 conf 阈值 ⇒ 该类进不了候选列表 | 不证明是定位/生成能力不足 | **G1 STAGE 4 / cls-supervision / P2 calibration** |
| 6 | scale ratio ≈1.90× | P3 `best_raw_area_ratio` 中位 1.90×，61.4% >1.5× | **存在的候选**尺寸偏大 | 对 F0（无候选）根本不适用；不证明 scale aug 能解决 | OASA 1.4 / 2.0 / 1536 / F1 |
| 7 | NMS non-causality | 30 F1 候选中仅 4 个本身 IoU≥0.5；仅 2/371 同时满足 IoU≥0.5 且受 top-100 影响 | NMS 不是瓶颈 | — | NMS 调参 |
| 8 | top-100 non-causality | 仅 5 small GT 受 top-100 影响；max_det 从未触发 | top-100 不是瓶颈 | — | top-k / max_det |
| 9 | OASA negative | 线上 48.134 / 47.610 < D′ 48.712 | 小目标 scale aug 无效 | — | OASA |
| 10 | P2 negative | 0.50232 vs 0.50928；配对 bootstrap CI 含 0 | 加检测尺度无效 | — | P2 |
| 11 | resolution negative | 1536 纯推理 −0.0053 | 分辨率无增益 | — | 1536 |
| 12 | capacity negative | F5 = YOLO11l@1280 = 0.52619 < F4 0.53273 | 容量杠杆证伪 | — | 容量 |
| 13 | cls-supervision negative | 终裁 RED；TAL 把 GT 类 target 峰值钉死为 maxCIoU(98.6%) | 监督信号不是瓶颈 | — | train-side cls supervision |
| 14 | **定义循环性**（本轮新增） | G86 ⊂ {final 无 ≥.5 框}；「final 无框」⟸ 类分数<阈值。实测 G86 **100%** <1e-3 | #4 与 #5 **不是两个独立观察**，是同一件事 | 不构成相互印证 | — |
| 15 | **匹配对照**（本轮新增） | G86 vs (同 class 同 size±20%) CONTROL：Δlogit **−17.611**，100%<0，CI[−18.09,−16.73] | 差异**不能**由 class/size 解释 | ⚠ **仍循环** —— CONTROL 定义为 `detected=1`，类分数必然活着 | — |

---

## 2. What CASE C Actually Means

对称分解（恒等式精确）：

```
G86      : rep −2.563   cls −7.063   Δb −0.113   Δlogit −9.739
CONTROL  : rep +5.394   cls −2.507   Δb −0.006   Δlogit +2.881
```

- **问 logit 水平**：分类器 2.8× 主导
- **问 G86−CONTROL 分离**（=塌缩本义）：**表征 −7.957 (63%)** / 分类器 −4.556 (36%)

机制读法：**分类器漂移是全局向下压的**（CONTROL 也被压 −2.51），**表征漂移是分组歧视的**
（G86 −2.56 vs CONTROL **+5.39**）。

**重要限度**：CASE C 描述的是**哪一侧在动**，不是**哪一侧可干预**。表征侧的历史杠杆
（attention P3、capacity F5、resolution 1536、single-modality）**全部已闭环**。

---

## 3. Remaining Mechanism Space

本轮把候选空间**缩小**了，没有扩大：

1. 失败阶段 = **分类分数**（不是定位、不是生成、不是后处理）—— 与 G1 终裁一致。
2. F0 的规模是**阈值产物**：conf=0.001、GT 类 sigmoid 中位 5.5e-08 ⇒ 相差 **4 个数量级**。
   即使把 conf 降到 1e-7，这些框也只是变成"存在但无意义"的候选。
3. 表征侧虽然是分离的主因（63%），但**指向的杠杆已全部证伪**。

⇒ **机制空间在"分类分数"处收敛，而该处已被显式关闭。**

---

## 4. Candidate Mechanism #1

**状态：REJECT — 不可证伪 / 实为阶段重述**

- **Mechanism**：`GT 类 sigmoid < conf 阈值 → 该类不进候选列表 → F0 → 末输出缺失`。
- **Supporting**：G86 100% <1e-3；small F0 95.8% <1e-3；与轨迹的 −16.7 nats 一致。
- **Contradicting**：无。
- **Why not closed**：**它就是已被关闭的那个阶段**。G1 终裁已把失败定位到 STAGE 4（classification/score）
  并 CLOSE；train-side cls supervision 已 RED。
- **判定**：`"类分数低导致 F0"` 是**阶段描述**，不是机制。它**没有回答"为什么这些对象的类分数是死的"**，
  因此不可证伪、无法据以设计干预。**REJECT。**

---

## 5. Candidate Mechanism #2

**状态：REJECT — renamed closed direction**

- **Mechanism（提议形式）**：`小目标 / 细长类的 GT 类分数在训练中被系统性压低 → 需要针对性增强其类监督或
  重加权 → 提升小目标 recall`。
- **Why rejected**：这是 **train-side cls supervision / class weighting / oversampling** 的换皮。
  历史已测：cls-supervision 终裁 RED；小目标 oversampling / crop 路径已证伪；F1（native-small replay，
  +57.6% 小目标曝光）线上 48.204 < 48.712 **被线上否决**。
- **判定**：`REJECT — renamed closed direction`。

---

## 6. Candidate Mechanism #3

**不提出。** 本轮证据不支持第三个候选，**不为凑数而造**（§14）。

---

## 7. G86/CONTROL × Scale Error Analysis（Part D）

### D1 — G86 是什么

| 维度 | G86 实测 |
|---|---|
| 定义 | `certified_prefinal_loss==1`：头部（**square** 几何 TAL 正样本 CIoU≥0.5）认证，但最终输出无 ≥0.5 框 |
| 类别 | person 25 / animal 16 / sign 13 / light 13 / bicycle 10 / garbage_can 3 / car 3 / ball 2 / seat 1 |
| 尺寸 | √area 中位 **50.9 px**（p10 35.1 / p90 81.1）—— 小到中，非 tiny |
| 图像 | 54 / 281 张 val 图；HR(1920×1080) 占 20.9% |
| 模态 | RGBID 5ch（与全体相同，无特殊性） |

### D2 — 是否有更差的 raw-generation 读数

**有，而且是二值的**（独立复算，用 `layers.npz` 的 pre-NMS 候选 × 原图像素坐标）：

| role | n | same-class raw IoU 中位 | ≥0.5 | ≥0.1 | 零重叠 |
|---|---:|---:|---:|---:|---:|
| **G86** | 86 | **0.000** | 0.0% | 0.0% | **100.0%** |
| CONTROL | 1070 | 0.819 | 100.0% | 100.0% | 0.0% |
| MISSED_NON86 | 164 | 0.244 | 7.9% | 57.9% | 35.4% |
| medium 全体 | 1320 | 0.785 | 82.0% | 88.3% | 10.9% |

**但是**：G86 的 GT 类 sigmoid 中位 **5.5e-08**，**100% < 1e-3 = conf 阈值**。
⇒ 零候选是**类分数低于阈值**的**必然结果**，不是独立的生成能力证据。

### D3 — 何时分化

来自快照轨迹（10-epoch 分辨率）：**epoch 0 已存在 −5.223**；csv ep11 两组几乎同步（Δsep −0.188）；
**ep21 起分化**；**ep230 后平台**。⚠ 分辨率不足以定位 ep1/2/5 的早期细节。

### D 的结论

**G86 × scale error 的相关性是定义性的，不是因果性的。** G86 之所以"零候选"，是因为它的类分数死了；
它之所以被选进 G86，是因为它末输出缺失；末输出缺失主要由类分数死亡造成。**闭环。**

---

## 8. Mechanisms Explicitly Rejected

| 候选（换皮形式） | 撞上的已关闭方向 |
|---|---|
| 类分数低 → 加强类监督 / 重加权 | train-side cls supervision (RED)、class weighting |
| F0 多 → 提高候选密度（anchor/P2/多尺度） | P2 (REJECTED)、1536、OASA |
| scale ratio 1.90× → scale aug | OASA 1.4/2.0、F1（线上否决） |
| 后处理残留 → NMS/top-k/conf 调参 | NMS 因果审计（2/371）、top-100（5/371）、conf=0.0001（线上 −2.729） |
| 表征侧分离主导 → attention / 容量 / 分辨率 | P3 attention、F5 容量、1536 |
| 单模态成分 → IR/RGB/Depth 侧重 | 单模态优化 NO_GO、互补性实测（97.15% TP 不依赖 IR/Depth） |

---

## 9. Decision Gate

```text
CASE 1 — NO ACTIONABLE MECHANISM
```

判定理由（逐条对应 §13 CASE 3 的六项要求）：

| 要求 | 满足？ |
|---|---|
| 机制与已关闭方向明确不同 | ✗ —— 收敛到"分类分数"，正是 G1 STAGE 4 已关闭处 |
| 至少两个**独立**观察支持 | ✗ —— 本轮证明原以为独立的两条（F0 统计 vs 类分数）实为同一件事（Evidence Ledger #14） |
| zero-GPU / cheap diagnostic 支持 | ✗ —— 本轮 zero-GPU 分析反而**否证**了独立性 |
| 可提出具体 intervention | ✗ —— 能提出的都是换皮 |
| intervention 有明确 causal prediction | ✗ |
| 有明确 stop criterion | — |

**CASE 3 六项全部不满足 ⇒ 不得进入训练。**

---

## 10. Recommended Next Action

```text
STOP.
不再启动任何优化方向的训练。
```

**为什么继续训练的期望值很低：**

1. **阶段已定位且已关闭**：失败在分类分数，不是定位/生成/后处理。该阶段的干预（cls 监督、重加权、
   score calibration）已全部试过并证伪。
2. **现象之间的"相互印证"经查是定义循环**。过去几轮反复出现的模式（把同一件事从不同口径数两遍，
   当作两条独立证据）在本轮被显式拆穿（#14、#15）。
3. **表征侧虽是分离主因（63%），但其杠杆已全部闭环**（attention / capacity / resolution / modality）。
4. **收益侧已无空间**：本地与线上 gap 的剩余部分主要由**指标口径**（−0.0308）而非可训练成分解释
   （见既有 local-online gap 分解）；线上 48.712 已是本架构在本阶段的稳定值。

**如果仍要花预算**，唯一有正当理由的**非训练**行动是**数据卫生**（不是优化）：
G86 这个群体横跨三个不同日期、不同几何（square `_v6_domains` / Oct-1 attribution / rect P3）的冻结产物，
其"认证"状态**从未在现行 rect 流水线上复现过**。建议做一次**零 GPU** 的复现核对；
若不复现，则 G86 及其轨迹结论应整体退役，以免后续分析继续建立在一个跨口径的集合上。

**这是清理，不是找入口。本轮不推荐任何 GPU 行动。**

---

## 附：本轮未做但可做（非必需）

- `tal_metrics.csv`（3.4 MB）未展开 —— 与 h/W 归因正交。
- 小目标 371 的逐 GT 轨迹（快照法的 `medium_cells` 只覆盖 1320 medium）—— 需另建 cells。
