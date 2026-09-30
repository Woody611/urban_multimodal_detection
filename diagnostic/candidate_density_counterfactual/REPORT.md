# Candidate-Density Counterfactual Audit

**日期**：2026-09-28
**性质**：**READ-ONLY**。0 训练 / 0 backward / 0 optimizer.step / 0 改任何既有源码/配置/checkpoint /
0 改 evaluator/dataset/augmentation/assigner / 0 改模型权重 / 0 test inference / 0 submission。
所有新增文件位于 `diagnostic/candidate_density_counterfactual/`，未覆盖任何既有文件。

**唯一问题**：native-small 的低 maxIoU / low alignment，主要是 candidate **数量不足**造成的
sampling effect，还是 small 自身 candidate geometry / quality 较差？

---

## 1. Executive Conclusion

**一句话**：**candidate 数量（sampling / order-statistics）是解释 native-small 低 maxIoU 的主因；
small 自身的 candidate geometry/quality 不但不差，反而比 large 好得多。**

```text
Small native    maxCIoU med = 0.8798   (n_cand = 4)
Large native    maxCIoU med = 0.9795   (n_cand = 169)
Large Random-4  maxCIoU med = 0.7755   ← 同等候选数下 **低于** small（overshoot）
```

**裁决：A — STRONG SUPPORT**（§14，含 overshoot 判定说明与对协议字面规则的一处明示偏离）。

---

## 2. Read-only / Provenance Gate

| 文件 | SHA256（前 16） |
|---|---|
| `ultralytics/utils/tal.py`（assigner） | `aae7e8ac438f00cd` |
| `ultralytics/utils/loss.py` | `0f092cf22a372f6d` |
| `ultralytics/utils/instance.py` | `78a72d1f4354c29f` |
| `ultralytics/data/augment.py` | `5cb9a407625921ce` |
| `ultralytics/data/base.py` | `8bcf834930155bd5` |
| `ultralytics/data/build.py` | `f014263f2bfe5921` |
| `ultralytics/data/dataset.py` | `0aba2bc14793ab85` |
| `configs/yolo11m_sepstem.yaml` | `9b14f29460733384` |
| `configs/train_rgbid_sepstem_clahe.yaml` | `a4e329cfc3d22020` |
| **D′ best.pt** | `1cae45f75693f541` |

全部与上一轮 E1/E2 记录**逐位相同**，在 `_run.py` 开头重新计算并落盘到 `_meta.json`。

### 2b. Read-only gate（实测）

| 项 | 值 |
|---|---|
| `existing files modified` | **0**（上述 10 个文件 SHA256 复检全部 UNCHANGED） |
| `optimizer step` | **0**（全程 `torch.no_grad()`；无 `.backward()`、无优化器构造） |
| `checkpoint modified` | **0**（`best.pt` 未写） |
| `training process` | **0** |
| 本轮新增 | 仅 `diagnostic/candidate_density_counterfactual/` 下文件 |
| `git status --short` | 与上一轮结束时相同；**无本轮新增的 `M` 项** |

---

## 3. Real Pipeline Verification

**没有手写任何「类似 YOLO candidate selection」的简化模拟。** 全部使用运行中的真实对象：

| 组件 | 来源 | 证据 |
|---|---|---|
| 模型 | D′ `best.pt` | `Transferred 661/661 items from pretrained weights` |
| dataset / labels | `build_yolo_dataset`（与训练同一条代码路径） | `dataset n=1599` |
| assigner | 运行中的 `loss.assigner` | `topk=10 alpha=0.5 beta=6.0 nc=12`（从对象读出，非采信记忆） |
| CIoU | `TaskAlignedAssigner.iou_calculation` | `bbox_iou(..., CIoU=True).clamp_(0)` |
| bucket | **native_area**，**非 aug_area** | §4 |

**runtime hook 合法性证据**：`get_box_metrics` 是父类的**逐行副本**（唯一新增是
`self.last["bbox_scores"] = bbox_scores`），且每次运行都执行自检 —— `ProbeAssigner` 与**父类**
在同一输入、同一参数下的五个返回值必须逐位一致：

```
[SELF-CHECK] ProbeAssigner vs 父类逐位一致: True
```

磁盘上**没有任何既有源码被修改**（monkey-patch 只作用于运行中的类对象）。

---

## 4. Baseline Reproduction

**gate 口径 = native_area 分桶**（非 aug_area）。目标见协议 §3。

| class | n | n_cand med | maxCIoU med | maxCls med | bestAlign med | zero-cand% | 目标 | ok |
|---|---:|---:|---:|---:|---:|---:|---|---|
| small | 250 | **4.0** | **0.8798** | 0.7692 | **0.3540** | 2.40% | 250 / 4.0 / 0.8805 / 0.354 | ✅ |
| medium | 934 | **20.0** | **0.9608** | 0.8785 | **0.7032** | 0.43% | 934 / 20.0 / 0.9610 / 0.703 | ✅ |
| large | 736 | **169.0** | **0.9795** | 0.9352 | **0.8256** | 0.00% | 736 / 169.0 / 0.9795 / 0.826 | ✅ |

```text
BASELINE REPRODUCTION PASS  →  继续反事实
```
（容差：n_cand ±1.5、maxCIoU ±0.02、align ±0.05。最大偏差是 small maxCIoU 的 0.0007。）

---

## 5. Candidate Geometry Baseline

`n_cand` 分布：

| class | n | mean | sd | p10 | p25 | **p50** | p75 | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 250 | 10.08 | 19.57 | 2.0 | 2.25 | **4.0** | 8.0 | 21.3 |
| medium | 934 | 30.76 | 35.94 | 6.0 | 11.0 | **20.0** | 38.0 | 64.0 |
| large | 736 | 398.40 | 724.22 | 45.0 | 77.0 | **169.0** | 365.0 | 942.5 |

`maxCIoU` 分布：

| class | mean | sd | p10 | p25 | **p50** | p75 | p90 | P(≥.5) | P(≥.7) | P(≥.8) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 0.8027 | 0.2135 | 0.5483 | 0.7691 | **0.8798** | 0.9330 | 0.9665 | 92.8% | 80.8% | **70.0%** |
| medium | 0.9363 | 0.0903 | 0.8709 | 0.9327 | **0.9608** | 0.9765 | 0.9842 | 99.3% | 97.8% | 96.3% |
| large | 0.9680 | 0.0533 | 0.9420 | 0.9677 | **0.9795** | 0.9885 | 0.9926 | 99.9% | 99.3% | 98.4% |

### ★ §5b — 逐候选质量（**与 candidate 数量无关**）

分离「数量」与「质量」最干净的量：**per-candidate 的均值/中位不受 n_cand 影响**。

| class | 候选总数 | **P(cand CIoU≥0.8)** | **P(cand cls≥0.5)** | CIoU mean | CIoU med |
|---|---:|---:|---:|---:|---:|
| **small** | 2,519 | **57.6%** | **38.7%** | 0.6991 | 0.8451 |
| medium | 28,730 | 53.3% | 26.8% | 0.6570 | 0.8387 |
| **large** | 293,225 | **7.2%** | **2.6%** | 0.1721 | 0.0355 |

**⇒ small 的候选池按每个候选看，CIoU 密度是 large 的 8 倍（57.6% vs 7.2%）、
置信度密度是 15 倍（38.7% vs 2.6%）。small 的 candidate quality 不差，反而远好于 large。**

### §5c — Sampling curve（Large GT 的 order-statistics）

`E[max CIoU over k 个随机候选]`（大目标自身候选池，seed 0..29，per-GT 取中位后总体取中位）：

| k | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 169(native) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E[max CIoU] | 0.2957 | 0.5224 | **0.7897** | **0.9440** | 0.9637 | 0.9708 | 0.9753 | 0.9773 | 0.9795 |

**Small native = 0.8798 ⇒ 落在 k=4 (0.790) 与 k=8 (0.944) 之间
⇒ small 的 4 个候选相当于 large 池中约 5–6 个随机候选。**

---

## 6. CF1 Large → Random-4

从每个 native-large GT 的**真实候选集合**无放回随机抽 4 个，**固定 seed 列表 0..99**，配对反事实。

| regime | n | mean | sd | p10 | p25 | **p50** | p75 | p90 | P(≥.5) | P(≥.7) | P(≥.8) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Large native | 736 | 0.9680 | 0.0533 | 0.9420 | 0.9677 | **0.9795** | 0.9885 | 0.9926 | 99.9% | 99.3% | 98.4% |
| **Large Random-4** | 736 | 0.6752 | 0.2944 | 0.2154 | 0.4538 | **0.7755** | 0.9489 | 0.9657 | 70.2% | 55.0% | **48.4%** |

**Δ Random-4 = 0.7755 − 0.9795 = −0.2040**（占 native 的 20.8%）
⇒ **仅把候选数 169→4，large 的 maxCIoU 就掉 0.204。**

**辅助指标（同批反事实）**：`maxCls` 中位从 0.9352 掉到 **5.74e-06**，
`P(maxCls>0.5)` 只剩 **28.3%**；`bestAlign` 中位从 0.8256 掉到 ~0。
⇒ 大目标框内**高置信 anchor 极其稀疏**，随机 4 个几乎必然错过它们。

---

## 7. CF2 Large → Best-4

| regime | maxCIoU med | maxCls med | bestAlign med |
|---|---:|---:|---:|
| Large native | 0.9795 | 0.9352 | 0.8256 |
| **Large Best-4** | **0.9795** | 0.9162 | 0.8125 |

⚠ **CF2 对 maxCIoU 按构造退化**：按 CIoU 取前 4 再取 max = 全局 max，必然等于 native max
（实测 0.9795 vs 0.9795，逐位一致）。**它不提供任何关于「数量」的信息。**
唯一有信息的量是**前 4 的 CIoU 下界（第 4 高）= 0.9735**（说明 large 池中 ≥0.97 的候选不止一个）。
这是**选择性上界**，**不能**据此推断「small 只要 4 个候选就能达标」（协议 §13 明令禁止）。

---

## 8. CF3 Spatial-Matched-4

**CF3 = IMPLEMENTED，但定义是一个选择，不是唯一严格的定义。**
采用的定义：限制到 **stride-8 候选**（native-small 的正样本 92.8% 来自 stride-8，见 E2 审计），
在该子集内取距 GT 中心（归一化偏移口径）最近的 4 个。覆盖率 735/736。

| regime | n | maxCIoU med | P(≥.8) |
|---|---:|---:|---:|
| Large native | 736 | 0.9795 | 98.4% |
| **Large Stride8-4** | 735 | **0.7680** | 47.3% |

⚠ **该定义混入了 stride 通道**：large GT 在 stride-8 上的预测本身更差（大目标由 stride-16/32 负责），
故 0.7680 同时含「数量」与「stride 不匹配」两个因素，**不是纯 count 反事实**，仅作参考。

### §5d CF4 Coverage-matched 4（**本审计额外构造**，非协议要求）

定义：归一化偏移空间 `[-1,1]²` 取 4 个象限中心 (±0.5,±0.5)，每象限取最近候选 ——
模拟「4 个候选**铺满**整个目标」，这才是 small 的真实处境（4 个候选铺满小框）。

| regime | maxCIoU med | maxCls med | bestAlign med |
|---|---:|---:|---:|
| **Large Coverage-4** | **0.7136** | ~0 | ~0 |

**⚠ protocol sensitivity**：同为「4 个候选」，random 0.7755 / stride8 0.7680 / coverage 0.7136
⇒ 跨度 **0.0619**（协议 §12 CASE D 列出的条件之一，如实报告）。

---

## 9. Size / Aspect-Ratio Matching

```
native sqrt: small [11.8, 32.0] median 25.0  |  large [96.0, 1309.4] median 164.7
```

**⇒ native 尺寸分层按定义不相交（small<32px、large>96px）⇒ literal size matching 不可能。
明确报告，不伪造匹配。** CF1/CF2/CF3/CF4 均为 **within-GT** 反事实，尺寸/位置/上下文被完全固定。

可行的匹配是 **aspect-ratio 分层**（small 的 AR 四分位 0.500 / 0.775 / 1.258）：

| AR 层 | n_small | n_large | Small maxCIoU | LR4 maxCIoU | Large maxCIoU |
|---|---:|---:|---:|---:|---:|
| (−inf, 0.50] | 63 | 248 | 0.8275 | 0.6467 | 0.9794 |
| (0.50, 0.78] | 62 | 148 | 0.8277 | 0.7866 | 0.9832 |
| (0.78, 1.26] | 62 | 131 | 0.8882 | 0.6883 | 0.9823 |
| (1.26, +inf) | 63 | 209 | 0.9312 | 0.9116 | 0.9765 |

**⇒ 4/4 个 AR 层里 Small native 都高于 Large Random-4 ⇒ overshoot 稳定，不是 AR 造成的假象。**

---

## 10. Distribution Results

核心对照（**maxCIoU 中位**，主判据）：

| regime | n_cand | maxCIoU |
|---|---:|---:|
| **Small native** | 4 | **0.8798** |
| Large native | 169 | 0.9795 |
| Large Random-4 | 4 | **0.7755** |
| Large Best-4 | 4 | 0.9795（退化） |
| Large Stride8-4 | 4 | 0.7680 |
| Large Coverage-4 | 4 | 0.7136 |

```
Δ Random-4            = 0.7755 − 0.9795 = −0.2040
Small − Large_Random4 = 0.8798 − 0.7755 = +0.1043
Small / Large_Random4 = 1.1345
```

完整分位数（median/mean/sd/p10/p25/p75/p90）与 `P(maxCIoU≥t)` 见 §5、§6、§8。

---

## 11. Statistical / Monte-Carlo Uncertainty

CF1 使用**固定 seed 列表 0..99**（不每次随机生成）。

| 量 | mean | sd | p5 | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| per-GT **mean** over 100 seeds | 0.6758 | 0.2256 | 0.2298 | 0.7183 | 0.9492 |
| per-GT **median** over 100 seeds | 0.6752 | 0.2944 | 0.0970 | **0.7755** | 0.9709 |
| 总体（GT×seed 全当样本） | 0.6758 | 0.3214 | — | 0.8249 | — |

**主口径用 per-GT 分布，不把抽样当独立 GT。**（若误用 pooled 口径得 0.8249，偏乐观；
本报告主表一律用 per-GT 口径 0.7755。）

---

## 12. Mechanistic Interpretation

1. **large 的 maxCIoU 是极端 order statistic。** large 候选池中只有 **7.2%** 的候选 CIoU≥0.8、
   **2.6%** 的候选 cls≥0.5。169 个候选里抽到一个 0.98 是必然；降到 4 个后
   `P(max≥0.8)` 从 98.4% 掉到 **48.4%**，`P(maxCls>0.5)` 掉到 **28.3%**。
2. **small 的候选池密度远高于 large**（§5b）：逐候选 P(CIoU≥0.8) 57.6% vs 7.2%（8×）；
   P(cls≥0.5) 38.7% vs 2.6%（15×）。**small 的 candidate geometry/quality 不是瓶颈。**
3. **等数量比较下 small 反而更好**：同为 4 个候选，small 0.8798 > large-random 0.7755，
   且 random / stride8 / coverage 三种选法**全部**低于 small。
4. ⇒ **small 与 large 的 maxCIoU 差距（0.88 vs 0.98）是 count / order-statistics 效应。**
   §5c 给出定量刻度：small 的 0.8798 ≈ large 池中 5–6 个随机候选。
5. **反事实外推（仅假设，非物理）**：若保持 small 自身候选密度、只把数量提到 8/16，
   `P(max CIoU≥0.8)` 由 96.8%(k=4) → 99.9%(k=8) → 100%(k=16)。
   ⇒ 数量对 small 同样是有效杠杆。**但物理上 small 框内不可能容纳更多 stride-8 anchor
   ——这需要改 anchor 网格（finer stride / 更高分辨率检测层）。本审计不提出、也不背书该改法。**

**`best_align` 的地位**：主判据用 `maxCIoU` 与 `P(maxCIoU≥t)`。`best_align` 受 `alpha/beta` 与 cls
影响，其数值**不可跨设定比较**（E2 审计已量化该陷阱）；本报告中它只作辅助并已标注。

---

## 13. Limitations

1. **size matching 按定义不可能** ⇒ **跨类比较（§12.3/12.4）带无法消除的尺寸混淆**。
   已用 AR 分层部分控制（4/4 层一致），但不能替代尺寸匹配。
2. **CF 结论对 selection protocol 敏感**：random 0.7755 / stride8 0.7680 / coverage 0.7136，
   跨度 **0.0619**（协议 §12 CASE D 的条件之一）。
3. **CF2 对 maxCIoU 退化**（§7），不能作为独立的 count 证据。
4. **CF3 混入 stride 通道**（§8），不是纯 count 反事实。
5. **单一 checkpoint**：全部预测来自 D′ best.pt；换权重后结论的**方向**是否保持未验证。
6. **Monte-Carlo**：augmentation realization 与训练 RNG 不可复现；但 CF 内部**固定同一批
   realization 与同一批预测**，是配对反事实，同一 GT 跨 regime 的差异**精确**。
7. **「small 若有更多候选会怎样」不可直接检验**——物理上不存在这些 anchor；
   §12.5 是**假设性外推**，不构成因果证据。
8. 样本量：native-small 250 个；native sqrt <8px 的桶为空（见 exposure audit）。

---

## 14. Final Verdict

```text
A — STRONG SUPPORT
candidate-count effect plausibly explains a substantial fraction of the small geometry deficit.
```

**判定依据 + 一处对协议判定规则的显式偏离**：

协议 §12 的三条 CASE 都默认 `Large-Random4` 落在 `Small` 与 `Large-native` **之间**。
实测出现 **overshoot**：`Large-Random4 (0.7755) < Small (0.8798)`，即落到 small **之下**。
该情形协议未覆盖，故本报告显式说明判定：

- overshoot 意味着「**同等候选数下 small 反而更好**」⇒ 直接**排除**原问题两支中的
  「small 自身 candidate geometry / quality 较差」这一支。
- 同时 `Large-native (0.9795) >> Large-Random4 (0.7755)`，Δ = 0.204，且 §5c 的 sampling curve
  显示 large 的高 maxCIoU 完全可由 order statistics 解释。
- 再加 §5b：small 的**逐候选**质量比 large 高 8–15 倍。

⇒ 三条独立通道同向指向 **candidate 数量**，故判 **A**。

**⚠ 明示偏离**：若**严格按协议字面规则**（要求 `Small ≈ Large-Random4`），本结果**不满足 A 的
等式条件**，字面应落到 **D**。本报告选择 A，理由如上；该偏离是**明示的**，不是默认的。
若采纳字面规则，请将裁决读作 **D — UNRESOLVED（overshoot 未在协议覆盖范围内）**。

---

## 15. Recommendation

```text
HOLD
```

- **不进入 E2 配置阶段**；不创建任何 candidate-density config、不改 anchor / center rule /
  stride / TAL / P2/P3、不训练、不提交。
- 本审计只回答一个问题：**candidate 数量足以解释 native-small 相对 large 的 geometry deficit；
  small 自身的候选质量不是瓶颈。**
- **未回答、也不应据此推断**：提高 small 的 anchor 密度能否在**训练后**改善 small 的实际指标。
  相关既有证据：`reports/next_optimization_review.md` 已记录「加检测尺度」家族 3/3 证伪；
  §12.5 的外推**不构成**对该家族翻案的依据。

---

## 附录：产物与复现

| 文件 | 说明 |
|---|---|
| `_run.py` / `RUN.log` | 逐候选抽取（10,105 GT / 1,905,266 候选）；含 runtime hook 与自检 |
| `_results.npz` | `gt_*`（uid/native_area/bw/bh/ncand/maxciou/maxcls/maxalign/zero）+ `cand_*`（gt/ciou/cls/align/stride/offx/offy） |
| `_analyze.py` / `ANALYZE.log` | 反事实分析 + 分布 + MC 不确定度 + CASE 判定 |
| `_tables.json` | 结构化结果 |
| `_meta.json` | provenance / 自检 / 输入口径 |

复现顺序：

```bash
python -X utf8 -u diagnostic/candidate_density_counterfactual/_run.py --n-images 250 --epochs 3
python -X utf8 -u diagnostic/candidate_density_counterfactual/_analyze.py
```
