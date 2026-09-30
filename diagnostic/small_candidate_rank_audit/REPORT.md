# Small Candidate Rank 5~16 Feasibility Audit

**日期**：2026-09-28
**性质**：**纯离线只读**。不训练 / 不 backward / 不 optimizer.step / 不加载 optimizer /
不改既有源码·config·checkpoint·evaluator / **不重新 forward** / **不重新 augmentation** /
不改 assigner / 不创建训练 config / 不提交。

**唯一问题**：对 native-small GT，当前 4 个 geometry candidates 之外，第 5～16 个潜在 candidate
是否仍有足够的几何 / 分类质量，使「增加 candidate density」存在真实可利用空间？

---

## 0. 最终裁决

```text
D — UNRESOLVED
STATUS = INSUFFICIENT_DATA
REASON = Existing artifact contains only accepted candidates
         (select_candidates_in_gts == True) and therefore cannot distinguish
         geometric candidate scarcity from candidate-generation exclusion.
```

**没有重新跑模型**（现有 NPZ 已包含逐候选 CIoU / cls / align / stride / 归一化偏移，
足以做池内分析；但不足以回答「新增候选」问题 —— 见 §2/§3）。

---

## 1. Executive Conclusion

| 问题 | 回答 |
|---|---|
| **Q1** small 的第 5～16 个潜在 candidate **是否存在**？ | **在本 artifact 里只能部分回答**：只对 `n_cand ≥ k` 的 GT 存在。**中位 small GT 的 n_cand = 4**，其 rank 5～16 **在当前 grid∩框内根本不存在**。 |
| **Q2** 若存在，是否有足够 CIoU + class score 构成 practical headroom？ | 在**池内**存在的那些（仅 20.0% 的 small GT 拥有未选中候选）：CIoU 尚可（中位 0.593，56.1% ≥0.5）**但 class score 几乎全为 0**（中位 0.0000，仅 14.1% ≥0.1）⇒ **几何 headroom 有、TAL-useful headroom 很有限**。 |
| **Q3** 现有证据是否足以支持 candidate-density intervention feasibility？ | **不足**。artifact 只含「已被中心约束接受」的候选，**无法区分 §14 Situation A 与 Situation B**。 |

⇒ 主裁决 **D**。**若把池内观察误当作对 Q1 的回答**，它指向 **B（geometric headroom only）**——
但那回答的是**另一个问题**（池内未被 topk 选中的候选质量），**不是**「增加密度会带来什么」。

---

## 2. 数据充分性检查（协议 §2）

`diagnostic/candidate_density_counterfactual/_results.npz`：

```
keys 16；折后 native GT 1920；候选 324,474
  gt_uid / gt_native_area / gt_bw / gt_bh / gt_ncand
  gt_maxciou / gt_maxcls / gt_maxalign / gt_zero
  cand_gt / cand_ciou / cand_cls / cand_align / cand_stride / cand_offx / cand_offy
dtype：int64 / float64；声明值 gt_ncand 与 cand 实际条数逐 GT 一致（已校验 True）
```

具备：GT identity ✔ / candidate 顺序确定性 ✔（按 anchor 索引升序，可复核）/ CIoU ✔ / cls ✔ /
align ✔ / stride ✔ / 归一化偏移 ✔。

**关键判定 —— candidate 是「全部潜在候选」还是「已接受候选」？**

`_run.py` 的抽取语句是：

```python
idx = np.flatnonzero(mig[g])      # mig = last["mask_in_gts"][0]
```

`mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)` 要求 **anchor 中心严格落在 GT 框内**
（`bbox_deltas.amin(3) > eps`）。**框外 anchor 全部没有保存。**

⇒ **artifact 只含「已接受」候选。**

---

## 3. 关键限制：无法把 rank 5～16 凭空创造出来（协议 §3/§14）

「增加 candidate density」意味着**新增 anchor**（更细网格 / 更高分辨率检测层），
这些新 anchor 会产生**当前 artifact 里不存在的候选**。它们属于：

- **Situation A**：真实 grid 中**存在**这些候选，但当前 `select_candidates_in_gts` 的中心约束把它们滤掉；
- **Situation B**：当前 stride / grid **本身就没有**这些候选。

**本 artifact 无法区分 A 与 B**，因为它只保存了通过中心约束的那一批。
⇒ 协议 §3「如果只保存 `select_candidates_in_gts == True`，立即 STOP」的条件**成立**。
**未做任何插值 / 模拟 / 猜测来生成第 5～16 个候选。**

---

## 4. 数据实际支持的那部分（有界补充观察，**不改变主裁决**）

以下统计全部在**池内**（`mask_in_gts==True`）完成，逐格标注 eligible n。

### §4.1 资格（native-small，n=250）

| 条件 | n | 占比 |
|---|---:|---:|
| `n_cand ≥ 5` | 123 | 49.2% |
| `n_cand ≥ 8` | 69 | 27.6% |
| `n_cand ≥ 12` | 45 | 18.0% |
| `n_cand ≥ 16` | 33 | **13.2%** |
| **`n_cand > topk(10)`** | **50** | **20.0%** ← 只有这些 GT 存在「未被选中的池内候选」 |

`n_cand` 分布：min 0 / p50 **4** / p75 8 / p90 21 / max 166。

### §4.2 固定总体逐 rank（`n_cand ≥ 16`，**n = 33**，rank 之间可比）

| rank | CIoU med | mean | p25 | p75 | P(≥.5) | P(≥.7) | P(≥.8) | cls med | P(cls≥.1) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9642 | 0.9452 | 0.9314 | 0.9743 | 100% | 100% | 97.0% | 0.0000 | 33.3% |
| 2 | 0.9592 | 0.9394 | 0.9302 | 0.9698 | 100% | 100% | 97.0% | 0.0658 | 45.5% |
| 3 | 0.9583 | 0.9351 | 0.9240 | 0.9660 | 100% | 100% | 97.0% | 0.0113 | 48.5% |
| 4 | 0.9522 | 0.9304 | 0.9218 | 0.9606 | 100% | 100% | 93.9% | 0.0028 | 45.5% |
| **5** | 0.9518 | 0.9270 | 0.9198 | 0.9590 | 100% | 100% | 93.9% | 0.8193 | 75.8% |
| 6 | 0.9467 | 0.9250 | 0.9170 | 0.9576 | 100% | 100% | 93.9% | 0.6499 | 54.5% |
| 7 | 0.9395 | 0.9223 | 0.9126 | 0.9565 | 100% | 100% | 93.9% | 0.7762 | 66.7% |
| 8 | 0.9345 | 0.9201 | 0.9122 | 0.9529 | 100% | 100% | 93.9% | 0.7618 | 63.6% |
| 9 | 0.9335 | 0.9185 | 0.9106 | 0.9518 | 100% | 100% | 93.9% | 0.4330 | 51.5% |
| 10 | 0.9323 | 0.9165 | 0.9104 | 0.9483 | 100% | 100% | 93.9% | 0.8236 | 84.8% |
| 11 | 0.9263 | 0.9150 | 0.9071 | 0.9466 | 100% | 100% | 93.9% | 0.7491 | 63.6% |
| 12 | 0.9262 | 0.9136 | 0.9028 | 0.9466 | 100% | 100% | 93.9% | 0.8079 | 75.8% |
| 13 | 0.9257 | 0.9112 | 0.8957 | 0.9438 | 100% | 100% | 93.9% | 0.0210 | 48.5% |
| 14 | 0.9224 | 0.9081 | 0.8948 | 0.9422 | 100% | 100% | 93.9% | 0.0052 | 45.5% |
| 15 | 0.9198 | 0.9065 | 0.8922 | 0.9407 | 100% | 100% | 93.9% | 0.0986 | 48.5% |
| 16 | 0.9132 | 0.8778 | 0.8849 | 0.9382 | 97% | 97% | 90.9% | 0.5927 | 51.5% |

**读法**：**在这 13.2% 的「密集 small GT」上，rank 5～16 的 CIoU 与 rank 1～4 相当**
（0.9518@rank5 → 0.9132@rank16，对比 0.9642@rank1）。CIoU 几乎不随 rank 衰减。
**cls 则参差**（中位在 0.0000 与 0.82 之间跳动）——说明同一 GT 的不同 anchor 置信度差异极大。

### §4.3 池内 headroom —— **正确口径**（当前 topk=10 未选中的候选）

```
有未选中候选的 small GT：50 / 250 (20.0%)；未选中候选共 1,215 个（每 GT 中位 12）
未选中候选 CIoU : med 0.5930  p25 0.2621  p75 0.8808   P(≥.5)=56.1%  P(≥.7)=44.4%
未选中候选 cls  : med 0.0000                            P(≥.1)=14.1%  P(≥.3)=13.3%
双门槛 useful   : CIoU≥0.5 & cls≥0.1 → 13.9% ;  CIoU≥0.7 & cls≥0.1 → 13.8%
per-GT ≥1 个 useful 未选中候选：47/50 = 94.0%
未选中候选 stride 构成：stride8 81.1% / stride16 12.3% / stride32 6.6%
```

⚠ **必须先记录一个同义反复**（它是本轮的**结构性事实**，不是发现）：
Rank-A 按 CIoU 降序后，`max_CIoU@K` 对一切 K **恒等于 rank-1**（实测 Δ 全为 0.0000）。
⇒ **新增候选不可能提高 max** —— 当前 max 已由池内最优候选取得。
⇒ 候选密度若要起作用，**只能通过增加正样本数量**，不能通过抬高 max。
（§5.2 的 `max@K` 表在 `_analyze.py` v1 中给出，已被判定为该同义反复，不进入结论。）

### §4.4 分层

| 层 | n | n_cand med | elig16 | 有未选中 | 未选中 CIoU med | 未选中 P(≥.5) |
|---|---:|---:|---:|---:|---:|---:|
| <S1 (p25) | 63 | 3 | 13 | 17 | 0.8624 | 85.6% |
| S2 (25–50%) | 62 | 3 | 3 | 8 | 0.6811 | 60.3% |
| S3 (50–75%) | 63 | 5 | 9 | 11 | 0.4321 | 44.4% |
| >S4 (p75+) | 62 | 6 | 8 | 14 | 0.5496 | 54.2% |
| AR<0.5 | 62 | 4 | 4 | 7 | 0.5338 | 53.4% |
| 0.5≤AR<1 | 96 | 4 | 4 | 9 | 0.8514 | 93.5% |
| 1≤AR<2 | 75 | 6 | 15 | 21 | 0.5061 | 50.3% |
| AR≥2 | 17 | 17 | 10 | 13 | 0.6884 | 59.2% |

**没有任何一层是「全部 GT 都有 rank 5～16」**；每层的「有未选中候选」格都只覆盖该层的一小部分。
⇒ 不能把 §4.2 的结果外推到全体 native-small。

### §4.5 stride 分解（§8）

| stride | rank1–4 | rank5–10 | rank11–16 | rank5–16 占比 | CIoU med(rank5–16) |
|---:|---:|---:|---:|---:|---:|
| 8 | 642 | 309 | 161 | **63.6%** | 0.9119 |
| 16 | 152 | 162 | 77 | 32.3% | 0.9170 |
| 32 | 19 | 20 | 10 | 4.1% | 0.8128 |

⇒ rank 5～16 **并非**主要来自 stride 16/32（stride-8 仍占 63.6%），
所以**不能**把它解释成「需要更粗的 stride」；但也**不能**据此声称「提高 stride-8 密度」——
因为**当前 stride-8 的候选已经被全部收进池里了**（框内的都在），真正的问题是**框外/更细网格**，
而那正是 §3 无法回答的部分。

---

## 5. 三层 headroom 证据（协议 §11/§12）

| Level | 判据 | 实测 | 结论 |
|---|---|---|---|
| **L1 geometry availability** | rank 5～16 中 CIoU≥0.5 的比例 | 池内未选中候选 P(≥.5)=56.1% | 存在非零几何 headroom（**但仅对 20% 的 GT**） |
| **L2 meaningful headroom** | `median(max@8 − max@4) > 0` | **恒为 0.0000**（同义反复，§4.3） | **不成立** —— max 不可能因新增候选而上升 |
| **L3 practical headroom** | 新增 rank 的 cls 是否合理 | 未选中候选 **cls 中位 0.0000**，仅 14.1% ≥0.1；双门槛 useful 13.9% | **不成立** —— 几何 headroom ≠ training-useful headroom |

**§12 双门槛（池内 rank 5～16）**：

| 定义 | 全部 rank5–16 | rank5–10 | rank11–16 | per-GT ≥1 个 |
|---|---:|---:|---:|---:|
| CIoU≥0.5 & cls≥0.05 | 70.5% | 77.6% | 56.5% | 46.4% |
| CIoU≥0.5 & cls≥0.10 | 68.7% | 76.0% | 54.4% | 46.0% |
| CIoU≥0.7 & cls≥0.10 | 67.9% | 74.7% | 54.4% | 45.2% |

（对照：rank1–4 上同一双门槛的 per-GT 比例 = **84.4%**）
⚠ 该表的 per-GT 分母是**全部 250 个 small GT**（含 80% 根本没有 rank>10 候选的 GT），
因此 46% 这个数字是「全部 small GT 中，能在池内找到 1 个 useful 的 rank5–16 候选」的比例。

---

## 6. Mechanistic Interpretation

1. **对中位 native-small GT（n_cand = 4），rank 5～16 在当前 grid∩框内不存在**，
   而 topk=10 > 4 ⇒ 它的全部候选**已经被选中**。对它而言，**池内 headroom 恰好为 0**。
   这一条覆盖 **80%** 的 native-small GT。
2. **只有 20% 的 small GT 拥有未被选中的池内候选**（n_cand > 10）。对这些 GT，
   未选中候选的**几何**质量尚可（CIoU 中位 0.593），但**分类分数几乎全为 0**
   （cls 中位 0.0000，只有 14.1% ≥0.1）⇒ **它们大概率无法成为有效 TAL 正样本**
   （TAL 的 `align = cls^0.5 · CIoU^6`，cls→0 时 align→0）。
3. **`max` 通道被证明是封闭的**：Rank-A 下 max_CIoU@K 恒等于 rank-1 ⇒
   任何「增加候选数量」都不能抬高几何上限。密度若有用，只能靠**增加正样本数量**。
4. **真正的开放问题**是「框外 / 更细网格上的新 anchor 质量如何」——
   这正是 §3 无法回答的 Situation A vs B。

---

## 7. Limitations

1. **artifact 只含已接受候选**（`mask_in_gts==True`），框外 anchor 未保存 ⇒ 无法分离 Situation A/B。
2. 池内观察的**资格人口很小且非随机**：`n_cand ≥ 16` 只有 33 个 GT（13.2%），
   `n_cand > topk` 只有 50 个（20.0%）；这些是「框内容纳更多 anchor」的偏斜子集，
   **不能外推到全体 native-small**。
3. `_analyze.py` v1 的两个方法学错误（`max@K` 同义反复、逐 rank 跨群体比较）已在文件中标注，
   其输出不进入裁决；修正版为 `_analyze2.py`。
4. 单一 checkpoint（D′ best.pt）；全部为相关性描述，无因果干预。
5. 未做假设检验；但 `max@K ≡ rank-1` 是**恒等式**，不是统计问题。
6. native-small 样本量 250（其中「密集」子集 33）。

---

## 8. Final Verdict（协议 §16 四分类）

```text
D — UNRESOLVED
```

**依据**：现有 artifact 无法恢复「增加 candidate density 会带来的**新增**候选」，
因此无法区分**几何候选稀缺**与**候选生成时的排除**（§3/§15）。

**补充说明（不改变裁决）**：若仅就**池内**可回答的那部分而言，证据形态是
**几何 headroom 存在、但 class score 过低** ⇒ 形态上接近 **B — GEOMETRIC HEADROOM ONLY**。
但该观察回答的是「topk 是否在丢弃池内好候选」，**不是**「增加密度是否有用」——
两者不可互换，故不据此改判。

---

## 9. Recommendation（协议 §17/§20）

```text
HOLD
```

- 本审计**不支持**进入 candidate-density intervention feasibility 研究：
  关键证据（框外 / 更细网格上的新 anchor 质量）在当前 artifact 中**不存在**。
- **不得**从本报告推出任何具体改法。明确禁止的写法包括（协议 §17）：
  「应该增加 P2」/「应该增加 stride-8 anchor」/「P2 一定会提升 mAP」/「下一步训练 candidate-density model」。
- **若要真正回答 Q1–Q3**，需要一个**新的、明确声明的**数据采集步骤
  （例如保存框外 anchor 的候选几何），而那属于**新的审计任务**，需另行批准；
  **本轮不提出、不设计、不执行。**

---

## 10. Provenance / 合规

```text
MODIFIED_EXISTING_FILES = 0
TRAINING               = 0
BACKWARD               = 0
OPTIMIZER_STEP         = 0
CHECKPOINT_MODIFIED    = 0
FORWARD_RERUN          = 0        ← 全程只读 NPZ，未加载模型
```

读取的既有 artifact（只读）：

| 文件 | SHA256（前 16） |
|---|---|
| `diagnostic/candidate_density_counterfactual/_results.npz` | `7b28dc555c3fffb6` |
| `diagnostic/candidate_density_counterfactual/_meta.json` | `b14d348272e22478` |
| `ultralytics/utils/tal.py` | `aae7e8ac438f00cd` |
| `runs/…_sepstem_clahe/weights/best.pt` | `1cae45f75693f541` |

`git status --short` 见 §10 末（与上一轮结束时相同，**无本轮新增的 `M` 项**）。

**产物**：`REPORT.md` / `_analyze.py`（v1，含已标注的两个方法学错误）/ `_analyze2.py`（修正版，结论来源）/
`_tables.json` / `_tables2.json` / `ANALYZE.log`。
（相对协议 §18 的文件清单多了 `_analyze2.py` 与 `_tables2.json` —— 因为 v1 必须保留为过程记录，
修正版不能覆盖它。）
