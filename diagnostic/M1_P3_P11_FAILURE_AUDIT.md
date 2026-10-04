# M1 RGB-only — P3 / P11 Failure Attribution Audit

**日期** 2026-10-04 · **性质** ZERO GPU / 不重新推理 / 不补数据
**脚本** `diagnostic/m1_p3_attribution.py` · 产出 `diagnostic/batch1_eval/_m1_p3.json`

---

## Executive Verdict

**M1 的第一瓶颈是 raw candidate generation（A），且与 D′ 属同一 failure stage。**
M1 的 loss 有 **75%** 落在「最终输出里根本没有同类 IoU≥0.5 的候选」（18.38% of GT），而 **official matching 只占 0.86%**。
D′ 相比 M1 的优势**不在 small**（small TP 仅 +6/371）、**在 medium（+33）**⇒ **D′ 的 multimodal gain 与 small-object candidate failure 基本不重合**。
**没有可证伪的 RGB-side intervention** ⇒ **NO-GO**。

---

## Gate

```text
M1_PROVENANCE_GATE = PASS
```
400 val / 398 有效 / 2 corrupt / GT 2807 · D′ 与 M1 的 TXT stem 集合一致 · M1 覆盖全部有效 GT 图 · p3 join miss = 0/2807 · 同一 evaluator（`official_eval`，top-100、conf 降序、贪心 one-to-one、IoU 0.50–0.95）。

---

## P3 — M1 Candidate Failure（IoU = 0.50）

| bucket | count | % GT |
|---|---:|---:|
| **TP@.50（matched）** | **2267** | **80.76%** |
| **F3 official-match fail** | **24** | **0.86%** |
| **F0∪F1∪F2（no candidate）** | **516** | **18.38%** |
| TOTAL | 2807 | 100.00% |

```text
M1 的 F0 / F1 / F2 = NOT AVAILABLE
  —— 最终 TXT 是 post-NMS + post-top100，raw(layerA)/pre-top100(layerB) 只有 D′ 有。
  —— 按 brief：不为了拿到该字段而重新推理。**不补、不猜。**
```

### ★ 一个 M1-internal 的证据，用来在不重跑的前提下判断 F0/F1 谁主导

把「no-candidate 桶」的 **size 剖面**与 D′ 已知的 F0 / F1 剖面比形状：

| 量 | small 占比 | large 占比 | **small/large 比** |
|---|---:|---:|---:|
| D′ **F0**（raw 无候选） | 95/371 = 25.6% | 51/1116 = 4.57% | **5.60** |
| D′ **F1**（NMS 消失） | 30/371 = 8.09% | 51/1116 = 4.57% | **1.77** |
| D′ F2（top-100） | 11/371 | 0/1116 | — |
| **M1 no-candidate 桶** | **155/371 = 41.8%** | **80/1116 = 7.17%** | **5.83** |

**⇒ M1 的 no-candidate 桶的 size 剖面（5.83）几乎精确匹配 F0（5.60），远超 F1（1.77）。**
（D′ 同口径 no-candidate：148/371 = 39.9% / 97/1116 = 8.69% → 比 4.59，同样靠 F0。）

**这是 shape-matching 推断，不是测量。** M1 自身的 F0/F1/F2 **仍标 NOT AVAILABLE**。

---

## P3 — M1 vs D′

⚠ **口径说明**：D′ 有**两套**数：
1. `p3_disappearance_audit/attribution.json` —— **square 几何**、独立 `_run_infer.py`，**含 F0/F1/F2 细分**；
2. 本表 —— 从**冻结 TXT**（`sepstem_clahe/best_full/results`，**rect 几何**）按**同一约定**重算的三态。

**M1 的 TXT 是 rect ⇒ 只有 (2) 与 M1 严格可比。** 不可把 (1) 混进这张表。

| metric | D′（rect，同约定） | M1（rect） | Δ(M1−D′) |
|---|---:|---:|---:|
| TP@.50 | 2295 | 2267 | **−28** |
| F3 official-match fail | 17 | 24 | **+7** |
| F0∪F1∪F2（no candidate） | 495 | 516 | **+21** |

**D′ 原生 P3（square，只读不重算）**：`F0=301, F1=160, F2=12, F3=39, MATCHED=2295`。
（其 no-candidate 构成 = **F0 64% / F1 34% / F2 2.5%**。）

### Q1 / Q2 回答

- **Q1（M1 是否明显更多 F0？）**：**在 no-candidate 桶上多 +21 GT**（516 vs 495，+0.75pp）。**量级很小**，且 M1 自身的 F0 不可测。
- **Q2（F1/F2/F3 呢？）**：**F3 只多 +7**（24 vs 17）；F1/F2 对 M1 不可测（D′ 侧 F1=5.7%、F2=0.4%）。
- **Q3（M1 的 small failure 是否更严重 ⇒ D′ 的 gain 部分来自 small？）**：
  **不。small TP：D′ 222 vs M1 216 ⇒ D′ 只多 +6/371。**
  **明确写：D′ 的 multimodal gain 与 small-object candidate-generation failure 基本不重合。**

---

## P11 — M1 Longitudinal

```text
NOT AVAILABLE — no inference / no GPU
```
`runs/urban_multimodal_det_modality_m1_rgb/weights/` 只有 `best.pt` 与 `last.pt`，**无 `epoch*.pt`**；
产出 D′ P11 那套 per-GT 纵向 dump **必须重新推理**（brief 明令禁止）。
**本地仅有 `results.csv` 的 300-epoch val 曲线**（单标量轨迹）：best@ep268 = 0.54828、last = 0.53891、**11 个 epoch 落在 best±0.002 内（平台，非尖峰）**。
**⇒ CASE D（P11 unavailable）。不从 P3 推断 P11。**

---

## Size × Failure（IoU = 0.50）

| size | GT | TP@.50 | F3 | F0∪F1∪F2 | TP 率 | D′ 同口径 TP/F3/noCand |
|---|---:|---:|---:|---:|---:|---|
| **small** | 371 | 216 | **0** | **155** | **58.2%** | 222 / 1 / 148 |
| medium | 1320 | 1031 | 8 | 281 | 78.1% | 1064 / 6 / 250 |
| large | 1116 | 1020 | **16** | **80** | **91.4%** | 1009 / 10 / 97 |

**recall 曲线（M1）**：`official match recall`（= TP 率）如上；`raw / post-NMS / post-top100 recall` **NOT AVAILABLE**（无 layerA/layerB）。

**⇒ M1 的损失集中在 small（41.8% 无候选）与 medium（21.3%）；large 只有 7.2%。**
**F3 反向集中在 large（16/24）** —— 即"候选在但没匹配上"主要发生在大目标（很少见，0.86% 总占比）。

---

## Class × Failure（IoU = 0.50）

| class | GT | TP@.50 | F3 | F0∪F1∪F2 | TP 率 |
|---|---:|---:|---:|---:|---:|
| **sign** | 150 | 99 | 0 | **51** | **66.0%** ← 最低 |
| **bicycle** | 106 | 73 | 2 | 31 | **68.9%** |
| seat | 107 | 85 | 0 | 22 | 79.4% |
| **animal** | 731 | 588 | **13** | 130 | 80.4% |
| person | 1045 | 844 | 8 | 193 | 80.8% |
| light | 244 | 200 | 1 | 43 | 82.0% |
| garbage_can | 57 | 47 | 0 | 10 | 82.5% |
| ball | 19 | 16 | 0 | 3 | 84.2% |
| car | 288 | 256 | 0 | 32 | 88.9% |
| boat | 28 | 27 | 0 | 1 | 96.4% |
| uav | 30 | 30 | 0 | 0 | 100.0% |
| tricycle | 2 | 2 | 0 | 0 | 100.0% |

**⇒ 最弱三类 sign / bicycle / seat 全是小目标主导类。**
**★ F3 高度集中：`animal` 独占 13/24（54%）**，而 sign / bicycle 中 0–2 个 ⇒ **official matching failure 不是 generic 机制，是个别类的 niche。**

---

## D′_ONLY / M1_ONLY Linkage

| 组 | n | F3 | F0∪F1∪F2 |
|---|---:|---:|---:|
| **D′_ONLY**（D′ 救回） | 129 | 11 | **118（91.5%）** |
| **M1_ONLY**（M1 救回） | 101 | 5（D′ 侧） | **96（95.0%）** |

**⇒ Q1/Q2：两个方向的 rescue 都压倒性地发生在「对方在该 GT 上没有产出同类 IoU≥0.5 候选」这一状态上**，而非 official matching。

**⇒ Q3：是的 —— 这是一次同类型 generic failure 的【净交换】，不是新的 modality-specific capability。**
两侧的 gross flow（129 vs 101）远大于净额（+28），且两边的失败**类型相同**、只是**发生在不同的 GT 上**。

**⇒ Q4：未发现 multimodal-specific mechanism。**

---

## Mechanism Decision

```text
A — raw candidate generation
```
依据：
1. **E（official matching）被直接排除**：F3 = 24/2807 = **0.86%**，且高度集中于 animal（54%），非 generic；
2. **D（top-100）在 D′ 上只有 12/2807 = 0.4%**；M1 无法测（TXT 已 cap），但上限不可能超过 D′ 的同一环节；
3. **C（NMS）在 D′ 上是 F1 = 160/2807 = 5.7%**，是次要项；
4. **A 是主导项**：no-candidate 桶 18.38%，且其 **size 剖面 5.83 精确匹配 F0 的 5.60**（F1 只有 1.77）。

**⚠ 保留**：M1 自身的 F0/F1/F2 细分 **NOT AVAILABLE**；上述第 4 点是 shape-matching 推断。若要坐实，需要一次带 `layerA/layerB` 落盘的 M1 推理 —— **本轮禁止，不做。**

---

## GPU Decision

```text
NO-GO
```
**理由**：机制虽已识别（A，raw candidate generation，small/medium 主导），但**没有可证伪的 intervention**：

- 已试过并**带线上判决**关闭的同靶点路线：**OASA**（线上 1.4=48.134 / 2.0=47.61，均 < D′ 48.712）、**F1 native-small replay**（线上 48.204，−0.508）、**1536**（无 D′ lineage 证据 + 瓶颈不对齐）、**YOLO11l 容量**（本地 −0.0065）、**P2 加检测尺度**（配对 bootstrap CI 全含 0）；
- 本轮允许范围内**没有**新的、单变量、可证伪的候选 —— 其余（loss/optimizer/score-calibration/head/NMS/TTA/augmentation/class-weight/oversampling）**要么已被 P6/P11 从数学上排除，要么被明令禁止**。

**⇒ GO 需要"mechanism + observable + intervention + falsification"四件齐全，现在缺 intervention 与 falsification 两件。**

---

## Recommended Next Experiments（按证据排序，最多 3）

| # | 候选 | 优先级 | 依据 |
|---|---|---|---|
| **1** | **M4 RGB+IR / M5 RGB+Depth** | **最高（成本已沉没，只差评测）** | 唯一剩下能回答"任何模态对 RGB 有没有增量"的实验；**必须用同步后的 `predict_rect.py` 评测**（否则重演 M2a 的 percentile 事故） |
| **2** | **RGB small/medium candidate-generation intervention** | **低（NO-GO，仅记录）** | small no-candidate 41.8% 是真实靶点，但 5 条同靶点路线全部已关闭、其中 2 条有线上判决 ⇒ **不排实验** |
| **3** | **RGB localization intervention** | **最低（NO-GO）** | F3 仅 0.86% ⇒ 匹配环节不是瓶颈；大目标的"重新定位"是 D′ 的优势而非 M1 的短板 |
| — | 新 fusion architecture | **排除** | Phase A + GT-level attribution 已判 Case A 主导（净 +28/2807、无稳定 phenotype） |
| — | RGB classification/head intervention | **排除** | P6/P11 已从单调性与时间上排除 |

**⚠ 若日后要为 #2 立项，必须先补一个 artifact**：M1 带 `layerA/layerB` 落盘的推理（把 F0/F1/F2 从推断变成测量）。**这本身就是唯一缺的观测**，且成本远低于训练。

---

## HARD STOP

不训练、不推理、不 smoke、不改代码/配置、不启动 M4/M5。
