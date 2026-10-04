# D′ Checkpoint Selection Audit

**日期** 2026-10-03 · **性质** READ-ONLY / ZERO-TRAINING / ZERO-FORWARD / ZERO-SUBMISSION
**incumbent** D′ = SepStem + IR-CLAHE · official mAP50-95 = **0.515281**

## Verdict

```text
CHECKPOINT SELECTION: NOT JUSTIFIED
GPU: NO-GO
```

**理由**：D′ lineage 里**只有 2 个 checkpoint 有官方口径结果**（best 0.515281 / last 0.506170），
唯一可比的替代品**低 0.0091**；其余 30 个 `epoch*.pt` **没有任何官方证据**，
且按已有 val 曲线**没有一个能追平 best.pt**。既无"可重复优势"的证据，方向也相反。

---

## A. Checkpoint inventory

| 项 | 值 |
|---|---|
| 目录 | `runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/` |
| epoch 快照 | **30 个**：`epoch0.pt … epoch290.pt`（步长 10），各 80.9 MB，**未 strip** |
| 另有 | `best.pt`（40.6 MB，**已 strip**，`epoch=-1`）、`last.pt`（同） |
| 文件名 vs 内部 epoch | **30/30 一致**（读取 checkpoint 内 `epoch` 字段核验，非按文件名） |
| 内含对象 | 全部为 `DetectionModel`（`ck["ema"]`，即 **EMA**），与 D′ 同一 lineage |
| `best_fitness` | 每个快照内记录：ep240 = 0.58569 → **ep250 = 0.59269** → ep290 仍 0.59269（**再无提升**） |
| 重复/损坏 | 无（shape/finite/identity 全部一致） |

**P5.1/P5.2 用的 30 个 `from_snapshots/epoch_*.npz` 与这 30 个 `.pt` 是同一批**（同 epoch 网格）。

## B. Validation trajectory（已有合法结果，`results.csv`）

| 项 | 值 |
|---|---|
| n | 300（每 epoch 一次训练内 val） |
| **峰值** | **csv ep246 = 0-based 245**，mAP50-95 = **0.56690** |
| 距峰值 ≤0.002 的 epoch 数 | **1**（即峰值自身）⇒ **尖峰，不是平台** |
| 距峰值 ≤0.005 的 epoch 数 | 10 |
| 末 10 epoch 均值 | 0.55425（比峰值低 **0.01265**） |
| 30 个快照中最高的 | 0-based **280** = 0.56151（**仍低于峰值 0.0054**） |

峰值附近（0-based）：242 → 0.56359；**245 → 0.56690**；248 → 0.56318；250 → 0.55957；280 → 0.56151。

## C. Official trajectory

```text
NO EXISTING OFFICIAL RESULT（除 best.pt 与 last.pt 外）
```

全库已存在的 `_official.json` 只有 3 个：

| 来源 | checkpoint | official mAP50-95 | 是否 D′ lineage |
|---|---|---:|---|
| （P3 audit 复现） | **D′ best.pt** | **0.515281** | ✅ |
| `rectlate10_pair/Dp_last` | **D′ last.pt** | **0.506170** | ✅ |
| `rectlate10_pair/RL_last` | rectlate10 last | 0.507549 | ❌ 异 lineage |
| `box2_infer_val` | box2 | 0.509456 | ❌ 异 lineage |

**30 个 `epoch*.pt` 没有任何官方结果。** 按 §5，**不为它们补跑推理**。

## D. Best checkpoint comparison

| | checkpoint | 0-based epoch | val mAP50-95 | **official mAP50-95** |
|---|---|---:|---:|---:|
| **current** | `best.pt` | **245** | 0.56690 | **0.515281** |
| alternative（唯一有官方结果的 D′ checkpoint） | `last.pt` | 299 | 0.55522 | **0.506170** |
| Δ | | | +0.01168 | **+0.00911** |

**唯一可比的一个替代 checkpoint 在官方口径下也差 0.0091** —— 方向一致（val 与 official 同向）。

## E. P5 correlation（仅 descriptive alignment，不做因果）

30 个快照点上，已有 val metric 与 P5 量的相关：

| 与 val mAP50-95 的相关 | r |
|---|---:|
| live dead rate | **+0.792** |
| GT 居中 logit | **+0.961** |
| 12 类均值 logit | **−0.886** |
| W top-1 能量 | **+0.974** |

| 0-based | val | live dead | 12类均值 | GT居中 | top1能量 |
|---:|---:|---:|---:|---:|---:|
| 110 | 0.52475 | 23.1% | −11.96 | 11.28 | 41.1% |
| 200 | 0.54615 | 28.1% | −13.51 | 12.74 | 44.6% |
| 280 | **0.56151** | 34.6% | −14.46 | 12.99 | 45.2% |
| 290 | 0.55930 | 35.0% | −14.46 | 12.85 | 45.2% |

**⇒ P5 的这些量全部是"训练进度"的单调函数**，与检测指标**同向**变化。

**这否证了一个隐含假设**：score vacuum 并不是一个与 metric 最优点对应的"病理" ——
它的恶化与 metric 的改善**同向**。因此**不存在一个"vacuum 还没恶化、metric 又更好"的甜点 checkpoint**。
（这正是 §7 想找的东西，答案是**不存在**。）

## F. Final decision

```text
KEEP CURRENT D′ BEST
```

**§9 GO 六条件逐条：**

| 条件 | 满足？ |
|---|---|
| 1 已有 checkpoint | ✅（30 个） |
| 2 同一训练 lineage | ✅ |
| 3 evaluator / geometry / prediction pipeline 合法 | ⚠ 30 个快照**无任何官方预测产物** |
| **4 官方口径存在可重复优势** | ❌ **无证据**；唯一可比的 last.pt 低 0.0091 |
| 5 优势非单次异常 | n/a |
| 6 不需重训 | ✅ |

⇒ **NOT JUSTIFIED**。

**额外强化**：即便假设 val→official 单调（2/2 点一致），30 个快照中 val 最高的 0-based 280
（0.56151）仍比 best.pt（0.56690）低 **0.0054**；按 best/last 的换算比（0.78）约折合官方 0.0042，
而官方口径的 checkpoint 噪声带 sd ≈ 0.003–0.004 ⇒ **处于噪声边缘且方向为负，不构成"优势"**。

---

## 附：本轮**没有**做的事

- 未训练 / 未 resume / 未 fine-tune / 未 forward 任何模型 / 未生成任何预测 / 未提交
- 未修改模型 / 数据 / augmentation / evaluator / 阈值
- 未为 30 个 checkpoint 补跑官方评测（§5 明确禁止）
- 未因"dead rate 更低 / 几何更漂亮"而当作收益（§10）—— 事实上本轮发现这些量与 metric **同向**，更不构成收益理由
