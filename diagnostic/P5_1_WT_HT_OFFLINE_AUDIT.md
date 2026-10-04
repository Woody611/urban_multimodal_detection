# P5.1 — Correct W_t × h_t Offline Audit

**日期** 2026-10-03 · **性质** 只读 / 零 GPU / 零训练 / 零 forward · **incumbent** D′ (0.515281)

> 标注：**[FACT]** artifact 直接支持 ／ **[INFERENCE]** 多 FACT 推出 ／ **[HYPOTHESIS]** 无直接证据

---

## 1. Artifact Integrity

### 1.1 `from_snapshots/` —— **不存在于本地**

```
ls diagnostic/full300_trajectory_probe/from_snapshots/  →  No such file or directory
```

P5.1 指定的那个 artifact（30 个带**正确 `head_W`** 的快照 npz）**未同步**。

### 1.2 本地实际可用的 19 个 npz（来自 FL-300 那轮）

| 项 | 结果 |
|---|---|
| epoch 数 / 编号 | 19 个：`0,1,2,5,10,20,30,50,75,100,125,150,175,200,225,250,275,289,299` |
| `head_W` shape | (12, 256) —— 19 个全部一致 |
| `head_b` shape | (12,) |
| `vec_h` shape | (1320, 256) |
| finite | **19/19 全部 True**（无 NaN/Inf） |
| `vec_keys` 跨 epoch 一致 | **19/19 True** |

### 1.3 W 是否随 epoch 变化 —— **§2 的 STOP 条件触发**

```
唯一 W hash 数 = 1 / 19        ⇒  W 恒定（陈旧）
唯一 b hash 数 = 1 / 19
```

而这个恒定 W **不是** `W_final`：

| | 值 |
|---|---:|
| `max|W_npz − W_final|` | 0.1210 |
| `‖W_npz‖` | **2.0146** |
| `‖W_final‖` | **2.5316** |
| `max|b_npz − b_final|` | 0.2511 |

**[INFERENCE]** 该恒定 W 是 `attach_model` 时刻（`on_pretrain_routine_end`，**训练前**）读到的值
⇒ 它就是 **`W_0`**，不是任何 `W_t (t>0)`。

```text
§2 STOP 条件触发 ⇒ 不得用这些 npz 做 live-head (Readout A) 分析
```

**但**：`W_0` 恒定反而使 **Readout C（frozen initial head）完全可用**，
配合 `best.pt` 的 **Readout B（frozen final head）**，三个读数里能拿到两个。

---

## 2. Identity Alignment

```text
identity_alignment = PASS
```

- 19 个 npz 的 `vec_keys` **逐位相同**，与 `medium_cells_canvas.json` 的 1320 个 key 顺序一致。
- class / size / geometry 由 label 文件独立重建（不依赖 npz），与 key 一一对应。
- **独立校验**：用手算 `W_0 @ vec_h + b_0` 对比 npz 的 `logit` 列 →
  `max|Δ| = 7.11e-15` ⇒ 列定义与我的实现完全一致（float64 舍入）。

---

## 3. Live-head trajectory (Readout A)

```text
NOT COMPUTABLE —— W_t 不存在于任何本地 artifact
```

唯一可用的 live 代理是 P5 已测的 `cls_center_max`（**真实 forward**，GT 中心邻域 anchor 的最大 sigmoid）：

| ep | 5 | 20 | 100 | 200 | 299 |
|---|---:|---:|---:|---:|---:|
| 中位 | 0.529 | 0.585 | 0.740 | 0.839 | 0.866 |
| dead(<1e-3) | 2.9% | 3.8% | 6.9% | 10.9% | **13.0%** |

⚠ **口径警告**：`cls_center_max` 是**邻域最大值**，而 Readout B/C 用的是**GT 单元格点值**。
两者**level 不可直接比较**（单元格点值系统性更严）。下文只比较**趋势**。

---

## 4. Frozen-final trajectory (Readout B)

`L_B(t) = W_final · h_t + b_final`（单点）：

| ep | 0 | 5 | 20 | 100 | 200 | 299 |
|---|---:|---:|---:|---:|---:|---:|
| GT logit 中位 | −5.564 | +0.149 | +0.026 | −0.063 | +0.144 | −0.801 |
| dead(<1e-3) | 41.1% | 12.7% | 15.5% | 23.9% | 28.4% | **35.1%** |

**[FACT]** B 的 dead 率从 ep5 的 12.7% 增长到 ep299 的 35.1%（**2.8×**）。

## 5. Frozen-initial trajectory (Readout C)

`L_C(t) = W_0 · h_t + b_0`：

| ep | 0 | 5 | 20 | 100 | 200 | 299 |
|---|---:|---:|---:|---:|---:|---:|
| GT logit 中位 | −2.097 | +0.146 | +0.521 | +1.206 | +2.167 | **+2.279** |
| dead(<1e-3) | 13.1% | 4.2% | 4.1% | 4.9% | 6.3% | **8.0%** |

**[FACT]** C 的 GT logit **单调改善**，dead 率仅 4.1%→8.0%（**1.9×**）。

### 5.1 三读数并排（dead 率）

| ep | LIVE (邻域max, 真 forward) | C: W_0 (单点) | B: W_final (单点) |
|---:|---:|---:|---:|
| 5 | 2.9% | 4.2% | 12.7% |
| 100 | 6.9% | 4.9% | 23.9% |
| 299 | **13.0%** | **8.0%** | **35.1%** |
| 增长倍数 (ep5→299) | **4.5×** | 1.9× | 2.8× |

**[FACT]** **两个固定读数都复现不出 live 的极化幅度**：C 几乎不极化（1.9×），B 极化但水平被邻域口径抬高。
**[INFERENCE]** 极化**依赖 live 的联合路径 W_t × h_t**，不能由任一固定 head 单独解释。

---

## 6. GT vs max-other vs margin

用两个极端 head 在 ep299 算 GT 类 / max-other / margin（单点）：

| readout | GT logit | max-other | margin |
|---|---:|---:|---:|
| C: W_0 | +2.28 | （见下） | — |
| B: W_final | −0.80 | （见下） | — |

（P5 已在 square 口径给出 dead GT 的 margin **+3.58**、max-other sigmoid≥1e-3 仅 **2.2%** ⇒ **Pattern A：全局真空**。）

**§9 四模式判定：Pattern A（全局 score vacuum）** —— GT logit ↓ 且 other logits 也 ↓，margin 不塌。

---

## 7. Representation / Head / Interaction decomposition

```text
不可能 —— 需要 W_t。§8 的对称 Shapley 分解全部依赖 (W_t, W_0, W_final) 三者。
```

**替代方案（两个可算的 counterfactual，各自数学精确，但不可相加）：**

- **纯表征**（固定 W_0，h_0→h_299）：`R = W_0·(h_299 − h_0)` 的 GT 分量
- **纯 head**（固定 h_299，W_final→W_0）：`H = (W_0 − W_final)·h_299 + (b_0 − b_final)` 的 GT 分量

---

## 8. Dead-tail analysis

**无法做**：live dead 需要 W_t；event alignment（dead-entry epoch）需要 live 的状态定义。

可报的是固定读数下的 dead 轨迹（§4/§5）。**UNKNOWN。**

## 9. New-face event analysis

P5 的 Jaccard（ep10∩ep299 = 0.242、ep20∩ep299 = 0.261、ep100∩ep299 = 0.503）是**用 live 代理**算的，
本轮**无法用正确 W_t 复核**。**UNKNOWN。**

---

## 10. Class / Size / Border / Density attribution

两个 counterfactual 的 GT 分量（ep299，单位 nats，全部 >0 表示"改善"）：

| cohort | n | **纯表征 Δ**（固定 W_0） | **纯 head Δ**（固定 h_299） |
|---|---:|---:|---:|
| 小 (sqrt<49) | 436 | +3.82 | +3.36 |
| 中 | 448 | +4.33 | +3.35 |
| 大 | 436 | +3.33 | **+7.50** |
| 近边 <0.126 | 221 | +3.39 | +5.32 |
| 远边 | 1099 | +3.90 | +3.95 |
| person | 482 | +2.70 | +2.68 |
| animal | 359 | +4.02 | +3.27 |
| car | 167 | +5.63 | +3.93 |
| sign | 84 | **+9.58** | +6.32 |
| light | 102 | +5.32 | **+8.33** |
| bicycle | 50 | +2.89 | +5.15 |
| garbage_can | 24 | **+9.57** | +3.16 |

**[FACT]** **两个 counterfactual 对 12/12 个 cohort 全为正** —— 在固定 head 下，表征漂移**改善**所有 cohort；
在固定 h 下，回到 W_0 也**改善**所有 cohort。

**[INFERENCE]** 因此**极化不可能来自任一单侧**：单侧效应全是"变好"。
极化的来源必须落在 **W_t 与 h_t 的联合演化**上。这与 §5.1 的结论一致。

**[FACT]** 大目标的 **head 效应**最大（+7.50，约为小的 2.2×）；近边组的 head 效应（+5.32）也高于远边（+3.95）。

---

## 11. Common-mode vacuum analysis（§16）—— 本轮最强的结果

在 ep299 上，把每 GT 的 12 类 logit 减去其**12 类均值**（去掉共同模）：

| readout | 12 类均值中位 | GT logit 中位 | **类居中 GT logit 中位** | **居中后 <0 占比** |
|---|---:|---:|---:|---:|
| C: W_0 | −8.72 | +2.28 | **+10.86** | **2.1%** |
| B: W_final | −14.51 | −0.80 | **+12.92** | **2.6%** |

**[FACT]** 在两个**极端不同的**分类头下：

- GT 类相对 12 类均值高出 **+10.9 ~ +12.9 nats**
- 只有 **2.1% / 2.6%** 的 GT 其类居中 logit 为负

⇒ **在 GT 单元格点上，类别判别几乎完美；"score death" 是共同模（整体电平）现象，不是判别失败。**

**[FACT]** 共同模电平**强烈依赖 head**：同一个 `h_299`，W_0 给 −8.72、W_final 给 −14.51（差 **5.79 nats**）。

**[INFERENCE]** ⇒ 共同模 = `(mean_c W_c)·h + mean_c b_c`，**head 对其有实质贡献**，
因此单靠表征侧无法解释 score death。

---

## 12. Negative Evidence

1. **两个 counterfactual 对所有 cohort 全为正** ⇒ 任何"单侧驱动"叙事被否（§10）。
2. **类居中后 <0 仅 2.1%/2.6%** ⇒ "分类判别失败"被否；P5 的"真空"结论在**两个极端 head** 下均成立，**不是某个 head 的偶然**。
3. **C（W_0）几乎不极化（1.9×），B 极化但基线被口径抬高** ⇒ 极化不能由固定 head 复现（§5.1）。
4. **口径混淆必须承认**：live 是邻域 max、B/C 是单点 ⇒ 三者的 **level 与 dead 率不可直接相减**，只能比趋势。

---

## 13. Mechanism Matrix

| Mechanism | Live evidence | Frozen-readout evidence | Decomposition | Temporal precedence | Strength | Status |
|---|---|---|---|---|---|---|
| **Representation-driven** | 有（live 极化） | **否**（C 几乎不极化；两 counterfactual 全为正） | **不可算** | 不可排序 | **Weak** | 不成立 |
| **Head-driven** | 有 | **否**（B 也复现不全） | **不可算** | 不可排序 | **Weak** | 不成立 |
| **Coupled** | 有 | **两个固定读数都复现不全** ⇒ 需要联合路径 | **不可算** | UNKNOWN | **Moderate** | 方向上最一致 |
| **Common-mode vacuum** | 有 | **是**（两极端 head 下居中 GT 均为 +11~+13） | 无需分解 | — | **Strong** | **确立** |
| **Class/size modulation** | 有 | 是（大目标 head 效应 +7.5 最大） | — | — | **Moderate** | 调制器 |

---

## 14. What This Changes From P5

- **新增确立**：score death = **共同模电平**现象（类居中 GT logit +11~+13、仅 2% 为负），
  在 **W_0 与 W_final 两个极端 head 下都成立** ⇒ 这不是某个 head 的偶然，也不是判别失败。
- **新增否证**："单侧驱动"（纯表征 或 纯 head）—— 两个 counterfactual 对全部 12 个 cohort 全为正。
- **新增定位**：共同模电平对 head 高度敏感（同 h 下 W_0 与 W_final 差 5.79 nats）。
- **未变**：Assignment 否证、极化、Jaccard 新面孔等 P5 结论不受影响（本轮不与之冲突）。

## 15. GPU Decision

```text
NO GPU
```

§20 五项逐条：

| 条件 | 满足？ |
|---|---|
| 1. W_t/h_t identity verified | ❌ **W_t 不存在**（`from_snapshots/` 未同步；本地 npz 的 W 恒定=W_0） |
| 2. decomposition numerical identity exact | ❌ 无法构造（缺 W_t） |
| 3. mechanism outcome-independent | ✅ |
| 4. temporal evidence coherent | ❌ |
| 5. mechanism has a specific intervention | ❌ |

**且按 §20 末段**：即便机制是"common-mode classifier drift"，**也不知道训练中什么造成 drift** ⇒ 仍然 NO GPU。

## 16. Final Verdict

```text
VERDICT D — Artifact limitation
```

缺的是**唯一一个**能完成 §8 分解的变量（`W_t`），而它**已存在于云端**：
`from_snapshots/epoch_*.npz`（30 个，~78 MB，修复后跑的，带正确 `head_W`）。

**本轮的净收获（不依赖该缺失变量）**：
> score death 是**共同模电平**现象而非判别失败 —— 已在两个极端分类头下各自验证。
> 且单侧 counterfactual 全部为正 ⇒ 极化位于 **W_t × h_t 的联合路径**上，不是任一侧。

**这不是申请 GPU**：它是"同步一个已存在的文件"。在 W_t 到位之前，机制问题无法推进；
到位之后，§7 的对称分解可以**完全离线**完成。

---

## 附：本轮未做（因 artifact 缺失）

- §7 对称 Shapley 分解（需 W_t）
- §8 dead-tail / §9 new-face event alignment（需 live 状态定义 = 需 W_t）
- Readout A 的任何逐 epoch 量
