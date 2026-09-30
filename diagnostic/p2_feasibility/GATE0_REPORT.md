# Gate 0 — Small-Object Root Cause + P2@1280 Feasibility Audit

**性质**：只读审查。0 训练 / 0 新推理 / 0 代码修改 / 0 配置修改 / 0 提交。
**产物**：`_small_root_cause.py`、`_small_root_cause.json`、本报告。全部只写 `diagnostic/p2_feasibility/`。

---

## 1. Executive conclusion

```text
NO-P2
```

**small-object 的首要问题不是 feature resolution。**

三条独立证据指向同一方向：

1. **在小目标内部，尺寸不区分成功/失败** —— miss 率在 0–20px 几乎平坦（31.2/31.2/27.0/26.9%），仅 ≥20px 才降到 12.6%；**类别**才区分（同尺寸下 animal 41.3% vs person 19.5%）。`sign` 更是完全不随尺度改善（16.7/15.5/**19.4%**）。
2. **P2@1280 已经在本数据集、本分辨率上真实执行过并被关闭** —— 官方口径 **0.50232 vs 0.50928 = −0.00695**，而该差值**恰等于 baseline 自身 9 个 checkpoint 的散布**（max−mean = 0.00680）⇒ NULL，非有害。终裁原文：`P2_REJECTED — 关闭 P2 方向`。
3. **它不是低风险实验** —— 需要为 3-stem 布局**新写一个 checkpoint 手术脚本**（新代码，非改配置），且推理侧 `max_nms`/`max_det` 截断行为会改变（见 §7 R5 / §8）。

同时**纠正我自己上一轮报告中的一个错误**：我说「89.8% 不成立」是**错的** —— 我测的是另一个三元组。按原始定义重算，**89.8% 完全复现**（见 §4）。

---

## 2. Small-object error taxonomy（D′，val n=371 small GT）

方法：`scripts/official_eval.py` 的 `load_split` / `read_pred_txt` / `apply_max_boxes` / `box_iou_np` 全部直接 import，不重新实现。small = native `area < 1024`。

| 类型 | 占比 | 判据 |
|---|---:|---|
| OK | **50.1%** | 同类 IoU≥0.5 且 conf≥0.1 |
| D — 定位好但置信低 | 10.0% | 同类 IoU≥0.5，conf<0.1 |
| L — 定位误差 | 12.4% | 0.1≤IoU<0.5 |
| C — 分类错误 | 0.8% | 其他类别 IoU≥0.5 |
| **H — 硬漏检** | **26.7%** | H2 83 / H1 14 / H4 2 |

**L 的子类**：L1 偏大 **30** / L3 中心偏移 5 / L2 偏小 4 / L4 长宽比 3。

### ⚠️ H2 不是「围绕 GT 的大框」——是虚假匹配

您要求复核的历史观察（「best same-class box 明显大于 GT」）。**数字对，机制解释错**。D′ 的 83 条 H2：

| 实测 | |
|---|---:|
| pred/GT 面积比 中位 | 9.92× |
| pred 面积 > GT 占比 | 94.0% |
| **中心偏移（GT 宽度为单位）** | **21.5 倍** |
| **任意类别** pred 最大 IoU | **0.000** |
| pred 置信度 中位 | **0.888** |
| pred 自身尺度 | small 18 / medium 38 / **large 27** |

那个「大框」在图像**完全不同的位置**、高置信、属中大目标。它只是「同类别中 IoU 最高者」，纯属虚假对应。
**⇒ H2 实质等同 H1。** 诚实的口径：

| 真实失败模式 | 占比 |
|---|---:|
| **无任何预测重叠**（任意类别 IoU<0.1） | **24.5%** |
| 同类弱重叠 0.1–0.5 | 12.4% |
| 同类良好但低置信 | 10.0% |
| 跨类重叠 | ~2.2% |
| 正常 | 50.1% |

### 尺度趋势（D′）

| | 无重叠率 |
|---|---:|
| small | **24.5%** |
| medium | 9.6% |
| large | 3.2% |

---

## 3. Small-object class breakdown

| class | small n | miss% | medium n | miss% | large n | miss% | 中位输入尺寸 |
|---|---:|---:|---:|---:|---:|---:|---:|
| person | 195 | 19.5% | 482 | 8.5% | 368 | 1.6% | 16.49 px |
| animal | 63 | **41.3%** | 359 | 7.8% | 309 | 3.2% | 15.10 px |
| **sign** | 30 | 16.7% | 84 | 15.5% | 36 | **19.4%** | 15.53 px |
| car | 18 | 27.8% | 167 | 2.4% | 103 | 1.9% | 33.87 px |
| uav | 18 | **0.0%** | 12 | 8.3% | 0 | — | 35.72 px |
| bicycle | 12 | 25.0% | 50 | 10.0% | 44 | 0.0% | 15.69 px |
| light | 11 | **54.5%** | 102 | 21.6% | 131 | 4.6% | 19.00 px |
| seat | 9 | 44.4% | 17 | 29.4% | 81 | 2.5% | 19.56 px |
| ball | 6 | 50.0% | 11 | 27.3% | 2 | 0.0% | 11.81 px |
| garbage_can | 6 | 16.7% | 24 | 20.8% | 27 | 11.1% | 14.82 px |
| boat | 3 | 0.0% | 12 | 0.0% | 13 | 0.0% | — |
| tricycle | 0 | — | 0 | — | 2 | 0.0% | — |

**关键对比**（尺寸几乎相同、miss 率差 2.1×，~3.5 SE）：
- `animal` 15.10px → 41.3%
- `person` 16.49px → **19.5%**

**`sign` 的 miss 率不随尺度下降**（16.7 / 15.5 / **19.4%**）⇒ 该类是外观/可判别性问题，与分辨率无关。

**绝对贡献**：person 占 small GT 的 52.6%，贡献 38/91 = 41.8% 的 miss；animal 贡献 26/91 = 28.6%。

---

## 4. Common-hard-miss verification

### ⚠️ 先纠正我上一轮的错误

我此前用 **D′ / OASA2.0 / OASA1.4** 三元组算出 55.8%，并据此说「89.8% 不成立」——**这是错的**。89.8% 来自**完全不同的三元组**（模态消融）与**不同定义**。两者不可比较。

| | 原始 89.8% | 我上一轮的 55.8% |
|---|---|---|
| 三元组 | `RGBID_baseline` / `RGBID_base_ir_dropped` / `RGBID_base_depth_dropped` | D′ / OASA2.0 / OASA1.4 |
| 问的问题 | full 模型的 FN 有多少是**跨模态配置**共同漏掉 | 漏检有多少是**跨增广/融合-stem 配置**共同漏掉 |
| FN 定义 | 逐类贪心 IoU≥0.5 未匹配 | 任意类别 IoU<0.1 |

### 原始 89.8% 的独立复算 —— **完全复现**

用原三元组、原定义、**不排除损坏图**（原表注明「本表未排除」）：

| | 我的复算 | 原审计 |
|---|---:|---:|
| GT 总数 | **2847** | 2847 ✓ |
| Full FN | 527 | 522 |
| 三配置共同漏掉 | 473 | 469 |
| **3-way common miss rate** | **89.8%** | **89.8%** ✓ |

> **89.8% 成立**（ΔFN 5 个来自贪心匹配的次序细节，比率逐位相同）。

### 结论的含义

**89.8% 的漏检跨模态配置不变** ⇒ 这些失败**与 IR/Depth 无关**，换模态配置救不回来。
但这**不**说明它们与 stride 有关 —— 模态无关 ≠ 分辨率相关。

---

## 5. Recall vs localization diagnosis

```text
Case M（Mixed），但失败模式上 recall 占优、机制上不是分辨率
```

- **失败模式**：无候选 24.5% > 定位 12.4% > 低置信 10.0% > 分类 ~2.2%
- **驱动因素**：不是尺寸（0–20px 平坦），是类别/外观
- **限制（必须同读）**：val 上 small 的尺寸分布很窄（P25–P75 = 13.3–19.8px），动态范围小，**可能压不出尺寸梯度** ⇒ 这是「不支持纯分辨率解释」，不是「证明分辨率无关」

---

## 6. Current model pyramid（`configs/yolo11m_sepstem.yaml` @1280）

| idx | module | out stride | H×W @1280 | ch |
|---:|---|---:|---:|---:|
| 0 | Silence→Identity | 1 | 1280 | 5 |
| 1/3/5 | SilenceChannel（切通道） | 1 | 1280 | 3 / 1 / 1 |
| 2 | Conv（RGB stem） | 2 | 640 | 48 |
| 4 / 6 | Conv（IR / Depth stem） | 2 | 640 | 8 / 8 |
| **7** | **Concat[2,4,6]** | 2 | 640 | **64** ← stem 融合点 |
| **8** | Conv [128,3,2] | **4** | **320** | **128** |
| **9** | **C3k2 [256,False,0.25]** | **4** | **320** | **256** ← **P2 骨干特征** |
| 10/11 | Conv + C3k2 | 8 | 160 | 256/512 |
| 12/13 | Conv + C3k2 | 16 | 80 | 512 |
| 14/15 | Conv + C3k2 | 32 | 40 | 512 |
| 16/17 | SPPF + C2PSA | 32 | 40 | 512 |
| 20 | C3k2（P4 颈） | 16 | 80 | 512 |
| **23** | **C3k2（P3 颈）** | **8** | **160** | **256** ← Detect 输入①|
| 26 | C3k2（P4 颈） | 16 | 80 | 512 ← Detect 输入② |
| 29 | C3k2（P5 颈） | 32 | 40 | 512 ← Detect 输入③ |
| 30 | **Detect [23,26,29]** | — | — | ch=(256,512,512) |

**stride-4 feature 已经存在，且已经物化可直接取用**：idx 9（C3k2，256ch @320）。
它被 idx 10 通过 `-1` 引用 ⇒ 落在 `self.save` 中 ⇒ `y[9]` 在 `_predict_once` 里已materialize，**可直接作为 `from` 目标，无需新增任何前向层**。

---

## 7. P2@1280 code feasibility

| 环节 | 结论 |
|---|---|
| **model** | 需在 Detect 前插入 P2 分支（Upsample(idx23) → Concat(,idx9) → C3k2[256,False]），Detect 输入改 `[...]` |
| **Detect** | ✅ **原生支持**：`nl = len(ch)`、per-level `ModuleList`、`forward` 循环 `range(self.nl)`、`self.stride` 由 dummy forward **实测**。**全仓无 3-level 硬编码**（唯一 `nl == 4` 在 `detect/train.py:277`，是 fork 的 4-level remap 表，说明 4-level 路径已存在） |
| **loss** | ✅ **零改动**：`v8DetectionLoss` 全程用 `self.stride.size(0)` 与列表推导，无 `range(nl)` |
| **assigner** | ✅ **零改动**：`TaskAlignedAssigner` 完全 level-agnostic（全部按 flat anchor id 索引，无 stride / 无 nl）。`make_anchors` 只需 `feats` 与 `stride` 同序 |
| **inference** | ✅ 结构上通用；⚠️ 但见下方 R5 |
| **evaluator** | ✅ 无需改（只看预测 TXT） |

**已被本仓库端到端验证过**：`reports/rgbid_p2_probe.md` 记录真实 run 输出 `Detect(nc=12, nl=4)`, `stride=[4,8,16,32]`, `(1,16,136000)` 且 `136000 == 320²+160²+80²+40²`。

### ⚠️ R1 — 真正的 blocker：`pretrained: yolo11m.pt` 对 4-level sepstem **必然硬失败**

`_remap_separate_stem` 按**索引差**映射 fusion Concat 之后的每一层。stock yolo11m 只有 24 层：

- 新增的 P2 颈层没有对应物 → `train.py:175-176` 抛 `RuntimeError`
- 且 Detect 索引从 30 移位，**所有 Detrect 键也会抛**（`train.py:165-174` 的例外只覆盖 `sv is not None`，不覆盖新层）

**唯一出路**：先造一个手术 remap 的 init checkpoint（仓库对 early-fusion 做过：`diagnostic/p2_probe/_make_init_scratch.py`），再让 `train.py:234-239` 的 early-exit 短路。**这需要为 3-stem 布局新写一份手术脚本** —— 是新代码，不是改配置。

### R5 — `max_nms` / `max_det` 截断行为改变

anchor 数 **33,600 → 136,000（4.05×）**。`ops.py:301-302` 的 `max_nms=30000` 剪枝会**频繁触发**；`max_det` 本身与层数无关，但 **P2 框尺度更小、置信更低，`max_det` 截断不再 level-neutral** —— 这正是您点名要警惕的风险。**其量级本轮未量化。**

---

## 8. Medium/large protection risk

**未建立**，不能声称安全：

- `TaskAlignedAssigner` 的 topk 是在**全部 anchor 上按 alignment metric 全局选**（`tal.py:157-190`），不设 per-level 配额
- 增加 25% 的 anchor（P2）后，P2 anchor 与 P3/P4/P5 **同池竞争**。中大目标的 P2 anchor 因预测框尺度不匹配而 alignment 低，**理论上**不会被选为正样本，但**本轮未量化验证**
- 反向风险同样存在：P2 anchor 稀释 topk 预算，**可能挤掉原本属于 P3 的 smal l正样本**

⇒ 这正是您 Part C 要求检查的风险点，**结论：存在且未量化**。

---

## 9. Compute / memory estimate

| 项 | Δ | 依据 |
|---|---:|---|
| params | **+741,196（+3.70%）** → 20,061,972 → 20,803,168 | 解析计算，且与仓库同类 run 实测逐位吻合 |
| 新 Detect 层 | 329,548 | cv2 188,736 + cv3 140,812 |
| 新颈 C3k2[512,256] | 411,648 | 与 probe 实测值一致 |
| **FLOPs @1280** | **+150.9 GFLOPs → ×1.55**（272.7 → ~423.6） | 同类 probe 实测 273.85 → 422.69（1.55×） |
| **epoch 时间** | **×1.03**（130.8s vs 127.2s） | probe 实测 —— FLOPs 1.55× 但时间只 1.03× |
| **显存** | **+2.5–3.5 GB @bs=8**（他人估算） | Tal assigner 中间张量随 `h*w` 线性增长（33,600→136,000） |
| batch=8 是否可跑 | A100-40G 预计可以（19G + ~3G） | 未实测 |

---

## 10. Gate decision

```text
NO-P2
```

**依据**：

| GO-P2 条件 | 判定 |
|---|---|
| ① small 问题明显包含 feature-resolution 成分 | ⚠️ **弱**：recall 确为最大失败模式（24.5%），但**小目标内部尺寸不区分**、类别才区分，`sign` 尺度不变 |
| ② backbone 有合理 P2 来源 | ✅ **满足**：idx 9 已存在且已物化 |
| ③ Detect/loss/assigner 原生支持 4-level | ✅ **满足**（零源码改动，已被 300 epoch 验证） |
| ④ 无中大目标正样本污染风险 | ❌ **未建立**（§8） |
| ⑤ 推理/evaluator 不需大改 | ❌ **部分**：源码通用，但 R5 截断行为改变**未量化**；且 R1 需**新写手术脚本** |

**决定性的一条（超出您五条条件之外）**：**P2@1280 已经在本数据集、本分辨率上执行过并被关闭** ——
官方口径 **0.50232 vs 0.50928 = −0.00695**，而该差值**恰等于 baseline 自身 9 个 checkpoint 的散布**（max−mean 0.00680）⇒ 与 checkpoint 选择噪声不可区分。终裁：`P2_REJECTED（无增益）—— 关闭 P2 方向`。

按您的 §D 定义：「如果分析表明主要问题不是 feature resolution … 那么不要做 P2」⇒ **NO-P2**。

⚠️ 我不声称「P2 一定无效」。我声称：**现有证据不支持把 P2 作为针对 small-object 问题的下一次训练**，且它不是低风险实验。

---

## 11. 下一步诊断方向（替代 P2）

证据指向：**失败由类别/外观驱动，不由尺寸驱动**。

**最值得先做的单件事**：把「无任何重叠」的 91 个 small GT 按**失败成因**分层，区分三类：
1. **遮挡/低对比/模糊** → 数据/标注问题，架构改不动
2. **邻近同类实例被检出、目标本身没检出**（instance confusion）→ 与 NMS/匹配有关，`diagnostic/small_object_error_attribution/` 已有 NMS-IoU 变体框架可复用
3. **同类同尺寸的对照类却能检出**（如 animal 41.3% vs person 19.5% @ 15-16px）→ 可判别性/容量问题

**为什么优先做这个**：`animal`（63 个 small GT，41.3% miss）与 `person`（195 个，19.5%）尺寸几乎相同而 miss 率差 2.1×（~3.5 SE）——
**这是全表信息量最大的一组对照**，且它直接排除了「分辨率」解释。

**已有资产可复用，不必新写**：`diagnostic/small_object_error_attribution/_attribution.py` + `_attr_rows.json`（已有 attribution harness），`_nms_ap.py` + `nms_iou0.5/0.6/0.8_md300`（已有 NMS 变体）。

**明确不建议**：
- 继续在 `scale`（OASA）/ 分辨率 / 检测尺度上加层 —— 三条路都已有证伪记录
- 靠 IR/Depth 补 small —— 89.8% 漏检跨模态配置不变，且 Depth 独有贡献在 small 上最低（0.92%）

---

## 12. 证据不足的声明

以下为**本轮未能判定**、不应作为决策依据的部分：

1. **尺寸效应无法排除** —— val 上 small 尺寸分布过窄（13.3–19.8px），压不出梯度。要严格判定需在**训练分布**（含 mosaic 的 0.5× 降采样，small 落到 ~8.6px）上度量，本轮未做
2. **§8 的中大目标正样本污染** —— 未量化
3. **R5 的 `max_det` 截断效应** —— 未量化
4. **89.8% 与 55.8% 的关系** —— 两者问的是不同问题，本轮只各自独立复算，未做联合解释

```text
本轮训练次数 = 0 ；新推理 = 0 ；提交 = 0
代码修改 = 0（仅新增 diagnostic/p2_feasibility/ 下的只读分析脚本）
```
