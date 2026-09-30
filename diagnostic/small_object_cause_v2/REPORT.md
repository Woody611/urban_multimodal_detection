# SMALL_OBJECT_CAUSE_ATTRIBUTION_V2

**性质**：只读深度诊断。0 训练 / 0 推理 / 0 新 prediction / 0 代码修改 / 0 提交。
**新文件**：`diagnostic/small_object_cause_v2/{_v2_records.py,_v2_matched.py,_v2_visual.py,_v2_attribute.py,_v2_records.json,_v2_visual.json,_v2_attribute.json,REPORT.md}`
**修改的既有文件**：**无**（所有脚本直接 `import scripts/official_eval.py` 与 `ultralytics.utils.patches.imread`，不改动它们）

---

## 0. Executive conclusion

```text
HOLD
```

**三条最硬的结论：**

1. **「91」的口径被查清**：91 = **同类** IoU==0；本次严格定义「**任意类别** IoU==0」= **70**。差额 21 个是有其他类别预测重叠、但无同类预测的目标。（§1）

2. **可视性/遮挡/模糊/对比度**这一整族原因**基本被否定**：尺寸匹配对照下，missed 的对象**遮挡更少**（occ 0.0000 vs 0.0509）、**更清晰**（Laplacian 方差 1238 vs 1042）、对比度/梯度/边缘密度**无差异**（比值 0.87–1.08）；**34.3% 的 miss 按同一可计算 proxy 属于 CLEAR**。叠加 §6 的 animal 尺寸反向趋势 → **不是分辨率、也不是数据可见性问题**。（§3）

3. **⚠️ 新发现（本轮最重要）：Depth 模态在小目标处结构性无效。**
   depth 零值占比**随尺度单调**：small **100.0%** / medium 56.8% / large 13.5%；small 处 79.2% **完全为 0**。同图同尺寸随机位置只有 40.3%，全图 28.0% —— **不是裁剪偏差**。
   与「距离受限深度传感器」一致。**这解释了 Depth 独有贡献仅 1.98%，并且使模态消融中「Depth → 置 0」这一臂在 small 上近似恒等变换**（§3.4）。

---

## 1. 91-GT population verification

严格按本次定义重算（`_v2_records.py`，直接 import `official_eval` 的 reader / IoU / cap）：

| 口径 | 数量 | 占 small GT (n=371) |
|---|---:|---:|
| small GT 总数 | **371** | 100% |
| 有**任意类别**预测重叠（IoU>0） | 301 | 81.13% |
| **任意类别 IoU == 0**（本次定义） | **70** | **18.87%** |
| 同类 IoU == 0 | **91** | **24.53%** |
| 同类 0 < IoU < 0.5 | 57 | 15.36% |
| 同类 IoU ≥ 0.5 | 223 | 60.11% |

### 与历史「91」的差异来源

历史「91」的口径是「**没有同类预测与该 GT 有任何重叠**」= `same-class IoU == 0`，**不是**「无任一类别重叠」。两者差 **21 个**目标 —— 这些目标有**其他类别**的预测与之重叠（例如 person 的小目标被 car/animal 框到）。

⇒ 两个数字都对，但**问的问题不同**。本轮后续（§3–§11）一律以**严格 70** 为归因对象，并同时给出 91 口径的对照。

---

## 2. Error taxonomy

每条 small GT 一行完整 record，已落盘 `_v2_records.json`，字段含：
`image_id / gt_id / cls / name / native_w,h / native_area / native_diag / sqrt_area / input_w,h / img_w,h / scale / best_same_class_iou / best_any_class_iou / best_same_class_conf / best_any_class_conf / best_any_class_pred_cls / best_{same,any}_class_area_ratio / center_dx_norm_gtw / center_dy_norm_gth / n_pred_total / n_pred_same / neighbor_gt_count / neighbor_same_class_count / neighbor_other_class_count / nearest_{same,other}_class_distance(+_norm) / near_{same,other}_class_detected / self_detected / border_margin_norm / aspect / n_pred_conf_ge_{0.001,0.01,0.05}`

> ⚠️ 上一轮我曾报告 H2「同类框明显大于 GT、overlap 很低」。本轮复核：**数字对，机制解释错**。
> D′ 的 83 条 H2，pred/GT 面积比中位 **9.92×**、中心偏移中位 **21.5 个 GT 宽度**、**任意类别最大 IoU 中位 0.000**、pred 置信度中位 **0.888** —— 那个「大框」在图像**完全别处**，是虚假对应。

---

## 3. Visibility / occlusion / blur / contrast

### 3.1 可计算 proxy，尺寸匹配对照（missed n=70 vs detected n=166）

| metric | missed 中位 | detected 中位 | 比值 |
|---|---:|---:|---:|
| `occ_proxy`（被其他 GT 框覆盖比例） | **0.0000** | 0.0509 | **0.00** ← missed 遮挡**更少** |
| `visible_lapvar`（模糊代理） | **1238.3** | 1041.9 | **1.19** ← missed **更清晰** |
| `visible_gradmean` | 77.78 | 79.71 | 0.98 |
| `visible_std`（局部对比度） | 38.82 | 36.18 | 1.07 |
| `visible_edgedens` | 0.1341 | 0.1535 | 0.87 |
| `infrared_std` | 18.33 | 17.36 | 1.06 |
| `infrared_gradmean` | 18.77 | 17.32 | 1.08 |

**⇒ 没有一项支持「missed 更暗/更糊/更被遮挡」。** 三项甚至朝**相反**方向。

### 3.2 可检测性分级（仅用可计算 proxy，**不含人工判断**）

| 分级 | missed (n=70) | detected (n=166) |
|---|---:|---:|
| CLEAR | **24 (34.3%)** | 68 (41.0%) |
| AMBIGUOUS | 32 (45.7%) | 72 (43.4%) |
| LOW_VISIBILITY | 14 (20.0%) | 26 (15.7%) |

**分布几乎相同。** 三分之一的 miss 与「被成功检出的目标」在这套 proxy 下**没有区别**。

### 3.3 明确的方法学限制

`occ_proxy` 是 **GT 框重叠代理，不是像素级遮挡分割**。本报告**不声称**「目标被遮挡/人类也无法识别」—— 没有可靠依据。同一可计算 proxy 下 missed 与 detected 无差异，**只能**说明「这族可计算代理不解释失败」。

### 3.4 ⚠️ 新发现：Depth 在 small 处结构性无效

用 pipeline 同一 reader（`ultralytics.utils.patches.imread`）读全部 398 张 val depth，**0 读取失败**：

| | 零值占比 中位 |
|---|---:|
| depth 全图 | 28.0% |
| **同图同尺寸随机位置** | **40.3%** |
| **small-GT 位置** | **100.0%**（≥90% 零值占 76.5%；**完全为 0 占 62.8%**） |
| 对照：visible / infrared 同位置 | 0.1% / 0.0% |

**按尺度单调**（这是机制性证据，不是相关性）：

| GT 尺度 | depth 零值占比 中位 | 完全为 0 |
|---|---:|---:|
| small (<1024) | **100.0%** | 79.2% |
| medium | 56.8% | 30.4% |
| large (>9216) | **13.5%** | 4.0% |

**与「距离受限深度传感器：小 = 远 = 超出量程 = 无效」一致。**

**⚠️ 这反过来削弱模态消融结论的 Depth 一臂**：消融的做法是把 Depth 通道**置常数**，而 small 处 depth 本来就是 0（中位 100%）⇒ 对该位置的输入**几乎没变**。
因此「89.8% 的漏检跨模态配置不变」中，**Depth 那一臂在 small 上接近恒等变换**，该数字对 small 而言部分是同义反复。**这不否定 89.8% 的计算，而是否定它作为「depth 对小目标无用」的证据强度。**

---

## 4. Neighbor-instance analysis

对 70 个严格无重叠目标，邻域半径 = 3 × GT 对角（native）：

| 归类 | 数量 | 占比 |
|---|---:|---:|
| `SCENE_DIFFICULTY_CANDIDATE`（邻域对象也全漏） | **28** | 40.0% |
| `ISOLATED_MISS`（邻域内无其它 GT） | **20** | 28.6% |
| `INSTANCE_SEPARATION_CANDIDATE`（邻域**同类已检出**） | **12** | 17.1% |
| `CLASS_CONFUSION_CANDIDATE`（邻域**异类已检出**） | **10** | 14.3% |

---

## 5. Same-image same-class matched analysis

在**同一张图**内、**同一类别**、`sqrt_area` 比值 ∈ [0.67, 1.5] 的 missed↔detected 配对：

- **128 对，来自 13 张图**（person 94 / animal 33 / bicycle 1）
- size ratio 中位 **1.001**（尺寸控制良好）
- detected 侧 best-any IoU 中位 **0.706**、conf 中位 **0.687** ⇒ 邻居**确实被良好检出**

| 对比量 | missed | detected |
|---|---:|---:|
| 输入空间 √area 中位 | **15.61 px** | 14.97 px |
| 边框距离 中位 | **138.5 px** | **199.0 px** |

**⇒ 同图同类同尺寸下，missed 的对象 (a) 略大、(b) 更靠近图像边框。**

**明确回答**：在相同 image + 相同 class + 相似尺寸条件下，**最稳定的差异不是尺寸、不是外观质量，而是「靠近边框」**。这指向**边界/上下文**机制（框被裁切、可用上下文更少），而非尺度机制。

⚠️ 仅 13 张图，场景数少；这是 E2，不是 E3。

---

## 6. Animal vs Person controlled analysis

| 控制层 | animal | person |
|---|---:|---:|
| 未控制 | n=63 **38.1%** | n=195 **15.4%** |
| **L1 尺寸分箱**（共同区间 10.76–20.91px） | | |
| 　10.8–14.1 px | n=20 30.0% | n=47 17.0% |
| 　14.1–17.5 px | n=12 41.7% | n=55 16.4% |
| 　17.5–20.9 px | n=21 **47.6%** | n=60 **11.7%** |
| 　**区间内合计** | **n=53 39.6%** | **n=162 14.8%** |
| L2 同分辨率 (1920×1080) | n=63 38.1% | n=188 14.9% |
| L3 同图共现 | n=30 **30.0%** | — |

### 判读：`CLASS_EFFECT`（E2–E3）

- 尺寸受控后差距**没有缩小**（39.6% vs 14.8%）
- **animal 的 miss 率随尺寸上升（30.0% → 41.7% → 47.6%），person 随尺寸下降（17.0% → 16.4% → 11.7%）**
  ⇒ **若分辨率是约束，这个方向是反的。这是对分辨率假说最直接的否证。**
- 同图共现时 animal 仍 30.0%（vs 未控制 38.1%）⇒ 部分场景效应，但**类别效应仍在**

**不声称**「animal 特征更难所以应该增强 animal」—— 见 §8 的暴露率分析，机制尚未确定。

---

## 7. Weak-candidate / confidence analysis

⚠️ **`predict_rect.py` 的 dump 下限是 `--conf 0.001`** ⇒ **`conf >= 0.0001` 这一档无法回答**（除非生成新 prediction，本轮禁止）。

| 检查（70 个无重叠目标） | 结果 |
|---|---:|
| 该图内存在 `conf >= 0.001` 的任意预测 | 70/70 = 100% |
| 该图内存在 `conf >= 0.01` | 65/70 = 92.9% |
| 该图内存在 `conf >= 0.05` | 64/70 = 91.4% |
| **该图内完全没有同类预测** | **10/70 = 14.3%** |

**分类：`NO_CANDIDATE` = 10，`WEAK_OR_OTHER` = 60。**

**关键判读**：这 60 个不是「在目标位置产生了极低置信候选」——它们**在该位置的任意类别 IoU 都是 0**，而图内别处有高置信同类框（§2 的 H2 分析：中心偏移 21.5 个 GT 宽度、置信 0.888）。
⇒ **不是 confidence/ranking 问题；模型根本没有在这些位置产生候选。** 这**排除**了「下调 conf 阈值」类方案。

---

## 8. Class frequency / exposure analysis

| class | train GT | train small | small 占该类 | small 图数 | val small | **val miss 率** |
|---|---:|---:|---:|---:|---:|---:|
| **uav** | 174 | **118** | **67.8%** | 97 | 18 | **0.0%** |
| **person** | 4421 | 703 | 15.9% | 193 | 195 | 15.4% |
| sign | 659 | 124 | 18.8% | 97 | 30 | 10.0% |
| bicycle | 550 | 79 | 14.4% | 25 | 12 | 16.7% |
| garbage_can | 225 | 39 | 17.3% | 25 | 6 | 16.7% |
| car | 1283 | 97 | 7.6% | 65 | 18 | 16.7% |
| boat | 107 | 17 | 15.9% | 7 | 3 | 0.0% |
| seat | 511 | 30 | 5.9% | 30 | 9 | 0.0% |
| ball | 70 | 22 | 31.4% | 17 | 6 | 16.7% |
| **animal** | 2328 | **107** | **4.6%** | 40 | 63 | **38.1%** |
| **light** | 1211 | **56** | **4.6%** | 25 | 11 | **54.5%** |

**⇒ 与「少样本」无关，而与「small 暴露率」有关：**
- `uav` 总 GT 只有 174（**最少**），但 **67.8% 是 small** → miss **0%**
- `animal` 总 GT 2328（**很多**），但只有 **4.6% 是 small** → miss **38.1%**
- `light` 同理：1211 GT，4.6% small → miss 54.5%

**⇒ 这不是「少样本」，是「少 small 样本」** —— 精确得多。但**机制未定**（可能是 small-animal 外观多样性不足，也可能是标注口径问题）。E1–E2。

---

## 9. Train-vs-val small-size distribution

用同一口径（long-side→1280 的输入空间 √area）统计：

| | n | P5 | P10 | P25 | 中位 | P75 | P90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **train small** | 1392 | 10.09 | 11.78 | 14.61 | **17.65** | 20.66 | 36.88 | 63.50 |
| **val small** | 371 | 10.03 | 11.27 | 13.32 | **17.13** | 19.84 | 33.94 | 59.13 |

**⇒ train 与 val 的 small 尺寸分布几乎重合。** 我上一轮说「val 尺寸范围窄」**不成立** —— 两者都窄，这是**数据本身的属性**。

**Mosaic 后尺寸**：复用既有审计记录（不重新模拟）—— `reports/OASA_STEP25_PRE_MOSAIC_BENCHMARK.md` 实测「最终输入空间 anchor √area **8.59 px**（baseline）」，即 mosaic 期 small ≈ **val 的 0.5×**。

⇒ **模型训练时见到的小目标约为 8.6px，验证时约 17px**，存在 ~2× 的系统尺度差。

---

## 10. Per-class small-object diagnosis

| class | val small | miss | 主要 cause（§11） | 备注 |
|---|---:|---:|---|---|
| person | 195 | 15.4% | SCENE_DIFFICULTY 10 / WEAK_FEATURE 8 / INSTANCE_SEP 4 | 绝对量最大 |
| animal | 63 | 38.1% | SCENE_DIFFICULTY 10 / **INSTANCE_SEP 6** / WEAK_FEATURE 3 | 尺寸反向 |
| light | 11 | 54.5% | WEAK_FEATURE 3 / OCCLUSION 2 / BLUR 1 | small 暴露率 4.6% |
| sign | 30 | 10.0% | WEAK_FEATURE 2 / CLASS_CONF 1 | **尺度不变（16.7/15.5/19.4%）** |
| car | 18 | 27.8% | CLASS_CONFUSION 2 / OCCLUSION 1 | |
| uav | 18 | 0.0% | — | small 暴露率 67.8% |
| bicycle/ball/garbage_can | 12/6/6 | 25/50/16.7% | 样本过少 | 不作推断 |

---

## 11. Primary-cause attribution

**对象：严格「任意类别 IoU==0」的 70 个 small GT。** 互斥 primary cause。

| Primary cause | Count | % | Evidence |
|---|---:|---:|---|
| `SCENE_DIFFICULTY` | 21 | **30.0%** | **E1** |
| `WEAK_FEATURE_EVIDENCE` | 17 | **24.3%** | **E1** |
| `INSTANCE_SEPARATION` | 11 | 15.7% | **E2** |
| `CLASS_CONFUSION` | 6 | 8.6% | E2 |
| `OCCLUSION` | 6 | 8.6% | E2（**且被对照否定**，§3.1） |
| `DATA_VISIBILITY` | 5 | 7.1% | E2 |
| `LOW_CONTRAST` | 3 | 4.3% | E2 |
| `BLUR` | 1 | 1.4% | E2 |
| `UNKNOWN` | 0 | 0% | — |

> **UNKNOWN 允许存在，本轮为 0** —— 但注意 `SCENE_DIFFICULTY`(E1) 与 `WEAK_FEATURE_EVIDENCE`(E1) 合计 **54.3%** 只有单一统计相关性支持，**不应被当成已确立的机制**。

---

## 12. Model-actionable vs non-actionable

| 家族 | 占比 | 可干预性 |
|---|---:|---|
| **SCENE_DIFFICULTY + WEAK_FEATURE_EVIDENCE** | **54.3%** | 未知（E1）—— 可能含大量样本/标注内在难度，**不能假定可干预** |
| **INSTANCE_SEPARATION + CLASS_CONFUSION** | **24.3%** | **可干预**（邻居已被检出，是分离/判别问题） |
| OCCLUSION + DATA_VISIBILITY + LOW_CONTRAST + BLUR | 21.4% | **基本不可干预**（数据侧）；且 occlusion 已被对照否定 |

**⇒ 模型可改善的上界约 24.3%（E2），其余 54.3% 机制不明。**

---

## 13. Evidence strength

| 结论 | 等级 |
|---|---|
| **89.8% 跨模态共同漏检 —— 但 Depth 一臂在 small 上近似恒等变换** | **E2**（复算 + depth 零值实测） |
| **Depth 在 small 处无效，且随尺度单调（测距机制）** | **E2**（398 图全量 + 随机位置对照 + 尺度单调） |
| **missed 与 detected 的可计算视觉 proxy 无差异；不是分辨率/可见性** | **E2**（尺寸匹配对照） |
| **animal vs person 的类别效应在尺寸/分辨率受控后仍存在，且尺寸趋势反向** | **E2–E3** |
| **不是 confidence/ranking 问题** | **E2**（70/70 图内有低阈值预测，且目标处置信为 0 重叠） |
| **missed 更靠近图像边框（同图同类同尺寸）** | **E1–E2**（仅 13 图） |
| **small 暴露率 ↔ miss 率 相关（uav 0% / animal 38.1%）** | **E1** |
| SCENE_DIFFICULTY 是最大单因 | **E1** |

---

## 14. Candidate next mechanisms（**不排名**）

### Candidate A — 拥挤/邻近小目标的实例分离
```text
target failure : INSTANCE_SEPARATION (15.7%, E2) + CLASS_CONFUSION (8.6%, E2) = 24.3%
mechanism      : 邻居同类实例被检出、目标实例未被检出；模型在同一局部场景中只能
                 选中部分实例（§2 的 H2 证据：中心偏移 21.5 个 GT 宽度）
support        : §4 直接观测（124 个中的 12+10）；§5 配对（13 图内同图同类已有成功检出）
falsify        : 若把 NMS-IoU / 匹配口径改变后该比例不降，则非分离问题
                 现有资产 diagnostic/small_object_error_attribution/_nms_ap.py + nms_iou0.5/0.6/0.8 可直接检验
risk to M/L    : 若通过改变 assigner/NMS 实现，可能影响中大目标的框竞争
complexity     : 中（且已有 NMS 变体实验框架可复用）
```

### Candidate B — small 暴露率不足的类（animal / light）的**外观多样性**
```text
target failure : animal 38.1%、light 54.5%；两者 small 暴露率均 4.6%（最低档）
mechanism      : 未定。可能是 small-animal 外观/姿态多样性不足，也可能是该类的
                 small 标注口径与视觉边界模糊（模型可能在"看不到"与"不认为是 animal"之间）
support        : §6 的尺寸/分辨率受控类别效应（E2–E3）+ §8 暴露率分层（E1）
falsify        : 若按 small 暴露率分层后 animal 仍显著高于同暴露率的类，则是类别本身
risk to M/L    : 数据侧改动，对 M/L 无直接结构风险；若改标注则影响面大
complexity     : 高（需先做标注/外观审计，本轮未做）
```

### Candidate C — Depth 通道的**条件性使用**
```text
target failure : F representation/fusion —— depth 在 small 处 100% 无效，
                 模型仍为其分配容量（stem 8ch）
mechanism      : 用一个 validity/远距离感知的门控，让 depth 在无效区不参与；
                 或直接评估「去掉 depth 通道」的 4ch 变体
support        : §3.4 的 E2 实测（单调 + 随机位置对照）
falsify        : 若 depth 在 medium/large 上的有效性并不转化为中大型目标增益，
                 则该路线的上限很低（已有模态消融显示 Depth 独有贡献仅 1.98%）
risk to M/L    : 低（改 5ch→4ch 是输入侧，但会改变 stem 结构）
complexity     : 中（4ch 早期融合已有先例）；但**与"提升 small"的关系是间接的**
```

**⚠️ 不做推荐、不排名。** 三个候选针对的失败家族不同（A=24.3% E2、B=类别效应 E2–E3、C=数据事实 E2），但**都不能覆盖占比最大的 54.3%（E1）**。

---

## 15. Training Gate

```text
HOLD
```

**理由**：占比最大的两个家族（合 54.3%）**只有 E1 支撑**，且 §14 的三个候选**没有一个直接针对它们**。此时开训等于用一次 300-epoch 实验去赌一个尚未确立的机制。

### 具体缺哪一个 diagnostic

**必须补的一项：`SCENE_DIFFICULTY` + `WEAK_FEATURE_EVIDENCE`（54.3%）到底是「内在难度」还是「模型盲区」。**

判别设计（**只读，不需训练**）：
1. 对 38 个（21+17）目标所在的图，统计**该图的整体 GT 密度 / small 密度 / 与其它已检出目标的最近距离**。
   - 若这些目标普遍处在**高密度**场景 → 指向拥挤/分离（可干预，且与 Candidate A 合流）
   - 若处在**稀疏**场景 → 更可能是局部外观/标注问题（不可干预）
2. 与 §5 的配对方法一致，按 `border_margin_norm` 分层——**验证「靠近边框」这条 E1–E2 线索**是否也出现在这 38 个上。

**若这两项把 54.3% 归入可干预的一侧，下一次训练才有明确的 target failure mechanism；否则应转向数据/标注侧。**

---

## 16. Files / scripts / data used

**新增（只读分析，均在 `diagnostic/small_object_cause_v2/`）**

```text
_v2_records.py       阶段 1/2/4/7/8/9
_v2_matched.py       阶段 5/6/7
_v2_visual.py        阶段 3/10（从已有 RGB/IR/Depth 裁局部，无可计算 proxy）
_v2_attribute.py     阶段 11/12/13
_v2_records.json     371 行 small GT 完整 record + stage1 计数 + train/val 尺寸统计
_v2_visual.json      missed 70 / detected 166 的 proxy 明细
_v2_attribute.json   70 条 primary cause
REPORT.md            本报告
```

**读取的既有资产（未改动）**

```text
diagnostic/sepstem_clahe/best_full/results/            D′ 的 400 个 val 预测 TXT
data/processed/rgbid_split_train/{images,labels}/val/  val GT + RGB/IR/Depth
data/processed/rgbid_split_train/{images,labels}/train/ train GT（阶段 8/9）
scripts/official_eval.py                               直接 import（reader/IoU/cap=100）
ultralytics/utils/patches.py:imread                    与 pipeline 同一 reader
reports/SINGLE_MODALITY_TO_RGBID_AUDIT.md              89.8% 的原始定义
reports/OASA_STEP25_PRE_MOSAIC_BENCHMARK.md            mosaic 期 8.59px（复用，未重模拟）
diagnostic/small_object_error_attribution/             NMS 变体框架（未使用，候选 A 可用）
```

---

## 17. Reproducibility

```text
training runs              : 0
model modifications        : 0
evaluator modifications    : 0
prediction generation      : 0
online submission          : 0
new diagnostic files       : 见 §16（8 个，均在 diagnostic/small_object_cause_v2/）
existing files modified    : 无
```

**复现命令**

```bash
python -X utf8 diagnostic/small_object_cause_v2/_v2_records.py
python -X utf8 diagnostic/small_object_cause_v2/_v2_matched.py
python -X utf8 diagnostic/small_object_cause_v2/_v2_visual.py
python -X utf8 diagnostic/small_object_cause_v2/_v2_attribute.py
```

---

## 附：本轮我自己犯过并更正的三个错误（记录备查）

1. **「89.8% 不成立」** —— 我上一轮用了 D′/OASA 三元组。按原始模态消融三元组复算为 473/527 = **89.8%，完全复现**。
2. **「91」的口径** —— 我说是「任意类别 IoU<0.10」。实为**同类 IoU==0**（91）；严格任意类别口径是 **70**。
3. **「val small 尺寸范围窄」** —— 实测 train 与 val 分布几乎重合（中位 17.65 vs 17.13），两者都窄，是数据属性而非 split 属性。

另有两处**测量代码 bug**（已在最终数字中修正）：depth 路径硬编码 `.jpg`（漏掉 373 个 `.png`/16-bit 文件）、visual 指标误从 records JSON 读取。
