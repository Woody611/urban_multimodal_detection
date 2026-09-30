# Depth Reliability Small-Miss Attribution Audit

**日期**：2026-09-28
**性质**：**READ-ONLY**。0 training / 0 backward / 0 optimizer.step / 0 改既有源码·config·checkpoint·
evaluator·dataset·labels·augmentation·inference / 0 提交 / 0 创建训练 config。
**FORWARD_RERUN = 0**（复用 `diagnostic/sepstem_clahe/best_full/results` 已落盘的 D′ 预测）。

**唯一假设**：native-small hard miss 是否系统性地处于 depth-invalid / depth-low-information 区域，
且该 reliability deficit 能否解释 small-object 的 detection failure？

---

## 0. 最终裁决

```text
D — UNRESOLVED
```

**触发原因：STOP B（native depth 与 GT 坐标无法可靠对应）。**

四项只读检验（`_alignment_check.py`）：

| 检验 | 结果 | 判读 |
|---|---|---|
| T1 像素级梯度相关 corr@(0,0) | **0.1065** | 对齐良好的 RGB-D 通常 0.3–0.6 ⇒ **过弱** |
| T2 位移搜索最佳 corr | **0.1448**（最佳位移中位 (dy,dx)=(0,16)） | 纯平移也解释不了 ⇒ **非简单错位** |
| T3 行剖面 corr(zero_frac, visible gradient) | 0.8837 | 仅说明**行剖面的共同趋势**，**不证明像素级对齐** |
| T4 `depth==0` 在 visible 天空掩码内的 lift | **1.184** | ≈1 ⇒ 零区与可见光结构**无强对应** |

⇒ **无法证明 HR depth 与 HR visible 在像素级对应**，因此「GT 框内 depth 全零」**不能**解释成
「目标区域没有深度回波」。协议 §23 **STOP B 成立**。

**但有一类结论对错位稳健**（两组用**同一套坐标映射**，是 apples-to-apples）：
**hard miss 与 successful 的 depth 有效性分布完全相同** ⇒ 即便在做得到的相对比较上，
depth reliability **也不区分 hard miss**。详见 §4。

---

## 1. STOP 条件逐条核查（协议 §23）

| 条件 | 状态 | 说明 |
|---|---|---|
| **STOP A** 无法恢复 G1/G2/G4 | **未触发** ✅ | 直接复用 `_v3_analysis.json["E1"]`（G1=38）与 `["controls"]`（已有的 matched control）；G4 = `_v2_records.json` 中 `native_area<1024 ∧ best_same_class_iou≥0.5 ∧ ∉G1/G2`。**未重新随机抽 control。** |
| **STOP B** native depth 与 GT 坐标无法可靠对应 | **⚠ 触发** | 见 §0 |
| **STOP C** 需要重新 forward | **未触发** ✅ | RGB/IR 对照用**图像域对比度**（无 forward）；已落盘预测足够 |
| **STOP D** depth 编码不明确 | **未触发** ✅ | `base.py:344-347` 明读：uint16 → `clip(d/19999*255,0,255).astype(uint8)`；**raw 0 = 无效**，19999 = 饱和常数 |
| **STOP E** depth 文件版本与训练不一致 | **未触发** ✅ | 与训练同源：`rgbid_split_train/images/{train,val}/depth` 由 `data/raw/**/depth` 硬链接而来；val 侧即评测侧所读 |

**坐标对应校验**：G1/G2 的 `native_area` 与 `_v2_records.json` 重算值**不一致数 = 0**
（2,847 条 GT 记录）⇒ **数值坐标本身可靠**；不可靠的是 depth↔visible 的**空间**对应。

---

## 2. 数据与口径

| 项 | 值 |
|---|---|
| split | `data/processed/rgbid_split_train`，val（400 图），与冻结 evaluator 同集 |
| G1 | **38** small hard miss（`_v3_analysis.json["E1"]`） |
| G2 | **38** 已有 matched control（class 一致 33/38；`sqrt(area)` 差中位 **0.64px**；aspect 差中位 0.286） |
| G4 / small_success | **194** |
| medium / large success | **0 / 0** —— `_v2_records.json` 的 371 条**全为 small**（native_area 119–1022），**没有 medium/large 记录**；即用落盘预测重算会引入第二套匹配口径，与 G1/G4 不可比，故**不伪造**（详见 §5 限制 5） |
| depth 层 | G1/G2 **全部为 HR `.png`（uint16, 1080×1920）** ⇒ 单一层，无跨层混淆 |
| 统计位置 | **native image coordinate**（归一化 GT × 原图尺寸），**不 resize**（§4） |
| patch 定义 | R1 = exact box；R2 = 1.5×；R3 = 2.0×（绕 GT 中心放大后裁剪）；ring = R3−R1 |

---

## 3. §19 REPLICATION（当前原始 depth / exact GT UID 重算，不引用旧数字）

```text
native-small GT 数 = 232（G1 38 + small_success 194）
GT box depth zero_frac: 中位 1.0000  p25 1.0000  p75 1.0000  p90 1.0000
完全 zero 的 small patch 比例 = 80.2%
全图 depth zero_frac 中位 = 0.4880
```

⇒ **历史观察复现**：small GT 位置 depth 零比例极高，且**80.2% 的 small patch 完全为零**
（全图仅 48.8% 为零）。**replication = YES**。
但见 §4 —— **这并不是 hard miss 的特征。**

补充结构（`_alignment_check.py`）：跨 373 张图叠加后，zero 频率 >0.75 的像素仅 **1.4%**
⇒ **不存在固定相机掩码**；零区主要是**每图特有**的，并呈**上多下少**的平滑梯度（天空/远处无回波）。

---

## 4. §17 ATTRIBUTION MATRIX

| group | n | GT zero | 1.5× zero | 2× zero | **ring zero** | valid ratio | rel deficit |
|---|---:|---:|---:|---:|---:|---:|---:|
| small_hard_miss | 38 | **1.0000** | 1.0000 | 1.0000 | **1.0000** | **0.0000** | 0.6539 |
| G2_matched_control | 29 | **1.0000** | 1.0000 | 1.0000 | **1.0000** | **0.0000** | 0.5947 |
| small_success | 194 | **1.0000** | 1.0000 | 1.0000 | **1.0000** | **0.0000** | 0.3918 |

（median）

**读法**：**所有组的 GT box、1.5×、2× 邻域、以及 ring 的 zero fraction 中位都是 1.0000**。
⇒ small GT 位于**远大于自身 2× 的大片 depth 无效区**内 —— 这是**场景级**而非目标级的无效；
且 **hard miss 与 success 完全一致**。

### §11/§12 配对比较（G1 vs G2，38 对）

| metric | G1 中位 | G2 中位 | **配对 Δ 中位** | Wilcoxon p | Cliff d | MWU p |
|---|---:|---:|---:|---:|---:|---:|
| GT zero | 1.0000 | 1.0000 | **0.0000** | 0.918 | 0.368 | 0.0057 |
| 1.5× zero | 1.0000 | 1.0000 | **0.0000** | 0.831 | 0.307 | 0.021 |
| 2× zero | 1.0000 | 1.0000 | **0.0000** | 0.935 | 0.269 | 0.044 |
| ring zero | 1.0000 | 1.0000 | **0.0000** | 0.935 | 0.270 | 0.043 |
| valid ratio | 0.0000 | 0.0000 | **0.0000** | 0.918 | 0.435 | 0.0011 |
| depth std | 649.64 | 566.73 | 326.47 | nan | −0.089 | 0.744 |
| depth IQR | 819.75 | 741.63 | 478.00 | nan | −0.156 | 0.568 |
| grad med | 0.0000 | 0.0000 | **0.0000** | 0.255 | 0.204 | 0.127 |
| enc-collapse | 0.0000 | 0.0000 | **0.0000** | nan | 0.600 | 0.0275 |

**核心**：**所有 zero-fraction 指标的配对 Δ 中位 = 0.0000，Wilcoxon p 0.83–0.94（无配对差）。**

Cliff's d（0.27–0.44）与 MWU p（0.001–0.04）看似"显著"，但其来源是**少数 G1 样本的 depth 反而有效**
（见 §6 class 分层：`light` G1 中位 0.0051 vs G2 1.0000；`sign` G1 0.5017 vs G2 0.9762）——
**方向与原假设相反**（hard miss 更**不**零）。这不是证据，是**尾部少数样本**。

> ⚠ `depth std / IQR` 只在 **raw 非零的 patch** 上有定义（约占 20% 的 GT），
> 分母很小且与 zero-fraction 高度耦合，**不作为主判据**。

---

## 5. 分层控制（协议 §13/§14/§15/§16）

### §14 class 分层（G1 vs G2 的 `r1_zero_frac` 配对差）

| class | n_pairs | G1 中位 | G2 中位 | Δ 中位 |
|---|---:|---:|---:|---:|
| person | 18 | 1.0000 | 1.0000 | 0.0000 |
| animal | 13 | 1.0000 | 0.9410 | 0.0000 |
| light | 3 | 0.0051 | 1.0000 | −0.9949 *(n<8，仅描述)* |
| sign | 2 | 0.5017 | 0.9762 | −0.4745 *(n<8，仅描述)* |
| garbage_can | 1 | 1.0000 | 1.0000 | 0.0000 *(n<8)* |
| ball | 1 | 1.0000 | 0.0000 | 1.0000 *(n<8)* |

⇒ 两个主类（person 18 对、animal 13 对，共 31/38 = 82%）的 **Δ 中位 = 0.0000**。
非零差异全部来自 **n<8** 的类，且**方向相反**。**无 class 分层下的关联。**

### §15 size 分层（G1 内部按 native sqrt 四分位，对照全体 small_success）

| stratum | n_G1 | G1 zero 中位 | n_succ | succ zero 中位 | Δ |
|---|---:|---:|---:|---:|---:|
| S1 (≤p25) | 10 | 1.0000 | 50 | 1.0000 | **0.0000** |
| S2 (25–50%) | 9 | 1.0000 | 19 | 1.0000 | **0.0000** |
| S3 (50–75%) | 9 | 1.0000 | 26 | 1.0000 | **0.0000** |
| S4 (>p75) | 10 | **0.7984** | 87 | 1.0000 | **−0.2016** |

⇒ 前三层完全一致；唯一的差异在**最大的四分位**，且方向是 **hard miss 更不零**（与假设相反）。

### §13 RGB / IR EVIDENCE 分层（图像域对比度，非模型特征；无 forward）

| 分层 | n_G1 | n_succ | G1 zero | succ zero | Δ |
|---|---:|---:|---:|---:|---:|
| RGB contrast 低(<中位) | 19 | 122 | 1.0000 | 1.0000 | **0.0000** |
| RGB contrast 高(≥中位) | 19 | 72 | 1.0000 | 1.0000 | **0.0000** |
| IR contrast 低(<中位) | 19 | 59 | 1.0000 | 1.0000 | **0.0000** |
| IR contrast 高(≥中位) | 19 | 135 | 1.0000 | 1.0000 | **0.0000** |

⇒ 在 RGB/IR 证据分层后，depth zero 的组间差**处处为 0**。

### §16 image-level baseline

`relative_depth_deficit = GT_zero − image_zero`：hard miss 0.6539 / G2 0.5947 / success 0.3918（中位）。
差异**全部由 image-level zero 驱动**（hard-miss 图像整体 65% 为零 vs success 图像 39%），
而非目标区域相对整图「异常更差」——因为**所有组的 GT zero 都已饱和于 1.0**，无从比较。

---

## 6. 唯一非零的信号：image-level，而非 target-level

| 量 | small_hard_miss | small_success |
|---|---:|---:|
| 全图 depth zero_frac（中位） | **0.6539** | **0.3918** |
| patch raw_std（中位，仅非零 patch） | 649.64 | 105.16 |

hard-miss 所在**图像**整体更零、depth 变化更大。但这**不是** §2 所问的「hard miss 更容易落在
depth-invalid 区域」——它是**图像/场景层面的构成差异**（hard miss 更可能出现在 depth 质量差的场景），
且 §13 分层后 target-level 差异为 0。**这属于 image-level confounding，不构成 target-level association。**

---

## 7. 三个问题的回答（协议 §25）

**Q1 — Depth reliability 是否与 native-small hard miss 显著相关？**

**否。** target-level 上：GT box / 1.5× / 2× / ring 的 zero fraction **中位在全部组都是 1.0000**，
配对 Δ 中位 **全部 = 0.0000**，Wilcoxon p 0.83–0.94。仅存在**图像层面**（非目标层面）的scene-level 差异。

**Q2 — 该关联能否在 class / size / image-level 分层后保持？**

**不存在可保持的关联。** class 分层（82% 的配对来自 person/animal）Δ=0；size 分层前三层 Δ=0、
第四层方向相反；RGB/IR 分层 Δ 处处为 0。

**Q3 — 是否达到 L2/L3，值得进入下一阶段 feasibility audit？**

**否**（且本轮裁决为 **D**，不是 L0 —— 见下）。

---

## 8. 证据等级（协议 §18）

```text
D — UNRESOLVED
```

**为什么不是 L0：** STOP B 使「GT 框内 depth 是否代表目标区域」**未被证实**。
零比例在两组都饱和于 1.0 —— 这**既可能**是「small 目标普遍无深度回波」，
**也完全可能**是「坐标映射系统性地把 GT 框投到了无效区」。二者无法区分，
因此不能签 L0（L0 要求「hard miss 与 successful 的 depth reliability 基本一致」这一**测量**可信）。

**对错位稳健的那部分结论**：两组用**同一套坐标映射**，是 apples-to-apples 比较 ⇒
**hard miss 与 successful 在 depth 有效性上不可区分**这一条**成立**。
故**若**后续证明坐标对应是可靠的，本数据**会**落到 **L0**。

---

## 9. Limitations

1. **STOP B**：HR depth 与 HR visible 的像素级对应未能证明（T1 0.107 / T2 0.145 / T4 lift 1.18）。
2. **零比例饱和**：全部三个组的 zero 中位都是 1.0000 ⇒ 该指标在本数据上**没有分辨力**；
   任何进一步的「比例」分析都无法区分。
3. **medium / large 对照缺失**：`_v2_records.json` 371 条**全为 small**，没有 medium/large；
   用落盘预测重算会引入第二套匹配口径，与 G1/G4 不可比 ⇒ **不伪造**（协议 §5 的 medium/large 行因此为空）。
4. **G2 匹配**：class 一致 33/38（5 对跨类），`sqrt(area)` 差中位 0.64px；未做 size/AR 的额外人工匹配
   （已有 matched control 优先，协议 §11）。
5. **RGB/IR 对照是图像域对比度**，不是模型内部 representation 响应（无 forward）。
   协议 §13 允许在无现成变量时报 `RGB/IR adjustment unavailable`；此处提供了**弱一档**的替代，
   已在 §5 明确标注。
6. **不可外推到「depth 有害」**：协议 §20 —— 本轮最多只能说「未检出关联」，
   不能推出「去掉 depth 会更好」，更不能推出「Depth Validity Mask 一定有效」。

---

## 10. Recommendation（协议 §24）

```text
HOLD
```

即使将来升级到 L2/L3：**也不训练**；最多只能建议
「Proceed to a separate Depth-Validity intervention feasibility audit」。
**不创建** 6-channel config / depth mask config / 训练脚本改动 / checkpoint / 实验 run。

**本轮的净结论**：在当前可测量的口径下，**depth reliability 与 native-small hard miss 无可检出的关联**；
且由于 STOP B，**该问题的绝对形式尚未被可靠测量**。若要让这个问题变成可回答的，
需要先解决 depth↔visible 的空间对应（那属于**新的基础设施审计任务**，需另行批准）。

---

## 11. Provenance / 合规

```text
MODIFIED_EXISTING_FILES = 0
TRAINING = 0
BACKWARD = 0
OPTIMIZER_STEP = 0
FORWARD_RERUN = 0
CHECKPOINT_MODIFIED = 0
```

| 读取对象 | SHA256（前 16） |
|---|---|
| `runs/…_sepstem_clahe/weights/best.pt` | `1cae45f75693f541` |
| dataset | `data/processed/rgbid_split_train`（val 400 图；labels/val/visible 400 txt；images/val/depth 400） |
| 预测落盘 | `diagnostic/sepstem_clahe/best_full/results`（400 文件，**= v2 records 的同一份 dump**） |
| `ultralytics/data/base.py`（depth 编码） | `8bcf834930155bd5` |
| `scripts/official_eval.py`（匹配/IoU 口径） | 见 `_tables.json` |

**产物**：`REPORT.md` / `_analyze.py` / `_alignment_check.py`（STOP B 判定）/
`_tables.json` / `_alignment.json` / `_results.npz` / `ANALYZE.log` / `ALIGNMENT.log`。
（相对协议 §21 的文件清单多了 `_alignment_check.py` / `_alignment.json` / `ALIGNMENT.log`
—— STOP B 需要一个**可复现的判定产物**，不能只在报告里叙述。）
