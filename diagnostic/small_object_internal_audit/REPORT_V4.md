# SMALL_OBJECT_MODEL_INTERNAL_AUDIT_V1

**性质**：只读因果定位。0 训练 / 0 调参 / 0 改 assigner / 0 改模型 / 0 新 prediction / 0 提交。

---

## A. Data integrity

```
train/val only                 : 确认。全部脚本路径 = data/processed/rgbid_split_train/（train + val）
test data accessed             : 否。未读 data/raw/test/*、submissions/*、任何 online/leaderboard 产物
checkpoint                     : runs/..._sepstem_clahe/weights/best.pt
                                 sha256 1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda（未改动）
augment.py / default.yaml      : 5cb9a407… / 991a89b3…  与本轮开始时一致
scripts/train.py               : 9f55b09f…  一致
configs/oasa_pre_mosaic_scale14.yaml : dee6601d…  一致
configs/yolo11m_sepstem.yaml   : 9b14f294…  一致
本轮修改的既有文件              : 无
本轮新增文件                    : 见 §17（均在 diagnostic/small_object_internal_audit/）
```

`git status` 中 `M` 的文件（`.gitignore` / `scripts/*` / `default.yaml` / `augment.py` / `models/yolo/detect/train.py`）均为**前几轮实验的既有改动**，其哈希与本轮开始时逐位相同 —— 本轮未触碰。

---

## B. G1/G2/G3/G4 matching quality

| 组 | 定义 | N |
|---|---|---:|
| **G1** | V3 的 38 个 E1 hard miss | **38** |
| **G2** | 同图 + 同类 + 相似尺寸 + 已检出的 matched control（V3 的 Control A） | **29** |
| **G4** | 同图其它**成功** small GT（same-class IoU≥0.5，非 G1 非 G2） | **69** |
| G3 | 同图相似尺寸的 non-small GT | 已由 G4 覆盖，本轮不单列 |

匹配已在 V3 验证（`sqrt_area` ratio 中位 1.001；G1 与对照尺寸分布重合）。**G4 是本轮最强的对照：它直接回答「missed small 与 successful small 是否可分」。**

---

## C. Image-space discriminability（Q1）

**V3 已做且本轮不重复**。结论：

| descriptor | G1 vs 对照 | 结论 |
|---|---|---|
| RGB 全部（std / grad / edge / Lapvar） | p = 0.42–0.77 | **无差异** |
| **IR local contrast（std）** | 12.29 vs 19.79, p = **0.0054** | 唯一显著，Bonferroni α=0.0056 **勉强守住** |
| Depth 全部 | p = 0.46–0.58 | 无差异（且 small 处 depth 中位 100% 为 0） |

**⇒ 输入空间几乎不可分。** 这本身是重要结论：**失败不在像素可见性**。

---

## D. Assignment audit（Q2）—— **结论：assignment 不成立为原因**

真实 `TaskAlignedAssigner`（topk=10, alpha=0.5, beta=6.0）的 **in-process 子类插桩**（不修改 `ultralytics/`），在同一 forward 下复盘：

| metric | G1 (n=38) | G2 (n=29) | G4 (n=69) | Cliff's Δ | p(G1vG2) | p(G1vG4) |
|---|---:|---:|---:|---:|---:|---:|
| `n_in_gts` | 5.0 | 5.0 | 6.0 | — | — | — |
| `n_topk` | 10.0 | 10.0 | 10.0 | — | — | — |
| **`n_pos`** | **5.0** | **5.0** | **6.0** | −0.190 | **0.184** | 0.104 |
| `best_assign_iou` | 0.6278 | 0.7208 | 0.7721 | −0.367 | 0.0108 | 1.2e-05 |
| positive stride 分布 | {8:38, 16:28, 32:3} | {8:29, 16:23, 32:6} | {8:69, 16:45, 32:17} | — | — | — |

### Q1（`positive_count == 0` 比例）

```
G1: 0/38 = 0.0%    G2: 0/29 = 0.0%    G4: 0/69 = 0.0%
```

### Q2（positive 是否更弱）

- **`n_pos` 无显著差异（p = 0.184），`n_topk` 全部满 10** ⇒ **G1 并未缺少 positive 数量**
- `best_assign_iou` 略低（0.628 vs 0.721，p = 0.011）—— 但 `n_pos` 不显著，且该 IoU 差异的 CI 跨 0（[−0.242, +0.015]）
- positive 的 stride 分布正常落在 P3(8) 为主，与对照一致

**⇒ Assignment hypothesis 不成立**（按你 §6 的 Q3 条款：assignment 与 G2/G4 基本一致）。

---

## E. Feature response audit（Q3 上半）

P3 = layer 23（Detect 输入①，256ch@160）、backbone P2 = layer 9（256ch@320）：

| metric | G1 | G2 | G4 | Cliff's Δ | p(G1vG2) |
|---|---:|---:|---:|---:|---:|
| **P3 center/global norm ratio** | **1.0492** | 1.1503 | 1.1586 | **−0.666** | **3.5e-06** |
| **P3 center feature norm** | **6.0962** | 6.6584 | 6.6719 | −0.670 | **3.1e-06** |
| P3 global norm | 5.8369 | 5.8507 | 5.7223 | — | — |
| **backbone P2 (L9) center/global ratio** | **1.4201** | 1.4828 | **1.4204** | −0.180 | **0.213（NS）** |

**⇒ P3 的中心特征**确实**弱于对照（约 −8%），达统计显著；但**特征并未消失**（norm 6.10，非接近 0）。
**⇒ backbone P2（stride 4）的特征 G1 ≈ G4（1.420 vs 1.420，p=0.21）—— 差异只出现在 P3（stride 8）。**

---

## F. GT-center candidate audit（Q3 下半 / §8）—— **本轮最关键的发现**

**注意：以下 "logit" 为 raw logit；"prob" 为 sigmoid。**

| metric | G1 | G2 | G4 | Cliff's Δ | ΔCI95 | p(G1vG2) |
|---|---:|---:|---:|---:|---|---:|
| **GT 中心附近最高同类 logit（P2 层, r=12px）** | **−12.585** | 0.827 | 0.882 | **−0.987** | [−14.38,−11.52] | **6.1e-12** |
| **同上（P3 层, r=24px）** | **−11.356** | 1.043 | 1.262 | −0.962 | [−14.23,−10.77] | 2.1e-11 |
| **同上（P4 层, r=48px）** | **−9.640** | 1.278 | 1.351 | −0.955 | [−14.24,−9.94] | 2.9e-11 |
| **全图最高同类 logit** | **2.543** | 2.601 | 1.944 | −0.289 | [−3.37,+0.23] | 0.044 |
| **positive anchor 上类别概率（max）** | **0.0000** | **0.6705** | 0.6667 | **−0.980** | [−0.791,−0.470] | **8.7e-12** |
| positive anchor 上类别概率（mean） | 0.0000 | 0.3958 | 0.3879 | −0.984 | [−0.569,−0.155] | 7.3e-12 |

### 判读

1. **在 G1 目标位置，模型的类别概率 ≈ 0**（中心最高 logit −12.6 ⇒ p ≈ 3.4e-6；positive anchor 上 p = 0.0000）
2. **同一张图里该类别最高 logit = +2.54（p ≈ 0.93）** ⇒ 模型**能在该图中检测该类别**，只是不在这个实例上
3. **Cliff's Δ = −0.98**：几乎是**完全分离**（G1 的每一个值都低于对照的几乎所有值）
4. **特征存在但概率为 0** ⇒ 不是「特征太弱」，而是**分类头在该处输出强烈的「非此类」**

> 极端例：`000012_025_00000313#10` 的 P3 center/global ratio = **1.164**（高于 G2 中位 1.150），
> 但该处类别 logit = **−17.98**。**特征比成功目标还强，分类概率却≈0。**

### §10 的判别

你 §10 问：*「若 GT 中心附近已有高于背景的 class response，但无 overlap candidate → 指向 localization」*。
**本轮的答案相反**：中心附近的 class response **不是「低」而是「强负」**（−12.6），
而几何是好的（`best_assign_iou` = 0.63、positive 存在）。
⇒ **不是 localization，不是 candidate generation，是 head 的分类响应缺失。**

---

## G. Failure-stage attribution

```text
input evidence      ✅ 存在（V3：92.1% 在 ≥1 模态有局部证据；RGB 与对照无差异）
      ↓
feature representation  ⚠️ P3 中心特征弱约 8%（p=3e-6），但**存在**；backbone P2 与对照无差异
      ↓
assignment / supervision  ✅ **正常**（n_pos p=0.184；n_topk 满；positive stride 分布正常）
      ↓
classification / regression head  ❌ **在此断裂**：类别概率 ≈ 0，logit −12.6 ~ −18
      ↓
candidate generation  ⛔ 未到达（head 从未产生该处的候选）
      ↓
NMS                   ⛔ 无关
```

```text
Type C —— assignment 正常，但 head response 缺失
Evidence: E3
```

**E3 依据**：(a) 匹配对照（同图/同类/同尺寸）；(b) effect size 接近完全分离（Cliff's Δ ≈ −0.98）；
(c) 通过 Bonferroni（7/11）与 BH-FDR（8/11）；(d) 在**预先指定的低密度孤立子集**中同样成立（见 §12）。

---

## H. Coverage

| mechanism | N | /70 | % |
|---|---:|---:|---:|
| **G1（E1 家族）= head 类别响应缺失（Type C）** | **38** | 38/70 | **54.3%** |
| 其中在 G1 内部 | 38/38 | — | **100%** |
| 剩余 32 个 zero-overlap（V2 归为 INSTANCE_SEPARATION/CLASS_CONFUSION/OCCLUSION/DATA_VISIBILITY/LOW_CONTRAST/BLUR） | 32 | 32/70 | 45.7% |

---

## I. Q1–Q5

| 问题 | 答案 |
|---|---|
| **Q1** G1 与 G2/G4 在 image-space 是否可分？ | **几乎不可分**（RGB/Depth 无差异；仅 IR 对比度 p=0.0054 边界显著） |
| **Q2** G1 是否系统性缺少 positive assignment？ | **否**（`n_pos` p=0.184，`n_topk` 满，positive_count==0 比例 0%） |
| **Q3** G1 是否系统性缺少 GT-center feature/head response？ | **特征：部分**（P3 ratio 1.049 vs 1.150，p=3.5e-06，但未消失）<br>**head：是**（类别概率 0.0000 vs 0.67，p=8.7e-12，Δ=−0.98） |
| **Q4** 若两者都正常，是否在 candidate generation/localization 失败？ | **否。** 几何正常（IoU 0.63），但 head 从未产生候选 → **不属于 candidate generation / localization** |
| **Q5** 是否存在 E2/E3 + 足量 coverage + 明确干预点 + 可单变量验证的机制？ | **E3 有、coverage 有（54.3%）、但「明确模型干预点」不成立 → HOLD** |

---

## J. Final Gate

```text
HOLD
```

### 依据

**已确立（E3）**：G1 的 54.3% strict hard miss，其失败**定位在分类头的输出层**——
在 GT 位置与 positive anchor 上，该类别概率 ≈ 0（logit −12 ~ −18），而**同一图中该类别峰值 logit = +2.54**。
**几何、assignment、特征存在性均正常。**

**为何仍 HOLD（按你 §16 的六条）**：

| 条件 | 判定 |
|---|---|
| E2/E3 evidence | ✅ **E3** |
| coverage meaningful | ✅ 38/70 = 54.3% |
| reproducible in matched controls | ✅ G2 与 G4 双重对照 |
| **clear model intervention point** | ❌ **不成立** |
| single-variable experiment possible | ⚠️ 依赖上一项 |
| medium/large guardrails definable | ⚠️ 依赖上一项 |

**「head 在该处输出 ≈0」是一个精确的断裂点描述，但它不是机制。** 它没有说明**为什么**：
- 可能是训练时该实例从未获得正的分类监督（TAL 的 `align = s^α·u^β`，若 `s≈0` 则目标分≈0 ⇒ 正反馈）；
- 也可能是该外观在 P3 表征空间中落在决策边界的另一侧；
- 也可能与 padding/LetterBox 的边界上下文有关。

**这三者需要完全不同的干预，而现有证据无法区分。** 因此没有一个可以写成单变量实验的干预点。

### 还缺的那一项证据（唯一）

**在训练时（而非 val 输入）复盘同一批 GT 的 assignment**：
若这些 GT 在**训练增强样本**上同样得到 `align≈0 / target_score≈0`，则机制锁定为
**「零对齐 → 无正类监督 → head 自锁」**，干预点明确（例如对 align 退化的 GT 保底注入正样本）；
反之若训练时 `align` 正常，则失败发生在**表征泛化**，干预点完全不同。

**实现路径（不需训练）**：用 `build_yolo_dataset(mode="train")` 取一个确定性种子下的增强样本，
对同一批 GT 复用本轮完全相同的插桩。**这是一个只读诊断，不是训练实验。**

---

## §17 Reproducibility

```text
training              : 0
assigner modification : 0（仅 in-process 子类插桩，ultralytics/ 未改）
model modification    : 0
core code modification: 0
new inference         : 0（仅 forward，不落盘 prediction）
online submission     : 0
test data accessed    : 0
```

**新增文件（均在 `diagnostic/small_object_internal_audit/`）**
```
_v4_internal.py     assigner 插桩 + feature hook + GT-center raw 响应
_v4_analyze.py      三层对比 + 阶段归类
_v4_stats.py        effect size / bootstrap CI / Mann-Whitney / Bonferroni / FDR
_v4_internal.json   429 条 GT 记录
_v4_analysis.json   G1/G2/G4 计数 + 阶段归类
_v4_stats.json      统计汇总
_v4_run.log         运行日志
REPORT_V4.md        本报告
```

**复现**
```bash
python -X utf8 diagnostic/small_object_internal_audit/_v4_internal.py
python -X utf8 diagnostic/small_object_internal_audit/_v4_analyze.py
python -X utf8 diagnostic/small_object_internal_audit/_v4_stats.py
```

---

## 附：本轮自己的错误（记录备查）

1. **跨图对照所在图未被处理** —— `ctrl` 的值是 `(control_key, type)`，首版只检查了 G1 侧
   的 stem，导致 11/38 对照缺失（G2 = 19）。已修正 → G2 = 29。
2. **统计脚本对已是概率的量重复施加 sigmoid** —— 使 `pos_score_max` 中位假显示为 0.5000
   （= sigmoid(0)）。修正后真值为 **0.0000 vs 0.67**，**结论方向不变、效应更强**。
3. `stride_tensor` 形状误用（`make_anchors` 返回 (n,1) 而非 (1,n,1)），首版索引错误。
4. 首版 `mask_pos` 未 `.bool()`（它是乘过 float `mask_gt` 的 float 张量）。

以上均已修正；最终数字来自修正后的运行。
