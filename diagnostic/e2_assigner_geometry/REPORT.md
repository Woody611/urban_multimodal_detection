# E2 前置可行性审计 — Assigner 几何

**日期**：2026-09-28
**性质**：**READ-ONLY**。0 训练 / 0 optimizer.step / 0 backward / 0 改任何既有源码或配置 /
0 改 checkpoint / 0 test inference / 0 submission。仅 `torch.no_grad()` 的只读 forward
与 assigner 的离线重算；**未修改 assigner**。

新增文件全部位于 `diagnostic/e2_assigner_geometry/`，未覆盖任何既有文件。

---

## OBSERVED FACTS

### 1. 真实 assigner（从**运行中的对象**读出，不采信记忆）

```
type        = TaskAlignedAssigner
topk        = 10
alpha       = 0.5
beta        = 6.0
num_classes = 12      eps = 1e-09
__init__ signature = (self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-09)
loss.py:438        = self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc,
                                                          alpha=0.5, beta=6.0)
v8DetectionLoss.__init__(self, model, tal_topk=10)   ⇒ tal_topk = 10
```
⇒ **(topk, alpha, beta) = (10, 0.5, 6.0)** 与 §2 已知一致 ✓（实测断言，非假设）

### 2. 真实 alignment 公式与 IoU 实现（源码行）

```
align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
def iou_calculation(self, gt_bboxes, pd_bboxes):
    return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)
```

⚠ **两点必须记录**：

1. `bbox_scores` 是**该 GT 类别的预测分数**（`pd_scores[ind[0], :, ind[1]]`），不是 max over classes。
   所以 `align = cls_of_that_class^alpha * CIoU^beta`。
2. **IoU 用的是 CIoU（不是纯 IoU），并且 `.clamp_(0)`**。CIoU = IoU − ρ²/c² − α·v 可以为负，
   被 clamp 成 0 后 `align = 0`。⇒ **中心偏离越大，CIoU 越可能先归零**，对 small GT 尤其不利。

### 3. 全流程结构（实读）

```
select_candidates_in_gts   : anchor **中心** 必须严格落在 GT 框内 (bbox_deltas.amin(3) > eps)
  ↓ mask_in_gts
get_box_metrics            : bbox_scores(该 GT 类) + CIoU.clamp(0)  → align_metric
  ↓
select_topk_candidates     : torch.topk(align_metric, self.topk)  ← **每个 GT 独立取 topk**
                             （并丢弃在 topk 内出现 >1 次的 anchor）
  ↓ mask_topk
mask_pos = mask_topk * mask_in_gts * mask_gt
  ↓
select_highest_overlaps    : 冲突消解（一个 anchor 归 IoU 最大的那个 GT）
  ↓
get_targets + norm_align_metric = align * pos_overlaps / pos_align_metrics
```

**关键结构推论（待 sweep 验证）**：
`mask_pos` 同时受 `mask_in_gts`（几何上限 = 落在框内的 anchor 中心数）与 `mask_topk` 约束。
⇒ 若某 GT 的 `n_cand = #(框内 anchor 中心) ≤ topk`，则它的**全部**候选都被选中，
   与排名无关 ⇒ 此时 **n_pos 对 alpha / beta / topk 完全不敏感**。

### 4. FULL_ASSIGNER_REPLAY = **TRUE**（并已自检）

`topk / alpha / beta` 是 `TaskAlignedAssigner` 的**实例属性**，在 `get_pos_mask` 与
`select_topk_candidates` 内部被读取。因此在**完全相同的输入张量**上修改这三个属性并重调
`_forward()`，等价于「用该组超参跑一次完整 assigner」——包含
candidate selection → top-k selection → 冲突消解 → target_scores 归一化的**全流程**。
**不是**只重算 alignment 公式。

**自检（每次运行都执行）**：`ProbeAssigner`（我的插桩子类）与**父类** `TaskAlignedAssigner`
在同一输入、同一 baseline 参数下的五个返回值必须**逐位一致**：

```
[SELF-CHECK] ProbeAssigner vs 父类逐位一致: True  (n_gt=14, fg=138)
```
（`ProbeAssigner` 仅覆写 `get_pos_mask`/`get_box_metrics`/`_forward` 以留下中间量，
 `get_box_metrics` 是父类的**逐行**副本。）

### 5. 审计过程中自查出的两个实现错误（已修，记录以备复核）

| # | 错误 | 根因 | 影响 |
|---|---|---|---|
| 1 | `IndexError: tensors used as indices must be long/int/byte/bool` | 我早前用 `grep -vE "^\s+(...\|mask_gt\|...)"` 读源码时，**把父类的第一行 `mask_gt = mask_gt.bool()` 过滤掉了**，复刻时漏抄 | 读源码的过滤失误，非 assigner 问题；已改为逐行副本 |
| 2 | `KeyError: 'bbox_scores'` | `get_pos_mask` 里 `self.last = dict(...)` 重新赋值，抹掉了 `get_box_metrics` 写入的 `bbox_scores` | 已改为 `self.last.update(...)` |
| 3 | 首版 0 条记录 | 忘记调用 `INSTR.install()`，lineage 挂载点未安装 ⇒ 全部样本因 uid 缺失被跳过 | 已修 |

### 6. Provenance

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

`git status`：`M` 项与上一轮 E1 结束时**完全相同**（无本轮新增改动）；
`MODIFIED_EXISTING_FILES = 0`（本轮）。完整 sha 清单见 `_e2_replay_meta.json`。

---

## REPRODUCTION

### §10 Gate — 必须复现 Exposure Audit 的 baseline

`alpha=0.5 beta=6 topk=10`，n_pos 用 **pre-conflict**（与 §13 同口径）：

| class | n | n_pos(pre) | n_pos(post) | align | §13 n_pos | §13 align | match |
|---|---:|---:|---:|---:|---:|---:|---|
| small | 250 | **4.0** | 4.0 | **0.354** | 4.0 | 0.357 | ✅ |
| medium | 934 | **10.0** | 10.0 | **0.703** | 10.0 | 0.688 | ✅ |
| large | 736 | **10.0** | 10.0 | **0.826** | 10.0 | 0.826 | ✅ |

```text
REPRODUCTION PASS  →  继续 sweep
```
（容差：n_pos ±1.0、align ±0.05。small align 差 0.003、medium 差 0.015，均在容差内。）

**一处口径说明（已实测不影响 gate）**：本 replay 的 `AG` 以 uid 为键 ⇒ 同一 native GT 在一个
epoch 内多次实例化时只保留最后一次；Exposure Audit 是**按实例**逐条记录。在既有监督数据上
实测该差异的影响：small `4.0→4.0 / 0.357→0.354`、medium `10.0→10.0 / 0.688→0.703`、
large `10.0→10.0 / 0.826→0.826` ⇒ 可忽略。

### 自检

`ProbeAssigner` 与父类在同一输入、同一 baseline 参数下五个返回值**逐位一致**：
`[SELF-CHECK] ProbeAssigner vs 父类逐位一致: True`

---

## SWEEP RESULTS

### §3 Alignment decomposition

| class | n | **n_cand 中位** | **maxIoU 中位** | **maxCls 中位** | maxAlign 中位 | align<.01 |
|---|---:|---:|---:|---:|---:|---:|
| **small** | 250 | **4.0** | **0.8805** | **0.7698** | 0.35699 | 6.8% |
| medium | 934 | 20.0 | 0.9610 | 0.8789 | 0.70466 | 0.7% |
| large | 736 | **169.0** | **0.9795** | **0.9352** | 0.82589 | 0.3% |

**对数加性分解** `Δlog(align) = α·Δlog(cls) + β·Δlog(IoU)`（small 相对 large，均为负 ⇒ 两项都是劣势）：

| α / β | Δlog(align) | α·Δlog(cls) | β·Δlog(IoU) | cls 占比 | **IoU 占比** |
|---|---:|---:|---:|---:|---:|
| 0.50 / **6**（baseline） | **−1.3875** | −0.1738 | **−1.2137** | 12.5% | **87.5%** |
| 0.50 / 4 | −0.9829 | −0.1738 | −0.8091 | 17.7% | 82.3% |
| 0.50 / 2 | −0.5783 | −0.1738 | −0.4046 | 30.0% | 70.0% |
| 0.75 / 6 | −1.4743 | −0.2606 | −1.2137 | 17.7% | 82.3% |
| 0.25 / 6 | −1.3006 | −0.0869 | −1.2137 | 6.7% | 93.3% |

⇒ 对应 §3 的选项：**A（IoU geometry）为主 + C（β 放大）**。分类分量（`cls`）只占 12.5%，
且提高 α 到 0.75 也只能把它推到 17.7%。**B（classification score 主导）不成立。**

### §4/§7/§8 全 27 组 sweep —— 三条硬结论

**结论 1：small 的 `n_pos` 在全部 27 组里恒为 4.0，`zero%` 恒为 2.4%。**

```
alpha beta topk  class   n  n_pos med  align med   zero%  align<.01%
 0.25    2    5  small 250        4.0      0.687    2.4%        4.4%
 0.50    6   10  small 250        4.0      0.354    2.4%        9.2%   <- baseline
 0.75    6   20  small 250        4.0      0.324    2.4%       10.8%
 ...（27 组全部为 4.0 / 2.4%）
```

**结论 2：small 的 `align` 只随 β（与 α）变，完全不随 topk 变。**

§7 β 专项（α=0.5, topk=10 固定）：

| β | class | n_pos | align | zero% | align<.01 | **small/large align** | small/medium |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | small | 4.0 | **0.635** | 2.4% | 5.2% | **0.6951** | 0.7523 |
| 2 | large | 10.0 | 0.914 | 0.0% | 0.7% | | |
| 4 | small | 4.0 | 0.472 | 2.4% | 6.8% | **0.5443** | 0.6125 |
| 4 | large | 10.0 | 0.867 | 0.0% | 0.8% | | |
| **6** | small | 4.0 | **0.354** | 2.4% | 9.2% | **0.4288** | 0.5035 |
| **6** | large | 10.0 | 0.826 | 0.0% | 0.8% | | |

**结论 3：topk 只改 medium/large 的 n_pos，且完全不给 align 带来任何改善。**

§8 topk 专项（α=0.5, β=6 固定）：

| topk | class | n_pos(pre) | n_pos(post) | align | zero% | align<.01 |
|---:|---|---:|---:|---:|---:|---:|
| 5 | small | **4.0** | 4.0 | 0.354 | 2.4% | 9.2% |
| 10 | small | **4.0** | 4.0 | 0.354 | 2.4% | 9.2% |
| 20 | small | **4.0** | 4.0 | 0.354 | 2.4% | 9.2% |
| 5 | medium | 5.0 | 5.0 | 0.703 | 0.4% | 1.7% |
| 10 | medium | 10.0 | 10.0 | 0.703 | 0.4% | 2.2% |
| 20 | medium | **19.0** | **18.0** | **0.703** | 0.4% | 2.4% |
| 5 | large | 5.0 | 5.0 | 0.826 | 0.0% | 0.4% |
| 10 | large | 10.0 | 10.0 | 0.826 | 0.0% | 0.8% |
| 20 | large | **20.0** | 20.0 | **0.826** | 0.0% | 0.8% |

⇒ **§8 的五个问题**：
1. small `n_pos` 是否从 4 显著增加？ **否 —— 恒为 4.0（27/27 组）**
2. `best_align` 是否改善？ **否（topk 完全不改变 align）**
3. `zero-pos` 是否下降？ **否（恒为 2.4%）**
4. medium/large 是否被过度增加 positives？ **是 —— 10 → 19/20（约 ×2）**
5. topk=20 是否只是增加低质量 positives？ **是 —— medium/large 的 align 一字不变（0.703 / 0.826），
   即新增的 9~10 个 positives 全部**低于**原有的最好那个，纯粹是数量膨胀**

---

## MECHANISTIC INTERPRETATION

### 1. 为什么 small 的 n_pos 对 α/β/topk 完全不敏感（机制）

`select_candidates_in_gts` 要求 anchor **中心**严格落在 GT 框内 ⇒ `n_cand` 是 `n_pos` 的**几何上限**。
实测 `n_cand` 中位：**small 4 / medium 20 / large 169**。

small 的 `n_cand = 4 < topk = 5`（最小的一档）⇒ 它的框内候选**本来就全部**被 topk 选中，
且 `align_metric` 在框外恒为 0（`get_box_metrics` 只在 `mask_gt` 内填值）⇒ **排名完全不参与**。
改变 α/β 只改变框内 4 个候选的**相对与绝对数值**，不改变「谁被选中」。

> ⇒ **对本数据集的 native-small GT，assigner 的超参 (α, β, topk) 在结构上无法改变其正样本集合。**

### 2. 为什么 β=2 时 small align「变好」却是假象

`align = cls^α · IoU^β`。small 的 `maxIoU = 0.8805 < 1`，故 `IoU^2 = 0.775 > IoU^6 = 0.466`。
把 β 从 6 降到 2，**每个候选的 align 都被同一个单调变换抬高**，于是
`small align 0.354 → 0.635`、`small/large 比 0.4288 → 0.6951`、`align<.01 9.2% → 5.2%` 全部"改善"。

**但候选集合、(cls, IoU) 的配对数、正样本归属一个都没变。** 这是纯粹的重标定，
不是「assigner 变得对 small 更友好」。

⇒ **`best_align` 不可跨 (α,β) 比较** —— 这是本审计最需要强调的方法学结论
（否则 §9 的第一条判据「best_align 有明确改善」会被这个假象直接满足）。

### 3. 劣势的真正来源

| 渠道 | small | large | 说明 |
|---|---:|---:|---|
| `n_cand` | 4 | 169 | 锚点中心落在框内的数量，**几何**决定 |
| `maxIoU` (CIoU) | 0.8805 | 0.9795 | 最好候选的定位质量 |
| `maxCls`（该类分数） | 0.7698 | 0.9352 | 分类置信 |
| → `maxAlign`（β=6） | 0.357 | 0.826 | 前三者经 `cls^0.5 · IoU^6` 合成 |

对数分解给出 **IoU 项贡献 87.5%**（β=6 时）。压低 β 会同时压低**所有人的** IoU 项，
所以 small/large 的比值改善只是「指数压缩了差距」，差距本身（Δlog IoU ≈ −0.202）没有变化。

---

## CANDIDATE STATUS

### §9 逐条判据（对全部 27 组）

| 判据 | small 侧 | 结论 |
|---|---|---|
| best_align 有明确改善 | 仅 β 改变时的**重标定**，非 assignment 改善 | ❌ |
| align<.01 不恶化 | 不变或"改善"，但同为重标定 | ⚠ 无信息 |
| zero-pos 不恶化 | 恒为 2.4% | ✅（但也没改善） |
| n_pos 增加不是单纯低质量膨胀 | small **n_pos 完全不变**；topk=20 对 medium/large 是纯膨胀 | ❌ |
| medium/large best_align 不系统性下降 | 不下降（topk 完全不影响 align） | ✅ |
| **mechanism：能解释为什么该 setting 改善 native-small 的 alignment** | **不能 —— 没有任何 setting 改变 small 的正样本集合** | ❌ |

```text
NOT A STRONG CANDIDATE
```

### §12 裁决

```text
C — NOT SUPPORTED
```

**依据**：改变 α/β/topk **无法**解决 native-small 的 alignment 劣势。决定性的机制事实是
small 的 `n_cand = 4 < topk` ⇒ 它的正样本集合由**几何**完全决定，与这三个超参无关；
而唯一让 `best_align` 数字变好的方向（降低 β）是单调重标定，不改变任何配对。

**注意**：C 说的是「这三个超参解决不了」，**不是**「small 的 alignment 劣势不存在」——
劣势存在且已在上一轮量化（native-small `align` 中位 0.357 vs large 0.826）。

---

## LIMITATIONS

1. **`n_pos` 口径**：主表用 **pre-conflict** 以匹配 §13；同时已记录 **post-conflict**
   （`n_pos(post)`）。两者在 baseline 完全一致；仅在 topk=20 时 medium 出现 19→18 的差异
   （冲突消解）。结论不受影响。
2. **`maxIoU` 差距混杂了两个因素**：small 的候选数少（4）本身就是「少抽几次」，
   而 max over 4 天然低于 max over 169。**本审计无法把「抽得少」与「抽得差」分开** —— 需要
   一个反事实实验（例如把 large GT 的候选下采样到 4 个）才能分离。这一点**未被解决**。
3. **单一 checkpoint**：所有预测来自 D′ best.pt。assigner 行为依赖模型输出，换模型会变。
4. **Monte-Carlo**：augmentation realization 与训练 RNG 不可复现（同上轮）。但 sweep 内部
   **固定了同一批 realization 与同一批预测**，是**配对比较**，故同一 GT 跨 setting 的差异是精确的。
5. **样本量**：250 图 × 3 epoch ⇒ 1920 个 native GT（small 250）。对中位数足够，对尾部（<8px）不足。
6. **未做假设检验**；但 27/27 组 n_pos 恒等、topk 对 align 的零效应都是**精确相等**，非统计问题。
7. **不能外推到别的 assigner 家族**（如 ATSS / SimOTA / 无锚框）——本审计只回答了
   「当前 TAL + 这三个超参」。

---

## RECOMMENDATION FOR NEXT STEP

```text
不建议进入 E2 配置阶段（不存在合格的 assigner setting 候选）。
```

如果仍要沿「small-object supervision」这条线继续，本审计**支持且仅支持**以下方向：
把问题重新表述为**锚框几何/候选密度**问题（small 的 `n_cand` 中位仅 4），
而不是 TAL 的超参问题。但**任何具体改法都未经检验，本审计不提出、也不背书**，
且须先注意 `reports/next_optimization_review.md` 已记录「加检测尺度」家族 3/3 证伪。

**下一轮之前必须先解决 LIMITATION #2**（分离「候选少」与「候选差」），否则无法判断
「提高 small 的候选密度」是否有意义。

```text
HOLD — 不训练 / 不改 assigner / 不生成 E2 config / 不提出多个训练变体。
```
