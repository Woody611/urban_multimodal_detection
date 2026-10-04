STATUS = PASS

# D′ Classification Representation / Modality Attribution Audit

只读。0 GPU / 0 training / 0 inference / **0 forward** / 0 backward / 0 evaluator /
0 源码·YAML·config·checkpoint 修改 / 0 覆盖既有 diagnostic。**C1–C9 全 PASS。**

```
PROVENANCE_STATUS = PASS
```

---

## §7/§14 — 关键前提：**存在 frozen feature artifact（本轮最重要的发现）**

`diagnostic/p3_feature_space/_v7_vectors.json`（54 MB，4,960 行 = T1 2153 + **V1 2807 = 全部 val GT**）

| 字段 | 语义（本轮已用 ckpt 权重逐行验证） |
|---|---|
| `p3` | `model.model[23]` 输出 —— **P3 颈特征（共享）**，256ch，`_v7_extract.py:51` forward_hook |
| `h` | `Detect.cv3[0][-1]` 的**输入** —— **分类头前置特征**，256ch，`:52` forward_pre_hook |
| `logit_decomp` | `W_gt·h + b_gt` = **GT 类 logit**（stride-8 分类分支，GT 中心 cell） |

**恒等式校验**：`max|resid| = 2.16e-06`（float32 精度），**4960/4960 行成立** ⇒ 字段语义 **PROVEN**。

```
FEATURE_TENSOR_AVAILABLE = YES
  scope = 仅 P3 与 cv3-前置 h 两级、仅 stride-8、仅 GT 中心 cell、仅 val
  无 per-modality 特征 / 无 modality-specific logits / 无梯度 ⇒ 模态归因 NOT_OBSERVABLE
```

---

## Q1 — cls 与 box 共享哪些 representation？

```
P3(256) / P4(512) / P5(1024) 颈特征
        ↓  同一张量 x[i]                       head.py:74
  ┌─────┴─────┐
 cv2[i]     cv3[i]                              head.py:47-49 / 53-60
 64ch       256ch
 Conv3x3x2  (DWConv+Conv)x2
 1x1        Conv2d(256,12,1)
```

- **共享**：L0–L23 全部 backbone + neck，以及 head 的**输入张量** `x[i]`。
- **分离**：两条**独立 `nn.Sequential`**，参数不共享；宽度 **64 vs 256**（分类 tower 宽 4×）。
- 无 DFL 之外的分支、无 attention/gating/dropout、无 modality-specific normalization。

```
CLS_BOX_REPRESENTATION = PARTIAL（shared backbone/neck/head-input；head 内两 tower 完全分离）
```

## Q2 — 三模态在哪第一次融合？SepStem+CLAHE 改了什么？

| 项 | 值 | 来源 |
|---|---|---|
| 合并 | `cv2.merge((b,g,r,ir,depth))` → **[B,G,R,IR,D] uint8** | `base.py:430` |
| **第一次融合** | **`Concat([2,4,6])` → 64ch**（RGB 48 + IR 8 + Depth 8） | `configs/yolo11m_sepstem.yaml:53` |
| 融合方式 | **concat**（非 add / weighted / attention） | yaml |
| 融合后 | 立即 `Conv(128,3,2)` 通道混合；此后**无 modality-specific 分支** | yaml |
| compression | 48+8+8 = 64 → 64 ⇒ ratio **1.00** ⇒ `STRUCTURAL_COMPRESSION_PRESENT = FALSE` | — |
| **CLAHE** | `base.py:334-335`，在 merge **之前**，**仅 IR**（RGB/Depth 不可能受影响），uint8[0,255] | 源码 |
| 模态 range | 三模态合并前**均为 uint8 [0,255]** ⇒ `RANGE_DIFFERENCE = NOT OBSERVED` | 源码 |

## Q3 — frozen evidence 是否显示 representation-level pattern？

**是，而且很强**（1332 个 medium val GT：86 实验组 / 1070 已检出对照 / 164 未检出非 86）。

**★ bias/feature 分解**（`GT类 logit = W·h + b`，b 为全位置共享常数）：

| 组 | n | GT 类 logit 中位 | **特征项 W·h 中位** | ‖h‖ 中位 | ‖p3‖ 中位 |
|---|---:|---:|---:|---:|---:|
| **G-86** | 86 | **−16.719** | **−6.965** | **40.38** | **6.15** |
| 对照（medium 已检出） | 1070 | **+0.883** | **+10.634** | 51.51 | 7.68 |
| 对照（**尺寸匹配** P5–P95） | 899 | +1.224 | +10.950 | 51.72 | 7.75 |
| medium 未检出（非 86） | 164 | −11.010 | −1.252 | 43.95 | 6.53 |

```
86 :  logit −16.719 = bias(−9.758) + 特征项(−6.965)
对照:  logit  +0.883 = bias(−9.758) + 特征项(+10.634)
⇒ 特征项差 = −17.915 nats（尺寸匹配后）；bias 项差 = 0.000 nats
⇒ 塌陷【完全在特征项】，bias 是共享常数，无法产生位置差异
```

**§13.2 class separation**（cos(h, 类均值)，类均值仅用对照组估计）：**7 个 n≥20 的类全部一致**
`0.62–0.70`（86）vs `0.83–0.97`（对照），Δ = **−0.15 ~ −0.34**。

**★ §12 直接可判别性检验**（nearest-centroid top-1 类别准确率）：

| 特征 | G-86 | 对照（已检出） | 对照（未检出） | 随机基线 |
|---|---:|---:|---:|---:|
| **`h`（cls 头前置）** | **0.4651** | **0.9065** | 0.7317 | 0.2907 |
| **`p3`（共享颈特征）** | **0.2558** | **0.6645** | 0.4817 | 0.2907 |

⇒ **①** `h` 的类别信息**减半**（0.465 vs 0.907），但仍**高于随机**（0.291）⇒ 不是"完全无法区分"。
⇒ **②** 共享的 `p3` 在 86 的 cell 上**已到/低于随机水平**（0.256 ≤ 0.291）。
⇒ **③** 同一份 P3 却被 box tower 用于产出 **CIoU ≥ 0.50 认证**的几何。
⇒ **④** 梯度结构：已检出 0.91 > 未检出非86 0.73 > 86 0.47（一致单调）。

## Q4 — 是否有合法 frozen feature artifact 支持模态对 classification 的影响？

**没有。** 无 per-modality feature、无 modality-specific logit、无逐模态梯度。
`diagnostic/modality_dropout/` 是**训练期 dropout 实验**，不是逐模态特征归因。
现有 `depth_reliability_small_miss/_results.npz` 只有 **GT 框内输入图像统计**（非特征）。

```
MODALITY_HARM = NOT_OBSERVABLE
```

## Q5 — class collapse 是否有稳定的 class-specific pattern？

**有差异，但描述性、且与频率无关**（`class_collapse_summary.csv`）：

| class | n_medium | n_86 | collapse_rate |
|---|---:|---:|---:|
| bicycle | 50 | 10 | 0.2000 |
| ball | 11 | 2 | 0.1818 |
| sign | 84 | 13 | **0.1548** |
| light | 102 | 13 | 0.1275 |
| garbage_can | 24 | 3 | 0.1250 |
| seat | 17 | 1 | 0.0588 |
| person | 482 | 25 | 0.0519 |
| animal | 359 | 16 | 0.0446 |
| car | 167 | 3 | 0.0180 |
| **boat** | 12 | **0** | 0.0000 |
| **uav** | 12 | **0** | 0.0000 |

⇒ **不做 ranking**。仅记录两点：`sign` 的 collapse_rate 最高（与上一轮 `cls_lim` 最多一致）；
**极稀有的 `boat`/`uav`（各 12 个 medium）collapse 率为 0，而最高频的 `person` 仍有 5.2%**
⇒ 频率**不预测** collapse（见 H5）。

## Q6 / §24 — Hypothesis matrix

| Hypothesis | Evidence required | Current evidence | Status |
|---|---|---|---|
| **H1** classification representation weak | feature / class separation | nearest-centroid `h` 0.465 vs 0.907；cos 差 −0.15~−0.34（7/7 类一致）；特征项 −17.9 nats（尺寸匹配后） | **PARTIALLY_SUPPORTED** |
| **H2** multimodal fusion harms class separation | modality feature evidence | **无 per-modality 特征/logit/梯度 artifact** | **NOT_OBSERVABLE** |
| **H3** TAL/optimization causes collapse | training trajectory | TAL 静态 bias 已证；**无 trajectory** | **NOT_OBSERVABLE** |
| **H4** shared representation explains divergence | shared arch + feature | 架构 SHARED+SEPARATE；**且共享 P3 在 86 的 cell 上 ≤ 随机（0.256）而 box 仍达 CIoU≥0.50** | **PARTIALLY_SUPPORTED** |
| **H5** class imbalance explains collapse | direct causal evidence | `boat`/`uav` 极稀有却 collapse=0；`person` 最高频仍 5.2% ⇒ 频率不预测 | **NOT_SUPPORTED** |

**H1/H4 为什么不是 SUPPORTED（必须写明的限制）**：
① 我们测的是**到类原型的距离与最近类均值准确率**，不是线性可分性的完整刻画；
② 冻结的 `h`/`p3` 是**训练后**产物，86 从未成为正样本 ⇒「表征弱」与「因未被选择所以没被训练」
是**同一快照的两种读法**，方向不可分离（与上一轮 `TRAINING_CAUSALITY = NOT_OBSERVABLE` 同源）；
③ 只有 stride-8 单层、GT 中心单点，无跨尺度/跨 cell 的一致性证据。

## Q7 — 是否值得做 GPU probe？

**值得，且只有一个问题值得问**（见下）。当前证据已把问题从"是不是 TAL/监督问题"收敛成一个
**可判定的二选一**：86 所在 cell 的共享特征在**训练开始时就**类别不可分（表征/架构），
还是**训练过程中**从可分退化为不可分（优化/选择反馈）。

---

## §16/§17 附：bias 初始化（结构性观察，非 bug）

```
head.py:141   b[-1].bias.data[:nc] = log(5/nc/(640/s)^2)     ；调用点 tasks.py:373
stride-8 初值 = −9.6395 ；D′ 训练后实测 mean = −9.7949（位移仅 −0.15）
imgsz=1280 的等效先验 = log(5/12/(1280/8)^2) = −11.03（比实际初值低 1.39 nats）
```
bias 是**每类一个、全图所有位置共享**的常数 ⇒ **无法解释同类内"有的检出、有的塌陷"**（位置无关项
不能产生位置相关差异）。它只影响整体尺度。初始化只影响训练早期，**不足以解释最终 epoch 的 1e-6**。

---

## ==================================================
## FINAL REPRESENTATION / MODALITY VERDICT
## ==================================================

```
FEATURE_TENSOR_AVAILABLE:
    YES（p3 / h / logit_decomp；仅 stride-8、GT 中心 cell、val 全覆盖；**无 per-modality 特征**）

CLS_BOX_REPRESENTATION:
    PARTIAL
    （共享 backbone + neck + P3/P4/P5 + head 输入张量；head 内 cv2(box,64ch) 与
      cv3(cls,256ch) 是两条完全独立的 tower；无其他 modality/class 专属路径）

FIRST_MODALITY_FUSION:
    Concat([2,4,6]) @ configs/yolo11m_sepstem.yaml:53  ->  64ch
    （RGB 48 + IR 8 + Depth 8；concat，非 add/attention；融合后立即 Conv(128,3,2) 通道混合）

H1_CLASSIFICATION_REPRESENTATION:   PARTIALLY_SUPPORTED
H2_MULTIMODAL_FUSION:               NOT_OBSERVABLE
H3_TAL_OPTIMIZATION:                NOT_OBSERVABLE
H4_SHARED_REPRESENTATION:           PARTIALLY_SUPPORTED
H5_CLASS_IMBALANCE:                 NOT_SUPPORTED

GPU_PROBE_JUSTIFIED:
    YES

IF YES:
    SINGLE_HYPOTHESIS:
        86 所在 cell 的共享 P3 特征的类别可分性，是【训练起点即不可分】还是
        【训练过程中由可分退化为不可分】？
        （即：表征/架构原因 vs 选择-优化反馈原因）

    MINIMAL_PROBE:
        一次带插桩的**短训练**（同一 D′ 配置、同 seed、≤ 30 epoch，不必跑满 300）。
        每 K 个 epoch 冻结评估一次，只用**已冻结的同一把尺子**：
          (a) 在 val 上取与 86 同位置的 cell，记录 p3 向量的 nearest-centroid 类别准确率；
          (b) 记录这些 GT 是否落在 TAL 的 mask_pos 内（正样本成员）；
          (c) 记录它们的 GT 类 logit。
        不新增任何模型/损失/数据改动；只加只读探针。

    REQUIRED_ARTIFACTS:
        - 每 K epoch 的 {cell 坐标, p3 向量, h 向量, GT 类 logit, 是否 mask_pos} 落盘（npz）
        - 该 probe 的 config/seed/hash 与 D′ 逐项一致声明
        - 上述量的时间序列（epoch × 指标），用于判定"何时"退化

    SUPPORT_CRITERION:
        p3 的类别准确率在 **epoch 0/1 就已 ≤ 随机**（≈0.29）⇒ 支持 **表征/架构原因**
        （H1/H4 升级为 SUPPORTED，且不含优化反馈的功劳）

    FALSIFICATION_CRITERION:
        p3 的类别准确率**早期显著高于随机、随后单调下降**，且下降与"失去 mask_pos 成员资格"
        同步 ⇒ 支持 **选择-优化反馈**（H3 的因果部分可升级）；若两者**不**同步 ⇒ 两者皆不成立，
        需回到表征侧寻找其他解释。

GPU USED: 0      TRAINING RUNS: 0      INFERENCE RUNS: 0      SOURCE MODIFICATIONS: 0
==================================================
```

---

## 生成文件（新目录，未覆盖任何既有目录）

```
diagnostic/cls_representation_modality_audit/
    input_manifest.txt / provenance.md
    architecture_path.md / cls_box_branch_audit.md / modality_fusion_audit.md
    normalization_audit.md / classification_init_audit.md
    existing_feature_inventory.csv          既有 frozen artifact 的字段语义清单
    per_gt_representation_audit.csv         1332 行（medium，含 86 标记）
    class_collapse_summary.csv              12 类描述性
    frozen_output_summary.csv
    size_matched_control.csv                ★ 尺寸匹配对照
    class_separation.csv                    ★ cos(h, 类均值)
    hypothesis_matrix.csv
    summary.md / summary.json / consistency_checks.txt / _run.log
    _audit.py / _size_matched.py / _size_matched.json
```

## 异常与修正（不省略）

1. **两次崩在 `sys.path`**：脚本以脚本目录为 `sys.path[0]`，`torch.load` 从 **site-packages** 解析
   `ultralytics`，取不到仓库的 `SilenceChannel` ⇒ ckpt unpickle 失败。
   **这正是 memory 里已记录的陷阱**（`ir-encoding-silent-fallback-incident` 同源：torch.load 用的是
   site-packages 的 ultralytics）。已改为把 `ROOT` 插入 `sys.path` 最前。
2. **一次列名用错文件**：`certified_prefinal_loss` 在 `medium_93_47_attribution/per_gt_attribution.csv`
   （第 33 列），不在 `medium_head_to_final_audit/per_gt_causal_audit.csv`。已修正。
3. **主审计的对照未按尺寸匹配**（86 的 native area 显著更小，MWP p=3e-3）。已补做尺寸匹配对照，
   **差异不被削弱反而略增**（−17.600 → −17.915 nats）⇒ 尺寸混杂被排除为该差异的解释。
4. 前四轮 diagnostic 的 SHA 在 C2 中比对，**全程未变**。
