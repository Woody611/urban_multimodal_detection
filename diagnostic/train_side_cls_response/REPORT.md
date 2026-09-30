# Train-side Native-small GT-class Response Audit

**日期**：2026-09-29
**性质**：**frozen-model read-only inference audit**。授权范围内执行了**一次** frozen D′ forward
（train 原图、augmentation 全 OFF、1599 张、全程 `model.eval()` + `torch.no_grad()`）。
0 training / 0 `model.train()` / 0 backward / 0 `autograd.grad` / 0 optimizer / 0 scheduler /
0 checkpoint 写入 / 0 既有文件修改 / 0 新 submission / 0 官方线上推理。

---

## 1. EXECUTIVE VERDICT

```text
Train native-small GT-class response:
   median logit = +1.4876   (n=1392, 全量 train 1599 图, eval 模式)
   89.1% > 0；仅 1.0% < −10；P10 = −0.128

H1 training-response-deficit:  NOT_SUPPORTED
   froze D′ 对 train native-small GT **已经**产生健康、甚至偏强的 GT-class 响应。

Relation to val G1:
   train native-small (+1.488) **高于** val native-small detected (+0.487)；
   而 val G1 在 **−14.634**。train→G1 落差 = **16.12 logit**。
   且该落差**只集中在 G1 那 38 个目标上**：同图同尺度的非 G1 小目标是 −1.227（正常）。

A/B supervision explanation:  INSUFFICIENT
   A/B 的赤字幅度是 10–34%（target 数值），无法解释 16 logit / ~7 个数量级的响应落差。

Next mechanism:
   H2 泛化 / 表征 — 获得支持（train 正常 + val 特定目标塌），**但本实验未定位是哪一种表征机制**。

Training recommendation:  RED
```

---

## 2. HARD GATES

| Gate | Result | Evidence |
|---|---|---|
| training = 0 | **PASS** | 脚本无 `train`/`optimizer`/`scheduler`/loss 构造；`RUN.log` 无训练字样 |
| backward = 0 | **PASS** | 无 `backward()` / `autograd.grad`；`torch.no_grad()` 全程 |
| optimizer = 0 | **PASS** | 无优化器构造 |
| checkpoint unchanged | **PASS** | `sha256sum -c _ckpt_before.txt` → OK；`RUN.log` 首尾 SHA 均为 `1cae45f7…fae4fda` |
| existing files unchanged | **PASS** | 以本轮开始时刻 `16:35:32` 为界全仓库 mtime 扫描 = **0**；`git status` 前后逐行相同；预测 TXT 目录 mtime 仍为 `2026-09-22` |
| no augmentation | **PASS** | OV 实测：`mosaic=0.0 / mixup=0.0 / copy_paste=0.0 / degrees=0.0 / shear=0.0 / perspective=0.0 / flipud=0.0`，`augment=False`，`mode='val'`；**未调用任何 `.train()`**（无 P3Attack 意义上的 train 模式） |
| class mapping verified | **PASS** | §11b：`label 行 cls` vs 落盘 `cls` 不一致 = 8/11552（且已定位为 dataset 去重导致的索引位移）；另见上一轮 channel 审计 27/27 PASS |
| coordinate mapping verified | **PASS** | §11b：离线重算 stride-8 cell 坐标，**98.04% 逐条精确**，残差 2.0% 已定位（96.9% 恰差 1 cell） |

---

## 3. ACTUAL RESPONSE EXTRACTION PATH

```text
source          ultralytics/nn/modules/head.py:50-61  （cv3 = ModuleList, 末层 nn.Conv2d(c3, nc, 1)）
本轮路径        `_run.py`：register_forward_hook 到 model.model[-1].cv3[li][-1]（li=0,1,2）
tensor          该 1×1 conv 的 **输出**，(1, nc=12, H, W)
                L0: (1,12,160,160)  stride 8
                L1: (1,12, 80, 80)  stride 16
                L2: (1,12, 40, 40)  stride 32
class channel   `out[0, cid, y, x]`，`cid = int(lab["cls"][j])` —— **GT 自己的类，不是 argmax**
空间索引        `sc = 1280.0 / ww`（ww = 该层特征图宽）
                `x = int(min(max(cx / sc, 0), ww - 1))`，`y = int(min(max(cy / sc, 0), hh - 1))`
                `cx = lab["bboxes"][j][0] * 1280`，`cy = lab["bboxes"][j][1] * 1280`
                ⇒ **floor 截断 + clamp**，与 `_v7_extract.py:68-74` 的 `sample()` **逐步相同**
sigmoid         `prob = 1/(1+exp(-logit))`（Check B：max|Δ| = 0.000e+00）
```

**为什么 hook 输出而不是把 head 切到 train 模式**：`Detect.forward` 仅当 `self.training` 时返回 raw 输出，
而 `__init__` 契约**禁止 `model.train()`**。hook `cv3[li][-1]` 的输出在 **eval 模式下同样执行**，
且 `W_c·h + b_c` 正是 `_v7_extract.py:52,135` 用的同一个量（该脚本取 `h` 再投影）。
⇒ **同一定义、零 `.train()` 调用、BN 全程 running stats（= 部署口径）。**

---

## 4. MAIN RESPONSE TABLE

### 4.1 来源 B —— **本轮 forward**（全量 1599 图、eval 模式）

| native scale | n GT | **L0 = stride 8 中位** | P25 | P75 | median σ | % < −10 | % > 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **small** | **1392** | **+1.4876** | +0.954 | +1.775 | 0.8157 | **1.0%** | **89.1%** |
| medium | 5536 | +1.8828 | +1.147 | +2.134 | 0.8679 | 12.5% | 78.7% |
| large\* | 4632 | −17.7045 | −19.855 | −16.023 | 0.0000 | 98.7% | 0.1% |

| native scale | n GT | **L1 = stride 16 中位** | P25 | P75 | % < −10 | % > 0 |
|---|---:|---:|---:|---:|---:|---:|
| small | 1392 | +0.6975 | −1.603 | +1.452 | 3.4% | 63.1% |
| medium | 5536 | **+1.9986** | +1.592 | +2.348 | 2.4% | 92.8% |
| **large** | 4632 | **+2.5511** | −14.780 | +2.886 | 33.5% | 65.3% |

| native scale | n GT | **L2 = stride 32 中位** | P25 | P75 | % < −10 | % > 0 |
|---|---:|---:|---:|---:|---:|---:|
| small | 1392 | −9.3643 | −13.643 | −3.081 | 46.7% | 6.3% |
| medium | 5536 | −13.3676 | −14.795 | −10.693 | 77.5% | 2.5% |
| large | 4632 | −13.6312 | −15.219 | +2.872 | 64.1% | 34.2% |

**\* large 的 L0 值不是"large 分类失败"**：`large` 在 **L1（stride 16）中位 +2.5511**，完全正常。
`_v7_vectors.logit_decomp` 只取 `cv3[0][-1]`（stride 8），因此它对 large 是**错误的层级** ——
上一轮报告 §5 的警告在本轮由**同一批数据的三层 logits 直接证实**。

### 4.2 来源 A —— 已有 `_v7_vectors.json` **T1 域**（train 原图、aug OFF、300 图、head=train 模式）

| native scale | n GT | median | P25 | P75 | median σ | % < −10 | % > 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **small** | 238 | **+1.3837** | +0.656 | +1.685 | 0.7996 | 1.7% | 87.4% |
| medium | 1015 | +1.7885 | +0.929 | +2.162 | 0.8567 | 12.3% | 78.3% |
| large\* | 900 | −17.8500 | −19.738 | −16.242 | 0.0000 | 98.6% | 0.2% |

### 4.3 两个来源一致

```text
来源 A（300 图, head=train 模式）native-small 中位 = +1.3837
来源 B（1599 图, eval 模式）    native-small 中位 = +1.4876
```

⇒ **BN 模式（batch vs running 统计）对该量的影响只有 +0.10 logit**（§11 Check A：ρ=+0.9836）。
⇒ 结论在**部署口径**下成立，不只是某个测量口径的产物。

---

## 5. TRAIN vs VAL COMPARISON

```text
Population                          n     median GT-class logit    来源
Train native-small (来源 B, L0)   1392            +1.4876          本轮 forward（全量, eval）
Train native-small (来源 A)        238            +1.3837          _v7_vectors T1（300 图）
Val native-small detected          223            +0.4873          _v7_vectors V1
Val native-small missed            148           -12.1473          _v7_vectors V1
Val native-medium detected        1070            +0.8828          _v7_vectors V1
Val native-medium missed           250           -14.1531          _v7_vectors V1
Val **G1 自身**                     38           -14.6341          _v7_vectors V1（= L17 audit 的 base_logit 中位）
```

⚠ **train / val 不是 paired samples**（不同图像、不同划分），**不做配对检验**。

**读法**：

```text
train native-small (+1.488)  >  val native-small detected (+0.487)
```

⇒ **frozen D′ 对训练集 native-small GT 的 GT-class 响应，比它对验证集上"被成功检出"的
   同尺度目标还要高约 1.0 logit。**
⇒ 而 val G1 在 −14.634 ⇒ **train→G1 落差 = 16.12 logit（概率上约 7 个数量级）**。

---

## 6. DISTRIBUTION ANALYSIS

### 6.1 分位数（native-small）

| 来源 | n | P10 | P25 | P50 | P75 | P90 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B（全量, eval）** | 1392 | −0.128 | +0.954 | **+1.488** | +1.775 | +1.950 | −20.32 | +2.66 |
| A（300 图） | 238 | −0.212 | +0.656 | +1.384 | +1.685 | +1.922 | −18.44 | +2.28 |

### 6.2 直方图（来源 B，n=1392）

```text
[ −60,−20)     1    0.1%
[ −20,−12)    10    0.7%
[ −12, −8)    17    1.2%
[  −8, −4)    10    0.7%
[  −4, −2)    27    1.9%
[  −2,  0)    87    6.2%   ###
[   0,  2)  1136   81.6%   #########################################
[   2, 20)   104    7.5%   ####
```

**低响应占比 (< −10) = 1.0%（14/1392）；高响应 (> 0) = 89.1%。**

### 6.3 是否 bimodal

**不是。** 形状是**以 +1.5 附近为单峰主导 + 一条很小的低响应尾**（低尾 1.0%）。
两个 mode **规模悬殊**，不构成协议 §8 意义上的「双峰结果」⇒ **不需要按双峰条款分别报告 mode 的数量/分位数/
是否按 class 或 image 聚集**。（来源 A 形状相同：81.9% 落在 (0,2)，低尾 1.7%。）

---

## 7. CLASS ANALYSIS（native-small，来源 B，n=1392）

| class | n small GT | median logit | P25 | P75 | % < −10 |
|---|---:|---:|---:|---:|---:|
| person | 706 | +1.3021 | +0.652 | +1.665 | 0.8% |
| boat | 17 | +1.1782 | +0.717 | +1.412 | 0.0% |
| animal | 107 | +1.6303 | +1.284 | +1.827 | 0.0% |
| seat | 30 | +1.7098 | +1.536 | +1.850 | 0.0% |
| sign | 124 | +1.6887 | +1.492 | +1.862 | 0.0% |
| bicycle | 76 | +1.5639 | +1.194 | +1.694 | 0.0% |
| car | 97 | +1.4754 | +1.161 | +1.710 | 0.0% |
| ball | 22 | +1.7078 | +0.748 | +2.069 | 4.5% |
| light | 56 | +1.5538 | +1.068 | +1.770 | 3.6% |
| garbage_can | 39 | +1.2874 | +0.901 | +1.738 | 0.0% |
| uav | 118 | +1.8571 | +1.575 | +2.002 | 4.2% |

（tricycle 在 native-small 里 n=0，未列出。）

**n≥30 的 9 个类**：median 范围 [+1.287, +1.857]，**类间跨度 = 0.570 logit**。
⇒ **0.570 logit ≪ train→G1 的 16.12 logit 落差** ⇒ **不是某几个类特有的问题**。

---

## 8. HYPOTHESIS MATRIX

| Hypothesis | Evidence | Verdict |
|---|---|---|
| **H1** training response deficit | train native-small 中位 **+1.488**（n=1392，全量，eval），89.1% > 0，**高于** val native-small detected (+0.487)；两个独立来源一致（+1.384 / +1.488）；类间跨度仅 0.570 | **C — Not supported** |
| **H2** feature / spatial support（泛化） | train 响应正常而 val G1 在 −14.634 ⇒ 落差 16.12 logit；且该落差**只集中在 G1 那 38 个目标**：同图同尺度非 G1 小目标中位 **−1.227**（正常），同图 medium 中位 **−0.493**（**比别处的 medium 还高**）⇒ **既非图像级、也非尺度级，是目标级** | **B — Partially supported**（方向明确；但本实验**未定位**是哪一种表征机制，也未证明因果） |
| supervision quantity as primary cause | A 是**事实**（n_pos 4 vs 10），但 train 响应健康 ⇒ 不可能是**主因** | **C — Not supported（as primary）** |
| target strength as primary cause | B 的赤字幅度 10–34%，而落差 16 logit ⇒ 幅度差一个数量级以上 | **C — Not supported（as primary）** |
| class-specific mechanism | 类间跨度 0.570 logit；`% < −10` 逐类 ≤4.5% | **C — Not supported** |

**派生的一条零成本佐证（用已有 `_v7_vectors` V1 全 2807 val GT，无新 forward）**：

| subset | n | median | % < −10 | % > 0 |
|---|---:|---:|---:|---:|
| G1 图内 native-small（全部） | 115 | −6.907 | 42.6% | 23.5% |
| 　其中 **G1 自身** | **38** | **−14.634** | **89.5%** | **0.0%** |
| 　其中 **同图非 G1** | **77** | **−1.227** | 19.5% | 35.1% |
| G1 图内 **medium** | 125 | −0.493 | 24.0% | — |
| 非 G1 图的 medium | 1195 | −1.034 | 28.0% | — |

⇒ 在**同一批图像**里：G1 的 38 个目标塌到 −14.6，而同图同尺度的其它 77 个小目标在 −1.2（正常），
medium 甚至略高于别处。**赤字不外溢 → 失败是「目标级」的，不是图像级、不是 image×scale。**

**跨来源锚点**：§8b 算出的 G1 median −14.634 与上一轮 L17 audit 的 `base_logit` 中位 **−14.634** 逐位一致
（那是另一次独立 forward）⇒ 本轮的 GT-class 测量与既有链条对齐。

---

## 9. WHAT THIS EXPERIMENT DOES NOT PROVE

1. **train response ≠ 因果**。测得"train 响应健康"只说明**响应存在**，不证明它与 val 失败之间存在因果关系。
2. **train / val 不是 paired**。两个 population 的并置对照，**不是**配对比较，也未做任何配对检验。
3. **L0 (P3/stride-8) response ≠ 整检测器响应**。对 small 而言 P3 是正确的层级；对 medium/large **不是**。
   §4 的三层表已给出正确层级的读数（large 在 L1 = +2.55），但**任何跨尺度的"检测器级"比较仍需谨慎**。
4. **不得从相关推因果**。例如"G1 落在 −14.6"与"训练侧监督正常"并存，**不能**推出某个具体表征机制。
5. **本实验不能区分**「该目标根本不可辨（标注/遮挡/过小）」与「可辨但表征失败」——
   两者都会表现为 GT-class 响应 ≈ 0。区分它们需要 image-domain / 人工核查级别的证据。
6. **口径 caveat（本轮新发现，回溯适用于整条 audit 链）**：见 §11.6。

---

## 10. FINAL DECISION

```text
RED
```

**不允许训练。**

理由：
- **H1 被明确否定**（train 响应健康且高于 val detected）。监督侧（A/B）的两条机制**被降级为"非主因"**。
- **H2 只到 B（partially supported）**，且本轮**没有定位**到任何具体可干预的表征机制。
- 此前唯一被识别为可干预的机制（抬升小目标类别响应，即 E1 / L17 路线）**已在前两轮实测收口**。
- ⇒ 没有任何机制达到「机制链充分」⇒ 不满足 GREEN 的条件。

**下一步（若将来授权）只能是一个明确指定的 read-only audit**，且本轮**不自行启动**。
按协议，RED 下不提出任何训练方案。

---

## 11. 方法与口径（含本轮自己的错误更正）

### 11.1 §11 SANITY CHECKS

| Check | 结果 |
|---|---|
| **A** class channel | 用**两个独立来源**在同一 2153 个 GT 上比同量：Spearman **ρ = +0.98355**，median\|Δ\| = 0.286。两者读的是**同一个类通道、同一空间位置**（Δ 来自 BN 模式差异）。另：上一轮 channel 审计已 27/27 PASS（`W[c]·h+b[c]` 复现 `logit_decomp`，max\|Δ\|=2.3e-06，换行则失败 median\|Δ\|=7.42） |
| **B** sigmoid | `prob_L0 == sigmoid(logit_L0)`，**max\|Δ\| = 0.000e+00** ✅ |
| **C** spatial coordinate | §11b 离线重算：**98.04% 逐条精确**（11326/11552）；不一致 226 条中 **96.9% 恰差 1 个 cell**（cell 边界 floor 的亚像素差，dataset 用 `min(math.ceil(w*r), imgsz)`），其余 7 条来自重复标签的索引位移 |
| **D** no augmentation | OV 实测值见 §2；`mosaic=0.0` 等全部为 0，`augment=False`，`rect=False` |
| **E** frozen checkpoint | 首尾 SHA256 均为 `1cae45f75693f541…fae4d959f40fae4fda` |

### 11.2 §10 FORWARD 数据口径

```text
train 图像文件        = 1600
dataset 载入          = 1599      （`003107.png` 坐标越界被 dataset 自动剔除）
实际处理              = 1599 / 1599（全部成功）
GT 行                 = 11560
丢弃                  = 0（无静默丢弃；无任何排除未记录）
native-small GT       = 1392
无法映射 native 的 GT  = 0
非有限 logit 的 GT     = 0
dataset 去重           = `000013_046_00000202.png`、`hehe_10_00000044.png` 各去 1 条重复 label
（原始 label 去重后 11567 行；forward 得 11560 行；差 7 = dataset 的去重/剔除口径）
```

### 11.3 我自己的三处错误（如实记录）

| # | 我写成 | 实测 | 更正 |
|---|---|---|---|
| 1 | `_run.py` 的 `coord_resid` 检查（告警 11446 条） | 它拿**原始图像归一化**坐标去比 dataset 的 **letterbox 画布归一化**坐标 —— 两者本就不同 | 该检查**作废**；改为 §11b 的离线重算（98.04% 精确） |
| 2 | 首版报告草稿里 native-large 的 L0 低值曾被当作"large 响应失败" | large 在 **L1 = +2.55** 正常；L0 只是层级不匹配 | 按 §5 要求标注为 **P3-only reference**，不参与结论 |
| 3 | 一度准备只跑 100 图试跑当结果 | 协议 §14 要求优先全量、不得静默改子集 | 改为**全量 1599 图**（38 分钟），试跑结果单独存为 `_rows_pilot100.json` 不参与结论 |

**没有任何一项更正改变 §8 的判定方向。**

### 11.4 并未使用（协议 §12 禁止）的替代

本轮**没有**用 val TXT、val logits、训练 loss log 来替代 train-side 测量。
来源 B 是**真的** train 图像 + train GT + frozen model 的 forward。
来源 A（`_v7_vectors` T1）本身就是**先前的** train 图像 + train GT + frozen model forward，不是替代物，
且它与来源 B 在 2153 个重叠 GT 上 ρ=+0.9836 互证。

### 11.5 forward 规模

**没有**"因为太大而改子集"：全量 1599 图在 38 分钟内完成（~1.9 s/图，CPU），故未触发 §14 的 STOP 条件。

### 11.6 ⚠ 口径 caveat：整条 audit 链用的是 **percentile**，而 D′ 训练声明的是 **CLAHE**

```text
`ultralytics/data/base.py:333`  ir_encoding = getattr(self.hyp, "ir_encoding", "percentile") or "percentile"
`ultralytics/data/build.py:104`  build_yolo_dataset(..., hyp=cfg)
所有既有 audit 脚本的 `get_cfg(overrides=OV)` 中 **OV 都没有 `ir_encoding` 键**
  ⇒ get_cfg 合并 default.yaml（`ir_encoding: percentile`）⇒ **落到 percentile 分支**
而 D′ 的 `configs/train_rgbid_sepstem_clahe.yaml` 显式声明 `ir_encoding: clahe`，
`scripts/train.py:499-500` 只在 train_cfg 有该键时才传下去。
```

⇒ **`_v7_vectors` / `_v6_domains` / `_v4_internal` / `_v11`–`_v14` / `_sge_*` / `_v5_replay` /
`candidate_density` / `e2` / 以及本轮，全部是在 percentile IR 上测的，不是 CLAHE。**
⇒ 本轮**刻意沿用 percentile**，以保证与既有 val 侧对照值（+0.487 / −12.147 等）**同口径可比**；
   若改按 CLAHE 重测，则两侧都要重测才能比较，而协议只授权**一次** forward。
⇒ 影响评估：train 与 val 的**相对**比较仍然成立（两者同口径）；绝对 logit 水平可能整体平移。
**这是本轮发现的、回溯适用于整条 audit 链的口径问题，如实记录，未做任何补救性 forward。**

---

## 12. Provenance

```text
TRAINING        = 0
BACKWARD        = 0
OPTIMIZER_STEP  = 0
CHECKPOINT_MODIFIED   = 0
EXISTING_FILES_MODIFIED = 0
PREDICTION_TXT_MODIFIED = 0

GIT_STATUS_BEFORE = 73 行  →  GIT_STATUS_AFTER = 73 行，逐行相同（唯一新增为本轮目录）
GIT_HEAD_BEFORE   = f659609c28f6b4f500212197d5229550a36d12d8
GIT_HEAD_AFTER    = f659609c28f6b4f500212197d5229550a36d12d8
CHECKPOINT_SHA256_before = 1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda
CHECKPOINT_SHA256_after  = 1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda
```

**「0 既有文件被改」的验证口径**：以本轮开始时刻 `2026-09-29 16:35:32` 为界，
全仓库（排除 `.git` / `__pycache__` / 本轮自身目录）扫描 mtime 晚于该时刻的文件 ⇒ **命中 0**。
预测 TXT 目录 `diagnostic/sepstem_clahe/best_full/results` 的 mtime 仍为 `2026-09-22 08:20:20`（未动）。
上一轮的 `diagnostic/train_side_cls_supervision/`（mtime `16:29:35`）**未被覆盖**。

### 读取对象 SHA256[:16]

| 对象 | SHA256[:16] |
|---|---|
| `runs/…_sepstem_clahe/weights/best.pt` | `1cae45f75693f541` |
| `configs/yolo11m_sepstem.yaml` | `9b14f29460733384` |
| `data/processed/rgbid_split_train/dataset.yaml` | `3bc36c31e913072f` |
| `diagnostic/p3_feature_space/_v7_vectors.json` | `977593ae00126100` |
| `diagnostic/small_object_domains/_v6_domains.json` | `3505d9160c47aae8` |
| `diagnostic/small_object_cause_v2/_v3_analysis.json` | `41d753f5fe24f372` |
| `diagnostic/p3_feature_space/_v11_ids.json` | `e761e673e239fa2a` |

### NEW_FILES（全部位于 `diagnostic/train_side_cls_response/`）

```text
REPORT.md              本报告
_run.py                单次冻结 forward（无 .train()；hook cv3 末层输出）
_analyze.py            分析（两来源交叉验证）
RUN.log                forward 运行日志（含首尾 checkpoint SHA）
ANALYZE.log            分析日志
_rows.json             forward 原始落盘（11560 行 × 3 层 logit/prob/坐标）
_rows_pilot100.json    吞吐试跑（100 图），**不参与任何结论**
_tables.json           结构化结果
_ckpt_before.txt       checkpoint sha256sum 基线
_git_before.txt         git status + HEAD（before）
_git_after.txt          git status + HEAD（after）
```

---

**本轮结束，停止。不训练、不设计干预、不再补充 forward、不自行启动下一轮。**
