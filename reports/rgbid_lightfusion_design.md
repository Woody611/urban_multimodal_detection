# RGBID Light Fusion — 设计审计报告

**日期**：2026-09-18
**性质**：**只读代码审计 + 预训练加载模拟**。未训练、未创建训练配置、未修改任何既有文件、未上传线上。
**结论**：**`LIGHT_FUSION_CONFIG_READY = NO`** —— 设计评审 **FAIL**，按 §7 / §11 的停止条件**不创建最终配置**。

---

## 0. 一句话结论

> **当前 RGBID baseline 的第一层 `Conv(5→64, k=3, s=2)` 本身已经就是「可学习的轻量通道融合层」**
> —— 它直接把 5 个模态通道（`[R,G,B,IR,Depth]`）线性混合成 64 通道，且已经带有
> **RGB 预训练权重 + IR/Depth = mean(R,G,B)** 的初始化。
> 在此之上再加一个 `5→3` projection 属于**重复实现**，并且**实测会彻底破坏预训练权重加载**
> （标准 load() 只匹配 **14/655**，`_transfer_rgb_pretrained` 返回 **0**，整个 backbone 退化为随机初始化）。
> **因此本阶段停止，不创建 `configs/yolo11m_rgbid_lightfusion.yaml` / `configs/train_rgbid_lightfusion.yaml`。**

---

## 1. Baseline architecture

**模型配置**：`configs/yolo11m_earlyfusion.yaml`（`ch: 5`，`nc: 12`）
**与 `ultralytics/cfg/models/11/yolo11.yaml` 的差异**：**仅 `ch: 3 → 5`**（外加 `nc` 与注释/文件名）。
backbone 与 head 的**每一行完全相同**。

### 1.1 数据流（实测追踪）

```
data 载入  base.py::_merge_channels_rgbid → cv2.merge((b,g,r,ir,d))   → HWC [B,G,R,IR,D]
变换      augment.py::Format._format_img (5ch 分支)                    → CHW [R,G,B,IR,D]
归一化    detect/train.py:159  batch["img"].float() / 255              → [B,5,H,W] ∈ [0,1]
─────────────────────────────────────────────────────────────────────────────────────
模型入口  DetectionModel.parse_model
          ch = [self.yaml["ch"]] = [5]        (tasks.py:347 / 991)
          首层 f = -1  →  c1 = ch[-1] = 5     (tasks.py:1067)
          c2 = make_divisible(min(64, 512) * width=1.0, 8) = 64
─────────────────────────────────────────────────────────────────────────────────────
第 0 层   Conv(5, 64, k=3, s=2, p=1) + BN + SiLU     ← 5ch 在此进入网络
          输出 64 × 640 × 640                       ← 此后即标准 YOLO11 backbone 通道
─────────────────────────────────────────────────────────────────────────────────────
```

### 1.2 逐条回答 §2 的 8 个问题

| # | 问题 | 实测答案 |
|---|---|---|
| 1 | 当前第一层是什么模块？ | **`Conv`**（`ultralytics/nn/modules/conv.py:41` = `Conv2d + BatchNorm2d + SiLU`） |
| 2 | 第一层 `in_channels`？ | **5** |
| 3 | 5ch 在哪里进入网络？ | **就在第 0 层**（`model.0`），由 `yaml["ch"]=5` 经 `parse_model` 赋给 `c1` |
| 4 | 第一层之后何时变成正常 backbone 通道？ | **第一层输出即刻**：`Conv(5→64)` 之后全部层是标准 YOLO11m 通道（64/128/256/512/1024） |
| 5 | 当前是否已有 5→C 的 projection？ | **已有。`5 → 64`**（k=3、s=2 的卷积核） |
| 6 | 是否已属可学习的模态融合？ | **是。** 每个输出通道 = 全部 5 个输入模态在 3×3 邻域内的**可学习线性组合**，随后 BN+SiLU。**这就是跨模态 channel mixing。** |
| 7 | 第一层参数量？ | **3,008**（conv 64×5×3×3 = 2,880 + BN 64×2 = 128），占总参数 **0.0150%** |
| 8 | 第一层 FLOPs？ | **2.359 G @1280**，占全模型 **0.862%**（全模型实测 **273.85 G**） |

### 1.3 已有的初始化（关键）

`ultralytics/models/yolo/detect/train.py:21 _transfer_rgb_pretrained()` **已专门为 RGBID 早期融合实现**：

```python
# RGBID early fusion: first Conv absorbs all 5 channels (no 1ch stem)
five_ch = next((m for m in stems if m.conv.in_channels == 5), None)
w = src.get("model.0.conv.weight")                      # 预训练 (64,3,3,3)
five_ch.conv.weight[:, :3].copy_(w)                     # RGB ← 预训练
five_ch.conv.weight[:, 3:].copy_(w.mean(dim=1, keepdim=True).repeat(1, n_aux, 1, 1))  # IR/D ← mean(R,G,B)
```

**实测验证（本次模拟）**：

```
Transferred 642/649 items from pretrained weights
RGBID early-fusion stem init: RGB=pretrained, IR/D=mean(R,G,B) (stem in=5)
  stem[:, :3] == 预训练 RGB 权重 : True
  stem[:, 3:] == mean(R,G,B)     : True
```

即：**除首层外 642/649 个张量按名字正常加载，首层由专用 remap 赋上 RGB 预训练 + IR/D 均值**。
这正是用户在 §6 所描述的初始化方案 —— **它已经在 baseline 中生效了。**

---

## 2. Proposed light fusion architecture

用户给出的候选：

| 候选 | 形式 | 评估 |
|---|---|---|
| **A（首选）** | `RGBID 5ch → Conv(5,3,1×1) → 原 YOLO11m` | **已实测否证（见 §3/§7）** |
| **B** | `5ch → 1×1 Conv → 原 backbone 第一层所需 C` | 与 A 同构，同样需要**插入一层** → 同样的 key 位移问题（§7）；若改为**原位替换** `model.0`，则 1×1 核会与现有 remap 的 3×3 复制逻辑形状冲突 |
| **C（gating）** | RGB/IR/Depth 上的极小 channel gate | 5 元素（或 5→5）门控**被首层卷积完全吸收**：把 `W[:, c, :, :]` 置零即等价于 gate(c)=0，BN 已提供逐输出通道缩放。**不构成新的表达能力**，且任何新模块仍会改变 key 布局 |

---

## 3. Exact architecture diff

### Candidate A 的实际结构（临时审计 yaml，非可用配置）

```diff
 backbone:
+  - [-1, 1, Conv, [3, 1, 1]]   # 0 新增：5 -> 3，1x1，stride 1
   - [-1, 1, Conv, [64, 3, 2]]  # 原 0 -> 现 1
   ...
 head:
-  - [[-1, 6], 1, Concat, [1]]      # 6 需 -> 7
-  - [[-1, 4], 1, Concat, [1]]      # 4 需 -> 5
-  - [[-1, 13], 1, Concat, [1]]     # 13 需 -> 14
-  - [[-1, 10], 1, Concat, [1]]     # 10 需 -> 11
-  - [[16, 19, 22], 1, Detect, [nc]] # -> [17, 20, 23]
```

**⚠️ 第一处实测偏差：`Conv(5,3,1,1)` 实际输出 8 通道，不是 3。**

```
第 0 层: Conv(5, 8, k=(1,1), s=(1,1))   权重 shape = (8, 5, 1, 1)
```

原因：`parse_model` 对 `base_modules` 无条件执行
`c2 = make_divisible(min(c2, max_channels) * width, 8)`，
`make_divisible(3, 8) = 8`（`utils/ops.py:130`）。
→ **「5→3」在 YAML 层面无法表达**，除非新增自定义模块（= 改源码）。

**⚠️ 第二处：插入一层使全网络绝对索引 +1**，`state_dict` key 整体位移
（`model.0.*` → `model.1.*`，依此类推）。这是 §7 失败的根因。

---

## 4. Parameter count

| 模型 | 总参数 | 首层 | 首层占比 | Δ vs baseline |
|---|---:|---:|---:|---:|
| **Baseline** `yolo11m_earlyfusion` | **20,063,412** | 3,008 | 0.0150% | — |
| **Candidate A** | **20,065,196** | 56 | 0.0003% | **+1,784 (+0.0089%)** |
| 1% 门限 | 200,634 | | | **PASS**（远低于门限） |

> 参数量门禁（§3 原则 3）**通过**，但这不是本设计被否证的原因。

---

## 5. FLOPs / complexity

| | FLOPs @1280×1280 | 首层 FLOPs | 首层占比 |
|---|---:|---:|---:|
| Baseline | **273.85 G** | 2.359 G | 0.862% |
| Candidate A | **275.51 G** | 0.033 G | 0.012% |

- 首层是**普通卷积**，输出空间尺寸 **640×640**，**O(HW) 复杂度**，**无任何 `HW × HW` 项**。
- 全部候选（A/B/C）均为逐像素或逐通道操作，**不涉及 CrossAttention / SelfAttention /
  CrossMLCA / CMA / Transformer block**，不存在 `[B, heads, HW, HW]` 张量。
  → **满足 §3 原则 2（禁止空间注意力）。**

---

## 6. Memory implication

参数量几乎不变，但 **Candidate A 会新增一张全分辨率特征图**，激活显存并非免费：

| 张量 | 每图元素数 | B=8, AMP(fp16) |
|---|---:|---:|
| 输入 5ch @1280×1280（两者都需要） | 8.19 M | 131 MB |
| **Candidate A 新增的 8ch @1280×1280（需保留用于反传）** | **13.11 M** | **≈ 210 MB** |

→ 在 A100-40GB 上约 **+0.5% 显存**。绝对量不大，但**与「参数量只增 0.0089%」给人的印象不成比例** ——
全分辨率层的内存代价与该层参数量无关，而与 `C × H × W × B` 成正比。

**Baseline 本身不在全分辨率上保留额外张量**（首层立刻 stride=2 降采样到 640×640）。

---

## 7. Pretrained weight loading analysis

**这是本设计的决定性环节。** 模拟完全复刻训练路径
`train.py:194-196`：`model.load(weights)` → `_transfer_rgb_pretrained(model, weights)`。

### 7.1 Baseline（对照组）

```
标准 intersect_dicts : 匹配 642/649   缺失 7   unexpected 0
源 model.0.conv.weight 被匹配: False（源 shape (64,3,3,3) vs 目标 (64,5,3,3)）
_transfer_rgb_pretrained 返回 = 5
  stem[:, :3] == 预训练 RGB 权重 : True
  stem[:, 3:] == mean(R,G,B)     : True
```

✅ **仅首层 1 个张量由专用 remap 补齐，其余 642 个按名字正常加载。backbone 预训练完整保留。**

### 7.2 Candidate A

```
标准 intersect_dicts : 匹配 14/655   缺失 641   unexpected 613
源 model.0.conv.weight 被匹配: False（源 shape (64,3,3,3) vs 目标 (8,5,1,1)）
_transfer_rgb_pretrained 返回 = 0            <-- 未迁移任何权重
  原因：five_ch.conv.weight.shape[0] = 8 ≠ 源 w.shape[0] = 64，条件不成立直接跳过；
        且随后 `if not any(in_channels == 1): return 0`
第一层权重是否仍为随机初始化: True (std=0.27034)
```

### 7.3 汇总（§5 / §7 要求的判定）

| 项 | Baseline | **Candidate A** |
|---|---|---|
| Original pretrained backbone loaded | ✅ **YES**（642/649） | ❌ **NO**（14/655） |
| Fusion layer newly initialized | ✅ YES（仅首层，且有预设初始化） | ⚠️ YES（但**整个 backbone 一起被重置**） |
| **Unexpected backbone reset** | ✅ **NO** | ❌ **YES**（641 个张量丢失 + 613 个 key 位移未匹配） |
| **Weight-loading simulation** | **PASS** | ❌ **FAIL** |

**根因**：在 backbone 前插入任何一层，都会使全部后续层的**绝对索引位移 +1**，
而 `state_dict` 以 `model.{index}.` 命名，`intersect_dicts` 按「key 名 + shape」双重匹配 →
**位移后无一命中**。唯一修法是**新增一段 index 重映射代码**（改源码），
而这恰恰是 §5 明确要求避免的情况。

> 按 §7 的规定：**「如果发现整个 backbone 因为结构修改而无法加载预训练：停止，不创建最终配置，先报告问题。」**
> → **本阶段在此停止。**

---

## 8. Initialization strategy

### 8.1 Baseline 现用的初始化（已生效，无需新增）

| 输入通道 | 初始化 | 依据 |
|---|---|---|
| `R, G, B`（索引 0,1,2） | **直接复制预训练 `model.0.conv.weight`** | 保住 COCO RGB 特征 |
| `IR, Depth`（索引 3,4） | **`mean(W_R, W_G, W_B)`**（对输入维求均值后 repeat） | 幅值与原 RGB 通道同尺度，避免新通道引入额外激活能量 |

### 8.2 若采用 Candidate A 会怎样

- `Conv(5,8,1,1)` 只能**随机初始化**（见 §7.2）。
- 「RGB identity + IR/D 小权重」的设想本身合理，但**它无法带来净收益**：
  随机/手工初始化的 1×1 层会把**预训练 RGB 输入打乱**，此后网络必须重新学习
  「如何恢复一个可用的 RGB 表示」，而 baseline 在这一点上**开局即无损**。
- **尺度问题（§6 特别提示）**：IR 经逐图 1%/99% 分位拉伸、Depth 为 8bit 映射且
  31–37% 像素为 0（见 `reports/rgbid_five_channel_pipeline_audit.md` §11-③），
  与 RGB 分布确不相同。但**现有 `mean(R,G,B)` 方案已经是在不引入额外尺度的前提下最保守的选择**，
  且已被 300 epoch 训练验证可用。**没有证据支持改用更复杂的手工权重。**

> 结论：**初始化层面没有可改进空间**，现有方案即为「简单、稳定、可解释」的答案。

---

## 9. Training configuration diff

**未创建。** 按 §7/§11 停止条件，本阶段不产出
`configs/yolo11m_rgbid_lightfusion.yaml` 与 `configs/train_rgbid_lightfusion.yaml`，
`runs/urban_multimodal_det_yolo11_rgbid_lightfusion/` 亦未创建。

（若将来获批，**计划中的** diff 仅有两处：
`model: configs/yolo11m_earlyfusion.yaml → 新模型 yaml`，`experiment_name → ..._rgbid_lightfusion`；
其余 `data / epochs=300 / imgsz=1280 / batch=8 / optimizer=SGD / lr0=0.005 / lrf=0.01 /
momentum / weight_decay / warmup_epochs=3 / cos_lr / close_mosaic=10 / mosaic / mixup /
box=7.5 / cls=0.5 / dfl=1.5 / seed=42 / deterministic=true` 全部照抄 baseline，一字不改。）

---

## 10. Smoke-test result

对**临时审计模型**（非新配置）执行了等价检查：

| # | 检查 | Baseline | Candidate A |
|---|---|---|---|
| 1 | YAML parse | ✅ PASS | ✅ PASS（临时 yaml） |
| 2 | Model construction | ✅ PASS（24 顶层模块） | ✅ PASS（25 顶层模块） |
| 3 | Dummy forward `[1,5,1280,1280]` | ✅ PASS，输出 `(1,16,33600)`，**NaN/Inf = 0** | ✅ PASS，输出 `(1,16,33600)`，**NaN/Inf = 0** |
| 4 | Weight-loading simulation | ✅ **PASS** | ❌ **FAIL**（14/655，backbone 被重置） |
| 5 | Parameter comparison | 20,063,412 | 20,065,196（**+0.0089%**） |
| 6 | Git diff | 见 §11 | 见 §11 |

> 注意第 3 项：**forward 能跑通并不代表设计成立** —— Candidate A 在结构上完全合法，
> 失效点在**权重加载**，这正是本次模拟要专门验证的。

---

## 11. Files changed

**既有文件：零改动。** 全程未修改任何 `.yaml`、`.py`、`results.csv`、checkpoint、提交包。

**新增文件（全部为审计产物）**：

| 文件 | 性质 | 说明 |
|---|---|---|
| `reports/rgbid_lightfusion_design.md` | 报告 | 本文件 |
| `diagnostic/lightfusion_design/_audit_and_sim.py` | 临时只读脚本 | 审计 + 加载模拟驱动，**不训练** |
| `diagnostic/lightfusion_design/yolo11m_candidateA_5to3.yaml` | 临时审计 yaml | **已加⛔头注**，明确标注「非可用配置，请勿训练」 |
| `diagnostic/lightfusion_design/`（目录） | 审计工作区 | 与 `configs/`、`runs/` 隔离 |

**`configs/` 与 `runs/` 下未新增任何文件。**

---

## 12. Safety / rollback

| 项 | 状态 |
|---|---|
| Baseline 模型 yaml `configs/yolo11m_earlyfusion.yaml` | ✅ **未修改** |
| Baseline 训练 yaml `configs/train_rgbird_ir_quicktest.yaml` | ✅ **未修改** |
| Baseline run 目录 `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/` | ✅ **未修改、未覆盖、未删除** |
| Baseline checkpoint（`best.pt` 等 12 个） | ✅ **未修改**（SHA256 已复核，见 `rgbid_checkpoint_selection_audit.md` §8） |
| Baseline `results.csv` | ✅ **未修改** |
| `ultralytics/cfg/default.yaml` | ✅ **未修改** |
| 数据处理 / loss / augmentation / optimizer | ✅ **未修改** |
| 正式提交 `submissions/rgbird_ir_quicktest/submission.zip` | ✅ **未修改**（`0c550773…1f756ce`） |
| 训练 / validation / resume / 权重下载 | ❌ **未执行** |
| 线上提交 | ❌ **未提交** |
| 正式线上保底 | ✅ **RGBID Early Fusion，线上 0.53180 不变** |

**回滚方式**：本阶段未产生需要回滚的改动。若要清理，直接删除
`diagnostic/lightfusion_design/` 与 `reports/rgbid_lightfusion_design.md` 即可，
**不会影响任何既有实验产物。**

---

## 13. 设计评审输出（§11 要求）

```text
LIGHT_FUSION_DESIGN_REVIEW

Current input:
  [R,G,B,IR,Depth]   (5ch, uint8/255 -> [0,1])

Current first layer:
  Conv(5 -> 64, k=3, s=2, p=1) + BatchNorm2d + SiLU      (model.0)
  —— 已经是「可学习的 5 通道跨模态线性混合 + 非线性」

Current first-layer parameters:
  3,008   (0.0150% of total)

Current first-layer FLOPs:
  2.359 G @1280    (0.862% of 273.85 G total)

Candidate:
  A —— 在 backbone 前插入 1x1 Conv(5 -> 3)，其后接原 3ch YOLO11m

Candidate A 与规格的两处偏差（实测）:
  (1) Conv(5,3,1,1) 实际输出 8 通道, 因 make_divisible(3, 8) == 8 —— “5->3” 在 YAML 层不可表达
  (2) 插入一层使全网络绝对索引 +1 -> state_dict key 整体位移

Additional parameters:
  +1,784  (20,063,412 -> 20,065,196)

Percentage of total YOLO11m parameters:
  +0.0089 %          (门限 1% -> PASS)

Spatial complexity:
  O(HW)              (无 HW x HW 项，无 attention)

Pretrained backbone preserved:
  NO                 <-- 标准 load() 仅匹配 14/655；_transfer_rgb_pretrained 返回 0

Expected newly initialized parameters:
  641/655 目标张量未被赋权（含全部 backbone），且 613 个源 key 因索引位移无对应目标

Weight-loading simulation:
  FAIL
```

**三项停止条件中命中两项**（`pretrained backbone preserved = NO`、`weight-loading simulation = FAIL`）
→ **按 §11 停止，不创建最终训练配置。**

---

## 14. 最终输出

```text
LIGHT_FUSION_CONFIG_READY

Training launched: NO
Baseline modified: NO
Baseline checkpoint modified: NO
Dataset modified: NO
Official submission modified: NO

Model config:
NOT CREATED  (设计评审未通过，按 §7/§11 停止)

Train config:
NOT CREATED  (同上)

Run directory:
NOT CREATED  (runs/urban_multimodal_det_yolo11_rgbid_lightfusion/ 不存在)

Design review:
FAIL
  - 首层 Conv(5->64,3x3) 已经就是可学习的轻量通道融合（§2「已经是」情形）
  - Candidate A 的 “5->3” 在 YAML 层不可表达（make_divisible -> 8）
  - Candidate A 破坏预训练加载（14/655，整网重置）

Weight-loading simulation:
Baseline     PASS  (642/649 + stem remap 5 个张量)
Candidate A  FAIL  (14/655, _transfer_rgb_pretrained -> 0)

Dummy forward:
PASS  (两者均 forward 正常、输出 (1,16,33600)、NaN/Inf = 0；但 forward 通过不代表设计成立)

Ready for training:
NO
```

---

## 15. 建议（不作为本次执行内容）

1. **不要在「加一层 5→3 / 5→C projection」这个方向上投入训练资源。**
   该杠杆在 baseline 设计里**已经被用掉了**（首层即 5→64 融合），
   再加一层是重复实现，且会破坏预训练加载。
2. 若仍希望从「融合」方向取得单变量增益，**唯一不破坏预训练加载的位置是在首层之后**，
   但这必然改 backbone / neck / head，**违反 §3 原则 1（只增加一个融合因素）**，故本次不提案。
3. 与本会话既有结论一致：真正有正期望的方向仍是
   **降低有效容量 / 增强归纳偏置**（过拟合诊断 `corr(train/cls,val/cls) = −0.785`），
   而**任何增加容量的改动都与证据方向相反**。
4. 若用户仍希望推进某个具体变体，**请先明确告知**，我再单独做设计评审；
   **在获得授权前，本阶段到此为止，不启动训练。**
