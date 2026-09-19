# RGBID 下一阶段优化 —— 只读审查与规划报告

**日期**：2026-09-18
**性质**：**只读审查 + 规划**。未训练、未改模型、未改训练配置、未创建大规模实验、未修改正式目录、未提交线上。
**本轮唯一产出**：本报告（+ `diagnostic/next_optimization_review/` 下的只读模拟脚本与临时探针 yaml，均不影响正式代码路径）。

---

## 0. 冻结声明（§一）

| 对象 | 本轮状态 |
|---|---|
| `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/` | ✅ 未触碰 |
| `…/weights/best.pt` | ✅ SHA256 `f9dddbfaca4cb08bde31d082eb0a54d499b8441eb1f97c6134a028caa6d83e55` 未变 |
| `…/weights/last.pt` | ✅ `cfa00836314e5e5eb87b35ff7dd6fc42d2e0291f81c5ffc7384957b51e88004f` 未变 |
| `submissions/rgbird_ir_quicktest/submission.zip` | ✅ `0c550773c8656108…` 未变 |
| 正式成绩 | ✅ online **0.53180** / official **0.50928** 不变 |
| 训练 / 提交 | ❌ 未执行 |

---

## 1. 代码结构审查（§1）

### 1.1 实际模型结构 —— 24 个顶层模块（实读实例化结果）

配置文件：`configs/yolo11m_earlyfusion.yaml`（`ch: 5`, `nc: 12`, scale m = `[0.50, 1.00, 512]`）

| idx | 模块 | 实际通道（实读） | 作用 |
|---:|---|---|---|
| **0** | **`Conv`** | **c1=5 → c2=64**, k=3, s=2 | **输入投影（5ch 入口）** |
| 1 | `Conv` | →128, s=2 | P2/4 |
| 2 | `C3k2` | 128→256 | |
| 3 | `Conv` | →256, s=2 | P3/8 |
| 4 | `C3k2` | 256→**256** | **backbone P3 特征** |
| 5 | `Conv` | →512, s=2 | P4/16 |
| 6 | `C3k2` | 512→**512** | **backbone P4 特征** |
| 7 | `Conv` | →512(max512), s=2 | P5/32 |
| 8 | `C3k2` | 512→512 | |
| 9 | `SPPF` | 512→512 | 多尺度池化 |
| 10 | **`C2PSA`** | 512→512 | **HW×HW 自注意力（P5 层）** |
| 11 | `Upsample` | | |
| 12 | `Concat` | `[-1, 6]` | FPN 上采样与 backbone P4 拼接 |
| 13 | `C3k2` | 1024→512 | |
| 14 | `Upsample` | | |
| 15 | `Concat` | `[-1, 4]` | **与 backbone P3 拼接** |
| **16** | **`C3k2`** | **1024→256** | **★ P3 检测特征（small 分支）** |
| 17 | `Conv` | 256→256, s=2 | 下采样 |
| 18 | `Concat` | `[-1, 13]` | PAN |
| **19** | **`C3k2`** | **768→512** | **★ P4 检测特征（medium）** |
| 20 | `Conv` | 512→512, s=2 | |
| 21 | `Concat` | `[-1, 10]` | PAN |
| **22** | **`C3k2`** | **1024→512** | **★ P5 检测特征（large）** |
| **23** | **`Detect`** | **ch=[256,512,512]** | 头部 |

### 1.2 P3/P4/P5 特征尺寸与 stride（实测）

```
Detect:  nc=12  nl=3  reg_max=16  no=76
         输入通道 ch = [256, 512, 512]
         stride    = [8.0, 16.0, 32.0]
```

stride 由 `ultralytics/nn/tasks.py:371` 用 256×256 dummy 前向实测得出：
`m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])`

在**当前正式 imgsz=1280** 下：

| 检测分支 | 层 | 特征图 | stride | 一个 32px 小目标占 |
|---|---:|---|---:|---|
| **P3（small）** | 16 | **160 × 160** | 8 | **4 × 4 格** |
| P4（medium） | 19 | 80 × 80 | 16 | 2 × 2 格 |
| P5（large） | 22 | 40 × 40 | 32 | 1 × 1 格 |

**小目标主要由 P3（层 16）预测** —— 这也解释了为什么瓶颈审计里 small AP50-95 = 0.05625 而 large = 0.61711。

### 1.3 P3 是否已包含三模态融合信息？

**是，而且是从第 0 层起就完全融合的。** 数据流：

```
base.py::_merge_channels_rgbid      → cv2.merge((b,g,r,ir,d))     HWC [B,G,R,IR,D]
augment.py::Format._format_img(5ch) → [:3][::-1] + [3:]            CHW [R,G,B,IR,D]
train.py:159                        → .float()/255
──────────────────────────────────────────────────────────────────────────
layer 0  Conv(5→64,3×3)   ← 5 个模态在这里被一次性线性混合
layer 1 ~ 22                     ← 之后**全部**是融合特征，无任何模态专用分支
layer 16 (P3)                    ← 因此 P3 必然包含三模态信息
```

**关键结论：网络中不存在独立的 RGB / IR / Depth 分支**（`base.py::_merge_channels_rgbid` 在载入时就把三模态合并成一张 5ch 图）。这一点直接决定了候选 B 的可行性（见 §3.2）。

### 1.4 当前是否已有可学习的输入投影？

**已有。** layer 0 = `Conv(5→64, 3×3, s2)`（`ultralytics/nn/modules/conv.py:41`），
即一个**可学习的 5→64 跨模态卷积**，且已带 `_transfer_rgb_pretrained` 的
「RGB=预训练 / IR,Depth=mean(R,G,B)」初始化（`ultralytics/models/yolo/detect/train.py:21`）。
**它是本网络唯一的输入级融合模块。**

### 1.5 是否存在可在 P3 处做轻量增强的安全插入点？

**存在，且只有一个位置是安全的：层 16 原位替换（不新增层）。**

| 方案 | 层索引 | 预训练迁移 | 可行性 |
|---|---|---|---|
| 在层 16 之后**插入**新层 | **全网络 +1 位移** | **实测崩塌**（项目已证：`reports/rgbid_lightfusion_design.md`，仅 14/655 匹配） | ❌ |
| 在 **层 16 内部**加子模块（`self.attn`） | **不变** | **实测 649/651，已有层 100% 迁移** | ✅ |
| 替换层 16 为新模块类（同索引） | 不变 | 需注册 parse_model 集合，否则构建失败（见 §5.10） | ⚠️ 可行但需源码改动 |

---

## 2. 当前模型已有的类似机制（§2）

### 2.1 机制盘点（实读）

| 机制 | 位置 | 说明 |
|---|---|---|
| **输入投影** | layer 0 `Conv(5→64,3×3)` | **可学习的 5 通道融合**，已有 |
| **Concat** | `conv.py:327`；用于 layer 12 / 15 / 18 / 21 | FPN/PAN 特征拼接 |
| **C2f / C3k2** | `block.py:237` / `block.py:737` | CSP 式通道分裂+交叉拼接+残差；本模型 **8 个 C3k2**，且 scale=m 时 `tasks.py:1081` 强制 `args[3]=True` → **全部使用 C3k** |
| **SPPF** | `block.py:182`；layer 9 | 多尺度池化 |
| **C2PSA / PSABlock / Attention** | `block.py:1073` / `993` / `937`；layer 10 | **HW×HW 自注意力**（`attn = qᵀk`，在 P5 的 40×40=1600 token 上） |
| **Channel Attention** | `conv.py:282`、`attention.py:689` | **代码存在，但本模型未使用** |
| **Spatial Attention** | `conv.py:297`、`attention.py:712` | 同上，未使用 |
| **CBAM** | `conv.py:313` | 同上，未使用 |
| **独立模态分支** | — | **不存在** |
| **多模态专用融合模块** | — | **除 layer 0 外不存在** |

### 2.2 是否「只是重复已有功能」？—— 分两层回答

**第一层（字面重复）：不重复。** 全网络**没有任何通道注意力 / SE / CBAM / CA**。
`configs/yolo11_midfusion_rgbd_cbam.yaml` 这个配置存在，但：

```
grep -l "cbam" runs/*/args.yaml   → 无任何命中
```

→ **CBAM 在本项目中从未被训练过**。因此「加通道注意力」在项目层面是**未试过的家族**，不是重复。

**第二层（功能重复）：高度可疑。** 三点必须正视：

1. **P3 已经是融合特征**（§1.3），对 P3 做通道重标定，是在**已被融合的特征**上再校准，
   而不是在补足模态信息。
2. **层 16 的 C3k2 本身就在做通道混合**：`cv1` 卷积 → split → 多个 `C3k`/`Bottleneck` → `cv2` 拼接。
   它是一个**通道维度的重加权机制**，与 SE 式通道注意力的功能区间高度重叠。
3. **最大的一次「小目标定向干预」已经失败**：E5 = RGBD mid-fusion **+ P2 检测层** @640
   → best mAP50-95 **0.47796**，而同期无 P2 的 E4 = **0.47783**，**Δ = +0.00013（实质为零）**。
   即：**给小目标专门加一个更高分辨率检测层，在本项目里没有产生收益。**

> **因此不能默认「加注意力一定有效」。** 相反，现有证据更偏向「P3 侧的特征/尺度干预已经试过，且为中性」。
> **本候选的收益预期必须标为「无项目证据，收益未知」。**

---

## 3. 三个候选方案分析（§3）

### 3.1 候选 A：P3 轻量增强

**概念**：`P3 feature → 轻量 1×1 Conv / Channel Attention → Detect`

| 检查项 | 实测/分析结果 |
|---|---|
| **最小可行改动** | 在**层 16 内部**添加一个 `self.attn`（通道注意力），`forward` 改为 `self.attn(super().forward(x))`。**不新增层** |
| **是否需修改 YAML** | 是（1 个新 model yaml 副本 + 1 个新 train yaml 副本）。**必须新建，不得改 `configs/yolo11m_earlyfusion.yaml`** |
| **是否导致层索引变化** | **不变**（实测：顶层模块数仍为 24；Detect 输入通道仍 `[256,512,512]`） |
| **是否破坏预训练迁移** | **不破坏 —— 已实测**（见 §5.9） |
| **参数量增加** | **+65,792 = +0.3268%**（`nn.Conv2d(256,256,1,bias=True)` = 256²+256） |
| **FLOPs 增加** | ≈ **+13.1 MFLOPs**（160×160 逐元素乘 6.55M MAC + 池化/1×1 卷积）÷ 273.85 G → **+0.005%**，可忽略 |
| **显存风险** | 极低。仅多一个 `[B,256,160,160]` 的中间张量（B=8, AMP ≈ **105 MB**） |
| **是否能只影响 P3** | **是**（只改层 16） |
| **是否需全量重训** | **建议是**（见 §7.3；与项目既有方法论一致：所有结构改动都是 1280 从零 300 epoch） |
| **是否适合从 best.pt 初始化** | **适合且已实测**：649/651 迁移，仅新增 attn 权重；若 attention 做 **identity 初始化**则训练起点**精确等于 baseline** |

### 3.2 候选 B：多模态 P3 融合（RGB/IR/Depth 分支级融合）

| 检查项 | 结论 |
|---|---|
| **当前是否保留独立 RGB/IR/Depth 分支** | **否。完全没有。**（§1.3：载入时即合并为 5ch 单张图） |
| **是否必须重构主干** | **是。** 要得到分支级特征，必须把 `load_and_preprocess_image` / `Format` / 全部 300+ 行数据管线与模型输入层重写为多输入分支结构 |
| **是否导致大量随机初始化** | **是。** 新主干（3 个 stem、3 条分支）几乎全部随机初始化，丢失当前 649/649 的预训练迁移 |
| **是否破坏已验证的 RGBID Early Fusion 优势** | **是。** RGBID 相对 RGBD mid-fusion 的 +0.02527（`reports/rgbid_inference_gain_audit.md`）正是来自早期融合；重构为分支融合等于回到已被 M6/F4 证明更弱的中期融合家族 |
| **是否值得在比赛截止前尝试** | **否。** 工作量 = 重做整个数据管线 + 主干 + 300 epoch 重训 × 多轮调试；**截止 2026-09-20 20:00 前不可能完成并验证** |

**结论：候选 B 应直接排除。**

### 3.3 候选 C：小目标定向同步 crop / oversampling

| 检查项 | 结论 |
|---|---|
| **配对规则** | `base.py` 中纯字符串替换：`file_path.replace(pairs_rgb, pairs_ir)` / `(pairs_rgb, pairs_depth)`（`base.py:315/316` 等）。命名同 stem，扩展名可不同 |
| **三模态尺寸是否一致** | **实测 180 组（60 train + 120 val）尺寸 100% 一致，0 组不一致** |
| **标签格式** | `labels/<split>/visible/<stem>.txt`，YOLO 归一化 `cls cx cy w h`（实读样本确认） |
| **Mosaic / RandomPerspective 调用顺序** | `v8_transforms`（`augment.py:3803`）→ `Compose([pre_transform(Mosaic), MixUp, alb, random_hsv, RandomFlip×2, RandomPerspective, LetterBox])` |
| **crop 应在 Mosaic 前还是后** | **前**（在 `get_image_and_label` 内、transforms 之前） |
| **crop 后是否需重算 labels** | 是，但**归一化坐标可直接线性变换**，无需反归一化 |
| **是否可能截断框** | 是。需定义策略（中心点在内则保留 + clip，或 IoU≥阈值保留） |
| **是否会让小目标比例过高** | 取决于概率参数；需设上限 |
| **是否破坏大目标/场景上下文** | 是（crop 会丢上下文）→ 必须用概率启用，不能全量 |
| **是否能只对含小目标图片启用** | 可以（`self.labels[i]` 里有 bbox，可预筛） |
| **是否能设最大 crop 次数** | 可以 |
| **是否能独立配置/独立 run** | 可以 |

**★ 一个决定性的有利事实（本项目结构红利）**：

`get_image_and_label`（`base.py`）在 `load_image(index)` 之后、`update_labels_info` 之前，
`label["img"]` **已经是合并后的 5 通道 HWC 图**，而 `label["bboxes"]` **仍是原始归一化坐标**。

```python
label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
#  ← 就在这里：img 是 [H,W,5]（三模态已合并），bboxes 仍为原始 normalized cxcywh
...
return self.update_labels_info(label)
```

→ **在此处裁剪，三模态同步是「构造性保证」的，不是「靠写对代码」保证的** ——
因为三个模态已经被合并进同一张数组，任何几何操作天然一致。
这直接回答了 §6 最担心的「模态对齐风险」。

**★ 一个必须正视的不利事实**：

1. `RandomPerspective` **已经带随机缩放**：`augment.py:1056`
   `s = random.uniform(1 - self.scale, 1 + self.scale)`，baseline `scale=0.5` → **s ∈ [0.5, 1.5]**。
   即「放大」机制**已经存在**，crop 只是把它做得更强、更有针对性。
2. Mosaic 使目标系统性变小：canvas = 2s×2s（`augment.py:3172`）随后被 resize 回 imgsz
   → **mosaic 开启期的有效目标尺度约为 0.5 × [0.5,1.5] = [0.25, 0.75]**。
   而 baseline 的 best epoch（237/300）**落在 mosaic 开启区**，即模型是在「目标被系统缩小」的条件下学出来的。
3. **同族干预已经 2/2 失败**：提高每目标像素数这一家族里，
   `1536 训练` 失败（官方 0.49401 / 0.50673，均 < 0.50928），
   `P2 检测层` 失败（E5 = +0.00013）。**crop-zoom 属于同一家族。**

---

## 4. 风险检查表（§四）

| 项目 | **候选 A：P3 轻量增强** | **候选 B：多模态 P3 融合** | **候选 C：小目标同步 crop** |
|---|---|---|---|
| **目标瓶颈是否匹配** | **不确定** —— 瓶颈是小目标定位，A 作用于 P3（小目标所在层），但机制是通道重标定而非尺度/定位 | **不确定** | **是** —— 直接改变有效目标尺度 |
| **是否修改模型结构** | 是（层 16 内加子模块） | 是（重构主干） | 否 |
| **是否破坏预训练权重** | **否 —— 已实测 649/651，已有层 100% 迁移** | **是 —— 主干几乎全部随机初始化** | 否 |
| **是否需要全量重训** | 是（建议 300 ep；见 §7.3） | 是 | 是 |
| **是否增加显存** | +≈105 MB @B=8 AMP（可忽略） | 大幅（多分支特征图） | 基本不变 |
| **是否增加 FLOPs** | **+0.005%** | 大幅 | 不变 |
| **是否存在数据对齐风险** | 无 | 高 | **无（构造性保证，见 §3.3）** |
| **是否有当前项目实验证据** | **无**（通道注意力从未训练过；最近似干预 E5 为 +0.00013） | **有（反向）**：早期融合已被验证优于中期融合 +0.02527 | **有（反向）**：同族的 1536 与 P2 均失败 |
| **预期收益依据** | **没有当前项目实验证据，收益未知。** | **没有当前项目实验证据，收益未知，且已知更差** | **没有当前项目实验证据，收益未知，且同族两次为负** |
| **主要失败模式** | ① 全局池化让通道权重由背景主导 → **抑制小目标信号**；② 与 C3k2 已有通道混合功能重叠 → 无净增益 | 破坏已验证的 early-fusion 优势；预训练优势丧失；工期不可行 | 与 RandomPerspective 已有 zoom 重叠；同族已 2/2 失败；大目标与上下文被破坏 |
| **实现复杂度** | **低** | **高** | **中** |
| **是否建议当前阶段执行** | **见 §7** | **否** | **见 §7** |

---

## 5. P3 轻量增强合理性专项审查（§五，10 问）

**1. 当前 P3 是在哪里生成的？**
layer 16（`C3k2`，1024→256），输入来自 layer 15 = `Concat([layer14 上采样输出, layer4 backbone-P3])`。

**2. P3 进入 Detect 前经过哪些模块？**
只有 layer 16 本身（`C3k2`）。layer 16 的输出**直接**喂给 layer 23 的 `Detect`（`configs/yolo11m_earlyfusion.yaml` 末行 `[[16,19,22], 1, Detect, [nc]]`）。

**3. P3 是否已经过 C2f/C3k2/PSA 等特征处理？**
**是。** `C3k2`（`block.py:737`，继承 `C2f`）就是 CSP 式特征处理块；且 scale=m 时全部使用 `C3k` 瓶颈。
**但 P3 路径上没有 PSA**（`C2PSA` 只在 layer 10，即 P5 层）。

**4. 加 CA 是增强有效信息，还是可能抑制弱小目标信号？**
**存在明确的抑制风险（机制层面推理，非项目实测）**：
`ChannelAttention`（`conv.py:282`）用 `nn.AdaptiveAvgPool2d(1)` 做全局平均。
对一个只占 4×4 格的小目标，全局平均被背景主导 → 得到的通道权重反映的是**背景统计**，
再乘回特征时会按背景而非目标来缩放通道。对小目标占比高的 P3 层，这个方向是**不利**的。

**5. CA 是否会让小目标特征变得更稀疏？**
**有可能。** 若背景主导的权重把承载小目标的通道压低，等效于稀疏化。
这是与目标相反的机制，**必须在设计里显式对冲**（见 §7.3 的 identity 初始化 + 只作用于 P3）。

**6. 1×1 Conv 是否会改变已有特征分布？**
会 —— 若随机初始化，等价于对已收敛特征做一次随机线性变换。
**但这一点可以完全消除**：把新模块初始化为**恒等映射**（见 §5.7），则训练起点与 baseline **逐值相同**。

**7. 从正式 best.pt 初始化时，新增模块如何初始化？**
**建议恒等初始化（identity init）**，使 `attn(x) ≡ x` 在 t=0 成立：
- 用残差形式 `out = x * (1 + tanh(fc(pool(x))))` 且 `fc` 零初始化 → t=0 时 gate≡1，**输出精确等于输入**；
- 或门控形式 `out = x * sigmoid(fc(pool(x)))`，`fc.bias` 取大正值（如 +6 → sigmoid≈0.9975）→ 近似恒等。
**注意**：`conv.py:282` 的现成 `ChannelAttention` 在 `fc` 零初始化时 `sigmoid(0)=0.5` → `out = 0.5x`，
**不是恒等**，会立刻把 P3 特征减半 → **直接采用现成类而不改初始化是错的**。

**8. 是否会造成 backbone 重新随机初始化？**
**不会 —— 已实测。** 见 §5.9。

**9. 是否能保证原有 642/649 左右的预训练迁移规模不显著下降？**
**已实测，且比 642/649 更好：**

```
=== 从正式 best.pt 迁移（源 = 5ch / nc12，与目标同构）===
  匹配 649/651   缺失 2
  其中新增 attn 权重（本就应缺失）= 2  -> ['model.16.attn.fc.bias', 'model.16.attn.fc.weight']
  其他缺失 = 0  -> 无
  => 原有层 100% 迁移: True

  对照：baseline 结构从 best.pt 迁移 649/649
  => 加 attn 后迁移量与 baseline 相同: True
```

**已有 649 个张量全部迁移，唯一缺失的就是新增注意力的 2 个权重。**

**10. 是否可以设计成不改变已有层索引的方式？**
**可以 —— 已实测**：仅给 layer 16 挂 `self.attn`，`state_dict` key 从 **649 → 651**，
**新增 `16.attn.fc.{weight,bias}`，消失 key = 无**；顶层模块数仍为 **24**。

**⚠️ 但实测发现一个必须记录的实现障碍**：
若以「**新模块类** + 新 yaml」实现（而不是给现成 `C3k2` 挂子模块），
`parse_model` 会**拒绝**它 —— 因为 `ultralytics/nn/tasks.py` 用
`if m in base_modules:` / `elif m in repeat_modules:` 这类 **frozenset 成员判定**，
新类不在其中 → 落到兜底分支 → 收到原始 yaml 参数 `(256, False)` 当作 `(c1, c2)` → `c2=False` → 0 通道 → 构建失败（已实测报错）。

→ **若采用新模块类，必须同时把新类加入 `parse_model` 的 `base_modules` 与 `repeat_modules` 两个 frozenset**（这属于源码改动，须在实施前明确列出）。
→ **若不改 parse_model**，替代方案是给 `C3k2.__init__` 增加一个带默认值的参数，
   并在 yaml 里补足位置参数（注意 `tasks.py:1081` 会强制 `args[3]=True`，第 4/5 个 yaml 槽位分别落到 `g`/`shortcut`）——
   **该方案更易出错，不推荐。**

---

## 6. 小目标 crop 方案可行性审查（§六，12 问）

1. **配对规则**：`base.py:315/316` 等处的 `file_path.replace(pairs_rgb, pairs_ir/pairs_depth)`，纯字符串替换目录名；同 stem，扩展名可不同（depth 有 `.png` 与 `.jpg` 两代）。
2. **三模态尺寸**：实测 180 组**完全一致**（0 组不一致）。
3. **标签格式**：`labels/<split>/visible/<stem>.txt`，`cls cx cy w h`（归一化）。
4. **Mosaic / RandomPerspective 调用顺序**：`augment.py:3803` `Compose([pre_transform(Mosaic), MixUp, alb, random_hsv, RandomFlip×2, RandomPerspective, LetterBox])`（`pre_transform` 内含 `Mosaic` + `CopyPaste` + `RandomPerspective(pre_transform=LetterBox)`）。
5. **crop 应在 Mosaic 前**（在 `get_image_and_label` 内、transforms 之前）。
6. **需要重算 labels**：是，但归一化坐标可直接线性变换：
   `cx' = (cx·W − x0)/(x1−x0)`、`w' = w·W/(x1−x0)`（y 同理）。
7. **截断框**：会产生。策略建议 —— 保留中心点落在 crop 区内的框并 clip 到边界；或保留 IoU≥0.3 的框。**必须显式定义，不能默认**。
8. **小目标比例过高**：需设上限（如单图 crop 后小目标占比上限）。
9. **破坏大目标/上下文**：是 → **必须用概率启用**（如 p=0.3），不能全量。
10. **是否只对含小目标图片启用**：可以（`self.labels[i]` 已含 bbox，可预筛）。
11. **最大 crop 次数**：可设（如每 epoch 每图最多 1 次）。
12. **独立配置/独立 run**：可以。

### 最小可行设计（**本轮不实现**）

```
位置：ultralytics/data/base.py::get_image_and_label，在 load_image 之后、update_labels_info 之前
      （与现有 apply_modality_dropout 同一插入点风格）

新增函数：apply_small_object_crop(im, bboxes, cfg, augment)
  输入 im: [H,W,5]（三模态已合并 → 几何同步是构造性保证）
       bboxes: [n,5] 归一化 cls,cx,cy,w,h
  逻辑：if not augment or not cfg.enabled: return im, bboxes
        if 图中不存在面积 < 阈值的目标: return im, bboxes     # 预筛
        if random.random() > cfg.prob: return im, bboxes
        在含小目标的区域随机取一个 crop 框（边长比例 ∈ [cfg.min_ratio, cfg.max_ratio]）
        裁剪 im，线性变换 bboxes，丢弃中心落在区外的框并 clamp 其余
  返回：crop 后的 im 与变换后的 bboxes

配置（新增，默认 disabled → 既有实验零影响）：
  small_object_crop: {enabled: false, prob: 0.3, area_thresh: 1024,
                      min_ratio: 0.5, max_ratio: 0.75}
```

**必须遵守的约束**：`im` 与 `bboxes` **同时**进、**同时**出（否则模态或标签会失配）。

---

## 7. 最终建议（§七）

### 7.1 当前最值得尝试的一个方向：**候选 A：P3 轻量增强**

**选择依据（只说有证据的部分）**：

1. **它是三个候选里唯一「已验证安全」的**。
   实测：层 16 原位挂子模块 → **顶层索引不变（24）、已有 key 名 0 丢失、从 best.pt 迁移 649/651（与 baseline 的 649/649 等量）**。
   候选 B 会丢失几乎全部预训练迁移；候选 C 虽无迁移问题但实现复杂度更高、且要改数据管线。
2. **它是唯一「未被本项目试过的家族」**（通道注意力从未训练过），
   而候选 B 已被 early-fusion 的既有结论反向否定，候选 C 所属的「提高每目标像素数」家族已 **2/2 失败**（1536 训练、P2 层）。
3. **成本最低**：+0.3268% 参数、+0.005% FLOPs、+105 MB 显存、实现复杂度低，**且不动数据管线**。
4. **可以做成「起点精确等于 baseline」的实验**：靠 identity 初始化（§5.7），
   使这个 300 epoch 训练的唯一变量真的是「多了一个通道重标定」，
   而不是「多了一个随机模块」。这是本项目此前多次实验栽跟头的地方（Probe 1 就是被随机扰动毁掉的）。

**必须同时接受的两个反面事实（不得淡化）**：

- **没有当前项目实验证据，收益未知。** 最近似的一次小目标定向结构干预（E5 = +P2 检测层）实测 **+0.00013**。
- 通道注意力的**全局池化机制对小目标有抑制风险**（§5.4/5.5），这是与目标相反的机制。

**→ 因此这是一个「低风险、低期望、信息量有限」的实验，不是「有把握的优化」。**
若判定为不值得，`暂不继续实验` 同样是本报告证据支持的选项（见 §7.4）。

### 7.2 当前不建议做的方向

| 方向 | 不建议理由（有项目证据） |
|---|---|
| **继续 1536 调参** | 两个 LR 探针均已官方实测：0.49401 / **0.50673**，**均 < 0.50928**；且两个探针 best epoch 都在 ep1/ep2 —— 无向上趋势 |
| **复杂 CrossMLCA / CMA** | ① 显存风险（`C2PSA` 的 `Attention` 是 HW×HW，P5 的 1600 token 尚可，**P3 的 25600 token 是 6.5×10⁸ 元素，不可行**）；② 瓶颈审计已证误差是**前景/背景**而非类间（跨类误判仅 0.70%），注意力不针对该瓶颈 |
| **盲目增加 box / dfl** | 已实测为负：`box=10.0` → 0.52093 < 同族 0.53273；`dfl=2.0` → 0.51894 < 0.53273；且误差分解显示 scale 均值 0.9971±0.0022（**无偏置可修**） |
| **modality dropout** | 已实测为负：官方 0.49029 vs 0.50928（−0.01899） |
| **ensemble / voting** | **规则明文禁止**（赛题 §十一(一)：禁止多模型简单集成、投票法、平均法） |
| **未审查的 HHA / 新预处理** | 无任何项目证据；且数据管线已有两代混存（1080×1920 uint16 与 360×640 uint8），新预处理会引入未验证的对齐风险 |
| **同时修改多个变量** | 本项目已多次因多变量混淆付出代价（G7 的 box/cls/dfl 三变量、1536 Probe 1 的 LR 混淆） |

### 7.3 下一步执行门槛（**在你明确回复「批准实施」之前，一律不执行**）

**若批准候选 A，实施前必须先把以下 8 项写定（本报告已给出大部分）：**

| # | 项 | 状态 |
|---|---|---|
| 1 | **审查报告完成** | ✅ 本报告 |
| 2 | **明确修改文件** | ① 新增 `configs/yolo11m_earlyfusion_p3attn.yaml`（baseline 模型 yaml 的副本 + 层 16 改为新模块）；② 新增 `configs/train_rgbid_1280_p3attn.yaml`（baseline 训练 yaml 副本，只改 `experiment_name` + `model`）；③ 源码：`ultralytics/nn/modules/block.py` 新增 `C3k2P3Attn`（含 **identity 初始化**）；④ **`ultralytics/nn/tasks.py` 把新类加入 `base_modules` 与 `repeat_modules` 两个 frozenset**（§5.10 实测必需）。**`configs/yolo11m_earlyfusion.yaml` 与 `configs/train_rgbird_ir_quicktest.yaml` 一字不改** |
| 3 | **明确独立实验目录** | `runs/urban_multimodal_det_yolo11_rgbid_p3attn/`（新目录，绝不触及正式目录） |
| 4 | **明确回滚方式** | 删除 2 个新 config + `runs/…_p3attn/`；源码改动集中在一个新类 + 两行 frozenset 增补，`git diff` 可精确回滚 |
| 5 | **明确权重初始化方式** | **identity 初始化**（§5.7），使 t=0 时 `attn(x) ≡ x`；其余层从 `yolo11m.pt` 走既有 `_transfer_rgb_pretrained`（5ch stem）路径 |
| 6 | **明确是否保持 1280** | **保持 imgsz=1280**（1536 已否决；1280 是唯一被官方验证过的分辨率） |
| 7 | **明确官方评测命令** | `predict_rect.py --mode rect --imgsz 1280 --conf 0.001 --iou 0.7 --max_det 300 --max_boxes 300` → `official_map.py --split_root data/processed/rgbid_split`（与 baseline 逐字同参） |
| 8 | **明确停止阈值** | official mAP50-95 **≤ 0.50928 → 判 `P3ATTN_REJECTED`**；0.50928 < x < 0.51428 → `NO_MEANINGFUL_GAIN`；≥ 0.51428 → `CANDIDATE`。**并且必须观察 small AP50-95（baseline 0.05625）是否上升**，否则即使总分微升也判为「未解决瓶颈」 |

**成本预估**：300 epoch @1280 从零训练 ≈ **11 小时**（baseline 实测 130.9 s/epoch）。
截止 **2026-09-20 20:00** 前时间够用，但**只够一次**。

### 7.4 备选结论：`暂不继续实验`

**这一选项同样被本报告的证据支持**，理由：

1. 已有两次独立审计得出 `NO_SAFE_CANDIDATE`（`reports/rgbid_final_bottleneck_audit.md`）与
   `NO_SAFE_OPPORTUNITY`（`reports/rgbid_final_optimization_opportunity_audit.md`）；
2. 候选 A 所属的「head/P3 侧干预」中，**最大的一次**（E5 加 P2 层）实测 **+0.00013**；
3. 候选 C 所属的「提高每目标像素数」家族已 **2/2 失败**；
4. 候选 A 自身的机制（全局池化通道重标定）对**小目标**有可论证的抑制方向。

**若你更看重「把剩余时间用于提交物完整性而非再一次低期望训练」，请直接选择此项。**
我不会自动推进任何一种。

---

## 8. 本轮合规状态

| 项 | 状态 |
|---|---|
| 训练 | ❌ 未执行 |
| 修改模型 / 源码 / YAML | ❌ 无（仅新增只读模拟脚本与临时探针 yaml） |
| 修改正式目录 / best.pt / last.pt / 提交文件 | ❌ 无 |
| 线上提交 | ❌ 未提交 |
| 本轮新增 | `reports/next_optimization_review.md`、`diagnostic/next_optimization_review/`（只读模拟脚本 + 临时探针 yaml） |

**等待你回复「批准实施」后才进入实施阶段；在那之前不训练、不改代码、不改 YAML、不提交。**
