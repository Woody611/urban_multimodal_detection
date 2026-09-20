# RGBID Fusion Bottleneck Audit

**日期**：2026-09-19 · **性质**：READ-ONLY（未训练、未改代码/配置、未创建实验目录）
**唯一问题**：当前 RGBID 的主要限制，是否可能来自 Fusion 本身？

---

## 1. Frozen Baseline

```text
Official RGBID mAP50-95 (线上榜单)        = 0.53180   ← OFFICIAL_FALLBACK
本地官方口径复现 (official_map.py, 400图) = 0.50928
训练内 val (fork, results.csv ep237)      = 0.55800
```

冻结链复核 16/17 完好，未触碰。

---

## 2. Actual RGBID Forward Path

model yaml = `configs/yolo11m_earlyfusion.yaml`｜权重 = `runs/..._rgbird_ir_quicktest/weights/best.pt`

**实测结构（`torch.load(best.pt)['model']`，非文件名推测）**：

```text
输入 5ch [B,G,R,IR,D]  ← base.py::_merge_channels_rgbid, cv2.merge
   ↓
Conv2d(5, 64, k=3, s=2)  ← 唯一与 YOLO11m 不同的层
   ↓
标准 YOLO11m backbone（C3k2/Conv/SPPF/C2PSA） → 标准 PAN-FPN head → Detect(nc=12) @ P3/P4/P5
```

| 检查 | 结果 |
|---|---|
| backbone / head / scales 与 `yolo11_visible.yaml`（3ch 标准）逐项相等 | **True / True / True** |
| 唯一差异 | `ch: 5` |
| **是否存在 fusion 模块** | **不存在** |
| gating / attention / softmax / routing / TensorSelector | **不存在** |

> 当前 RGBID **没有"融合模块"**。"融合" = 在 `_merge_channels_rgbid` 里把 5 通道 `cv2.merge` 成一个张量；
> 之后 99.985% 的网络与单模态 YOLO11m 完全相同。

---

## 3. Modality Feature Flow

RGB → 输入 ch0–2｜IR → ch3｜Depth → ch4。三者**全部进入**（沿 forward 确认
`weight.shape = (64,5,3,3)`），但**仅在输入通道维区分**；stem 之后再无任何按模态区分的结构。

---

## 4. Fusion Stages

类型 = **early fusion**（输入级通道拼接）｜位置 = **仅 1 处**（stem 之前的通道维）｜
P3/P4/P5 = **无融合**（这些层看到的是已混合张量）。**不存在 multi-stage fusion。**

---

## 5. Channel / Capacity Analysis

| Stage | RGB | IR | Depth | 融合输入 |
|---|---:|---:|---:|---:|
| 输入（融合前） | 3 | 1 | 1 | 5 |
| stem 后（融合后） | 已不可分 | | | 64 |

**参数量归因（实测 best.pt）**：全模型 20,063,412 参数；唯一可按模态归因的块（stem conv）为
**3,008（0.0150%）**；其余 99.985% 均作用于已混合张量，**无法按模态归因**。

**stem 逐输入通道权重能量（L2，实测）**：

| 通道 | L2 | 占比 |
|---|---:|---:|
| ch0 B | 3.5645 | 29.38% |
| ch1 G | 3.4648 | 28.55% |
| ch2 R | 2.8496 | 23.48% |
| **ch3 IR** | 1.4219 | **11.72%** |
| **ch4 Depth** | 0.8335 | **6.87%** |
| RGB 合计 | — | **81.41%** |

> 模型在唯一可区分模态处把 **81.4%** 权重能量给了 RGB（通道数分配是 60/20/20，权重却是 81/12/7
> —— 模型**主动进一步压低** IR 与 Depth）。**但不构成"瓶颈"证明**：与 §6 消融（两者边际贡献均为**正**）
> 方向一致，更像**信息量分布的结果**而非原因，**因果方向不可判**。

---

## 6. Modality Suppression Check

搜索 softmax / attention / gating / modality weight / mask / routing：**全模型无任何此类机制**。
模态权重未被显式设计（由共享 stem 自由学习）；训练路径亦无抑制 —— 官方 `args.yaml`
无 `modality_dropout` / `ir_encoding` 键（均默认关闭）。

> 按 brief §6，无法证明"IR 被压制"→ 记 **`NOT PROVEN`**（消融显示 IR/Depth 边际贡献均为正）。
> §5 的权重失衡是**无约束优化的结果**，不是被设计出来的抑制机制。

---

## 7. Historical Coverage

| 融合方式 | 配置 | 是否训练 | 结果 |
|---|---|---|---|
| **early**（当前官方） | `yolo11m_earlyfusion.yaml` | ✅ 多轮 | 0.55800 @1280/300ep |
| **middle** | `yolo11_midfusion.yaml` | ⚠️ 1 次（`yolo11_multimodal4`） | 0.29554 @**640/200ep**（弱配方，不可比） |
| **late** | `yolo11_latefusion.yaml` | ❌ **从未训练** | — |

RGBD（4ch）线则全面使用 middle fusion（E4–E9 / F1–F5）。

```text
NO_NEW_MECHANISM（early/mid/late 家族配置均已存在） ｜ GAP = 三者从未在同配方下对比过
```

> 缺口真实存在，但**不等于"融合是瓶颈"** —— 只说明"early 的选择从未被验证"；
> 而验证它需更换架构，brief §11 明令禁止（"重新设计整个 fusion"）。

---

## 8. Candidate Bottlenecks

### Candidate A — RGBID 光度增强被静默跳过（config 声明失效）

```text
问题    ：config 声明 hsv_h=0.015 / hsv_s=0.7 / hsv_v=0.4 / erasing=0.4，但 5ch 下全部不生效
          → 训练实际为零光度增强（对比 RGBD(4ch) 走 Albumentations4C + RandomHSV4C，完整生效）。
代码位置：augment.py:3800-3802  if hyp.channels == 5: alb = Albumentations(p=0)
          augment.py:3308      RandomHSV.__call__: if img.shape[-1] != 3: return labels
机制    ：albumentations 不支持 >4ch（p 显式置 0）；RandomHSV 对非 3ch 直接 return。
          这是**融合表示选择（5ch 拼接）的下游代价**，属"融合本身"的间接后果。
已有证据：代码 + 实测（RandomHSV 作用于 5ch 数组 → 逐位不变；Albumentations(p=0).transform is None）
证据等级：CODE_LEVEL_RISK（"失效"已证；"影响最终指标"未证）｜ 重复历史？：否
验证需改 ：写 5ch 感知的 RandomHSV（只作用于 ch0-2）+ 开 alb 的 5ch 分支
```

**决定性反证（本次实测）**：官方 run **无过拟合特征** —— ep180-199 → ep240-300 期间
`train/box_loss` 0.5278→0.4017、`train/cls_loss` 0.3167→0.2506 **持续下降**，
**同时** val mAP50-95 0.53279→0.55003 **同步上升**，到 ep300 仍在获益。
→ 光度增强通常要解决的"过拟合"在这里**不存在**，该缺陷**缺乏机制支撑**。

### Candidate B — IR/Depth 填充值为 114（语义不当）

```text
问题    ：5ch 分支中 IR 与 Depth 被填充 114
代码位置：augment.py:1097-1107（RandomPerspective，aux borderValue=114）
          augment.py:1662-1667（LetterBox，5ch 落入 else → np.full(fill_value=114)）
机制    ：depth 编码 clip(mm/19999*255) 且 mm<300→0（0 = 无效）。填充 114 ≈ "约 8.9m 处有实体表面"
证据等级：CODE_LEVEL_RISK
重复历史？：**是 —— E9 已实测**（depth padding 114→0，4ch 版）0.50521 vs 0.50878 → 证伪
```

### Candidate C — stem 模态权重失衡（RGB 81.4% / IR 11.7% / Depth 6.9%）

```text
证据等级：CODE_LEVEL_RISK（已实测，因果不可判）
可低风险单变量修复？：**否** —— 需加模态专属 stem 或改融合结构 = 架构变更
```

---

## 9. Decision

**Finding**：

```text
NO_CLEAR_FUSION_BOTTLENECK
```

**理由**：

1. **没有 fusion 模块可失效** —— 是 `ch:5` 标准 YOLO11m（backbone/head/scales 逐项相等），也不存在门控/注意力/路由来压制模态（§2、§6）。
2. **模态不是被架构压制的**，而是被优化自然降权（§5）；IR/Depth 消融边际贡献**均为正**（+0.00937 / +0.01175）。
3. **唯一"新且低风险"的候选 A 无机制支撑** —— 光度增强的典型作用是抗过拟合，而官方 run 实测**无过拟合特征**（train 与 val 同向改善至 ep300）；B 重复已证伪的 E9；C 需架构变更且 mid/late fusion 已存在。
4. **上限受限于信息分布而非融合设计**：97.15% 的 TP 在移除 IR+Depth 后仍在，89.8% 的漏检与模态无关（`SINGLE_MODALITY_TO_RGBID_AUDIT.md` §6）—— **更好的融合结构触及不到这部分误差。**

| Candidate | Evidence | New mechanism? | Low risk? | Single variable? | Worth one training? |
|---|---|---|---|---|---|
| A 光度增强失效 | CODE_LEVEL_RISK | 是 | 中（需新写 5ch aug） | 是 | **否**（无过拟合，机制不成立） |
| B 填充值 114 | CODE_LEVEL_RISK | 否（=E9） | 低 | 是 | **否**（已证伪方向） |
| C stem 权重失衡 | CODE_LEVEL_RISK | 否（=mid/late） | 低 | 否（架构变更） | **否**（违反 §11） |

**无候选通过 §11 门槛。** 按 brief §18「没有，就停止」，不创造实验。

---

## 10. Next Action

无。不提出实验。

**保留观察（非本次提案）**：Candidate A 记录的「5ch 融合表示使光度增强静默失效」是**此前未被记录**的
config/实现不一致。它本身不足以支撑一次训练；若未来出现**过拟合证据**（train 持续降而 val 持续降），
它应是第一顺位排查项 —— **但本次实测的官方 run 曲线不支持该前提**。

```text
OFFICIAL_FALLBACK = 0.53180（未变更）
```

---

```text
DECISION: NO_TRAINING_RECOMMENDED
```
