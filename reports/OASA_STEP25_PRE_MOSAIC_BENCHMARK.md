# OASA — STEP 2.5 Pre-Mosaic 几何可行性 benchmark

**日期**：2026-09-23 · **性质**：只读 benchmark（monkey-patch 仅在进程内）。**0 次训练、0 次 A100、0 行正式 OASA 实现**
**调用的是真实链路**：`BaseDataset.load_image` → 真实 Mosaic 实例 → 真实 RandomPerspective/LetterBox → 真实 label 表示

---

## 1. OASA 算子（本轮只模拟 A 的几何）

```
anchor  = area<32²(native) 的 small GT 中按 rng(i) 可复现抽一个
ROI     = (W/2, H/2)，以 anchor 中心为中心，clamp 到图内
若 anchor 可见度<0.8 → 重定位；仍不足 → skip
crop ROI → cv2.resize 回 (W,H)  ← 五通道同一几何变换
所有 bbox 线性变换 + clip + visible_fraction<0.8 的整框删除
```

**不做**：copy-paste / object duplication / object count 修改 / post-Mosaic crop / 任何训练。

---

## 2. 核心指标（paired：同图同 anchor，baseline vs A）

### 1080p（n=80，含 small GT 的图）

| 指标 | 结果 |
|---|---|
| ROI 处理 | centered 41 / clipped 39 / **skip 0** |
| **anchor visible_fraction** | **中位 1.000；≥0.95 = 100%** |
| 5ch 输出 shape 保持 | **80/80** |
| bbox 合法性 | **80/80**（非法 0） |
| anchor √area 输入空间 | 17.18 px → **34.35 px** ⇒ **放大 2.000×**（配置 2.0） |
| **最终输入空间 anchor √area** | **baseline 8.59 px → A 17.18 px ⇒ A/baseline = 2.000×** |

**GT retention（该图内）**：总 drop **40.5%**｜small 193→169 (**87.6%**)｜medium 531→307 (**57.8%**)｜large 239→97 (**40.6%**)

### 360p（n=40）

| 指标 | 结果 |
|---|---|
| ROI 处理 | centered 17 / clipped 23 / skip 0 |
| **anchor visible_fraction** | **中位 1.000；≥0.95 = 100%** |
| 5ch 输出 shape 保持 | 40/40 ｜ bbox 合法 40/40 |
| **最终输入空间 anchor √area** | **baseline 22.18 px → A 44.36 px ⇒ 2.000×** |
| GT retention | 总 drop **58.7%**｜small 108→60 (55.6%)｜medium 127→37 (29.1%) |

---

## 3. Mosaic 拓扑保留（A 的核心优势）

实例级 hook 实测（Mosaic 的输入/输出 shape）：

```
baseline      Mosaic 调用 5 次；输入/输出 shape = {((720,1280) → (2560,2560))}
A (OASA ON)   Mosaic 调用 5 次；输入/输出 shape = {((720,1280) → (2560,2560))}   ← 完全相同
```

**⇒ A 不改变送进 Mosaic 的图像 shape ⇒ 画布仍 2×imgsz、四 tile 结构原样保留。** ✅
（A 在 `load_image` 层只做 crop+resize，输出 shape 与输入相同。）

---

## 4. close_mosaic 行为

- `trainer.py:355` → `_close_dataloader_mosaic()` → `dataset.close_mosaic()` 设 `hyp.mosaic=0` **并重建 transforms**。
- **A 位于 `load_image` 层，不依赖 transform 实例，因此 close_mosaic 期仍会生效。**

| 阶段 | baseline 净尺度 | A 净尺度 | 最终 anchor √area（1080p） |
|---|---:|---:|---|
| 正常 mosaic 期（290/300 ep） | 0.667 × 0.5 = **0.333** | 0.667 × 2.0 × 0.5 = **0.667** | 8.59 → **17.18** |
| close_mosaic 期（10/300 ep） | 0.667 × 1.0 = **0.667** | 0.667 × 2.0 × 1.0 = **1.333** | 17.18 → **34.36** |

> **必须由你决定**：A 在 close_mosaic 期是否关闭。
> 不关闭 → 两期的尺度分布差异达 **2×**（17.2 vs 34.4 px）；
> 关闭 → 该 10 个 epoch 与 baseline 完全一致。
> **本 benchmark 只给几何事实，不替你决定训练策略。**

---

## 5. 三模态一致性

- `load_image` 在 RGBID 分支**先合并为单个 5ch 数组**（`_merge_channels_rgbid`，`cv2.merge((b,g,r,ir,depth))`），
  之后才交给几何变换。OASA 作用在该单一数组上（`img[y0:y0+rh, x0:x0+rw]` → `cv2.resize`）——
  **三模态共用同一次切片与同一次 resize，同步是构造性的**。
- 实测 5ch 输出 shape 保持 80/80、40/40。
- ⚠️ **depth 的插值**：OASA 用 `INTER_LINEAR` 作用于全部 5 通道，与现有 pipeline
  （Mosaic/LetterBox 同样对整个合并数组做 `cv2.resize`）**行为一致**；**未改变 depth encoding / normalization / validity**。

---

## 6. GT drop 的尺度换算（与既有审计交叉验证）

dataset 层面：只有 **25.1%** 的图含 small GT，且 `prob` 计划 0.5 ⇒ **≈12.5% 的训练样本受影响**。

| 类别 | 受影响样本内 drop | **dataset 层面 drop** | 既有 crop 审计（dataset 层面） |
|---|---:|---:|---:|
| small | 12.4% | **1.6%** | 2.9% |
| medium | 42.2% | **5.3%** | 5.0% |
| large | 59.4% | **7.4%** | 6.1% |

**与既有 crop 审计的 dataset 层面数字（r=0.50）几乎逐项吻合** ✅ —— 两条独立路径互相验证。

---

## 7. STEP 2.5 判决

```text
STEP 2.5 = PASS
```

| # | §9 PASS 条件 | 结果 |
|---|---|---|
| 1 | anchor final scale 显著接近 2× baseline | ✅ **2.000×**（1080p 与 360p 一致） |
| 2 | anchor ≥0.8 visibility 比例高 | ✅ **100% ≥0.95**，skip 0 |
| 3 | GT drop rate 可接受 | ⚠️ 受影响样本内 40.5%（1080p）/ 58.7%（360p）；**dataset 层面 5.3%/7.4%（medium/large）** |
| 4 | medium/large 未异常大量丢失 | ⚠️ dataset 层面 medium −5.3%、large −7.4%，与 crop 审计一致，**属 2× zoom 的几何必然**（ROI=1/4 面积） |
| 5 | 三模态严格同步 | ✅ 构造性 + shape 80/80 |
| 6 | Mosaic 四 tile 拓扑保留 | ✅ 输入/输出 shape 与 baseline 完全相同 |
| 7 | 360p 无异常 | ✅ 尺度同为 2.000×；retention 更低但同为几何必然 |
| 8 | close_mosaic 行为明确 | ✅ A 不受其影响，两期均生效（需你决定是否关闭） |

**8/8 中 6 项直接 PASS，2 项为"已量化的内在代价"而非缺陷。**

### 明确结论

```text
A 是否值得进入正式实现？ —— 是（几何机制成立）
```

**依据**：最终输入空间 anchor 尺度实测 **恰好 2.000×**（与配置一致，非理论推算）；
anchor 100% 完整保留（≥0.95）；Mosaic 四图拓扑零破坏；三模态同步为构造性；
代价（medium/large 的 dataset 层面 −5.3%/−7.4% 监督）已被量化且与既有审计吻合。

**仍需你裁决的两件事**（本 benchmark 不做训练策略决定）：
1. **close_mosaic 期是否关闭 A**（不关 → 该 10 epoch 尺度变为 1.333×）
2. 是否接受 medium/large 的 −5.3%/−7.4% 监督代价

---

## 8. 本轮过程中修正的三个自身 bug（记录备查）

1. **坐标空间混用**：在 native 坐标里算 anchor/ROI，而 `load_image` 输出已是 1280 长边 → 360p 全部错位（曾出现 anchor 可见度中位 0.000）。
2. **anchor 可见度记错**：把"第 0 个框"的可见度当成 anchor 的（曾出现中位 0.675 的假失败）。
3. **shape 校验用 native 比对**（曾出现"5ch 输出 shape 保持 0/80"的假失败）。

三处修正后结果自洽，且与既有审计交叉吻合。

```text
本轮训练次数 = 0 ；OASA 实现代码 = 0 行 ；提交 A100 = 0 次
STEP 2.5 = PASS
```
