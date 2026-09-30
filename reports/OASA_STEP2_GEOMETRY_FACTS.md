# OASA — STEP 2 几何事实（instrumented 实测）

**日期**：2026-09-23 · **性质**：只读实测。**0 行 OASA 实现、0 次训练、0 次 A100**
**方法**：**实例级 instrumentation**（不依赖 GT rank-match、不依赖视觉 marker）

---

## 1. 真实 transform 实例树（递归枚举，非索引假设）

```
ds.transforms                                   Compose
├─ transforms[0]                                Compose           ← 即 pre_transform
│   ├─ [0] Mosaic                               id=…856128  p=1.0   ← 唯一的 Mosaic 实例
│   ├─ [1] CopyPaste                            p=0.0
│   └─ [2] RandomPerspective                    id=…884704
│           └─ pre_transform  LetterBox         id=…886576
├─ transforms[1] MixUp  p=0.0
│       └─ pre_transform → **同一个 Compose 对象**（id 相同）⇒ Mosaic 只有一个实例，树里出现两次
├─ transforms[2] Albumentations  p=0             (5ch 关闭)
├─ transforms[3] RandomHSV                        (对非 3ch 直接 return)
├─ transforms[4] RandomFlip  p=0.0
├─ transforms[5] RandomFlip  p=0.5
└─ transforms[6] Format
```

**ON/OFF 控制点（已证实作用于真实实例）**：
```python
ds.transforms.transforms[0].transforms[0].p = 0.0 / 1.0
```
> 之前 `ds.hyp.mosaic = 0` **完全无效**——`Mosaic(dataset, imgsz, p=hyp.mosaic)` 的 `p` 在 `build_transforms` 时即固化。

**调用顺序由实例级 hook 实测确认**：`Mosaic → RandomPerspective(pre_transform=LetterBox)`，与源码顺序一致。

---

## 2. ★ 实测几何链（1080p 与 360p 各取真实训练样本）

```
=== 1080p  源图 native shape=(1080, 1920) ===
  Mosaic OFF   final tensor=(5,1280,1280)
      Mosaic            (720,1280) -> (720,1280)     scale=1.0000
      LetterBox         (720,1280) -> (1280,1280)    scale=1.7778   ← 仅 H 方向补方，W 1280→1280
      RandomPerspective (720,1280) -> (1280,1280)    scale=1.7778
  Mosaic ON    final tensor=(5,1280,1280)
      Mosaic            (720,1280) -> (2560,2560)    scale=3.5556   ← 画布 2×imgsz
      RandomPerspective (2560,2560) -> (1280,1280)   scale=0.5000   ← 关键：×0.5
```

**决定性发现**：**进入 Mosaic 的图像已经是 `(720,1280)`** —— 即 `load_image` **已把源图缩放到长边 = imgsz(1280)**，早于任何几何增广。
（1080p: ×0.667；360p: ×2.000。360p 实测同样得到 `(720,1280)`。）

---

## 3. STEP 2 要求的最终表

| source resolution | Mosaic | **final scale P25** | **median** | **P75** | Mosaic incremental |
|---|---|---:|---:|---:|---:|
| **1920×1080** | OFF | **0.667** | **0.667** | **0.667** | *baseline* |
| **1920×1080** | ON | **0.333** | **0.333** | **0.333** | **0.500×** |
| **640×360** | OFF | **2.000** | **2.000** | **2.000** | *baseline* |
| **640×360** | ON | **1.000** | **1.000** | **1.000** | **0.500×** |

（P25=P75=median 是因为链路上唯一的乘法因子都是确定性的：resize 与 canvas→final 都是固定比例。
  逐样本随机的只有 `RandomPerspective.scale∈[0.5,1.5]`，本表为隔离它已置 0。）

**其它要求项**：

| 项 | 值 |
|---|---|
| Mosaic canvas size | **2560 × 2560**（= 2 × imgsz） |
| 每 tile scale 分布 | **1.000**（tile 按 `resized_shape` 原尺寸放入画布，**Mosaic 内部不缩放**） |
| downstream geometric scale | `RandomPerspective`(含 `LetterBox(1280)`)：OFF ×1.0 / ON **×0.5** |
| 实际调用的 Mosaic 实例路径 | `ds.transforms.transforms[0].transforms[0]`（id …856128） |
| 实际调用的 RandomPerspective 路径 | `ds.transforms.transforms[0].transforms[2]`（id …884704） |
| 实际调用的 LetterBox 路径 | `ds.transforms.transforms[0].transforms[2].pre_transform`（id …886576） |

---

## 4. 数学 sanity check（§6）—— **与既有审计吻合**

```
1080p: native small √area ≈ 25.53 px
  Mosaic ON : 25.53 × 0.667 × 0.5 = 8.51 px    ← 审计记录 8.8 px  ✅ 吻合
  Mosaic OFF: 25.53 × 0.667       = 17.03 px   ← 审计记录 17.7 px ✅ 吻合
```

⇒ **既有审计（`rgbid_small_object_crop_audit.md` §8）的 mosaic 结论是正确的：mosaic 增量因子 = 0.5×。**

---

## 5. ⚠️ 对我自己 Phase-1 报告的一处更正（重要）

Phase-1 报告 §2 断言：

> 「候选 A（pre-Mosaic）会被 Mosaic 抵消 2.00× ⇒ 净 1.00×，因此无效」

**这个结论是错的。** 它建立在"Mosaic 作用于 native 分辨率的 tile"这一错误模型上。
实测事实是：**`load_image` 在任何几何增广之前已把图像缩放到长边 1280**。

正确的链路与两种候选的净尺度：

| 方案 | 链路 | 净 native→final | 最终 small √area |
|---|---|---|---|
| baseline（mosaic 期） | 0.667 × **1.0** × 0.5 | **0.333** | **8.5 px** |
| **候选 A**（pre-Mosaic ×2） | 0.667 × **2.0** × 0.5 | **0.667** | **17.0 px** |
| **候选 B**（post-Mosaic 裁 2560→1280） | 0.667 × 0.5 × **2.0** | **0.667** | **17.0 px** |

> **⇒ 候选 A 与候选 B 的净尺度收益完全相同（都是 2.0× 相对 mosaic 期 baseline）。**
> 差别只在**拓扑**：
> - **A**：被放大的 tile 与另外 3 个正常 tile 一起进 Mosaic ⇒ **mosaic 四图拼接完全保留**
> - **B**：从 2560 画布裁 1280 窗口 ⇒ 覆盖约 1/4 画布 ⇒ **该样本实际退化为近单 tile**
>
> 因此 **A 在你原本关心的"12.5% 样本失去 mosaic"这一点上更优，且同样没有 pre-Mosaic 抵消问题。**
> 你此前选 B 的依据（我的"A 会被抵消"结论）已被实测推翻 —— **这一点需要你重新裁决。**

---

## 6. close_mosaic 行为

```python
# engine/trainer.py:355
if epoch == (self.epochs - self.args.close_mosaic):
    self._close_dataloader_mosaic()
# dataset.py:197 close_mosaic(hyp): hyp.mosaic = 0.0 ... 并**重建 transforms**
```

- 正常 mosaic 期（**290/300 epoch**）：`Mosaic.p=1.0` ⇒ baseline 最终 small = **8.5 px**
- close_mosaic 期（**10/300 epoch**）：`Mosaic.p=0.0` 且 transforms **重建** ⇒ baseline 最终 small = **17.0 px**

⇒ **baseline 在最后 10 个 epoch 本来就已看到 17.0 px。** OASA 的作用是把**mosaic 期的 12.5% 样本**提到同一水平。
⇒ 若未来 OASA 放在 post-Mosaic 位置，**close_mosaic 期 OASA 会自动失效**（没有画布可裁）——
   这是预期的（该期本来就是 17.0 px），但必须在实现时显式处理，不能报错。

---

## 7. PASS / FAIL 门禁

| # | 条件 | 状态 |
|---|---|---|
| 1 | 已证明 ON/OFF 控制真的作用于实际 Mosaic instance | ✅ `transforms[0].transforms[0].p`（id 已核对） |
| 2 | 已从真实 transform arithmetic 得到 Mosaic incremental scale | ✅ **0.500×** |
| 3 | 1080p 的最终 input-space scale 已可靠测出 | ✅ **0.333 (ON) / 0.667 (OFF)** |
| 4 | close_mosaic 行为已解释 | ✅ trainer.py:355 + dataset.py:197 |
| 5 | 不依赖 GT rank-match | ✅ 实例级 hook |
| 6 | 不依赖视觉 marker | ✅ |
| 7 | 没有训练 | ✅ |
| 8 | 没有修改 OASA 实现 | ✅ 0 行 |

```text
STEP 2 = PASS
```

---

## 8. 机制结论（仅事实，不替你决定 OASA）

```text
1. Mosaic 是否真的使 source object 在最终 input space 缩小？   → 是
2. 增量因子是多少？                                          → 0.500×（1080p 与 360p 一致）
3. 因此 pre-Mosaic ×2 是否会被 Mosaic 抵消？                  → 不会。净 0.667，恰好抵消 mosaic 的 0.5
4. post-Mosaic ×2 是否仍有理论净放大？                        → 是。净 0.667，与 pre-Mosaic 等价
```

**"post-Mosaic ×2 理论机制成立"仅指几何比例成立，不代表实验有效** —— 效果仍需一次训练才能判定。

---

## 9. 需要你重新裁决的一件事

你选 B 的理由第 1 条是「Pre-Mosaic OASA 已确认 OASA ×2 → Mosaic ×0.5 → net ≈1.0×」。
实测显示该前提的**中间步骤成立但结论不成立**：net = 0.667×(2.0)×0.5 = **0.667，不是 1.0**，
因为缩放链的起点不是 native 而是**已缩放到 1280 的图像**。

⇒ **A 与 B 在净尺度上等价，A 额外保住 mosaic 拓扑、且不动 `augment.py`（改 `base.py`）。**

```text
本轮训练次数 = 0 ；OASA 实现代码 = 0 行 ；提交 A100 = 0 次
STEP 2 = PASS ；下一步待你重新裁决 候选 A / 候选 B
```
