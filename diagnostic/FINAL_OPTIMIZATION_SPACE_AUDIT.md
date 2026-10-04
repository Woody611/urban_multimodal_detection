# Final Optimization Space Audit

**日期** 2026-10-03 · **性质** READ-ONLY / ZERO-GPU / ZERO-TRAINING / ZERO-FORWARD

## Verdict

```text
VALID GPU CANDIDATES: 0
GPU: NO-GO
```

```text
NO VALID GPU CANDIDATE REMAINS.
KEEP D′ = 0.515281.
```

---

## A. Experiment ledger

Δ 列口径：**新测试集**（复赛换集）内用线上分；旧集条目另标。跨测试集分数**禁止相减**。

| # | Direction | Experiment | Best official / online | Δ vs D′ | Evidence strength | Status | Reason |
|---|---|---|---|---|---|---|---|
| 1 | baseline RGBD @1024 | E4–E7 系 | 旧集 online 0.53180 | — | A | CLOSED-A（被超越） | 后续 1280/CLAHE/SepStem 逐个超越 |
| 2 | **resolution** | F4 = 1280 | official 0.53273（本地最高） | 已并入 D′ | A | **IN-BASELINE** | 单变量兑现；D′ 即 @1280 |
| 3 | **IR-CLAHE** | D | 新集 online **47.985** | +0.0235（旧集口径） | A | **IN-BASELINE** | 历史上最大单点增益 |
| 4 | **SepStem** | D′ | 新集 online **48.712** | **incumbent** | A | **IN-BASELINE** | 三路轻量 stem，Δparams −0.0072% |
| 5 | capacity | F5 = YOLO11l@1280 | 0.52619 | −0.0065 | A | CLOSED-A | 容量杠杆证伪 |
| 6 | scale aug（小目标） | OASA 1.4 / 2.0 | 48.134 / 47.610 | −0.578 / −1.102 | A | CLOSED-A | 两尺度均低于 D′ |
| 7 | 加检测尺度 | P2 head probe | 0.50232 | −0.00696 | A | CLOSED-A | 配对 bootstrap 四指标 CI 全含 0 |
| 8 | 分辨率微调 | 1536 finetune | 纯推理 −0.0053 | 负 | A | CLOSED-A | 无增益 |
| 9 | 注意力 | P3 attention @L16 | 0.50071 / 0.51084 | 负 | B | CLOSED-A | 协议偏离（freeze 不冻 BN），confound 主导 |
| 10 | 分类监督 | train-side cls supervision | 终裁 RED | 负 | A | CLOSED-A | TAL 已把 GT 类 target 钉死为 maxCIoU(98.6%) |
| 11 | 失败阶段 | G1 Stage-4 | CLOSE | — | A | CLOSED-B | 定位到 classification/score 阶段并关闭 |
| 12 | box loss | box=10 / box=15 | 0.52093 / fork 5/5 负 | 负 | A | CLOSED-A | box gain 轴 3 个负点 |
| 13 | dfl loss | dfl=2.0 | 0.51894 | 负 | A | CLOSED-A | 同上 |
| 14 | 区域增益 | E1 RegionResponseGain | official 0.50988 | −0.00540 | A | CLOSED-A | 配对 bootstrap CI[−0.0193,+0.0054] 含 0 |
| 15 | 模态 dropout | modality dropout | 0.49022 | 负 | A | CLOSED-A | 逐像素证人齐全 |
| 16 | 小目标复现 | F1 native-small replay | 新集 online **48.204** | **−0.508** | A | CLOSED-A | 线上否决 |
| 17 | 单模态路径 | IR-only / RGB-only / Depth-only | NO_GO | — | A | CLOSED-B | 97.15% TP 不依赖 IR/Depth |
| 18 | 小目标 crop/oversample | crop audit / oversample | 负 | — | A | CLOSED-A | 路径穷尽 |
| 19 | TTA | hflip / multi-scale | −0.00637 / +0.00432 但 small AP ~0 | 负 | A | CLOSED-A | 退化框 65→188 |
| 20 | 后处理 | NMS / top-k / max_det | — | — | A | CLOSED-B | NMS 因果 2/371、top-100 5/371、max_det 从未触顶 |
| 21 | conf 阈值 | conf=0.0001 | 旧集 50.4510 vs 53.1800 | **−2.729** | A | CLOSED-A | 线上实证淘汰 |
| 22 | 集成 | Ensemble 3:1 | 未提交 | — | A | **规则禁止** | 赛题明令禁止多模型集成 |
| 23 | 训练侧 rect 尾巴 | RectLate10 | 投影 ≈48.0 | ≈−0.7 | A | CLOSED-A | last.pt 混合两变量 |
| 24 | box gain 复刻 | Box×2 | fork 5/5 负，门槛缺口 0.0190 | 负 | A | CLOSED-A | — |
| 25 | **checkpoint selection** | 30 个 epoch*.pt | 无任何官方结果 | — | A | **CLOSED-C** | 唯一可比的 last.pt 低 0.0091；val 尖峰 |
| 26 | **score dynamics** | P5 | — | — | A | **CLOSED-B** | assignment 饱和；真空非误分类；是"极化" |
| 27 | **W/h 分解** | P5.1 | — | — | A | **CLOSED-B** | head-driven；R 对 12/12 cohort 为正 |
| 28 | **W 动力学** | P5.2 | — | — | A | **CLOSED-B** | mechanism NOT IDENTIFIED |
| 29 | 融合设计 | LightFusion | 未创建 | — | A | CLOSED-C | 现有 `Conv(5→64)` **本身即**可学习融合；加投影摧毁预训练加载（14/655、remap 0） |
| 30 | 深度可靠性 | depth_reliability audit | — | — | A | CLOSED-C | 与 native-small hard miss **无可检出关联** |

## B. CLOSED directions（最终关闭原因，去重）

- **CLOSED-A（官方/线上 negative）**：#1 #5–#8 #9 #10 #12–#16 #18 #19 #21 #23 #24
- **CLOSED-B（机制审计证明不成立）**：#11 #17 #20 #26 #27 #28
- **CLOSED-C（无正式训练，但有强反证/条件不支持）**：#25 #29 #30
- **规则禁止**：#22

**CLOSED-B 的关键条目值得单列**，因为它们关闭的是**想法**而不是某次运行：

| 被关闭的想法 | 关闭依据 |
|---|---|
| 「NMS / top-100 是瓶颈」 | NMS 因果预算 2/371、top-100 5/371、max_det 从未触发 |
| 「scale aug / 多尺度能救小目标」 | OASA 两尺度线上负；且 F0（无候选）根本没有 scale 可言 |
| 「assignment 驱动 score death」 | P5：`mask_pos ≡ 1.000`、`n_pos ≡ 10`、`ciou` 平坦 —— 该层**无自由度** |
| 「细长目标是机制」 | P5：长宽比 >3.2 的 dead 率 **8.3% < 方形 14.8%**（方向相反） |
| 「表征漂移是主因」 | P5.1：`R = +3.75` 对 **12/12 cohort 全为正**，类居中 logit 反而 6.28→+12.85 |
| 「类频率驱动 score」 | P5.2：`corr(频率, Δ(W·h)) = −0.021`（频率只解释 row 范数，不解释 logit） |
| 「存在 vacuum 未恶化而 metric 更好的 checkpoint」 | Checkpoint audit §7：P5 各量与 val metric **同向**（corr +0.79~+0.97） |

## C. OPEN-WEAK

以下两项理论上仍"没跑过"，但**证据密度不足以消耗一次 GPU**：

| 方向 | independent evidence | contradictory evidence | mechanism specificity | single-run falsifiable |
|---|---|---|---|---|
| **5ch 光度增强失效**（`hsv_h/s/v`、`erasing` 在 5 通道下 inert） | 1（代码事实，`rgbid-5ch-photometric-aug-inert`） | **1**（官方 run 无过拟合特征 ⇒ 缺失的正则不是被测出的瓶颈） | weak | yes |
| **CLAHE 超参**（`clipLimit` / `tileGridSize`） | 1（CLAHE 本身有效） | 1（无证据表明当前 2.0 次优） | weak | **no（需 3–5 点 sweep）** |

```text
not sufficient for GPU
```

**理由**：两项都是**给一个已证明不是瓶颈的量加码**。光度增强那一项的预期增益被"无过拟合"直接抵消；
CLAHE 超参没有独立证据指向某个更优值，且违反 §4-Q7（多尺度 sweep ⇒ REJECT）。

## D. OPEN-STRONG

```text
NONE.
```

**特别检查过、但一个都不成立的"看起来像候选"的方向：**

| 看似候选 | 为何不成立 |
|---|---|
| **head 干预**（classifier LR / weight decay / temperature / logit scale） | §5 明文禁止。且 P5/P5.1/P5.2 **只证明 head 在变，不证明改它会提高 metric**。更关键：Checkpoint audit §7 显示 vacuum 与 metric **同向** ⇒ 这是**反证**，不是候选 |
| **bias 冻结的修复**（P5.2 的 M6） | 同上；且"为何冻结"未识别（需梯度，artifact 里没有） |
| **补评 30 个 epoch\*.pt 的官方分** | 不是优化实验；且 §2 显示 **30 个快照中 val 最高的仍比 best.pt 低 0.0054** ⇒ 无超越可能 |
| **延长训练 / 换 seed** | 曲线在 ep230 后平台（末 10 epoch 均值比峰值低 0.01265）；且 §4-Q7 需多 seed ⇒ REJECT |
| **融合结构再设计** | LightFusion 已 FAIL，根因是预训练加载（remap 返回 0）—— 任何新增前置层都会撞同一堵墙 |
| **深度通道编码改造** | depth_reliability 审计：与 hard miss **无可检出关联** |

## E. Final candidate

```text
N/A —— VALID GPU CANDIDATES = 0
```

## F. If no candidate

```text
NO VALID GPU CANDIDATE REMAINS.
KEEP D′ = 0.515281.
```

**不再提出"可以顺便试试"的实验。**

---

## 附：为什么这不是"没找到"，而是"已经找完了"

1. **增益来源已全部在模型里**。历史上真正推动线上分的只有三件事 ——
   **1280 分辨率**、**IR-CLAHE**、**SepStem** —— 三者都是当前 incumbent 的组成部分。
   除它们之外，**没有任何一个方向产生过官方或线上的正向 Δ**。

2. **剩余误差的机制已定位到"无自由度"的层**。P3（F0 主导）→ P4（F0 是 conf 阈值效应）
   → P5（assignment 饱和、真空非误分类、极化）→ P5.1（head-driven，表征在改善）
   → P5.2（W 旋转 + bias 冻结，但 upstream 未识别）。每一环都把候选空间**缩小**，从未扩大。

3. **P5 线提供的是反证而非候选**。Checkpoint audit §7 的 corr(val, live_dead) = **+0.792**
   说明这些"看起来像病理"的量与 metric **同向**。因此**任何以消除它们为目标的干预，其预期增益为负或零**。

4. **§4-Q7 淘汰了最后的 sweep 类方向**。CLAHE 超参、seed、延长训练都需要多次运行才可能可信 ——
   按硬规则直接 REJECT。

5. **规则侧也已封顶**。集成被禁；推理参数（conf/iou/max_det）已证无空间。

⇒ 正确决定是 **STOP EXPERIMENTATION**。这是实验结论，不是放弃。
