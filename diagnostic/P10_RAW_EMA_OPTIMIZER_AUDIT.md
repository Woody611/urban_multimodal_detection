# P10 — Raw-vs-EMA / Optimizer Dynamics Audit

**日期** 2026-10-03 · **性质** ZERO GPU / 只读源码 + 已有 checkpoint CPU 代数 · 无 forward / backward / 训练 / 推理 / 验证 / 新 checkpoint
**对象** `runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/epoch{0,10,...,290}.pt`（29 个快照，D′ lineage，300ep / `save_period:10`）
**产出** `diagnostic/p10_out/*.json`、脚本 `diagnostic/p10_raw_ema_optimizer_audit.py` / `p10_stage2.py` / `p10_timeline.py`

---

## 1. Executive verdict

```text
MECHANISM STILL OPEN
GPU NO-GO
```

**本轮把 P9 剩下的"budget 不可测"这一环向前推进了一大步**（EMA 被排除、初始化腿被闭合、梯度组成被直接测量），
但**因果链的时间顺序是反的** —— W_mean 动得比 logit 塌得早，且早期 W_mean 暴涨期间 dead rate 反而**下降**。
因此**不申请 GPU**。

---

## 2. D′ optimizer configuration

| component | value | source | confidence |
|---|---|---|---|
| optimizer | **SGD + nesterov** | `trainer.py:807-808`；`epoch0.pt.optimizer` | EXACT |
| lr0 | **0.005** | args.yaml | EXACT |
| lrf | **0.01**（cos） | args.yaml；`args.yaml` `cos_lr: true` | EXACT |
| momentum/beta | **0.937**（warmup 0.8→0.937） | args.yaml + 实测 `group_moms` | EXACT |
| weight decay | **5e-4**（仅 w 组；bias/BN 组 **0**） | `trainer.py:815-816`；实测 `group_wd=[0, 5e-4, 0]` | EXACT |
| warmup | **3.0 epoch = 600 iter**（`nb=200`，`nw=max(3×200,100)`） | `trainer.py:326`；`nb` 由 `updates` 反推 | EXACT |
| `warmup_bias_lr` | **0.1**（仅 group0=bias） | args.yaml；实测 ep0 `lr_g0=0.0685` | EXACT |
| `warmup_momentum` | 0.8 → 0.937 | args.yaml；实测 ep0 `mom=0.8454` | EXACT |
| EMA decay | **`0.9999·(1−e^{−t/2000})`**，动态 ramp | `torch_utils.py:546` | EXACT |
| EMA update freq | **每个 optimizer step 一次** | `trainer.py:587-595 optimizer_step()` | EXACT |
| AMP | `amp: true`（fp16 autocast + GradScaler） | args.yaml；`trainer.py:583` | EXACT |
| accumulation | **8**（`round(64/8)`），warmup 内 1→8 ramp | `trainer.py:301,369` | EXACT |
| grad clipping | `clip_grad_norm_(max_norm=10.0)` | `trainer.py:585` | EXACT |
| batch / nbs / imgsz / epochs | 8 / 64 / 1280 / 300 | args.yaml | EXACT |
| **parameter groups** | **[0]=bias (114 个, wd 0)** · **[1]=weights (115, wd 5e-4)** · **[2]=BN (108, wd 0)** | 实测 `group_sizes`，`build_optimizer` | EXACT |

**★ classifier W（`model.30.cv3.0.2.weight`, Conv2d(256→12,1×1)）落在 group[1]（wd=5e-4）；classifier bias 落在 group[0]。**
**两组 lr 在 warmup 后完全相同（`lr_g0 == lr_g1`，实测）** ⇒ 分类头**没有**特殊 LR / 特殊 weight decay。

**LR 轨迹（实测 `optimizer.param_groups[*].lr`）**

| ep | 0 | 10 | 30 | 60 | 100 | 150 | 200 | 250 | 290 |
|---|---|---|---|---|---|---|---|---|---|
| lr_g1(weights) | 0.001658 | 0.004986 | 0.004879 | 0.004527 | 0.003762 | 0.002525 | 0.001287 | 0.000382 | **0.000064** |
| lr_g0(bias) | **0.068492** | 同上 | 〃 | 〃 | 〃 | 〃 | 〃 | 〃 | 〃 |

**★ warmup 的前 3 个 epoch 里 bias 组拿到最高 0.1 的 LR（≈20× base），weights 组从 0 爬升。**
**但 bias 几乎不动（见 §3）⇒ 这一段 LR 优势在本模型上基本是惰性的。**

**updates（EMA 步数，直接读自 checkpoint `["updates"]`）**

| ep | 0 | 10 | 20 | … | 290 | 300(外推) |
|---|---|---|---|---|---|---|
| updates | **109** | 384 | 634 | +250/10ep | **7384** | ≈7634 |

`nb = 200` batch/epoch（109→384 跨 10 epoch 而 warmup 期 accumulate<8）；warmup 后 **25 updates/epoch**。
**独立交叉验证**：`trajectory_metadata.json` 的 runtime 记录 `ep0 updates_total=109, batches=200, this_epoch=25` —— **与 checkpoint 逐位一致**。

---

## 3. Initialization audit

### 3.1 源码

`ultralytics/nn/modules/head.py:143-144`（Detect 头）：

```python
a[-1].bias.data[:] = 1.0                                        # box 分支
b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)   # ★ cls 分支
```

⇒ **分类 bias 是确定性初始化**，与数据、预训练、随机种子**都无关**。

### 3.2 理论值 vs 实测（三个 stride 分支）

| 分支 | 理论 init `log(5/12/(640/s)²)` | 实测 ep0 `b_mean` | 实测 ep290 `b_mean` | **全程漂移** | `b_std` ep0→ep290 |
|---|---:|---:|---:|---:|---:|
| **cv3.0 (stride 8)** | **−9.6395** | **−9.7331** | −9.7949 | **−0.062** | 0.0506→0.0654 |
| **cv3.1 (stride 16)** | **−8.2532** | **−8.3743** | −8.4434 | **−0.069** | — |
| **cv3.2 (stride 32)** | **−6.8669** | **−7.0146** | −7.0882 | **−0.074** | — |

### 3.3 结论

```text
b ≈ −9.7 是【初始化】，不是【训练结果】。
```

- 三个分支**各自命中理论值**，误差 0.10–0.15（= 前 1 个 epoch 的移动量）。
- 之后 **300 个 epoch 只再动 0.06–0.07（≤2%）**，且 `b_std ≈ 0.05–0.065`（12 个类几乎相同）。
- **⇒「负 logit 真空」的地板是一个初始化常数**（YOLO 的 `5 objects / nc / stride²` 先验），
  **训练没有制造它，也没有消除它**。

**★ 这把 brief §7/§15 要求区分的两件事干净地分开了：**

| 现象 | 归属 |
|---|---|
| **负 logit 地板（≈ −9.7 / −8.3 / −6.9）** | **initialization**（100% 解释，无需训练） |
| **`W_mean` +188%** | **optimization**（bias 全程冻结，与它无关） |

**⇒「vacuum 的时间起点」= epoch 0 之前的初始化；「W_mean growth」= 训练产生的，两者是不同机制，不得混算。**

### 3.4 W 的初始化

`cv3.*` 权重来自 `parse_model` 的 kaiming_uniform（`nn.Conv2d` 默认），**不经预训练**：
`yolo11m.pt` 的 head 是 `nc=80`，形状不匹配 ⇒ `intersect_dicts` 丢弃。
随机 12×256 kaiming 的 **row-mean 理论范数 ≈ 0.1667**，实测 ep0 `‖W_mean‖ = 0.1696`（+1.7%）—— 两者不可区分。

---

## 4. Raw vs EMA

```text
RAW-vs-EMA comparison = UNMEASURABLE
```

**源码级确认**（`trainer.py:520-537` `save_model()`）：

```python
ckpt = {"epoch":…, "model": None,          # ★ resume and final checkpoints derive from EMA
        "ema": deepcopy(self.ema.ema).half(), "updates":…,
        "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())), …}
```

- **29/29 快照的 `"model"` 均为 `None`** ⇒ **raw 权重从未落盘**（`best.pt`/`last.pt` 经 `strip_optimizer` 后 `model := ema`，同样是 EMA）。
- 全部快照是 **fp16 EMA**；文件 80.8 MB（含 optimizer）vs stripped 40.6 MB。

**因此 P5.1/P5.2 里的 `W_t` 100% 是 EMA**（probe 元数据亦记 `probe_target: trainer.ema.ema`）。
**但 EMA 不是一个强平滑器 —— 见 §5。**

---

## 5. EMA 实现与"是否制造了 ep0→ep10 的表型"

### 5.1 精确公式（`torch_utils.py:551-562`）

```python
self.updates += 1
d = 0.9999 * (1 - math.exp(-self.updates / 2000))     # ★ per-UPDATE ramp, tau=2000
v *= d;  v += (1 - d) * msd[k].detach()               # v = EMA, msd = raw
```

**更新频率 = 每个 optimizer step**（`trainer.py:587-595`），**不是每 batch、不是每 epoch**。

### 5.2 decay 是否动态 —— 是，且被精确读出

| ep | 0 | 10 | 50 | 100 | 150 | 200 | 250 | 290 |
|---|---|---|---|---|---|---|---|---|
| updates | 109 | 384 | 1384 | 2634 | 3884 | 5134 | 6384 | 7384 |
| **decay / update** | 0.0530 | 0.1747 | 0.4994 | 0.7320 | 0.8565 | 0.9231 | 0.9588 | **0.9750** |
| **等效记忆** | 1.1 upd | 1.2 upd | 2.0 upd | 3.7 upd | 7.0 upd | 13.0 upd | 24.3 upd | **40 upd ≈ 1.6 ep** |

**★ ep290 时"10 个 epoch 以前"的更新只占 `e_290` 权重的 `0.0011`（0.11%）。**

### 5.3 结论：**EMA 是透明的，H3 被排除**

- `updates=109` 时（ep0）**init 权重占 `e_ep0` 的 0.0000**；换句话说 **ep0 快照 ≈ epoch 0 最后一 step 的 raw**，不是平均。
- 用**精确的 per-update decay 序列**对一个**线性 raw ramp** 做滤波仿真：

| 累计完成度 | ep10 | ep50 | ep100 | ep200 |
|---|---:|---:|---:|---:|
| raw（线性，构造值） | 3.78% | 17.53% | 34.71% | 69.07% |
| **EMA-of-linear** | **3.80%** | 17.61% | 34.85% | 69.27% |
| **实测 EMA** | **37.41%** | 71.77% | 87.69% | 98.67% |

**⇒ 滤波器对平滑信号的传递率 ≈ 1.005（几乎恒等）。EMA 无法凭空制造 10× 的 front-loading。**
**观测到的 ep0→ep10 front-loading 是 raw 轨迹自身的性质 ⇒ H3（EMA 造成观测表型）NOT SUPPORTED。**

---

## 6. Early-motion audit

`model.30.cv3.0.2.weight`（stride-8 分类器）：

| interval | Δ‖W_mean‖ | Δ‖W_cent‖ | ΔW/‖W‖ | lr | EMA decay(末端) | 方向变化 cos |
|---|---:|---:|---:|---:|---:|---:|
| ep0→ep1…10 | **+0.1195** | −0.0443 | +70.4% | 0.00499 | 0.175 | **cos(Wm₀,Wm₁₀)=0.7732（39°）** |
| ep10→20 | +0.0428 | −0.0018 | +14.8% | 0.00499 | 0.272 | 0.9967 |
| ep20→30 | +0.0287 | +0.0050 | +8.7% | 0.00495 | 0.357 | 0.9923 |
| ep30→50 | +0.0383 | +0.0064 | +10.6% | 0.00488 | 0.499 | 0.9856 |
| ep50→100 | +0.0508 | +0.0118 | +12.7% | 0.00467 | 0.732 | 0.9754 |
| ep100→200 | +0.0351 | −0.0143 | +7.8% | 0.00376 | 0.923 | 0.9672 |
| ep200→290 | +0.0042 | −0.0071 | +0.9% | 0.00129 | 0.975 | 0.9659 |

**绝对位移占比**：ep0→10 = **37.4%**，ep0→50 = 71.8%，ep0→100 = 87.7%（P5.2 用 `‖W_mean(t)−W_mean(0)‖` 记作 **46.5% @ ep10**，差异来自它含 39° 旋转的贡献）。

**★★ 两相结构（本轮新发现）**

| 相 | 区间 | W_mean 方向 | W_mean 幅度 |
|---|---|---|---|
| **I 重定向相** | ep0 → ep10 | **旋转 39°**（cos 0.7732） | +70% |
| **II 纯放大相** | ep10 → ep290 | **几乎不动**：相邻 cos **≥ 0.9917，0/28 为负**；cos(Wm₁₀, Wm₂₉₀)=**0.9659** | 再 +69% |

`ln` 增长对半：ln(0.2890/0.1696)=0.5330 / ln(0.4889/0.2890)=0.5250 ⇒ **50.4% / 49.6%**。

**⇒ 共同模方向在前 ~10 个 epoch 内被"设定"，之后只是沿固定方向的幅值增长（并随 LR 衰减饱和）。**

---

## 7. Optimizer-state audit（★ 本轮唯一"新数据源"）

`save_model()` 把 `optimizer.state_dict()` 整个写进快照 ⇒ **SGD momentum buffer 可用**（336 个 state entry，fp16；`m_t = 0.937·m_{t−1} + g_t`，**是 gradient 的低通代理，memory ≈ 16 step ≈ 0.6 epoch**，且**不受 EMA 平滑**）。

| ep | ‖W_mean‖ | ‖W_cent‖ | **‖M_mean‖** | **‖M_cent‖** | `cos(M_mean, W_mean)` | `cos(M_cent, W_cent)` | **M_mean/(M_cent/√12)** |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1696 | 1.9272 | 0.3340 | 0.8305 | **−0.239** | +0.058 | 1.393 |
| 10 | 0.2890 | 1.8829 | 0.1160 | 0.8541 | **−0.385** | +0.138 | 0.471 |
| 50 | 0.3988 | 1.8925 | 0.0551 | 0.5786 | **−0.385** | +0.150 | 0.330 |
| 100 | 0.4496 | 1.9043 | 0.0530 | 0.6016 | **−0.309** | +0.157 | 0.305 |
| 200 | 0.4847 | 1.8900 | 0.0325 | 0.5494 | **−0.286** | +0.054 | 0.205 |
| 290 | 0.4889 | 1.8829 | 0.1734 | 0.8739 | **−0.141** | +0.422 | 0.687 |

**全 29 快照统计（stride-8）**

| 量 | mean | median | 负号数 |
|---|---:|---:|---:|
| `cos(M_mean, W_mean)` | **−0.1981** | −0.2168 | **26/29** |
| `cos(M_cent, W_cent)` | **+0.0220** | +0.0227 | **14/29**（≈ 掷硬币） |
| `‖M_mean‖/(‖M_cent‖/√12)` | 0.354 | 0.288 | — |
| 10-epoch 持续性 `cos(M_mean(t), M_mean(t−10))` | +0.1134 | — | 11/28 |
| 10-epoch 持续性 `cos(M_cent(t), M_cent(t−10))` | +0.0639 | — | 11/28 |

**★★★ 这是本轮的核心定量结果：**

1. **`cos(M_mean, W_mean) < 0` 在 26/29 个快照成立** ⇒ 共同模梯度**系统性地**指向 `−W_mean`；
   梯度下降取 `−g` ⇒ **更新持续地把 `‖W_mean‖` 推大**。**⇒ 这就是 +188% 的"持续性"来源。**
2. **`cos(M_cent, W_cent) ≈ +0.02，符号 14/29 随机** ⇒ 判别性梯度**与判别性参数无系统关系** ⇒ 逐 epoch 互相抵消 ⇒ **这就是 `‖W_cent‖` 基本持平（−2.3%）的来源。**
3. **但梯度不是"共同模占优"**：`‖M_mean‖/(‖M_cent‖/√12)` 中位 **0.288** ⇒ **梯度的逐行偏离分量比共享分量大 ~3.5×**。
   **⇒ 不对称的来源是【符号持续性】，不是【幅度】。**

**激励–响应量级检查（把 §7 与 §6 接起来，粗模型、标 INFERENCE）**

`Δ‖W_mean‖ ≈ −lr·250·(1+0.937)·(M_mean·Ŵ_mean)`：

| interval | proj (M_mean·Ŵ) | pred ΔW | obs ΔW | obs/pred |
|---|---:|---:|---:|---:|
| 0→10 | −0.0799 | +0.1935 | +0.1195 | 0.62 |
| 10→20 | −0.0446 | +0.1077 | +0.0428 | 0.40 |
| 100→110 | −0.0164 | +0.0299 | +0.0061 | 0.20 |
| 200→210 附近中位 | — | — | — | ≈0.3 |

**符号 24/26 一致；幅度系统高估 2–5×（median ratio ≈0.4）。**
**⇒ 量级一致，但不是精确 budget。不据此下因果结论。**

---

## 8. Hypothesis table

| hypothesis | evidence | status |
|---|---|---|
| **H1 真实 gradient 就是 common-mode 占优** | `‖M_mean‖/(‖M_cent‖/√12)` 中位 **0.288** ⇒ 偏离分量大约 3.5× | **NOT SUPPORTED** |
| **H2 optimizer / momentum 把它变成 common-mode** | momentum 是**线性低通**（DC 增益 1/(1−m)），**不能旋转方向**；但它对持续性分量增益 15.9×、对零均值分量衰减 ⇒ **放大不对称、不制造不对称**。真正的不对称在梯度里：`cos(M_mean,W_mean)` 26/29 为负 vs `cos(M_cent,W_cent)` 14/29 | **PARTIALLY SUPPORTED**（"放大"部分支持，"制造"部分不支持） |
| **H3 EMA 制造了观测表型** | 记忆 1.2→40 upd（0.05→1.6 ep）；>10ep 权重 **0.0011**；线性 ramp 滤波传递率 **1.005**；ep0 快照 init 权重 **0.0000** | **NOT SUPPORTED** |
| **H4 初始化 / 早期瞬态** | **bias：完全由 init 解释**（三分支命中 `log(5/12/(640/s)²)`，之后冻结 ≤2%）·**W_mean 方向：39° 旋转全部发生在 ep0–10，之后 0/28 反向** ·**幅度：ep0–10 只占 37%（绝对）/50%（log）** | **PARTIALLY SUPPORTED**（bias 腿 FULL；W_mean 腿 PARTIAL） |
| **H5 组合** | 不同腿由不同因素主导：地板=init，方向=早期瞬态，幅值放大=持续性梯度×momentum | **SUPPORTED（但这种"H5"不是机制，是分类）** |

---

## 9. Causal-chain audit

```text
initialization                          [SUPPORTED]      §3：b=log(5/nc/(640/s)^2)，三 stride 命中，之后冻结
      ↓
classification gradient                 [SUPPORTED]      §7：momentum buffer 直接可读；common-mode 符号持续 26/29
      ↓
optimizer (SGD+nesterov, mom .937)      [SUPPORTED]      §2：参数组/LR/momentum 全部实测
      ↓
raw W                                   [UNMEASURABLE]   §4：29/29 快照 "model": None
      ↓
EMA                                     [SUPPORTED]      §5：公式+精确 updates；记忆 1.2–40 upd ⇒ 近似恒等
      ↓
W_mean ↑188% / W_cent 持平              [SUPPORTED]      §6：0.1696→0.4889；1.9272→1.8829
      ↓
GT logit 共同模 −8.67 → −14.46          [PARTIAL]        §10：**时间上滞后于 W_mean**（ep10: 46.5% vs 18.3%）
      ↓
F0 / dead rate ↑                        [NOT SUPPORTED]  §10：**早期 dead rate 反而 14.77% → 11.82%**
```

**⇒ 链条在最后两条边上断裂：不是"没有数据"，是"数据方向相反"。**

---

## 10. Temporal ordering（§13；用 P5.2 的**有效**轨迹 + P10 checkpoint 的 W）

| ep | ‖W_mean‖ | `dWm`(P5.2) | `dWm`% | **gm（共同模 logit 电平）** | **gm%** | **live_dead** | **dead%** |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1696 | 0.0000 | 0.0% | −8.6698 | 0.0% | **14.77%** | 0.0% |
| **10** | 0.2890 | 0.1911 | **46.5%** | −9.7315 | **18.3%** | **11.82%** | **−14.6%** |
| 20 | 0.3318 | 0.2395 | 58.3% | −9.6291 | 16.6% | 13.33% | −7.1% |
| 50 | 0.3988 | 0.3128 | 76.2% | −10.7174 | 35.4% | 17.35% | +12.7% |
| 100 | 0.4496 | 0.3677 | 89.5% | −11.8744 | 55.3% | 22.42% | +37.8% |
| 200 | 0.4847 | 0.4059 | 98.8% | −13.5134 | 83.6% | 28.11% | +65.9% |
| 290 | 0.4889 | 0.4107 | 100% | −14.4617 | 100% | 35.00% | 100% |

（`gm` = 共同模 logit 电平；`live_dead` = 用当前 W_t 时的 dead 率。均取自 P5.2 有效表。）

```text
PRECEDENCE:
  W_mean / dWm   ——  领先         （ep10 已完成 46.5%）
  gm（共同模 logit）—— 滞后         （同点仅 18.3%；50% 分位约在 ep60–65）
  dead rate      ——  滞后最多     （同点 **低于** 基线；50% 分位约在 ep120）
  val mAP50-95   ——  与 W_mean 同向上升（ep0 0.25725 → ep10 0.43047 → ep290 …）

★ 关键反证：ep0→ep10 期间 W_mean 完成了 46.5% 的总位移，
  但 dead rate 从 14.77% 降到 11.82%（mAP 0.257→0.430）。
  ⇒ 【早期 W_mean 暴涨】与【vacuum】不是同一件事，方向相反。
```

**仅报告 precedence / lag / simultaneity，不作 causal claim。**
`corr(‖W_mean‖, gm)` 之类的同向相关（P5 已记 `corr(val, live_dead)=+0.79`）**全部不予因果解读**。

---

## 11. GPU Gate

| gate | status |
|---|---|
| **upstream mechanism identified** | **PARTIAL** —— 识别出「**持续性符号**（26/29）而非幅度（0.288）」这一机制类，以及「方向在 ep0–10 设定后冻结」；**但持续性梯度的来源未识别** |
| **explains `W_mean` +188%** | **PARTIAL** —— 定性成立（26/29 同号）；定量只到量级（obs/pred ≈ 0.4，误差 2–5×） |
| **explains centered-flat** | **YES** —— `cos(M_cent,W_cent)=+0.022`，符号 14/29，逐 epoch 抵消 |
| **temporal precedence** | **❌ 反向** —— W_mean 领先，logit / dead-rate 滞后；早期 dead rate 反而下降 |
| **explains logit decline** | **NO** —— decline 的主体（81.7%）发生在 W_mean 已完成 46.5% 之后 |
| **minimal intervention** | **NO** —— 未产生新的、可证伪且针对 D′ 的干预 |
| **GPU candidate** | **NO** |

```text
GPU NO-GO
```

---

## 附：本轮实际执行的核查

1. 读 `trainer.py:514-548` ⇒ **`"model": None`**，raw 权重不落盘；`optimizer.state_dict()` **落盘**。
2. 读 `torch_utils.py:542-562` ⇒ EMA 公式 `0.9999·(1−e^{−t/2000})`，**每 optimizer step 一次**。
3. 读 `trainer.py:760-816, 355-395` ⇒ SGD+nesterov；**group0 = bias（`warmup_bias_lr=0.1`, wd=0）**；warmup 3ep / 600 iter。
4. 读 `head.py:143-144` ⇒ 分类 bias 初始化 `log(5/nc/(640/s)²)`。
5. 装载 **29 个 `epoch*.pt`**（CPU、`weights_only=False`、只读），提取 `epoch / updates / ema.state_dict / optimizer.state_dict`，**逐快照比对 `updates`**。
6. **三个 stride 分支**的 W/bias 轨迹 + 与理论 init 对照。
7. **momentum buffer** 的 common/centered 分解、与 W 的余弦、10-epoch 持续性。
8. **EMA 滤波权重**显式计算 + 对线性 raw ramp 的仿真。
9. 与 **P5.2 有效轨迹**（`from_snapshots` / `p5_2_temporal.csv`）对齐成统一时间轴。
10. **交叉验证**：probe 的 `head_W/head_b` 与 checkpoint 直读**逐位相同**（‖W_mean‖=0.1696、‖W_cent‖=1.9272、b_mean=−9.7331、b_std=0.0506）；
    `trajectory_metadata.json` 的 `updates_total` 与 checkpoint `updates` 一致。

**未做**：任何 forward / backward / 训练 / 推理 / 验证 / 源码或配置修改 / 新 checkpoint。
**未用**模拟值顶替缺失 artifact —— `raw W` 明确记 `UNMEASURABLE`。

---

## 最终回答（brief 的收尾问题）

> **我们到底有没有证据知道 `W_mean` +188% 是怎么来的？**

**部分有，但不足以支持实验。**

- **知道**：它不是 EMA 造的（EMA 近似恒等，§5）；不是 bias 造的（bias 是初始化常数且冻结，§3）；
  不是"共同模梯度更大"造的（梯度反而是判别性占优 3.5×，§7）；
  而是"**共同模梯度符号持续（26/29）而判别性梯度无方向（14/29）**"造的（§7）；
  方向在 **ep0→ep10 内设定后冻结**（0/28 反向，§6）。
- **不知道**：那个持续性符号**从哪来**（是 `target_scores_sum` 归一化？是背景 anchor 的系统性偏置？是 5 通道输入的共同漂移？）——
  本轮的 artifact 无法区分。
- **且时间顺序是反的**：W_mean 领先、logit/dead-rate 滞后，早期 W_mean 暴涨期间 dead rate **下降**（§10）。
  ⇒ 把 `W_mean ↑188%` 当成 F0 的成因，**与现有时间证据不符**。

**⇒ `MECHANISM STILL OPEN` / `GPU NO-GO`。不提出训练实验。**
