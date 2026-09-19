# RGBID P2 Detection Head Probe —— 报告（Design A，from-scratch 正式结果）

**日期**：2026-09-19
**状态**：✅ **Design A 已完整跑完 300/300 epoch（无早停），两个 checkpoint 均已按预注册口径评测。**
**判定**：**`P2_REJECTED`（无增益）—— 关闭 P2 方向，正式保底维持 `0.53180`。**
**正式保底全程冻结**：`best.pt` SHA256 `f9dddbfa…d83e55` / 线上 **0.53180** / 离线官方 **0.50928**（本轮已复核未变）。

---

## 0. 结论速览

| 项 | 结果 |
|---|---|
| 训练 | ✅ **300 / 300 epoch 完整跑完**，10.902 h，130.8 s/ep，**未触发早停** |
| 内部 val best | ep247，mAP50-95 **0.55856**（baseline ep237 = 0.55800，**+0.00056**） |
| **官方口径 mAP50-95** | **P2_best = 0.50232** vs baseline **0.50928** → **Δ = −0.00695** |
| 官方口径 P2_last | 0.48944 → Δ = −0.01983 |
| small / medium / large | **+0.00236 / +0.00724 / −0.01399** |
| 预注册判定（本轮 pre-flight §12.8：P2 ≤ 0.50928 → 关闭方向） | ✅ **触发 → 关闭 P2 方向** |
| 配对 bootstrap（400 次） | **四个指标（总 + S/M/L）95% CI 全部含 0** → **无一显著**；Δsmall `+0.00236 < 1σ(0.00266)` |
| **关键判读** | P2 落在 **baseline 自身 9 个 checkpoint 的均值上**（0.50232 vs 0.50248，−0.00016）；<br>−0.00695 的差距 ≈ **baseline 自身 checkpoint 散布的 max−mean = 0.00680** |
| 结论 | **P2 不带来增益，也未被证明有害** —— 属于**第三次同族中性结果** |

---

## 1. 实验目的

回答：**在当前 RGBID 5ch 早期融合 @1280 上，只增加 P2（stride 4）检测尺度，能否改善官方 mAP50-95（尤其 small）？**

**此前置反证（未变）**：P2 在本项目已试过一次 —— `e5_yolo11m_rgbd_p2`（RGBD mid-fusion + P2 @640）best **0.47796** vs `e4_yolo11m_rgbd` **0.47783**，**Δ = +0.00013（实质为零）**。

---

## 2. 实验设计：Design A 为什么是唯一干净的单变量

模型 `configs/yolo11m_rgbid_p2_probe.yaml` = `configs/yolo11m_earlyfusion.yaml` + P2 分支：

```
- [16, 1, nn.Upsample, [None, 2, "nearest"]]  # 23  -> 320x320
- [[-1, 2], 1, Concat, [1]]                    # 24  cat backbone P2(layer2) -> 512ch
- [-1, 2, C3k2, [256, False]]                  # 25  P2/4-xsmall
- [[25, 16, 19, 22], 1, Detect, [nc]]          # 26  Detect(P2,P3,P4,P5)
```

**实测**：顶层模块 27（baseline 24）；`Detect(nc=12, nl=4, no=76)`；**stride = [4,8,16,32]**；
Detect 输入 `[P2(1,76,320,320), P3(1,76,160,160), P4(1,76,80,80), P5(1,76,40,40)]`；`layer 0–22 索引逐字不变`。
参数量 **20,804,608**（本次从 checkpoint 实读复核，+3.70%）；FLOPs @1280 **422.69 G**（云端 fused 实测；baseline 273.85 G）。

训练配置 `configs/train_rgbid_p2_probe.yaml` 与 baseline `configs/train_rgbird_ir_quicktest.yaml` **恰好 3 处差异**：

```
name        urban_multimodal_det_yolo11_rgbird_ir_quicktest -> urban_multimodal_det_yolo11_rgbid_p2_probe
patience    80 -> 0                 （禁用早停，保证 300 epoch schedule 可比）
pretrained  yolo11m.pt -> diagnostic/p2_probe/p2_probe_init_scratch.pt
```

**其余全部逐字不变**：`dataset` · `imgsz=1280` · `epochs=300` · `lr0=0.005` · `optimizer=SGD(0.937, 5e-4)` ·
`lrf=0.01` · `cos_lr` · `warmup_epochs=3` · `box/cls/dfl=7.5/0.5/1.5` · `close_mosaic=10` · `batch=8` · `seed=42` · `nbs=64`。

**初始化等价性（pre-flight 已逐张量验证，见 附录 A）**：`p2_probe_init_scratch.pt` 就是 `yolo11m.pt` 的重索引版本 ——
backbone/neck 528 键与 baseline **逐位一致**；5ch stem 走**同一** `_transfer_rgb_pretrained`
（`stem[:3]` = 预训练 RGB，`stem[3]=stem[4]` = mean(RGB)）；Detect 按尺度重映射 115 张量（与 baseline 同构）。

---

## 3. 训练完成情况

```
300 epochs completed in 10.902 hours.
Results saved to runs/urban_multimodal_det_yolo11_rgbid_p2_probe
```

| 项 | baseline | P2 |
|---|---:|---:|
| 完成 epoch | 300 | **300** |
| 早停 | patience=80（未触发） | patience=0（**结构性禁用**） |
| 速度 | 127.2 s/ep | 130.8 s/ep（**1.03×**） |
| 内部 val best epoch | ep237 | **ep247** |

**内部 val 曲线（`results.csv`）**：

| ep | **1** | 50 | 100 | 150 | 200 | 237 | 247 | 270 | 300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline mAP50-95 | 0.22871 | 0.47184 | 0.51156 | 0.52916 | 0.53306 | **0.55800** | 0.54652 | 0.54491 | 0.55091 |
| **P2 mAP50-95** | 0.22494 | 0.47728 | 0.50144 | 0.51321 | 0.53865 | 0.54323 | **0.55856** | 0.54682 | 0.54557 |
| baseline mAP75 | 0.22447 | 0.49773 | 0.53795 | 0.55262 | 0.57582 | **0.61859** | 0.57807 | 0.56258 | 0.58738 |
| **P2 mAP75** | 0.21341 | 0.46064 | 0.50724 | 0.52489 | 0.54863 | 0.57481 | 0.57192 | 0.57303 | 0.56833 |

**平台期统计（ep200–300）**：

| | best fitness | ep200-300 mean | ep200-300 max | mAP75 mean | mAP50-95 mean |
|---|---:|---:|---:|---:|---:|
| baseline | **0.58371** (ep237) | 0.57417 | 0.58371 | **0.58932** | **0.54787** |
| P2 | **0.58425** (ep247) | 0.57121 | 0.58425 | 0.56976 | 0.54472 |

- **best fitness 几乎相同**（0.58425 vs 0.58371，**+0.00054**），但 **P2 的平台均值更低**（0.57121 vs 0.57417）；
- **最大差异在 mAP75**：P2 的 mAP75 曲线系统性低约 0.02，且**没有 baseline 那种偶发尖峰**
  （baseline 在 ep237=0.61859 / ep290=0.61738 出现尖峰，而 P2 最高仅 0.60366）；
- 两者 top-5 fitness epoch 都聚成紧密簇（baseline ep237/258/279/287/259；P2 ep247/252/253/248/251），
  说明**不存在"P2 还没收敛"的情况**。

---

## 4. 官方口径评测（按 pre-flight §12.7/§12.8 预注册执行）

评测链与 baseline **逐字同参**：
`predict_rect.py --mode rect --imgsz 1280 --conf 0.001 --iou 0.7 --max_det 300 --max_boxes 300`
→ `official_map.py --split_root data/processed/rgbid_split`（官方口径 = mean + tail0 + conf 匹配）。
尺寸分桶与既有实验一致（`32²/96²`，`diagnostic/p3attn_probe/_sml_eval.py`）。

| 实验 | epoch | mAP50 | mAP75 | **mAP50-95** | P | R | 预测框数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline（正式 `best.pt`）** | 237 | **0.77601** | **0.56386** | **0.50928** | **0.8578** | 0.7029 | **4824** |
| **P2 `best.pt`** | 247 | 0.76710 | 0.50582 | **0.50232** | 0.8213 | 0.7075 | 4471 |
| **P2 `last.pt`** | 300 | 0.75584 | 0.50561 | **0.48944** | 0.8260 | 0.7036 | 4604 |

| size | Baseline | P2 `best.pt` | **Δ** | P2 `last.pt` | Δ |
|---|---:|---:|---:|---:|---:|
| small | 0.05625 | **0.05861** | **+0.00236** | 0.05891 | +0.00266 |
| medium | 0.17984 | **0.18708** | **+0.00724** | 0.17648 | −0.00336 |
| large | **0.61711** | 0.60312 | **−0.01399** | 0.59473 | −0.02238 |

| | ΔmAP50 | ΔmAP75 | **ΔmAP50-95** |
|---|---:|---:|---:|
| P2 `best.pt` | −0.00891 | −0.05804 | **−0.00695** |
| P2 `last.pt` | −0.02017 | −0.05824 | −0.01983 |

**读数**：
- 两个 checkpoint **都低于 baseline**；P2 自身 `last.pt` 比 `best.pt` 低 **0.01288**（0.48944 vs 0.50232）；
  （注：baseline 的 `last.pt` 未在这条官方链上评过，故**不**做跨模型的 last-drop 比较。）
- 预测框数**下降**（4471 / 4604 vs 4824）：新增 stride-4 头**没有**带来更多框，反而更少；
  精度 P 明显下降（0.8213 vs 0.8578），召回 R 基本持平（0.7075 vs 0.7029）；
- 尺寸分解显示 P2 是**"large 换 medium"**：medium **+0.00724**、large **−0.01399**、small 微升 **+0.00236**。

---

## 5. 关键判读：0.00695 的差距来自哪里

**这一步决定了结论的措辞 —— 从"P2 有害"改为"P2 无增益"。**

仓库里已有 baseline **同一模型** 在 ep200/230/240/250/260/270/280/290 的官方评测记录
（`diagnostic/checkpoint_selection_audit/_ep*.eval.txt`，与本次**同一条评测链**）：

| checkpoint | ep200 | ep230 | **ep237（正式）** | ep240 | ep250 | ep260 | ep270 | ep280 | ep290 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official mAP50-95 | 0.49714 | 0.50369 | **0.50928** | 0.50097 | 0.49858 | 0.50146 | 0.49978 | 0.50396 | 0.50742 |
| official mAP75 | 0.52266 | 0.55612 | **0.56386** | 0.50724 | 0.50090 | 0.55457 | 0.51150 | 0.54721 | 0.54549 |

**同一个 baseline 模型，仅换 checkpoint，官方 mAP50-95 就在 0.49714–0.50928 之间波动**
：`mean = 0.50248`、`sd = 0.00401`、`max − mean = 0.00680`。

> **P2_best = 0.50232，与 baseline 自身 9 个 checkpoint 的均值 0.50248 相差 −0.00016。**
> 即：**P2 模型恰好落在 baseline 的"典型 checkpoint"水平上**；
> 而作为比较基准的 0.50928 是**同一 baseline 在这 9 个里的最大值**。
> 观测到的 −0.00695 ≈ baseline 自身 checkpoint 散布的 max−mean（0.00680）。

**采样边界（必须写清楚）**：这 9 个点只覆盖 ep200–290 的 9/91 个 epoch，
所以 0.50928 **不是** baseline 的真实峰值上界 —— 更强的 baseline checkpoint 可能存在。
这不削弱结论：结论用的是**散布量级**（sd 0.004 / max−mean 0.007），
而散布的估计对"还有没有更高点"不敏感。
（同理，P2 也只评了 2 个 checkpoint，它自己的峰值上界同样未知 —— 两边对称。）

**因此不能说"P2 使模型变差"**。准确表述是：

- P2 的 best.pt 官方成绩 = baseline 的**典型** checkpoint 成绩；
- 它**没有超过 baseline 的峰值**，而判据要求超过峰值才能继续；
- mAP75 的 −0.058 里含有**选择运气**成分（baseline 的 ep237 恰是它自己的 mAP75 尖峰）。

**另外两个不能忽略的观察**：
1. **large AP 在两个 checkpoint 上同向下降**（−0.01399 / −0.02238），且内部 val mAP75 在整条曲线上
   系统性低 ~0.02 —— 这是**方向一致**的，不像纯噪声；
   但 §6 的 bootstrap 显示其 Δ 的 95% CI 仍含 0（boot mean −0.00823），
   所以**只能说"有下降的迹象"，不能说"已证实 large 退化"**。
2. **small AP 三次同向微升**（随机 P2 分支 +0.00102、本次 +0.00236 / +0.00266）
   —— 同一方向，但**幅度低于噪声**（§6：Δsmall = +0.00236 < 1σ = 0.00266）。

---

## 6. 不确定性：图像级配对 bootstrap

预注册判定规则本身不要求这一步，但它直接回答本探针的科学命题
「**P2 是否真的改善小目标**」。做法：**配对**重采样（同一次抽到的图像集合同时用于两个模型，
消除图像难度方差），官方口径 + 相同尺寸分桶，400 次迭代。

**正确性自检**：快路径在全量（不重采样）下与 `official_map.evaluate` **逐位一致**
（Baseline 0.509275 / P2 0.502321，`exact=True`），并精确复现 `_sml_eval.py` 的 S/M/L 值。
脚本：`diagnostic/p2_probe/_bootstrap_delta.py`（`--verify` 可复跑自检）。

**结果（400 次，seed=42）**：

| 指标 | point Δ | boot mean | 2.5% | 97.5% | **P(Δ ≤ 0)** | σ(Δ) |
|---|---:|---:|---:|---:|---:|---:|
| **mAP50-95** | −0.00695 | −0.00349 | **−0.01264** | **+0.00555** | 0.752 | 0.00485 |
| **small** | +0.00236 | +0.00162 | **−0.00341** | **+0.00643** | 0.275 | 0.00266 |
| **medium** | +0.00724 | +0.00435 | **−0.00379** | **+0.01349** | 0.142 | 0.00440 |
| **large** | −0.01399 | −0.00823 | **−0.02198** | **+0.00505** | 0.865 | 0.00700 |

**尺度参考（单模型重采样的 σ，即指标本身对图像抽样的敏感度）**：
`mAP50-95 = 0.02461`、`small = 0.00631`、`medium = 0.01175`、`large = 0.03721`。
配对后 Δ 的 σ（0.00485）比单模型 σ（0.02461）**小 5 倍** —— 说明配对确实去掉了图像难度方差，
剩下的就是模型差异本身。

**读数**：

1. **四个指标（总 + 三个尺寸桶）的 95% CI 全部包含 0** → 在图像抽样意义上，
   **P2 与 baseline 的差异无一达到统计显著**。
2. **Δsmall = +0.00236 < 1σ(0.00266)** —— "P2 改善小目标"**没有得到支持**。
   （§4 里 small 的两次同向微升 +0.00236 / +0.00266，合起来看是**方向一致但幅度低于噪声**。）
3. **Δmedium = +0.00724 的 95% CI 下界 −0.00379 仍含 0** —— 不能声称 medium 有改善。
   （用 20 次迭代时该 CI 曾一度不含 0；扩到 400 次后消失 —— 那是**小样本分位数的假象**，
   记录在此以免后人误引。）
4. ⚠️ **一个必须标出的系统性现象：四个指标的 boot mean 都弱于 point Δ、一致向 0 收缩**
   （总 −0.00695→−0.00349、small +0.00236→+0.00162、medium +0.00724→+0.00435、large −0.01399→−0.00823）。
   **我没有把它归因清楚**，只记录两种候选解释：
   (a) AP 是 pooled 的 rank-based 统计量，对重采样**不是无偏的**，bootstrap 均值本身有衰减
   （已知性质），因此 **boot mean 不应被当作"修正后的 point 估计"**；
   (b) point Δ 恰好落在重采样分布的不利尾部。
   无论 (a) 还是 (b)，**要读的是百分位 CI，而四个 CI 都含 0** —— 判定不变。
   若要严格区分，需要的是**独立验证集**（本项目没有），重采样本身无法分辨。

> **诚实的边界**：bootstrap 的 CI 只覆盖「**图像抽样**」这一个方差来源，
> **不覆盖**「checkpoint 选择」这个方差来源（那由 §5 的 9 个 checkpoint 散布单独刻画，sd=0.00401）。
> 把两者放在一起，本节结论与 §5 完全一致：**差距处于噪声量级，P2 无增益**。

---

## 7. 最终判定

```text
P2_REJECTED —— 无增益（no gain），非有害（not harmful）
```

**依据（双轨，方向一致）**：

1. **预注册规则（pre-flight §12.8）**：P2 official mAP50-95 = **0.50232 ≤ 0.50928**
   → 规则明文规定"**关闭 P2 方向**"。`last.pt` 更低（0.48944），不构成翻案依据。
2. **不确定性分析**：配对 bootstrap 的 Δ(总) 95% CI **包含 0**
   → **没有统计学证据表明 P2 优于 baseline**。

**方向关闭的具体含义**：
- 不再重跑、不再调 P2（不试 lr0、patience、混 init 等任何变体）；
- 正式提交方案**维持 `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt`**，
  线上 **0.53180** / 离线官方 **0.50928**，SHA256 `f9dddbfa…d83e55`（本轮复核未变）。

**必须如实记录的边界**：
- 本结论**只否证"P2 带来增益"**，**不**证明"P2 结构本身有缺陷"——
  受限于 §5，−0.00695 与 baseline 自身 checkpoint 散布同量级；
- **小目标方向至此 3 次同族中性**：E5 @640 = +0.00013、1536 = small 未改善、本次 P2 = Δsmall +0.00236（< 1σ）。
  small AP 的绝对值仍是本任务的硬瓶颈（**0.056**，与 large 的 0.617 相差一个数量级），
  但**"加检测尺度"这条路径已被三次独立实验穷尽**，后续若继续攻小目标需换机制（如裁剪/超分/损失重加权），
  而不是再加尺度或加容量（容量杠杆亦已在 F5 证伪）。

---

## 7.1 建议

**建议：就此冻结 `0.53180`，把剩余时间（至截止 09-20 20:00 约 32 h）全部投入提交物打包与最终核对，不再开新实验。**

理由：

1. 本轮是**最后一个"证据对口"的候选**（误差归因指向小尺度框未被生成，P2 直接在该尺度生成框），
   它已被干净地否证（单变量、双轨结论一致）；
2. 剩余未验证的杠杆（容量 F5、集成 官方明令禁止、conf=0.0001 线上实测 −2.729、1536 纯推理 −0.0053）
   均已单独证伪，没有现成的"下一个"；
3. 新开任何实验都意味着**再次承担 pre-flight 判断失误的风险**（附录 B 已发生一次，代价 3 h），
   而收益上界只能是与 baseline 同量级的噪声内波动。

**如果仍要继续做实验**，唯一有信息量、且**成本可控**的形式是：
在**已有 checkpoint** 上做零训练量的推理侧单变量（如多尺度 TTA 的官方口径复算），
且必须**先用官方口径本地复现**再考虑上线 —— 这条纪律来自 `conf=0.0001` 的教训
（本地官方口径 +0.0049，线上 −2.729）。

---

## 8. 合规状态

| 项 | 状态 |
|---|---|
| 正式 `best.pt` SHA256 | ✅ `f9dddbfaca4cb08bde31d082eb0a54d499b8441eb1f97c6134a028caa6d83e55` —— **与记录一致，未变** |
| 正式 `last.pt` SHA256 | ✅ `cfa00836314e5e5eb87b35ff7dd6fc42d2e0291f81c5ffc7384957b51e88004f` |
| 正式 run 目录 | ✅ 未触碰 |
| 源码 / 正式配置 | ✅ 本轮未修改（`scripts/*` 与 `ultralytics/*` 的改动为本会话之前既有状态） |
| 提交 | ❌ 未生成 submission |
| 新增 | `runs/urban_multimodal_det_yolo11_rgbid_p2_probe/`（独立目录）、`diagnostic/p2_probe/p2_scratch_best|_last/`（评测预测）、`diagnostic/p2_probe/_bootstrap_delta.py`、本报告 |

⚠️ **一处数据处置需记录**：Design A 训练**覆盖了首次运行（81 epoch 早停）的同名 run 目录**。
首次运行的 `results.csv` / `results.png` / `weights/` 已不存在。其结论性产出**未丢失**
（内部曲线表见 附录 B；其 best.pt=ep1 的官方评测预测仍在 `diagnostic/p2_probe/p2_best_full/`，数字见 附录 B）。

---

## 9. 最终结果块

```text
RGBID P2 PROBE RESULT (Design A, from-scratch, 300/300 epochs)

Formal (frozen):
mAP50 = 0.77601   mAP75 = 0.56386   mAP50-95 = 0.50928   online = 0.53180

P2 best.pt (epoch 247):
mAP50 = 0.76710   mAP75 = 0.50582   mAP50-95 = 0.50232   preds = 4471

P2 last.pt (epoch 300):
mAP50 = 0.75584   mAP75 = 0.50561   mAP50-95 = 0.48944   preds = 4604

Delta best.pt vs Formal:
mAP50     = -0.00891
mAP75     = -0.05804
mAP50-95  = -0.00695
small     = +0.00236
medium    = +0.00724
large     = -0.01399

Internals (best epoch):
Formal  ep237  mAP50=0.81513  mAP75=0.61859  mAP50-95=0.55800
P2      ep247  mAP50=0.81550  mAP75=0.57192  mAP50-95=0.55856

Verdict:
P2_REJECTED (no gain, not harmful)

Reason:
1) Pre-registered rule (pre-flight §12.8): 0.50232 <= 0.50928  -> close the P2 direction.
2) Paired bootstrap: Delta(mAP50-95) 95% CI includes 0 -> no evidence of gain.
3) Calibration: P2_best (0.50232) equals baseline's own 9-checkpoint mean (0.50248);
   the -0.00695 shortfall ~= baseline's own checkpoint scatter (max-mean = 0.00680).
   => The gap is checkpoint-selection scale, NOT proof that the P2 architecture is faulty.

Action:
Freeze the formal submission at runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt
(SHA256 f9dddbfa...d83e55, online 0.53180). Do not pursue P2 further.
Small-object direction: exhausted via "add a detection scale" (E5 +0.00013, 1536 no gain, P2 -0.00695).
```

---

# 附录 A —— Design A Pre-flight 记录（已完成并通过）

## A.1 逐层初始化等价性（逐张量 `torch.equal`，非 key 计数推断）

从 **`yolo11m.pt`** 分别构建 baseline 与 P2，对**共同存在的** backbone/neck 参数逐张量比对：

```
键数: baseline=649  P2=743
共同键 = 528    逐位一致 = 528    不一致 = 0     -> True
layer 0-22（backbone + 原 FPN/PAN neck）共同键 = 528，逐位一致 = True
仅 baseline 有 = 121 {layer 23: 121}        <- 旧的 3 尺度 Detect（索引位移）
仅 P2 有       = 215 {layer 25: 54, layer 26: 161}   <- 新增 P2 neck + 新 4 尺度 Detect
```

**5ch stem 逐通道验证（两个模型都必须满足）**：

| 模型 | `stem[:3]` == 预训练 RGB | `stem[3]` == mean(RGB) | `stem[4]` == mean(RGB) | IR == Depth |
|---|---|---|---|---|
| baseline | ✅ True | ✅ True | ✅ True | ✅ True |
| P2 | ✅ True | ✅ True | ✅ True | ✅ True |

## A.2 修正了一处对 P2 不利的初始化不对称

**发现**：若 P2 直接使用 `pretrained: yolo11m.pt`，因 Detect 索引 **23 → 26** 位移，
其 Detect 的 161 个键**全部按名不匹配**：

| 从 `yolo11m.pt` 构建 | Detect 键 | 获得 COCO 预训练值 | 随机 |
|---|---:|---:|---:|
| baseline | 121 | **115** | 6（仅 nc 相关 `cv3.{0,1,2}.2`） |
| P2（未修正） | 161 | **0** | **161** |

→ P2 **少获得 861,894 个 COCO 预训练的框回归塔（cv2）参数**，这是**对 P2 不利**的不对称。

**修正**：checkpoint 手术（`diagnostic/p2_probe/_make_init_scratch.py`）把 `yolo11m.pt` 的 Detect
**按检测尺度重映射**到新的 index 26：

```
Detect 尺度重映射: 成功 115 张量, 形状不符跳过 6
   skip model.26.cv3.{1,2,3}.2.{weight,bias}   目标(12,256,1,1) vs 源(80,256,1,1)   <- nc 不同，与 baseline 的随机集完全一致
```

修正后 baseline 与 P2 的 Detect 初始化来源**同构**：
`remap 115（P3/P4/P5 的 cv2+cv3-tower+dfl） / 新增 40（仅 P2 尺度分支，随机） / nc 随机 6`。

## A.3 Early stopping 关闭（源码实读）

`ultralytics/utils/torch_utils.py:762`：
```python
self.patience = patience or float("inf")   # patience=0 -> inf -> 永不触发
stop = delta >= self.patience
```
→ **`patience: 0` 语义 = 禁用早停**。本次训练确实跑满 300 epoch（见 §3）。

## A.4 one-batch sanity check（CPU 实测，全部通过）

```
forward OK: 4 个尺度输出 [(2,76,64,64),(2,76,32,32),(2,76,16,16),(2,76,8,8)]
loss = 23.8875  finite = True   items = [3.7531, 4.5024, 3.6883]
backward + optimizer.step OK   梯度总和 = 2,525,584.41   非零 = True
val forward OK   out = (1,16,5440)   NaN = 0
```

**正式 imgsz=1280 的形状打印（hook 实读）**：

```
P2 = [B, 76, 320, 320]
P3 = [B, 76, 160, 160]
P4 = [B, 76,  80,  80]
P5 = [B, 76,  40,  40]
Detect stride = [4.0, 8.0, 16.0, 32.0]
nc = 12   nl = 4   no = 76
输出 = (1, 16, 136000)   其中 136000 == 320²+160²+80²+40²  ✔
```

---

# 附录 B —— 首次运行（Design A 之前）的失败记录

**这一节保留，因为它是一个有教学价值的失败，且责任在我。**

## B.1 实际结果：81 / 300，被早停

```
81 epochs completed in 3.068 hours.
EarlyStopping: Training stopped early as no improvement observed in last 80 epochs.
               Best results observed at epoch 1, best model saved as best.pt.
```

| ep | **1** | 10 | 25 | 50 | 81 |
|---|---:|---:|---:|---:|---:|
| mAP50-95 | **0.54787** | 0.50173 | 0.49673 | 0.50128 | 0.52368 |
| mAP50 | 0.80622 | 0.78688 | 0.77084 | 0.76395 | — |
| mAP75 | 0.60262 | 0.49346 | 0.52825 | 0.53534 | — |

- **best epoch = 1**（warmup 的极低 LR 就是最高点）；速度 136.3 s/epoch；3.07 h 后 `patience=80` 触发早停。

## B.2 根因：初始化点与学习率的组合

```
起点 = p2_probe_init.pt  = 【已经收敛的 RGBID 模型】 + 随机 P2 分支
lr0  = 0.005             = 【从零训练】的学习率
```

以从零训练的 LR 重新加热一个已收敛的权重 → 第 1 个 epoch 就是最高点 → 之后 LR 升到 5e-3 迅速破坏 →
80 个 epoch 不足以恢复 → 早停。**该配置在数学上不可能跑出与 baseline 可比的结果**
（baseline 自身 best 出现在 ep237）。

## B.3 这是我的判断失误

我在 1536 实验里**已经识别过同型的失败模式**，但在 P2 的 pre-flight 报告里，
把 `lr0=0.005` 只列为"保持不变"，**没有对"收敛起点 + 从零 LR"这一组合提出预警**，
也没有使用用户给出的逃生口（"如果发现其他变量必须改变，请先停止并报告原因"）。
**导致浪费了一次 3 小时的运行。**

## B.4 首次运行的评测数字（**不得作为 P2 的成绩引用**）

| 实验 | mAP50 | mAP75 | mAP50-95 | small | medium | large |
|---|---:|---:|---:|---:|---:|---:|
| Baseline（正式） | 0.77601 | 0.56386 | **0.50928** | 0.05625 | 0.17984 | 0.61711 |
| 首次运行（`best.pt` = **ep1**） | 0.76410 | 0.54503 | 0.49679 | 0.05726 | 0.17177 | 0.61132 |

该 `best.pt` 只是「**收敛 baseline + 随机 P2 分支 + 1 个 warmup epoch**」，
**不是**"训练好的 P2 模型"，因此**不构成对"P2 检测头是否有效"的检验**。
（唯一中性观察：**即使 P2 分支是随机的，small AP 仍 +0.00102** —— 与本次 +0.00236 同向。）
