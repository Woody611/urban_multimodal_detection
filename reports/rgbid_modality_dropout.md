# RGBID + Modality Dropout 实验报告（终版）

**日期**：2026-09-18
**状态**：**`MODALITY_DROPOUT_REJECTED`**
**性质**：单变量训练实验（仅新增 Modality Dropout），已完成 300 epoch 训练 + 官方口径全量评测 + 模态消融 + checkpoint 审计。
**未提交线上。**

---

## 0. 结论速览

| 项 | 值 |
|---|---|
| 实验目录 | `runs/urban_multimodal_det_yolo11_rgbid_modality_dropout/` |
| best.pt | 40,635,557 B，SHA256 `30560164f53840310d0a05f8efe784f6f538dcc4cffb70663d73bbb12e982396`（对应 **ep262**） |
| **官方 mAP50-95** | **0.49029** |
| baseline 官方 mAP50-95 | **0.50928** |
| **Δ（官方）** | **−0.01899** |
| 内部 mAP50-95 | 0.54697 vs baseline 0.55800（Δ −0.01103） |
| 线上 | ❌ **未提交** |
| 判定 | **`MODALITY_DROPOUT_REJECTED`** |

> **该方案在官方口径下全面退化**（mAP50 −0.01989 / mAP75 −0.09917 / mAP50-95 −0.01899）。
> 主要重灾区是 **官方 mAP75**。**不提交线上、不作为后续模型基础。**
> **本结论不解释为「IR / Depth 没有价值」** —— 它只说明
> **「在当前 RGBID 模型上做输入级整模态随机清空」这一策略不适合**。原因分解见 §5–§6。

---

## 1. 口径声明（务必先读）

本报告严格区分五类数字，**不得互相替代**：

| # | 口径 | 来源 | 是否可用于决策 |
|---|---|---|---|
| ① | **Ultralytics 内部验证** | 训练期 `results.csv` 的 `metrics/*` | ⚠️ 仅作趋势参考 |
| ② | **官方赛题离线口径** | `scripts/official_map.py`（mean + tail0 + conf 匹配） | ✅ **本报告主判据** |
| ③ | **当前正式线上成绩** | 用户上传后官方返回 | ✅ 最终事实 |
| ④ | 模态消融（推理时人为 drop） | 本报告 §4 | ⚠️ 仅分析用，**不得提交** |
| ⑤ | fork 口径（trapz + ramp + IoU 匹配） | `official_map.py` 同脚本 | ⚠️ 仅作对照 |

**当前正式线上保底（不变）**：

```
RGBID Early Fusion  (runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt)
online mAP50-95 = 0.53180
```

> 本会话已两次证明**本地离线口径不足以预测线上**（conf=0.0001：本地官方 +0.00488 → 线上 **−2.729** 分）。
> 故本报告**只做离线判定，不据此推断线上**，也**不自动生成任何线上候选包**。

---

## 2. 权重身份确认（§1 要求）

**确认：本次评估权重确实来自「RGBID + Modality Dropout」，不是旧 baseline。**

| 项 | 值 |
|---|---|
| 实验目录 | `runs/urban_multimodal_det_yolo11_rgbid_modality_dropout/` |
| best.pt 路径 | `runs/urban_multimodal_det_yolo11_rgbid_modality_dropout/weights/best.pt` |
| best.pt SHA256 | `30560164f53840310d0a05f8efe784f6f538dcc4cffb70663d73bbb12e982396` |
| last.pt SHA256 | `5d799210bcb314241963bb74c3c53dff455e6cc0ff423d1b1bfad6ce247fca3f` |
| 训练配置 | `configs/train_rgbid_modality_dropout.yaml` |
| 模型配置 | `configs/yolo11m_earlyfusion.yaml` |
| 完成 epoch | **300 / 300**（`results.csv` 300 行；best = ep262） |
| 实际 dropout 配置 | `enabled=True, ir_drop_prob=0.075, depth_drop_prob=0.075, mutually_exclusive=True, apply_train_only=True, ir_fill=125, depth_fill=0` |

**对照旧 baseline**：

| | 大小 (B) | SHA256 |
|---|---:|---|
| Modality Dropout best.pt | 40,635,557 | `30560164…e982396` |
| **RGBID baseline best.pt** | 40,635,237 | `f9dddbfaca4cb08bde31d082eb0a54d499b8441eb1f97c6134a028caa6d83e55` |

两者 SHA256 与文件大小均不同 → **确为两个不同权重**。
且 baseline 的 SHA256 与冻结记录 `f9dddbfa…d83e55` **逐位一致** → baseline 未被覆盖。

**dropout 确实在训练中生效的证据链**：`args.yaml` 实读含上述 `modality_dropout` 块；
而 `ultralytics/data/build.py:104` 把 `self.args` 整个作为 `hyp` 传给 dataset
（`train.py:143 → build_yolo_dataset(self.args, ...)`），故 dataset 必然读到该配置。

---

## 3. 官方赛题评测（§2 要求）

评测脚本 `scripts/official_map.py`，参数 `--split_root data/processed/rgbid_split`，
官方口径 = 101 点算术平均 + 尾部取 0 + conf 顺序贪心匹配。**与 baseline 记录值同脚本同参数。**

> 回归校验：先用 baseline `best.pt` 跑同一脚本，得到 **0.77601 / 0.56386 / 0.50928**，
> 与历史记录**逐位一致**后，才用于新模型。

| 指标 | baseline | Modality Dropout | Δ |
|---|---:|---:|---:|
| **official mAP50** | 0.77601 | **0.75612** | **−0.01989** |
| **official mAP75** | 0.56386 | **0.46469** | **−0.09917** |
| **official mAP50-95** | **0.50928** | **0.49029** | **−0.01899** |
| micro P（IoU0.5, F1 工作点） | 0.85783 | 0.84427 | −0.01356 |
| micro R（同上） | 0.70289 | 0.71072 | +0.00783 |
| R@0.5 small / medium / large | 0.67116 / 0.81364 / 0.86470 | 0.66577 / 0.81970 / 0.84409 | −0.00539 / +0.00606 / −0.02061 |
| R@0.5:0.95 small / medium / large | 0.38248 / 0.50871 / 0.65134 | 0.37871 / 0.51735 / 0.64211 | −0.00377 / +0.00864 / −0.00923 |
| 预测框总数 | 4,824 | 4,344 | −480 |

```
delta_official = 0.49029 − 0.50928 = −0.01899
```

**判定：`official mAP50-95 = 0.49029 ≤ 0.50928` → `MODALITY_DROPOUT_REJECTED`。**

### 3.1 Ultralytics 内部口径（仅对照）

| | best ep | mAP50 | mAP75 | mAP50-95 | P | R |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 237 | 0.81513 | 0.61859 | 0.55800 | 0.83856 | 0.74785 |
| Modality Dropout | 262 | 0.80580 | 0.54218 | 0.54697 | 0.86772 | 0.71931 |
| Δ | | −0.00933 | −0.07641 | −0.01103 | +0.02916 | −0.02854 |

**两个口径方向一致（均为负），不存在「内部正 / 官方负」的分歧。**
且两个口径都指向同一形态：**precision 升、recall 降，高 IoU 指标恶化最重**。

---

## 4. 模态作用分析（§3 要求）

**只改变推理输入模态，不重训、不改权重、不改验证集、不改评测参数。**
填充值与训练期一致：IR → 125，Depth → 0，shape 恒为 `[B,5,H,W]`。
实现：`scripts/modality_ablation_infer.py`（新增，复用冻结的
`predict_rect._preprocess_rect` 与 `predict._to_chw/_format_lines`，未修改任何冻结文件）。
**该三组结果仅供分析，不得用于比赛提交。**

| 条件 | mAP50 | mAP75 | **mAP50-95** | P | R | R@0.5 S/M/L | R@0.5:0.95 S/M/L |
|---|---:|---:|---:|---:|---:|---|---|
| **Base** Full | 0.77601 | 0.56386 | **0.50928** | 0.8578 | 0.7029 | .6712/.8136/.8647 | .3825/.5087/.6513 |
| **Base** IR-drop | 0.77385 | 0.55128 | **0.49990** | 0.8294 | 0.7154 | .6631/.8152/.8629 | .3771/.5070/.6454 |
| **Base** Depth-drop | 0.77081 | 0.55179 | **0.49753** | 0.8209 | 0.7154 | .6792/.8061/.8548 | .3841/.5024/.6416 |
| **New** Full | 0.75612 | 0.46469 | **0.49029** | 0.8443 | 0.7107 | .6658/.8197/.8441 | .3787/.5173/.6421 |
| **New** IR-drop | 0.75282 | 0.47209 | **0.48839** | 0.8301 | 0.7207 | .6604/.8174/.8468 | .3776/.5160/.6404 |
| **New** Depth-drop | 0.75575 | 0.47323 | **0.49039** | 0.8181 | 0.7292 | .6685/.8197/.8423 | .3819/.5166/.6423 |

### 4.1 模态依赖（Full − 单模态清空）

| | Full − IR-drop | 相对 | Full − Depth-drop | 相对 |
|---|---:|---:|---:|---:|
| **baseline** | **+0.00937** | 1.84% | **+0.01175** | 2.31% |
| **Modality Dropout** | **+0.00190** | 0.39% | **−0.00010** | −0.02% |

**机制确实生效了**：dropout 训练把 IR 依赖压到约 1/5，把 Depth 依赖压到 0。
模型不再依赖这两个模态 —— **但代价是它在 Full 条件下也变弱了**。

### 4.2 一个刺眼的对照

```
baseline 在 Depth 被整通道清空时 : 0.49753
Modality Dropout 在完整输入时    : 0.49029
                         差值    : +0.00723
```

**baseline 被砍掉一个模态后，仍然优于 Modality Dropout 用全部模态的结果。**
说明 dropout 带来的不是「鲁棒」，而是「整体能力下降 + 不再使用被 drop 的模态」。

### 4.3 如何解读（避免过度推论）

- 本消融测的是**训练好的模型在某个通道被中性化后的行为**，是**行为学探针**，
  不等于证明了模型「因果地使用了该模态」。
- IR 的填充值 125 是 IR 通道均值量级（实测 train IR 均值 0.459 / val 0.498，125→0.490），
  Depth 的 0 本就有 31–37% 的像素占比 —— **两个填充值都在分布内**，
  因此「掉分」更可能来自**信息缺失**，而非**分布外异常输入**。
- **本结果不支持「IR 无价值」**：baseline 对 IR 的依赖是正的（+0.0094），
  IR 在 baseline 中确实被使用。只是**当前这种清空式 dropout 训练没能把它变成增益**。

---

## 5. 逐类分析（§4 要求）

官方口径 AP50-95，Full 条件，按 GT 框数升序：

| 类 | GT 框 | 占 GT | GT 图 | AP50-95 (Base) | AP50-95 (New) | **Δ** |
|---|---:|---:|---:|---:|---:|---:|
| tricycle | **2** | 0.1% | 1 | 0.8020 | 0.7020 | **−0.1000** |
| ball | **19** | 0.7% | 17 | 0.4139 | 0.3504 | **−0.0634** |
| boat | 28 | 1.0% | 14 | 0.4863 | 0.5004 | **+0.0140** |
| uav | **30** | 1.1% | 23 | 0.5731 | 0.5229 | **−0.0502** |
| garbage_can | **57** | 2.0% | 46 | 0.5338 | 0.4828 | **−0.0510** |
| bicycle | 106 | 3.8% | 43 | 0.3806 | 0.3917 | +0.0111 |
| seat | 107 | 3.8% | 51 | 0.6059 | 0.6198 | +0.0139 |
| sign | 150 | 5.3% | 83 | 0.3784 | 0.3696 | −0.0088 |
| light | 244 | 8.7% | 85 | 0.4950 | 0.4898 | −0.0053 |
| car | 288 | 10.3% | 69 | 0.5107 | 0.5080 | −0.0028 |
| animal | 731 | 26.0% | 184 | 0.4921 | 0.4917 | −0.0003 |
| person | 1045 | 37.2% | 219 | 0.4394 | 0.4544 | **+0.0150** |
| **合计** | 2807 | 100% | — | 0.50928 | 0.49029 | −0.01899 |

### 对照提问项

| 提问 | 结论 |
|---|---|
| **mAP50 上升但 mAP50-95 下降？** | ❌ **不成立**。两个口径**全面下降**：官方 mAP50 **−0.01989**、官方 mAP50-95 **−0.01899**。 |
| **person** | **+0.0150**（上升，且 person 占 37.2% 标注） |
| **sign** | **−0.0088**（小幅下降） |
| **bicycle** | **+0.0111**（上升） |
| **ball** | **−0.0634**（显著下降，但只有 19 个 GT 框） |
| **small recall 下降？** | **小幅下降**。R@0.5 −0.0054、R@0.5:0.95 −0.0038；medium 反而 +0.006~+0.009，large −0.009~−0.021。 |
| **Depth dropped 明显下降？** | 对 **New**：**几乎没有**（Full−Depth = −0.0001）。对 baseline：**+0.0118**。 |
| **IR dropped 明显下降？** | 对 **New**：很小（+0.0019）。对 baseline：**+0.0094**。 |

**关键观察**：损失几乎全部集中在**最稀有的 5 个类**（tricycle / ball / uav / garbage_can，合计占 GT 标注 **3.9%**），
而占 **96.1%** 标注的其余类（person / animal / car / light / sign / seat / bicycle / boat）
**合计是持平甚至上升的**。官方口径对 12 个类**等权平均**，
因此 **2 个 GT 框的 tricycle（0.1%）与 1045 个框的 person（37.2%）在总分里权重完全相同**
（各 1/12 = 8.3%）。

---

## 6. 「mAP75 崩了 0.0992」的机制核查（重要更正）

按 IoU 阈值拆开官方口径（12 类等权平均）：

| IoU | Base | New | Δ |
|---:|---:|---:|---:|
| 0.50 | 0.77601 | 0.75612 | −0.01989 |
| 0.55 | 0.74545 | 0.73241 | −0.01303 |
| 0.60 | 0.71076 | 0.70064 | −0.01012 |
| 0.65 | 0.68350 | 0.67142 | −0.01208 |
| 0.70 | 0.64081 | 0.62162 | −0.01919 |
| **0.75** | 0.56386 | 0.46469 | **−0.09917** |
| 0.80 | 0.39263 | 0.38542 | −0.00721 |
| 0.85 | 0.29150 | 0.30538 | **+0.01388** |
| 0.90 | 0.19175 | 0.20697 | **+0.01522** |
| 0.95 | 0.09647 | 0.05824 | −0.03823 |

0.75 处孤立下陷 −0.0992，而邻居只有 −0.019 / −0.007 —— **这不是平滑的定位退化形态**。
逐类 × 逐阈值拆解后，机制明确：

| 类 | Base 的 AP 台阶 | New 的 AP 台阶 | 0.75 处 Δ | 对总 Δ 的贡献 |
|---|---|---|---|---:|
| **tricycle**（2 个 GT 框） | 1.0000 → 0.5050 的台阶落在 **IoU 0.80** | 台阶提前到 **IoU 0.75** | −0.4950 | **−0.0413** |
| **uav**（30 个 GT 框） | IoU0.75 = 0.7323 | IoU0.75 = **0.4070** | −0.3253 | **−0.0271** |
| 其余 10 类合计 | — | — | — | **−0.0308** |

**两个类合计贡献了 0.75 处 −0.0992 的 69%。**

> **修正后的结论**：官方 **mAP75 −0.0992 并非「高 IoU 定位质量全面崩坏」**，
> 而是**极稀有类（tricycle 仅 2 个 GT 框、uav 30 个框）在等权平均下的一次「台阶错位」** ——
> 由于样本极少，这两类的 AP 是**离散台阶函数**，
> 模型只要多/少检出一个框，台阶就会整体挪动一个 0.05 的 IoU 档位，产生 0.33–0.50 的单类跳变。
> 同理，IoU=0.95 处的 −0.0382 也主要由 tricycle 驱动（0.5050 → 0.0000）。
>
> **真实的、可解释的退化**是 IoU 0.50–0.70 区间那条**平稳的 −0.010 ~ −0.020** 带，
> 以及 **precision↑ / recall↓** 的形态 —— 即模型变得更保守，而不是「定位崩了」。

---

## 7. Checkpoint 审计（§4 要求，未重训）

`runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/`：

| checkpoint | 大小 (B) | SHA256 | epoch |
|---|---:|---|---:|
| `best.pt` | 40,635,237 | `f9dddbfaca4cb08bde31d082eb0a54d499b8441eb1f97c6134a028caa6d83e55` | 237 |
| `last.pt` | 40,635,237 | `cfa00836314e5e5eb87b35ff7dd6fc42d2e0291f81c5ffc7384957b51e88004f` | 300 |

- ❌ **无 `epoch_*.pt`**（该目录下 `find` 仅这两个 `.pt`）。
  `args.yaml` 有 `save_period: 10`，理论上云端应有 `epoch10.pt … epoch290.pt`；
  本地这份 run 目录里没有 —— **若需中间 checkpoint，需回云端 `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/` 取**。
- ❌ **无独立 EMA 权重文件**（EMA 只存在 checkpoint 内部字段，不单独落盘）。
- ✅ `best.pt != last.pt`（SHA256 不同）。

**同一官方口径比较（`predict_rect.py --mode rect`，同 imgsz / conf / NMS / official_map 参数）**：

| checkpoint | epoch | official mAP50 | official mAP75 | **official mAP50-95** |
|---|---:|---:|---:|---:|
| `best.pt` | 237 | 0.77601 | 0.56386 | **0.50928** |
| `last.pt` | 300 | 0.76163 | 0.53053 | **0.50290** |
| **Δ (last − best)** | | −0.01438 | −0.03333 | **−0.00638** |

**结论：官方口径确认 `best.pt` (ep237) 优于 `last.pt` (ep300)，差 −0.00638。**
（内部口径 last−best = −0.00709，方向一致。）
→ **继续使用 `best.pt` 作为正式保底权重**，不得改用 `last.pt`。

（同样记录 Modality Dropout 自身的 best=ep262 / last=ep300，内部 mAP50-95 0.54697 / 0.53431。）

---

## 8. 这个结论说明了什么 / 不说明什么

**说明：**

1. **「在输入层把整模态通道随机清空」这个策略，在当前 RGBID 早期融合模型上不适合** ——
   300 epoch 单变量训练后在官方口径下净退化 −0.01899。
2. 该策略的**机制按设计生效了**：IR 依赖 0.0094 → 0.0019，Depth 依赖 0.0118 → −0.0001。
   即：**它确实削弱了模态依赖，但削弱模态依赖本身没有带来精度收益。**
3. 退化主要落在**极稀有类**（tricycle / ball / uav / garbage_can，合计 3.9% 标注），
   而占 96.1% 标注的类基本持平或上升。

**不说明：**

- ❌ **不说明 IR / Depth 没有价值。** baseline 对 IR (+0.0094)、Depth (+0.0118) 的依赖都是**正的**。
- ❌ 不说明「模态融合方向走错了」。
- ❌ 不说明其它 dropout 变体（更低概率 / 部分区域 drop / 特征级 drop / 只 drop 一个模态）也会失败。
  **本实验只否证了「ir_drop_prob=0.075 + depth_drop_prob=0.075 + 整通道清空 + 互斥」这一组具体设定。**
- ❌ 不说明线上会怎样。**离线 ≠ 线上**（本会话已有 +0.0049 → −2.729 的反例）。

---

## 9. 后续处置

| 动作 | 决定 |
|---|---|
| 提交 Modality Dropout 到线上 | ❌ **禁止** |
| 继续训练 7.5% dropout | ❌ **禁止** |
| 继续训练 3% dropout | ❌ **禁止** |
| 改 IR fill 后重跑本实验 | ❌ **禁止** |
| 覆盖 RGBID baseline | ❌ **禁止**（baseline 保持不动） |
| 删除历史实验 | ❌ **禁止** |
| 自动生成线上提交包 | ❌ **禁止** |
| **正式保底** | ✅ **RGBID Early Fusion，线上 mAP50-95 = 0.53180（不变）** |

---

## 10. 合规状态（全部重新核验）

| 对象 | 状态 |
|---|---|
| `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/`（正式实验目录） | ✅ 未修改 |
| baseline `best.pt` / `last.pt` | ✅ **未覆盖**（SHA256 与冻结记录一致） |
| F4 `best.pt` | ✅ 未变动 |
| RGBID 正式 `submission.zip` | ✅ 未变动、未被覆盖 |
| 冻结 `scripts/predict_rect.py` / `predict.py` / `official_map.py` | ✅ 未修改 |
| `ultralytics/data/base.py` / `cfg/default.yaml` / `scripts/train.py` | 三处 dropout 改动为**本实验必需**，已核验；`dataset.py` 无残留 |
| Modality Dropout 权重 | ✅ 保留于独立目录，**未被用于生成任何提交包** |
| 历史实验产物 | ✅ 未删除、未移动 |
| 线上提交 | ❌ 未执行 |

**本次新增只读/分析文件**（不改动任何既有产物）：
`scripts/modality_ablation_infer.py`、`scripts/modality_dropout_official_eval.py`、
`scripts/rgbid_pipeline_audit.py`、`diagnostic/modality_dropout/*`。

---

## 11. 最终输出

```text
Experiment:
RGBID + Modality Dropout

Status:
MODALITY_DROPOUT_REJECTED

Baseline official mAP50-95:
0.50928

Dropout official mAP50-95:
0.49029

Delta:
-0.01899

Main degradation:
official mAP75 -0.09917
（逐类拆解后集中于 2 个极稀有类 tricycle(2 框)/uav(30 框) 的 AP 台阶错位，
  占该阈值降幅的 69%；IoU 0.50-0.70 的真实退化带为 -0.010 ~ -0.020）

Online submission:
NOT SUBMITTED

Current best official online model:
RGBID Early Fusion
online mAP50-95 = 0.53180
```
