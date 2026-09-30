# OASA SCALE=1.4 — OFFICIAL 300E RESULT

## Experiment

| 项 | 值 |
|---|---|
| run | `runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_oasa_s14` |
| epochs | 300（执行史见下） |
| model | YOLO11m_sepstem（`configs/yolo11m_sepstem.yaml`） |
| dataset | `data/processed/rgbid_split_train`（1600 train / 400 val，其中 1 train + 2 val corrupt） |
| seed | 42 |
| scale | **1.4** |
| prob | 0.5 |
| small_area | 1024（native 口径） |
| min_visible | 0.8 |

## Provenance

| 项 | 值 |
|---|---|
| git commit | `f659609c28f6b4f500212197d5229550a36d12d8` |
| `augment.py` SHA256 | `5cb9a407625921ce3f12ffc13df2d703`（与冻结值一致） |
| 训练配置 | `configs/oasa_pre_mosaic_scale14.yaml`，SHA256 `dee6601da39309d091164ddf3ee0056` |
| **续训配置** | `configs/oasa_pre_mosaic_scale14_resume.yaml`（与上者只差 `resume`/`resume_path` 两行） |
| **`models/yolo/detect/train.py`** | `8b4178be947acdc031878132a8b5ef79` → **`b021354f541bbb3c7770eb341c852bbe`** ⚠️ |
| `best.pt` SHA256 | `055795ad884e62a96f57803c7f58db608992e81c236f830349c3005c52054bbe` |
| `last.pt` SHA256 | `d7f78057a58af443ab1d006fbd834174a244037cf8e3f4cada46b19470b9684f` |
| 推理前后 `best.pt` | SHA256 **逐字相同**（未被改动） |

### ⚠️ 两项必须记录的执行史偏差

**（1）300 epoch = 226 完整 + ep227 重跑 + ep228–300 续训**

原训练在 epoch 227 跑到 38% 时被意外中断，从 `last.pt`（226 结束态）续训。
`resume` 恢复 model / optimizer / EMA / best_fitness / LR 调度，但**不恢复 dataloader shuffle 状态**，因此 ep227–300 的样本顺序与「不中断跑完」不同。

- **LR 调度连续性已验证**：ep227 实测 `0.00075668`，余弦理论值 `0.000756` ⇒ 调度逐位接上
- epoch 编号 1→300 完整、无重复、无跳变

**（2）续训需修补 `_transfer_rgb_pretrained` 的 resume 守卫（+17 行，无删除）**

原代码分派只依据**目标模型**的 stem，假定**源**是单分支 stock COCO。resume 时源是本模型的 sepstem checkpoint（`model.0` 是 `Identity`，RGB stem 在 `model.2`），于是 `_remap_separate_stem` 找不到 `model.0.conv.weight` 而 `RuntimeError`。

- **不能靠绕行解决**：即使越过 stem 检查，remap 也会把已训好的 Detect cv3 分类分支按 `nc` 重新初始化，抹掉分类头
- **惰性证明（已测）**：修复后 `model.load()` 的 661/661 张量与源 ckpt **逐位相同**；stock→sepstem 路径输出与首训日志**逐字相同**（`550` tensors、`61/661`）
- 项目 preflight 在此修改后仍 `ALL CHECKS PASSED`

## Runtime Contract

| 项 | 状态 | 证据 |
|---|---|---|
| native-space small criterion | ✅ | `areas_native` 换算在位 |
| pre-Mosaic insertion | ✅ | `transforms[0]` = ObjectScaleAug |
| Mosaic ON | ✅ | `mosaic_on=1` 全程（ep1–290） |
| **close_mosaic OFF** | ✅ | `Closing dataloader mosaic` 后：`mosaic_on=0 eligible=0 applied=0 skip_nomosaic=4000`（吃满），`seen` 从 200 重新起算 |
| validation OFF | ✅ | val transforms = `['LetterBox','Format']` |
| inference OFF | ✅ | `LoadImagesAndVideos` + `LetterBox`；推理日志 `use_simotm=RGBID channels=5 ir_encoding=clahe` |
| **OASA 实际 GT drop** | 实测 **17.14%**（`gt 133320→110474`） | 离线 simulation 预测 17.51%（Δ0.37pp）✅ |

## Training Curve（fork 口径，**仅诊断**）

| 模型 | best mAP50-95 | best epoch | final |
|---|---:|---:|---:|
| D′ | 0.56690 | 246 | 0.55522 |
| OASA 2.0 | 0.55409 | **195** | 0.54437 |
| **OASA 1.4** | **0.56070** | **263** | 0.54795 |

**分段 Δ vs D′**：

| 窗口 | Δ(1.4−D′) | Δ(2.0−D′) |
|---|---:|---:|
| ep1–100 | **+0.00455** | −0.00232 |
| ep101–200 | −0.00206 | −0.00677 |
| **ep201–290** | **−0.00309** | **−0.01941** |
| ep291–300（close_mosaic） | −0.00684 | −0.01000 |

⇒ **1.4 把后程分叉压缩了 84%**（−0.0194 → −0.0031），峰值从 ep195 推迟到 ep263。
**离线 simulation 的机制预测被真实训练证实**：减少监督损失 ⇒ 延缓饱和。

## Official Evaluation

| Model | Official mAP50-95 | mAP50 | mAP75 | pred 框数 | clamped |
|---|---:|---:|---:|---:|---:|
| D′ SepStem+CLAHE | **0.51528** | 0.7783 | 0.5110 | — | — |
| OASA 2.0 | 0.51122 | 0.78756 | 0.55208 | 5357 | 5 |
| **OASA 1.4** | **0.50739** | 0.78890 | **0.50197** | 4534 | 3 |

评测链：`scripts/official_map.py`（未修改），`official_eval.MAX_BOXES_PER_IMAGE = 100` 冻结默认未覆盖；
`scripts/test_official_eval.py` regression tests **ALL PASS**；prediction dump 参数与 OASA-2.0 完全一致。

## Deltas

```
Δ(1.4 − D′)   = 0.50739 − 0.51528 = −0.00789
Δ(1.4 − 2.0)  = 0.50739 − 0.51122 = −0.00383
Δ(2.0 − D′)   = 0.51122 − 0.51528 = −0.00406   （§0 冻结值，未重测）
```

## Fork Diagnostic（**次要证据**）

| 口径 | D′ | OASA 2.0 | OASA 1.4 | 排序 |
|---|---:|---:|---:|---|
| fork（results.csv best） | 0.56690 | 0.55409 | 0.56070 | D′ > **1.4 > 2.0** |
| fork（对预测重算） | 0.56596 | 0.54761 | 0.55601 | D′ > **1.4 > 2.0** |
| **官方（cap=100，判据）** | **0.51528** | **0.51122** | **0.50739** | D′ > **2.0 > 1.4** |

> ⚠️ **两个口径对 1.4 vs 2.0 给出相反排序。** fork 口径在本项目已有多次符号翻转记录；
> 本次是又一个实例。**以官方口径为准**，且不能用 fork 的「1.4 更好」去推翻官方结论。

## Interpretation

### 1. Observed fact（事实）

- OASA 1.4 官方 mAP50-95 = **0.50739**，低于 D′（0.51528）**−0.00789**，也低于 OASA 2.0（0.51122）**−0.00383**
- 两个 OASA 尺度**都**未超过 D′
- 官方 mAP50：1.4 = 0.78890 **≥** 2.0 = 0.78756 ≥ D′ = 0.7783
- 官方 mAP75：1.4 = **0.50197**，显著低于 2.0 = 0.55208

### 2. Diagnostic evidence（诊断）

- **fork 与官方排序冲突**（D′>1.4>2.0 vs D′>2.0>1.4），因此 fork 的任何正向信号不能作为判据
- 1.4 的机制确实生效：实测 GT drop 与 simulation 吻合到 0.37pp；后程分叉压缩 84%；峰值推迟 68 个 epoch
- 1.4 的劣势集中在 **mAP75**（−0.050 vs 2.0），而 mAP50 持平甚至略优 ⇒ 影响在**定位精度**而非检出
- 无多种子方差估计，Δ 量级（0.004–0.008）与历史噪声带同阶，**不声称统计显著性**

### 3. Possible mechanism（推测，未验证）

`gt` 保留提升确实兑现为「更晚饱和」，但未兑现为官方指标。可能原因（**均为推测**）：
- 1.4× 放大对定位精度的增益不足以补偿 ROI 裁剪带来的**上下文丢失**（1.4 的 ROI 仍裁掉 49% 画面）
- OASA 改变的是**输入尺度分布**，而官方 mAP50-95 对定位敏感；尺度扰动可能反而损害框回归精度
- 真实瓶颈可能不在监督量，而在别处（例如 F1 诊断记录的「困难类硬漏检」）

**不把「监督损失下降」写成「因此 AP 一定提高」—— 事实证明没有提高。**

## Submission Decision

```text
不提交 OASA 1.4。
```

按 §8 判定规则：`OASA 1.4 (0.50739) < D′ (0.51528)` ⇒ **在冻结官方口径下未观察到性能增益**。

结合 OASA 2.0 (0.51122)：**两个尺度均低于 D′，且 1.4 更低。OASA 路线在冻结官方口径下未产生可交付增益。**

**路线诊断（客观）**：OASA 的机制假设（减少监督损失 → 延缓饱和 → 提升指标）**前半段被证实、后半段被证伪**。
既然「把监督损失减半」（1.4 vs 2.0：影响 −84% 后程分叉）都没能换来官方提升，
**监督损失不是这条线的瓶颈**，继续在 `scale` 上取值不太可能改变结论。

**按 §9，本轮不追加任何 OASA 参数实验（不扫 1.2/1.3/1.5，不加种子，不改 evaluator）。**

## Training

```text
训练已完成：300 epoch（226 完整 + ep227 重跑 + ep228–300 续训）
本轮新增提交 = 0 ；线上评测 = 0
```

## 产物

```
diagnostic/oasa_scale14/oasa14_best/results/                    400 个预测 TXT
diagnostic/oasa_scale14/_oasa14_best.infer.log                 推理日志
diagnostic/oasa_scale14/_oasa14_best.official_map.txt          官方评测结果
diagnostic/oasa_scale14/_oasa14_best.test_official_eval.txt    regression test (ALL PASS)
diagnostic/oasa_scale14/_local_official_eval.sh                复现命令
runs/..._oasa_s14/oasa_s14_0923_1820.log                       完整训练日志（含 OASA marker）
```
