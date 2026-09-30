# Separate-Stem Fusion —— 最终报告

**日期**：2026-09-22
**实验**：D' = IR-CLAHE + Separate-Stem Fusion

---

## A. Experiment identity

```text
D  (基线) = IR-CLAHE + 单层 Conv(5→64,3,2) 早期融合
D' (候选) = IR-CLAHE + 三路轻量 Stem (48/8/8) + Concat 融合
```

| | fusion | IR 预处理 | 其余 |
|---|---|---|---|
| **D** | `configs/yolo11m_earlyfusion.yaml` | CLAHE | 完全相同 |
| **D'** | `configs/yolo11m_sepstem.yaml` | CLAHE | 完全相同 |

---

## B. Variable control —— 单变量成立

D 与 D' 的差异经独立审计（4 审计员 + 对抗性复核，76 项检查）确认为**恰好两项**：

```text
model_config（= fusion architecture）  +  experiment_name
```

| 检查 | 结果 |
|---|---|
| `args.yaml` 逐项对比（除 `model`/`name`/`save_dir`） | **差异 0 项** |
| `ir_encoding` | **`clahe`**（args.yaml 与 ckpt 内嵌 `train_args` 两处一致） |
| 架构 | 31 层 / 20,061,972 参数 / ch=5 / nc=12 |
| Δparams / ΔFLOPs vs D | **−1,440（−0.0072%）/ −0.43%** |

> ⚠️ 本轮**曾发生过一次污染**：首次训练在云上用了「`_clahe` 的 experiment_name + 缺 `ir_encoding`」的配置，
> `base.py:333` 静默回落 percentile，D vs D' 变成两个变量。该次产物已如实改名为
> `runs/..._rgbid_sepstem_percentile` 并排除在本报告之外；成因已清理（见 §C）。

---

## C. Dry-run / 防呆

| 项 | 结果 |
|---|---|
| train-entry dry-run（真实 `scripts/train.py`、1 真实 batch） | `TRAIN_ENTRY_READY` |
| 真实 batch | `(8,5,1280,1280)` · 1 batch · 1 optimizer.step |
| remap 在真实入口下生效 | ep1 mAP50-95 = 0.256（from-scratch 应 ≈0.001） |

**污染成因已清理（3 道拦截，全部实测有效）**：

1. `scripts/train.py` 硬守卫 —— RGBID 缺 `ir_encoding` 直接 `SystemExit`（白名单只豁免冻结的历史配置）
2. `scripts/preflight_check_train_config.py` —— 云上训练前跑，走**与正式训练相同的解析路径**，
   并复刻 `check_dict_alignment`（正是云上次报错的那个函数）
3. 命名去歧义 —— 删掉易混淆的配置名，`ir_encoding` 显式化并移到文件上方

---

## D. Pretrained audit

| 项 | 结果 |
|---|---|
| backbone 结构对应 exact-match | **245 / 245** |
| head 结构对应 exact-match | **287 / 287** |
| 合计 | **532 / 538 = 98.9%**（与 D 完全相同） |
| 合法例外 | Detect `.cv3.` 分类分支 **6** 个 tensor（源 COCO `nc=80` → 本数据集 `nc=12`） |
| 三个 stem 初始化 | RGB←`stock[:48]`、IR/Depth←`mean(stock[:,0:3])[:8]`，**逐位相等 `max\|Δ\|=0`** |

---

## E. Training result

`runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/` · 300 epoch 无早停（`patience: 0`）

| | best ep | mAP50-95 | mAP50 | mAP75 | P | R | final |
|---|---:|---:|---:|---:|---:|---:|---:|
| D | 263 | 0.56806 | 0.81879 | 0.57624 | 0.87429 | 0.72746 | 0.55612 |
| **D'** | 246 | **0.56690** | 0.82487 | 0.57471 | 0.86387 | **0.75251** | 0.55522 |
| Δ | | **−0.00116** | +0.00608 | −0.00153 | −0.01042 | **+0.02505** | |

曲线 ep240 后进入平台期。**工作点明显右移：recall ↑、precision ↓。**

**本地官方口径**（同一 400 图 val、同一 pipeline）：

| 口径 | D | D' | Δ |
|---|---:|---:|---:|
| mAP50 | 0.77119 | 0.78021 | **+0.00902** |
| mAP75 | 0.50161 | 0.51241 | **+0.01080** |
| mAP50-95 | 0.51076 | 0.51657 | **+0.00581** |
| 框数 | 4341 | 4654 | +313 |

**三个本地口径符号互相矛盾**（训练内 val −0.0012 / fork −0.0001 / 官方 +0.0058），
幅度均落在本项目已记录的 checkpoint 波动带（~0.008–0.015）内 → **本地无法判定**，只能等线上。

---

## F. Online result —— ★ 测试集已更换

**复赛启用了新的测试集**，因此旧基准 `0.5553` **不再可比**。必须用**同一新测试集**上的 D 作对照：

| 模型 | 新测试集线上分 |
|---|---:|
| D（Early Fusion + IR-CLAHE） | **47.985** |
| **D'（Separate-Stem + IR-CLAHE）** | **48.712** |
| **Δ(D' − D)** | **+0.727**（相对 **+1.52%**） |

> 历史参考（旧测试集，仅存档不可比）：官方 baseline 53.1800、D 55.53。
> 新测试集整体分数下移约 5–7 分，属测试集难度/口径变化，不影响 D vs D' 的同集对照。

---

## G. Final verdict

```text
SEPARATE_STEM_ONLINE_GAIN_CONFIRMED
```

按 brief §24 情况 A（D' > 同测试集 baseline），**Separate-Stem 在 IR-CLAHE 已固定的前提下
确实优于原始 5ch early fusion，线上 Δ = +0.727。**

### 附带结论：本地指标的可信度记录被再一次改写

| 实验 | 官方口径（本地） | fork 口径（本地） | 线上 | 谁的方向对 |
|---|---|---|---|---|
| conf=0.0001 | +0.0049 | — | **−2.729** | 都错 |
| D（IR-CLAHE） | +0.0015 | +0.0100 | **+2.35**(旧集) | **都对本** |
| **D'（Separate-Stem）** | **+0.0058** | **−0.0001** | **+0.727** | **官方对，fork 无分辨率** |

**要点**：fork 口径这次的 Δ(−0.0001) **落在它自身的噪声带内**（本项目记录 ~0.008–0.015），
即它这次**没有给出信息**，而不是"给错了方向"。此前"方向信 fork"的裁定需要限定为：

> fork 只在 **|Δ| 超出其噪声带（约 0.008）** 时可作判据；带内视为**无信息**，
> 此时官方口径仍有方向参考价值（本次即如此）。

---

## 提交包 provenance

| 项 | 值 |
|---|---|
| submission.zip | 349,545 B · sha256 `830e0c552615852f86a51005249c7b814844d1283c6310632903f3a16380e4d8` |
| checkpoint best.pt | sha256 `1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda` |
| model config | `9b14f2946073338414b4441784b6df87a868b982` |
| train config | `a4e329cfc3d220206448cdd3377c1b3f5126c50f` |
| base.py / loaders.py | `8bcf8349…` / `f84bdcf2…` |
| detect/train.py / predict_rect.py | `8b4178be…` / `c55f7744…` |
| 预测 | 1000 图 / **10,926 框** / 每图 ≤100（合规，超限 0） |

**未触碰**：`OFFICIAL_FALLBACK` `f9dddbfa…`、D 的 `best.pt` `3c2f942c…`、0.53180 正式包 `0c550773…`。

---

```text
SEPARATE_STEM_ONLINE_GAIN_CONFIRMED
D  (新测试集) = 47.985
D' (新测试集) = 48.712
Δ             = +0.727
```
