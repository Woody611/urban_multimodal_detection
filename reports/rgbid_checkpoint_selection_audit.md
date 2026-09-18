# RGBID Checkpoint Selection 审计（只读验证）

**日期**：2026-09-18
**对象**：`runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/`
**性质**：**只读验证**。未训练、未修改任何 `.yaml` / 源码 / `results.csv`、未修改或覆盖任何 checkpoint、未替换 `best.pt`、未生成新 run、未上传线上。
**目的**：回答一个问题 —— **ep237 的单点 best 是否只是噪声尖峰？ep276–290 稳定平台中的 checkpoint 在官方赛题口径下是否优于 `best.pt`？**

---

## 1. Executive Summary

> **没有任何后期 checkpoint 在官方赛题口径下超过 `best.pt`。**
> ep240 / ep260 / ep280 / ep290 的 official mAP50-95 分别为
> **0.50097 / 0.50146 / 0.50396 / 0.50742**，**全部低于 `best.pt`(ep237) 的 0.50928**，
> 最接近的 ep290 仍低 **−0.00186**。
>
> 结论：**`NO OFFICIAL IMPROVEMENT`**。当前 `best.pt` 保持为正式保底权重（线上 0.53180）**不变**；
> **没有证据**支持改变 checkpoint selection 策略，**不建议**为此投入训练预算。

**同时必须诚实记录**：本会话此前的诊断报告 `rgbid_baseline_late_training_diagnosis.md` 曾基于
**内部指标**的 10-epoch 滚动均值推测「ep276–290 才是真正的稳定最优平台」。
**本次官方口径实测直接否证了该推测** —— 该平台在官方指标上**不如** ep237。
这正是引入本次验证的价值：**内部指标的稳定平台判断不能直接迁移到官方赛题指标。**

---

## 2. Checkpoint Inventory

目录：`runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/`
（命名格式为 `epoch{N}.pt`，**无下划线**）

| 文件 | 大小 (B) | mtime | SHA256 | ckpt.epoch | 本次是否评估 |
|---|---:|---|---|---:|---|
| `best.pt` | 40,635,237 | 2026-09-17 06:01:16 | `f9dddbfaca4cb08bde31d082eb0a54d499b8441eb1f97c6134a028caa6d83e55` | −1（已 strip） | ✅（复用既有正式结果） |
| `last.pt` | 40,635,237 | 2026-09-17 06:01:25 | `cfa00836314e5e5eb87b35ff7dd6fc42d2e0291f81c5ffc7384957b51e88004f` | −1（已 strip） | 未评（此前已评：0.50290） |
| `epoch200.pt` | 80,894,656 | 2026-09-18 08:54:21 | `24bce27474084dd519f81748f457459e8db87707d53fac2caeae210833b68dd0` | 200 | 未评（用户未要求） |
| `epoch210.pt` | 80,896,000 | 2026-09-18 08:54:58 | `ef2f0f44529f76f7c9454d283d2d18775b48025dcd433d97b2c3272c9d6ba349` | 210 | 未评（用户未要求） |
| `epoch220.pt` | 80,897,408 | 2026-09-18 08:54:54 | `fec092e6b4cb16cb5b59be7b72b9a7bec7ba0efd7e5ccb770c95d960eb6fd32d` | 220 | 未评（用户未要求） |
| `epoch230.pt` | 80,898,752 | 2026-09-18 08:55:31 | `18b389d541d7fa96d5f23a0fb6fae83dfdebf9f02cedba359719379c6da13a5a` | 230 | 未评（用户未要求） |
| **`epoch240.pt`** | 80,900,096 | 2026-09-18 08:55:58 | `1c2df30b0223e1f4a09550068b5e348ca33ea86e8d0f3d4308af56fcd711e156` | 240 | ✅ |
| `epoch250.pt` | 80,901,504 | 2026-09-18 08:56:08 | `1296f8f33007e305a4f70df8737b531499c57838520373919a72f6f19a0ae0f0` | 250 | 未评（用户未要求） |
| **`epoch260.pt`** | 80,902,848 | 2026-09-18 08:56:12 | `f21638cb0b329d696d3b24cb25ba95dbb670371a8f69c735529030bf2acb503c` | 260 | ✅ |
| `epoch270.pt` | 80,904,256 | 2026-09-18 08:56:10 | `8075fd0e8da0594d3866580746722ca24c9297cf725c97a9fb9cbbf65277cc46` | 270 | 未评（用户未要求） |
| **`epoch280.pt`** | 80,905,600 | 2026-09-18 08:56:17 | `cf1401efcdc9a9f6a5b736c76f8a02a6d794b62b106518fab32cb2334b231c02` | 280 | ✅ |
| **`epoch290.pt`** | 80,907,008 | 2026-09-18 08:56:16 | `5733d825d180ced7115cd1f772c742bf580d8ddc10ee520fa1ca8786a458d9a6` | 290 | ✅ |

**NOT_FOUND：无。** 用户点名的 7 个（230/240/250/260/270/280/290）全部存在，另有 200/210/220。
（本轮按用户后续指示，**只评 240 / 260 / 280 / 290**。）

**结构核验**：每个 `epoch*.pt` 的 `epoch` 字段与文件名一致；均含 `ema` 与 `optimizer`
（故约 80.9 MB，而 `best/last.pt` 经 `strip_optimizer` 后为 40.6 MB、`epoch = -1`）。
两者都含 `ema`，而 `attempt_load_one_weight` **优先取 `ema`**，故所有 checkpoint 的评估口径一致。

**来自 checkpoint 自身的独立佐证**（不依赖对 `results.csv` 的重新推导）：

| checkpoint | ckpt.epoch | 内嵌 `best_fitness` |
|---|---:|---:|
| epoch200 | 200 | 0.57665 |
| epoch210 | 210 | 0.57688 |
| epoch220 | 220 | 0.57688 |
| epoch230 | 230 | 0.57697 |
| **epoch240** | 240 | **0.58372** |
| epoch250 … **epoch290** | 250–290 | **0.58372（此后一直未变）** |

`best_fitness` 自 ep240 起锁定在 **0.58372** 直到 ep290 再未上升；
ep237 的 fitness = 0.58371，而 ep238–300 的最大 fitness = 0.58308。
**即训练循环自己记录的历史最优，在 ep237 之后再未被刷新。**

---

## 3. Evaluation Protocol

**完全复用既有正式命令，未发明任何新口径。** 该命令即记录于
`reports/rgbid_inference_gain_audit.md`、复现出 **0.77601 / 0.56386 / 0.50928** 的那一条：

```bash
# 1) 推理（冻结的提交管线）
python scripts/predict_rect.py \
  --weights <checkpoint> \
  --train_config configs/train_rgbird_ir_quicktest.yaml \
  --source data/processed/rgbid_split/images/val/visible \
  --mode rect --imgsz 1280 --conf 0.001 --iou 0.7 --max_det 300 --max_boxes 300 \
  --batch 16 --output diagnostic/checkpoint_selection_audit/<name>

# 2) 官方赛题口径（mean + tail0 + conf 匹配）
python scripts/official_map.py \
  --results diagnostic/checkpoint_selection_audit/<name>/results \
  --split_root data/processed/rgbid_split
```

**`best.pt` 未重跑** —— 其结果目录 `diagnostic/rgbid_inference_gain/RGBID_baseline/results`
仍在磁盘上，且此前已用同一脚本复算并**逐位重现** 0.77601 / 0.56386 / 0.50928。
（避免无意义的重复计算。）

**`args.yaml` 口径核对**：`imgsz=1280`、`iou=0.7`、`max_det=300`、`half=false`、`augment=false`、
`save_json=false`、`channels=5`、`use_simotm=RGBID`、`rect=false`（训练期；val 内部走 rect）——
与上述命令一致。所有 checkpoint 的预测框总数均落在 4,344–4,497 区间，无异常。

**未使用的口径**：训练日志 `results.csv` 的 `metrics/mAP50-95(B)` **不作为结论依据**，
仅在 §6 中用于偏移分析。

---

## 4. Official Evaluation Results

官方赛题口径（`official_map.py`，同 `--split_root`）。缺失项一律标 `NOT_FOUND`，未填任何假数据。

| Checkpoint | Epoch | mAP50 | mAP75 | **mAP50-95** | P | R | R-S | R-M | R-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`best.pt`** | **237** | **0.77601** | **0.56386** | **0.50928** | 0.8578 | 0.7029 | 0.6712 | 0.8136 | 0.8647 |
| `epoch240.pt` | 240 | 0.76881 | 0.50724 | 0.50097 | 0.8620 | 0.6986 | 0.6819 | 0.8182 | 0.8557 |
| `epoch260.pt` | 260 | 0.76973 | 0.55457 | 0.50146 | 0.8323 | 0.7196 | 0.6685 | 0.8114 | 0.8566 |
| `epoch280.pt` | 280 | 0.77176 | 0.54721 | 0.50396 | 0.8526 | 0.7004 | 0.6846 | 0.8106 | 0.8539 |
| `epoch290.pt` | 290 | 0.77480 | 0.54549 | 0.50742 | 0.8426 | 0.7132 | 0.6819 | 0.8152 | 0.8620 |
| `epoch230.pt` | 230 | NOT_FOUND（未评） | | | | | | | |
| `epoch250.pt` | 250 | NOT_FOUND（未评） | | | | | | | |
| `epoch270.pt` | 270 | NOT_FOUND（未评） | | | | | | | |
| `epoch200/210/220.pt` | 200/210/220 | NOT_FOUND（未评） | | | | | | | |

> P / R 为官方口径下 IoU=0.5、按类别 p 的 micro 平均在 F1 最大工作点上的取值。
> R-S / R-M / R-L 为 COCO 定义（small <32²、medium 32²–96²、large >96²）在 IoU=0.5 官方 conf 顺序贪心匹配下的 recall。

**相对 `best.pt` 的差值**：

| Checkpoint | Δ mAP50 | Δ mAP75 | **Δ mAP50-95** |
|---|---:|---:|---:|
| `epoch240.pt` | −0.00720 | −0.05662 | **−0.00830** |
| `epoch260.pt` | −0.00628 | −0.00929 | **−0.00782** |
| `epoch280.pt` | −0.00425 | −0.01665 | **−0.00531** |
| `epoch290.pt` | −0.00121 | −0.01837 | **−0.00186** |

**全部为负。** 四个后期 checkpoint **无一超过 0.50928**。

---

## 5. Best vs Plateau

| | `best.pt` (ep237) | `epoch280` | `epoch290` |
|---|---:|---:|---:|
| **official mAP50-95** | **0.50928** | 0.50396 | 0.50742 |
| Δ vs best | — | **−0.00531** | **−0.00186** |
| official mAP75 | 0.56386 | 0.54721 (−0.01665) | 0.54549 (−0.01837) |
| official mAP50 | 0.77601 | 0.77176 (−0.00425) | 0.77480 (−0.00121) |
| P (micro) | 0.8578 | 0.8526 | 0.8426 |
| R (micro) | 0.7029 | 0.7004 | 0.7132 |
| R-S | 0.6712 | **0.6846（最高）** | 0.6819 |
| R-M | 0.8136 | 0.8106 | 0.8152 |
| R-L | **0.8647（最高）** | 0.8539 | 0.8620 |

**三点观察**：

1. **`best.pt` 在 mAP50-95 / mAP75 / mAP50 三项上均为最高**，三项一致指向同一结论。
2. **ep290 是最接近的挑战者**（−0.00186），且它的 **micro recall 反而最高（0.7132 vs 0.7029）**，
   但 **mAP75 明显更低（−0.01837）** —— 典型的「检得更全、框得更松」，在官方 mAP 上净亏。
3. **small recall 上 ep280/ep290 更好**（0.6846/0.6819 vs 0.6712），
   但 **large recall 上 `best.pt` 最好**（0.8647）。后期 checkpoint 的收益集中在最小的目标上，
   不足以补偿高 IoU 与大目标的损失。

---

## 6. Selection Analysis

### 6.1 训练日志与官方指标的系统性偏移

| Epoch | 内部 mAP50-95 | 官方 mAP50-95 | Δ（official − internal） |
|---:|---:|---:|---:|
| 237 | 0.55800 | 0.50928 | **−0.04872** |
| 240 | 0.54524 | 0.50097 | **−0.04427** |
| 260 | 0.55529 | 0.50146 | **−0.05383** |
| 280 | 0.55541 | 0.50396 | **−0.05145** |
| 290 | 0.55503 | 0.50742 | **−0.04761** |

```
偏移均值 = -0.04918    std = 0.00327    范围 = [-0.05383, -0.04427]    跨度 = 0.00957
```

- **偏移方向恒定**：官方始终比内部低约 **0.044–0.054**，**没有任何一次反号**。
  这与两个口径的定义差异（101 点算术平均 + 尾部取 0 + conf 顺序匹配 vs trapz + ramp + IoU 匹配）一致。
- **但偏移并不稳定**：跨度 **0.00957** 已经**大于官方值域跨度 0.00830**，也大于内部跨度 0.01276 的 3/4。
  即**不能用「官方 ≈ 内部 − 0.049」做换算**来做 checkpoint 选择。
- **秩相关**：`Spearman(internal, official) = +0.700`
  （internal 秩 `[1,5,3,2,4]`，official 秩 `[1,5,4,3,2]`，顺序 237→240→260→280→290）。
  首名（237）与末名（240）两端口径一致，但**中间三个（260/280/290）的排序被完全打乱**：
  内部认为 280 > 260 > 290，官方认为 290 > 280 > 260。

> **方法论含义**：在本数据上，**内部指标可以在"最大值选点"这个粗粒度上工作
> （237 在两口径都是第 1），但不能可靠地分辨平台内部各 epoch 的优劣**
> —— 偏移抖动（0.0096）与官方真实差距（0.0083）同量级。

### 6.2 ep237 是否只是噪声尖峰？

**两个层面要分开回答：**

| 命题 | 证据 | 判定 |
|---|---|---|
| ep237 的**内部指标**是噪声抬高的 | 相对 ep237–290 均值 z=+1.71；领先第 2 名仅 0.00093 < 噪声步长 1/3；ep236→237 单步 +0.01185 而 ep237→238 回吐 −0.00595 | ✅ **成立** |
| ep237 的**权重**是候选中最好的 | 官方 mAP50-95 0.50928，高于全部 4 个后期 checkpoint；内嵌 `best_fitness` 自 ep240 起再未刷新 | ✅ **成立** |

**两个命题并不矛盾**：ep237 的内部读数确实被验证集噪声抬高，
但**它的权重在官方口径下仍然是最优的那个**。
换句话说 —— **"ep237 是噪声尖峰"这一说法，不足以推出"后期 checkpoint 更好"。**
本次实测正是对后半个推测的否证。

### 6.3 276–290 是否存在稳定平台？

- **内部指标层面：存在。** 10-epoch 滚动均值在 ep276–290 达 0.55457（全区间最高），
  且 std 仅 0.00181（最小）。这一点在 `rgbid_baseline_late_training_diagnosis.md` §3.2 已确立，**仍然成立**。
- **官方指标层面：该平台不优于 ep237。** ep280 = 0.50396、ep290 = 0.50742，
  均低于 0.50928。**内部稳定性没有转化为官方成绩。**

> ⚠️ **对本会话此前结论的更正**：`rgbid_baseline_late_training_diagnosis.md` §6.2 曾建议
> 「若云端存在 `epoch280/290.pt`，官方评测可回答平台是否更好」。
> **现在答案已经给出：不更好。** 该建议的答案是否定的，不应据此改变选点策略。

### 6.4 官方指标是否支持选择后期 checkpoint？

**不支持。** 4/4 后期 checkpoint 均低于 `best.pt`，且差距方向完全一致（Δ mAP50、Δ mAP75、
Δ mAP50-95 **三项全部为负**）。这不是单指标的偶然波动。

---

## 7. Recommendation

**属于 §Q3 的「情况 B」**：

> ep280 / ep290 ≤ best.pt。当前 ep237 虽然可能存在一定随机波动，
> 但**至少没有证据证明后期 checkpoint 在官方指标上更优**。
> **不要继续在 checkpoint selection 上浪费训练预算。**

据此：

| 项 | 决定 |
|---|---|
| 更换 checkpoint 选择策略 | ❌ **不做**（情况 B，无证据支持） |
| 引入 official-metric-based checkpoint selection | ❌ **不引入** —— 前提（官方要求后期更好）未被满足；且需改训练代码，本次禁止 |
| 用 `epoch280/290.pt` 替换 `best.pt` | ❌ **禁止**，且无收益（低 −0.00531 / −0.00186） |
| `best.pt` 作为正式保底 | ✅ **保持不变**（线上 0.53180） |
| 后续训练预算投向 | 维持 `rgbid_baseline_late_training_diagnosis.md` §6.1 的排序：**保持当前策略**（首选）；**轻量融合结构**（唯一有正期望的方向）。**checkpoint selection 已从候选项中划掉。** |

**唯一的后续可选项（非训练、非本次任务范围）**：若将来确实关心选点稳健性，
可考虑 **对平台内多个 checkpoint 的权重做平均**（如 ep270/280/290），
但那**需要先做一次官方口径验证**才能判断，且**不属于本轮任务**，本次不做。

---

## 8. 合规状态

| 项 | 状态 |
|---|---|
| 训练 | ❌ **未启动任何训练** |
| `.yaml` 修改 | ❌ **无** |
| Python 源码修改 | ❌ **无** |
| `results.csv` 修改 | ❌ **无** |
| checkpoint 修改 / 覆盖 / 删除 / 重命名 | ❌ **无**（`best.pt` / 全部 `epoch*.pt` SHA256 已复核不变） |
| `best.pt` 替换 | ❌ **未替换** |
| 新训练 run | ❌ **未生成** |
| optimizer / resume / train | ❌ **未运行** |
| 线上提交 | ❌ **未提交** |
| 正式线上保底 | ✅ `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt`（线上 mAP50-95 = **0.53180**）**保持不变** |

**本次新增产物**：`reports/rgbid_checkpoint_selection_audit.md`（本报告）与
`diagnostic/checkpoint_selection_audit/`（推理结果与评测日志，纯新增目录）。

---

## 9. 最终输出

```text
CHECKPOINT_AUDIT_COMPLETE

Found checkpoints:
  best.pt, last.pt,
  epoch200.pt, epoch210.pt, epoch220.pt, epoch230.pt, epoch240.pt,
  epoch250.pt, epoch260.pt, epoch270.pt, epoch280.pt, epoch290.pt
  （用户点名的 7 个全部存在；NOT_FOUND = 无）

Evaluated (official protocol: predict_rect --mode rect + official_map):
  best.pt(ep237)   mAP50=0.77601  mAP75=0.56386  mAP50-95=0.50928
  epoch240.pt      mAP50=0.76881  mAP75=0.50724  mAP50-95=0.50097
  epoch260.pt      mAP50=0.76973  mAP75=0.55457  mAP50-95=0.50146
  epoch280.pt      mAP50=0.77176  mAP75=0.54721  mAP50-95=0.50396
  epoch290.pt      mAP50=0.77480  mAP75=0.54549  mAP50-95=0.50742

Best official mAP50-95:
  0.50928  (best.pt / epoch 237)

Current formal best:
  0.50928 offline / 0.53180 online

Does any late checkpoint beat 0.50928?
  NO   (closest: epoch290 = 0.50742, delta = -0.00186)

Does this justify changing checkpoint selection?
  NO   (Case B)

Training launched:
  NO

Config modified:
  NO

Historical checkpoint modified:
  NO

Online submission:
  NO
```
