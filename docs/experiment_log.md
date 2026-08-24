# 实验记录

本文档集中记录项目各轮实验的配置、结果与分析。每完成一轮完整训练实验，更新下方表格并在对应小节补充详细记录。

> ⚠️ **代码迁移说明**：Exp-001 在项目早期的「自定义模型」代码库上完成，现已迁移到 ultralytics/YOLOv11 方案。下文引用的 `models/baseline.py`、`BaselineDetector`、`DetectionLoss`、`ModelEMA`、`_bn_initialized` 等文件/符号已随迁移移除，仅作历史记录保留；实验结果数据与结论仍有效。当前实现见 `README.md` 与 `docs/model_design.md`。

## 实验记录表

| 实验编号 | 日期 | 模型版本 | 输入模态 | 主要修改 | 训练配置 | mAP@0.5 | mAP@0.5:0.95 | 结论 | 备注 |
| :--- | :---: | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| Exp-001 | 2026-08-19 ~ 08-21 | Baseline（CSPDarknet-s + PAN + DecoupledHead） | RGB（Visible） | 建立单模态基准，跑通完整 pipeline | lr=1e-2 / bs=16 / 300 ep / SGD+Cos / AMP+EMA | 0.1512 | 0.0809 | 整体 mAP 偏低；原始模型优于 EMA；boat、tricycle 的 AP=0 | EMA 结果受 BN 同步 bug 影响 |

## 实验记录规范

每次完成一轮完整训练实验，需要填写表格，记录内容说明：

1. **实验编号**：自增编号，例如 Exp-001、Exp-002
2. **日期**：实验执行的日期
3. **模型版本**：Baseline / 多模态融合版本等，对应代码版本
4. **输入模态**：RGB / RGB+Infrared / RGB+Infrared+Depth
5. **主要修改**：本次实验改动点，如修改融合策略、调整数据集、修改骨干网络等
6. **训练配置**：引用 config 文件版本，记录关键超参（学习率、batch size、epoch 等）
7. **mAP@0.5、mAP@0.5:0.95**：测试集评估指标
8. **结论**：简单总结本次实验现象，效果变好/变差，出现的问题
9. **备注**：其他补充，例如报错、异常现象、对比参照对象

> 禁止编造实验数据，没有跑完的实验表格留空。

---

## Exp-001：Baseline（RGB 单模态）详细记录

> 面向城市场景的视觉多模态目标检测 · 第一轮 Baseline
> 实验标识：`urban_multimodal_det_v1`

### 1. 实验目的

本轮实验用于建立 **RGB（可见光）单模态目标检测基准（Baseline）**，目标如下：

1. 跑通完整的「数据加载 → 模型训练 → 权重保存 → 评估」基础流程（pipeline）；
2. 产出一个可复现的 RGB 单模态检测基线成绩，作为后续多模态融合实验的对照基准；
3. 为后续三组融合实验提供性能对比参照：
   - RGB + 红外（Infrared）
   - RGB + 深度（Depth）
   - RGB + 红外 + 深度（Infrared + Depth）

本实验只验证单模态 baseline 模型，不涉及多模态融合。

### 2. 实验配置

以下配置为 Exp-001 执行时仓库中的真实配置（`configs/dataset.yaml`、`configs/model.yaml`、`configs/train.yaml`）；此后代码已迁移到 ultralytics/YOLOv11，其中 `dataset.yaml` 已改为 ultralytics 格式。

| 项目 | 配置 |
| :--- | :--- |
| 实验名称 | `urban_multimodal_det_v1` |
| 输入模态 | Visible（RGB，3 通道）单模态 |
| 数据集 | `data/raw/train`，共 2000 个空间对齐样本（visible / infrared / depth / labels 各 2000） |
| 训练 / 验证切分 | 无独立 val，从 train 按 `val_ratio=0.2` + `seed=42` 在 stem 级别切分 → 训练 1600 / 验证 400 |
| 输入尺寸 | 640 × 640（letterbox） |
| 类别数量 | 12（与 `dataset.yaml` 一致） |
| 模型结构 | CSPDarknet-s 主干 + PAN 颈部 + DecoupledHead 解耦检测头 |
| 检测范式 | anchor-free，YOLOX 风格解耦头，YOLOv5 风格框解码 |
| 训练轮数 | `epochs=300`（最佳权重保存在 epoch 178） |
| 优化器 | SGD，`lr=1e-2`，`momentum=0.937`，`weight_decay=5e-4`，`nesterov=true` |
| 参数分组 | backbone lr×0.1，neck / head lr×1.0 |
| 学习率调度 | 预热 3 epoch（1e-6 → 1e-2）+ CosineAnnealingLR（`min_lr_ratio=0.01`） |
| 混合精度 | AMP（FP16）开启 |
| EMA | 开启，`decay=0.9999` |
| 批量大小 | `batch_size=16`，`num_workers=4` |
| 设备 | CUDA（单卡） |
| 预训练权重 | 无（`pretrained.enabled=false`，随机初始化） |
| 损失函数 | DetectionLoss：obj 用 focal BCE（全网格）、cls 用 focal BCE（仅正样本 one-hot）、reg 用 CIoU（仅正样本），`γ=1.5`、`α=0.25` |
| 评估指标 | `mAP@0.5`、`mAP@0.5:0.95`（COCO 101 点插值 AP，类别内取均值，无 GT 类别忽略） |
| 评估后处理 | `conf_thres=0.001`、`iou_thres=0.6`、`max_det=300`（默认） |
| checkpoint | `runs/urban_multimodal_det_v1/weights/best.pth` |

### 3. Baseline 模型说明（历史实现）

Exp-001 使用的 Baseline 模型为自定义 **`BaselineDetector`**（原 `models/baseline.py`，该文件已随代码迁移移除），其数据流为：

```
Visible (RGB, 3 通道) → CSPDarknet 主干 → PAN 颈部 → DecoupledHead → 三尺度预测
```

关键点：

- **仅使用 Visible/RGB** 作为输入；`configs/model.yaml` 的 `modality.enabled` 仅包含 `"visible"`；
- **不使用 Infrared（红外）和 Depth（深度）**，二者仅保留配置占位，未参与前向计算；
- 主干 `CSPDarknet` 采用 `variant="s"`（`depth_multiple=0.33`、`width_multiple=0.50`），输出 P3 / P4 / P5 三个尺度（stride 8 / 16 / 32，通道经 `width_multiple=0.50` 缩放后为 [128, 256, 512]），P5 末端含 SPPF；
- 检测头 `DecoupledHead` 在每个尺度上分成 cls / reg / obj 三个独立分支（隐藏通道 256）。

### 4. 实验结果

三个实验均在 **验证集 400 个样本**、epoch 178 的 `best.pth` 权重上评估（val_loss 均为 0.628788）：

| 实验名称 | 权重类型 | max_det | mAP@0.5 | mAP@0.5:0.95 | 备注 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| 实验 1：EMA 权重评估 | EMA | 300 | 0.1225 | 0.0698 | val_loss=0.628788 |
| 实验 2：原始模型权重评估 | 原始模型（model） | 300 | **0.1512** | **0.0809** | val_loss=0.628788，num_preds=120000 |
| 实验 3：原始模型 + max_det=100 | 原始模型（model） | 100 | 0.1491 | 0.0803 | val_loss=0.628788，num_preds=40000 |

说明：实验 2、3 均通过 `--no_ema` 加载 checkpoint 中的原始 `model` 权重；`best_score=0.081602` 为训练期代理指标（checkpoint 内记录）。

### 5. EMA 与原始模型对比

| 权重类型 | mAP@0.5:0.95 |
| :--- | :---: |
| EMA 权重 | 0.0698 |
| 原始模型（model） | **0.0809** |

对比结论：

- 在相同验证集与评估设置下，**原始模型（mAP@0.5:0.95 = 0.0809）优于 EMA 权重（0.0698）**；
- 因此 **后续 Baseline 结果采用原始模型（`--no_ema`）作为最佳结果**，EMA 权重仅作参考记录。

> ⚠️ **补充说明（代码核对后）**：EMA 权重反而不如原始模型，属异常现象。代码层面已定位原因——当时的 `scripts/train.py` 的 `ModelEMA` 曾存在 BatchNorm 运行统计量未同步的 bug（`6c52ca3` 已修复），当时的 `scripts/evaluate.py` 也为此加入了 `_bn_initialized` 退化回退保护。该结果很可能反映的是修复前训练的 EMA 权重退化，详见「实验结果分析」。

### 6. max_det 实验分析

在原始模型权重上，比较 `max_det` 由默认 300 调整为 100 的影响：

| max_det | mAP@0.5 | mAP@0.5:0.95 | num_preds |
| :---: | :---: | :---: | :---: |
| 300 | 0.1512 | **0.0809** | 120000 |
| 100 | 0.1491 | 0.0803 | 40000 |

结论：

- `max_det` 从 300 降到 100 后，mAP@0.5:0.95 由 0.0809 降至 0.0803，仅下降 0.0006；
- 说明在当前验证集上，`max_det` 从 300 调整为 100 对整体检测性能影响较小；
- 后续实验应根据比赛最终预测数量限制进行设置，并保持不同实验之间的评估设置一致。

### 7. 各类别 AP@0.5

以下为 **原始模型**（实验 2）在验证集上的 12 类逐类 AP@0.5：

| 类别编号 | 类别 | AP@0.5 |
| :---: | :--- | :---: |
| 0 | person 行人 | 0.0926 |
| 1 | boat 船 | 0.0000 |
| 2 | animal 动物 | 0.0718 |
| 3 | seat 座椅 | 0.4169 |
| 4 | sign 标识 | 0.0675 |
| 5 | bicycle 双轮车 | 0.0342 |
| 6 | car 汽车 | 0.2456 |
| 7 | ball 球 | 0.1115 |
| 8 | light 灯 | 0.3207 |
| 9 | garbage_can 垃圾桶 | 0.0335 |
| 10 | uav 无人机 | 0.4205 |
| 11 | tricycle 三轮车 | 0.0000 |

简单分析（仅依据实际结果，不推断原因）：

- **表现较好**：uav（0.4205）、seat（0.4169）、light（0.3207）、car（0.2456）；
- **表现中等**：ball（0.1115）、person（0.0926）、animal（0.0718）、sign（0.0675）；
- **表现较弱**：bicycle（0.0342）、garbage_can（0.0335）；
- **AP@0.5 = 0**：boat（0.0000）、tricycle（0.0000），说明当前模型在验证集上对这两个类别未获得有效的检测性能。具体原因尚需结合类别样本分布、目标尺度以及预测结果进一步分析。

整体而言各类别 AP 普遍偏低，说明单模态 Baseline 仍有较大提升空间，也为后续多模态融合提供了可观测的改进空间。

### 8. 最终 Baseline 结论

当前 RGB 单模态 Baseline 在标准验证设置下的最佳结果为：

- 权重类型：原始模型（`--no_ema`）
- epoch：178
- mAP@0.5：0.1512
- mAP@0.5:0.95：0.0809

在将 `max_det` 限制为 100 的附加实验中，mAP@0.5:0.95 为 0.0803。

因此，后续模型实验将以 0.0809 作为当前 Baseline 的参考成绩，同时在最终测试/提交阶段遵循比赛规定的预测数量限制。

### 9. 后续实验方向

当前 Baseline 阶段已经完成，后续进入多模态融合实验阶段。

按照比赛计划，后续将开展以下实验：

1. RGB + 红外（Infrared）
2. RGB + 深度（Depth）
3. RGB + 红外 + 深度（Infrared + Depth）

通过对比不同模态组合的检测性能，分析红外和深度信息对城市场景目标检测的增益。

本阶段不再修改 `train.py` 或重新训练 RGB Baseline。后续多模态实验将在独立的 Fusion 实验中进行。

### 附录 A：权重文件清单

| 文件 | 说明 |
| :--- | :--- |
| `runs/urban_multimodal_det_v1/weights/best.pth` | 最佳权重（epoch 178，`best_score=0.081602`） |
| `runs/urban_multimodal_det_v1/weights/last.pth` | 最新 checkpoint（含 optimizer / scheduler / scaler） |
| `runs/urban_multimodal_det_v1/weights/`epoch_50.pth`、`epoch_60.pth`、`epoch_70.pth`、`epoch_80.pth`、`epoch_90.pth` | 每 10 epoch 的中间权重 |

### 附录 B：复现命令

```bash
# 实验 1：EMA 权重评估
python scripts/evaluate.py --weights runs/urban_multimodal_det_v1/weights/best.pth

# 实验 2：原始模型权重评估（Baseline 最佳结果）
python scripts/evaluate.py --weights runs/urban_multimodal_det_v1/weights/best.pth --no_ema

# 实验 3：原始模型 + max_det=100
python scripts/evaluate.py --weights runs/urban_multimodal_det_v1/weights/best.pth --no_ema --max_det 100
```

---

## 实验结果分析

> 以下基于 Exp-001 的训练结果与代码核对，分析当前 Baseline 反映的问题与改进方向。

### 1. 总体性能判断

mAP@0.5 = 0.1512、mAP@0.5:0.95 = 0.0809，对 12 类城市场景目标检测任务而言明显偏低，尚不具备实用价值。这个量级说明当前 Baseline 存在**系统性训练问题**，而非仅靠「更多 epoch」就能补齐。

### 2. 反映的主要问题

1. **正样本分配过于稀疏（最可疑的主因）**
   `DetectionLoss`（当时的 `scripts/train.py`）采用最简化的目标分配：每个 GT 只分配到其中心所在网格，三个尺度各分配一次，即每个 GT 仅 3 个正样本，无 simOTA、无多 anchor 偏移。相比 YOLOX 的 simOTA（每个 GT 动态分配数十个正样本），监督信号严重不足，导致回归与分类学习不充分，是 mAP 偏低的首要嫌疑点。

2. **无预训练 + 小数据 + 小模型，易欠拟合**
   `pretrained.enabled=false` 从零随机初始化；训练集仅 1600 张、12 类（平均约 133 张/类）。CSPDarknet-s 属轻量骨干，检测任务通常依赖 ImageNet/COCO 预训练来获得良好的初始特征，从零 + 小数据极易欠拟合。

3. **EMA 权重退化（已定位、已修复）**
   EMA 结果（0.0698）反而不如原始模型（0.0809），违背「EMA 通常不差于原始模型」的常识。代码核对确认：当时的 `scripts/train.py` 的 `ModelEMA` 曾存在 BatchNorm 运行统计量未同步的 bug（commit `6c52ca3` 已修复），当时的 `scripts/evaluate.py` 亦为此加入 `_bn_initialized` 退化回退。因此该 EMA 结果很可能反映修复前训练的退化权重；同时需注意 `best.pth` 若按 EMA 保存，训练期选优可能基于退化模型，应在修复后重新评估确认。

4. **部分类别 AP=0 / 接近 0**
   boat、tricycle 的 AP@0.5 = 0；bicycle（0.0342）、garbage_can（0.0335）接近 0。需进一步核查这些类别的样本数量、目标尺度与标注质量，判断是样本不足还是类间混淆。

5. **细节存疑点**
   三个实验的 val_loss 完全相同（0.628788），但 EMA 与原始模型权重不同、前向输出理应略有差异。由于 val_loss 中 obj 项（全网格 focal BCE）占主导，理论上可能接近，但精确到 6 位小数完全相同仍建议复核评估脚本是否确实加载了不同权重。

### 3. 下一步改进方向

1. 用 YOLOX 式 simOTA（或至少增加每个 GT 的正样本数）替换简化的中心网格分配；
2. 接入预训练骨干，或补充 mosaic/mixup 等数据增强、适当延长训练；
3. 在 EMA 修复后重新评估 `best.pth`，确认真实最佳成绩；
4. 统计各类别样本分布，对 boat、tricycle 等 tail 类做针对处理（类别平衡采样或 focal 参数调优）；
5. 统一并固化评估配置（`conf_thres`、`max_det`），并在日志中规范记录执行日期与关键指标。
