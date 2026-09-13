# 模型设计说明文档

> 本文件描述**当前实际实现**的模型设计。
> 实验过程、逐轮指标与结论见 [`docs/experiment_log.md`](experiment_log.md)；评估口径与当前基线摘要见 [`README.md`](../README.md) 第 7 章。
>
> 最近更新：2026-09-13

---

## 1. 任务概述

本项目为 Urban-Multimodal-Detection 城市场景多模态目标检测任务。
输入为空间已对齐的 RGB（可见光）、Infrared（红外）、Depth（深度）三种图像；
输出为目标物体的检测框、类别与置信度。检测共包含 12 个城市场景目标类别
（类别名与顺序见 `configs/dataset.yaml` 的 `names`）。

---

## 2. 当前主线模型：YOLO11m + RGB + Depth 中期融合（E7）

### 2.1 概览

| 项目 | 值 |
| :--- | :--- |
| 模型配置 | `configs/yolo11m_midfusion_rgbd_concat_res.yaml` |
| 输入模态 | **RGB（可见光 3 通道）+ Depth（深度 1 通道）**，融合为 4 通道 `[R, G, B, D]` |
| 类别数 | `nc = 12` |
| 模型规模 | `m`（`scales` 中 `m: [0.50, 1.00, 512]`） |
| 参数量 / 计算量 | **30,309,364（约 30.3M）/ 292.37 GFLOPs**（实测，`imgsz = 1024`） |
| 检测尺度 | **P3 / P4 / P5**（不含 P2） |
| 训练分辨率 | `imgsz = 1024` |
| best mAP@0.5:0.95 | **0.50878**（epoch 257，训练内置 `model.val()`） |
| 权重路径 | `runs/urban_multimodal_det_e7_yolo11m_rgbd_1024/weights/best.pt` |

> 注：该 yaml 头部注释写的是「30.3M 参数 / 115.11 GFLOPs」，其中参数量与实测一致，但 **115.11 GFLOPs 不是 `imgsz=1024` 口径**（按分辨率面积折算 292.37 × (640/1024)² ≈ 114.2）。以实测值 **292.37 GFLOPs @ 1024** 为准。

### 2.2 网络结构

采用**双分支骨干 + 逐尺度中期融合 + 共享检测头**的结构，全部由 fork 已有的模块搭建，未引入自定义模块：

```
输入 4 通道 [R, G, B, D]
        │
      Silence                                  # 恒等占位（模块索引 0）
        │
   ┌────┴─────────────────────────┐
   │ SilenceChannel[0,3]          │ SilenceChannel[3,4]
   │ → visible 分支（3 通道）      │ → depth 分支（1 通道）
   │   Conv s2 → Conv s2 → C3k2   │   Conv s2 → Conv s2 → C3k2
   │   → Conv s2 → C3k2  (vis P3) │   → Conv s2 → C3k2  (depth P3)
   └────┬─────────────────────────┘
        │  P3 融合： fused_P3 = depth_P3 + Conv1×1( Concat(vis_P3, depth_P3) )
        │
        ├─ 两分支各自继续下采样至 P4 → 同样的残差融合 → fused_P4
        └─ 两分支各自继续下采样至 P5 → 同样的残差融合 → fused_P5
                                          │
                                    SPPF → C2PSA
                                          │
                              共享 PAN-FPN 颈部（head）
                                          │
                              Detect(P3, P4, P5)
```

模块说明（均在 fork 内实现）：

| 模块 | 位置 | 作用 |
| :--- | :--- | :--- |
| `Silence` | `ultralytics/nn/modules/conv.py:358` | 恒等映射，作为多分支的公共起点占位 |
| `SilenceChannel[c_start, c_end]` | `ultralytics/nn/modules/conv.py:364` | 按通道切片，`x[..., c_start:c_end, :, :]`，用于把 4 通道拆成 visible(0:3) 与 depth(3:4) |
| `Concat` | `ultralytics/nn/modules/conv.py:327` | 通道维拼接 |
| `ADD(alpha)` | `ultralytics/nn/modules/block.py:2433` | `x1 + alpha * x2`，用于残差相加 |

### 2.3 融合方式：残差式、以 depth 为基底

这是本模型最关键的设计特征。P3 / P4 / P5 三个尺度上的融合均为：

```
fused = depth_Pn  +  Conv1×1( Concat(vis_Pn, depth_Pn) )
        └─ 基底 ──┘    └────── 可学习增量 delta ──────┘
```

对应 yaml 中的三行（以 P3 为例，模块索引 13–15）：

```yaml
- [[6, 12], 1, Concat, [1]]     # 13  拼接 vis_P3(节点6) 与 depth_P3(节点12)
- [-1, 1, Conv, [512, 1, 1]]   # 14  delta = 1×1 Conv 降维到 512
- [[14, 12], 1, ADD, [1.0]]     # 15  fused_P3 = delta + 1.0 * depth_P3
```

**语义**：以 **depth 分支为基底**，visible 分支通过 1×1 卷积提供一个**残差增量**，即模型被设计为「**偏向 depth**」，让 visible 做修正。

> 该设计取向与项目早期单模态实验中「depth 单模态表现不低于 RGB 单模态」的观察一致（`runs/urban_multimodal_det_yolo11_depth` = 0.31209、`runs/urban_multimodal_det_yolo11_rgb_baseline` = 0.30267、`runs/urban_multimodal_det_yolo11_infrared_v2` = 0.12162，均取自各自 `results.csv`）。
> ⚠️ 需注意：这三个运行属早期阶段（约 200 epoch、早于 E 系列），**其训练日志与 `args.yaml` 未随产物保留**，无法确认它们构成严格配对的对照实验，因此上述数字**仅作设计取向的旁证，不作为正式结论**。

### 2.4 输入与输出

**输入**：4 通道 `[R, G, B, D]`（CHW）。原始数据为 BGR 的 visible 图与单通道 depth 图，由 fork 的数据管线在通道维拼接后交给 `Format` 转换——前 3 通道做 BGR→RGB 翻转，第 4 通道 depth 保持原序。

**输出**：P3 / P4 / P5 三个尺度上的目标边界框、类别（12 类）与置信度。

**Depth 预处理**（已实验确认的正确配置）：

| 环节 | 配置 |
| :--- | :--- |
| 读取 | `cv2.IMREAD_UNCHANGED`（保留 uint16） |
| 归一化 | uint16 → `<300` 置 0（无深度信号）→ `/19999*255` → uint8 |
| resize | **`INTER_LINEAR`** |
| letterbox padding | **114** |

> **为什么 padding 必须是 114 而不是 0**：depth 中的 0 本身带有「无深度信号 / 背景」语义（数据集中 JPG 约 73%、PNG 约 5.78% 的像素为 0）。若用 0 填充，模型无法区分「填充区」与「合法背景」，会在图像边缘学到虚假的 depth 梯度。114 是 depth 值的离群点，天然充当「忽略此 padding」的哨兵。该结论由 E9 实验证伪得出，详见 `experiment_log.md`。

### 2.5 预训练权重迁移（`_transfer_rgb_pretrained`）

**问题**：ultralytics 默认的 `model.load()` 按「键名 + 形状」匹配权重（`intersect_dicts`）。RGBD 融合模型在骨干前插入了 `Silence` / `SilenceChannel`，使 visible 分支的模块索引整体偏移，且 depth 分支的模块穿插其中，**导致 visible 分支几乎匹配不到任何预训练键**。

**解决**：在 `ultralytics/models/yolo/detect/train.py:21` 实现 `_transfer_rgb_pretrained()`，按**结构对应关系**逐模块重映射：

1. 通过检测头数量 `nl` 选择映射表（`nl == 3` 用 P3/P4/P5 表，`nl == 4` 用 P2/P3/P4/P5 表）；
2. 逐模块把单分支 COCO 权重（如 `yolo11m.pt`）拷贝到融合模型的对应位置（覆盖 visible 分支 + 共享的 SPPF / C2PSA / 颈部 / Detect）；
3. **depth 分支与三处 1×1 融合卷积保持随机初始化**；
4. **depth stem 特殊初始化**：`W_depth = mean(W_R, W_G, W_B)`，bias 直接复制自 RGB stem。

**触发条件**：仅当模型骨干中含有 `in_channels == 1` 的 `Conv`（即 RGBD 融合模型）时才执行；单分支模型直接返回 0，交回标准 `load()` 处理，不受影响。

### 2.6 模型规模由文件名决定（重要约定）

fork 通过 `guess_model_scale()` **从 yaml 文件名中用正则提取规模字母**（n / s / m / l / x）。
**若文件名不含这些字母，会静默回退到 nano 规模**，不报错、不警告。

这正是 `configs/` 下模型文件命名为 `yolo11s_midfusion_rgbd_concat_res.yaml`、`yolo11m_midfusion_rgbd_concat_res.yaml` 的原因——**历史上一批实验曾因文件名缺少规模字母而全部静默跑在 nano 上**。新增模型配置时务必让文件名带正确的规模字母。

---

## 3. 数据管线

多模态读取由 fork 的加载器实现，通过 **CLI 参数**控制（不写在 `dataset.yaml` 里），由 `scripts/train.py` 从训练配置注入：

| 参数 | 取值 | 作用 |
| :--- | :--- | :--- |
| `use_simotm` | `"RGBD"` | 选择融合模式，产 4 通道图 |
| `channels` | `4` | 门控 4 通道的增强分支 |
| `pairs_rgb_ir` | `["visible", "depth"]` | 第二模态的路径替换规则 |
| `pairs_rgb_depth` | `["visible", "depth"]` | 同上（三模态预留） |

**读取机制**：以 `train` 指定的 `visible` 目录为主输入读取 RGB；depth 图由**路径字符串替换**得到（把路径中的 `visible` 换成 `depth`），要求两目录同名同构。

**数据划分**：原始数据只有 `train` 与 `test`，无独立 val。由 `scripts/train.py` 的 `_split_train_val_depth()` 从 `train` 按 `val_ratio=0.2` + `seed=42` 在 **stem 级别**（按原始文件名分组，避免同源图泄漏）切分出 val，结果写入 `data/processed/depth_split_train/`（幂等，已存在则复用）：

```
1600 训练 / 400 验证
```

---

## 4. 实验演进（E0–E9）

完整指标与逐项分析见 [`docs/experiment_log.md`](experiment_log.md)。演进主线：

| 方向 | 结论 |
| :--- | :--- |
| 模型规模 | n → s → m 逐级带来增益 |
| 模态 | RGBD 双模态优于 RGB 单模态（同规模下 +0.01350） |
| 输入分辨率 | 640 → 768 → 1024 逐级有效，是后期主要增益来源 |
| P2 检测层 | **两次独立证伪**（s 规模 −0.07459；m 规模仅 +0.00013），不采用 |
| depth resize | `INTER_NEAREST` **证伪**（−0.00769） |
| depth padding | `padding=0` **证伪**（−0.00357） |

**当前主线 = E7（YOLO11m + RGBD 中期融合 + `imgsz=1024`），best mAP@0.5:0.95 = 0.50878。**

**评估口径**：正式指标统一以**训练内置 `model.val()`** 为准；`scripts/predict.py` 仅作推理/结果检查工具。两者已完成一致性验证（398 张有效图 / 2807 GT 上差值 +0.00111）。

---

## 5. 遗留代码（未接入训练流程）

以下文件为项目早期「自定义模型」阶段的产物，**目前没有被任何代码 import**，保留仅作历史参考：

| 文件 | 内容 | 状态 |
| :--- | :--- | :--- |
| `models/fusion.py` | `ConcatFusion` / `MultiScaleConcatFusion` | 未接入 |
| `models/attention.py` | `CrossModalAttention` / `MultiScaleCrossModalAttention` | 未接入 |
| `configs/model.yaml`、`configs/model_fusion.yaml` | 早期融合方案配置 | 仅被 `models/fusion.py` 的文档字符串提及，**无脚本引用** |

**当前的多模态融合完全由 `configs/yolo11*_midfusion_rgbd_*.yaml` + fork 内建模块（`Silence` / `SilenceChannel` / `Concat` / `ADD`）实现**，是实际生效的方案。

---

## 6. 后续优化方向

推理侧优化（TTA / conf / NMS IoU）已探索完毕并明确不再优先：

- **TTA 未证明有稳定增益**（实测 −0.01174）；
- 后处理参数调参**不计入模型提升**（conf 项），或**未达门槛**（IoU 0.6 仅 +0.00207）。

当前优化重点为**训练阶段**。E7 训练曲线显示 `best` 出现在 epoch 257 后即进入平台期且 train/val 明显分叉，同时 `close_mosaic=10` 关闭 mosaic 的末 10 轮（ep291–300）出现陡降台阶（净损约 −0.0113）。因此正在进行严格单变量 A/B 实验：

```
Phase 6-A：close_mosaic 10 → 0（其余训练条件全部保持 E7 不变）
配置：configs/train_e7_close_mosaic0.yaml
输出：runs/urban_multimodal_det_e7_close_mosaic0/
```

后续方向将依据该实验的实际结果决定，暂不预设。
