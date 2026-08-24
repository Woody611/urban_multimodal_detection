# 模型设计说明文档

## 1. 任务概述

本项目为 Urban-Multimodal-Detection 城市场景多模态目标检测任务。
输入为空间已对齐的 RGB（可见光）、Infrared（红外）、Depth（深度）三种图像；
输出为目标物体的检测框、类别与置信度。检测共包含 12 个城市场景目标类别
（类别名与顺序见 `configs/dataset.yaml` 的 `names`）。

## 2. Baseline 模型（当前实现）

当前 Baseline 采用 **ultralytics YOLOv11** 单模态检测，输入仅 RGB（可见光，3 通道）。
训练不再使用自定义训练循环，而是直接调用 ultralytics 的 `YOLO.train()`：

- 模型结构：`configs/yolo11_visible.yaml`（YOLOv11，`ch=3`、`nc=12`）；
- 数据配置：`configs/dataset.yaml`（ultralytics 格式，12 类）；
- 训练超参：`configs/train.yaml`（由 `scripts/train.py` 映射为 ultralytics 参数）；
- 训练入口：`scripts/train.py`，评估 `scripts/evaluate.py`，推理 `scripts/predict.py`。

数据流：

1. 将 RGB 图像送入 YOLOv11 骨干网络，提取多尺度特征（P3 / P4 / P5）；
2. 经 PAN 颈部与 Detect 检测头完成边界框回归与类别分类；
3. 输出目标检测结果。

Baseline 的作用是建立单模态参照基准，用来对比后续多模态融合模型的效果。

## 3. 多模态融合模型（规划中）

多模态模型将同时接收 RGB、Infrared、Depth 三种对齐图像。融合方案在
`configs/model.yaml` / `configs/model_fusion.yaml` 中描述，融合模块已在
`models/` 中以「独立、自包含」的纯 PyTorch 形式实现，尚未接入训练流程：

- `models/fusion.py` —— `ConcatFusion` / `MultiScaleConcatFusion`（通道拼接融合）；
- `models/attention.py` —— `CrossModalAttention` / `MultiScaleCrossModalAttention`
  （以 visible 为 query、infrared / depth 为 key / value 的非对称交叉注意力）。

多模态数据流（规划）：

1. 三个模态分别经各自的特征提取分支，独立提取对应模态的多尺度特征；
2. 在指定特征层对多模态特征执行融合（ConcatFusion 或 CrossModalAttention）；
3. 将融合后的综合特征送入检测头；
4. 检测头完成边界框回归与类别分类，输出最终检测结果。

> 融合方案严格参考 `configs/model_fusion.yaml` 配置，不额外增加未确定的模型结构。

## 4. 输入与输出

### 输入

- RGB：可见光彩色图像（3 通道）
- Infrared：红外图像
- Depth：深度图像

三张图像已经完成空间对齐，可以直接送入模型。

### 输出

- 各目标的边界框坐标
- 目标所属类别（共 12 类）
- 预测置信度

## 5. 模型设计思路

1. 先用 ultralytics YOLOv11 搭建 RGB 单模态 Baseline，验证整套训练、推理、评估流程可正常跑通，拿到单模态性能基准；
2. 在 Baseline 基础上接入红外、深度两个模态，用 `models/fusion.py` / `models/attention.py` 做多模态特征融合；
3. 对比单模态与多模态的检测效果，验证多模态信息对城市场景检测是否带来增益。

## 6. 后续优化

后续将根据实际实验结果对模型进行调整与优化，暂不预设具体优化方向与实验效果。
