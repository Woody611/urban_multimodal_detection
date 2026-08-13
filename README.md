# 面向城市场景的视觉多模态目标检测

---

# 1. 项目简介

本项目面向复杂城市场景下的视觉多模态目标检测任务，以 RGB（可见光）、Infrared（红外）和 Depth（深度）三种空间对齐视觉模态作为输入，通过深度学习目标检测模型实现城市环境中多类别目标的定位与识别。

模型输出目标类别（Class Label）、边界框（Bounding Box）以及预测置信度（Confidence）。

---

# 2. 数据集类别

项目面向城市复杂环境目标检测任务，包含以下12类目标：

| 类别编号 | 类别 |
| :---: | :--- |
| 0 | person 行人 |
| 1 | boat 船 |
| 2 | animal 动物 |
| 3 | seat 座椅 |
| 4 | sign 标识 |
| 5 | bicycle 双轮车 |
| 6 | car 汽车 |
| 7 | ball 球 |
| 8 | light 灯 |
| 9 | garbage can 垃圾桶 |
| 10 | uav 无人机 |
| 11 | tricycle 三轮车 |

---

# 3. 环境配置

```bash
conda create -n urban_multimodal python=3.10 -y
conda activate urban_multimodal
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124   # 有独显用 cu124，无独显改 cpu
pip install -r requirements.txt
```

---

# 4. 项目目录结构

```
Urban-MultiModal-Object-Detection

├── configs
│   ├── dataset.yaml
│   ├── model.yaml
│   └── train.yaml
│
├── data
│   └── raw
│
├── docs
│   ├── model_design.md
│   └── experiment_log.md
│
├── experiments
│   ├── baseline
│   └── fusion
│
├── models
│
├── scripts
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── utils
│
├── weights
│
├── README.md
├── requirements.txt
└── environment.yml
```

---

# 5. 团队分工

|成员|负责文件/目录|最终交付物|
|-|-|-|
|成员一（队长）|README.md<br>configs/<br>models/<br>scripts/train.py<br>scripts/predict.py|项目方案、核心代码、最终模型、提交版本|
|成员二|data/<br>models/<br>scripts/evaluate.py<br>experiments/|数据处理代码、训练代码、模型权重、实验结果|
|成员三|docs/<br>README.md（辅助）|model_design.md、experiment_log.md、技术报告|

---

# 6. 引用与参考资料

1. Cao Y, Bin J, Hamari J, et al.  
   Multimodal Object Detection by Channel Switching and Spatial Attention[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 403-411.

2. Redmon J, Divvala S, Girshick R, et al.  
   You Only Look Once: Unified, Real-Time Object Detection[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 779-788.

3. Cheng C, Xu T, Wu X J, et al.  
   EvaNet: Towards More Efficient and Consistent Infrared and Visible Image Fusion Assessment[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

4. Tang Z, Xie Y, Xu T, et al.  
   Learning Bi-Directional Fusion and Deformation-Sensitive Loss for RGB-T Tiny Object Detection[J]. Information Fusion, 2025: 103985.

5. Zhu X F, Xu T, Pan Y, Gu J, Li X, Lu J, et al.  
   Collaborating Vision, Depth, and Thermal Signals for Multi-Modal Tracking: Dataset and Algorithm[C]//The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025.
