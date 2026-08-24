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
| 9 | garbage_can 垃圾桶 |
| 10 | uav 无人机 |
| 11 | tricycle 三轮车 |

### 数据格式说明

- 图像扩展名：`visible` / `infrared` / `depth` 均为 **`.png` 与 `.jpg` 混存**；
- 标注格式：YOLO 格式 `<class_id> <norm_cx> <norm_cy> <norm_w> <norm_h>`（空格分隔，坐标归一化到 [0,1]）；
- 数据划分：仅含 `train` 与 `test`，无独立 `val`，验证集需从 `train` 切分。

---

# 3. 环境配置

```bash
conda create -n urban_multimodal python=3.10 -y
conda activate urban_multimodal
# CUDA 版 PyTorch（GPU 训练必需；无 NVIDIA GPU 或仅跑流程可装 CPU 版）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

说明：

- **本地 fork 的硬依赖**：项目使用仓库内自带的 `ultralytics/` fork（含 RGBT 多模态融合实现），其 `nn/tasks.py`、`nn/modules/attention.py`、`data/base.py` 等直接 `import timm` / `einops` / `efficientnet_pytorch` / `thop` / `psutil`，均已列入 `requirements.txt` 与 `environment.yml`。
- **GPU 训练需 CUDA 版 torch**：上方的 cu124 安装需 NVIDIA 驱动 ≥ 525.60.13（可用 `nvidia-smi` 确认）。驱动较旧时把 `cu124` 换成 `cu121` 或 `cu118`；无 GPU 则用 CPU 版 `pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1`。
- **使用本地 fork**：脚本通过 `sys.path` 优先导入仓库根目录的 `ultralytics/`，而非 pip 安装的官方 ultralytics，多模态相关能力以本地 fork 为准。

---

# 4. 项目目录结构

```
Urban-Multimodal-Detection

├── configs
│   ├── dataset.yaml           # ultralytics 数据配置 (path / train / val / test / nc / names)
│   ├── train.yaml             # 训练超参，由 scripts/train.py 映射到 ultralytics
│   ├── yolo11_visible.yaml    # YOLOv11 单模态 baseline 模型 (nc=12)
│   ├── model.yaml             # 自定义模型设计（融合方案参考）
│   └── model_fusion.yaml      # 三模态融合设计 (fusion_method 等)
│
├── data
│   ├── raw
│   │   ├── train
│   │   │   ├── visible/       # 可见光 (RGB)
│   │   │   ├── infrared/      # 红外
│   │   │   ├── depth/         # 深度
│   │   │   └── labels/        # YOLO 格式标注
│   │   └── test
│   │       ├── visible/
│   │       ├── infrared/
│   │       └── depth/
│   └── processed              # train.py 切分生成的 visible_split/（不入库）
│
├── docs
│   ├── model_design.md
│   └── experiment_log.md
│
├── experiments
│   ├── README.md
│   ├── baseline/              # 各实验材料（配置快照、metrics.json 等）
│   └── fusion/
│
├── models
│   ├── attention.py           # CrossModalAttention（未来融合模块）
│   └── fusion.py              # ConcatFusion（未来融合模块）
│
├── scripts
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── ultralytics                # 本地 fork（含 RGBT 多模态融合实现）
│
├── weights                    # 训练权重（已 gitignore）
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
