# 面向城市场景的视觉多模态目标检测

> **当前状态（2026-10）**：主线已转到 **RGBID 三模态**，incumbent = **D′**（Separate-Stem 三路轻量 stem + Concat，`configs/yolo11m_sepstem.yaml`），official METRIC_A mAP@0.5:0.95 = **0.515281**、线上 **48.712**；复赛提交 D′+IRaug 线上 **48.951**。
> 早期 **RGBD 中期融合** 线峰值 = **F4**（YOLO11m RGBD @ `imgsz=1280`），`model.val()` mAP@0.5:0.95 = **0.53273**。注意两套口径（`model.val()` vs official METRIC_A）**不可直接比较**，详见 §7.1。
> 完整实验记录见 [`docs/experiment_log.md`](docs/experiment_log.md) 与 `reports/`、`diagnostic/` 下的各报告。

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
│   ├── dataset.yaml                     # ultralytics 数据配置 (path / train / val / test / nc / names)
│   ├── train*.yaml                      # 各实验训练超参（train_e*.yaml / train_f*.yaml / train_modality_m*.yaml / train_rgbid_*.yaml 等）
│   ├── yolo11*.yaml                     # 各模型配置（visible / midfusion_rgbd_* / earlyfusion / sepstem / modality{2,3,4}ch 等）
│   ├── MODALITY_BASELINE_README.md      # M1–M7 模态基线矩阵说明
│   ├── model.yaml / model_fusion.yaml   # 早期融合方案参考（未接入训练）
│
├── data
│   ├── raw
│   │   ├── train/{visible,infrared,depth,labels}
│   │   └── test/{visible,infrared,depth}
│   └── processed                        # train.py 切分生成的 *_split_train/（不入库）
│
├── docs
│   ├── model_design.md
│   └── experiment_log.md
│
├── experiments                         # 各实验可复现材料（配置快照、结果审计）
│   ├── README.md
│   ├── f4_1280/  f5_l1280/
│   └── yolo11_rgbd_loss_ablation/       # Loss 消融（含 README.md 与 l2_result_audit.md）
│
├── reports                             # 各实验 / 终审报告（F4/F5/D′/复赛/冻结审计等）
│
├── diagnostic                          # 诊断与审计（模态归因、3-modality 就绪审计等）
│
├── models
│   ├── attention.py                    # CrossModalAttention（未接入）
│   └── fusion.py                       # ConcatFusion（未接入）
│
├── scripts                             # 训练/评测/推理/审计入口（train.py、evaluate.py、predict*.py、official_eval.py 等）
│
├── ultralytics                         # 本地 fork（8.3.75，含 RGBT/RGBD/RGBID 多模态实现）
│
├── rematch                             # 复赛提交包（D′+IRaug，自包含，见 rematch/README.md）
│
├── weights                             # 训练权重（已 gitignore）
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

---

# 7. 当前基线与评估口径（2026-10 更新）

> 本章为实验结论摘要。完整配置、逐项结果与分析见 [`docs/experiment_log.md`](docs/experiment_log.md)（E/F 系列）与 `reports/`、`diagnostic/` 下的各报告（F4/F5、Loss 消融、D/D′、M 系列、复赛）。

## 7.1 评估口径说明（重要）

- **E / F 系列**（RGBD 中期融合）：指标统一以训练内置 `model.val()` 的 `best mAP@0.5:0.95` 为准（`conf=0.001`、`iou=0.7`、`max_det=300`、`rect=False`）。
- **D′ / M 系列**（RGBID 三模态、模态基线）：指标统一以 `scripts/official_eval.py` 的 **official METRIC_A** 为准（top-100、IoU 0.50–0.95、101 点插值），用于跨配置横向比较与线上对标。
- 两套口径**不可直接比较**：`model.val()` mAP 与 official METRIC_A 是不同度量。`scripts/predict*.py` 定位为推理 / 结果检查工具，不作为正式指标来源。

## 7.2 实验演进主线

### F 系列（RGBD 中期融合，`model.val()` 口径）

| 实验 | 唯一变量 | best mAP@0.5:0.95 | 结论 |
| :--- | :--- | :---: | :--- |
| F1 | lr0 0.01→0.005（@1024） | 0.51955 | 确立新基线（旧基线 E7 = 0.50878） |
| F2 | SGD→AdamW | 无产物 | 未采用 |
| F3a | lr0 0.005→0.004 | — | 证伪 |
| **F4** | **imgsz 1024→1280** | **0.53273** | **RGBD 线峰值（+0.01318）** |
| F5 | YOLO11m→l @1280 | 0.52619 | 容量杠杆证伪 |

> depth 通道结论：resize 保持 `INTER_LINEAR`、padding 保持 **114**（depth 中的 0 有「无深度信号」语义）。相关变体（`INTER_NEAREST`、padding=0）均已证伪。

### Loss 消融（基于 F4，唯一变量 box/cls/dfl）

| 实验 | 权重 | best mAP@0.5:0.95 | Δ vs F4 |
| :--- | :--- | :---: | :---: |
| F4（默认） | box=7.5 / cls=0.5 / dfl=1.5 | 0.53273 | — |
| L1 | box=10.0 | 0.52093 | −0.01180 |
| L2 | dfl=2.0 | 0.51894 | −0.01379 |

结论：**默认损失权重为当前最优，loss 权重不是提升杠杆**（见 `experiments/yolo11_rgbd_loss_ablation/`）。

### RGBID 三模态（official METRIC_A 口径）

- D = 单层 `Conv(5→64,3,2)` early fusion（`configs/yolo11m_earlyfusion.yaml`）；
- **D′ = 三路轻量 Stem（48/8/8）+ Concat（`configs/yolo11m_sepstem.yaml`）＝ incumbent**，official mAP@0.5:0.95 = **0.515281**、线上 **48.712**（详见 `reports/RGBID_SEPSTEM_CLAHE_FINAL_REPORT.md`）。

### M1–M7 模态基线矩阵（Fusion Phase 0）

单/双/三模态基线以支撑模态归因，见 `configs/MODALITY_BASELINE_README.md`：M1–M6 均 READY（M4 = 0.50597），M7 = D′（incumbent）。

### 复赛（rematch）

D′ + IR-specific gamma/noise 增强：线上 **48.951**（vs D′ 48.712，+0.239）；但本地两口径均为负（official −0.00100 / fork −0.00299，`rematch/submission/RESULTS.md` 判 **NOT SUPPORTED**），线上差异无法与单次轨迹扰动区分。详见 `rematch/README.md`。

## 7.3 已探索并明确不再优先的方向（inference-side）

| 方向 | 结果 | 结论 |
| :--- | :--- | :--- |
| TTA（原图 + 水平翻转） | mAP@0.5:0.95 −0.01174 | 负向，不采用 |
| conf 阈值调参 | 0.001→0.01 约 +0.026 | 属纯推理调整，不计入模型提升 |
| NMS IoU 调参 | +0.00207 | 低于 +0.005 门槛，不采用 |

**结论：不再优先优化 inference-side，也不通过修改 conf / IoU / NMS 等推理参数来人为追求更高指标。**

## 7.4 当前状态与下一步

- **主线已从 RGBD 中期融合转到 RGBID 三模态**（D′ 为 incumbent，复赛提交 D′+IRaug）。
- 3-modality 下一步（M4-preserving minimal residual Depth adapter）**当前不可执行**：`_transfer_rgb_pretrained` 无对应 remap 分支，且合同门有盲区。见 `diagnostic/FINAL_3MODALITY_READINESS_AUDIT.md`（结论 NOT READY）。
- 已证伪 / 关闭的杠杆：P2 检测层、容量（F5）、loss 权重（L1/L2）、TTA、conf/IoU/NMS。
