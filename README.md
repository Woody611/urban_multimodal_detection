# 面向城市场景的视觉多模态目标检测

> **当前状态（2026-09-14）**：主线基线为 **F1**（YOLO11m + RGB+Depth 中期融合，`imgsz=1024`，`lr0=0.005`），训练内置 `model.val()` 的 `best mAP@0.5:0.95 = 0.51955`（epoch 287）；上一版 E7（`lr0=0.01`）= 0.50878，F1 相对 E7 **+0.01077**。
> **正式实验指标统一以 `model.val()` 为准**，`scripts/predict.py` 仅作推理/结果检查工具。推理侧优化（TTA / conf / NMS IoU）已探索完毕，当前优化重点为**训练阶段**。
> 完整实验记录与结论见 [`docs/experiment_log.md`](docs/experiment_log.md)。

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

---

# 7. 当前基线与评估口径（2026-09-13 更新）

> 本章为实验结论摘要。完整配置、逐项结果与分析见 [`docs/experiment_log.md`](docs/experiment_log.md)。

## 7.1 当前主线基线：F1（E7 + lr0 减半）

| 项目 | 值 |
| :--- | :--- |
| 实验 | **F1**（上一版 E7，见下方对照） |
| 模型 | YOLO11m + RGB + Depth 中期融合（`configs/yolo11m_midfusion_rgbd_concat_res.yaml`，`ch=4`、P3/P4/P5、约 30.3M 参数） |
| 输入 | 4 通道 `[R,G,B,D]`，`imgsz=1024` |
| 数据 | 从 `train` 按 `val_ratio=0.2` + `seed=42` 在 stem 级别切分 → 1600 训练 / 400 验证 |
| 训练 | SGD，`lr0=5e-3`（唯一变量：E7 的 `lr0=1e-2` 减半），`batch=8`，`epochs=300`，`patience=80`，AMP |
| **best mAP@0.5:0.95** | **0.51955**（epoch 287） |
| best mAP@0.5 | 0.76883 |
| 权重 | `runs/urban_multimodal_det_e7_lr0half/weights/best.pt` |

| 对照 | E7（lr0=0.01） | F1（lr0=0.005） | Δ |
| :--- | :---: | :---: | :---: |
| best mAP@0.5:0.95 | 0.50878 @ ep257 | 0.51955 @ ep287 | **+0.01077** |
| best mAP@0.5 | 0.75573 | 0.76883 | +0.01310 |
| 末窗口 box gap（分叉） | 0.8453 | 0.8288 | −0.0165 |

关键配置结论：**depth 通道 resize 保持 `INTER_LINEAR`、padding 保持 114**。depth 中的 0 本身带有「无深度信号」语义（JPG 中约 73%、PNG 中约 5.78% 的像素为 0），padding=0 会让模型在图像边缘学到虚假的 depth 梯度；114 是 depth 的离群值，天然充当「忽略此 padding」的哨兵。相关变体（`INTER_NEAREST`、padding=0）均已实验证伪。

## 7.2 评估口径约定

1. **正式实验指标统一以 `model.val()`（训练内置 val）的结果为准**：`conf=0.001`、`iou=0.7`、`max_det=300`、`rect=False`。
2. **`scripts/predict.py` 定位为推理 / 结果检查工具**（产出提交 TXT 与 `submission.zip`、做输出自检），**不作为正式排行榜指标来源**。
3. 两种流程已完成一致性验证：在同一批 398 张有效图 / 2807 个 GT 上，`predict.py` mAP@0.5:0.95 = 0.49898，`model.val()` = 0.49787，**差值 +0.00111**；Precision / Recall / mAP@0.5 / mAP@0.75 差异均在 0.0016 以内。可认为两种评估流程基本一致。

> 📌 **复现说明**：7.2 / 7.3 中属于 `predict.py` 侧的数字，由 `scripts/predict.py` 输出、`scripts/phase2_map.py` 复算而得；其依赖的预测 TXT 位于 `.gitignore` 的 `submissions/` 临时目录，已清理，**需重跑推理才能复算**。完整复现命令见 [`docs/experiment_log.md`](docs/experiment_log.md) 的「复现说明」。`model.val()` 侧数字可用 `scripts/phase2_val_ref.py` 直接复现；E7 训练指标来自 `runs/*/results.csv`，始终可复现。

## 7.3 已探索并明确不再优先的方向（inference-side）

| 方向 | 结果 | 结论 |
| :--- | :--- | :--- |
| **TTA**（原图 + 水平翻转） | mAP@0.5:0.95 **0.48724** vs 无 TTA **0.49898**，**−0.01174** | **未证明有稳定增益，为负向**。有害指纹：Precision 升、Recall 降、mAP@0.75 暴跌 −0.049（双重 NMS 过度抑制）。比赛提交使用无 TTA 结果 |
| **conf 阈值调参** | 0.001 → 0.01 约 +0.026 | 属**纯推理置信阈值调整**，按项目约定**不计入模型提升** |
| **NMS IoU 调参** | IoU 0.6 vs 0.7 仅 **+0.00207** | **低于 +0.005 门槛，不采用**。主线固定 `iou=0.7` |

**结论：不再优先优化 inference-side，也不用修改 conf / IoU / NMS 等推理参数来人为追求更高指标。**

## 7.4 当前优化重点（training-side）：Phase 6-A

E7 训练曲线分析显示：`best` 出现在 epoch 257，此后进入平台期且 train/val 明显分叉（box loss gap 由 0.599 扩大到 0.845）；同时 `close_mosaic=10` 所关闭 mosaic 的**末 10 轮（ep291–300）出现陡降台阶**，均值 0.49303 vs 前 10 轮 0.50433，**净损约 −0.0113**。

因此进行严格单变量 A/B 实验：

```
E7 baseline（close_mosaic = 10）
        ↓
Phase 6-A：唯一变量 close_mosaic: 10 → 0
        ↓
其余训练条件（模型结构 / loss / optimizer / lr / epochs / patience / batch /
imgsz / seed / 数据集与切分 / RGB-Depth 预处理 / 其它数据增强）全部保持 E7 不变
```

| 项目 | 值 |
| :--- | :--- |
| 实验配置 | `configs/train_e7_close_mosaic0.yaml` |
| 输出目录 | `runs/urban_multimodal_det_e7_close_mosaic0/`（独立目录，不覆盖 E7） |
| 启动命令 | `python scripts/train.py --model_config configs/yolo11m_midfusion_rgbd_concat_res.yaml --train_config configs/train_e7_close_mosaic0.yaml` |
| 状态 | **已完成，证伪** |

### 结果与结论

| 指标 | E7（`close_mosaic=10`） | E7m0（`close_mosaic=0`） |
| :--- | :---: | :---: |
| **best mAP@0.5:0.95** | **0.50878** @ ep257 | **0.50878** @ ep257 |
| ep250–290 平台均值 | 0.50493 | 0.50493 |
| **ep291–300 末 10 轮均值** | **0.49303**（陡降） | **0.50457**（持平） |

- **逐 epoch 对照**：ep1–290 完全一致，仅 ep291–300（`close_mosaic=10` 关闭 mosaic 的那 10 轮）不同。
- **权重逐张量对照**：两个 `best.pt` 的 913 个模型张量逐 bit 相同（最大绝对差 0.0）。
- **结论**：`close_mosaic=0` **不提升 mAP**（0.50878 → 0.50878）。最优 epoch 都在 257，远早于 mosaic 关闭点（291），关不关 mosaic 都碰不到 best。Phase 5 的「末 10 轮陡降」确系 `close_mosaic` 造成，但发生在 best 之后，对上报指标无影响。`close_mosaic=0` 只抹平了尾巴曲线，是**曲线形状效果，不是指标提升**。

> ⚠️ 该实验**禁止混入** TTA、conf、IoU、NMS 参数、学习率、patience、epochs、模型结构、数据集等因素，以保证与 E7 的可比性。

## 7.5 实验逻辑链

```
E7 baseline（mAP@0.5:0.95 = 0.50878）
        ↓
predict.py / model.val() 一致性验证（差值 +0.00111）
        ↓
确认评估流程没有造成明显指标偏差
        ↓
TTA 未证明有稳定 mAP@0.5:0.95 增益（实测 −0.01174）
        ↓
不再优先优化 inference-side
        ↓
转向 training-side optimization
        ↓
Phase 6-A：close_mosaic = 0 单变量 A/B 实验（证伪，best 0.50878 不变）
        ↓
Phase 6-B-F1：lr0 减半（0.01 → 0.005）→ 正向 +0.01077，新基线 0.51955
        ↓
Phase 6-B-F2：optimizer SGD → AdamW（待跑）
```
