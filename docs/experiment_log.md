# 实验记录

本文档集中记录项目各轮实验的配置、结果与分析。每完成一轮完整训练实验，更新下方表格并在对应小节补充详细记录。

> 🔖 **当前状态速览（2026-09-13）**
> - **当前主线基线 = E7**（YOLO11m + RGBD 中期融合 + imgsz=1024），`best mAP@0.5:0.95 = 0.50878`（epoch 257）。
> - **正式实验指标统一以 `model.val()` 为准**；`scripts/predict.py` 定位为推理/结果检查工具，不作为正式指标来源。二者一致性验证见下文「E7 评估流程一致性验证」。
> - **推理侧优化（TTA / conf / NMS IoU）已探索完毕并明确不再优先**：TTA 未证明有稳定增益，后处理调参不计入模型提升。详见对应小节。
> - **当前优化重点已转向训练阶段**，正在执行 Phase 6-A 单变量 A/B 实验（`close_mosaic` 10 → 0），见文末。
> - 上述数字均取自 `runs/*/results.csv` 实际产物，非估算值。

> ⚠️ **代码迁移说明**：Exp-001 在项目早期的「自定义模型」代码库上完成，现已迁移到 ultralytics/YOLOv11 方案。下文引用的 `models/baseline.py`、`BaselineDetector`、`DetectionLoss`、`ModelEMA`、`_bn_initialized` 等文件/符号已随迁移移除，仅作历史记录保留；实验结果数据与结论仍有效。当前实现见 `README.md` 与 `docs/model_design.md`。

## 实验记录表

| 实验编号 | 日期 | 模型版本 | 输入模态 | 主要修改 | 训练配置 | mAP@0.5 | mAP@0.5:0.95 | 结论 | 备注 |
| :--- | :---: | :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| Exp-001 | 2026-08-19 ~ 08-21 | Baseline（CSPDarknet-s + PAN + DecoupledHead） | RGB（Visible） | 建立单模态基准，跑通完整 pipeline | lr=1e-2 / bs=16 / 300 ep / SGD+Cos / AMP+EMA | 0.1512 | 0.0809 | 整体 mAP 偏低；原始模型优于 EMA；boat、tricycle 的 AP=0 | EMA 结果受 BN 同步 bug 影响 |

### 实验记录表（YOLOv11 / E 系列，2026-09-10 ~ 2026-09-12）

迁移到 ultralytics YOLOv11 fork 后的主线实验。全部指标取自各实验 `runs/<实验名>/results.csv` 中 **`best` 权重所在 epoch**（训练内置 `model.val()`），`best epoch` 为该 mAP@0.5:0.95 最大值的轮次。

| 实验编号 | 模型 | 输入模态 | 关键变量 | best epoch | mAP@0.5 | mAP@0.5:0.95 | 结论 |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| E0 | YOLO11n | RGB | 建立 YOLOv11 单模态基准 | 236 | 0.55653 | 0.32286 | 迁移方案跑通，作为后续基线起点 |
| E1 | YOLO11s | RGB | 模型规模 n → s | 288 | 0.69085 | 0.44069 | 规模提升带来大幅增益 |
| E2 | YOLO11s | RGB + Depth | 引入 Depth 第二模态（RGBD 4ch） | 238 | 0.70672 | 0.45419 | 双模态优于单模态（+0.01350） |
| E3 | YOLO11s | RGB + Depth | 在 E2（s 规模）基础上增加 P2 检测层 | 227 | 0.63593 | 0.37960 | **负向**，相对 E2 −0.07459，未采用 |
| E4 | YOLO11m | RGB + Depth | 模型规模 s → m | 286 | 0.73059 | 0.47783 | 规模继续带来增益 |
| E5 | YOLO11m | RGB + Depth | 在 E4（m 规模）基础上增加 P2 检测层 | 264 | 0.73234 | 0.47796 | 相对 E4 仅 +0.00013，无实质增益 |
| E6 | YOLO11m | RGB + Depth | 输入分辨率 640 → 768 | 213 | 0.73566 | 0.48893 | 分辨率提升有效（相对 E4/E5 均 640：+0.01110） |
| **E7** | **YOLO11m** | **RGB + Depth** | **输入分辨率 768 → 1024** | **257** | **0.75573** | **0.50878** | **当前主线基线，分辨率提升继续有效（+0.01985）** |
| E8 | YOLO11m | RGB + Depth | depth resize INTER_LINEAR → INTER_NEAREST | 228 | 0.75482 | 0.50109 | **证伪**，相对 E7 −0.00769 |
| E9 | YOLO11m | RGB + Depth | depth padding 114 → 0 | 273 | 0.75505 | 0.50521 | **证伪**，相对 E7 −0.00357 |

**主线结论**：

- **模型规模**：n → s → m 逐级带来增益（E0 → E1 → E4）；
- **模态**：RGBD 双模态优于 RGB 单模态（E1 vs E2，同 s 规模下 +0.01350）；
- **输入分辨率**：640 → 768 → 1024 逐级有效（E4/E5 → E6 → E7），是 E 系列后期最主要的增益来源；
- **P2 检测层被两次独立证伪**：s 规模（E3）大幅负向 −0.07459，m 规模（E5）仅 +0.00013。**P2 不带来增益**。

> 注：E0–E3 的 `args.yaml` 未随产物保留（仅存 `results.csv` 与部分训练日志），上表中的模型/输入/变量依据各实验日志中的 `engine/trainer:` 配置行与运行目录名核对；E0 无日志，其定义依据同构的 E1（`configs/yolo11s_visible.yaml`）反推为 n 规模 RGB。

**E8 / E9 证伪要点**（两者指纹同构：mAP50 几乎不动、mAP75 大跌、P 降、R 升）：

- **E8（NEAREST 下采样）**：1080×1920 → 576×1024 的最近邻下采样产生块状锯齿，损害高 IoU 定位，mAP75 −0.047。
- **E9（depth padding=0）**：**depth 中的 0 本身有语义**（JPG 中 73%、PNG 中 5.78% 的像素为 0，表示无深度信号/背景）。padding=0 使填充区与合法背景无法区分，模型在图像边缘学到虚假的 depth 梯度；114 是 depth 的离群值，天然充当「忽略此 padding」的哨兵。
- **结论：depth resize 保持 INTER_LINEAR、padding 保持 114 是正确配置，不是 bug。** 后续单变量实验必须以 E7 = 0.50878 为唯一基准。

---

## E7 评估流程一致性验证（`predict.py` vs `model.val()`）

### 1. 动机

`scripts/predict.py` 走的是「自包含推理循环」——直接用 fork 的 `LoadImagesAndVideos` 读多模态图、`DetectionModel` 前向、`non_max_suppression` + `scale_boxes` 解码，再自行写提交 TXT；**不经过** `model.predict()` / `model.val()`（后者会受 checkpoint args 残留影响，且 predictor 的 `setup_source` 未转发 `pairs_rgb_depth`）。因此必须验证：**这套独立推理链路与训练内置 val 是否等价**，否则无法用 predict 的结果判断提交质量。

### 2. 方法

以 E7 `best.pt` 为唯一权重，在同一批 val split 数据上分别跑两条链路，用同一套指标原语（`DetMetrics` + `box_iou` + 与 `DetectionValidator.match_predictions` 完全一致的类感知贪心匹配）计算 mAP：

- 链路 A：`scripts/predict.py`（输出 TXT 后由 `scripts/phase2_map.py` 复算指标）；
- 链路 B：`scripts/phase2_val_ref.py` 调用 `model.val()`。

评估参数对齐：`conf=0.001`、`iou=0.7`、`max_det=300`、`imgsz=1024`、`rect=False`、`agnostic_nms=False`、`use_simotm=RGBD`、4 通道。

**对齐前先修平了两处口径差异**：
1. `predict.py` 的 `non_max_suppression` 显式加 `multi_label=True`（与 val 一致；原默认 `False`）；
2. 指标脚本复刻 `verify_image_label` 的越界检查，跳过 corrupt 图像（否则会多算 GT 实例）。

### 3. 结果（同一批 **398 张有效图 / 2807 个 GT**）

| 指标 | `predict.py` | `model.val()` | 差值 |
| :--- | :---: | :---: | :---: |
| Precision (B) | 0.87026 | 0.87036 | −0.00010 |
| Recall (B) | 0.68778 | 0.68786 | −0.00008 |
| mAP@0.5 | 0.74813 | 0.74705 | +0.00108 |
| mAP@0.75 | 0.54350 | 0.54197 | +0.00153 |
| **mAP@0.5:0.95** | **0.49898** | **0.49787** | **+0.00111** |

补充事实：
- val split 共 400 张，其中 **2 张因标注越界被剔除**（`000050.png`、`003817.png`，坐标 1.0034 / 1.0019），两条链路均处理 398 张、2807 个 GT，GT 数完全一致；
- `predict.py` 侧共 7908 个预测框；其中 **9 张图的框数正好被 `max_boxes=100` 截断**（单图最大 268 框），是 `predict.py` 与 val 的残余差异来源之一；
- 已验证等价的环节：LetterBox 几何、通道顺序 `[R,G,B,D]`、/255 归一化、depth 归一化、`scale_boxes` 坐标还原、`conf`/`iou`/`max_det`/`agnostic`。

### 4. 结论

**`predict.py` 与 `model.val()` 两种评估流程基本一致**：mAP@0.5:0.95 差值 +0.00111，P/R/mAP50/mAP75 差异均在 0.0016 以内，**可以认为评估流程未造成明显指标偏差**。残余差异可归因于两个方向相反的已知因素（`predict.py` 的 `max_boxes=100` 截断压低指标、小图 `scaleup=True` 抬高指标），非 bug。

📌 **复现说明（重要）**

本节以及下文「TTA 实验」「推理侧后处理调参」三节中**属于 `predict.py` 侧的数字**，均由 `scripts/predict.py` 生成的预测 TXT（`submissions/predict_*/results/*.txt`）经 `scripts/phase2_map.py` 复算得到。该目录属 `.gitignore` 的临时产物，**已清理**，因此这些数字**目前无法直接复算**，需按下述命令重跑推理（CPU 单次约 20 分钟，`--tta` 约 40 分钟）：

```bash
# 一致性验证（无 TTA）
python scripts/predict.py --source data/processed/depth_split_train/images/val/visible \
    --output_dir submissions/phase2
python scripts/phase2_map.py --results submissions/phase2/predict_<时间戳>/results

# TTA 实验（仅多一个 --tta）
python scripts/predict.py --source data/processed/depth_split_train/images/val/visible \
    --output_dir submissions/phase3 --tta
python scripts/phase2_map.py --results submissions/phase3/predict_<时间戳>/results

# conf 扫描（离线即可，无需重跑推理；IoU 扫描需用 --iou 重跑 predict）
python scripts/phase2_map.py --results <上面的 results 目录> --min_conf 0.005
```

`model.val()` 侧的数字不受影响，可直接用 `scripts/phase2_val_ref.py` 复现。E7 训练指标（0.50878 等）来自 `runs/*/results.csv`，始终可复现。

---

## TTA 实验（已证伪，不采用）

### 1. 方法

在 `scripts/predict.py` 上启用 `--tta`：原图 + 水平翻转两次推理，翻转结果水平还原后按 IoU=0.7 做逐类 NMS 合并。其余条件与一致性验证完全一致（E7 `best.pt`、398 张有效图 / 2807 GT、`conf=0.001`、`iou=0.7`）。

### 2. 结果

| 指标 | 无 TTA | TTA | 差值 |
| :--- | :---: | :---: | :---: |
| Precision (B) | 0.87026 | 0.89050 | +0.02024 |
| Recall (B) | 0.68778 | 0.66200 | −0.02578 |
| mAP@0.5 | 0.74813 | 0.74528 | −0.00285 |
| mAP@0.75 | 0.54350 | 0.49456 | **−0.04894** |
| **mAP@0.5:0.95** | **0.49898** | **0.48724** | **−0.01174** |

### 3. 结论

**TTA 未能提升 mAP@0.5:0.95，实测为负向（−0.01174）**，且高于「值得采用」的 +0.005 门槛的反面。有害指纹明确：Precision 升、Recall 降、**mAP@0.75 暴跌（−0.049）**，属典型「过度抑制」——`predict.py` 的 TTA 是两次独立 NMS 后再做一次逐类 NMS 合并（双重 NMS），把本应保留的高 IoU 正样本误删，尤其损害定位精度。此外 TTA 使框数增至 10713，`max_boxes=100` 截断的图从 9 张增至 15 张（单图最大 417 框）。

**因此：TTA 不作为主要优化方向，比赛提交使用无 TTA 的 `predict.py` 结果。**

> 📌 **复现说明**：上表 TTA 侧数字来自 `submissions/` 下已清理的预测 TXT，重跑方式见上一节的「复现说明」（`--tta` 分支）。

---

## 推理侧后处理调参（conf / NMS IoU）—— 不计入模型提升

在 E7 的 predict 结果上离线做后处理扫描（`conf` 可离线复现，因为按置信度降序的 NMS 中低置信框永不压制高置信框，故「提高 conf 阈值」与「对已存结果做后过滤」严格等价）：

| NMS IoU \ conf | 0.001 | 0.003 | 0.005 | 0.01 |
| :---: | :---: | :---: | :---: | :---: |
| 0.6 | 0.50105 | 0.51355 | 0.51973 | **0.52818** |
| **0.7（E7 口径）** | 0.49898 | 0.51115 | 0.51719 | 0.52542 |
| 0.8 | 0.49577 | 0.50697 | 0.51258 | 0.52093 |

（表内为 mAP@0.5:0.95）

- **conf 项**：随阈值升高单调上升（0.001 → 0.01 约 +0.026）。**但这是纯粹的推理置信阈值调整，按项目约束「推理置信阈值调整不属于 mAP 提升方案」，不计入模型提升。**
- **NMS IoU 项**：在 `conf=0.001` 口径下，IoU 0.6 / 0.7 / 0.8 → 0.50105 / 0.49898 / 0.49577。IoU=0.6 相对 0.7 仅 **+0.00207**，**低于 +0.005 门槛，不采用**。

**结论：不通过修改 conf / IoU / NMS 等推理参数来人为追求更高指标。** 主线固定为与训练内置 val 同口径的 `conf=0.001` + `iou=0.7`。

> 📌 **复现说明**：上表同样基于 `submissions/` 下已清理的预测 TXT；`conf` 列为离线过滤（`--min_conf`），`IoU` 行需用 `--iou` 重跑 `predict.py`。详见上文的「复现说明」。

---

## 评估口径约定（2026-09-13 起生效）

1. **正式实验指标统一以 `model.val()` 的结果为准**（训练内置 val，`conf=0.001`、`iou=0.7`、`max_det=300`、`rect=False`）。
2. **`scripts/predict.py` 定位为推理 / 结果检查工具**，用于产出提交文件与做输出自检，**不作为正式排行榜指标的来源**。
3. 已通过一致性验证：两者 mAP@0.5:0.95 差值 +0.00111，可比。

---

## Phase 6-A：`close_mosaic` 单变量 A/B 实验（已完成，证伪）

### 1. 动机

对 E7 训练曲线做只读分析后发现：

| 区间 | mAP@0.5:0.95 均值 |
| :--- | :---: |
| ep281–290（mosaic 开启） | 0.50433 |
| **ep291–300（`close_mosaic=10` 关闭 mosaic）** | **0.49303** |

ep290 → ep291 出现陡降台阶（0.50326 → 0.49453），恰好落在 `close_mosaic=10` 触发的「关闭 mosaic」边界上，**末 10 轮净损约 −0.0113**。同时 E7 后半程已进入平台期且 train/val 明显分叉（box loss gap 由 0.599 扩大到 0.845；val/cls_loss 最低点在 ep112、val/dfl_loss 在 ep78 后持续回升），表明延长训练无收益，需从**训练阶段**寻找改进点。

### 2. 唯一变量

```
close_mosaic: 10 → 0
```

含义：整个训练过程都不关闭 Mosaic。**其余训练条件全部保持 E7 不变**，包括但不限于：模型结构（YOLO11m RGBD 中期融合，P3/P4/P5）、loss、optimizer（SGD + momentum 0.937 + wd 5e-4）、学习率（lr0=1e-2，CosineAnnealingLR）、`epochs=300`、`patience=80`、`batch=8`、`imgsz=1024`、`seed=42`、数据集与 train/val 切分（`val_ratio=0.2`、`seed=42`）、RGB/Depth 预处理、所有其它数据增强。

**明确禁止混入该实验的变量**：TTA、conf、IoU、NMS 参数、学习率、patience、epochs、模型结构、数据集、数据增强（除 `close_mosaic` 本身）。

实验配置：`configs/train_e7_close_mosaic0.yaml`
输出目录：`runs/urban_multimodal_det_e7_close_mosaic0/`（独立目录，不覆盖 E7）
启动命令：

```bash
python scripts/train.py \
    --model_config configs/yolo11m_midfusion_rgbd_concat_res.yaml \
    --train_config configs/train_e7_close_mosaic0.yaml
```

### 3. 结果（2026-09-13，从零重跑 300 epoch，`resume: false`）

> 之前多次启动失败的根因已查明：**GPU 上残留了多个训练进程（17.51 GiB + 14.19 GiB）把 31.73 GiB 显存占满**（`torch.cuda.OutOfMemoryError`，仅剩 22 MiB 空闲），且多个进程并发写同一 `run` 目录导致 `results.csv` 出现重复 epoch 行、`last.pt`/`epoch*.pt` 被互相覆盖。清理残留进程后从零重跑，一次跑满 300 epoch 无异常。

| 指标 | E7（`close_mosaic=10`） | E7m0（`close_mosaic=0`） |
| :--- | :---: | :---: |
| **best mAP@0.5:0.95** | **0.50878** @ ep257 | **0.50878** @ ep257 |
| best mAP@0.5 | 0.75573 | 0.75573 |
| ep250–290 平台均值 | 0.50493 | 0.50493 |
| **ep291–300 末 10 轮均值** | **0.49303**（陡降） | **0.50457**（持平） |

**逐 epoch 对照**：ep1–290 完全一致（0 差异），仅 ep291–300 这 10 轮不同——恰好对应 `close_mosaic=10` 关闭 mosaic 的那 10 轮。

**权重逐张量对照**：两个 `best.pt` 的 913 个模型张量，**0 个有差异、最大绝对差 = 0.0**（逐 bit 相同）。唯一差异是 checkpoint 元数据（`name`、`close_mosaic` 两个字段）。

### 4. 结论：证伪，`close_mosaic` 不提升 mAP

1. **best 指标不变**：0.50878 → 0.50878。最优 epoch 都是 257，**远早于 mosaic 关闭点（291）**，关不关 mosaic 都碰不到 best。
2. Phase 5 观察到的「末 10 轮陡降」确系 `close_mosaic=10` 造成（0.50493→0.49303），但它发生在 best 之后，**对上报指标无影响**。
3. `close_mosaic=0` 只是把尾巴曲线抹平（0.50457 持平），属**曲线形状的化妆品效果**，不是指标提升。

**一句话**：`close_mosaic` 不是 mAP 的瓶颈——最优解早在 ep257 就出现。这条优化路径（关闭 mosaic）走到头了，训练侧需要另找改进点。

---

## 实验逻辑链（当前阶段）

```
E7 baseline（mAP@0.5:0.95 = 0.50878，主干基线）
        ↓
predict.py / model.val() 一致性验证（差值 +0.00111）
        ↓
确认评估流程没有造成明显指标偏差
   → 正式指标以 model.val() 为准，predict.py 仅作推理/检查工具
        ↓
TTA 未证明有稳定 mAP@0.5:0.95 增益（实测 −0.01174）
   → 后处理调参（conf / IoU）亦不计入模型提升
        ↓
不再优先优化 inference-side
        ↓
转向 training-side optimization
        ↓
Phase 6-A：close_mosaic = 0 单变量 A/B 实验（证伪，best 0.50878 不变）
```

---

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
