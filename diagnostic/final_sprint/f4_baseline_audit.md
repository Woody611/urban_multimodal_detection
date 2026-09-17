# F4 基准审计报告（最终冲刺 · 第一任务）

**日期**：2026-09-16
**范围**：只做基准审计，不训练任何新模型。
**结论**：F4 best.pt 确认无误；统一验证口径三源一致；submission pipeline 已冻结且校验通过。

---

## 1. F4 best.pt 确认

| 项 | 值 |
|---|---|
| 权重路径 | `runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt` |
| 模型 | `configs/yolo11m_midfusion_rgbd_concat_res.yaml`（YOLO11m RGBD Mid-Fusion，30.34M / 456.83 GFLOPs@1280） |
| best epoch | **251**（fitness 最优 = mAP50-95 最大，同一 epoch） |
| results.csv best mAP50-95 | **0.53273** |
| args.yaml 关键字段 | `box=7.5 / cls=0.5 / dfl=1.5 / batch=8 / imgsz=1280 / epochs=300 / seed=42 / nbs=64 / lr0=0.005 / lrf=0.01 / SGD(mom=0.937,wd=5e-4) / cos_lr / warmup_epochs=3 / amp / close_mosaic=10 / use_simotm=RGBD / channels=4 / pretrained=yolo11m.pt` |

---

## 2. F4 统一验证（三源口径，互相对照）

统一口径：`imgsz=1280`、`rect=True`（model.val() 强制）、同一 val 集（`depth_split_train`，398 图 / 2807 实例）、fork DetMetrics、`conf=0.001 / iou=0.7 / max_det=300`。

### 2.1 总体指标

| 指标 | 训练内置 val（results.csv ep251，batch=8） | rect 独立复现（predict_rect + phase2_map，batch=16） | 逐类重跑（f4_val_per_class，batch=4） |
|---|---|---|---|
| mAP50-95 | **0.53273** | 0.53264 | ≈0.533 |
| mAP50 | 0.78026 | 0.77863 | — |
| mAP75 | 0.59124 | 0.58069 | ≈0.581 |
| Precision | 0.84029 | 0.83948 | — |
| Recall | 0.72177 | 0.72090 | — |
| F1 | 0.77653 | 0.77568 | — |

**口径结论**：
- **mAP50-95 为稳健主口径**：三源差 ≤0.0003（batch 无关、rect 无关），是跨实验可比的唯一可靠指标。
- **mAP75 有 ~0.01 的 batch 伪差**（batch=8 → 0.59124，batch=4/16 → 0.58069），逐类/独立复现的 mAP75 仅作参考，不以 batch≠8 的 mAP75 做跨实验比较。
- rect 独立复现训练 val 到 **Δ=0.00009**，证明 submission pipeline 的预测与 model.val() 逐框一致（5152 框与 val save_txt 完全一致）。

### 2.2 逐类 AP（batch=4 重跑口径，`reports/f4_class_metrics.csv`）

| class | images | instances | P | R | AP50 | AP75 | AP50-95 |
|---|---|---|---|---|---|---|---|
| person | 219 | 1045 | 0.827 | 0.699 | 0.784 | 0.510 | 0.490 |
| boat | 14 | 28 | 0.662 | 0.698 | 0.667 | 0.416 | 0.430 |
| animal | 184 | 731 | 0.886 | 0.726 | 0.837 | 0.595 | 0.552 |
| seat | 51 | 107 | 0.878 | 0.738 | 0.811 | 0.757 | 0.688 |
| sign | 83 | 150 | 0.822 | 0.580 | 0.674 | 0.443 | 0.430 |
| bicycle | 43 | 106 | 0.753 | 0.519 | 0.624 | 0.491 | 0.424 |
| car | 69 | 288 | 0.807 | 0.771 | 0.808 | 0.578 | 0.530 |
| ball | 17 | 19 | 0.915 | 0.566 | 0.678 | 0.430 | 0.427 |
| light | 85 | 244 | 0.759 | 0.713 | 0.700 | 0.576 | 0.510 |
| garbage_can | 46 | 57 | 0.954 | 0.702 | 0.815 | 0.564 | 0.559 |
| uav | 23 | 30 | 0.904 | 0.939 | 0.951 | 0.615 | 0.555 |
| tricycle | 1 | 2 | 0.905 | 1.000 | 0.995 | 0.995 | 0.798 |

**困难类（AP50-95 低）**：bicycle 0.424、ball 0.427、sign 0.430、boat 0.430、person 0.490 —— 正是小目标类（4/5 为诊断已确认的硬漏检小目标），tricycle 仅 2 实例无统计意义。

---

## 3. Submission pipeline 检查（已冻结，未改动）

| 环节 | 状态 | 说明 |
|---|---|---|
| RGB/Depth 预处理 | ✅ | `predict_rect.py::_preprocess_rect`：4ch [B,G,R,D] 合并 → load_image 长边 1280（ceil，无封顶）→ rect LetterBox（scaleup=False） |
| rect 画布 | ✅ | stride-32 对齐最小矩形，ar=0.5625 → 736×1312（`_compute_rect_shape` 复刻 set_rectangle 公式，回归测试全 PASS） |
| ratio_pad | ✅ | 显式 `((r,r),(left,top))` 传入 scale_boxes（修复了 `scale_boxes(ratio_pad=None)` 重算的 latent bug） |
| 坐标还原 | ✅ | rect 路径 `scale_boxes(..., ratio_pad=真实值)`，round-trip 误差 0.000px |
| NMS | ✅ | `non_max_suppression(conf=0.001, iou=0.7, max_det=300, multi_label=True)` |
| confidence | ✅ | conf=0.001（与 val 一致，不截断低置信） |
| submission zip | ✅ | `submission_rect_fixed/submission.zip`（1000 entries / 297KB / 7890 框 / 4 空文件 / 0 格式错误 / 0 越界 / 0 类别越界） |

**冻结状态**：`scripts/predict_rect.py` 与 `scripts/predict.py` 均未改动（predict.py 仅存在 .bak 副本，工作副本未变）。manifest `frozen_at: 2026-09-15`。**未发现需修复的 bug。**

---

## 4. 本地 vs 线上（关键现实检查）

| 口径 | mAP50-95 |
|---|---|
| 本地训练 val（主口径） | 0.53273 |
| 本地 rect 独立复现 | 0.53264 |
| 线上 rect 提交（评分 51.0510） | ≈0.5105 |
| **本地→线上 gap** | **≈0.022** |

- 目标线上 **0.570** 按当前 gap（≈0.022）外推，需本地主口径达到 **≈0.59** —— 这是一个很大的跨距（F4 本地 0.533，需 +0.057）。
- ⚠️ 简报中「当前线上约 0.516」与已确认的评分 51.0510（=0.5105）存在 0.005 出入，请以实际榜单为准；后续实验的「本地→线上」换算需用同一次提交的实测 gap 校准（见简报第十一部分）。

---

## 5. 审计结论与待确认

1. **F4 best.pt = ep251，本地主口径 mAP50-95 = 0.53273**（本报告所有后续对比的基准）。
2. **统一验证口径已固定**：跨实验只比较「训练内置 val 的 mAP50-95（results.csv）」，逐类用 `f4_val_per_class.py`（batch=4）口径，mAP75 不跨 batch 比较。
3. **submission pipeline 已冻结且完整**，无 bug，无需改动。
4. 后续 IR 快速可行性实验（RGB+IR+Depth 5ch early fusion）的**唯一判据 = 训练内置 val mAP50-95 相对 0.53273**。

**本报告完成后：停止，不训练。**（等待指示是否进入第二任务 IR 可行性实验。）
