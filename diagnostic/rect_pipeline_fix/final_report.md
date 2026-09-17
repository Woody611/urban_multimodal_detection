# F4 提交推理 Pipeline 修复 · 最终报告（阶段一~八）

- **日期**：2026-09-15
- **模型**：F4 = YOLO11m RGBD Mid-Fusion @1280（`runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt`）
- **任务**：修复 F4 提交推理 pipeline（rect letterbox + 显式 ratio_pad），完成线上复测前审计
- **约束遵守**：未训练 F6、未改模型结构、未改 F4 checkpoint、未删旧实验、未覆盖 scripts/predict.py、未提交线上

---

## 0. 一句话结论

`model.val()` 强制 `rect=True`（矩形 736×1312 letterbox + 显式 ratio_pad），而原 `scripts/predict.py` 用 square 1280×1280 + 丢弃 ratio_pad。新建 `scripts/predict_rect.py` 复刻 val 的 rect 路径后，**本地 val mAP50-95 从 square 的 0.51835 恢复到 0.53264（+0.01429），精确复现 model.val() 0.53273（Δ=0.00009）**。

---

## 1. 各阶段产物

| 阶段 | 内容 | 产物 |
|---|---|---|
| 一 | 只读代码检查 | `diagnostic/pipeline_code_path_report.md` |
| 二 | 最小侵入 rect letterbox 修复 | `scripts/predict_rect.py`（新文件，不覆盖 predict.py） |
| 三 | 独立诊断目录 | `diagnostic/rect_pipeline_fix/`（含 code_diff_summary.md 等） |
| 四 | 严格 A/B | `validation_rect_metrics.json` / `validation_square_metrics.json` |
| 五 | 逐图一致性 | `per_image_comparison.csv` + `rect_vs_square_report.md` |
| 六 | 回归测试 | `scripts/regression_rect_letterbox.py` + `regression_test.log`（全 PASS） |
| 七 | 提交生成（不提交） | `submission_rect_fixed/` + `submission_manifest.json` |
| 八 | 本报告 | `final_report.md` |

---

## 2. 修改文件清单

**新增（未改动任何原提交文件）**：
- `scripts/predict_rect.py` — rect letterbox 修复版推理（A/B + 提交生成）
- `scripts/compare_rect_square.py` — 逐图一致性对比工具
- `scripts/regression_rect_letterbox.py` — 回归测试
- `diagnostic/rect_pipeline_fix/` — 全部诊断产物（独立目录）

**未修改**：`scripts/predict.py`、F4 checkpoint、训练配置、所有旧实验。

---

## 3. 阶段四 A/B 结果（val 集，fork 指标，conf=0.001/iou=0.7/max_det=300/max_boxes=300）

| 指标 | rect（修复） | square（原 pipeline） | Δ |
|---|---|---|---|
| **mAP50-95** | **0.53264** | 0.51835 | **+0.01429** |
| mAP50 | 0.77863 | 0.76720 | +0.01143 |
| mAP75 | 0.58069 | 0.56000 | +0.02069 |
| Precision | 0.83948 | 0.83657 | +0.00291 |
| Recall | 0.72090 | 0.70990 | +0.01100 |
| 预测框数 | 5152 | 6955 | −1803 |

**关键**：
- rect 预测框数 5152 与 val save_txt **完全一致**（394/394 框数一致，坐标到 1e-6）。
- square 过检 +1803 框，mAP 损失 −0.01429 —— 与审计预测的 −0.0141 一致，修复完全兑现。

## 4. 阶段五 逐图一致性（28 张图覆盖 7/8 维度）

| 维度 | 命中 | 结论 |
|---|---|---|
| 小图 360×640 | 27 | rect 框数 ≤ square（如 00000095：11 vs 21） |
| 大图 1080×1920 | 373 | rect 明显更少（如 000003_080_00000409：24 vs 38） |
| 宽图（横向） | 400 | val 集 100% 横向 |
| 高图（纵向） | 0 | val 集无纵向图（回归测试已覆盖） |
| 密集 GT≥8 | 130 | rect < square |
| 稀疏 GT≤2 | 113 | 多数 rect ≤ square |
| 无效 Depth | 80 | 全在小图 |
| 小目标 | 265 | rect < square |

汇总：28 图 rect 总框数 **212** vs square **316**（Δ −104，平均 −3.71/图），方向一致，无反向异常。

## 5. 阶段六 回归测试（全 PASS）

- 矩形画布 stride-32 对齐（大图/小图/纵向/宽图/方图，全部 `(736,1312)`/`(1312,736)`/`(672,1312)` 等）。
- 坐标 round-trip **0.000px**（大图/小图/纵向/宽图，中心框/角落框）。
- RGB/Depth 共享同一 ratio/pad（4ch 合并）。
- square scaleup=False 暴露 `scale_boxes(ratio_pad=None)` 重算 latent bug（gain=2.0 vs 真实 1.0）—— 印证 rect 路径必须显式传 ratio_pad。

## 6. 阶段七 提交生成（未提交）

- 命令：`python scripts/predict_rect.py --source data/raw/test/visible --mode rect --max_boxes 100`
- 输出：`diagnostic/rect_pipeline_fix/submission_rect_fixed/results/*.txt`（1000 张测试图，7890 框）
- 校验：1000 文件、0 坐标越界、0 class_id 越界、0 行格式错误、0 图被 max_boxes=100 截断
- 打包：`submission_rect_fixed/submission.zip`（1000 entries，297 KB）
- 参数：imgsz=1280, conf=0.001, iou=0.7, max_det=300, max_boxes=100, augment=False, half=False
- 状态：**已生成，未提交线上**（见 submission_manifest.json）

## 7. 阶段八 结论与建议

1. **修复有效且无副作用**：rect letterbox + 显式 ratio_pad 精确复现 model.val()（0.53264 ≈ 0.53273），比原 square pipeline 高 **+0.01429**（fork 口径）。
2. **线上预期**：审计三分解中「指标口径 −0.0308（不可修）+ rect −0.0141（已修）+ 测试分布 +0.0140」。修复后官方口径预期 ≈ 0.48787 + 0.0141 ≈ 0.502，考虑测试集小图占比更高（15.5% vs 6.75%，放大 2× 更受益于 rect），**线上预期 0.502~0.516**。
3. **建议**：确认后提交 `submission_rect_fixed` 的 zip；本报告所有产物均在独立目录，未触碰任何原提交文件。

---

## 附：8 阶段回复速览（供确认）

- **修改文件**：新增 `scripts/predict_rect.py`（+ 2 个诊断脚本），未改 predict.py/checkpoint/训练配置
- **A/B结果**：rect 0.53264 vs square 0.51835（Δ +0.01429，fork 口径）
- **rect mAP50-95**：0.53264（≈ model.val() 0.53273）
- **square mAP50-95**：0.51835
- **是否复现 model.val**：是（Δ=0.00009）
- **是否生成正式提交**：是（submission_rect_fixed，max_boxes=100，未提交）
- **是否建议确认后提交**：建议确认后提交
