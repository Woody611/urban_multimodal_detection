# F4 rect 提交版本 · 最终提交说明（提交前冻结）

- **日期**：2026-09-15
- **状态**：**已冻结，未提交线上**（`ready_for_manual_submission`）

---

## 1. 使用权重

- 模型：YOLO11m RGBD Mid-Fusion（F4）
- 权重：`runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt`
- **未修改 checkpoint**。

## 2. 使用预测脚本

- `scripts/predict_rect.py`（新建，未覆盖原 `scripts/predict.py`）
- 复用原 `predict.py` 的 `_to_chw`（4ch [R,G,B,D]）、`_format_lines`（6 位小数归一化写盘）、`_load_yaml`/`_parse_pairs`，保证输出格式与旧提交逐字节兼容。

## 3. 完整运行命令

```bash
python scripts/predict_rect.py \
    --source data/raw/test/visible \
    --mode rect \
    --max_boxes 100 \
    --batch 16 \
    --output diagnostic/rect_pipeline_fix/submission_rect_fixed
```

## 4. 实际推理参数（源自运行日志，非伪造）

| 参数 | 值 |
|---|---|
| imgsz | 1280 |
| rect | true（矩形 letterbox） |
| conf | 0.001 |
| iou | 0.7 |
| max_det | 300 |
| max_boxes | 100 |
| augment / TTA | false（无任何增强/TTA） |
| half | false（float32 推理，`.float()/255`） |
| ratio_pad_explicit | true（scale_boxes 显式传 ratio_pad） |
| 输入通道 | 4ch [R,G,B,D]（visible+depth 合并） |

## 5. rect letterbox 说明

`model.val()` 在 `engine/model.py:598` 强制 `rect=True`，把每张图 letterbox 到 **stride-32 对齐的最小矩形画布**（本测试集全部宽高比 0.5625 → **736×1312**），而非 square 1280×1280。

`predict_rect.py --mode rect` 复刻 val 的两步预处理：
1. `load_image(rect_mode=True)`：长边缩放到 1280（`ceil`，无 scaleup 封顶，小图放大 2×）；
2. rect `LetterBox`（`scaleup=False` + `center=True`）padding 到 736×1312。

内容像素与 square 完全一致，仅画布/填充不同；此差异导致原 square pipeline 过检 +1803 框、mAP 损失 −0.01429。

## 6. ratio_pad 显式传递说明

`LetterBox(image=im)` 只返回图像、**丢弃 ratio_pad**；原 `predict.py:399` 的 `scale_boxes(..., ratio_pad=None)` 会自行重算 `gain=min(imgsz/h0, imgsz/w0)` 并假设 square + 居中 pad —— 改 rect 后会坐标错位。

`predict_rect.py` 显式保存 `ratio_pad = ((r, r), (left, top))`（r = load_image 缩放比，left/top = rect padding 偏移），并传给 `scale_boxes(..., ratio_pad=...)`，坐标精确还原（回归测试 round-trip 误差 0.000px）。

## 7. 测试集与预测量

| 项 | 值 |
|---|---|
| 测试图片数 | 1000（155 小图 360×640 + 845 大图 1080×1920） |
| 预测文件数 | 1000（每图一个同名 TXT） |
| 预测框总数 | 7890 |
| 空文件（无检测） | 4（`000013_020_00000285`、`000014_012_00000001`、`003403`、`hehe_161_000002_021_00000001`，均真实存在、conf=0.001 下无框） |

## 8. 完整性检查结果（全通过）

| 检查项 | 结果 |
|---|---|
| 测试图片数量 = 1000 | ✓ |
| 每张测试图都有预测文件 | ✓（缺失 0） |
| 多余文件 | 0 |
| 空文件 | 4（合法无检测，非错误） |
| 非法 class id | 0（全部 ∈ [0,11]） |
| bbox ∈ [0,1] | ✓（cx/cy/w/h 全部在 [0,1]） |
| confidence ∈ [0,1] | ✓ |
| 每行字段 = `class_id cx cy w h confidence` | ✓（0 行字段数≠6） |
| NaN / Inf | 0 |
| 越界框 | 152 框边缘距 [0,1] 边界 ≤ 5e-7（6 位小数舍入噪声，非真实越界） |
| 重复图片 ID | 0 |
| 文件名大小写问题 | 0 |
| zip 目录结构 | `results/<name>.txt`（1000 条目，0 异常，0 目录条目） |

## 9. zip 路径

```
diagnostic/rect_pipeline_fix/submission_rect_fixed/submission.zip   (297628 bytes, 1000 entries)
```

## 10. 结论

**未提交线上**。本版本已通过全部完整性检查，等待人工确认后提交。所有产物在独立 `diagnostic/rect_pipeline_fix/` 目录，未触碰任何原提交文件 / F4 checkpoint / 训练配置 / 旧实验。
