# 修复代码差异摘要（阶段二/三）

- **日期**：2026-09-15
- **模型**：F4 = YOLO11m RGBD Mid-Fusion @1280（`runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt`）
- **原脚本**：`scripts/predict.py`（未修改）
- **新脚本**：`scripts/predict_rect.py`（独立新增，最小侵入，不覆盖 predict.py）

---

## 1. 只改了什么

| 维度 | predict.py（原） | predict_rect.py（修复） |
|---|---|---|
| letterbox 目标 | square `(1280,1280)` | **rect**（stride=32 对齐最小矩形，本集 ar=0.5625 → **736×1312**） |
| 预处理 | `LetterBox(new_shape=(imgsz,imgsz), scaleup=True)` | 复刻 `model.val()` 两步：`load_image(rect_mode=True)` 长边→1280（ceil、无封顶）+ rect `LetterBox(scaleup=False)` |
| ratio_pad | **丢弃**（`letterbox(image=im)` 只返回 img） | **真实保留并显式传入** `scale_boxes` |
| scale_boxes | `scale_boxes(..., orig_shape)`（自行重算 gain/pad） | `scale_boxes(..., orig_shape, ratio_pad=真实值)` |
| 批推理 | 每张图独立（batch 内同 shape 假设成立） | 按**画布 shape 分组**后 stack（同宽高比同 rect shape） |
| 其余（conf/iou/imgsz/max_det/通道/写盘） | 不变 | **逐字节复用**（`from predict import _to_chw, _format_lines, ...`） |

## 2. 关键代码差异

### 2.1 预处理（rect）

```python
# _preprocess_rect：load_image(rect_mode=True) + rect LetterBox
h0, w0 = im.shape[:2]
r = imgsz / max(h0, w0)                       # 长边 → 1280
w = min(math.ceil(w0 * r), imgsz)             # ceil（非 round）
h = min(math.ceil(h0 * r), imgsz)
im = cv2.resize(im, (w, h), cv2.INTER_LINEAR) # 无 scaleup 封顶（小图放大 2×）

rect_h, rect_w = _compute_rect_shape(h0, w0, imgsz, stride)  # set_rectangle 公式
ratio = min(rect_h / h, rect_w / w); ratio = min(ratio, 1.0)  # scaleup=False
# ... center padding 到 (rect_h, rect_w) ...
ratio_pad = ((r, r), (left, top))             # 真实 ratio_pad
```

### 2.2 矩形画布公式（复刻 base.py set_rectangle）

```python
def _compute_rect_shape(h0, w0, imgsz, stride, pad=0.5):
    ar = h0 / w0
    shape = [ar, 1.0] if ar < 1 else [1.0, 1.0 / ar]
    rect_h = int(np.ceil(shape[0] * imgsz / stride + pad)) * stride
    rect_w = int(np.ceil(shape[1] * imgsz / stride + pad)) * stride
    return rect_h, rect_w
```

验证：ar=0.5625（1080/1920、360/640）→ `ceil(0.5625*1280/32+0.5)=ceil(23.0)=23` → 736；`ceil(1.0*1280/32+0.5)=ceil(40.5)=41` → 1312。✓

### 2.3 scale_boxes（rect 模式显式传 ratio_pad）

```python
# 复刻 val.py:121-127
o[:, :4] = scale_boxes(x.shape[2:], o[:, :4], orig_hw, ratio_pad=ratio_pad)
# square 对照组则复刻 predict.py:399（不传 ratio_pad）
```

## 3. 为什么必须显式传 ratio_pad

- `scale_boxes(ratio_pad=None)` 会重算 `gain=min(imgsz/h0, imgsz/w0)` 且假设 **scaleup=True + square + 居中 pad**。
- 改 rect 后：画布是 736×1312，内容 720×1280 落在 (top=8, left=16)，重算逻辑会**错误**地把 736×1312 当正方形、居中 pad → 坐标错位。
- 传真实 `ratio_pad=((r,r),(left,top))` 后，scale_boxes 直接用 (r,left,top) 反变换 → 坐标精确还原。

## 4. 最小侵入保证

- `scripts/predict.py` **零改动**（原提交脚本保持不变）。
- F4 checkpoint **零改动**。
- 训练配置 **零改动**。
- 新增文件只有 `scripts/predict_rect.py` + 本 `diagnostic/rect_pipeline_fix/` 目录。
- 输出格式/通道/参数与 predict.py 完全一致（直接 import 复用），保证 A/B 只差「rect vs square + 是否显式 ratio_pad」。

## 5. 回归测试（阶段六，已通过）

| 用例 | 画布 | round-trip 误差 |
|---|---|---|
| 大图 1080×1920 中心 | 736×1312 | 0.000 px |
| 大图 1080×1920 左上/右下角 | 736×1312 | 0.000 px |
| 小图 360×640（放大 2×） | 736×1312 | 0.000 px |
| 纵向图 1920×1080 | 1312×736 | 0.000 px |
| 宽图 1000×2000（ar=0.5） | 672×1312 | 0.000 px |
| RGB/Depth 同步（合并 4ch 共享 ratio/pad） | — | PASS |
| square scaleup=True/False | 1280×1280 | 自洽（scaleup=False 暴露重算 latent bug，见下） |

> square `scaleup=False` 的 ratio_pad 重算给出 gain=2.0/pad=(0,280)，而真实 letterbox 是 gain=1.0/pad=(320,460) —— 这正是 `scale_boxes(ratio_pad=None)` 的 latent bug 铁证，也证明修复必须显式传 ratio_pad。
