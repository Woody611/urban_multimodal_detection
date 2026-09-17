# Pipeline 代码路径报告（阶段一 · 只读检查）

- **日期**：2026-09-15
- **模型**：F4 = YOLO11m RGBD Mid-Fusion @1280（`runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt`）
- **结论一句话**：`model.val()` 通过 `rect=True` 把输入 letterbox 到 **矩形 736×1312** 并显式传真实 `ratio_pad`；`scripts/predict.py` 用 **square 1280×1280** 且 `letterbox(image=im)` 丢弃 `ratio_pad`、`scale_boxes` 自行猜测 gain/pad。这是本地 val（0.53273）与提交 pipeline（≈0.51341）之间 −0.0141 的**唯一**代码级差异。

---

## 1. model.val() 的实际输入尺寸

- **不是 1280×1280**，而是**矩形，随批内宽高比变化**。
- 本 val 集 400 张图宽高比全部相同（1080/1920 = 360/640 = **0.5625**），`set_rectangle()` 算出的 batch_shape = **（736, 1312）**。
- 铁证：val dump 早期报错 `RuntimeError: ... expected input[8, 0, 736, 1312] ...`（输入张量 [B, C, H=736, W=1312]）。

## 2. model.val() 如何执行 rect=True（完整链路）

| 步骤 | 位置 | 内容 |
|---|---|---|
| 1 | [model.py:598](ultralytics/engine/model.py#L598) | `custom = {"rect": True}` 作为 `model.val()` 方法级默认，覆盖 args.yaml 里的 `rect: false` |
| 2 | [validator.py:156](ultralytics/engine/validator.py#L156) | `if not pt: self.args.rect = False`；pt=True → rect 保持 True |
| 3 | [val.py:243-248](ultralytics/models/yolo/detect/val.py#L243-L248) | `build_yolo_dataset(self.args, ...)`（未显式传 rect）→ build.py 用 `cfg.rect` |
| 4 | [build.py:96-105](ultralytics/data/build.py#L96-L105) | `rect = cfg.rect or rect` → True 传给 `YOLODataset(rect=True)` |
| 5 | [base.py:86-88](ultralytics/data/base.py#L86-L88) | `if self.rect: self.set_rectangle()` |
| 6 | [base.py:491-514](ultralytics/data/base.py#L491-L514) | `set_rectangle()`：按宽高比排序分批，`batch_shapes = ceil(shape*imgsz/stride + 0.5)*stride` → 本集 **(736, 1312)** |
| 7 | [base.py:529-530](ultralytics/data/base.py#L529-L530) | `label["rect_shape"] = self.batch_shapes[self.batch[index]]` |
| 8 | [augment.py:1597](ultralytics/data/augment.py#L1597) | `LetterBox` 读 `new_shape = labels.pop("rect_shape", self.new_shape)` → 目标 = **(736, 1312)** |

## 3. load_image(rect_mode=True) 的实际缩放逻辑

[base.py:385-389](ultralytics/data/base.py#L385-L389)：

```python
r = self.imgsz / max(h0, w0)          # 1280 / 长边
if r != 1:
    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
```

- **长边缩放到 1280，无 scaleup 封顶**（大图 1080×1920→720×1280 缩 0.667×；小图 360×640→720×1280 **放 2.0×**）。
- 用 `math.ceil`（非 round），并 `min(..., imgsz)` 封顶。
- 在 RGBD 模式下，`load_and_preprocess_image` 先把 visible+depth 合并成 4ch `[B,G,R,D]`，**再对这个合并图整体 resize** → RGB 与 Depth 共享同一缩放参数。

## 4. LetterBox 的实际返回值（关键陷阱）

[augment.py:1593-1678](ultralytics/data/augment.py#L1593-L1678) 的 `__call__`：

- `letterbox(image=im)`（predict.py 的调用方式）：`labels={}` 为空，走到 `else: return img` → **只返回图像，丢弃 ratio_pad**。
- `letterbox(labels={...})`（dataset 的调用方式）：`labels` 非空 → 走到 `return labels` → **返回含 `img`/`ratio_pad`/`resized_shape` 的 dict**。

## 5. ratio_pad 在哪里产生

1. [base.py:525-528](ultralytics/data/base.py#L525-L528) `get_image_and_label()`：
   ```python
   label["ratio_pad"] = (resized_shape[0]/ori_shape[0], resized_shape[1]/ori_shape[1])
   ```
   即缩放比（大图 (720/1080, 1280/1920)=(0.667,0.667)，小图 (720/360,1280/640)=(2.0,2.0)）。
2. [augment.py:1669-1670](ultralytics/data/augment.py#L1669-L1670) `LetterBox` 追加 pad：
   ```python
   labels["ratio_pad"] = (labels["ratio_pad"], (left, top))
   ```
   最终 `ratio_pad = ((ratio, ratio), (left, top))`。val 大图 = `((0.667,0.667),(16,8))`。

## 6. scale_boxes() 当前是否使用真实 ratio_pad

| 路径 | 调用 | 是否真实 ratio_pad |
|---|---|---|
| val（正确） | [val.py:121-127](ultralytics/models/yolo/detect/val.py#L121-L127) `scale_boxes(imgsz, predn[:,:4], ori_shape, ratio_pad=pbatch["ratio_pad"])` | **是** |
| predict.py（当前） | [predict.py:399](scripts/predict.py#L399) `scale_boxes(x.shape[2:], out[i][:,:4], orig_shapes[i])` | **否** → [ops.py:110-115](ultralytics/utils/ops.py#L110-L115) 自行重算 `gain=min(img1_h/img0_h, img1_w/img0_w)`、`pad` 居中 |

- predict.py 的 `letterbox(image=im)` 已经丢掉了 ratio_pad（见 §4），所以即便想传也没有来源。
- 当前 scaleup=True + square 1280×1280 下，重算的 gain/pad 恰好等于真实值，**坐标侥幸正确**；一旦改 rect 或 scaleup=False，立即错位。

## 7. 当前提交 pipeline 的输入/padding/坐标还原

- **输入画布**：square `1280×1280`（[predict.py:356-357](scripts/predict.py#L356-L357) `LetterBox(new_shape=(imgsz, imgsz), auto=False, scaleup=True, center=True, stride=32)`）。
- **缩放**：`scaleup=True` → 小图放大 2×（r=min(1280/h,1280/w)），大图缩 0.667×。
- **padding**：`center=True` → 上下居中（大图/放大后的小图都是 720×1280 内容 + 上下各 280px）。
- **坐标还原**：`scale_boxes(x.shape[2:], boxes, orig_shape)` 无 ratio_pad → 重算 gain=min(1280/h0,1280/w0)、pad 居中（见 §6）。
- **写盘**：`_format_lines` 归一化到 [0,1]、clamp、6 位小数、`max_boxes=100` 截断。

## 8. RGB 与 Depth 是否用完全相同的 resize/padding

**是，两边都一致**：

- val：`load_and_preprocess_image` 先 `cv2.merge((b,g,r,depth))` 成 4ch，再整体 resize + letterbox → RGB/Depth 同一 ratio、同一 pad。
- predict.py：`LoadImagesAndVideos` 先 `cv2.merge((b,g,r,depth))` 成 4ch（[loaders.py:688-689](ultralytics/data/loaders.py#L688-L689)），`_to_chw` 转 [R,G,B,D] 后整体 letterbox → RGB/Depth 同一 ratio、同一 pad。

→ **RGBD 读取/合并/缩放不是差距来源**，无需在修复中拆分 RGB/Depth 分别处理（它们天然共享参数，因为合并后是单张 4ch 图）。

---

## 附：关键结论映射（审计 8 问 → 答案）

| 问题 | 答案 |
|---|---|
| model.val() 实际输入尺寸 | 矩形 **736×1312**（本集），非 1280×1280 |
| rect=True 如何执行 | model.py:598 强制 → set_rectangle 算 batch_shapes → LetterBox 读 rect_shape |
| load_image 缩放逻辑 | 长边→1280，ceil，无 scaleup 封顶（小图放大 2×） |
| LetterBox 返回值 | `image=` 只返回 img（丢 ratio_pad）；`labels=` 返回含 ratio_pad 的 dict |
| ratio_pad 产生处 | base.py:525-528 生成 ratio，augment.py:1669-1670 追加 (left,top) |
| scale_boxes 用真实 ratio_pad？ | val 是；predict.py **否**（自行重算） |
| 提交 pipeline 输入/padding/坐标 | square 1280×1280 + scaleup=True + center，无 ratio_pad 反变换 |
| RGB/Depth resize/padding 一致？ | **是**，两边都是先 merge 成 4ch 再整体缩放 |
