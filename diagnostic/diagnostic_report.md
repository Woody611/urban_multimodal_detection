# 本地验证指标 vs 赛题提交 Pipeline 一致性审计报告

- **基准模型**：F4 = YOLO11m RGBD Mid-Fusion @1280，best mAP50-95 = **0.53273**（`runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt`，results.csv ep251）
- **线上提交分数**：**0.501890**
- **本地 vs 线上差距**：**0.03084**（绝对）
- **日期**：2026-09-15
- **范围**：只做诊断；不训练新模型、不改 F4 checkpoint、不改训练配置、不删已有实验；所有新输出在独立 `diagnostic/` 目录。

---

## 0. 结论速览

差距 0.03084 由**三项**组成（可加和到 ±0.0001）：

| # | 因素 | 数值 | 性质 |
|---|---|---|---|
| 1 | **指标口径差异**（fork 训练指标 → 赛题官方指标） | **−0.0308** | 主因，**非 bug**（赛题指标定义如此） |
| 2 | **rect=True（矩形 letterbox 736×1312）vs 提交 square letterbox（1280×1280）** | **−0.0141**（val 上） | 真实 pipeline 差异，作用于**全部 400 张图**，可修复 |
| 3 | 测试集分布差异 | **+0.0140** | 测试集略易，**非 bug** |

−0.0308 − 0.0141 + 0.0140 = **−0.0309 ≈ −0.03084** ✓

**最重要的新发现（本报告定稿结论）**：`model.val()` 在 [model.py:598](ultralytics/engine/model.py#L598) **强制 `rect=True`**（矩形推理），把每张图 letterbox 到**矩形 736×1312**（而非 1280×1280 正方形）；而 `scripts/predict.py` 用 **square 1280×1280**。二者的「内容像素」完全相同（都把长边归一化到 720×1280），仅画布尺寸/填充不同，造成 forward 特征图对齐差异 → 预测框数量与置信度系统性偏移 → **−0.0141**。这是本地 val 与提交 pipeline 之间除指标口径外的唯一真实不一致点。

---

## 1. 三个关键数字（fork 指标口径，val 集，conf=0.001）

| 来源 | mAP50-95 | P | R | mAP50 | mAP75 | 说明 |
|---|---|---|---|---|---|---|
| A = 训练内置 val（model.val()，rect=True） | **0.53273** | 0.84029 | 0.72177 | 0.78026 | 0.59124 | results.csv ep251 |
| B1 = 提交 pipeline（predict.py，scaleup=True + square） | **0.51863** | 0.83657 | 0.70990 | 0.76733 | 0.56043 | 真实提交配置 |
| B2 = predict.py scaleup=False（**坐标 bug 未修**） | 0.46465 | 0.74426 | 0.62039 | 0.67256 | 0.50566 | 是 bug 不是干净对比 |
| B3 = predict.py scaleup=False（**坐标已修**） | **0.51341** | — | — | — | — | 见 §4.3 |

- A → B1 = **−0.0141**：唯一来源是 **rect=True（矩形）vs square（正方形）**（见 §4），**不是 scaleup**。
- B1 → B3 = −0.0052：这是 **scaleup 的效应**（B1 放大小图、B3 不放大），且 B3 与 val 的语义**不一致**（val 的 load_image 本来就放大小图，见 §4.1）。

---

## 2. 任务一/二/六证据：参数与类别审计

### 2.1 F4 训练配置 + best.pt（任务一）

`runs/.../f4_1280/args.yaml` 实读：`imgsz=1280, batch=8, conf=null, iou=0.7, max_det=300, half=false, rect=false, single_cls=false, agnostic_nms=false, channels=4, use_simotm=RGBD, pairs_rgb_ir=[], pairs_rgb_depth=[]`。

- `conf=null` 在 val 时被 [validator.py:101-102](ultralytics/engine/validator.py#L101-L102) 回填为 **0.001**。
- `half=false` → val 用 **float32** 输入（与 predict.py 的 `float()/255` 一致，无 fp16 差异）。
- **`rect=false` 是训练配置里的字段，但 `model.val()` 在运行时被 [model.py:598](ultralytics/engine/model.py#L598) 的 `custom={"rect": True}` 覆盖为 True**（val 方法级默认值优先于 args.yaml）。因此训练日志里的 mAP 全部是在 `rect=True` 下算出的。

### 2.2 提交预测脚本（任务二）—— 实际代码

`scripts/predict.py`（唯一真实提交脚本）。关键参数：

| 参数 | 值 | 位置 |
|---|---|---|
| conf | **0.001** | predict.py:289 |
| iou | **0.7** | predict.py:291 |
| imgsz | **1280** | predict.py:293 |
| batch | 16 | predict.py:294 |
| max_det (NMS) | **300** | predict.py:295 |
| max_boxes (写盘截断) | 100 | predict.py:297 |
| letterbox | auto=False, scaleFill=False, **scaleup=True**, center=True, stride=32, **new_shape=(1280,1280) square** | predict.py:356-357 |
| 通道 | `_to_chw`：[R,G,B,D] | predict.py:113-135 |
| 归一化 | `float() / 255.0` | predict.py:142 |

### 2.3 类别映射（任务六）—— 正确

- `dataset.yaml` 12 类、`model.names` 12 类、训练 label class_id 与提交 class_id 均为 **0-based [0,11]**，无错位（假设 9 排除）。

---

## 3. 核心发现一：指标口径差异（主因，非 bug）

训练内置 val 用 **fork 指标**（trapz 积分 + ramp 尾段 + 按 IoU 降序匹配），赛题线上用 **官方指标**（101 点算术平均 + 零尾段 + 按置信度顺序匹配）。对同一批 predict.py 预测（scaleup=True）测出来：

| 指标组合 | mAP50-95 |
|---|---|
| 官方（mean + zero tail + conf 匹配） | **0.48787** |
| mean + ramp + conf | 0.51821 |
| trapz + ramp + iou（fork 复刻） | 0.51491 |
| trapz + zero + conf | 0.48773 |

三个子项对 mAP50-95 的影响：**tail zero vs ramp = −0.0303**（主导）、match conf vs iou = −0.0037、mean vs trapz = −0.0005。

→ 0.51863（fork）− 0.0308 ≈ 0.48787（官方）。**这一项是 0.03084 差距的绝对大头，是赛题指标定义决定的，无法靠推理参数「修」。**

---

## 4. 核心发现二：rect=True（矩形 letterbox）vs square（正方形）—— −0.0141 的真正来源

### 4.1 关键事实链（全部已实测钉死）

1. [model.py:598](ultralytics/engine/model.py#L598)：`model.val()` 强制 `custom = {"rect": True}`。
2. [base.py:491-513](ultralytics/data/base.py#L491-L513) `set_rectangle()`：按宽高比把图排序分批，`batch_shapes = ceil(shape * imgsz/stride + 0.5) * stride`。val 集全部图宽高比相同（1080/1920 = 360/640 = **0.5625**），得到 batch_shape = **（736, 1312）**。
3. [base.py:385-389](ultralytics/data/base.py#L385-L389) `load_image(rect_mode=True)`：**先把长边 resize 到 1280（无 scaleup 封顶）**——大图 1080×1920→720×1280（×0.667），小图 360×640→720×1280（**×2.0 放大**）。
4. [augment.py:1597](ultralytics/data/augment.py#L1597)：`LetterBox` 读 `labels.pop("rect_shape", self.new_shape)` → 目标形状是 **736×1312**（不是 1280×1280）。
5. 铁证：val dump 早期报错 `expected input[8, 0, 736, 1312]` —— val 的输入张量确实是 **736×1312**。
6. 决定性复现：对同一张大图，走「load_image ceil-resize 到 720×1280 → LetterBox(736×1312)」的预测框数与 val save_txt **逐张完全一致**（2/24/5 全 MATCH）；而 LetterBox(1280×1280) 的框数系统性偏高（8/38/1）。

### 4.2 为什么是 rect、不是 scaleup

- val 的 `LetterBox` 虽写 `scaleup=False`（dataset.py:181），但它在 `load_image` **之后**才运行，此时图已被 load_image 把长边放大/缩小到 720×1280，`scaleup=False` 只会阻止「再放大」（永远不发生，因为长边已 =1280）。
- 因此 **val 对小图的有效行为是「放大到 720×1280」**，与 predict.py 的 `scaleup=True` **完全一致**。`scaleup` 标志的差异是**红鲱鱼**。
- 真正差异：val 把 720×1280 内容放进 **736×1312 矩形画布（上下各 8px、左右各 16px 填充）**；predict.py 放进 **1280×1280 正方形画布（上下各 280px 填充）**。内容像素相同，但填充量与内容在特征图上的落点不同 → stride 下采样后特征对齐不同 → 预测框数量/置信度偏移 → **−0.0141**。

### 4.3 干净 scaleup=False 复算（坐标已修）—— 实为回归，不是对齐

`diagnostic_predict_scaleup_false_fixed.py`（scaleup=False + 显式 ratio_pad）最终实测 **0.51341**，而非此前预测的 0.53273。原因：

- 它用 **square 1280×1280**（未复刻 rect），且 **scaleup=False 不放大 27 张小图**（与 val 的 load_image 放大相悖）。
- 所以 B3 = 0.51341 = A(0.53273) − rect(−0.0141) − 小图不放大(−0.0052)。它同时偏离 val 两处，**不是**干净对齐。
- 坐标 bug 修复确实让它从 B2=0.46465 涨回 +0.0488，但没解决 rect 差异。

---

## 5. 假设逐条判定（10 条）

| # | 假设 | 判定 | 证据 |
|---|---|---|---|
| 1 | 测试集分布差异 | **非 bug**（+0.0140，测试略易） | 0.48787→0.501890 |
| 2 | 提交推理参数 | **基本一致**；唯一实质差异是 **square vs val 的 rect=True** | §2.2 / §4 |
| 3 | RGBD 预处理不一致 | **排除** | loaders.py:671-689 与 base.py 逐字节等价 |
| 4 | letterbox/resize 不一致 | **成立，机制是 rect=True**（非 scaleup） | §4.1 铁证 + §4.2 |
| 5 | 坐标反变换错误 | **成立（latent bug）**，当前被 scaleup=True 掩盖，非本次差距主因 | §7.3 |
| 6 | confidence 阈值过高 | **排除** | conf 均 0.001；调 conf 违规且方向相反 |
| 7 | NMS 参数不同 | **排除** | iou=0.7 / max_det=300 / multi_label=True 一致 |
| 8 | max_det 限制 | **排除** | NMS 300 一致；仅 4 图超 max_boxes=100 写盘截断 |
| 9 | 类别 ID/顺序错误 | **排除** | 0-based [0,11] 一致 |
| 10 | 提交文件格式问题 | **排除** | `<cid> cx cy w h conf` 6 位小数、clamp [0,1] |

### 5.1 假设 6 补充：conf 扫描（方向相反）

| conf | fork mAP50-95 | 官方 mAP50-95 |
|---|---|---|
| 0.001 | 0.51863 | 0.48787 |
| 0.01 | 0.54338 | 0.48041 |
| 0.25 | 0.56468 | 0.45736 |

fork 随 conf 升高而**涨**，官方随 conf 升高而**跌**。**调 conf 无法缩小线上差距，且违规——双重否决。**

---

## 6. 任务五证据：RGBD 运行时预处理（已打印，无异常）

`diagnostic_preprocess.py` 首图 00000045.jpg（360×640）实测：

- visible：BGR uint8，360×640；depth：uint8，74.3% 零值，mean=25.242（.jpg 深度走 uint8 分支不归一化，正确）
- 通道 [R,G,B,D] CHW，/255 前逐通道统计正常

→ 预处理数值与 fork 训练/val 语义一致，无隐藏 depth 问题（val depth 373 png + 27 jpg 与 visible 完全配对，`_resize_images` 为 no-op）。

---

## 7. 建议（按优先级）

1. **【接受】0.0308 的指标口径差不可消除**：赛题官方指标（mean+zero tail+conf 匹配）与本地 fork 指标（trapz+ramp+IoU 匹配）的定义差。报告/日志/团队沟通都应**用官方指标口径**复算，不要用 fork 的 0.53 直接对标线上 0.50。

2. **【可做，收益 ~+0.014】让 predict.py 复刻 val 的 rect=True letterbox**：当前提交用 square 1280×1280，val 用 rect 736×1312，差 −0.0141。把提交 pipeline 改成与 val 一致的「load_image 长边 1280 → 矩形 letterbox（最小矩形 + stride 对齐 padding）」即可消除这一项，预计 fork 口径 B1 从 0.51863 回到 ≈0.5327、官方口径从 0.48787 回到 ≈0.5019，**线上预期 ≈0.516**。
   - **注意**：必须同时把真实 `ratio_pad` 传给 `scale_boxes`（见第 3 条），否则坐标反变换错误会立刻暴露。
   - 先在 val 上 A/B（fork + 官方两个口径都报），确认 +0.014 兑现后再上测试集。

3. **【修 latent bug】predict.py:399 的 `scale_boxes` 未传 ratio_pad**：`scale_boxes` 在 `ratio_pad=None` 时重算 `gain=min(...)` 且**假设 scaleup=True + square + 居中 pad**。当前 scaleup=True 下恰好与真实 letterbox 一致（坐标正确），但一旦改 letterbox（如第 2 条的 rect、或切 scaleup=False）就会坐标错位。最小侵入：让 `_preprocess_batch` 返回每张图的 `ratio_pad`，`scale_boxes` 显式传参。

4. **【不建议】**：调 conf / NMS / max_det / TTA 刷分（违规或无效）；改 F4 模型结构（越界）。

---

## 8. 复算结论（已补，替代旧预测）

| 项 | 值 | 判定 |
|---|---|---|
| `phase2_map` 逻辑 @ val 自身 dump 的预测 | **0.53264** | ≈ val 0.53272 → **phase2_map 的 GT/指标口径正确** |
| `phase2_map` 逻辑 @ 干净 scaleup=False predict（B3） | **0.51341** | 偏离 val 两处（square + 小图不放大），非对齐 |

**结论**：−0.019 的「预测差距」已定位——不是 GT/指标问题，而是 predict.py 的 square letterbox（vs val 的 rect 736×1312）导致 forward 输出不同、预测框数量系统性偏高（6927 vs val 5152）。§0 的三项分解即最终答案。

---

## 附：诊断产物清单

| 文件 | 用途 |
|---|---|
| `scripts/diagnostic_preprocess.py` | 任务五：运行时 RGBD 预处理数值打印 |
| `scripts/diagnostic_compare_input.py` | 逐图对比 predict/val 输入张量（head0 逐字节一致，head1 为 list 需展平） |
| `scripts/diagnostic_predict_scaleup_false.py` | 早期 scaleup=False 测试（含坐标 bug，0.46465） |
| `scripts/diagnostic_predict_scaleup_false_fixed.py` | 干净 scaleup=False（修 ratio_pad），实测 0.51341 |
| `scripts/diagnostic_val_dump.py` | 决定性实验：model.val(save_txt) 复算，定位 metric/GT 口径正确 |
| `diagnostic/scaleup_false_fixed/results/` | B3 的 TXT 输出 |
| `diagnostic/val_dump/val_preds/labels/` | val 自身 dump 的预测 |
| 本报告 `diagnostic/diagnostic_report.md` | 审计结论 |
