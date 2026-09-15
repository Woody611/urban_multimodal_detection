# F4 训练前审计报告 — YOLO11m RGBD Mid Fusion @1280

> 阶段：**训练前**（配置审计 + 单变量 diff + 冒烟测试）。正式 300 epoch 尚未启动，等待确认。
> 基线：F1 = `runs/urban_multimodal_det_e7_lr0half/`，best mAP@0.5:0.95 = **0.51955**（ep287）。
> 审计依据：F1 落盘的 `runs/urban_multimodal_det_e7_lr0half/args.yaml`（权威来源，非 yaml 文件名）。

---

## 1. F1 / F4 配置 diff

F1 的实际训练配置是 [`configs/train_e7_lr0half.yaml`](../configs/train_e7_lr0half.yaml)（由 F1 的 `args.yaml` 反查确认）。
F4 配置 [`configs/train_f4_1280.yaml`](../configs/train_f4_1280.yaml) 是它的逐行副本。

### 1.1 文件级 diff（`diff -u`）

除注释外，实质差异只有两行：

```diff
-experiment_name: "urban_multimodal_det_e7_lr0half"
+experiment_name: "urban_multimodal_det_yolo11_rgbd_f4_1280"   # 仅输出目录，非训练变量
-image_size: [1024, 1024]
+image_size: [1280, 1280]                                       # ★ 唯一实验变量
```

### 1.2 语义级 diff（更严格）

不看 yaml 文本，而是直接调用 `scripts/train.py::_build_train_kwargs` 构造两边真正传给
`model.train()` 的 kwargs，再逐项比对 —— 这能抓到文本 diff 看不出的隐藏映射差异：

```
kwargs key         F1                           F4
------------------------------------------------------------------------------
imgsz              1024                         1280                          <== DIFF
name               urban_multimodal_det_e7_lr0half  urban_multimodal_det_yolo11_rgbd_f4_1280  <== DIFF
------------------------------------------------------------------------------
功能性差异项数: 2 / 共 24 项
```

**24 个 kwarg 中只有 2 项不同**，其中 `name` 只是输出目录、不是训练变量。
**唯一训练变量 = `imgsz: 1024 → 1280`**。符合理想 diff。

### 1.3 未出现在 kwargs 里的参数 —— 已确认走默认值

F1 的 `args.yaml` 显示 `deterministic: true`、`close_mosaic: 10`、`lrf: 0.01`、`nbs: 64`、
`box/cls/dfl = 7.5/0.5/1.5`、全部增强值等，但它们**不在** `_build_train_kwargs` 输出里
（train.py 只在 train_cfg 显式提供时才覆盖）。已逐一核对 [`ultralytics/cfg/default.yaml`](../ultralytics/cfg/default.yaml)：

| 参数 | fork 默认值 | F1 args.yaml 实测 | 一致 |
| :--- | :---: | :---: | :---: |
| deterministic | True (:29) | true | ✅ |
| close_mosaic | 10 (:33) | 10 | ✅ |
| lrf | 0.01 (:93) | 0.01 | ✅ |
| nbs | 64 (:104) | 64 | ✅ |
| box / cls / dfl | 7.5 / 0.5 / 1.5 (:99-101) | 同 | ✅ |
| mosaic / mixup / copy_paste | 1.0 / 0.0 / 0.0 (:116-118) | 同 | ✅ |
| degrees / translate / scale / shear / perspective | 0 / 0.1 / 0.5 / 0 / 0 (:108-112) | 同 | ✅ |
| flipud / fliplr | 0.0 / 0.5 (:113-114) | 同 | ✅ |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 (:105-107) | 同 | ✅ |
| erasing / crop_fraction / auto_augment | 0.4 / 1.0 / randaugment (:121-122,120) | 同 | ✅ |
| warmup_epochs / momentum / bias_lr | 3 / 0.8 / 0.1 (:97-98 + train_cfg) | 同 | ✅ |
| rect / multi_scale / fraction / dropout | False / False / 1.0 / 0.0 | 同 | ✅ |

**结论：F1 与 F4 的默认值完全一致，不存在隐藏的第二个变量。**

### 1.4 F1 实际训练管线（审计确认）

| 项 | F1 实际值 |
| :--- | :--- |
| model yaml | `configs/yolo11m_midfusion_rgbd_concat_res.yaml` |
| 模型 scale | **m**（文件名正则 `guess_model_scale`，冒烟测试实测返回 `'m'`） |
| 参数量 | 30.34M / 45 层（@640 基准 114.21 GFLOPs） |
| 数据 | `data/processed/depth_split_train/dataset.yaml`（由 `val_ratio=0.2, seed=42` 切分生成） |
| RGBD 管线 | `use_simotm="RGBD"`, `channels=4`, `pairs=[visible, depth]` |
| fusion 位置 | P3/P4/P5 三处 `fused = depth + Conv(Concat(rgb, depth))` |
| optimizer | SGD (nesterov 源码硬编码) + momentum 0.937 + wd 5e-4 |
| lr | lr0=0.005, lrf=0.01, cos_lr=True |
| batch / imgsz | 8 / 1024 |
| epochs / patience | 300 / 80 |
| seed / deterministic | 42 / true |
| AMP | true |
| val | 训练内置 val，conf=0.001, iou=0.7, max_det=300 |
| resume | false（从零训练） |

---

## 2. F4 实际训练配置

| 项 | F4 值 | 与 F1 |
| :--- | :--- | :---: |
| experiment_name | `urban_multimodal_det_yolo11_rgbd_f4_1280` | 独立目录 |
| model yaml | `configs/yolo11m_midfusion_rgbd_concat_res.yaml` | 同 |
| **imgsz** | **1280** | **★ 唯一变量** |
| batch_size | 8（待显存确认） | 同 |
| epochs / patience | 300 / 80 | 同 |
| learning_rate (lr0) | 5.0e-3 | 同 |
| optimizer | SGD, momentum 0.937, wd 5.0e-4 | 同 |
| scheduler | CosineAnnealingLR (cos_lr=True) | 同 |
| warmup_epochs | 3 | 同 |
| seed / deterministic | 42 / true | 同 |
| amp | true | 同 |
| use_simotm / channels | RGBD / 4 | 同 |
| val_ratio | 0.2 | 同 |
| resume | **false（绝不 resume F1）** | 同 |
| pretrained | `yolo11m.pt` | 同 |
| 增强 / loss gain / NMS | 全部走默认（见 §1.3） | 同 |

启动命令：

```bash
python scripts/train.py \
    --model_config configs/yolo11m_midfusion_rgbd_concat_res.yaml \
    --train_config configs/train_f4_1280.yaml
```

> 说明：`--dataset_config` 未指定，默认 `configs/dataset.yaml`；因 `val_ratio=0.2 > 0` 且 val 无标注，
> train.py 会用 `_split_train_val_depth(dataset_cfg, 0.2, 42)` **确定性重建**切分。
> 已实测重建结果为同一路径 `depth_split_train`、同样 1600 train / 400 val —— 与 F1 完全一致（见 §4.4）。

---

## 3. GPU / batch / effective batch / 显存

### 3.1 梯度累积：框架已内置，无需改任何代码

[`ultralytics/engine/trainer.py:301`](../ultralytics/engine/trainer.py#L301)：

```python
self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
```

`nbs=64`（nominal batch size，默认值，F1/F4 相同）：

| batch | accumulate | **effective batch** | weight_decay 自动补偿 |
| :---: | :---: | :---: | :---: |
| 8 | 8 | **64** | 是（保持恒定） |
| 4 | 16 | **64** | 是（保持恒定） |
| 2 | 32 | **64** | 是（保持恒定） |

**结论：batch 从 8 降到 4 时，effective batch 仍是 64，`lr0` 无需改动，框架不改。**
这与「不要因为 batch 改变就自行修改 lr0」的要求天然吻合 —— `lr0=0.005` 保持不变。

> ⚠️ 一个诚实的保留：梯度累积让**梯度**等价于 batch=64，但 **BatchNorm 统计量仍按 micro-batch 计算**
> （batch=8 或 4），与真正的 batch=64 不同。这是 batch 变化带来的唯一真实副作用。
> 但注意：F4 无论如何都已经在改 imgsz，BN 统计量本就随分辨率变化，**只要 F4 内部保持 batch=8 就等于 F1 的 batch**，
> 反而不引入这个副作用。所以**优先 batch=8**。

### 3.2 显存估算（云端 32GB）

模型侧固定开销（与 batch/imgsz 无关）：

| 项 | 大小 |
| :--- | ---: |
| 参数 (30.34M fp32) | 121 MB |
| 梯度 | 121 MB |
| SGD momentum 缓冲 | 121 MB |
| EMA（权重副本） | 121 MB |
| AMP master weights | ~60 MB |
| **合计** | **≈ 0.5 GB** |

激活值随 `batch × H × W` 线性增长。相对 F1(1024) 的比例：

```
(1280/1024)² = 1.5625  →  激活值内存约为 F1 的 1.56 倍
```

计算量同步放大（@640 基准 114.21 GFLOPs）：

| 分辨率 | GFLOPs/图 |
| :---: | ---: |
| 1024（F1） | 292.4 |
| **1280（F4）** | **456.8** |

**判断**：模型侧仅 0.5 GB，激活值约为 F1 的 1.56 倍。F1 在云端可跑，32GB 卡上 **1280 + batch=8 大概率可跑**，
且很可能还有余量。但这属于**推断**，不是实测 —— **必须在云端用 `nvidia-smi` 确认**。
若实测 OOM，降到 batch=4（effective batch 仍为 64，lr0 不动）。

### 3.3 目标硬件：V100 32GB（用户确认）

| 项 | 值 | 对 F4 的影响 |
| :--- | :--- | :--- |
| 架构 | Volta (SM 7.0) | — |
| 显存 | 32 GB HBM2, ~900 GB/s | 容量充足；F1(1024/batch8) 已在同卡跑通 |
| fp16 张量核心 | ✅ 支持 | **`amp: true` 可用**，与 F1 一致，不改 |
| bf16 | ❌ **Volta 不支持 bf16** | 无影响 —— ultralytics AMP 用 fp16 而非 bf16，F1 同此 |
| TF32 | ❌ 不支持（Ampere+ 才有） | 无影响（本实验走 fp16 AMP，不用 TF32） |

**结论**：V100 32GB 对 `1280 + batch=8 + amp=true` 没有架构级障碍。显存容量充裕（模型侧仅 0.5GB，
激活值 1.56× F1），**预期可跑**；仍以云端 `nvidia-smi` 实测为准。

### 3.4 时间成本（由 F1 实测外推，非猜测）

F1 的 `runs/urban_multimodal_det_e7_lr0half/results.csv` 有每 epoch 累计耗时列，实测稳态：

| epoch | 累计 time(s) | 单 epoch 耗时 |
| :---: | ---: | ---: |
| 10 | 829.8 | (829.8−96.4)/9 ≈ 81.5 s |
| 100 | 8316.5 | (8316.5−829.8)/90 ≈ 83.2 s |
| 200 | 16379.9 | (16379.9−8316.5)/100 ≈ **80.6 s** |
| 300 | 24443.8 | (24443.8−16379.9)/100 ≈ **80.6 s** |

F1 全 300 epoch 实测 **24443.8 s = 6.79 小时**（稳态 80.6 s/epoch，含每 epoch 的 val）。

F4 外推（计算量与激活值同按面积缩放，`(1280/1024)² = 1.5625`）：

```
80.6 × 1.5625 ≈ 126 s/epoch   →   300 epoch ≈ 37,800 s ≈ 10.5 小时
```

**预估 F4 约 10.5~11.5 小时**（比 F1 的 6.8 小时多约 55%）。
参考：F1 best 在 ep287，未触发 `patience=80` 早停；F4 若提前分叉可能早停，实际耗时可能更短。

---

## 4. 冒烟测试结果

脚本：[`scripts/f4_smoke_test.py`](../scripts/f4_smoke_test.py)。
本机为 **CPU-only**（torch 2.4.1+cpu，无 CUDA），GPU 显存无法本地测量，其余全部实测。

**结果：20 PASS / 0 FAIL**（`--batch 2`；batch=8 因 CPU 内存 >20GB 不可行，见 §4.5）

| # | 检查项 | 结果 | 实测证据 |
| :---: | :--- | :---: | :--- |
| 1 | RGBD 数据读取正常 | ✅ PASS | dataset=1599 图，batch 正常取出 |
| 2 | 4-channel 输入正常 | ✅ PASS | `img.shape=(2, 4, 1280, 1280)` |
| 3 | 1280 resize/letterbox 正常 | ✅ PASS | `HxW=1280x1280`，dtype=uint8 |
| 3b | depth 通道有真实信号 | ✅ PASS | mean=72.0, std=80.4, range 0~255（非全 0 / 非全 114） |
| 4 | Mid Fusion 正常 | ✅ PASS | scale=`m`(文件名正则)；ch=4；nc=12；30.34M 参数 / 45 层；SilenceChannel×2 + ADD×3 齐全 |
| 5 | forward 正常 | ✅ PASS | P3/P4/P5 = `[(2,76,160,160), (2,76,80,80), (2,76,40,40)]` |
| 5b | 特征图层级 = stride 8/16/32 @1280 | ✅ PASS | 160/80/40 完全对应 |
| 6 | backward 正常 | ✅ PASS | 462 个参数张量全部有梯度、全部有限、全部非零 |
| 6b | 融合真的在训（depth 分支收到梯度） | ✅ PASS | depth 分支 stem \|grad\|=7.00e-01 |
| 7 | loss 正常 | ✅ PASS | total=34.044，[box,cls,dfl]=[5.786, 6.964, 4.273]，有限且为正 |
| 8 | validation 正常 | ✅ PASS | 真实 validator 全流程跑完 398 图，578.6s，无报错 |
| 9 | 显存不 OOM | ⚠️ **SKIP（本地无 CUDA）** | 须云端验证；估算见 §3.2 |
| 10 | 无 tensor shape mismatch | ✅ PASS | 全流程无 shape 报错；NMS postprocess 正常解析 |

### 4.1 关键实测数字

- 未训练权重跑全量 val@1280：**mAP50-95 = 0.00000**（随机权重，数值无意义 —— 只证明流程跑通）
- val 速度（CPU）：1278.1 ms/图 推理
- **validator 报告的 val 规模 = 398 图 / 2807 GT**，与 F1 误差分析报告中的 `398 图 / 2807 GT` **完全一致**
  → 独立佐证 F1 与 F4 用的是**同一个 val 划分**
- 训练单步（batch=2 @1280, CPU）：forward 3.5s + backward 7.4s

### 4.2 冒烟测试中发现并修正的两个自身错误（记录以备复核）

1. 最初把训练态 `model(x)` 的返回值当成 `(preds, loss)`。实测本 fork 训练态 forward **只返回 3 层原始特征图**，
   loss 由 `criterion` 单独计算（`BaseModel.loss(batch, preds)`，即 trainer 的真实调用路径）。
   修正前「loss」是 P4 原始 logits 之和（-454538，负数且无意义），修正后才是真 loss。
2. `non_max_suppression` 的形参名在本 fork 是 `conf_thres` / `iou_thres`，不是 `conf` / `iou`。

### 4.3 数据侧观察（**不修改**，如实报告）

训练集扫描有 3 条既有警告（F1 完全相同，非 F4 引入、不构成变量）：

```
003107.png              : ignoring corrupt image/label: non-normalized or out of bounds coordinates [1.0065]
000013_046_00000202.png : 1 duplicate labels removed
hehe_10_00000044.png    : 1 duplicate labels removed
```

- `003107.png` 因标签越界被**整图丢弃** → dataset 实际 1599 张（非 1600）
- 另有 2 张图各删除 1 个重复框

这些与 F1 逐字相同（同数据、同管线），因此**不破坏 A/B 对照**。按本次实验要求「不修改标签」，不做任何修复。

### 4.4 数据切分一致性验证（实测）

`_split_train_val_depth(dataset.yaml, val_ratio=0.2, seed=42)` 重建结果：

```
生成路径: data/processed/depth_split_train/dataset.yaml   ← 与 F1 同一路径
  images/train/visible   1600 文件
  images/val/visible      400 文件
```

确定性重建、路径与数量均与 F1 一致 → **F4 与 F1 用同一份数据划分**。

### 4.5 为什么本地用 batch=2 而非 8

batch=8 @1280 在 CPU 上前向即耗尽内存，进程 segfault（本机 33.9GB 总内存，可用 22.7GB）。
这是**本地 CPU 内存**限制，与 GPU 显存无关，不能外推。
用 batch=2 完成全部前后向/损失/梯度验证 —— 这些结论与 batch 大小无关（形状、层级、梯度连通性都不变）。

---

## 5. 待办：正式训练后需要补的报告项

以下为训练完成后才能填写的项（当前尚无数据）：

```text
 5. 最终 best mAP50-95        → 对照 F1 的 0.51955
 6. best epoch                → 对照 F1 的 287
 7. mAP50 / AP75 / P / R
 8. person / sign / bicycle / ball 四类逐项（AP50-95 / AP50 / AP75 / P / R）
 9. 训练稳定性：best 是否孤峰、前后是否有平台、最后 30 epoch 均值、best 后是否下降、train/val loss 异常
10. F1 vs F4 对比 + 是否建议继续 1280 方向 + 下一步建议
```

判据（预登记，避免事后调整标准）：

| 档位 | F4 best mAP50-95 | 结论 |
| :--- | :---: | :--- |
| 强成功 | ≥ 0.535 | 1280 对当前任务有明显价值 |
| 明显有效 | 0.530 ~ 0.535 | 分辨率是重要杠杆，继续 1280 方向 |
| 小幅有效 | 0.524 ~ 0.530 | 有收益，需权衡显存/训练成本 |
| 基本无效 | < 0.524 | 单纯提分辨率收益有限，转向容量 / Depth 表达 |

第一判断标准：**官方 mAP@0.5:0.95**。

---

## 6. 合规确认

| 约束 | 状态 |
| :--- | :--- |
| 不 resume F1 / 不加载 F1 best.pt / last.pt | ✅ `resume: false`，`model_path = --model_config` 的 yaml |
| 独立实验目录 | ✅ `runs/urban_multimodal_det_yolo11_rgbd_f4_1280/`，未覆盖 F1/F2/F3a |
| 唯一变量 imgsz | ✅ 语义 diff 证实 24 项中仅 imgsz 为训练变量 |
| 不加入 YOLO11l / ValidMask / P2 / attention / 新 loss / TTA 等 | ✅ 全部未加入 |
| 不改 conf/NMS 刷分 | ✅ conf=0.001 / iou=0.7 / max_det=300 未动 |
| 不改 lr0 / epochs / close_mosaic / 增强 / 数据 / 标签 / depth 预处理 / 划分 | ✅ 未改（仅报告 §4.3 既有警告） |
| 不修改训练框架代码 | ✅ 未改 train.py 或任何 ultralytics 源码；仅新增独立脚本与配置 |
| 未启动正式训练 | ✅ 等待确认 |
