# Classification bias initialization audit

| 项 | 值 |
|---|---|
| 机制 | `Detect.bias_init`（`ultralytics/nn/modules/head.py:137-141`） |
| 调用点 | `ultralytics/nn/tasks.py:373` → `m.bias_init()`（注释 "only run once"） |
| 公式 | `b[-1].bias.data[:nc] = log(5 / nc / (640/s)**2)` |
| stride-8 初值 | **-9.6395** |
| D′ 训练后实测 | `model.30.cv3.0.2.bias` min=-9.8906 max=-9.6719 **mean=-9.7949** |
| 位移 | **-0.1554**（300 epoch） |
| 是否 class-specific | 是（每类一个 bias 值），但**同一类在全图所有位置共享同一 bias** |
| box 分支对照 | `cv2.0.2.bias` mean = 0.9979（初值 1.0，已明显训练） |

## 意义（严格措辞）

```
GT 类 logit(x) = W_gt · h(x) + b_gt        （h = cv3 前置特征，已用 ckpt 逐行验证的精确等式）
                 └ 位置相关 ┘   └ 位置无关 ┘
```
- **b_gt ≈ -9.79 是全图共享的负偏移**，因此它**不能解释同一类内"有的检出、有的塌陷"**
  —— 位置无关项无法产生位置相关的差异。
- 但它可以解释**整体尺度**：要让 sigmoid 达到 0.001（推理 conf 阈值），需 `W·h > 6.91`… 即 logit > −6.9。
- 预训练权重自带 `bias_init` 的初值；该初值按 640px 的锚点密度推导，而本实验 imgsz=1280
  ⇒ 等效先验对应 `log(5/12/(1280/8)^2) = -11.0258`，比实际初值低 **1.39 nats**。
  **这是一个结构性观察，不是 bug**；且初始化只影响训练早期，**不足以解释最终 epoch 的 1e-6**。
