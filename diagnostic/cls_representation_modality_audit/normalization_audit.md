# Normalization / activation / range audit（纯代码审计）

| 项 | 值 | 来源 |
|---|---|---|
| 模态合并 dtype/range | 全部 uint8 **[0,255]** | `base.py:430-433` |
| IR 16-bit 处理 | `* (255/65535)` → uint8，或 percentile 拉伸 | `base.py:323-327` |
| Depth 16-bit 处理 | `/19999*255` → uint8 | `base.py:347,361` |
| RGB | `cv2.split` 后原样 uint8 | `base.py:431` |
| 后续归一化 | 训练：`img/255`（trainer `preprocess_batch`）；推理：`torch.from_numpy(...)/255.0`（`predict_rect.py:253`） |
| 模态间 range 一致性 | **三模态在合并前均为 uint8 [0,255]** ⇒ `RANGE_DIFFERENCE = NOT_OBSERVED` | — |
| backbone normalization | 各 `Conv` 内含 **BatchNorm2d**（ultralytics `Conv` 默认 `bn=True`） | `nn/modules/conv.py` |
| 激活 | SiLU（`Conv` 默认 `act=True`） | 同上 |
| classification 头 | **无** 独立 normalization / dropout / attention / gating；末尾直接 `Conv2d(256,12,1)` | `head.py:53-60` |
| 分类概率 | 输出为 **logit**，sigmoid 在推理侧（NMS）与 loss 侧分别施加 | `ops.py:250`, `loss.py:522` |

**结论（严格）**：三模态进入融合前**数值范围一致**（都 uint8 [0,255]）⇒ 不存在"range mismatch"证据。
但**这不等于**特征层面无冲突 —— 后者需 feature 证据。
