# Modality fusion audit

| 项 | 值 | 来源 |
|---|---|---|
| 三模态合并在哪 | `_merge_channels_rgbid`（`base.py:430`），`cv2.merge((b,g,r,ir,depth))` | 源码 |
| 通道序 | **[B, G, R, IR, D]**，全部 uint8 [0,255] | 源码 |
| 各模态 stem | RGB→Conv(48,3,2)；IR→Conv(8,3,2)；Depth→Conv(8,3,2) | yaml:48,50,52 |
| stem 参数独立性 | **独立**（三个不同 `Conv` 实例） | yaml |
| **第一次融合** | **`Concat([2,4,6])` → 64ch**（`yaml:53`） | 源码 |
| 融合方式 | **concat**（非 add / 非 weighted / 非 attention） | 源码 |
| 融合后 | 立即 `Conv(128,3,2)` ⇒ **通道混合**；此后**无任何 modality-specific 分支** | 源码 |
| 融合前通道数 | 48 + 8 + 8 = **64** | — |
| 融合后通道数 | **64**（L7）→ L8 扩张至 128 | — |
| compression_ratio | **1.00**（64→64）⇒ `STRUCTURAL_COMPRESSION_PRESENT = FALSE` | — |

## IR-CLAHE

| 项 | 值 |
|---|---|
| 位置 | `base.py:334-335`，在 `_merge_channels_rgbid` **之前** |
| 作用域 | **仅 IR** —— 注释与代码均确认 RGB/Depth 不可能被影响（`base.py:332`） |
| 与 percentile 的关系 | **互斥**，非叠加（`base.py:329-330`） |
| dtype / range | in/out 均 uint8 [0,255]，与 percentile 相同 |
| train/val 一致性 | 同一 `load_and_preprocess_image` 路径；`ir_encoding` 取自 `self.hyp` |
| 推理/提交侧 | **另一份拷贝** `ultralytics/data/loaders.py`（由 `predict_rect.py --train_config` 传 `ir_encoding`） |

## §6 / §21 措辞纪律

- `MODALITY_INTERACTION_EXISTS = TRUE`（代码可证：三模态在 L7 融合，之后共享全部计算）。
- `MODALITY_HARM = NOT_OBSERVABLE` —— **无任何 frozen artifact 含 per-modality feature / modality-specific
  logits / per-modality 梯度**。`diagnostic/modality_dropout/` 是**训练期 dropout 实验**，不是逐模态特征归因。
- ⇒ 不得写 "IR hurts classification" / "depth harms" / "CLAHE causes collapse"。
