# cls / box branch audit

```
P3/P4/P5 neck features (256 / 512 / 1024 ch)
              ↓  同一张量 x[i]  (head.py:74)
   ┌──────────┴──────────┐
 cv2[i]                cv3[i]
 (box tower)         (cls tower)
 64ch                256ch
 Conv3x3 x2          (DWConv+Conv) x2
 Conv2d(64,64,1)     Conv2d(256,12,1)
   ↓                     ↓
 4*reg_max=64          12 类 logits
```

## CLS_BOX_REPRESENTATION = SHARED (backbone/neck) + SEPARATE (head tower)

- **共享**：L0–L23 全部 backbone + neck，以及 `x[i]`（head 的输入张量）。
- **分离**：`cv2` 与 `cv3` 是**两个独立的 `nn.Sequential`**（`head.py:47-60`），参数不共享；
  宽度 **64 vs 256**；算子不同（普通 3x3 vs DWConv+1x1）。
- **不存在** DFL 专用分支之外的其他 modality/class 专用路径；**不存在** attention / gating / dropout；
  **不存在** modality-specific normalization。

## 严格措辞（§12 要求）

CIoU 高只证明 `GEOMETRIC_OUTPUT_CAPABILITY = OBSERVED`；`cls ≈ 1e-6` 只证明
`CLS_OUTPUT_COLLAPSE = OBSERVED`。二者是 **output-level dissociation**。
`REPRESENTATION_LEVEL_DIVERGENCE` 需要 feature 证据 —— 本轮**部分**具备（见 per_gt_representation_audit.csv，
仅 P3 与 cv3 前置 h 两个层级、仅 stride-8、仅 GT 中心 cell）。
