# Mechanism report — Full-300 trajectory

```text
CASE = D
representation_effect = 0.000 nats
classifier_effect     = 0.000 nats
MATERIAL_NATS         = 2.0
```

## 分离度序列（csv epoch 口径）

| csv ep | Δ_logit | Δ_feature | Δ_h_norm | Δ_cosH | Δ_W_norm | Δ_cosW |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -16.3627 | -16.4955 | -7.1301 | -0.0 | 0.0 | 0.0 |
| 2 | -16.3627 | -16.4955 | -7.1301 | -0.0 | 0.0 | 0.0 |

## 首次分叉

- gap_widens_by_1.0nats: not reached
- gap_widens_by_2.0nats: not reached
- gap_widens_by_5.0nats: not reached

## 因果限制

- 本分析是**单条轨迹**，无干预 ⇒ 只能给 temporal association，不能给因果。
- close_mosaic 前后的差异只能报 temporal association（§17）。
- 观测点分辨率为相邻两点区间，不得假装更精确（§16）。
