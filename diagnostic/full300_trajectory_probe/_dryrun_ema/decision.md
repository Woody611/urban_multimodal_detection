# DECISION — Probe-P3-Trajectory

```text
CASE = A

REPRESENTATION_INITIAL_WEAKNESS= SUPPORTED
TRAINING_DEGRADATION        = NOT_SUPPORTED
TAL_SELECTION_FEEDBACK      = NOT_SUPPORTED
P3_WEAK_H_RECOVERY          = NOT_SUPPORTED
SYNCHRONIZED_MECHANISM      = SUPPORTED
CAUSALITY_LEVEL             = mechanistically supported (single trajectory, no intervention)
```

## 时间序列

| epoch | P3 acc(G86) | P3 acc(CTRL) | h acc(G86) | mask_pos率 | cls logit | raw_align |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6364 | 0.8571 | 0.6364 | 1.000 | -17.089 | 0.00003 |
| 1 | 0.6364 | 0.8571 | 0.6364 | 1.000 | -18.605 | 0.00002 |

random baseline（control 的多数类先验）= 0.8571

## 时间顺序

- 首次 P3 退化：None
- 首次 cls 退化：None
- 首次 alignment 退化：None
- 首次 mask_pos 丢失：None

> 本判决由 _probe.py::_decide 依预注册阈值自动生成；阈值写在代码内，事前固定。
