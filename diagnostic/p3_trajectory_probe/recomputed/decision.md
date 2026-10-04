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
| 0.0 | 0.186 | 0.4215 | 0.1047 | 1.000 | -9.833 | 0.00910 |
| 1.0 | 0.2326 | 0.4533 | 0.2209 | 1.000 | -10.019 | 0.01454 |
| 2.0 | 0.2326 | 0.5112 | 0.1163 | 1.000 | -9.841 | 0.01225 |
| 5.0 | 0.2558 | 0.4991 | 0.2326 | 1.000 | -9.752 | 0.01117 |
| 10.0 | 0.1628 | 0.4645 | 0.2674 | 1.000 | -9.822 | 0.00566 |
| 15.0 | 0.2209 | 0.4804 | 0.1512 | 1.000 | -9.805 | 0.00994 |
| 20.0 | 0.2326 | 0.5009 | 0.1744 | 1.000 | -9.719 | 0.00916 |
| 25.0 | 0.2093 | 0.4794 | 0.3023 | 1.000 | -9.874 | 0.00365 |
| 30.0 | 0.2209 | 0.4794 | 0.2326 | 1.000 | -9.613 | 0.00501 |

random baseline（control 的多数类先验）= 0.3617

## 时间顺序

- 首次 P3 退化：None
- 首次 cls 退化：None
- 首次 alignment 退化：10.0
- 首次 mask_pos 丢失：None

> 本判决由 _probe.py::_decide 依预注册阈值自动生成；阈值写在代码内，事前固定。
