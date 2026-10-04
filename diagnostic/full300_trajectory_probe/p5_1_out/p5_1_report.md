# P5.1 — W_t × h_t Offline Decomposition Report

## 硬结论

1. W_t 是否真的存在？            **是**（唯一 W hash = 30）
2. identity 是否 PASS？          **是**（key 1320，跨 30 epoch 逐位一致）
3. decomposition identity 是否 PASS？ **是**（max|R+H−T| = 3.55e-15）
4. live 极化是否在同口径复现？   见 p5_1_wt_ht_summary.csv 的 live_dead_lo 列
5. R / H 主导？                  **末观测点** R=+3.753 H=-4.002 ⇒ **Head-driven**；全程中位 R=+3.466 H=-2.955 ⇒ Head-driven
6. common-mode vacuum 是否成立？ 居中<0 占比 = 2.3%
7. dead-tail persistent/rotating？ Jaccard 中位 = 0.721
8. 是否足够设计 GPU intervention？ **否**（见 §23 门）

## 口径声明

- 三个读数（live / frozen_initial / frozen_final）**全部为 GT 单元格点值**，同一口径。
- 历史的 `cls_center_max` 是邻域最大值，**不在本表内混用**。
- **decomposition ≠ causality**：本分解只回答总 logit change 落在哪条状态变量路径上，
  不是干预实验。

## §9 公式说明

- 规格原文版（表征项用 W_final）max|R_spec+H−T| = **1.174e+01**（非零）；
  其残差恒等于 ½·(W_final−W_0)_gt·(h_t−h_0)。
- 本报告主分解使用可通过恒等式的 Shapley 标准形式，max|R+H−T| = 3.553e-15。

## GPU Gate

```
NO GPU
```