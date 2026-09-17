# 首图调试报告：rect 预测 vs model.val() save_txt（阶段五前置验证）

- **日期**：2026-09-15
- **对照对象**：`diagnostic/val_dump/val_preds/labels/`（model.val() 自身 save_txt 的 394 个 TXT）
- **被验证对象**：`diagnostic/rect_pipeline_fix/validation_rect_predictions/results/`（predict_rect.py --mode rect）

---

## 1. 框数逐张比对（前 10 张，全部 MATCH）

| image | val_save_txt 框数 | rect_pred 框数 | 判定 |
|---|---|---|---|
| 00000045 | 11 | 11 | OK |
| 00000057 | 9 | 9 | OK |
| 00000063 | 9 | 9 | OK |
| 00000095 | 11 | 11 | OK |
| 00000106 | 8 | 8 | OK |
| 00000123 | 12 | 12 | OK |
| 00000171 | 3 | 3 | OK |
| 00000180 | 4 | 4 | OK |
| 00000234 | 3 | 3 | OK |
| 00000257 | 3 | 3 | OK |

## 2. 全量框数比对

- val save_txt 共 **394** 个 TXT（val 集 400 图 − 6 张无检测）。
- rect_pred 与 val save_txt **框数 100% 一致（394/394）**。

## 3. 坐标比对：差异全部 ≤ 1e-6（纯舍入精度）

val save_txt 写 **7 位小数**（ultralytics `save_one_txt` 默认），predict.py 的 `_format_lines`
写 **6 位小数**（赛题提交格式）。二者底层浮点值**完全相同**，逐框最大差 4e-7 ~ 1e-6：

```
val : 10 0.418704 0.695113 0.0709525 0.0587597 ...   # 7 位小数
rect: 10 0.418704 0.695113 0.070952  0.058760  ...   # 6 位小数（= 同一浮点值）
      maxd = 4.0e-07
```

- 坐标级匹配（6 位小数对齐）：137~147/394；**5 位小数：214/394**——不匹配全是 6↔7 位舍入边界，
  非真实坐标偏移（maxd ≤ 1e-6）。
- 结论：**rect pipeline 的预测框与 model.val() 的预测框逐张、逐框一致**（到浮点舍入精度）。

## 4. 结论

`predict_rect.py --mode rect` 在 val 集上**逐框复现 model.val()**：
- mAP50-95 = 0.53264 ≈ model.val() 0.53273（Δ=0.00009，浮点/排序舍入）。
- 框数 100% 一致（394/394）。
- 坐标一致到 1e-6。
