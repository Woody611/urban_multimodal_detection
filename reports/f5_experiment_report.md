# F5 实验报告：YOLO11l RGBD 中期融合 @1280（容量杠杆，m→l 单变量）

- **实验定义**：F5 = F4 唯一改动 **模型规模 m → l**（scale）。保持 `imgsz=1280`，验证「容量」杠杆。
- **基线**：F4 = `runs/urban_multimodal_det_yolo11_rgbd_f4_1280`（YOLO11m RGBD @1280，best mAP50-95=0.53273）。
- **结果目录**：`runs/urban_multimodal_det_yolo11_rgbd_f5_l1280/`（独立目录，未 resume F4，从头训练）。
- **结论速览**：best mAP50-95 **0.52619**（−0.00654，−1.23%）。**容量杠杆证伪**——l 未提升、反而略降；签名是 mAP50↑ 但 mAP75 大幅↓（定位变差）。

---

## 1. F4 / F5 配置 diff（唯一变量）

对 F5 落盘的 `args.yaml` 与 F4 的 `args.yaml` 做语义 diff：

| 字段 | F4 | F5 | 说明 |
|---|---|---|---|
| **model / scale** | yolo11m yaml / **m** | yolo11l yaml / **l** | ✅ 唯一实验变量 |
| pretrained | yolo11m.pt | yolo11l.pt | 随 scale 联动 |
| name / save_dir | …_f4_1280 | …_f5_l1280 | 仅目录命名 |

其余**全部一致**：`imgsz=1280, epochs=300, patience=80, batch=8, lr0=0.005, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3, cos_lr=true, close_mosaic=10, amp=true, nbs=64, seed=42, deterministic=true, optimizer=SGD, use_simotm=RGBD, channels=4, pairs=[visible,depth], box=7.5/cls=0.5/dfl=1.5, mosaic=1.0, erasing=0.4` 等。

→ **单变量成立**。且注意：`args.yaml` 显示 F5 实际 **batch=8**（训练前审计计划的是 batch=4 保显存，实际跑的是 8）——这反而让单变量更纯粹（batch 与 F4 完全一致），并印证了训练前审计 §3.2 的纠正：l 仅为 m 的 1.26×，batch=8 在 V100 32GB 可行。

---

## 2. F5 实际训练配置（args.yaml 实读）

| 项 | 值 |
|---|---|
| model / scale | configs/yolo11l_midfusion_rgbd_concat_res.yaml / **l** |
| 参数量 / 计算量 | 37.57M / 571.34 GFLOPs @1280（fused 摘要；thop 冒烟 575.95） |
| imgsz / batch | 1280 / **8** |
| optimizer / lr0 / lrf | SGD(动量0.937, wd=5e-4) / 0.005 / 0.01 |
| scheduler | cos_lr, warmup 3 ep |
| epochs / patience | 300 / 80 |
| nbs / 梯度累积 | 64 / 8（有效 batch=64） |
| amp / close_mosaic | true / 10 |
| use_simotm / channels | RGBD / 4 |
| pretrained / seed | yolo11l.pt / 42 |

---

## 3. 冒烟测试结果（开训前）

`scripts/f5_smoke_test.py`，**15 PASS / 0 FAIL**（本地 CPU，batch=2）。关键项：scale='l'、nc=12、params=37.62M、顶层模块数=45（与 m 一致 → `_transfer_rgb_pretrained` 硬编码 index 仍有效）、SilenceChannel×2+ADD×3、forward P3/P4/P5=160/80/40、loss 有限、**depth 分支 stem 收到梯度**（|grad|=1.06e+01）。详见 `reports/f5_pre_training_audit.md`。

---

## 4. 最终指标（results.csv 实读）

| 项 | F4 (m) | F5 (l) | Δ | 相对 |
|---|---|---|---|---|
| **best mAP50-95** | **0.53273** | **0.52619** | **−0.00654** | −1.23% |
| **best epoch** | 251 | 228 | — | — |
| mAP50 | 0.78026 | 0.78835 | +0.00809 | +1.04% |
| mAP75 | 0.59124 | 0.55207 | **−0.03917** | **−6.62%** |
| Precision | 0.84029 | 0.84308 | +0.00279 | — |
| Recall | 0.72177 | 0.72164 | −0.00013 | — |

**关键签名**：mAP50（宽松 IoU）略升、**mAP75（严格 IoU）大幅降**、mAP50-95（均值）净降。
→ l 在 1600 张训练集上**轻微过拟合**，框回归更噪（定位精度变差），严格 IoU 掉分拖累平均分。

---

## 5. 稳定性 / 平台期（窗口均值）

| 窗口 | F4 | F5 | Δ |
|---|---|---|---|
| ep150–200 | 0.50355 | 0.50397 | +0.00042 |
| ep200–250 | 0.52309 | 0.51764 | **−0.00545** |
| ep250–300 | 0.52292 | 0.52012 | −0.00280 |
| ep290–300 | 0.51882 | 0.51425 | −0.00457 |
| 末 30 ep | 0.52191 | 0.51879 | −0.00312 |

- F5 best 超出其 ±10 ep 均值 **+0.00613**（F4 为 +0.00860）；落在 best±0.005 内的 epoch 数 **39/300**（F4 仅 10/300）。
- 即 F5 的 best **比 F4 更「实在」**（不是靠单点运气），但即便如此，**平台期也稳定地略低于 F4**——不是「best 尖峰刚好没超过」，而是**全程矮一截**。
- `val/box_loss` 末段约 0.514，训练健康、无发散（同 F4 的尾部回落是 close_mosaic=10 的已知效应）。

---

## 6. 逐类结果（待补）

> 逐类 val 正在本地重跑（复刻训练内置 val：conf=0.001/iou=0.7/max_det=300/imgsz=1280，batch=4），
> 完成后补入 `reports/f5_class_metrics.csv` 及下表。当前先给 F4 逐类作参照。

| 类 | F4 AP50 | F4 AP75 | F4 AP50-95 | F5 AP50-95 | Δ |
|---|---|---|---|---|---|
| person | 0.784 | 0.510 | 0.490 | — | — |
| sign | 0.674 | 0.443 | 0.430 | — | — |
| bicycle | 0.624 | 0.491 | 0.424 | — | — |
| ball | 0.678 | 0.430 | 0.427 | — | — |

---

## 7. 判级（对照预登记判据）

训练前审计 §5 预登记的判据（best 口径）：

| 档位 | 阈值 | F5 best |
|---|---|---|
| 强成功 | ≥ 0.545 | |
| 明显有效 | 0.538 ~ 0.545 | |
| 小幅有效 | 0.534 ~ 0.538 | |
| **基本无效** | **< 0.534** | **0.52619 ✅ 落入此档** |

**判级：基本无效（负收益）。** 容量杠杆（m→l）没有兑现，反而 −0.0065。若按平台期口径（末30 ep −0.003）同样为负。

---

## 8. 结论：容量假设被证伪

F4 报告 §15 的假设是「1280 后 Recall 上升、Precision 下降（bicycle −0.078 / ball −0.060），说明 m 容量见顶，加容量可保住 Recall、补回 Precision」。

**F5 证伪了它**：
1. **Precision 没有补回来**：P 0.84308 vs 0.84029（+0.003，等于没动）。1280 的 precision 下降是**高分辨率带来更多小框 → FP 固有增加**，不是 m 判别能力不足。
2. **Recall 也没有继续涨**：R 0.72164 vs 0.72177（持平）。m 的 recall 已到该分辨率下的饱和，l 加不出更多。
3. **唯一实质变化是 mAP75 大跌**（−0.039）：l 的额外 7.3M 参数在 1600 张上过拟合，框回归更噪、定位更差。

→ **容量不是瓶颈，数据量/分辨率-正则化平衡才是。** 在 1600 张训练集上，m（30.34M）已经够用，l（37.62M）只会过拟合。继续加容量（x / 更深）方向应放弃。

**保留 F4（m@1280）为提交基线。** `predict.py` 已指回 F4，无需改。

---

## 9. 下一步建议

F1 误差诊断（`f1-error-diagnosis.md`）的实验排序是「1280 → YOLO11l → 深度掩码」。现在 ①1280 ✅ 兑现、②YOLO11l ❌ 证伪，剩下 ③深度掩码，但需结合本实验的**新证据**重新定位：

- **真实短板从「漏检」转向「定位精度」**：F4@1280 的 mAP75=0.591、F5 又掉到 0.552，严格 IoU 的定位是当前损失的大头。下一杠杆应瞄准**框回归精度/定位**，而非容量。
- **候选方向（按优先级）**：
  1. **深度掩码 / depth 表达**（原诊断 ③）：用 depth 通道做前景/背景先验，可能同时改善定位与抑制 FP，且不增加模型容量——与「m 已够用」的结论自洽。
  2. **定位专项**：如更高精度的 anchor-free 回归头损失（IoU/GIoU/CIoU 变体）、或 DFL 超参微调——直接针对 mAP75 掉分。
  3. **数据/正则**：1600 张偏少，l 过拟合是证据；若坚持加容量，需先扩数据或加强正则（但这与「不追求容量」矛盾，仅作背景说明）。
- **不建议**：继续加容量（x/更深）、450ep 延训（F4/F5 在 ep250 后均已平台回落）、TTA/conf 刷分（违规）。

---

## 附注（口径）

- **评估标准**：全程以训练内置 val（`results.csv`）为准，未用 `evaluate.py`。
- **batch 差异**：F5 实际 batch=8（与 F4 一致），逐类 val 本地重跑为 batch=4（与 F4 逐类同口径，主指标 mAP50-95 在 batch 8↔4 间已知吻合到 0.0003）。
- **产出物**：`reports/f5_pre_training_audit.md`（训练前审计）、`scripts/f5_smoke_test.py`、`scripts/f5_val_per_class.py`、`reports/f5_class_metrics.csv`。
