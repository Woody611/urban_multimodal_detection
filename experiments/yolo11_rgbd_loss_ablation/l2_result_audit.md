# L2（dfl=2.0）结果审计报告

**日期**：2026-09-16
**状态**：✅ 确认为 L2，mAP50-95 ≈ 0.519，**差于 F4**；结论为负向结果，可用于 Loss 消融。

---

## 0. 结论（TL;DR）

- 本次验证对应实验 **L2（唯一变量 dfl 1.5→2.0）**，权重为 **best.pt**（非 last.pt）。
- `args.yaml` 全部关键字段正确：`box=7.5 / cls=0.5 / dfl=2.0 / batch=8 / imgsz=1280 / epochs=300 / seed=42`。
- 独立验证 mAP50-95 = **0.519**，与训练期 best（0.51872 @ep289 / max 0.51894 @ep280）一致。
- 与 F4（0.53273）差值 **−0.0137**，与 L1（0.52093）也差 **−0.0022** → **dfl=2.0 反而更差**。
- **可以用于 Loss 消融结论**：box=10.0（L1）与 dfl=2.0（L2）均劣于默认值（box=7.5 / dfl=1.5），默认损失权重为最优。

---

## 1. 逐项审计

### Q1 实验目录
`runs/yolo11_rgbd_loss_l2_df12/`

> ⚠️ 注意：**文件夹名有笔误** `df12`，但 `args.yaml` 内 `name: yolo11_rgbd_loss_l2_dfl2`、
> `save_dir: runs/yolo11_rgbd_loss_l2_dfl2`。即云端训练确实写入了 `dfl2` 目录，本地拷贝时被
> 重命名为 `df12`（`l` 误写成 `1`）。**仅目录名笔误，内容无误**，不影响结论。

### Q2 best.pt / last.pt
**best.pt**。
- last.pt = ep300：mAP50-95=0.50997，P=0.83857，R=0.701 → 与本次验证（0.519/0.812/0.727）明显不符。
- best.pt = ep289（fitness 最优）：mAP50-95=0.51872，P=0.81236，R=0.72726 → 与本次验证吻合。

### Q3 F4 / L1 / L2
**L2（dfl=2.0）**。`args.yaml` 中 `box=7.5 / cls=0.5 / dfl=2.0`，且 `model=yolo11m_midfusion_rgbd_concat_res.yaml`。

### Q4 args.yaml 关键字段
| 字段 | 值 | 期望（L2） | 判定 |
|---|---|---|---|
| box | 7.5 | 7.5 | ✅ |
| cls | 0.5 | 0.5 | ✅ |
| dfl | 2.0 | 2.0 | ✅ |
| batch | 8 | 8 | ✅ |
| imgsz | 1280 | 1280 | ✅ |
| epochs | 300 | 300 | ✅ |
| seed | 42 | 42 | ✅ |

（其余与 F4 一致：lr0=0.005、SGD momentum=0.937/wd=5e-4、cos_lr、warmup_epochs=3、
patience=80、use_simotm=RGBD、channels=4、pretrained=yolo11m.pt）

### Q5 验证设置
| 项 | 判定 | 依据 |
|---|---|---|
| 相同验证集 | ✅ | Images=398 / Instances=2807 = depth_split_train val 切分（val_ratio=0.2），与 F4/L1 一致 |
| imgsz=1280 | ✅ | args.yaml imgsz=1280；GFLOPs=456.86（YOLO11m）且指标与 results.csv 1280 吻合 |
| rect=True | ✅ | `model.val()` 强制 rect=True（engine/model.py:598），evaluate.py 走同一路径 |
| F4 相同指标口径 | ✅ | 数值与 fork DetMetrics 结果一致，mAP50-95 可直接与 F4=0.53273 比较 |

### Q6 最佳 epoch
- **fitness 最优（= best.pt）**：**ep289**，mAP50-95=0.51872，mAP50=0.79156，mAP75=0.53259，P=0.81236，R=0.72726
- **mAP50-95 最大**：ep280，0.51894
- 末 20 ep 均值 0.51333 / 末 30 ep 均值 0.51501（无尖峰，训练平稳收敛）

### Q7 全类指标与逐类 AP
**all-class**（独立验证，本次提交）：
| P | R | mAP50 | mAP75 | mAP50-95 |
|---|---|---|---|---|
| 0.812 | 0.727 | 0.791 | 0.534 | 0.519 |

**逐类 AP50-95**：`results.csv` 只记录聚合指标，**不含逐类**。需在云端 GPU 跑一次：
```bash
python scripts/evaluate.py \
  --weights runs/yolo11_rgbd_loss_l2_dfl2/weights/best.pt \
  --train_config configs/train_l2_dfl2.yaml
```
（本地 Windows 为纯 CPU，imgsz=1280 全验证集太慢，不在本地跑；逐类 AP 不影响本次结论。）

### Q8 与 F4 差值
| 实验 | 唯一变量 | best mAP50-95 | Δ vs F4 |
|---|---|---|---|
| F4 | 默认 box=7.5 / dfl=1.5 | 0.53273 | — |
| L1 | box=10.0 | 0.52093 | **−0.01180** |
| L2 | dfl=2.0 | 0.51894（max）/ 0.51872（best.pt）/ 0.519（独立验证） | **−0.01379 ~ −0.01401** |

---

## 2. 是否可用于 Loss 消融结论

**可以，作为负向结论。**

- L2（dfl=2.0）mAP50-95 ≈ 0.519，**低于** F4（0.53273）约 −0.0137，也**低于** L1（0.52093）约 −0.0022。
- 结合 L1（box=10.0 → 0.52093，−0.0118）：**两个方向（加大 box、加大 dfl）都劣于默认值**。
- 结论：ultralytics 默认损失权重 **box=7.5 / cls=0.5 / dfl=1.5** 在该任务上为当前最优，Loss 权重不是提升杠杆。

---

## 3. 后续动作

- ❌ 不启动 L3、不启动其他实验（YOLO11l/x、RGBD+IR、HHA、Inverse Depth、TTA 等）。
- ⏸️ 等待确认后再决定是否回填 `comparison.csv` / `README.md` 结果表（F4/L1/L2 三行合一）。
- 📌 若需逐类 AP 分析，在云端执行 Q7 的 evaluate.py 命令。
