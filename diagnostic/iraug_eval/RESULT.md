# D′ + IR-specific Augmentation — 官方口径评测结果（本地，CPU）

**日期**: 2026-10-05 · **评测链**（与 D′ 0.51528 完全同一条）：
```
predict_rect.py --mode rect --conf 0.001 --iou 0.7 --imgsz 1280 --max_boxes 100 --max_det 300 --device cpu
  → <out>/results/*.txt
official_eval.py --metric A  (398 imgs / 2807 GT，两模型完全相同)
```
产物：`Dp_best/`（控制）、`IRAug_best/`（实验）；`run_eval.log`、`run_eval.sh`。

---

## 结果

| 模型 | official (metric A) | fork 口径 | mAP50 | mAP75 |
|---|---|---|---|---|
| **D′ best（控制）** | **0.51528** | **0.56609** | 0.77826 | 0.51096 |
| **D′+IRaug best** | **0.51428** | **0.56310** | 0.78788 | 0.52786 |
| **Δ** | **−0.00100** | **−0.00299** | +0.00963 | +0.01690 |

**控制组逐位复现了已知的 incumbent**：official `0.51528` ✅、fork `0.56609` ✅
（两者都与历史记录一致）⇒ **今天的评测链正确、可直接比较**。

## 判读

1. **两个口径同号（都为负）** ⇒ 符号可信。
   （同架构族内 fork 口径 4/4 跟上线上符号，官方口径曾 4 次给错符号；此处不需取舍。）
2. **幅度落在噪声带内**（该 val 上 sd ≈ 0.003–0.004）：Δ_official = −0.00100、Δ_fork = −0.00299。
3. 结论：**IR gamma/noise 增强（0.75–1.35 / p=0.35 / σ=0.015 / p=0.20）没有带来可检测的增益**；
   也**不构成明确的损害**。假设 **NOT SUPPORTED**。

## ⚠ 不要解读 per-class / per-IoU 明细

val split 的类别分布极不平衡：

| class | GT | class | GT |
|---|---|---|---|
| person | 1073 | seat | 107 |
| animal | 733 | boat | 28 |
| car | 288 | uav | 30 |
| light | 244 | ball | 19 |
| sign | 150 | garbage_can | 57 |
| bicycle | 116 | **tricycle** | **2** |

`tricycle` 只有 **2 个 GT** ⇒ 它那 −0.10000 的「AP 暴跌」是 2 实例的离散跳动，无统计含义。
boat / ball / uav 同理。per-IoU 上 IoU=0.70 (−0.0268) 与 0.95 (−0.0404) 的反向也是
小样本 + 单种子的抖动，与 mAP50/mAP75 的上升**不构成连贯机制**。

## 决策

- **不作为改进提交**。两个口径同号为负且幅度在噪声内 ⇒ 没有提交理由。
- 不要再在这条线上盲扫参数：单种子在 sd≈0.003 的噪声带里分辨不了 0.001–0.003 的效应。
  若一定要分辨，需要**两臂各跑 seed-2**，成本 ≈ 24 GPU·h，而当前点估计为负。
- 线上不做外推（跨测试集禁止相减，local→online 也不是常数比）。

## 附：那次 threading 报错与本结果无关

run 目录体检：300/300 epoch 连续、NaN/Inf=0、LR 正常衰减到 5.01e-5、
best≠last（SHA 不同）、架构与 D′ 一致、`train_args.ir_gamma=[0.75,1.35]` 确已生效；
中位 epoch 间隔 138.9s vs D′ 131.6s = **+5.5%**（= 每图多一次 pow+noise 的开销），
**数据量正常**（若扫描残了只会更快）。异常形如 `Thread._bootstrap_inner→run` 的线程外壳，
最可能是 `@threaded plot_images`（plotting.py:2184，daemon，只画 `val_batch*.jpg`）。
D′ 与 IR-Aug 的 run 目录**都没有** `val_batch*.jpg` ⇒ 既有问题，非本次引入。
