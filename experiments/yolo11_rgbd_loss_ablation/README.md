# 第一轮 Loss 消融实验（基于 F4）

- **基线**：F4 = YOLO11m RGBD Mid-Fusion @1280，best mAP50-95 = **0.53273**（ep251）
- **权重**：`runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt`
- **状态**：配置已建立、`scripts/train.py` 最小补丁已应用并验证；L1 已完成（box=10.0 → 0.52093，差于 F4），L2 待跑

---

## 1. 实验设计（单变量）

三个实验都严格复用 F4 的实际训练配置 `configs/train_f4_1280.yaml`，**唯一实验变量是 loss 权重**
（ultralytics 默认 `box=7.5 / cls=0.5 / dfl=1.5`，在 `loss.py:588-590` 作为 `loss[0/1/2]` 的增益乘子）。
三个实验统一 `batch_size=8`（与 F4、L1 一致），不构成实验变量。

| 实验 | box | cls | dfl | 相对 L0 的唯一变量 |
|---|---|---|---|---|
| L0 baseline | 7.5 | 0.5 | 1.5 | —（默认值，复现 F4） |
| L1 box10 | **10.0** | 0.5 | 1.5 | box 7.5 → 10.0 |
| L2 dfl2 | 7.5 | 0.5 | **2.0** | dfl 1.5 → 2.0 |

> 全部在 imgsz=1280、YOLO11m RGBD Mid-Fusion 模型、`use_simotm=RGBD` 4ch 管线下运行；
> 其余超参（SGD momentum=0.937/wd=5e-4、lr0=0.005、cos_lr、warmup_epochs=3、
> epochs=300、patience=80、batch=8、seed=42、amp、val_ratio=0.2、nbs=64 默认）完全一致。
>
> **batch 说明**：三个实验统一 `batch_size=8`（与 F4、L1 一致），`nbs=64` 不变 →
> `accumulate = round(nbs/batch) = 8`，有效 batch 64，`lr0` 不用调。L1 已按 batch=8 跑完，
> L0/L2 同样保持 batch=8，确保三者（及 F4）单一变量可比。

---

## 2. 配置文件（位于根目录 `configs/`）

| 实验 | 配置路径 |
|---|---|
| L0 | `configs/train_l0_baseline.yaml` |
| L1 | `configs/train_l1_box10.yaml` |
| L2 | `configs/train_l2_dfl2.yaml` |

配套产物（仍在 `experiments/yolo11_rgbd_loss_ablation/`）：
- `comparison_template.csv` — 结果回填表
- `test_loss_config.py` — 配置回归测试（不训练，仅验证 box/cls/dfl 链路）
- `README.md` — 本说明

---

## 3. 训练命令（**先不要执行**）

```bash
# L0 基线（应复现 F4，≈0.53273）
python scripts/train.py --model_config configs/yolo11m_midfusion_rgbd_concat_res.yaml \
    --train_config configs/train_l0_baseline.yaml

# L1 box=10.0
python scripts/train.py --model_config configs/yolo11m_midfusion_rgbd_concat_res.yaml \
    --train_config configs/train_l1_box10.yaml

# L2 dfl=2.0
python scripts/train.py --model_config configs/yolo11m_midfusion_rgbd_concat_res.yaml \
    --train_config configs/train_l2_dfl2.yaml
```

（`--dataset_config` 走默认 `configs/dataset.yaml`，与 F4 一致；`use_simotm=RGBD` 会触发
`_split_train_val_depth` 生成 `data/processed/depth_split_train/dataset.yaml`，同 F4。）

---

## 4. 预计输出目录

| 实验 | 输出目录（`project`/`name` 由 `checkpoint.save_dir` 解析） |
|---|---|
| L0 | `runs/yolo11_rgbd_loss_l0_baseline/`（`weights/best.pt`, `weights/last.pt`, `args.yaml`） |
| L1 | `runs/yolo11_rgbd_loss_l1_box10/` |
| L2 | `runs/yolo11_rgbd_loss_l2_dfl2/` |

---

## 5. 已解决：train.py 转发 box/cls/dfl（最小补丁）

`scripts/train.py` 的 `_build_train_kwargs` 末尾（`return kwargs` 之前）已加入：

```python
    # loss 权重（box/cls/dfl）：可选覆盖
    # 未配置时保持 Ultralytics 默认值，不改变现有实验行为
    for key in ("box", "cls", "dfl"):
        if key in train_cfg:
            kwargs[key] = float(train_cfg[key])
```

链路已核实并通过回归测试（`test_loss_config.py`，PASS / exit 0）：

```
config box/cls/dfl → _build_train_kwargs kwargs → get_cfg(overrides)
→ trainer.args → v8DetectionLoss.hyp = model.args → self.hyp.box/cls/dfl
```

未配置 box/cls/dfl 的旧 train.yaml（如 F4）不会新增 override，仍走 ultralytics 默认 7.5/0.5/1.5。

---

## 6. 单变量与兼容性自检结论

- **单变量**：L0/L1/L2 逐字段 diff 仅 `experiment_name` 与一个 loss 字段（box 或 dfl）不同，
  其余（含 batch=8）三组完全一致 → 满足单变量原则（补丁已应用，box/cls/dfl 会真正生效）。
- **梯度累积**：`nbs=64` 为框架默认（`trainer.py:301 accumulate = round(nbs/batch)=8`，batch=8），
  非项目自定义逻辑，三组一致，不构成第二变量。
- **config 落盘**：`args.yaml` 由 ultralytics 自动写入 run 目录，训练后可直接核验
  box/cls/dfl 是否真正生效（这是补丁生效的最终判据）。
