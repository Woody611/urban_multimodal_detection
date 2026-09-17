# G7 定位损失实验报告（RGBID Loss Localization）

**日期**：2026-09-17
**实验名**：`G7_RGBID_loss_localization`
**目录（预定）**：`runs/urban_multimodal_det_yolo11_rgbid_g7_loss_localization/`
**最终判定**：**`BLOCKED_NO_GPU`**

---

## 0. 结论

> # `G7 = BLOCKED_NO_GPU`
>
> **本机无 GPU，无法执行训练。** 按你的明确指令「如果 GPU 不可用：不要在 CPU 上训练，
> 直接 G7 = BLOCKED_NO_GPU，不要降低 imgsz、epoch、batch 后伪造一个可比实验」——
> **未训练、未创建实验目录、未生成任何 best.pt。**
>
> **但 G7 的全部可执行产物已准备就绪**（配置文件 + 校验 + 训练命令），
> 云端 GPU 一旦可用即可直接开跑，无需任何额外改动。

---

## 1. GPU 与执行环境（对应报告项 3、4、5）

### 1.1 环境实测

| 检查项 | 结果 |
|---|---|
| `nvidia-smi` | **command not found** |
| `torch.__version__` | **2.4.1+cpu** |
| `torch.cuda.is_available()` | **False** |
| `torch.cuda.device_count()` | **0** |
| `torch.version.cuda` | **None** |
| `~/.ssh/` | 仅 `known_hosts`，**无密钥、无 config** → 无已配置的云端执行通道 |

**结论**：云端 GPU（历史使用的 AutoDL V100）**在本会话中不可访问**，无远程执行通道。

### 1.2 若强行在 CPU 上训练的成本（实测）

| 项 | 本机 CPU | 云端 V100 |
|---|---|---|
| 单图 forward (1280, 5ch) | 0.89 s | — |
| 单图 fwd+bwd（估） | 2.66 s | — |
| 每 epoch（1600 图） | **≈71 分钟** | **133.8 s（2.2 分钟）** |
| 倍率 | **≈ 32×** | — |
| **300 epoch 全量** | **≈355 小时 ≈ 14.8 天** | ≈10.6–11.2 小时 |

→ 按指令**不在 CPU 上跑**。

### 1.3 报告项对照

| 报告项 | 值 |
|---|---|
| **3. 训练时间** | **未训练**（预估 V100 ≈10.6–11.2 h / CPU ≈355 h） |
| **4. GPU** | **无**（`cuda.is_available=False`，`device_count=0`） |
| **5. peak VRAM** | **未测得**（无 GPU；参照 baseline F4@1280 batch=8 在 V100 32GB 未 OOM） |

---

## 2. Baseline 配置（对应报告项 1）

| 项 | 值 |
|---|---|
| 实验名 | `urban_multimodal_det_yolo11_rgbird_ir_quicktest` |
| 权重 | `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt` |
| SHA256 | `f9dddbfaca4cb08bde31d082eb0a54d499b8441eb1f97c6134a028caa6d83e55` |
| 模型 | YOLO11m + RGBID 早期融合，5ch `[R,G,B,IR,D]` |
| 配置 | `configs/yolo11m_earlyfusion.yaml` + `configs/train_rgbird_ir_quicktest.yaml` |
| imgsz / batch / nbs | 1280 / 8 / 64（有效 batch 64） |
| epochs / patience | 300 / 80 |
| optimizer | SGD（momentum 0.937，wd 5e-4），lr0 0.005，cos_lr，warmup 3 ep |
| amp / seed / deterministic | true / 42 / true |
| 增强 | 全部 ultralytics 默认（mosaic 1.0, fliplr 0.5, scale 0.5, erasing 0.4, close_mosaic 10） |
| **loss 权重** | **box=7.5, cls=0.5, dfl=1.5**（走 ultralytics 默认，config 中无此键） |
| 线上成绩 | **mAP@0.5:0.95 = 0.53180** |

---

## 3. G7 配置（对应报告项 2）

**配置已生成**：`configs/train_rgbid_g7_loss_localization.yaml`
**SHA256**：见下方 §6 校验

### 3.1 当前 RGBID 基线与 G7 的完整差异（语义 diff，已程序化验证）

| 变更项 | baseline | G7 |
|---|---|---|
| `box` | **7.5**（ultralytics 默认，键缺失） | **9.0** |
| `cls` | **0.5**（ultralytics 默认，键缺失） | **0.35** |
| `dfl` | **1.5**（ultralytics 默认，键缺失） | **2.0** |
| `experiment_name` | `..._rgbird_ir_quicktest` | `..._rgbid_g7_loss_localization` |

**其余全部逐字节一致**（已程序化核验，共 21 项）：

```
OK  seed=42           OK  pretrained=yolo11m.pt    OK  epochs=300
OK  patience=80       OK  batch_size=8             OK  image_size=[1280,1280]
OK  learning_rate=0.005                            OK  warmup_epochs=3
OK  optimizer={SGD, momentum 0.937, wd 5e-4}       OK  scheduler=CosineAnnealingLR
OK  amp=True          OK  val_ratio=0.2
OK  use_simotm=RGBID  OK  pairs_rgb_ir=[visible,infrared]
OK  pairs_rgb_depth=[visible,depth]                OK  channels=5
OK  checkpoint（save_dir/save_interval/save_best/save_last/resume）
```

### 3.2 训练管线转发验证（已实测）

调用 `scripts/train.py::_build_train_kwargs` 实际展开 G7 与 baseline 的 kwargs 并逐项对比：

| kwargs 项 | baseline | G7 |
|---|---|---|
| `box` | `<无>`（→ 默认 7.5） | **9.0** |
| `cls` | `<无>`（→ 默认 0.5） | **0.35** |
| `dfl` | `<无>`（→ 默认 1.5） | **2.0** |
| `name` | `...rgbird_ir_quicktest` | `...rgbid_g7_loss_localization` |
| `imgsz` / `epochs` / `batch` / `lr0` / `optimizer` / `cos_lr` / `seed` / `workers` / `device` / `data` / `pretrained` / `amp` / `patience` / `momentum` / `weight_decay` / `warmup_epochs` / `save_period` / `pairs_*` / `channels` / `use_simotm` | — | **全部一致** |

**差异项共 4 个：3 个 loss 权重（唯一实验变量）+ 1 个 `name`（独立输出目录，非训练变量）。** ✅ 单变量成立。

---

## 4. 指标对照表（对应报告项 6–15）

| 报告项 | 指标 | Baseline RGBID | G7 | Δ |
|---|---|---:|---:|---:|
| 7 | mAP50 | 0.81513 | **未测** | — |
| 9 | **mAP50-95** | **0.55800** | **未测** | — |
| 11 | mAP75 | 0.61859 | **未测** | — |
| 13 | Precision | 0.83856 | **未测** | — |
| 13 | Recall | 0.74785 | **未测** | — |
| 6 | best epoch | 237 | **未测** | — |
| 14 | **ΔmAP50-95** | — | — | **未测** |
| 15 | **ΔmAP75** | — | — | **未测** |

> Baseline 数字取自 `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/results.csv` 的 best 行（epoch 237）。
> G7 未训练，故无任何指标。

---

## 5. 最终判定（对应报告项 16）

> # `BLOCKED_NO_GPU`
>
> 未训练、未产出 best.pt、未产出 results.csv、未创建实验目录。
> **不判定 CLEAR_GAIN / NO_CLEAR_GAIN** —— 无数据，不做任何推断。

| 判定选项 | 适用性 |
|---|---|
| `CLEAR_GAIN` | ❌ 未训练 |
| `NO_CLEAR_GAIN` | ❌ 未训练 |
| `FAILED` | ❌ 未尝试 |
| **`BLOCKED_NO_GPU`** | ✅ **本机无 GPU，按指令直接判定** |

---

## 6. 已准备就绪的交付物（云端可直接开跑）

| 文件 | 状态 | SHA256 |
|---|---|---|
| `configs/train_rgbid_g7_loss_localization.yaml` | ✅ 已生成、已验证 | 见下 |
| `configs/yolo11m_earlyfusion.yaml` | ✅ 已存在（复用，未改动） | `edc95f46…d9399` |

### 云端训练命令（复制即用）

```bash
python scripts/train.py \
  --model_config configs/yolo11m_earlyfusion.yaml \
  --train_config configs/train_rgbid_g7_loss_localization.yaml
```

- 输出目录：`runs/urban_multimodal_det_yolo11_rgbid_g7_loss_localization/`（**新目录，不覆盖 baseline**）
- 预估时长：V100 约 **10.6–11.2 小时**（300 epoch）

### 云端执行前检查清单

1. ☐ 云端已同步 `configs/train_rgbid_g7_loss_localization.yaml`
2. ☐ 云端 `ultralytics/models/yolo/detect/train.py` 已含 5ch 预训练 remap（否则 E6 的初始化失效）
3. ☐ `runs/urban_multimodal_det_yolo11_rgbid_g7_loss_localization/` **不存在**（避免覆盖）
4. ☐ baseline 目录 `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/` 只读保留
5. ☐ 启动日志确认 `box=9.0 cls=0.35 dfl=2.0`

### 训练后需回填的指标

best epoch / mAP50 / mAP50-95 / mAP75 / P / R / 训练时长 / peak VRAM，
并计算 ΔmAP50-95 与 ΔmAP75，然后按 §5 重新判定 `CLEAR_GAIN` 或 `NO_CLEAR_GAIN`。

---

## 7. 合规与冻结记录

### 7.1 本次未触碰的冻结产物（SHA256 复核全部 `OK`）

| 对象 | 状态 |
|---|---|
| 正式 `best.pt`（RGBID 0.53180 对应） | **未修改、未覆盖** |
| F4 备用 `best.pt` | **未修改、未覆盖** |
| 正式 submission.zip | **未修改、未覆盖** |
| `scripts/predict_rect.py` | **未修改** |
| `scripts/predict.py` | **未修改** |
| 历史实验 runs/（含历史 E6–E9） | **未删除任何目录** |

### 7.2 关于路径的再次说明

你在指令中引用的正式路径 `runs/urban_multimodal_det_yolo11_rgbid_earlyfusion/`
**在仓库中不存在**。按你此前的指示「以仓库现有正式 RGBID best.pt 为准」，
本报告所有 baseline 引用均指向 `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt`。
（该结论已在 `reports/final_submission_audit.md` §0 记录，本次复核仍然成立。）

### 7.3 未执行的动作

未训练 / 未创建实验目录 / 未生成 best.pt / 未上传 / 未提交 /
未修改 `predict_rect.py` / 未覆盖 baseline 任何产物 / 未执行 E8、E9、P2、TTA、Ensemble、
conf 搜索、IoU 搜索、max_det 搜索、CrossMLCA/CMA。

---

## 8. 下一步

按你的指令「完成 G7 后立即停止，等待人工决策」，**现停止**。

**需要你的决定**：

1. **提供云端 GPU 资源**（或把本报告 §6 的命令在云端执行），然后回填指标后重新判定；
2. 若在 **2026-09-20 20:00 截止前**无法获得 GPU，则 G7 无法在本轮完成，
   建议维持现状（RGBID baseline 线上 **0.53180** 为正式方案）。

**未获明确授权，不执行任何训练，不创建任何实验目录。**
