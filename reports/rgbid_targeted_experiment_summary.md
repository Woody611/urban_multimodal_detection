# RGBID 定向实验汇总（E6 → E7 → E8 → E9）

**日期**：2026-09-17
**正式基线**：RGBID Early Fusion（5ch `[RGB, IR, Depth]`，YOLO11m，imgsz=1280），线上 **mAP@0.5:0.95 = 0.53180**
**本报告范围**：仅比较 RGBID baseline / E6 / E7 / E8 / E9。不重新展开 P2、旧分辨率搜索、TTA、Ensemble、conf、IoU、max_det、CrossMLCA/CMA。

---

## 0. 结论速览

| 实验 | 状态 | VERDICT |
|---|---|---|
| **E6** 5-channel pretrained initialization | ✅ **已检查完成** | **`SKIPPED_ALREADY_IMPLEMENTED`** |
| **E7** Localization Loss | ⛔ **未执行（环境阻塞）** | `SKIPPED` |
| **E8** 1536 resolution fine-tune | ⛔ **未执行（环境阻塞）** | `SKIPPED` |
| **E9** Lightweight Mid-Fusion | ⛔ **未执行（环境阻塞）** | `SKIPPED` |

**阻塞原因**：本机无 GPU（`torch 2.4.1+cpu`，`CUDA available = False`，GPU 数 = 0）。
实测 CPU 训练比云端 V100 慢约 **32×**，E7/E9 各需约 **15 天**，E8 需约 **2.5~4 天**。
按任务书「如果显存不足导致训练无法稳定进行，立即停止」的要求，**不执行 E7/E8/E9**。

---

## 1. E6 检查结果（本阶段唯一完成的实验）

### 1.1 任务要求的四项确认

| # | 检查项 | 结论 |
|---|---|---|
| 1 | 5-channel 第一层是否继承 YOLO11m RGB 预训练权重 | ✅ **是**（实测 cos 0.9251~0.9691） |
| 2 | RGB 前 3 通道是否直接使用原始预训练权重 | ✅ **是**（代码 `weight[:, :3].copy_(w)` 直接拷贝） |
| 3 | IR / Depth 两通道如何初始化 | ✅ **`mean(W_R, W_G, W_B)`**（代码 `w.mean(dim=1, keepdim=True).repeat(1, n_aux, 1, 1)`） |
| 4 | 是否因 `in_channels=5` 导致整个输入 stem 随机初始化 | ❌ **不存在此问题** |

### 1.2 代码证据

`ultralytics/models/yolo/detect/train.py::_transfer_rgb_pretrained`（L56–74）已实现 5ch 早融合分支：

```python
# ---- RGBID early fusion: first Conv absorbs all 5 channels (no 1ch stem) ----
five_ch = next((m for m in stems if getattr(m.conv, "in_channels", None) == 5), None)
if five_ch is not None and not any(getattr(m.conv, "in_channels", None) == 1 for m in stems):
    w = src.get("model.0.conv.weight")
    if w is not None and w.shape[1] == 3 and w.shape[0] == five_ch.conv.weight.shape[0]:
        n_aux = five_ch.conv.weight.shape[1] - 3
        with torch.no_grad():
            five_ch.conv.weight[:, :3].copy_(w)                      # RGB 预训练
            five_ch.conv.weight[:, 3:].copy_(
                w.mean(dim=1, keepdim=True).repeat(1, n_aux, 1, 1))   # IR/D = mean(R,G,B)
        return 5
```

**调用条件**（`get_model()`，L194–197）：只要 `weights` 非空即调用，且
trainer 用 `self.args.pretrained` 兜底加载（避免 yaml-built 模型 `self.ckpt` 为空时静默跳过）。

### 1.3 权重实证证据（决定性的）

直接比较 **训练后的 RGBID best.pt** 与 **YOLO11m 原始 stem**，逐输出 filter 展平后算余弦相似度：

| 对照情形 | cos 均值 |
|---|---:|
| RGBID ch0 (R) vs 预训练 ch0 | **+0.9691** |
| RGBID ch1 (G) vs 预训练 ch1 | **+0.9573** |
| RGBID ch2 (B) vs 预训练 ch2 | **+0.9251** |
| 随机初始化 vs 预训练（20 次均值） | **+0.0035**（范围 −0.093 ~ +0.047） |
| 对照：错位通道 RGBID-R vs 预训练-G | +0.3628 |
| 对照：错位通道 RGBID-G vs 预训练-B | +0.5667 |
| ch3 (IR) vs `mean(R,G,B)` | +0.5201 |
| ch4 (Depth) vs `mean(R,G,B)` | +0.3853 |

**判读**：
- RGB 三通道经过 **300 epoch 训练**后仍与预训练 stem 保持 cos ≈ 0.93~0.97，
  而随机初始化基线仅 **0.0035** —— 若 stem 是随机初始化的，300 轮训练不可能收敛出
  与预训练**逐 filter 一一对应**的结构。**这证明 RGB 预训练权重确实被使用。**
- IR / Depth 通道与 `mean(R,G,B)` 的 cos 为 0.52 / 0.39，符合「以均值初始化、
  再随各自模态分布分化」的预期。
- 错位通道对照（0.36 / 0.57）高于随机基线，是因为预训练 stem 各通道本身共享边缘检测结构；
  但同通道匹配（0.93~0.97）与错位匹配（0.36~0.57）分离清晰。

> **附带澄清**：IR 快速实验报告曾提示「云端可能未同步该修复」。
> 上述权重实证**已解决该疑问** —— 云端训练确实应用了该初始化，无需重跑。

### 1.4 E6 判定

> ## `E6 = SKIPPED_ALREADY_IMPLEMENTED`
> 当前 RGBID 已正确采用「RGB 权重保留 + IR/D 均值初始化」。
> **不做重复实验。**

### 1.5 E6 输出项对照（任务要求）

| 要求项 | 状态 |
|---|---|
| best.pt | ✅ 已存在（即正式基线权重，**未新建、未覆盖**） |
| results.csv | ✅ 已存在 |
| config | ✅ `configs/train_rgbird_ir_quicktest.yaml` + `configs/yolo11m_earlyfusion.yaml` |
| mAP50 / mAP50-95 / mAP75 / P / R / best epoch | ✅ 见下方基线行（无新增实验，故无差值） |

---

## 2. 对照表

| 实验 | mAP50 | **mAP50-95** | mAP75 | Precision | Recall | best epoch | 本地 vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| **RGBID baseline** | 0.81513 | **0.55800** | 0.61859 | 0.83856 | 0.74785 | 237 | — |
| E6 | — | — | — | — | — | — | **无新增（`SKIPPED_ALREADY_IMPLEMENTED`）** |
| E7 | — | — | — | — | — | — | 未执行 |
| E8 | — | — | — | — | — | — | 未执行 |
| E9 | — | — | — | — | — | — | 未执行 |

> 基线数字取自 `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/results.csv` 的 best 行（epoch 237）。
> 该权重 SHA256 `f9dddbfaca4cb08bde31d082eb0a54d499b8441eb1f97c6134a028caa6d83e55`。

### 2.1 逐实验输出项（任务要求的 11 项）

| 项 | E6 | E7 | E8 | E9 |
|---|---|---|---|---|
| EXPERIMENT | 5ch pretrained init 检查 | Localization Loss | 1536 fine-tune | Lightweight Mid-Fusion |
| BASELINE_MAP5095 | 0.55800 | 0.55800 | 0.55800 | 0.55800 |
| NEW_MAP5095 | — (无需实验) | 未执行 | 未执行 | 未执行 |
| DELTA | 0 | 未执行 | 未执行 | 未执行 |
| MAP75_DELTA | — | 未执行 | 未执行 | 未执行 |
| RECALL_DELTA | — | 未执行 | 未执行 | 未执行 |
| PARAMS | 20.06 M（沿用） | — | — | — |
| FLOPS | ≈457 GFLOPs @1280（沿用） | — | — | — |
| TRAIN_TIME | — | 估算 ≈355 h（CPU） | 估算 ≈59–95 h（CPU） | 估算 ≈355 h（CPU） |
| GPU_MEMORY | — | **本机无 GPU** | **本机无 GPU** | **本机无 GPU** |
| VERDICT | `SKIPPED_ALREADY_IMPLEMENTED` | `SKIPPED` | `SKIPPED` | `SKIPPED` |

---

## 3. E7 / E8 / E9 未执行原因（环境阻塞）

### 3.1 环境事实（实测）

| 项 | 值 |
|---|---|
| torch | **2.4.1+cpu** |
| `torch.cuda.is_available()` | **False** |
| GPU 数量 | **0** |
| CPU 逻辑核 | 14 |
| 实测 1280/5ch 单图前向 | **0.89 s** |
| 实测 fwd+bwd（估算） | **2.66 s / 图** |
| 估算每 epoch（1600 图） | **≈71 分钟** |
| 对照：云端 V100 实测 | **133.8 s / epoch（2.2 分钟）** |
| **CPU / V100 倍率** | **≈ 32×** |

### 3.2 各实验的 CPU 训练成本（不含 dataloader 开销，实际更慢）

| 实验 | 配置 | 估算耗时 |
|---|---|---|
| E7 | 300 epoch 全量（与 baseline 同条件） | **≈355 小时 ≈ 14.8 天** |
| E8 | 50~80 epoch fine-tune @1536（更慢，约 1.44× 像素） | **≈85–137 小时 ≈ 3.5–5.7 天** |
| E9 | 300 epoch 新架构 | **≈355 小时+ ≈ 15 天以上** |

### 3.3 为什么不"降低标准凑合跑"

任务书明确：「如果显存不足导致训练无法稳定进行，立即停止 E8，不要为了 1536 修改模型结构」。
同理，为避免把「设备差异」误判为「方法结论」，**不在 CPU 上跑缩水版实验**：
- 缩短 epoch / 减小 batch / 降低分辨率都会破坏「与 baseline 单变量可比」的前提；
- 其结果无法与云端训练的 baseline 直接比较，会产生误导性结论。

### 3.4 解除阻塞所需

要继续 E7/E8/E9，需要以下之一（**请由你决定并明确授权**）：

1. **云端 GPU 资源**（历史使用 AutoDL V100 32GB）——推荐，与 baseline 条件一致，可比性最强；
2. 本机接入可用 GPU；
3. 明确接受「缩水条件实验」，并同意其结论**仅作方向性参考、不作为单变量判据**（不推荐）。

**若走路线 1，还需你提供**（任务书第四阶段要求的 7 项中本机无法确认的两项）：
- 云端 GPU 型号 / 显存（决定 E8 的 1536 是否可行）
- 可用训练时长预算（E7 在 V100 上约 10.6 小时）

---

## 4. 风险与合规记录

### 4.1 本次未触碰的任何冻结产物

| 冻结对象 | 状态 |
|---|---|
| 正式 `best.pt` | **未修改、未覆盖** |
| 正式 submission (`submissions/rgbird_ir_quicktest/submission.zip`) | **未修改、未覆盖** |
| `scripts/predict_rect.py` | **未修改** |
| 历史实验 runs/ | **未删除任何目录** |
| 线上提交 | **未执行** |

本次唯一新建文件：本报告。全程只读检查（加载权重做统计比较，未写入）。

### 4.2 已证伪方向（本报告不重新展开）

P2 检测层、旧分辨率搜索（640/768/1024）、TTA、Ensemble、conf / IoU / max_det、CrossMLCA/CMA ——
均已在此前阶段证伪或裁决，本阶段不再涉及。

### 4.3 当前剩余风险

| # | 风险 |
|---|---|
| 1 | E7/E8/E9 未经实验，**目前没有任何证据表明它们会带来提升** |
| 2 | 若 E8（1536）在云端执行，显存需求未验证；1024→1280 的边际收益已有递减迹象 |
| 3 | 单次 seed=42 的 best 是尖峰（F4 的 0.53273 超出 ±10ep 均值 0.0086），**不能按 best 预期收益** |
| 4 | 当前正式成绩 0.53180 的完整链条已验证（本地→线上迁移率 84.2%），任何新实验都不应破坏它 |

---

## 5. ⚠️ 命名冲突风险（必须在开训前解决）

`runs/` 下**已存在**历史目录 `e6_* / e7_* / e8_* / e9_*`，但其含义与本任务新定义的 E6/E7/E8/E9 **完全不同**：

| 现有目录 | 日期 | 内容 |
|---|---|---|
| `urban_multimodal_det_e6_yolo11m_rgbd_768` | 2026-09-11 | RGBD 中期融合 @768 |
| `urban_multimodal_det_e7_yolo11m_rgbd_1024` | 2026-09-12 | RGBD 中期融合 @1024 |
| `urban_multimodal_det_e7_lr0half` | 2026-09-14 | RGBD 中期融合 lr0=0.005 |
| `urban_multimodal_det_e7_close_mosaic0` | 2026-09-13 | RGBD 中期融合 close_mosaic=0 |
| `urban_multimodal_det_e7_f3a_lr004` | 2026-09-14 | RGBD 中期融合 lr0=0.004 |
| `urban_multimodal_det_e8_yolo11m_rgbd_nearest` | 2026-09-12 | RGBD depth INTER_NEAREST（已证伪） |
| `urban_multimodal_det_e9_yolo11m_rgbd_depthpad0` | 2026-09-12 | RGBD depth padding=0（已证伪） |

这些都是 **RGBD 4ch 中期融合**（best.pt ≈ 61 MB），而新实验是 **RGBID 5ch 早期融合**（best.pt ≈ 40 MB）。

**风险**：若新实验沿用 `e7_*` / `e8_*` / `e9_*` 命名，会与历史目录混淆，
且违反「每个实验必须单独保存目录」（约束 #10）与「不删除历史实验」（约束 #8）的可追溯性要求。

**建议命名方案**（开训前请确认）：

```
runs/urban_multimodal_det_g7_rgbid_box9_dfl2/     # E7 Localization Loss
runs/urban_multimodal_det_g8_rgbid_1536ft/        # E8 1536 fine-tune
runs/urban_multimodal_det_g9_rgbid_midfusion/     # E9 Lightweight Mid-Fusion
```

（E6 无需目录——判定为 `SKIPPED_ALREADY_IMPLEMENTED`。）

---

## 6. 下一步

按任务书「完成一个实验后先停止并报告结果」的要求，**E6 已完成并报告，现停止**。

**等待你的决定**：

1. 是否确认 **E6 = `SKIPPED_ALREADY_IMPLEMENTED`**；
2. 是否提供**云端 GPU 资源**并授权继续 E7（E7 在 V100 上约 10.6 小时）；
3. 确认新实验的**目录命名方案**（§5），避免与历史 E6–E9 撞名；
4. 若暂无 GPU 资源，建议维持现状（RGBID baseline 0.53180 为正式方案）。

**未经明确授权，不执行任何训练，不创建任何实验目录。**
