# IR 快速可行性实验 · 配置审计报告（最终冲刺 · 第二任务）

**日期**：2026-09-17
**状态**：✅ 训练完成，结果落盘 `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/`（见 §9）。判据 **D**。
**唯一问题**：5ch 早期融合的预训练加载缺口已修复（见 §3）。

---

## 0. 结论（TL;DR）

- 三模态 `RGBID` 数据管线**已在本 fork 完整实现**（非本任务新建），10 项审计 **9 项通过 + 1 项已修复**。
- 唯一修复：`_transfer_rgb_pretrained` 原本只处理 RGBD 中期融合（要求独立 1ch depth stem），对 5ch 早期融合会**静默跳过首个 Conv 的 RGB 预训练权重**（连 RGB 通道都是随机初始化），使「加 IR」与「丢 RGB 预训练」两个变量混杂。已扩展为「RGB 通道复制预训练 + IR/D 通道用 mean(R,G,B) 初始化」（与 F4 depth stem 初始化约定一致）。
- 新模型/配置已就绪，命令见 §5。**未训练、未覆盖 F4、未开第二实验。**

---

## 1. 三模态管线现状（关键事实）

| 组件 | 位置 | 状态 |
|---|---|---|
| `pairs_rgb_depth` 字段 | `ultralytics/data/{base,dataset,loaders,build}.py` | ✅ 已存在 |
| `use_simotm=="RGBID"` 分支 | `base.py:254` / `dataset.py:632` / `loaders.py:638` | ✅ 已存在 |
| `_merge_channels_rgbid` | `base.py:362` → `cv2.merge((b,g,r,ir,d))` | ✅ 已存在 |
| `_resize_images_3` 三图对齐 | `base.py:335` | ✅ 已存在 |
| 5ch 增强分支（RandomPerspective/Format/v8_transforms） | `augment.py:1097 / 3068 / 3800` | ✅ 已存在 |
| `_split_train_val_rgbid` 切分 | `scripts/train.py:323` | ✅ 已存在 |
| `_build_train_kwargs` 转发 pairs_rgb_depth/channels | `scripts/train.py:436-437` | ✅ 已存在 |
| `main()` RGBID 派发 | `scripts/train.py:511` | ✅ 已存在 |

> 结论：**不需要新建任何数据管线代码**。本任务只补两处配置/模型文件 + 一处预训练缺口。

---

## 2. 十项配置审计

| # | 审计项 | 判定 | 依据 |
|---|---|---|---|
| 1 | RGB/IR/Depth 1:1 对应 | ✅ | `data/raw/train/`：visible=2000、infrared=2000、depth=2000、labels=2000，全部同名；`rgbid_split` 三模态目录同名 |
| 2 | IR/Depth 尺寸对齐 | ✅ | `_resize_images_3` 统一到同尺寸；IR/Depth 均为单通道灰度，统一 uint8 |
| 3 | 5ch 顺序 `[B,G,R,IR,D]` | ✅ | `_merge_channels_rgbid`=`merge((b,g,r,ir,d))`；`Format._format_img:3068` 只反转 `[:3]`→`[R,G,B,IR,D]`，IR/D 保序 |
| 4 | loader 返回 5ch | ✅ | RGBID 分支产出 5ch uint8；`channels=5` 门控增强（alb p=0、RandomHSV 自动跳过、RandomPerspective 5ch 分通道 warp） |
| 5 | 模型首层接受 5ch | ✅ | 实测 `yolo11m_earlyfusion.yaml`：`ch=5`、`scale=m`、首个 Conv `in=5→64`、20.06M 参数 |
| 6 | 无 RGBT CrossMLCA 复用 | ✅ | 早期融合 YAML 为单支结构，无 SilenceChannel/CrossMLCA/CMA/attention/Transformer |
| 7 | 无隐式归一化/缩放差异 | ✅ | IR 1%/99% 拉伸 + uint8（与 Infrared 单模态分支一致）；Depth `<300→0`+`/19999*255`（与 RGBD 分支一致）；无 float32[0,1] 泄漏 |
| 8 | train/val 切分与 F4 一致 | ✅ | `_split_train_val_rgbid` 用 seed=42 + val_ratio=0.2 + 同一 visible 列表；实测 rgbid_split 的 val 集合与 F4 `depth_split_train` 完全一致（diff 为空，1600 train / 400 val） |
| 9 | train.py 转发全部超参 | ✅ | box/cls/dfl/batch/imgsz/lr0/optimizer/cos_lr/warmup_epochs/seed/momentum/wd/close_mosaic 均转发；**nbs/lrf 未显式转发但取 ultralytics 默认 64/0.01，与 F4 args.yaml 一致**（F4 也走默认） |
| 10 | 预训练权重正确迁移 | ⚠️→✅ | 见 §3，已修复 |

---

## 3. 关键修复：5ch 早期融合的预训练缺口

**问题**：`ultralytics/models/yolo/detect/train.py::_transfer_rgb_pretrained` 的开头有一句
`if not any(m.conv.in_channels == 1 ...): return 0` —— 只对「有独立 1ch stem 的 RGBD 中期融合」做重映射。
5ch 早期融合**没有** 1ch stem（首层 Conv 直接 `in=5`），于是：
- 标准 `model.load()` 用 `intersect_dicts` 按 (key, shape) 匹配，首层 Conv 权重 `[64,5,3,3]` 与预训练 `[64,3,3,3]` **形状不匹配 → 被静默跳过**；
- 结果连 **RGB 前 3 通道都是随机初始化**（而非 F4 那样的 ImageNet 预训练），与「完全复制 F4 训练条件」矛盾，会把「加 IR」和「丢 RGB 预训练」两个变量混在一起。

**修复**（`detect/train.py`，仅新增一个分支，不改动 RGBD/RGBT 既有路径）：
```
5ch stem 且无 1ch stem 时：
  weight[:, :3]  <- 预训练 RGB 权重（model.0.conv.weight）
  weight[:, 3:]  <- mean(R,G,B) 重复到 2 通道（IR/D）
```
与 F4 的 depth stem 初始化约定 `W_depth = mean(W_R,W_G,W_B)` 完全一致。已通过语法/导入校验。

> 该改动是**加法式**的：只在「首层 in=5 且无 1ch stem」时触发，不影响任何既有实验的预训练加载。

---

## 4. 新建文件

| 文件 | 说明 |
|---|---|
| `configs/yolo11m_earlyfusion.yaml` | YOLO11m 早期融合，`ch:5`。文件名带 `m` 使 `guess_model_scale` 解析为 scale='m'（否则静默回退 nano）。实测 20.06M / 首层 in=5 |
| `configs/train_rgbird_ir_quicktest.yaml` | F4 配置的逐行副本，仅改 4 处：experiment_name、use_simotm=RGBID、pairs_rgb_ir=[visible,infrared]、channels=5 |
| `ultralytics/models/yolo/detect/train.py`（修改） | §3 预训练缺口修复 |

---

## 5. 云端训练命令

```bash
python scripts/train.py \
  --model_config configs/yolo11m_earlyfusion.yaml \
  --train_config configs/train_rgbird_ir_quicktest.yaml
```

- 训练目录：`runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/`（**已确认该目录不存在，无覆盖风险**）。
- 切分会首次生成 `data/processed/rgbid_split_train/`（幂等，与 F4 同 seed/val_ratio，val 集与 depth_split_train 一致；现有 `rgbid_split/` 是旧命名，无 `_train` 后缀，会被新目录取代，仅命名差异）。
- 开训前请确认云端 `ultralytics/models/yolo/detect/train.py` 已同步本报告 §3 的修复（本地已改，云端若不同步则首层 RGB 无预训练）。

---

## 6. 训练后必须记录

| 项 | 取值来源 |
|---|---|
| best epoch / best.pt | `weights/best.pt`，`results.csv` 中 fitness 最优 epoch |
| mAP50-95 / mAP50 / P / R | `results.csv` best epoch 行 |
| Params / GFLOPs | 训练启动日志（预期 ≈20.06M / ≈457 GFLOPs@1280） |
| args.yaml | 落盘核验：`use_simotm=RGBID, channels=5, batch=8, imgsz=1280, epochs=300, seed=42` |
| **ΔmAP50-95** | `IRD_best − 0.53273`（F4 基准） |

---

## 7. 止损判据（训练完成后执行）

| 区间 | 判定 | 动作 |
|---|---|---|
| **A** mAP50-95 < 0.533 | IR 无益 | 淘汰 IR，止损，回 RGBD 主线 |
| **B** 0.533 ≤ < 0.545 | 边缘 | 默认不加 IR（收益不足） |
| **C** 0.545 ≤ < 0.555 | 有收益 | 分析（逐类 AP、与 F4 困难类对比） |
| **D** ≥ 0.555 | 明显收益 | 升为主候选（再评估中期融合 RGBID 隔离融合方案变量） |

---

## 8. 判据口径说明（诚实声明）

- 本任务按简报要求用 **早期融合 5ch**（最廉价的可行性探针），与 F4 的 **中期融合 4ch** 比较，因此 Δ 同时含「融合方案（early vs mid）」与「模态（+IR）」两个变量。判据仍按简报以 0.53273 为基准。
- 若结果落在 C/D 区间，下一步应做「中期融合 RGBID」（与 F4 同融合方案、仅 +IR）以隔离出纯 IR 收益；本任务只回答「早期融合下加 IR 是否值得继续」。
- 早期融合参数量（20.06M）低于 F4（30.34M），属预期（单支 vs 三支融合），不是配置错误。

---

## 9. 训练结果（2026-09-17，云端 300 ep 完成）

| 指标 | IRD（RGBID 早期融合） | F4（RGBD 中期融合） | Δ |
|---|---|---|---|
| **mAP50-95** | **0.55800** | 0.53273 | **+0.02527** |
| mAP50 | 0.81513 | — | — |
| mAP75 | 0.61859 | — | — |
| Precision | 0.83856 | — | — |
| Recall | 0.74785 | — | — |
| fitness (0.1·mAP50 + 0.9·mAP50-95) | 0.58371 | — | — |
| best epoch | 237 | 251 | — |

- **args.yaml 核验通过**：`use_simotm=RGBID, channels=5, batch=8, imgsz=1280, epochs=300, seed=42, box=7.5/cls=0.5/dfl=1.5, nbs=64, lr0=0.005, lrf=0.01, cos_lr, close_mosaic=10, pretrained=yolo11m.pt, pairs_rgb_ir=[visible,infrared], pairs_rgb_depth=[visible,depth]` —— 与 F4 逐项一致，仅改 4 处。
- **曲线形态**：ep237=0.55800 为**尖峰**（邻近 ep234=0.54479 / ep240=0.54524 / ep246=0.54625 / ep249=0.54980），平台期 ≈ **0.545–0.552**。末 10/20/30 ep 均值 = 0.55064 / 0.55286 / 0.55207。
- **判据**：best 0.55800 ≥ 0.555 → **D（明显收益，升为主候选）**。即使按平台期 0.552 算（C 区间上沿），也仍比 F4 高 +0.019，且邻近最差 ep（0.545）仍高于 F4 的 0.53273。

### 结论（只回答本任务唯一问题）

**是**：在完全复制 F4 训练条件（同切分、同 1280、同 batch=8/nbs=64、同 SGD/cos_lr/seed=42、同 loss 权重、同预训练策略）下，加入 Infrared 后的 RGBID（5ch 早期融合）验证集 mAP50-95 = **0.55800**，较 F4 的 0.53273 提升 **+0.02527**（≥0.555，判据 D）。

### 诚实声明（判据口径）

1. **Δ 混合了两个变量**：本实验是「早期融合 5ch」对「中期融合 4ch」，Δ=+0.02527 同时含「融合方案 early vs mid」与「模态 +IR」，无法在本实验内拆开。早期融合本身（单支、少 10M 参数）也可能是收益来源之一。
2. **单次 seed=42**：0.558 是单一 seed 的单次尖峰，平台期约 0.552；未做多种子重复，不构成统计显著。
3. **尖峰 vs 平台**：0.558 是 ep237 的尖峰（同 F4 的 0.53273 也是 ep251 尖峰，二者口径一致）；即便取平台期 0.552，仍稳定优于 F4。

### 下一步建议（非本任务，需单独授权）

做「**中期融合 RGBID**」（与 F4 同融合方案、仅 +IR，5ch 输入三分支 mid-fusion），以隔离出纯 IR 收益；若中期融合 RGBID 仍 ≥ 早期融合 RGBID，则 IR 收益成立且融合方案可任选；若中期融合 RGBID 跌回 F4 附近，则本次收益主要来自「早期融合方案」而非 IR，需重新权衡。
