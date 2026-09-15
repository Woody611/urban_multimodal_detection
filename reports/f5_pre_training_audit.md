# F5 训练前审计报告 — YOLO11l RGBD Mid Fusion @1280

> 阶段：**训练前**（配置审计 + 单变量 diff + 冒烟测试）。正式 300 epoch 尚未启动，等待确认。
> 基线：F4 = `runs/urban_multimodal_det_yolo11_rgbd_f4_1280/`，best mAP@0.5:0.95 = **0.53273**（ep251，尖峰；平台期 +0.006~0.011）。
> 审计依据：F4 落盘的 `runs/urban_multimodal_det_yolo11_rgbd_f4_1280/args.yaml`（权威来源）。

---

## 0. 实验定义

**F5 = F4 的唯一改动：模型规模 m → l（scale）。** 保持 `imgsz=1280` 不变，验证「容量」杠杆——
F4 在 1280 下 Recall 上来但 Precision 掉（bicycle −0.078 / ball −0.060），说明 m 容量见顶，l 有望保住 Recall、补回 Precision。

---

## 1. F4 / F5 配置 diff

F4 实际训练配置 = `configs/train_f4_1280.yaml`。F5 配置 [`configs/train_f5_l1280.yaml`](../configs/train_f5_l1280.yaml) 是它的逐行副本。

### 1.1 文件级 diff

除注释外，实质差异只有三行：

```diff
-experiment_name: "urban_multimodal_det_yolo11_rgbd_f4_1280"
+experiment_name: "urban_multimodal_det_yolo11_rgbd_f5_l1280"   # 仅输出目录，非训练变量
-pretrained: "yolo11m.pt"
+pretrained: "yolo11l.pt"                                        # ★ 唯一实验变量（模型规模 m→l）
-batch_size: 8
+batch_size: 4                                                  # 显存安全（见 §3），非实验变量
```

### 1.2 语义级 diff

| 项 | F4 | F5 | 性质 |
| :--- | :--- | :--- | :--- |
| **模型 scale** | **m**（30.34M） | **l**（37.62M） | ★ **唯一实验变量** |
| pretrained | yolo11m.pt | yolo11l.pt | 随 scale 联动（RGB 分支预训练） |
| batch_size | 8 | 4 | 显存安全，`accumulate` 补偿（见 §3.1） |
| experiment_name / save_dir | …_f4_1280 | …_f5_l1280 | 仅独立目录，非超参 |

**唯一训练变量 = 模型规模 m→l。** `batch_size 8→4` 由梯度累积补偿（effective batch 恒 64，lr0 不动），
`name/save_dir` 只服务独立目录。其余（imgsz=1280、epochs=300、patience=80、lr0=0.005、SGD、amp、增强、loss gain、数据划分、RGBD 管线、conf/iou/max_det）**全部与 F4 逐字一致**。

### 1.3 m→l 到底改了什么（关键，避免重蹈 F4 报告 §15 的估算错误）

| scales 项 | m | l | 是否变化 |
| :--- | :---: | :---: | :---: |
| depth | 0.50 | **1.00** | ✅ 变 |
| width | 1.00 | 1.00 | ❌ **不变** |
| max_channels | 512 | 512 | ❌ 不变 |

- **width 对 m 和 l 都是 1.00**——所以 l 的通道数、特征图空间尺寸、stem/下采样卷积（640×640 那几层）**与 m 完全一致**。
- depth 只影响 `C3k2` / `C2PSA` 的**内部 repeat**（parse_model 把 n 塞进 args 后重置 n=1），**顶层模块序号 0..44 不变**（45 个模块）。
- 因此：① `_transfer_rgb_pretrained` 的硬编码 index pairs 对 l 依然有效；② l 的算力/显存增幅是**约 1.26×，不是 4×**。

---

## 2. F5 实际训练配置

| 项 | F5 值 | 与 F4 |
| :--- | :--- | :---: |
| experiment_name | `urban_multimodal_det_yolo11_rgbd_f5_l1280` | 独立目录 |
| model yaml | `configs/yolo11l_midfusion_rgbd_concat_res.yaml` | **★ m→l** |
| pretrained | `yolo11l.pt`（云端自动下载，同 F4 下载 yolo11m.pt 机制） | 联动 |
| **imgsz** | **1280** | 同 |
| batch_size | **4** | 8→4（显存） |
| epochs / patience | 300 / 80 | 同 |
| learning_rate (lr0) | 5.0e-3 | 同 |
| optimizer / scheduler | SGD(momentum 0.937, wd 5e-4) / cos_lr | 同 |
| warmup_epochs | 3 | 同 |
| seed / deterministic | 42 / true | 同 |
| amp | true | 同 |
| use_simotm / channels | RGBD / 4 | 同 |
| val_ratio | 0.2 | 同 |
| resume | **false（绝不 resume F4）** | 同 |
| 增强 / loss gain / NMS / 数据划分 | 全部走默认，与 F4 逐字一致 | 同 |

启动命令：

```bash
python scripts/train.py \
    --model_config configs/yolo11l_midfusion_rgbd_concat_res.yaml \
    --train_config configs/train_f5_l1280.yaml
```

> `--dataset_config` 未指定，默认 `configs/dataset.yaml`；`val_ratio=0.2 > 0` 且 val 无标注，
> train.py 用 `_split_train_val_depth(dataset_cfg, 0.2, 42)` 确定性重建切分 —— 与 F4 同一路径 `depth_split_train`、同样 1600 train / 400 val。

---

## 3. GPU / batch / effective batch / 显存

### 3.1 梯度累积（框架内置，无需改代码）

[`ultralytics/engine/trainer.py:301`](../ultralytics/engine/trainer.py#L301)：
`accumulate = max(round(nbs / batch_size), 1)`，`nbs=64`：

| batch | accumulate | **effective batch** | weight_decay 补偿 | lr0 |
| :---: | :---: | :---: | :---: | :---: |
| 8（F4） | 8 | **64** | 是 | 0.005 |
| **4（F5）** | 16 | **64** | 是 | **0.005 不变** |

**结论：batch 8→4 后 effective batch 仍 64，lr0 不动，框架不改。** 唯一真实副作用是 BatchNorm
统计量按 micro-batch=4 而非 8 计算（梯度已由累积等价于 64）。已知影响很小：F4 逐类重跑 batch=4 与
训练 batch=8 的 mAP50-95 吻合到 0.0003。选 4 是为了给 30h 级云端任务留显存余量（见 §3.2）。

### 3.2 显存估算（V100 32GB）

**固定开销（与 batch 无关，fp32 + AMP）：**

| 项 | m（F4） | l（F5） |
| :--- | ---: | ---: |
| 参数 | 30.34M → 121 MB | 37.62M → 150 MB |
| 梯度 | 121 MB | 150 MB |
| SGD momentum 缓冲 | 121 MB | 150 MB |
| EMA（权重副本） | 121 MB | 150 MB |
| AMP fp16 前向副本 | ~60 MB | ~75 MB |
| **合计** | **≈ 0.55 GB** | **≈ 0.68 GB** |

**激活值（随 batch 线性）：**

- F4@batch=8 已全程跑通 300 ep **未 OOM**（云端实测，权威证据）。
- F5 激活值相对 F4 的放大 ≈ **1.26×**（GFLOPs 比，保守上界）。真实值更低：depth 只翻倍了
  C3k2/C2PSA 内部层，而占激活大头的 stem/下采样卷积（640×640、320×320）**通道数不变、层数不变**。

**估算（推断，非实测，须云端 `nvidia-smi` 确认）：**

| 方案 | 峰值显存估算 | 判定 |
| :--- | :--- | :--- |
| l@1280 batch=8 | ≈ 25~30 GB | 大概率可跑，但余量小（1.26× F4@batch8） |
| **l@1280 batch=4（已选）** | **≈ 13~17 GB** | **充裕**，且 effective batch 仍 64 |

> 选 batch=4 的原因：F4 报告 §15 曾按「4×」错误估算预言 batch=8 必 OOM。实测纠正后（1.26×）batch=8
> 其实可能可跑，但 30h 级任务 OOM 的代价是白烧 GPU 小时。batch=4 + accumulate=16 在不改 lr0 的前提下
> 把显存砍半，代价仅是单卡利用率略降（§3.3）。若你偏好与 F4 严格同 batch（彻底消除 BN micro-batch 差异），
> 把 `train_f5_l1280.yaml` 的 `batch_size` 改回 8 即可，风险已由 1.26× 实测值降到很低。

### 3.3 计算量与时间（实测 GFLOPs，thop）

| 分辨率 | m GFLOPs | l GFLOPs | 比 |
| :---: | ---: | ---: | :---: |
| 640 | 114.21 | 143.99 | 1.26× |
| 1024 | 292.37 | 368.6 | 1.26× |
| **1280** | **456.83** | **575.95** | **1.26×** |

- F4 实测稳态 ≈ 126 s/epoch → 300 ep ≈ 10.5 h。
- F5 = 1.26× F4 → ≈ **159 s/epoch @batch=8**；@batch=4 因累积步数翻倍、单卡利用率略降，≈ **14~15 h / 300 ep**。
- **纠正**：F4 报告 §15 曾估「~1.8k GFLOPs、约 30h、batch=8 必 OOM」——**该估算是错的**（见 §7 勘误），实际仅 1.26×（575.95 GFLOPs）。

---

## 4. 冒烟测试结果

脚本：[`scripts/f5_smoke_test.py`](../scripts/f5_smoke_test.py)。本机 CPU-only（torch 2.4.1+cpu），GPU 显存无法本地测量。

**结果：15 PASS / 0 FAIL**（`--batch 2`）

| # | 检查项 | 实测证据 |
| :---: | :--- | :--- |
| 1 | scale 由文件名解析为 `l` | `guess_model_scale='l'` |
| 2 | 类别数 nc=12 | `Detect.nc=12` |
| 3 | 参数为 l 量级 (>35M) | **params=37.62M**，顶层模块数=45 |
| 4 | 融合结构完整 | SilenceChannel×2 + ADD×3 齐全 |
| 5 | 顶层模块数=45（m 一致 → 重映射 index 有效） | `len(model.model)=45` |
| 6 | stride 正常 | `stride=32` |
| 7 | RGBD 数据读取正常 | batch 正常取出 |
| 8 | 4-channel 输入正常 | `img.shape[1]=4` |
| 9 | 1280 letterbox 正常 | `HxW=1280x1280` |
| 10 | depth 通道有实际信号 | std>1e-3 |
| 11 | forward 正常（P3/P4/P5） | 3 层输出 |
| 12 | 特征图 160/80/40 @1280 | 与 stride 8/16/32 对应 |
| 13 | loss 正常（有限且为正） | total 有限为正 |
| 14 | backward 正常（梯度有限） | 全部参数张量有梯度且有限 |
| 15 | **融合真的在训（depth 分支 stem 收到梯度）** | depth stem \|grad\|=1.06e+01 |

GFLOPs（thop 实测）：**@640 = 143.99**，**@1280 = 575.95**。

### 4.1 与 F4 冒烟测试的差异（如实说明）

F4 冒烟测试还含一项**全量 validator 跑 398 图**的检查（随机权重 mAP=0.000 只证流程通）。F5 冒烟测试
**未含该 val 项**。理由：val 管线只依赖数据划分（与 F4 同一 `depth_split_train`）与 Detect 头（顶层结构
45 模块不变），与模型 scale 无关，F4 已证过一遍。若需严格复刻，可给 F5 冒烟测试补上同款 val 检查——但
对本实验（唯一变量 m→l）不构成新风险。

---

## 5. 待办：正式训练后需要补的报告项

```text
 5. 最终 best mAP50-95        → 对照 F4 的 0.53273
 6. best epoch                → 对照 F4 的 251
 7. mAP50 / AP75 / P / R
 8. person / sign / bicycle / ball 四类逐项（AP50-95 / AP50 / AP75 / P / R）
 9. 训练稳定性：best 是否孤峰、平台期窗口均值、末 30 ep、loss 是否发散
10. F4 vs F5 对比 + 是否建议继续容量方向 + 下一步建议
```

判据（预登记，避免事后调整标准；对照 [f4-1280-result](../memory/f4-1280-result.md) 的「≥0.540」）：

| 档位 | F5 best mAP50-95 | 结论 |
| :--- | :---: | :--- |
| 强成功 | ≥ 0.545 | 容量杠杆显著，可考虑 x / 更深 |
| 明显有效 | 0.538 ~ 0.545 | 容量是有效杠杆，继续 |
| 小幅有效 | 0.534 ~ 0.538 | 收益有限，需权衡成本 |
| 基本无效 | < 0.534 | 容量边际收益到头，转向其他方向 |

> 注：F4 best 0.53273 是尖峰，平台期约 0.523。故 F5 判据以「平台期 +0.011（≈0.534）」为有效下界，
> 而非简单比 0.53273。

第一判断标准：**训练内置 val 的 mAP@0.5:0.95（results.csv）**，不用 evaluate.py。

---

## 6. 合规确认

| 约束 | 状态 |
| :--- | :--- |
| 不 resume F4 / 不加载 F4 best.pt / last.pt | ✅ `resume: false`，`model_path = --model_config` 的 yaml |
| 独立实验目录 | ✅ `runs/urban_multimodal_det_yolo11_rgbd_f5_l1280/`，未覆盖 F4 |
| 唯一变量 = 模型规模 m→l | ✅ 语义 diff 证实；batch 8→4 由 accumulate 补偿、lr0 不动 |
| RGB-Depth 双分支中期融合整体架构保持不变 | ✅ 模型 yaml 顶层结构逐字同 m（仅文件名 scale） |
| 不改 lr0 / epochs / close_mosaic / 增强 / 数据 / 标签 / depth 预处理 / 划分 / conf / NMS | ✅ 全部未改 |
| 不修改训练框架 / 模型代码 | ✅ 未改 train.py / 任何 ultralytics 源码；仅新增独立 yaml + 冒烟脚本 |
| 推理置信阈值调参不当作 mAP 提升 | ✅ conf=0.001 / iou=0.7 / max_det=300 未动 |
| 未启动正式训练 | ✅ 等待确认 |

---

## 7. 勘误：F4 报告 §15 关于 l 成本的错误估算

F4 实验报告 §15 曾写「l@1280 计算量约为 m@1280 的 **4×**（~1.8k GFLOPs），batch=8 大概率 OOM，需 batch=2，单 run 约 30h」。

**此估算错误。** 实测（thop，本报告 §3.3）：

- l@1280 = **575.95 GFLOPs** = m 的 **1.26×**（不是 4×，也不是 ~1.8k）。
- l@1024 = 368.6 GFLOPs；l@640 = 143.99 GFLOPs。
- params 37.62M = m 的 1.24×。

**根因**：误以为 m→l 会翻倍 width。实际 scales 表里 `m=[0.50,1.00,512]` 与 `l=[1.00,1.00,512]` 的
**width 都是 1.00**，只有 depth 0.50→1.00 变化，而 depth 只加厚 C3k2/C2PSA 内部层、不动通道宽度。
因此 l 的真实代价是 +26% 算力、+24% 参数，训练时间 ≈ 14~15h（不是 30h），batch=4 已足够稳妥
（batch=8 也大概率可跑）。本报告 §3.2/§3.3 已按正确值重估。
