# 面向城市场景的视觉多模态目标检测 —— 复赛提交说明

> **提交方案**：`D′ + IR-specific gamma/noise augmentation`
> **线上得分**：**48.951**（同一测试集上，D′ 基线为 48.712）
> 本文档面向评审/复现者，包含方案说明、权重下载、运行配置、文件清单、运行命令与注意事项。

---

## 1. 项目简介

### 1.1 任务

面向城市场景的多模态目标检测。输入为**空间对齐**的三种模态图像，输出目标的**类别、边界框、置信度**。

### 1.2 数据（三模态）

| 模态 | 目录 | 说明 |
|---|---|---|
| 可见光 RGB | `visible/` | 彩色图像（BGR 3 通道） |
| 红外 IR | `infrared/` | 灰度图，**8-bit**（3 通道同样值复制，`cv2.IMREAD_GRAYSCALE` 读取） |
| 深度 Depth | `depth/` | 8-bit（16-bit 深度按 `d/19999*255` 映射，`<300mm` 置 0） |

原始数据布局（**需自行准备，未随包提供**）：

```
data/raw/
├── train/{visible,infrared,depth}/   各 2000 张
├── train/labels/                     2000 个 .txt
└── test /{visible,infrared,depth}/   各 1000 张（当前复赛测试集，无标注）
```

标注格式为 YOLO：`<class_id> <norm_cx> <norm_cy> <norm_w> <norm_h>`（坐标归一化到 [0,1]）。

训练用的切分由 `val_ratio: 0.2` 自动生成到 `data/processed/rgbid_split_train/`
（**1600 train / 400 val**，三模态同名对齐）：`images/{train,val}/{visible,infrared,depth}` + `labels/`。

### 1.3 类别（12 类，class id 即顺序）

| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 名称 | person | boat | animal | seat | sign | bicycle | car | ball | light | garbage_can | uav | tricycle |

---

## 2. 模型权重下载链接

> ### ⬇️ **`<待填写 —— 模型权重下载地址>`**
>
> 下载后放到 `rematch/weights/`，见 [`weights/README.md`](weights/README.md)。

| 文件 | SHA-256 | 用途 |
|---|---|---|
| `best.pt` | `5b99b6e5b20dca9a73e1183c5fda1cba2aba2d48d26366bff0704f530e911478` | **本次提交所用权重** |
| `last.pt` | `d2029dac8545a71e99e00d23dc2b4e32d733949b9a24c1c99e9f66720be2a455` | 第 300 epoch（对照） |

### 2.1 复现训练还需两项**外部资源**（不随包提供）

| 资源 | 放置位置 | 获取方式 |
|---|---|---|
| **数据集** | `data/raw/{train,test}/{visible,infrared,depth}` | 赛题方提供；布局见 §1.2 |
| **预训练骨干 `yolo11m.pt`** | 项目根目录（与 `configs/` 同级） | Ultralytics 官方发布：`https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt`（本实验所用副本 SHA-256 = `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`，40,684,120 字节） |

> `pretrained: "yolo11m.pt"` 写在两份训练配置里。缺该文件时 Ultralytics 会尝试联网下载；
> 离线环境请预先放到项目根目录。**仅做推理/评测（§6.2 / §6.3）不需要它**，只需要 §2 的 `best.pt`。

---

## 3. 算法方案

### 3.1 输入表示：单塔 5 通道早期融合

三种模态在**数据加载阶段**就合并成一张 **5 通道** uint8 图像，此后全程只有一个骨干网络：

```
[B, G, R, IR, D]        ← load_and_preprocess_image（RGBID 分支）输出的原始通道序
        │
        ├─ 训练画布: rect resize 长边→1280 → mosaic/几何增强 → /255
        │
[R, G, B, IR, D]        ← Format 张量序（BGR→RGB 翻转，IR/D 保持原位）
```

> ⚠ 数组序是 `[B,G,R,IR,D]`，**进网张量序是 `[R,G,B,IR,D]`**（`Format._format_img` 做 BGR→RGB）。
> IR 恒在 **idx 3**、Depth 恒在 **idx 4**，两者不参与通道翻转。

### 3.2 网络结构：Separate-Stem 三路轻量 stem + YOLO11m

不做单层 5→64 卷积混合，而是**先按模态切分**，各走一个轻量 stem 升维后 Concat：

```
输入 5ch
 ├─ SilenceChannel[0:3] → Conv(3→48, 3, s2)  → RGB   [B,48,H/2,W/2]
 ├─ SilenceChannel[3:4] → Conv(1→ 8, 3, s2)  → IR    [B, 8,H/2,W/2]
 └─ SilenceChannel[4:5] → Conv(1→ 8, 3, s2)  → Depth [B, 8,H/2,W/2]
                                     Concat ↓
                            [B,64,H/2,W/2] → 其后为**未改动的** YOLO11m 主干/颈/头
```

- 参数量 **20,061,972**，GFLOPs@1280 **272.68**
- 设计动机：三条模态各有独立的第一层卷积核，避免共享权重被某一模态主导
  （实测权重分配 RGB 81.4% / IR 11.7% / Depth 6.9%）
- **唯一变量纪律**：与基线的差异**只有** fusion 结构本身（+ 实验名）

### 3.3 IR 预处理：CLAHE

红外在交给网络之前统一做对比度处理，二者**互斥（非叠加）**：

| 取值 | 行为 |
|---|---|
| `percentile`（默认） | 1%/99% 分位数线性拉伸 |
| **`clahe`（本方案）** | `CLAHE(clipLimit=2.0, tileGridSize=(8,8))` |
| `raw` | 不处理 |

**位置**：在 raw IR 上执行，早于 geometric augmentation / normalization / 通道合并。

### 3.4 ★ 本次实验的唯一自变量：IR-specific gamma/noise 增强

这是**相对 D′ 基线唯一的改动**（4 个新配置键，其余全部逐字不变）。

**作用对象**：CLAHE **之后**、通道合并**之前**的单通道 IR —— 即**模型实际看到的 IR 表示**。

```
raw IR → CLAHE(uint8) → 【gamma → noise】 → merge 5ch → resize → /255 → SepStem
```

**数值定义（锁死）**：

| 键 | 值 | 定义 |
|---|---|---|
| `ir_gamma` | `[0.75, 1.35]` | 命中时在该区间**均匀采样** γ；`x=im/255∈[0,1]` → `x**γ` → clip → ×255 → uint8 |
| `ir_gamma_probability` | `0.35` | 命中 gamma 的概率（**每次图像加载**） |
| `ir_noise_std` | `0.015` | 高斯噪声 σ，单位是**归一化 [0,1] 域**（= 3.825 灰度级） |
| `ir_noise_probability` | `0.20` | 命中噪声的概率 |

**参数依据（来自本项目数据实测，非照搬）**：

- `γ∈[0.75,1.35]` 诱导的曝光伸缩跨度 **1.84×**，**小于**本数据图与图之间天然曝光跨度 **2.69×** ⇒ 落在数据自身流形内；
  超出该范围的 γ≤0.5 / ≥2.0 会诱导 4.3× 跨度，**已排除**。
- `σ=0.015`（3.825 灰度级）相对 CLAHE 输出的 σ≈60 灰度级，SNR≈15.7，属温和传感器噪声。
- 概率语义经实测：Mosaic 的 buffer + `self.ims` 缓存使**每训练样本恰好 1.0 次**预处理调用
  ⇒ 概率**就是**「被增强图像的比例」，**不存在 Mosaic 放大**。

**train-only 保证（三重）**：
1. val dataset 的 `augment=False` 直接短路返回；
2. 推理路径 `loaders.py::LoadImagesAndVideos` **根本不调用**该函数（走自己的内联 IR 分支）；
3. 四个键在 `default.yaml` 中默认「恒等 + 关闭」⇒ 未显式配置的实验**不消耗任何 RNG**，逐位不变。

### 3.5 训练配方

| 项 | 值 |
|---|---|
| 骨干 | YOLO11m（预训练 `yolo11m.pt` 迁移到 3-stem 布局） |
| imgsz / batch | 1280 × 1280 / 8 |
| 优化器 | SGD，lr0 **0.005**，momentum 0.937，weight_decay 5e-4 |
| 调度 | CosineAnnealingLR，warmup 3 epoch |
| epochs / patience | **300 / 0**（禁用早停） |
| AMP | 开启 |
| seed | **42** |
| 几何增强 | mosaic 1.0、close_mosaic 10、translate 0.1、scale 0.5、fliplr 0.5 |
| 其他光度增强 | **无**（5ch 下 Albumentations p=0、RandomHSV 自动跳过） |

---

## 4. 运行配置（环境）

| | 训练环境（实测） | 推理/评测环境（实测，CPU 亦可） |
|---|---|---|
| OS | Linux (AutoDL) | Windows 11 / Linux |
| Python | 3.10.8 | 3.8.15 |
| torch | 2.1.2+cu121 | 2.4.1+cpu |
| CUDA / GPU | 12.1 / NVIDIA A100-PCIE-40GB | 无（纯 CPU 可跑通全流程） |
| opencv-python | — | 5.0.0 |
| fork 版本 | Ultralytics **8.3.75**（本地 fork） | 同 |

安装：

```bash
pip install -r rematch/requirements.txt
```

> ⚠ **不要 `pip install ultralytics`** —— 本方案用的是随包提供的**本地 fork**（`./ultralytics/`）。
> 详见 §8 注意事项 1。

---

## 5. 项目文件说明

```
rematch/
├── README.md                    ← 本文档
├── requirements.txt             ← 环境依赖（已按实际 import 校正）
│
├── ultralytics/                 ← ★ 本地 fork（算法核心，Ultralytics 8.3.75 改写版）
│   ├── cfg/default.yaml         自定义键：ir_encoding / ir_gamma* / ir_noise* / modality_dropout / OASA ...
│   ├── data/
│   │   ├── base.py              ★ 数据集与 IR 预处理：apply_ir_encoding / apply_ir_augmentation / RGBID 分支
│   │   ├── loaders.py           ★ 推理侧多模态加载（含**独立的** IR-CLAHE 内联实现）
│   │   └── augment.py           几何/光度增强、5ch letterbox、Format（张量通道序在此确定）
│   ├── models/yolo/detect/      ★ 预训练权重迁移到 3-stem 布局（_remap_separate_stem）
│   ├── nn/                      网络模块（含 SilenceChannel）
│   └── engine/trainer.py        训练主循环
│
├── scripts/                     ← 入口脚本（**仅保留本实验可复跑所需的 8 个**，导入闭包已核）
│   ├── train.py                 ★ 训练入口（含 RGBID 缺 ir_encoding 的硬失败守卫）
│   ├── preflight_check_train_config.py  开训前自检（import scripts.train）
│   ├── predict_rect.py          ★ 提交/评测推理（rect letterbox，复刻 model.val 的预处理）
│   ├── predict.py               自包含推理循环（predict_rect.py import 其工具函数）
│   ├── official_eval.py         ★ 官方口径 mAP 评测器（纯 TXT 打分）
│   ├── official_map.py          官方 vs fork 双口径对照（import official_eval）
│   ├── check_submission.py      提交前 10 点自检 + 打包 zip（import official_eval）
│   └── make_candidate_submission.py  候选提交包生成（不自动上传；校验逻辑内联）
│
├── configs/                     ← 配置（**仅保留本实验所需的 4 个**）
│   ├── yolo11m_sepstem.yaml                  ★ 模型配置（3-stem 48/8/8 + Concat），两臂共用
│   ├── train_rgbid_sepstem_clahe_iraug.yaml  ★ 本方案训练配置（唯一自变量所在）
│   ├── train_rgbid_sepstem_clahe.yaml        D′ 基线训练配置（**对照臂**，复现对比需要）
│   └── dataset.yaml                          数据集根路径与类别（train.py 的默认 --dataset_config）
│
├── diagnostic/ir_aug_audit/     ← 本实验的方法审计与改动留痕
│   ├── IR_AUG_P0_REPORT.md      插入点 / 数值域 / 参数依据的完整审计
│   ├── FINAL_LAUNCH_GATE.md     15 项开训前硬门的结果
│   ├── verify_run_args.py       ★ 训练启动后 30 秒核对实际生效超参（可直接跑 run_record/args.yaml）
│   └── diff_*.patch             本次代码改动的完整 diff（base.py / train.py）
│
├── weights/                     ← 放入下载的 best.pt / last.pt（见 §2）
│   └── README.md
│
├── run_record/                  ← ★ 本次训练的实际运行记录（可自证超参）
│   ├── args.yaml               训练时**实际解析**出的配置（非配置文件原文）
│   ├── results.csv             300 epoch 逐 epoch 指标
│   └── results.png             训练曲线
│
└── submission/                  ← ★ 本次提交结果
    ├── submission.zip           ★ 提交包（1000 个 results/<stem>.txt）
    ├── _submission_record.json  提交记录（全部 SHA、命令、超参）
    └── RESULTS.md               本地双口径评测 + 线上结果
```

> `run_record/args.yaml` 是**证据文件**：它由训练时真正传给框架的配置落盘而来，
> 可用来验证「配置注释/文件名所声称的」与「实际生效的」是否一致：
> ```bash
> cd rematch
> python diagnostic/ir_aug_audit/verify_run_args.py run_record/args.yaml
> # 期望末尾: IR_AUG_ACTIVE = YES
> ```

---

## 6. 运行 / 测试命令

> 所有命令的工作目录均为 **`rematch/`**，且需先按 §1.2 准备好 `data/`。

### 6.1 训练（GPU，约 12 小时 / A100-40GB）

```bash
# 0) 开训前自检（与正式训练同一条配置解析路径）
python scripts/preflight_check_train_config.py \
    --model_config configs/yolo11m_sepstem.yaml \
    --train_config configs/train_rgbid_sepstem_clahe_iraug.yaml \
    --expect-ir-encoding clahe
# 期望: RESULT: ALL CHECKS PASSED (exit 0)

# 1) 训练
python scripts/train.py \
    --model_config configs/yolo11m_sepstem.yaml \
    --train_config configs/train_rgbid_sepstem_clahe_iraug.yaml
```

**开训后立刻**核对实际生效的超参（防止配置静默回落，见 §8-4）：

```bash
python diagnostic/ir_aug_audit/verify_run_args.py \
    runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_iraug/args.yaml
# 期望: IR_AUG_ACTIVE = YES
```

### 6.2 验证集评测（CPU 即可，400 张约 7 分钟）

```bash
python scripts/predict_rect.py \
    --weights weights/best.pt \
    --train_config configs/train_rgbid_sepstem_clahe_iraug.yaml \
    --source data/processed/rgbid_split_train/images/val/visible \
    --mode rect --conf 0.001 --iou 0.7 --imgsz 1280 --max_det 300 --max_boxes 100 \
    --output diagnostic/eval_iraug

# 官方口径（复刻线上指标）
python -X utf8 scripts/official_eval.py \
    --results diagnostic/eval_iraug/results \
    --images data/processed/rgbid_split_train/images/val/visible \
    --labels data/processed/rgbid_split_train/labels/val/visible \
    --metric A --json diagnostic/eval_iraug/_official.json
```

### 6.3 生成提交包（CPU，1000 张约 17 分钟）

```bash
# 推理
python scripts/predict_rect.py \
    --weights weights/best.pt \
    --train_config configs/train_rgbid_sepstem_clahe_iraug.yaml \
    --source data/raw/test/visible \
    --mode rect --conf 0.001 --iou 0.7 --imgsz 1280 --max_det 300 --max_boxes 100 \
    --output submissions/my_candidate

# 自检 + 打包
python scripts/check_submission.py \
    --results submissions/my_candidate/results \
    --zip submissions/my_candidate/submission.zip --expect 1000

python scripts/make_candidate_submission.py \
    --results submissions/my_candidate/results \
    --source data/raw/test/visible --expect 1000 \
    --out submissions/my_candidate --accept-degenerate-wh
```

`--accept-degenerate-wh` 见 §8-8。

---

## 7. 结果

### 7.1 本地 400 张验证集（与基线同一条评测链）

| 模型 | 官方口径 (metric A) | fork 口径 | mAP50 | mAP75 |
|---|---:|---:|---:|---:|
| **D′**（基线，控制组） | 0.51528 | 0.56609 | 0.77826 | 0.51096 |
| **D′ + IR aug**（本方案） | 0.51428 | 0.56310 | 0.78788 | 0.52786 |
| Δ | −0.00100 | −0.00299 | +0.00963 | +0.01690 |

> 控制组逐位复现了基线已知值（0.51528 / 0.56609）⇒ 评测链可信。

### 7.2 线上（1000 张测试集，同一评测集）

| 模型 | 线上分 |
|---|---:|
| D′（基线） | 48.712 |
| **D′ + IR aug（本方案）** | **48.951** |
| Δ | **+0.239** |

### 7.3 ⚠ 结果解读（请务必连同 §8-9 一起看）

- 线上 **+0.239 是真实差异**（同 seed，线上对固定模型的评测是确定性的，非评测噪声）。
- **但不能归因于该增强本身**：加增强会改变随机数流、扰动整条训练轨迹，这与「换一个 seed」
  造成的扰动**量级相同**；而本项目**从未测过跑间/种子方差**，因此该差异**无法与单次轨迹扰动区分**。
- 同时，两组本地口径在本例中**都判错了符号**（都判负、线上为正）⇒ 本地口径在
  ±0.5 线上分的粒度上不可靠。**这是仪器局限，不是本方案的论据。**

---

## 8. 注意事项（重要）

1. **不要 `pip install ultralytics`**
   本方案使用随包的本地 fork（`./ultralytics/`，8.3.75）。`scripts/*.py` 会把项目根插入
   `sys.path[0]` 以确保导入本地 fork；但若同时装了 PyPI 版，仍可能造成版本冲突或误导入。

2. **`ultralytics/data/` 在原仓库中不被 git 追踪**
   原仓库 `.gitignore` 有一条无锚定的 `data/` 规则，会连带吞掉 `ultralytics/data/`。
   因此 **fork 的改动无法通过 git 传递**，必须**按文件完整拷贝**（本包即如此）。
   同理：改动前后请用 SHA-256 而非 `git diff` 校验。

3. **IR 预处理在代码里有 3 处副本**
   训练侧 `ultralytics/data/base.py::apply_ir_encoding`、
   推理侧 `ultralytics/data/loaders.py`（**内联实现，不调用上面的函数**）、
   以及 `dataset.py` 中一处未被使用的旧实现。
   **改一处不会自动生效到另外两处**；验证必须同时覆盖训练与 `LoadImagesAndVideos`。

4. **`ir_encoding` 必须显式声明**
   RGBID 配置若缺该键会**静默回落 `percentile`**（历史上曾因此白跑 11 小时）。
   现已加硬守卫：`scripts/train.py` 缺键即失败，`preflight_check_train_config.py` 也会拦。
   **判据是看 `args.yaml` 里的实际值，不是看配置文件写了什么。**

5. **模型 scale 由文件名正则决定**
   `guess_model_scale` 用正则 `yolo[v]?\d+([nslmx])` 解析**文件名**。
   模型 yaml 文件名必须含 `yolo11m`，否则**静默回退 nano（2.59M 参数）**，训练照跑但模型小一个量级。

6. **`albumentations` 是否安装会影响 RNG 流**
   本方案 5 通道下 `Albumentations(p=0)` 完全不生效，因此**非必需**。
   但若安装，其 `__call__` 每样本会多消耗一次 `random` 抽样（功能不变，RNG 流不同）
   ⇒ **跨机器的逐位复现会失效**。要严格复现，请保持与训练机一致的安装状态。

7. **提交硬性约束**
   官方规则：**禁用多模型集成**；**每图最多 100 个框**（推理已用 `--max_boxes 100` 截断）。
   `check_submission.py` 会校验这两点及其他 8 项。

8. **退化框（w/h=0）是既有行为，不要"修复"**
   提交 TXT 中存在少量 `w=0 或 h=0` 的框，来源是冻结推理链（`predict.py::_format_lines` → `scale_boxes`）
   在图像边界处的既有行为，历史提交同样存在且已线上评分过。打包时需加 `--accept-degenerate-wh`。

9. **不要用本地 Δ 的符号下结论**
   本例中两组本地口径都判负、线上为正。**任何本地数值都不得直接当作线上分数**，
   也不要在 ±0.5 线上分的粒度上依赖本地口径的符号。要判断某改动是否有效，
   必须做**重复实验（另一 seed）+ 对照臂方差**。

10. **路径配置需按本机调整**
    `configs/dataset.yaml` 中的 `path` 与训练时生成的 `dataset.yaml` 含**绝对路径**，
    换机器时需同步修改。

11. **本包只收录了「本实验可复跑」所需的文件，不是整个开发仓库**
    原开发仓库有 92 个配置、50 个脚本（含大量其他实验与诊断工具）。本包按**导入闭包**裁剪为
    **4 个配置 + 8 个脚本**，足以完整复跑 D′ 基线与 IR-Aug 两臂、评测、并打包提交。
    ⇒ 因此 `scripts/predict_rect.py` / `predict.py` / `official_eval.py` 里的**默认参数**
    （`DEFAULT_WEIGHTS`、默认 `--train_config`、默认 `--images/--labels`）指向**未收录**的
    其他实验产物，在本包中不适用 —— 运行 §6 的命令时**必须显式传**
    `--weights` / `--train_config` / `--source` / `--images` / `--labels`
    （§6 给出的命令均已显式传入，可直接复制执行）。

12. **包内**不含**数据集与预训练骨干**（见 §2.1），这是有意为之（体积 + 版权）。
    除此之外，`rematch/` 内每个文件都有明确用途，**没有占位或冗余文件**。

---

## 8.1 本包完整性自检（交付前实测，非声称）

只用 `rematch/` 内的文件、CPU、不训练，跑通了完整链路：

| 检查 | 结果 |
|---|---|
| `import ultralytics` 解析到本包 fork | ✅ 8.3.75，来自 `rematch/ultralytics/` |
| 两臂 config 经真实解析路径展开 | ✅ 对照臂 `ir_gamma=[1.0,1.0] p=0`；实验臂 `[0.75,1.35] p=0.35 / σ=0.015 p=0.20` |
| 数据集可加载（三模态 5ch） | ✅ 1599 train 样本 |
| 样本张量形状/dtype | ✅ `(5, 1280, 1280)` `torch.uint8` |
| **IR augmentation 实际触发** | ✅ `calls=1`，gamma/noise 计数按概率递增 |
| 模型可从 yaml 构建 + 前向 | ✅ 参数量 **20,061,972**（与实跑 run 完全一致） |
| 8 个入口脚本可导入 | ✅ 8/8 |
| 开训前自检 | ✅ `RESULT: ALL CHECKS PASSED` (exit 0) |
| 交付权重与运行配置自洽 | ✅ `verify_run_args.py run_record/args.yaml` → `IR_AUG_ACTIVE = YES` |
| 提交包未被破坏 | ✅ `submission.zip` sha256 `c704d545…`，1000 条目，CRC 无错 |

⇒ **配合 §2.1 的两项外部资源（数据 + `yolo11m.pt`），本包可完整复跑本实验。**

---

## 9. 引用与许可

- 本地 `ultralytics/` 为 Ultralytics YOLO 的改写分支，**遵循其原始 AGPL-3.0 许可**
  （源码文件头保留 `Ultralytics 🚀 AGPL-3.0 License` 声明）。使用/分发请遵守该许可。
- 本项目其余代码为参赛自研。
