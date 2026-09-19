# Experiment D — IR CLAHE 单变量审计与实施报告

**日期**：2026-09-18
**实验定义**：`RGBID@1280 baseline` vs `RGBID@1280 + IR CLAHE`（**唯一变量 = IR 对比度处理**）
**状态**：**代码已实施 + 冒烟测试全部 PASS**；**训练未启动**（等 GPU 空闲）。
**正式保底全程冻结**。

---

## 0. 结论速览

| 项 | 结果 |
|---|---|
| 唯一变量 | **IR 对比度处理：`percentile` → `clahe`**（互斥替换，非叠加）✅ |
| RGB 是否变化 | ❌ **不变**（40/40 逐位一致） |
| Depth 是否变化 | ❌ **不变**（40/40 逐位一致） |
| IR 是否变化 | ✅ **变化**（40/40 全部不同） |
| 5ch shape | ✅ `(H,W,5)` → Format → `(5,H,W)`，**不变** |
| dtype / range | ✅ 均 uint8 `[0,255]`，与 percentile **完全相同**；后续 `.float()/255` 未改 |
| Albumentations 对 5ch 的隐式风险 | ✅ **无** —— 见 §10 |
| baseline 是否仍逐位可复现 | ✅ 默认 `percentile`，`else` 分支是**原代码逐字** |
| 训练配置差异 | ✅ 与 baseline **恰好 2 处**：`ir_encoding`（变量）+ `name`（目录） |
| 冒烟测试 | ✅ **全部 PASS** |
| run directory | `runs/urban_multimodal_det_yolo11_rgbid_ir_clahe/`（**新目录**） |

---

## 1. 当前 percentile IR preprocessing 的具体代码位置

**文件**：`ultralytics/data/base.py` · **函数**：`BaseDataset.load_and_preprocess_image()` · **分支**：`elif use_simotm == 'RGBID'`（L313 起）

**改动前（原代码，实读）**：

```python
# 红外统一到 uint8 单通道 + 1%/99% 分位数对比度拉伸（复用 Infrared 单模态逻辑）
if im_infrared.dtype == np.uint16:
    im_infrared = (im_infrared.astype(np.float32) * (255.0 / 65535.0)).astype(np.uint8)
elif im_infrared.dtype != np.uint8:
    im_infrared = im_infrared.astype(np.float32)
    lo, hi = im_infrared.min(), im_infrared.max()
    im_infrared = ((im_infrared - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(im_infrared)).astype(np.uint8)
lo, hi = np.percentile(im_infrared, (1.0, 99.0))                                    # ← L328
if hi - lo > 1:                                                                     # ← L329
    im_infrared = np.clip((im_infrared.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)  # ← L330
```

**执行位置**：在 `IMREAD_GRAYSCALE` 读取 + dtype 统一 **之后**，在 `_resize_images_3` / `_merge_channels_rgbid`（`cv2.merge((b,g,r,ir,d))`）**之前**。

---

## 2. CLAHE 应该插入的具体位置

**同一处**，作为 `percentile` 的**互斥替代**（不是叠加）：

```python
# 红外对比度处理（可切换；默认 percentile = 既有行为，保持 baseline 逐位可复现）
#   percentile: 1%/99% 分位数线性拉伸（原实现）
#   clahe     : CLAHE(clipLimit=2.0, tileGridSize=(8,8)) —— 与 percentile **互斥**，非叠加
# 此处 im_infrared 已统一为 uint8 单通道；CLAHE 的 in/out dtype 与 range 均与 percentile 相同。
# 该步骤在 _merge_channels_rgbid **之前**，因此只作用于 IR，RGB/Depth 不可能被影响。
ir_encoding = getattr(getattr(self, "hyp", None), "ir_encoding", "percentile") or "percentile"
if ir_encoding == "clahe":
    im_infrared = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(im_infrared)
else:
    lo, hi = np.percentile(im_infrared, (1.0, 99.0))
    if hi - lo > 1:
        im_infrared = np.clip((im_infrared.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
```

**关键**：CLAHE 作用对象是 **单通道 `im_infrared`**，此时 **5ch 数组还不存在** ——
所以 §4 禁止的「对 5-channel array 做 CLAHE」在这个位置**在结构上不可能发生**。

---

## 3. 准备修改哪些文件

| # | 文件 | 性质 | SHA256 前 24 | git 跟踪 |
|---|---|---|---|---|
| 1 | `ultralytics/data/base.py` | RGBID 分支的 IR 处理改为二选一 | `8bcf834930155bd58a99ccfc` | ❌ **UNTRACKED**（`.gitignore` 的 `data/`） |
| 2 | `ultralytics/cfg/default.yaml` | 新增 `ir_encoding: percentile` 默认键 | `6f91a2770b4e3cc7b393d98d` | ✅ TRACKED |
| 3 | `scripts/train.py` | 转发 `ir_encoding` | `80f6cdec1e52d41258ca7726` | ✅ TRACKED |
| 4 | `configs/train_rgbid_ir_clahe.yaml` | **新增**训练配置 | `2106385634b01a057357cc5c` | ❌ UNTRACKED |

**未改动**：`configs/yolo11m_earlyfusion.yaml`（模型）、`configs/train_rgbird_ir_quicktest.yaml`（baseline）、
任何 checkpoint、任何数据文件、`augment.py`、`dataset.py`、loss、augmentation、optimizer。

---

## 4. 每个文件具体修改什么

### 4.1 `ultralytics/data/base.py`（+8/−0，把原有 3 行包进 `else`）

`else` 分支内的三行是**原代码逐字**，仅缩进一级。**baseline 路径的数值输出与改动前完全一致**（已实测，见 §7）。

### 4.2 `ultralytics/cfg/default.yaml`（+4 行，纯增量）

```yaml
# IR 对比度处理方式（RGBID 专用；默认 percentile = 既有行为 -> 对既有实验零影响）
#   percentile: 1%/99% 分位数线性拉伸（原实现）
#   clahe     : CLAHE(clipLimit=2.0, tileGridSize=(8,8))，与 percentile **互斥**（不是叠加）
ir_encoding: percentile
```

### 4.3 `scripts/train.py`（+5 行，纯增量）

```python
# IR 对比度处理方式（可选）：仅在配置中存在时转发；未配置时走 default.yaml 的
# percentile 默认值 -> 既有实验行为不变。
if "ir_encoding" in train_cfg:
    kwargs["ir_encoding"] = str(train_cfg["ir_encoding"])
```

### 4.4 `configs/train_rgbid_ir_clahe.yaml`（新增）

`configs/train_rgbird_ir_quicktest.yaml` 的副本 + `experiment_name` 改名 + `ir_encoding: clahe`。

**程序化核验的 kwargs 差异 = 恰好 2 处**：

```
ir_encoding     <缺>                                                clahe
name            urban_multimodal_det_yolo11_rgbird_ir_quicktest     urban_multimodal_det_yolo11_rgbid_ir_clahe
共 2 处差异
```

其余 `imgsz=1280 / batch=8 / epochs=300 / optimizer=SGD / lr0 / lrf / warmup / loss / close_mosaic / 数据 / 增强 / seed / workers / AMP / patience` **逐字相同**。

---

## 5–9. 隔离性验证（逐位实测，n=40 张 val 样本）

| # | 检查 | 结果 |
|---|---|---|
| **5** | CLAHE 是否只作用于 IR | ✅ **是** —— 调用点在 `_merge_channels_rgbid` 之前，对象是单通道 `im_infrared` |
| **6** | RGB 是否完全不变 | ✅ **40/40 逐位一致** |
| **7** | Depth 是否完全不变 | ✅ **40/40 逐位一致** |
| **8** | 5ch shape 是否保持 | ✅ `(360,640,5)` 不变；`Format` 后 `(5,360,640)` 不变 |
| **9** | dtype / range 是否安全 | ✅ CLAHE 与 percentile 的 in/out **都是 uint8 [0,255]**；实测两路径 `min/max = 0/255`；后续 `.float()/255`（`train.py:159`）**未改** |

**IR 通道的分布变化**（应变化，量级可比）：

| | baseline (percentile) | D (clahe) |
|---|---:|---:|
| 均值 | 87.41 | 85.19 |
| 标准差 | 67.18 | 61.34 |
| 唯一值数 | 221.1 | **253.7** |

→ CLAHE 使**全局对比度略降**（std −5.8）但**局部灰度层次更多**（唯一值 +33）—— 符合 CLAHE 的预期行为。

**Format 之后**（训练/推理共用同一实现）：

```
baseline -> (5, 360, 640) uint8   ch3 唯一值=199
D        -> (5, 360, 640) uint8   ch3 唯一值=254
[PASS] Format 后 RGB(前3) 仍完全一致
[PASS] Format 后 Depth(第5) 仍完全一致
[PASS] Format 后 IR(第4) 发生变化
```

**end-to-end DataLoader（含 collate）**：

```
ir_encoding=percentile  batch img shape=(4,5,640,640) dtype=torch.uint8 min/max=0/255  5ch=OK
ir_encoding=clahe       batch img shape=(4,5,640,640) dtype=torch.uint8 min/max=0/255  5ch=OK
```

---

## 10. 是否存在 Albumentations 对 5ch 数据的隐式处理风险？

**✅ 无。已逐条实读确认：**

| 检查点 | 实读结果 |
|---|---|
| 5ch 路径为什么关闭 CLAHE？ | `augment.py:3801` `if hyp.channels == 5: alb = Albumentations(p=0)` —— **因为 albumentations 不支持 >4ch**，所以整条 alb 链被禁用 |
| 注释掉的 `A.CLAHE` 能否复用？ | ❌ **不能**：① 它已被注释（`augment.py:1825` 附近为 `# A.CLAHE(p=0.01)`）；② 属 `Albumentations`/`Albumentations4C`，在 5ch 下 `p=0` 不执行；③ 它是**随机**增强（`p=0.01`）且作用于**整图**，与我们要的"确定性、IR-only"语义不符 |
| `RandomHSV` 会不会动 5ch？ | ✅ **不会** —— `augment.py:3305` 首行 `if img.shape[-1] != 3: return labels`，5ch **直接返回** |
| 几何增强是否一致？ | ✅ 未改动（Mosaic / RandomPerspective / LetterBox 全部不碰） |

> **即：当前 5ch 训练路径的增强对像素值零改动**；本次 IR CLAHE 是数据读取阶段的**确定性**变换，
> 与增强管线**完全正交**，不引入第二个变量。

**§5 CLAHE 参数**：项目内**没有任何既有 CLAHE 参数**（`A.CLAHE` 全被注释）。
故按用户指定固定一组：**`clipLimit = 2.0`, `tileGridSize = (8, 8)`**，**不做超参搜索**。

---

## 11. run directory

```
runs/urban_multimodal_det_yolo11_rgbid_ir_clahe/
```
**新目录**，与正式 baseline 目录、A（P2）目录、以及 `*_rgb_baseline2`/`*_rgbt` 等**均不同**。

---

## 12. 完整训练命令

```bash
cd /root/autodl-tmp/urban_multimodal_detection
export OMP_NUM_THREADS=8

# ① 目标目录必须不存在（不得覆盖任何既有运行）
test ! -d runs/urban_multimodal_det_yolo11_rgbid_ir_clahe && echo "OK 目录不存在" \
  || { echo "!! 已存在，中止"; exit 1; }

# ② 训练（与 baseline 同配方：300 ep / 1280 / batch 8 / SGD lr0=0.005 / cos_lr / close_mosaic=10）
nohup python scripts/train.py \
  --model_config configs/yolo11m_earlyfusion.yaml \
  --train_config configs/train_rgbid_ir_clahe.yaml \
  > ir_clahe_train.log 2>&1 &
```

**需要同步到云端的文件**：

```
ultralytics/data/base.py              8bcf834930155bd58a99ccfc
ultralytics/cfg/default.yaml          6f91a2770b4e3cc7b393d98d
scripts/train.py                      80f6cdec1e52d41258ca7726
configs/train_rgbid_ir_clahe.yaml     2106385634b01a057357cc5c
```

**训练后官方评测**（与 baseline 逐字同参；本机 CPU 执行）：

```bash
python scripts/predict_rect.py \
  --weights runs/urban_multimodal_det_yolo11_rgbid_ir_clahe/weights/best.pt \
  --train_config configs/train_rgbid_ir_clahe.yaml \
  --source data/processed/rgbid_split/images/val/visible \
  --mode rect --imgsz 1280 --conf 0.001 --iou 0.7 --max_det 300 --max_boxes 300 \
  --batch 16 --output diagnostic/ir_clahe/best_full
python scripts/official_map.py --results diagnostic/ir_clahe/best_full/results --split_root data/processed/rgbid_split
python diagnostic/p3attn_probe/_sml_eval.py \
  "Baseline"=diagnostic/rgbid_inference_gain/RGBID_baseline/results \
  "IR-CLAHE"=diagnostic/ir_clahe/best_full/results
```

**判定（预注册）**：官方 mAP50-95 **> 0.50928** → 候选；**≤ 0.50928** → 关闭该方向。
baseline 对照：`0.77601 / 0.56386 / 0.50928`；small `0.05625` / medium `0.17984` / large `0.61711`。

---

## 13. 回滚方式

```bash
# 已跟踪的两个文件
git checkout -- ultralytics/cfg/default.yaml scripts/train.py

# 未跟踪的两个（.gitignore 的 data/ + 新配置）
# base.py：把 §4.1 的 5 行改回原 3 行（原代码见 §1）
rm -f configs/train_rgbid_ir_clahe.yaml
```

---

## 14. ⚠️ 两条必须提醒的事项

### 14.1 GPU 资源冲突

**你现在正在云端跑 Experiment A（P2，300 epoch ≈11.4 h）。**
**300-epoch 训练无法并行** —— 请等 A 结束（或明确放弃 A）后再启动 D，否则两个进程会争抢同一张 GPU，
两者的速度与显存都不可控。

### 14.2 `patience` 的取舍（我在实施中**忠于 baseline 保留了 80**）

D 的配置**逐字复制** baseline，包含 `patience: 80`。
- baseline 自己的 best 在 **ep237**，`patience=80` 下运行到 **ep300** 是**被 epoch 上限**终止的（不是被早停）→ 所以 80 对 baseline-like 曲线**无影响** ✓
- 但**若 D 的曲线提前见顶**（例如 ep150 后不再提升），它会在 **ep230** 被早停 —— 少跑 70 epoch。
  这不会影响 `best.pt` 的有效性（两者都用各自的 best），但**总预算不再严格相同**。
- 若你希望消除该风险：把配置里的 `patience: 80` 改为 `patience: 0`（禁用早停）即可 —— **一行改动，我等你指示**。

---

## 15. 合规状态

| 项 | 状态 |
|---|---|
| 正式 `best.pt` / `last.pt` / `submission.zip` / 两个 baseline 配置 | ✅ **SHA256 全部未变** |
| 正式 run 目录 | ✅ 未触碰 |
| 模型结构 / P2 | ✅ **未改**（D 用 `configs/yolo11m_earlyfusion.yaml`，**不含 P2**）|
| loss / optimizer / augmentation（除 IR preprocessing 本身） / depth / RGB | ✅ **未改** |
| 训练 | ❌ **未启动** |
| 新增 | `configs/train_rgbid_ir_clahe.yaml`、`diagnostic/ir_clahe/_smoke_test.py`、本报告 |
