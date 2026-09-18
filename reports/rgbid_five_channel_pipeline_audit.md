# RGBID 五通道数据链路审计

**日期**：2026-09-18
**性质**：**只读审计**。未训练、未修改任何数据/配置/权重/冻结产物。
**脚本**：`scripts/rgbid_pipeline_audit.py`（本次新增，只读）
**原始输出**：`diagnostic/modality_dropout/_rgbid_pipeline_audit.txt`

---

## 0. 结论速览

| 项 | 结果 |
|---|---|
| 审计项 | S1–S10 共 **10 段 / 22 项检查** |
| **FAIL** | **0** |
| WARN | 1（训练期 val 方形画布 vs 推理 rect 画布，**既有已知设计差异**） |
| 通道序 `[B,G,R,IR,D]` → `[R,G,B,IR,D]` | ✅ 逐通道验证通过 |
| 训练管线 vs 推理管线通道序 | ✅ **逐元素完全一致** |
| 5 通道 padding 值 | ✅ LetterBox / RandomPerspective / Mosaic **三处均为 114**，无 E9 残留 |
| 深度两条预处理分支 | ✅ 均被真实数据触发并逐像素验证通过 |
| 归一化 | ✅ 全通道统一 `.float()/255`，无逐通道 mean/std |
| **发现的问题** | 无阻断性缺陷；4 项需记录的结构性事实（见 §11） |

---

## 1. S1 原始文件格式与三模态配对

| split | visible | infrared | depth | 三模态交集 |
|---|---:|---:|---:|---:|
| train | 1600 | 1600 | 1600 | **1600** |
| val | 400 | 400 | 400 | **400** |

✅ 一一对应，无缺模态样本。

**⚠️ 数据是「混代」的**（同一 split 内并存两种来源）：

| 模态 | 格式 A | 格式 B |
|---|---|---|
| visible | `uint8 (1080,1920,3)` .jpg | `uint8 (360,640,3)` .jpg |
| infrared | `uint8 (1080,1920,3)` .jpg | `uint8 (360,640,3)` .jpg |
| depth | **`uint16 (1080,1920,1)` .png** | `uint8 (360,640,3)` .jpg |

抽样 80 张（val 全量）中：1080×1920 代占 53，360×640 代占 27（train 前三同理）。
**两代在通道数、位深、扩展名上都不同** —— 这是本链路必须同时处理两条深度分支的根因。

---

## 2. S2–S5 融合与通道序

样本 `00000045.jpg`，`load_and_preprocess_image(use_simotm="RGBID")` 实读：

```
合并后: shape=(360, 640, 5)  dtype=uint8
```

**逐通道相等性验证**（与独立读取的原图比对）：

| 融合通道 | 期望来源 | 结果 |
|---|---|---|
| ch0 | 可见光 **B** | ✅ |
| ch1 | 可见光 **G** | ✅ |
| ch2 | 可见光 **R** | ✅ |
| ch3 | 红外（拉伸后） | ✅ |
| ch4 | 深度（取 ch0） | ✅ |

✅ ch0/ch1/ch2 互不相同（排除「三通道被灰度复制」这类退化）。
✅ 融合图 `(H,W)` == 可见光原图 `(H,W)`（三模态已对齐，无尺寸错位）。

来源代码：`base.py::_merge_channels_rgbid` → `cv2.merge((b, g, r, ir, d))`。

---

## 3. S3 红外 1%/99% 分位数拉伸

```
IR raw: min/max = 0/255     1%/99% 分位 = 0.0/197.0     拉伸后 min/max = 0/255
```

✅ 融合 ch3 **逐元素等于**独立复算的拉伸结果。
✅ ch3 唯一值 199（未被拉伸成常量）。

> 拉伸公式（`base.py` RGBID 分支）：`clip((ir - p1) * 255/(p99 - p1), 0, 255)`，
> 作用在**每张图自身**的分位数上，因此**同一物理红外强度在不同图上映射到的灰度不同**。

---

## 4. S4 深度两条分支（均实测触发）

### 4.1 8bit 分支（`.jpg`，360×640×3）

```
raw dtype=uint8 -> 取 ch0 -> shape=(360,640)
✅ 融合 ch4 == 深度第 0 通道
```

`im_depth[im_depth<300]=0` 与 `/19999*255` **不执行**（被 `if im_depth.dtype != np.uint8` 挡住）。
三通道逐像素相同（灰度复制），取 ch0 无信息损失。

### 4.2 16bit 分支（`.png`，1080×1920）

样本 `000002_080_00000048.png`：

```
原图 : shape=(1080,1920) dtype=uint16  min/max = 0/19999   <300mm 像素占比 = 33.97%
期望 : <300 -> 0, 再 clip(v/19999*255, 0, 255)
实际 : 融合 ch4 min/max = 0/255，0 值占比 = 33.97%（704,355 个 0 像素）
✅ 逐元素完全等于期望映射
```

**结论**：`<300mm → 0` 这条「无信号哨兵」在 16bit 数据上**真实执行**；
8bit 数据的 0 值则是**上游数据制备阶段**已完成的等价映射结果。

---

## 5. S6 LetterBox 逐通道 padding

```
train  LetterBox(1280, scaleup=True ) : out=(1280,1280,5)  各通道众数 = [114,114,114,114,114]  ✅
val    LetterBox(1280, scaleup=False) : out=(1280,1280,5)  各通道众数 = [114,114,114,114,114]  ✅
```

**探针实验**（内容全 7、尺寸 64×128 → letterbox 到 128×128，确保一定补边）：

```
左上角 = [114,114,114,114,114]   ← 补边区
中心   = [7,7,7,7,7]             ← 内容区
```

✅ 5 通道补边值一致，且确实为 padding 而非内容。

> **代码路径说明（重要）**：`LetterBox.__call__` 中 `channels == 5` **不落入**
> `value = (114,114,114)` 那个 3 元组分支，而是落入末尾的
> `else: # multispectral` 分支：
> ```python
> pad_img = np.full((h+top+bottom, w+left+right, c), fill_value=114, dtype=img.dtype)
> ```
> 该分支显式对**全部 c 个通道**填 114，因此 np/cv2 的 Scalar 4 元组循环问题**不会触发**。
> （若将来有人把 5ch 改走 `copyMakeBorder`，3 元组 `value` 会让 cv2 按 Scalar(114,114,114,0)
> 循环取值 → 第 4 通道会变成 0。**属于脆弱点，但当前不可达。**）

---

## 6. S7 Mosaic 画布填充

```
Mosaic.gray_value = 114
5 通道画布初值 = [114,114,114,114,114]  ✅
```

与 LetterBox padding 值一致（同为 114），三处 padding 语义统一。

---

## 7. RandomPerspective 的 5 通道分支（静态核对）

`augment.py:1097`：

```python
elif channels == 5:
    # RGBID: 前 3ch 用 (114,114,114)，后 2ch(IR+D) 用标量 114（cv2 不接受 5 元组 borderValue）
    rgb = cv2.warpAffine(img[:, :, :3], M[:2], dsize=self.size, borderValue=value)
    aux = cv2.warpAffine(img[:, :, 3:], M[:2], dsize=self.size, borderValue=114)
    img = np.concatenate((rgb, aux), axis=2)
```

✅ 同样落到「5 通道全部 114」。

**E9 残留核查**：E9 的 3 处 `depth padding=0` 修改全部位于 **`channels == 4`（RGBD）** 分支
（`RandomPerspective` / `LetterBox` / `Mosaic` 的 4ch 判断），当前 `augment.py` 已回滚：

```
diff augment.py.bak_20260912_222420 augment.py
  < elif channels == 4:  ... borderValue=0      (备份含 E9)
  > elif channels in [1, 3, 4]:                 (当前已回滚)
```

✅ **当前本地 augment.py 无 E9 残留**；且**即使有，也只影响 4ch RGBD，不会影响 RGBID 5ch**。
（`augment.py` 被 `.gitignore` 忽略、不随 git 同步云端 —— 见 §11-④。）

---

## 8. S8 训练管线 vs 推理管线（关键一致性验证）

对**同一张 letterbox 输出**分别走两条路径：

| 路径 | 实现 | 输出 |
|---|---|---|
| 训练/验证 | `Format._format_img`（`augment.py`） | `(5, 1280, 1280) uint8` |
| 推理/提交 | `predict._to_chw`（`predict.py`） | `(5, 1280, 1280) uint8` |

```
✅ 两条路径 shape 一致
✅ 两条路径逐元素完全一致（np.array_equal == True）
✅ 输出 ch0 == 可见光 R（BGR→RGB 已翻转）
✅ 输出 ch3 == IR（未翻转）
✅ 输出 ch4 == Depth（未翻转）
```

> 训练侧 `Format` 的 5ch 分支：
> ```python
> img3c = np.ascontiguousarray(img.transpose(2,0,1)[:3,:,:][::-1])  # BGR -> RGB
> img2c = img.transpose(2,0,1)[3:,:,:]                              # IR, D 保持原序
> ```
> 推理侧 `predict._to_chw` 对 4/5ch 做 `np.concatenate([im[:3][::-1], im[3:]])` —— 语义等价。
> **推理管线已按训练端的正确语义实现**（`predict.py` 注释中亦声明「fork 的 5ch 分支会错误地整体反转」，
> 但当前 `Format` 的 5ch 分支已正确，两侧一致）。

---

## 9. S9 归一化

```
.float()/255 -> range [0.0000, 1.0000]
各通道均值 = [0.4509, 0.4513, 0.4501, 0.4301, 0.3981]   (B, G, R, IR, Depth)
```

✅ 全通道统一 `/255`，**无逐通道均值减除 / 方差归一化**（与 `detect/train.py:159` 一致）。

---

## 10. S10 训练 / 推理几何差异（WARN，既有设计）

| 路径 | 画布 |
|---|---|
| 训练期 val（`build_transforms` 的 `augment=False` 分支） | `LetterBox((1280,1280), scaleup=False)` → **方形 1280×1280** |
| 提交/官方评测（`predict_rect --mode rect`） | `_compute_rect_shape` → **rect 736×1312**（长边 1280，stride=32 对齐） |

⚠️ 这意味着**训练期用于选 best.pt 的指标（方形）与提交时的实际推理几何（rect）不同**。
这是既有事实，且已有线上验证：rect 修复在线上兑现 **+0.862 分**（官方 51.0510）。
本次审计**未做任何修改**，仅记录。

---

## 11. 需要记录的结构性事实（非缺陷，但会影响结论解释）

**① 数据集是混代的。** 同一 split 内并存 `1920×1080` 与 `640×360` 两代源图，
depth 并存 `uint16 .png` 与 `uint8 .jpg`。任何「单一预处理假设」的分析都必须先确认覆盖了两代。

**② 红外对比度拉伸是逐图自适应的。** 每张图用自身 1%/99% 分位拉伸，
因此 IR 通道的**绝对灰度不跨图可比**；`ir_fill=125` 是「训练集 IR 均值」这一说法需要谨慎 ——
均值本身随图而变（本次实测 train IR 均值 116.9 / val 127.0）。

**③ 各通道实测量级**（本次实测，200 train / 120 val 张）：

| 通道 | train 均值 (/255) | train 0 值占比 | val 均值 (/255) | val 0 值占比 |
|---|---:|---:|---:|---:|
| B / G / R | 0.427 / 0.436 / 0.416 | 0.0–0.1% | 0.429 / 0.444 / 0.422 | 0.0–0.1% |
| **IR** | **0.459** | 5.3% | **0.498** | 2.9% |
| **Depth** | **0.233** | **36.9%** | 0.313 | **31.0%** |
| padding 值 114 | 0.447 | — | 0.447 | — |

> 注：`ir_fill=125` → 0.490，与 IR 实测均值（0.459–0.498）同量级，属**分布内**填充；
> `depth_fill=0` → 0.000，而 depth 本就有 31–37% 的 0 值像素，同样属**分布内**。
> **两个填充值的选择站得住脚。**
> （Modality Dropout 报告 §3 记的「IR 125.02 / depth 零值 31.9%」与本次实测同量级，
> 差异来自抽样与两代数据混合比例。）

**④ `ultralytics/data/augment.py` 不受 git 管理（`.gitignore:4: data/`）。**
本地已核验无 E9 残留，但**云端副本无法从本机验证**。若将来再做以 RGBD 为基线的 A/B，
建议在云端先跑一次 `grep -n "E9" ultralytics/data/augment.py`（应无输出）。
**对 RGBID 5ch 无影响**（见 §7）。

---

## 12. 审计未做什么

未训练 / 未修改任何数据、配置、权重、冻结产物 / 未重新生成任何预测 /
未修改 `predict.py`、`predict_rect.py`、`augment.py`、`base.py` / 未上传线上。

新增文件仅一个只读脚本：`scripts/rgbid_pipeline_audit.py`。
