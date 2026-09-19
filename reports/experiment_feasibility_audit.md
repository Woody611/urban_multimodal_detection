# 后续实验可行性审计（只读）

**日期**：2026-09-18
**性质**：**只读审计**。未修改任何代码、未训练、未生成提交、未动任何权重。
**标注约定**：`【实读】`= 从本仓库文件/代码直接读出 · `【外部摘要】`= 来自公开仓库的搜索摘要（**非逐字代码，可信度较低**）· `【推测】`= 机制推理 · `【建议】`= 行动方案

---

## 0. 结论速览

| 实验 | 能做 | 干净单变量 | 需改文件 | 主要风险 | 建议 |
|---|---|---|---|---|---|
| **A. RGBID@1280 + P2** | ✅ | ✅ | 0（已完成，含 init 手术） | 低；期望值低（同族 2 次中性） | **唯一推荐立即执行** |
| **B. Depth Validity Mask (6ch)** | ⚠️ 技术上可行 | ❌ **做不到** | **5 个源码 + 2 个配置** | **高**：6ch 会静默触发 3 处错误分支（见 §4） | **暂缓** |
| **C. Depth Encoding (inverse/log/minmax)** | ✅ | ✅ | **1 个源码**（`base.py` 一处） | 中：数值稳定性 + 两代数据语义不一致 | **可做，列 P1** |
| **D. IR CLAHE** | ✅ | ✅ | **1 个源码**（`base.py` 一处） | 低：CPU 开销很小 | **可做，列 P1** |

---

## 1. 当前代码结构理解（§五，全部实读）

### 1.1 五个关键文件的实际状态

| 文件 | 作用 | 本次审计读到的关键行 |
|---|---|---|
| `configs/yolo11m_earlyfusion.yaml` | 模型结构 | `ch: 5`；`scales.m = [0.50, 1.00, 512]`；24 层；首层 `Conv(5→64,3,2)`；`Detect[[16,19,22]]` |
| `configs/train_rgbird_ir_quicktest.yaml` | 训练配置 | 无 `aug` 段（全走默认）；`epochs=300, batch_size=8, imgsz=[1280,1280], lr0=5e-3, warmup=3, patience=80` |
| `scripts/train.py` | 训练入口 | `_build_train_kwargs()` 把 YAML 映射为 `YOLO.train(**kwargs)`；只转发白名单键 |
| `scripts/predict_rect.py` | 冻结推理管线 | `--mode rect` 复刻 `model.val()`；`_to_chw` 走 `predict.py` |
| `ultralytics/data/base.py` | 数据读取 | `load_and_preprocess_image()` 的 `RGBID` 分支（L313–341）；`_merge_channels_rgbid`（L421）|

### 1.2 三模态的实际数据流（实读）

```
base.py::load_and_preprocess_image(use_simotm='RGBID')          # L313
  im_visible  = imread(path)                        # BGR 3ch
  im_infrared = imread(path.replace(...), GRAYSCALE) # 单通道
                  -> 1%/99% 分位数拉伸               # L327-329  ★ 当前已有 percentile 拉伸
  im_depth    = imread(path.replace(...), UNCHANGED)
                  -> if dtype!=uint8: im[im<300]=0; clip(im/19999*255)   # L336-338
                  -> if ndim==3: 取 ch0
  _resize_images_3(vis, ir, dep)                     # 三模态统一尺寸
  _merge_channels_rgbid: cv2.merge((b,g,r,ir,d))     # L421 → HWC [B,G,R,IR,D]
─────────────────────────────────────────────────────────────────
augment.py::Format._format_img 5ch 分支: [:3][::-1] + [3:]   → CHW [R,G,B,IR,D]
train.py:159:  batch["img"].float() / 255
─────────────────────────────────────────────────────────────────
model.0:  Conv(5 → 64, k=3, s=2, p=1)
```

### 1.3 ★ 三条对本次审计最关键的实读事实

1. **IR 已做对比度增强**：`base.py:327-329` 的 **1%/99% 分位数拉伸**——
   这正是 CityViMD-Net 里 `ir_encoding` 三个选项中的 **`percentile`** 那一档。
2. **Depth 已做线性截断归一化**：`im[im<300]=0; clip(im/19999*255)` ——
   与 CityViMD-Net 的默认 `clip(/20000,0,1)` **本质相同**（截断点 19999 vs 20000 mm 的 0.005% 差异可忽略）。
3. **CLAHE 在 5ch 路径上被显式关闭**：`augment.py:3801`
   ```python
   if hyp.channels == 5:
       # RGBID 5ch：albumentations 不支持 >4ch，关闭模糊/CLAHE；RandomHSV 对非 3ch 自动跳过
       alb = Albumentations(p=0)
   ```
   → **当前 5ch 训练完全没有 CLAHE**（`A.CLAHE` 那几行在 `Albumentations`/`Albumentations4C` 里也**已被注释掉**：`augment.py:1825` 附近为 `# A.CLAHE(p=0.01)`）。

### 1.4 关于 P2 的历史实验（§五-10）

```
E5 = runs/urban_multimodal_det_e5_yolo11m_rgbd_p2   model=configs/yolo11m_midfusion_rgbd_concat_res_p2.yaml
E4 = runs/urban_multimodal_det_e4_yolo11m_rgbd       model=configs/yolo11m_midfusion_rgbd_concat_res.yaml
E5 vs E4 的 args 差异 = 仅 model / name / save_dir   => 【实读】干净单变量 ✓
E5 best mAP50-95 = 0.47796   E4 = 0.47783   Δ = +0.00013
```

**但 E5 与当前 baseline 是不同架构族**：E5 是 **RGBD 双分支中期融合**（`ch: 4`，`Silence`/`SilenceChannel` 分支结构），
**imgsz=640，batch=16**；当前 baseline 是 **RGBID 单 stem 早期融合**（`ch: 5`），**imgsz=1280，batch=8**。

**§五-10 的问题：「P2@640 几乎没提升，是 P2 无效还是 640 下小目标信息不足？」**

**从代码无法判定** —— 这两个解释在现有数据上**不可分离**。但可以给出机制层面的量化：

| | P2 特征图 | P3 特征图 | 中位小目标（√area 17.7px@1280）在 P2 上占 | 在 P3 上占 |
|---|---:|---:|---:|---:|
| **imgsz=640**（E5） | 160×160 | 80×80 | **2.2 格** | 1.1 格 |
| **imgsz=1280**（本项目） | 320×320 | 160×160 | **4.4 格** | 2.2 格 |

【推测】640 下即使加了 P2，中位小目标也只剩 **2.2 个特征格** —— 低于检测头能稳定工作的尺度。
1280 下是 **4.4 格**，是 640 的两倍。**这是"P2 在 640 失效、在 1280 可能不同"的一个机制性理由，但不是证据。**
**结论：E5 的结果不能外推到 1280 的 RGBID；必须实测。**

---

## 2. Experiment A（RGBID@1280 + P2）—— **✅ 可实施**

### 2.1 逐问回答（§三）

| # | 问题 | 实读答案 |
|---|---|---|
| 1 | backbone 是否已产生 P2？ | **已产生** —— layer 2 `C3k2(128→256)` 输出 stride-4 的 P2 backbone 特征 |
| 2 | 为什么 Detect 没用？ | **不是"没产生"，是"没接出去"** —— P2 特征只被 layer 15 的 Concat 用于构造 P3（`[-1, 4]`），没有任何 head 路径使用它 |
| 3 | 只改 head/Detect 还是 neck 也要？ | **neck 也要** —— 必须新增一条 `Upsample → Concat(P2 backbone) → C3k2` 把 P2 接出去。已在 `configs/yolo11m_rgbid_p2_probe.yaml` 实现（layer 23/24/25） |
| 4 | P2 @1280 feature map | **320×320**（hook 实读） |
| 5 | P2 stride | **4**（`Detect.stride = [4,8,16,32]` 实读） |
| 6 | 显存增加 | **待 GPU 实测**。FLOPs 273.85 → **422.69 G（1.55×）**；【推测】batch=8 约 +2.5~3.5 GB |
| 7 | 低层背景噪声导致 P 下降？ | **是真实风险**（这正是 P2 的经典代价）。缓解：`bias_init` 已按 stride 给 P2 最保守的先验（见下） |
| 8 | Detect 能否直接 3 尺度 → 4 尺度？ | **可以** —— 实测 `nl=4, no=76, stride=[4,8,16,32]`，输出 `(1,16,136000)` |
| 9 | 是否有隐含假设？ | **未发现**：`bias_init`(`head.py:142`) 用 `zip(cv2,cv3,stride)` 逐尺度；`v8DetectionLoss`(`loss.py:422`) 用 `self.stride=m.stride` 且 `feats=feats[:stride.size(0)]`；`nl=len(ch)` 全部自适应 |
| 10 | checkpoint/resume/validation/export 正常？ | **validation 已实测正常**（官方评测跑通）。checkpoint 加载已实测（743/743）。**resume / export 未测**（本实验不需要） |

**§三-9 补充实测**（一个值得记录的副作用）：
`bias_init` 的类偏置公式 `log(5/nc/(640/s)²)` 在 4 个尺度上给出：

| 尺度 | stride | 初始 cls bias |
|---|---:|---:|
| **P2** | **4** | **−11.02**（最保守） |
| P3 | 8 | −9.64 |
| P4 | 16 | −8.25 |
| P5 | 32 | −6.87 |

→ P2 分支**初始置信度最低**（这是 ultralytics 的标准公式，非 bug），意味着 P2 需要更多训练才能"开口"。

### 2.2 §四 实验隔离核验

| 项 | 状态 |
|---|---|
| 数据 split / label / resize / augmentation / normalization | ✅ 未动（模型层实验，不碰数据） |
| optimizer / lr / scheduler / epochs / batch / seed / AMP / wd / warmup | ✅ 未动（已程序化核验：与 baseline 仅 3 处差异，见 §7） |
| 模型：loss / attention / neck 其它部分 / backbone | ✅ 未动（P2 是**追加**在末尾，layer 0–22 逐位不变） |

### 2.3 隐藏风险

1. **期望值低**：同族已有两次中性结果（E5 @640 = +0.00013；1536 = small AP 未改善）。
2. **FLOPs 1.55×**：训练时长与推理时长都显著上升。
3. **早停陷阱（已修）**：`patience` 必须设为 `0`（否则 best=早期 → 早停）。
4. **索引位移的初始化陷阱（已修）**：Detect 索引 23→26 会让 P3/P4/P5 的框回归塔丢失 COCO 预训练 —— 已用尺度重映射补齐（`p2_probe_init_scratch.pt`，115 张量）。

### 2.4 状态

**pre-flight 已全部完成并通过**（逐层 init 等价性 528/528 逐位一致、one-batch sanity check、stride/形状实读）。
**只差 GPU feasibility + 训练。**

---

## 3. Experiment B（Depth Validity Mask，5ch → 6ch）—— **⚠️ 技术上可行，但做不到干净单变量**

### 3.1 CityViMD-Net 的实现方式【外部摘要】

> `data.depth_validity_mask`：官方数据约定把 depth 0 或过小视为无效。
> 启用后 depth **输出两个通道** `[normalized depth, 0/1 validity mask]`，输入变为 `[B,6,H,W]`；
> 并且必须把 `model.in_channels.depth` 设为 2（模型构建会校验一致性）。
> 在所有 depth encoding 下，**无效深度统一输出为 0**。

### 3.2 我们的数据是否存在无效 depth？—— **是，且两代数据不一致**【实读】

| 数据代 | 文件数 | depth 格式 | **无效像素占比** | 有效深度均值 |
|---|---:|---|---:|---:|
| 16-bit `.png`（1920×1080） | 1478 / 1600 | uint16 单通道 mm | **31.5%**（`<300mm`） | 7717 mm |
| 8-bit `.jpg`（640×360） | 122 / 1600 | uint8 3ch 灰度 | **74.3%**（`==0`） | 不可得（mm 信息已丢失） |

> ⚠️ **这是 Experiment B 的核心障碍**：两代数据的"有效"定义与占比差异巨大（68.5% vs 25.7% 有效）。
> 8-bit 代**已经没有 mm 信息**，只能判断 `>0`；16-bit 代用 `>=300mm`。
> **同一个 mask 通道在两代数据上语义不同** → 模型学到的是混合信号。

### 3.3 ⚠️ 隐藏风险（本次审计最重要的发现）：**6ch 会在 3 处静默触发错误分支**

| # | 位置 | 实读代码 | 后果 |
|---|---|---|---|
| **1** | `augment.py::Format._format_img` 6ch 分支 | `[:3][::-1]` + `[3:][::-1]` | `[B,G,R,IR,D,M]` → **`[R,G,B,M,D,IR]`** —— 后 3 通道被**反转**，IR/Depth/Mask 位置全错 |
| **2** | `scripts/predict.py::_to_chw` 6ch 分支 | `im[:3][::-1]`, `im[3:][::-1]` | **同样的反转** → 与训练侧一致（所以不会崩），但语义同样是 `[R,G,B,M,D,IR]` |
| **3** | `augment.py::v8_transforms` (`channels==6`) | `alb=Albumentations4C(p=1.0)`；`random_hsv=RandomHSV6C(hgain=0.015, sgain=0.7, vgain=0.4)` | **`RandomHSV6C` 会把 `img[:,:,3:]` 当成"灰度三通道组"做 HSV 变换**（`cv2.cvtColor(gray, COLOR_BGR2HSV)`）→ 对 `[IR,D,M]` 三个**不同物理量**做色相/饱和度变换 = **跨模态污染**。注意 `sgain=0.7`、`vgain=0.4` 都不为 0，**每次都会触发** |

> 也就是说：**把 `ch: 5` 改成 `ch: 6` 会同时改变 5 件事** ——
> ①通道数 ②`Format` 的通道序 ③`Albumentations`→`Albumentations4C`
> ④`RandomHSV`→`RandomHSV6C`（且它污染模态通道）⑤stem 5→6 的预训练加载。
> **这不是"只加一个 mask 通道"，而是一次改动 5 个变量的实验，且其中 3 个是静默的。**

### 3.4 要做成干净实验，必须改的 5 个文件

| # | 文件 | 改动 |
|---|---|---|
| 1 | `ultralytics/data/augment.py` | `Format._format_img` 6ch 分支改为 `[:3][::-1] + [3:]`（保持 `[R,G,B,IR,D,M]`） |
| 2 | `ultralytics/data/augment.py` | `v8_transforms` 增加 `channels==6` 的处理，**镜像 `channels==5`**：`alb=Albumentations(p=0)` + 禁用 RandomHSV |
| 3 | `scripts/predict.py` | `_to_chw` 6ch 分支改为 `[im[:3][::-1], im[3:]]` |
| 4 | `ultralytics/data/base.py` | 新增 `RGBIDM` 分支（或 `_merge_channels_rgbid` 加开关）：在 `<300→0` **之前**生成 mask，`cv2.merge((b,g,r,ir,d,m))` |
| 5 | `ultralytics/models/yolo/detect/train.py` | `_transfer_rgb_pretrained` 支持 **6ch stem**（当前条件写死 `in_channels == 5`） |
| + | 新增 2 个配置 | `configs/yolo11m_earlyfusion_rgbidm.yaml`（`ch: 6`）+ 对应 train yaml |

### 3.5 §二 逐问回答

| # | 问题 | 答案 |
|---|---|---|
| 2 | 是否适合迁移到我们当前 YOLO11m？ | **设计理念适合**（两者都是 5ch 早融合 + 单 stem），但**实施代价高**（5 文件 + 3 处静默陷阱） |
| 4 | mask 应该在哪里生成？ | `base.py::load_and_preprocess_image` 的 RGBID 分支，**在 `im[im<300]=0` 之前**（16bit 用 `>=300`；8bit 用 `>0`） |
| 5 | 离线生成 6ch 数据 vs dataloader 动态？ | **dataloader 动态**更优（不动 1600×3 数据文件、可随时改阈值、与 CityViMD-Net 的"配置开关"一致） |
| 6 | 首层/stem 如何修改？ | `Conv(5→64)` → `Conv(6→64)`；`ch: 5` → `ch: 6`；**`make_divisible` 不受影响**（`ch` 直接作为 `c1`，不参与 `make_divisible`，见 `tasks.py:347/1067`） |
| 7 | 是否影响预训练加载？ | **影响** —— `_transfer_rgb_pretrained` 的 `in_channels == 5` 条件会失配 → **5ch stem 的初始权重不会写入** → 必须改第 5 个文件 |
| 8 | 能否保持干净单变量？ | ❌ **不能**（除非先改完上述 5 个文件；改完后"加 mask"本身才是单变量） |
| 9 | 显存/速度增加？ | **可忽略**：stem FLOPs +20%（2880→3456 MACs/输出像素），全模型影响 **<0.1%**；显存几乎不变 |
| 10 | 是否值得在截止前做？ | **不值得** —— 见 §7 优先级 |

---

## 4. Experiment C（Depth Encoding）—— **✅ 可做，且是干净单变量**

### 4.1 逐问回答

| # | 问题 | 实读答案 |
|---|---|---|
| 1 | 当前 Depth 数值范围与分布 | 16bit 代：0–19999 mm，**31.5% 为 0**，有效均值 7717 mm；8bit 代：uint8 0–255，**74.3% 为 0** |
| 2 | 当前 pipeline 是否已做 normalization？ | **已做**：`im[im<300]=0` → `clip(im/19999*255, 0, 255)` → uint8；之后 trainer 统一 `.float()/255`。**等价于 [0,1] 的线性截断归一化** |
| 3 | 直接切 inverse/log 会不会重复归一化或产生异常？ | **会**。① 当前已在 uint8 域，inverse 需在**原始 mm 域**做才能有意义 → 必须**替换**而非追加；② `1/d` 对 31.5–74.3% 的 0 像素**除零**；③ `log(0) = −inf` |
| 4 | 哪种最容易实现？ | **`linear`（现状）与 `minmax`（逐图归一化）** —— 都只改 `base.py` 一处，输出仍为 uint8 1ch |
| 5 | 哪种最适合严格 ablation？ | **`minmax`（逐图 min-max 归一化）** —— 单变量、无数值奇点、与我们 IR 已有的"逐图百分位拉伸"同类思路，可解释性强 |
| 6 | 能否只改 preprocessing、保持模型结构完全不变？ | **✅ 能** —— 输出仍是 **1ch uint8** → `cv2.merge` 仍 5ch → stem 不变 → **预训练加载不变** |
| 7 | 训练成本？ | **≈0**（纯数值变换，无额外 IO；CPU 开销 <1 ms/图） |
| 8 | 数值稳定性？ | `inverse`/`log` 有除零/−inf；**`minmax` 无奇点**（`(d−dmin)/(dmax−dmin)`，全 0 图需特判） |

### 4.2 ⚠️ 必须注意的语义陷阱：**两代数据的 encoding 不可比**

- 16-bit 代：可以从 **mm** 出发做任意 encoding（inverse 有意义）
- 8-bit 代：**mm 信息已丢失**（上游已做 `clip(mm/19999*255)`），对它做 inverse 只是对**已归一化的灰度**取倒数 —— **物理意义不同**

→ **任何非 linear 的 encoding 在两代数据上语义不一致。** 这**不阻止实验**（baseline 也有同样问题），
但意味着结果的解释必须谨慎。

---

## 5. Experiment D（IR CLAHE）—— **✅ 可做，且是干净单变量**

### 5.1 逐问回答

| # | 问题 | 实读答案 |
|---|---|---|
| 1 | 当前 IR 格式/范围/dtype | **8-bit** 灰度存成 3ch JPEG（两代：1920×1080 `.png` 与 640×360 `.jpg`）；读取时 `IMREAD_GRAYSCALE` → 单通道 uint8 0–255 |
| 2 | 当前 pipeline 是否已做对比度增强？ | **已做** —— `base.py:327-329` 的 **1%/99% 分位数拉伸**（即 CityViMD-Net 的 `percentile` 选项） |
| 3 | CLAHE 应放在哪里？ | `base.py::load_and_preprocess_image` 的 RGBID 分支，**替换** 现有的 percentile 拉伸（或在其后追加，二选一，必须明确） |
| 4 | 是否影响训练/验证一致性？ | **只要同时改 `load_and_preprocess_image`，train 与 val 走同一函数 → 天然一致** ✓（CityViMD-Net 也强调"训练推理共享同一 encoding 实现"） |
| 5 | 能否只改 IR preprocessing？ | **✅ 能** —— 输出仍 1ch uint8 → 5ch 不变 → 模型/预训练/结构全不动 |
| 6 | CPU 开销？ | **实测 4.5–4.9 ms/图**（1920×1080，clipLimit=2.0/3.0，tile 8×8）→ 1600 图/epoch 增加 **+7.2~7.8 s**（baseline 130.9 s/epoch 的 **+5.5~6.0%**）。**且 `workers=4` 并行 → 若 dataloader 非瓶颈，wall-clock 影响 ≈ 0** |
| 7 | 是否改变数据分布使 baseline 不再严格可比？ | **会，但这正是实验的目的** —— CLAHE 改变 IR 的局部对比度分布。严格来说这是"我们主动引入的变量"，可接受 |

### 5.2 与当前 pipeline 的关系

**注意**：我们**不是**在做"从 raw 到 CLAHE"的对比 —— 我们当前**已经用了 percentile**。
所以 Experiment D 的真实语义是 **`percentile` → `clahe` 的替换**（不是 raw → clahe）。

---

## 6. §六 代码级对照表（CityViMD-Net vs 我们）

> **可信度声明**：CityViMD-Net 一列全部来自 **WebSearch 返回的仓库摘要**（`github.com` 的 WebFetch 被本机网络策略拦截）。
> **非逐字代码**，数值细节（如 19999 vs 20000）可能有偏差，**实施前必须人工核对源码**。

| 项目 | **CityViMD-Net**【外部摘要】 | **我们当前**【实读】 | 是否一致 |
|---|---|---|---|
| RGB channels | 3ch 8-bit PNG → `/255` | 3ch（BGR JPEG/PNG）→ `.float()/255` | ✅ 等价 |
| IR channels | **1ch**，8-bit PNG → GRAYSCALE → `/255` | **1ch**，GRAYSCALE → **1%/99% 分位拉伸** → 3ch→取 1ch | ⚠️ **不一致**：我们多了一层 percentile 拉伸 |
| Depth channels | **1ch**，16-bit PNG → `clip(/20000,0,1)` | **1ch**，`<300→0` → `clip(/19999*255)` → uint8 | ✅ 近似等价（截断点差 0.005%，域不同但都线性） |
| **Depth validity mask** | **有开关**（`data.depth_validity_mask`）→ depth 变 2ch，输入 6ch | **无** | ❌ **我们没有** |
| Depth encoding | **有开关**：`linear`(默认)/`inverse`/`log`/`minmax` | **仅 linear**（硬编码） | ❌ 我们只有 1/4 |
| IR preprocessing | **有开关**：`raw`(默认)/`clahe`/`percentile` | **percentile**（硬编码） | ⚠️ 我们用的是它的第 3 档，但**不可切换** |
| Input resolution | 未从摘要中获得 | 1280 | 未找到 |
| **P2** | 未从摘要中获得 | **无**（A 实验将加） | 未找到 |
| P3/P4/P5 | "PAN + decoupled heads"（未给尺度细节） | P3/P4/P5 = layer 16/19/22 | 未找到细节 |
| Fusion position | **早期融合**（5ch 单 backbone） | **早期融合**（5ch 单 stem） | ✅ **同族** |
| Backbone | CSPDarknet-style，**单一共享** | YOLO11m backbone | ✅ 同族 |
| Neck | PAN | FPN+PAN | ✅ 同族 |
| Head | decoupled heads | YOLO11 `Detect`（decoupled） | ✅ 同族 |
| Loss | 未从摘要中获得 | CIoU + DFL（ultralytics 默认） | 未找到 |
| Augmentation | 几何变换三模态同步；RGB 颜色增强与 IR gamma 只改像素 | Mosaic + RandomPerspective（5ch 路径关闭 alb/HSV） | ⚠️ 结构不同 |

**另外 CityViMD-Net 还有两个我们没有的开关**（摘要提及）：
- `data.test_blacklist`：**测试集哈希黑名单**，train/val loader 拒绝测试集图像 —— **合规性设计**，值得借鉴思路
- `train.augment.modality_dropout`：训练期随机把 IR 或 Depth 整图置零 —— **本项目已实测为负**（官方 0.49029 vs 0.50928）

> **最重要的一条**：CityViMD-Net 的**默认配置**（5ch 早期融合 + 单 backbone + linear depth + raw IR）
> 与我们**高度同族**；而它的**默认 IR 是 `raw`**、**默认 depth encoding 是 `linear`**、
> **validity mask 是关闭的** —— 也就是说，**它的"基线默认"与我们的"当前实现"在 IR 上不同、在 depth 上相同。**
> **不能因为"它有某个功能"就认为迁移过来一定更好 —— 它的默认值恰恰是"不启用"这些功能。**

---

## 7. §七 实验可行性矩阵

| 实验 | 是否能做 | 需要修改什么 | 是否单变量 | 风险 | 显存影响 | 时间影响 | **推荐优先级** |
|---|---|---|---|---|---|---|---|
| **A. RGBID@1280 + P2** | ✅ **能** | **0 个源码**（模型 yaml + train yaml + init ckpt 均已就绪） | ✅ **是** | **低**（pre-flight 全通过；同族 2 次中性） | FLOPs **1.55×**；【推测】+2.5~3.5 GB | 训练 **≈11.5 h**（实测 136.3 s/ep） | **P0** |
| **B. Depth Validity Mask (6ch)** | ⚠️ 技术上能 | **5 个源码 + 2 个配置** | ❌ **否**（改 `ch` 会静默触发 3 处错误分支） | **高**（`RandomHSV6C` 跨模态污染；两代 mask 语义不一致） | **<0.1%**（可忽略） | ≈0 | **P3（暂缓）** |
| **C. Depth minmax/inverse/log** | ✅ **能** | **1 个源码**（`base.py` depth 分支） | ✅ **是**（输出仍 1ch uint8） | **中**（inverse/log 有奇点；两代语义不一致） | 0 | ≈0 | **P1**（推荐 **minmax**） |
| **D. IR CLAHE** | ✅ **能** | **1 个源码**（`base.py` IR 分支） | ✅ **是**（输出仍 1ch uint8） | **低**（CPU +6% 且被 4 workers 吸收；train/val 天然一致） | 0 | CPU +7.2~7.8 s/ep（≈0 wall-clock） | **P1** |

**优先级依据**（非主观）：

| 判据 | A (P2) | B (mask) | C (depth enc) | D (IR CLAHE) |
|---|---|---|---|---|
| 与当前瓶颈（小目标定位）相关性 | **高**（直接在小尺度生成框） | 中（改善模态质量，非尺度） | 中（改变 depth 语义表示） | **低**（IR 不是主瓶颈） |
| 改动大小 | **0 源码** | 5 源码 | **1 源码** | **1 源码** |
| 实验可解释性 | **高**（单变量、逐层 init 已证等价） | **低**（5 变量） | **高** | **高** |
| 训练成本 | 11.5 h | 11 h | 11 h | 11 h |
| OOM 风险 | 中（FLOPs 1.55×，待实测） | 无 | 无 | 无 |
| 截止时间（剩 ~46 h） | **可负担** | 可负担但会挤占 | 可负担 | 可负担 |

---

## 8. §八 逐项回答

### 8.1 当前代码结构理解
见 §1。关键：**5ch 单 stem 早期融合**；IR 已做 percentile 拉伸；Depth 已做线性截断；**CLAHE 在 5ch 路径被显式关闭**。

### 8.2–8.5 Experiment A/B/C/D 是否可实施
- **A ✅ 可实施**（pre-flight 全通过，0 源码改动）
- **B ⚠️ 技术上可实施但不应现在做**（5 源码 + 3 处静默陷阱 + 两代数据语义冲突）
- **C ✅ 可实施**（1 源码；推荐 `minmax`）
- **D ✅ 可实施**（1 源码；低风险）

### 8.6 每个实验需要改哪些文件

| 实验 | 文件 |
|---|---|
| **A** | 无源码改动。`configs/yolo11m_rgbid_p2_probe.yaml` + `configs/train_rgbid_p2_probe.yaml` + `diagnostic/p2_probe/p2_probe_init_scratch.pt`（均已就绪） |
| **B** | `augment.py`(×2 处) + `scripts/predict.py` + `base.py` + `models/yolo/detect/train.py` + 2 个新配置 |
| **C** | `base.py::load_and_preprocess_image` 的 **depth 处理 3 行**（L336-338），替换为 `minmax` |
| **D** | `base.py::load_and_preprocess_image` 的 **IR 拉伸 3 行**（L327-329），替换为 `CLAHE` |

### 8.7 是否存在隐藏风险
**是，两条最重要的**：
1. **§3.3 的 6ch 静默陷阱** —— 改 `ch: 6` 会同时触发 `Format` 通道反转、`Albumentations4C`、`RandomHSV6C`（后者会跨模态污染）。**这是本次审计最有价值的发现。**
2. **两代数据不一致** —— depth 无效像素占比 31.5% vs 74.3%；8-bit 代已丢失 mm 信息。任何 depth 侧的改动（B 与 C）都会在两代数据上产生**不同语义**。

### 8.8 哪些实验应该暂缓
- **B（Depth Validity Mask）暂缓** —— 除非截止前有充裕时间且愿意承担 5 文件改动 + 3 处静默陷阱的风险
- **C 的 `inverse` / `log` 暂缓** —— 有除零/−inf 奇点，且在两代数据上语义不一致；若要做 C，**只做 `minmax`**

### 8.9 如果只剩有限训练时间，优先验证哪几个方向
**只剩 46 h，一次 300-epoch 训练 ≈11.5 h → 最多 2 个实验（含评测与决策时间）。**

**建议顺序**：
1. **P0 = Experiment A（P2）** —— 唯一与瓶颈机制对口、0 源码改动、pre-flight 已全通过
2. **P1 = 视 A 的结果二选一**：若 A 有效 → 沿 P2 加码；若 A 无效 → 做 **D 或 C(minmax)** 中**改动最小、最不可能出错**的一个

**不推荐在截止前做 B。**

### 8.10 每个实验如何做到「只改一个变量」

| 实验 | 「只改一个变量」的具体做法 |
|---|---|
| **A** | **已经做到**：唯一变量 = 有无 P2 检测尺度。layer 0–22 逐位不变；init 逐层等价（528/528 逐位一致）；训练配方仅 `name`/`patience`(早停开关)/`pretrained`(同源重索引) 三处，且后者是使 init **更一致**、不是新变量 |
| **C** | 只改 `base.py` 的 depth 分支**一处**（L336-338）；**输出仍是 1ch uint8** → `cv2.merge` 仍 5ch → stem/预训练/模型结构/其它训练参数全部不动。**注意**：必须**替换**现有归一化（不是叠加），否则会重复归一化 |
| **D** | 只改 `base.py` 的 IR 分支**一处**（L327-329）；同样输出仍是 1ch uint8 → 其余全不动。**注意**：这是 `percentile → CLAHE` 的**替换**，不是 `raw → CLAHE` |
| **B** | **无法只改一处** —— 必须先完成 5 处源码改动**并逐一验证**（尤其是 `RandomHSV6C` 必须被禁用），此后"加 mask"才是单变量。**建议改完后先跑单 epoch 冒烟**确认 6ch 数据通路的通道序与增强分支，再全量训练 |

---

## 9. 我的一句话建议

> **截止前只做 Experiment A（P2），其余三个暂缓。**
> A 是唯一「与瓶颈机制对口 + 0 源码改动 + pre-flight 已证单变量」的候选；
> B 看似最有"新意"，但这次审计发现**改 `ch: 6` 会静默启用一个会跨模态污染 IR/Depth 的增强分支**，且两代数据的 mask 语义不一致 —— **在只剩 46 h 的情况下不值得冒险**。
> 若 A 的官方 mAP50-95 ≤ 0.50928，则按预案**冻结 0.53180 结案**，把剩余时间用于提交物核对。

---

## 10. 本轮审计的合规声明

| 项 | 状态 |
|---|---|
| 修改代码 / 配置 / 模型 / 数据 / 权重 | ❌ **全部未改** |
| 训练 / 提交 | ❌ 未执行 |
| 执行的命令 | 仅静态读取（读 YAML/PY、grep、逐张量数值比对、CPU 上的 CLAHE 计时 与 label/depth 统计） |
| 正式产物 | ✅ `best.pt` `f9dddbfa…` / `last.pt` `cfa00836…` / `submission.zip` `0c550773…` / 两个 baseline 配置 **SHA256 全部未变** |
| 新增 | `reports/experiment_feasibility_audit.md`（本报告） |
| 网络 | ⚠️ `WebFetch(github.com)` **被网络策略拦截**；`WebSearch` 返回了该仓库的架构摘要（已在 §6 标注为「外部摘要」，非逐字代码） |

**Sources**（CityViMD-Net 信息来自以下搜索结果的摘要）：
- [Dessert613/CityViMD-Net — GitHub](https://github.com/Dessert613/CityViMD-Net)
- [CityViMD-Net/docs/architecture.md](https://github.com/Dessert613/CityViMD-Net/blob/main/docs/architecture.md)
