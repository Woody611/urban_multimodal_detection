# Depth↔RGB Spatial Alignment Infrastructure Audit

**日期**：2026-09-29
**性质**：**READ-ONLY INFRASTRUCTURE AUDIT**。0 training / 0 backward / 0 optimizer.step /
0 checkpoint / 0 改任何既有 source·config·data·label·depth 原文件 / 0 重跑 forward /
0 重生成 prediction / 0 重定义 G1-G4。新增文件仅位于 `diagnostic/depth_alignment_audit/`。

**唯一问题**：depth 与 visible、GT label 处于什么坐标系？训练 pipeline 在哪一步做了
resize/crop/registration？上一轮 **STOP B** 是「数据未配准」还是「审计坐标链路用错」？

---

## FINAL VERDICT

```text
FINAL VERDICT: B — visible / depth raw 文件本身没有 pixel registration
STOP_B_STATUS: CONFIRMED   （且已归因：来源在数据侧，不是审计坐标链路）
```

**（附 E 式保留）** 严格按协议 §17「任一关键环节无法确定即判 E」：**depth 自身的物理坐标系统
（depth pixel (x,y) 对应哪条 scene ray）无法确定**，且**无法排除源数据在交付前被某个未知
变换预配准过**。若采用该严格读法，裁决应读作 **E**，其中「raw 文件未 pixel register」是**证据
最强的一支**。两种读法给出**同一个可执行结论**（见 §7 Q7）。

**一句话**：训练 pipeline 里**没有任何** modality-specific 几何操作；
`cv2.merge` 那一步**假设**了 raw depth 与 raw visible 共享同一像素网格，而这个假设**实测不成立**。

---

## 0. 七问速答（协议 §16）

| # | 问题 | 回答 |
|---|---|---|
| **Q1** | raw visible 与 raw depth 是否 pixel-registered？ | **否**（见 §5；A=0.1105 / B=0.1304 / Canny Jaccard=0.0232） |
| **Q2** | 是否来自同一 sensor coordinate system？ | **无法确定**；证据与「不同传感器 / 未配准」一致，**不能**据此断言 sensor 型号 |
| **Q3** | 若不同，是否存在 calibration / registration transform？ | **仓库与数据集内 NOT_FOUND**（§4） |
| **Q4** | 训练 pipeline 是否应用了该 transform？ | **没有应用任何 transform**（因为不存在；且 pipeline 内没有任何 modality-specific 几何操作，§3） |
| **Q5** | GT visible 坐标能否无歧义映射到 depth 坐标？ | **GT → visible：可以**（唯一、明确）。**GT → depth：不可以**（缺 depth 的坐标定义） |
| **Q6** | 上一轮「80.2% small GT fully-zero」是否仍有物理意义？ | **作为「模型 5ch 输入的第 5 通道在该位置为 0」有意义**（pipeline 确实按同一网格拼接）；**但不能**解释为「目标在场景中没有深度回波」 |
| **Q7** | 下一步 | **FIX DATA/REGISTRATION INFRASTRUCTURE** |

---

## 1. Provenance / 合规（协议 §19）

```text
MODIFIED_EXISTING_FILES = 0
TRAINING       = NO
BACKWARD       = NO
OPTIMIZER_STEP = NO
MODEL_FORWARD  = NO
PREDICTION_FILES_MODIFIED = NO
EXISTING_FILES_CHANGED = 0
```

**FILES_READ**（只读）
`ultralytics/data/base.py`（`load_and_preprocess_image` / `_resize_images_3` / `_merge_channels_rgbid` /
`load_image`）、`ultralytics/data/augment.py`（几何链，上一轮已审计）、`scripts/train.py`、
`configs/train_rgbid_sepstem_clahe.yaml`、`configs/yolo11m_sepstem.yaml`、
`data/raw/**`（header 只读）、`docs/`、`README.md`、赛题 PDF（原始字节扫描）、
`diagnostic/sepstem_clahe/best_full/results`（未修改）。

**FILES_CREATED**（全部在 `diagnostic/depth_alignment_audit/`）

| 文件 | SHA256（前 16） |
|---|---|
| `_analyze.py` | `10e44ce99de41b19` |
| `_confound_check.py` | `877d2d76032fb98c` |
| `_tables.json` | `ed3024ddb049e440` |
| `_confound.json` | `eda6626be4f7f1dc` |
| `ALIGNMENT.log` | `72b073e46e2a0fb0` |
| `CONFOUND.log` | `e28afba793d13eb1` |
| `REPORT.md` | `05d58861fda2bcc0` |

**既有文件复检**：`EXISTING_FILES_CHANGED = 0`
（`ultralytics/data/base.py`、`augment.py`、`tal.py`、两份 config、D′ `best.pt`、
`diagnostic/sepstem_clahe/best_full/results/00000045.txt` 的 SHA256 **全部未变**）。
`git status --short` 的 10 个 `M` 项与本轮开始前**完全相同**。

---

## 2. 真实数据语义复核（协议 §15）

`base.py:344-347`（RGBID 分支，**实读**）：

```python
if im_depth.dtype != np.uint8:
    im_depth = im_depth.astype(np.float32)
    im_depth[im_depth < 300] = 0.0                        # 无效深度(<300mm) 置 0
    im_depth = np.clip(im_depth / 19999.0 * 255.0, 0, 255).astype(np.uint8)
```

**只读实测（raw train HR，20 张）**：

| 量 | 中位 |
|---|---:|
| `== 0` | **29.95%** |
| `< 300` | **29.95%**（与 `==0` **完全相同**） |
| `>= 300` | 70.05% |
| `== 19999` | **0.00%** |

⇒ 三个结论：
1. **raw 中不存在 1..299 的值** ⇒ `im_depth[im_depth<300]=0` 是**惰性防御代码**，不是有效阈值
   ⇒ 上一轮用 `d == 0` 当 invalid 掩码 **是正确的**（当时的担心是多余的）。
2. `19999` 是**归一化分母**，几乎从不作为实际像素值出现 ⇒ **不是饱和截断**。
3. 无 min-max / 无 NaN·Inf 处理；normalization 发生在 RGBID merge **之前**（对 depth 单通道做），
   然后才 `cv2.merge`。**训练时 depth 没有被当作普通图像 resize**（见 §3）。

---

## 3. Coordinate Chain（从代码逐行建立，协议 §3）

### 3.1 图像链

```
raw visible  (H0×W0×3, BGR,  uint8)   ─┐
raw infrared (H0×W0,     mono, uint8/16)─┼─ cv2.merge  ← base.py:430-433
raw depth    (H0×W0,     mono, uint16) ─┘   [纯通道拼接；**零几何操作**]
                                            ↑ 前置条件：三者 H,W **完全相同**
                                            （cv2.merge 对尺寸不一致会直接抛错）
        ↓  base.py:340 _resize_images_3
           [**仅当** h_vis!=h_inf or w_vis!=w_inf or h_vis!=h_dep or w_vis!=w_dep 才 resize；
            实测三者恒同（§5）⇒ 该函数**恒为 no-op**]
        ↓  base.py:350 _merge_channels_rgbid → 5ch [B,G,R,IR,D]
        ↓  base.py:435 load_image：对**整个 5ch 数组一次** cv2.resize（长边→1280，等比）
        ↓  augment.py：Mosaic / RandomPerspective / LetterBox 作用于**同一个 5ch 数组**
        ↓  Format → model input
```

**关键**：从 `cv2.merge` 之后的每一步都作用在**同一个 5 通道数组**上
⇒ **pipeline 内部不存在任何 modality-specific 的 resize / crop / pad / registration**。
唯一未经证明的假设是 `cv2.merge` 那一步：**raw depth 与 raw visible 共享同一像素网格**。

### 3.2 label 链

```
YOLO normalized (cls cx cy w h ∈[0,1])   ← labels/val/visible/<stem>.txt
        ↓ × (W0, H0)   （native visible 像素尺寸）
native visible pixel coordinates
        ↓ 与图像**同一套** transforms（Mosaic/RandomPerspective/LetterBox）
model input coordinates
```

**结论**：`GT → visible` 的映射**唯一且明确**；`GT → depth` 的映射
**完全依赖 §3.1 那个未证明的假设**。

### 3.3 协议 §9 专项：是否存在「visible 做 letterbox、depth 做直接 resize」这类几何不一致？

**不存在。** 逐条证据：
- `_resize_images_3` 对三者用**各自的** `r = imgsz / max(h,w)`，但**只在形状不同时**才动手；
  实测三者恒同 ⇒ **从不执行**。
- `load_image` 的 resize 作用在**合并后的 5ch 数组**上 ⇒ 五通道共享同一几何。
- LetterBox 属 `augment.py` 的 `RandomPerspective.pre_transform`，作用在 labels dict 的 `img`（5ch 数组）上。
- 无 EXIF orientation（§5）⇒ 无旋转歧义。

---

## 4. Registration / Calibration 元数据检索（协议 §6）

| 检索位置 | 结果 |
|---|---|
| `docs/`、`README.md`、`configs/`（关键词 intrinsic/extrinsic/calibrat/registrat/homograph/camera_matrix/内参/外参/标定/配准） | **0 命中**（两条命中是 CBAM/mosaic 注释，无关） |
| `data/raw/` 下非图像文件 | 仅 `data/raw/test初赛/{visible,infrared,depth}.zip`；**无任何元数据文件** |
| 仓库内 `*calib*` / `*intrinsic*` / `*register*` 文件名 | **0** |
| 全仓 grep "alignment/registration" | 命中**全部是本次/历次 diagnostic 自身**（`best_align` 等），非元数据 |
| 赛题 PDF 原始字节 ASCII 扫描 | `1920`=0、`1080`=0、`depth`=0、`intrinsic`=0、`calib`=0、`regist`=0、`sensor`=0 |

```text
REGISTRATION_METADATA = NOT_FOUND
RGB_DEPTH_CALIBRATION = UNKNOWN
```

**未猜测、未用 correlation 反推 calibration**（协议 §10/§18）。

---

## 5. Raw file level 对应表（协议 §5，**全量 2400 stem**，header-only 只读）

| split | n_stem | 缺失 stem | 三方 (W,H) 组合 | 三方同尺寸 |
|---|---:|---|---|---|
| train | **2000** | visible=0, ir=0, depth=0 | `(1920,1080)×3` n=1851；`(640,360)×3` n=149 | **全部 True** |
| val | **400** | visible=0, ir=0, depth=0 | `(1920,1080)×3` n=373；`(640,360)×3` n=27 | **全部 True** |

| 模态 | 格式 / 模式 |
|---|---|
| visible | `PNG/RGB` 1851 + `JPEG/RGB` 149 |
| infrared | `PNG/RGB` 1851 + `JPEG/RGB` 149 |
| **depth** | **`PNG/I;16`** 1851（16-bit 灰度）+ `JPEG/RGB` 149 |

**EXIF orientation：前 200 样本 × 3 模态全为 `None`** ⇒ 无旋转歧义。

⇒ 三条硬事实：
1. **1:1:1 stem 配对，0 缺失**（三模态按 stem 严格对应）。
2. **只有 2 种分辨率层**，且**每一种层内三模态尺寸完全相同**
   ⇒ 不存在「visible 1920×1080 而 depth 640×360」这类跨层错配。
3. **没有 registration 信息**——只有「尺寸相同」。

> ⚠ 协议 §6：**尺寸相同 ≠ 同坐标系**。上面三条只证明 **raster size 相同**。

---

## 6. 空间关系实证（协议 §7/§8，raw 层，n=40 HR）

预注册 **primary metric = Sobel 幅值 Pearson r（8×8 下采样）**；
secondary = Canny 边缘 Jaccard。**不以 secondary 改判 primary。**

| 量 | 中位 | 参照 |
|---|---:|---|
| **A** 全像素梯度 r @(0,0) | **0.1105** | 配准良好的 RGB-D 通常 **0.3–0.6** |
| 最佳平移后 A | **0.1659**（提升 +0.0553） | |
| 最佳位移 中位 (dy,dx) | **(0, −24)** | |
| **最佳位移标准差 (dy,dx)** | **(12.5, 20.5) px** | **高度分散 ⇒ 单一全局平移无法对齐** |
| **secondary: Canny Jaccard** | **0.0232** | 配准良好通常 0.1–0.3 |

### 6.1 排除「低相关只是度量假象」（`_confound_check.py`）

raw depth 的梯度幅值被 **validity 边界**（0↔非零跳变）主导，而那些跳变**未必**对应 visible 边缘。
故低相关**可能**是度量假象。**预注册三量直接检验**：

| 量 | 中位 | 若真配准的预期 |
|---|---:|---|
| **A** 全像素梯度 r | 0.1105 | — |
| **B** **valid-only** 梯度 r（排除 validity 边界） | **0.1304** | **应显著 > 0.3** |
| **C** validity mask vs visible 亮度 r | −0.0559 | 无效区应对应场景特征（天空/过曝）|

**B − A = +0.0198** ⇒ 排除 validity 边界后**几乎不变**
⇒ **低相关不是度量假象**，而是真实的未配准。

---

## 7. STOP B 来源分类（协议 §11）

| CASE | 判定 | 证据 |
|---|---|---|
| **A** 审计脚本用了错误坐标转换 | **排除** | 上一轮审计的映射 = **归一化 GT × raw 尺寸 → 直接索引 raw depth**，与 pipeline 的 `cv2.merge` **逐字一致**（都是「raw depth 与 raw visible 同网格」这一假设）。且**无替代映射可用**：最佳平移高度分散（std 20.5px），平移也救不回来（A→0.166，仍远低于 0.3） |
| **B** raw 文件本身没有 pixel registration | **◀ 主判** | A=0.1105 / B=0.1304 / Canny=0.0232 / 最佳平移不稳定；且 **REGISTRATION_METADATA = NOT_FOUND** |
| **C** 有 calibration 但 pipeline 没做 | **排除** | **不存在任何 calibration / registration 元数据**可供应用（§4）。pipeline 内也确实没有 modality-specific 几何操作（§3.1）——即「该做的事」本身不存在定义 |
| **D** 有 registration、pipeline 也正确，只是 correlation metric 不合适 | **排除** | §6.1 的 B（valid-only）**直接**反驳：换掉度量后相关仍 ≈0.13。这不是指标选择问题 |
| **E** 信息不足 | **部分适用** | 适用于「depth 自身的物理坐标系统是什么 / 源数据是否被未知变换预配准过」——**无法确定** |

---

## 8. Evidence 汇总（协议 §12 要求的 OLD/CORRECT/EVIDENCE）

```text
OLD_MAPPING:
    depth_pixel(x, y) := 与 visible_pixel(x, y) 同一位置的深度
    （即：raw depth 与 raw visible 共享像素网格；等价于 pipeline 的 cv2.merge 假设）

CORRECT_MAPPING:
    ???  —— 本审计**无法给出**正确的对应关系。
    证据只支持「identity 映射不成立」，不支持任何具体的替代映射
    （无 calibration、无 registration 元数据、无单一平移能对齐）。
    按协议 §6/§10：**不使用 correlation 反推 calibration**，故不提供猜测性映射。

EVIDENCE:
    E1 全量 2400 stem：1:1:1 配对、仅 2 种分辨率层、层内三模态尺寸恒同、EXIF 全 None   (§5)
    E2 代码链：cv2.merge 零几何；_resize_images_3 恒 no-op；load_image 对 5ch 数组一次 resize (§3)
    E3 raw 层 A=0.1105 / best-shift=0.1659(std 20.5px) / Canny Jaccard=0.0232            (§6)
    E4 confound check：valid-only 梯度 r=0.1304（B−A=+0.0198）⇒ 非度量假象            (§6.1)
    E5 registration/calibration 元数据全域 NOT_FOUND                                    (§4)
    E6 depth 语义复核：0=invalid、无 1..299、19999 几乎不出现 ⇒ 上一轮 invalid 掩码正确  (§2)

PREVIOUS_DEPTH_AUDIT_INVALID = YES
    失效范围：上一轮**所有 target-level 的物理解释**（例如「80.2% small GT patch 完全 zero」
    被读作「目标区域无深度回波」）**不成立**——因为 depth 与 visible 不共享坐标系。
    仍然成立的部分：**(a)** 「模型实际吃到的 5ch 输入中，第 5 通道在该位置为 0」这一陈述
    （pipeline 确实按 identity 网格拼接）；**(b)** 「hard miss 与 successful 在该量上不可区分」
    这一**相对**结论（两组用同一映射，是 apples-to-apples）。
```

**未重跑上一轮 depth reliability audit**（协议 §12/§14）。

---

## 9. 严格成功标准核查（协议 §17）

| 环节 | 状态 |
|---|---|
| raw file correspondence | **明确** ✅（§5，2400 stem 全量） |
| label coordinate | **明确** ✅（§3.2） |
| visible coordinate | **明确** ✅ |
| **depth coordinate** | **不明确** ❌（仅知 raster 尺寸；物理坐标系统未知） |
| resize / crop / pad | **明确** ✅（pipeline 内三者皆无 modality-specific 操作） |
| **registration transform** | **不存在于 pipeline** ✅；**源数据是否预配准过：未知** ❌ |
| calibration existence | **明确为 NOT_FOUND** ✅ |
| training pipeline 是否使用 | **明确：没有** ✅ |
| STOP B 来源 | **明确：数据侧（B）** ✅ |

⇒ **COORDINATE_CHAIN = PARTIALLY_EXPLAINED**（2 项未定）。
⇒ 按协议 §17 严格读法 **VERDICT = E**；按 §11 证据归类 **CASE B**。
**两者给出同一可执行结论。**

---

## 10. 协议 §18：明确不做哪些推论

本报告**未**使用、也不认可以下任一推论：
`high correlation = alignment proven` / `same resolution = alignment proven` /
`same filename = alignment proven` / `same zero pattern = same coordinate system` /
`Depth-only low AP = Depth useless` / `GT box zero depth = target has no depth`。

特别地：**「尺寸相同」只说明 raster size 相同**；本报告的全部结论建立在
§6/§6.1 的**内容级**证据（A/B/C 三量与 Canny Jaccard）之上，而非尺寸或命名。

---

## 11. Limitations

1. **depth 的物理坐标系统不可确定**：无 calibration、无 registration 元数据、
   且禁止用 correlation 反推 ⇒ 只能说「identity 映射不成立」，说不出**正确的**映射是什么。
2. **无法排除源数据预配准**：若提供方在交付前做过某个未知变换（不同 FOV 的 homography 等），
   本审计无法识别；但**可以**断定交付后的文件**不满足 identity 对应**。
3. 相关性检验在 **HR train 层 40 张**上做（与上一轮同层、同协议）；LR 层（176 stem）未单独检验
   ——其 depth 是 **JPEG/RGB 3 通道**，语义另需确认。
4. 未检验 depth 与 **IR** 的对应关系（本轮范围外）。
5. `_resize_images_3` 在「三模态形状不同」时**各自独立** resize，存在潜在几何不一致；
   本数据集**恰好不触发**该分支，但**代码路径本身是有风险的**（若将来数据层混合分辨率，
   该函数不会报错，只会静默产生 modality-specific 缩放）。**记录为基础设施风险。**

---

## 12. Recommendation

```text
NEXT ACTION = FIX DATA/REGISTRATION INFRASTRUCTURE
```

**Depth 路线状态**：

```text
BLOCKED BY REGISTRATION INFRASTRUCTURE
```

- **不要**继续做 depth validity mask / depth-only training / depth reliability attribution /
  depth target-level analysis —— 因为 **target-level depth validity 目前没有定义良好的物理意义**（协议 §13）。
- **不要**创建任何 config / 脚本改动 / checkpoint / 实验 run。
- 若要解锁，需要的是**基础设施任务**（非本审计范围，需另行批准），例如取得 RGB↔Depth 的
  intrinsic/extrinsic 或确认源数据的配准状态——**本审计不提出具体方案、不设计、不执行**。

**DEPTH_ALIGNMENT_CONFIRMED = NO**（协议 §14 的条件未满足，故**不**进入「重跑 depth reliability audit」）。

---

## 13. 产物

| 文件 | 说明 |
|---|---|
| `_analyze.py` | §5 全量 header 对应表 + §2 depth 语义 + §6 raw 层对齐检验 |
| `_confound_check.py` | §6.1 排除「低相关是 validity 边界度量假象」 |
| `_tables.json` | 结构化：raw_correspondence / alignment / depth_semantics / registration_metadata |
| `_confound.json` | A/B/C 三量 + 逐图明细 |
| `ALIGNMENT.log` / `CONFOUND.log` | 运行日志 |

（相对协议 §12 的文件清单多了 `_confound_check.py` / `_confound.json` / `CONFOUND.log`
与 `_tables.json` —— §6.1 的 confound 检验是**判定 CASE B vs CASE D 的决定性证据**，
必须留成可复现产物，不能只在报告里叙述。）
