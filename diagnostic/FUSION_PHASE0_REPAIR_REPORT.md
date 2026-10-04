# Fusion Phase 0 — Infrastructure Repair Report

**日期** 2026-10-03 · **性质** Infrastructure Repair + Zero-GPU Validation
**基线** `HEAD = 5f1a8d5`（branch `yyy`）· **未 commit**
**本轮不做**：GPU / forward / 正式 inference / 训练 / 优化策略 / loss·optimizer·architecture 改动（除下方明确列出的 infrastructure fix）

---

## Executive Verdict

```text
READY
```

5 个基础设施缺陷全部修复并验证；D′ 语义零改动（两个 config SHA 逐位不变、RGBID/sepstem 预处理与 remap 逐位不变）；一个越界的"改进"（歧义布局守卫）在回归扫描中被发现会打挂 **15 个既有 RGBD mid-fusion 模型**，**已回退**。全部 CPU 矩阵 PASS（29 + 20 + 8 + 7 + 12 项）。

**但报告里最重要的一句话是 §G**：`ultralytics/data/base.py` 与 `ultralytics/data/loaders.py` **被 `.gitignore:11` 忽略、不在 git 里**——本轮最关键的三个修复（IR 分派 / Depth·Infrared·IRD 图片分支 / RGBD pair 统一）**对 `git diff` 完全不可见**，只能靠 SHA-256 追踪。

---

## A. Infrastructure changes

### A-1 · ISSUE A — IR preprocessing dispatch 错误

```
Issue      : use_simotm='Infrared'（IR 单模态，训练与推理两侧）硬编码 1%/99% percentile，
             完全不读 ir_encoding ⇒ `ir_encoding: clahe` 被静默忽略，M2a 实际跑成 M2b。
Root cause : IR 对比度处理被复制成 3 份独立实现，只有 RGBID 分支实现了 ir_encoding 分派；
             scripts/train.py 的硬防呆（:638）只拦 use_simotm=="RGBID"。
Files      : ultralytics/data/base.py        （'Infrared' 分支 + 新增 apply_ir_encoding）
             ultralytics/data/loaders.py     （'Infrared' 视频路径 + 图片路径 + 新增 RGBIR）
Exact      : 1) 新增模块级单实现 apply_ir_encoding(im, ir_encoding)，支持
                "percentile"（逐字复刻旧式）/ "clahe" / "raw"，未知取值 **raise ValueError**
                （不再静默回落 percentile）；
             2) base.py 的 'Infrared'、'RGBID' 两个分支改为调用它；
             3) loaders.py 的 'Infrared'（视频）与新增 'Infrared'（图片）改为调用它
                （改用类里**早已存在**却从未被用的 self.ir_encoding）。
Risk       : 低。对 "clahe"/"percentile" 输出逐位相同（对 8 张真实 IR 图对拍旧实现验证）；
             唯一行为变化是"未知取值"由静默→抛错。现有配置中 ir_encoding 只出现 clahe/percentile/缺省。
验证       : loader_tests C.1/C.2/C.3/C.4、repair_tests T1.*/T2.*
```

### A-2 · ISSUE A′（新发现，比 ISSUE A 更严重）— 推理图片路径缺 3 个模态分支

```
Issue      : loaders.py 的 **图片路径**（predict_rect.py 走的那条）此前**没有** 'Depth' 与
             'Infrared' 分支 ⇒ 二者落到最后的 else: imread(path) ⇒ **读到 visible 原图**。
             即 M2（IR）/M3（Depth）即使训练正确，**推理时会用 RGB 图评测**，且无任何报错。
             视频路径（同文件 ~526）一直有这两个分支 ⇒ 两条路径此前不一致。
             （'IRD' 也不存在。）
Root cause : 图片路径与视频路径是两份独立的分派表，长期漂移；无人维护图片侧。
Files      : ultralytics/data/loaders.py
Exact      : 在图片路径新增 'Depth' / 'Infrared' / 'IRD' 三个分支（IR 侧走 apply_ir_encoding）。
Risk       : 低（纯新增分支；此前这些 mode 在图片路径上是"读错图"，没有正确行为可破坏）。
验证       : loader_tests B.* 与 T3.5（显式定位 image 代码块并断言三者存在）
```

### A-3 · ISSUE B — 4ch / 2ch 预训练迁移缺失

```
Issue      : _transfer_rgb_pretrained 的分派写死 `in_channels == 5`；4ch/2ch 单 stem 不匹配任何
             分支 ⇒ 落到 `if not any(in_channels == 1): return 0` ⇒ **无日志、无 raise**。
             而 model.load() 已因 shape 不符丢掉 model.0.conv.weight
             ⇒ 全文只有那 1 个张量是 kaiming 随机初始化，其余 642 个张量都是预训练。
Root cause : 原分支只为 RGBID(5ch) 服务；条件用字面量 5 而不是"非 3 的输入 stem"。
Files      : ultralytics/models/yolo/detect/train.py
Exact      : 1) multi-ch 分支：条件改为"**第一个 Conv** 的 in_channels > 3"，
                ch[0:3] = 预训练 RGB，ch[3:] = mean(W_R,W_G,W_B)（与 RGBD/sepstem aux-stem 同一约定）；
             2) 新增 narrow-ch 分支：in_channels ∈ {1,2} 且无 3ch stem
                ⇒ 全通道 = mean(W_R,W_G,W_B)（与 _remap_separate_stem 的 aux 处理一致）；
             3) 两个分支在源张量缺失/形状不符时 **raise RuntimeError**，不再静默；
             4) 每个分支打印 in/out、来源通道区间、mean/std/finite、newly_init/transferred 计数。
Risk       : 中→低（已回归）。**首次实现误写成"任意 in_channels > 3 的 Conv"**，
             会在 3ch 模型上命中 model.1(64ch) 并 raise；已收紧为"只看第一个 Conv"。
验证       : repair_tests T4.*（4ch RGB==stock 且 aux 非零；2ch 全通道==mean(RGB)）
```

### A-4 · ISSUE C — IR+Depth 数据管线缺失

```
Issue      : base.py / loaders.py / dataset.py 的 merge 只有 (RGB+IR)/(RGB+IR+RGB)/(RGB+IR+D)/(RGB+D)，
             **没有 (IR+D)**；M6 无法构造。
Root cause : 新增模态组合时未同步三处 loader。
Files      : ultralytics/data/base.py（'IRD' 分支 + _merge_channels_ir_depth）
             ultralytics/data/loaders.py（图片路径 'IRD' 分支）
             scripts/train.py（'IRD' 加入 rgbid_split 分派组）
Exact      : 通道序**固定 [IR, D]**（cv2.merge((im_infrared, im_depth))，不复制不交换）；
             IR 走 apply_ir_encoding，Depth 走与 RGBID 逐字相同的 /19999×255 映射；
             复用 rgbid_split_train（该 split 三模态目录齐全）。
Risk       : 低（纯新增）。
验证       : repair_tests T3.1/T3.2（合成 IR=11 / D=22 ⇒ ch0==11, ch1==22）、loader_tests A.IRD/B.IRD
```

### A-5 · ISSUE D — augmentation 不公平

```
Issue      : 光度增强按 **通道数** 硬分派：3ch → Albumentations(p=1.0)；4ch → Albumentations4C(p=1.0)；
             5ch(D′) → Albumentations(p=0)。⇒ D′ 无光度增强，3ch/4ch 单模态基线有
             (Blur/MedianBlur/ToGray/CLAHE 各 p=.01)，两者在 augmentation 层不可比。
Root cause : 分派是隐式写在 build_transforms 里的常量，没有配置出口。
Files      : ultralytics/data/augment.py、ultralytics/cfg/default.yaml、scripts/train.py
Exact      : 新增 **opt-in** 键 `albumentations_p`（default.yaml 默认 null）。
             null ⇒ 完全保持既有行为（逐位）；非 null ⇒ 强制 alb = Albumentations4C(p)（4ch）
             或 Albumentations(p)（其余）。7 个 modality 配置写 `aug.albumentations_p: 0.0` 与 D′ 对齐。
Risk       : 低。null 默认使所有既有配置（含 D′）行为**逐位不变**。
             ⚠ 若云环境未装 albumentations 包，该差异本就是 no-op（本机 ModuleNotFoundError）。
验证       : repair_tests T6.*；contract gate 强制 3ch/4ch 必须写 0.0
```

### A-6 · ISSUE E — model scale 显式性

```
Issue      : scale 只能由 **文件名正则** 决定；yaml 内写 `scale:` 会被 yaml_model_load 覆盖。
Root cause : tasks.py:1317 `d["scale"] = guess_model_scale(path)` 无条件覆写。
Files      : 无（**不能从 yaml 修**）—— 改用命名纪律 + 合同门强制。
Exact      : 1) 所有 modality 模型 yaml 命名为 `yolo11m_modality{2,3,4}ch.yaml`（含 scale 字符）；
             2) 合同门断言"文件名含 scale 字符且 == 'm' 且 scales.m == [0.50,1.00,512]"。
Risk       : ——
验证       : repair_tests T5.*（5 个 yaml 的 scale/depth/width/max_ch 全部 = D′ 的 m/[0.5,1.0,512]）
             ⚠ 附带确认既有 `configs/yolo11_visible.yaml` 的 guess_model_scale 返回 ''（nano 回退陷阱）
```

### A-7 · 额外修复 — RGBD 的 depth pair 在 base.py / loaders.py 不一致

```
Issue      : base.py 的 'RGBD' 用 `pairs_ir` 做 depth 路径替换，loaders.py 用 `pairs_depth`。
             若配置写 pairs_rgb_ir=["visible","infrared"] ⇒ **训练把红外当深度读**。
Root cause : 两处实现各自演化。
Files      : ultralytics/data/base.py
Exact      : base.py 改用 pairs_depth（= loaders.py 的行为）。
Risk       : 无。对既有 RGBD 配置是 no-op（train_multimodal_rgbd*.yaml 的两个 pair 都是 ["visible","depth"]）。
验证       : loader_tests A.RGBD（修复前该用例因读 infrared 而 FileNotFoundError）
```

### A-8 · 新增 — `RGBIR` 模式（为 M4 提供正确的 IR 编码）

```
Issue      : 既有 'RGBT'(4ch) 的 IR 分支**不做任何对比度处理**（raw），无法表达 D′ 的 IR-CLAHE。
             M4 若用 RGBT 则"移除 Depth"与"IR 编码改变"混在一起。
Root cause : RGBT 是历史模式，其 IR 预处理从未接入 ir_encoding。
Files      : ultralytics/data/base.py、ultralytics/data/loaders.py、scripts/train.py
Exact      : 新增 'RGBIR'（[B,G,R,IR]，IR 走 apply_ir_encoding），**不改动既有 RGBT**
             （避免静默改变历史 RGBT 实验语义）。M4 配置改用 RGBIR。
Risk       : 低（纯新增；RGBT 未动）。
验证       : 合同门 M4 行、loader_tests A.RGBIR/B.RGBIR
```

### A-9 · ★ 越界改动 — 已回退

```
Issue      : 我在 _transfer_rgb_pretrained 里加过一条"1 个 3ch stem + 1 个 1ch stem ⇒ raise"的歧义守卫，
             本意是防止静默误映射。
发现       : 全 configs 回归扫描显示该布局是 **15 个既有 RGBD mid-fusion 模型**
             （yolo11*_midfusion_rgbd_*.yaml、yolo11_rgbt.yaml）的**合法**布局，
             它们本就依赖下一个分支的 concat_res 硬编码索引表。守卫会让它们全部失败。
处置       : **已回退**，改为在源码里留下注释记录残留风险。
理由       : 超出 infrastructure scope（brief §11/§12 的 STOP 条件）。
回归       : 回退后 18 个含 1ch stem 的模型全部 OK；仅剩 2 个 **pre-existing** 失败
             （yolo11_latefusion / yolo11_midfusion，在 sepstem-Concat 校验处抛出，与本轮无关）。
```

---

## B. CPU validation matrix（brief §9）

`✅` = 有可执行的自动断言并通过 · `n/a` = 该模态不涉及

| Test | M1 | M2a | M2b | M3 | M4 | M5 | M6 | M7(D′) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| YAML parse | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Model construction（无 forward） | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Explicit YOLO11m（scale=m, d/w/mx 同 D′） | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Correct input channels（config==yaml==模式） | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Pretrained transfer（非静默） | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| IR preprocessing | n/a | ✅ | ✅ | n/a | ✅ | n/a | ✅ | ✅ |
| Depth preprocessing | n/a | n/a | n/a | ✅ | n/a | ✅ | ✅ | ✅ |
| Merge provenance（通道序） | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Augmentation contract | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Train loader（单图） | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Val loader（同上，同一函数） | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Inference loader（LoadImagesAndVideos） | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Evaluator compatibility（100 框上限/官方口径） | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Contract gate | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**汇总**：`fusion_phase0_repair_tests.py` **29/29** · `fusion_phase0_loader_tests.py` **20/20** ·
`validate_modality_contract.py --all` **8/8** · `fusion_phase0_validate.py` **7/7** ·
`fusion_phase0_contract_selftest.py` **12/12**

---

## C. Pretrained transfer audit

`yolo11m.pt`（40,684,120 B · sha 见仓库根）为唯一源；**无任何模态使用随机初始化**。

| Model | Input ch | Stem shape | transferred | newly initialised | `model.0.conv.weight` 来源 | Detect(cv3) |
|---|---:|---|---:|---:|---|---|
| M1/M2a/M2b/M3 | 3 | (64,3,3,3) | **0**（标准 `load()` 已全量） | 0 | `intersect_dicts` **643/649 = 99.1%** | 6 张量未迁移（nc 80→12） |
| M4/M5 | 4 | (64,4,3,3) | **5** | **1 stem**（4 个新输入通道里的 1 个 AUX） | ch[0:3]=预训练 RGB；ch[3]=mean(W_R,W_G,W_B) | 6 张量未迁移 |
| M6 | 2 | (64,2,3,3) | **5** | **1 stem**（2 个通道全部） | 两通道都 = mean(W_R,W_G,W_B) | 6 张量未迁移 |
| **M7 = D′** | 5 | (48,3,3,3) + (8,1,3,3)×2 | **550** | 3 stems | `_remap_separate_stem`（RGB=stock model.0 前缀 `[:48]`；aux=灰度化 stock stem） | 6 张量未迁移 |

**可追踪性**（A-3 要求）：两个新分支都打印 `in=.. out=.. ch[0:3]=pretrained RGB (shape) | ch[3:n]=mean(R,G,B) | mean=.. std=.. finite=True | newly_init_tensors=1 transferred_tensors=4`。
实测（4ch）：`mean=-0.000858 std=0.094826 finite=True`；（2ch）：`mean=-0.000858 std=0.070213 finite=True`。

**公平性结论**：D′ 与 3ch 基线同为全量预训练、同源；两条路径未迁移的**都恰好是 Detect cv3 的 6 个张量** ⇒ 分类头在两边同等重新初始化。
**M4/M5/M6 的 AUX 通道 = `mean(W_R,W_G,W_B)`，与 D′ 的 aux stem 初始化约定一致** ⇒ 不是"随机 vs 预训练"的不公平。

---

## D. Preprocessing audit

| 路径 | 代码位置 | 实际执行 |
|---|---|---|
| **M2a** train/val | `base.py` `'Infrared'` | `uint16→uint8`（如有）→ **`apply_ir_encoding(im, hyp.ir_encoding)`** → `stack×3` |
| **M2a** inference | `loaders.py` 图片路径 `'Infrared'`（**新增**） | 同上（用 `self.ir_encoding`） |
| **M2b** train/val/inference | 同上，`ir_encoding=percentile` | 1%/99% 分位线性拉伸（**与修复前逐位相同**） |
| **D′ (M7)** train/val | `base.py` `'RGBID'` | IR：`apply_ir_encoding(ir, 'clahe')` → CLAHE(2.0,(8,8))；D：`<300→0` → `/19999×255` → uint8；merge `[B,G,R,IR,D]` |
| **D′** inference | `loaders.py` 图片路径 `'RGBID'`（未改动） | 同训练侧 |
| **M4** train/inference | `base.py` / `loaders.py` `'RGBIR'`（新增） | IR 走 `apply_ir_encoding`，`merge [B,G,R,IR]` |
| **M6** train/inference | `'IRD'`（新增） | IR 走 `apply_ir_encoding`；D 同 RGBID；`merge [IR,D]` |
| （既有 `RGBT`） | **未改动** | IR = raw（该分支从不做对比度处理） |

**逐位回归证据**：对 **8 张真实 IR 图**，新实现 vs 旧实现内联复刻 ——
`clahe` 与 `percentile` **均逐位相同**（repair_tests T2.1/T2.2）；
D′ 通道序 `[B,G,R,IR,D]` 由合成张量证明（T2.3，row0 = [10,20,30,44,55]）。

---

## E. Augmentation audit

| Pipeline | Albumentations | HSV | Geometric | Channel-specific |
|---|---|---|---|---|
| **D′ 5ch** | `Albumentations(p=0)` ⇒ **OFF** | `RandomHSV` 但 `shape[-1]!=3` ⇒ **直接 return（OFF）** | Mosaic / RandomPerspective / LetterBox（default） | 无（alb 关，HSV 关） |
| **M1/M2a/M2b/M3 3ch · 修复前** | `Albumentations(p=1.0)` ⇒ **ON**（Blur/MedianBlur/ToGray/CLAHE 各 .01） | `RandomHSV` **ON**（.015/.7/.4） | 同 D′ | 是 ⇒ 与 D′ 不可比 |
| **M1/M2a/M2b/M3 3ch · 修复后** | `albumentations_p: 0.0` ⇒ **OFF** | `hsv_*=0` ⇒ **OFF** | 同 D′ | 已消除 |
| **M4/M5 4ch · 修复前** | `Albumentations4C(p=1.0)` ⇒ **ON** | `RandomHSV4C` **ON** | 同 D′ | 是 |
| **M4/M5 4ch · 修复后** | `0.0` ⇒ **OFF** | `hsv_*=0` ⇒ **OFF** | 同 D′ | 已消除 |
| **M6 2ch** | 默认 `Albumentations(p=1.0)` ⇒ 覆盖为 **OFF** | `RandomHSV` 因 `shape[-1]!=3` 本就 inert | 同 D′ | 已消除 |

**必须保留的 modality 差异（不是不公平）**：通道数本身（3/4/2/5）、IR 的输入表示（单通道复制 vs 拼接）、IR 的对比度编码（clahe vs percentile vs raw，**由 `ir_encoding` 显式选择**）、Depth 的 mm→uint8 映射。
**应统一且已统一**：几何增强全部策略、光度增强开关、Mosaic/close_mosaic、训练预算、evaluator。

⚠ **cloud/local 差异不得假设**：`Albumentations` / `Albumentations4C` 需要安装 `albumentations` 包；
**本机未安装（ModuleNotFoundError）⇒ 本机恒为 no-op**。云上是否安装**未确认**，因此 7 个配置一律显式 `albumentations_p: 0.0`，
使两种环境下的 modality 基线都与 D′ 对齐。**这是本轮唯一无法在本机闭环确认的项**（见 §G）。

---

## F. Contract Gate

`scripts/validate_modality_contract.py` —— 任一字段不符即 `AssertionError`（不是 warning）。

**正常 PASS 示例**：

```text
Modality Experiment Contract Gate   (reference = D′ (SepStem + IR-CLAHE))
  ✅ M1   Gray2BGR  ch=3 [B,G,R]        split=visible_split      stem_in=3 remap=0   intersect=643/649
  ✅ M2a  Infrared  ch=3 [IR,IR,IR]     split=rgbt_split         stem_in=3 remap=0   intersect=643/649
  ✅ M2b  Infrared  ch=3 [IR,IR,IR]     split=rgbt_split         stem_in=3 remap=0   intersect=643/649
  ✅ M3   Depth     ch=3 [D,D,D]        split=depth_split_train  stem_in=3 remap=0   intersect=643/649
  ✅ M4   RGBIR     ch=4 [B,G,R,IR]     split=rgbt_split         stem_in=4 remap=5   intersect=642/649
  ✅ M5   RGBD      ch=4 [B,G,R,D]      split=depth_split_train  stem_in=4 remap=5   intersect=642/649
  ✅ M6   IRD       ch=2 [IR,D]         split=rgbid_split_train  stem_in=2 remap=5   intersect=642/649
  ✅ M7   RGBID     ch=5 [B,G,R,IR,D]   split=rgbid_split_train  stem_in=3 remap=550 intersect=61/661
CONTRACT 8/8 PASS
```

**故意破坏（`diagnostic/fusion_phase0_contract_selftest.py`，对 M1 临时副本注入，用后即删）**：

| 注入的缺陷 | 结果 |
|---|---|
| *(baseline M1 无缺陷)* | **PASS**（对照成立） |
| `epochs 300 → 30` | CAUGHT `epochs: got=30 expected=300` |
| `channels 3 → 4` | CAUGHT `channels(config) vs mode产出: got=4 expected=3` |
| `batch_size 8 → 16` | CAUGHT |
| `lr0 5e-3 → 1e-2` | CAUGHT |
| `seed 42 → 0` | CAUGHT |
| `val_ratio 0.2 → 0.5` | CAUGHT |
| **模型 yaml 去掉 scale 字符** | CAUGHT `model yaml 文件名含 scale 字符`（= nano 静默回退陷阱） |
| **`use_simotm=Depth` 却声明 `ir_encoding=clahe`** | CAUGHT `ir_encoding 是否被该模式消费`（= IR 静默回落陷阱） |
| `aug.hsv_h 0.0 → 0.015` | CAUGHT |
| `aug.albumentations_p 0.0 → 1.0` | CAUGHT |
| `pretrained yolo11m.pt → yolo11n.pt` | CAUGHT |

```text
self-test: 12/12 符合预期（baseline PASS + 全部缺陷 CAUGHT）
临时文件残留: 无
```

---

## G. GPU readiness

```text
READY
```

### 下一阶段建议的 GPU smoke 顺序（**本轮不运行**）

1. **M1 RGB-only** — `yolo11m_modality3ch.yaml` + `train_modality_m1_rgb.yaml`
2. **M3 Depth-only** — 同上 + `train_modality_m3_depth.yaml`
3. **M2a IR-only + CLAHE** — 同上 + `train_modality_m2a_ir_clahe.yaml`
4. （Batch-1 出结果后）M2b / M4 / M5 / M6

**开跑前必须做的两件事**：
1. 在 **1–2 epoch 的 smoke run** 上确认 `Gray2BGR`（M1）与 `Infrared+clahe`（M2a）真的按预期读图 —— 本机已静态+合成验证，但云端实读未验。
2. **确认云环境是否安装 `albumentations`**，并把结果记入实验日志。

### 已知残留（不阻塞，但须记录）

| # | 残留 | 影响 | 状态 |
|---|---|---|---|
| R1 | `use_simotm='RGBT'` 的 IR 是 raw，且不消费 `ir_encoding` | 历史 RGBT 实验的语义；本轮**故意未改** | 已在合同门标 `consumes_ir=False` |
| R2 | `yolo11_latefusion.yaml` / `yolo11_midfusion.yaml` 在 `_remap_separate_stem` 的 Concat 校验处 raise | **pre-existing**，与本轮无关 | 未处理（越界） |
| R3 | "1×3ch + 1×1ch" 布局依赖 RGBD 硬编码索引表 | 新增同类模型时可能静默误映射 | 已在源码留注释（曾尝试加守卫，因打挂 15 个既有模型而回退） |
| R4 | `ultralytics/data/*.py` 被 `.gitignore:11` 忽略 | **本轮最关键的三处修复对 `git diff` 不可见** | 只能用 SHA-256 追踪（见下表） |
| R5 | 云上 `albumentations` 是否安装未确认 | 决定 3ch/4ch 是否真的与 D′ 对齐 | 配置已强制 `albumentations_p: 0.0`，两种环境都对齐 |

---

## H. Git / SHA discipline

```text
HEAD (round start) = 5f1a8d5802aa49adf35e8fa6fb112a1030b3e6d3   branch yyy
未 commit（遵守 brief）
```

### Intended modified files（本轮真正修改）

| File | pre-repair sha256 | post-repair sha256 | tracked? |
|---|---|---|---|
| `ultralytics/data/base.py` | `8bcf8349…871d9` | **`bf380a22…76afe`** | ❌ **gitignored** |
| `ultralytics/data/loaders.py` | `f84bdcf2…c3c1c` | **`56c07c85…c3362`** | ❌ **gitignored** |
| `ultralytics/models/yolo/detect/train.py` | `f4619800…238cb` | `6538ab6f…30ce9` | ✅ |
| `ultralytics/data/augment.py` | `53c75148…531c5` | `abfc90f5…1282b` | ✅（**已 track**，此前被 force-add 过） |
| `ultralytics/cfg/default.yaml` | — | `cf06d3a6…d3d4` | ✅ |
| `scripts/train.py` | — | `fa25bf12…722e` | ✅（+12 行，仅在既有基础上） |
| `scripts/validate_modality_contract.py` | 新增 | `010422b6…0bff` | ✅ untracked |
| `configs/train_modality_m{1,2a,2b,3,4,5,6}*.yaml` | 新增/修改 | — | ✅ untracked |
| `configs/MODALITY_BASELINE_README.md`、`diagnostic/fusion_phase0_*.py` | 新增 | — | ✅ untracked |

### Untouched D′ files（明确未修改）

| File | sha256（与轮前一致） |
|---|---|
| `configs/train_rgbid_sepstem_clahe.yaml` | **`a4e329cf…86e12` ✅ 未变** |
| `configs/yolo11m_sepstem.yaml` | **`9b14f294…3fdd8` ✅ 未变** |
| `configs/train_rgbid_sepstem_clahe_rectlate10.yaml` | `edc16cc4…463e12` ✅ 未变 |
| `runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt` | 未被打开写入 |
| `data/processed/**` | 未被修改（split 走幂等复用） |

**预存在的 working-tree 修改（**不是本轮**）**：`scripts/predict_rect.py`（74 行）、`submissions/rgbid_sepstem_clahe_f1_candidate_infer.log`（1 行）—— 两者在本轮开始前的 `git diff --stat` 中已存在，本轮未触碰。

### ⚠ 超出 scope 的改动：1 项，**已回退**（见 A-9）

---

## Regression checks（brief §10）

| # | 检查 | 结果 |
|---|---|---|
| **R1** | D′ config / model YAML SHA 不变 | ✅ 两个 SHA 逐位不变 |
| **R1** | D′ 输入通道 = 5、通道序 `[B,G,R,IR,D]`、IR = CLAHE | ✅ T2.3 + C.3 + C.4 |
| **R1** | D′ pretrained remap = 550 | ✅ T4（`remap=550`） |
| **R2** | M2a 真 CLAHE / M2b 真 percentile（不同分支） | ✅ C.1 `maxdiff=92` |
| **R3** | M4/M5 stem ≠ 全随机 | ✅ T4：`remap=5`、`RGB==stock:True`、`aux_nonzero:True` |
| **R4** | M6 通道序 `[IR,D]` 非 `[D,IR]` | ✅ T3.1（合成 IR=11/D=22 ⇒ ch0=11, ch1=22） |
| **R5** | 全部 modality = YOLO11m | ✅ T5（scale=m、depth 0.5、width 1.0、max_ch 512） |
| **额外** | 全 configs 模型 yaml 的 remap 回归 | ✅ 18 OK；2 个 pre-existing raise（R2 残留） |

---

## Decision Memo

```text
1. IR dispatch 正确?      是。apply_ir_encoding 单实现；Infrared/RGBIR/IRD/RGBID 四条路径统一读
                          ir_encoding；未知取值 raise。M2a 现在真的是 CLAHE。
2. 4ch 预训练迁移正确?    是。ch[0:3]=预训练 RGB，ch[3:]=mean(RGB)（4ch）；全通道=mean(RGB)（2ch）；
                          源缺失即 raise。实测 remap=5，RGB 通道逐位等于 stock。
3. M6 merge 正确?         是。[IR,D] 由合成常量图证明，train/val/inference 三路一致。
4. augmentation 契约清楚? 是。差异已定位到「通道数触发的 Albumentations/HSV 分派」，
                          已用 opt-in 的 albumentations_p + hsv_*=0 统一到 D′。
5. YOLO11m 显式?          是（且已确认 yaml 内写 scale 无效，只能靠文件名 + 合同门）。
6. 合同门有效?            是。12/12 注入缺陷全部 CAUGHT，baseline PASS；每次真实违规都 raise。
7. 可以开 GPU smoke 吗?   可以，但先做两件事：1–2 epoch 的通道/预处理 smoke；
                          确认云上是否装了 albumentations。
8. 最需要记住的一件事?    本轮最关键的 three fixes（base.py / loaders.py）**不在 git 里**
                          （.gitignore:11）。想复核只能对 SHA-256。
```
