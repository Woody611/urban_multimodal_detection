# Final 3-Modality Experiment — Readiness Audit

**日期** 2026-10-04 · **性质** ZERO GPU / ZERO forward / ZERO training / 未改动任何代码与配置
**Baseline** D′ = `urban_multimodal_det_yolo11_rgbid_sepstem_clahe`，本地 official mAP50-95 = **0.515281**，线上 48.712
**目的** 回答：Seed-2 结束后 30 分钟内，能否启动一个 **single-shot RGB+IR+Depth 最小实验**

---

## Executive Verdict

```text
READY / NOT READY :  NOT READY
```

**理由（一句话）**：**所有能用现有 config 表达的 3-modality 配置都已经训练过了**（D 与 D′）；
任何**新的** 3-modality 布局（含 Phase F CASE C 指定的
"M4-preserving minimal residual Depth adapter"）**拿不到预训练权重**，
而且其中**最自然的那一个会让合同门误判为 PASS**（实测：body 只有 9.7% 是预训练，55% 是随机/错映射）。

⇒ **30 分钟 SLA 当前不成立**；启动前必须先落一个 remap 分支。

---

## §0 Provenance / SHA / git

```text
HEAD                       5f1a8d5802aa49adf35e8fa6fb112a1030b3e6d3
branch                     yyy
```

| SHA-256 (前16) | 文件 |
|---|---|
| `67cb9fe02757fd59` | `scripts/train.py` （modified, tracked） |
| `9d704300143a5786` | `scripts/predict_rect.py` （modified, tracked） |
| `e82128abc082ed05` | `scripts/official_eval.py` |
| `bf380a22d2491f73` | `ultralytics/data/base.py` ⚠ **gitignored（`.gitignore:11 ultralytics/data/*`）** |
| `56c07c8520cf5b25` | `ultralytics/data/loaders.py` ⚠ **gitignored** |
| `abfc90f53f75f7ad` | `ultralytics/data/augment.py` （tracked，已 modified） |
| `6538ab6f3a0ca4a3` | `ultralytics/models/yolo/detect/train.py` （modified, tracked） |
| `cf06d3a61a5cc9a9` | `ultralytics/cfg/default.yaml` （modified, tracked） |
| `9b14f29460733384` | `configs/yolo11m_sepstem.yaml` （= D′ 的 model yaml） |
| `edc95f4665ead847` | `configs/yolo11m_earlyfusion.yaml` （= D 的 model yaml） |
| `a4e329cfc3d22020` | `configs/train_rgbid_sepstem_clahe.yaml` （= D′ 的 train config） |
| `1bca25e659ff85f2` | `diagnostic/gpu_smoke_gate.py` |
| `283c28f006bbe834` | `scripts/validate_modality_contract.py` |

**⚠ git 盲区**：本次审计最关键的两个文件（`base.py` / `loaders.py`）**不在 git 里**
（`.gitignore:11` 的 `ultralytics/data/*` 无锚定）。`git status` 看不到它们，只有 SHA-256 可校验。
上游 `yolo11m.pt` 存在（`attempt_load_one_weight` **优先 EMA**）。

---

## §1 现有 3-modality 资产盘点（回答"还剩哪个格子没跑"）

| ID | 模态 | model yaml | train config | 状态 |
|---|---|---|---|---|
| **D** | RGB+IR+Depth（5ch 单 stem） | `yolo11m_earlyfusion.yaml` | `train_rgbid_ir_clahe.yaml` | **已训练 2026-09-19** |
| **D′** | RGB+IR+Depth（sepstem 3ch/1ch/1ch） | `yolo11m_sepstem.yaml` | `train_rgbid_sepstem_clahe.yaml` | **已训练 2026-09-22 = incumbent** |
| — | RGB+IR+Depth（percentile 对照） | `yolo11m_sepstem.yaml` | `train_rgbid_sepstem_percentile.yaml` | 已训练 2026-09-20 |
| — | sepstem stem 宽 40/16/8 | `yolo11m_sepstem_40_16_8.yaml` | `train_rgbid_sepstem_40_16_8_clahe.yaml` | **从未训成**（`runs/` 目录为空）—— **stem 宽 = 超参数搜索，被本轮 brief 禁止** |
| — | sepstem + E1 | `yolo11m_sepstem_e1.yaml` | `train_rgbid_sepstem_clahe_e1.yaml` | 已训练 2026-09-28（E1 路线已线上否决） |
| — | 5ch + P3 attn / P2 probe | `yolo11m_earlyfusion_p3attn.yaml` / `yolo11m_rgbid_p2_probe.yaml` | 同名 | 已训练；**P2/P3 = 明令禁止** |

**recipe 比对结论**：`train_rgbid_ir_clahe.yaml` 与 M-series 的 canonical block
**逐项相同**（300ep / patience0 / batch8 / 1280 / seed42 / SGD 0.937/5e-4 / CosineAnnealingLR /
amp / val_ratio 0.2 / yolo11m.pt），唯一差别是 M-series 显式写了
`aug.hsv_*=0 + albumentations_p: 0` —— 而 **5ch 下这两个开关本来就是惰性的**
（`RandomHSV` 因 `shape[-1]!=3` 直接 return；5ch 走 `Albumentations(p=0)`）。
⇒ **一个"5ch 的 M-series 配置"在行为上逐位等于 D′（sepstem）或 D（earlyfusion）。**

```text
⇒ 3-modality 这一行在矩阵里【没有空位】：两个格子都已被已训练、已评测的模型占据。
```

---

## §2 ★ 决定性实验：新布局能否拿到预训练？（本机实测）

复现 `validate_modality_contract.py` 的加载路径
（`attempt_load_one_weight` → `model.load()` → `_transfer_rgb_pretrained`），
统计目标模型有多少张量**逐位等于** stock `yolo11m.pt` 的张量。
脚本与 3 个候选 yaml 原型在 `%TEMP%\f3m_audit\`（**未进入仓库**）。

| 模型 | 布局 | 分派到的分支 | `n_ret` | 逐位 == stock | **未预训练** |
|---|---|---|---:|---:|---:|
| **D** | 5ch 单 stem | multi-ch stem | 5 | **536/543 = 98.7%** | 1.3% |
| **D′** | 3ch+1ch+1ch sepstem | `_remap_separate_stem` | 550 | **532/553 = 96.2%** | 3.8% |
| **candA** | D′ 布局，Depth stem 换 `ZeroConv2d` | **RGBD concat_res 索引表** | 9 | **58/550 = 10.5%** | **54.5%** |
| **candB** | M4 4ch stem ‖ Depth 1ch stem → Concat | **RGBD concat_res 索引表** | 44 | **97/548 = 17.7%** | **50.5%** |
| **candC** | **M4 4ch stem + `ZeroConv2d` Depth 残差 ADD** | multi-ch（**只写了 stem**） | 5 | **53/545 = 9.7%** | **55.0%** |

**机制**：
1. 任何"三路切分"都必须先放 `Silence` + `SilenceChannel`，**层号整体位移** ⇒
   `model.load()` 的名字匹配率从 98.9% 崩到 **9.2–9.7%**；
2. 能修这种位移的只有两条分支：① 恰好 **1×3ch + 2×1ch**（= D′ 的 sepstem）；
   ② RGBD 的**硬编码索引表**；
3. **"4ch stem + 额外 1ch 分支"两者都不匹配** ⇒ 静默落进 RGBD 索引表 ⇒ 错映射；
4. candC 更隐蔽：它命中 multi-ch 分支，**该分支只写第一个 Conv 就 return**，
   后面的位移层全部没被碰过 ⇒ body 仅 9.7% 预训练。

### ★★ 合同门盲区（本轮最有价值的发现）

`validate_modality_contract.py` 对非 3ch 的判据只有一行：

```python
need(n_ret > 0 and stem_changed, "非 3ch stem 必须被显式初始化", ...)
```

实测三个候选过门情况：

| 候选 | `n_ret` | `stem_changed` | **合同门** | **真实 body 预训练率** |
|---|---:|---|---|---|
| candA | 9 | False | ❌ FAIL（拦住） | 10.5% |
| candB | 44 | False | ❌ FAIL（拦住） | 17.7% |
| **candC** | **5** | **True** | **✅ PASS** | **9.7%** |

```text
⇒ candC = Phase F CASE C 指定的那个形状（M4-preserving minimal residual Depth adapter），
   会【通过】现有合同门，却带着 ~90% 随机初始化的主干开训。
   这与历史上"三-stem [3,1,1] 静默退化为从零训练（body 仅 0.2%）"是同一类事故。
```

**结论：在补 remap 分支之前，candC 这一类 yaml 绝不允许进入 `configs/`。**

---

## §3 Depth pipeline（现有实现，逐行核对）

`base.py` RGBID 分支（`base.py:378-411`）与 `Depth`/`IRD`/`RGBD` 分支共用同一段深度映射：

```python
if im_depth.ndim == 3:
    im_depth = im_depth[..., 0]
if im_depth.dtype != np.uint8:          # ← 关键门
    im_depth = im_depth.astype(np.float32)
    im_depth[im_depth < 300] = 0.0      # 无效深度 <300mm 置 0
    im_depth = np.clip(im_depth / 19999.0 * 255.0, 0, 255).astype(np.uint8)
```

- 通道序：`_merge_channels_rgbid` = `cv2.merge((b, g, r, im_infrared, im_depth))` ⇒ **[B,G,R,IR,D]** ✓
- 推理侧：`loaders.py` 的 image 路径已有 `Depth`/`RGBID` 分支（Infrastructure Repair 补的） ✓
- **`channels: 5` 门控** 5ch 增强分支；`RandomHSV` 与 `Albumentations` 在 5ch 下惰性 ✓

### ★ HR / LR 双分层（**训练集里也有**，此前只记录了 val）

| split | depth PNG (HR) | depth JPEG (LR) | HR 尺寸/dtype | LR 尺寸/dtype |
|---|---:|---:|---|---|
| **train** | **1478** | **122**（7.6%） | 1920×1080, uint16, colortype 0 | 640×360, 8-bit JPEG |
| **val** | 373 | 27（6.75%） | 同上 | 同上 |

- 三个模态目录的 **stem 集合逐位一致**（visible == infrared == depth），LR 子集三者同集合 ✓
- **`im_depth.dtype != np.uint8` 这道门使 LR 与 HR 走完全不同的公式**：
  HR 经 `<300` 置零 + `/19999*255`；**LR 的 uint8 直接透传**，
  无 `<300`、无 `/19999` ⇒ **同一通道在两分层里是两种量纲**。
- 这是 **D′ 自身的既有行为**，不是新引入的问题；任何 3-modality 实验都会继承它。
- **本机 `albumentations` 未安装** ⇒ `Albumentations(p=?)` 在本机恒为 no-op；云上是否安装**未确认**。

---

## §4 输入通道 / 融合位置 / evaluator

| 项 | 值 |
|---|---|
| 输入通道 | `RGBID` ⇒ **5ch [B,G,R,IR,D]**（`channels: 5`，模型 yaml `ch: 5`） |
| 融合位置 | **early fusion**：D = 单层 `Conv(5→64,3,2)`；D′ = 3 路轻量 stem `48/8/8` + `Concat` → 64 |
| 参数量 | D 20,063,412 ；D′ 20,061,972（Δ = −1,440 = −0.0072%） |
| evaluator | `scripts/predict_rect.py`（`--mode rect --conf 0.001 --iou 0.7 --imgsz 1280 --max_boxes 100`）→ `scripts/official_eval.py --metric A` |
| 指标口径 | official METRIC_A，top-100，conf 降序贪心 one-to-one，IoU 0.50–0.95，101 点插值 |
| 公平性 | 5 个 split 逐位同划分（1600/400）；`_transfer_rgb_pretrained` 两条既有路径均已验证 |

---

## §5 预计耗时（基于本机 `results.csv` 实测）

| 配置 | 通道 | 300ep 总时长 | 首 epoch |
|---|---:|---:|---:|
| M1 RGB | 3 | 16,569.7 s = **4.60 h** | 71.1 s |
| M4 RGB+IR | 4 | 18,892.6 s = **5.25 h** | 109.5 s |
| M5 RGB+Depth | 4 | 17,072 s = **4.74 h** | 79.0 s |
| D′ RGB+IR+Depth | 5 | 40,483.6 s = **11.25 h** ⚠ | 205.7 s |

⚠ D′ 那次跑在**旧容器**（M2a 事故期间已换到 `autodl-container-b05511bdfa-75798c3e`）。
按当前容器的 3ch→4ch 边际（+14%）外推：

```text
5ch 3-modality 单次训练 ≈ 6.0–6.3 h  （当前容器）
+ predict_rect(400 val) ≈ 0.2 h（本机 CPU 口径 26–30 min/1000 图；GPU 更快）
+ official_eval ≈ <2 min
⇒ 端到端 ≈ 6.5 h
```

---

## §6 唯一可执行配置（**在 remap 分支落地之前不可执行**）

实验身份：**M4-preserving minimal residual Depth adapter**（= Phase F CASE C 指定的形状）
—— 以 **D′ (0.515281)** 为唯一 baseline，对照 **M4 (0.50597)** 已落盘结果。

**(a) model yaml**（原型 `%TEMP%\f3m_audit\yolo11m_candC_m4stem_zeroinit_depth_residual.yaml`）

```yaml
ch: 5
nc: 12
scales: {n: [0.50,0.25,1024], s: [0.50,0.50,1024], m: [0.50,1.00,512], l: [1.00,1.00,512], x: [1.00,1.50,512]}
backbone:
  - [-1, 1, Silence, []]                     # 0  5ch 透传
  - [0, 1, SilenceChannel, [0, 4]]           # 1  → [B,G,R,IR] 4ch  == M4 的输入
  - [-1, 1, Conv, [64, 3, 2]]                # 2  ← M4 的 stem（4ch，已预训练映射）
  - [0, 1, SilenceChannel, [4, 5]]           # 3  → Depth 1ch
  - [-1, 1, ZeroConv2d, [64, 3, 2, 1]]       # 4  ★ 零初始化 Depth adapter（t=0 时为 0）
  - [[2, 4], 1, ADD, [1.0]]                  # 5  fused = M4_stem + 0  ⇒ t=0 与 M4 逐位同源
  # …… 其后与 yolo11m_earlyfusion.yaml 的 L1..L10 逐项相同（层号 +5）
head:  # …… 同 earlyfusion，引用索引 +5，Detect [[21,24,27]]
```

> `ZeroConv2d` 与 `ADD` **都已注册在 `parse_model`**（`tasks.py:1176` / `:1125`）⇒ **零新代码**。
> ⚠ `ZeroConv2d` 继承 `nn.Conv2d`，**`padding` 默认为 0**（不像 ultralytics `Conv` 取 `k//2`），
> 必须写 `[64, 3, 2, 1]`，否则 Concat/ADD 处尺寸差 1 而报错。

**(b) train config**：`train_rgbid_sepstem_clahe.yaml` 的逐行副本，只改 `experiment_name`。
（5ch ⇒ 无需写 `aug:` 段；`ir_encoding: clahe` 必须保留，`train.py:655` 会硬失败兜底。）

**(c) 启动命令（唯一）**

```bash
python scripts/train.py --model_config configs/yolo11m_<新名>.yaml \
                        --train_config configs/train_modality_m4_ir_depth_residual.yaml
```

**(d) 评测（必须 official 口径，且 `predict_rect.py` 必须带 `--train_config`）**

```bash
python scripts/predict_rect.py --weights runs/<exp>/weights/best.pt \
        --train_config configs/train_modality_m4_ir_depth_residual.yaml \
        --source data/processed/rgbid_split_train/images/val/visible \
        --mode rect --output diagnostic/f3m_eval --pack
python -X utf8 scripts/official_eval.py --results diagnostic/f3m_eval/predictions.txt \
        --images data/processed/rgbid_split_train/images/val/visible \
        --labels data/processed/rgbid_split_train/labels/val/visible --metric A
```

---

## §7 唯一阻塞项与最小解封路径

**阻塞项**：`_transfer_rgb_pretrained`（`ultralytics/models/yolo/detect/train.py:204-340`）
没有任何分支覆盖"带 `Silence` 前缀、stem 数 ≠ {1×3ch + 2×1ch}、也不是 concat_res 索引表"的布局。
源码里已用注释登记了这个残留风险（`:316-322`），并说明"新增此类模型时必须同步检查本函数的分派条件"。

**最小解封（~30 行，一个分支）** —— 结构上照抄 `_remap_separate_stem` 的**已验证写法**：
1. 从 yaml 定位融合算子（3 输入 Concat 或 ADD）⇒ 得到 `offset`；
2. `offset` 之后的层按 `i ↔ i-offset` 整体映射到 stock L1..；
3. **任何缺对应 / shape 不符 / 歧义一律 `raise`**（`_remap_separate_stem` 已是这个语义，照搬即可）；
4. 4ch stem 用现成的 multi-ch 语义（`ch[0:3]=预训练 RGB`，`ch[3]=mean(RGB)`）；
5. `ZeroConv2d` 保持零初始化（不得被 remap 覆盖）。

**并必须同步加固合同门**：把 `need(n_ret > 0 and stem_changed, ...)` 换成
**"逐位 == stock 的占比 ≥ 阈值"**（本次实测：好布局 96.2–98.7%，坏布局 9.7–17.7%，
阈值取 80% 有 >15pp 的余量），否则 candC 那类模型仍会静默过门。

---

## §8 与 Phase F 证据的关系（必须并列陈述）

本审计**只回答"能不能跑"，不回答"该不该跑"**。同时记录 Phase F 的结论以免误用：

- `DEPTH_ATTRIBUTION = NOT_SUPPORTED`，`NEXT_GPU = NO-GO`；
- `100`(D′ 独有) vs `010`(M4 独有) 的 Depth quality 分离 **size-matched 后全不显著**（p 0.50–0.97）；
- image-level Spearman：`valid_fraction` 与 D′ 净增益 **rho = −0.1479, p = 0.0043（方向相反）**；
- D′ − M4 = −0.00931，CI [−0.02036, +0.00943] 含 0。

⇒ 即使解封后能跑，这个实验的**预期收益按现有证据是负向的**；它唯一正当的用途是**证伪**。

---

## §9 结论

```text
READY / NOT READY :  NOT READY
BLOCKER           :  _transfer_rgb_pretrained 无对应分支（静默错映射 / 合同门盲区）
30 分钟 SLA       :  当前不成立
```

**若要满足 SLA**，需要在 Seed-2 结束**之前**完成：① remap 分支；② 合同门加固；
③ 用 `%TEMP%\f3m_audit\probe_layouts.py` 这类探针把 body 覆盖率从 9.7% 提到 >95% 并留证。
**这三件事本轮一律未做**（brief 明令不修改代码）。

---

## HARD STOP

不训练、不推理、不 smoke、不改代码/配置、不启动 M4 seed-2、不干扰正在运行的 Seed-2。
