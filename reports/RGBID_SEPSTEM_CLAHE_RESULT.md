# D' = IR-CLAHE + Separate-Stem —— 训练与本地评测结果

**日期**：2026-09-22 · **性质**：训练后诊断（未提交、未修改任何既有资产）
**对象**：重跑后的 `runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/`（`ir_encoding` 污染已修复）

---

## 1. 单变量是否成立 —— 成立 ✅

| 检查 | 结果 |
|---|---|
| **`ir_encoding`** | **`clahe`**（`args.yaml` 与 ckpt 内嵌 `train_args` **两处一致**） |
| `model` | `configs/yolo11m_sepstem.yaml`（31 层 / 20,061,972 参数 / ch=5 / nc=12） |
| 其余全部超参 vs D | `args.yaml` 逐项对比，**除 `model`/`name`/`save_dir` 外差异 0 项** |
| D 的资产 | 未触碰（`best.pt` sha256 `3c2f942c…` 未变） |
| `OFFICIAL_FALLBACK` | 未变（`f9dddbfa…`，与 FREEZE_MANIFEST 一致） |

**D vs D' 的差异恰好两项：`model_config`（fusion architecture）＋ `experiment_name`。**

### Provenance

| 项 | SHA256（前 40） |
|---|---|
| D' `weights/best.pt` | `1cae45f75693f54146e35c5fa076c0a6f78ca1cf` |
| D' `weights/last.pt` | `b8398b875316c727907e268fb4a3f3f53dc12b16` |
| `configs/yolo11m_sepstem.yaml` | `9b14f2946073338414b4441784b6df87a868b982` |
| `configs/train_rgbid_sepstem_clahe.yaml` | `a4e329cfc3d220206448cdd3377c1b3f5126c50f` |

---

## 2. 三个口径的对比（同一 400 图 val / 2807 GT）

| 口径 | D | **D'** | Δ |
|---|---:|---:|---:|
| **训练内 val**（results.csv best） | 0.56806 @ep263 | **0.56690 @ep246** | **−0.00116** |
| **fork 口径**（official_map 同批预测） | 0.56607 | **0.56596** | **−0.00011** |
| **官方口径**（official_map） | 0.51076 | **0.51657** | **+0.00581** |

官方口径分项：

| 指标 | D | D' | Δ |
|---|---:|---:|---:|
| mAP50 | 0.77119 | 0.78021 | **+0.00902** |
| mAP75 | 0.50161 | 0.51241 | **+0.01080** |
| mAP50-95 | 0.51076 | 0.51657 | **+0.00581** |
| 框数 | 4341 | 4654 | +313 |

（D' 评测命令与 D **逐字同构**，仅 `weights` / `train_config` / `output` 三处路径不同；
推理日志自证 `use_simotm=RGBID channels=5 ir_encoding=clahe`。）

---

## 3. **关键发现：两个口径再次符号相反**

```
训练内 val (fork)  :  D' 比 D 差  −0.00116
fork 重算           :  D' 比 D 差  −0.00011   （≈ 完全打平）
官方口径            :  D' 比 D 好  +0.00581   ← 符号相反
```

这是本项目已记录的 **fork vs 官方口径分歧** 的又一次实例（见记忆
`fork-vs-official-metric-divergence` / `metric-instrument-fork-over-official`）。
本项目对该分歧的既有裁定是：**方向信 fork，官方口径不可单独作否决依据**。

按此裁定：fork 说 **−0.0001 ≈ 完全打平**，因此本地证据**不支持"D' 优于 D"**。

### 但要与 D 的先例一起读

D 当年的本地官方口径是 **+0.00148**（CI 含 0，本地判 `D_REJECTED`），**线上却是 +0.02350**。
即官方口径**大幅低估**了 D。所以"官方口径 +0.00581"不能直接换算成线上增益
（记忆 `fork-vs-official-metric-divergence` 已明确：本地→线上的 gap 不是恒定偏移）。

**结论：本地三个口径给出的信号互相矛盾（−0.0012 / −0.0001 / +0.0058），
幅度全部落在本项目已记录的 checkpoint 波动范围内（~0.008–0.015）。
本地无法判定胜负。**

---

## 4. 训练曲线（D'）

```
ep200-219  0.54870
ep220-239  0.55277
ep240-259  0.55990   ← best 0.56690 @ep246
ep260-279  0.55926
ep280-299  0.55856
```

与 D 同形：ep240 后进入平台期，300 epoch 预算用尽，无早停（`patience: 0`）。

---

## 5. 判定

按 brief §21/§24，**最终判定只能由线上结果给出**：

```text
D  online = 0.5553
D' online = ?        ← 尚未提交

D' > 0.5553  -> SEPARATE_STEM_ONLINE_GAIN_CONFIRMED
D' ≈ 0.5553  -> SEPARATE_STEM_NO_CLEAR_ONLINE_GAIN
D' < 0.5553  -> SEPARATE_STEM_ONLINE_REGRESSION
```

**本地证据不足以预测**：三个口径符号互相矛盾，且 D 的先例证明官方口径会大幅低估。
是否值得花一次提交由你决定；若要提交，命令与 D 逐字同构（仅三处路径不同）：

```bash
python scripts/predict_rect.py \
  --weights runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt \
  --train_config configs/train_rgbid_sepstem_clahe.yaml \
  --source data/raw/test/visible \
  --mode rect --imgsz 1280 --conf 0.001 --iou 0.7 --max_det 300 --max_boxes 100 \
  --batch 16 --output submissions/rgbid_sepstem_clahe_candidate
python scripts/check_submission.py \
  --results submissions/rgbid_sepstem_clahe_candidate/results \
  --zip submissions/rgbid_sepstem_clahe_candidate/submission.zip --expect 1000
```

---

## 6. 提交包 provenance（2026-09-22 生成，待上传）

与 D **逐字同构**的唯一区别是 `weights` / `train_config` / `output` 三处路径：

```bash
python scripts/predict_rect.py \
  --weights runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt \
  --train_config configs/train_rgbid_sepstem_clahe.yaml \
  --source data/raw/test/visible \
  --mode rect --imgsz 1280 --conf 0.001 --iou 0.7 --max_det 300 --max_boxes 100 \
  --batch 16 --output submissions/rgbid_sepstem_clahe_candidate
```

推理日志自证：`use_simotm=RGBID channels=5 ir_encoding=clahe`。

| 项 | 值 |
|---|---|
| experiment name | `urban_multimodal_det_yolo11_rgbid_sepstem_clahe` |
| **submission.zip** | 349,545 B · sha256 `830e0c552615852f86a51005249c7b814844d1283c6310632903f3a16380e4d8` |
| checkpoint `best.pt` | 40,641,641 B · sha256 `1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda` |
| model config | `9b14f2946073338414b4441784b6df87a868b982`（`configs/yolo11m_sepstem.yaml`） |
| train config | `a4e329cfc3d220206448cdd3377c1b3f5126c50f`（`configs/train_rgbid_sepstem_clahe.yaml`） |
| 预处理链 base.py | `8bcf834930155bd58a99ccfc5e87113398f740aeb720b5827ff05ba45c5871d9` |
| 预处理链 loaders.py | `f84bdcf29081e6f95b58fbf19c255d1da52281d27c107746d0661eaf6d8c3c1c` |
| remap `detect/train.py` | `8b4178be947acdc031878132a8b5ef79f73019de403dd8925441e652ab89ae85` |
| 推理 `predict_rect.py` | `c55f7744e3a40e71e8ff38da5030c39adeb4344d68ead95ed82b56097e867e18` |
| `cfg/default.yaml` | `6f91a2770b4e3cc7b393d98d2cb5f2cae8fec4371ef67e310beff6a8f6045431` |

**预测统计**（`check_submission.py` → `ALL PASS`）：

| 项 | 值 |
|---|---|
| TXT 数 / 与 test 图 1:1 | 1000 / 1000 ✅ |
| 空 TXT（无检测） | 11 |
| 格式错误 / 类别越界 / 坐标越界 | 0 / 0 / 0 ✅ |
| **总框数** | **10,926**（D 候选 6,388；0.53180 正式包 7,370） |
| 每图框数 mean/median/max | 10.93 / 8 / **100** |
| **每图 ≤100 框（赛题硬规则）** | **max=100，超限图 0** ✅ |
| 触顶 100 框的图 | 3（截断） |

**未触碰**：`submissions/rgbird_ir_quicktest/`（0.53180 正式包，sha256 `0c550773…` 与 FREEZE_MANIFEST 一致）、
`submissions/rgbid_d_ir_clahe_candidate/`（D 候选包，sha256 `96881002…`）、D 的 `best.pt`（`3c2f942c…`）、
`OFFICIAL_FALLBACK`（`f9dddbfa…`）。

---

```text
D  online            = 0.5553   （未变更）
OFFICIAL_FALLBACK    = 0.53180  （未变更）
D  best.pt           = 3c2f942c…（未变更）
本轮修改的既有文件    = 0
待办                = 上传 submissions/rgbid_sepstem_clahe_candidate/submission.zip
```
