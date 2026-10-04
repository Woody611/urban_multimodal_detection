# PROVENANCE — Probe-P3-Trajectory

```text
BASELINE = D′ SepStem + IR-CLAHE

MODEL STRUCTURE = UNCHANGED
LOSS            = UNCHANGED
TAL             = UNCHANGED
DATA            = UNCHANGED
AUGMENTATION    = UNCHANGED
OPTIMIZER       = UNCHANGED
SCHEDULER       = UNCHANGED
SEED            = UNCHANGED
BATCH           = UNCHANGED
IMGSZ           = UNCHANGED

PROBE_ONLY      = YES
```

## 冻结 SHA256

| 文件 | SHA256 |
|---|---|
| model yaml (configs/yolo11m_sepstem.yaml) | `9b14f2946073338414b4441784b6df87a868b982be32ac8d50b6d1c4a17f3dd8` |
| train config (configs/train_rgbid_sepstem_clahe.yaml) | `a4e329cfc3d220206448cdd3377c1b3f5126c50f13c907ada522d98725486e12` |
| train entry (scripts/train.py) | `064a83e56e692203e9ae40a74703c992f831ebd5eed4f15e44136d70dca6ba9b` |
| loss (ultralytics/utils/loss.py) | `0f092cf22a372f6d5c5e7197d7adde51cbb23c02ac4522978f42df5b23a6543b` |
| TAL (ultralytics/utils/tal.py) | `aae7e8ac438f00cda88d7e151795b3f78270ec4ed56b883f413fd16233b7e527` |
| data base (ultralytics/data/base.py) | `8bcf834930155bd58a99ccfc5e87113398f740aeb720b5827ff05ba45c5871d9` |
| augment (ultralytics/data/augment.py) | `53c75148fc168a5390b4f50523bcd9c1ba7379e5de6430f2875d8827953531c5` |
| D-prime best.pt | `1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda` |
| frozen feature artifact (_v7_vectors.json) | `977593ae0012610004077b7fa58cd2cdd9e2a59dc2d90044b5e65ab3b6db0496` |
| G-86 source (per_gt_attribution.csv) | `62dbc551f4703bb3162baed16266f793b6e9cba4f21509236afee186bfc0c605` |
| probe (_probe.py) | `edd98c8e7b61c8ec3326ec45ca52e0da5c0ad8a1a297cd67f53825296c44fd5c` |
| probe runner (_run_probe.py) | `30a5bed87b786c20cd6b44c69448f4dae430630ed3030d9ec92375f666a4c15e` |
| integrity gate (_integrity_gate.py) | `ffb472a4f7c3e2c4044ff409d89cd275a5b545ed08a615ab8aa5aee18fa64561` |
| G-86 freeze (_g86.py) | `226dc13873186782731900bc910fad2f43eef9a47b8a3c01f92956afd13c8271` |

## D′ args.yaml（逐项，作为 SCHEDULER/BATCH/IMGSZ/增强 的不变证据）

```text
epochs          = 300
batch           = 8
imgsz           = 1280
seed            = 42
optimizer       = SGD
lr0             = 0.005
lrf             = 0.01
momentum        = 0.937
weight_decay    = 0.0005
warmup_epochs   = 3.0
cos_lr          = True
close_mosaic    = 10
mosaic          = 1.0
scale           = 0.5
translate       = 0.1
degrees         = 0.0
shear           = 0.0
perspective     = 0.0
flipud          = 0.0
fliplr          = 0.5
mixup           = 0.0
copy_paste      = 0.0
rect            = False
ir_encoding     = clahe
channels        = 5
use_simotm      = RGBID
pretrained      = yolo11m.pt
resume          = False
cls_pw          = []
dfl             = 1.5
box             = 7.5
cls             = 0.5
device          = 0
workers         = 4
```

## 与 D′ 的**唯一差异**

| 项 | D′ | 本 probe | 是否训练变量 |
|---|---|---|---|
| `name`（输出目录） | `urban_multimodal_det_yolo11_rgbid_sepstem_clahe` | `urban_multimodal_det_yolo11_rgbid_sepstem_clahe_PROBE30` | **否**（只决定 run 目录；RectLate10 轮已实测确认） |
| 训练长度 | 300 ep 跑满 | 300 ep **配置不变**，回调在 ep30 后 `trainer.stop=True` | **否**（前 30 ep 的 LR/增强/相位与 D′ 逐位相同） |
| 其余全部 | — | — | **无差异** |

## G-86 冻结集合

- 来源：`diagnostic\medium_93_47_attribution\per_gt_attribution.csv`  sha256 `62dbc551f4703bb3162baed16266f793b6e9cba4f21509236afee186bfc0c605`
- 定义：`per_gt_attribution.csv 的 certified_prefinal_loss == 1（与上一轮 representation audit 同一批）`
- n = **86**，分布在 **54** 张 val 图
- cell 映射：`_v7_extract.py 口径：canvas = 归一化中心 × 1280；P3 160×160 ⇒ cell = floor(c/8)`
- **未重筛、未改阈值、未增删任何 GT**；key 唯一性已断言通过。

## Probe 完整性与停机

- `_integrity_gate.py`（本机 CPU 合成输入）结果：**PROBE_INTEGRITY = PASS**
  G1 forward 逐位不变 / G6 loss 逐位不变 / G2 RawTAL≡父类 / G3 raw_align 未被掩码 /
  G4 RNG 不变 / G5 无梯度+参数未变 / **G8 BN buffer 副作用已由快照+还原消除** / G7 特征语义同口径
- 停机：`on_fit_epoch_end` 中 `if epoch >= 30: trainer.stop = True`（`trainer.py:435,453` 检查该标志）


## 运行状态

```text
PROBE_INTEGRITY = PASS（本机 CPU 合成输入，integrity_gate.log）
TRAINING RUN   = NOT STARTED —— 本机无 GPU（torch 2.4.1+cpu, cuda unavailable）
                 30 epoch 在云端 ≈ 65 min（D′ 稳态 130.7 s/epoch）；本机 CPU ≈ 34.9 h
DECISION       = INCONCLUSIVE（尚无轨迹数据）
NEXT EXPERIMENT= STOP
```

新增冻结/交付文件：
- `g86_frozen.json`  sha `cc0b7551cf224239631c5073c76b8f6c60358cb4711e4d04b0fd43d6cc70ddd4`
- `medium_cells.json` sha `5d0ba324616bd48d8c7b9a01300eb867f4bc410ed26b78d247f62f2de5223c6b`
- `_probe.py`         sha `edd98c8e7b61c8ec3326ec45ca52e0da5c0ad8a1a297cd67f53825296c44fd5c`
- `_run_probe.py`     sha `30a5bed87b786c20cd6b44c69448f4dae430630ed3030d9ec92375f666a4c15e`
- `_integrity_gate.py` sha `ffb472a4f7c3e2c4044ff409d89cd275a5b545ed08a615ab8aa5aee18fa64561`
- `_g86.py`           sha `226dc13873186782731900bc910fad2f43eef9a47b8a3c01f92956afd13c8271`
- `_medium_cells.py`  sha `9ba902930ddddde979ed562de97b1336b1adba0960f772e31e187b39b1bc1f91`


## 云端首跑后的修复（2026-10-01）

1. **设备混用（致命）**：`self._W/self._b`（cuda）与 `_sample()` 返回的 numpy(CPU) 混算，
   云端在 ep0 的 `on_fit_epoch_end` 报 `Expected all tensors to be on the same device`。
   已改为**纯 numpy**（`_Wn/_bn`）并在真实路径加快速失败断言。
   ⚠ **CPU-only 本机无法复现此类 bug**（本地 `_W` 恰好是 CPU tensor）——dry-run 当时 PASS 却漏掉它。
2. **`dump_epoch` 类型 bug（致命）**：把所有字段都往 float64 转，含字符串 `key`/`role` ⇒ 第 1 个观测 epoch 就崩。
   已按数值/字符串分流。
3. **`_decide` 判序 bug**：CASE C 是 CASE A 的子情形却后判 ⇒ 真实 C 会被误报成 A。已改为**先判 C**，
   并按 §13 要求 h **稳定**（`h_decline < 0.10`）。
4. **`finalize` 总体错位**：`epoch_summary.csv`/`decision.*` 的中位数算在**全部 1320 个 medium cell** 上，
   而非规格 §9 要求的 **G-86** ⇒ 该两个产物答非所问。已按 `role` 分组修正；
   已在跑的 run 用 `_redecide.py` 离线重算（`per_gt_trajectory.csv` 带 role，可精确过滤）。
5. `_run_probe.py` 传错 JSON（`g86_frozen.json` 无 `role`）→ 已改为 `medium_cells.json`。
6. 日志曾把 `1320`（全部 medium cell）写成 “G-86 records” → 已改为分别打印。
