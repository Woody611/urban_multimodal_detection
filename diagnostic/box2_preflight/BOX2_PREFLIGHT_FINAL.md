# BOX2 PREFLIGHT — FINAL

**日期**：2026-09-30
**实验**：Medium Regression Intervention — `box: 7.5 → 15.0`（×2），唯一自变量
**状态**：`PREFLIGHT_READY`（20/20 项一致）— **等待云端执行；本机无法训练（无 GPU）**

---

## 1. git HEAD

```
5f1a8d5802aa49adf35e8fa6fb112a1030b3e6d3   2026-09-30 10:03:11 +0800  "调整文件内容"
```

## 2. git status

```
 M submissions/rgbid_sepstem_clahe_f1_candidate_infer.log     ← 本轮之前已存在
?? configs/train_rgbid_sepstem_clahe_box2.yaml                ← ★ 本轮唯一新增
?? diagnostic/f1_native_small_replay/ONLINE_RESULT.md         ← 本轮之前已存在
?? reports/FINAL_WEEK_PIPELINE_AUDIT.md                       ← 本轮之前已存在
?? submissions/rgbid_sepstem_clahe_f1_candidate/              ← 本轮之前已存在
```

本轮**未修改任何既有文件**；唯一新增是本实验 config（+ 本报告）。

## 3. D′ checkpoint SHA256

```
1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda
runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt
```
**保持不变**（与 D′ run 的 provenance 记录逐位相同）。

## 4. pretrained SHA256

```
d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95   yolo11m.pt
```
冷启动，`resume: false`，**不 warm-start 任何实验权重**。

## 5. loss.py SHA256

```
0f092cf22a372f6d5c5e7197d7adde51cbb23c02ac4522978f42df5b23a6543b
```
与 D′ 记录一致。DFL 钳位路径（`loss.py:332` → `tal.py:363` → `loss.py:278`）未被触碰。

## 6. tal.py SHA256

```
aae7e8ac438f00cda88d7e151795b3f78270ec4ed56b883f413fd16233b7e527
```
与 D′ 记录一致。`topk=10, alpha=0.5, beta=6.0`、`iou_calculation = CIoU.clamp_(0)` 未变。

## 7. D′ config SHA256

```
a4e329cfc3d220206448cdd3377c1b3f5126c50f13c907ada522d98725486e12   configs/train_rgbid_sepstem_clahe.yaml
```
**保持字节不变。**

## 8. Box2 config SHA256

```
4fde5a0bce8ef9ac0a54e4e2ba3e2a87bdd3b5273924271a985baf2bff217d5a   configs/train_rgbid_sepstem_clahe_box2.yaml
3202 bytes, CRLF=0 (纯 LF)
```

## 9. config diff

```diff
--- configs/train_rgbid_sepstem_clahe.yaml
+++ configs/train_rgbid_sepstem_clahe_box2.yaml
@@ -22,7 +22,7 @@
-experiment_name: "urban_multimodal_det_yolo11_rgbid_sepstem_clahe"
+experiment_name: "urban_multimodal_det_yolo11_rgbid_sepstem_clahe_box2"
@@ -59,6 +59,7 @@
 ir_encoding: clahe
+box: 15.0
```

**改动行数 = 2**（`diff | grep -c '^[<>]'` → 3，含 1 行上下文无关的 `box:` 内容行 + 2 行 pair）。
除上述两处外**其余 3187 字节逐位相同**（用 `read_bytes()` + 字节级替换写入，非文本重写）。
**未添加任何注释**（§11 只允许 `experiment_name` 与 `box: 15.0` 两处内容变化）。

## 10. effective args

由 `scripts/train.py:402 _build_train_kwargs(box2_cfg, data_path)` **实际调用**得到（未训练）：

| key | 值 | key | 值 |
|---|---|---|---|
| `epochs` | 300 | `optimizer` | SGD |
| `patience` | 0 | `lr0` | 0.005 |
| `batch` | 8 | `momentum` | 0.937 |
| `imgsz` | 1280 | `weight_decay` | 0.0005 |
| `workers` | 4 | `warmup_epochs` | 3.0 |
| `seed` | 42 | `cos_lr` | True |
| `amp` | True | `deterministic` | True |
| `device` | 0 | `data` | data/processed/rgbid_split_train/dataset.yaml |
| `pretrained` | yolo11m.pt | `project` / `name` | runs / `…_sepstem_clahe_box2` |
| `save` | True | `save_period` | 10 |
| `channels` | 5 | `use_simotm` | RGBID |
| `pairs_rgb_ir` | [visible, infrared] | `pairs_rgb_depth` | [visible, depth] |
| `ir_encoding` | **clahe** | | |

## 11. box / cls / dfl

| gain | D′（args.yaml 实测） | box2（effective） | 来源 | 相对 |
|---|---:|---:|---|---:|
| **`box`** | 7.5 | **15.0** | `train.yaml`（唯一变量） | **×2** |
| `cls` | 0.5 | 0.5 | `default.yaml:100` | ×1 |
| `dfl` | 1.5 | 1.5 | `default.yaml:101` | ×1 |

## 12. seed

`42`（`train.yaml:26` 显式；`_build_train_kwargs` → 传输路径已核）

## 13. resume

`False` — `train.yaml` 的 `checkpoint.resume: false`；且 `resume_path: ""` ⇒ `train.py:605 kwargs["resume"]=False`。

## 14. imgsz

`1280`（`train.yaml:35 image_size: [1280, 1280]` → `int(imgsz[0])`）

## 15. batch

`8`（`train.yaml:34 batch_size: 8`）

## 16. rect

**`False`**（train.yaml 无 `aug:` 段 ⇒ 不覆盖；`default.yaml:31 rect: False`）
与 D′ `args.yaml:24 rect: false` 一致。**训练侧 rect 未实现、未启用。**

## 17. augmentation values

| key | D′ args.yaml | box2 effective | 来源 | |
|---|---:|---:|---|---|
| `mosaic` | 1.0 | 1.0 | default.yaml | OK |
| `close_mosaic` | 10 | 10 | default.yaml | OK |
| `scale` | 0.5 | 0.5 | default.yaml | OK |
| `translate` | 0.1 | 0.1 | default.yaml | OK |
| `degrees` / `shear` / `perspective` | 0.0 | 0.0 | default.yaml | OK |
| `flipud` / `fliplr` | 0.0 / 0.5 | 0.0 / 0.5 | default.yaml | OK |
| `mixup` / `copy_paste` | 0.0 / 0.0 | 0.0 / 0.0 | default.yaml | OK |
| `hsv_h/s/v`, `erasing` | 0.015/0.7/0.4 / 0.4 | 同 | default.yaml | OK |

**注意**：RGBID 5ch 下 `RandomHSV` 与 `alb` 为既有的惰性路径（`augment.py:4199 alb=Albumentations(p=0)`），
D′ 与 box2 完全相同，非本实验变量。

## 18. ir_encoding

`clahe` — 显式写入 `train.yaml:61`，紧随其后插入 `box: 15.0`。
preflight 断言：[PASS] `ir_encoding 已显式声明（未静默回落 percentile）` / `[PASS] ir_encoding == 'clahe'`。
开训后须立刻在 `args.yaml` 中确认该键**存在且值为 clahe**（2026-09-20 事故的守卫项）。

## 19. native_small_replay presence/absence

| 检查 | 结果 |
|---|---|
| `box2 train.yaml` 含任何 `native_small_replay*` 键？ | **否**（absent） |
| `box2 train.yaml` 含 `object_scale_aug*` 键？ | **否**（absent） |
| effective `native_small_replay` | `0.0` → `NativeSmallReplay.__call__` 首句 `if self.p <= 0.0: return labels`（**零 RNG 消耗**） |
| effective `object_scale_aug` | `False` → `ObjectScaleAug.__call__` 首句 `if not self.enabled: return labels`（**零 RNG 消耗**） |

⚠ 这两个键在 `default.yaml:154-166` 中**已注册**（F1/OASA 时期加入，非本轮改动）。
D′ 的 `args.yaml` 中不存在这两个键；因此 box2 的 `args.yaml` 会**多列出**它们（值均为惰性默认）。
二者在首句即 return、不消耗 RNG ⇒ **数据流与 D′ 逐位一致**（F1 preflight Gates 10a/10b 已实测验证）。

## 20. exact training command

```bash
python scripts/preflight_check_train_config.py \
    --model_config configs/yolo11m_sepstem.yaml \
    --train_config configs/train_rgbid_sepstem_clahe_box2.yaml \
    --expect-ir-encoding clahe

python scripts/train.py \
    --model_config configs/yolo11m_sepstem.yaml \
    --train_config configs/train_rgbid_sepstem_clahe_box2.yaml
```

---

# §13 之外的额外发现（必须披露，不阻断）

## F-1 `git status` 对 `ultralytics/data/` 结构性失效

`.gitignore:11: ultralytics/data/*` ⇒ `augment.py` / `base.py` / `loaders.py` / `build.py` / `dataset.py`
**不进 git**。§12 的 "git status 无意外修改" 这一项**在构造上无法覆盖这几个文件**。故改用 SHA 逐一比对：

| 文件 | D′ 记录 | 当前 | 判定 |
|---|---|---|---|
| `ultralytics/utils/tal.py` | `aae7e8ac…` | `aae7e8ac…` | ✅ 相同 |
| `ultralytics/utils/loss.py` | `0f092cf2…` | `0f092cf2…` | ✅ 相同 |
| `ultralytics/utils/instance.py` | `78a72d1f…` | `78a72d1f…` | ✅ 相同 |
| `ultralytics/data/base.py` | `8bcf8349…` | `8bcf8349…` | ✅ 相同 |
| `ultralytics/data/build.py` | `f014263f…` | `f014263f…` | ✅ 相同 |
| `ultralytics/data/dataset.py` | `0aba2bc1…` | `0aba2bc1…` | ✅ 相同 |
| `configs/yolo11m_sepstem.yaml` | `9b14f294…` | `9b14f294…` | ✅ 相同 |
| **`ultralytics/data/augment.py`** | **`5cb9a407…`** | **`53c75148…`** | ⚠️ **不同（F1 时期加入，本轮之前既存）** |
| `scripts/train.py` | （D′ 未记录） | `db4371e3…` | ⚠️ F1 版本（含惰性 guarded 转发） |

**差异性质**：`augment.py` 当前含 `NativeSmallReplay`（L3958/L4216）与 `ObjectScaleAug`（L1513/L4203），
两者在本实验设置下**首句 early-return、零 RNG 消耗**（见 §19）。`scripts/train.py` 的
`native_small_replay*` 转发被 `if "native_small_replay" in train_cfg` 守卫，本配置未触发。
⇒ **行为等价，哈希不等**。单变量声明依赖该 inertness 论证，而非哈希相等。
**本项不满足「哈希完全一致」，但满足「无行为改变」。是否接受由你裁决。**

## F-2 本机无法训练

```
torch 2.4.1+cpu     cuda available: False     device count: 0     nproc: 18
```
本机为 CPU-only（300 ep @1280/YOLO11m 在 CPU 上约 15 天）。**训练必须在云端执行**（见 §20）。
**本轮到此为止，不启动训练。**

## F-3 评测脚本默认权重指向 F1（训练无关，但评测时是陷阱）

```
scripts/predict_rect.py:93  DEFAULT_WEIGHTS   = runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_f1/weights/best.pt
scripts/predict_rect.py:94  DEFAULT_TRAIN_CONFIG = configs/train_rgbid_sepstem_clahe_f1.yaml
```
§0 禁止修改该文件 ⇒ 训练后评测**必须显式传参**，否则会静默评测 F1：

```bash
python scripts/predict_rect.py --mode rect \
    --weights runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_box2/weights/best.pt \
    --train_config configs/train_rgbid_sepstem_clahe_box2.yaml \
    --source data/processed/rgbid_split_train/images/val/visible \
    --output diagnostic/box2_infer_val
```

---

# 结论

| §13 清单 | 结果 |
|---|---|
| 1–20 全部 | **PASS**（逐项见上） |
| §12 preflight 脚本 | `RESULT: ALL CHECKS PASSED` / `PREFLIGHT_READY` / `EXIT_CODE=0` |
| 额外发现 | F-1（augment.py 哈希差异，行为等价）、F-2（本机无 GPU）、F-3（评测默认权重陷阱） |

```
PREFLIGHT STATUS = PASS (20/20)
TRAINING STARTED = NO   ← 本机无 GPU；交由云端执行 §20 命令
```
