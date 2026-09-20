# D' 污染清理 + 重跑准备

**日期**：2026-09-20 · **性质**：清理污染成因并加装拦截；**未训练**

---

## 1. 根因

D' 训练（云上）使用的那份 train config 带有 `experiment_name: ..._rgbid_sepstem_clahe`，
但**没有 `ir_encoding` 键**。于是：

```text
scripts/train.py:_build_train_kwargs   if "ir_encoding" in train_cfg: ...   ← 键不存在 → 不传
        ↓
ultralytics/cfg/default.yaml           无 ir_encoding                        ← 不在 namespace
        ↓
ultralytics/data/base.py:333           getattr(hyp, "ir_encoding", "percentile")  ← 静默回落
        ↓
实际 IR 预处理 = percentile（不是 CLAHE）
```

**关键：这条链上没有任何一处会报错。** 于是 "IR-CLAHE + Separate-Stem" 被训成了
"percentile + Separate-Stem"，D vs D' 差了两个变量。

**为什么本地 dry-run 没拦住**：dry-run 用的是**本地**配置（含 clahe），
而云上跑的是**另一份**配置。**dry-run 天然覆盖不了"云上配置 ≠ 本地配置"这一类风险。**

---

## 2. 已做的 4 项清理

### 2.1 消除命名歧义（根本诱因）

| 动作 | 说明 |
|---|---|
| 删除 `configs/train_rgbid_sepstem.yaml` | 这个文件名"看起来就是" sepstem 配置，却**没有** `ir_encoding`（= percentile）—— 正是本事故的诱因 |
| 新建 `configs/train_rgbid_sepstem_percentile.yaml` | 同样内容，但名字**明说**是 percentile，并**显式**写 `ir_encoding: percentile`（不再依赖默认值） |
| `configs/train_rgbid_sepstem_clahe.yaml` | 把 `ir_encoding: clahe` 从**文件末尾**移到**紧跟 `channels: 5`**（末尾位置在手工复制/截断时最容易丢），并加了醒目注释 |

现状（三个 sepstem 训练配置，`ir_encoding` 全部显式）：

```
configs/train_rgbid_sepstem_clahe.yaml          ir_encoding=clahe        ← D' 用这个
configs/train_rgbid_sepstem_percentile.yaml     ir_encoding=percentile
configs/train_rgbid_sepstem_40_16_8_clahe.yaml  ir_encoding=clahe        （D''，尚未训练）
```

### 2.2 在 `scripts/train.py` 加**硬守卫**（不再静默回落）

```python
if str(kwargs.get("use_simotm", "")) == "RGBID" and "ir_encoding" not in kwargs:
    if Path(args.train_config).name not in _LEGACY_RGBID_PERCENTILE_CONFIGS:
        raise SystemExit("[train] FATAL: RGBID 训练配置 ... 未显式声明 ir_encoding ...")
```

- **白名单**（`_LEGACY_RGBID_PERCENTILE_CONFIGS`）只含建于 `ir_encoding` 出现**之前**的历史配置，
  尤其 `train_rgbird_ir_quicktest.yaml`（**已冻结的官方 baseline，FREEZE_MANIFEST 保护，不得修改**）。
- 白名单之外的所有 RGBID 配置都必须显式声明，否则**开训第一秒就 SystemExit**。

### 2.3 新增云上自检脚本 `scripts/preflight_check_train_config.py`

用**与正式训练完全相同的解析路径**（`_build_train_kwargs`）解析配置，断言：

```
use_simotm == RGBID          channels == 5
ir_encoding 已显式声明        ir_encoding == 期望值
模型 scale == 'm'            模型可构建（打印参数量/层数/nc）
输出目录是否已存在（会 +1 递增 → 产物落错地方）
```

退出码 0 = 可开训；非 0 = **禁止开训**。

### 2.4 把已污染 run 改成**如实命名**

```
runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe
  → runs/urban_multimodal_det_yolo11_rgbid_sepstem_percentile
```

权重 / `results.csv` / `args.yaml` **一个字节未改**，只改了目录名，
并在目录内写入 `_RENAME_NOTE.md` 说明原因。
好处有二：目录名与内容一致；`..._sepstem_clahe` 这个名字**空出来给真正的 CLAHE 重跑**。

---

## 3. 拦截器验证（4 项测试，全部实测）

| # | 测试 | 期望 | 实测 |
|---|---|---|---|
| 1 | preflight on **clahe** 配置，期望 clahe | PASS | ✅ `ALL CHECKS PASSED` / exit 0 |
| 2 | preflight on **percentile** 配置，期望 clahe | FAIL | ✅ `[FAIL] ir_encoding == 'clahe' — 'percentile'` / exit 1 |
| 3 | `scripts/train.py` + RGBID 配置**缺** `ir_encoding` | 硬失败 | ✅ `SystemExit: FATAL ... 未显式声明 ir_encoding` / exit 1 |
| 4 | 冻结官方 baseline **不得**被误拦 | allowed | ✅ 白名单命中 `allowed`（另 5 个含显式键的配置也全部 `allowed`） |

测试 4 特意用**不启动训练**的方式验证白名单逻辑，避免误建 `runs/..._ir_quicktest2`。

---

## 4. 重跑前的端到端复验（dry-run）

对**修正后**的配置重跑 train-entry dry-run（1 个真实 batch 后停止）：

```text
scale=m  ch=5  nc=12  params=20,061,972
ir_encoding (from dataset)      : clahe          ← 关键
real batch input shape          : (8, 5, 1280, 1280)
remap backbone/head             : 245/245 , 287/287
stem init equality              : {rgb:True, ir:True, depth:True}
stem gradients |g|sum           : rgb 3877.65 / ir 721.21 / depth 485.04
batches=1  optimizer.step=1
三个 stem 的 bn.bias 全部改变     : True
```

> ⚠️ dry-run 输出里那两行 `IR sample vs PERCENTILE / CLAHE mean|d|` 是**已知的构造错误**
> （把 percentile 与 CLAHE 串起来算；二者实为**互斥**），**不作为判据**。
> 判据是 ①`dataset hyp.ir_encoding == 'clahe'`、②独立的 8 假设逐像素对照
> （`H6 CLAHE@原分辨率→resize` 得 `mean|d| = 0.0000`，且 visible 对照组同样为 0，证明复刻正确）。

---

## 5. 需要你处理的一件事（**未动，等你决定**）

```
runs/urban_multimodal_det_yolo11_rgbid_sepstem_40_16_8_clahe/   ← 空目录（0 个文件）
```

这是 D''（40/16/8 stem 分配）预留的目录，但**里面一个文件都没有**（训练还没跑）。
它存在会导致 D'' 开训时 `exist_ok=false` → ultralytics 自动递增到
`..._sepstem_40_16_8_clahe2`，**产物落错名字**。

preflight 已把它标为 `[WARN]`。建议 `rmdir` 掉这个空目录（非破坏性，里面确实没有文件），
但我**没有替你删** —— 请确认。

---

## 6. 重跑：云上操作顺序

```bash
# 0) 确保以下文件已同步到云（这三份是本轮的关键）
#    configs/yolo11m_sepstem.yaml
#    configs/train_rgbid_sepstem_clahe.yaml
#    scripts/train.py                       ← 含新硬守卫
#    scripts/preflight_check_train_config.py ← 新增

# 1) 训练前必须先跑自检 —— 它读的就是即将使用的那两份配置
python scripts/preflight_check_train_config.py \
    --model_config configs/yolo11m_sepstem.yaml \
    --train_config configs/train_rgbid_sepstem_clahe.yaml \
    --expect-ir-encoding clahe
#    必须看到 PREFLIGHT_READY / exit 0；否则禁止开训

# 2) 开训
python scripts/train.py \
    --model_config configs/yolo11m_sepstem.yaml \
    --train_config configs/train_rgbid_sepstem_clahe.yaml

# 3) 开训后**立刻**核对（第一分钟就能查，不要等 11 小时）
grep -n "ir_encoding" runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/args.yaml
#    必须出现：ir_encoding: clahe        ← 若整条缺失，说明守卫被绕过/代码没同步，立即停机
```

---

## 7. 状态

```text
污染成因      = 已清理（命名歧义已消除 + 硬守卫 + 云上自检脚本）
已污染 run    = 改名为 ..._sepstem_percentile（内容零改动，如实命名）
重跑目标      = runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe（名字已释放）
端到端复验    = PASS（dry-run: ir_encoding=clahe，remap 245/245 & 287/287，1 batch/1 step）
本轮训练      = 0（未启动；训练在云端由你执行）

D  online            = 0.5553   （未变更）
OFFICIAL_FALLBACK    = 0.53180  （未变更）
FREEZE_MANIFEST      = 16/17（唯一不符为既有 predict_rect.py 漂移，非本轮造成）
```
