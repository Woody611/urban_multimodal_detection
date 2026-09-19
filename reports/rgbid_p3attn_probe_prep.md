# P3 Identity Attention Probe —— 审查与准备报告

**日期**：2026-09-18
**性质**：**准备阶段**。已完成代码审查、实现、单元测试、权重迁移验证、冻结参数验证、初始一致性验证、配置与命令准备。
**状态**：**未训练**。等待你回复「批准启动 Probe」。正式保底全程冻结。

---

## 0. 结论速览

| 项 | 结果 |
|---|---|
| 原位插入、层索引不变 | ✅ 顶层模块数 **24 不变**；Detect 仍在 23 |
| Detect 输入通道 / stride / 输出 shape | ✅ `[256,512,512]` / `[8,16,32]` **不变** |
| **identity 初始化误差** | ✅ **0.000e+00（逐位精确恒等）** |
| 权重迁移 | ✅ 从正式 best.pt **649/651**，缺失的恰为新增 attn 2 个权重 |
| **「只训练新增 attention」** | ✅ **可实现，且无需改 trainer** —— 配置里一个 `freeze` 列表即可，实测 trainable = **2 个张量 / 65,792 参数（0.3268%）** |
| 初始输出一致性 | ✅ layer16 输出 / Detect 输入 / 最终预测 **全部 0.000e+00** |
| 单元测试 | ✅ **11/11 PASS，0 FAIL** |
| 剩余风险 | ⚠️ 冻结层（含 backbone）的 **BN running stats 仍会更新**（框架行为）——见 §5.3，已有事后只读诊断方案 |
| 训练 | ❌ **未启动** |

---

## 1. 代码修改清单（§7-1）

**改动 2 个文件，全部为纯增量，默认关闭时行为与原代码逐字一致。**

### 1.1 `ultralytics/nn/modules/block.py` —— `+31 / −1`

| 位置 | 改动 |
|---|---|
| `block.py:737` 之前 | **新增类 `P3IdentityAttn`**（+22 行） |
| `block.py:740` | `C3k2.__init__` 签名末尾**新增带默认值的参数** `attn=False` |
| `block.py:746` | 新增 `self.attn = P3IdentityAttn(c2) if attn else None` |
| `block.py:748` | **新增 `C3k2.forward`**（原类无 forward，继承 `C2f.forward`）：`x = super().forward(x); return self.attn(x) if self.attn is not None else x` |

**向后兼容性证据**：
- 全仓库（`configs/*.yaml` + `ultralytics/cfg/models/11/*.yaml`）中 **C3k2 最多只用到 3 个 yaml 参数槽位** → 增加第 8 个带默认值的构造参数不可能触碰任何既有配置。
- `attn=False` 时 `self.attn is None`（不是子模块，不产生任何 state_dict 键）、`forward` 返回 `super().forward(x)` 原值。
- **回归实测**：baseline 模型键数仍为 **649**；对 `yolo11m.pt` 迁移仍为 **642/649**；`_transfer_rgb_pretrained` 仍返回 **5** —— 与本会话改动前记录的数值完全一致。

### 1.2 `scripts/train.py` —— `+6 / −0`

在 `_build_train_kwargs` 中 `modality_dropout` 转发之后，新增：

```python
# 冻结层（可选）：仅在配置中存在时转发。ultralytics 的 freeze 语义是
# "冻结名字含 model.<x>. 的参数"，x 可混用层序号(int)与子模块路径(str)，
# 因此可用混合列表精确表达"只训练某几个新增参数"。未配置时行为不变。
if "freeze" in train_cfg:
    kwargs["freeze"] = train_cfg["freeze"]
```

**未配置 `freeze` 时行为不变** —— 既有全部实验配置都不含该键。

### 1.3 明确未改动

`ultralytics/nn/tasks.py`、`ultralytics/engine/trainer.py`、`ultralytics/cfg/default.yaml`、
`ultralytics/data/*`、`configs/yolo11m_earlyfusion.yaml`、`configs/train_rgbird_ir_quicktest.yaml`、任何 checkpoint、任何提交文件 —— **一字未改**。

> 说明：规划阶段曾判断「新模块类必须注册进 `parse_model` 的 `base_modules`/`repeat_modules` 两个 frozenset」。
> 实际实施改为**给现成 `C3k2` 增加一个默认参数**，因此**完全不需要动 `tasks.py`** —— 比原计划更小、更安全。

---

## 2. Identity Attention 实现（§7-2）

```python
class P3IdentityAttn(nn.Module):
    """P3 channel recalibration with an exact identity initialization.

    ``out = x * (1 + tanh(fc(avgpool(x))))`` with ``fc`` zero-initialized gives ``out == x``
    bit-exactly at t=0, while the gradient w.r.t. ``fc`` stays non-zero
    (``d(out)/d(fc.weight) = x * avgpool(x) * (1 - tanh^2(0)) = x * avgpool(x)``).
    This is deliberate: ``x * sigmoid(fc(...))`` zero-initialized starts at ``0.5x``
    and would perturb the pretrained P3 feature before training even begins.
    Complexity is O(HW); no ``HW x HW`` attention map.
    """

    def __init__(self, c):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(c, c, 1, 1, 0, bias=True)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return x * (1.0 + torch.tanh(self.fc(self.pool(x))))
```

**为什么是 `1 + tanh` 而不是 `sigmoid`**：
`fc` 零初始化 → `sigmoid(0) = 0.5` → `out = 0.5x`，**会把 P3 特征直接减半**（不是恒等）。
`1 + tanh(0) = 1.0` → `out = x * 1.0`，IEEE-754 下 `x * 1.0` 与 `x` **逐位相同**。
门控范围 `(0, 2)`，可同时上调与下调通道。

**梯度活性**：`d(out)/d(fc.weight) = x · avgpool(x) · (1 − tanh²(0)) = x · avgpool(x) ≠ 0`。
不是 `x * 0` 那种死门控。

**时间复杂度 O(HW)**：全局池化 + 1×1 卷积，**不产生任何 `HW × HW` 张量**。

---

## 3. 单元测试结果（§7-3）—— 11/11 PASS，0 FAIL

```
§2 Identity Attention —— 单元测试
  [PASS] A1 输出 shape 与输入完全一致   (2, 256, 40, 40)
  [PASS] A2 dtype 一致   torch.float32
  [PASS] A3 device 一致   cpu
  [PASS] A4 初始为精确恒等 max|x*-x| < 1e-6   max_abs_diff = 0.000e+00
  [PASS] A5 无 NaN/Inf   nan=0 inf=0
  [PASS] A6 初始门控 ≡ 1（非 sigmoid 的 0.5）   gate range [1.0000,1.0000]
  [PASS] A7 fc.weight 梯度存在且非零   |grad|=5.8602e+04
  [PASS] A8 fc.bias   梯度存在且非零   |grad|=1.0834e+04
  [PASS] A9 对输入的梯度非零（不是死门）   |grad|=8.1920e+05
  [PASS] A10 参数被扰动后输出改变（模块有效）   max_abs_diff = 1.052e-01
  [PASS] A11 对照：现成 ChannelAttention 零初始化 = 0.5x（故不可直接用）   max|ref-0.5x| = 0.000e+00
```

A4 的 `0.000e+00` 是**逐位精确**（不是「小于阈值」）；A11 是反面证据 —— 证明为什么没有直接复用 `conv.py:282` 的现成通道注意力。

---

## 4. 权重迁移结果（§7-4）

源 = **正式保底 `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt`**（5ch / nc12 / ep237）。

```
§3 权重迁移
  Transferred 642/649 items from pretrained weights          <- baseline 对 yolo11m.pt（回归对照）
  [PASS] B1  baseline 键数仍为 649
  [PASS] B2  baseline 对 yolo11m.pt 迁移数仍为 642/649（与改动前一致）
  [PASS] B3  baseline 的 _transfer_rgb_pretrained 仍返回 5
  Transferred 649/651 items from pretrained weights          <- Probe 对正式 best.pt
  [PASS] B4  attn 模型键数 = 651
  [PASS] B5  匹配 649/651
  [PASS] B6  未匹配项全部是新增 attn 权重
             attn=['model.16.attn.fc.bias','model.16.attn.fc.weight']   other=[]
  [PASS] B7  Detect head 权重完整匹配（121 个键全部匹配）
  [PASS] B8  backbone 前 4 层完整匹配（72 个键，无大面积随机初始化）
  [PASS] B9  层索引不变（顶层模块数 24，Detect 仍在 23）
  [PASS] B10 层 16 非 attn 的 key 与 baseline 逐字一致（54 个键）
  [PASS] B11 Detect 输入通道仍为 [256,512,512]
  [PASS] B12 stride 仍为 [8,16,32]
  [PASS] B13 参数增量 < 1%   Δ=+65,792 (+0.3279%)
```

**即：原有权重 100% 匹配，唯一未匹配的就是新增注意力本身；不存在 backbone/head 大面积随机初始化；层索引与 Detect 结构完全不变。**

---

## 5. 冻结参数结果（§7-5）

### 5.1 框架能力核查（实读 `ultralytics/engine/trainer.py:239-258`）

```python
freeze_list = (self.args.freeze if isinstance(self.args.freeze, list)
               else range(self.args.freeze) if isinstance(self.args.freeze, int) else [])
always_freeze_names = [".dfl"]
freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
for k, v in self.model.named_parameters():
    if any(x in k for x in freeze_layer_names):
        v.requires_grad = False
    elif not v.requires_grad and v.dtype.is_floating_point:
        v.requires_grad = True          # ← 强制把未列出的参数设回可训练
```

**关键**：`freeze` 走的是**参数名子串匹配** `f"model.{x}."`，而 `x` 可以是 **str**。
因此可以传**混合列表**，用 `'16.cv1'` / `'16.cv2'` / `'16.m'` 精确冻结层 16 的既有子模块，
**只留下 `model.16.attn.*` 不匹配**。

另外确认：`build_optimizer`（`trainer.py:791`）**不按 `requires_grad` 过滤**，会把所有参数放进优化器分组；
但 `requires_grad=False` 的参数 `.grad = None`，SGD 直接跳过（不含 weight decay）→ **冻结有效**。

### 5.2 配置与实测（用配置里真实的 freeze 列表复刻 trainer 逻辑）

配置中的列表：
```yaml
freeze: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,'16.cv1','16.cv2','16.m',17,18,19,20,21,22,23]
```

实测结果：

```
trainable 张量 = 2 个, 元素 = 65,792 / 20,129,204 (0.3268%)
   model.16.attn.fc.weight   (65,536)
   model.16.attn.fc.bias        (256)
-> 恰好只有 attn: True
frozen 张量 = 331 个（含 backbone / neck / head / Detect / DFL）
```

**✅ 「只有新增 attention 参数 trainable」已达成，且 `ultralytics/engine/trainer.py` 一字未改。**

### 5.3 ⚠️ 必须记录的残留风险：冻结层（含 backbone）的 BN running stats 仍会更新

`freeze()` 只设 `requires_grad=False`，**不把 BN 置为 `eval()`**；而每个 epoch 开头 `self.model.train()`
会把**所有** BN 置回训练模式 → **被冻结层的 `running_mean/running_var` 仍随批次更新**。

影响界定（须如实说明）：
- 训练期的前向用**批统计**，因此**反传到 attn 的梯度不受影响**；
- 但**推理期用 running stats** → 被冻结 backbone 的 BN 统计会在 10 个 epoch 内发生漂移，
  理论上**最终官方成绩的差异可能部分来自 BN 漂移，而非 attn**。

**处置（不改变实验，只加诊断）**：训练结束后做一次**只读事后诊断** ——
逐层比较 Probe `best.pt` 与正式 `best.pt` 的 BN `running_mean/running_var` 相对漂移：
- 漂移可忽略（如层最大相对漂移 < 1e-3）→ attn 归因成立；
- 漂移显著 → 判定为「BN 漂移主导」，本次探针**不能**作为 attn 有效的证据。

该诊断纯读取两个 checkpoint，不需训练、不改配置。**若你希望在本阶段就消除该风险，
唯一办法是改 `trainer.py` 让冻结层 BN 进入 eval —— 那属于额外源码改动，需你单独批准。**

---

## 6. 初始输出一致性结果（§7-6）

两个模型**加载同一份正式 `best.pt`**，输入**同一个** `torch.rand(1,5,1280,1280)`（seed=0），eval 模式：

```
layer 16 输出      shape=(1, 256, 160, 160)                        max_abs_err = 0.000e+00
Detect 输入        3 个张量 [(1,76,160,160),(1,76,80,80),(1,76,40,40)]  max_abs_err = 0.000e+00
最终预测输出       [(1,16,33600), [(1,76,160,160),...]]            max_abs_err = 0.000e+00
                                                                  max_rel_err = 0.000e+00
  [PASS] C1/C2/C3 逐值一致 (<1e-6)   [PASS] C4 无 NaN/Inf
```

**三处全部逐位相同（0.000e+00）** → identity 初始化使 Probe 在 t=0 时**功能上完全等于 baseline**，
因此任何后续变化都只能来自被训练的 65,792 个 attn 参数。

---

## 7. 训练命令（§7-7）

```bash
cd /root/autodl-tmp/urban_multimodal_detection
export OMP_NUM_THREADS=8

test ! -d runs/urban_multimodal_det_yolo11_rgbid_p3attn_probe \
  && echo "OK 输出目录不存在" || { echo "!! 已存在，中止"; exit 1; }

python scripts/train.py \
  --model_config configs/yolo11m_earlyfusion_p3attn.yaml \
  --train_config configs/train_rgbid_p3attn_probe.yaml \
  2>&1 | tee p3attn_probe_train.log
```

需要先同步到云端：`configs/yolo11m_earlyfusion_p3attn.yaml`、`configs/train_rgbid_p3attn_probe.yaml`、
`ultralytics/nn/modules/block.py`、`scripts/train.py`。

**训练配置（与 baseline 的差异，10 项，逐项说明）**：

| key | baseline | Probe | 性质 |
|---|---|---|---|
| `epochs` | 300 | **10** | 第一阶段短探针（你的指定值） |
| `lr0` | 0.005 | **1e-4** | 只训练新增参数，用微调级 LR（你的指定值） |
| `warmup_epochs` | 3.0 | **0.0** | 你的指定值 |
| `freeze` | 缺省 | **混合列表** | 实现「只训练 attn」 |
| `pretrained` | `yolo11m.pt` | **正式 best.pt** | 从保底权重初始化 |
| `name` | `…_rgbird_ir_quicktest` | `…_rgbid_p3attn_probe` | 输出目录 |
| `box` / `cls` / `dfl` | 缺省(=7.5/0.5/1.5) | 显式 7.5/0.5/1.5 | **取值与 baseline 默认值完全相同**，仅为审计显式写出 |
| `close_mosaic` | 缺省(=10) | 显式 10 | **取值与 baseline 默认值相同** |

**未改动**：`imgsz=[1280,1280]`、`batch_size=8`、`optimizer=SGD(m=0.937, wd=5e-4)`、`lrf=0.01`、
`cos_lr`、`nbs=64`、`seed=42`、`deterministic=true`、`amp`、`channels=5`、`use_simotm=RGBID`、
数据切分、数据管线、loss、augmentation（除显式写出的 close_mosaic=10，取值不变）。

> **成本预估**：10 epoch @1280 batch8 ≈ **22 分钟**（baseline 实测 130.9 s/epoch）。

---

## 8. 官方评测命令（§7-8）

与 baseline **逐字同参**，仅 `--weights` 指向 Probe：

```bash
python scripts/predict_rect.py \
  --weights runs/urban_multimodal_det_yolo11_rgbid_p3attn_probe/weights/best.pt \
  --train_config configs/train_rgbid_p3attn_probe.yaml \
  --source data/processed/rgbid_split/images/val/visible \
  --mode rect --imgsz 1280 --conf 0.001 --iou 0.7 --max_det 300 --max_boxes 300 \
  --batch 16 --output diagnostic/p3attn_probe/probe_best_full

python scripts/official_map.py \
  --results diagnostic/p3attn_probe/probe_best_full/results \
  --split_root data/processed/rgbid_split

python scripts/modality_dropout_official_eval.py \
  --results diagnostic/p3attn_probe/probe_best_full/results \
  --split_root data/processed/rgbid_split --tag "P3IdentityAttn-probe" \
  --json diagnostic/p3attn_probe/eval_probe_best.json     # 含 Small/Medium/Large recall
```

**Small/Medium/Large AP50-95** 用既有 `diagnostic/final_bottleneck_audit/_analyze.py` 的尺寸分桶逻辑计算
（baseline 对照值：**small 0.05625 / medium 0.17984 / large 0.61711**）。

**判据（预先固定）**：
| official mAP50-95 | 判定 |
|---|---|
| ≤ 0.50928 | `P3ATTN_REJECTED` |
| 0.50928 < x < 0.51428 | `P3ATTN_NO_MEANINGFUL_GAIN` |
| ≥ 0.51428 | `P3ATTN_CANDIDATE` |
**并且必须同时看 small AP50-95 是否高于 0.05625** —— 否则即使总分微升也判为「未解决瓶颈」。

---

## 9. 回滚方式（§7-9）

**全部改动可精确回滚，不影响正式产物：**

```bash
# 1) 源码（两个文件均被 git 跟踪）
git checkout -- ultralytics/nn/modules/block.py scripts/train.py

# 2) 新增的 2 个配置
rm -f configs/yolo11m_earlyfusion_p3attn.yaml configs/train_rgbid_p3attn_probe.yaml

# 3) 实验产出与诊断产物
rm -rf runs/urban_multimodal_det_yolo11_rgbid_p3attn_probe
rm -rf diagnostic/p3attn_probe
```

**回滚后校验**：`block.py` 应回到 `d5160d4`、`train.py` 回到其改动前版本；
`configs/yolo11m_earlyfusion.yaml` = `edc95f4665ead847…`、`configs/train_rgbird_ir_quicktest.yaml` = `0fb7b43c5702cc83…`、
`best.pt` = `f9dddbfaca4cb08b…` —— 本轮已复核，三者**均未改变**。

---

## 10. 预计输出目录（§7-10）

```
runs/urban_multimodal_det_yolo11_rgbid_p3attn_probe/       <- 训练输出（新目录，与正式目录完全隔离）
diagnostic/p3attn_probe/probe_best_full/                    <- 官方评测预测
diagnostic/p3attn_probe/eval_probe_best.json                <- 官方口径指标（含 S/M/L）
p3attn_probe_train.log                                      <- 云端训练日志
```

---

## 11. 一处需要你知悉的期望管理（诚实说明）

本探针的**可训练容量非常小**：`model.16.attn.fc` 只有 65,536 + 256 = 65,792 个参数，
作用是给 P3 的 256 个通道各学一个标量门控（范围 (0,2)）。

- 它的**主要价值是判定性的**：确认「P3 通道重标定」这个方向是否存在**任何**可学到的增益；
- **不是**一个有把握的精度提升手段 —— **没有当前项目实验证据，收益未知**；
- 最接近的一次小目标定向结构干预（E5 = +P2 检测层）实测 **+0.00013**。

若 10 epoch 后官方 mAP50-95 落在 `P3ATTN_NO_MEANINGFUL_GAIN` 区间，且 small AP 无改善，
则按 §8 判据应直接结案，**不进入更长训练**。

---

## 12. 合规状态

| 项 | 状态 |
|---|---|
| 训练 | ❌ **未启动** |
| 正式目录 `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/` | ✅ 未触碰 |
| `best.pt` / `last.pt` / `submission.zip` | ✅ SHA256 未变 |
| `configs/yolo11m_earlyfusion.yaml` / `configs/train_rgbird_ir_quicktest.yaml` | ✅ SHA256 未变 |
| `ultralytics/nn/tasks.py` / `engine/trainer.py` / `cfg/default.yaml` / `data/*` | ✅ 一字未改 |
| 线上提交 | ❌ 未提交 |
| 后台进程 | ❌ 未启动 |
| 本轮新增 | `reports/rgbid_p3attn_probe_prep.md`、`configs/yolo11m_earlyfusion_p3attn.yaml`、`configs/train_rgbid_p3attn_probe.yaml`、`diagnostic/p3attn_probe/`、对 `block.py`(+31/−1) 与 `train.py`(+6/−0) 的增量改动 |

**等待你回复「批准启动 Probe」。在此之前：不训练、不修改正式目录、不提交线上、不启动后台进程。**
