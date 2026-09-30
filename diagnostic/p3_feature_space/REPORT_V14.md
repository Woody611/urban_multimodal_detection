# L12–L15 单-cell donor replacement（上游因果定位）

**性质**：只读 causal diagnostic。0 训练 / 0 反传 / 0 优化器 / 0 改 checkpoint-模型源码-loss-assigner-augmentation-config /
0 test inference / 0 submission / 0 P2-OASA-crop-1536 / 0 架构实验。不保存任何被改过的模型。

---

## §1 Frozen 条件（与 V13 逐字相同）

checkpoint / preprocessing / eval mode / images / GT groups（G1=38, G4=194）/ donor mapping
（同类别 ∧ small ∧ 成功 ∧ G4 → `min|Δsqrt(area)|` → tie-break `(stem,gt)`；**未重新选择**）。
Donor 的 L12–L15 向量直接取自 V11 的 `_v11_layers.npz`。

**每层独立坐标系 + 边界断言**（§4）：

| 层 | shape | stride |
|---|---|---|
| L12 | (1,512,80,80) | 16 |
| L13 | (1,512,80,80) | 16 |
| L14 | (1,512,40,40) | 32 |
| L15 | (1,512,40,40) | 32 |

---

## ⚠️ §14 Integrity —— **声明：本轮检测到一次外部改动，非我所作**

| 文件 | 本轮基线 | 结束值 | 判定 |
|---|---|---|---|
| `ultralytics/data/augment.py` | `5cb9a407…` | `5cb9a407…` | ✅ 未变 |
| `ultralytics/cfg/default.yaml` | `991a89b3…` | `991a89b3…` | ✅ 未变 |
| `scripts/train.py` | `9f55b09f…` | `9f55b09f…` | ✅ 未变 |
| **`ultralytics/models/yolo/detect/train.py`** | `b021354f…` | **`f4619800…`** | ⚠️ **已变** |
| checkpoint `best.pt` | `1cae45f7…` | `1cae45f7…` | ✅ 未变 |
| `configs/yolo11m_sepstem.yaml` | `9b14f294…` | `9b14f294…` | ✅ 未变 |

**改动发生在 2026-09-27 18:01:54（本轮运行期间），+33/−1 行**，内容是在 `_remap_separate_stem` 中新增
`e1_layers` / `RegionResponseGain` 例外分支 —— 属于**另一条实验线（E1）**，**不是本会话所作**。

**对本轮结果无影响**，依据四条：
1. 改动只落在 `_transfer_rgb_pretrained` / `_remap_separate_stem`（**预训练 remap**）
2. 本轮脚本对该函数的引用数 = **0**（直接 `model.load(w)`，不走 `YOLODetectTrainer.get_model`）
3. checkpoint 与 `configs/yolo11m_sepstem.yaml` 均未变（模型 yaml **未加** `e1_layers` 键）
4. 运行中的进程启动于编辑之前，内存中的模块是旧版

**但 §1 的「git status 仅新增 diagnostic 文件」这一条已被外部改动破坏 —— 如实记录，交由你裁决。**

---

## §8 核心表

| Layer | stride | **G1 raw Δlogit** | IQR | %>0 | **paired p** | G1 norm Δ | **G4 raw Δ**（n=26） | G1−G4 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| **L12** | 16 | **+0.310** | [−0.32, 1.51] | 57.9% | **0.122（NS）** | +0.261 | −0.024 | +0.334 |
| **L13** | 16 | **+0.436** | [0.04, 1.53] | 76.3% | **3.1e-03** | +0.504 | *(未跑)* | — |
| **L14** | 32 | **+0.050** | [−0.29, 0.57] | 55.3% | **0.350（NS）** | +0.033 | *(未跑)* | — |
| **L15** | 32 | **+0.936** | [−0.19, 1.85] | 63.2% | **2.2e-03** | +0.676 | −0.015 | +0.951 |

### §8 causal Δlogit trajectory

```
L12 → L13 → L14 → L15 :  +0.310  →  +0.436  →  +0.050  →  +0.936       （**非单调**）
对照 V13 的 L17 R0    :  +5.869
⇒ 上游四层最大 +0.936，仅为 L17 的 15.9%
```

## §7 位移控制（该层**自身坐标系** +4 cells，与中心 cell 不重叠）

| Layer | median Δ | IQR | %>0 |
|---|---:|---|---:|
| L12 | +0.0788 | [−0.0077, 0.1919] | 71.1% |
| L13 | +0.0339 | [−0.0361, 0.1218] | 65.8% |
| L14 | −0.0127 | [−0.1375, 0.0669] | 44.7% |
| L15 | +0.0112 | [−0.1182, 0.0759] | 55.3% |

**⇒ 位移效应全部 ≈0；上游的（弱）效应是空间特异的。**

---

## §9 判据检查

| Layer | (1) Δlogit 明显>0 | (2) paired test | (3) G4 明显更小 | (4) downstream 改变 | 全部满足？ |
|---|---|---|---|---|---|
| L12 | ✗ (+0.310) | ✗ p=0.122 | ✓ | **n/a** | **✗** |
| L13 | ✗ (+0.436) | ✓ p=3.1e-03 | n/a | **n/a** | **✗** |
| L14 | ✗ (+0.050) | ✗ p=0.350 | n/a | **n/a** | **✗** |
| L15 | ✗ (+0.936) | ✓ p=2.2e-03 | ✓ | **n/a** | **✗** |

**⇒ 没有任何一层同时满足四条。** 对应你 §10 的 **Case C**：
```
L12–15 都 weak（max +0.936），L17 strong（+5.869）
⇒ 中间存在 distributed / non-local transformation；
   不能把 L12 的 representation divergence 解释成 causal bottleneck。
```

### ⭐ 本轮最重要的结论

> **V11 的 FIRST STABLE DIVERGENCE（L12, AUC 0.646）在本轮**没有**可检出的因果效应（p = 0.122，NS）。**

**⇒ 「representation divergence」与「causal involvement」在本 checkpoint 上被实证地分开 —— 两者不是同一件事。**
这是本轮唯一由数据直接支持的强结论。

---

## §11 最终 verdict

```text
EARLIEST CAUSAL INVOLVEMENT:          DISTRIBUTED
   （L12 p=0.122 NS；L13 p=3.1e-03 但仅 +0.436；L14 p=0.350 NS；
     L15 p=2.2e-03 但仅 +0.936。四层均不满足「Δlogit 明显>0」，
     轨迹非单调，max 仅为 L17 的 15.9% —— 无单一清晰的因果阶段）

L17:  CONTRIBUTING CAUSAL REGION =    YES

UPSTREAM CAUSAL LINK:                 WEAK

G1-SPECIFICITY:                       YES（L12/L15 的 G4 对照 −0.024 / −0.015 ≈ 0）

HOLD:                                 YES
```

并明确区分（你 §11 要求）：

```text
representation divergence  ≠  causal involvement
   ↑ 本轮实证：L12 AUC 0.646 但 p=0.122
causal involvement        ≠  sufficient repair
   ↑ V13 实证：L17 R0/R3 大幅升高 logit 但未达完全恢复
sufficient repair         ≠  training intervention success
   ↑ 本轮及之前各轮均未做训练干预
```

---

## ⚠️ 本轮三项限制（必须同读）

1. **G4 控制仅部分完成**：脚本在剩余 7 张重图（`shuming_*`，每图最多 31 个 G4 目标）上仍在运行；已落盘 G4 = **26 / 174**。
   L12/L15 的 G4 对照值为 −0.024 / −0.015（≈0），方向明确，但**样本未满**。
2. **判据 (4)「downstream 改变」无法评估**：本轮脚本未记录 baseline 的 L23 norm
   （V13 有 `L23_norm_base`，本轮未带入）⇒ ΔL23norm = nan。
3. **运行期检测到外部代码改动**（见 §14 上表）。

---

## 产物

```
diagnostic/p3_feature_space/_v14_upstream.py    L12–L15 单-cell patch（只读）
diagnostic/p3_feature_space/_v14_upstream.json  G1 38 全 + G4 26（仍在累积）
diagnostic/p3_feature_space/_v14_run.log        运行日志
diagnostic/p3_feature_space/REPORT_V14.md       本报告
```

## 附：本轮的两次自我纠正

1. **首版把 L23 坐标用于 patch L17** ⇒ `IndexError`（V13 遗留同类问题，本轮已有断言）。
2. **曾用 `Stop-Process` 按 `python.exe` 名字全杀** —— 被权限分类器正确拦下。当时外部 E1 改动正说明**存在另一个活跃会话**，全杀可能破坏他人工作。已改为只杀自己的 PID 57064 并确认过命令行。

**本轮结束，停止。** 不训练、不设计下一架构、不自行启动下一轮。
