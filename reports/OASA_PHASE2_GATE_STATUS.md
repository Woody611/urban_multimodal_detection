# OASA — STEP 1 / STEP 2 门禁状态

**日期**：2026-09-23 · **性质**：未训练、未实现 OASA；仅 STEP 1 的 .gitignore 例外 + 只读实测
**裁决**：采用**候选 B（Post-Mosaic）**

---

## STEP 1 — augment.py gitignore 风险：**PASS** ✅

**现状（改前）**：`git check-ignore -v` → `.gitignore:4:data/`；`git ls-files` → 空。
根因：`.gitignore` 第 4 行 **无锚定 `data/`**，同时命中仓库根 `data/` 与 `ultralytics/data/`。

**最小范围 exception（已实施）**：

```gitignore
data/                      # 原规则（未动语义）
!ultralytics/data/         # 放行目录本身（git 无法 re-include 被排除目录下的文件）
ultralytics/data/*         # 内容仍忽略
!ultralytics/data/augment.py   # 仅放行 OASA 需要改的这一个文件
```

**验收（实测）**：

| 检查 | 结果 |
|---|---|
| `git ls-files ultralytics/data/augment.py` | **`ultralytics/data/augment.py`** ✅ 能找到 |
| `git status --short` | **`A  ultralytics/data/augment.py`** ✅ 已进入暂存（新增文件） |
| `git diff --cached --stat` | `augment.py \| 3810 +++` ✅ 修改可见 |
| **其他 `ultralytics/data/*.py`** | base/loaders/dataset/build/utils **仍被忽略** ✅ 最小范围 |
| 仓库根 `data/` | 仍被忽略（`.gitignore:9:data/`）✅ 行为未变 |

> ⚠️ `git check-ignore -v ultralytics/data/augment.py` 仍会**打印**第 12 行那条规则 —— 那是**否定规则本身被匹配到**，
> 不代表文件被忽略。权威判据是 `git ls-files`（能找到）与 `git status`（显示 `A`）。
>
> ⚠️ **未处理的同源风险**：`ultralytics/data/{base,loaders,dataset}.py` 仍未被追踪。
> 其中 `loaders.py` 是**评测侧**预处理、`base.py` 是**训练侧**——正是 CLAHE 事故里发生分歧的那一对。
> 本次按"最小范围"只放行 augment.py；**该风险独立存在，建议另行立项**。

**云端同步**：本仓库无自动同步机制，云端靠手动上传 `ultralytics/` 整个目录。
`augment.py` 现已被 git 追踪，`git diff` 可核对；但**云端是否拿到新版仍需开训前 grep 核对**（见 §风险 R2）。

---

## STEP 2 — Post-Mosaic 几何实测：**未通过（INCOMPLETE）** ❌

### 已获得的可信结果（方法正确、样本充分、确定性）

关闭全部几何抖动（`RandomPerspective.scale/translate/degrees/shear/perspective = 0`），
只留 LetterBox，用 **rank-matched**（排序后逐框比）测量：

| 分辨率 | n_img | 几何尺度中位 | p25 | p75 |
|---|---:|---:|---:|---:|
| **1080p**（92.4% 数据） | **80/80** | **0.667** | 0.667 | 0.667 |
| **360p** | **80/80** | **2.000** | 2.000 | 2.000 |

- 0.667 = 1280/1920，2.000 = 1280/640 —— **就是纯 LetterBox 的缩放比，完全确定性**。
- ⇒ **1080p 的 small GT 在最终输入空间 = 25.53 × 0.667 = 17.02 px**。
  **这与项目既有审计的"1280 输入后 √area median ≈ 17.7 px"一致** ✅ 基线前提成立。

### ❌ 未测出的关键量：**Mosaic 的增量因子**

`mosaic ON` 时最终图含 4 张图的框，`len(final) != len(native)` 几乎必然成立 →
**80 张里只有 6 张能 rank-match**，该数字**不可信**。

**因此：Post-Mosaic OASA 能否真的带来 ~2× 最终尺度增益，尚未实测。**

### ⚠️ 顺带发现：既有审计的「mosaic 缩小 2.00×」结论可疑

- 我实测的 **mosaic ON/OFF 尺度比在可信样本上是 ≈1.0**（360p：ON=2.000 / OFF=2.000，完全一致）。
- 而审计（`rgbid_small_object_crop_audit.md` §8）断言 mosaic 使目标缩小 2.00×，并据此得出 `CROP_NO_GO`。
- 若 mosaic 实际不缩小，那么 **CROP_NO_GO 的第一条决定性论据（"crop 放大 < mosaic 缩小"）需要重检**。

> **这条分歧必须在本实验启动前解决** —— 因为 Post-Mosaic OASA 的**全部合理性**都建立在
> "mosaic 把目标压到 0.5×，所以要在 mosaic 之后再放大回来"这个前提上。
> 如果 mosaic 根本不缩小，Post-Mosaic OASA 的机制论证就不成立。

### 我自己的测量方法走过的三个坑（已修正，记录备查）

1. **总体不匹配**：拿"最终图里所有框的中位"对比"native 的 small 框中位" → 苹果比橘子（首次测量得 2.07× 的假象）。
2. **`Mosaic.p` 在构造时固化**：`ds.hyp.mosaic = 0` **完全不生效**（`Mosaic(dataset, imgsz, p=hyp.mosaic)` 在 build 时取值）。
   我前 3 轮"mosaic OFF"其实全是 mosaic ON，ON/OFF 比 ≈1.0 是这个 bug 造成的。
   **正确做法**：`ds.transforms.transforms[0].transforms[0].p = 0.0`。
3. **marker 注入法失败**：白方块被 mosaic/RandomPerspective 裁切或旋转，30 张里只有 3 张可测。

### 要把 STEP 2 做完，需要一个不同的测量方法（建议）

不得再用"框跟踪"。建议改为**直接读几何参数**：
hook `Mosaic.__call__`（记录画布尺寸）与 `LetterBox.__call__`（记录输入→输出尺寸），
对同一张图分别取 mosaic ON/OFF 的画布与 letterbox 比 → **mosaic 增量因子 = (letterbox_in_OFF / imgsz) / (letterbox_in_ON / imgsz)**。
（我尝试过 hook，但未生效——需先确认 `Compose` 持有的实例类与方法解析路径，再实施。）

---

## 门禁状态（对应你给的 10 项）

| # | 检查 | 状态 |
|---|---|---|
| 1 | augment.py 不再被 gitignore 吞掉 | ✅ **PASS** |
| 2 | git status/diff 能看到修改 | ✅ **PASS** |
| 3 | 实际训练环境读取的是修改后的 augment.py | ⬜ 未验证（需云端 grep） |
| 4 | **post-Mosaic insertion confirmed** | ⬜ **未实现** |
| 5 | **final input scale 实测** | ❌ **INCOMPLETE**（mosaic 增量因子未测出） |
| 6 | anchor retention ≥ 0.8 | ⬜ 未测 |
| 7 | bbox 合法 | ⬜ 未测 |
| 8 | RGB/IR/Depth 空间一致 | ⬜ 未测 |
| 9 | visual smoke test | ⬜ 未做 |
| 10 | 1~4 batch smoke test | ⬜ 未做 |

```text
结论：STEP 2 未通过 → 按你的门禁规则，不进入实现，不提交 A100。
阻塞项两项：
  ① mosaic 的增量尺度因子未实测（且既有审计的 2.00× 结论与之冲突，需先解决）
  ② Final-input ×2 的机制主张尚无实测支撑
本轮训练次数 = 0 ；实现 OASA 代码行数 = 0
```
