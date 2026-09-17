# 最终方案冻结审计报告

**日期**：2026-09-17
**性质**：只读归档检查。**未训练、未推理、未上传、未提交、未修改任何冻结文件、未生成新提交包、未删除任何历史结果。**
**校验清单**：`reports/FREEZE_MANIFEST.sha256`（只读，可用于日后核验产物是否被意外修改）

---

## A. 正式线上成绩

| 方案 | checkpoint | 线上 mAP50-95 |
|---|---|---:|
| **RGBID Early Fusion**（正式保底） | `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt` | **0.53180** |
| F4 RGBD（备选基准） | `runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt` | 0.51051 |

线上提升：RGBID 相对 F4 **+0.02129**。

---

## B. 当前正式提交候选

| 方案 | 状态 | 说明 |
|---|---|---|
| **RGBID Early Fusion** | ✅ **正式保底方案** | 线上 0.53180，唯一正式提交候选 |
| Ensemble 3:1 | ❌ **不提交** | 官方本地口径 +0.00568，但额外 +2100 FP；fork 口径 −0.01278；用户 2026-09-17 裁定不提交 |
| TTA（HFlip / MultiScale） | ❌ **不提交** | 官方口径 −0.00636 / +0.00432；fork 口径均负；一级判据未通过 |
| 1536 训练候选 | ❌ **不提交（且未验证）** | 从未训练、从未评估，仅作为未验证方向记录 |

---

## C. 文件完整性

### C.1 关键产物

| 文件 | 存在 | 字节 | mtime | SHA256 |
|---|---|---|---|---|
| RGBID best.pt | ✅ | 40,635,237 | 2026-09-17 06:01:16 | `f9dddbfaca4cb08bde31d082eb0a54d499b8441eb1f97c6134a028caa6d83e55` |
| F4 best.pt | ✅ | 61,331,325 | 2026-09-15 07:25:22 | `989c449ad7a2298a64fd7d43ef2763b0ce777b948dcb754486081b3c93fccfac` |
| RGBID 正式提交 zip | ✅ | 289,016 | 2026-09-17 06:37:58 | `0c550773c86561089a9fe70ac020ecb67e12febe70093bbb7686aa97d1f756ce` |
| Ensemble 3:1 候选 zip | ✅ | 340,634 | 2026-09-17 13:23:26 | `cee8e90205abeefc2b8593c5ff30dbf49634a4bcfa212b599978274d0ac75d97` |
| **冻结** `scripts/predict_rect.py` | ✅ | 11,286 | **2026-09-15 22:34:28** | `656aa58789a467e003517a3f6ca1f05ab25dc065cf5861a8df66468f90528be8` |
| `scripts/predict.py` | ✅ | 20,269 | 2026-09-15 12:54:23 | `ffc1edd08aa0d22b2e007ef76a83b7358ddb12e623aeafd1788a7d11236e1f13` |

> **是否被修改**：**否**。
> `predict_rect.py` 的 mtime 为 **2026-09-15 22:34:28**，早于本会话开始时间（2026-09-17 上午），
> 且 `git diff` 对其无 tracked 修改。两个 checkpoint 的 mtime 分别为 09-17 06:01（训练产出时刻）
> 与 09-15 07:25（训练产出时刻），均早于会话开始，**本次会话未写入任何权重**。

### C.2 提交 zip 内容校验

| 项 | RGBID 正式提交 | Ensemble 3:1 候选 |
|---|---|---|
| zip 是否存在 | ✅ | ✅ |
| zip 内条目数 | **1000** | **1000** |
| CRC 完整性 | ✅ PASS | ✅ PASS |
| zip 内命名与 `results/` 一致 | ✅ PASS | ✅ PASS |
| zip 内命名与测试源图逐一对应 | ✅ PASS | ✅ PASS |
| results 目录 TXT 数 | 1000 / 1000 | 1000 / 1000 |
| **重复文件名** | **0** | **0** |
| **遗漏图片** | **0** | **0** |
| 多余图片 | 0 | 0 |
| 空文件数量 | 6 | 2 |
| 总检测框数 | 7370 | 10212 |
| TXT 格式非法 | 0 | 0 |
| 非法类别 (∉[0,11]) | **0** | **0** |
| 坐标越界 [0,1] | **0** | **0** |
| 非法 confidence (>1) | **0** | **0** |
| NaN / Inf | 0 | 0 |
| conf 最大值 | 0.981424 | 0.981424 |
| **退化框 (w/h ≤ 0)** | **155** | **226** |

**关于退化框**：155 行（正式）与 226 行（候选）均为**冻结 pipeline 的既有行为**
（`predict.py::_format_lines` 对 `scale_boxes` 裁到边界外的框不设最小边长），
conf 均 ≤ 0.047，排在置信度末尾。**用户已于 2026-09-17 裁定接受、不修复**
（理由：历史线上有效提交同样含此类记录，修复需改动冻结代码、会破坏已验证提交链条）。
详见 `reports/rgbid_f4_ensemble_3to1_final_audit.md` §5.3。

### C.3 仓库变更状态

`git status` 与**本会话开始时的快照一致**：仅 2 个 tracked 文件为 `M`
（`scripts/train.py`、`ultralytics/models/yolo/detect/train.py`，**均为会话前既存改动**，
本会话未再触碰），其余均为 untracked。

本会话新增的全部为**审计脚本与报告**，不含任何冻结文件：
- 脚本：`complementarity_analysis.py`、`conf_sensitivity_scan.py`、`ensemble_final_candidate.py`、
  `inference_audit.py`、`inference_gain_summary.py`、`make_candidate_submission.py`、
  `rgbid_error_analysis.py`、`validate_submission_txt.py`
- 报告：`rgbid_inference_gain_audit.md`、`rgbid_f4_ensemble_3to1_final_audit.md`、
  `next_stage_decision_audit.md`、`rgbid_error_analysis.{json,txt}`、`FREEZE_MANIFEST.sha256`
- 数据：`diagnostic/rgbid_inference_gain/`（推理产物）、`submissions/rgbid_f4_ensemble_3to1_candidate/`

---

## D. 冻结文件清单

以下产物**一律冻结，不得再修改**：

| # | 冻结对象 | 路径 | 说明 |
|---|---|---|---|
| 1 | **冻结推理脚本** | `scripts/predict_rect.py` | 提交 pipeline 核心，SHA256 `656aa587…528be8` |
| 2 | **RGBID 权重** | `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt` | 正式线上方案，SHA256 `f9dddbfa…d83e55` |
| 3 | **F4 权重** | `runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt` | 备选基准，SHA256 `989c449a…3fccfac` |
| 4 | **正式 RGBID 提交产物** | `submissions/rgbird_ir_quicktest/submission.zip` + `results/` | 线上 0.53180 对应版本，SHA256 `0c550773…1f756ce` |
| 5 | **Ensemble 3:1 候选包** | `submissions/rgbid_f4_ensemble_3to1_candidate/submission.zip` | 离线研究候选，**保留但不提交**，SHA256 `cee8e902…c75d97` |

附带冻结（同为提交链条组成部分）：
`scripts/predict.py`、`configs/train_rgbird_ir_quicktest.yaml`、`configs/yolo11m_earlyfusion.yaml`、
`configs/train_f4_1280.yaml` —— 校验值同见 `reports/FREEZE_MANIFEST.sha256`。

---

## E. 风险记录

| # | 风险 | 状态 / 说明 |
|---|---|---|
| 1 | **官方评分器未获得独立脚本校验** | 仓库内**不存在**赛题官方评分脚本。`scripts/official_map.py` 是按赛题文字转写的重实现，**与真实评分器是否逐位一致未经证实**。本阶段全部"官方口径"数字均建立在此实现之上。 |
| 2 | **剩余提交次数未知** | 仓库内无任何提交次数记录，**无法确认**。无法评估试错代价。 |
| 3 | **覆盖 / 撤回规则未知** | 仓库内无相关说明，**无法确认**。若提交不可撤回，则任何新提交都是单向下行风险。 |
| 4 | **Ensemble 3:1 的 FP 增长** | 官方本地口径虽为正（+0.00568），但换取 +108 TP 的代价是 **+2100 FP**（框数 4824→7031，Precision 0.83887→0.79877）。增益幅度（+0.00568）远小于已知的本地—线上系统偏移（≈+0.020），该交易在测试集上是否同样成立**未经测试集验证**。 |
| 5 | **Ensemble 在 fork 口径下为负** | 同一方案 fork 口径为 **−0.01278**，逐类几乎全降。两口径符号相反且阈值鲁棒（见 `reports/rgbid_inference_gain_audit.md` §6.3）。若真实评分器更接近 fork 口径，该方案会掉分。 |
| 6 | **1536 尚未验证** | 分辨率 1536 从未训练、从未评估。它是唯一未被历史实验消耗的杠杆（1280 已用、容量 F5 已证伪、P2 两次证伪），但收益不确定且成本高。**目前不做。** |
| 7 | **当前不继续试错** | 推理侧已证无安全增益（`reports/next_stage_decision_audit.md` §4：现有框集合 TP=2288 已是上限，任何后处理都无法新增真阳性）；训练侧杠杆大多已消耗。结合风险 2、3，**继续试错的期望收益为负**。 |

---

## F. 本轮冻结结论

> **正式方案**：RGBID Early Fusion，线上 **0.53180**，保持不动。
> **备选**：F4 RGBD，线上 0.51051，保持不动。
> **不提交**：Ensemble 3:1、TTA、1536 候选。
> **不进行任何新实验。**

本阶段执行的动作**仅限只读检查**，外加新建一个只读校验清单
`reports/FREEZE_MANIFEST.sha256` 与一份归档报告（本文件）。

**未执行**：训练 / resume / finetune / 1536 实验 / 新 TTA / 新 Ensemble / conf 搜索 /
IoU 搜索 / max_det 搜索 / 修改模型结构 / 修改 `predict_rect.py` / 修改冻结 submission pipeline /
上传 / 线上提交 / 自动轮询 / 自动执行后续任务。

**未删除**任何历史实验目录，**未覆盖**任何 best.pt，**未生成**新的提交包。

**等待用户新的明确指令。**
