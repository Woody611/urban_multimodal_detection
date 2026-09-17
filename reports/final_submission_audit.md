# 最终提交审计报告（Submission Freeze Audit）

**日期**：2026-09-17
**比赛截止**：2026-09-20 20:00
**性质**：冻结与提交准备。**未训练、未推理、未上传、未提交、未修改任何冻结文件、未改动任何提交内容、未删除任何历史结果。**
**校验清单**：`reports/FREEZE_MANIFEST.sha256`（17 个关键对象，含校验命令）

---

## 0. ⚠️ 路径更正（必须首先说明）

任务书中给出的 RGBID 路径为：

```
runs/urban_multimodal_det_yolo11_rgbid_earlyfusion/weights/best.pt
```

**该路径在仓库中不存在。** 已核验 `ls`：`No such file or directory`；
`runs/` 下也不存在任何 `rgbid*` 命名的独立目录。

按任务书「如果实际路径不同，请以仓库现有正式 RGBID best.pt 为准」，
本报告的正式路径为：

```
runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt
```

**佐证该权重即线上 0.53180 对应版本**（三条独立证据）：
1. `results.csv` 最优行为 **epoch 237，mAP50-95 = 0.55800**，与线上成绩分析记录一致；
2. `diagnostic/final_sprint/rgbid_online_score_analysis.md` §2 已逐项确认
   「线上 53.18 → 该 zip → 该 best.pt」链条无断点；
3. 该 zip 的生成脚本为冻结的 `scripts/predict_rect.py --mode rect`。

---

## 1. 当前正式模型与成绩

| 项 | 值 |
|---|---|
| **当前正式模型** | **RGBID Early Fusion**（5ch `[B,G,R,IR,D]` 早期融合） |
| 权重路径 | `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt` |
| **对应线上成绩** | **mAP@0.5:0.95 = 0.53180** |
| 备用模型 | F4 RGBD（4ch 中期融合） |
| 备用路径 | `runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt` |
| **备用线上成绩** | **mAP@0.5:0.95 = 0.51051** |

---

## 2. 最终冻结清单（阶段 A）

| # | 对象 | 存在 | 字节 | 修改时间 | SHA256 |
|---|---|---|---:|---|---|
| 1 | **RGBID 正式 best.pt** | ✅ | 40,635,237 | 2026-09-17 06:01:16 | `f9dddbfaca4cb08bde31d082eb0a54d499b8441eb1f97c6134a028caa6d83e55` |
| 2 | **F4 备用 best.pt** | ✅ | 61,331,325 | 2026-09-15 07:25:22 | `989c449ad7a2298a64fd7d43ef2763b0ce777b948dcb754486081b3c93fccfac` |
| 3 | **RGBID 正式 submission.zip** | ✅ | 289,016 | 2026-09-17 06:37:58 | `0c550773c86561089a9fe70ac020ecb67e12febe70093bbb7686aa97d1f756ce` |
| 4 | Ensemble 3:1 候选 zip | ✅ | 340,634 | 2026-09-17 13:23:26 | `cee8e90205abeefc2b8593c5ff30dbf49634a4bcfa212b599978274d0ac75d97` |
| 5 | **冻结** `scripts/predict_rect.py` | ✅ | 11,286 | **2026-09-15 22:34:28** | `656aa58789a467e003517a3f6ca1f05ab25dc065cf5861a8df66468f90528be8` |
| 6 | `scripts/predict.py` | ✅ | 20,269 | 2026-09-15 12:54:23 | `ffc1edd08aa0d22b2e007ef76a83b7358ddb12e623aeafd1788a7d11236e1f13` |
| 7 | 类别配置 `configs/dataset.yaml` | ✅ | 3,338 | 2026-08-25 19:37:05 | `fd326f30f53a0e7abcb974e44c48891e9eef041d38967477cb01a32b99466599` |
| 8 | RGBID 切分 `data/processed/rgbid_split/dataset.yaml` | ✅ | 293 | 2026-09-02 20:37:33 | `b30032e74fd95b9301c56a8a290927883a8efbd3c510814eb41bc4f89879cecf` |
| 9 | RGBID 训练配置 | ✅ | 2,583 | 2026-09-16 17:38:37 | `0fb7b43c5702cc833ec13ff0b1c5c0d7a1cab44fa687101065d5a92450430db1` |
| 10 | RGBID 模型配置 | ✅ | 2,685 | 2026-09-16 17:37:42 | `edc95f4665ead847e759b5dc7c871ddc4090f20186b9718bb3cfd8a0b76d9399` |
| 11 | RGBID `args.yaml` | ✅ | 1,837 | 2026-09-17 06:00:55 | `d074a44cb374f06ab3abd41e8455cd72d14ac7bcf31efcffa2419e94fc1291b2` |
| 12 | RGBID `results.csv` | ✅ | 39,742 | 2026-09-17 06:00:58 | `29125f32715688f94d20960767518d3c1a2b93485041858edf9705e90ba18361` |
| 13 | F4 `args.yaml` | ✅ | 1,833 | 2026-09-15 07:24:42 | `732d5e96ad9eb295db61c79712b869865a5fbefd40731a253a6aea5fca60ddec` |
| 14 | `README.md` | ✅ | 12,974 | 2026-09-14 07:33:53 | `7d0f3b80d872913ae04e9dcf57727592267a57f1a0b0cee6d092f8bd01d1a906` |
| 15 | `docs/experiment_log.md` | ✅ | 32,098 | 2026-09-14 07:32:14 | `36ef3171207c8669a9b54fceb4287be7ebc30e9344f0ca2d2a470325833e3b81` |
| 16 | `docs/model_design.md` | ✅ | 11,584 | 2026-09-13 11:14:30 | `62908153d383c1e1b03854cb756d6f97ef4db7de2ea12654b5f63d7957ed0f4d` |
| 17 | `experiments/README.md` | ✅ | 1,502 | 2026-08-25 09:27:50 | `d97a6b029131cc5aac0eed3baefa9b7bcef4dc3992fcd63080d89c7f45d2e8f0` |

### 2.1 与线上成绩的对应关系

| 对象 | 是否与当前线上成绩对应 | 依据 |
|---|---|---|
| RGBID best.pt | ✅ **是**（0.53180） | `results.csv` best=ep237/0.55800；线上分析报告链条闭环 |
| RGBID submission.zip | ✅ **是**（0.53180） | 由该 best.pt + 冻结 `predict_rect.py --mode rect` 生成；1000 TXT 与测试集 1:1 |
| F4 best.pt | ✅ **是**（0.51051） | `runs/.../weights/best.pt`，best=ep251/0.53273（本地） |
| Ensemble 3:1 候选 zip | ❌ **无线上成绩** | 从未提交，仅离线审计 |
| `predict_rect.py` | ✅ 提交链条组成部分 | mtime 早于本会话，未被修改 |

**未覆盖、未删除任何已确认文件。**

---

## 3. RGBID 正式提交包完整性审计（阶段 B，14 项）

**审计对象**：`submissions/rgbird_ir_quicktest/submission.zip` + 同目录 `results/`

| # | 检查项 | 结果 | 判定 |
|---|---|---|---|
| 1 | 文件数量 | **1000**（期望 1000） | ✅ PASS |
| 2 | 空 TXT | **6**（无检测的图像，格式允许） | ✅ |
| 3 | TXT 行格式 | 非法 **0** 行（全部 6 字段） | ✅ PASS |
| 4 | 类别 id 范围 | 越界 **0**（全部 ∈ [0,11]） | ✅ PASS |
| 5 | confidence 合法性 | conf>1 = **0**，conf<0 = **0**；范围 **[0.001000, 0.981424]** | ✅ PASS |
| 6 | 坐标越界 | 越界 [0,1] = **0**；框体超出图像边界 = **0** | ✅ PASS |
| 7 | 重复文件名 | **0** | ✅ PASS |
| 8 | 缺失图片对应结果 | **0** | ✅ PASS |
| 9 | 多余结果 | **0** | ✅ PASS |
| 10 | NaN / Inf | **0** | ✅ PASS |
| 11 | **退化框 w=0 或 h=0** | **155** | ⚠️ 见 §3.2 |
| 12 | 是否修改 pipeline 自动修复 | **否，原样记录** | ✅ 遵守 |
| 13 | zip 目录结构 | 条目 1000，**0 个目录条目**，顶层前缀唯一为 `results`，非 `.txt` 文件 0，全部为 `results/<name>.txt` | ✅ PASS |
| 14 | CRC / integrity test | `zipfile.testzip()` → **None（无损坏）** | ✅ PASS |

**其他一致性**：总检测框数 **7370**；zip 内文件名集合与 `results/` 目录**一致**；与测试源图 1000 张**逐一对应**。

### 3.2 退化框（w=0 或 h=0）—— 原样记录，不修改

| 项 | 值 |
|---|---|
| 数量 | **155** |
| conf 均值 | 0.00256 |
| conf 中位 | 0.00155 |
| conf 最大 | **0.01907** |

**样例（原样摘录）**：
```
000003_014_00000081.txt: 9 0.000000 0.505914 0.000000 0.110441 0.007882
000003_014_00000081.txt: 9 0.000000 0.447289 0.000000 0.097126 0.007105
000003_014_00000081.txt: 6 1.000000 0.402351 0.000000 0.103281 0.005089
```

**来源（已追溯）**：冻结的 `predict.py::_format_lines` 会把 `scale_boxes`
裁剪到图像边界外的框 clamp 到 [0,1]，但**不设最小边长**，因此产生零宽/零高框。

**这是冻结 pipeline 的既有行为，不是本次引入**：该 zip 本身即线上 0.53180 的有效提交，
其内已含这 155 行。**未修改 `predict_rect.py`，未重新生成结果，未做任何"修复"。**

---

## 4. 提交规则核对（阶段 C）

### 4.1 仓库内可确认的规则

| 项 | 规则 | 来源 |
|---|---|---|
| **submission.zip 命名** | `submission.zip` | `scripts/predict.py:444`、`diagnostic/rect_pipeline_fix/final_submission_readme.md:86` |
| **zip 内目录结构** | `results/<name>.txt`（每条目一层，无目录条目） | 同上；本次实测确认 |
| **TXT 格式** | 每行 6 个值，空格分隔：`class_id cx cy w h conf` | `scripts/predict.py:7-10` |
| **类别编号** | `class_id ∈ [0, 11]`，共 12 类，顺序固定 | `configs/dataset.yaml`（nc=12 + names 0..11）；`scripts/predict.py:71` |
| **confidence 范围** | 期望落在 [0,1]（仓库未成文规定上界；`predict.py` 依赖 sigmoid 输出天然 ≤1，未显式截断） | ⚠️ **部分确认**——无比赛原文佐证 |
| **坐标格式** | 归一化到 [0,1]，保留 **6 位小数** | `scripts/predict.py:9-10`、`scripts/check_submission.py` |
| **文件数量** | = 测试图像数 = **1000** | `data/raw/test/visible` 实测 1000；与 zip 条目数一致 |
| **类别映射** | 0 person / 1 boat / 2 animal / 3 seat / 4 sign / 5 bicycle / 6 car / 7 ball / 8 light / 9 garbage_can / 10 uav / 11 tricycle | `configs/dataset.yaml` |

### 4.2 无法确认的规则（标记 UNKNOWN，不推断）

| 项 | 状态 |
|---|---|
| **提交次数（剩余/每日限额）** | ❌ **UNKNOWN** —— 仓库、文档、脚本中均无任何记录 |
| **是否允许覆盖 / 撤回提交** | ❌ **UNKNOWN** —— 无任何记录 |
| **最终截止时间** | ⚠️ 由用户提供：**2026-09-20 20:00**。仓库内无独立记录，**未做交叉验证** |
| confidence 上界的比赛原文规定 | ❌ **UNKNOWN** —— 仅有仓库内实现约定 |
| 官方评分器实现 | ❌ **UNKNOWN** —— 仓库内无官方脚本；`scripts/official_map.py` 为按赛题文字转写的**重实现**，与真实评分器是否逐位一致未经证实 |

> ⚠️ 由于「提交次数」与「覆盖规则」UNKNOWN，**任何提交的试错代价无法评估**。

---

## 5. 与历史正式提交的一致性（阶段 D）

本次**未生成新的提交包**。理由与依据：

1. 现有 `submissions/rgbird_ir_quicktest/submission.zip` **就是线上获得 0.53180 的那一份**；
   重新生成会产生一个**没有线上成绩对应**的新产物，反而破坏可追溯性。
2. 重新生成需重跑推理，属于任务书明确排除的方向之外的多余动作；
   且 5ch 路径无需修改（历史已验证）。
3. 因此本阶段对该 zip **只做确认与校验**，不做重新生成。

**一致性结论**：zip 内 1000 个 TXT 与测试集 1000 图 1:1 对应；格式、类别、坐标、confidence
全部通过审计；退化框数量（155）与历史记录一致；文件未被修改（SHA256 `0c550773…`）。

---

## 6. 未修改冻结 pipeline 的说明

| 冻结对象 | 状态 |
|---|---|
| `scripts/predict_rect.py` | **未修改**（mtime 2026-09-15 22:34:28，早于本会话） |
| `scripts/predict.py` | **未修改**（mtime 2026-09-15 12:54:23） |
| RGBID `best.pt` | **未修改、未覆盖**（mtime 2026-09-17 06:01:16 = 训练产出时刻） |
| F4 `best.pt` | **未修改、未覆盖**（mtime 2026-09-15 07:25:22 = 训练产出时刻） |
| 正式 RGBID submission.zip / results | **未修改、未覆盖** |
| Ensemble 3:1 候选包 | **未修改**，保留归档 |
| 历史实验 runs/ | **未删除任何目录** |

本阶段新建文件**仅为**：
- `reports/final_submission_audit.md`（本报告）
- `reports/FREEZE_MANIFEST.sha256`（只读校验清单）

---

## 7. 已明确放弃的实验方向

| 方向 | 放弃依据 |
|---|---|
| **1536 分辨率训练** | 未验证；收益不确定、成本高；当前保底链条已验证 |
| **P2 检测层** | 两次独立证伪（E3 −0.07459；E5 仅 +0.00013） |
| **F5 容量（YOLO11l@1280）** | 已证伪（0.52619 < 0.53273，−0.0065） |
| **学习率搜索** | 已优化（lr0 0.01→0.005，+0.01077） |
| **close_mosaic 搜索** | 已证伪（关闭后 best 不变） |
| **conf 阈值搜索** | 官方口径下单调为负（0.50928→0.50450→0.49972→0.49570）；项目约定不计入提升；诊断报告定性为违规 |
| **IoU 阈值搜索** | 已测 0.6 vs 0.7 仅 +0.00207，低于 +0.005 门槛 |
| **max_det 搜索** | 非瓶颈（单图最大 82 框，0 张触发截断） |
| **TTA（HFlip / MultiScale）** | 官方口径 −0.00636 / +0.00432；fork 口径均负；一级判据未通过 |
| **新 Ensemble / WBF** | Ensemble 3:1 已测且用户裁定不提交；不再搜索 |

---

## 8. 当前剩余风险

| # | 风险 | 说明 |
|---|---|---|
| 1 | **官方评分器未获独立脚本校验** | 仓库内无官方脚本；`official_map.py` 为文字转写的重实现，与真实评分器一致性未证实 |
| 2 | **剩余提交次数 UNKNOWN** | 无法评估试错代价 |
| 3 | **覆盖/撤回规则 UNKNOWN** | 若不可撤回，任何提交均为单向下行风险 |
| 4 | Ensemble 3:1 的 FP 增长 | +108 TP 换 **+2100 FP**（框数 4824→7031，Precision 0.83887→0.79877）；增益 +0.00568 远小于本地—线上系统偏移（≈+0.020） |
| 5 | Ensemble 在 fork 口径下为负 | −0.01278，逐类几乎全降；两口径符号相反且阈值鲁棒 |
| 6 | 1536 尚未验证 | 唯二未消耗的杠杆之一，但收益不确定 |
| 7 | **退化框 155 行** | 冻结 pipeline 既有行为，已裁定接受；若比赛侧对退化框有硬性校验则存在理论风险（但该 zip 已实际获得 0.53180，**实测被接受**） |
| 8 | 当前不继续试错 | 推理侧已证无安全增益；结合风险 2、3，继续试错期望收益为负 |

---

## 9. 最终输出

| 项 | 值 |
|---|---|
| **1. FINAL_MODEL** | `runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt`（RGBID Early Fusion，线上 0.53180） |
| **2. FINAL_SUBMISSION_ZIP** | `submissions/rgbird_ir_quicktest/submission.zip`（289,016 B，1000 条目，SHA256 `0c550773c86561089a9fe70ac020ecb67e12febe70093bbb7686aa97d1f756ce`） |
| **3. BACKUP_MODEL** | `runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt`（F4 RGBD，线上 0.51051） |
| **4. AUDIT_REPORT** | `reports/final_submission_audit.md`（本文件）+ `reports/FREEZE_MANIFEST.sha256` |

### 人工操作建议

- **当前正式提交候选**：`submissions/rgbird_ir_quicktest/submission.zip`（即线上 0.53180 那一份，已审计通过，**不要重新生成**）。
- **需要人工保留的文件**：`reports/FREEZE_MANIFEST.sha256`（含 17 项 SHA256，日后可校验产物是否被改动）；连同两个 `best.pt` 与上述 zip。
- **是否具备提交条件**：**具备** —— 包完整、格式合法、链条已验证、成绩已知。
  但**提交次数与覆盖规则 UNKNOWN**，请你在提交前自行确认这两项。
- **仍需人工确认的事项**：
  1. 比赛后台的**剩余提交次数**与**是否允许覆盖**；
  2. **截止时间 2026-09-20 20:00** 以比赛官方页面为准（仓库内无记录）；
  3. 是否接受 155 行退化框（该 zip 已实测获得 0.53180，判为可接受）。

**未执行**：训练 / 上传 / 修改模型 / 修改 `predict_rect.py` / 修改 submission 内容 /
自行提交 / 自行轮询后台 / 新 Ensemble / 新 TTA / 1536 / 删除历史结果。

**等待你的明确授权后才可执行最终比赛提交。**
