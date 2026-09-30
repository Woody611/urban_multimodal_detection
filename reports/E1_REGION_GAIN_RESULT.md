# E1 = D′ + L12–L17 Region Response Gain —— 本地评测与提交包

**日期**：2026-09-28 · **性质**：训练后诊断 + 候选提交包生成（**未上传**）
**对象**：`runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_e1/`（300 epoch 跑完，`best.pt` = ep253）
**合同**：`reports/E1_EXPERIMENT_CONTRACT.md`（21/21 门 PASS，EXPERIMENT READY）

> **一句话结论：E1 在全部四个本地口径上一致低于 D′ —— 这是本项目迄今最干净的本地负结果。**
> 提交包已按赛题格式生成并通过全部自检，但**本地证据不支持 E1 优于 D′**，是否花一次线上提交由用户定夺。

---

## 1. 单变量是否成立 —— 成立 ✅

| 检查 | 结果 |
|---|---|
| ckpt 内嵌 `model.yaml` | `e1_enabled: true`、`e1_layers: [13, 15, 17]`、`e1_kernel: 3`、`ch: 5`、`nc: 12`、`scale: m` |
| 参数量 | **20,077,332** = 合同预期值（D′ 20,061,972，Δ +15,360 = **+0.0766%**） |
| E1 模块挂载位点 | 实测 31 个 top-level 模块中，L13 `C3k2`、L15 `C3k2`、L17 `C2PSA` **各含 1 个 `RegionResponseGain`** |
| `ir_encoding` | **`clahe`**（ckpt `train_args` 与推理日志两处一致） |
| 前向 | `(1,5,736,1312)` → `(1,16,19803)` + P3/P4/P5 三尺度特征图 (76ch 各) ✅ |
| 学到的增益幅值 | \|dw\| 均值 0.004–0.007 / 最大 0.126（tanh 后 ≈ **+0.7%**）—— 增益确实被学到，但极温和 |

`configs/train_rgbid_sepstem_clahe_e1.yaml` 与 D′ 的 `train_rgbid_sepstem_clahe.yaml` 逐字相同（仅 `experiment_name`），
唯一变量由 CLI 的 `--model_config configs/yolo11m_sepstem_e1.yaml` 承载 —— **差异恰好两项，单变量成立**。

---

## 2. 四口径 A/B（同环境、同 400 图 val / GT 2807）

评测链：`scripts/predict_rect.py --mode rect --imgsz 1280 --conf 0.001 --iou 0.7 --max_det 300 --max_boxes 300`
→ `scripts/official_map.py --split_root data/processed/rgbid_split`（metric A，cap 100）。
**D′ 控制组用同一份 `best.pt`（sha256 `1cae45f7…`）在本环境重跑**，以排除代码漂移。

| 口径 | D′（控制组重跑） | **E1** | Δ (E1 − D′) |
|---|---:|---:|---:|
| **官方 mAP50-95** | 0.51528 | **0.50988** | **−0.00540** |
| 官方 mAP50 | 0.77826 | 0.77029 | −0.00797 |
| 官方 mAP75 | 0.51096 | 0.50403 | −0.00693 |
| **fork mAP50-95** | 0.56609 | **0.56270** | **−0.00339** |
| 训练内 val best | 0.56690 @ep246 | 0.56546 @ep253 | −0.00144 |

**测量链自洽性**：D′ 控制组重跑得 **0.51528**，与 E1 合同声明的 baseline（0.51528）**逐位一致**
⇒ 本次 A/B 的对照值可信，上述 Δ 不是代码漂移造成的伪差。

落盘：`diagnostic/e1_region_gain/best_full/`（E1）、`diagnostic/e1_region_gain/ctrl_dprime/`（D′控制）。

### 为什么这次不能套用「本地负 → 线上可能正」的先例

本项目历史上确有本地判负、线上判正的先例（D：本地判 REJECTED → 线上 +0.02350；
D′：训练内 val −0.00116 → 线上 +0.727）。但**那两次的特征是「口径之间符号矛盾」**，
而 E1 是**四个口径同号**——这正是本项目至今最干净的负结果，先例的可迁移性弱。

**但须保留一条**：fork 口径「与线上同序」的保证**只在同架构族内**成立（跨架构会错排，见 D vs D′）。
E1 相对 D′ 是**跨架构**差分（多挂 3 个模块），故 fork 这一票的证据力**弱于往常**，
0.00339 的小差距不宜直接判死。

---

## 3. Provenance

| 项 | SHA256 |
|---|---|
| E1 `weights/best.pt` | `f3929ee65681eaed36d6ae0f4628141180ccc10c02425b081d5c7b4d8ac18d26` |
| D′ `weights/best.pt`（控制组） | `1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda` |
| `configs/yolo11m_sepstem_e1.yaml` | `b2cf656027e2235696d6a1247503dc7e0348529b7fc27987de1e6371b151f217` |
| `configs/train_rgbid_sepstem_clahe_e1.yaml` | `99cfc5793b2df348a41ad5b4176425911fd3109f4254473f11fa7f0a34653840` |
| `scripts/predict_rect.py`（本次改默认值后） | `36858acb59058e89b67dcc491d6c68d36672efb9990b0f96b95cc3163e77bd93` |

**`predict_rect.py` 的改动**：仅 `DEFAULT_WEIGHTS` / `DEFAULT_TRAIN_CONFIG` 两个常量 + 注释。
逐行核对（`git diff` 去注释）确认**无任何逻辑改动** —— 且控制组与 E1 都显式传 `--weights/--train_config/--max_boxes`，
推理数学与产出 D′ 的冻结链逐位等价。

---

## 4. 提交包（2026-09-28 生成，**待上传**）

```bash
python scripts/predict_rect.py \
  --weights runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_e1/weights/best.pt \
  --train_config configs/train_rgbid_sepstem_clahe_e1.yaml \
  --source data/raw/test/visible \
  --mode rect --imgsz 1280 --conf 0.001 --iou 0.7 --max_det 300 --max_boxes 100 \
  --batch 16 --output submissions/rgbid_sepstem_clahe_e1_candidate

python scripts/check_submission.py \
  --results submissions/rgbid_sepstem_clahe_e1_candidate/results \
  --zip submissions/rgbid_sepstem_clahe_e1_candidate/submission.zip --expect 1000
```

推理日志自证：`use_simotm=RGBID channels=5 ir_encoding=clahe`，1000 张图 → 1000 个 TXT。

| 项 | 值 |
|---|---|
| **submission.zip** | 348,035 B · sha256 `af888514e452c1d43501d3eb83f822d23d454e982980334b2484b95b373324af` |
| zip 成员 | 1000 个，全部 `results/<stem>.txt`，无损坏成员 |
| 自检 | **ALL PASS**（TXT 1000/1000、格式错误 0、类别越界 0、坐标越界 0） |
| 检测框总数 | 10,857（均值 10.86 / 中位 8 / 峰 100） |
| 有框图片 | 981 / 1000（19 张无检测） |
| 每图上限 | 最大 100 = §九(二) 硬上限，**无图超限** |

---

## 5. 判定

```
本地证据（四口径一致）：E1  <  D′
线上参考：               D′ = 48.712（已是当前最优）
```

- **不建议**以 E1 替换 D′ 作为默认最优模型 —— 本地四口径无一支持。
- 若要花一次线上提交，须明确这是一次**低期望值赌博**：赌的是「跨架构差分下 fork 口径失效」这一条
  尚未被证伪的可能性，而非赌 E1 的机制本身。
- **本地最优仍是 D′**；`predict_rect.py` 的默认值现指向 E1，理由是用户要求以 E1 出包。
  若后续判定 E1 无增益，应把默认值回退到 D′（`runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt`
  + `configs/train_rgbid_sepstem_clahe.yaml`）。
