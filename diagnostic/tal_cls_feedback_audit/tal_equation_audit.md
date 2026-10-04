# TAL equation audit（本轮重新读取源码，不手抄旧报告）

| 项 | 实际值 | 来源 |
|---|---|---|
| 文件 | `ultralytics/utils/tal.py` | sha `aae7e8ac438f00cda88d7e151795b3f78270ec4ed56b883f413fd16233b7e527` |
| `alpha` | **0.5** | `loss.py:438` 实例化 `TaskAlignedAssigner(..., alpha=0.5, ...)` |
| `beta` | **6.0** | 同上 `beta=6.0` |
| `topk` | **10** | `loss.py:438` `tal_topk=10` |
| 变量名 | `align_metric` | `tal.py:150` |
| 公式 | `align_metric = bbox_scores.pow(alpha) * overlaps.pow(beta)` | `tal.py:150` |
| cls 来源 | `bbox_scores[mask_gt] = pd_scores[ind0, :, ind1][mask_gt]` | `tal.py:147` ⇒ **GT 类那一列** |
| cls 形式 | `pred_scores.detach().sigmoid()` | `loss.py:522` ⇒ **sigmoid 概率、已 detach** |
| geom 来源 | `iou_calculation = bbox_iou(..., CIoU=True).squeeze(-1).clamp_(0)` | `tal.py:155` ⇒ **CIoU** |
| 候选池 | `mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)` | `tal.py:125` |
| top-k | `mask_topk = select_topk_candidates(align_metric, topk_mask=...)` | `tal.py:126` |
| 多 GT 冲突 | `select_highest_overlaps(mask_pos, overlaps, n_max_boxes)` | `tal.py:106` |
| 正样本 | `mask_pos = mask_topk * mask_in_gts * mask_gt` | `tal.py:128` |

```text
align(a,g) = cls_sigmoid(a, g_cls)^0.5 * CIoU(a,g)^6
pos(g)     = topk10( a : align(a,g), a in in_gts(g) )   再解多 GT 冲突
```

**注意**：`select_topk_candidates` 在**全部 anchor** 上取 top-k；非 in-gts 的 anchor `bbox_scores=0`
且 `overlaps=0` ⇒ `align=0`，随后被 `* mask_in_gts` 掩掉 ⇒ 实际等价于在池内按 align 取前 10。

**字段语义陷阱（本轮 C1 实测发现）**：`tal.py:110` 的 `align_metric *= mask_pos` 是 in-place，
探针持有同一 tensor ⇒ dump 出的 `cand_align` 是 **post-mask_pos** 版本（非正样本被置 0），
**不能用于排序**。排序须用重算的 `align_raw = cls^0.5 * CIoU^6`。
