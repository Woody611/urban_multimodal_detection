# TAL code audit — D′ 实际使用的 assigner

来源：`ultralytics/utils/tal.py`  sha256 = `aae7e8ac438f00cda88d7e151795b3f78270ec4ed56b883f413fd16233b7e527`
（与 D′ provenance 记录 `aae7e8ac438f00cd…` 一致）
超参：`loss.py:438` `TaskAlignedAssigner(topk=10, num_classes=12, alpha=0.5, beta=6.0)`

## A. positive assignment 的输入（`loss.py:521-527` 实际调用）
```
target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
    pred_scores.detach().sigmoid(),                       # (b, h*w, nc)  **sigmoid 后、已 detach**
    (pred_bboxes.detach() * stride_tensor),               # (b, h*w, 4)   解码框 × stride，**已 detach**
    anchor_points * stride_tensor,                        # anchor 中心（像素）
    gt_labels, gt_bboxes, mask_gt)
```

## B. classification score 在 TAL 中的用法（`tal.py:133-151`）
```
bbox_scores[mask_gt] = pd_scores[ind0, :, ind1][mask_gt]   # 取 **GT 类别那一列** 的 sigmoid 概率
align_metric = bbox_scores.pow(alpha) * overlaps.pow(beta)  # alpha=0.5, beta=6.0
```
⇒ 用的是 **sigmoid 概率**（非 logit）、**仅 GT 类**、`detach()`（无梯度回传）。**无 objectness 参与。**

## C. bbox quality（`tal.py:153-155`）
```
iou_calculation = bbox_iou(gt, pd, xywh=False, CIoU=True).squeeze(-1).clamp_(0)   # CIoU
```

## D. positive candidate 选法（`tal.py:124-131`）
```
mask_in_gts  = select_candidates_in_gts(anc_points, gt_bboxes)   # anchor 中心严格在 GT 框内
mask_topk    = select_topk_candidates(align_metric, topk=10)     # 按 align_metric 取前 10
mask_pos     = mask_topk * mask_in_gts * mask_gt
```
再经 `select_highest_overlaps` 解决一个 anchor 对多 GT 的冲突。

## 简化公式
```
align(a,g) = cls_sigmoid(a, g_cls)^0.5 * CIoU(a,g)^6
pos(g)     = topk10{ a : align(a,g), a ∈ in_gts(g) }
```
