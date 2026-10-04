# Classification target / loss audit

来源：`ultralytics/utils/loss.py`  sha256 = `0f092cf22a372f6d5c5e7197d7adde51cbb23c02ac4522978f42df5b23a6543b`
（与 D′ provenance 记录 `0f092cf22a372f6d…` 一致）

## 1. target_bboxes / target_labels / target_scores 的实际构造

`tal.py:215-240`（`get_targets`）：
```
target_labels = gt_labels.long().flatten()[target_gt_idx]      # 该 anchor 被分配的 GT 类
target_scores = zeros(...); target_scores.scatter_(2, target_labels[...,None], 1)   # ← 先置 **hard 1.0**
target_scores = torch.where(fg_mask[...,None].repeat(...) > 0, target_scores, 0)    # 非 fg 置 0
```
`tal.py:110-116`（归一化）：
```
align_metric *= mask_pos
pos_align_metrics = align_metric.amax(-1, keepdim=True)          # 每个 GT 的 max align
pos_overlaps      = (overlaps * mask_pos).amax(-1, keepdim=True) # 每个 GT 的 **max CIoU over positives**
norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics+eps)).amax(-2).unsqueeze(-1)
target_scores = target_scores * norm_align_metric
```

### ⇒ 分类 target 的精确形式（**非 hard 1、非纯 IoU**）
```
target[a, g_cls] = 1.0 * ( align[a] / max_a align ) * max_overlaps(g)
                 = ( align[a] / max_a align ) * maxCIoU_over_positives(g)
```
在 **align 最大的那个 anchor** 上 `align/max = 1` ⇒ **峰值分类 target = 该 GT 的 maxCIoU over positives**。
（该恒等式是代码级的，不依赖任何经验假设。）

## 2. 归一化

`loss.py:526`  `target_scores_sum = max(target_scores.sum(), 1)`
`loss.py:533`  `loss[1] = loss_cls.sum() / target_scores_sum`
⇒ 分母只含 **非零 target** 的求和 → 背景 anchor 对分母无贡献（不存在"背景稀释"）。

## 3. classification loss

`loss.py:415`  `self.bce = nn.BCEWithLogitsLoss(reduction="none")`
`loss.py:527`  `loss_cls = self.bce(pred_scores, target_scores.to(dtype))   # (b, h*w, nc)`
`loss.py:533`  `loss[1] = loss_cls.sum() / target_scores_sum ...  × hyp.cls(0.5)`
`cls_pw`：D′ 的 args.yaml 为 `[]` ⇒ `len([]) != nc(12)` ⇒ `self.cls_pw = None` ⇒ **无类别加权**。

### 逐元素梯度
```
BCE(logit, t) = -[ t·log σ + (1-t)·log(1-σ) ]        ∂BCE/∂logit = σ - t
```
⇒ 当预测 σ ≈ 0 而 target t ≈ maxCIoU（0.5–0.9）时，**梯度接近其上限**。
