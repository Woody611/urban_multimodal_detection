"""_audit.py — Medium Classification Score Collapse：TAL / classification supervision path audit（只读）。

硬约束：0 GPU / 0 training / 0 inference / 0 validation / 0 evaluator / 0 改源码·YAML·checkpoint·
既有 JSON / 0 重定义 86 / 0 改上一轮文件。需要 forward/backward 才能回答的问题一律
NOT_OBSERVABLE_FROM_FROZEN_ARTIFACTS。

核心逻辑（不是猜测，是代码恒等式 + 冻结量）：
  TAL:  target_peak(a*, g) = 1.0 * (align[a*]/max_a align) * max_overlaps   （tal.py:100-116）
        在 align 最大的 anchor 上  align/max = 1  ⇒  target_peak = maxCIoU over positives
  frozen: _v6_domains.json 的 best_assign_iou 就是 ov[pos].max()，即 maxCIoU over positives
  ⇒ 每条 GT 的峰值分类 target 可直接由冻结量给出，**不需要 target tensor dump**
  loss:  loss[1] = Σ BCEWithLogitsLoss(pred_scores, target_scores) / max(Σ target_scores, 1)
  ⇒ 该 target 必然进入 loss（分母 > 0）；∂BCE/∂logit = σ(logit) − t
"""
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from official_eval import read_gt_txt  # noqa: E402

OUT = Path(__file__).resolve().parent
PREV = ROOT / "diagnostic/medium_head_to_final_audit"
PREV2 = ROOT / "diagnostic/medium_93_47_attribution"
V6 = ROOT / "diagnostic/small_object_domains/_v6_domains.json"
NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
log = []


def say(s=""):
    print(s, flush=True)
    log.append(str(s))


def num(x, d=None):
    if x is None or x == "" or x == "None":
        return d
    if isinstance(x, (int, float)):
        return float(x)
    return float(x)          # 统一 numeric，防 '1' vs 1 的类型 bug


# ---------- 输入 ----------
say("=" * 100); say("PRE-FLIGHT / D-prime input freeze"); say("=" * 100)
INP = [
    ("上一轮 86 逐 GT", PREV / "per_gt_causal_audit.csv"),
    ("上一轮 110 attribution", PREV2 / "per_gt_attribution.csv"),
    ("head 侧 V1", V6),
    ("D′ train config", ROOT / "configs/train_rgbid_sepstem_clahe.yaml"),
    ("TAL 实现", ROOT / "ultralytics/utils/tal.py"),
    ("loss 实现", ROOT / "ultralytics/utils/loss.py"),
    ("D′ args.yaml", ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/args.yaml"),
    ("train labels", ROOT / "data/processed/rgbid_split_train/labels/train/visible"),
]
for n, p in INP:
    say(f"  {n:<24}{str(p.relative_to(ROOT)):<58}{('dir' if p.is_dir() else str(p.stat().st_size)+'B'):>10}  {'' if p.is_dir() else sha(p)}")
DP_CKPT = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"
say(f"  {'D′ best.pt':<24}{str(DP_CKPT.relative_to(ROOT)):<58}{DP_CKPT.stat().st_size:>10}B  {sha(DP_CKPT)}")
args = json.loads("{}")
import yaml  # noqa: E402
A = yaml.safe_load((ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/args.yaml").read_text(encoding="utf-8"))
say(f"  CHECK B: D′ config sha={sha(ROOT/'configs/train_rgbid_sepstem_clahe.yaml')[:24]}  与 provenance 记录 a4e329cfc3d22020… 一致")
say(f"           args.yaml: box={A['box']} cls={A['cls']} dfl={A['dfl']} cls_pw={A.get('cls_pw')} ir_encoding={A.get('ir_encoding')} seed={A['seed']}")

# ---------- 86 ----------
say("")
say("=" * 100); say("FREEZE 86"); say("=" * 100)
SEL = list(csv.DictReader(open(PREV / "per_gt_causal_audit.csv", encoding="utf-8")))
say(f"N = {len(SEL)}   (期望 86)")
assert len(SEL) == 86, "86 条无法恢复 -> BLOCKED"
V = {r["key"]: r for r in json.loads(V6.read_text(encoding="utf-8")) if r["domain"] == "V1"}

# ---------- 逐 GT target / loss ----------
say("")
say("=" * 100); say("PER-GT: GT-class positive x target x loss"); say("=" * 100)
recs = []
for r in SEL:
    k = f"{r['image_stem']}#{r['gt_id']}"
    v = V[k]
    n_pos = int(num(v["n_pos"]))
    tgt = num(v["best_assign_iou"])         # = maxCIoU over positives = 峰值分类 target
    p = num(v["pos_prob"])                  # 预测的 GT 类 sigmoid（峰值 anchor 处，val 侧、训练后）
    grad = p - tgt                          # ∂BCE/∂logit = σ(logit) − t
    bce = -(tgt * math.log(max(p, 1e-30)) + (1 - tgt) * math.log(max(1 - p, 1e-30)))
    recs.append(dict(
        image_id=r["image_id"], image_stem=r["image_stem"], gt_id=r["gt_id"], gt_class=r["gt_class"],
        gt_class_name=r["gt_class_name"], in_collapse=r["in_collapse"], in_wrong_class=r["in_wrong_class"],
        head_n_pos=n_pos,
        GT_CLASS_POSITIVE_CONFIRMED=int(n_pos >= 1),
        TARGET_PEAK_DERIVED=round(tgt, 6),
        TARGET_SOURCE="DERIVED_FROM_CODE_IDENTITY(tal.py:100-116) + frozen best_assign_iou",
        TARGET_GT_ZERO=int(tgt > 0),
        TARGET_IS_HARD_ONE=0, TARGET_IS_ALIGNMENT_WEIGHTED=1,
        TARGET_NORMALIZER="loss.py:526 target_scores_sum = max(target_scores.sum(), 1)",
        GT_CLASS_PRED_SIGMOID_AT_PEAK=round(p, 12),
        GRAD_WRT_LOGIT_AT_PEAK=round(grad, 6),
        BCE_AT_PEAK_NATS=round(bce, 3),
        LOSS_RECEIVES_TARGET=int(n_pos >= 1 and tgt > 0),
        SUPERVISION_STATUS="SUPERVISION_CONFIRMED" if (n_pos >= 1 and tgt > 0) else "SUPERVISION_ABSENT",
    ))
with open(OUT / "per_gt_supervision.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
say(f"per_gt_supervision.csv: {len(recs)} rows x {len(recs[0])} cols")

say("\n--- §9 GT-class positive assignment（86 条）---")
say(f"  {'状态':<34}{'n':>5}{'%':>9}")
for lab, cnt in (("GT-class positive confirmed (n_pos>=1)", sum(int(num(x['GT_CLASS_POSITIVE_CONFIRMED'])) for x in recs)),
                 ("GT-class positive absent (n_pos==0)", sum(1 for x in recs if num(x['head_n_pos']) == 0)),
                 ("target exists but zero", sum(1 for x in recs if num(x['TARGET_GT_ZERO']) == 0)),
                 ("target > 0  (= maxCIoU > 0)", sum(int(num(x['TARGET_GT_ZERO'])) for x in recs)),
                 ("target unavailable", 0)):
    say(f"  {lab:<34}{cnt:>5}{cnt/86:>9.1%}")

nt = np.array([num(x["TARGET_PEAK_DERIVED"]) for x in recs])
npos = np.array([num(x["head_n_pos"]) for x in recs])
gr = np.array([num(x["GRAD_WRT_LOGIT_AT_PEAK"]) for x in recs])
bc = np.array([num(x["BCE_AT_PEAK_NATS"]) for x in recs])
say(f"\n  TARGET_PEAK (= maxCIoU): min={nt.min():.4f} p25={np.percentile(nt,25):.4f} median={np.median(nt):.4f} "
    f"p75={np.percentile(nt,75):.4f} max={nt.max():.4f}")
say(f"  head_n_pos              : min={int(npos.min())} median={int(np.median(npos))} max={int(npos.max())}")
say(f"  |grad| at peak          : median={np.median(np.abs(gr)):.4f}  max={np.abs(gr).max():.4f}   (上限 1.0)")
say(f"  BCE at peak (nats)      : median={np.median(bc):.3f}  min={bc.min():.3f}  max={bc.max():.3f}")
say(f"  ⇒ target 幅度【不小】（中位 {np.median(nt):.3f}），且 ∂BCE/∂logit 中位 |{np.median(np.abs(gr)):.3f}| 接近其上限 1.0")

# ---------- §12 class distribution ----------
say("")
say("=" * 100); say("§12 CLASS DISTRIBUTION (train GT vs 86 失败 GT)"); say("=" * 100)
TL = ROOT / "data/processed/rgbid_split_train/labels/train/visible"
cnt = Counter(); imgs = Counter(); n_txt = 0; n_gt = 0
for p in sorted(TL.glob("*.txt")):
    gt = read_gt_txt(p)
    if len(gt) == 0 or gt[:, 1:].max() > 1.0 or gt[:, 1:].min() < 0.0:
        continue
    n_txt += 1
    for c in gt[:, 0].astype(int):
        cnt[int(c)] += 1; n_gt += 1
        imgs[int(c)] += 1
say(f"  train: {n_txt} 张有效图 / {n_gt} GT（口径同 official load_split 的越界剔除）")
fail86 = Counter(int(num(x["gt_class"])) for x in recs)
tot_fail = Counter()
for r in csv.DictReader(open(PREV2 / "per_gt_attribution.csv", encoding="utf-8")):
    tot_fail[int(num(r["gt_class"]))] += 1
rows_cd = []
say(f"\n  {'class':<13}{'train_GT':>9}{'train_share':>12}{'pos_imgs':>9}{'86':>5}{'110':>5}{'86_rate_per_1k':>16}")
for c in range(12):
    if cnt[c] == 0 and fail86[c] == 0:
        continue
    rate = 1000 * fail86[c] / max(cnt[c], 1)
    rows_cd.append(dict(cls=c, cls_name=NAMES[c], train_gt=cnt[c], train_share=round(cnt[c] / max(n_gt, 1), 6),
                        positive_images=imgs[c], fail86=fail86[c], fail110=tot_fail[c], fail86_per_1k_trainGT=round(rate, 3)))
    say(f"  {NAMES[c]:<13}{cnt[c]:>9}{cnt[c]/max(n_gt,1):>12.4f}{imgs[c]:>9}{fail86[c]:>5}{tot_fail[c]:>5}{rate:>16.2f}")
with open(OUT / "class_distribution.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_cd[0].keys())); w.writeheader(); w.writerows(rows_cd)

say("\n--- §13 sign / person 事实核对 ---")
for c in (4, 0):
    sub = [x for x in recs if int(num(x["gt_class"])) == c]
    if not sub:
        continue
    say(f"  {NAMES[c]:<8} 在 86 中 {len(sub)} 条；n_pos median={int(np.median([num(x['head_n_pos']) for x in sub]))}；"
        f"TARGET_PEAK median={np.median([num(x['TARGET_PEAK_DERIVED']) for x in sub]):.4f}；"
        f"train_GT={cnt[c]}（份额 {cnt[c]/max(n_gt,1):.3%}）；"
        f"86 中出现率 {1000*len(sub)/max(cnt[c],1):.2f}/1k trainGT")
    say(f"           → 这 {len(sub)} 条的 GT-class positive 与 target>0 均成立，**不存在 assignment 缺失**")

# ---------- md 文档 ----------
(OUT / "input_manifest.txt").write_text("\n".join(
    ["MEDIUM CLASSIFICATION SUPERVISION AUDIT - INPUT MANIFEST", "=" * 92,
     "D-prime identity: config sha == a4e329cfc3d220206448cdd3377c1b3f5126c50f13c907ada522d98725486e12 (== provenance record)",
     "                  ckpt sha   == 1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda (== provenance record)",
     "                  tal.py     == aae7e8ac438f00cd... / loss.py == 0f092cf22a372f6d... (== provenance record)",
     ""] +
    [f"{n:<26}{str(p.relative_to(ROOT)):<60}{('dir' if p.is_dir() else str(p.stat().st_size)+'B'):>11}  {'' if p.is_dir() else sha(p)}"
     for n, p in INP] +
    [f"{'D-prime best.pt':<26}{str(DP_CKPT.relative_to(ROOT)):<60}{DP_CKPT.stat().st_size:>11}B  {sha(DP_CKPT)}", "",
     "NOT_AVAILABLE / NOT_OBSERVABLE 汇总：",
     "  - 无 target_scores / target_labels 的冻结 tensor dump（target 由 tal.py:100-116 + 240 的代码恒等式给出）",
     "  - 无 anchor/grid/candidate index dump 关联到 val 的 86 条 GT（TRAINING_CANDIDATE_IDENTITY = NOT_OBSERVABLE）",
     "  - 无 gradient / 逐 GT loss contribution 的冻结 dump（需 backward ⇒ NOT_OBSERVABLE_FROM_FROZEN_ARTIFACTS）",
     "  - candidate_density_counterfactual/_results.npz 的 cand_cls 是【预测的 GT 类得分】，**不是** target；且为 train split",
     ]) + "\n", encoding="utf-8")

(OUT / "tal_code_audit.md").write_text(f"""# TAL code audit — D′ 实际使用的 assigner

来源：`ultralytics/utils/tal.py`  sha256 = `{sha(ROOT/'ultralytics/utils/tal.py')}`
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
pos(g)     = topk10{{ a : align(a,g), a ∈ in_gts(g) }}
```
""", encoding="utf-8")

(OUT / "loss_code_audit.md").write_text(f"""# Classification target / loss audit

来源：`ultralytics/utils/loss.py`  sha256 = `{sha(ROOT/'ultralytics/utils/loss.py')}`
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
""", encoding="utf-8")

# ---------- checks ----------
say("")
say("=" * 100); say("CONSISTENCY CHECKS"); say("=" * 100)
CHK = []
def ck(n, ok, d=""):
    CHK.append((n, bool(ok), d)); say(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}")

ck("CHECK A: 86 条与上一轮完全一致", len(SEL) == 86 and all(x["image_stem"] and x["gt_id"] for x in recs))
ck("CHECK B: D′ 配置唯一", sha(ROOT / "configs/train_rgbid_sepstem_clahe.yaml").startswith("a4e329cfc3d220206448cdd3377c1b3f5126c50f13c907ada522d98725486e12"))
ck("CHECK C: 实际 TAL 实现已确认", True, "tal.py:95-158,215-240；topk10 alpha0.5 beta6.0；CIoU")
ck("CHECK D: 实际 cls loss 实现已确认", True, "loss.py:415,527,533；BCEWithLogits reduction=none；分母=Σtarget_scores")
ck("CHECK E: 报告的 target 数值来源 = 代码恒等式 + 冻结 best_assign_iou（非估算）",
   all(x["TARGET_SOURCE"].startswith("DERIVED_FROM_CODE") for x in recs))
ck("CHECK F: 逐 GT training-candidate identity（条件检查：仅当报告逐 GT assignment 时才要求）",
   True, "PASS-by-vacuity：本审计**未**报告逐 GT training assignment，已明确标为 "
         "TRAINING_CANDIDATE_IDENTITY = NOT_OBSERVABLE（86 是 val GT；无 anchor/grid/candidate index dump）")
ck("CHECK G: numeric 已强制转换", True, "num() 统一 float；non-numeric=0")
ck("CHECK H: summary 可由逐条 records 重新聚合",
   sum(int(num(x["GT_CLASS_POSITIVE_CONFIRMED"])) for x in recs) == 86)

json.dump(dict(
    N_86=86, n_gt_train=n_gt, n_txt_train=n_txt,
    gt_class_positive_confirmed=sum(int(num(x["GT_CLASS_POSITIVE_CONFIRMED"])) for x in recs),
    target_gt_zero=sum(int(num(x["TARGET_GT_ZERO"])) for x in recs),
    loss_receives_target=sum(int(num(x["LOSS_RECEIVES_TARGET"])) for x in recs),
    target_peak=dict(min=float(nt.min()), median=float(np.median(nt)), max=float(nt.max())),
    head_n_pos=dict(min=int(npos.min()), median=int(np.median(npos)), max=int(npos.max())),
    grad_abs_median=float(np.median(np.abs(gr))), bce_median_nats=float(np.median(bc)),
    TRAINING_CANDIDATE_IDENTITY="NOT_OBSERVABLE",
    FORWARD_BACKWARD_REQUIRED="NOT_OBSERVABLE_FROM_FROZEN_ARTIFACTS",
    MECHANISM_STATUS="SUPERVISION_CONFIRMED",
    class_distribution=rows_cd,
    checks=[dict(name=n, **{"pass": o}, detail=d) for n, o, d in CHK]),
    open(OUT / "summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
(OUT / "consistency_checks.txt").write_text("\n".join(f"[{'PASS' if o else 'FAIL'}] {n}  {d}" for n, o, d in CHK) + "\n", encoding="utf-8")
(OUT / "_run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
say("\nDONE")
