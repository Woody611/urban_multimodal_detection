"""_analyze.py — Training-side Classification Supervision Audit（READ-ONLY / NO-FORWARD）

唯一问题：为什么 native-small GT 在**训练阶段**获得的 classification supervision，
最终没有转化为 validation G1 的 GT-class response？

硬约束（违反即 INVALID）：0 training / 0 forward / 0 backward / 0 optimizer step /
0 修改既有文件 / 0 重定义 G1-G2-G4 / 0 重跑已完成实验。

只用：真实源码 + 已有 artifact（`_sge_supervision.json`、`_v5_replay.json`、
`_sge_native_meta.json`、`_v6_domains.json`、`_v2_records.json`、train/val labels）。
只写 diagnostic/train_side_cls_supervision/。
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SGE = ROOT / "diagnostic/small_gt_exposure"
V5 = ROOT / "diagnostic/small_object_train_replay"
DOM = ROOT / "diagnostic/small_object_domains"
V2D = ROOT / "diagnostic/small_object_cause_v2"
P3 = ROOT / "diagnostic/p3_feature_space"
TR_LAB = ROOT / "data/processed/rgbid_split_train/labels/train/visible"
TR_IMG = ROOT / "data/processed/rgbid_split_train/images/train/visible"
VA_LAB = ROOT / "data/processed/rgbid_split_train/labels/val/visible"
CKPT = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}
SMALL_A, MEDIUM_A = 1024.0, 9216.0
BOOT = 20000
SEED = 20260929
LOG: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def boot_ci(v, stat=np.median, n=BOOT, seed=SEED):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"))
    r = np.random.default_rng(seed)
    d = stat(v[r.integers(0, len(v), size=(n, len(v)))], axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def desc(v, name="") -> dict:
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(name=name, n=0)
    lo, hi = boot_ci(v)
    return dict(name=name, n=int(len(v)), median=float(np.median(v)),
                q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
                p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                mean=float(np.mean(v)), boot95=[lo, hi])


def mwu_p(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return float("nan")
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:  # noqa: BLE001
        return float("nan")


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 4:
        return float("nan"), float("nan")
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(a, b)
        return float(r), float(p)
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")


def buck(a) -> str:
    return "small" if a < SMALL_A else ("medium" if a < MEDIUM_A else "large")


def main() -> int:
    say("=" * 112)
    say("TRAINING-SIDE CLASSIFICATION SUPERVISION AUDIT —— READ-ONLY / NO-FORWARD")
    say("=" * 112)
    prov = {str(p): sha16(p) for p in
            (CKPT, ROOT / "ultralytics/utils/tal.py", ROOT / "ultralytics/utils/loss.py",
             ROOT / "ultralytics/nn/modules/head.py", ROOT / "configs/train_rgbid_sepstem_clahe.yaml",
             SGE / "_sge_supervision.json", V5 / "_v5_replay.json",
             SGE / "_sge_native_meta.json", DOM / "_v6_domains.json", V2D / "_v2_records.json",
             P3 / "_v7_vectors.json")}
    say("[provenance]")
    for k, v in prov.items():
        say(f"  {v}  {k}")

    # ================================================================ §3 源码事实链
    say("")
    say("=" * 112)
    say("[§3] ACTUAL CODE PATH（本项目 fork 的真实源码，非 YOLO 常识）")
    say("=" * 112)
    say("  `ultralytics/utils/tal.py:118-127`  TaskAlignedAssigner.get_pos_mask")
    say("      mask_in_gts  = select_candidates_in_gts(anc_points, gt_bboxes)")
    say("      align_metric, overlaps = get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes,")
    say("                                               mask_in_gts * mask_gt)")
    say("      mask_topk = select_topk_candidates(align_metric, topk_mask=...)")
    say("      mask_pos  = mask_topk * mask_in_gts * mask_gt")
    say("  `tal.py:139-152`  get_box_metrics")
    say("      ind[1] = gt_labels.squeeze(-1)")
    say("      bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]     # = sigmoid 后的同类分")
    say("      overlaps[mask_gt]    = iou_calculation(gt, pd_boxes)            # CIoU, clamp(0)")
    say("      align_metric = bbox_scores.pow(alpha) * overlaps.pow(beta)       # alpha=.5 beta=6")
    say("  `tal.py:109-118`  _forward 的 **target 构造 + 归一化**（本轮核心）")
    say("      target_labels, target_bboxes, target_scores = get_targets(...)")
    say("      align_metric *= mask_pos                    # ← in-place")
    say("      pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)   # 每 GT 的 max align")
    say("      pos_overlaps      = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # 每 GT 的 max CIoU")
    say("      norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + eps)).amax(-2)")
    say("      target_scores = target_scores * norm_align_metric")
    say("  `tal.py:226-241`  get_targets 里 target_scores 的初值")
    say("      target_scores = zeros((b, h*w, nc)); target_scores.scatter_(2, target_labels, 1)  # ← one-hot **1.0**")
    say("      target_scores = where(fg_mask>0, target_scores, 0)                                  # 只留正样本")
    say("")
    say("  ⇒ **raw target（正样本、GT 类通道）= 1.0；归一化后：**")
    say("       target[a] = 1.0 × [ align[a, g_a] · maxCIoU[g_a] / maxAlign[g_a] ]")
    say("     其中 g_a = anchor a 被分配到的 GT（`select_highest_overlaps` 已保证唯一）。")
    say("  ⇒ **在 GT g 的 max-align 正样本上：target = maxAlign · maxCIoU / maxAlign = maxCIoU[g]**")
    say("  ⇒ **『每个 GT 的峰值分类 target == 它最好的正样本与它的 CIoU』**（下面数值验证）")
    say("")
    say("  `ultralytics/utils/loss.py:523-535`  分类 loss 的 reduction")
    say("      target_scores_sum = max(target_scores.sum(), 1)        # 全 batch、全 anchor、全类求和")
    say("      self.bce = nn.BCEWithLogitsLoss(reduction='none')      # loss.py:415")
    say("      loss_cls = self.bce(pred_scores, target_scores.to(dtype))   # (b, h*w, nc)")
    say("      if self.cls_pw is not None: ...                        # loss.py:430 getattr(h,'cls_pw',None)")
    say("      loss[1] = loss_cls.sum() / target_scores_sum")
    say("  `configs/train_rgbid_sepstem_clahe.yaml` 无 `cls_pw` 键 ⇒ `self.cls_pw = None`（类别加权未启用）")
    say("  `loss.py:526`  `loss.sum() * batch_size`（__call__ 里），随后 trainer 除以累计量 —— 不改变 GT 间相对权重")

    # ================================================================ §4 加载 artifact
    say("")
    say("=" * 112)
    say("[§4] ARTIFACT 语义核对（不轻信字段名）")
    say("=" * 112)
    sup = json.loads((SGE / "_sge_supervision.json").read_text(encoding="utf-8"))
    R = [r for r in sup["rows"] if "err" not in r]
    say(f"  `_sge_supervision.json`: rows={len(sup['rows'])}（有效 {len(R)}），"
        f"n_images={sup['n_images']} epochs={sup['epochs']} seed={sup['base_seed']}")
    say("    来源 `_sge_supervision.py:62`：`self.last.update(target_scores=ts)` —— **stash 的是 forward 的"
        "返回值**，即 `tal.py:116` 归一化**之后**的 target_scores ✓")
    say("    字段：cls / n_pos / best_align / pos_target_max / pos_strides / aug_w,h,area / "
        "**native_area** / layer(HR|LR)")
    say("    ⇒ **有 native_area** ⇒ 可以按 native 尺度分桶（协议要求）。")
    v5 = json.loads((V5 / "_v5_replay.json").read_text(encoding="utf-8"))
    V5R = v5["rows"]
    say(f"  `_v5_replay.json`: rows={len(V5R)} seed={v5['seed']} "
        f"stat={v5['stat']}")
    say("    字段额外含 `pos_target_mean` / `target_sum_all` / `mean_align` / `best_assign_iou`")
    say("    ⚠ `_v5_replay.py:270-271` 的 `bucket` 用 **aug_area**（增强后），**不是 native**")
    say("    ⚠ `_v5_replay.py:264` 自述「身份追踪不可行」：`stat.map_bad = 244/250` ⇒ "
        "**`src_stem` 不可用于回连 native**")
    say("")
    # 语义陷阱 1：target_sum_all 不是 per-GT
    say("  ⚠ **语义陷阱 A**：`target_sum_all = target_scores[0, :, cid].sum()`（`_v5_replay.py:133,145`）")
    say("     是 **`cid` 这个类通道在全部 anchor 上的和** —— 只要同图里有**同类** GT，就互相污染。")
    say("     实测反例：")
    b = max(V5R, key=lambda r: r["target_sum_all"])
    say(f"       n_pos={b['n_pos']}  pos_target_mean={b['pos_target_mean']:.4f}  "
        f"best_assign_iou={b['best_assign_iou']:.4f}  target_sum_all={b['target_sum_all']:.4f}")
    say(f"       若它是 per-GT 的量，应 ≈ n_pos×mean = {b['n_pos']*b['pos_target_mean']:.4f}；"
        f"实测 {b['target_sum_all']:.4f} ⇒ **不是 per-GT 量**")
    say("     ⇒ **本轮不使用 `target_sum_all` 作为 per-GT 质量**。per-GT 目标质量改用")
    say("       **`n_pos × pos_target_mean`**（= `ts_g[pos]` 的和，`pos` 是该 GT 自己的正样本）✓")
    say("     ⇒ 但 `target_sum_all` 可用于**重建 batch 分母**：按 (mosaic_sample, cls) 去重后求和。")
    say("")
    # 语义陷阱 2：pos_target_max == best_assign_iou
    ptm = np.array([r["pos_target_max"] for r in V5R], float)
    bai = np.array([r["best_assign_iou"] for r in V5R], float)
    dd = np.abs(ptm - bai)
    say("  ★ **结构恒等式验证**（§3 推出 `pos_target_max == best_assign_iou`）：")
    say(f"     `_v5_replay` n={len(V5R)}：max|Δ| = {dd.max():.3e}   median|Δ| = {np.median(dd):.3e}"
        f"   #(|Δ|>1e-5) = {int((dd > 1e-5).sum())}")
    say("     ⇒ **证实**：每个 GT 的**峰值分类目标 = 它最好的正样本框与它的 CIoU**。")
    say("     ⇒ 几何质量**直接封顶**分类监督强度（target 初值是 1.0，被归一化拉低到 CIoU 水平）。")
    bad = np.where(dd > 1e-5)[0]
    say(f"     ⚠ 例外 {len(bad)}/{len(V5R)}（{len(bad)/len(V5R):.2%}），其中 "
        f"{int((ptm[bad] > bai[bad]).sum())} 例 pos_target_max **高于** best_assign_iou")
    say("       真因：插桩取的是 `get_pos_mask` 的 **mask_pos（冲突消解前）**，而 "
        "`select_highest_overlaps` 事后会把部分 anchor 改判给别的 GT；")
    say("       被改判走的那部分 anchor 上，`ts_g[pos]` 反映的是**同图同类另一个 GT**的目标。")
    say("       ⇒ 这是**插桩侧的测量瑕疵**，不是 loss 公式的性质（公式在 98.6% 上精确复现）。")
    say("       量级：|Δ| p50 = 0.000，p95 = "
        f"{np.percentile(np.abs(dd),95):.4f}，max = {np.abs(dd).max():.4f} ⇒ 不影响 §5 的尺度形态。")

    # ================================================================ §5 监督表（native 分桶）
    say("")
    say("=" * 112)
    say("[§5] SUPERVISION TABLE —— native 分桶（用 `_sge_supervision.json` 的 native_area）")
    say("=" * 112)
    nat = np.array([r["native_area"] for r in R if r.get("native_area")], float)
    say(f"  有效并含 native_area 的行 = {len(nat)} / {len(R)}")
    tab = {}
    say(f"  {'scale':<8}{'n_GT':>7}{'n_pos med':>11}{'best_align med':>16}"
        f"{'pos_target_max med':>20}{'layer HR%':>11}")
    for b in ("small", "medium", "large"):
        rs = [r for r in R if r.get("native_area") and buck(r["native_area"]) == b]
        if not rs:
            continue
        npos = np.array([r["n_pos"] for r in rs], float)
        ba = np.array([r["best_align"] for r in rs], float)
        pt = np.array([r["pos_target_max"] for r in rs], float)
        hr = np.mean([r.get("layer") == "HR" for r in rs])
        tab[b] = dict(n=len(rs), n_pos=desc(npos), best_align=desc(ba),
                      pos_target_max=desc(pt), hr_frac=float(hr))
        say(f"  {b:<8}{len(rs):>7}{np.median(npos):>11.0f}{np.median(ba):>16.4f}"
            f"{np.median(pt):>20.4f}{hr:>11.1%}")
    s_, m_, l_ = (tab.get("small"), tab.get("medium"), tab.get("large"))
    if s_ and l_:
        say("")
        say(f"  small vs medium  n_pos：MWU p={mwu_p([r['n_pos'] for r in R if r.get('native_area') and buck(r['native_area'])=='small'], [r['n_pos'] for r in R if r.get('native_area') and buck(r['native_area'])=='medium']):.3g}")
        say(f"  small vs large   n_pos：MWU p={mwu_p([r['n_pos'] for r in R if r.get('native_area') and buck(r['native_area'])=='small'], [r['n_pos'] for r in R if r.get('native_area') and buck(r['native_area'])=='large']):.3g}")
    say("")
    say("  ⚠ `_v5_replay` 只有 aug 尺度 ⇒ 其表按 aug_area 分桶，**明确标注为 aug-bucket**（混入增强）")
    say("  `_v5_replay`（aug_area 分桶）补充列（per-GT 目标质量 = n_pos × pos_target_mean）：")
    say(f"  {'aug-bucket':<11}{'n_GT':>7}{'n_pos med':>11}{'mean_align med':>16}"
        f"{'sum_align med':>15}{'pos_target_mean med':>21}{'target_mass med':>17}")
    tab2 = {}
    for b in ("small", "medium", "large"):
        rs = [r for r in V5R if r["bucket"] == b]
        if not rs:
            continue
        npos = np.array([r["n_pos"] for r in rs], float)
        pm = np.array([r["pos_target_mean"] for r in rs], float)
        ma = np.array([r["mean_align"] for r in rs], float)
        sa = np.array([r["n_pos"] * r["mean_align"] for r in rs], float)
        mass = np.array([r["n_pos"] * r["pos_target_mean"] for r in rs], float)
        tab2[b] = dict(n=len(rs), n_pos=desc(npos), mean_align=desc(ma), sum_align=desc(sa),
                       pos_target_mean=desc(pm), target_mass=desc(mass))
        say(f"  {b:<11}{len(rs):>7}{np.median(npos):>11.0f}{np.median(ma):>16.4f}"
            f"{np.median(sa):>15.4f}{np.median(pm):>21.4f}{np.median(mass):>17.4f}")
    say("")
    say("  §4 要求的 8 项逐项可得性：")
    say(f"    {'#':<3}{'量':<36}{'来源':<52}{'状态'}")
    avail = [
        (1, "positive anchor 数 n_pos", "_sge_supervision / _v5_replay", "AVAILABLE"),
        (2, "raw TAL alignment（每正样本）", "只有 max/mean，逐 anchor 未落盘", "PARTIAL"),
        (3, "max alignment", "_sge_supervision.best_align / _v5_replay.best_align", "AVAILABLE"),
        (4, "sum alignment", "= n_pos × mean_align（_v5_replay，导出量）", "DERIVED"),
        (5, "normalized target score", "pos_target_max / pos_target_mean（归一化后 ✓）", "AVAILABLE"),
        (6, "target score sum（per-GT）", "`target_sum_all` **不是 per-GT**（语义陷阱 A）；"
                                        "改用 n_pos×pos_target_mean", "DERIVED"),
        (7, "target score mean（per-GT）", "pos_target_mean", "AVAILABLE"),
        (8, "target score max（per-GT）", "pos_target_max（98.6% 精确 = best_assign_iou）", "AVAILABLE"),
    ]
    for a, b_, c_, d_ in avail:
        say(f"    {a:<3}{b_:<36}{c_:<52}{d_}")

    # ================================================================ §4/§6 normalization 分析
    say("")
    say("=" * 112)
    say("[§6] NORMALIZATION ANALYSIS —— small 是否因归一化获得**额外**的 supervision deficit？")
    say("=" * 112)
    say("  `norm_align_metric[a] = align[a]/maxAlign[g] × maxCIoU[g]`  ⇒  target_max[g] = maxCIoU[g]")
    say("  ⇒ 归一化的效果是：**把 raw one-hot 1.0 拉到该 GT 的 CIoU 水平**。")
    say("  ⇒ 因此 absolute target 水平 ∝ 几何质量，而 **不是** small 特有的惩罚。")
    say("     问题是 small 的 CIoU 是否显著更低。用 val 侧（`_v6_domains` V1，非配对）与 train 侧各查一次：")
    dom = json.loads((DOM / "_v6_domains.json").read_text(encoding="utf-8"))
    V1 = {r["key"]: r for r in dom if r["domain"] == "V1"}
    v3 = json.loads((V2D / "_v3_analysis.json").read_text(encoding="utf-8"))
    ids = json.loads((P3 / "_v11_ids.json").read_text(encoding="utf-8"))
    G1 = list(v3["E1"]); G2 = [v[0] for v in v3["controls"].values()]; G4 = list(ids["G4"])
    v7 = {r["key"]: r for r in json.loads((P3 / "_v7_vectors.json").read_text(encoding="utf-8"))["rows"]
          if r["domain"] == "V1"}
    say("")
    say(f"  {'set':<26}{'n':>6}{'best_assign_iou med':>21}{'pos_target_max 预测值':>24}")
    for nm, ks in (("val G1（hard miss）", G1), ("val G2（matched ctrl）", G2), ("val G4（success）", G4)):
        vv = np.array([V1[k]["best_assign_iou"] for k in ks if k in V1], float)
        say(f"  {nm:<26}{len(vv):>6}{np.median(vv):>21.4f}{np.median(vv):>24.4f}")
    say("  `_sge_supervision`（train，native-small）的 pos_target_max 见 §5 表。")
    say("")
    say("  ⚠ **关键限定**：G1/G2/G4 是 **validation** GT；train-side artifact 是 **train** GT。")
    say("     二者**不可配对**。下面所有 G1/G2/G4 的引用都是 **population 级对比**，")
    say("     不是「G1 在训练时收到了多少监督」。协议 §10 明令禁止后者。")

    # ================================================================ §7 背景稀释
    say("")
    say("=" * 112)
    say("[§7] BACKGROUND DILUTION —— 是否存在证据证明 small 的分类梯度被背景 anchor 稀释？")
    say("=" * 112)
    say("  从源码可以直接判定**归一化并不按 anchor 数除**：")
    say("      loss[1] = Σ_{(b,a,c)} BCE(logit, target) / Σ_{(b,a,c)} target      （loss.py:535 / :523）")
    say("      分母只含 **target 非零** 的项（背景项 target=0，不贡献分母）。")
    say("  ⇒ 每个正样本项的梯度权重是**共享的 1/D**，**与同图 anchor 总数无关**。")
    say("  ⇒ **「背景 anchor 越多 ⇒ 某 GT 的梯度被按比例稀释」在源码层面不成立**（NOT SUPPORTED）。")
    say("  ⇒ 背景项的作用是**叠加一个把全类 logit 往下压的梯度**（balance 问题），不是除法稀释。")
    say("")
    say("  结构性对照（可由源码计数，无需 artifact）：")
    n_anchor = 160 * 160 + 80 * 80 + 40 * 40
    _samp = sorted({r["mosaic_sample"] for r in V5R})
    _npos_per_img = [sum(r["n_pos"] for r in V5R if r["mosaic_sample"] == i) for i in _samp]
    say(f"     单图 anchor 总数（1280 输入，stride 8/16/32）= {n_anchor}")
    say(f"     每图 class-channel 项 = {n_anchor} × 12 = {n_anchor*12}")
    say(f"     `_v5_replay` 每图正样本数中位 = {np.median(_npos_per_img):.0f}"
        f"（{len(_samp)} 个样本）")
    say(f"     ⇒ **非零 target 项 : 全部 class-channel 项 ≈ "
        f"1:{ (n_anchor*12)/max(np.median(_npos_per_img),1):.0f}**")
    say("     （每个正样本贡献 **恰好 1** 个非零 target；其余 11 个同类通道项 target=0）")
    say("     ⇒ 这是**项数**比，**不是梯度幅度比** —— 梯度幅度还取决于误差 (σ(z) − t)。")
    say("  ⚠ 协议 §11：**没有 gradient artifact ⇒ 不能声称 small 收到的梯度更弱**。")
    say("     本轮结论限于「**target 数值**层面」，不涉及梯度幅度。⇒ 梯度层面的稀释 = **NOT PROVEN**")

    # ================================================================ §8 batch 分母与 per-GT share
    say("")
    say("=" * 112)
    say("[§8] BATCH 分母重建 与 per-GT TARGET-MASS SHARE")
    say("=" * 112)
    den = defaultdict(float)
    for r in V5R:
        den[(r["mosaic_sample"], r["cls"])] = r["target_sum_all"]   # 每 (样本,类) 唯一
    D_of = defaultdict(float)
    for (i, c), v in den.items():
        D_of[i] += v
    mass = {id(r): r["n_pos"] * r["pos_target_mean"] for r in V5R}
    share = np.array([mass[id(r)] / max(D_of[r["mosaic_sample"]], 1e-9) for r in V5R])
    V5R2 = []
    for r, s in zip(V5R, share):
        d2 = dict(r)
        d2["_share"] = float(s)
        d2["_D"] = float(D_of[r["mosaic_sample"]])
        V5R2.append(d2)
    say(f"  重建：按 (mosaic_sample, cls) 去重求和 = batch 分母 D（loss.py:523 的 target_scores.sum()）")
    say(f"    可重建的样本数 = {len(D_of)}；D 中位 = {np.median(list(D_of.values())):.3f}")
    say(f"  per-GT target-mass share = (n_pos × pos_target_mean) / D")
    say(f"  {'aug-bucket':<11}{'n':>6}{'share med':>12}{'share p90':>11}{'mass med':>11}{'D med':>10}")
    shr = {}
    for b in ("small", "medium", "large"):
        rs = [r for r in V5R2 if r["bucket"] == b]
        if not rs:
            continue
        sv = np.array([r["_share"] for r in rs], float)
        mv = np.array([r["n_pos"] * r["pos_target_mean"] for r in rs], float)
        dv = np.array([r["_D"] for r in rs], float)
        shr[b] = desc(sv)
        say(f"  {b:<11}{len(rs):>6}{np.median(sv):>12.5f}{np.percentile(sv,90):>11.5f}"
            f"{np.median(mv):>11.4f}{np.median(dv):>10.2f}")
    say("")
    say("  同图**等分基准**归一：`share_i × n_GT_in_image`（1.0 = 恰好拿到等分份额）")
    ng = defaultdict(int)
    for r in V5R2:
        ng[r["mosaic_sample"]] += 1
    say(f"  {'aug-bucket':<11}{'n':>6}{'share×n_GT med':>17}{'p10':>9}{'p90':>9}")
    equal = {}
    for b in ("small", "medium", "large"):
        rs = [r for r in V5R2 if r["bucket"] == b]
        if not rs:
            continue
        ev = np.array([r["_share"] * ng[r["mosaic_sample"]] for r in rs], float)
        equal[b] = desc(ev)
        say(f"  {b:<11}{len(rs):>6}{np.median(ev):>17.4f}{np.percentile(ev,10):>9.4f}"
            f"{np.percentile(ev,90):>9.4f}")
    say("")
    say("  ⚠ 读法：share 是**分母归一化后的 target 质量占比**，即该 GT 的正样本项在")
    say("     该图分类 loss 中的**权重份额**（同图 GT 之间可比）。它**不是梯度测量**。")
    say("  ⚠ **aug-bucket vs native-bucket 的分歧必须点明**：")
    say("     按 native 分桶时 small 的 n_pos 中位是 **4**（vs large 10）—— 数量赤字明显；")
    say("     按 aug 分桶时 small 的 n_pos 中位是 **9**（vs large 10）—— 数量赤字几乎消失。")
    say("     差异来自 **mosaic/RandomPerspective 的放大**：native-small 目标被放大后进入 aug-medium/large。")
    say("     ⇒ 任何用 aug 尺度做的 small/large 比较都会**系统性低估数量赤字**（与既有 MEMORY 一致）。")

    # ================================================================ §9 class imbalance
    say("")
    say("=" * 112)
    say("[§9] CLASS IMBALANCE（train labels 直接统计；native_area 分桶）")
    say("=" * 112)
    tot = Counter(); small = Counter(); imgs = defaultdict(set)
    from PIL import Image
    n_files = 0
    for ip in sorted(p for p in TR_IMG.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}):
        lp = TR_LAB / f"{ip.stem}.txt"
        if not lp.exists():
            continue
        n_files += 1
        try:
            with Image.open(ip) as im:
                W, H = im.size
        except Exception:  # noqa: BLE001
            continue
        for ln in lp.read_text(encoding="utf-8").splitlines():
            f = ln.split()
            if len(f) < 5:
                continue
            c = int(float(f[0]))
            a = float(f[3]) * W * float(f[4]) * H
            tot[c] += 1
            if a < SMALL_A:
                small[c] += 1
                imgs[c].add(ip.stem)
    T = sum(tot.values()); S = sum(small.values())
    say(f"  train 图像 {n_files} 张；GT 总数 {T}；其中 native-small {S}（{S/max(T,1):.1%}）")
    say(f"  {'class':<13}{'total GT':>10}{'small GT':>10}{'small ratio':>12}{'small imgs':>11}"
        f"{'total share':>12}")
    cls_tab = {}
    for c in range(12):
        cls_tab[NAMES[c]] = dict(total=tot[c], small=small[c],
                                 small_ratio=small[c] / max(tot[c], 1),
                                 small_imgs=len(imgs[c]), total_share=tot[c] / max(T, 1))
        say(f"  {NAMES[c]:<13}{tot[c]:>10}{small[c]:>10}{small[c]/max(tot[c],1):>12.1%}"
            f"{len(imgs[c]):>11}{tot[c]/max(T,1):>12.1%}")
    say("")
    say("  ⚠ 不做 12 类的显著性解释（协议 §12）。只报频率与比例。")

    # ================================================================ §12/§13 关系
    say("")
    say("=" * 112)
    say("[§12/§13] 关系检查（train population；不出显著性结论）")
    say("=" * 112)
    say("  (a) train 侧：native-small 的 per-GT target mass vs 其 CIoU")
    rs_s = [r for r in V5R2 if r["bucket"] == "small"]
    mv = np.array([r["n_pos"] * r["pos_target_mean"] for r in rs_s], float)
    cv = np.array([r["best_assign_iou"] for r in rs_s], float)
    npv = np.array([r["n_pos"] for r in rs_s], float)
    r1, p1 = spearman(mv, cv); r2, p2 = spearman(mv, npv)
    say(f"     n={len(rs_s)}  mass vs best_assign_iou：Spearman ρ={r1:+.4f} p={p1:.3g}")
    say(f"     n={len(rs_s)}  mass vs n_pos           ：Spearman ρ={r2:+.4f} p={p2:.3g}")
    say("  (b) class 频率 vs train 侧 small 监督质量（按类聚合，n=12 类）")
    fr, mq = [], []
    for c in range(12):
        rr = [r for r in V5R2 if r["cls"] == c and r["bucket"] == "small"]
        if len(rr) < 5:
            continue
        fr.append(tot[c] / max(T, 1))
        mq.append(float(np.median([r["n_pos"] * r["pos_target_mean"] for r in rr])))
    r3, p3 = spearman(fr, mq)
    say(f"     可用于该检验的类数 = {len(fr)}（每类 small 行 ≥5）")
    say(f"     Spearman ρ={r3:+.4f}  p={p3:.3g}   ⇒ **n 太小，不作结论**（协议 §9/§12）")

    # ================================================================ §10 G1/G2/G4
    say("")
    say("=" * 112)
    say("[§10] G1 / G2 / G4 —— 严格区分 train population 与 validation population")
    say("=" * 112)
    say("  **不可做的推断**：G1/G2/G4 是 validation GT，train-side artifact 里没有它们的身份。")
    say("     因此**不能**写出「G1 在训练时收到了 X」。下面只是两个 population 的**并置对照**。")
    say("")
    v1_bi = np.array([V1[k]["best_assign_iou"] for k in G1 if k in V1], float)
    v1_tm = np.array([V1[k]["best_assign_iou"] for k in G1 if k in V1], float)  # = 预测的 target_max
    tr_small_pt = np.array([r["pos_target_max"] for r in R
                            if r.get("native_area") and buck(r["native_area"]) == "small"], float)
    say(f"  {'population':<40}{'n':>6}{'median':>10}{'p10':>9}{'p90':>9}")
    say(f"  {'train native-small: pos_target_max':<40}{len(tr_small_pt):>6}"
        f"{np.median(tr_small_pt):>10.4f}{np.percentile(tr_small_pt,10):>9.4f}"
        f"{np.percentile(tr_small_pt,90):>9.4f}")
    say(f"  {'val G1: best_assign_iou (=target_max 预测)':<40}{len(v1_bi):>6}"
        f"{np.median(v1_bi):>10.4f}{np.percentile(v1_bi,10):>9.4f}{np.percentile(v1_bi,90):>9.4f}")
    v1_pt = np.array([V1[k]["pos_prob"] for k in G1 if k in V1], float)
    say(f"  {'val G1: 实测 GT-class sigmoid (pos_prob)':<40}{len(v1_pt):>6}"
        f"{np.median(v1_pt):>10.6f}{np.percentile(v1_pt,10):>9.2e}{np.percentile(v1_pt,90):>9.2e}")
    say("")
    say(f"  ⇒ **关键落差**：train native-small 的 target 峰值中位 {np.median(tr_small_pt):.3f}，")
    say(f"     而 val G1 的同类响应中位 {np.median(v1_pt):.2e} —— 相差约 "
        f"{np.log10(max(np.median(tr_small_pt),1e-9)/max(np.median(v1_pt),1e-12)):.0f} 个数量级。")
    say("     **监督被「下达」了（target≈0.6），响应却没有形成。**")

    # ================================================================ §13b 判别性检查
    say("")
    say("=" * 112)
    say("[§13b] ★ 判别性检查 —— 赤字是「尺度相关」还是「只对 hard-miss 子集成立」？")
    say("=" * 112)
    say("  若纯 supervision 赤字（A/B）能解释，则**同尺度的成功目标**也应显示更低的 GT 类 logit。")
    say("  用 val 侧全部 2807 GT 的 `logit_decomp`（GT 中心 cell 的 W_c·h+b_c），按 native 分桶 × 检出状态：")
    say("")
    say(f"  {'native bucket':<14}{'detected':<10}{'n':>6}{'GT类 logit 中位':>17}"
        f"{'sigmoid':>12}{'train 侧 pos_target_max 中位':>28}")
    # 全 val GT 表（native） + 检出状态（来自冻结 D′ TXT，纯 IoU 计算，无 forward）
    import numpy as np2  # noqa: F401
    from official_eval import read_pred_txt as _rpt, norm_xywh_to_xyxy as _n2x,         apply_max_boxes as _amb, box_iou_np as _biou, read_gt_txt as _rgt, _imsize as _ims
    from pathlib import Path as _P
    VALIMG = ROOT / "data/processed/rgbid_split_train/images/val/visible"
    PRED = ROOT / "diagnostic/sepstem_clahe/best_full/results"
    VALGT = {}
    for ip in sorted(x for x in VALIMG.iterdir()
                     if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}):
        W_, H_ = _ims(ip)
        if W_ <= 1 or H_ <= 1:
            continue
        gt = _rgt(VA_LAB / f"{ip.stem}.txt")
        if len(gt) == 0:
            continue
        gb, gc = _n2x(gt, W_, H_)
        _, pr, _ = _rpt(PRED / f"{ip.stem}.txt")
        pr = _amb(pr, 100)
        if len(pr):
            pb, pc = _n2x(pr, W_, H_)
        else:
            pb, pc = np.zeros((0, 4), np.float32), np.zeros(0, int)
        for j in range(len(gb)):
            a = max(float(gb[j, 2] - gb[j, 0]), 0) * max(float(gb[j, 3] - gb[j, 1]), 0)
            same = np.where(pc == gc[j])[0] if len(pc) else np.zeros(0, int)
            iou = _biou(gb[j:j + 1], pb[same])[0] if len(same) else np.zeros(0, np.float32)
            nw = float(gt[j, 3]); nh = float(gt[j, 4])          # 归一化 w/h
            VALGT[f"{ip.stem}#{j}"] = dict(native_area=float(a), cls=int(gc[j]),
                                           detected=bool(len(iou) and float(iou.max()) >= 0.5),
                                           wnorm=float(nw), hnorm=float(nh),
                                           fullspan=bool(nw >= 0.9 or nh >= 0.9))
    say(f"  自建 val GT 表：n={len(VALGT)}（native_area + 检出状态来自冻结 TXT 的纯 IoU 计算）")
    n_fs = sum(1 for r in VALGT.values() if r["fullspan"])
    say("  ⚠ **必须先声明该测量的层级范围**（本轮新发现，源自实测异常）：")
    say("     `_v7_extract.py:52`  `last = model.model[-1].cv3[0][-1]` ⇒ `logit_decomp` 是")
    say("     **stride-8（P3）那一条 class 分支**的 logit —— 它**不是**「模型对任意尺度的类别响应」。")
    say("     实测证据：输入 sqrt≥96 的 670 个 GT，`logit_decomp` 中位 **−17.22**、**p90 = −14.64、无一 > 0**；")
    say("     而这些 GT 显然是被检出的（大目标 AP@0.50 ≈ 0.79）。")
    say("     原因：**大目标由 stride-16/32 分支负责**，P3(stride-8) 本来就不为大目标输出。")
    say("     抽检 3 例证实 v6 与 v7 一致（−15.3/−15.5/−17.6 vs −17.1/−16.3/−19.3）⇒ **不是 bug，是层级范围**。")
    say("     ⇒ **large 一行在本表不可解释**，只作「该测量不适用于 large」的证据保留。")
    say("     ⇒ small 是本测量的**正确层级**（stride-8 就是小目标的分支）⇒ small 结论有效。")
    say(f"     另剔除「跨全幅标签」（`w_norm≥0.9` 或 `h_norm≥0.9`）：n={n_fs}/{len(VALGT)}（其次要原因，中心不在目标上）。")
    lv = {}
    for b in ("small", "medium", "large"):
        for det in (True, False):
            ks = [k for k, r in VALGT.items()
                   if buck(r["native_area"]) == b and r["detected"] == det and k in v7
                   and not r["fullspan"]]
            if len(ks) < 10:
                continue
            lg = np.array([v7[k]["logit_decomp"] for k in ks], float)
            lv[(b, det)] = desc(lg)
            say(f"  {b:<14}{str(det):<10}{len(lg):>6}{np.median(lg):>17.4f}"
                f"{1/(1+np.exp(-np.median(lg))):>12.4f}"
                f"{(np.median([r['pos_target_max'] for r in R if r.get('native_area') and buck(r['native_area'])==b]) if b in tab else float('nan')):>28.4f}")
    say("")
    say("  （下表已剔除全幅框；large 行因层级范围不可解释，不参与结论）")
    a1 = lv.get(("small", True), {}).get("median", float("nan"))
    a2 = lv.get(("medium", True), {}).get("median", float("nan"))
    a3 = lv.get(("large", True), {}).get("median", float("nan"))
    b1 = lv.get(("small", False), {}).get("median", float("nan"))
    say(f"  ⇒ **检出的** GT：native-small {a1:+.3f} vs medium {a2:+.3f} vs large {a3:+.3f}")
    say(f"  ⇒ **未检出的** native-small = {b1:+.3f}")
    m1 = lv.get(("medium", False), {}).get("median", float("nan"))
    say(f"    差（未检出 − 检出）= {b1-a1:+.3f} logit")
    say(f"  ⇒ medium 的未检出 = {m1:+.3f} ⇒ **『未检出』的 logit 水平在 small 与 medium 上几乎相同**")
    say(f"     （{b1:+.2f} vs {m1:+.2f}），而『检出』的差异也小（{a1:+.2f} vs {a2:+.2f}）。")
    say("  ⇒ **val GT-class 响应是『检出/未检出』两态，且该两态的水平与 native 尺度基本无关。**")
    say("     而监督侧却是**尺度渐变**的（n_pos 4→10、target 0.87→0.96）⇒")
    say("     **尺度渐变的监督赤字无法解释尺度不变的响应两态。**")
    say("")
    say("  ⇒ **读法**：如果 A/B（监督数量/强度赤字）是主因，那么 small 的**成功**目标也应当明显低；")
    say("     如果 small 成功目标的 logit 与 large 相差不大、而『未检出的小目标』塌到很低，")
    say("     那么赤字**不是尺度整体性的**，而是集中在 hard-miss 子集上 ⇒ supervision 赤字不足以解释。")

    say("")
    say("  [§13c] val GT-class logit（stride-8 分支）的**分布形状**（native-small，剔除全幅框）")
    ks_all = [k for k, r in VALGT.items()
              if buck(r["native_area"]) == "small" and not r["fullspan"] and k in v7]
    lgv = np.array([v7[k]["logit_decomp"] for k in ks_all], float)
    bins = [(-40, -8), (-8, -4), (-4, -2), (-2, 0), (0, 1), (1, 3)]
    say(f"     n={len(lgv)}   分箱直方图：")
    for lo, hi in bins:
        m = (lgv >= lo) & (lgv < hi)
        say(f"       [{lo:>4},{hi:>3})  n={int(m.sum()):>4}  {100*m.mean():>6.1f}%  "
            f"{'#'*int(round(60*m.mean()))}")
    _mid = float(((lgv >= -8) & (lgv < 0)).mean())
    say(f"     ⇒ **读法更正（我上一版写成『两态、中间为空』，是过度断言）**：")
    say(f"        实测中间区间 (−8, 0) 占 **{_mid:.1%}**，(−2,0) 单箱就有 12.9% ⇒")
    say("        **分布是宽的、两峰但没有空档**。`+0.49` 与 `−12.15` 是**分组中位数**，")
    say("        **不是**总体上的两态分离。不得据此声称『存在一个离散的坏态』。")
    say("        可支持的弱陈述：small 的响应分布**明显比 medium 更左移且更宽**。")
    say("")
    say("  [§13d] native 尺度 × 未检出率（val，剔除全幅框）")
    say(f"     {'bucket':<10}{'n':>6}{'未检出':>8}{'未检出率':>10}")
    for b in ("small", "medium", "large"):
        rs = [r for r in VALGT.values() if buck(r["native_area"]) == b and not r["fullspan"]]
        if not rs:
            continue
        nd = sum(1 for r in rs if not r["detected"])
        say(f"     {b:<10}{len(rs):>6}{nd:>8}{nd/len(rs):>10.1%}")
    say("     ⇒ 未检出率随尺度**单调下降**（39.6% → 18.8% → 8.8%）。")
    say("     ⇒ 结合 §13b：尺度主要影响『**有多少个**落在低响应区』，而**低响应区本身有多低**")
    say("        在 small 与 medium 上大致相同（−12.15 vs −14.15）。")
    say("        ⇒ **尺度渐变的监督赤字（A/B）与『低响应区的高度』对不上**，")
    say("          但它**可以**解释『有多少个掉进去』这一半 —— 这两半必须分开陈述。")

    # class 频率 vs val 响应
    say("")
    say("  [§9b] class 频率 vs **val** GT-class 响应（n=12 类，描述性）")
    fr2, vr = [], []
    for c in range(12):
        ks = [k for k, r in VALGT.items() if r["cls"] == c and k in v7]
        if len(ks) < 10:
            continue
        fr2.append(tot[c] / max(T, 1))
        vr.append(float(np.median([v7[k]["logit_decomp"] for k in ks])))
    r5, p5 = spearman(fr2, vr)
    say(f"     可用于该检验的类数 = {len(fr2)}")
    say(f"     Spearman ρ(train 频率, val GT类 logit 中位) = {r5:+.4f}  p={p5:.3g}"
        f"  ⇒ **n=12，不作结论**")

    (OUT / "_tables.json").write_text(json.dumps(dict(
        provenance=prov, supervision_native=tab, supervision_aug=tab2,
        identity_pos_target_max_eq_best_assign_iou=dict(
            n=len(V5R), max_abs=float(dd.max()), median_abs=float(np.median(dd)),
            n_gt_1e5=int((dd > 1e-5).sum())),
        share_by_aug_bucket=shr, share_within_image_equal_basis=equal, class_imbalance=cls_tab,
        relations=dict(mass_vs_ciou_small=[r1, p1], mass_vs_npos_small=[r2, p2],
                       classfreq_vs_mass=[r3, p3, len(fr)]),
        g1_population=dict(n=int(len(v1_bi)),
                           train_small_pos_target_max_med=float(np.median(tr_small_pt)),
                           val_g1_best_assign_iou_med=float(np.median(v1_bi)),
                           val_g1_pos_prob_med=float(np.median(v1_pt))),
        n_anchor=int(n_anchor),
        native_x_detected={f"{k[0]}|{k[1]}": v for k, v in lv.items()},
        classfreq_vs_val_response=[r5, p5, len(fr2)],
    ), indent=2, ensure_ascii=False, default=lambda o: int(o) if isinstance(o, np.integer)
        else (float(o) if isinstance(o, np.floating) else (bool(o) if isinstance(o, np.bool_) else None))),
        encoding="utf-8")
    say("")
    say(f"[saved] {OUT/'_tables.json'}")
    (OUT / "AUDIT.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
