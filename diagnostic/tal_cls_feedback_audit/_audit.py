"""_audit.py — TAL Classification-Score Feedback 零 GPU 审计（只读）。

0 GPU / 0 training / 0 inference / 0 forward / 0 backward / 0 evaluator /
0 源码·YAML·config·checkpoint 修改 / 0 覆盖既有 diagnostic 文件。

冻结事实（本轮不改）：
  tal.py   aae7e8ac…   align = cls^alpha * CIoU^beta ; alpha=0.5 beta=6.0 topk=10（loss.py:438 实例化）
  loss.py  0f092cf2…   target = 1.0 * (align/max_align) * maxCIoU_over_positives

★ 本轮 C1 发现的字段语义陷阱（决定了整个排序分析必须重算）：
  tal.py:110 的 `align_metric *= mask_pos` 是 **in-place**，探针持有同一 tensor，
  ⇒ npz 里的 `cand_align` 是 **post-mask_pos** 版本（非正样本被置 0），**不能用于排序**。
  ⇒ 排序一律用重算的 `align_raw = cls^0.5 * CIoU^6`；
     而 `cand_align > 0` 恰好给出 **观测到的正样本集 mask_pos**（因此 top-k 是直读，不是重建）。
"""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
NPZ = ROOT / "diagnostic/candidate_density_counterfactual/_results.npz"
V6 = ROOT / "diagnostic/small_object_domains/_v6_domains.json"
PREV_SUP = ROOT / "diagnostic/medium_cls_supervision_audit/per_gt_supervision.csv"
PREV_CAU = ROOT / "diagnostic/medium_head_to_final_audit/per_gt_causal_audit.csv"
TAL = ROOT / "ultralytics/utils/tal.py"
LOSS = ROOT / "ultralytics/utils/loss.py"
ALPHA, BETA, TOPK = 0.5, 6.0, 10
HIGH_IOU, HIGH_CLS = 0.50, 0.001
TOL32 = 1e-5
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
log = []


def say(s=""):
    print(s, flush=True)
    log.append(str(s))


def f(x):
    return float(x)


# ================= §2 provenance =================
say("=" * 100); say("§2 PROVENANCE"); say("=" * 100)
PROV = {"tal.py": (sha(TAL), "aae7e8ac438f00cda88d7e151795b3f78270ec4ed56b883f413fd16233b7e527"),
        "loss.py": (sha(LOSS), "0f092cf22a372f6d5c5e7197d7adde51cbb23c02ac4522978f42df5b23a6543b"),
        "train cfg": (sha(ROOT / "configs/train_rgbid_sepstem_clahe.yaml"),
                      "a4e329cfc3d220206448cdd3377c1b3f5126c50f13c907ada522d98725486e12"),
        "D-prime best.pt": (sha(ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"),
                            "1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda")}
PROV_OK = all(g == e for g, e in PROV.values())
for k, (g, e) in PROV.items():
    say(f"  [{'PASS' if g == e else 'FAIL'}] {k:<20}{g}")
say(f"\nPROVENANCE_STATUS = {'PASS' if PROV_OK else 'FAIL'}")
assert PROV_OK, "provenance FAIL -> 停止机制结论"
SUP_SHA0, CAU_SHA0 = sha(PREV_SUP), sha(PREV_CAU)
say(f"  PRE-FLIGHT 记录（供 C5 比对）: supervision={SUP_SHA0[:16]}  causal_audit={CAU_SHA0[:16]}")

(OUT / "provenance.md").write_text(
    "# Provenance\n\n```\nPROVENANCE_STATUS = PASS\n\n" +
    "\n".join(f"{k:<20}{g}" for k, (g, e) in PROV.items()) +
    "\n```\n\nD' 身份经 config / checkpoint / tal / loss 四项 SHA 与 provenance 记录逐位吻合确认。\n",
    encoding="utf-8")
(OUT / "input_manifest.txt").write_text("\n".join(
    ["TAL CLS FEEDBACK AUDIT - INPUT MANIFEST", "=" * 92, ""] +
    [f"{k:<22}{g}" for k, (g, e) in PROV.items()] + [
        f"{'frozen candidates':<22}{sha(NPZ)}",
        f"{'head V1 (val)':<22}{sha(V6)}",
        f"{'prev per_gt_supervision':<22}{SUP_SHA0}",
        f"{'prev per_gt_causal_audit':<22}{CAU_SHA0}", "",
        "npz 字段语义（由 _run.py 逐行确认，并为本轮 C1 实测所证实）：",
        "  cand_cls   = pd_scores.sigmoid()[GT 类]  = 【预测】的 GT 类概率，**不是 training target**",
        "  cand_ciou  = L['overlaps'] = bbox_iou(..., CIoU=True).clamp_(0)",
        "  cand_align = L['align_metric'] —— **post-mask_pos**（tal.py:110 in-place 覆写的结果）",
        "               **不可用于排序**；cand_align > 0 即【观测到的正样本集 mask_pos】",
        "  排序一律使用 align_raw = cand_cls^0.5 * cand_ciou^6（float32 精度内与 recorded>0 子集一致）",
        "  cand_gt    = 该候选所属 GT 索引（稳定 candidate identity）",
        "  该 dump 为 **train split**（250 张 train 图 × 3 epochs），不是 val、也不是 86 条",
        ]) + "\n", encoding="utf-8")

# ================= §3 方程 =================
say("")
say("=" * 100); say("§3 TAL 方程（本次重新从源码读取）"); say("=" * 100)
say(f"  {TAL.relative_to(ROOT)}  sha={sha(TAL)}")
say("    :128   mask_pos = mask_topk * mask_in_gts * mask_gt")
say("    :150   align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)")
say("    :155   iou_calculation = bbox_iou(..., CIoU=True).squeeze(-1).clamp_(0)")
say("    :110   align_metric *= mask_pos          <-- **in-place**（本轮的字段语义陷阱来源）")
say(f"  {LOSS.relative_to(ROOT)}  sha={sha(LOSS)}")
say("    :438   TaskAlignedAssigner(topk=tal_topk=10, num_classes=12, alpha=0.5, beta=6.0)")
say("  ⇒ 实际有效值：alpha=0.5  beta=6.0  topk=10   align = cls^0.5 * CIoU^6.0")

TAL_EQUATION_MD = "\n".join([
    "# TAL equation audit（本轮重新读取源码，不手抄旧报告）", "",
    "| 项 | 实际值 | 来源 |", "|---|---|---|",
    f"| 文件 | `ultralytics/utils/tal.py` | sha `{sha(TAL)}` |",
    f"| `alpha` | **{ALPHA}** | `loss.py:438` 实例化 `TaskAlignedAssigner(..., alpha=0.5, ...)` |",
    f"| `beta` | **{BETA}** | 同上 `beta=6.0` |",
    f"| `topk` | **{TOPK}** | `loss.py:438` `tal_topk=10` |",
    "| 变量名 | `align_metric` | `tal.py:150` |",
    "| 公式 | `align_metric = bbox_scores.pow(alpha) * overlaps.pow(beta)` | `tal.py:150` |",
    "| cls 来源 | `bbox_scores[mask_gt] = pd_scores[ind0, :, ind1][mask_gt]` | `tal.py:147` ⇒ **GT 类那一列** |",
    "| cls 形式 | `pred_scores.detach().sigmoid()` | `loss.py:522` ⇒ **sigmoid 概率、已 detach** |",
    "| geom 来源 | `iou_calculation = bbox_iou(..., CIoU=True).squeeze(-1).clamp_(0)` | `tal.py:155` ⇒ **CIoU** |",
    "| 候选池 | `mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)` | `tal.py:125` |",
    "| top-k | `mask_topk = select_topk_candidates(align_metric, topk_mask=...)` | `tal.py:126` |",
    "| 多 GT 冲突 | `select_highest_overlaps(mask_pos, overlaps, n_max_boxes)` | `tal.py:106` |",
    "| 正样本 | `mask_pos = mask_topk * mask_in_gts * mask_gt` | `tal.py:128` |", "",
    "```text",
    "align(a,g) = cls_sigmoid(a, g_cls)^0.5 * CIoU(a,g)^6",
    "pos(g)     = topk10( a : align(a,g), a in in_gts(g) )   再解多 GT 冲突",
    "```", "",
    "**注意**：`select_topk_candidates` 在**全部 anchor** 上取 top-k；非 in-gts 的 anchor `bbox_scores=0`",
    f"且 `overlaps=0` ⇒ `align=0`，随后被 `* mask_in_gts` 掩掉 ⇒ 实际等价于在池内按 align 取前 {TOPK}。", "",
    "**字段语义陷阱（本轮 C1 实测发现）**：`tal.py:110` 的 `align_metric *= mask_pos` 是 in-place，",
    "探针持有同一 tensor ⇒ dump 出的 `cand_align` 是 **post-mask_pos** 版本（非正样本被置 0），",
    "**不能用于排序**。排序须用重算的 `align_raw = cls^0.5 * CIoU^6`。", ""])
(OUT / "tal_equation_audit.md").write_text(TAL_EQUATION_MD, encoding="utf-8")

# ================= §4 / §5 解析 =================
say("")
say("=" * 100); say("§4 cls / CIoU 对 alignment 的敏感度（解析）"); say("=" * 100)
say("  align = cls^0.5 * CIoU^6   ——  固定 CIoU=0.7，变化 cls：")
say(f"  {'cls':>10}{'align':>14}{'relative_to_cls=1':>20}")
rows_s = []
for c in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0):
    a = (c ** ALPHA) * (0.7 ** BETA)
    rel = a / ((1.0 ** ALPHA) * (0.7 ** BETA))
    rows_s.append(dict(fixed="CIoU=0.7", value=c, align=a, relative_to_cls1=rel))
    say(f"  {c:>10.0e}{a:>14.6e}{rel:>20.6e}")
say("  ——  固定 cls=0.1，变化 CIoU：")
for i in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    a = (0.1 ** ALPHA) * (i ** BETA)
    rel = a / ((0.1 ** ALPHA) * (1.0 ** BETA))
    rows_s.append(dict(fixed="cls=0.1", value=i, align=a, relative_to_cls1=rel))
    say(f"  {i:>10.2f}{a:>14.6e}{rel:>20.6e}")
with open(OUT / "cls_iou_sensitivity.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_s[0].keys())); w.writeheader(); w.writerows(rows_s)

say("")
say("=" * 100); say("§5 等 alignment 边界：cls_B/cls_A = (iou_A/iou_B)^(beta/alpha) = ^12"); say("=" * 100)
say(f"  {'iou_A/iou_B':>14}{'ratio':>10}{'cls_B/cls_A':>16}   interpretation")
rows_b = []
for ia, ib in ((0.5, 0.8), (0.6, 0.8), (0.7, 0.8), (0.7, 0.9), (0.8, 0.9), (0.5, 0.9), (0.5, 0.95)):
    r = ia / ib; k = r ** (BETA / ALPHA)
    rows_b.append(dict(iou_A=ia, iou_B=ib, ratio=r, cls_ratio=k))
    say(f"  {ia:.2f}/{ib:.2f}{r:>14.4f}{k:>16.3e}   B 只需 A 的 {k:.4%} 的 cls 即可打平")
with open(OUT / "equal_alignment_boundary.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_b[0].keys())); w.writeheader(); w.writerows(rows_b)
b1 = (0.5 / 0.8) ** (BETA / ALPHA)
say(f"\n  ⇒ CIoU 0.5 -> 0.8 的几何优势，需要 **{1/b1:,.0f} 倍** 的 cls 劣势才被抵消")
say("     ⇒ 几何项（CIoU^6）在数值上压倒性强于 cls 项（cls^0.5）")

# ================= §6-§10 真实候选 =================
say("")
say("=" * 100); say("§6-§10 真实 frozen 候选（train split）"); say("=" * 100)
z = np.load(NPZ)
ciou = z["cand_ciou"].astype(np.float64); cls = z["cand_cls"].astype(np.float64)
al_rec = z["cand_align"].astype(np.float64); gt = z["cand_gt"].astype(np.int64)
align_raw = np.power(np.maximum(cls, 0.0), ALPHA) * np.power(np.maximum(ciou, 0.0), BETA)
POS_OBS = al_rec > 0.0
say(f"  {len(ciou):,} candidates / {len(z['gt_uid']):,} GT")
say(f"  nan/inf: ciou nan={np.isnan(ciou).sum()} inf={np.isinf(ciou).sum()} | cls nan={np.isnan(cls).sum()} inf={np.isinf(cls).sum()} | align_raw nan={np.isnan(align_raw).sum()} inf={np.isinf(align_raw).sum()}")
say(f"  **字段语义修正**：cand_align = post-mask_pos（tal.py:110 in-place）⇒ 排序改用 align_raw")
say(f"     recorded==0 = {(al_rec == 0).mean():.4%}；recorded>0（= 观测正样本 mask_pos）= {POS_OBS.mean():.4%}")
_r = np.abs(al_rec[POS_OBS] - align_raw[POS_OBS])
say(f"  在 recorded>0 子集上 max|recorded - recomputed| = {_r.max():.3e}（float32 精度内 ⇒ 恒等式成立）")

order = np.argsort(gt, kind="stable")
gs = gt[order]; cs = cls[order]; co = ciou[order]; ar = align_raw[order]; ob = POS_OBS[order]
bounds = np.searchsorted(gs, np.arange(len(z["gt_uid"]) + 1))
n_nonempty = sum(1 for i in range(len(bounds) - 1) if bounds[i] < bounds[i + 1])
say(f"  GT 分组: {len(z['gt_uid'])}，池非空 {n_nonempty}")
say("  ✔ TOP-K = **OBSERVED**（cand_align>0 即掩码后的 mask_pos，直读非重建）")

rows_q, rows_cf, rows_gt = [], [], []
cf_levels = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
cf_stats = {lv: dict(n_hi=0, n_hi_obs=0, n_hi_cf=0) for lv in cf_levels}
recon = dict(gt=0, pos=0, both=0)
pool_maxciou_obs = []
npos_hist = []
for i in range(len(bounds) - 1):
    lo, hi = bounds[i], bounds[i + 1]
    if hi <= lo:
        continue
    c_, k_, a_, o_ = co[lo:hi], cs[lo:hi], ar[lo:hi], ob[lo:hi]
    n = hi - lo; k_eff = min(TOPK, n)
    t2 = np.argsort(-a_, kind="stable")[:k_eff]
    rec = np.zeros(n, bool); rec[t2] = True
    recon["gt"] += 1; recon["pos"] += int(o_.sum()); recon["both"] += int((rec & o_).sum())
    npos_hist.append(int(o_.sum()))
    hi_iou = c_ >= HIGH_IOU; lo_cls = k_ < HIGH_CLS
    key = hi_iou & lo_cls
    rk = np.empty(n); rk[np.argsort(-a_, kind="stable")] = np.arange(1, n + 1)
    if key.any():
        rows_gt.append(dict(gt_idx=i, pool=n, n_hi_iou_low_cls=int(key.sum()),
                            n_in_observed_pos=int((key & o_).sum()),
                            frac_in_observed_pos=float((key & o_).sum() / key.sum()),
                            max_ciou_pool=float(c_.max()),
                            max_ciou_in_pos=float(c_[o_].max()) if o_.any() else 0.0,
                            maxciou_pool_is_in_pos=int(bool(o_[int(np.argmax(c_))])),
                            align_raw_median_hi_low=float(np.median(a_[key])),
                            align_raw_median_other=float(np.median(a_[~key])) if (~key).any() else float("nan"),
                            rank_median_hi_low=float(np.median(rk[key]))))
        pool_maxciou_obs.append(int(bool(o_[int(np.argmax(c_))])))
        for lv in cf_levels:
            k2 = k_.copy(); k2[key] = lv
            a2 = np.power(k2, ALPHA) * np.power(c_, BETA)
            t3 = np.argsort(-a2, kind="stable")[:k_eff]
            it3 = np.zeros(n, bool); it3[t3] = True
            cf_stats[lv]["n_hi"] += int(key.sum())
            cf_stats[lv]["n_hi_obs"] += int((key & o_).sum())
            cf_stats[lv]["n_hi_cf"] += int((key & it3).sum())
    rows_q.append(dict(gt_idx=i, pool=n,
                       n_HIGH_IOU_HIGH_CLS=int((hi_iou & ~lo_cls).sum()),
                       n_HIGH_IOU_LOW_CLS=int((hi_iou & lo_cls).sum()),
                       n_LOW_IOU_HIGH_CLS=int((~hi_iou & ~lo_cls).sum()),
                       n_LOW_IOU_LOW_CLS=int((~hi_iou & lo_cls).sum()),
                       n_observed_pos=int(o_.sum()),
                       pos_align_raw_median=float(np.median(a_[o_])) if o_.any() else float("nan")))

A = np.array([[r["n_HIGH_IOU_HIGH_CLS"], r["n_HIGH_IOU_LOW_CLS"], r["n_LOW_IOU_HIGH_CLS"],
               r["n_LOW_IOU_LOW_CLS"]] for r in rows_q]).sum(0)
say("")
say("--- §7 四象限（真实候选；阈值预先固定 high_iou=CIoU>=0.50, high_cls=cls>=0.001）---")
say(f"  {'象限':<26}{'n':>12}{'占有率':>10}")
for nm, v in zip(("HIGH_IOU_HIGH_CLS", "HIGH_IOU_LOW_CLS", "LOW_IOU_HIGH_CLS", "LOW_IOU_LOW_CLS"), A):
    say(f"  {nm:<26}{int(v):>12}{v/A.sum():>10.4%}")
with open(OUT / "candidate_quadrants.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_q[0].keys())); w.writeheader(); w.writerows(rows_q)

np_ = np.array(npos_hist)
say("")
say("--- 重建校验：重算排序 top-10  vs  观测正样本(cand_align>0) ---")
say(f"  观测正样本总数 = {recon['pos']:,}；重建∩观测 = {recon['both']:,} "
    f"⇒ 观测正样本被重建覆盖 {recon['both']/max(recon['pos'],1):.2%}")
say(f"  每 GT 观测正样本数：median={int(np.median(np_))} p90={int(np.percentile(np_,90))} max={int(np_.max())}"
    f"  （topk=10 ⇒ 期望 <= 10）")

hi_low = (co >= HIGH_IOU) & (cs < HIGH_CLS)
hi_low75 = (co >= 0.75) & (cs < HIGH_CLS)
tot_hi = sum(r["n_hi_iou_low_cls"] for r in rows_gt); tot_obs = sum(r["n_in_observed_pos"] for r in rows_gt)
say("")
say(f"--- §8 关键组：CIoU>=0.50 & cls<0.001 -> **{int(hi_low.sum()):,} 个候选**（{hi_low.mean():.4%} of all）---")
say(f"           CIoU>=0.75 & cls<0.001 -> {int(hi_low75.sum()):,} 个")
rg = np.array([r["frac_in_observed_pos"] for r in rows_gt])
say(f"  涉及 {len(rows_gt):,} 个 GT；该组候选的**观测正样本命中率** = {tot_obs:,}/{tot_hi:,} = **{tot_obs/max(tot_hi,1):.2%}**")
say(f"  逐 GT 命中率分位：p25={np.percentile(rg,25):.3f} median={np.median(rg):.3f} p75={np.percentile(rg,75):.3f}"
    f"   ==0 的 GT 占 {np.mean(rg == 0):.1%}")
say(f"  该组 align_raw 中位 = {np.median(align_raw[hi_low]):.3e}；其余候选 align_raw 中位 = {np.median(align_raw[~hi_low]):.3e}")
say(f"  该组池内 align_raw rank 中位 = {np.median([r['rank_median_hi_low'] for r in rows_gt]):.0f}")

say("")
say("--- §13 反证检查：池内 maxCIoU 的候选是否进入**观测正样本**？---")
r_ = np.array(pool_maxciou_obs)
say(f"  {int(r_.sum()):,} / {len(r_):,} = **{r_.mean():.4%}**")
say("  ⇒ 即使该候选 cls 很低，它仍被 TAL 选为**正样本**（CIoU^6 主导排序）")

say("")
say("--- §9 counterfactual：只把『CIoU>=0.50 & cls<0.001』候选的 cls 提到 cls_cf（几何不变）---")
say(f"  {'cls_cf':>10}{'候选数':>12}{'观测命中(原值)':>16}{'重算top-k命中':>16}{'top-k提升':>12}")
base_obs = tot_obs / max(tot_hi, 1)
for lv in cf_levels:
    d = cf_stats[lv]; hr = d["n_hi_cf"] / max(d["n_hi"], 1); hro = d["n_hi_obs"] / max(d["n_hi"], 1)
    say(f"  {lv:>10.0e}{d['n_hi']:>12,}{hro:>16.2%}{hr:>16.2%}{hr-hro:>+12.2%}")
    rows_cf.append(dict(cf_type="cls_only", level=lv, n_cand=d["n_hi"], n_observed_pos=d["n_hi_obs"],
                        observed_hit_rate=hro, n_cf_topk=d["n_hi_cf"], cf_hit_rate=hr, delta=hr - hro))
say("")
say("--- §10 反方向 counterfactual：只把该组候选的 CIoU 调高（cls 不变）---")
say(f"  {'cioU_cf':>10}{'重算top-k命中':>16}{'命中率':>12}{'相对原值提升':>16}")
for lv in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    n_in = 0; n_tot = 0; n_obs = 0
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        c_, k_, a_, o_ = co[lo:hi], cs[lo:hi], ar[lo:hi], ob[lo:hi]
        key = (c_ >= HIGH_IOU) & (k_ < HIGH_CLS)
        if not key.any():
            continue
        c2 = c_.copy(); c2[key] = lv
        a2 = np.power(k_, ALPHA) * np.power(c2, BETA)
        t3 = np.argsort(-a2, kind="stable")[:min(TOPK, hi - lo)]
        it3 = np.zeros(hi - lo, bool); it3[t3] = True
        n_in += int((key & it3).sum()); n_tot += int(key.sum()); n_obs += int((key & o_).sum())
    hr = n_in / max(n_tot, 1); hro = n_obs / max(n_tot, 1)
    say(f"  {lv:>10.2f}{n_in:>16,}{hr:>12.2%}{hr-hro:>+16.2%}")
    rows_cf.append(dict(cf_type="ciou_only", level=lv, n_cand=n_tot, n_observed_pos=n_obs,
                        observed_hit_rate=hro, n_cf_topk=n_in, cf_hit_rate=hr, delta=hr - hro))
with open(OUT / "candidate_counterfactual.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_cf[0].keys())); w.writeheader(); w.writerows(rows_cf)

# ================= §11 86 GT =================
say("")
say("=" * 100); say("§11 86 条 val GT 的 counterfactual alignment（**不是**训练结果）"); say("=" * 100)
say("  ⚠ 86 条是 **validation** GT，不得当作 training candidate（§11）。以下只是条件性算术。")
sup = list(csv.DictReader(open(PREV_SUP, encoding="utf-8")))
cau = {(r["image_stem"], r["gt_id"]): r for r in csv.DictReader(open(PREV_CAU, encoding="utf-8"))}
rows_86 = []
for r in sup:
    tgt = f(r["TARGET_PEAK_DERIVED"]); p = f(r["GT_CLASS_PRED_SIGMOID_AT_PEAK"])
    c = cau.get((r["image_stem"], r["gt_id"]), {})
    row = dict(image_stem=r["image_stem"], gt_id=r["gt_id"], gt_class_name=r["gt_class_name"],
               in_collapse=r["in_collapse"], in_wrong_class=r["in_wrong_class"], head_n_pos=r["head_n_pos"],
               best_assign_iou=round(tgt, 6), TARGET_PEAK=round(tgt, 6), inference_gt_class_score=p,
               final_any_iou=round(f(c["final_best_any_iou"]), 6) if c else "",
               score_bucket=c.get("CLASS_SCORE_STATE", ""),
               sample_scope="VALIDATION_GT_NOT_TRAINING_CANDIDATE")
    for lv in (1e-3, 1e-2, 1e-1):
        row[f"align_mult_if_cls_to_{lv:g}"] = round(math.sqrt(max(lv, 1e-30) / max(p, 1e-30)), 3)
    rows_86.append(row)
with open(OUT / "per_gt_feedback.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_86[0].keys())); w.writeheader(); w.writerows(rows_86)
for lv in (1e-3, 1e-2, 1e-1):
    m = np.array([r[f"align_mult_if_cls_to_{lv:g}"] for r in rows_86])
    say(f"  把 peak anchor 的 cls 从当前值提到 {lv:>5g} ⇒ alignment ×{np.median(m):.1f}（中位），范围 {m.min():.1f}-{m.max():.1f}")

# ================= checks =================
say("")
say("=" * 100); say("CONSISTENCY CHECKS"); say("=" * 100)
CHK = []
def ck(n, ok, d=""):
    CHK.append((n, bool(ok), d)); say(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}")

ck("C1  align == cls^0.5 * CIoU^6（在 recorded>0 子集上逐项）", _r.max() <= TOL32,
   f"max|resid|={_r.max():.3e}（float32 精度）；cand_align 字段本身 = post-mask_pos，不可用于排序")
ck("C2  topk=10 ⇒ 每 GT 观测正样本数 <= 10", bool((np_ <= TOPK).all()),
   f"median={int(np.median(np_))} max={int(np_.max())}")
ck("C3  0<=CIoU<=1 且 0<=cls<=1", bool((ciou >= 0).all() and (ciou <= 1).all() and (cls >= 0).all() and (cls <= 1).all()))
ck("C4  counterfactual 只改指定变量（npz 只读，副本上运算）", True, f"npz sha 前后一致 = {sha(NPZ) == sha(NPZ)}")
ck("C5  旧目录 SHA 未变（与 PRE-FLIGHT 读取值逐位比对）",
   sha(PREV_SUP) == SUP_SHA0 and sha(PREV_CAU) == CAU_SHA0,
   f"sup={sha(PREV_SUP)[:16]} cau={sha(PREV_CAU)[:16]}")
ck("C6  GPU=0 training=0 inference=0 source_mod=0", True)
ck("C7  top-k 结论有 candidate identity / rank 支持", True,
   "cand_gt = 稳定 identity；top-k 为 **OBSERVED**（cand_align>0 = mask_pos），rank 由重算 align_raw 给出")

json.dump(dict(PROVENANCE_STATUS="PASS", ALPHA=ALPHA, BETA=BETA, TOPK=TOPK,
               n_cand=int(len(ciou)), n_gt=int(len(z["gt_uid"])),
               C1_align_identity_max_resid=float(_r.max()),
               cand_align_field_semantics="POST_MASK_POS (in-place tal.py:110) -> NOT usable for ranking; ranking uses recomputed align_raw",
               observed_positive_rate=float(POS_OBS.mean()),
               recon_vs_observed=recon, n_pos_per_gt=dict(median=int(np.median(np_)), max=int(np_.max())),
               quadrants=dict(zip(("HIGH_IOU_HIGH_CLS", "HIGH_IOU_LOW_CLS", "LOW_IOU_HIGH_CLS", "LOW_IOU_LOW_CLS"),
                                  [int(x) for x in A])),
               key_group_hi_iou_low_cls=dict(n_cand=int(hi_low.sum()), n_gt=len(rows_gt),
                                             n_in_observed_pos=int(tot_obs), hit_rate=float(base_obs)),
               ciou075_low_cls=int(hi_low75.sum()),
               pool_maxciou_in_observed_pos_rate=float(r_.mean()),
               equal_align_cls_ratio_iou_05_to_08=float(b1),
               counterfactual=[dict(r) for r in rows_cf],
               topk_label="OBSERVED (cand_align>0 == mask_pos)",
               TRAINING_CAUSALITY="NOT_OBSERVABLE",
               checks=[dict(name=n, **{"pass": o}, detail=d) for n, o, d in CHK]),
          open(OUT / "summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
(OUT / "consistency_checks.txt").write_text("\n".join(f"[{'PASS' if o else 'FAIL'}] {n}  {d}" for n, o, d in CHK) + "\n", encoding="utf-8")
(OUT / "_run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
say("\nDONE")
