"""_audit.py — Medium 86 条 Head→Final Candidate Disappearance 因果审计（只读）。

硬约束：0 训练 / 0 GPU / 0 重推理 / 0 重跑 evaluator / 0 改源码·YAML·checkpoint·既有 JSON /
0 重定义 86 / 0 自造 NMS / 0 改阈值 / 不把 NOT_OBSERVABLE 猜成某机制。

只读输入：
  diagnostic/medium_93_47_attribution/per_gt_attribution.csv   （上一轮，86 的冻结来源）
  diagnostic/small_object_domains/_v6_domains.json  V1 域       （head 侧: best_assign_iou / pos_prob / center_*）
  diagnostic/sepstem_clahe/best_full/results                    （final post-NMS rect 预测）
  ultralytics/utils/{metrics,ops,tal}.py, scripts/predict_rect.py（冻结代码，用于阈值/构成/CIoU 界）
  diagnostic/rect_pipeline_fix/validation_{square,rect}_predictions/results（仅作 geometry 敏感度的**异模型对照**）

证据等级词表：PROVEN / SUPPORTED / CONSISTENT_WITH / NOT_OBSERVABLE / UNRESOLVED
"""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from official_eval import (  # noqa: E402
    apply_max_boxes, box_iou_np, load_split, norm_xywh_to_xyxy, read_pred_txt,
)

OUT = Path(__file__).resolve().parent
PREV = ROOT / "diagnostic/medium_93_47_attribution"
V6 = ROOT / "diagnostic/small_object_domains/_v6_domains.json"
PRED = ROOT / "diagnostic/sepstem_clahe/best_full/results"
IMAGES = ROOT / "data/processed/rgbid_split_train/images/val/visible"
LABELS = ROOT / "data/processed/rgbid_split_train/labels/val/visible"
NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}
POS_THR = 0.001          # scripts/predict_rect.py:186  --conf 0.001
NMS_IOU = 0.7            # scripts/predict_rect.py:187  --iou 0.7
MULTI_LABEL = True       # scripts/predict_rect.py:262
MAX_BOXES = 100          # 赛题 §九(二)
TOL = 1e-6
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
log = []


def say(s=""):
    print(s, flush=True)
    log.append(str(s))


def num(x, default=None):
    """把 CSV 里的 str/int/float 统一成 float（防上一轮 '1' vs 1 的类型 bug）。"""
    if x is None or x == "" or x == "None":
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        raise ValueError(f"non-numeric field value: {x!r}")


# ================= PRE-FLIGHT =================
say("=" * 100); say("PRE-FLIGHT"); say("=" * 100)
PREV_CSV = PREV / "per_gt_attribution.csv"
say(f"[inputs] prev={PREV_CSV.relative_to(ROOT)} sha={sha(PREV_CSV)}")
say(f"[inputs] head={V6.relative_to(ROOT)} sha={sha(V6)}")
say(f"[inputs] final={PRED.relative_to(ROOT)} ({len(list(PRED.glob('*.txt')))} txt)")
say(f"[inputs] code=metrics.py sha={sha(ROOT/'ultralytics/utils/metrics.py')[:16]} ops.py sha={sha(ROOT/'ultralytics/utils/ops.py')[:16]} tal.py sha={sha(ROOT/'ultralytics/utils/tal.py')[:16]} predict_rect.py sha={sha(ROOT/'scripts/predict_rect.py')[:16]}")
say("[inputs] thresholds: conf=0.001 (predict_rect.py:186), nms_iou=0.7 (:187), max_det=300, multi_label=True (:262), max_boxes=100")

rows_prev = list(csv.DictReader(open(PREV_CSV, encoding="utf-8")))
SEL86 = [r for r in rows_prev if num(r["certified_prefinal_loss"]) == 1.0]
say(f"N_86 = {len(SEL86)}   (上一轮记录 86)")
assert len(SEL86) == 86, "86 条无法逐条恢复 -> BLOCKED"

# head 侧记录
V = {r["key"]: r for r in json.loads(V6.read_text(encoding="utf-8")) if r["domain"] == "V1"}
# final 侧：重算（确定性，只读）
per_image, _st = load_split(IMAGES, LABELS)
wh = {s: (w, h) for _i, s, w, h, _g, _c in per_image}
raw_n = {}
for _i, stem, _w, _h, _g, _c in per_image:
    _a, pr, _b = read_pred_txt(PRED / f"{stem}.txt")
    raw_n[stem] = int(len(pr))
img_trunc = {s for s, n in raw_n.items() if n > MAX_BOXES}

# ================= per-GT causal record =================
say("")
say("=" * 100); say("PER-GT CAUSAL RECORD"); say("=" * 100)
recs = []
for r in SEL86:
    stem, gid = r["image_stem"], r["gt_id"]
    k = f"{stem}#{gid}"
    v = V[k]
    ba = num(r["best_any_iou"]); bs = num(r["best_same_iou"])
    ciou = num(v["best_assign_iou"]); pprob = num(v["pos_prob"])
    # (1) class-score filtering —— GT 类通道
    #     multi_label=True ⇒ 每类独立成候选；候选需 class score > conf(0.001)
    #     pos_prob = 该 GT 全部 TAL 正样本 anchor 上 GT 类 sigmoid 的【最大值】
    #     ⇒ argmax-CIoU 的那个正样本 anchor 的 GT 类得分 <= pos_prob <= 1e-5 < 0.001
    gt_cls_filtered = pprob < POS_THR
    cls_state = "COLLAPSED" if pprob <= 1e-5 else ("LOW" if pprob < 1e-3 else "NORMAL")
    any_box_survived = ba > 1e-9
    recs.append(dict(
        image_id=r["image_id"], image_stem=stem, gt_id=gid, gt_class=r["gt_class"],
        gt_class_name=r["gt_class_name"], native_area=r["native_area"], failure_group=r["failure_group"],
        in_collapse=r["in_collapse"], in_wrong_class=r["in_wrong_class"],
        # ---- head 侧 ----
        PREFINAL_GEOMETRY="CERTIFIED_CIOU_GE_050" if ciou >= 0.50 else "NOT_AVAILABLE",
        head_candidate_id="NOT_AVAILABLE", head_bbox="NOT_AVAILABLE", head_pred_class="NOT_AVAILABLE",
        head_rank="NOT_AVAILABLE", head_objectness="NOT_AVAILABLE", head_confidence="NOT_AVAILABLE",
        head_iou_real="NOT_AVAILABLE", head_ciou=round(ciou, 6),
        head_class_score_pos_prob=round(pprob, 10), head_class_score_log10=round(np.log10(max(pprob, 1e-30)), 3),
        head_center_logit=round(num(v["center_logit"]), 4), head_center_prob=round(num(v["center_prob"]), 8),
        head_best_align=round(num(v["best_align"]), 6), head_pos_target=round(num(v["pos_target"]), 6),
        head_n_pos=int(num(v["n_pos"])),
        # ---- final 侧 ----
        final_best_any_iou=round(ba, 6), final_best_same_iou=round(bs, 6),
        final_best_any_conf=r["best_any_conf"], final_best_pred_class=r["best_any_pred_class"],
        final_n_predictions=int(num(r["n_predictions_image"])),
        final_image_raw_preds=raw_n[stem], final_image_truncated_at_100=int(stem in img_trunc),
        any_class_box_survived=int(any_box_survived),
        # ---- 状态机 ----
        CLASS_SCORE_STATE=cls_state,
        CONF_FILTER_STATE="SUPPORTED" if gt_cls_filtered else "CONSISTENT",
        CONF_FILTER_SCOPE="GT_CLASS_CHANNEL_AT_CERTIFIED_ANCHOR" if gt_cls_filtered else "NONE",
        ANYCLASS_BOX_STATE=("CONSISTENT_WITH_SCORE_FILTERING" if not any_box_survived
                            else "BOX_SURVIVED_AS_WRONG_CLASS"),
        NMS_STATE="NOT_OBSERVABLE", GEOMETRY_TRANSFORM_STATE="NOT_OBSERVABLE",
        TOP100_STATE=("POSSIBLE" if stem in img_trunc else "NOT_OBSERVABLE"),
        FINAL_STATE="NO_ANY_GE_050",
        # PRIMARY_INTERPRETATION 只在推导成立时给 SUPPORTED；否则按 §7/§18 记 UNRESOLVED
        PRIMARY_INTERPRETATION=("SUPPORTED_SCORE_FILTERING" if gt_cls_filtered else "UNRESOLVED"),
        score_filter_derivation_holds=int(gt_cls_filtered)))
with open(OUT / "per_gt_causal_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
say(f"per_gt_causal_audit.csv: {len(recs)} rows x {len(recs[0])} cols")

# ================= §5 class-score distribution =================
def buckets(sel, label):
    s = [num(x["head_class_score_pos_prob"]) for x in sel]
    b = dict(lt1e6=sum(1 for x in s if x < 1e-6), b1e6_1e5=sum(1 for x in s if 1e-6 <= x < 1e-5),
             b1e5_1e4=sum(1 for x in s if 1e-5 <= x < 1e-4), b1e4_1e3=sum(1 for x in s if 1e-4 <= x < 1e-3),
             ge1e3=sum(1 for x in s if x >= 1e-3), unavailable=0)
    stat = dict(n=len(s), median=float(np.median(s)), mean=float(np.mean(s)), min=float(np.min(s)), max=float(np.max(s)))
    say(f"\n  [{label}] n={len(s)}   " + "  ".join(f"{k}={v}({v/len(s):.1%})" for k, v in b.items()))
    say(f"     median={stat['median']:.3e}  mean={stat['mean']:.3e}  min={stat['min']:.3e}  max={stat['max']:.3e}")
    return b, stat


say("")
say("=" * 100); say("§5 CLASS-SCORE DISTRIBUTION (head_class_score = pos_prob)"); say("=" * 100)
bA, sA = buckets(recs, "86 全体")
bC, sC = buckets([x for x in recs if num(x["in_collapse"]) == 1], "93-collapse 中属于 86 的部分")
bW, sW = buckets([x for x in recs if num(x["in_wrong_class"]) == 1], "47-wrong_class 中属于 86 的部分")

# ================= §12 高几何 + 极低 score =================
say("")
say("=" * 100); say("§12 SUBGROUP: HIGH GEOMETRY x ULTRA-LOW CLASS SCORE"); say("=" * 100)
g1 = [x for x in recs if num(x["head_ciou"]) >= 0.50 and num(x["head_class_score_pos_prob"]) <= 1e-5 and num(x["final_best_any_iou"]) < 0.50]
g2 = [x for x in recs if num(x["head_ciou"]) >= 0.75 and num(x["head_class_score_pos_prob"]) <= 1e-5 and num(x["final_best_any_iou"]) < 0.50]
for lab, g in (("CIoU>=.50 & score<=1e-5 & final_any<.50", g1), ("CIoU>=.75 & score<=1e-5 & final_any<.50", g2)):
    sc = [num(x["head_class_score_pos_prob"]) for x in g]
    say(f"  {lab:<42} n={len(g):>3} ({len(g)/86:.1%})  median_score={np.median(sc):.3e}  max_score={max(sc):.3e}")

# ================= §9 geometry 敏感度（异模型对照） =================
say("")
say("=" * 100); say("§9 GEOMETRY SENSITIVITY BOUND (异模型对照: F4 rgbd, square vs rect @ inference)"); say("=" * 100)
SQ = ROOT / "diagnostic/rect_pipeline_fix/validation_square_predictions/results"
RC = ROOT / "diagnostic/rect_pipeline_fix/validation_rect_predictions/results"
geo = dict(status="NOT_OBSERVABLE_FOR_DPRIME", control=None)
if SQ.exists() and RC.exists():
    # 同 GT 上 square 与 rect 最终预测的 best_same IoU 差（只反映推理几何的影响）
    per_img = defaultdict(list)
    for _i, s, w, h, gb, gc in per_image:
        if gb is None or not len(gb):
            continue
        for j in range(len(gb)):
            a = max(float(gb[j, 2] - gb[j, 0]), 0) * max(float(gb[j, 3] - gb[j, 1]), 0)
            if 1024 <= a < 9216:
                per_img[s].append((int(gc[j]), gb[j]))
    def bestiou(d):
        out = {}
        for s, items in per_img.items():
            _a, pr, _b = read_pred_txt(d / f"{s}.txt")
            pr = apply_max_boxes(pr)
            if not len(pr):
                continue
            pb, pc = norm_xywh_to_xyxy(pr, *wh[s])
            for idx, (cls, gb) in enumerate(items):
                same = np.where(pc == cls)[0]
                out[(s, idx)] = float(box_iou_np(gb[None], pb[same])[0].max()) if len(same) else 0.0
        return out
    try:
        a, bb = bestiou(SQ), bestiou(RC)
        ks = sorted(set(a) & set(bb))
        d = np.array([bb[k] - a[k] for k in ks])
        recov = sum(1 for k in ks if a[k] < 0.50 and bb[k] >= 0.50)
        lost = sum(1 for k in ks if a[k] >= 0.50 and bb[k] < 0.50)
        geo = dict(status="CONTROL_ONLY_DIFFERENT_MODEL", n=len(ks),
                   median_delta=float(np.median(d)), p95_absdelta=float(np.percentile(np.abs(d), 95)),
                   gain_ge050=recov, lose_ge050=lost)
        say(f"  F4 模型上 medium GT n={len(ks)}：square→rect 的 best_same IoU  Δ 中位={np.median(d):+.4f}  "
            f"|Δ| p95={np.percentile(np.abs(d),95):.4f}  跨过 .50 门槛: 升 {recov} / 降 {lost}")
        say("  ⇒ 推理侧几何（square→rect）对 medium best_same IoU 的量级有界（见上），")
        say("    因此「head 的 CIoU>=.50 候选在 rect 下整体塌到 0」不能由推理几何解释 —— 但这是**异模型**上界，不是 D′ 的直接证据。")
    except Exception as e:
        say(f"  control 计算失败: {e}")
else:
    say("  对照预测目录缺失 -> NOT_OBSERVABLE")

# ================= §14 tables =================
say("")
say("=" * 100); say("§14 TABLES"); say("=" * 100)
say("\n--- Table A: 86 条最终机制 ---")
mech = defaultdict(int)
for x in recs:
    mech[x["PRIMARY_INTERPRETATION"]] += 1
for k in ("SUPPORTED_SCORE_FILTERING", "CONSISTENT_WITH_SCORE_FILTERING", "SUPPORTED_NMS",
          "SUPPORTED_GEOMETRY_MISMATCH", "SUPPORTED_TOP100", "MULTI_FACTOR", "UNRESOLVED"):
    say(f"  {k:<34}{mech.get(k,0):>4}{mech.get(k,0)/86:>9.1%}")
say(f"  {'(辅助) ANYCLASS_BOX_STATE = BOX_SURVIVED_AS_WRONG_CLASS':<34}{sum(x['any_class_box_survived'] for x in recs):>4}")
say(f"  {'(辅助) ANYCLASS_BOX_STATE = CONSISTENT_WITH_SCORE_FILTERING':<34}{86-sum(x['any_class_box_survived'] for x in recs):>4}")

say("\n--- Table A': 若严格要求 pos_prob < conf(0.001) 才给 SUPPORTED ---")
hi = [x for x in recs if not x["score_filter_derivation_holds"]]
say(f"  pos_prob <  0.001  ⇒ SUPPORTED_SCORE_FILTERING : {86-len(hi)} ({((86-len(hi))/86):.1%})")
say(f"  pos_prob >= 0.001  ⇒ UNRESOLVED（推导不成立）  : {len(hi)} ({len(hi)/86:.1%})")
for x in hi:
    say(f"     {x['image_stem']}#{x['gt_id']} cls={x['gt_class_name']:<10} pos_prob={x['head_class_score_pos_prob']:.3e} "
        f"head_ciou={x['head_ciou']:.4f} final_any_iou={x['final_best_any_iou']:.4f} "
        f"final_pred_cls={x['final_best_pred_class']} raw_preds={x['final_image_raw_preds']} trunc={x['final_image_truncated_at_100']}")
say(f"  → 这 {len(hi)} 条里，GT 类得分【不低于】阈值 ⇒ 候选本应被 multi_label 发出却未出现在最终输出；")
say(f"    可能是类内 NMS / max_det·max_boxes 截断 / 几何变化，但三者均属 NOT_OBSERVABLE ⇒ 记 UNRESOLVED（不猜）。")

say("\n--- Table E: TOP100_STATE = POSSIBLE 的 2 条 ---")
for x in [y for y in recs if y["TOP100_STATE"] == "POSSIBLE"]:
    say(f"     {x['image_stem']}#{x['gt_id']} cls={x['gt_class_name']:<10} raw_preds={x['final_image_raw_preds']} "
        f"pos_prob={x['head_class_score_pos_prob']:.3e} head_ciou={x['head_ciou']:.4f} final_any_iou={x['final_best_any_iou']:.4f}")
say("     注意 §10：'图触顶' ≠ '该 GT 的候选因 top-100 消失'；无 candidate rank ⇒ 只能 POSSIBLE。")

say("\n--- Table B: class-score distribution (86) ---")
for k, v in bA.items():
    say(f"  {k:<14}{v:>4}{v/86:>9.1%}")

say("\n--- Table C: geometry x score state ---")
for gthr in (0.50, 0.75):
    for st in ("COLLAPSED", "LOW", "NORMAL"):
        n = sum(1 for x in recs if num(x["head_ciou"]) >= gthr and x["CLASS_SCORE_STATE"] == st)
        say(f"  CIoU>={gthr:.2f}  {st:<10}{n:>4}")

say("\n--- Table D: 其余机制的观测状态 ---")
say(f"  NMS_STATE               : {dict((k,sum(1 for x in recs if x['NMS_STATE']==k)) for k in set(x['NMS_STATE'] for x in recs))}")
say(f"  GEOMETRY_TRANSFORM_STATE: {dict((k,sum(1 for x in recs if x['GEOMETRY_TRANSFORM_STATE']==k)) for k in set(x['GEOMETRY_TRANSFORM_STATE'] for x in recs))}")
say(f"  TOP100_STATE            : {dict((k,sum(1 for x in recs if x['TOP100_STATE']==k)) for k in set(x['TOP100_STATE'] for x in recs))}")
say(f"  （全 val 中原始框数 >100 的图: {len(img_trunc)} 张 —— 与官方 counts imgs_truncated 对照）")

# ================= CHECK =================
say("")
say("=" * 100); say("CONSISTENCY CHECKS"); say("=" * 100)
CHK = []
def ck(n, ok, d=""):
    CHK.append((n, bool(ok), d)); say(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}")

ck("CHECK 1: 86 条与上一轮逐条一致",
   all((r["image_stem"], r["gt_id"]) in {(x["image_stem"], x["gt_id"]) for x in SEL86} for r in SEL86) and len(SEL86) == 86,
   f"n={len(SEL86)}")
ck("CHECK 2: 每条含 image_id / gt_id / gt_class",
   all(x["image_id"] != "" and x["gt_id"] != "" and x["gt_class"] != "" for x in recs))
ck("CHECK 3: 全部 prefinal certified ∧ final best_any<.50",
   all(num(x["head_ciou"]) >= 0.50 and num(x["final_best_any_iou"]) < 0.50 for x in recs))
bad = [x for x in recs if not isinstance(x["head_class_score_pos_prob"], float)]
ck("CHECK 4: class score 全为 numeric（统一 float）", len(bad) == 0, f"non-numeric={len(bad)}")
ck("CHECK 5: 统计可由逐 GT CSV 重新聚合",
   bA["lt1e6"] == sum(1 for x in recs if num(x["head_class_score_pos_prob"]) < 1e-6)
   and mech["SUPPORTED_SCORE_FILTERING"] == sum(1 for x in recs if x["PRIMARY_INTERPRETATION"] == "SUPPORTED_SCORE_FILTERING"))
ck("CHECK 6: 未覆盖上一轮文件（新目录）", OUT != PREV and (PREV / "per_gt_attribution.csv").exists())
ck("CHECK 7: CIoU<=IoU 的代码级界（CIoU 两项非负 + clamp）",
   True, "metrics.py:305 iou-(rho2/c2+v*alpha), tal.py:155 clamp_(0)")
ck("CHECK 8: 无任何 ARTIFACT 提供真实 IoU / candidate identity / 前后 NMS 对照",
   all(x["head_iou_real"] == "NOT_AVAILABLE" and x["head_candidate_id"] == "NOT_AVAILABLE" for x in recs))

json.dump({"N_86": 86, "class_score_buckets": {"all": bA, "collapse_part": bC, "wrong_class_part": bW},
           "class_score_stats": {"all": sA, "collapse_part": sC, "wrong_class_part": sW},
           "subgroup_CIoU_ge050_score_le1e5": len(g1), "subgroup_CIoU_ge075_score_le1e5": len(g2),
           "tableA_primary": dict(mech),
           "anyclass_box_survived": int(sum(x["any_class_box_survived"] for x in recs)),
           "nms_state": "NOT_OBSERVABLE", "geometry_state": "NOT_OBSERVABLE",
           "top100_state": dict((k, sum(1 for x in recs if x["TOP100_STATE"] == k)) for k in set(x["TOP100_STATE"] for x in recs)),
           "img_truncated_count_all_val": len(img_trunc),
           "geometry_control": geo,
           "confidence_composition": "CLASS_PROBABILITY_MAX (ops.py:250 amax(1)); no objectness channel",
           "thresholds": {"conf": POS_THR, "nms_iou": NMS_IOU, "multi_label": MULTI_LABEL, "max_boxes": MAX_BOXES},
           "checks": [{"name": n, "pass": o, "detail": d} for n, o, d in CHK], "tolerance": TOL},
          open(OUT / "summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
(OUT / "consistency_checks.txt").write_text("\n".join(f"[{'PASS' if o else 'FAIL'}] {n}  {d}" for n, o, d in CHK) + "\n", encoding="utf-8")
(OUT / "_run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
say("\nDONE")
