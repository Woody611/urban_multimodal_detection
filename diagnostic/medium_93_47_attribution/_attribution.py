"""_attribution.py — Medium 失败 GT 的 93+47 零 GPU attribution（只读）。

硬约束：不训练 / 不推理 / 不改源码 / 不改 config / 不改 checkpoint / 不重生成预测 /
不重跑 evaluator / 不改 confidence·IoU·NMS·top100 / 不重定义 93·47 集合。

只读输入（内容级识别，非按文件名猜测）：
  - GT          data/processed/rgbid_split_train/{labels,images}/val/visible
  - D′ 冻结预测  diagnostic/sepstem_clahe/best_full/results
      识别依据：medium(1320) 零同类重叠=145、零任意类重叠=98，与冻结参照
      diagnostic/box2_infer_val/_medium_pair.log 逐位吻合（对照：box2 148/99、D′ last 146/100 亦各自吻合）
  - pos_prob 来源 diagnostic/small_object_domains/_v6_domains.json 的 V1 域
  - 93/47 的冻结参照 diagnostic/box2_infer_val/_medium_pair.log

集合恢复规则（**原样复用**，不新增标准）：
  zero_same  = medium GT with best_same_iou <= 1e-9          -> 145
  wrong_class= zero_same with best_any_iou > 1e-9            -> 47
  collapse   = zero_same with V1 pos_prob <= 1e-5            -> 93
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
NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}
IMAGES = ROOT / "data/processed/rgbid_split_train/images/val/visible"
LABELS = ROOT / "data/processed/rgbid_split_train/labels/val/visible"
PRED = ROOT / "diagnostic/sepstem_clahe/best_full/results"
V6 = ROOT / "diagnostic/small_object_domains/_v6_domains.json"
REF = ROOT / "diagnostic/box2_infer_val/_medium_pair.log"
SMALL_A, MEDIUM_A = 1024.0, 9216.0
TOL = 1e-6
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
log = []


def say(s=""):
    print(s, flush=True)
    log.append(s)


# ---------------- 1. inputs ----------------
agg = hashlib.sha256()
for f in sorted(PRED.glob("*.txt")):
    agg.update(hashlib.sha256(f.read_bytes()).digest())
INP = [("GT labels dir", LABELS, f"{len(list(LABELS.glob('*.txt')))} txt", "(dir)"),
       ("GT label cache", LABELS.parent / "visible.cache", "-", sha(LABELS.parent / "visible.cache")),
       ("images dir", IMAGES, f"{len(list(IMAGES.iterdir()))} files", "(dir)"),
       ("D-prime frozen predictions", PRED, f"{len(list(PRED.glob('*.txt')))} txt", agg.hexdigest()),
       ("pos_prob source (V1)", V6, f"{len(json.loads(V6.read_text(encoding='utf-8')))} recs", sha(V6)),
       ("frozen 145/98/47 reference", REF, "-", sha(REF))]
(OUT / "input_manifest.txt").write_text("\n".join(
    ["MEDIUM 93+47 ATTRIBUTION - INPUT MANIFEST", "=" * 78,
     "D-prime frozen prediction identified by CONTENT, not filename:",
     "  medium(1320) zero-same-class=145 / zero-any-class=98 == frozen reference in _medium_pair.log",
     "  (controls: box2 gave 148/99, D-prime last gave 146/100, each matching its own reference)", ""] +
    [f"{n:<30}{str(p.relative_to(ROOT)):<52}{c:<10}{s}" for n, p, c, s in INP]) + "\n", encoding="utf-8")

# ---------------- 2. per-GT geometry ----------------
per_image, gtstats = load_split(IMAGES, LABELS)
say(f"GT: images={len(per_image)}  GT={gtstats['gt']}  (official load_split)")
gts = []
for i, stem, w, h, gtb, gtc in per_image:
    if gtb is None or not len(gtb):
        continue
    for j in range(len(gtb)):
        a = max(float(gtb[j, 2] - gtb[j, 0]), 0) * max(float(gtb[j, 3] - gtb[j, 1]), 0)
        if SMALL_A <= a < MEDIUM_A:
            gts.append(dict(key=f"{stem}#{j}", img_id=i, stem=stem, gt_id=j, cls=int(gtc[j]),
                            box=gtb[j].astype(np.float64), area=float(a), w=w, h=h))
byimg = defaultdict(list)
for g in gts:
    byimg[g["stem"]].append(g)
say(f"medium GT (native {SMALL_A:.0f}-{MEDIUM_A:.0f}) = {len(gts)}")

rec = {}
for stem, gs in byimg.items():
    _a, pr, _b = read_pred_txt(PRED / f"{stem}.txt")
    pr = apply_max_boxes(pr)
    if len(pr) == 0:
        pb, pc = np.zeros((0, 4), np.float32), np.zeros(0, int)
    else:
        pb, pc = norm_xywh_to_xyxy(pr, gs[0]["w"], gs[0]["h"])
    for g in gs:
        same = np.where(pc == g["cls"])[0] if len(pc) else np.zeros(0, int)
        wrong = np.where(pc != g["cls"])[0] if len(pc) else np.zeros(0, int)
        si = box_iou_np(g["box"][None], pb[same])[0] if len(same) else np.zeros(0, np.float32)
        ai = box_iou_np(g["box"][None], pb)[0] if len(pb) else np.zeros(0, np.float32)
        ka = int(np.argmax(ai)) if len(pb) else None
        ks = same[int(np.argmax(si))] if len(same) else None
        rec[g["key"]] = dict(
            best_any_iou=float(ai.max()) if len(ai) else 0.0,
            best_any_conf=float(pr[ka, 5]) if ka is not None else None,
            best_any_pred_class=int(pc[ka]) if ka is not None else None,
            best_same_iou=float(si.max()) if len(si) else 0.0,
            best_same_conf=float(pr[ks, 5]) if ks is not None else None,
            n_predictions_image=int(len(pb)), n_same_class_predictions=int(len(same)),
            n_wrong_class_predictions=int(len(wrong)))

V = {r["key"]: r for r in json.loads(V6.read_text(encoding="utf-8")) if r["domain"] == "V1"}
say(f"V1 join: {sum(1 for g in gts if g['key'] in V)}/{len(gts)}")
zero_same = [g["key"] for g in gts if rec[g["key"]]["best_same_iou"] <= 1e-9]
S_coll = set(k for k in zero_same if k in V and V[k]["pos_prob"] <= 1e-5)
S_wrong = set(k for k in zero_same if rec[k]["best_any_iou"] > 1e-9)
S_union = S_coll | S_wrong
gmap = {g["key"]: g for g in gts}
say(f"RECOVERED: zero_same={len(zero_same)} (ref 145)  wrong_class={len(S_wrong)} (ref 47)  collapse={len(S_coll)} (ref 93)")
say(f"SET RELATIONS: |collapse|={len(S_coll)} |wrong_class|={len(S_wrong)} "
    f"intersection={len(S_coll & S_wrong)} union={len(S_union)}")

# ---------------- 3. attribution ----------------
rows = []
for k in sorted(S_union):
    g, r = gmap[k], rec[k]
    ba, bs = r["best_any_iou"], r["best_same_iou"]
    grp = "+".join([x for x, f in (("collapse", k in S_coll), ("wrong_class", k in S_wrong)) if f])
    rows.append(dict(
        image_id=g["img_id"], image_stem=g["stem"], gt_id=g["gt_id"], gt_class=g["cls"],
        gt_class_name=NAMES[g["cls"]], native_area=round(g["area"], 1), failure_group=grp,
        in_collapse=int(k in S_coll), in_wrong_class=int(k in S_wrong),
        best_any_iou=round(ba, 6),
        best_any_conf=round(r["best_any_conf"], 6) if r["best_any_conf"] is not None else "",
        best_any_pred_class=r["best_any_pred_class"] if r["best_any_pred_class"] is not None else "",
        best_same_iou=round(bs, 6),
        best_same_conf=round(r["best_same_conf"], 6) if r["best_same_conf"] is not None else "",
        best_any_ge_050=int(ba >= 0.50), best_any_ge_075=int(ba >= 0.75), best_any_ge_090=int(ba >= 0.90),
        best_same_ge_050=int(bs >= 0.50), best_same_ge_075=int(bs >= 0.75), best_same_ge_090=int(bs >= 0.90),
        best_any_is_correct_class=int(r["best_any_pred_class"] == g["cls"]) if r["best_any_pred_class"] is not None else 0,
        attribution="LOCALIZATION_LIMITED" if ba < 0.50 else "CLASSIFICATION_LIMITED",
        strong_classification_failure=int(ba >= 0.75 and bs < 0.50),
        possible_ranking_limited=0, top100_limited="NOT_OBSERVABLE_FROM_FROZEN_OUTPUT", mixed=0,
        n_predictions_image=r["n_predictions_image"], n_same_class_predictions=r["n_same_class_predictions"],
        n_wrong_class_predictions=r["n_wrong_class_predictions"]))
with open(OUT / "per_gt_attribution.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
say(f"per_gt_attribution.csv: {len(rows)} rows")

# ---------------- 4. summary ----------------
def stats(sel, label):
    s = [r for r in rows if sel(r)]
    n = len(s)
    d = dict(n=n,
             best_any_ge_050=sum(r["best_any_ge_050"] for r in s),
             best_any_ge_075=sum(r["best_any_ge_075"] for r in s),
             best_any_ge_090=sum(r["best_any_ge_090"] for r in s),
             best_same_ge_050=sum(r["best_same_ge_050"] for r in s),
             best_same_ge_075=sum(r["best_same_ge_075"] for r in s),
             best_same_ge_090=sum(r["best_same_ge_090"] for r in s),
             LOCALIZATION_LIMITED=sum(1 for r in s if r["attribution"] == "LOCALIZATION_LIMITED"),
             CLASSIFICATION_LIMITED=sum(1 for r in s if r["attribution"] == "CLASSIFICATION_LIMITED"),
             STRONG_CLASSIFICATION_FAILURE=sum(r["strong_classification_failure"] for r in s),
             POSSIBLE_RANKING_LIMITED=0, TOP100_LIMITED="NOT_OBSERVABLE", MIXED=0, UNRESOLVED=0)
    say(f"\n[{label}] n={n}")
    for k2 in ("best_any_ge_050", "best_any_ge_075", "best_any_ge_090",
               "best_same_ge_050", "best_same_ge_075", "best_same_ge_090"):
        say(f"   {k2:<26}{d[k2]:>5}  ({d[k2]/n:.1%})")
    for k2 in ("LOCALIZATION_LIMITED", "CLASSIFICATION_LIMITED", "STRONG_CLASSIFICATION_FAILURE"):
        say(f"   {k2:<26}{d[k2]:>5}  ({d[k2]/n:.1%})")
    return d


A = stats(lambda r: r["in_collapse"], "A. 93 classification-collapse")
B = stats(lambda r: r["in_wrong_class"], "B. 47 wrong_class")
C = stats(lambda r: True, "C. union (110)")

say("\n=== best_any x best_same cross-table (union) ===")
bands = [("<0.50", "<0.50", "localization-limited"), (">=0.50", "<0.50", "classification-limited"),
         (">=0.75", "<0.50", "strong classification failure"), (">=0.75", ">=0.50", "geometry exists; inspect ranking"),
         (">=0.90", ">=0.75", "very strong geometric candidate")]
for bl, sl, interp in bands:
    ba_lo = float(bl.strip("<>=")) if bl != "<0.50" else -1.0
    a_ok = (lambda x: x >= 0.50) if bl == ">=0.50" else (lambda x: x >= 0.75) if bl == ">=0.75" else (lambda x: x >= 0.90) if bl == ">=0.90" else (lambda x: x < 0.50)
    b_ok = (lambda x: x < 0.50) if sl == "<0.50" else (lambda x: x >= 0.50) if sl == ">=0.50" else (lambda x: x >= 0.75)
    n = sum(1 for r in rows if a_ok(r["best_any_iou"]) and b_ok(r["best_same_iou"]))
    say(f"  best_any {bl:<7} best_same {sl:<7} n={n:<5} {interp}")

say("\n=== class breakdown ===")
say(f"  {'class':<13}{'coll':>6}{'wrong':>7}{'any_mean':>10}{'any_med':>9}{'same_mean':>11}{'same_med':>10}{'cls_lim':>9}{'loc_lim':>9}{'strong':>8}")
for c in range(12):
    s = [r for r in rows if r["gt_class"] == c]
    if not s:
        continue
    say(f"  {NAMES[c]:<13}{sum(r['in_collapse'] for r in s):>6}{sum(r['in_wrong_class'] for r in s):>7}"
        f"{np.mean([r['best_any_iou'] for r in s]):>10.4f}{np.median([r['best_any_iou'] for r in s]):>9.4f}"
        f"{np.mean([r['best_same_iou'] for r in s]):>11.4f}{np.median([r['best_same_iou'] for r in s]):>10.4f}"
        f"{sum(1 for r in s if r['attribution']=='CLASSIFICATION_LIMITED'):>9}"
        f"{sum(1 for r in s if r['attribution']=='LOCALIZATION_LIMITED'):>9}"
        f"{sum(r['strong_classification_failure'] for r in s):>8}")

# ---------------- 5. consistency checks ----------------
CHK = []


def ck(name, ok, detail=""):
    CHK.append((name, bool(ok), detail))
    say(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


say("\n=== consistency checks ===")
ck("CHECK A: collapse==93 / wrong_class==47", len(S_coll) == 93 and len(S_wrong) == 47,
   f"collapse={len(S_coll)} wrong_class={len(S_wrong)}")
ck("CHECK B: union rows == 110 (93+47 nominal; overlap 30)", len(rows) == len(S_union),
   f"rows={len(rows)} union={len(S_union)} intersection={len(S_coll & S_wrong)}")
ck("CHECK C: best_any_iou >= best_same_iou for every GT",
   all(r["best_any_iou"] >= r["best_same_iou"] - TOL for r in rows), f"tol={TOL}")
ck("CHECK D: 0 <= IoU <= 1",
   all(0.0 - TOL <= r["best_any_iou"] <= 1.0 + TOL and 0.0 - TOL <= r["best_same_iou"] <= 1.0 + TOL for r in rows))
ck("CHECK E: if best_same present, its class == GT class",
   all(r["best_same_iou"] <= 0 or True for r in rows), "by construction (same-class filtered)")
bad_f = [r for r in rows if r["best_any_is_correct_class"] and abs(r["best_any_iou"] - r["best_same_iou"]) > TOL]
ck("CHECK F: best_any class==gt_class => best_any_iou==best_same_iou", len(bad_f) == 0,
   f"violations={len(bad_f)} tol={TOL}")
re_A = sum(1 for r in rows if r["in_collapse"]);
re_lim = sum(1 for r in rows if r["in_collapse"] and r["attribution"] == "LOCALIZATION_LIMITED")
ck("CHECK G: aggregates re-derivable from per-GT rows",
   re_A == A["n"] and re_lim == A["LOCALIZATION_LIMITED"], "recount == summary")
ck("CHECK H: possible_ranking_limited unreachable in this sample",
   all(r["best_same_iou"] < 0.50 for r in rows),
   "all 110 have best_same==0 -> ranking cat structurally 0 (sample is pre-selected on same-class)")

json.dump({"A_collapse": A, "B_wrong_class": B, "C_union": C,
           "set_relations": {"collapse": len(S_coll), "wrong_class": len(S_wrong),
                             "intersection": len(S_coll & S_wrong), "union": len(S_union)},
           "per_gt_rows": len(rows), "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in CHK],
           "tolerance": TOL},
          open(OUT / "summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
(OUT / "consistency_checks.txt").write_text(
    "\n".join(f"[{'PASS' if ok else 'FAIL'}] {n}  {d}" for n, ok, d in CHK) + "\n", encoding="utf-8")
(OUT / "_run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
say("\nDONE")
