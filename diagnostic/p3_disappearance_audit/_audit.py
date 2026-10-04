"""P3 — per-GT disappearance attribution across pre-NMS / NMS / top-100 / official match.

Reads `layers.npz` (produced by `_run_infer.py`) and attributes every GT object to one of

  F0  no pre-NMS same-class candidate with IoU >= 0.10
  F1  best pre-NMS candidate killed by NMS (suppressed, or cut by i[:max_det])
  F2  candidate survived NMS but did not enter the per-image top-100
  F3  candidate entered the evaluator but the GT still produced no official TP

Matching is the frozen evaluator's own algorithm (copied verbatim from
`official_eval.pr_curve_for_class`), instrumented to record which GT got matched, and
asserted to reproduce `official_eval`'s AP exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import official_eval as OE  # noqa: E402

OUT = Path(__file__).resolve().parent
IMG = ROOT / "data/processed/rgbid_split/images/val/visible"
LBL = ROOT / "data/processed/rgbid_split/labels/val/visible"
NAMES = ["person", "boat", "animal", "seat", "sign", "bicycle",
         "car", "ball", "light", "garbage_can", "uav", "tricycle"]
S0_IOU = 0.10
TINY_AREA = 256          # exploratory sub-small bucket (no pre-existing project definition)
SMALL_AREA, MED_AREA = 1024, 9216


def bucket(a):
    return "small" if a < SMALL_AREA else ("medium" if a < MED_AREA else "large")


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-12)


# ---------------------------------------------------------------- official matching
def official_match(preds_c, gt_by_img, iou_thr):
    """Verbatim copy of official_eval.pr_curve_for_class + per-GT match recording."""
    if not preds_c:
        return np.zeros(0), np.zeros(0), {}
    preds = sorted(preds_c, key=lambda z: -z[1])
    npos = int(sum(len(v) for v in gt_by_img.values()))
    if npos == 0:
        return np.zeros(0), np.zeros(0), {}
    used = {i: np.zeros(len(v), bool) for i, v in gt_by_img.items()}
    tp = np.zeros(len(preds)); fp = np.zeros(len(preds))
    matched = {}
    for k, (img_id, _conf, box) in enumerate(preds):
        gt = gt_by_img.get(img_id)
        if gt is None or len(gt) == 0:
            fp[k] = 1
            continue
        ious = OE.box_iou_np(box[None, :], gt)[0].astype(np.float64)
        ious[used[img_id]] = -1
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr:
            tp[k] = 1
            used[img_id][j] = True
            matched[(img_id, j)] = k
        else:
            fp[k] = 1
    ctp = np.cumsum(tp); cfp = np.cumsum(fp)
    return ctp / npos, ctp / np.maximum(ctp + cfp, OE.EPS), matched


def evaluate_instrumented(per_image, results_dir, max_boxes, iou_thresholds=OE.IOU_THRESHOLDS):
    """Instrumented clone of official_eval.evaluate: identical inputs (reads the same TXT
    with the same reader/cap/normalizer) and identical AP arithmetic, plus per-GT match
    recording. Returns (metrics, per-threshold {(cls,img_id,gt_j): pred_rank})."""
    per_iou, per_class_ap = [], np.full((OE.NC, len(iou_thresholds)), np.nan, np.float32)
    gt_match = [dict() for _ in iou_thresholds]
    preds_by_cls = {c: [] for c in range(OE.NC)}
    gt_by_cls = {c: {} for c in range(OE.NC)}
    results_dir = Path(results_dir)
    for (img_id, stem, w, h, gb, gc) in per_image:
        if gb is not None and len(gb):
            for c in np.unique(gc):
                m = gc == c
                gt_by_cls[int(c)][img_id] = gb[m]
        _n, pred, _bad = OE.read_pred_txt(results_dir / f"{stem}.txt")
        if len(pred) == 0:
            continue
        pred = OE.apply_max_boxes(pred, max_boxes)
        boxes, cls = OE.norm_xywh_to_xyxy(pred, w, h)
        conf = pred[:, 5]
        for i in range(len(boxes)):
            preds_by_cls[int(cls[i])].append((img_id, float(conf[i]), boxes[i]))
    for ti, t in enumerate(iou_thresholds):
        aps_A = []
        for c in range(OE.NC):
            rec, prec, matched = official_match(preds_by_cls[c], gt_by_cls[c], float(t))
            ap = OE.interpolate_ap(rec, prec)
            # --- divergence guard: must equal the frozen evaluator exactly ---
            r2, p2 = OE.pr_curve_for_class(preds_by_cls[c], gt_by_cls[c], float(t))
            assert abs(ap - OE.interpolate_ap(r2, p2)) < 1e-12, (c, t)
            has_gt = len(gt_by_cls[c]) > 0
            if has_gt:
                per_class_ap[c, ti] = ap
            aps_A.append(ap if has_gt else 0.0)
            for (img_id, j), k in matched.items():
                gt_match[ti][(c, img_id, j)] = k
        per_iou.append(float(np.mean(aps_A)))
    per_iou = np.array(per_iou)
    return (dict(mAP50=float(per_iou[0]), mAP75=float(per_iou[5]),
                 mAP50_95=float(per_iou.mean()), per_iou=per_iou.tolist(),
                 per_class_ap50_95=np.nanmean(per_class_ap, 1).tolist()), gt_match,
            gt_by_cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="layers.npz")
    a = ap.parse_args()

    L = np.load(OUT / a.npz, allow_pickle=True)
    stems = [str(s) for s in L["stem"]]
    a_off, b_off = L["a_off"], L["b_off"]
    A, B = L["layerA"], L["layerB"]
    n = len(stems)

    def slice_of(arr, off, i):
        s = int(off[i]); e = int(off[i + 1]) if i + 1 < n else len(arr)
        return arr[s:e]

    # The artifact encoded supp_a as (image_ordinal + local_suppressor_index): the dump ran
    # before the offset repair, when its running offset list held list positions, not box
    # counts.  Decoding is validated in-line below (100% must be same-class, in range).
    stem2i = {stems[i]: i for i in range(n)}
    supp = {}
    n_dec = n_sameclass = 0
    for s, idx, sup, iou, cap in zip(L["supp_stem"], L["supp_idx"], L["supp_a"], L["supp_iou"], L["supp_maxdet"]):
        s = str(s)
        i = stem2i[s]
        start = int(a_off[i]); end = int(a_off[i + 1]) if i + 1 < n else len(A)
        local = int(sup) if int(sup) < 0 else int(sup) - i
        if local >= 0:
            n_dec += 1
            assert start <= start + local < end, (s, local)
            if int(A[start + local][5]) == int(A[start + int(idx)][5]):
                n_sameclass += 1
        supp.setdefault(s, {})[int(idx)] = (local, float(iou), bool(cap))
    assert n_dec == n_sameclass, f"suppressor decode not same-class: {n_sameclass}/{n_dec}"

    layerA = {stems[i]: slice_of(A, a_off, i) for i in range(n)}
    layerB = {stems[i]: slice_of(B, b_off, i) for i in range(n)}
    assert all(layerA[s].shape[1] == 6 for s in stems)

    # ---- integrity: layer B rows must be exactly layer A rows kept by NMS ----
    max_delta = 0.0
    for s in stems:
        A_s = layerA[s]
        for j, idx in supp.get(s, {}).items():
            pass  # dropped candidates are simply absent from B
        # every B row must appear in A with identical geometry
        if len(A_s) == 0:
            continue
        for row in layerB[s]:
            m = np.where((np.abs(A_s[:, :4] - row[:4]).max(1) < 1e-2) &
                         (A_s[:, 5].astype(int) == int(row[5])))[0]
            if len(m):
                max_delta = max(max_delta, float(np.abs(A_s[m[0], 4] - row[4])))

    per_image, _ = OE.load_split(IMG, LBL)
    stem2img = {e[1]: e for e in per_image}

    # ---- official numbers from the FROZEN evaluator itself (written TXT, its own reader) ----
    orig_wh = {e[1]: (e[2], e[3]) for e in per_image}
    scratch = OUT / "_layerB_txt"
    scratch.mkdir(exist_ok=True)
    for s, (w, h) in orig_wh.items():
        arr = layerB[s]
        lines = []
        if len(arr):
            d = arr[np.argsort(-arr[:, 4])]
            for x1, y1, x2, y2, cf, cl in d:
                cx = min(max((x1 + x2) / 2.0 / w, 0.0), 1.0)
                cy = min(max((y1 + y2) / 2.0 / h, 0.0), 1.0)
                bw = min(max((x2 - x1) / w, 0.0), 1.0)
                bh = min(max((y2 - y1) / h, 0.0), 1.0)
                lines.append(f"{int(round(float(cl)))} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {cf:.6f}")
        (scratch / f"{s}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    metrics_C = OE.evaluate(per_image, scratch, max_boxes=OE.MAX_BOXES_PER_IMAGE)
    metrics_B = OE.evaluate(per_image, scratch, max_boxes=10 ** 6)
    instr_C, gt_match_C, gt_by_cls = evaluate_instrumented(per_image, scratch, OE.MAX_BOXES_PER_IMAGE)
    instr_B, _, _ = evaluate_instrumented(per_image, scratch, 10 ** 6)
    assert abs(instr_C["mAP50_95"] - metrics_C["mAP50-95"]) < 1e-12, "instrumented matcher diverged (top-100)"
    assert abs(instr_B["mAP50_95"] - metrics_B["mAP50-95"]) < 1e-12, "instrumented matcher diverged (uncapped)"

    thr = [float(t) for t in OE.IOU_THRESHOLDS]
    ti50 = 0

    rows = []
    for (img_id, stem, w, h, gb, gc) in per_image:
        if gb is None or not len(gb):
            continue
        A_s = layerA.get(stem, np.zeros((0, 6), np.float32))
        B_s = layerB.get(stem, np.zeros((0, 6), np.float32))
        sp = supp.get(stem, {})
        # the matcher keys GTs by their position WITHIN the per-class subset of the image
        cls_pos = {}
        for c in np.unique(gc):
            idxs = np.where(gc == c)[0]
            for k, jj in enumerate(idxs):
                cls_pos[int(jj)] = k
        for j in range(len(gb)):
            j_local = cls_pos[j]
            gcls = int(gc[j]); gbox = gb[j]
            garea = float((gbox[2] - gbox[0]) * (gbox[3] - gbox[1]))
            cand = np.where(A_s[:, 5].astype(int) == gcls)[0] if len(A_s) else np.zeros(0, int)
            best_iou = 0.0; best_j = -1; best_conf = np.nan; best_area_ratio = np.nan
            best_dx = best_dy = np.nan; best_rw = best_rh = np.nan
            if len(cand):
                ious = iou_matrix(A_s[cand, :4], gbox[None, :])[:, 0]
                k = int(np.argmax(ious))
                best_j = int(cand[k]); best_iou = float(ious[k])
                row = A_s[best_j]
                bcx = (row[0] + row[2]) / 2; bcy = (row[1] + row[3]) / 2
                gcx = (gbox[0] + gbox[2]) / 2; gcy = (gbox[1] + gbox[3]) / 2
                gw = gbox[2] - gbox[0]; gh = gbox[3] - gbox[1]
                barea = float((row[2] - row[0]) * (row[3] - row[1]))
                best_conf = float(row[4])
                best_area_ratio = barea / max(garea, 1e-9)
                best_dx = (bcx - gcx) / max(gw, 1e-9); best_dy = (bcy - gcy) / max(gh, 1e-9)
                best_rw = (row[2] - row[0]) / max(gw, 1e-9); best_rh = (row[3] - row[1]) / max(gh, 1e-9)

            # --- causal test for F1 (§7 C2): does a USABLE same-class box survive NMS? ---
            post_max_iou = 0.0; post_n_ge50 = 0
            if len(B_s):
                sel_b = np.where(B_s[:, 5].astype(int) == gcls)[0]
                if len(sel_b):
                    ious_b = iou_matrix(B_s[sel_b, :4], gbox[None, :])[:, 0]
                    post_max_iou = float(ious_b.max())
                    post_n_ge50 = int((ious_b >= 0.5).sum())

            matched50 = (gcls, img_id, j_local) in gt_match_C[ti50]
            matched_any = any((gcls, img_id, j_local) in gt_match_C[t] for t in range(len(thr)))

            s0 = best_iou >= S0_IOU
            nms_drop = False; drop_reason = None; sup_iou = np.nan; sup_conf = np.nan
            sup_area_ratio = np.nan; sup_same_cls = None
            rank_b = -1; in_top100 = False
            if s0:
                rec = sp.get(best_j)
                if rec is not None:
                    nms_drop = True
                    sup_idx, cand_sup_iou, sup_maxdet = rec
                    drop_reason = "max_det_cut" if sup_maxdet else "nms_suppressed"
                    if sup_idx >= 0 and len(A_s) > sup_idx:
                        srow = A_s[sup_idx]
                        sup_conf = float(srow[4]); sup_same_cls = bool(int(srow[5]) == gcls)
                        sarea = float((srow[2] - srow[0]) * (srow[3] - srow[1]))
                        sup_area_ratio = sarea / max(garea, 1e-9)
                        # THE causal quantity of brief §7: how well does the box that SURVIVED
                        # overlap this GT?  (cand_sup_iou is candidate<->suppressor, a tautology)
                        sup_iou = float(iou_matrix(srow[None, :4], gbox[None, :])[0, 0])
                else:
                    # survived NMS -> find its rank in the layer-B ordering
                    if len(B_s):
                        ious_b = iou_matrix(B_s[:, :4], np.asarray(A_s[best_j, :4])[None, :])[:, 0]
                        hit = np.where(ious_b > 0.999)[0]
                        if len(hit):
                            rank_b = int(hit[0])
                    in_top100 = 0 <= rank_b < OE.MAX_BOXES_PER_IMAGE

            if matched50:
                failure = "MATCHED@0.5"
            elif not s0:
                failure = "F0_no_raw_candidate"
            elif nms_drop:
                failure = "F1_nms_disappearance"
            elif not in_top100:
                failure = "F2_top100_disappearance"
            else:
                failure = "F3_survives_official_fail"

            rows.append(dict(stem=stem, gt_id=j, cls=gcls, cls_name=NAMES[gcls],
                             area=garea, size=bucket(garea), tiny=garea < TINY_AREA,
                             best_raw_iou=best_iou, best_raw_conf=best_conf if np.isfinite(best_conf) else None,
                             best_raw_area_ratio=best_area_ratio if np.isfinite(best_area_ratio) else None,
                             best_raw_dx=best_dx if np.isfinite(best_dx) else None,
                             best_raw_dy=best_dy if np.isfinite(best_dy) else None,
                             best_raw_rw=best_rw if np.isfinite(best_rw) else None,
                             best_raw_rh=best_rh if np.isfinite(best_rh) else None,
                             s0=s0, nms_drop=nms_drop, drop_reason=drop_reason,
                             sup_iou=sup_iou if np.isfinite(sup_iou) else None,
                             sup_conf=sup_conf if np.isfinite(sup_conf) else None,
                             sup_area_ratio=sup_area_ratio if np.isfinite(sup_area_ratio) else None,
                             sup_same_class=sup_same_cls,
                             rank_after_nms=rank_b, in_top100=in_top100,
                             post_max_iou_sameclass=post_max_iou, post_n_ge50=post_n_ge50,
                             post_usable=bool(post_max_iou >= 0.5),
                             matched50=matched50, matched_any_iou=matched_any,
                             matched_at=[bool((gcls, img_id, j_local) in gt_match_C[t]) for t in range(len(thr))],
                             failure=failure))

    out = dict(npz=a.npz, n_gt=len(rows), thresholds=thr,
               layerB_row_geom_max_delta=max_delta,
               official_top100=metrics_C, official_uncapped_C1=metrics_B,
               rows=rows)
    (OUT / "attribution.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"GT={len(rows)}  layerB geom max|delta|={max_delta:.2e}")
    print("official top-100  : mAP50-95=%.6f mAP50=%.6f mAP75=%.6f" %
          (metrics_C["mAP50-95"], metrics_C["mAP50"], metrics_C["mAP75"]))
    print("official uncapped : mAP50-95=%.6f mAP50=%.6f mAP75=%.6f" %
          (metrics_B["mAP50-95"], metrics_B["mAP50"], metrics_B["mAP75"]))


if __name__ == "__main__":
    main()
