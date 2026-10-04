"""P2 §14 — G86 / CONTROL / MISSED_NON86 paired score+rank analysis.

Pure read-side: applies a score transform in memory (no file writes, no eval),
then for every GT object in the frozen role audit reports the best same-class
prediction's raw score, calibrated score, per-image rank, top-100 membership and
CIoU.  Reads only; the frozen evaluator is not touched.

  python diagnostic/p2_score_calibration/p2_g86.py --method pow_2.0
  python diagnostic/p2_score_calibration/p2_g86.py --method raw
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/gyt/AIC/urban_multimodal_detection")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "diagnostic" / "p2_score_calibration"))
import official_eval as OE  # noqa: E402
from p2_calibration import make_transform, RAW, LBL, IMG, OUT  # noqa: E402

AUD = ROOT / "diagnostic" / "cls_representation_modality_audit" / "per_gt_representation_audit.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="raw")
    a = ap.parse_args()
    fn = make_transform(a.method)

    # ---- image size lookup (multi-extension safe) ----
    img_path = {p.stem: p for p in IMG.iterdir() if p.suffix.lower() in OE.IMG_EXTS}
    size_cache: dict[str, tuple[int, int]] = {}

    def imsize(stem):
        if stem not in size_cache:
            size_cache[stem] = OE._imsize(img_path[stem])
        return size_cache[stem]

    # ---- GT boxes (native xyxy) keyed by stem, in raw label-file order ----
    gt_boxes: dict[str, np.ndarray] = {}
    for p in sorted(LBL.glob("*.txt")):
        gt = OE.read_gt_txt(p)
        if len(gt) == 0:
            gt_boxes[p.stem] = np.zeros((0, 5), np.float32)
            continue
        w, h = imsize(p.stem)
        b, c = OE.norm_xywh_to_xyxy(gt, w, h)
        gt_boxes[p.stem] = np.concatenate([c[:, None], b], 1)

    # ---- raw predictions, transformed in memory ----
    def preds_of(stem):
        path = RAW / f"{stem}.txt"
        if not path.exists():
            return np.zeros((0, 6), np.float32)
        rows = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            t = line.split()
            if len(t) < 6:
                continue
            v = [float(x) for x in t[:6]]
            v[5] = float(fn(v[5], int(v[0])))
            rows.append(v)
        return np.array(rows, np.float32) if rows else np.zeros((0, 6), np.float32)

    audit = list(csv.DictReader(open(AUD, encoding="utf-8")))
    per_image_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def image_ctx(stem):
        if stem not in per_image_cache:
            arr = preds_of(stem)
            if len(arr):
                order = np.lexsort((arr[:, 0], -arr[:, 5]))   # same tie-break as frozen evaluator
                arr = arr[order]
            per_image_cache[stem] = arr
        return per_image_cache[stem]

    recs = []
    for r in audit:
        stem = r["image_stem"]
        gid = int(r["gt_id"])
        gcls = int(float(r["gt_class"]))
        gb = gt_boxes.get(stem)
        if gb is None or gid >= len(gb):
            continue
        box = gb[gid][1:]
        arr = image_ctx(stem)
        cls_i = arr[:, 0].astype(int) if len(arr) else np.zeros(0, int)
        sel = np.where(cls_i == gcls)[0]

        def best(idx):
            if not len(idx):
                return dict(ciou=0.0, conf=float("nan"), rank=-1, in_top=False, pred_cls=-1)
            w, h = imsize(stem)
            nb, _ = OE.norm_xywh_to_xyxy(arr[idx], w, h)
            ious = OE.box_iou_np(nb, box[None, :])[:, 0]
            j = int(np.argmax(ious))
            rk = int(idx[j])
            return dict(ciou=float(ious[j]), conf=float(arr[rk, 5]), rank=rk,
                        in_top=rk < OE.MAX_BOXES_PER_IMAGE, pred_cls=int(arr[rk, 0]))

        sc = best(sel)
        ac = best(np.arange(len(arr)))
        role = ("G86" if int(float(r["in_86"])) == 1
                else ("CONTROL" if int(float(r["detected"])) == 1 else "MISSED_NON86"))
        recs.append(dict(stem=stem, gt_id=gid, cls=gcls, role=role,
                         native_area=float(r["native_area"]),
                         gt_cls_sigmoid_stride8=float(r["cls_sigmoid_stride8"]),
                         sc_ciou=sc["ciou"], sc_conf=sc["conf"], sc_rank=sc["rank"],
                         sc_in_top100=sc["in_top"], sc_pred_cls=sc["pred_cls"],
                         any_ciou=ac["ciou"], any_conf=ac["conf"], any_rank=ac["rank"],
                         any_in_top100=ac["in_top"], any_pred_cls=ac["pred_cls"]))

    def med(v):
        v = [x for x in v if x is not None]
        return float(np.median(v)) if v else None

    def agg(role):
        s = [x for x in recs if x["role"] == role]
        sc_pos = [x for x in s if x["sc_ciou"] > 0]
        sc_strong = [x for x in s if x["sc_ciou"] >= 0.5]
        sc_low = [x for x in sc_strong if x["sc_conf"] < 1e-3]
        any_pos = [x for x in s if x["any_ciou"] > 0]
        any_strong = [x for x in s if x["any_ciou"] >= 0.5]
        return {
            "n": len(s),
            "sameclass": {
                "n_with_pred": len(sc_pos), "n_ciou_ge_0p5": len(sc_strong),
                "n_ciou_ge_0p5_and_predscore_lt_1e-3": len(sc_low),
                "median_ciou": med([x["sc_ciou"] for x in sc_pos]),
                "median_conf_at_ciou_ge_0p5": med([x["sc_conf"] for x in sc_strong]),
                "n_in_top100": sum(x["sc_in_top100"] for x in sc_pos)},
            "anyclass": {
                "n_with_overlap": len(any_pos), "n_ciou_ge_0p5": len(any_strong),
                "median_ciou": med([x["any_ciou"] for x in any_pos]),
                "median_conf_at_ciou_ge_0p5": med([x["any_conf"] for x in any_strong]),
                "n_in_top100_at_ciou_ge_0p5": sum(x["any_in_top100"] for x in any_strong)},
            "median_gt_class_sigmoid_stride8": med([x["gt_cls_sigmoid_stride8"] for x in s]),
            "n_gt_class_sigmoid_lt_1e-3": sum(x["gt_cls_sigmoid_stride8"] < 1e-3 for x in s),
        }

    rep = {"method": a.method,
           "roles": {r: agg(r) for r in ["G86", "CONTROL", "MISSED_NON86"]},
           "records": recs}
    print(json.dumps({k: v for k, v in rep.items() if k != "records"}, indent=1, ensure_ascii=False))
    out = OUT / f"g86_{a.method}.json"
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print("[saved]", out)


if __name__ == "__main__":
    main()
