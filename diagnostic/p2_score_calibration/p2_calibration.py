"""P2 — Offline prediction score calibration (CPU-only, prediction-only).

Strict contract:
  * only the 6th column (confidence) of each prediction TXT may change
  * first 5 columns (class + normalized xywh) are copied token-for-token
  * frozen official evaluator (scripts/official_eval.py) is imported, never modified
  * no training / no inference / no GPU

Usage:
  python diagnostic/p2_score_calibration/p2_calibration.py --stage global
  python diagnostic/p2_score_calibration/p2_calibration.py --stage classwise
  python diagnostic/p2_score_calibration/p2_calibration.py --stage holdout --method cls_c6_g1.5
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/gyt/AIC/urban_multimodal_detection")
sys.path.insert(0, str(ROOT / "scripts"))
import official_eval as OE  # noqa: E402  frozen evaluator, single source of truth

RAW = ROOT / "diagnostic" / "sepstem_clahe" / "best_full" / "results"
IMG = ROOT / "data" / "processed" / "rgbid_split" / "images" / "val" / "visible"
LBL = ROOT / "data" / "processed" / "rgbid_split" / "labels" / "val" / "visible"
OUT = ROOT / "diagnostic" / "p2_score_calibration"
SCRATCH = OUT / "preds_scratch"
SEED = 20261003
NC = 12
CLASS_NAMES = ["person", "boat", "animal", "seat", "sign", "bicycle",
               "car", "ball", "light", "garbage_can", "uav", "tricycle"]

EPS = 1e-12


# ---------------------------------------------------------------- raw loading
def read_raw(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        t = s.split()
        if len(t) < 6:
            continue
        rows.append(t)
    return rows


RAWROWS: dict[str, list[list[str]]] = {}


def load_all_raw():
    global RAWROWS
    for p in sorted(RAW.glob("*.txt")):
        RAWROWS[p.stem] = read_raw(p)
    return RAWROWS


# ---------------------------------------------------------------- transforms
def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def logit(s):
    s = min(max(s, EPS), 1.0 - EPS)
    return math.log(s / (1.0 - s))


def make_transform(name: str):
    """Return fn(conf, cls) -> new conf. Raises for unknown names."""
    if name == "raw":
        return lambda s, c: s
    if name.startswith("pow_"):
        g = float(name[4:])
        return lambda s, c: s ** g
    if name.startswith("temp_"):
        T = float(name[5:])
        return lambda s, c: sigmoid(logit(s) / T)
    if name.startswith("cls_c"):
        # cls_c<c>_g<g>  -> exponent g applied only to class c
        body = name[5:]
        c0, g = body.split("_g")
        c0, g = int(c0), float(g)
        return lambda s, c: (s ** g) if c == c0 else s
    raise ValueError(name)


# ---------------------------------------------------------------- scratch I/O
def write_preds(fn, stems):
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        rows = RAWROWS.get(stem)
        if rows is None:
            continue
        lines = [" ".join(t[:5] + [repr(float(fn(float(t[5]), int(float(t[0])))))]) for t in rows]
        (SCRATCH / f"{stem}.txt").write_text(("\n".join(lines) + "\n") if lines else "",
                                             encoding="utf-8")


def integrity_check(stems):
    """bbox / class / count must be identical to raw. Only col-6 may differ."""
    bad = []
    for stem in stems:
        rows = RAWROWS.get(stem, [])
        npath = SCRATCH / f"{stem}.txt"
        new = read_raw(npath) if npath.exists() else []
        if len(new) != len(rows):
            bad.append((stem, "count", len(rows), len(new)))
            continue
        for a, b in zip(rows, new):
            if a[:5] != b[:5]:
                bad.append((stem, "geom", a[:5], b[:5]))
                break
    return bad


# ---------------------------------------------------------------- split
def build_split():
    all_stems = sorted(p.stem for p in IMG.iterdir()
                       if p.suffix.lower() in OE.IMG_EXTS)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(all_stems))
    n_cal = int(round(len(all_stems) * 0.8))
    cal = sorted(all_stems[i] for i in perm[:n_cal])
    hold = sorted(all_stems[i] for i in perm[n_cal:])
    return {"seed": SEED, "n_total": len(all_stems),
            "calibration": cal, "holdout": hold}


def load_split_json():
    return json.loads((OUT / "calibration_split.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- eval helpers
def eval_subset(per_image, stems_set, results_dir):
    sub = [e for e in per_image if e[1] in stems_set]
    return OE.evaluate(sub, results_dir)


def top100_sets(stems):
    """per-image top-100 stem->set of (cls,cx,cy,w,h) keyed by conf rank."""
    out = {}
    for stem in stems:
        rows = read_raw(SCRATCH / f"{stem}.txt")
        if not rows:
            out[stem] = []
            continue
        arr = np.array([[float(x) for x in r[:6]] for r in rows], np.float32)
        keep = OE.apply_max_boxes(arr, 100)
        out[stem] = [tuple(r[:5]) for r in keep]
    return out


def raw_top100_sets(stems):
    out = {}
    for stem in stems:
        rows = RAWROWS.get(stem, [])
        if not rows:
            out[stem] = []
            continue
        arr = np.array([[float(x) for x in r[:6]] for r in rows], np.float32)
        keep = OE.apply_max_boxes(arr, 100)
        out[stem] = [tuple(r[:5]) for r in keep]
    return out


def summarize(r):
    return {"mAP50-95": r["mAP50-95"], "mAP50": r["mAP50"], "mAP75": r["mAP75"],
            "per_iou": r["per_iou"], "per_class": r["per_class_ap50_95"],
            "counts": r["counts"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["split", "global", "classwise", "holdout", "full", "top100", "audit"])
    ap.add_argument("--method", default=None)
    a = ap.parse_args()

    load_all_raw()
    per_image, _ = OE.load_split(IMG, LBL)

    OUT.mkdir(parents=True, exist_ok=True)

    if a.stage == "split":
        sp = build_split()
        (OUT / "calibration_split.json").write_text(json.dumps(sp, indent=1), encoding="utf-8")
        print("total=%d calib=%d holdout=%d" % (sp["n_total"], len(sp["calibration"]), len(sp["holdout"])))
        return

    all_stems = sorted(RAWROWS.keys())

    if a.stage == "audit":
        pass  # handled below without needing the split
    else:
        sp = load_split_json()
        cal_set, hold_set = set(sp["calibration"]), set(sp["holdout"])
        full_set = set(sp["calibration"]) | set(sp["holdout"])

    if a.stage == "global":
        rows_out = []
        for name in ["raw", "pow_0.5", "pow_0.75", "pow_1.0", "pow_1.25", "pow_1.5", "pow_2.0",
                     "temp_0.5", "temp_0.75", "temp_1.0", "temp_1.25", "temp_1.5", "temp_2.0"]:
            t0 = time.time()
            write_preds(make_transform(name), all_stems)
            bad = integrity_check(all_stems)
            r = eval_subset(per_image, cal_set, SCRATCH)
            s = summarize(r)
            rows_out.append({"method": name, "split": "calibration", "integrity_bad": len(bad), **s})
            print("%-10s mAP50-95=%.6f mAP50=%.4f mAP75=%.4f  integrity_bad=%d  %.1fs"
                  % (name, s["mAP50-95"], s["mAP50"], s["mAP75"], len(bad), time.time() - t0))
        base = rows_out[0]["mAP50-95"]
        for r in rows_out:
            r["d_mAP50-95"] = r["mAP50-95"] - base
        (OUT / "global_results.json").write_text(json.dumps(rows_out, indent=1), encoding="utf-8")
        print("baseline(calib raw) mAP50-95=%.6f" % base)
        return

    if a.stage == "classwise":
        rows_out = []
        gammas = [0.75, 1.25, 1.5]
        # raw reference on calibration split
        write_preds(make_transform("raw"), all_stems)
        r0 = eval_subset(per_image, cal_set, SCRATCH)
        base = r0["mAP50-95"]
        rows_out.append({"method": "raw", "split": "calibration", "cls": None, "gamma": None,
                         **summarize(r0)})
        print("raw calib=%.6f" % base)
        for c0 in range(NC):
            for g in gammas:
                name = f"cls_c{c0}_g{g}"
                write_preds(make_transform(name), all_stems)
                bad = integrity_check(all_stems)
                r = eval_subset(per_image, cal_set, SCRATCH)
                s = summarize(r)
                rows_out.append({"method": name, "split": "calibration", "cls": c0, "gamma": g,
                                 "integrity_bad": len(bad), **s})
                print("%-14s %-11s g=%.2f  mAP50-95=%.6f  Δ=%+.6f"
                      % (name, CLASS_NAMES[c0], g, s["mAP50-95"], s["mAP50-95"] - base))
        for r in rows_out:
            r["d_mAP50-95"] = r["mAP50-95"] - base
        (OUT / "classwise_results.json").write_text(json.dumps(rows_out, indent=1), encoding="utf-8")
        return

    if a.stage in ("holdout", "full"):
        stems_set = hold_set if a.stage == "holdout" else full_set
        res = {}
        for name in ["raw", a.method]:
            write_preds(make_transform(name), all_stems)
            bad = integrity_check(all_stems)
            r = eval_subset(per_image, stems_set, SCRATCH)
            res[name] = {"bad": len(bad), **summarize(r)}
            print("%-16s %-8s mAP50-95=%.6f mAP50=%.4f mAP75=%.4f" %
                  (name, a.stage, r["mAP50-95"], r["mAP50"], r["mAP75"]))
        res["delta_mAP50-95"] = res[a.method]["mAP50-95"] - res["raw"]["mAP50-95"]
        print("Δ(%s) = %+.6f" % (a.method, res["delta_mAP50-95"]))
        (OUT / f"{a.stage}_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        with open(OUT / f"{a.stage}_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method", "split", "mAP50-95", "mAP50", "mAP75", "per_iou"])
            for name in ["raw", a.method]:
                w.writerow([name, a.stage, res[name]["mAP50-95"], res[name]["mAP50"],
                            res[name]["mAP75"], res[name]["per_iou"]])
        return

    if a.stage == "top100":
        names = ["raw"] + (a.method.split(",") if a.method else [])
        for name in names:
            write_preds(make_transform(name), all_stems)
            new = top100_sets(all_stems)
            old = raw_top100_sets(all_stems)
            changed_imgs, enter, leave = 0, 0, 0
            detail = []
            for stem in all_stems:
                so, sn = set(old[stem]), set(new[stem])
                if so != sn:
                    changed_imgs += 1
                    enter += len(sn - so)
                    leave += len(so - sn)
                    detail.append({"stem": stem, "entered": len(sn - so), "left": len(so - sn),
                                   "n_raw": len(old[stem]), "n_new": len(new[stem])})
            print("%-14s changed_images=%d entered=%d left=%d" % (name, changed_imgs, enter, leave))
            for d in sorted(detail, key=lambda z: -(z["entered"] + z["left"]))[:10]:
                print("   ", d)
        return

    if a.stage == "audit":
        # raw integrity + score distribution + boxes/image, full set
        n_img = len(all_stems)
        n_box = sum(len(v) for v in RAWROWS.values())
        confs = np.array([float(t[5]) for v in RAWROWS.values() for t in v])
        cls_all = np.array([int(float(t[0])) for v in RAWROWS.values() for t in v])
        bx = np.array([len(v) for v in RAWROWS.values()])
        import hashlib
        h = hashlib.sha256()
        for stem in all_stems:
            h.update(stem.encode())
            for t in RAWROWS[stem]:
                h.update((" ".join(t)).encode())
        rep = {
            "images": n_img, "raw_boxes": int(n_box),
            "conf_min": float(confs.min()), "conf_max": float(confs.max()),
            "conf_median": float(np.median(confs)),
            "conf_p95": float(np.percentile(confs, 95)),
            "conf_p99": float(np.percentile(confs, 99)),
            "boxes_per_image_min": int(bx.min()), "boxes_per_image_max": int(bx.max()),
            "boxes_per_image_mean": float(bx.mean()), "images_gt100": int((bx > 100).sum()),
            "class_hist": {CLASS_NAMES[c]: int((cls_all == c).sum()) for c in range(NC)},
            "has_full_class_scores": False,
            "pred_bundle_sha256": h.hexdigest(),
            "eval_images_used": len(per_image),
        }
        (OUT / "prediction_audit.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps(rep, indent=1, ensure_ascii=False))
        return


if __name__ == "__main__":
    main()
