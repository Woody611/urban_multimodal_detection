"""scripts/inference_gain_summary.py — 汇总各推理方案的 mAP / 逐类 AP / 框统计。

口径与 scripts/phase2_map.py 完全一致（DetMetrics + 与 model.val() 相同的贪心匹配），
即「与 model.val() 可比」的 fork 口径；同时用 official_map.py 的官方口径作为对照列。

用法:
  python scripts/inference_gain_summary.py --runs baseline=dir1 hflip=dir2 ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils.metrics import DetMetrics, box_iou  # noqa: E402
from phase2_map import match_predictions, _read_txt, _norm_xywh_to_native_xyxy  # noqa: E402
from official_map import collect, evaluate, curve_for_class, ap_from_curve, IOUV_OFFICIAL, NC as _ONC  # noqa: E402

NC = 12
NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def fork_metrics(results_dir: Path, split_root: Path):
    images_dir = split_root / "images" / "val" / "visible"
    labels_dir = split_root / "labels" / "val" / "visible"
    tp_parts, conf_parts, cls_parts, tgt_parts = [], [], [], []
    seen, no_pred, n_corrupt, n_gt = 0, 0, 0, 0
    n_boxes = 0
    for img_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        stem = img_path.stem
        seen += 1
        h, w = cv2.imread(str(img_path)).shape[:2]
        gt = _read_txt(labels_dir / f"{stem}.txt", with_conf=False)
        if len(gt) and (gt[:, 1:].max() > 1.0 or gt[:, 1:].min() < 0.0):
            n_corrupt += 1
            continue
        gt_boxes, gt_cls = _norm_xywh_to_native_xyxy(gt, w, h)
        n_gt += len(gt_boxes)
        pred = _read_txt(results_dir / f"{stem}.txt", with_conf=True)
        n_boxes += len(pred)
        pred_cls = pred[:, 0].astype(np.float32)
        pred_conf = pred[:, 5].astype(np.float32)
        pred_boxes, _ = _norm_xywh_to_native_xyxy(pred[:, :5], w, h)
        npr, nl = len(pred_boxes), len(gt_boxes)
        if npr == 0:
            no_pred += 1
            if nl:
                tgt_parts.append(gt_cls)
            continue
        if nl:
            iou = box_iou(torch.from_numpy(gt_boxes), torch.from_numpy(pred_boxes)).cpu().numpy()
            correct = match_predictions(pred_cls, gt_cls, iou)
        else:
            correct = np.zeros((npr, 10), dtype=bool)
        tp_parts.append(correct)
        conf_parts.append(pred_conf)
        cls_parts.append(pred_cls)
        tgt_parts.append(gt_cls)

    tp = np.concatenate(tp_parts) if tp_parts else np.zeros((0, 10), dtype=bool)
    conf = np.concatenate(conf_parts) if conf_parts else np.zeros((0,), np.float32)
    pred_cls = np.concatenate(cls_parts) if cls_parts else np.zeros((0,), np.float32)
    target_cls = np.concatenate(tgt_parts) if tgt_parts else np.zeros((0,), np.float32)

    det = DetMetrics(save_dir=PROJECT_ROOT, plot=False, names=NAMES)
    det.process(tp=tp, conf=conf, pred_cls=pred_cls, target_cls=target_cls)
    P, R, mAP50, mAP75, mAP5095 = det.mean_results()
    per_class = {}
    for ci, c in enumerate(det.ap_class_index):
        c = int(c)
        per_class[c] = float(det.box.ap[ci])
    return {
        "precision": float(P), "recall": float(R),
        "mAP50": float(mAP50), "mAP75": float(mAP75), "mAP50-95": float(mAP5095),
        "per_class_ap50_95": per_class,
        "n_images": seen, "n_corrupt": n_corrupt, "n_valid": seen - n_corrupt,
        "n_gt": n_gt, "n_pred": n_boxes, "n_empty": no_pred,
    }


def official_metrics(results_dir: Path, split_root: Path):
    """官方口径总体 + 逐类 AP50-95（101 点 mean + tail0 + conf 匹配）。"""
    images_dir = split_root / "images" / "val" / "visible"
    labels_dir = split_root / "labels" / "val" / "visible"
    gt, preds, st = collect(images_dir, labels_dir, results_dir, 0.0, drop_corrupt=True)
    r = evaluate(gt, preds, avg="mean", tail="zero", match="conf")

    per_class = {}
    for c in range(NC):
        if len(gt[c]) == 0:
            continue          # 无 GT 类别忽略（与官方一致）
        aps = []
        for t in IOUV_OFFICIAL:
            rec, prec = curve_for_class(preds[c], gt[c], float(t), match="conf")
            aps.append(ap_from_curve(rec, prec, avg="mean", tail="zero"))
        per_class[c] = float(np.mean(aps))

    return {"official_mAP50": r["mAP50"], "official_mAP75": r["mAP75"],
            "official_mAP50-95": r["mAP50-95"], "official_per_class_ap50_95": per_class}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=dir")
    ap.add_argument("--split_root", default="data/processed/rgbid_split")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    split_root = (PROJECT_ROOT / args.split_root).resolve()
    all_res = {}
    for spec in args.runs:
        name, d = spec.split("=", 1)
        d = (PROJECT_ROOT / d).resolve()
        if not (d / "results").is_dir():
            print(f"[skip] {name}: 无 results 目录 {d}")
            continue
        m = fork_metrics(d / "results", split_root)
        m.update(official_metrics(d / "results", split_root))
        all_res[name] = m

    base = all_res.get("RGBID_baseline")
    f4 = all_res.get("F4_baseline")

    hdr = (f"{'Model':<22}{'mAP50':>9}{'mAP50-95':>10}{'mAP75':>9}{'P':>9}{'R':>9}"
           f"{'ΔvsRGBID':>10}{'ΔvsF4':>9}{'boxes':>8}{'empty':>7}")
    print("=" * len(hdr))
    print("推理增益审计汇总（fork 口径 = 与 model.val() 可比）")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for name, m in all_res.items():
        d1 = m["mAP50-95"] - base["mAP50-95"] if base else 0.0
        d2 = m["mAP50-95"] - f4["mAP50-95"] if f4 else 0.0
        print(f"{name:<22}{m['mAP50']:>9.5f}{m['mAP50-95']:>10.5f}{m['mAP75']:>9.5f}"
              f"{m['precision']:>9.5f}{m['recall']:>9.5f}{d1:>+10.5f}{d2:>+9.5f}"
              f"{m['n_pred']:>8}{m['n_empty']:>7}")
    print("-" * len(hdr))
    print("\n官方口径对照（101 点 mean + tail0 + conf 匹配）:")
    print(f"{'Model':<22}{'mAP50':>9}{'mAP50-95':>10}")
    for name, m in all_res.items():
        print(f"{name:<22}{m['official_mAP50']:>9.5f}{m['official_mAP50-95']:>10.5f}")

    print("\n逐类 AP50-95（fork 口径）:")
    classes = sorted(NAMES)
    head = f"{'Model':<22}" + "".join(f"{NAMES[c][:9]:>11}" for c in classes)
    print(head)
    for name, m in all_res.items():
        row = f"{name:<22}" + "".join(
            f"{m['per_class_ap50_95'].get(c, float('nan')):>11.4f}" for c in classes)
        print(row)

    print("\n逐类 AP50-95（官方口径）:")
    print(head)
    for name, m in all_res.items():
        row = f"{name:<22}" + "".join(
            f"{m['official_per_class_ap50_95'].get(c, float('nan')):>11.4f}" for c in classes)
        print(row)

    print("\n相对 RGBID baseline 的逐类 ΔAP50-95（官方口径，升/降类数统计）:")
    for name, m in all_res.items():
        if name == "RGBID_baseline" or base is None:
            continue
        deltas = []
        for c in classes:
            b = base["official_per_class_ap50_95"].get(c)
            v = m["official_per_class_ap50_95"].get(c)
            if b is None or v is None:
                continue
            deltas.append(v - b)
        up = sum(1 for d in deltas if d > 0)
        dn = sum(1 for d in deltas if d < 0)
        print(f"  {name:<22} 升 {up}/{len(deltas)}  降 {dn}/{len(deltas)}  "
              f"均值Δ {np.mean(deltas):+.5f}  中位Δ {np.median(deltas):+.5f}")

    if args.out_json:
        outp = (PROJECT_ROOT / args.out_json).resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(all_res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[json] -> {outp}")


if __name__ == "__main__":
    main()
