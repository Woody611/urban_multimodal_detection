"""scripts/conf_sensitivity_scan.py — 离线 conf 敏感性诊断（只读分析，不改任何 pipeline）。

目的：判断 ensemble / TTA 在 fork 口径下的损失是否由「低置信 FP 长尾」造成。
做法：对各方案已落盘的 results/*.txt 按 min_conf 过滤后再算 mAP（等价于提高 NMS 的
conf_thres），观察 fork / 官方两种口径随 min_conf 的变化。

注意：这是**诊断**，不是提交配置。冻结 pipeline 的 conf=0.001 未被修改。

用法:
  python scripts/conf_sensitivity_scan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference_gain_summary import fork_metrics, official_metrics  # noqa: E402
import cv2  # noqa: E402
from phase2_map import match_predictions, _read_txt, _norm_xywh_to_native_xyxy  # noqa: E402
from ultralytics.utils.metrics import DetMetrics, box_iou  # noqa: E402

G = PROJECT_ROOT / "diagnostic" / "rgbid_inference_gain"
SPLIT = PROJECT_ROOT / "data" / "processed" / "rgbid_split"
NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def fork_metrics_at(results_dir: Path, split_root: Path, min_conf: float):
    """fork 口径，带 min_conf 过滤。"""
    images_dir = split_root / "images" / "val" / "visible"
    labels_dir = split_root / "labels" / "val" / "visible"
    tp_parts, conf_parts, cls_parts, tgt_parts = [], [], [], []
    n_boxes = 0
    for img_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        stem = img_path.stem
        h, w = cv2.imread(str(img_path)).shape[:2]
        gt = _read_txt(labels_dir / f"{stem}.txt", with_conf=False)
        if len(gt) and (gt[:, 1:].max() > 1.0 or gt[:, 1:].min() < 0.0):
            continue
        gt_boxes, gt_cls = _norm_xywh_to_native_xyxy(gt, w, h)
        pred = _read_txt(results_dir / f"{stem}.txt", with_conf=True)
        if min_conf > 0 and len(pred):
            pred = pred[pred[:, 5] >= min_conf]
        n_boxes += len(pred)
        pred_cls = pred[:, 0].astype(np.float32)
        pred_conf = pred[:, 5].astype(np.float32)
        pred_boxes, _ = _norm_xywh_to_native_xyxy(pred[:, :5], w, h)
        if len(pred_boxes) == 0:
            if len(gt_boxes):
                tgt_parts.append(gt_cls)
            continue
        if len(gt_boxes):
            iou = box_iou(torch.from_numpy(gt_boxes),
                          torch.from_numpy(pred_boxes)).cpu().numpy()
            correct = match_predictions(pred_cls, gt_cls, iou)
        else:
            correct = np.zeros((len(pred_boxes), 10), dtype=bool)
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
    return float(mAP5095), float(P), float(R), n_boxes


import torch  # noqa: E402


def official_at(results_dir: Path, split_root: Path, min_conf: float):
    """官方口径，带 min_conf 过滤（复用 official_map.collect 的 min_conf 参数）。"""
    from official_map import collect, evaluate
    images_dir = split_root / "images" / "val" / "visible"
    labels_dir = split_root / "labels" / "val" / "visible"
    gt, preds, st = collect(images_dir, labels_dir, results_dir, min_conf, drop_corrupt=True)
    r = evaluate(gt, preds, avg="mean", tail="zero", match="conf")
    return r["mAP50-95"], r["mAP50"]


def main():
    variants = ["RGBID_baseline", "RGBID_tta_hflip", "RGBID_tta_multiscale",
                "RGBID_f4_ens_1to1", "RGBID_f4_ens_2to1", "RGBID_f4_ens_3to1"]
    confs = [0.0, 0.005, 0.02, 0.05]

    print("=" * 96)
    print("离线 conf 敏感性诊断（只读分析；冻结 pipeline 的 conf=0.001 未改动）")
    print("=" * 96)
    print(f"{'Model':<22}{'min_conf':>9}{'fork mAP50-95':>15}{'fork P':>9}{'fork R':>9}"
          f"{'boxes':>8}{'official50-95':>14}{'official50':>12}")
    print("-" * 96)
    results = {}
    for v in variants:
        d = G / v / "results"
        if not d.is_dir():
            continue
        for mc in confs:
            f95, fp, fr, nb = fork_metrics_at(d, SPLIT, mc)
            o95, o50 = official_at(d, SPLIT, mc)
            results[(v, mc)] = (f95, o95)
            print(f"{v:<22}{mc:>9.3f}{f95:>15.5f}{fp:>9.5f}{fr:>9.5f}{nb:>8}"
                  f"{o95:>14.5f}{o50:>12.5f}", flush=True)
        print(flush=True)

    print("=" * 96)
    print("各 conf 下相对 RGBID_baseline(同 conf) 的 ΔmAP50-95")
    print("=" * 96)
    print(f"{'Model':<22}" + "".join(f"{('c=%g' % mc):>16}" for mc in confs))
    for v in variants:
        if v == "RGBID_baseline":
            continue
        row = f"{v:<22}"
        for mc in confs:
            if (v, mc) in results and ("RGBID_baseline", mc) in results:
                df = results[(v, mc)][0] - results[("RGBID_baseline", mc)][0]
                do = results[(v, mc)][1] - results[("RGBID_baseline", mc)][1]
                row += f"{df:>+8.5f}/{do:>+7.5f}"
            else:
                row += f"{'-':>16}"
        print(row)
    print("(格式: fork Δ / official Δ)")


if __name__ == "__main__":
    main()
