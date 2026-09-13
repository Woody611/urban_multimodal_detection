"""scripts/phase2_map.py — 从 predict.py 的 TXT 检测结果计算 mAP，与 model.val() 可比。

Phase 2 复现用：predict.py 只写提交 TXT、不计算 mAP。本脚本读取 predict.py 在
validation split 上生成的 results/*.txt，用 fork 自身的指标原语（DetMetrics +
box_iou + 与 DetectionValidator.match_predictions 完全一致的贪心匹配）计算
mAP50 / mAP75 / mAP50-95 / Precision / Recall，保证与训练内置 val 完全可比。

约定（与 val.py 的 update_metrics 逐一对应）:
  - 每图 label 读取 labels/val/visible/<stem>.txt（归一化 [0,1]，native 归一化）；
  - 每图 pred  读取 results/<stem>.txt（`class_id cx cy w h conf`，native 归一化）；
  - 两者都乘 native (H,W) 还原为 native 像素 xyxy；
  - box_iou(gt, pred) + match_predictions 得到 correct [N,10]；
  - 用 DetMetrics.process 累加，mean_results 输出 P/R/mAP50/mAP75/mAP50-95。

用法:
  python scripts/phase2_map.py --results submissions/predict_YYYYMMDD_HHMMSS/results \
      [--split_root data/processed/depth_split_train]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils.metrics import DetMetrics, box_iou  # noqa: E402

NC = 12
IOUV = np.linspace(0.5, 0.95, 10)  # 与 DetectionValidator.iouv 一致
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def match_predictions(pred_classes, true_classes, iou_np):
    """复刻 ultralytics/engine/validator.py:226 match_predictions（类感知贪心匹配）。

    Args:
        pred_classes (np.ndarray): [N] 预测类别。
        true_classes (np.ndarray): [M] 目标类别。
        iou_np (np.ndarray): [M, N] 预测与目标的成对 IoU。

    Returns:
        np.ndarray: [N, 10] 布尔正确矩阵（10 个 IoU 阈值）。
    """
    correct = np.zeros((pred_classes.shape[0], IOUV.shape[0]), dtype=bool)
    correct_class = true_classes[:, None] == pred_classes  # [M, N]
    iou = iou_np * correct_class  # 错误类别清零
    for i, threshold in enumerate(IOUV.tolist()):
        matches = np.nonzero(iou >= threshold)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def _read_txt(path, with_conf):
    """读取 YOLO 格式 txt。

    with_conf=True  → 每行 `cls cx cy w h conf`，返回 [n, 6]。
    with_conf=False → 每行 `cls cx cy w h`，返回 [n, 5]。
    """
    if not path.exists():
        n = 6 if with_conf else 5
        return np.zeros((0, n), dtype=np.float32)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < (6 if with_conf else 5):
            continue
        rows.append([float(x) for x in parts[: (6 if with_conf else 5)]])
    n = 6 if with_conf else 5
    if not rows:
        return np.zeros((0, n), dtype=np.float32)
    return np.array(rows, dtype=np.float32).reshape(-1, n)


def _norm_xywh_to_native_xyxy(arr, w, h):
    """(n,5)[cls,cx,cy,w,h]（native 归一化）→ native 像素 xyxy + cls。

    返回 (boxes [n,4], cls [n])。
    """
    if len(arr) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    cx, cy, bw, bh = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    x1 = (cx - bw / 2.0) * w
    y1 = (cy - bh / 2.0) * h
    x2 = (cx + bw / 2.0) * w
    y2 = (cy + bh / 2.0) * h
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32), arr[:, 0].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="从 predict.py 结果计算 mAP（与 model.val() 可比）")
    ap.add_argument("--results", required=True, help="predict.py 生成的 results 目录")
    ap.add_argument("--split_root", default="data/processed/depth_split_train",
                    help="val split 根目录（含 images/val/visible + labels/val/visible）")
    ap.add_argument("--min_conf", type=float, default=0.0,
                    help="额外按置信度过滤预测框（等价于更高的 NMS conf_thres，离线 conf 扫描用）")
    ap.add_argument("--names", action="store_true", help="额外打印每类 AP")
    args = ap.parse_args()

    split_root = (PROJECT_ROOT / args.split_root).resolve()
    images_dir = split_root / "images" / "val" / "visible"
    labels_dir = split_root / "labels" / "val" / "visible"
    results_dir = (PROJECT_ROOT / args.results).resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"val visible 目录不存在: {images_dir}")
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results 目录不存在: {results_dir}")

    tp_parts, conf_parts, pred_cls_parts, target_cls_parts = [], [], [], []
    seen, no_pred, n_corrupt = 0, 0, 0

    img_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    for img_path in img_files:
        stem = img_path.stem
        seen += 1
        h, w = cv2.imread(str(img_path)).shape[:2]  # 与 loader(cv2.imread) 同源，native HxW

        gt = _read_txt(labels_dir / f"{stem}.txt", with_conf=False)  # [n,5]
        # 复刻 verify_image_label 的越界检查：label 坐标 cx/cy/w/h 越界则整图判 corrupt，
        # 与 model.val() 一致地跳过该图（否则会多算 GT 实例导致 mAP 不可比）。
        if len(gt) and (gt[:, 1:].max() > 1.0 or gt[:, 1:].min() < 0.0):
            n_corrupt += 1
            continue
        gt_boxes, gt_cls = _norm_xywh_to_native_xyxy(gt, w, h)      # native xyxy + cls

        pred = _read_txt(results_dir / f"{stem}.txt", with_conf=True)  # [n,6]
        if args.min_conf > 0:
            pred = pred[pred[:, 5] >= args.min_conf]  # 等价于更高的 NMS conf_thres
        pred_cls = pred[:, 0].astype(np.float32)
        pred_conf = pred[:, 5].astype(np.float32)
        pred_boxes, _ = _norm_xywh_to_native_xyxy(pred[:, :5], w, h)  # 前5列 [cls,cx,cy,w,h]

        npr = len(pred_boxes)
        nl = len(gt_boxes)
        if npr == 0:
            no_pred += 1
            if nl:
                target_cls_parts.append(gt_cls)
            continue

        if nl:
            iou = box_iou(
                torch.from_numpy(gt_boxes),
                torch.from_numpy(pred_boxes),
            ).cpu().numpy()  # [M, N]
            correct = match_predictions(pred_cls, gt_cls, iou)  # [N, 10]
        else:
            correct = np.zeros((npr, 10), dtype=bool)

        tp_parts.append(correct)
        conf_parts.append(pred_conf)
        pred_cls_parts.append(pred_cls)
        target_cls_parts.append(gt_cls)

    tp = np.concatenate(tp_parts) if tp_parts else np.zeros((0, 10), dtype=bool)
    conf = np.concatenate(conf_parts) if conf_parts else np.zeros((0,), dtype=np.float32)
    pred_cls = np.concatenate(pred_cls_parts) if pred_cls_parts else np.zeros((0,), dtype=np.float32)
    target_cls = np.concatenate(target_cls_parts) if target_cls_parts else np.zeros((0,), dtype=np.float32)

    print("=" * 56)
    print("Phase 2 — predict.py 结果 mAP（与 model.val() 同指标）")
    print("=" * 56)
    print(f"val 图像      : {seen}")
    print(f"跳过 corrupt  : {n_corrupt}")
    print(f"有效图像      : {seen - n_corrupt}")
    print(f"无检测图像    : {no_pred}")
    print(f"GT 实例       : {len(target_cls)}")
    print(f"预测实例      : {len(pred_cls)}")

    if len(tp) and tp.any():
        names = {i: f"class_{i}" for i in range(NC)}
        det = DetMetrics(save_dir=PROJECT_ROOT, plot=False, names=names)
        det.process(tp=tp, conf=conf, pred_cls=pred_cls, target_cls=target_cls)
        P, R, mAP50, mAP75, mAP5095 = det.mean_results()
        print("-" * 56)
        print(f"Precision (B) : {P:.5f}")
        print(f"Recall    (B) : {R:.5f}")
        print(f"mAP50         : {mAP50:.5f}")
        print(f"mAP75         : {mAP75:.5f}")
        print(f"mAP50-95      : {mAP5095:.5f}")
        if args.names:
            print("-" * 56)
            print("每类 AP50-95:")
            apc = det.ap_class_index
            for ci, c in enumerate(apc):
                print(f"  class {int(c):2d}: {det.box.ap[ci]:.5f}")
        print("=" * 56)
    else:
        print("（无有效预测，mAP 无法计算）")


if __name__ == "__main__":
    main()
