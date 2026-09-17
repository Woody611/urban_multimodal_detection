"""scripts/diagnostic_val_dump.py — 决定性实验：用 model.val() 的 save_txt 复算 mAP。

目的（定位 −0.019 系统性差距的归属）：
  - 已知：predict.py 与 model.val() 的输入张量逐字节一致、forward head0 一致。
  - 问题：干净的 scaleup=False predict（0.51341）仍比训练内置 val（0.53273）低 −0.019。
  - 本脚本让 model.val() 用 save_txt=True/save_conf=True 把「验证器自己的预测」
    按 predict.py 完全相同的 `<cls> cx cy w h conf`（native 归一化）写到磁盘，
    再用 phase2_map.py 的同一套逻辑（label TXT 直读 GT + DetMetrics + match_predictions）
    复算 mAP：
      * 若 ≈ 0.53273 → phase2_map 的 GT/pred 口径与 val 一致，−0.019 来自 predict.py 输出本身；
      * 若 ≈ 0.51341 → phase2_map 的 GT（label TXT 直读）与 val 的 GT（round-trip）有差异。

只读诊断：不训练、不改模型/配置；输出写到 diagnostic/val_dump/。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.metrics import DetMetrics  # noqa: E402

from phase2_map import (  # noqa: E402
    _read_txt,
    _norm_xywh_to_native_xyxy,
    match_predictions,
    IMG_EXTS,
)

WEIGHTS = "runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt"
SPLIT_ROOT = PROJECT_ROOT / "data/processed/depth_split_train"
OUT = PROJECT_ROOT / "diagnostic/val_dump"


def run_phase2_logic(labels_dir: Path, results_dir: Path):
    """复刻 phase2_map.py::main 的度量逻辑，返回 mAP50-95。"""
    images_dir = SPLIT_ROOT / "images" / "val" / "visible"
    tp_parts, conf_parts, pred_cls_parts, target_cls_parts = [], [], [], []
    seen, no_pred, n_corrupt = 0, 0, 0

    for img_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        stem = img_path.stem
        seen += 1
        import cv2
        h, w = cv2.imread(str(img_path)).shape[:2]

        gt = _read_txt(labels_dir / f"{stem}.txt", with_conf=False)
        if len(gt) and (gt[:, 1:].max() > 1.0 or gt[:, 1:].min() < 0.0):
            n_corrupt += 1
            continue
        gt_boxes, gt_cls = _norm_xywh_to_native_xyxy(gt, w, h)

        pred = _read_txt(results_dir / f"{stem}.txt", with_conf=True)
        pred_cls = pred[:, 0].astype(np.float32)
        pred_conf = pred[:, 5].astype(np.float32)
        pred_boxes, _ = _norm_xywh_to_native_xyxy(pred[:, :5], w, h)

        npr = len(pred_boxes)
        nl = len(gt_boxes)
        if npr == 0:
            no_pred += 1
            if nl:
                target_cls_parts.append(gt_cls)
            continue

        if nl:
            from ultralytics.utils.metrics import box_iou
            iou = box_iou(torch.from_numpy(gt_boxes), torch.from_numpy(pred_boxes)).cpu().numpy()
            correct = match_predictions(pred_cls, gt_cls, iou)
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

    print(f"  [phase2 逻辑] 图像={seen} corrupt={n_corrupt} 无检测={no_pred} "
          f"GT={len(target_cls)} pred={len(pred_cls)}")
    if len(tp) and tp.any():
        names = {i: f"class_{i}" for i in range(12)}
        det = DetMetrics(save_dir=PROJECT_ROOT, plot=False, names=names)
        det.process(tp=tp, conf=conf, pred_cls=pred_cls, target_cls=target_cls)
        P, R, mAP50, mAP75, mAP5095 = det.mean_results()
        return mAP5095
    return float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(WEIGHTS)

    # 1. 让验证器自己 dump 预测（与 predict.py 相同格式）
    print("运行 model.val(save_txt=True, save_conf=True)...")
    metrics = model.val(
        data=str(SPLIT_ROOT / "dataset.yaml"),
        split="val",
        save_txt=True,
        save_conf=True,
        save_hybrid=False,
        imgsz=1280,
        conf=0.001,
        iou=0.7,
        max_det=300,
        batch=4,
        device="cpu",
        plots=False,
        save_json=False,
        use_simotm="RGBD",
        channels=4,
        pairs_rgb_ir=["visible", "depth"],
        pairs_rgb_depth=["visible", "depth"],
        project=str(OUT),
        name="val_preds",
        exist_ok=True,
    )
    print(f"[model.val] mAP50-95 = {metrics.box.map:.5f}")

    # 2. 找到 save_txt 的 labels 目录
    labels_dir = OUT / "val_preds" / "labels"
    if not labels_dir.is_dir():
        # fallback：ultralytics 可能写到 runs/detect/val_preds/labels
        labels_dir = PROJECT_ROOT / "runs" / "detect" / "val_preds" / "labels"
    print(f"[save_txt] labels_dir = {labels_dir} (存在={labels_dir.is_dir()})")

    # 3. 用 phase2_map 的同一套逻辑复算 val 自己 dump 的预测
    gt_dir = SPLIT_ROOT / "labels" / "val" / "visible"
    m = run_phase2_logic(gt_dir, labels_dir)
    print(f"[phase2 逻辑 @ val 自身预测] mAP50-95 = {m:.5f}")

    # 4. 对照：用同一逻辑复算 干净 scaleup=False predict 的输出
    fixed_dir = PROJECT_ROOT / "diagnostic/scaleup_false_fixed/results"
    if fixed_dir.is_dir():
        m2 = run_phase2_logic(gt_dir, fixed_dir)
        print(f"[phase2 逻辑 @ clean scaleup=False predict] mAP50-95 = {m2:.5f}")


if __name__ == "__main__":
    main()
