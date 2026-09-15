"""scripts/f1_val_metrics_dump.py — F1 验证集逐类指标 + 混淆矩阵 dump（供误差分析）。

复刻训练内置 val 的评估口径：conf=0.001 / iou=0.7 / max_det=300 / imgsz=1024 / rect=False。
输出 reports/f1_val_metrics.json（逐类 AP50/AP75/AP50-95/P/R + 混淆矩阵 + 总体指标）。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402

WEIGHTS = "runs/urban_multimodal_det_e7_lr0half/weights/best.pt"
DATA = "data/processed/depth_split_train/dataset.yaml"
OUT = ROOT / "reports" / "f1_val_metrics.json"

model = YOLO(WEIGHTS)
names = dict(model.names) if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}

metrics = model.val(
    data=DATA,
    imgsz=1024,
    conf=0.001,
    iou=0.7,
    max_det=300,
    batch=8,
    device="cpu",
    plots=False,
    save_json=False,
    use_simotm="RGBD",
    channels=4,
    pairs_rgb_ir=["visible", "depth"],
    pairs_rgb_depth=["visible", "depth"],
    project=str(ROOT / "runs" / "audit"),
    name="f1_val",
    exist_ok=True,
)

box = metrics.box
# 混淆矩阵：ConfusionMatrix.matrix 为 [nc+1, nc+1]（最后一行/列是 background/FP/FN）
cm = getattr(box.confusion_matrix, "matrix", None)
cm = np.asarray(cm).tolist() if cm is not None else None

def tolist(x):
    x = np.asarray(x)
    return x.tolist()

out = {
    "overall": {
        "map50": float(box.map50),
        "map75": float(box.map75),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    },
    "per_class": {
        "names": names,
        "ap50": tolist(box.ap50),          # 每类 AP50
        "ap": tolist(box.ap),              # 每类 AP50-95
        "maps": tolist(box.maps),          # 每类 [AP50, AP75, AP50-95]
        "p": tolist(box.p),                # 每类 precision
        "r": tolist(box.r),                # 每类 recall
    },
    "confusion_matrix": cm,
    "speed": {k: float(v) for k, v in (box.speed or {}).items()},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[val] 已写出 {OUT}")
print(f"[val] 总体: mAP50={box.map50:.5f}  mAP75={box.map75:.5f}  mAP50-95={box.map:.5f}  P={box.mp:.5f}  R={box.mr:.5f}")
print(f"[val] 逐类 AP50 / AP75 / AP50-95:")
for i, n in sorted(names.items()):
    ap50, ap75, ap = box.maps[i]
    print(f"   {i:>2} {n:<12} {ap50:.4f} / {ap75:.4f} / {ap:.4f}")
