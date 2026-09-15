"""scripts/f4_val_per_class.py — F4 best.pt 逐类指标 dump（与 F1 逐类 CSV 对齐口径）。

复刻训练内置 val：conf=0.001 / iou=0.7 / max_det=300 / imgsz=1280 / rect=False。
输出 reports/f4_class_metrics.csv（逐类 AP50/AP75/AP50-95/P/R）。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402

WEIGHTS = "runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt"
DATA = "data/processed/depth_split_train/dataset.yaml"
OUT = ROOT / "reports" / "f4_class_metrics.csv"

model = YOLO(WEIGHTS)
names = dict(model.names) if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}

metrics = model.val(
    data=DATA, imgsz=1280, conf=0.001, iou=0.7, max_det=300, batch=4,
    device="cpu", plots=False, save_json=False,
    use_simotm="RGBD", channels=4,
    pairs_rgb_ir=["visible", "depth"], pairs_rgb_depth=["visible", "depth"],
    project=str(ROOT / "runs" / "audit"), name="f4_val", exist_ok=True,
)

box = metrics.box
# fork Metric: ap50/ap75/ap 是 @property 返回逐类 np.ndarray；p/r 为逐类 np.ndarray。
# 注意：不是 box.maps[i]（maps 是标量属性），逐类取 box.ap50[i] / box.ap75[i] / box.ap[i]。
rows = []
for i in sorted(names):
    ap50 = float(box.ap50[i])
    ap75 = float(box.ap75[i])
    ap = float(box.ap[i])
    rows.append({
        "class_id": i, "class": names[i],
        "AP50": round(ap50, 4), "AP75": round(ap75, 4), "AP50_95": round(ap, 4),
        "precision": round(float(box.p[i]), 4), "recall": round(float(box.r[i]), 4),
    })

with OUT.open("w", encoding="utf-8") as f:
    f.write("class_id,class,AP50,AP75,AP50_95,precision,recall\n")
    for r in rows:
        f.write(f"{r['class_id']},{r['class']},{r['AP50']},{r['AP75']},{r['AP50_95']},{r['precision']},{r['recall']}\n")

print(f"[val] 总体: mAP50={box.map50:.5f}  mAP75={box.map75:.5f}  mAP50-95={box.map:.5f}  P={box.mp:.5f}  R={box.mr:.5f}")
print(f"[val] 已写出 {OUT}")
for r in rows:
    print(f"  {r['class_id']:>2} {r['class']:<12} AP50={r['AP50']:.4f} AP75={r['AP75']:.4f} "
          f"AP50-95={r['AP50_95']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}")
