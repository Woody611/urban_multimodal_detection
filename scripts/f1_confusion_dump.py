"""scripts/f1_confusion_dump.py — F1 验证集混淆矩阵 dump（含 0~11 类 + background）。

手动推理 + fork ConfusionMatrix，产出 reports/f1_confusion_matrix.json。
评估口径：conf=0.001 / iou=0.7 / max_det=300 / imgsz=1024，与训练内置 val 一致。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.data.loaders import LoadImagesAndVideos  # noqa: E402
from ultralytics.utils.ops import non_max_suppression, scale_boxes  # noqa: E402
from ultralytics.utils.metrics import ConfusionMatrix  # noqa: E402

WEIGHTS = "runs/urban_multimodal_det_e7_lr0half/weights/best.pt"
SRC = ROOT / "data" / "processed" / "depth_split_train" / "images" / "val" / "visible"
LBL = ROOT / "data" / "processed" / "depth_split_train" / "labels" / "val" / "visible"
IMGSZ, CONF, IOU, MAX_DET = 1024, 0.001, 0.7, 300
USE_SIMOTM, CHANNELS, PAIRS = "RGBD", 4, ["visible", "depth"]
OUT = ROOT / "reports" / "f1_confusion_matrix.json"

model = YOLO(WEIGHTS)
nn = model.model.to("cpu").eval()
nc = int(getattr(nn, "nc", 12))
stride = int(nn.stride.max())
lb = LetterBox(new_shape=(IMGSZ, IMGSZ), auto=False, scaleFill=False, scaleup=True, center=True, stride=stride)

cm = ConfusionMatrix(nc=nc, conf=CONF, iou_thres=0.45, task="detect")

def load_gt(stem, h, w):
    f = LBL / f"{stem}.txt"
    if not f.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    box, cls = [], []
    for line in f.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c, cx, cy, bw, bh = int(float(p[0])), *map(float, p[1:5])
        x1 = (cx - bw/2)*w; y1 = (cy - bh/2)*h; x2 = (cx + bw/2)*w; y2 = (cy + bh/2)*h
        box.append([x1, y1, x2, y2]); cls.append(c)
    return np.array(box, dtype=np.float32), np.array(cls, dtype=np.float32)

loader = LoadImagesAndVideos(str(SRC), batch=8, use_simotm=USE_SIMOTM, imgsz=IMGSZ,
                             pairs_rgb_ir=PAIRS, pairs_rgb_depth=PAIRS)
n_det = n_gt = 0
for paths, imgs, _ in loader:
    orig_shapes = [im.shape[:2] for im in imgs]
    arrs = []
    for im in imgs:
        imc = np.ascontiguousarray(lb(image=im).transpose(2, 0, 1))
        arrs.append(np.concatenate([imc[:3][::-1], imc[3:]], axis=0))
    x = torch.from_numpy(np.stack(arrs)).float() / 255.0
    with torch.no_grad():
        preds = nn(x)
    out = non_max_suppression(preds, CONF, IOU, nc=nc, max_det=MAX_DET, multi_label=True)
    for j, p in enumerate(paths):
        stem = Path(p).stem
        oh, ow = orig_shapes[j]
        gt_box, gt_cls = load_gt(stem, oh, ow)
        det = out[j]
        # fork process_batch(detections[N,6], gt_bboxes[M,4], gt_cls[M]) 期望 torch tensor
        gt_box_t = torch.from_numpy(gt_box).float()
        gt_cls_t = torch.from_numpy(gt_cls).float()
        if det is not None and len(det):
            d = det.clone(); d[:, :4] = scale_boxes(x.shape[2:], d[:, :4], orig_shapes[j])
            det_t = d  # 已是 torch [N,6] xyxy+conf+cls
        else:
            det_t = None
        cm.process_batch(det_t, gt_box_t, gt_cls_t)
        n_det += 0 if det_t is None else len(det_t)
        n_gt += len(gt_cls_t)

mat = cm.matrix  # [nc+1, nc+1]，fork 索引为 matrix[pred, gt]：最后一行=背景(pred=bg→FN)，最后一列=背景(gt=bg→FP)
OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump({"matrix": mat.astype(int).tolist(), "nc": nc, "n_det": n_det, "n_gt": n_gt,
           "index_semantics": "matrix[pred, gt]; row/col nc = background; col nc = FP(pred class, gt bg); row nc = FN(gt class, pred bg)"},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[cm] 已写出 {OUT}  (total det={n_det}, gt={n_gt})")
print("[cm] 混淆矩阵(行=Pred, 列=GT, 最后一行=背景FN, 最后一列=背景FP):")
print("        " + " ".join(f"{i:>5}" for i in range(nc)) + "     bg")
for i in range(nc + 1):
    r = ("cls" + str(i)) if i < nc else "bg"
    print(f"  {r:>5} " + " ".join(f"{mat[i,j]:>5}" for j in range(nc)) + f"   {mat[i,nc]:>5}")
