"""scripts/diagnostic_predict_scaleup_false_fixed.py — 干净的 scaleup=False 推理（修正 ratio_pad）。

目的（审计任务四/五的收尾）：
  验证 predict.py 的坐标反变换 bug（scale_boxes 未传 ratio_pad，导致 scaleup=False
  时小图 gain 被错误重算为 scaleup=True 的 2.0）。本脚本用 scaleup=False + 显式把
  letterbox 的真实 ratio_pad 传给 scale_boxes，输出应与训练内置 val（0.53273）吻合，
  从而证明 0.46465（diagnostic_predict_scaleup_false.py）是坐标 bug，而非预处理差异。

只读诊断：不改模型/配置/训练；输出写到 diagnostic/scaleup_false_fixed/results/。
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
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.data.loaders import LoadImagesAndVideos  # noqa: E402
from ultralytics.utils.ops import non_max_suppression, scale_boxes  # noqa: E402

WEIGHTS = "runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt"
VAL_VIS = PROJECT_ROOT / "data/processed/depth_split_train/images/val/visible"
OUT_DIR = PROJECT_ROOT / "diagnostic/scaleup_false_fixed/results"
IMGSZ = 1280
CONF, IOU, MAX_DET, MAX_BOXES = 0.001, 0.7, 300, 100


def _to_chw(im: np.ndarray) -> np.ndarray:
    """与 scripts/predict.py::_to_chw 一致。"""
    im = np.ascontiguousarray(im.transpose(2, 0, 1))
    c = im.shape[0]
    if c in (4, 5):
        im = np.concatenate([im[:3][::-1], im[3:]], axis=0)
    elif c == 3:
        im = im[::-1]
    return np.ascontiguousarray(im)


def _letterbox_ratio_pad(im: np.ndarray, imgsz: int, scaleup: bool):
    """复刻 LetterBox 的缩放比与 (left, top) pad，供 scale_boxes 正确反变换。"""
    h, w = im.shape[:2]
    r = min(imgsz / h, imgsz / w)
    if not scaleup:
        r = min(r, 1.0)
    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))
    dw = (imgsz - new_unpad_w) / 2.0
    dh = (imgsz - new_unpad_h) / 2.0
    left = round(dw - 0.1)
    top = round(dh - 0.1)
    return (r, r), (left, top)


def _format_lines(dets, orig_h: int, orig_w: int, max_boxes: int):
    lines = []
    if dets is None or len(dets) == 0:
        return lines
    d = dets.detach().cpu().numpy()
    d = d[np.argsort(-d[:, 4])][:max_boxes]
    for x1, y1, x2, y2, conf, cls in d:
        cid = int(round(float(cls)))
        if cid < 0 or cid > 11:
            continue
        cx = (x1 + x2) / 2.0 / orig_w
        cy = (y1 + y2) / 2.0 / orig_h
        w = (x2 - x1) / orig_w
        h = (y2 - y1) / orig_h
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}")
    return lines


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(WEIGHTS)
    nn = model.model.to("cpu").eval()
    nc = int(getattr(nn, "nc", 12))
    letterbox = LetterBox(new_shape=(IMGSZ, IMGSZ), auto=False, scaleup=False,
                          center=True, stride=32)
    loader = LoadImagesAndVideos(
        str(VAL_VIS), batch=16, use_simotm="RGBD", imgsz=IMGSZ,
        pairs_rgb_ir=["visible", "depth"], pairs_rgb_depth=["visible", "depth"],
    )

    for paths, imgs, _info in loader:
        orig = [im.shape[:2] for im in imgs]
        ratio_pads = [_letterbox_ratio_pad(im, IMGSZ, scaleup=False) for im in imgs]
        arrs = [_to_chw(letterbox(image=im)) for im in imgs]
        x = torch.from_numpy(np.stack(arrs)).float() / 255.0
        with torch.no_grad():
            preds = nn(x)
        out = non_max_suppression(preds, CONF, IOU, nc=nc, max_det=MAX_DET, multi_label=True)
        for i, p in enumerate(paths):
            if out[i] is not None and len(out[i]):
                out[i][:, :4] = scale_boxes(x.shape[2:], out[i][:, :4], orig[i],
                                            ratio_pad=ratio_pads[i])
            lines = _format_lines(out[i], orig[i][0], orig[i][1], MAX_BOXES)
            (OUT_DIR / (Path(p).stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    n = len(list(OUT_DIR.glob("*.txt")))
    print(f"[clean scaleup=False] 完成 {n} 张 → {OUT_DIR}")


if __name__ == "__main__":
    main()
