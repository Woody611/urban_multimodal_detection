"""scripts/diagnostic_preprocess.py — RGBD 预处理一致性诊断（审计任务五）。

对照 predict.py 与 fork 训练/val 的读图+letterbox+通道顺序，打印运行时数值证据：
  - visible（BGR uint8）逐通道 min/max/mean/std
  - depth（uint16→uint8 归一化）min/max/mean/std、零值占比
  - merge 后 [B,G,R,D] 4ch 逐通道统计
  - letterbox（同 predict.py：auto=False/scaleup=True/center=True）后的形状、pad 量、缩放比
  - _to_chw 后 [R,G,B,D] CHW 逐通道统计（/255 前）
  - F4 模型首图前向 + NMS 的预测框数与前几框样例

只读诊断，不写模型/配置/结果文件。输出打印到 stdout（可重定向到 diagnostic/）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils.ops import non_max_suppression  # noqa: E402

import torch  # noqa: E402

WEIGHTS = "runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt"
VAL_VIS = PROJECT_ROOT / "data/processed/depth_split_train/images/val/visible"
IMGSZ = 1280


def _stats(name: str, arr: np.ndarray):
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        ch = [arr]
    else:
        ch = [arr[..., i] for i in range(arr.shape[2])]
    for i, c in enumerate(ch):
        z = float((c == 0).mean())
        print(f"  [{name}] ch{i} shape={c.shape} min={c.min():.2f} max={c.max():.2f} "
              f"mean={c.mean():.3f} std={c.std():.3f} zero_ratio={z:.3f}")


def _to_chw(im: np.ndarray) -> np.ndarray:
    """与 scripts/predict.py::_to_chw 完全一致。"""
    im = np.ascontiguousarray(im.transpose(2, 0, 1))
    c = im.shape[0]
    if c == 1:
        pass
    elif c == 3:
        im = im[::-1]
    elif c in (4, 5):
        im = np.concatenate([im[:3][::-1], im[3:]], axis=0)
    elif c == 6:
        im = np.concatenate([im[:3][::-1], im[3:][::-1]], axis=0)
    else:
        im = im[::-1]
    return np.ascontiguousarray(im)


def main():
    files = sorted(p for p in VAL_VIS.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    first = files[0]
    stem, suffix = first.stem, first.suffix
    vis_path = first
    dep_path = VAL_VIS.parent / "depth" / f"{stem}{suffix}"

    print("=" * 72)
    print(f"首图 {stem}")
    print("=" * 72)

    # 1. visible（fork 用 cv2.imread → BGR uint8）
    vis = cv2.imread(str(vis_path))
    print(f"[visible] file={vis_path}")
    print(f"  dtype={vis.dtype} shape(HWC)={vis.shape}")
    _stats("vis_BGR", vis)

    # 2. depth（fork：IMREAD_UNCHANGED，uint16 → <300→0, /19999*255, clip uint8）
    dep_raw = cv2.imread(str(dep_path), cv2.IMREAD_UNCHANGED)
    print(f"[depth] file={dep_path}")
    print(f"  raw dtype={dep_raw.dtype} shape={dep_raw.shape}")
    if dep_raw.ndim == 3:
        dep_raw = dep_raw[..., 0]
    if dep_raw.dtype != np.uint8:
        dep = dep_raw.astype(np.float32)
        dep[dep < 300] = 0.0
        dep = np.clip(dep / 19999.0 * 255.0, 0, 255).astype(np.uint8)
    else:
        dep = dep_raw
    _stats("depth_u8", dep)

    # 3. merge [B,G,R,D]（fork _merge_channels 语义）
    b, g, r = cv2.split(vis)
    merged = cv2.merge((b, g, r, dep))
    print(f"[merged] shape(HWC)={merged.shape} dtype={merged.dtype}")
    _stats("merged", merged)

    # 4. letterbox（同 predict.py）
    lb = LetterBox(new_shape=(IMGSZ, IMGSZ), auto=False, scaleFill=False,
                   scaleup=True, center=True, stride=32)
    lb_img = lb(image=merged)
    print(f"[letterbox] out_shape={lb_img.shape}")
    print(f"  缩放比 r=min(1280/H,1280/W)={min(IMGSZ / merged.shape[0], IMGSZ / merged.shape[1]):.4f}")
    new_h, new_w = lb_img.shape[:2]
    # pad 量（与 letterbox 中心填充一致）
    pad_h = new_h - int(round(merged.shape[0] * min(IMGSZ / merged.shape[0], IMGSZ / merged.shape[1])))
    pad_w = new_w - int(round(merged.shape[1] * min(IMGSZ / merged.shape[0], IMGSZ / merged.shape[1])))
    print(f"  pad(top/bottom, left/right)≈({pad_h // 2},{pad_h - pad_h // 2}),({pad_w // 2},{pad_w - pad_w // 2})")
    _stats("lb", lb_img)

    # 5. _to_chw → [R,G,B,D] CHW
    chw = _to_chw(lb_img)
    print(f"[to_chw] shape(CHW)={chw.shape}")
    for i in range(chw.shape[0]):
        c = chw[i].astype(np.float32)
        print(f"  [CHW] ch{i} min={c.min():.2f} max={c.max():.2f} mean={c.mean():.3f} std={c.std():.3f}")

    # 6. F4 模型前向 + NMS（首图）
    model = YOLO(WEIGHTS)
    nn = model.model.eval()
    x = torch.from_numpy(chw[None]).float() / 255.0
    with torch.no_grad():
        preds = nn(x)
    out = non_max_suppression(preds, 0.001, 0.7, nc=12, max_det=300, multi_label=True)
    det = out[0]
    print(f"[F4 forward] P3/P4/P5 shapes={[tuple(p.shape) for p in preds]}")
    print(f"[F4 NMS] 首图预测框数={0 if det is None else len(det)}")
    if det is not None and len(det):
        d = det.detach().cpu().numpy()
        print("  前 5 框 (xyxy, conf, cls):")
        for row in d[:5]:
            print(f"    xyxy=({row[0]:.0f},{row[1]:.0f},{row[2]:.0f},{row[3]:.0f}) "
                  f"conf={row[4]:.3f} cls={int(row[5])}")
    print("=" * 72)


if __name__ == "__main__":
    main()
