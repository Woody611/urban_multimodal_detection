"""scripts/diagnostic_compare_input.py — 逐图对比 predict.py 与 model.val() 的模型输入张量。

定位 −0.019 系统性差距：对同一张图，手动复刻两条 pipeline 的读图+merge+letterbox+
通道重排，直接比 float 张量（/255 后）。predict 侧用 _to_chw；val 侧用 fork 的
Format._format_img（真实类）。若张量一致 → 差距在 forward/NMS/度量；不一致 → 在预处理。

只读诊断。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox, Format  # noqa: E402

WEIGHTS = "runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt"
IMGSZ = 1280
VIS = PROJECT_ROOT / "data/processed/depth_split_train/images/val/visible"
DEP = PROJECT_ROOT / "data/processed/depth_split_train/images/val/depth"


def _to_chw(im):
    im = np.ascontiguousarray(im.transpose(2, 0, 1))
    c = im.shape[0]
    if c in (4, 5):
        im = np.concatenate([im[:3][::-1], im[3:]], axis=0)
    elif c == 3:
        im = im[::-1]
    return np.ascontiguousarray(im)


def _read_merged(vp: Path):
    """与 loaders.py/base.py RGBD 分支完全一致的读图+merge。"""
    vis = cv2.imread(str(vp))  # BGR uint8
    dep = cv2.imread(str(DEP / vp.name), cv2.IMREAD_UNCHANGED)
    if dep.ndim == 3:
        dep = dep[..., 0]
    if dep.dtype != np.uint8:
        dep = dep.astype(np.float32)
        dep[dep < 300] = 0.0
        dep = np.clip(dep / 19999.0 * 255.0, 0, 255).astype(np.uint8)
    b, g, r = cv2.split(vis)
    return cv2.merge((b, g, r, dep))  # [B,G,R,D] uint8


def main():
    model = YOLO(WEIGHTS)
    nn = model.model.to("cpu").eval()

    lb = LetterBox(new_shape=(IMGSZ, IMGSZ), auto=False, scaleup=False, center=True, stride=32)
    fmt = Format(bbox_format="xywh", normalize=True, return_mask=False,
                 return_keypoint=False, return_obb=False, batch_idx=True,
                 mask_ratio=4, mask_overlap=True, bgr=0.0)

    stems = []
    for ext in ["png", "jpg"]:
        stems += sorted(p.stem for p in VIS.glob(f"*.{ext}"))[:2]

    for stem in stems:
        vp = VIS / f"{stem}{'.png' if (VIS / f'{stem}.png').exists() else '.jpg'}"
        merged = _read_merged(vp)

        # --- predict pipeline ---
        lb_img = lb(image=merged)
        pred_x = torch.from_numpy(_to_chw(lb_img)[None]).float() / 255.0

        # --- val pipeline ---
        val_img = lb(image=merged.copy())
        val_tensor = fmt._format_img(val_img)  # 直接取 Format 的核心：HWC→CHW + 通道重排
        val_x = val_tensor.float()[None] / 255.0

        d = (pred_x - val_x).abs()
        print(f"{stem}: pred{tuple(pred_x.shape)} val{tuple(val_x.shape)} "
              f"max|diff|={d.max().item():.6e} mean={d.mean().item():.6e} "
              f"n_diff_px={(d > 0).sum().item()}")

        with torch.no_grad():
            o1 = nn(pred_x)
            o2 = nn(val_x)
        h1 = o1 if isinstance(o1, (list, tuple)) else [o1]
        h2 = o2 if isinstance(o2, (list, tuple)) else [o2]
        for lvl, (a, b) in enumerate(zip(h1, h2)):
            print(f"    head{lvl} max|diff|={(a - b).abs().max().item():.6e}")


if __name__ == "__main__":
    main()
