"""scripts/phase2_val_ref.py — 用 model.val() 在 E7 val split 上得到参考 mAP。

Phase 2 复现的参考基准：以 E7 best.pt 在 depth_split_train 的 val 集上运行 model.val()，
参数与 E7 训练内置 val 一致（rect=False、conf=0.001、iou=0.7、max_det=300、
use_simotm=RGBD、4ch），输出 mAP50/75/50-95/P/R。

用法:
  python scripts/phase2_val_ref.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402

WEIGHTS = "runs/urban_multimodal_det_e7_yolo11m_rgbd_1024/weights/best.pt"
DATA = "data/processed/depth_split_train/dataset.yaml"


def main():
    model = YOLO(WEIGHTS)
    metrics = model.val(
        data=DATA,
        imgsz=1024,
        batch=8,
        conf=0.001,
        iou=0.7,
        max_det=300,
        rect=False,          # E7 训练内置 val 用 rect=False（关键，不能默认 True）
        half=False,
        augment=False,
        single_cls=False,
        agnostic_nms=False,
        split="val",
        use_simotm="RGBD",
        pairs_rgb_ir=["visible", "depth"],
        pairs_rgb_depth=["visible", "depth"],
        channels=4,
        device="cpu",
        workers=0,           # Windows 下避免多进程数据加载
        plots=False,
        verbose=True,
    )
    print("=" * 56)
    print("model.val() 参考指标 (E7 best.pt @ depth_split_train val):")
    for k, v in metrics.results_dict.items():
        print(f"  {k}: {v:.5f}")
    print("=" * 56)


if __name__ == "__main__":
    main()
