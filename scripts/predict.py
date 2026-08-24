"""scripts/predict.py — YOLOv11 目标检测预测脚本（ultralytics）。

职责：加载训练好的权重（best.pt）→ 调用 ultralytics 的 ``model.predict()``
对输入图片/目录做推理 → 输出带检测框的结果图片并保存。

用法:
    python scripts/predict.py                     # 预测 test/visible（默认）
    python scripts/predict.py --source data/raw/test/visible
    python scripts/predict.py --weights runs/<exp>/weights/best.pt --conf 0.3
    python scripts/predict.py --save_txt          # 额外输出 YOLO 格式 txt

输入:
    - visible 单模态（当前 baseline 阶段）: --source test/visible 或任意图片目录；
    - visible + infrared + depth 多模态: 待 fusion 模型落地后，把 --weights 指向
      fusion 模型、--source 指向多模态输入目录即可，脚本本身无需改动。

输出:
    带检测框的结果图片，保存到 project/name/（默认 runs/detect/predict/）。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 保证 `python scripts/predict.py` 时能导入项目内的 ultralytics / scripts
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402

# 复用 train.py 的配置读取 / 设备解析，保证与训练、评估口径一致
from scripts.train import _load_yaml, _resolve_device, _resolve_template  # noqa: E402


def _default_source(dataset_cfg: dict) -> str:
    """由 dataset.yaml 的 path + test 解析默认输入目录（test/visible）。"""
    path = dataset_cfg.get("path", "")
    test = dataset_cfg.get("test", "test/visible")
    return str(Path(path) / str(test))


def _parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11 目标检测预测")
    parser.add_argument("--weights", type=str, default=None,
                        help="权重文件路径；默认 train.yaml checkpoint.save_dir/best.pt")
    parser.add_argument("--source", type=str, default=None,
                        help="输入图片或目录；默认 dataset.yaml 的 test/visible")
    parser.add_argument("--dataset_config", type=str, default="configs/dataset.yaml",
                        help="dataset.yaml 路径，默认 configs/dataset.yaml")
    parser.add_argument("--train_config", type=str, default="configs/train.yaml",
                        help="train.yaml 路径，默认 configs/train.yaml")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda | cpu；cuda 时按 train.yaml gpu_ids 选择")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="推理尺寸，默认 train.yaml image_size")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值（默认 0.25）")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="NMS IoU 阈值（默认 0.7）")
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="结果输出根目录（默认 runs/detect）")
    parser.add_argument("--name", type=str, default="predict",
                        help="本次预测目录名（默认 predict，重名自动递增）")
    parser.add_argument("--save_txt", action="store_true",
                        help="额外输出 YOLO 格式标注 txt")
    return parser.parse_args()


def main():
    os.chdir(PROJECT_ROOT)

    args = _parse_args()
    train_cfg = _load_yaml(args.train_config)
    dataset_cfg = _load_yaml(args.dataset_config)

    device = "cpu" if str(args.device).lower() == "cpu" else _resolve_device(train_cfg)
    _image_size = train_cfg.get("image_size", [640, 640])
    imgsz = (args.imgsz or
             (int(_image_size[0]) if isinstance(_image_size, (list, tuple)) else int(_image_size)))

    # ---- 权重路径 ----
    if args.weights is None:
        ckpt_cfg = train_cfg.get("checkpoint", {})
        save_dir = PROJECT_ROOT / _resolve_template(
            ckpt_cfg.get("save_dir", "runs/${experiment_name}/weights"), train_cfg)
        weights_path = save_dir / "best.pt"
    else:
        weights_path = Path(args.weights)

    if not weights_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    # ---- 输入源 ----
    source = args.source or _default_source(dataset_cfg)
    if not Path(source).exists():
        raise FileNotFoundError(f"输入源不存在: {source}")

    print(f"[predict] weights={weights_path}")
    print(f"[predict] source={source}")
    print(f"[predict] device={device} imgsz={imgsz} conf={args.conf}")

    # ---- 推理 + 保存结果图片 ----
    model = YOLO(str(weights_path))
    results = model.predict(
        source=source,
        save=True,                 # 保存带检测框的结果图片
        save_txt=args.save_txt,    # 可选：YOLO 格式 txt
        conf=args.conf,
        iou=args.iou,
        imgsz=imgsz,
        device=device,
        project=args.project,
        name=args.name,
        verbose=True,
    )

    # 结果保存目录（ultralytics 自动递增 name）
    save_dir = Path(args.project) / args.name
    n_imgs = sum(1 for r in results if r is not None)
    print(f"[predict] 完成 {n_imgs} 张图片，结果保存至 → {save_dir.resolve()}")


if __name__ == "__main__":
    main()
