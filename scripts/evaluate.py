"""scripts/evaluate.py — 模型评估入口（YOLOv11 / YOLOv11-RGBT）。

职责：加载训练好的 best.pt → 调用 ultralytics 的 ``model.val()`` 在验证集上评估
→ 输出 mAP@0.5 / mAP@0.5:0.95 / precision / recall（及各类别指标），并把结果
以机器可读的 metrics.json 保存到 experiments/<实验名>/。

不再自实现解码 / NMS / mAP，全部交给 ultralytics 框架处理。

用法:
    python scripts/evaluate.py
    python scripts/evaluate.py --weights runs/urban_multimodal_det_v1/weights/best.pt
    python scripts/evaluate.py --device cpu --experiment baseline
    # RGBT 双模态 best.pt 验证（use_simotm 默认取 train_rgbt.yaml）
    python scripts/evaluate.py --weights runs/urban_multimodal_det_yolo11_rgbt/weights/best.pt \
                               --train_config configs/train_rgbt.yaml

说明:
    - 默认权重取 train.yaml checkpoint.save_dir/best.pt（与 train.py 一致）。
    - 验证集与训练一致：复用 data/processed/{visible_split,rgbt_split}/dataset.yaml
      （由 train.py 按 val_ratio + seed 切分生成）；RGBT 实验（use_simotm=RGBT）
      复用 rgbt_split，其余复用 visible_split，保证与训练期验证划分完全相同、
      无数据泄漏。
    - 多模态参数 use_simotm / pairs_rgb_ir 显式传入 model.val()（ultralytics 的
      _reset_ckpt_args 不会从 checkpoint 恢复这两个字段，默认回落 SimOTMBBS），
      否则 RGBT best.pt 会按 3ch 加载导致输入错位。
    - 结果保存到 experiments/<experiment_name>/metrics.json（按 experiments/README.md
      规范），并同步打印到终端。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# 保证 `python scripts/evaluate.py` 时能导入项目内的 ultralytics / scripts
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402

# 复用 train.py 的配置读取 / 设备解析 / val 切分，保证评估与训练口径完全一致
from scripts.train import (  # noqa: E402
    _load_yaml,
    _resolve_device,
    _resolve_template,
    _split_train_val,
    _split_train_val_rgbt,
    _val_has_labels,
)

SPLIT_YAML = "data/processed/visible_split/dataset.yaml"
SPLIT_YAML_RGBT = "data/processed/rgbt_split/dataset.yaml"


# ============================================================
# 基础工具
# ============================================================

def _to_py(v):
    """递归把 numpy / torch 标量转成 Python 原生类型，便于 json 序列化。"""
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.ndarray):
        return [_to_py(x) for x in v.tolist()]
    if isinstance(v, torch.Tensor):
        return _to_py(v.detach().cpu().numpy())
    if isinstance(v, (list, tuple)):
        return [_to_py(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_py(x) for k, x in v.items()}
    return v


def _resolve_val_data(dataset_cfg_path: str, dataset_cfg: dict,
                      train_cfg: dict, use_simotm: str = "SimOTMBBS") -> str:
    """确定验证数据 yaml：优先复用训练切分，必要时按相同逻辑生成。

    RGBT 实验（use_simotm=RGBT）复用 rgbt_split，否则复用 visible_split，
    保证验证数据与训练期完全一致、无数据泄漏。
    """
    split_yaml = SPLIT_YAML_RGBT if use_simotm == "RGBT" else SPLIT_YAML

    # 1) 训练期已生成的切分直接复用（保证与训练验证集一致）
    if Path(split_yaml).exists():
        return split_yaml

    # 2) dataset.yaml 的 val 本身有标注则直接用
    if _val_has_labels(dataset_cfg):
        return dataset_cfg_path

    # 3) 否则按 train.yaml 的 val_ratio + seed 切分（与 train.py 相同逻辑）
    val_ratio = float(train_cfg.get("val_ratio", 0.0))
    if val_ratio <= 0:
        # 无标注 val 且不切分 → 直接评估原 val（会恒得 mAP=0）
        print("[warn] val 无标注且 val_ratio<=0，将直接评估 dataset.yaml 的 val。")
        return dataset_cfg_path
    if use_simotm == "RGBT":
        return str(_split_train_val_rgbt(dataset_cfg, val_ratio,
                                         int(train_cfg.get("seed", 42))))
    return str(_split_train_val(dataset_cfg, val_ratio,
                                int(train_cfg.get("seed", 42))))


# ============================================================
# 指标提取
# ============================================================

def _extract_metrics(metrics) -> dict:
    """从 ultralytics DetMetrics 提取 mAP / precision / recall 及各类别指标。"""
    box = metrics.box
    names = getattr(metrics, "names", None) or getattr(box, "names", {}) or {}

    mp = float(box.mp)
    mr = float(box.mr)
    map50 = float(box.map50)
    map5095 = float(box.map)
    fitness = float(getattr(metrics, "fitness", 0.0) or 0.0)

    ap_class_index = [int(c) for c in getattr(box, "ap_class_index", [])]
    p = np.asarray(box.p) if len(box.p) else np.array([])
    r = np.asarray(box.r) if len(box.r) else np.array([])
    ap50 = np.asarray(box.ap50) if len(box.ap50) else np.array([])
    ap = np.asarray(box.ap) if len(box.ap) else np.array([])

    per_class = {}
    for i, c in enumerate(ap_class_index):
        per_class[str(c)] = {
            "name": names.get(c, str(c)),
            "precision": _to_py(p[i]) if i < len(p) else None,
            "recall": _to_py(r[i]) if i < len(r) else None,
            "ap50": _to_py(ap50[i]) if i < len(ap50) else None,
            "ap50_95": _to_py(ap[i]) if i < len(ap) else None,
        }

    speed = getattr(metrics, "speed", None)
    return {
        "precision": mp,
        "recall": mr,
        "mAP50": map50,
        "mAP50-95": map5095,
        "fitness": fitness,
        "per_class": per_class,
        "speed": _to_py(speed) if speed else None,
    }


# ============================================================
# 结果保存
# ============================================================

def _save_results(exp_dir: Path, payload: dict) -> Path:
    exp_dir.mkdir(parents=True, exist_ok=True)
    out_path = exp_dir / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_py(payload), f, ensure_ascii=False, indent=2)
    return out_path


# ============================================================
# 主流程
# ============================================================

def _parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11 目标检测模型评估")
    parser.add_argument("--weights", type=str, default=None,
                        help="权重文件路径；默认 train.yaml checkpoint.save_dir/best.pt")
    parser.add_argument("--dataset_config", type=str, default="configs/dataset.yaml",
                        help="dataset.yaml 路径，默认 configs/dataset.yaml")
    parser.add_argument("--train_config", type=str, default="configs/train.yaml",
                        help="train.yaml 路径，默认 configs/train.yaml")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda | cpu；cuda 时按 train.yaml gpu_ids 选择")
    parser.add_argument("--experiment", type=str, default=None,
                        help="结果保存目录名，默认 train.yaml experiment_name")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="覆盖 train.yaml 的 batch_size")
    parser.add_argument("--image_size", type=int, default=None,
                        help="覆盖 train.yaml 的 image_size")
    parser.add_argument("--conf", type=float, default=None,
                        help="置信度阈值（默认走 ultralytics val 的 0.001）")
    parser.add_argument("--iou", type=float, default=None,
                        help="NMS 的 IoU 阈值（默认 ultralytics 的 0.7）")
    parser.add_argument("--max_det", type=int, default=None,
                        help="每张图保留的最大检测数（默认 300）")
    parser.add_argument("--use_simotm", type=str, default=None,
                        help="多模态模式；默认取 train_config 的 use_simotm（SimOTMBBS/RGBT）")
    parser.add_argument("--pairs_rgb_ir", type=str, default=None,
                        help="第二模态映射 'a,b'；默认取 train_config 的 pairs_rgb_ir")
    return parser.parse_args()


def main():
    os.chdir(PROJECT_ROOT)

    args = _parse_args()
    train_cfg = _load_yaml(args.train_config)
    dataset_cfg_path = args.dataset_config
    dataset_cfg = _load_yaml(dataset_cfg_path)

    experiment_name = args.experiment or str(
        train_cfg.get("experiment_name", "urban_multimodal_det_v1"))
    device = "cpu" if str(args.device).lower() == "cpu" else _resolve_device(train_cfg)
    _image_size = train_cfg.get("image_size", [640, 640])
    imgsz = (args.image_size or
             (int(_image_size[0]) if isinstance(_image_size, (list, tuple)) else int(_image_size)))
    batch = args.batch_size or int(train_cfg.get("batch_size", 16))

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

    # ---- 多模态参数：显式覆盖（否则 val 会回落 SimOTMBBS 默认 3ch） ----
    use_simotm = args.use_simotm or str(train_cfg.get("use_simotm", "SimOTMBBS"))
    if args.pairs_rgb_ir:
        pairs_rgb_ir = [x.strip() for x in args.pairs_rgb_ir.split(",")]
    else:
        pairs_rgb_ir = list(train_cfg.get("pairs_rgb_ir", ["visible", "infrared"]))

    # ---- 验证数据 ----
    data_path = _resolve_val_data(dataset_cfg_path, dataset_cfg, train_cfg, use_simotm)

    print(f"[evaluate] weights={weights_path}")
    print(f"[evaluate] data={data_path}")
    print(f"[evaluate] use_simotm={use_simotm} pairs_rgb_ir={pairs_rgb_ir}")
    print(f"[evaluate] device={device} imgsz={imgsz} batch={batch}")

    # ---- 加载 best.pt 并验证 ----
    model = YOLO(str(weights_path))
    val_kwargs = {
        "data": data_path,
        "device": device,
        "imgsz": imgsz,
        "batch": batch,
        "split": "val",
        "plots": False,
        "verbose": True,
        "use_simotm": use_simotm,
        "pairs_rgb_ir": pairs_rgb_ir,
    }
    if args.conf is not None:
        val_kwargs["conf"] = args.conf
    if args.iou is not None:
        val_kwargs["iou"] = args.iou
    if args.max_det is not None:
        val_kwargs["max_det"] = args.max_det

    metrics = model.val(**val_kwargs)

    # ---- 提取并展示 ----
    result = _extract_metrics(metrics)

    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"weights   : {weights_path}")
    print(f"data      : {data_path}")
    print(f"device    : {device}")
    print("-" * 60)
    print(f"precision    : {result['precision']:.4f}")
    print(f"recall       : {result['recall']:.4f}")
    print(f"mAP@0.5      : {result['mAP50']:.4f}")
    print(f"mAP@0.5:0.95 : {result['mAP50-95']:.4f}")
    print("-" * 60)
    print("per-class (precision / recall / AP@0.5 / AP@0.5:0.95):")
    for cid, v in result["per_class"].items():
        def _f(x):
            return f"{x:.4f}" if isinstance(x, float) else "n/a"
        print(f"  {cid:>2} {v['name']:<12} {_f(v['precision'])} / {_f(v['recall'])} "
              f"/ {_f(v['ap50'])} / {_f(v['ap50_95'])}")
    print("=" * 60 + "\n")

    # ---- 保存到 experiments/<experiment>/metrics.json ----
    payload = {
        "experiment": experiment_name,
        "weights": str(weights_path),
        "data": data_path,
        "device": device,
        "imgsz": imgsz,
        "split": "val",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {k: result[k] for k in ("precision", "recall", "mAP50", "mAP50-95", "fitness")},
        "per_class": result["per_class"],
        "speed": result["speed"],
    }
    out_path = _save_results(PROJECT_ROOT / "experiments" / experiment_name, payload)
    print(f"[evaluate] 结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
