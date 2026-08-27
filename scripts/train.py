"""scripts/train.py — YOLOv11 / YOLOv11-RGBT 训练入口（ultralytics）。

职责：读取 configs/{train,dataset}.yaml → 映射为 ultralytics YOLO.train() 参数 →
调用 YOLO("<model yaml>").train(...) 完成训练。

支持两种模式（由 train.yaml 的 use_simotm 决定）:
    - visible 单模态 (use_simotm=SimOTMBBS, 默认) → configs/yolo11_visible.yaml +
      data/processed/visible_split/dataset.yaml
    - RGBT 双模态 (use_simotm=RGBT) → configs/yolo11_rgbt.yaml (ch=4) +
      data/processed/rgbt_split/dataset.yaml

不再实现自定义训练循环；训练 / 验证 / checkpoint / EMA / AMP / 学习率调度等
全部交给 ultralytics 框架处理。

用法:
    # RGB baseline（单模态）
    python scripts/train.py
    python scripts/train.py --model_config configs/yolo11_visible.yaml \
                            --train_config configs/train.yaml \
                            --dataset_config configs/dataset.yaml
    # Experiment C: RGBT 双模态
    python scripts/train.py --model_config configs/yolo11_rgbt.yaml \
                            --train_config configs/train_rgbt.yaml \
                            --dataset_config configs/dataset.yaml

说明:
    - 当前数据无独立 val（dataset.yaml 的 val 指向无标注的 test/visible，会恒得
      mAP=0），故按 train.yaml 的 val_ratio 从 train 切分出带标注的 val 集：
      visible 模式生成 data/processed/visible_split/dataset.yaml，
      RGBT 模式生成 data/processed/rgbt_split/dataset.yaml（visible+infrared 配对）。
    - checkpoint 由 ultralytics 保存到 project/name/weights/{last,best,epoch_*}.pt，
      与 train.yaml 的 checkpoint.save_dir（runs/${experiment_name}/weights）对应。
    - 多模态参数 use_simotm / pairs_rgb_ir 由 train.yaml 传入，显式覆盖 ultralytics
      默认的 SimOTMBBS（3ch），保证 RGBT 实验真正以 4ch [B,G,R,IR] 输入训练。
"""
from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402

MODEL_YAML = "configs/yolo11_visible.yaml"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# 基础工具
# ============================================================

def _load_yaml(path):
    """读取 yaml 配置文件，返回解析后的字典。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_template(s, cfg):
    """把路径模板中的 ``${key}`` 替换为 train.yaml 顶层同名值。"""
    return re.sub(r"\$\{(\w+)\}", lambda m: str(cfg.get(m.group(1), m.group(0))), str(s))


def _resolve_device(train_cfg) -> str:
    """train.yaml 的 device / gpu_ids → ultralytics device 字符串。

    ultralytics 接受 "cpu"、"0"（单卡）、"0,1,2,3"（多卡）；不接受 "cuda"。
    """
    device = str(train_cfg.get("device", "cuda")).lower()
    if device == "cpu":
        return "cpu"
    gpu_ids = train_cfg.get("gpu_ids") or [0]
    return ",".join(str(int(i)) for i in gpu_ids)


# ============================================================
# 数据：无标注 val 时按 val_ratio 从 train 切分
# ============================================================

def _val_has_labels(dataset_cfg) -> bool:
    """dataset.yaml 的 val 目录下是否存在标注文件。"""
    val_dir = Path(dataset_cfg.get("path", "")) / str(dataset_cfg.get("val", ""))
    return any(val_dir.rglob("*.txt")) if val_dir.is_dir() else False


def _link_or_copy(src: Path, dst: Path) -> None:
    """优先硬链接（省磁盘、免拷贝），失败回退到拷贝。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _split_train_val(dataset_cfg, val_ratio: float, seed: int) -> Path:
    """从 train 切分出带标注的 val 集，返回生成的 dataset yaml 路径。

    采用 ultralytics 约定布局: <root>/images/{train,val} + <root>/labels/{train,val}。
    已切分过则复用（幂等）。
    """
    src_dir = Path(dataset_cfg["path"]) / str(dataset_cfg["train"])
    # 标注目录：本项目标签位于 <train父目录>/labels（如 train/labels），
    # 而非与图像同目录；若不存在则回退到图像同目录（labels 紧邻图像）。
    labels_src = src_dir.parent / "labels"
    if not labels_src.is_dir():
        labels_src = src_dir
    root = PROJECT_ROOT / "data" / "processed" / "visible_split"
    out_yaml = root / "dataset.yaml"
    if out_yaml.exists():
        return out_yaml

    images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    rng = random.Random(seed)
    rng.shuffle(images)
    n_val = max(1, int(round(len(images) * val_ratio)))
    val_set = set(images[:n_val])

    for p in images:
        split = "val" if p in val_set else "train"
        _link_or_copy(p, root / "images" / split / p.name)
        _link_or_copy(labels_src / p.with_suffix(".txt").name,
                      root / "labels" / split / p.with_suffix(".txt").name)

    yaml.safe_dump(
        {
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "nc": dataset_cfg.get("nc", 12),
            "names": dataset_cfg.get("names"),
        },
        open(out_yaml, "w", encoding="utf-8"),
        allow_unicode=True,
    )
    print(f"[data] 已从 train 切分 val（val_ratio={val_ratio}）→ {out_yaml}")
    return out_yaml


def _split_train_val_rgbt(dataset_cfg, val_ratio: float, seed: int) -> Path:
    """为 RGBT 双模态切分独立数据，返回 data/processed/rgbt_split/dataset.yaml。

    布局（满足 YOLOv11-RGBT fork 的两条读取约定）:
      rgbt_split/
        images/{train,val}/visible/     # 可见光主输入（含 'visible'）
        images/{train,val}/infrared/    # 红外第二模态（与 visible 同名）
        labels/{train,val}/visible/     # 标注（与 visible 同名，.txt）
    约定 1（第二模态路径）: base.py 用 file_path.replace('visible','infrared')
        得到红外图 → 需 images/{split}/infrared/ 与 images/{split}/visible/ 同名。
    约定 2（标注路径）: utils.py 的 img2label_paths 把 /images/ 替换为 /labels/
        再改扩展名为 .txt → 需 labels/{split}/visible/*.txt。

    与 visible_split 使用相同 seed + val_ratio，保证 RGBT 与 RGB baseline 的
    train/val 划分完全一致、无数据泄漏。已切分过则复用（幂等）。
    """
    src_dir = Path(dataset_cfg["path"]) / str(dataset_cfg["train"])  # train/visible
    infrared_src = src_dir.parent / "infrared"                        # train/infrared
    labels_src = src_dir.parent / "labels"                            # train/labels
    if not labels_src.is_dir():
        labels_src = src_dir
    if not infrared_src.is_dir():
        raise FileNotFoundError(f"红外目录不存在，无法生成 RGBT split: {infrared_src}")

    root = PROJECT_ROOT / "data" / "processed" / "rgbt_split"
    out_yaml = root / "dataset.yaml"
    if out_yaml.exists():
        return out_yaml

    images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    rng = random.Random(seed)
    rng.shuffle(images)
    n_val = max(1, int(round(len(images) * val_ratio)))
    val_set = set(images[:n_val])

    for p in images:
        split = "val" if p in val_set else "train"
        # visible 主输入
        _link_or_copy(p, root / "images" / split / "visible" / p.name)
        # infrared 第二模态（同名）
        _link_or_copy(infrared_src / p.name, root / "images" / split / "infrared" / p.name)
        # 标注（images→labels 替换约定 → labels/{split}/visible/*.txt）
        _link_or_copy(labels_src / p.with_suffix(".txt").name,
                      root / "labels" / split / "visible" / p.with_suffix(".txt").name)

    yaml.safe_dump(
        {
            "path": str(root),
            "train": "images/train/visible",
            "val": "images/val/visible",
            "nc": dataset_cfg.get("nc", 12),
            "names": dataset_cfg.get("names"),
        },
        open(out_yaml, "w", encoding="utf-8"),
        allow_unicode=True,
    )
    print(f"[data] 已从 train 切分 RGBT val（val_ratio={val_ratio}）→ {out_yaml}")
    return out_yaml


# ============================================================
# train.yaml → ultralytics 参数映射
# ============================================================

def _build_train_kwargs(train_cfg, data_path):
    """把 train.yaml 映射为 ultralytics YOLO.train() 参数。"""
    ckpt = train_cfg.get("checkpoint", {})
    sched = train_cfg.get("scheduler", {})
    opt = train_cfg.get("optimizer", {})

    # checkpoint 目录: train.yaml 的 runs/${experiment_name}/weights
    #   ↔ ultralytics 的 project/name/weights
    save_path = Path(_resolve_template(
        ckpt.get("save_dir", "runs/${experiment_name}/weights"), train_cfg))

    imgsz = train_cfg.get("image_size", [640, 640])
    if isinstance(imgsz, (list, tuple)):
        imgsz = int(imgsz[0])

    kwargs = {
        "data": str(data_path),
        "epochs": int(train_cfg.get("epochs", 300)),
        "patience": int(train_cfg.get("patience", 100)),
        "batch": int(train_cfg.get("batch_size", 16)),
        "imgsz": imgsz,
        "device": _resolve_device(train_cfg),
        "workers": int(train_cfg.get("num_workers", 4)),
        "optimizer": str(opt.get("type", "auto")),
        "lr0": float(train_cfg.get("learning_rate", 1e-2)),
        "warmup_epochs": float(train_cfg.get("warmup_epochs", 3.0)),
        "seed": int(train_cfg.get("seed", 42)),
        "amp": bool(train_cfg.get("amp", True)),
        "save": bool(ckpt.get("save_best", True) or ckpt.get("save_last", True)),
        "save_period": int(ckpt.get("save_interval", -1)),
        "project": str(save_path.parent.parent),
        "name": save_path.parent.name,
    }

    # 学习率调度: CosineAnnealingLR → ultralytics cos_lr
    if str(sched.get("type", "")).lower() == "cosineannealinglr":
        kwargs["cos_lr"] = True
    if opt.get("momentum") is not None:
        kwargs["momentum"] = float(opt["momentum"])
    if opt.get("weight_decay") is not None:
        kwargs["weight_decay"] = float(opt["weight_decay"])

    # 多模态: 显式传递 use_simotm / pairs_rgb_ir。
    # ultralytics 默认 use_simotm=SimOTMBBS（灰度+模糊合并为 3ch）；RGBT 实验必须
    # 显式覆盖为 RGBT，否则训练/验证会回落到默认 3ch 加载，导致 4ch 模型输入错位。
    kwargs["use_simotm"] = str(train_cfg.get("use_simotm", "SimOTMBBS"))
    kwargs["pairs_rgb_ir"] = list(train_cfg.get("pairs_rgb_ir", ["visible", "infrared"]))

    return kwargs


# ============================================================
# 主流程
# ============================================================

def _parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11 visible 单模态 Baseline 训练")
    parser.add_argument("--model_config", type=str, default=MODEL_YAML,
                        help="YOLOv11 模型 yaml 路径，默认 configs/yolo11_visible.yaml")
    parser.add_argument("--train_config", type=str, default="configs/train.yaml",
                        help="训练配置路径，默认 configs/train.yaml")
    parser.add_argument("--dataset_config", type=str, default="configs/dataset.yaml",
                        help="数据集配置路径，默认 configs/dataset.yaml")
    return parser.parse_args()


def main():
    # 使 configs 中的相对路径基于项目根解析
    os.chdir(PROJECT_ROOT)

    args = _parse_args()
    train_cfg = _load_yaml(args.train_config)
    dataset_cfg_path = args.dataset_config
    dataset_cfg = _load_yaml(dataset_cfg_path)

    # ---- data: 无标注 val 时按 val_ratio 切分 train，否则直接用原 dataset.yaml ----
    data_path = dataset_cfg_path
    val_ratio = float(train_cfg.get("val_ratio", 0.0))
    use_simotm = str(train_cfg.get("use_simotm", "SimOTMBBS"))
    if val_ratio > 0 and not _val_has_labels(dataset_cfg):
        if use_simotm == "RGBT":
            # RGBT: 独立切分，生成 visible+infrared 配对的 rgbt_split（不碰 visible_split）
            data_path = str(_split_train_val_rgbt(
                dataset_cfg, val_ratio, int(train_cfg.get("seed", 42))))
        else:
            data_path = str(_split_train_val(
                dataset_cfg, val_ratio, int(train_cfg.get("seed", 42))))

    kwargs = _build_train_kwargs(train_cfg, data_path)

    # ---- model / resume ----
    ckpt = train_cfg.get("checkpoint", {})
    resume = bool(ckpt.get("resume", False))
    resume_path = str(ckpt.get("resume_path", "") or "")
    if resume and resume_path:
        model_path, kwargs["resume"], kwargs["exist_ok"] = resume_path, True, True
    else:
        model_path = args.model_config
        kwargs["resume"] = False

    print(f"[train] model={model_path}")
    print(f"[train] data={data_path}")
    print(f"[train] use_simotm={kwargs.get('use_simotm')} "
          f"pairs_rgb_ir={kwargs.get('pairs_rgb_ir')}")
    print(f"[train] device={kwargs['device']} epochs={kwargs['epochs']} "
          f"batch={kwargs['batch']} imgsz={kwargs['imgsz']} "
          f"opt={kwargs['optimizer']} cos_lr={kwargs.get('cos_lr', False)}")

    model = YOLO(model_path)
    model.train(**kwargs)


if __name__ == "__main__":
    main()
