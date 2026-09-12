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


_AUG_SUFFIX_RE = re.compile(r"_(?:aug|bal)\d+$")


def _base_stem(path: Path) -> str:
    """返回原图 stem：去掉 `_augN` / `_balN` 后缀（若有）。

    离线增强脚本把第 i 个变体命名为 `<stem>_aug<i>`，类别过采样脚本把第 i 个
    副本命名为 `<stem>_bal<i>`。据此把同一原图的所有变体/副本归为一组，
    避免随机 shuffle 把近重复样本切到 train/val 两侧造成泄漏。
    """
    return _AUG_SUFFIX_RE.sub("", path.stem)


def _split_by_group(files, val_ratio, seed):
    """按【原图 stem】分组切分，返回 (train_files, val_files)。

    - 切分单位是原图 stem 数（train:val = 0.8:0.2），而非文件数。
    - 同一原图 stem 的所有文件（原图 + _augN/_balN 副本）进同一侧，杜绝近重复泄漏。
    - train 侧保留该 stem 的全部文件（原图 + 副本），供过采样/增强。
    - val 侧只保留原图（不含 _balN/_augN 副本），保证验证集干净、与 baseline 同分布，
      否则过采样副本会被计入 val 导致验证集膨胀、mAP 不可比（2026-09 depth2 已踩坑）。
    - 兼容无 _aug*/_bal* 后缀的数据：每个 stem 只有一个文件，行为与旧版一致。
    """
    groups = {}
    for f in files:
        groups.setdefault(_base_stem(f), []).append(f)
    stems = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(stems)
    n_val = max(1, int(round(len(stems) * val_ratio)))
    val_stems = set(stems[:n_val])
    train_files, val_files = [], []
    for s in stems:
        if s in val_stems:
            val_files.extend(f for f in groups[s] if f.stem == s)  # 仅原图，剔除 _balN/_augN
        else:
            train_files.extend(groups[s])  # 原图 + 全部副本
    return train_files, val_files


def _resolve_raw_root(dataset_cfg) -> Path:
    """返回可移植的原始数据根目录（data/raw）。

    dataset.yaml 的 path 字段可能是平台相关的绝对路径（如 Windows 下的
    d:/gyt/.../data/raw），在 Linux 上不存在。这里优先使用配置的 path；若该目录
    不存在则回退到项目根目录下的 data/raw，避免硬编码平台相关路径。
    """
    cfg_root = Path(dataset_cfg.get("path", ""))
    return cfg_root if cfg_root.is_dir() else PROJECT_ROOT / "data" / "raw"


def _split_train_val(dataset_cfg, val_ratio: float, seed: int) -> Path:
    """从 train 切分出带标注的 val 集，返回生成的 dataset yaml 路径。

    采用 ultralytics 约定布局: <root>/images/{train,val} + <root>/labels/{train,val}。
    已切分过则复用（幂等）。
    """
    src_dir = _resolve_raw_root(dataset_cfg) / str(dataset_cfg["train"])
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
    train_files, val_files = _split_by_group(images, val_ratio, seed)

    for p in train_files:
        _link_or_copy(p, root / "images" / "train" / p.name)
        _link_or_copy(labels_src / p.with_suffix(".txt").name,
                      root / "labels" / "train" / p.with_suffix(".txt").name)
    for p in val_files:
        _link_or_copy(p, root / "images" / "val" / p.name)
        _link_or_copy(labels_src / p.with_suffix(".txt").name,
                      root / "labels" / "val" / p.with_suffix(".txt").name)

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
    src_dir = _resolve_raw_root(dataset_cfg) / str(dataset_cfg["train"])  # train/visible
    infrared_src = src_dir.parent / "infrared"                        # train/infrared
    labels_src = src_dir.parent / "labels"                            # train/labels
    if not labels_src.is_dir():
        labels_src = src_dir
    if not infrared_src.is_dir():
        raise FileNotFoundError(f"红外目录不存在，无法生成 RGBT split: {infrared_src}")

    # 切分目录按来源子目录命名（train vs train_aug），避免切换 dataset 配置时复用旧缓存
    src_key = Path(str(dataset_cfg["train"])).parent.name
    root = PROJECT_ROOT / "data" / "processed" / f"rgbt_split_{src_key}"
    out_yaml = root / "dataset.yaml"
    if out_yaml.exists():
        return out_yaml

    images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    train_files, val_files = _split_by_group(images, val_ratio, seed)

    for p in train_files:
        # visible 主输入
        _link_or_copy(p, root / "images" / "train" / "visible" / p.name)
        # infrared 第二模态（同名）
        _link_or_copy(infrared_src / p.name, root / "images" / "train" / "infrared" / p.name)
        # 标注（images→labels 替换约定 → labels/train/visible/*.txt）
        _link_or_copy(labels_src / p.with_suffix(".txt").name,
                      root / "labels" / "train" / "visible" / p.with_suffix(".txt").name)
    for p in val_files:
        _link_or_copy(p, root / "images" / "val" / "visible" / p.name)
        _link_or_copy(infrared_src / p.name, root / "images" / "val" / "infrared" / p.name)
        _link_or_copy(labels_src / p.with_suffix(".txt").name,
                      root / "labels" / "val" / "visible" / p.with_suffix(".txt").name)

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


def _split_train_val_depth(dataset_cfg, val_ratio: float, seed: int) -> Path:
    """为 Depth 单模态切分独立数据，返回 data/processed/depth_split/dataset.yaml。

    布局（与 RGBT split 相同约定，第二模态换成 depth）:
      depth_split/
        images/{train,val}/visible/     # 可见光主输入（含 'visible'）
        images/{train,val}/depth/       # 深度第二模态（与 visible 同名）
        labels/{train,val}/visible/     # 标注（与 visible 同名，.txt）
    Depth 分支用 file_path.replace('visible','depth') 得到深度图 → 需
    images/{split}/depth/ 与 images/{split}/visible/ 同名。
    使用相同 seed + val_ratio，保证与 RGB baseline 划分一致、无数据泄漏。
    """
    src_dir = _resolve_raw_root(dataset_cfg) / str(dataset_cfg["train"])  # train/visible
    depth_src = src_dir.parent / "depth"                             # train/depth
    labels_src = src_dir.parent / "labels"                           # train/labels
    if not labels_src.is_dir():
        labels_src = src_dir
    if not depth_src.is_dir():
        raise FileNotFoundError(f"深度目录不存在，无法生成 Depth split: {depth_src}")

    # 切分目录按来源子目录命名（train vs train_aug），避免切换 dataset 配置时复用旧缓存
    src_key = Path(str(dataset_cfg["train"])).parent.name
    root = PROJECT_ROOT / "data" / "processed" / f"depth_split_{src_key}"
    out_yaml = root / "dataset.yaml"
    if out_yaml.exists():
        return out_yaml

    images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    train_files, val_files = _split_by_group(images, val_ratio, seed)

    for p in train_files:
        # visible 主输入
        _link_or_copy(p, root / "images" / "train" / "visible" / p.name)
        # depth 第二模态（同名）
        _link_or_copy(depth_src / p.name, root / "images" / "train" / "depth" / p.name)
        # 标注（images→labels 替换约定 → labels/train/visible/*.txt）
        _link_or_copy(labels_src / p.with_suffix(".txt").name,
                      root / "labels" / "train" / "visible" / p.with_suffix(".txt").name)
    for p in val_files:
        _link_or_copy(p, root / "images" / "val" / "visible" / p.name)
        _link_or_copy(depth_src / p.name, root / "images" / "val" / "depth" / p.name)
        _link_or_copy(labels_src / p.with_suffix(".txt").name,
                      root / "labels" / "val" / "visible" / p.with_suffix(".txt").name)

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
    print(f"[data] 已从 train 切分 Depth val（val_ratio={val_ratio}）→ {out_yaml}")
    return out_yaml


def _split_train_val_rgbid(dataset_cfg, val_ratio: float, seed: int) -> Path:
    """为 RGBID 三模态（RGB+IR+Depth）切分独立数据，返回 data/processed/rgbid_split/dataset.yaml。

    布局（RGBID 5ch 融合用；第二/第三模态分别靠 'visible'→'infrared' / 'visible'→'depth'
    路径替换读取，因此三目录必须同名）:
      rgbid_split/
        images/{train,val}/visible/     # 可见光主输入（含 'visible'）
        images/{train,val}/infrared/    # 红外第二模态（与 visible 同名）
        images/{train,val}/depth/       # 深度第三模态（与 visible 同名）
        labels/{train,val}/visible/     # 标注（与 visible 同名，.txt）
    使用相同 seed + val_ratio，保证与 RGB baseline 划分一致、无数据泄漏。幂等。
    """
    src_dir = _resolve_raw_root(dataset_cfg) / str(dataset_cfg["train"])  # train/visible
    infrared_src = src_dir.parent / "infrared"                        # train/infrared
    depth_src = src_dir.parent / "depth"                             # train/depth
    labels_src = src_dir.parent / "labels"                           # train/labels
    if not labels_src.is_dir():
        labels_src = src_dir
    if not infrared_src.is_dir():
        raise FileNotFoundError(f"红外目录不存在，无法生成 RGBID split: {infrared_src}")
    if not depth_src.is_dir():
        raise FileNotFoundError(f"深度目录不存在，无法生成 RGBID split: {depth_src}")

    # 切分目录按来源子目录命名（train vs train_aug），避免切换 dataset 配置时复用旧缓存
    src_key = Path(str(dataset_cfg["train"])).parent.name
    root = PROJECT_ROOT / "data" / "processed" / f"rgbid_split_{src_key}"
    out_yaml = root / "dataset.yaml"
    if out_yaml.exists():
        return out_yaml

    images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    train_files, val_files = _split_by_group(images, val_ratio, seed)

    for p in train_files:
        _link_or_copy(p, root / "images" / "train" / "visible" / p.name)
        _link_or_copy(infrared_src / p.name, root / "images" / "train" / "infrared" / p.name)
        _link_or_copy(depth_src / p.name, root / "images" / "train" / "depth" / p.name)
        _link_or_copy(labels_src / p.with_suffix(".txt").name,
                      root / "labels" / "train" / "visible" / p.with_suffix(".txt").name)
    for p in val_files:
        _link_or_copy(p, root / "images" / "val" / "visible" / p.name)
        _link_or_copy(infrared_src / p.name, root / "images" / "val" / "infrared" / p.name)
        _link_or_copy(depth_src / p.name, root / "images" / "val" / "depth" / p.name)
        _link_or_copy(labels_src / p.with_suffix(".txt").name,
                      root / "labels" / "val" / "visible" / p.with_suffix(".txt").name)

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
    print(f"[data] 已从 train 切分 RGBID val（val_ratio={val_ratio}）→ {out_yaml}")
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
    # RGBID 三模态：额外传递深度映射与输入通道数（channels 门控 5ch 增强分支）
    kwargs["pairs_rgb_depth"] = list(train_cfg.get("pairs_rgb_depth", ["visible", "depth"]))
    kwargs["channels"] = int(train_cfg.get("channels", 3))
    # 类别加权（可选）：每类正样本权重，解决类别不平衡（2026-09 方法2，只作用于 cls loss）
    if "cls_pw" in train_cfg:
        kwargs["cls_pw"] = list(train_cfg["cls_pw"])

    # 数据增强超参（可选，由 train.yaml 的 aug 段控制；未配置则用 ultralytics 默认）。
    # 红外等非 RGB 模态需关闭 hsv_h/hsv_s、降低 mosaic/mixup 等，见 train_infrared.yaml。
    aug = train_cfg.get("aug", {})
    for key in ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
                "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup",
                "copy_paste", "copy_paste_mode", "erasing", "crop_fraction", "close_mosaic"):
        if key in aug:
            kwargs[key] = aug[key]

    # 预训练权重：False=从头训练；True=默认；字符串=指定权重文件（如 "yolo11n.pt"）。
    # 注意：ultralytics setup_model 只认 str/Path 才真正加载权重，bool True 是死配置（不加载任何权重）。
    if "pretrained" in train_cfg:
        p = train_cfg["pretrained"]
        kwargs["pretrained"] = p if isinstance(p, str) else bool(p)

    # warmup 细节（可选）
    if "warmup_bias_lr" in train_cfg:
        kwargs["warmup_bias_lr"] = float(train_cfg["warmup_bias_lr"])
    if "warmup_momentum" in train_cfg:
        kwargs["warmup_momentum"] = float(train_cfg["warmup_momentum"])

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
        if use_simotm in ("RGBT", "Infrared"):
            # RGBT(4ch 融合) 与 Infrared(单模态) 都靠 visible->infrared 路径替换，
            # 复用 visible+infrared 配对的 rgbt_split（不碰 visible_split）。
            data_path = str(_split_train_val_rgbt(
                dataset_cfg, val_ratio, int(train_cfg.get("seed", 42))))
        elif use_simotm in ("Depth", "RGBD"):
            # Depth(单模态) 与 RGBD(RGB+Depth 双模态融合) 都靠 visible->depth 路径替换，
            # 复用 visible+depth 配对的 depth_split（不碰 visible_split）。
            data_path = str(_split_train_val_depth(
                dataset_cfg, val_ratio, int(train_cfg.get("seed", 42))))
        elif use_simotm == "RGBID":
            # RGBID: 独立切分，生成 visible+infrared+depth 配对的 rgbid_split
            data_path = str(_split_train_val_rgbid(
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
