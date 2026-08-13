# ============================================================
# Urban Multi-Modal Object Detection — Dataset & DataLoader
# ============================================================
# 按 stem 对齐 visible / infrared / depth 三模态，兼容 .png/.jpg 混存；
# train 含 labels (YOLO 格式)，test 无；无独立 val，由 train 切分。
#
# 用法:
#   from utils.dataset import create_dataloaders
#   loaders = create_dataloaders("configs/dataset.yaml", image_size=[640, 640])
# ============================================================

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "load_dataset_config",
    "collect_split",
    "split_train_val",
    "MultimodalDetectionDataset",
    "collate_fn",
    "create_dataloaders",
]


def load_dataset_config(cfg_path: Union[str, os.PathLike]) -> dict:
    """读取 dataset.yaml，把单扩展名字符串统一为列表。"""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for spec in cfg["modalities"].values():
        if isinstance(spec["ext"], str):
            spec["ext"] = [spec["ext"]]
    return cfg


def _index_by_stem(root: Path, dirname: str, exts: Sequence[str]) -> Dict[str, Path]:
    """目录下指定扩展名文件 -> {stem: 路径}，同名多扩展名取先出现的。"""
    index: Dict[str, Path] = {}
    d = root / dirname
    if d.is_dir():
        for ext in exts:
            for p in sorted(d.glob(f"*{ext}")):
                index.setdefault(p.stem, p)
    return index


def collect_split(cfg: dict, split: str) -> List[dict]:
    """收集 split 下三模态对齐的样本 -> [{stem, paths, label_path?}]。"""
    root = Path(cfg["dataset_root"])
    sc = cfg[split]
    idx = {m: _index_by_stem(root, sc[m], s["ext"]) for m, s in cfg["modalities"].items()}
    if sc.get("labels"):
        idx["labels"] = _index_by_stem(root, sc["labels"], cfg["label"]["ext"])

    stems = set.intersection(*(set(v) for v in idx.values()))
    samples = []
    for stem in sorted(stems):
        s = {"stem": stem, "paths": {m: idx[m][stem] for m in cfg["modalities"]}}
        if "labels" in idx:
            s["label_path"] = idx["labels"][stem]
        samples.append(s)

    for m in cfg["modalities"]:
        n = len(idx[m]) - len(stems)
        if n > 0:
            print(f"[dataset] {split}/{m}: {n} 个文件因跨模态无法对齐而被跳过")
    return samples


def split_train_val(samples: List[dict], val_ratio: float = 0.2, seed: int = 42):
    """按比例随机切分 train -> (train, val)。"""
    perm = np.random.default_rng(seed).permutation(len(samples))
    val_idx = set(perm[: int(len(samples) * val_ratio)].tolist())
    return (
        [s for i, s in enumerate(samples) if i not in val_idx],
        [s for i, s in enumerate(samples) if i in val_idx],
    )


def _load_image(path: Path, channels: int, dtype: str,
                image_size: Optional[Sequence[int]], norm: bool) -> torch.Tensor:
    """读取单图 -> 归一化 tensor (C, H, W) float32。"""
    flag = cv2.IMREAD_COLOR if channels == 3 else cv2.IMREAD_UNCHANGED
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    if channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 3:
        img = img[:, :, 0] if dtype == "uint16" else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if image_size is not None:
        img = cv2.resize(img, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
    if img.ndim == 2:
        img = img[..., None]

    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
    if norm and np.issubdtype(img.dtype, np.integer):
        t /= float(np.iinfo(img.dtype).max)
    return t


def _read_labels(path: Path) -> torch.Tensor:
    """YOLO 标注 -> tensor (N, 5) float32。"""
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                labels.append([float(x) for x in p[:5]])
    return torch.tensor(labels, dtype=torch.float32).reshape(-1, 5)


class MultimodalDetectionDataset(Dataset):
    """返回 {stem, images{visible,infrared,depth}, labels, paths}。"""

    def __init__(self, cfg: dict, split: str, samples: Optional[List[dict]] = None,
                 image_size: Optional[Sequence[int]] = None, norm: bool = True):
        self.modalities = cfg["modalities"]
        self.samples = samples if samples is not None else collect_split(cfg, split)
        self.image_size = image_size
        self.norm = norm

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        images = {
            m: _load_image(p, self.modalities[m]["channels"], self.modalities[m]["dtype"],
                           self.image_size, self.norm)
            for m, p in s["paths"].items()
        }
        return {
            "stem": s["stem"],
            "images": images,
            "labels": _read_labels(s["label_path"]) if s.get("label_path") else None,
            "paths": s["paths"],
        }


def collate_fn(batch: List[dict]) -> dict:
    """图像按模态堆叠，labels 保持变长 list。"""
    mods = list(batch[0]["images"])
    return {
        "stems": [b["stem"] for b in batch],
        "images": {m: torch.stack([b["images"][m] for b in batch]) for m in mods},
        "labels": [b["labels"] for b in batch],
    }


def create_dataloaders(cfg_path: Union[str, os.PathLike], image_size: Optional[Sequence[int]] = None,
                       val_ratio: float = 0.2, seed: int = 42, batch_size: int = 16,
                       num_workers: int = 4, shuffle: bool = True, pin_memory: bool = True) -> Dict[str, DataLoader]:
    """构建 train / val / test 三个 DataLoader (val 从 train 切分)。"""
    cfg = load_dataset_config(cfg_path)
    train_s, val_s = split_train_val(collect_split(cfg, "train"), val_ratio, seed)
    datasets = {
        "train": MultimodalDetectionDataset(cfg, "train", train_s, image_size=image_size),
        "val": MultimodalDetectionDataset(cfg, "train", val_s, image_size=image_size),
        "test": MultimodalDetectionDataset(cfg, "test", image_size=image_size),
    }
    loaders = {
        name: DataLoader(ds, batch_size=batch_size, shuffle=(shuffle and name == "train"),
                         num_workers=num_workers, pin_memory=pin_memory,
                         collate_fn=collate_fn, drop_last=(name == "train"))
        for name, ds in datasets.items()
    }
    print(f"[dataset] train={len(datasets['train'])} "
          f"val={len(datasets['val'])} test={len(datasets['test'])}")
    return loaders


if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "dataset.yaml")
    loaders = create_dataloaders(cfg_path, image_size=[640, 640], batch_size=4)
    b = next(iter(loaders["train"]))
    for m, t in b["images"].items():
        print(f"images['{m}']: {tuple(t.shape)} {t.dtype}")
    print(f"labels: {len(b['labels'])} 张, 首个 {tuple(b['labels'][0].shape)}")
