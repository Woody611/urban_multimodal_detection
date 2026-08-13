"""多模态目标检测数据集与 DataLoader。

数据读取 / letterbox / 张量化在 utils.preprocess.py；本模块负责：
- 按 stem 对齐 visible / infrared / depth 三模态；
- 构造 Dataset，并在取样本时对三种模态 + label 做一致的 letterbox 几何变换；
- 提供配置加载、train→val 切分、DataLoader 工厂与 collate。

用法:
  from utils.dataset import create_dataloaders
  loaders = create_dataloaders("configs/dataset.yaml", image_size=[640, 640])
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from .preprocess import (
    read_rgb,
    read_depth,
    compute_letterbox,
    letterbox_rgb,
    letterbox_depth,
    letterbox_boxes,
    image_to_tensor,
    depth_to_tensor,
)

__all__ = [
    "load_dataset_config",
    "split_train_val",
    "MultiModalDataset",
    "collate_fn",
    "create_dataloaders",
]


def load_dataset_config(cfg_path: Union[str, os.PathLike]) -> dict:
    """读取 dataset.yaml，并把扩展名键统一为 ``exts`` 列表。

    兼容旧写法 ``ext: ".png"`` / ``ext: [".png", ".jpg"]``；
    若已存在 ``exts`` 则原样保留。
    """
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for spec in cfg["modalities"].values():
        if "exts" not in spec and "ext" in spec:
            ext = spec.pop("ext")
            spec["exts"] = [ext] if isinstance(ext, str) else list(ext)
    return cfg


class MultiModalDataset(Dataset):
    def __init__(self, config, split: str = "train", target_size=(640, 640), root=None):
        """构建数据集。

        Args:
            config: 解析后的 dataset.yaml 字典。
            split: "train" | "val" | "test"。
            target_size: (H, W)，三种模态统一 letterbox 到该尺寸。
            root: 覆盖 dataset_root（可选，默认取 config["dataset_root"]）。
        """
        super().__init__()
        self.config = config
        self.split = split
        self.target_size = tuple(target_size)  # (H, W)

        dataset_root = Path(root) if root is not None else Path(config["dataset_root"])
        split_cfg = config[split]
        ext_map = {name: spec["exts"] for name, spec in config["modalities"].items()}

        self.visible_dir = dataset_root / split_cfg["visible"]
        self.infrared_dir = dataset_root / split_cfg["infrared"]
        self.depth_dir = dataset_root / split_cfg["depth"]
        self.labels_dir = dataset_root / split_cfg["labels"] if "labels" in split_cfg else None

        # 各模态 stem -> 文件路径
        self.vis_map = self._discover(self.visible_dir, ext_map["visible"])
        self.ir_map = self._discover(self.infrared_dir, ext_map["infrared"])
        self.dep_map = self._discover(self.depth_dir, ext_map["depth"])

        # 三模态 stem 交集；train/val 再与 label 交集
        common = set(self.vis_map) & set(self.ir_map) & set(self.dep_map)
        self.lab_map = None
        if self.labels_dir is not None:
            self.lab_map = self._discover(self.labels_dir, [".txt"])
            common &= set(self.lab_map)

        self.samples = sorted(common)
        self.has_labels = self.lab_map is not None

    @staticmethod
    def _discover(directory: Path, exts):
        """返回 {stem: 文件路径}，仅保留扩展名在 exts 内的文件。"""
        mapping = {}
        if directory is None or not directory.exists():
            return mapping
        exts = {e.lower() for e in exts}
        for p in directory.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                mapping[p.stem] = p
        return mapping

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem = self.samples[idx]

        vis_raw = read_rgb(self.vis_map[stem])
        ir_raw = read_rgb(self.ir_map[stem])
        dep_raw = read_depth(self.dep_map[stem])
        orig_h, orig_w = vis_raw.shape[:2]
        orig_size = (orig_h, orig_w)

        # 只算一次几何参数，三种模态 + label 严格共用（空间对齐）
        geo = compute_letterbox(orig_size, self.target_size)

        vis = letterbox_rgb(vis_raw, self.target_size, geo)
        ir = letterbox_rgb(ir_raw, self.target_size, geo)
        dep = letterbox_depth(dep_raw, self.target_size, geo)

        item = {
            "visible": image_to_tensor(vis),
            "infrared": image_to_tensor(ir),
            "depth": depth_to_tensor(dep),
            "stem": stem,
            "orig_size": orig_size,
            "letterbox": geo,
        }
        if self.has_labels:
            label = self._read_labels(self.lab_map[stem])
            if label.shape[0] > 0:
                label[:, 1:5] = torch.from_numpy(
                    letterbox_boxes(label[:, 1:5].numpy(), orig_size, self.target_size, geo)
                )
            item["label"] = label
        return item

    @staticmethod
    def _read_labels(path: Path) -> torch.Tensor:
        """读取 YOLO 归一化标签 -> (N, 5) float32: [class_id, cx, cy, w, h]。

        返回原始 [0, 1] 归一化坐标；__getitem__ 中会再套用与图像
        完全一致的 letterbox 几何变换（见 letterbox_boxes）。
        """
        boxes = []
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(x) for x in parts[1:])
            boxes.append([cls, cx, cy, w, h])
        if not boxes:
            return torch.zeros((0, 5), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32)

    def summary(self):
        return {
            "split": self.split,
            "num_samples": len(self.samples),
            "has_labels": self.has_labels,
            "target_size": self.target_size,
        }


def split_train_val(samples: Sequence[str], val_ratio: float = 0.2, seed: int = 42):
    """按比例随机切分 stem 列表 -> (train, val)。

    用于无独立 val 划分的数据版本；若 dataset.yaml 已提供 val，则无需调用。
    """
    arr = np.asarray(sorted(samples))
    perm = np.random.default_rng(seed).permutation(len(arr))
    n_val = int(len(arr) * val_ratio)
    val = {arr[i] for i in perm[:n_val]}
    train = [s for s in arr if s not in val]
    return train, sorted(val)


def collate_fn(batch):
    """DataLoader 自定义 collate：堆叠图像，pad 变长 label。

    直接 torch 默认 collate 会因 label 的框数不一致（如 [11,5] 与 [9,5]）
    而报 "stack expects each tensor to be equal size"，故在此显式处理。

    约定：
    - visible / infrared / depth 堆叠为 (B, C, H, W)。
    - label pad 到 batch 内最大框数，填充值 -1（class_id 与坐标均 -1）；
      另返回 num_labels (B,) 记录每个样本的真实框数，供模型/损失做 mask。
    - stem / orig_size / letterbox 为非张量字段，保持为 list。
    - test 无 label 时，label / num_labels 字段不出现。
    """
    keys = batch[0].keys()
    out = {}
    for k in ("visible", "infrared", "depth"):
        if k in keys:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    for k in ("stem", "orig_size", "letterbox"):
        if k in keys:
            out[k] = [b[k] for b in batch]
    if "label" in keys:
        labels = [b["label"] for b in batch]
        out["num_labels"] = torch.tensor([L.shape[0] for L in labels], dtype=torch.long)
        max_n = max(L.shape[0] for L in labels)
        if max_n > 0:
            padded = torch.full((len(batch), max_n, 5), -1.0, dtype=torch.float32)
            for i, L in enumerate(labels):
                if L.shape[0] > 0:
                    padded[i, : L.shape[0]] = L
            out["label"] = padded
        else:
            out["label"] = torch.full((len(batch), 0, 5), -1.0, dtype=torch.float32)
    return out


def create_dataloaders(
    cfg_path: Union[str, os.PathLike],
    image_size: Sequence[int] = (640, 640),
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """构建 train / val / test 三个 DataLoader。

    - val 优先取 dataset.yaml 中的 ``val`` 划分；若配置无 val，则用
      split_train_val 从 train 按 val_ratio 切分。
    """
    cfg = load_dataset_config(cfg_path)
    target_size = tuple(image_size)  # (H, W)

    train_ds = MultiModalDataset(cfg, split="train", target_size=target_size)
    if "val" in cfg:
        val_ds = MultiModalDataset(cfg, split="val", target_size=target_size)
    else:
        train_stems, val_stems = split_train_val(train_ds.samples, val_ratio, seed)
        val_set = set(val_stems)
        train_idx = [i for i, s in enumerate(train_ds.samples) if s not in val_set]
        val_idx = [i for i, s in enumerate(train_ds.samples) if s in val_set]
        train_ds = Subset(train_ds, train_idx)
        val_ds = Subset(MultiModalDataset(cfg, split="train", target_size=target_size), val_idx)
    test_ds = MultiModalDataset(cfg, split="test", target_size=target_size)

    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
    loaders = {
        name: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(shuffle and name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=(name == "train"),
        )
        for name, ds in datasets.items()
    }
    print(f"[dataset] train={len(datasets['train'])} "
          f"val={len(datasets['val'])} test={len(datasets['test'])}")
    return loaders


if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "dataset.yaml")
    loaders = create_dataloaders(cfg_path, image_size=[640, 640], batch_size=4)
    b = next(iter(loaders["train"]))
    for k, v in b.items():
        if torch.is_tensor(v):
            print(f"{k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"{k}: list len={len(v)}")
