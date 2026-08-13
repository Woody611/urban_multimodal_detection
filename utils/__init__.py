"""utils 包：三模态数据集加载与图像预处理工具。

子模块:
    dataset     — 按 stem 对齐三模态的数据集与 DataLoader
    preprocess  — 模态无关的图像读取 / letterbox / 张量化纯函数

本文件把两个子模块对外的函数集中导出，便于:
    from utils import create_dataloaders, read_rgb, ...
"""

from .dataset import (
    collate_fn,
    create_dataloaders,
    load_dataset_config,
    split_train_val,
    MultiModalDataset,
)
from .preprocess import (
    compute_letterbox,
    depth_to_tensor,
    image_to_tensor,
    letterbox_boxes,
    letterbox_depth,
    letterbox_rgb,
    read_depth,
    read_rgb,
)

__all__ = [
    # dataset
    "load_dataset_config",
    "split_train_val",
    "MultiModalDataset",
    "collate_fn",
    "create_dataloaders",
    # preprocess
    "read_rgb",
    "read_depth",
    "compute_letterbox",
    "letterbox_rgb",
    "letterbox_depth",
    "letterbox_boxes",
    "image_to_tensor",
    "depth_to_tensor",
]
