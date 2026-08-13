"""数据预处理工具：读取、letterbox 缩放、张量化。

本模块提供模态无关的纯函数，供 Dataset 与推理共用。约定：

- 尺寸一律用 (H, W) 表示。
- visible / infrared 统一读取为 3 通道 uint8。
- depth PNG 读取为单通道 uint16（原始深度，单位 mm）。
- depth JPG 原始为 3 通道 uint8（三通道内容一致，已核实），
  读取后转成单通道 uint8 灰度。
- 缩放采用 letterbox（等比例缩放 + 居中 padding），保持纵横比；
  三种模态必须使用同一份几何参数（compute_letterbox 只算一次）。
- depth 用最近邻缩放、padding 填 0，避免引入插值深度值。
- depth 不做任何「灰度值→毫米 / 归一化 / 量化」换算
  （其 physical_scale 未知，见 configs/dataset.yaml 的 depth.jpg.physical_scale）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def read_rgb(path) -> np.ndarray:
    """读取可见光/红外为 3 通道 uint8，shape (H, W, 3)。

    适配 .png / .jpg，内部统一 convert("RGB") 保证 3 通道。
    """
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return arr


def read_depth(path) -> np.ndarray:
    """读取深度图为单通道。

    - .png / 其它：单通道 uint16（原始深度，单位 mm）。
    - .jpg / .jpeg：原始 3 通道 uint8（三通道内容一致），
      转换为单通道 uint8 灰度，不进行任何尺度换算。
    """
    p = Path(path)
    with Image.open(p) as im:
        if p.suffix.lower() in (".jpg", ".jpeg"):
            arr = np.asarray(im.convert("L"), dtype=np.uint8)
        else:
            arr = np.asarray(im)
    return arr


def compute_letterbox(orig_size, target_size):
    """计算等比例缩放 + 居中 padding 的 letterbox 几何参数。

    Args:
        orig_size: (H, W) 原始尺寸。
        target_size: (H, W) 目标尺寸。
    Returns:
        dict:
            scale   : 缩放比（浮点，min 边长比，因此不会拉伸）
            new_size: 缩放后尺寸 (new_H, new_W)，已取整
            pad     : (pad_left, pad_top, pad_right, pad_bottom)
    """
    H_orig, W_orig = orig_size
    H_t, W_t = target_size
    r = min(W_t / W_orig, H_t / H_orig)
    new_W = round(W_orig * r)
    new_H = round(H_orig * r)
    pad_w = W_t - new_W
    pad_h = H_t - new_H
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    return {
        "scale": r,
        "new_size": (new_H, new_W),
        "pad": (pad_left, pad_top, pad_right, pad_bottom),
    }


def letterbox_rgb(img: np.ndarray, target_size, geometry=None) -> np.ndarray:
    """RGB 图像 letterbox：等比例缩放（双线性）+ 0 padding，返回 uint8 (H_t, W_t, 3)。"""
    H_orig, W_orig = img.shape[:2]
    geo = geometry if geometry is not None else compute_letterbox((H_orig, W_orig), target_size)
    new_H, new_W = geo["new_size"]
    pl, pt, pr, pb = geo["pad"]

    im = Image.fromarray(np.asarray(img), mode="RGB")
    im = im.resize((new_W, new_H), Image.Resampling.BILINEAR)
    arr = np.asarray(im, dtype=np.uint8)
    return np.pad(arr, ((pt, pb), (pl, pr), (0, 0)), mode="constant", constant_values=0)


def letterbox_depth(depth: np.ndarray, target_size, geometry=None) -> np.ndarray:
    """深度图 letterbox：等比例缩放（最近邻）+ 0 padding。

    最近邻不引入新的插值深度值；padding 填 0（对应“无深度/无效”哨兵值）。
    返回单通道数组，dtype 与输入一致（uint16 或 uint8）。
    """
    H_orig, W_orig = depth.shape[:2]
    geo = geometry if geometry is not None else compute_letterbox((H_orig, W_orig), target_size)
    new_H, new_W = geo["new_size"]
    pl, pt, pr, pb = geo["pad"]

    im = Image.fromarray(np.asarray(depth))
    im = im.resize((new_W, new_H), Image.Resampling.NEAREST)
    arr = np.asarray(im)
    return np.pad(arr, ((pt, pb), (pl, pr)), mode="constant", constant_values=0)


def letterbox_boxes(boxes, orig_size, target_size, geometry=None):
    """将归一化 [cx, cy, w, h] 同步应用 letterbox 几何变换。

    变换与 letterbox_rgb / letterbox_depth 使用同一份几何参数，
    保证 label 与图像严格对齐。

    Args:
        boxes: (N, 4) float32，归一化中心点 + 宽高，相对原始尺寸。
        orig_size: (H, W) 原始图像尺寸。
        target_size: (H, W) 目标尺寸。
        geometry: compute_letterbox 结果（可选，避免重复计算）。
    Returns:
        (N, 4) float32，变换后仍为归一化坐标（相对 target_size）。
    """
    H_t, W_t = target_size
    geo = geometry if geometry is not None else compute_letterbox(orig_size, target_size)
    new_H, new_W = geo["new_size"]
    pl, pt, _, _ = geo["pad"]

    out = np.asarray(boxes, dtype=np.float32).copy()
    out[:, 0] = (out[:, 0] * new_W + pl) / W_t  # cx
    out[:, 1] = (out[:, 1] * new_H + pt) / H_t  # cy
    out[:, 2] = out[:, 2] * new_W / W_t         # w
    out[:, 3] = out[:, 3] * new_H / H_t         # h
    return out


def image_to_tensor(img: np.ndarray, normalize: bool = True) -> torch.Tensor:
    """(H, W, 3) uint8 -> (3, H, W) float32，可选归一化到 [0, 1]。

    用 torch.tensor 复制数据，避免 PIL 返回的只读数组触发
    "non-writable tensors" 警告。
    """
    t = torch.tensor(np.asarray(img)).permute(2, 0, 1).contiguous().float()
    if normalize:
        t = t / 255.0
    return t


def depth_to_tensor(depth: np.ndarray) -> torch.Tensor:
    """(H, W) -> (1, H, W) float32，保留原始数值，不做任何尺度换算。"""
    t = torch.tensor(np.asarray(depth)).unsqueeze(0).float()
    return t
