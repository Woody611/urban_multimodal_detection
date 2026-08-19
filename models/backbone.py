"""Backbone：CSPDarknet 多尺度特征编码器。

CSPDarknet（单模态）与 MultiModalBackbone（多模态，每模态一个分支），
输出三个尺度特征 P3 / P4 / P5（stride 8 / 16 / 32）供 Neck / Fusion 使用。
深度、宽度由 depth_multiple / width_multiple 缩放，不写死类别数。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

__all__ = [
    "Conv",
    "Bottleneck",
    "C3",
    "SPPF",
    "CSPDarknet",
    "MultiModalBackbone",
    "build_backbone",
]

# variant -> (depth_multiple, width_multiple)，与 YOLOv5 缩放惯例一致
VARIANT_SCALES: Dict[str, Tuple[float, float]] = {
    "n": (0.33, 0.25),
    "s": (0.33, 0.50),
    "m": (0.67, 0.75),
    "l": (1.00, 1.00),
    "x": (1.33, 1.25),
}

# 各 stage 的基准通道数（未缩放），依次对应 stem 及 stage1~4 输出
_BASE_CHANNELS: List[int] = [64, 128, 256, 512, 1024]
# 各 stage（stage1~4）中 C3 的 Bottleneck 重复次数（未缩放）
_BASE_DEPTHS: List[int] = [3, 6, 9, 3]

# P3/P4/P5 各尺度：基准通道（未缩放）与下采样倍率，Neck / Head / Fusion 复用
DEFAULT_CHANNELS: List[int] = _BASE_CHANNELS[2:]   # [256, 512, 1024]
DEFAULT_STRIDES: Tuple[int, int, int] = (8, 16, 32)


def autopad(k: int, p: Optional[int] = None) -> int:
    """卷积 same-padding：k//2（奇数核）保证输入输出尺寸不变。"""
    if p is None:
        p = k // 2
    return p


def make_divisible(x: float, divisor: int = 8) -> int:
    """把通道数向上取整到 divisor 的倍数，保证硬件友好。"""
    return int(round(x / divisor) * divisor)


class Conv(nn.Module):
    """Conv2d + BatchNorm2d + SiLU 的标准卷积块。"""

    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1,
                 p: Optional[int] = None, g: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """标准残差瓶颈：1x1 降维 -> 3x3 卷积 -> 残差连接。"""

    def __init__(self, in_ch: int, out_ch: int, shortcut: bool = True,
                 g: int = 1, e: float = 0.5):
        super().__init__()
        hidden = int(out_ch * e)
        self.cv1 = Conv(in_ch, hidden, 1, 1)
        self.cv2 = Conv(hidden, out_ch, 3, 1, g=g)
        self.add = shortcut and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C3(nn.Module):
    """CSP 瓶颈块：输入经两个 1x1 卷积分支，一支走 n 个 Bottleneck，
    另一支直连，concat 后再 1x1 融合。"""

    def __init__(self, in_ch: int, out_ch: int, n: int = 1,
                 shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        hidden = int(out_ch * e)
        self.cv1 = Conv(in_ch, hidden, 1, 1)
        self.cv2 = Conv(in_ch, hidden, 1, 1)
        self.cv3 = Conv(2 * hidden, out_ch, 1, 1)
        self.m = nn.Sequential(
            *(Bottleneck(hidden, hidden, shortcut, g, e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast：级联三个 5x5 最大池化并 concat，
    在不损失分辨率的前提下增大感受野。"""

    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        hidden = in_ch // 2
        self.cv1 = Conv(in_ch, hidden, 1, 1)
        self.cv2 = Conv(hidden * 4, out_ch, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


class CSPDarknet(nn.Module):
    """CSPDarknet 主干网络。

    输出三个尺度特征图，对应 stride 8 / 16 / 32 的 P3 / P4 / P5，
    通道数分别为 width_multiple * [256, 512, 1024]（再取整到 8 的倍数）。

    Args:
        in_channels: 输入通道数。visible / infrared 为 3，depth 为 1。
        depth_multiple: 深度缩放因子，作用于各 stage 的 Bottleneck 数量。
        width_multiple: 宽度缩放因子，作用于各 stage 的通道数。
        use_sppf: 是否在 P5 末端附加 SPPF 增大感受野。
    """

    def __init__(self, in_channels: int = 3, depth_multiple: float = 0.33,
                 width_multiple: float = 0.50, use_sppf: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple

        ch = [make_divisible(c * width_multiple) for c in _BASE_CHANNELS]
        depth = [max(1, round(n * depth_multiple)) for n in _BASE_DEPTHS]

        # stem: 6x6 下采样卷积（stride 2）
        self.stem = Conv(in_channels, ch[0], k=6, s=2, p=2)

        # stage1: stride 4
        self.stage1 = nn.Sequential(
            Conv(ch[0], ch[1], k=3, s=2),
            C3(ch[1], ch[1], n=depth[0]),
        )
        # stage2: stride 8 -> P3
        self.stage2 = nn.Sequential(
            Conv(ch[1], ch[2], k=3, s=2),
            C3(ch[2], ch[2], n=depth[1]),
        )
        # stage3: stride 16 -> P4
        self.stage3 = nn.Sequential(
            Conv(ch[2], ch[3], k=3, s=2),
            C3(ch[3], ch[3], n=depth[2]),
        )
        # stage4: stride 32 -> P5
        stage4 = [
            Conv(ch[3], ch[4], k=3, s=2),
            C3(ch[4], ch[4], n=depth[3]),
        ]
        if use_sppf:
            stage4.append(SPPF(ch[4], ch[4], k=5))
        self.stage4 = nn.Sequential(*stage4)

        # 输出各尺度通道数，供 Neck 使用
        self.out_channels: List[int] = [ch[2], ch[3], ch[4]]
        self.strides: List[int] = list(DEFAULT_STRIDES)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)   # stride 8
        p4 = self.stage3(p3)  # stride 16
        p5 = self.stage4(p4)  # stride 32
        return p3, p4, p5


class MultiModalBackbone(nn.Module):
    """多模态主干网络：为每种模态实例化一个独立 CSPDarknet 分支。

    各分支结构相同、共享同一 depth / width 缩放，仅输入通道不同
    （visible=3、infrared=3、depth=1）。输出为每模态一个 (P3, P4, P5)
    三元组的列表，供后续 Fusion 使用。

    Args:
        modality_channels: {模态名: 输入通道数}，键的顺序决定分支顺序。
        depth_multiple / width_multiple / use_sppf: 与 CSPDarknet 相同，
            由 build_backbone 解析配置后传入。
    """

    def __init__(self, modality_channels: Dict[str, int],
                 depth_multiple: float = 0.33, width_multiple: float = 0.50,
                 use_sppf: bool = True):
        super().__init__()
        self.modality_channels = dict(modality_channels)
        self.modalities = list(self.modality_channels.keys())

        self.branches = nn.ModuleDict({
            name: CSPDarknet(in_channels=ch,
                             depth_multiple=depth_multiple,
                             width_multiple=width_multiple,
                             use_sppf=use_sppf)
            for name, ch in self.modality_channels.items()
        })

        # 各分支输出通道一致（同一 width_multiple），取第一个分支作参考
        ref = self.branches[self.modalities[0]]
        self.out_channels: List[int] = list(ref.out_channels)
        self.strides: List[int] = list(ref.strides)

    def forward(self, images: Dict[str, torch.Tensor]) -> List[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """前向传播。

        Args:
            images: {模态名: 图像张量 [B, C, H, W]}，键需覆盖 self.modalities。

        Returns:
            每模态一个 (P3, P4, P5) 三元组，顺序与 self.modalities 一致。
        """
        return [self.branches[name](images[name]) for name in self.modalities]


def build_backbone(backbone_cfg: Optional[Dict] = None,
                   in_channels: Union[int, Dict[str, int]] = 3) -> nn.Module:
    """根据 configs/model.yaml 的 backbone 配置构建主干网络。

    Args:
        backbone_cfg: model.yaml 中 ``backbone`` 子字典。为 None 时使用默认配置。
        in_channels:
            - int: 构建单模态 CSPDarknet（baseline，如 visible=3 / depth=1）。
            - dict: 构建多模态 MultiModalBackbone，键为模态名、值为该模态
              输入通道数，如 {"visible": 3, "infrared": 3, "depth": 1}。
    """
    cfg = backbone_cfg or {}
    depth_multiple = cfg.get("depth_multiple")
    width_multiple = cfg.get("width_multiple")

    # 未显式指定缩放因子时，回退到 variant 预设（默认 "s"）
    if depth_multiple is None or width_multiple is None:
        variant = cfg.get("variant", "s")
        d_default, w_default = VARIANT_SCALES.get(variant, VARIANT_SCALES["s"])
        depth_multiple = depth_multiple if depth_multiple is not None else d_default
        width_multiple = width_multiple if width_multiple is not None else w_default

    use_sppf = cfg.get("use_sppf", True)

    if isinstance(in_channels, dict):
        return MultiModalBackbone(
            in_channels, depth_multiple=depth_multiple,
            width_multiple=width_multiple, use_sppf=use_sppf,
        )

    return CSPDarknet(
        in_channels=in_channels,
        depth_multiple=depth_multiple,
        width_multiple=width_multiple,
        use_sppf=use_sppf,
    )


if __name__ == "__main__":
    # 快速自检：验证输入输出形状与通道数
    net = CSPDarknet(in_channels=3, depth_multiple=0.33, width_multiple=0.50)
    dummy = torch.randn(2, 3, 640, 640)
    p3, p4, p5 = net(dummy)
    for name, feat in zip(("P3", "P4", "P5"), (p3, p4, p5)):
        print(f"{name}: {tuple(feat.shape)}  stride={net.strides[('P3', 'P4', 'P5').index(name)]}")
    print("out_channels:", net.out_channels)
