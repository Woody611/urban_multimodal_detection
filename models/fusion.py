"""models/fusion.py — 多模态特征融合模块（纯 PyTorch 实现）。

独立、自包含：不依赖项目内其它 models 模块，也不依赖 ultralytics。

在 YOLOv11 多模态检测中，融合位于「多模态 backbone 输出」与「检测 head 输入」
之间：visible / infrared / depth 三个分支各自经 backbone 编码出多尺度特征
P3 / P4 / P5（stride 8 / 16 / 32），ConcatFusion 把同尺度特征沿通道维拼接后
用 1x1 卷积投影回单模态通道数，使下游 head 保持与单模态完全一致的接口。

约定（由数据侧保证，本模块不做校验）:
    - 三个模态特征已空间对齐（同一 letterbox 几何），各尺度分辨率一致；
    - 各模态 backbone 已把输入通道统一到相同宽度，故同尺度通道数均为 C
      （depth 的 1 通道输入在 backbone stem 内已升到 C）。

模块:
    ConcatFusion          —— 单尺度拼接融合：N 个 [B,C,H,W] → 1 个 [B,C,H,W]。
    MultiScaleConcatFusion—— 多尺度拼接融合：N 个 (P3,P4,P5) → 1 个 (P3,P4,P5)，
                             对应 YOLOv11 backbone ↔ head 的接口。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

__all__ = [
    "ConcatFusion",
    "MultiScaleConcatFusion",
    "build_fusion",
]

Tensor = torch.Tensor


def _project(in_channels: int, out_channels: int,
             norm: bool = True, act: bool = True) -> nn.Module:
    """1x1 卷积投影层：in_channels -> out_channels，可选 BatchNorm2d + SiLU。"""
    layers: List[nn.Module] = [nn.Conv2d(in_channels, out_channels, 1, bias=False)]
    if norm:
        layers.append(nn.BatchNorm2d(out_channels))
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class ConcatFusion(nn.Module):
    """单尺度通道拼接融合（Concat Fusion）。

    把 N 个模态的特征图沿通道维拼接，再用 1x1 卷积投影回单模态通道数，
    输出形状与单个模态输入一致，可直接接回单模态 head。

    Args:
        in_channels: 单模态特征通道数 C。
        num_modalities: 参与融合的模态数（默认 3: visible / infrared / depth）。
        project: 是否用 1x1 卷积投影回 C。True 输出 [B, C, H, W]；
            False 输出 [B, N*C, H, W]。
        norm: 投影层是否加 BatchNorm2d。
        act: 投影层是否加 SiLU。

    Example:
        >>> fusion = ConcatFusion(in_channels=64)   # YOLOv11n P3 通道数
        >>> v, i, d = torch.randn(2, 64, 80, 80), torch.randn(2, 64, 80, 80), torch.randn(2, 64, 80, 80)
        >>> out = fusion(v, i, d)                    # [2, 64, 80, 80]
    """

    def __init__(self, in_channels: int, num_modalities: int = 3,
                 project: bool = True, norm: bool = True, act: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_modalities = int(num_modalities)
        self.project = bool(project)
        self.proj = (
            _project(self.num_modalities * self.in_channels,
                     self.in_channels, norm, act)
            if project else nn.Identity()
        )
        self.out_channels = (
            self.in_channels if project else self.num_modalities * self.in_channels
        )

    def forward(self, *modalities: Tensor) -> Tensor:
        """融合同尺度的 N 个模态特征。

        Args:
            *modalities: N 个 [B, C, H, W] 特征张量，如 visible / infrared / depth。

        Returns:
            融合特征 [B, C, H, W]（project=True）或 [B, N*C, H, W]。
        """
        if len(modalities) != self.num_modalities:
            raise ValueError(
                f"ConcatFusion 期望 {self.num_modalities} 个模态特征，"
                f"收到 {len(modalities)} 个"
            )
        return self.proj(torch.cat(modalities, dim=1))


class MultiScaleConcatFusion(nn.Module):
    """多尺度通道拼接融合（YOLOv11 backbone ↔ head 接口）。

    对 P3 / P4 / P5 每个尺度分别做一次 ConcatFusion，返回同尺度的
    (P3, P4, P5)，可直接送入检测 head（PAN Neck / Detect）。

    Args:
        in_channels: 单模态 backbone 多尺度输出通道 [c3, c4, c5]，
            例如 YOLOv11n 为 [64, 128, 256]。
        num_modalities: 参与融合的模态数（默认 3）。
        project / norm / act: 透传给每个尺度的 ConcatFusion。

    Example:
        >>> fusion = MultiScaleConcatFusion(in_channels=[64, 128, 256])
        >>> # 每个模态一个 (P3, P4, P5) 三元组
        >>> vis = (torch.randn(2, 64, 80, 80), torch.randn(2, 128, 40, 40), torch.randn(2, 256, 20, 20))
        >>> ir  = (torch.randn(2, 64, 80, 80), torch.randn(2, 128, 40, 40), torch.randn(2, 256, 20, 20))
        >>> dep = (torch.randn(2, 64, 80, 80), torch.randn(2, 128, 40, 40), torch.randn(2, 256, 20, 20))
        >>> f3, f4, f5 = fusion(vis, ir, dep)        # 各尺度仍为 [B, C_s, H_s, W_s]
    """

    def __init__(self, in_channels: Sequence[int], num_modalities: int = 3,
                 project: bool = True, norm: bool = True, act: bool = True):
        super().__init__()
        self.in_channels = list(in_channels)
        self.num_modalities = int(num_modalities)
        self.fusions = nn.ModuleList(
            ConcatFusion(c, num_modalities, project, norm, act)
            for c in self.in_channels
        )
        self.out_channels = [f.out_channels for f in self.fusions]

    def forward(self, *modalities: Sequence[Tensor]) -> Tuple[Tensor, ...]:
        """融合 N 个模态的多尺度特征。

        Args:
            *modalities: N 个模态特征，每个为 (P3, P4, P5) 三元组。

        Returns:
            融合后的 (P3, P4, P5)。
        """
        if len(modalities) != self.num_modalities:
            raise ValueError(
                f"MultiScaleConcatFusion 期望 {self.num_modalities} 个模态特征，"
                f"收到 {len(modalities)} 个"
            )
        n_scales = len(self.in_channels)
        return tuple(
            self.fusions[s](*[m[s] for m in modalities]) for s in range(n_scales)
        )


def build_fusion(fusion_cfg: Optional[Dict] = None,
                 in_channels: Optional[Union[int, Sequence[int]]] = None,
                 num_modalities: int = 3) -> nn.Module:
    """根据 configs/model_fusion.yaml 的 fusion_method 配置构建融合模块。

    Args:
        fusion_cfg: ``fusion_method`` 子字典（architecture / position）。
            为 None 时使用默认的 ConcatFusion。
        in_channels: 单模态特征通道数。int 时构建单尺度 ConcatFusion；
            Sequence[int] 时构建多尺度 MultiScaleConcatFusion。
            为 None 时默认 YOLOv11n 的 [64, 128, 256]（仅供独立自检）。
        num_modalities: 参与融合的模态数（默认 3）。
    """
    cfg = fusion_cfg or {}
    architecture = cfg.get("architecture", "ConcatFusion")

    if in_channels is None:
        in_channels = [64, 128, 256]

    if architecture == "ConcatFusion":
        if isinstance(in_channels, int):
            return ConcatFusion(in_channels, num_modalities=num_modalities)
        return MultiScaleConcatFusion(
            list(in_channels), num_modalities=num_modalities,
            project=cfg.get("project", True),
        )

    raise NotImplementedError(
        f"Unsupported fusion architecture: {architecture!r}. "
        f"Currently only 'ConcatFusion' is implemented."
    )


if __name__ == "__main__":
    # 快速自检：单尺度 + 多尺度两种接口的输入输出形状与通道数。
    print("== 单尺度 ConcatFusion (project=True) ==")
    f1 = ConcatFusion(in_channels=64)
    v, i, d = (torch.randn(2, 64, 80, 80) for _ in range(3))
    out = f1(v, i, d)
    print(f"  输入 3 x {tuple(v.shape)} -> 输出 {tuple(out.shape)}, "
          f"out_channels={f1.out_channels}")

    print("== 单尺度 ConcatFusion (project=False) ==")
    f2 = ConcatFusion(in_channels=64, project=False)
    out2 = f2(v, i, d)
    print(f"  输出 {tuple(out2.shape)}, out_channels={f2.out_channels}")

    print("== 多尺度 MultiScaleConcatFusion ==")
    fm = MultiScaleConcatFusion(in_channels=[64, 128, 256])
    vis = (torch.randn(2, 64, 80, 80), torch.randn(2, 128, 40, 40),
           torch.randn(2, 256, 20, 20))
    ir = tuple(torch.randn_like(t) for t in vis)
    dep = tuple(torch.randn_like(t) for t in vis)
    f3, f4, f5 = fm(vis, ir, dep)
    print(f"  输出 P3 {tuple(f3.shape)}, P4 {tuple(f4.shape)}, P5 {tuple(f5.shape)}")
    print(f"  out_channels={fm.out_channels}")
