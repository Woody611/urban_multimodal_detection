"""Neck / Fusion：PAN 多尺度特征融合 + ConcatFusion 多模态拼接融合。

PAN 在 P3/P4/P5 上做 FPN 自顶向下 + PAN 自底向上融合，输出同尺度增强特征；
ConcatFusion 把多模态特征按尺度 concat 后投影回原通道，供 PAN 复用。
两者都按 in_channels 解耦，不写死类别数、不做 loss / 解码。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .backbone import C3, Conv, DEFAULT_CHANNELS, DEFAULT_STRIDES, make_divisible

__all__ = [
    "PAN",
    "ConcatFusion",
    "build_neck",
    "build_fusion",
]

class PAN(nn.Module):
    """Path Aggregation Network 多尺度特征融合 Neck。

    结构分两阶段：
    1. 自顶向下（FPN）：P5 经 1x1 卷积降维后上采样，与 P4 融合；
       再降维上采样与 P3 融合，得到富含语义的高层信息向低层传递。
    2. 自底向上（PAN）：将上述低层特征下采样，依次与对应的高层 lateral
       特征融合，得到空间细节向高层回传的增强特征。

    Args:
        in_channels: 输入三个尺度的通道数 [c3, c4, c5]，通常取自
            backbone.out_channels。
        depth_multiple: 深度缩放因子，作用于每个 C3 融合块内的 Bottleneck
            数量（基准为 3，与 YOLOv5 neck 一致）。
    """

    def __init__(self, in_channels: Sequence[int],
                 depth_multiple: float = 0.33):
        super().__init__()
        c3, c4, c5 = in_channels
        # 每个融合 C3 块的 Bottleneck 重复次数（基准 3）
        n = max(1, round(3 * depth_multiple))

        # ---- 自顶向下（FPN） ----
        # P5 lateral：c5 -> c4
        self.reduce_p5 = Conv(c5, c4, 1, 1)
        # concat(P4, up(reduce_p5)) -> c4
        self.td_p4 = C3(c4 + c4, c4, n=n, shortcut=False)
        # P4 lateral：c4 -> c3
        self.reduce_p4 = Conv(c4, c3, 1, 1)
        # concat(P3, up(reduce_p4)) -> c3  == P3_out
        self.td_p3 = C3(c3 + c3, c3, n=n, shortcut=False)

        # ---- 自底向上（PAN） ----
        # 下采样 P3_out，与 reduce_p4 的 lateral 融合 -> c4  == P4_out
        self.down_p4 = Conv(c3, c3, 3, 2)
        self.bu_p4 = C3(c3 + c3, c4, n=n, shortcut=False)
        # 下采样 P4_out，与 reduce_p5 的 lateral 融合 -> c5  == P5_out
        self.down_p5 = Conv(c4, c4, 3, 2)
        self.bu_p5 = C3(c4 + c4, c5, n=n, shortcut=False)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # 输出各尺度通道数，供 Detection Head 使用
        self.out_channels: List[int] = [c3, c4, c5]
        self.strides: List[int] = list(DEFAULT_STRIDES)

    def forward(self, features: Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        List[torch.Tensor],
    ]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3, p4, p5 = features

        # ---- 自顶向下 ----
        lat5 = self.reduce_p5(p5)                      # c4，lateral（供 PAN 复用）
        t4 = self.td_p4(torch.cat([p4, self.upsample(lat5)], dim=1))  # c4

        lat4 = self.reduce_p4(t4)                      # c3，lateral（供 PAN 复用）
        p3_out = self.td_p3(torch.cat([p3, self.upsample(lat4)], dim=1))  # c3

        # ---- 自底向上 ----
        p4_out = self.bu_p4(torch.cat([self.down_p4(p3_out), lat4], dim=1))  # c4
        p5_out = self.bu_p5(torch.cat([self.down_p5(p4_out), lat5], dim=1))  # c5

        return p3_out, p4_out, p5_out


def build_neck(neck_cfg: Optional[Dict] = None,
               in_channels: Optional[Sequence[int]] = None) -> nn.Module:
    """根据 configs/model.yaml 的 neck 配置构建特征融合 Neck。

    Args:
        neck_cfg: model.yaml 中 ``neck`` 子字典。为 None 时使用默认配置。
        in_channels: 输入三个尺度的通道数 [c3, c4, c5]。通常直接传
            ``backbone.out_channels``。为 None 时按 neck 的 width_multiple
            对基准通道 [256, 512, 1024] 缩放得到默认值。

    说明:
        扩展为 fusion 模型时，把融合后的多模态特征通道列表传入 in_channels
        即可复用本 Neck，无需改动内部实现。
    """
    cfg = neck_cfg or {}
    architecture = cfg.get("architecture", "PAN")
    depth_multiple = cfg.get("depth_multiple", 0.33)

    if in_channels is None:
        width_multiple = cfg.get("width_multiple", 0.50)
        in_channels = [make_divisible(c * width_multiple) for c in DEFAULT_CHANNELS]

    if architecture == "PAN":
        return PAN(in_channels, depth_multiple=depth_multiple)

    raise NotImplementedError(
        f"Unsupported neck architecture: {architecture!r}. "
        f"Currently only 'PAN' is implemented."
    )


class ConcatFusion(nn.Module):
    """通道拼接融合（middle fusion）。

    对每个尺度，把 N 个模态的特征图沿通道维 concat，再用 1x1 卷积投影回
    单模态通道数，得到与单个 backbone 输出一致的 [c3, c4, c5]，使下游
    PAN Neck 可直接复用 baseline 配置。

    Args:
        in_channels: 单个模态 backbone 的多尺度输出通道 [c3, c4, c5]。
        num_modalities: 参与融合的模态数（默认 3）。
        project: 是否用 1x1 卷积把 concat 后的通道投影回原通道数。
            True 时输出 [c3, c4, c5]；False 时输出 [N*c3, N*c4, N*c5]。
    """

    def __init__(self, in_channels: Sequence[int],
                 num_modalities: int = 3, project: bool = True):
        super().__init__()
        self.in_channels = list(in_channels)
        self.num_modalities = int(num_modalities)
        self.project = project

        if project:
            self.projs = nn.ModuleList(
                Conv(self.num_modalities * c, c, k=1, s=1) for c in self.in_channels
            )
        else:
            self.projs = None

        self.out_channels = (
            list(in_channels) if project
            else [self.num_modalities * c for c in in_channels]
        )

    def forward(self, modalities: Sequence[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """融合多个模态的多尺度特征。

        Args:
            modalities: 每模态一个 (p3, p4, p5) 三元组，长度等于
                num_modalities。各模态对应尺度分辨率必须一致（数据侧已通过
                同一 letterbox 几何保证空间对齐）。

        Returns:
            融合后的 (p3, p4, p5)。
        """
        if len(modalities) != self.num_modalities:
            raise ValueError(
                f"期望 {self.num_modalities} 个模态特征，收到 {len(modalities)} 个"
            )

        n_scales = len(self.in_channels)
        fused = []
        for s in range(n_scales):
            cat = torch.cat([m[s] for m in modalities], dim=1)
            fused.append(self.projs[s](cat) if self.project else cat)
        return tuple(fused)


def build_fusion(fusion_cfg: Optional[Dict] = None,
                 in_channels: Optional[Sequence[int]] = None,
                 num_modalities: int = 3) -> nn.Module:
    """根据 configs/model.yaml 的 fusion_method 配置构建融合模块。

    Args:
        fusion_cfg: model.yaml 中 ``fusion_method`` 子字典。为 None 时使用默认配置。
        in_channels: 单个模态 backbone 的多尺度输出通道 [c3, c4, c5]。
            为 None 时使用基准通道 [256, 512, 1024]（仅供独立自检）。
        num_modalities: 参与融合的模态数。
    """
    cfg = fusion_cfg or {}
    architecture = cfg.get("architecture", "ConcatFusion")

    if in_channels is None:
        in_channels = DEFAULT_CHANNELS

    if architecture == "ConcatFusion":
        return ConcatFusion(
            in_channels,
            num_modalities=num_modalities,
            project=cfg.get("project", True),
        )

    raise NotImplementedError(
        f"Unsupported fusion architecture: {architecture!r}. "
        f"Currently only 'ConcatFusion' is implemented."
    )


if __name__ == "__main__":
    # 快速自检：验证输入输出形状与通道数
    from .backbone import build_backbone

    backbone = build_backbone(in_channels=3)
    neck = build_neck(in_channels=backbone.out_channels)

    dummy = torch.randn(2, 3, 640, 640)
    p3, p4, p5 = backbone(dummy)
    print("backbone out:", [tuple(f.shape) for f in (p3, p4, p5)])

    n3, n4, n5 = neck((p3, p4, p5))
    for name, feat in zip(("P3_out", "P4_out", "P5_out"), (n3, n4, n5)):
        print(f"{name}: {tuple(feat.shape)}")
    print("out_channels:", neck.out_channels)
