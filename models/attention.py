"""models/attention.py — Cross-Modal Attention（跨模态注意力，纯 PyTorch 实现）。

独立、自包含：不依赖项目内其它 models 模块，也不依赖 ultralytics。

针对三模态（visible / infrared / depth）目标检测设计，语义为「非对称交叉注意力」：
    - visible  作为 query（主模态，决定关注什么）；
    - infrared、depth 作为 key / value（辅助模态，提供被关注的信息）；
    - 输出增强后的 visible feature，形状与输入 visible 一致，可直接送入
      YOLOv11 检测头（PAN Neck / Detect）。

与 models/fusion.py 的 ConcatFusion（无参、按通道拼接）互补：
    ConcatFusion         —— 三模态等权拼接，无注意力权重；
    CrossModalAttention  —— visible 自适应地挑选 infrared / depth 中的有用信息。

模块:
    CrossModalAttention          —— 单尺度交叉注意力：3 个 [B,C,H,W] → 1 个 [B,C,H,W]。
    MultiScaleCrossModalAttention—— 多尺度封装：3 个 (P3,P4,P5) → 1 个 (P3,P4,P5)，
                                    对应 YOLOv11 backbone ↔ head 的接口。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

__all__ = [
    "CrossModalAttention",
    "MultiScaleCrossModalAttention",
    "build_cma",
]

Tensor = torch.Tensor


def _resolve_heads(dim: int, num_heads: int) -> int:
    """保证 num_heads 能整除 dim；否则回退到不超过 num_heads 的最大因子。"""
    if dim % num_heads == 0:
        return num_heads
    for h in range(num_heads, 0, -1):
        if dim % h == 0:
            return h
    return 1


class CrossModalAttention(nn.Module):
    """跨模态多头注意力：visible 作 query，infrared + depth 作 key/value。

    把每个空间位置视为一个 token，visible 的每个位置对 infrared / depth 的
    （可下采样的）位置集合做 softmax 注意力，聚合出辅助信息后经残差连接
    加回 visible，得到增强后的 visible 特征。

    Args:
        dim: 特征通道数 C（三个模态同尺度通道一致）。
        num_heads: 注意力头数（默认 8；若不能整除 dim 会自动回退）。
        head_dim: 每头维度，默认 dim // num_heads。
        qk_scale: 缩放因子，默认 head_dim ** -0.5。
        kv_stride: 对 infrared / depth 的空间下采样步长，用于降低 K/V token
            数量、控制 O(Nq * Nk) 的内存开销（P3 大尺度下建议 >= 2）。
        qkv_bias: Q/K/V 投影卷积是否带 bias。
        attn_drop: 注意力矩阵 dropout 概率。
        proj_drop: 输出投影 dropout 概率。

    Example:
        >>> cma = CrossModalAttention(dim=64, kv_stride=2)
        >>> v, i, d = torch.randn(2, 64, 80, 80), torch.randn(2, 64, 80, 80), torch.randn(2, 64, 80, 80)
        >>> out = cma(v, i, d)          # [2, 64, 80, 80]，增强后的 visible
    """

    def __init__(self, dim: int, num_heads: int = 8,
                 head_dim: Optional[int] = None, qk_scale: Optional[float] = None,
                 kv_stride: int = 1, qkv_bias: bool = False,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = _resolve_heads(self.dim, int(num_heads))
        self.head_dim = int(head_dim) if head_dim is not None else self.dim // self.num_heads
        self.scale = float(qk_scale) if qk_scale is not None else self.head_dim ** -0.5
        self.kv_stride = int(kv_stride)

        # Q 由 visible 投影；K/V 由 [infrared; depth] 拼接后投影（两种辅助模态联合提供 K/V）
        self.to_q = nn.Conv2d(self.dim, self.num_heads * self.head_dim, 1, bias=qkv_bias)
        self.to_k = nn.Conv2d(self.dim * 2, self.num_heads * self.head_dim, 1, bias=qkv_bias)
        self.to_v = nn.Conv2d(self.dim * 2, self.num_heads * self.head_dim, 1, bias=qkv_bias)

        self.proj = nn.Conv2d(self.num_heads * self.head_dim, self.dim, 1)
        self.kv_pool = nn.AvgPool2d(self.kv_stride) if self.kv_stride > 1 else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout2d(proj_drop)

        self.out_channels = self.dim

    def forward(self, visible: Tensor, infrared: Tensor, depth: Tensor) -> Tensor:
        """以 visible 为 query，融合 infrared / depth 的辅助信息。

        Args:
            visible:  [B, C, H, W]
            infrared: [B, C, H, W]
            depth:    [B, C, H, W]

        Returns:
            增强后的 visible 特征 [B, C, H, W]。
        """
        B, C, H, W = visible.shape
        Nq = H * W
        h, d = self.num_heads, self.head_dim

        # Q: visible -> [B, h, Nq, d]
        q = (self.to_q(visible)
             .reshape(B, Nq, h, d).permute(0, 2, 1, 3))

        # 辅助模态：下采样后沿通道拼接，再投影 K / V
        aux = torch.cat([self.kv_pool(infrared), self.kv_pool(depth)], dim=1)  # [B, 2C, Hk, Wk]
        Hk, Wk = aux.shape[-2:]
        Nk = Hk * Wk
        k = (self.to_k(aux)
             .reshape(B, Nk, h, d).permute(0, 2, 1, 3))  # [B, h, Nk, d]
        v = (self.to_v(aux)
             .reshape(B, Nk, h, d).permute(0, 2, 1, 3))  # [B, h, Nk, d]

        # 交叉注意力: visible 每个位置 -> 辅助模态位置集合
        attn = (q @ k.transpose(-2, -1)) * self.scale     # [B, h, Nq, Nk]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                     # [B, h, Nq, d]
        out = (out.transpose(1, 2)
               .reshape(B, Nq, self.num_heads * self.head_dim)
               .permute(0, 2, 1)
               .reshape(B, C, H, W))

        return visible + self.proj_drop(self.proj(out))


class MultiScaleCrossModalAttention(nn.Module):
    """多尺度跨模态注意力（YOLOv11 backbone ↔ head 接口）。

    对 P3 / P4 / P5 每个尺度分别做一次 CrossModalAttention，返回增强后的
    (P3, P4, P5) visible 特征，可直接送入检测 head。

    Args:
        in_channels: 单模态 backbone 多尺度输出通道 [c3, c4, c5]，
            例如 YOLOv11n 为 [64, 128, 256]。
        num_heads / head_dim / qk_scale / qkv_bias / attn_drop / proj_drop:
            透传给每个尺度的 CrossModalAttention。
        kv_strides: 各尺度 K/V 下采样步长；int 时所有尺度一致，
            Sequence[int] 时逐尺度指定（P3 建议较大以控制内存）。

    Example:
        >>> cma = MultiScaleCrossModalAttention(in_channels=[64, 128, 256], kv_strides=[2, 1, 1])
        >>> vis = (torch.randn(2, 64, 80, 80), torch.randn(2, 128, 40, 40), torch.randn(2, 256, 20, 20))
        >>> ir  = tuple(torch.randn_like(t) for t in vis)
        >>> dep = tuple(torch.randn_like(t) for t in vis)
        >>> f3, f4, f5 = cma(vis, ir, dep)      # 各尺度仍为 [B, C_s, H_s, W_s]
    """

    def __init__(self, in_channels: Sequence[int], num_heads: int = 8,
                 head_dim: Optional[int] = None, qk_scale: Optional[float] = None,
                 kv_strides: Union[int, Sequence[int]] = 1,
                 qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.in_channels = list(in_channels)
        if isinstance(kv_strides, int):
            kv_strides = [kv_strides] * len(self.in_channels)
        self.kv_strides = list(kv_strides)

        self.attns = nn.ModuleList(
            CrossModalAttention(c, num_heads=num_heads, head_dim=head_dim,
                                qk_scale=qk_scale, kv_stride=s,
                                qkv_bias=qkv_bias, attn_drop=attn_drop,
                                proj_drop=proj_drop)
            for c, s in zip(self.in_channels, self.kv_strides)
        )
        self.out_channels = [a.out_channels for a in self.attns]

    def forward(self, visible: Sequence[Tensor], infrared: Sequence[Tensor],
                depth: Sequence[Tensor]) -> Tuple[Tensor, ...]:
        """对三个模态的多尺度特征做逐尺度交叉注意力。

        Args:
            visible / infrared / depth: 各为 (P3, P4, P5) 三元组。

        Returns:
            增强后的 visible (P3, P4, P5)。
        """
        return tuple(
            self.attns[s](visible[s], infrared[s], depth[s])
            for s in range(len(self.in_channels))
        )


def build_cma(cma_cfg: Optional[Dict] = None,
              in_channels: Optional[Union[int, Sequence[int]]] = None) -> nn.Module:
    """根据配置构建跨模态注意力模块。

    Args:
        cma_cfg: 配置子字典（可含 num_heads / kv_stride(s) 等）。
            为 None 时使用默认配置。
        in_channels: 单模态特征通道数。int 时构建单尺度 CrossModalAttention；
            Sequence[int] 时构建多尺度 MultiScaleCrossModalAttention。
            为 None 时默认 YOLOv11n 的 [64, 128, 256]（仅供独立自检）。
    """
    cfg = cma_cfg or {}
    if in_channels is None:
        in_channels = [64, 128, 256]

    if isinstance(in_channels, int):
        return CrossModalAttention(
            in_channels,
            num_heads=cfg.get("num_heads", 8),
            kv_stride=cfg.get("kv_stride", 1),
        )
    return MultiScaleCrossModalAttention(
        list(in_channels),
        num_heads=cfg.get("num_heads", 8),
        kv_strides=cfg.get("kv_strides", cfg.get("kv_stride", 1)),
    )


if __name__ == "__main__":
    # 快速自检：单尺度 / 下采样 / 多尺度三种接口的输入输出形状。
    print("== 单尺度 CrossModalAttention (kv_stride=1) ==")
    cma1 = CrossModalAttention(dim=64)
    v, i, d = (torch.randn(2, 64, 80, 80) for _ in range(3))
    out = cma1(v, i, d)
    print(f"  3 x {tuple(v.shape)} -> {tuple(out.shape)}, out_channels={cma1.out_channels}")

    print("== 单尺度 CrossModalAttention (kv_stride=2) ==")
    cma2 = CrossModalAttention(dim=64, kv_stride=2)
    out2 = cma2(v, i, d)
    print(f"  K/V 下采样 2x -> {tuple(out2.shape)}")

    print("== 多尺度 MultiScaleCrossModalAttention ==")
    cma_m = MultiScaleCrossModalAttention(in_channels=[64, 128, 256],
                                          kv_strides=[2, 1, 1])
    vis = (torch.randn(2, 64, 80, 80), torch.randn(2, 128, 40, 40),
           torch.randn(2, 256, 20, 20))
    ir = tuple(torch.randn_like(t) for t in vis)
    dep = tuple(torch.randn_like(t) for t in vis)
    f3, f4, f5 = cma_m(vis, ir, dep)
    print(f"  P3 {tuple(f3.shape)}, P4 {tuple(f4.shape)}, P5 {tuple(f5.shape)}")
    print(f"  out_channels={cma_m.out_channels}")
