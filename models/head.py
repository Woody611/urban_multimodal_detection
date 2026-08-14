"""Detection Head：多尺度检测预测层。

在 P3/P4/P5 上为每个尺度预测 cls/reg/obj，返回 Tuple[HeadPredictions, ...]。
类别数 num_classes 由配置传入（不写死）；本模块不做解码 / NMS / loss。

每尺度输出（DecoupledHead 为例，输入 [B, C, H, W]）：
  cls : [B, num_classes, H, W]  分类 logit（未过激活）
  reg : [B, 4, H, W]            (cx, cy, w, h) 回归量，anchor-free
  obj : [B, 1, H, W]            目标性 logit
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .backbone import Conv, DEFAULT_CHANNELS, DEFAULT_STRIDES

__all__ = [
    "HeadPredictions",
    "HeadBranch",
    "DecoupledHead",
    "CoupledHead",
    "build_head",
]

# Head 内部隐藏通道数（各分支共享的中间宽度），与 YOLOX 惯例一致
_DEFAULT_HIDDEN: int = 256
# 每个分支中 3x3 卷积的重复次数
_DEFAULT_NUM_CONVS: int = 2


class HeadPredictions(NamedTuple):
    """单个检测尺度的预测结果。

    Attributes:
        cls: 类别 logit，shape [B, num_classes, H, W]。
        reg: 框回归量，shape [B, 4, H, W]，通道顺序 (cx, cy, w, h)。
        obj: 目标性 logit，shape [B, 1, H, W]。
    """
    cls: torch.Tensor
    reg: torch.Tensor
    obj: torch.Tensor


class HeadBranch(nn.Module):
    """单个预测分支：若干 3x3 卷积 + 一个 1x1 输出卷积。

    输入是共享 stem 之后的特征（通道数已降为 hidden），输出对应通道数
    （分类为 num_classes，回归为 4，目标性为 1）。
    """

    def __init__(self, hidden_ch: int, out_ch: int, num_convs: int = 2):
        super().__init__()
        self.convs = nn.Sequential(
            *(Conv(hidden_ch, hidden_ch, k=3, s=1) for _ in range(num_convs))
        )
        # 最后一层用纯 Conv2d（无 BN/激活），保持 logit 线性输出
        self.pred = nn.Conv2d(hidden_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred(self.convs(x))


class DecoupledHead(nn.Module):
    """解耦检测头（YOLOX 风格）。

    每个尺度共享一个 1x1 stem 将特征降维到 hidden_ch，随后分成三个
    相互独立的分支，分别预测类别 / 框回归 / 目标性。解耦的好处是
    分类与定位任务互不干扰，通常带来更高的精度。

    Args:
        in_channels: 三个尺度输入通道数 [c3, c4, c5]，通常取
            ``neck.out_channels``。
        num_classes: 类别数（不含背景），由配置传入。
        hidden_channels: 各分支共享的隐藏通道数，默认 256。
        num_convs: 每个分支 3x3 卷积的重复次数，默认 2。
        strides: 各尺度下采样倍率 [8, 16, 32]，仅作元信息保存，
            供 loss / 解码使用，Head 本身不做解码。
    """

    def __init__(self, in_channels: Sequence[int], num_classes: int,
                 hidden_channels: int = _DEFAULT_HIDDEN,
                 num_convs: int = _DEFAULT_NUM_CONVS,
                 strides: Sequence[int] = DEFAULT_STRIDES):
        super().__init__()
        self.in_channels = list(in_channels)
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels)
        self.num_convs = int(num_convs)
        self.strides = list(strides)

        if len(self.in_channels) != len(self.strides):
            raise ValueError(
                f"in_channels 与 strides 长度不一致: "
                f"{len(self.in_channels)} vs {len(self.strides)}"
            )

        self.stems = nn.ModuleList()
        self.cls_branches = nn.ModuleList()
        self.reg_branches = nn.ModuleList()
        self.obj_branches = nn.ModuleList()

        for c in self.in_channels:
            self.stems.append(Conv(c, hidden_channels, k=1, s=1))
            self.cls_branches.append(HeadBranch(hidden_channels, num_classes, num_convs))
            self.reg_branches.append(HeadBranch(hidden_channels, 4, num_convs))
            self.obj_branches.append(HeadBranch(hidden_channels, 1, num_convs))

    def forward(self, features: Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        List[torch.Tensor],
    ]) -> Tuple[HeadPredictions, ...]:
        if len(features) != len(self.in_channels):
            raise ValueError(
                f"期望 {len(self.in_channels)} 个尺度特征，收到 {len(features)} 个"
            )

        preds = []
        for feat, stem, cls_branch, reg_branch, obj_branch in zip(
                features, self.stems, self.cls_branches,
                self.reg_branches, self.obj_branches):
            x = stem(feat)
            preds.append(HeadPredictions(
                cls=cls_branch(x),
                reg=reg_branch(x),
                obj=obj_branch(x),
            ))
        return tuple(preds)


class CoupledHead(nn.Module):
    """耦合检测头（YOLOv5 风格）。

    每个尺度用一个 1x1 卷积直接预测 ``num_classes + 5`` 个通道：
    前 4 通道为框回归、第 5 通道为目标性、其余 num_classes 通道为
    类别 logit。为保持与 DecoupledHead 完全一致的输出接口，这里把
    该卷积输出按通道切成 cls / reg / obj 三部分返回。

    Args:
        in_channels: 三个尺度输入通道数 [c3, c4, c5]。
        num_classes: 类别数（不含背景），由配置传入。
        strides: 各尺度下采样倍率，仅作元信息保存。
    """

    def __init__(self, in_channels: Sequence[int], num_classes: int,
                 strides: Sequence[int] = DEFAULT_STRIDES):
        super().__init__()
        self.in_channels = list(in_channels)
        self.num_classes = int(num_classes)
        self.strides = list(strides)

        if len(self.in_channels) != len(self.strides):
            raise ValueError(
                f"in_channels 与 strides 长度不一致: "
                f"{len(self.in_channels)} vs {len(self.strides)}"
            )

        out_ch = num_classes + 5
        self.preds = nn.ModuleList(
            nn.Conv2d(c, out_ch, 1) for c in self.in_channels
        )

    def forward(self, features: Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        List[torch.Tensor],
    ]) -> Tuple[HeadPredictions, ...]:
        if len(features) != len(self.in_channels):
            raise ValueError(
                f"期望 {len(self.in_channels)} 个尺度特征，收到 {len(features)} 个"
            )

        preds = []
        for feat, pred in zip(features, self.preds):
            y = pred(feat)                    # [B, num_classes + 5, H, W]
            reg = y[:, :4]                    # 框回归 (cx, cy, w, h)
            obj = y[:, 4:5]                   # 目标性
            cls = y[:, 5:]                    # 类别 logit
            preds.append(HeadPredictions(cls=cls, reg=reg, obj=obj))
        return tuple(preds)


def build_head(head_cfg: Optional[Dict] = None,
               in_channels: Optional[Sequence[int]] = None,
               num_classes: Optional[int] = None) -> nn.Module:
    """根据 configs/model.yaml 的 detector_head 配置构建检测头。

    Args:
        head_cfg: model.yaml 中 ``detector_head`` 子字典。为 None 时使用默认配置。
        in_channels: 三个尺度输入通道数 [c3, c4, c5]，通常传
            ``neck.out_channels``。为 None 时使用基准通道 [256, 512, 1024]
            （仅供独立自检；实际使用时务必与 Neck 输出对齐）。
        num_classes: 类别数，通常传 ``model.yaml`` 的 ``num_classes``。

    说明:
        扩展为 fusion 模型时，把融合后的多尺度特征通道列表传入
        in_channels，并传入相同的 num_classes，即可复用本 Head。
    """
    cfg = head_cfg or {}
    architecture = cfg.get("architecture", "DecoupledHead")
    strides = cfg.get("strides", list(DEFAULT_STRIDES))
    num_scales = cfg.get("num_scales", len(strides))

    if in_channels is None:
        in_channels = DEFAULT_CHANNELS

    if num_classes is None:
        raise ValueError(
            "num_classes 必须由模型配置传入（configs/model.yaml 中为 12），"
            "Head 不写死类别数。"
        )

    if num_scales != len(in_channels):
        raise ValueError(
            f"detector_head.num_scales({num_scales}) 与输入尺度数 "
            f"({len(in_channels)}) 不一致。"
        )

    if architecture == "DecoupledHead":
        return DecoupledHead(
            in_channels,
            num_classes=num_classes,
            hidden_channels=cfg.get("hidden_channels", _DEFAULT_HIDDEN),
            num_convs=cfg.get("num_convs", _DEFAULT_NUM_CONVS),
            strides=strides,
        )

    if architecture == "CoupledHead":
        return CoupledHead(in_channels, num_classes=num_classes, strides=strides)

    raise NotImplementedError(
        f"Unsupported detector head architecture: {architecture!r}. "
        f"Currently 'DecoupledHead' and 'CoupledHead' are implemented."
    )


if __name__ == "__main__":
    # 快速自检：backbone -> neck -> head 完整前向，验证各尺度输出形状
    from .backbone import build_backbone
    from .neck import build_neck

    backbone = build_backbone(in_channels=3)
    neck = build_neck(in_channels=backbone.out_channels)
    head = build_head(in_channels=neck.out_channels, num_classes=12)

    dummy = torch.randn(2, 3, 640, 640)
    feats = neck(backbone(dummy))
    preds = head(feats)

    for name, pred in zip(("P3", "P4", "P5"), preds):
        print(f"{name}: cls={tuple(pred.cls.shape)} "
              f"reg={tuple(pred.reg.shape)} obj={tuple(pred.obj.shape)}")
    print("num_classes:", head.num_classes)
    print("strides:", head.strides)
