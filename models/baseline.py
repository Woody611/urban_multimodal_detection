"""检测器：BaselineDetector（visible 单模态）+ FusionDetector（三模态融合）。

两者共享 Neck/Head 与配置，区别在 backbone 分支数与是否插入 Fusion：
  BaselineDetector: visible          → Backbone   → Neck → Head
  FusionDetector:   visible/ir/depth → Backbone×3 → Fusion → Neck → Head
build_model 按 model.yaml 的 model_type 二选一构建。
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import yaml

from .backbone import build_backbone
from .neck import build_neck, build_fusion
from .head import HeadPredictions, build_head

__all__ = [
    "BaselineDetector",
    "FusionDetector",
    "load_model_config",
    "build_baseline",
    "build_fusion_model",
    "build_model",
]

# Baseline 输入通道数（visible 为 RGB 3 通道），仅作缺省值，
# 实际以 configs/model.yaml 的 modality.visible.in_channels 为准。
_DEFAULT_VISIBLE_CHANNELS: int = 3
# Baseline 类别数缺省值，实际以 configs/model.yaml 的 num_classes 为准。
_DEFAULT_NUM_CLASSES: int = 12
# 各模态输入通道数缺省值，与 configs/model.yaml 的 modality 子配置一致。
_DEFAULT_MODALITY_CHANNELS: Dict[str, int] = {
    "visible": 3,
    "infrared": 3,
    "depth": 1,
}


def load_model_config(cfg_path: Union[str, os.PathLike]) -> dict:
    """读取 model.yaml（yaml.safe_load），返回解析后的字典。"""
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class BaselineDetector(nn.Module):
    """visible 单模态检测器：Backbone → Neck → Head。

    Args:
        num_classes: 类别数（不含背景），由配置传入。
        backbone_cfg / neck_cfg / head_cfg: model.yaml 中对应子字典。
        in_channels: visible 输入通道数（默认 3）。
    """

    def __init__(self, num_classes: int = _DEFAULT_NUM_CLASSES,
                 backbone_cfg: Optional[Dict] = None,
                 neck_cfg: Optional[Dict] = None,
                 head_cfg: Optional[Dict] = None,
                 in_channels: int = _DEFAULT_VISIBLE_CHANNELS):
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)

        # 阶段一：Backbone（visible 单模态）
        self.backbone = build_backbone(backbone_cfg, in_channels=self.in_channels)

        # 阶段二：Neck（复用 backbone 输出的多尺度通道）
        self.neck = build_neck(neck_cfg, in_channels=self.backbone.out_channels)

        # 阶段三：Detection Head（复用 neck 输出的多尺度通道）
        self.head = build_head(
            head_cfg,
            in_channels=self.neck.out_channels,
            num_classes=self.num_classes,
        )

        self.strides = list(self.head.strides)

    def forward(self, x: torch.Tensor) -> Tuple[HeadPredictions, ...]:
        """前向传播。

        Args:
            x: visible 图像张量，shape [B, 3, H, W]。

        Returns:
            与检测尺度一一对应的预测结果 ``Tuple[HeadPredictions, ...]``，
            每个 HeadPredictions 含 cls / reg / obj 三个 logit tensor。
        """
        features = self.backbone(x)    # (p3, p4, p5)
        features = self.neck(features)  # (p3_out, p4_out, p5_out)
        return self.head(features)      # (pred_s, pred_m, pred_l)


def build_baseline(model_cfg: Optional[Dict] = None) -> BaselineDetector:
    """按 model.yaml 配置构建 BaselineDetector（类别数与 visible 通道取自配置）。"""
    cfg = model_cfg or {}

    num_classes = cfg.get("num_classes", _DEFAULT_NUM_CLASSES)
    in_channels = (
        cfg.get("modality", {})
        .get("visible", {})
        .get("in_channels", _DEFAULT_VISIBLE_CHANNELS)
    )

    return BaselineDetector(
        num_classes=num_classes,
        backbone_cfg=cfg.get("backbone"),
        neck_cfg=cfg.get("neck"),
        head_cfg=cfg.get("detector_head"),
        in_channels=in_channels,
    )


class FusionDetector(nn.Module):
    """三模态融合检测器：Backbone×3 → Fusion → Neck → Head。

    Args:
        num_classes: 类别数（不含背景），由配置传入。
        modality_channels: 各模态输入通道数，如 {"visible": 3, "infrared": 3,
            "depth": 1}，键的顺序决定 forward 参数顺序。
        backbone_cfg / neck_cfg / head_cfg / fusion_cfg: model.yaml 中对应子字典。
    """

    def __init__(self, num_classes: int = _DEFAULT_NUM_CLASSES,
                 modality_channels: Optional[Dict[str, int]] = None,
                 backbone_cfg: Optional[Dict] = None,
                 neck_cfg: Optional[Dict] = None,
                 head_cfg: Optional[Dict] = None,
                 fusion_cfg: Optional[Dict] = None):
        super().__init__()
        self.num_classes = int(num_classes)
        self.modality_channels = dict(
            modality_channels or _DEFAULT_MODALITY_CHANNELS
        )
        self.modalities = list(self.modality_channels.keys())

        # 阶段一：多模态 Backbone（每模态独立分支）
        self.backbone = build_backbone(
            backbone_cfg, in_channels=self.modality_channels
        )

        # 阶段二：Fusion（三份多尺度特征 → 一份 [c3, c4, c5]）
        self.fusion = build_fusion(
            fusion_cfg,
            in_channels=self.backbone.out_channels,
            num_modalities=len(self.modalities),
        )

        # 阶段三/四：Neck + Head（与 baseline 完全一致，复用配置）
        self.neck = build_neck(neck_cfg, in_channels=self.fusion.out_channels)
        self.head = build_head(
            head_cfg,
            in_channels=self.neck.out_channels,
            num_classes=self.num_classes,
        )

        self.strides = list(self.head.strides)

    def forward(self, visible: torch.Tensor,
                infrared: torch.Tensor,
                depth: torch.Tensor) -> Tuple[HeadPredictions, ...]:
        """前向传播。

        Args:
            visible: 可见光图像，shape [B, 3, H, W]。
            infrared: 红外图像，shape [B, 3, H, W]。
            depth: 深度图，shape [B, 1, H, W]。

        Returns:
            与检测尺度一一对应的 ``Tuple[HeadPredictions, ...]``。
        """
        images = dict(zip(self.modalities, (visible, infrared, depth)))
        feats = self.backbone(images)   # list of (p3, p4, p5)，每模态一个
        fused = self.fusion(feats)      # (p3, p4, p5)
        fused = self.neck(fused)
        return self.head(fused)


def build_fusion_model(model_cfg: Optional[Dict] = None) -> FusionDetector:
    """按 model.yaml 配置构建 FusionDetector（类别数与各模态通道取自配置）。"""
    cfg = model_cfg or {}

    num_classes = cfg.get("num_classes", _DEFAULT_NUM_CLASSES)
    modality_cfg = cfg.get("modality", {})
    modality_channels = {
        name: modality_cfg.get(name, {}).get("in_channels", default_ch)
        for name, default_ch in _DEFAULT_MODALITY_CHANNELS.items()
    }

    return FusionDetector(
        num_classes=num_classes,
        modality_channels=modality_channels,
        backbone_cfg=cfg.get("backbone"),
        neck_cfg=cfg.get("neck"),
        head_cfg=cfg.get("detector_head"),
        fusion_cfg=cfg.get("fusion_method"),
    )


def build_model(model_cfg: Optional[Dict] = None) -> nn.Module:
    """统一入口：按 model.yaml 的 ``model_type`` 构建 baseline 或 fusion 检测器。"""
    cfg = model_cfg or {}
    model_type = cfg.get("model_type", "baseline")

    if model_type == "baseline":
        return build_baseline(cfg)
    if model_type == "fusion":
        return build_fusion_model(cfg)

    raise ValueError(
        f"Unsupported model_type: {model_type!r}. "
        f"Expected 'baseline' or 'fusion'."
    )


if __name__ == "__main__":
    # 快速自检：从 configs/model.yaml 构建 baseline，跑通 backbone → neck → head，
    # 并打印各尺度预测输出形状，验证与配置一致（num_classes=12）。
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "model.yaml"
    )
    cfg = load_model_config(config_path)
    model = build_baseline(cfg)

    dummy = torch.randn(2, model.in_channels, 640, 640)
    preds = model(dummy)

    print("num_classes:", model.num_classes)
    print("strides:", model.strides)
    for name, pred in zip(("P3", "P4", "P5"), preds):
        print(f"{name}: cls={tuple(pred.cls.shape)} "
              f"reg={tuple(pred.reg.shape)} obj={tuple(pred.obj.shape)}")
