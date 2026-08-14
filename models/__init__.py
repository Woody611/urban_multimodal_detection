"""models 包：目标检测模型（Backbone / Neck / Fusion / Head / 检测器）。

概念名 → 实际类名：
    Backbone       → CSPDarknet / MultiModalBackbone
    Neck           → PAN
    Fusion         → ConcatFusion
    Detection Head → DecoupledHead / CoupledHead
    检测器          → BaselineDetector / FusionDetector
"""
from .backbone import (
    C3,
    Bottleneck,
    Conv,
    CSPDarknet,
    MultiModalBackbone,
    SPPF,
    build_backbone,
)
from .neck import (
    PAN,
    ConcatFusion,
    build_neck,
    build_fusion,
)
from .head import (
    CoupledHead,
    DecoupledHead,
    HeadBranch,
    HeadPredictions,
    build_head,
)
from .baseline import (
    BaselineDetector,
    FusionDetector,
    build_baseline,
    build_fusion_model,
    build_model,
    load_model_config,
)

__all__ = [
    # backbone
    "Conv",
    "Bottleneck",
    "C3",
    "SPPF",
    "CSPDarknet",
    "MultiModalBackbone",
    "build_backbone",
    # neck
    "PAN",
    "ConcatFusion",
    "build_neck",
    "build_fusion",
    # head
    "HeadPredictions",
    "HeadBranch",
    "DecoupledHead",
    "CoupledHead",
    "build_head",
    # baseline
    "BaselineDetector",
    "FusionDetector",
    "build_baseline",
    "build_fusion_model",
    "build_model",
    "load_model_config",
]
