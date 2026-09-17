"""scripts/verify_ir_pretrained_remap.py — IR 早期融合首层权重迁移验证（最终冲刺 · 第二任务）。

职责：不训练，只验证「5ch 早期融合模型的首个 Conv 权重」是否真的从 YOLO11m RGB
预训练迁移，而非整个 5ch 卷积随机初始化。

完全复刻 detect/train.py::get_model 的加载路径：
    model = DetectionModel(yaml)
    weights, _ = attempt_load_one_weight('yolo11m.pt')
    model.load(weights)                      # intersect_dicts：跳过 5ch stem（形状不匹配）
    _transfer_rgb_pretrained(model, weights) # 新增分支：修复 5ch stem

断言：
    channel 0-2 == 预训练 RGB 权重（model.0.conv.weight）
    channel 3-4 == mean(R,G,B) 重复到 2 通道
"""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
from ultralytics.models.yolo.detect.train import _transfer_rgb_pretrained


def main():
    yaml_path = "configs/yolo11m_earlyfusion.yaml"

    # 1) 实例化 5ch 早期融合模型
    model = DetectionModel(yaml_path, nc=12)
    stem = model.model[0].conv.weight  # [64, 5, 3, 3]
    print("=" * 64)
    print("[1] 模型实例化")
    print(f"    ch           = {model.yaml['ch']}")
    print(f"    scale        = {model.yaml.get('scale')}")
    print(f"    first Conv   = in_channels={model.model[0].conv.in_channels}, "
          f"out_channels={model.model[0].conv.out_channels}")
    print(f"    stem shape   = {tuple(stem.shape)}")
    print(f"    params       = {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 2) 加载源单分支 YOLO11m（3ch）
    src_model, _ = attempt_load_one_weight("yolo11m.pt")
    src = src_model.state_dict()
    w_src = src["model.0.conv.weight"]  # [64, 3, 3, 3]
    print("=" * 64)
    print(f"[2] 源 YOLO11m 预训练首层 = {tuple(w_src.shape)} (in_channels=3)")

    # 3) BEFORE：init 后（= load 跳过后的状态，二者相同，因为 5ch weight 被 intersect_dicts 跳过）
    w_before = stem.detach().clone()
    diff_before_rgb = (w_before[:, :3] - w_src).abs().max().item()
    print("=" * 64)
    print("[3] 迁移前（随机初始化）")
    print(f"    |stem[:,:3] - pretrained_RGB|_max = {diff_before_rgb:.4f}  "
          f"(期望 > 0，即尚未迁移)")

    # 4) 复刻 get_model 的加载路径
    model.load(src_model)  # intersect_dicts：跳过 5ch stem
    n_remap = _transfer_rgb_pretrained(model, src_model)  # 新增分支
    w_after = model.model[0].conv.weight.detach()

    # 5) AFTER：逐通道核对
    mean_w = w_src.mean(dim=1, keepdim=True)  # [64,1,3,3]
    rgb_ok = torch.equal(w_after[:, :3], w_src)
    aux_ok = torch.equal(w_after[:, 3:], mean_w.repeat(1, 2, 1, 1))
    diff_after_rgb = (w_after[:, :3] - w_src).abs().max().item()
    diff_after_aux = (w_after[:, 3:] - mean_w).abs().max().item()

    print("=" * 64)
    print(f"[4] 迁移后（_transfer_rgb_pretrained 返回 n_remap={n_remap}）")
    print(f"    channel 0-2 == pretrained RGB ? {rgb_ok}   (|Δ|_max={diff_after_rgb:.2e})")
    print(f"    channel 3-4 == mean(R,G,B)    ? {aux_ok}   (|Δ|_max={diff_after_aux:.2e})")
    print("=" * 64)

    # 6) 结论
    if rgb_ok and aux_ok:
        print("✅ 验证通过：首层 5ch 权重 = RGB 预训练 + IR/D mean 初始化，非随机。")
        return 0
    print("❌ 验证失败：首层权重迁移不符合预期。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
