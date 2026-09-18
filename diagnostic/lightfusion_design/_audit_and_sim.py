"""diagnostic/lightfusion_design/_audit_and_sim.py — 轻量融合设计审计 + 预训练加载模拟（只读，不训练）。

内容：
  1. 当前 RGBID early-fusion 第一层是什么 / in_channels / 参数量 / FLOPs
  2.  标准 load() + _transfer_rgb_pretrained() 的实际加载结果（transferred / missing / unexpected）
  3. Candidate A（5->3 1x1 projection + 原 3ch YOLO11m）的同样审计
  4. 两者 dummy forward（B=1, 1280）与参数量对比

不训练、不修改任何既有文件。临时 yaml 写在 diagnostic/lightfusion_design/ 下。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, intersect_dicts  # noqa: E402
from ultralytics.models.yolo.detect.train import _transfer_rgb_pretrained                 # noqa: E402

PRETRAINED = ROOT / "yolo11m.pt"
NC = 12
BASELINE_YAML = ROOT / "configs" / "yolo11m_earlyfusion.yaml"
CAND_A_YAML = Path(__file__).resolve().parent / "yolo11m_candidateA_5to3.yaml"

CAND_A = """# 临时审计用 yaml（Candidate A：5->3 1x1 projection + 原 3ch YOLO11m）
# 在 baseline 的 backbone 前插入一层 Conv(5,3,1,1)，其余所有层的绝对索引 +1。
ch: 5
nc: 12
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]
backbone:
  - [-1, 1, Conv, [3, 1, 1]]   # 0  <-- 新增：5 -> 3，1x1，stride 1
  - [-1, 1, Conv, [64, 3, 2]]  # 1
  - [-1, 1, Conv, [128, 3, 2]] # 2
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 7], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 5], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]
  - [[17, 20, 23], 1, Detect, [nc]]
"""


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def build_and_report(tag, cfg, ch):
    print("=" * 78)
    print(f"[{tag}]  cfg={cfg}")
    print("=" * 78)
    model = DetectionModel(str(cfg), ch=ch, nc=NC, verbose=False)
    core = model.model
    first = core[0]
    print(f"  模型顶层模块数 = {len(core)}")
    print(f"  第 0 层: {type(first).__name__}")
    print(f"     conv.in_channels  = {first.conv.in_channels}")
    print(f"     conv.out_channels = {first.conv.out_channels}")
    print(f"     k / s / p         = {first.conv.kernel_size} / {first.conv.stride} / {first.conv.padding}")
    print(f"     该层参数 = {n_params(first):,}")
    print(f"     该层权重 shape = {tuple(first.conv.weight.shape)}")
    print(f"  总参数 = {n_params(model):,}")
    return model


def simulate_load(tag, model, note=""):
    """复刻 trainer.get_model() 的加载路径：load() + _transfer_rgb_pretrained()。"""
    print(f"\n  --- [{tag}] 预训练加载模拟 {note} ---")
    src, _ = attempt_load_one_weight(str(PRETRAINED))
    src_sd = src.float().state_dict()

    # 标准 load() 路径
    before = model.state_dict()
    inter = intersect_dicts(src_sd, before)
    n_missing = len(before) - len(inter)
    src_keys = set(src_sd)
    tgt_keys = set(before)
    unexpected = sorted(src_keys - tgt_keys)
    print(f"    标准 intersect_dicts: 匹配 {len(inter)}/{len(before)}  缺失 {n_missing}")
    print(f"    整网 unexpected(源有目标无) 数 = {len(unexpected)}")
    stem_key = "model.0.conv.weight"
    print(f"    源 {stem_key} 是否被 intersect 匹配: {stem_key in inter}")
    if stem_key in src_sd:
        print(f"       源 shape = {tuple(src_sd[stem_key].shape)}   目标 shape = {tuple(before[stem_key].shape)}")

    # 完整路径
    model.load(src)
    n = _transfer_rgb_pretrained(model, src)
    print(f"    _transfer_rgb_pretrained 返回 = {n}")

    # 校验第一层是否真被赋上
    w = model.model[0].conv.weight.data
    src_w = src_sd.get(stem_key)
    if w.shape[1] == 5 and src_w is not None and src_w.shape[0] == w.shape[0]:
        ok_rgb = torch.equal(w[:, :3], src_w)
        exp_aux = src_w.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
        ok_aux = torch.allclose(w[:, 3:], exp_aux)
        print(f"    stem[:, :3] == 预训练 RGB 权重 : {ok_rgb}")
        print(f"    stem[:, 3:] == mean(R,G,B)     : {ok_aux}")
        print(f"    -> 5ch stem 已用预训练 RGB + 均值初始化 IR/D: {ok_rgb and ok_aux}")
    else:
        print(f"    !! 第一层 shape={tuple(w.shape)}，与源 {stem_key}{tuple(src_w.shape) if src_w is not None else None} "
              f"输出通道不匹配 -> _transfer_rgb_pretrained 无法为它赋预训练权重")
        # 该层是否被标准 load() 赋上任何值？
        li = model.model[0]
        w0 = li.conv.weight.data
        print(f"    第一层权重是否仍为随机初始化: {bool(w0.std().item() != 0)}  (std={w0.std().item():.5f})")
    return n


def flops_of(model, ch, imgsz=1280):
    try:
        from thop import profile
    except Exception:
        return None
    x = torch.zeros(1, ch, imgsz, imgsz)
    model.eval()
    with torch.no_grad():
        macs, _ = profile(model, inputs=(x,), verbose=False)
    return macs * 2  # FLOPs


def dummy_forward(tag, model, ch, imgsz=1280, bs=1):
    print(f"\n  --- [{tag}] dummy forward B={bs} @ {ch}x{imgsz}x{imgsz} ---")
    model.eval()
    x = torch.rand(bs, ch, imgsz, imgsz)
    with torch.no_grad():
        y = model(x)
    out = y[0] if isinstance(y, (list, tuple)) else y
    bad = 0 if torch.isfinite(out).all() else int((~torch.isfinite(out)).sum())
    print(f"    forward PASS, output shape = {tuple(out.shape)}, NaN/Inf = {bad}")
    return out


def main():
    print("\n########## PART 1: 当前 RGBID early-fusion baseline ##########")
    base = build_and_report("BASELINE (RGBID early fusion)", BASELINE_YAML, ch=5)
    p_base = n_params(base)
    simulate_load("BASELINE", base)
    f_base = flops_of(base, 5)
    print(f"    FLOPs @1280 = {f_base/1e9:.2f} G" if f_base else "    (thop 不可用，跳过 FLOPs)")
    dummy_forward("BASELINE", base, ch=5)

    print("\n\n########## PART 2: Candidate A (5->3 1x1 + 原 3ch YOLO11m) ##########")
    CAND_A_YAML.write_text(CAND_A, encoding="utf-8")
    cand = build_and_report("CANDIDATE A", CAND_A_YAML, ch=5)
    p_cand = n_params(cand)
    simulate_load("CANDIDATE A", cand, note="(注意：新增层导致全网络 key 位移)")
    f_cand = flops_of(cand, 5)
    print(f"    FLOPs @1280 = {f_cand/1e9:.2f} G" if f_cand else "    (thop 不可用，跳过 FLOPs)")
    dummy_forward("CANDIDATE A", cand, ch=5)

    print("\n\n########## PART 3: 参数量对比 ##########")
    print(f"  Baseline     总参数 = {p_base:,}")
    print(f"  Candidate A  总参数 = {p_cand:,}")
    print(f"  Δ = {p_cand - p_base:+,}    ({(p_cand-p_base)/p_base*100:+.4f}%)")
    print(f"  1% 门限 = {p_base*0.01:,.0f}  ->  {'PASS' if abs(p_cand-p_base) < p_base*0.01 else 'FAIL'}")


if __name__ == "__main__":
    main()
