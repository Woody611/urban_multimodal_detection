"""diagnostic/p3attn_probe/_probe_tests.py — P3 Identity Attention Probe 的单元测试与一致性测试。

只读测试：不训练、不写 checkpoint、不修改任何正式产物。
覆盖用户要求的 §2 identity 初始化、§3 权重迁移、§4 冻结参数、§6 初始输出一致性。

用法:  python diagnostic/p3attn_probe/_probe_tests.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ultralytics.nn.tasks as T                                        # noqa: E402
from ultralytics.nn.modules.block import C3k2, P3IdentityAttn           # noqa: E402

BASE_YAML = ROOT / "configs/yolo11m_earlyfusion.yaml"
ATTN_YAML = Path(__file__).resolve().parent / "yolo11m_p3attn_probe.yaml"
BEST_PT = ROOT / "runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt"
YOLO11M = ROOT / "yolo11m.pt"
NC, CH = 12, 5

FAILS: list[str] = []


def ck(name, cond, detail=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def sec(t):
    print("\n" + "=" * 84); print(t); print("=" * 84)


# ---------------------------------------------------------------- §2 identity
def test_identity_attn():
    sec("§2 Identity Attention —— 单元测试")
    torch.manual_seed(0)
    c = 256
    m = P3IdentityAttn(c)
    x = torch.randn(2, c, 40, 40)

    y = m(x)
    d = (y - x).abs().max().item()
    ck("A1 输出 shape 与输入完全一致", tuple(y.shape) == tuple(x.shape), f"{tuple(y.shape)}")
    ck("A2 dtype 一致", y.dtype == x.dtype, f"{y.dtype}")
    ck("A3 device 一致", y.device == x.device, f"{y.device}")
    ck("A4 初始为精确恒等 max|x*-x| < 1e-6", d < 1e-6, f"max_abs_diff = {d:.3e}")
    ck("A5 无 NaN/Inf", bool(torch.isfinite(y).all()), f"nan={int(torch.isnan(y).sum())} inf={int(torch.isinf(y).sum())}")

    # 门控是否真在 1 附近（证明是 1+tanh 而非 sigmoid）
    g = (1.0 + torch.tanh(m.fc(m.pool(x)))).detach()
    ck("A6 初始门控 ≡ 1（非 sigmoid 的 0.5）", torch.allclose(g, torch.ones_like(g)), f"gate range [{g.min():.4f},{g.max():.4f}]")

    # 梯度可反传：fc 与输入都必须有非零梯度
    m.zero_grad(); x2 = x.clone().requires_grad_(True)
    m(x2).sum().backward()
    gw = m.fc.weight.grad
    ck("A7 fc.weight 梯度存在且非零", gw is not None and float(gw.abs().sum()) > 0,
       f"|grad|={float(gw.abs().sum()):.4e}" if gw is not None else "grad=None")
    ck("A8 fc.bias 梯度存在且非零", m.fc.bias.grad is not None and float(m.fc.bias.grad.abs().sum()) > 0,
       f"|grad|={float(m.fc.bias.grad.abs().sum()):.4e}" if m.fc.bias.grad is not None else "grad=None")
    ck("A9 对输入的梯度非零（不是死门）", x2.grad is not None and float(x2.grad.abs().sum()) > 0,
       f"|grad|={float(x2.grad.abs().sum()):.4e}" if x2.grad is not None else "grad=None")

    # 非恒等性：把 fc 扰动后输出必须改变（证明模块真的可学）
    with torch.no_grad():
        m.fc.weight.normal_(0, 0.02)
    y2 = m(x)
    ck("A10 参数被扰动后输出改变（模块有效）", (y2 - x).abs().max().item() > 1e-4,
       f"max_abs_diff = {(y2 - x).abs().max().item():.3e}")

    # 对照：现成的 conv.py ChannelAttention 零初始化会得到 0.5x
    from ultralytics.nn.modules.conv import ChannelAttention
    ca = ChannelAttention(c)
    with torch.no_grad():
        ca.fc.weight.zero_(); ca.fc.bias.zero_()
    ref = ca(x)
    ck("A11 对照：现成 ChannelAttention 零初始化 = 0.5x（故不可直接用）",
       torch.allclose(ref, 0.5 * x), f"max|ref-0.5x| = {(ref-0.5*x).abs().max().item():.3e}")


# ---------------------------------------------------------------- build models
def build(yml, tag):
    m = T.DetectionModel(str(yml), ch=CH, nc=NC, verbose=False)
    print(f"  [{tag}] 顶层模块数={len(m.model)}  参数={sum(p.numel() for p in m.parameters()):,}")
    return m


def load_from(model, w):
    src, _ = T.attempt_load_one_weight(str(w))
    sd = model.state_dict(); src_sd = src.float().state_dict()
    inter = T.intersect_dicts(src_sd, sd)
    model.load(src)
    return src_sd, sd, inter


# ---------------------------------------------------------------- §3 migration
def test_migration(base, attn):
    sec("§3 权重迁移")
    # 3.1 baseline 回归：对 yolo11m.pt 必须仍是 642/649（本会话早前实测值）
    _, bsd, bi = load_from(base, YOLO11M)
    ck("B1 baseline 键数仍为 649", len(bsd) == 649, f"{len(bsd)}")
    ck("B2 baseline 对 yolo11m.pt 迁移数仍为 642/649（与改动前一致）",
       len(bi) == 642, f"{len(bi)}/{len(bsd)}")
    from ultralytics.models.yolo.detect.train import _transfer_rgb_pretrained
    base2 = T.DetectionModel(str(BASE_YAML), ch=CH, nc=NC, verbose=False)
    r = _transfer_rgb_pretrained(base2, T.attempt_load_one_weight(str(YOLO11M))[0])
    ck("B3 baseline 的 _transfer_rgb_pretrained 仍返回 5", r == 5, f"{r}")

    # 3.2 attn 模型：对正式 best.pt
    src_sd, asd, ai = load_from(attn, BEST_PT)
    miss = sorted(set(asd) - set(ai))
    only_attn = [k for k in miss if ".attn." in k]
    other = [k for k in miss if ".attn." not in k]
    ck("B4 attn 模型键数 = 651", len(asd) == 651, f"{len(asd)}")
    ck("B5 匹配 649/651", len(ai) == 649, f"{len(ai)}/{len(asd)}")
    ck("B6 未匹配项全部是新增 attn 权重", len(only_attn) == 2 and not other,
       f"attn={only_attn} other={other}")

    det_keys = [k for k in asd if k.startswith("model.23.")]
    ck("B7 Detect head 权重完整匹配", all(k in ai for k in det_keys), f"Detect 键 {len(det_keys)} 个，全部匹配")
    bb_keys = [k for k in asd if k.startswith(("model.0.", "model.1.", "model.2.", "model.3."))]
    ck("B8 backbone 前 4 层完整匹配（无大面积随机初始化）", all(k in ai for k in bb_keys), f"{len(bb_keys)} 个")
    ck("B9 层索引不变（顶层模块数 24 且 Detect 仍在 23）",
       len(attn.model) == 24 and type(attn.model[23]).__name__ == "Detect")

    # 3.3 结构等价：层 16 除 attn 外必须完全一致
    b16 = [k for k in base.state_dict() if k.startswith("model.16.")]
    a16 = [k for k in attn.state_dict() if k.startswith("model.16.") and ".attn." not in k]
    ck("B10 层 16 非 attn 的 key 与 baseline 逐字一致", sorted(b16) == sorted(a16), f"n={len(b16)}")
    d = attn.model[23]
    ck("B11 Detect 输入通道仍为 [256,512,512]",
       [d.cv2[i][0].conv.in_channels for i in range(3)] == [256, 512, 512])
    ck("B12 stride 仍为 [8,16,32]", d.stride.tolist() == [8.0, 16.0, 32.0], f"{d.stride.tolist()}")
    pb = sum(p.numel() for p in base.parameters()); pa = sum(p.numel() for p in attn.parameters())
    ck("B13 参数增量 < 1%", (pa - pb) / pb < 0.01, f"Δ={pa-pb:+,} ({(pa-pb)/pb*100:+.4f}%)")


# ---------------------------------------------------------------- §6 consistency
def _clone(o):
    if torch.is_tensor(o):
        return o.detach().clone()
    if isinstance(o, (list, tuple)):
        return type(o)(_clone(v) for v in o)
    return o


def _maxdiff(a, b, acc=None):
    """递归求两个（可能嵌套的）输出结构之间的最大绝对误差。"""
    if acc is None:
        acc = [0.0]
    if torch.is_tensor(a) and torch.is_tensor(b):
        if a.shape == b.shape:
            acc[0] = max(acc[0], (a.float() - b.float()).abs().max().item())
        return acc
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        for u, v in zip(a, b):
            _maxdiff(u, v, acc)
    return acc


def _shapes(o):
    if torch.is_tensor(o):
        return tuple(o.shape)
    if isinstance(o, (list, tuple)):
        return [_shapes(v) for v in o]
    return type(o).__name__


def test_consistency(base, attn):
    sec("§6 初始输出一致性（同一输入，baseline vs identity-attn）")
    src, _ = T.attempt_load_one_weight(str(BEST_PT))
    base.load(src); attn.load(src)          # attn 的 2 个权重保持零初始化
    base.eval(); attn.eval()

    caps = {}
    def hook(tag, on_input=False):
        def f(_m, inp, out):
            caps[tag] = _clone(inp[0] if on_input else out)
        return f

    hb = [base.model[16].register_forward_hook(hook("b16")),
          base.model[23].register_forward_hook(hook("bdet_in", on_input=True))]
    ha = [attn.model[16].register_forward_hook(hook("a16")),
          attn.model[23].register_forward_hook(hook("adet_in", on_input=True))]

    torch.manual_seed(0)
    x = torch.rand(1, CH, 1280, 1280)
    with torch.no_grad():
        yb = base(x); ya = attn(x)
    for h in hb + ha:
        h.remove()

    d16 = _maxdiff(caps["b16"], caps["a16"])[0]
    print(f"\n  layer 16 输出      shape={_shapes(caps['b16'])}  max_abs_err = {d16:.3e}")
    ck("C1 layer 16 输出逐值一致 (<1e-6)", d16 < 1e-6, f"{d16:.3e}")

    ddet = _maxdiff(caps["bdet_in"], caps["adet_in"])[0]
    n = len(caps["bdet_in"]) if isinstance(caps["bdet_in"], (list, tuple)) else 1
    print(f"  Detect 输入(P3/P4/P5, {n} tensor)  形状={_shapes(caps['bdet_in'])}  max_abs_err = {ddet:.3e}")
    ck("C2 Detect 输入逐值一致 (<1e-6)", ddet < 1e-6, f"{ddet:.3e}")

    dp = _maxdiff(yb, ya)[0]
    ref = torch.cat([t.float().abs().flatten() for t in (yb if isinstance(yb, (list, tuple)) else [yb])
                     if torch.is_tensor(t)]) if isinstance(yb, (list, tuple)) else yb.float().abs().flatten()
    rel = dp / max(float(ref.max()), 1e-12)
    print(f"  最终预测输出      shape={_shapes(yb)}  max_abs_err = {dp:.3e}  max_rel_err = {rel:.3e}")
    ck("C3 最终预测输出逐值一致 (<1e-6)", dp < 1e-6, f"abs={dp:.3e} rel={rel:.3e}")
    ck("C4 输出无 NaN/Inf", bool(torch.isfinite(torch.cat([t.float().flatten() for t in
        (yb if isinstance(yb, (list, tuple)) else [yb]) if torch.is_tensor(t)])).all()))


# ---------------------------------------------------------------- §4 freeze
def test_freeze(base, attn):
    sec("§4 冻结参数分析 —— 「只训练新增 attention」在本框架下能否实现")
    src, _ = T.attempt_load_one_weight(str(BEST_PT))
    attn.load(src)
    total = sum(p.numel() for p in attn.parameters())
    l16 = attn.model[16]
    n16 = sum(p.numel() for p in l16.parameters())
    nattn = sum(p.numel() for p in l16.attn.parameters())
    print(f"  全模型参数            = {total:,}")
    print(f"  layer 16 参数         = {n16:,}  ({n16/total*100:.4f}%)")
    print(f"    其中新增 attn       = {nattn:,}  ({nattn/total*100:.4f}%)")
    print(f"    其中既有 C3k2       = {n16-nattn:,}")
    print(f"    Detect(23) 参数     = {sum(p.numel() for p in attn.model[23].parameters()):,}")
    print()
    print("  方案对照（trainable 参数数）：")
    print(f"    freeze=null（全模型）                 -> {total:,}  ❌ 用户明确禁止")
    print(f"    freeze=[除16外全部层]                 -> {n16:,}  ⚠️ 含既有 C3k2，非「只有 attn」")
    print(f"    只训练 attn（需改 trainer）           -> {nattn:,}  ✅ 用户要求的目标")


def main():
    test_identity_attn()
    base = build(BASE_YAML, "baseline")
    attn = build(ATTN_YAML, "p3attn")
    test_migration(base, attn)
    test_consistency(base, attn)
    test_freeze(base, attn)
    sec("汇总")
    print(f"  FAIL: {len(FAILS)}" + ("" if not FAILS else "  -> " + str(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
