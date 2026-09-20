"""scripts/verify_sepstem_40_16_8_pretrained.py — Separate-Stem 40/16/8 预训练迁移验证（只读，不训练）。

D'' 的单变量对照物是 D'（48/8/8）。本脚本回答三个问题：

  1. D'' 的预训练迁移是否与 D' 同等级（backbone/head/stem 逐 tensor）？
  2. 64ch 预算再分配后，runtime graph 是否仍然严格 40/16/8 -> 64，无 silent fallback？
  3. 三条 stem 是否都真正参与 forward（dependency）与 backward（gradient）？
  另外附带 D' vs D'' 的参数量对照（架构决定，不需要 checkpoint）。

不训练、不创建 run、不改任何配置。退出码 0 = 全部通过。

dataset.yaml 按存在性解析（`rgbid_split_train/` 优先，回退 `rgbid_split/`），
不硬编码 —— 硬编码会在另一台机器上直接 FileNotFoundError。

用法:  python scripts/verify_sepstem_40_16_8_pretrained.py [--data <dataset.yaml>]
"""
import argparse
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, guess_model_scale  # noqa: E402
from ultralytics.models.yolo.detect.train import _transfer_rgb_pretrained  # noqa: E402

MODEL_YAML = PROJECT_ROOT / "configs" / "yolo11m_sepstem_40_16_8.yaml"
BASELINE_YAML = PROJECT_ROOT / "configs" / "yolo11m_sepstem.yaml"  # D' = 48/8/8
PRETRAINED = "yolo11m.pt"

# dataset.yaml 的候选路径，按「真实训练实际用的那份」优先。
# scripts/train.py::_split_train_val_rgbid 生成的目录名是 `rgbid_split_<src_key>`，
# src_key = train 目录的父目录名（通常是 `train`）-> `rgbid_split_train`。
# 仓库里另有一份更早的 `rgbid_split/`（已进 FREEZE_MANIFEST）。两者都可能存在，
# 因此这里按存在性解析而不是硬编码——硬编码会在另一台机器上直接崩。
DATASET_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "rgbid_split_train" / "dataset.yaml",
    PROJECT_ROOT / "data" / "processed" / "rgbid_split" / "dataset.yaml",
]

# 由 main() 设置
DATASET_YAML = None


def resolve_dataset_yaml(explicit=None):
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"--data {p} does not exist")
        return p
    for p in DATASET_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "no RGBID dataset.yaml found. Tried:\n  "
        + "\n  ".join(str(p) for p in DATASET_CANDIDATES)
        + "\nRun scripts/train.py once (or pass --data) to generate the split."
    )

EXPECT = {"rgb": 40, "ir": 16, "depth": 8}

FAILURES = []
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return bool(ok)


def build_via_real_trainer(yaml_path):
    """走 DetectionTrainer.get_model() 的正式路径（project 指向 %TEMP%，不碰 runs/）。"""
    from ultralytics.models.yolo.detect import DetectionTrainer

    tmp = Path(tempfile.mkdtemp(prefix="sepstem40168_verify_"))
    trainer = DetectionTrainer(
        overrides={
            "model": str(yaml_path),
            "data": str(DATASET_YAML),
            "pretrained": PRETRAINED,
            "project": str(tmp),
            "name": "verify",
            "exist_ok": True,
            "epochs": 1,
            "imgsz": 1280,
            "batch": 1,
            # 固定 CPU：本脚本是 build/remap/graph 审计，不是训练。
            # 固定 CPU 让证据与机器无关，也避免审计里 `torch.equal(model_w, stock_w)`
            # 出现 cuda/cpu 设备不匹配。GPU 机器上依然可用，只是这一步慢一些。
            "device": "cpu",
            "workers": 0,
        }
    )
    return trainer.get_model(cfg=str(yaml_path)), tmp


def stem_roles(model):
    """IR/Depth 由喂给该 stem 的 SilenceChannel c_start 判定（3=IR, 4=Depth），不看模块位置。"""
    from ultralytics.nn.modules.conv import Conv

    layers = list(model.yaml.get("backbone", [])) + list(model.yaml.get("head", []))
    idx_of = {id(m): i for i, m in enumerate(model.model)}

    def feeder_c_start(i):
        fr = layers[i][0]
        j = i - 1 if fr == -1 else fr
        if not isinstance(j, int) or j < 0 or j >= len(layers):
            return None
        mod = model.model[j]
        return int(getattr(mod, "c_start", -1)) if type(mod).__name__ == "SilenceChannel" else None

    rgb = ir = dep = None
    for m in model.model:
        if not isinstance(m, Conv):
            continue
        if m.conv.in_channels == 3:
            rgb = m
        elif m.conv.in_channels == 1:
            cs = feeder_c_start(idx_of[id(m)])
            if cs == 3:
                ir = m
            elif cs == 4:
                dep = m
    return rgb, ir, dep, layers, idx_of


def concat_idx(layers):
    for i, lyr in enumerate(layers):
        if str(lyr[2]) == "Concat" and isinstance(lyr[0], (list, tuple)) and len(lyr[0]) == 3:
            return i
    return None


def main():
    global DATASET_YAML

    ap = argparse.ArgumentParser(description="D'' = Separate-Stem 40/16/8 预训练迁移验证（只读）")
    ap.add_argument("--data", default=None,
                    help="RGBID dataset.yaml 路径；默认自动解析 rgbid_split_train/ 再回退 rgbid_split/")
    a = ap.parse_args()

    print("=" * 74)
    print("SEPARATE-STEM 40/16/8 PRETRAINED REMAP VERIFICATION (read-only, no training)")
    print("=" * 74)

    DATASET_YAML = resolve_dataset_yaml(a.data)
    print(f"\n[0] dataset.yaml -> {DATASET_YAML}")
    from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: F401  (import-time sanity)
    print(f"    model.yaml   -> {MODEL_YAML}")
    print(f"    pretrained   -> {PRETRAINED}")

    # ---------------- 1. scale ----------------
    print("\n[1] model scale (filename-derived; silent nano fallback guard)")
    sc = guess_model_scale(str(MODEL_YAML))
    check("scale == 'm'", sc == "m", f"got {sc!r}")
    print(f"  note: yolo11m.pt stock stem out_channels = "
          f"{attempt_load_one_weight(PRETRAINED)[0].state_dict()['model.0.conv.weight'].shape[0]}")

    # ---------------- 2. build via real trainer path ----------------
    print("\n[2] build via DetectionTrainer.get_model()  (real remap path)")
    model, tmpdir = build_via_real_trainer(MODEL_YAML)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    check("built OK", model is not None)
    check("input ch == 5", model.yaml.get("ch") == 5, f"ch={model.yaml.get('ch')}")
    check("nc == 12", int(getattr(model.model[-1], "nc", -1)) == 12)
    check("n layers == 31", len(model.model) == 31, f"{len(model.model)}")

    rgb, ir, dep, layers, idx_of = stem_roles(model)
    check("found RGB/IR/Depth stems", None not in (rgb, ir, dep))
    ci = concat_idx(layers)
    check("fusion Concat located at layer 7", ci == 7, f"cat_idx={ci}")

    # ---------------- 3. stem widths ----------------
    print("\n[3] stem widths == 40 / 16 / 8, sum == 64")
    got = {"rgb": rgb.conv.weight.shape[0], "ir": ir.conv.weight.shape[0],
           "depth": dep.conv.weight.shape[0]}
    check("RGB out == 40", got["rgb"] == EXPECT["rgb"], f"{got['rgb']}")
    check("IR  out == 16", got["ir"] == EXPECT["ir"], f"{got['ir']}")
    check("Depth out == 8", got["depth"] == EXPECT["depth"], f"{got['depth']}")
    check("40 + 16 + 8 == 64", sum(got.values()) == 64, f"{sum(got.values())}")
    check("RGB stem in_channels == 3", rgb.conv.in_channels == 3)
    check("IR  stem in_channels == 1", ir.conv.in_channels == 1)
    check("Depth stem in_channels == 1", dep.conv.in_channels == 1)

    # ---------------- 4. stem initialisation (tensor level) ----------------
    print("\n[4] stem initialisation == stock stem prefix (NOT random)")
    src = attempt_load_one_weight(PRETRAINED)[0].state_dict()
    sw = src["model.0.conv.weight"]
    check("stock stem is 3-channel", sw.shape[1] == 3, f"in={sw.shape[1]}")
    check("stock stem out >= 40 (prefix source exists)", sw.shape[0] >= 40, f"out={sw.shape[0]}")

    exp = {}
    for kind, stem in (("rgb", rgb), ("ir", ir), ("depth", dep)):
        oc = stem.conv.weight.shape[0]
        if kind == "rgb":
            exp[kind] = sw[:oc].to(stem.conv.weight.dtype)
            rule = f"stock[:{oc}]"
        else:
            exp[kind] = sw[:oc].float().mean(dim=1, keepdim=True).to(stem.conv.weight.dtype)
            rule = f"mean(RGB)[:{oc}]"
        eq = torch.equal(stem.conv.weight, exp[kind])
        check(f"{kind:5s} stem == {rule}", eq,
              f"max|Δ|={(stem.conv.weight - exp[kind]).abs().max().item():.3e}")
        check(f"{kind:5s} stem is NOT random (|W|sum > 0)", stem.conv.weight.abs().sum().item() > 0)
        # BN must come from the same prefix
        check(f"{kind:5s} BN == stock BN[:{oc}]",
              torch.equal(stem.bn.weight, src["model.0.bn.weight"][:oc]))

    # ---------------- 5. backbone / head pretrained equality ----------------
    print("\n[5] backbone/head pretrained equality (structural correspondence)")
    offset = ci
    csd = model.state_dict()
    n_bb = len(model.yaml["backbone"])
    body_hit = body_tot = head_hit = head_tot = 0
    for i in range(offset + 1, len(layers)):
        s = i - offset
        prefix = f"model.{i}."
        for k, v in csd.items():
            if not k.startswith(prefix) or v.ndim == 0:
                continue
            sv = src.get(f"model.{s}." + k[len(prefix):])
            if sv is None or tuple(sv.shape) != tuple(v.shape):
                continue
            if i >= n_bb:
                head_tot += 1
                head_hit += int(torch.equal(v, sv))
            else:
                body_tot += 1
                body_hit += int(torch.equal(v, sv))
    body_pct = 100.0 * body_hit / max(1, body_tot)
    head_pct = 100.0 * head_hit / max(1, head_tot)
    check(f"backbone exact-match {body_hit}/{body_tot} = {body_pct:.1f}%", body_pct >= 98.0)
    check(f"head     exact-match {head_hit}/{head_tot} = {head_pct:.1f}%", head_pct >= 98.0)

    # corresponding-tensor coverage: how many candidate tensors have a stock counterpart at all
    cov = miss = 0
    for i in range(offset + 1, len(layers)):
        s = i - offset
        prefix = f"model.{i}."
        for k, v in csd.items():
            if not k.startswith(prefix) or v.ndim == 0:
                continue
            sv = src.get(f"model.{s}." + k[len(prefix):])
            if sv is not None and tuple(sv.shape) == tuple(v.shape):
                cov += 1
            else:
                miss += 1
    print(f"  corresponding tensors: {cov} matched, {miss} not (Detect cv3 class branch, nc 80->12)")

    # ---------------- 6. runtime graph audit ----------------
    print("\n[6] runtime graph (real [B,5,1280,1280] input)")
    x = torch.rand(2, 5, 1280, 1280)
    caps = {}
    hooks = [
        rgb.register_forward_hook(lambda m, i, o: caps.__setitem__("rgb", tuple(o.shape))),
        ir.register_forward_hook(lambda m, i, o: caps.__setitem__("ir", tuple(o.shape))),
        dep.register_forward_hook(lambda m, i, o: caps.__setitem__("depth", tuple(o.shape))),
        model.model[ci].register_forward_hook(lambda m, i, o: caps.__setitem__("cat", tuple(o.shape))),
        model.model[ci + 1].register_forward_hook(
            lambda m, i, o: caps.__setitem__("next_in", tuple(i[0].shape))),
    ]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    check("input shape == (2,5,1280,1280)", True, f"{tuple(x.shape)}")
    check("RGB   stem out == (2,40,640,640)", caps["rgb"] == (2, 40, 640, 640), f"{caps['rgb']}")
    check("IR    stem out == (2,16,640,640)", caps["ir"] == (2, 16, 640, 640), f"{caps['ir']}")
    check("Depth stem out == (2,8,640,640)", caps["depth"] == (2, 8, 640, 640), f"{caps['depth']}")
    check("Concat out == (2,64,640,640)", caps["cat"] == (2, 64, 640, 640), f"{caps['cat']}")
    check("next backbone layer RECEIVES 64 ch", caps["next_in"][1] == 64, f"{caps['next_in']}")
    check("next layer conv.in_channels == 64", model.model[ci + 1].conv.in_channels == 64,
          f"{model.model[ci + 1].conv.in_channels}")
    check("no silent channel fallback (40+16+8 == next layer c1)",
          caps["cat"][1] == model.model[ci + 1].conv.in_channels)

    # ---------------- 7. runtime modality dependency ----------------
    print("\n[7] runtime modality dependency (perturb ONE modality, output must move)")
    xs = torch.rand(1, 5, 256, 256)
    with torch.no_grad():
        base = torch.cat([t.flatten() for t in model(xs) if torch.is_tensor(t)])

    def delta(chans):
        g = torch.Generator().manual_seed(0)
        xp = xs.clone()
        xp[:, chans] = xp[:, chans] + torch.randn(xp[:, chans].shape, generator=g) * 3.0
        with torch.no_grad():
            o = torch.cat([t.flatten() for t in model(xp) if torch.is_tensor(t)])
        return (o - base).abs().max().item()

    for label, ch in (("RGB", [0, 1, 2]), ("IR", [3]), ("Depth", [4])):
        d = delta(ch)
        check(f"{label} dependency > 0", d > 0, f"max|Δ|={d:.3e}")

    # ---------------- 8. gradient dependency ----------------
    print("\n[8] gradient dependency (all three stems, finite & non-zero)")
    model.train()
    model.zero_grad()
    out = model(xs)
    loss = sum(t.float().pow(2).sum() for t in out if torch.is_tensor(t))
    loss.backward()
    for label, m in (("RGB", rgb), ("IR", ir), ("Depth", dep)):
        g = m.conv.weight.grad
        ok = g is not None and torch.isfinite(g).all() and g.abs().sum().item() > 0
        check(f"{label:5s} stem gradient ok", ok,
              "None" if g is None else f"|g|sum={g.abs().sum().item():.6g}")
        gb = m.bn.weight.grad
        check(f"{label:5s} BN   gradient ok",
              gb is not None and torch.isfinite(gb).all() and gb.abs().sum().item() > 0,
              "None" if gb is None else f"|g|sum={gb.abs().sum().item():.6g}")
    model.eval()

    # ---------------- 9. D' vs D'' parameter comparison ----------------
    print("\n[9] parameter comparison D' (48/8/8) vs D'' (40/16/8)")
    d1, _ = build_via_real_trainer(BASELINE_YAML)
    n_d1 = sum(p.numel() for p in d1.parameters())
    n_d2 = n_params
    conv_delta = -(48 - 40) * 3 * 9 + (16 - 8) * 1 * 9   # RGB -216, IR +72  -> -144
    bn_delta = -(48 - 40) * 2 + (16 - 8) * 2             # RGB -16,  IR +16  ->    0
    print(f"  D'  params (48/8/8)  = {n_d1:,}")
    print(f"  D'' params (40/16/8) = {n_d2:,}")
    print(f"  Δ = {n_d2 - n_d1:+,}  (expected {conv_delta + bn_delta:+,}: conv {conv_delta:+,} + BN {bn_delta:+,})")
    check("Δ params == analytic (conv -144 + BN 0 = -144)", n_d2 - n_d1 == conv_delta + bn_delta,
          f"actual {n_d2 - n_d1:+,}")
    check("D'' params within 0.01% of D'",
          abs(n_d2 - n_d1) / n_d1 < 1e-4, f"{100 * abs(n_d2 - n_d1) / n_d1:.5f}%")

    from ultralytics.utils.torch_utils import get_flops

    f1, f2 = get_flops(d1, imgsz=1280), get_flops(model, imgsz=1280)
    print(f"  D'  GFLOPs@1280 = {f1:.2f}   D'' GFLOPs@1280 = {f2:.2f}   Δ = {f2 - f1:+.4f}")
    check("GFLOPs essentially unchanged (<0.1%)",
          abs(f2 - f1) / max(f1, 1e-9) < 1e-3, f"{100 * abs(f2 - f1) / max(f1, 1e-9):.4f}%")

    # ---------------- 10. temp checkpoint save/reload ----------------
    print("\n[10] temp checkpoint save/reload (structure preserved)")
    ckpt = Path(tempfile.mkdtemp(prefix="sepstem40168_ckpt_")) / "tmp.pt"
    torch.save({"model": model}, ckpt)
    from ultralytics import YOLO

    rm = YOLO(str(ckpt)).model
    check("reload: ch == 5", rm.yaml.get("ch") == 5)
    check("reload: 31 layers", len(rm.model) == 31, f"{len(rm.model)}")
    check("reload: params identical", sum(p.numel() for p in rm.parameters()) == n_params)

    # ---------------- 11. baseline regression (D' and RGBD must be untouched) ----------------
    print("\n[11] baseline regression (D' + RGBD paths unchanged by this experiment)")
    d1_stems = stem_roles(d1)
    check("D' stems still 48 / 8 / 8",
          (d1_stems[0].conv.weight.shape[0], d1_stems[1].conv.weight.shape[0],
           d1_stems[2].conv.weight.shape[0]) == (48, 8, 8),
          f"{d1_stems[0].conv.weight.shape[0]}/{d1_stems[1].conv.weight.shape[0]}/{d1_stems[2].conv.weight.shape[0]}")

    _p = PROJECT_ROOT / "configs" / "yolo11m_midfusion_rgbd_concat_res.yaml"
    mm = DetectionModel(str(_p), verbose=False)
    _t = [m for m in mm.model if type(m).__name__ == "Conv" and m.conv.in_channels == 3]
    _o = [m for m in mm.model if type(m).__name__ == "Conv" and m.conv.in_channels == 1]
    check("RGBD midfusion still 3ch+1ch (sepstem branch predicate FALSE)",
          len(_t) == 1 and len(_o) == 1, f"three={len(_t)} ones={len(_o)}")
    w, _ = attempt_load_one_weight(PRETRAINED)
    nm = _transfer_rgb_pretrained(mm, w)
    check("RGBD midfusion still remapped (returns > 0)", nm > 0, f"returned {nm}")

    # ---------------- summary ----------------
    print("\n" + "=" * 74)
    n_fail = len(FAILURES)
    print(f"RESULT: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    for f in FAILURES:
        print("  FAILED:", f)
    print("REMAP_READY_FOR_TRAINING" if n_fail == 0 else "REMAP_NOT_READY")
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
