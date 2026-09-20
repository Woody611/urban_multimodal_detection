"""scripts/verify_sepstem_pretrained.py — Separate-Stem Fusion 预训练迁移验证（只读，不训练）。

职责：不训练、不创建 run、不改任何配置，只回答一个问题 ——
「configs/yolo11m_sepstem.yaml 走正式训练的加载路径后，是否真的拿到了 YOLO11m 预训练权重？」

覆盖：
  1. scale == 'm'（防静默回退 nano）
  2. 通过 DetectionTrainer.get_model() 的真实路径构建（非手工调用 remap）
  3. 三个 stem 的确定性初始化逐 tensor 验证
  4. backbone/head 结构对应 tensor exact-match >= 98%
  5. runtime modality dependency（RGB / IR / Depth 都能改变输出）
  6. gradient dependency（三个 stem 梯度非 None / 有限 / 非零）
  7. Concat = 48+8+8 = 64
  8. temp checkpoint save/reload 后结构保持（写入 %TEMP%，不污染 runs/）

用法:  python scripts/verify_sepstem_pretrained.py
退出码: 0 = 全部通过；1 = 有 FAIL
"""
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, guess_model_scale  # noqa: E402
from ultralytics.models.yolo.detect.train import _transfer_rgb_pretrained  # noqa: E402

MODEL_YAML = PROJECT_ROOT / "configs" / "yolo11m_sepstem.yaml"
DATASET_YAML = PROJECT_ROOT / "data" / "processed" / "rgbid_split" / "dataset.yaml"
PRETRAINED = "yolo11m.pt"

FAILURES = []
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return bool(ok)


def build_via_real_trainer():
    """走 DetectionTrainer.get_model() 的正式路径。

    project 指向 %TEMP%，以免 BaseTrainer 在仓库 runs/ 下建目录。
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    tmp = Path(tempfile.mkdtemp(prefix="sepstem_verify_"))
    trainer = DetectionTrainer(
        overrides={
            "model": str(MODEL_YAML),
            "data": str(DATASET_YAML),
            "pretrained": PRETRAINED,
            "project": str(tmp),
            "name": "verify_sepstem",
            "exist_ok": True,
            "epochs": 1,
            "imgsz": 1280,
            "batch": 1,
            "device": "cpu",
            "workers": 0,
        }
    )
    model = trainer.get_model(cfg=str(MODEL_YAML))
    return model, tmp


def stems_of(model):
    from ultralytics.nn.modules.conv import Conv

    return [m for m in model.model if isinstance(m, Conv)]


def stem_roles(model):
    """Re-derive IR/Depth from the feeder SilenceChannel c_start (same rule as the remap)."""
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
    for m in stems_of(model):
        if m.conv.in_channels == 3:
            rgb = m
        elif m.conv.in_channels == 1:
            cs = feeder_c_start(idx_of[id(m)])
            if cs == 3:
                ir = m
            elif cs == 4:
                dep = m
    return rgb, ir, dep, layers, idx_of


def tail_mapping(model, layers):
    """candidate layer i (> offset) <-> stock layer (i - offset); offset = fusion Concat index."""
    cat_idx = None
    for i, lyr in enumerate(layers):
        if str(lyr[2]) == "Concat" and isinstance(lyr[0], (list, tuple)) and len(lyr[0]) == 3:
            cat_idx = i
            break
    return cat_idx


def main():
    print("=" * 72)
    print("SEPARATE-STEM PRETRAINED REMAP VERIFICATION (read-only, no training)")
    print("=" * 72)

    # ---------------- 1. scale ----------------
    print("\n[1] model scale")
    sc = guess_model_scale(str(MODEL_YAML))
    check("scale == 'm'", sc == "m", f"got {sc!r}")

    # ---------------- 2. real trainer path ----------------
    print("\n[2] build via DetectionTrainer.get_model()")
    model, tmpdir = build_via_real_trainer()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    check("built OK", model is not None)
    check("params ≈ 20.06M", abs(n_params - 20063412) < 20000, f"{n_params}")
    check("input ch == 5", model.yaml.get("ch") == 5, f"ch={model.yaml.get('ch')}")
    check("nc == 12", int(getattr(model.model[-1], "nc", -1)) == 12)
    check("n layers == 31", len(model.model) == 31, f"{len(model.model)}")

    rgb, ir, dep, layers, idx_of = stem_roles(model)
    check("found RGB/IR/Depth stems", None not in (rgb, ir, dep))
    cat_idx = tail_mapping(model, layers)
    check("fusion Concat located", cat_idx == 7, f"cat_idx={cat_idx}")

    # ---------------- 3. stems ----------------
    print("\n[3] stem initialisation (tensor-level)")
    src = torch.load(PRETRAINED, map_location="cpu", weights_only=False)["model"].state_dict()
    sw = src["model.0.conv.weight"]

    exp_rgb = sw[: rgb.conv.weight.shape[0]].to(rgb.conv.weight.dtype)
    exp_ir = (sw[: ir.conv.weight.shape[0]].float().mean(dim=1, keepdim=True)
              .to(ir.conv.weight.dtype))
    exp_dep = (sw[: dep.conv.weight.shape[0]].float().mean(dim=1, keepdim=True)
               .to(dep.conv.weight.dtype))
    check("RGB stem == stock[:48]", torch.equal(rgb.conv.weight, exp_rgb),
          f"max|Δ|={(rgb.conv.weight - exp_rgb).abs().max().item():.3e}")
    check("IR stem == mean(RGB)[:8]", torch.equal(ir.conv.weight, exp_ir),
          f"max|Δ|={(ir.conv.weight - exp_ir).abs().max().item():.3e}")
    check("Depth stem == mean(RGB)[:8]", torch.equal(dep.conv.weight, exp_dep),
          f"max|Δ|={(dep.conv.weight - exp_dep).abs().max().item():.3e}")
    check("IR != Depth-random (both nonzero)",
          ir.conv.weight.abs().sum() > 0 and dep.conv.weight.abs().sum() > 0)
    check("RGB BN == stock BN[:48]", torch.equal(rgb.bn.weight, src["model.0.bn.weight"][:48]))
    check("IR BN == stock BN[:8]", torch.equal(ir.bn.weight, src["model.0.bn.weight"][:8]))

    # ---------------- 4. backbone / head equality ----------------
    print("\n[4] backbone/head pretrained equality (structural correspondence)")
    offset = cat_idx
    csd = model.state_dict()
    body_hit = body_tot = 0
    head_hit = head_tot = 0
    n_bb = len(model.yaml["backbone"])
    for i in range(offset + 1, len(layers)):
        s = i - offset
        prefix = f"model.{i}."
        for k, v in csd.items():
            if not k.startswith(prefix) or v.ndim == 0:
                continue
            sv = src.get(f"model.{s}." + k[len(prefix):])
            if sv is None or tuple(sv.shape) != tuple(v.shape):
                continue
            is_head = i >= n_bb
            if is_head:
                head_tot += 1
                head_hit += int(torch.equal(v, sv))
            else:
                body_tot += 1
                body_hit += int(torch.equal(v, sv))
    body_pct = 100.0 * body_hit / max(1, body_tot)
    head_pct = 100.0 * head_hit / max(1, head_tot)
    check(f"backbone exact-match {body_hit}/{body_tot} = {body_pct:.1f}%", body_pct >= 98.0)
    check(f"head exact-match {head_hit}/{head_tot} = {head_pct:.1f}%", head_pct >= 98.0)

    # ---------------- 5. runtime modality dependency ----------------
    print("\n[5] runtime modality dependency")
    x = torch.rand(1, 5, 128, 128)
    with torch.no_grad():
        base = torch.cat([t.flatten() for t in model(x) if torch.is_tensor(t)])

    def delta(chans):
        g = torch.Generator().manual_seed(0)
        xp = x.clone()
        xp[:, chans] = xp[:, chans] + torch.randn(xp[:, chans].shape, generator=g) * 3.0
        with torch.no_grad():
            o = torch.cat([t.flatten() for t in model(xp) if torch.is_tensor(t)])
        return (o - base).abs().max().item()

    for label, ch in (("RGB", [0, 1, 2]), ("IR", [3]), ("Depth", [4])):
        d = delta(ch)
        check(f"{label} dependency > 0", d > 0, f"max|Δ|={d:.3e}")

    # ---------------- 6. gradient dependency ----------------
    print("\n[6] gradient dependency")
    model.train()
    model.zero_grad()
    out = model(x)
    loss = sum(t.float().pow(2).sum() for t in out if torch.is_tensor(t))
    loss.backward()
    for label, m in (("RGB", rgb), ("IR", ir), ("Depth", dep)):
        g = m.conv.weight.grad
        ok = g is not None and torch.isfinite(g).all() and g.abs().sum().item() > 0
        check(f"{label} stem gradient ok", ok,
              "None" if g is None else f"|g|sum={g.abs().sum().item():.4g}")
    model.eval()

    # ---------------- 7. concat ----------------
    print("\n[7] concat channels")
    caps = {}
    hs = [
        model.model[cat_idx].register_forward_hook(
            lambda mo, i, o: caps.__setitem__("cat_out", tuple(o.shape))),
        model.model[cat_idx + 1].register_forward_hook(
            lambda mo, i, o: caps.__setitem__("next_in", tuple(i[0].shape))),
    ]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    check("concat out == 64 channels", caps.get("cat_out", (0, 0))[1] == 64, f"{caps.get('cat_out')}")
    check("backbone receives 64 channels", caps.get("next_in", (0, 0))[1] == 64, f"{caps.get('next_in')}")
    check("next layer conv in_channels == 64", model.model[cat_idx + 1].conv.in_channels == 64)

    # ---------------- 8. save / reload ----------------
    print("\n[8] temp checkpoint save/reload")
    ckpt = Path(tempfile.mkdtemp(prefix="sepstem_ckpt_")) / "sepstem_tmp.pt"
    torch.save({"model": model}, ckpt)
    from ultralytics import YOLO

    reloaded = YOLO(str(ckpt))
    rm = reloaded.model
    check("reload: ch == 5", rm.yaml.get("ch") == 5)
    check("reload: 31 layers", len(rm.model) == 31, f"{len(rm.model)}")
    check("reload: layer0 is Identity/Silence", type(rm.model[0]).__name__ in ("Identity", "Silence"))
    check("reload: layer1 is SilenceChannel", type(rm.model[1]).__name__ == "SilenceChannel")
    check("reload: 3 stems present", len([m for m in rm.model
                                          if type(m).__name__ == "Conv" and m.conv.in_channels in (1, 3)]) == 3)
    check("reload: params identical",
          sum(p.numel() for p in rm.parameters()) == n_params)

    # ---------------- 9. baseline regression ----------------
    # `_transfer_rgb_pretrained` is SHARED code. Adding the separate-stem branch must not
    # change the two pre-existing layouts. Re-check them here.
    print("\n[9] baseline regression (existing layouts must be unchanged)")
    import re
    from ultralytics.models.yolo.detect.train import _transfer_rgb_pretrained as tr

    def _load(path):
        m = DetectionModel(str(path), verbose=False)
        w, _ = attempt_load_one_weight(PRETRAINED)
        m.load(w)
        return m, w, tr(m, w)

    def _body_match(m, offset):
        sd = m.state_dict()
        hit = tot = 0
        for i in range(1, 24):
            for k, v in src.items():
                if not k.startswith(f"model.{i}.") or v.ndim == 0:
                    continue
                cv = sd.get(f"model.{i + offset}." + k.split(".", 2)[2])
                if cv is None or tuple(cv.shape) != tuple(v.shape):
                    continue
                tot += 1
                hit += int(torch.equal(cv, v))
        return hit, tot

    # 9a. RGBID early fusion (ch:5, no 1ch stem) -> branch 1, returns 5, offset 0
    me, _, ne = _load(PROJECT_ROOT / "configs" / "yolo11m_earlyfusion.yaml")
    he, te = _body_match(me, 0)
    check("earlyfusion still remapped (returns 5)", ne == 5, f"returned {ne}")
    check(f"earlyfusion body match {he}/{te} = {100.0*he/max(1,te):.1f}%", 100.0 * he / max(1, te) >= 98.0)

    # 9b. RGBD mid-fusion (one 3ch + one 1ch) -> the new branch predicate must be FALSE,
    # which is what proves its code path is byte-identical to before this change.
    _p = PROJECT_ROOT / "configs" / "yolo11m_midfusion_rgbd_concat_res.yaml"
    mm = DetectionModel(str(_p), verbose=False)
    _t = [m for m in mm.model if type(m).__name__ == "Conv" and m.conv.in_channels == 3]
    _o = [m for m in mm.model if type(m).__name__ == "Conv" and m.conv.in_channels == 1]
    check("RGBD midfusion stems are 3ch+1ch (sepstem branch predicate FALSE)",
          len(_t) == 1 and len(_o) == 1, f"three={len(_t)} ones={len(_o)}")
    mm, _, nm = _load(_p)
    check("RGBD midfusion still remapped (returns >0)", nm > 0, f"returned {nm}")

    # ---------------- summary ----------------
    print("\n" + "=" * 72)
    n_fail = len(FAILURES)
    print(f"RESULT: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    for f in FAILURES:
        print("  FAILED:", f)
    print("REMAP_READY_FOR_TRAINING" if n_fail == 0 else "REMAP_NOT_READY")
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
