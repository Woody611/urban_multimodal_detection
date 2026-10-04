"""diagnostic/p1_remap_audit.py — P1 audit: residual-Depth remap + fail-closed contract gate.

ZERO GPU · ZERO training · ZERO inference on real data. CPU only (forced, see the
CUDA guard below). Only model construction, state_dict bookkeeping, tensor equality
and one dummy 64x64 CPU forward for the identity-at-initialization proof.

Sections
  1  baseline (git HEAD / status / SHAs)
  2  pre-P1 reference module (my three insertions stripped out) for true before/after
  3  prototypes (canonical candC + negative-test variants), written to a scratch dir
  4  mapping coverage table  (D / D' / M4 / candC)         -> report section 4
  5  mapping manifest + critical layer audit               -> report sections 5, 6
  6  identity-at-initialization probe                      -> report section 7
  7  negative / fail-closed tests                          -> report section 8
  8  regression scan over every model config in configs/    -> report section 9

Usage:
  python -X utf8 diagnostic/p1_remap_audit.py --out diagnostic/p1_remap_audit
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # MUST precede the torch import

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pretrained_coverage import (  # noqa: E402
    KIND_DERIVED, KIND_EXACT, KIND_NEW, KIND_ZERO, load_stock, manifest, measure,
)
from ultralytics import YOLO  # noqa: E402
import ultralytics.models.yolo.detect.train as TR  # noqa: E402

CUDA_USED = "NO"
THRESHOLD = 0.80
GIT = r"C:\Program Files\Git\cmd\git.exe"

# ============================================================
# Prototypes. Embedded here so this audit is reproducible from the repo alone;
# they are written to a scratch dir and are NOT promoted into configs/.
# ============================================================
_SCALES = """scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]
"""

_BODY_TAIL = """  - [-1, 1, Conv, [128, 3, 2]] # {a}
  - [-1, 2, C3k2, [256, False, 0.25]] # {b}
  - [-1, 1, Conv, [256, 3, 2]] # {c}
  - [-1, 2, C3k2, [512, False, 0.25]] # {d}
  - [-1, 1, Conv, [512, 3, 2]] # {e}
  - [-1, 2, C3k2, [512, True]] # {f}
  - [-1, 1, Conv, [1024, 3, 2]] # {g}
  - [-1, 2, C3k2, [1024, True]] # {h}
  - [-1, 1, SPPF, [1024, 5]] # {i}
  - [-1, 2, C2PSA, [1024]] # {j}
"""
_HEAD_TAIL = """head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, {b6}], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]] # {h1}
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, {b4}], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]] # {h2}
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, {h1}], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]] # {h3}
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, {b10}], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]] # {h4}
  - [[{p3}, {p4}, {p5}], 1, Detect, [nc]]
"""


def _body_tail(base: int) -> str:
    return _BODY_TAIL.format(**{k: base + i for i, k in enumerate("abcdefghij")})


def _head_tail(base: int) -> str:
    # backbone indices referenced by the head, expressed relative to `base`
    # (derived from yolo11m_earlyfusion.yaml where base == 0 for the 5ch stem)
    return _HEAD_TAIL.format(
        b6=base + 6, b4=base + 4, b10=base + 10,
        h1=base + 13, h2=base + 16, h3=base + 19, h4=base + 22,
        p3=base + 16, p4=base + 19, p5=base + 22,
    )


CANDC = (
    "# candC — M4-preserving minimal residual Depth adapter\n"
    "ch: 5\nnc: 12\n" + _SCALES +
    "backbone:\n"
    "  - [-1, 1, Silence, []] # 0\n"
    "  - [0, 1, SilenceChannel, [0, 4]] # 1  -> [B,G,R,IR] == M4 input\n"
    "  - [-1, 1, Conv, [64, 3, 2]] # 2  <- M4 stem\n"
    "  - [0, 1, SilenceChannel, [4, 5]] # 3  -> Depth 1ch\n"
    "  - [-1, 1, ZeroConv2d, [64, 3, 2, 1]] # 4  zero-init adapter\n"
    "  - [[2, 4], 1, ADD, [1.0]] # 5  fused = M4_stem + 0\n"
    + _body_tail(5)
    + _head_tail(5)
)

NEG_A1 = CANDC.replace(
    "  - [[2, 4], 1, ADD, [1.0]] # 5  fused = M4_stem + 0\n",
    "  - [[2, 4], 1, ADD, [1.0]] # 5  fused = M4_stem + 0\n"
    "  - [-1, 1, nn.Identity, []] # 5b injected extra layer -> body offset now misaligned\n",
)
NEG_A2 = CANDC.replace("[0, 1, SilenceChannel, [4, 5]] # 3  -> Depth 1ch",
                       "[0, 1, SilenceChannel, [3, 4]] # 3  WRONG: this slice is IR, not Depth")


def git(*a: str) -> str:
    r = subprocess.run([GIT, "-C", str(ROOT), *a], capture_output=True, text=True)
    return (r.stdout or r.stderr).strip()


def sha(p: Path, n: int = 16) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:n] if p.exists() else "MISSING"


def build_pre_p1_module(dst: Path) -> Path:
    """Reconstruct the pre-P1 `train.py` by removing the three P1 insertions."""
    src = (ROOT / "ultralytics/models/yolo/detect/train.py").read_text(encoding="utf-8")
    markers = [
        ("# ============================================================\n"
         "# Residual-Depth adapter remap (2026-10-04, P1)",
         "def _transfer_rgb_pretrained(model, weights):"),
        ("    # ---- residual-Depth adapter (M4 stem + zero-init Depth + ADD) ----",
         "    # ---- early fusion, single stem absorbing N channels"),
        ("    # ---- fail-closed guard: a Silence-prefixed layout with a >3ch Conv stem ----",
         "    if not any(getattr(m.conv, \"in_channels\", None) == 1 for m in stems):"),
        ("        # Fail-closed: this branch writes the stem and returns, and it can only",
         "        n_aux = multi_ch.conv.weight.shape[1] - 3"),
    ]
    for start, end in markers:
        i = src.find(start)
        j = src.find(end)
        assert i != -1 and j != -1 and i < j, f"marker not found: {start[:48]!r}"
        src = src[:i] + src[j:]
    dst.write_text(src, encoding="utf-8")
    return dst


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def transfer(cfg_path, stock, fn=None):
    """Construct + load + transfer. Returns (net, n_ret, error)."""
    net = YOLO(str(cfg_path)).model
    net.load(stock)
    fn = fn or TR._transfer_rgb_pretrained
    try:
        n = fn(net, stock)
        return net, n, None
    except Exception as e:  # noqa: BLE001
        return net, None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="diagnostic/p1_remap_audit")
    ap.add_argument("--workdir", default=None)
    A = ap.parse_args()
    out = ROOT / A.out
    out.mkdir(parents=True, exist_ok=True)
    work = Path(A.workdir) if A.workdir else Path(tempfile.gettempdir()) / "f3m_audit"
    work.mkdir(parents=True, exist_ok=True)

    assert not torch.cuda.is_available(), "P1 must run CPU-only"
    REPORT: dict = {"CUDA_USED": CUDA_USED, "THRESHOLD": THRESHOLD}

    # ---------------- 1. baseline ----------------
    print("=" * 104)
    print("§1  BASELINE")
    print("=" * 104)
    files = [
        "ultralytics/models/yolo/detect/train.py", "ultralytics/nn/tasks.py",
        "ultralytics/data/base.py", "ultralytics/data/loaders.py", "ultralytics/data/augment.py",
        "ultralytics/cfg/default.yaml", "scripts/train.py", "scripts/predict_rect.py",
        "scripts/official_eval.py", "scripts/validate_modality_contract.py",
        "scripts/pretrained_coverage.py", "configs/yolo11m_sepstem.yaml",
        "configs/yolo11m_earlyfusion.yaml", "configs/yolo11m_modality4ch.yaml",
        "configs/yolo11m_modality3ch.yaml", "configs/yolo11m_modality2ch.yaml", "yolo11m.pt",
    ]
    sha_tbl = {f: sha(ROOT / f) for f in files}
    for f, h in sha_tbl.items():
        print(f"  {h}  {f}")
    REPORT["baseline"] = {"HEAD": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                          "sha256_16": sha_tbl,
                          "gitignored": [f for f in files
                                         if git("check-ignore", f)]}
    print(f"  HEAD={REPORT['baseline']['HEAD']}")
    print(f"  GITIGNORED (invisible to git status): {REPORT['baseline']['gitignored']}")

    # ---------------- 2. prototypes ----------------
    protos = {"candC": CANDC, "candC_negA1_extra_layer": NEG_A1, "candC_negA2_wrong_depth_slice": NEG_A2}
    paths = {}
    for nm, txt in protos.items():
        p = work / f"yolo11m_{nm}.yaml"
        p.write_text(txt, encoding="utf-8")
        paths[nm] = p
    print(f"\n  prototypes written to {work}  (NOT promoted into configs/)")
    REPORT["prototypes"] = {k: str(v) for k, v in paths.items()}

    # ---------------- 3. pre-P1 reference ----------------
    pre = load_module(build_pre_p1_module(work / "train_pre_p1.py"), "train_pre_p1")
    print(f"  pre-P1 reference module built: {work / 'train_pre_p1.py'}")

    stock = load_stock(str(ROOT / "yolo11m.pt"))
    src = stock.state_dict()

    # ---------------- 4. coverage table ----------------
    print("\n" + "=" * 104)
    print("§4  MAPPING COVERAGE  (exact_match_ratio over expected pretrained tensors)")
    print("=" * 104)
    MODELS = [
        ("D", ROOT / "configs/yolo11m_earlyfusion.yaml"),
        ("D'", ROOT / "configs/yolo11m_sepstem.yaml"),
        ("M4", ROOT / "configs/yolo11m_modality4ch.yaml"),
        ("candC", paths["candC"]),
    ]
    cov_rows = {}
    print(f"  {'model':<7}{'family':<26}{'expected':>9}{'exact':>7}{'cov_tensor':>11}{'cov_param':>11}  gate")
    for nm, p in MODELS:
        net, n_ret, err = transfer(p, stock)
        if err:
            print(f"  {nm:<7}RAISED  {err[:70]}")
            cov_rows[nm] = {"error": err}
            continue
        rep = measure(net, src, THRESHOLD)
        cov_rows[nm] = {k: v for k, v in rep.items() if k != "_records"}
        cov_rows[nm]["n_ret"] = n_ret
        print(f"  {nm:<7}{rep['family']:<26}{rep['n_expected']:>9}{rep['n_exact']:>7}"
              f"{rep['coverage_tensors']:>11.4f}{rep['coverage_params']:>11.4f}  {'PASS' if rep['ok'] else 'FAIL'}")
        if not rep["ok"]:
            print(f"          failed: {[k for k, v in rep['checks'].items() if not v]}")
    REPORT["coverage"] = cov_rows

    # ---------------- 5. manifest + critical layers ----------------
    print("\n" + "=" * 104)
    print("§5/§6  MANIFEST + CRITICAL LAYER AUDIT  (candC)")
    print("=" * 104)
    net_c, _, err = transfer(paths["candC"], stock)
    assert err is None, err
    recs = manifest(net_c, src)
    (out / "mapping_manifest_candC.json").write_text(
        json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
    rep_c = measure(net_c, src, THRESHOLD)
    for r in recs:
        r["group"] = ("stem" if r["kind"] == KIND_DERIVED else
                      "Depth ZeroConv" if r["kind"] == KIND_ZERO else
                      "new (Detect classifier)" if r["kind"] == KIND_NEW else "backbone/neck/head")
    from collections import Counter
    print(f"  manifest rows = {len(recs)}  -> {out / 'mapping_manifest_candC.json'}")
    print(f"  {'group':<24}{'kind':<14}{'tensors':>9}{'params':>12}")
    for (g, k), c in sorted(Counter((r["group"], r["kind"]) for r in recs).items()):
        n = sum(r["numel"] for r in recs if r["group"] == g and r["kind"] == k)
        print(f"  {g:<24}{k:<14}{c:>9}{n:>12,}")
    print(f"  {'TOTAL':<24}{'':<14}{len(recs):>9}{sum(r['numel'] for r in recs):>12,}")
    print("\n  critical layers:")
    for n in rep_c["stem_notes"]:
        print(f"    · {n}")
    zc = [r for r in recs if r["kind"] == KIND_ZERO]
    zc_mod = net_c.model[4]
    print(f"    · Depth ZeroConv2d: tensors={len(zc)} "
          f"max|W|={float(zc_mod.weight.abs().max()):.1e} "
          f"max|b|={float(zc_mod.bias.abs().max()) if zc_mod.bias is not None else float('nan'):.1e} "
          f"status={[r['status'] for r in zc]}")
    print(f"    · ADD: type={type(net_c.model[5]).__name__} learnable_params="
          f"{sum(p.numel() for p in net_c.model[5].parameters())} alpha={net_c.model[5].a}")
    REPORT["manifest_rows"] = len(recs)
    REPORT["stem_notes"] = rep_c["stem_notes"]
    REPORT["zero_conv"] = dict(
        tensors=len(zc), max_abs_weight=float(zc_mod.weight.abs().max()),
        max_abs_bias=(float(zc_mod.bias.abs().max()) if zc_mod.bias is not None else None),
        status=[r["status"] for r in zc])
    REPORT["add_layer"] = dict(type=type(net_c.model[5]).__name__,
                               params=sum(p.numel() for p in net_c.model[5].parameters()),
                               alpha=net_c.model[5].a)

    # ---------------- 6. identity at initialization ----------------
    print("\n" + "=" * 104)
    print("§7  IDENTITY-AT-INITIALIZATION PROBE  (CPU, 64x64 dummy input, no CUDA)")
    print("=" * 104)
    m4 = YOLO(str(ROOT / "configs/yolo11m_modality4ch.yaml")).model
    m4.load(stock)
    TR._transfer_rgb_pretrained(m4, stock)
    m4.eval(); net_c.eval()
    g = torch.Generator().manual_seed(0)
    x5 = torch.randn(1, 5, 64, 64, generator=g)
    with torch.no_grad():
        m4_stem = m4.model[0](x5[:, :4])
        t = net_c.model[0](x5)
        rgbir = net_c.model[1](t)
        dep = net_c.model[3](t)
        c_stem = net_c.model[2](rgbir)
        z_out = net_c.model[4](dep)
        fused = net_c.model[5]([c_stem, z_out])

    def cmp(a, b, tag):
        d = (a - b).abs()
        rel = float(d.max() / (b.abs().max() + 1e-12))
        print(f"  {tag:<38} max={float(d.max()):.3e} mean={float(d.mean()):.3e} rel={rel:.3e} "
              f"{'EXACT' if torch.equal(a, b) else 'DIFFERS'}")
        return dict(max_abs=float(d.max()), mean_abs=float(d.mean()), rel=rel, exact=bool(torch.equal(a, b)))

    print(f"  depth adapter output: max|z| = {float(z_out.abs().max()):.3e}  "
          f"(depth input max|d| = {float(dep.abs().max()):.3f})")
    r_layers = dict(
        stem_vs_m4=cmp(c_stem, m4_stem, "candC M4 stem vs M4 model.0"),
        fused_vs_m4=cmp(fused, m4_stem, "candC fused(ADD) vs M4 stem"),
        zero_vs_zeros=cmp(z_out, torch.zeros_like(z_out), "ZeroConv2d(Depth) vs 0"),
    )
    r_layers["status"] = "PASS" if (r_layers["fused_vs_m4"]["exact"] and r_layers["zero_vs_zeros"]["exact"]) else "FAIL"
    print(f"  IDENTITY_AT_INIT = {r_layers['status']}")
    REPORT["identity"] = r_layers

    # ---------------- 7. negative / fail-closed ----------------
    print("\n" + "=" * 104)
    print("§8  NEGATIVE / FAIL-CLOSED TESTS")
    print("=" * 104)
    neg = {}

    # A1: injected extra layer -> body offset misaligned
    _, n_ret, err = transfer(paths["candC_negA1_extra_layer"], stock)
    neg["A1 extra layer after ADD"] = dict(expected="RAISE", observed=("RAISE:" + err[:90]) if err else f"NO RAISE (n_ret={n_ret})",
                                           passed=bool(err))
    # A2: depth fed by the IR slice
    _, n_ret, err = transfer(paths["candC_negA2_wrong_depth_slice"], stock)
    neg["A2 depth fed by IR slice"] = dict(expected="RAISE", observed=("RAISE:" + err[:90]) if err else f"NO RAISE (n_ret={n_ret})",
                                           passed=bool(err))
    # B1/B2: delete an expected source mapping
    for tag, drop in (("B1 delete stock stem", "model.0.conv.weight"),
                      ("B2 delete mid-body tensor", "model.10.cv1.conv.weight")):
        s2 = dict(src)
        s2.pop(drop, None)
        net2 = YOLO(str(paths["candC"])).model
        net2.load(stock)
        try:
            TR._transfer_rgb_pretrained(net2, s2)
            neg[tag] = dict(expected="RAISE", observed="NO RAISE", passed=False)
        except Exception as e:  # noqa: BLE001
            neg[tag] = dict(expected="RAISE", observed=f"RAISE: {type(e).__name__}: {str(e)[:80]}", passed=True)
    # C: run the PRE-P1 dispatcher on candC -- reproduces the original silent
    # degradation (no raise, stem-only transfer). The gate must reject it.
    net3, n_ret3, err3 = transfer(paths["candC"], stock, fn=pre._transfer_rgb_pretrained)
    if err3 is None:
        rep3 = measure(net3, src, THRESHOLD)
        neg["C pre-P1 dispatcher on candC"] = dict(
            expected="silent (no raise) but GATE FAIL",
            observed=f"no raise, n_ret={n_ret3}, coverage_t={rep3['coverage_tensors']:.4f}, "
                     f"gate={'PASS' if rep3['ok'] else 'FAIL'}",
            passed=bool(n_ret3 == 5 and (not rep3["ok"]) and rep3["coverage_tensors"] < THRESHOLD))
    else:
        neg["C pre-P1 dispatcher on candC"] = dict(expected="silent (no raise) but GATE FAIL",
                                                   observed=f"unexpected raise: {err3[:80]}", passed=False)
    # C2: new code but the new branch disabled -> the multi-ch fail-closed guard must catch it
    saved = TR._find_residual_depth_adapter
    TR._find_residual_depth_adapter = lambda m: None
    try:
        _n4, n_ret4, err4 = transfer(paths["candC"], stock)
    finally:
        TR._find_residual_depth_adapter = saved
    neg["C2 new code, branch disabled"] = dict(
        expected="RAISE (multi-ch fail-closed guard)",
        observed=(f"RAISE: {err4[:80]}" if err4 else f"NO RAISE (n_ret={n_ret4})"),
        passed=bool(err4))
    for k, v in neg.items():
        print(f"  {'✅' if v['passed'] else '❌'} {k:<38} expect={v['expected']:<36} got={v['observed']}")
    REPORT["negative_tests"] = neg

    # ---------------- 8. regression scan ----------------
    print("\n" + "=" * 104)
    print("§9  REGRESSION SCAN — every model config in configs/, pre-P1 vs post-P1")
    print("=" * 104)
    yamls = sorted((ROOT / "configs").glob("yolo11*.yaml"))
    rows, regressions = [], []
    print(f"  {'config':<46}{'before n_ret':>13}{'after n_ret':>12}{'before cov':>12}{'after cov':>11}  Δ")
    for p in yamls:
        try:
            yaml_peek = p.read_text(encoding="utf-8", errors="ignore")
            if "backbone:" not in yaml_peek or "head:" not in yaml_peek:
                continue
        except Exception:  # noqa: BLE001
            continue
        n_b, nb, eb = transfer(p, stock, fn=pre._transfer_rgb_pretrained)
        n_a, na, ea = transfer(p, stock)
        cb = measure(n_b, src, THRESHOLD)["coverage_tensors"] if not eb else float("nan")
        ca = measure(n_a, src, THRESHOLD)["coverage_tensors"] if not ea else float("nan")
        same = (nb == na) and (eb == ea) and (str(cb) == str(ca) or (cb != cb and ca != ca))
        rows.append(dict(cfg=p.name, before_n=nb, after_n=na, before_cov=cb, after_cov=ca,
                         before_err=eb, after_err=ea, unchanged=same))
        if not same:
            regressions.append(rows[-1])
        print(f"  {p.name:<46}{str(nb):>13}{str(na):>12}"
              f"{(f'{cb:.4f}' if cb == cb else 'n/a'):>12}{(f'{ca:.4f}' if ca == ca else 'n/a'):>11}"
              f"  {'=' if same else '⚠ CHANGED'}")
    print(f"\n  scanned={len(rows)}  regressions={len(regressions)}")
    REPORT["regression"] = dict(scanned=len(rows), regressions=regressions)
    REPORT["regression_rows"] = rows

    # ---------------- verdict ----------------
    gate_all = (
        all((cov_rows.get(m, {}).get("ok", False)) for m in ("D", "D'", "M4", "candC"))
        and r_layers["status"] == "PASS"
        and all(v["passed"] for v in neg.values())
        and not regressions
        and CUDA_USED == "NO"
    )
    REPORT["P1_STATUS"] = "PASS" if gate_all else "FAIL"
    REPORT["3M_READY"] = "YES" if gate_all else "NO"
    (out / "p1_audit.json").write_text(json.dumps(REPORT, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n" + "=" * 104)
    print(f"P1_STATUS = {REPORT['P1_STATUS']}   3M_READY = {REPORT['3M_READY']}   CUDA_USED = {CUDA_USED}")
    print(f"[json] {out / 'p1_audit.json'}")
    return 0 if gate_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
