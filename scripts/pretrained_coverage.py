"""scripts/pretrained_coverage.py — pretrained-transfer coverage: measurement, not bookkeeping.

Why this exists (2026-10-04, P1)
-------------------------------
``_transfer_rgb_pretrained`` used to be validated with ``n_ret > 0 and stem_changed``.
Both are satisfiable while ~90% of the network is still randomly initialised:
the ``multi_ch`` branch writes the input stem and returns, so any layout whose
Silence prefix shifts the module indices scores ``n_ret = 5`` and ``stem_changed = True``
with a body that never received a single pretrained tensor (measured: 9.7%).

This module answers the only question that matters instead:

    For every target parameter tensor, is it bit-equal to the stock tensor it is
    **supposed** to have come from?

The intended source is re-derived here from the model's own yaml + module graph,
independently of the remap implementation, so a bug in the remap cannot hide by
also being a bug in its self-report.

Pure CPU. No forward pass unless the caller asks for one. No training.

Public API
----------
``analyse(net)``   -> layout description (family, offset, stem roles, zero/detect idx)
``manifest(net, src)`` -> per-tensor records (target, source, shape, reason, status)
``measure(net, src)``  -> summary dict (ratios, gate booleans)
``gate(model, src, threshold)`` -> (ok, report) fail-closed verdict
"""
from __future__ import annotations

from typing import Any

import torch

FAMILY_SINGLE = "single_stem"  # one Conv stem absorbing all modalities (D, M4, M3, ...)
FAMILY_SEPSTEM = "separate_stem"  # 1x3ch + 2x1ch stems + Concat (D')
FAMILY_RESIDUAL_DEPTH = "residual_depth_adapter"  # 4ch M4 stem + ZeroConv2d(Depth) + ADD
FAMILY_RGBD_TABLE = "rgbd_concat_res_table"  # hard-coded index table
FAMILY_UNKNOWN = "unknown"

KIND_EXACT = "PRETRAINED"  # must be bit-equal to a named stock tensor
KIND_DERIVED = "DERIVED"  # built from stock (stem slices), checked separately
KIND_ZERO = "ZERO_INIT"  # must be exactly zero
KIND_NEW = "NEW_RANDOM"  # legitimately fresh (Detect cv3 for a new nc)
KIND_IGNORE = "IGNORED"  # scalar / non-parametric

# grouping used by the human-readable report
GROUP_OF_KIND = {
    KIND_EXACT: "backbone/neck/head",
    KIND_DERIVED: "stem",
    KIND_ZERO: "Depth ZeroConv",
    KIND_NEW: "new (Detect cv3)",
    KIND_IGNORE: "other",
}


def _ctype(m) -> str:
    return type(m).__name__


def _layers(net) -> list:
    return list((net.yaml or {}).get("backbone", [])) + list((net.yaml or {}).get("head", []))


def _conv_stems(net) -> list:
    from ultralytics.nn.modules.conv import Conv

    return [m for m in net.model if isinstance(m, Conv)]


def analyse(net) -> dict[str, Any]:
    """Re-derive the fusion layout from the module graph. Never guesses silently."""
    mods = list(net.model)
    layers = _layers(net)
    stems = _conv_stems(net)
    idx_of = {id(m): i for i, m in enumerate(mods)}

    zero_idx = [i for i, m in enumerate(mods) if _ctype(m) == "ZeroConv2d"]
    n_in3 = [m for m in stems if getattr(m.conv, "in_channels", None) == 3]
    n_in1 = [m for m in stems if getattr(m.conv, "in_channels", None) == 1]
    n_wide = [m for m in stems if (getattr(m.conv, "in_channels", None) or 0) > 3]
    n_narrow = [m for m in stems if getattr(m.conv, "in_channels", None) in (1, 2)]

    info: dict[str, Any] = dict(
        family=FAMILY_UNKNOWN,
        offset=None,
        stems={},
        zero_idx=zero_idx[0] if len(zero_idx) == 1 else None,
        detect_idx=len(layers) - 1,
        nc=int(getattr(net.model[-1], "nc", -1)),
        n_layers=len(layers),
        reasons=[],
    )

    # ---- residual-Depth adapter: 1 ZeroConv2d + ADD fed by it ----
    if len(zero_idx) == 1:
        zi = zero_idx[0]
        adds = [
            i
            for i, m in enumerate(mods)
            if _ctype(m) == "ADD" and isinstance(layers[i][0], (list, tuple)) and zi in layers[i][0]
        ]
        if len(adds) == 1:
            ai = adds[0]
            other = [f for f in layers[ai][0] if f != zi]
            if len(other) == 1:
                si = int(other[0])
                info.update(family=FAMILY_RESIDUAL_DEPTH, offset=ai)
                info["stems"] = {si: "derived"}
                info["zero_idx"] = zi
                info["reasons"].append(
                    f"ZeroConv2d@L{zi} + ADD@L{ai} fed by [L{si}, L{zi}] -> residual-Depth adapter; "
                    f"body L{ai + 1}.. <- stock 1.."
                )
                return info
        info["family"] = FAMILY_UNKNOWN
        info["reasons"].append(f"ZeroConv2d@L{zi} present but no unique ADD consumes it")
        return info

    # ---- separate-stem: exactly 1x3ch + 2x1ch, fused by a 3-input Concat ----
    if len(n_in3) == 1 and len(n_in1) == 2:
        cat = None
        for i, lyr in enumerate(layers):
            if str(lyr[2]) == "Concat" and isinstance(lyr[0], (list, tuple)) and len(lyr[0]) == 3:
                cat = i
                break
        if cat is not None:
            info.update(family=FAMILY_SEPSTEM, offset=cat)
            info["stems"] = {
                idx_of[id(n_in3[0])]: "derived_rgb",
                idx_of[id(n_in1[0])]: "derived_aux",
                idx_of[id(n_in1[1])]: "derived_aux",
            }
            info["reasons"].append(
                f"1x3ch + 2x1ch stems fused by 3-input Concat@L{cat} -> separate-stem; body L{cat + 1}.. <- stock 1.."
            )
            return info

    # ---- single stem absorbing every modality (D / M1 / M3 / M4 / M5 / M6) ----
    # `offset = 0` so that layer i maps to stock layer i; the stem itself is just
    # layer 0 and is handled by the body rule unless it is a multi-channel stem
    # (which needs the derived-slice check instead of verbatim equality).
    if stems and not n_in1 and len(n_in3) <= 1:
        first = stems[0]
        fi = idx_of[id(first)]
        fin = getattr(first.conv, "in_channels", None) or 0
        info.update(family=FAMILY_SINGLE, offset=0)
        info["stems"] = {} if fin == 3 else {fi: "derived"}
        info["reasons"].append(
            f"single {fin}ch Conv stem@L{fi} -> single-stem family; every layer Lk <- stock Lk"
            + ("" if fin == 3 else " (stem checked as derived slices)")
        )
        return info

    # ---- RGBD concat_res hard-coded index table ----
    if n_in1 or n_wide or n_narrow:
        info["family"] = FAMILY_RGBD_TABLE
        info["reasons"].append(
            "layout has 1ch/narrow/wide Conv stems but matches none of single/sepstem/residual-depth "
            "-> would be handled by the hard-coded RGBD concat_res index table (NOT verifiable here)"
        )
    return info


def _stem_check(net, src, info) -> tuple[bool, list[str]]:
    """Verify the derived stem tensors slice-by-slice against stock model.0."""
    mods = list(net.model)
    notes: list[str] = []
    ok = True
    sw = src.get("model.0.conv.weight")
    if sw is None:
        return False, ["stock model.0.conv.weight missing"]
    for li, role in info["stems"].items():
        stem = mods[li]
        t = stem.conv.weight
        out = t.shape[0]
        cin = t.shape[1]
        if role == "derived_rgb" and cin != 3:
            ok = False
            notes.append(f"L{li} ({role}): in_channels={cin} != 3 for an RGB stem")
            continue
        if out > sw.shape[0]:
            ok = False
            notes.append(f"L{li} ({role}): out={out} exceeds stock out={sw.shape[0]}")
            continue
        with torch.no_grad():
            if cin == 3:
                # verbatim stock stem (separate-stem RGB branch)
                good = torch.equal(t, sw[:out])
                why = "ch == stock RGB stem"
            elif cin > 3:
                # single-stem multi-channel: ch[0:3] verbatim, aux channels = mean(R,G,B)
                good = torch.equal(t[:, :3], sw[:out]) and torch.equal(
                    t[:, 3:], sw[:out].float().mean(dim=1, keepdim=True).repeat(1, cin - 3, 1, 1).to(t.dtype)
                )
                why = "ch[0:3] == stock RGB, ch[3:] == mean(R,G,B)"
            else:
                exp = sw[:out].float().mean(dim=1, keepdim=True).repeat(1, cin, 1, 1).to(t.dtype)
                good = torch.equal(t, exp)
                why = "all ch == mean(R,G,B)"
        ok = ok and bool(good)
        notes.append(f"L{li} ({role}, in={cin}, out={out}): {'OK' if good else 'MISMATCH'} [{why}]")
        notes.append(f"L{li} ({role}): {'OK' if good else 'MISMATCH'}")
        # BN must come from stock model.0.bn (prefix-sliced)
        for bk, bv in stem.bn.state_dict().items():
            sv = src.get(f"model.0.bn.{bk}")
            if sv is None:
                ok = False
                notes.append(f"L{li} bn.{bk}: no stock counterpart")
                continue
            expv = sv if sv.ndim == 0 else sv[:out]
            if not torch.equal(bv, expv):
                ok = False
                notes.append(f"L{li} bn.{bk}: MISMATCH vs stock model.0.bn.{bk}")
    return ok, notes


def manifest(net, src) -> list[dict[str, Any]]:
    """Per-tensor intended mapping + verification. Returns one record per non-scalar target tensor."""
    info = analyse(net)
    mods = list(net.model)
    layers = _layers(net)
    offset = info["offset"]
    stem_idx = set(info["stems"])
    zero_idx = info["zero_idx"]
    detect_idx = info["detect_idx"]
    nc = info["nc"]
    sd = net.state_dict()

    # index every stock tensor by (shape, bytes) so we can detect mis-mappings
    stock_index: dict[tuple, list[tuple[str, torch.Tensor]]] = {}
    for k, v in src.items():
        if hasattr(v, "shape") and v.ndim > 0:
            stock_index.setdefault(tuple(v.shape), []).append((k, v))

    recs: list[dict[str, Any]] = []
    for k, v in sd.items():
        if v.ndim == 0:
            continue
        parts = k.split(".", 2)
        li = int(parts[1]) if len(parts) == 3 and parts[0] == "model" else -1
        suffix = parts[2] if len(parts) == 3 else k
        rec = dict(target_name=k, shape=tuple(v.shape), numel=v.numel(), layer=li)

        if zero_idx is not None and li == zero_idx:
            rec.update(kind=KIND_ZERO, source_name=None, mapping_reason="Depth residual adapter (ZeroConv2d)")
            rec["status"] = "ZERO_OK" if torch.count_nonzero(v).item() == 0 else "ZERO_VIOLATED"
        elif li in stem_idx:
            rec.update(kind=KIND_DERIVED, source_name="model.0.*", mapping_reason=f"stem ({info['stems'][li]})")
            rec["status"] = "DERIVED"
        elif offset is not None and li > offset:
            s = li - offset
            sk = f"model.{s}." + suffix
            sv = src.get(sk)
            rec.update(source_name=sk, mapping_reason=f"body/neck/head: L{li} <- stock L{s}")
            # Only Detect's *classifier* output layer is legitimately fresh; its
            # preceding 3x3 convs (cv3.0.0.0 etc.) are transferred like any other
            # layer. Mirror the remap's own condition rather than blanket-tagging
            # every ".cv3." key -- blanket tagging mis-flagged 30 tensors in D'.
            _cv3_fresh = (
                li == detect_idx
                and ".cv3." in k
                and sv is not None
                and tuple(sv.shape[1:]) == tuple(v.shape[1:])
                and v.shape[0] == nc
                and sv.shape[0] != nc
            )
            if _cv3_fresh:
                rec.update(kind=KIND_NEW, mapping_reason=f"Detect classifier re-initialised for nc={nc}")
                rec["status"] = "NEW_RANDOM"
            elif sv is None:
                rec.update(kind=KIND_EXACT, status="UNRESOLVED_SOURCE")
            elif tuple(sv.shape) != tuple(v.shape):
                rec.update(kind=KIND_EXACT, status="SHAPE_MISMATCH")
            else:
                rec.update(kind=KIND_EXACT)
                rec["status"] = "EXACT" if torch.equal(v, sv) else "NOT_EQUAL"
        else:
            rec.update(kind=KIND_IGNORE, source_name=None, mapping_reason="fusion plumbing (no params)", status="IGNORED")

        # unexpected mapping: a tensor that is NOT derived from stock by rule, yet is
        # a verbatim copy of one. DERIVED tensors are excluded -- a stock prefix IS
        # their expected value, and _stem_check verifies them slice by slice.
        if rec["kind"] in (KIND_NEW, KIND_ZERO) and torch.count_nonzero(v).item() != 0:
            hits = [n for n, c in stock_index.get(tuple(v.shape), ()) if torch.equal(v, c)]
            if hits:
                rec["status"] = f"UNEXPECTED_STOCK_COPY:{hits[0]}"
        recs.append(rec)
    return recs


def measure(net, src, threshold: float = 0.80) -> dict[str, Any]:
    """Coverage summary + fail-closed gate booleans."""
    info = analyse(net)
    recs = manifest(net, src)
    exact = [r for r in recs if r["kind"] == KIND_EXACT]
    n_exact = sum(1 for r in exact if r["status"] == "EXACT")
    numel_exact = sum(r["numel"] for r in exact if r["status"] == "EXACT")
    numel_expected = sum(r["numel"] for r in exact)

    stem_ok, stem_notes = _stem_check(net, src, info)
    zero_recs = [r for r in recs if r["kind"] == KIND_ZERO]
    zero_ok = all(r["status"] == "ZERO_OK" for r in zero_recs)

    shape_mismatch = [r["target_name"] for r in exact if r["status"] == "SHAPE_MISMATCH"]
    unresolved = [r["target_name"] for r in exact if r["status"] == "UNRESOLVED_SOURCE"]
    unexpected = [r["target_name"] for r in recs if str(r.get("status", "")).startswith("UNEXPECTED_STOCK_COPY")]
    not_equal = [r["target_name"] for r in exact if r["status"] == "NOT_EQUAL"]

    ratio = (n_exact / len(exact)) if exact else float("nan")
    ratio_p = (numel_exact / numel_expected) if numel_expected else float("nan")

    checks = {
        "pretrained_exact_match_ratio>=threshold": bool(exact) and ratio >= threshold,
        "critical_stem_mapping": bool(stem_ok),
        "zero_init_exact": bool(zero_ok) or not zero_recs,
        "shape_mismatch==0": not shape_mismatch,
        "unresolved_expected_mapping==0": not unresolved,
        "unexpected_mapping==0": not unexpected,
        "not_equal==0": not not_equal,
    }
    return dict(
        family=info["family"],
        reasons=info["reasons"],
        threshold=threshold,
        n_expected=len(exact),
        n_exact=n_exact,
        coverage_tensors=ratio,
        coverage_params=ratio_p,
        counts={
            "PRETRAINED": len(exact),
            "DERIVED": sum(1 for r in recs if r["kind"] == KIND_DERIVED),
            "ZERO_INIT": len(zero_recs),
            "NEW_RANDOM": sum(1 for r in recs if r["kind"] == KIND_NEW),
            "other": sum(1 for r in recs if r["kind"] == KIND_IGNORE),
        },
        problems=dict(
            shape_mismatch=shape_mismatch[:20],
            unresolved=unresolved[:20],
            unexpected=unexpected[:20],
            not_equal=not_equal[:20],
        ),
        stem_notes=stem_notes,
        checks=checks,
        ok=all(checks.values()),
        _records=recs,
    )


def load_stock(path: str = "yolo11m.pt"):
    """Load the stock checkpoint the way the trainer does (``attempt_load_one_weight`` prefers EMA)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return (ck.get("ema") or ck["model"]).float()


def gate(cfg_path: str, stock, threshold: float = 0.80) -> tuple[bool, dict[str, Any]]:
    """Construct a model from ``cfg_path``, run the real transfer path, then measure.

    ``stock`` is the stock module (preferred) or its state_dict.
    """
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import _transfer_rgb_pretrained

    src = stock.state_dict() if hasattr(stock, "state_dict") else stock
    net = YOLO(str(cfg_path)).model
    try:
        net.load(stock if hasattr(stock, "state_dict") else {"model": stock})
        _transfer_rgb_pretrained(net, stock)
    except Exception as e:  # noqa: BLE001
        return False, dict(
            family="RAISED",
            error=f"{type(e).__name__}: {e}",
            ok=False,
            checks={"transfer_completed": False},
            coverage_tensors=float("nan"),
            coverage_params=float("nan"),
        )
    rep = measure(net, src, threshold=threshold)
    rep["checks"]["transfer_completed"] = True
    rep["ok"] = all(rep["checks"].values())
    return rep["ok"], rep


if __name__ == "__main__":
    import json
    import sys

    cfg = sys.argv[1]
    stock = load_stock("yolo11m.pt")
    ok, rep = gate(cfg, stock)
    rep.pop("_records", None)
    print(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    print(f"\nPRETRAINED_GATE = {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)
