# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
from copy import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


def _remap_separate_stem(model, src, rgb_stem, aux_stems):
    """Explicit pretrained remap for the Separate-Stem fusion layout (3ch + 1ch + 1ch).

    Layout (see ``configs/yolo11m_sepstem.yaml``)::

        Silence -> SilenceChannel[0,3] -> Conv(3->C_rgb) -\
                -> SilenceChannel[3,4] -> Conv(1->C_ir)   >- Concat -> stock YOLO11 backbone
                -> SilenceChannel[4,5] -> Conv(1->C_d)  -/

    Deterministic rules — no index guessing, no silent fallback:

    * **RGB stem** (in=3, out=C_rgb) <- stock stem ``model.0`` **output-channel prefix**
      ``[:C_rgb]`` (conv weight and all BN params).
    * **IR / Depth stems** (in=1, out=C_*) <- grayscale-reduced stock stem
      ``mean(stock_w[:, 0:3], dim=1, keepdim=True)`` then output-channel prefix ``[:C_*]``.
      Which aux stem is IR vs Depth is read from the ``c_start`` of the ``SilenceChannel``
      that *feeds* it (3 = infrared, 4 = depth) — never from module position.
    * Every layer after the fusion ``Concat`` is copied from stock layer
      ``(i - concat_idx)``; ``concat_idx`` is **derived from the yaml**, not hardcoded.

    Raises ``RuntimeError`` on any missing / mismatched / ambiguous item. This function
    never leaves a tensor silently randomly initialised.
    """
    layers = list(model.yaml.get("backbone", [])) + list(model.yaml.get("head", []))

    # ---- locate the fusion Concat (the three-input one) ----
    cat_idx = None
    for i, lyr in enumerate(layers):
        if str(lyr[2]) == "Concat" and isinstance(lyr[0], (list, tuple)) and len(lyr[0]) == 3:
            cat_idx = i
            break
    if cat_idx is None:
        raise RuntimeError(
            "[sepstem remap] could not locate a 3-input fusion Concat in the model yaml; refusing to guess"
        )
    offset = cat_idx  # candidate layer i (i > offset)  <->  stock layer (i - offset)

    idx_of = {id(m): i for i, m in enumerate(model.model)}

    # ---- the Concat must exactly reference the three stem layers ----
    expected_from = sorted(idx_of[id(m)] for m in [rgb_stem] + list(aux_stems))
    got_from = sorted(layers[cat_idx][0])
    if got_from != expected_from:
        raise RuntimeError(
            f"[sepstem remap] fusion Concat at layer {cat_idx} references {got_from}, "
            f"but the detected stems are at {expected_from}; refusing to guess"
        )

    # ---- identify IR vs Depth from each aux stem's feeder SilenceChannel ----
    def _feeder_c_start(i):
        fr = layers[i][0]
        j = i - 1 if fr == -1 else fr
        if not isinstance(j, int) or j < 0 or j >= len(layers):
            return None
        mod = model.model[j]
        return int(getattr(mod, "c_start", -1)) if type(mod).__name__ == "SilenceChannel" else None

    ir_stem = depth_stem = None
    for m in aux_stems:
        i = idx_of[id(m)]
        cs = _feeder_c_start(i)
        if cs == 3:
            ir_stem = m
        elif cs == 4:
            depth_stem = m
        else:
            raise RuntimeError(
                f"[sepstem remap] aux stem at layer {i} is not fed by a SilenceChannel with "
                f"c_start in (3, 4) (got {cs}); refusing to guess IR vs Depth"
            )
    if ir_stem is None or depth_stem is None:
        raise RuntimeError("[sepstem remap] failed to identify both IR and Depth stems")

    # ---- stock stem ----
    stock_w = src.get("model.0.conv.weight")
    if stock_w is None:
        raise RuntimeError("[sepstem remap] stock stem 'model.0.conv.weight' not present in pretrained weights")
    if stock_w.shape[1] != 3:
        raise RuntimeError(f"[sepstem remap] expected a 3-channel stock stem, got in_channels={stock_w.shape[1]}")
    stock_out = stock_w.shape[0]

    n = 0
    with torch.no_grad():
        # ---------- the three stems ----------
        for stem, kind in ((rgb_stem, "rgb"), (ir_stem, "ir"), (depth_stem, "depth")):
            li = idx_of[id(stem)]
            out_c = stem.conv.weight.shape[0]
            if out_c > stock_out:
                raise RuntimeError(
                    f"[sepstem remap] {kind} stem out={out_c} exceeds stock stem out={stock_out}"
                )
            if kind == "rgb":
                if stem.conv.in_channels != 3:
                    raise RuntimeError(f"[sepstem remap] RGB stem in_channels={stem.conv.in_channels}, expected 3")
                new_w = stock_w[:out_c].clone()
            else:
                if stem.conv.in_channels != 1:
                    raise RuntimeError(f"[sepstem remap] {kind} stem in_channels={stem.conv.in_channels}, expected 1")
                # Deterministic grayscale reduction, then output-channel prefix.
                # Computed in float32 regardless of the source checkpoint dtype (yolo11m.pt
                # stores fp16), so the result is bit-reproducible and testable.
                new_w = (
                    stock_w[:out_c].float().mean(dim=1, keepdim=True).to(stem.conv.weight.dtype).clone()
                )
            stem.conv.weight.copy_(new_w)
            n += 1
            for bk, bv in stem.bn.state_dict().items():
                sk = f"model.0.bn.{bk}"
                if sk not in src:
                    raise RuntimeError(f"[sepstem remap] stock '{sk}' missing for {kind} stem BN")
                sv = src[sk]
                if sv.ndim == 0:
                    bv.copy_(sv)
                else:
                    if sv.shape[0] < out_c:
                        raise RuntimeError(f"[sepstem remap] stock '{sk}' has {sv.shape[0]} < {out_c} entries")
                    bv.copy_(sv[:out_c])
                n += 1
            LOGGER.info(f"sepstem remap: {kind} stem (layer {li}, out={out_c}) initialised")

        # ---------- everything after the fusion Concat ----------
        csd = model.state_dict()
        detect_idx = len(layers) - 1
        nc = int(getattr(model.model[-1], "nc", -1))
        n_cls_untrained = 0
        n_e1_new = 0
        # E1 layers are read from the model's OWN yaml (the same keys parse_model used), never
        # from the filename or a hardcoded list, so the exception below can only ever fire for a
        # layer the config itself declared.
        e1_layers = {int(x) for x in (getattr(model, "yaml", {}) or {}).get("e1_layers", []) or []}
        with torch.no_grad():
            for i in range(offset + 1, len(layers)):
                s = i - offset
                prefix = f"model.{i}."
                for k, v in csd.items():
                    if not k.startswith(prefix) or v.ndim == 0:
                        continue
                    sk = f"model.{s}." + k[len(prefix) :]
                    sv = src.get(sk)
                    if sv is not None and tuple(sv.shape) == tuple(v.shape):
                        v.copy_(sv)
                        n += 1
                        continue
                    # A SECOND legitimately non-transferable block: the E1 region response gain
                    # (RegionResponseGain) bolted inside a layer listed in the model yaml's
                    # `e1_layers`. It is a NEW module by construction — a stock COCO checkpoint can
                    # never contain a counterpart for it. Constrained to (declared layer index) AND
                    # (`.e1.` submodule) AND (the layer really carries an `e1` attribute), so any
                    # other missing key still raises below.
                    if i in e1_layers and ".e1." in k and getattr(model.model[i], "e1", None) is not None:
                        n_e1_new += 1
                        continue
                    # The other declared exception: Detect's class branch
                    # (cv3). The source checkpoint was trained with a different number of
                    # classes, so its cls head is re-initialised for this dataset — the same
                    # thing stock ultralytics does for every new-nc fine-tune.
                    # Declared explicitly, constrained by shape/nc assertions, and logged;
                    # anything else raises.
                    if (
                        i == detect_idx
                        and ".cv3." in k
                        and sv is not None
                        and tuple(sv.shape[1:]) == tuple(v.shape[1:])
                        and v.shape[0] == nc
                        and sv.shape[0] != nc
                    ):
                        n_cls_untrained += 1
                        continue
                    if sv is None:
                        raise RuntimeError(f"[sepstem remap] no pretrained counterpart for '{k}' (looked for '{sk}')")
                    raise RuntimeError(
                        f"[sepstem remap] shape mismatch for '{k}': candidate {tuple(v.shape)} vs stock {sk} {tuple(sv.shape)}"
                    )

    LOGGER.info(
        f"Separate-stem pretrained remap: {n} tensors transferred "
        f"(3 stems + layers {offset + 1}..{len(layers) - 1} <- stock 1..{len(layers) - 1 - offset}); "
        f"E1 region response gain (layers {sorted(e1_layers) or 'none'}): {n_e1_new} new tensors; "
        f"Detect cv3 class branch re-initialised for nc={nc}: {n_cls_untrained} tensors not transferred"
    )
    return n


# ============================================================
# Residual-Depth adapter remap (2026-10-04, P1)
# ------------------------------------------------------------
# Layout (the "candC" prototype): a 5ch input is split into the M4 input
# ``[B,G,R,IR]`` (4ch) and the raw Depth channel (1ch). The M4 branch keeps a
# **4ch Conv stem**; Depth goes through a **zero-initialised** ``ZeroConv2d`` and
# is ADDed onto the stem output::
#
#     Silence -> SilenceChannel[0,4] -> Conv(4->C)         ---\
#             -> SilenceChannel[4,5] -> ZeroConv2d(1->C)   --- ADD -> stock body
#
# At t=0 the fused tensor is ``M4_stem(x) + 0``, so the initialised network is
# functionally identical to M4 carrying the same pretrained stem.
#
# Why this branch has to exist: this layout contains **no 1ch ``Conv`` stem**, so
# before this branch existed it fell into the ``multi_ch`` branch below -- which
# writes the first Conv only and returns -- leaving the entire index-shifted body
# randomly initialised. Measured pretrained coverage was 9.7% while the old
# contract gate still reported PASS.
#
# Fail-closed by construction: the layout is read from the model yaml, every
# mapping is shape-checked against the stock checkpoint, and anything ambiguous,
# missing or mismatched raises instead of falling back.
# ============================================================


def _find_residual_depth_adapter(model):
    """Locate the ``M4 stem + ZeroConv2d(Depth) + ADD`` layout.

    Returns ``(stem_idx, zc_idx, add_idx, silch_rgb_idx, silch_depth_idx)`` or
    ``None`` when the model has no ``ZeroConv2d`` at all (=> not this family, the
    pre-existing dispatch is untouched). Raises -- never guesses -- if a
    ``ZeroConv2d`` is present but the surrounding fusion is not exactly this
    layout.
    """
    mods = list(model.model)
    layers = list(model.yaml.get("backbone", [])) + list(model.yaml.get("head", []))
    zc = [i for i, m in enumerate(mods) if type(m).__name__ == "ZeroConv2d"]
    if not zc:
        return None  # not this family

    def _raise(msg):
        raise RuntimeError(f"[residual-depth remap] {msg}; refusing to guess")

    if len(zc) != 1:
        _raise(f"expected exactly one ZeroConv2d, found {len(zc)} at layers {zc}")
    zc_idx = zc[0]
    zc_mod = mods[zc_idx]
    if zc_mod.in_channels != 1:
        _raise(f"ZeroConv2d at layer {zc_idx} has in_channels={zc_mod.in_channels}, expected 1")

    # ---- ADD layers whose inputs include the ZeroConv2d ----
    adds = [
        i
        for i, m in enumerate(mods)
        if type(m).__name__ == "ADD" and isinstance(layers[i][0], (list, tuple)) and zc_idx in layers[i][0]
    ]
    if len(adds) != 1:
        _raise(f"expected exactly one ADD fed by the ZeroConv2d, found {len(adds)} at layers {adds}")
    add_idx = adds[0]

    # ---- the other ADD input must be the M4 (4ch) stem ----
    other = [f for f in layers[add_idx][0] if f != zc_idx]
    if len(other) != 1:
        _raise(f"ADD at layer {add_idx} has inputs {list(layers[add_idx][0])}, expected 2")
    stem_idx = int(other[0])
    stem = mods[stem_idx]
    if type(stem).__name__ != "Conv" or getattr(stem.conv, "in_channels", None) != 4:
        _raise(
            f"ADD at layer {add_idx} input {stem_idx} is {type(stem).__name__} with "
            f"in_channels={getattr(getattr(stem, 'conv', None), 'in_channels', None)}, expected a 4ch Conv stem"
        )

    # ---- both branches must be fed by the expected SilenceChannel slices ----
    def _feeder(i):
        fr = layers[i][0]
        j = i - 1 if fr == -1 else fr
        if not isinstance(j, int) or j < 0 or j >= len(layers):
            return None
        mod = mods[j]
        if type(mod).__name__ != "SilenceChannel":
            return None
        return j, int(getattr(mod, "c_start", -1)), int(getattr(mod, "c_end", -1))

    f_stem = _feeder(stem_idx)
    f_zc = _feeder(zc_idx)
    if f_stem is None or f_stem[1:] != (0, 4):
        _raise(f"M4 stem at layer {stem_idx} is not fed by a SilenceChannel[0,4] (got {f_stem})")
    if f_zc is None or f_zc[1:] != (4, 5):
        _raise(f"ZeroConv2d at layer {zc_idx} is not fed by a SilenceChannel[4,5] (got {f_zc})")
    # parse_model rewrites a leading `Silence` to `nn.Identity` (and mutates the
    # yaml in place), so accept either spelling at the root.
    if type(mods[0]).__name__ not in ("Identity", "Silence"):
        _raise(
            f"layer 0 is {type(mods[0]).__name__!r}, expected a 'Silence'/nn.Identity passthrough "
            f"(the 5ch tensor the two SilenceChannel slices reference)"
        )
    LOGGER.info(
        f"residual-depth adapter layout: 5ch root -> SilenceChannel[0,4]@{f_stem[0]} -> M4 stem@{stem_idx}; "
        f"SilenceChannel[4,5]@{f_zc[0]} -> ZeroConv2d@{zc_idx}; ADD@{add_idx}"
    )
    return stem_idx, zc_idx, add_idx


def _remap_residual_depth_adapter(model, src, stem_idx, zc_idx, add_idx):
    """Remap stock COCO weights into the residual-Depth adapter layout.

    * M4 stem (in=4): ``ch[0:3]`` <- pretrained RGB, ``ch[3]`` <- ``mean(W_R,W_G,W_B)``,
      BN copied from stock ``model.0.bn`` (``load()`` cannot reach it: the Silence
      prefix shifted the module name to ``model.<stem_idx>.bn``).
    * ZeroConv2d (Depth): **must already be exactly zero** and is never written.
    * Everything after the ADD: layer ``i`` <- stock layer ``i - add_idx``, shape-checked.

    Raises on any missing / mismatched / ambiguous tensor.
    """
    from ultralytics.nn.modules.conv import Conv  # noqa: F401  (documented layout)

    layers = list(model.yaml.get("backbone", [])) + list(model.yaml.get("head", []))
    mods = list(model.model)
    stem = mods[stem_idx]
    zc = mods[zc_idx]
    offset = add_idx

    stock_w = src.get("model.0.conv.weight")
    if stock_w is None:
        raise RuntimeError("[residual-depth remap] stock stem 'model.0.conv.weight' not present in pretrained weights")
    if stock_w.shape[1] != 3:
        raise RuntimeError(f"[residual-depth remap] expected a 3-channel stock stem, got in_channels={stock_w.shape[1]}")
    if stock_w.shape[0] != stem.conv.weight.shape[0]:
        raise RuntimeError(
            f"[residual-depth remap] M4 stem out={stem.conv.weight.shape[0]} != stock out={stock_w.shape[0]}"
        )

    n = 0
    n_zero = 0
    with torch.no_grad():
        # ---- 1. M4 stem: RGB prefix + mean(R,G,B) on the aux channel ----
        stem.conv.weight[:, :3].copy_(stock_w)
        stem.conv.weight[:, 3:].copy_(stock_w.float().mean(dim=1, keepdim=True).to(stem.conv.weight.dtype))
        n += 1
        for bk, bv in stem.bn.state_dict().items():
            sk = f"model.0.bn.{bk}"
            sv = src.get(sk)
            if sv is None:
                raise RuntimeError(f"[residual-depth remap] stock '{sk}' missing for the M4 stem BN")
            if sv.ndim == 0:
                bv.copy_(sv)
            elif sv.shape[0] == bv.shape[0]:
                bv.copy_(sv)
            else:
                raise RuntimeError(
                    f"[residual-depth remap] stock '{sk}' has {tuple(sv.shape)} vs target {tuple(bv.shape)}"
                )
            n += 1

        # ---- 2. Depth adapter must be an exact zero-init, and stays zero ----
        for zk, zv in list(zc.state_dict().items()):
            if torch.count_nonzero(zv).item() != 0:
                raise RuntimeError(
                    f"[residual-depth remap] ZeroConv2d parameter '{zk}' is not exactly zero "
                    f"(max_abs={float(zv.abs().max()):.3e}); the residual adapter must be identity at t=0"
                )
            n_zero += 1
        LOGGER.info(
            f"residual-depth remap: M4 stem at layer {stem_idx} <- stock model.0 (ch[0:3]=RGB, ch[3]=mean(RGB)); "
            f"ZeroConv2d at layer {zc_idx} left at exact zero ({n_zero} tensors)"
        )

        # ---- 3. body + neck + head: layer i <- stock (i - offset) ----
        csd = model.state_dict()
        detect_idx = len(layers) - 1
        nc = int(getattr(model.model[-1], "nc", -1))
        n_cls = 0
        for i in range(offset + 1, len(layers)):
            s = i - offset
            prefix = f"model.{i}."
            for k, v in csd.items():
                if not k.startswith(prefix) or v.ndim == 0:
                    continue
                sk = f"model.{s}." + k[len(prefix) :]
                sv = src.get(sk)
                if sv is not None and tuple(sv.shape) == tuple(v.shape):
                    v.copy_(sv)
                    n += 1
                    continue
                if (
                    i == detect_idx
                    and ".cv3." in k
                    and sv is not None
                    and tuple(sv.shape[1:]) == tuple(v.shape[1:])
                    and v.shape[0] == nc
                    and sv.shape[0] != nc
                ):
                    n_cls += 1
                    continue
                if sv is None:
                    raise RuntimeError(
                        f"[residual-depth remap] no pretrained counterpart for '{k}' (looked for '{sk}')"
                    )
                raise RuntimeError(
                    f"[residual-depth remap] shape mismatch for '{k}': candidate {tuple(v.shape)} vs stock {sk} {tuple(sv.shape)}"
                )

    LOGGER.info(
        f"Residual-depth adapter remap: {n} tensors transferred "
        f"(M4 stem layer {stem_idx} + layers {offset + 1}..{len(layers) - 1} <- stock 1..{len(layers) - 1 - offset}); "
        f"ZeroConv2d (layer {zc_idx}) = exact zero; Detect cv3 re-initialised for nc={nc}: {n_cls} tensors"
    )
    return n


def _transfer_rgb_pretrained(model, weights):
    """Transfer single-branch COCO pretrained weights into fusion models.

    The default ``model.load()`` matches weights by (key-name, shape) via
    ``intersect_dicts``, so any stem whose input-channel count differs from the
    pretrained 3ch is silently skipped. Two fusion families are handled here:

    - RGBD mid-fusion (separate 1ch depth stem): the RGB branch is prefixed by
      ``Silence``/``SilenceChannel``, shifting its indices by +2, so ``load()``
      matches ~0 RGB keys. Remap module-by-module by structural correspondence,
      then init the depth stem with ``W_depth = mean(W_R, W_G, W_B)``.
    - RGBID early fusion (single 5ch stem): every layer except the first Conv
      matches ``load()`` exactly; only the first Conv weight (5ch vs 3ch) is
      skipped. Copy the pretrained RGB weights into the first 3 channels and init
      the IR/depth channels with ``mean(W_R, W_G, W_B)``.

    Single-branch models (no 1ch or 5ch stem) are left untouched: standard
    ``load()`` already handles them. Returns the number of tensors transferred.
    """
    from pathlib import Path

    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.tasks import attempt_load_one_weight

    stems = [m for m in model.model if isinstance(m, Conv)]

    # source single-branch state_dict
    if isinstance(weights, (str, Path)):
        src_model, _ = attempt_load_one_weight(weights)
        src = src_model.state_dict()
    elif isinstance(weights, dict):
        src = weights["model"] if "model" in weights else weights
    else:
        src = weights.state_dict()

    # ---- 源已是目标架构（resume / 从融合 .pt 微调）→ 无需 remap ----
    # 下面的分派全部基于**目标模型**的 stem，并假定**源**是单分支 stock COCO。
    # 若源已经逐个带齐了目标的全部键且 shape 一致，说明 `model.load()` 已经完成全部工作，
    # 没有任何东西需要 remap —— 而 remap 要找的那个 stock stem 根本不存在。
    # 缺少本守卫时，resume 一个 sepstem/融合 run 会抛：
    #   RuntimeError: [sepstem remap] stock stem 'model.0.conv.weight' not present
    # 更严重的是：即使它侥幸越过 stem，也会把已训好的 Detect cv3 分类分支按 nc 重新初始化，
    # 直接抹掉分类头。所以这里必须跳过，不能让它继续。
    _tgt = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    _src = {k: tuple(v.shape) for k, v in src.items() if hasattr(v, "shape")}
    if _tgt and all(_src.get(k) == s for k, s in _tgt.items()):
        LOGGER.info(
            f"pretrained source already matches the target architecture "
            f"({len(_tgt)}/{len(_tgt)} keys) -> skip stock->fusion remap"
        )
        return 0

    # ---- residual-Depth adapter (M4 stem + zero-init Depth + ADD) ----
    # Dispatched FIRST and by an unambiguous marker: no existing config in this
    # repo uses ZeroConv2d. Without this branch the layout falls into `multi_ch`
    # below, which writes the first Conv and returns -- leaving the whole
    # index-shifted body randomly initialised (measured 9.7% coverage) while the
    # old contract gate still reported PASS.
    _rda = _find_residual_depth_adapter(model)
    if _rda is not None:
        return _remap_residual_depth_adapter(model, src, *_rda)

    # ---- early fusion, single stem absorbing N channels (no 1ch stem) ----
    # 2026-10-03 修复：原条件写死 `in_channels == 5`（只为 RGBID 服务），4ch / 2ch 单 stem
    # 不匹配任何分支 ⇒ 落入下面的 `return 0`（无日志、无 raise），而 `model.load()` 已因 shape
    # 不符丢掉 `model.0.conv.weight` ⇒ 首个 Conv **静默保持随机初始化**。
    # 现在统一处理 N != 3 的单 stem 布局，并在无法确定策略时**抛错**。
    # ⚠ 必须只看**第一个** Conv（输入 stem）：`in_channels > 3` 也会命中模型深处的
    #   Conv（如 64/128ch），那是普通层不是 stem。原实现写 `== 5` 恰好躲过了这个坑。
    _first = stems[0] if stems else None
    _fin = (getattr(_first.conv, "in_channels", None) or 0) if _first is not None else 0
    multi_ch = _first if _fin > 3 else None
    narrow_ch = _first if _fin in (1, 2) else None
    if multi_ch is not None and not any(getattr(m.conv, "in_channels", None) == 1 for m in stems):
        # 通道序约定 [B, G, R, AUX...]：前 3 通道吃预训练 RGB，其余 = mean(W_R,W_G,W_B)。
        w = src.get("model.0.conv.weight")
        if w is None or w.shape[1] != 3 or w.shape[0] != multi_ch.conv.weight.shape[0]:
            raise RuntimeError(
                f"[multi-ch stem remap] cannot initialise in_channels="
                f"{multi_ch.conv.in_channels} stem: stock stem 'model.0.conv.weight' "
                f"{'missing' if w is None else tuple(w.shape)} vs target out="
                f"{multi_ch.conv.weight.shape[0]}. 拒绝静默保留随机初始化。"
            )
        # Fail-closed: this branch writes the stem and returns, and it can only
        # rely on `load()` to have matched the rest of the graph **by name**. That
        # is true only when the stem sits at layer 0 and no Silence/SilenceChannel
        # prefix has shifted the indices. If the stem is anywhere else, the body is
        # not matched by name and would stay randomly initialised -- raise instead.
        _mi = next((i for i, m in enumerate(model.model) if m is multi_ch), None)
        if _mi != 0:
            raise RuntimeError(
                f"[multi-ch stem remap] the {multi_ch.conv.in_channels}ch stem is at layer {_mi}, not 0: "
                f"the Silence/Identity prefix shifted the module names, so `load()` cannot match the body "
                f"and this branch would leave it randomly initialised. Refusing to guess."
            )
        n_aux = multi_ch.conv.weight.shape[1] - 3
        with torch.no_grad():
            multi_ch.conv.weight[:, :3].copy_(w)  # RGB 预训练
            multi_ch.conv.weight[:, 3:].copy_(
                w.mean(dim=1, keepdim=True).repeat(1, n_aux, 1, 1)
            )  # AUX = mean(W_R,W_G,W_B)（与 RGBD depth-stem / sepstem aux-stem 同一约定）
        gw = multi_ch.conv.weight
        LOGGER.info(
            f"multi-ch stem init: in={multi_ch.conv.in_channels} out={gw.shape[0]} | "
            f"ch[0:3]=pretrained RGB ({tuple(w.shape)}) | ch[3:{gw.shape[1]}]=mean(R,G,B) | "
            f"mean={float(gw.mean()):+.6f} std={float(gw.std()):.6f} "
            f"finite={bool(torch.isfinite(gw).all())} | newly_init_tensors=1 transferred_tensors=4"
        )
        return 5

    if narrow_ch is not None and not any(getattr(m.conv, "in_channels", None) == 3 for m in stems):
        # 纯低频模态（IR+Depth 等，无 RGB）：每个输入通道都吃**灰度化** stock stem，
        # 与 _remap_separate_stem 对 aux stem 的处理完全一致。
        w = src.get("model.0.conv.weight")
        if w is None or w.shape[1] != 3 or w.shape[0] != narrow_ch.conv.weight.shape[0]:
            raise RuntimeError(
                f"[narrow-ch stem remap] cannot initialise in_channels="
                f"{narrow_ch.conv.in_channels} stem: stock stem 'model.0.conv.weight' "
                f"{'missing' if w is None else tuple(w.shape)} vs target out="
                f"{narrow_ch.conv.weight.shape[0]}. 拒绝静默保留随机初始化。"
            )
        with torch.no_grad():
            narrow_ch.conv.weight.copy_(
                w.mean(dim=1, keepdim=True).repeat(1, narrow_ch.conv.weight.shape[1], 1, 1)
            )
        gw = narrow_ch.conv.weight
        LOGGER.info(
            f"narrow-ch stem init (no RGB in input): in={narrow_ch.conv.in_channels} "
            f"out={gw.shape[0]} | all ch = mean(W_R,W_G,W_B) | "
            f"mean={float(gw.mean()):+.6f} std={float(gw.std()):.6f} "
            f"finite={bool(torch.isfinite(gw).all())} | newly_init_tensors=1 transferred_tensors=4"
        )
        return 5

    # ---- 关于 "1 个 3ch stem + 1 个 1ch stem" 的布局 ----
    # 该布局（yolo11*_midfusion_rgbd_*.yaml / yolo11_rgbt.yaml 等 15 个既有模型）
    # **合法地**依赖下面 RGBD concat_res 分支的硬编码索引表完成 remap。
    # 2026-10-03 Infrastructure Repair 曾在此处加过一条 "歧义即 raise" 的守卫，
    # 回归扫描显示它会让这 15 个既有模型全部失败 ⇒ **已回退**（属超出 infrastructure scope 的改动）。
    # 残留风险（已知、未修）：若将来新增一个"1×3ch + 1×1ch"但**不是** concat_res 索引表布局的
    # 模型，它会被这里静默误映射。新增此类模型时必须同步检查本函数的分派条件。

    # ---- Separate-Stem fusion (exactly one 3ch stem + exactly two 1ch stems) ----
    # Must be dispatched BEFORE the RGBD branch below: that branch keys off the mere
    # presence of a 1ch stem and would otherwise swallow this layout, remapping through a
    # hardcoded concat_res index table that does not correspond to it.
    _three = [m for m in stems if getattr(m.conv, "in_channels", None) == 3]
    _ones = [m for m in stems if getattr(m.conv, "in_channels", None) == 1]
    if len(_three) == 1 and len(_ones) == 2:
        return _remap_separate_stem(model, src, _three[0], _ones)

    # ---- fail-closed guard: a Silence-prefixed layout with a >3ch Conv stem ----
    # Only the two layouts above know how to remap a multi-channel stem. Anything
    # else that reaches this point would be silently swallowed by the RGBD index
    # table below (its (fusion idx -> stock idx) table does not correspond to any
    # other topology). Measured on the candB prototype: 50.5% of the body left
    # randomly initialised, with no exception raised.
    # ⚠ only the FIRST Conv is an input stem -- `in_channels > 3` also matches deep
    # Convs (64/256/512ch) which are ordinary layers, exactly the trap the multi-ch
    # branch above already documents.
    _wide = [stems[0]] if stems and (getattr(stems[0].conv, "in_channels", None) or 0) > 3 else []
    if _wide:
        _ins = sorted({int(m.conv.in_channels) for m in _wide})
        _wide_ids = {id(m) for m in _wide}
        _li = [i for i, m in enumerate(model.model) if id(m) in _wide_ids]
        raise RuntimeError(
            f"[fusion remap] unsupported fusion layout: found Conv stem(s) with in_channels={_ins} "
            f"at layer(s) {_li} together with 1ch stem(s), and this layout matches none of the "
            f"implemented remap families (multi-ch single stem / narrow-ch / 1x3ch+2x1ch separate-stem / "
            f"residual-Depth adapter). The RGBD concat_res index table does NOT apply here and would "
            f"silently leave most of the body randomly initialised. Refusing to guess."
        )

    if not any(getattr(m.conv, "in_channels", None) == 1 for m in stems):
        return 0  # not an RGBD fusion model -> nothing to remap

    # (fusion module idx -> single-branch yolo11 idx) for concat_res mid-fusion.
    # The layout shifts between the 3-scale (Detect P3/P4/P5) and 4-scale (P2
    # variant) architectures, so select by the number of detection heads.
    nl = getattr(model.model[-1], "nl", 3)
    if nl == 4:  # Detect(P2, P3, P4, P5)
        pairs = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 19: 5, 20: 6, 26: 7, 27: 8,
                 33: 9, 34: 10, 37: 13, 40: 16, 44: 17, 46: 19, 47: 20,
                 49: 22, 50: 23}
    else:  # Detect(P3, P4, P5)
        pairs = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 16: 5, 17: 6, 23: 7, 24: 8,
                 30: 9, 31: 10, 34: 13, 37: 16, 38: 17, 40: 19, 41: 20,
                 43: 22, 44: 23}
    n = 0
    with torch.no_grad():
        for fi, si in pairs.items():
            prefix = f"model.{fi}."
            for fname, fparam in model.state_dict().items():
                if fname.startswith(prefix) and fparam.ndim > 0:
                    sname = f"model.{si}." + fname[len(prefix):]
                    sv = src.get(sname)
                    if sv is not None and sv.shape == fparam.shape:
                        fparam.copy_(sv)
                        n += 1

    # depth stem: W_depth = mean(W_R, W_G, W_B)
    rgb = next((m for m in stems if m.conv.in_channels == 3), None)
    depth = next((m for m in stems if m.conv.in_channels == 1), None)
    if rgb is not None and depth is not None and rgb.conv.weight.shape[0] == depth.conv.weight.shape[0]:
        with torch.no_grad():
            depth.conv.weight.copy_(rgb.conv.weight.mean(dim=1, keepdim=True))
            if depth.conv.bias is not None and rgb.conv.bias is not None:
                depth.conv.bias.copy_(rgb.conv.bias)
        LOGGER.info(
            f"RGBD depth-stem init: W_depth = mean(W_R,W_G,W_B) "
            f"(rgb in={rgb.conv.in_channels} -> depth in={depth.conv.in_channels})"
        )
    LOGGER.info(f"RGBD pretrained remap: transferred {n} tensors")
    return n


class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, use_simotm=self.args.use_simotm,pairs_rgb_ir=self.args.pairs_rgb_ir,pairs_rgb_depth=self.args.pairs_rgb_depth)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights is None and isinstance(self.args.pretrained, (str, Path)):
            # `YOLO(yaml).train(pretrained=...)` reaches here with weights=None (self.ckpt
            # is empty for yaml-built models), which silently skips weight loading and the
            # RGBD remap below. Fall back to the trainer's `pretrained` arg so weights load.
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        if weights:
            model.load(weights)
            _transfer_rgb_pretrained(model, weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
            use_simotm=self.args.use_simotm,  # 2025-01-05
        )
        # 'yzc' 2025-03-03
        if self.args.use_simotm in ("RGBT", "RGBRGB6C"):
            plot_images(
                images=batch["img"],
                batch_idx=batch["batch_idx"],
                cls=batch["cls"].squeeze(-1),
                bboxes=batch["bboxes"],
                paths=batch["im_file"],
                fname=self.save_dir / f"train_batch{ni}_ir.jpg",
                on_plot=self.on_plot,
                use_simotm=self.args.use_simotm,  # 2025-01-05
                ir_show=True  # 显示红外图像以及标签
            )

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Get batch size by calculating memory occupation of model."""
        train_dataset = self.build_dataset(self.trainset, mode="train", batch=16)
        # 4 for mosaic augmentation
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        return super().auto_batch(max_num_obj)
