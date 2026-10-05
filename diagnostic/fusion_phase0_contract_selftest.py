"""fusion_phase0_contract_selftest.py — 证明 Contract Gate 不是装饰。

对 configs/train_modality_m1_rgb.yaml 的**临时副本**逐个注入真实缺陷，
确认 gate 每次都 raise。所有临时文件在 finally 中删除，不留痕。

ZERO GPU / ZERO forward / ZERO training。
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("gate", ROOT / "scripts/validate_modality_contract.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

BASE_T = "configs/train_modality_m1_rgb.yaml"
BASE_M = "configs/yolo11m_modality3ch.yaml"
TMP_T = "configs/_tmp_selftest_train.yaml"
TMP_M = "configs/yolo11m_tmp_selftest_model.yaml"
TMP_M_NOSCALE = "configs/tmp_selftest_model_noscale.yaml"   # 文件名里没有 scale 字符

FAULTS = [
    ("epochs 300 -> 30",
     lambda t, m: t.update(epochs=30), None),
    ("channels 3 -> 4（config 与 model yaml 不一致）",
     lambda t, m: t.update(channels=4), None),
    ("batch_size 8 -> 16",
     lambda t, m: t.update(batch_size=16), None),
    ("lr0 5e-3 -> 1e-2",
     lambda t, m: t.update(learning_rate=1.0e-2), None),
    ("seed 42 -> 0",
     lambda t, m: t.update(seed=0), None),
    ("val_ratio 0.2 -> 0.5",
     lambda t, m: t.update(val_ratio=0.5), None),
    ("模型 yaml 去掉 scale 字符（触发 nano 静默回退）",
     lambda t, m: None, "noscale"),
    ("use_simotm=Depth 却声明 ir_encoding=clahe（该模式不消费它）",
     lambda t, m: t.update(use_simotm="Depth", ir_encoding="clahe"), None),
    ("aug.hsv_h 0.0 -> 0.015（破坏光度增强对齐）",
     lambda t, m: t.update(aug={**t.get("aug", {}), "hsv_h": 0.015}), None),
    ("aug.albumentations_p 0.0 -> 1.0（3ch 重新打开光度增强）",
     lambda t, m: t.update(aug={**t.get("aug", {}), "albumentations_p": 1.0}), None),
    ("pretrained yolo11m.pt -> yolo11n.pt",
     lambda t, m: t.update(pretrained="yolo11n.pt"), None),
]


# ============================================================
# P1 (2026-10-04): coverage-specific faults.
# The old criterion `n_ret > 0 and stem_changed` is satisfiable while the body is
# still randomly initialised (the candC incident: n_ret=5, stem_changed=True,
# real coverage 0.0019). These two cases pin that down permanently.
# ============================================================
BASE_T5 = "configs/train_rgbid_sepstem_clahe.yaml"  # M7 / D′, ch=5
BASE_M_SEPSTEM = "configs/yolo11m_sepstem.yaml"
TMP_T5 = "configs/_tmp_selftest_train5.yaml"
TMP_C = "configs/yolo11m_tmp_selftest_resdepth.yaml"


def _shift(fr, d):
    if isinstance(fr, list):
        return [_shift(x, d) for x in fr]
    return fr + d if isinstance(fr, int) and fr >= 0 else fr


def residual_depth_layout_from_sepstem(orig_m: str) -> dict:
    """Rewrites the D′ sepstem yaml (L0..L7 fusion block) into the residual-Depth layout.

    Keeps the stock body/head untouched and shifts their graph references by -2, so
    the result is exactly the 'M4 stem + ZeroConv2d(Depth) + ADD' topology.
    """
    m = yaml.safe_load(orig_m)
    block = [
        [-1, 1, "Silence", []],
        [0, 1, "SilenceChannel", [0, 4]],
        [-1, 1, "Conv", [64, 3, 2]],
        [0, 1, "SilenceChannel", [4, 5]],
        [-1, 1, "ZeroConv2d", [64, 3, 2, 1]],
        [[2, 4], 1, "ADD", [1.0]],
    ]
    m["backbone"] = block + [[_shift(l[0], -2), *l[1:]] for l in m["backbone"][8:]]
    m["head"] = [[_shift(l[0], -2), *l[1:]] for l in m["head"]]
    return m


def main():
    orig_t = (ROOT / BASE_T).read_text(encoding="utf-8")
    orig_m = (ROOT / BASE_M).read_text(encoding="utf-8")
    results = []
    try:
        # 先确认 baseline 是 PASS（否则后面的 FAIL 无意义）
        try:
            gate.check("BASELINE", BASE_T, BASE_M)
            results.append(("(baseline M1 无缺陷)", "PASS", "gate 通过 —— 对照成立"))
        except AssertionError as e:
            results.append(("(baseline M1 无缺陷)", "UNEXPECTED-FAIL", str(e)))

        for name, mutate, mode in FAULTS:
            t = yaml.safe_load(orig_t)
            m = yaml.safe_load(orig_m)
            mutate(t, m)
            (ROOT / TMP_T).write_text(yaml.safe_dump(t, allow_unicode=True, sort_keys=False), encoding="utf-8")
            mpath = TMP_M
            if mode == "noscale":
                (ROOT / TMP_M_NOSCALE).write_text(orig_m, encoding="utf-8")
                mpath = TMP_M_NOSCALE
            else:
                (ROOT / TMP_M).write_text(orig_m, encoding="utf-8")
            try:
                gate.check("FAULT", TMP_T, mpath)
                results.append((name, "NOT-CAUGHT", "gate 竟然放行 —— 这是缺陷"))
            except AssertionError as e:
                results.append((name, "CAUGHT", str(e)[:120]))
            except Exception as e:
                results.append((name, "CAUGHT", f"{type(e).__name__}: {str(e)[:110]}"))
            finally:
                for p in (TMP_T, TMP_M, TMP_M_NOSCALE):
                    (ROOT / p).unlink(missing_ok=True)

        # ---- P1 coverage cases: residual-Depth layout, honest vs silently-no-op remap ----
        import ultralytics.models.yolo.detect.train as _tr
        orig_t5 = (ROOT / BASE_T5).read_text(encoding="utf-8")
        (ROOT / TMP_T5).write_text(orig_t5, encoding="utf-8")
        orig_sepstem = (ROOT / BASE_M_SEPSTEM).read_text(encoding="utf-8")
        (ROOT / TMP_C).write_text(
            yaml.safe_dump(residual_depth_layout_from_sepstem(orig_sepstem), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        # control: the real remap must PASS on this layout
        try:
            gate.check("RD-CONTROL", TMP_T5, TMP_C)
            results.append(("P1 对照：residual-Depth 布局 + 真实 remap", "PASS", "gate 通过 —— 对照成立"))
        except AssertionError as e:
            results.append(("P1 对照：residual-Depth 布局 + 真实 remap", "UNEXPECTED-FAIL", str(e)[:140]))
        except Exception as e:  # noqa: BLE001
            results.append(("P1 对照：residual-Depth 布局 + 真实 remap", "UNEXPECTED-FAIL", f"{type(e).__name__}: {str(e)[:110]}"))
        # fault: emulates the pre-P1 dispatch, which reported success while doing nothing
        _saved = _tr._transfer_rgb_pretrained
        _tr._transfer_rgb_pretrained = lambda model, weights: 5  # "只写 stem 就 return"
        try:
            try:
                gate.check("RD-FAULT", TMP_T5, TMP_C)
                results.append(("P1 缺陷：旧式静默 remap（n_ret=5 但 body 未继承）", "NOT-CAUGHT", "gate 竟然放行 —— 这是缺陷"))
            except AssertionError as e:
                results.append(("P1 缺陷：旧式静默 remap（n_ret=5 但 body 未继承）", "CAUGHT", str(e)[:140]))
            except Exception as e:  # noqa: BLE001
                results.append(("P1 缺陷：旧式静默 remap（n_ret=5 但 body 未继承）", "CAUGHT", f"{type(e).__name__}: {str(e)[:110]}"))
        finally:
            _tr._transfer_rgb_pretrained = _saved
            for p in (TMP_T5, TMP_C):
                (ROOT / p).unlink(missing_ok=True)
    finally:
        for p in (TMP_T, TMP_M, TMP_M_NOSCALE, TMP_T5, TMP_C):
            (ROOT / p).unlink(missing_ok=True)

    print("=" * 100)
    print("Contract Gate self-test —— 逐个注入真实缺陷，确认 gate 会 raise")
    print("=" * 100)
    for n, s, d in results:
        icon = "✅" if s in ("PASS", "CAUGHT") else "❌"
        print(f"  {icon} [{s:<15}] {n}")
        if d:
            print(f"        └─ {d}")
    nbad = sum(1 for _, s, _ in results if s not in ("PASS", "CAUGHT"))
    print("=" * 100)
    print(f"self-test: {len(results)-nbad}/{len(results)} 符合预期（baseline PASS + 全部缺陷 CAUGHT）")
    leftover = [p for p in (TMP_T, TMP_M, TMP_M_NOSCALE) if (ROOT / p).exists()]
    print(f"临时文件残留: {leftover or '无'}")
    return 0 if nbad == 0 and not leftover else 1


if __name__ == "__main__":
    raise SystemExit(main())
