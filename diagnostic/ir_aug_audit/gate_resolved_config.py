# -*- coding: utf-8 -*-
"""§5 — Resolved-config Single-variable Gate（比较**解析后**的配置，不是 YAML 文本）。

解析路径 = 与正式训练**完全相同**：
    scripts.train._build_train_kwargs(train_cfg, data_path)   → kwargs
    ultralytics.cfg.get_cfg(DEFAULT_CFG_DICT, overrides=kwargs)  → trainer 用的 args 命名空间

然后逐 key 比较 D′ 与 D′-IRAug 的 **全量 resolved namespace**（含 default.yaml 的默认值，
因此"YAML 里没写"的字段也会被真正比较）。

允许不同的只有: experiment_name / save_dir / name / project 派生项 + 4 个 IR aug 键。
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yaml  # noqa: E402
from ultralytics.cfg import get_cfg, DEFAULT_CFG_DICT  # noqa: E402
from scripts.train import _build_train_kwargs, resolve_data_path  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_CFG = "configs/train_rgbid_sepstem_clahe.yaml"
IRAU_CFG = "configs/train_rgbid_sepstem_clahe_iraug.yaml"
ALLOWED_DIFF = {"experiment_name", "save_dir", "name", "project",
                "ir_gamma", "ir_gamma_probability", "ir_noise_std", "ir_noise_probability"}


def resolve(path):
    tc = yaml.safe_load(open(path, encoding="utf-8")) or {}
    data_path = resolve_data_path(tc, "configs/dataset.yaml")
    kwargs = _build_train_kwargs(tc, data_path)
    args = get_cfg(DEFAULT_CFG_DICT, overrides=kwargs)
    return tc, kwargs, args


def norm(v):
    """把命名空间/列表值归一成可比较形式。"""
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {k: norm(x) for k, x in sorted(v.items())}
    if isinstance(v, os.PathLike):
        return str(v)
    return v


def main():
    tc_b, kw_b, a_b = resolve(BASE_CFG)
    tc_i, kw_i, a_i = resolve(IRAU_CFG)

    keys_b, keys_i = set(vars(a_b)), set(vars(a_i))
    only_b, only_i = keys_b - keys_i, keys_i - keys_b

    diffs = {}
    for k in sorted(keys_b & keys_i):
        vb, vi = norm(getattr(a_b, k)), norm(getattr(a_i, k))
        if vb != vi:
            diffs[k] = {"baseline": vb, "iraug": vi}

    unexpected = {k: v for k, v in diffs.items() if k not in ALLOWED_DIFF}
    missing_from_iraug = {k: getattr(a_b, k) for k in only_b}
    extra_in_iraug = {k: getattr(a_i, k) for k in only_i}

    res = {
        "n_resolved_keys": len(keys_b | keys_i),
        "resolved_data_path": {"baseline": a_b.data, "iraug": a_i.data},
        "raw_kwargs_diff": {k: {"baseline": kw_b.get(k, "<absent>"), "iraug": kw_i.get(k, "<absent>")}
                            for k in sorted(set(kw_b) | set(kw_i))
                            if norm(kw_b.get(k, "<absent>")) != norm(kw_i.get(k, "<absent>"))},
        "resolved_diff": diffs,
        "unexpected_diff": unexpected,
        "keys_only_in_baseline": missing_from_iraug,
        "keys_only_in_iraug": extra_in_iraug,
    }
    res["PASS"] = (not unexpected) and (not missing_from_iraug) and (not extra_in_iraug)

    print(f"resolved keys compared: {res['n_resolved_keys']}")
    print(f"resolved data path : both = {a_b.data == a_i.data}  ({a_i.data})")
    print("\n--- raw kwargs 差异（_build_train_kwargs 输出）---")
    for k, v in res["raw_kwargs_diff"].items():
        print(f"  {k:<22} {v['baseline']!r}  ->  {v['iraug']!r}")
    print("\n--- resolved namespace 差异（含 default.yaml 默认值）---")
    for k, v in diffs.items():
        tag = "ALLOWED" if k in ALLOWED_DIFF else "*** UNEXPECTED ***"
        print(f"  [{tag}] {k:<22} {v['baseline']!r}  ->  {v['iraug']!r}")
    print(f"\nonly-in-baseline: {missing_from_iraug}")
    print(f"only-in-iraug   : {extra_in_iraug}")
    print(f"\nSINGLE_VARIABLE_GATE = {'PASS' if res['PASS'] else 'FAIL'}")

    with open(os.path.join(HERE, "gate_resolved_config.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False, default=str)
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
