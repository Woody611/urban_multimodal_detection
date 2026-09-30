"""_args_check.py — OASA scale=1.4 正式实验的开训前 args 落地核验。

性质：只读、不训练。做三件事：
  ① 用 scripts/train.py **自己的** `_load_yaml` + `_build_train_kwargs` 构建 kwargs，
     再经 ultralytics `get_cfg` 解析成最终 training args（即真正送进 A100 的那一份）。
  ② 断言 5 个 OASA 键 + ir_encoding / channels / use_simotm / epochs / mosaic /
     close_mosaic / lr0 / seed / batch / imgsz / name 的期望值。任一不符 → 退出码 1。
  ③ 把 scale=1.4 与 scale=2.0 两份配置的**最终 args** 逐项 diff，证明差异只有两项
     （object_scale_aug_scale + name）—— 这是 §3「保持条件不变」的机器证明，
     强于 yaml 文本 diff。
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ultralytics.cfg import get_cfg  # noqa: E402

spec = importlib.util.spec_from_file_location("_trainmod", ROOT / "scripts/train.py")
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

DATA = "data/processed/rgbid_split_train/dataset.yaml"
CFG_20 = "configs/oasa_pre_mosaic.yaml"
CFG_14 = "configs/oasa_pre_mosaic_scale14.yaml"

EXPECT = {
    "object_scale_aug": True,
    "object_scale_aug_prob": 0.5,
    "object_scale_aug_scale": 1.4,
    "object_scale_aug_small_area": 1024.0,
    "object_scale_aug_min_visible": 0.8,
    "ir_encoding": "clahe",
    "channels": 5,
    "use_simotm": "RGBID",
    "epochs": 300,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "lr0": 0.005,
    "seed": 42,
    "batch": 8,
    "imgsz": 1280,
    "name": "urban_multimodal_det_yolo11_rgbid_sepstem_clahe_oasa_s14",
    "patience": 0,
    "cos_lr": True,
    "val": True,
}
# 只读诊断项：不参与断言，仅打印
SHOW = ["optimizer", "lrf", "weight_decay", "warmup_epochs", "momentum", "workers",
        "fliplr", "translate", "scale", "degrees", "shear", "perspective", "mixup",
        "copy_paste", "hsv_h", "hsv_s", "hsv_v", "erasing", "rect", "project"]


def resolved(cfg_path):
    tc = tm._load_yaml(str(ROOT / cfg_path))
    kw = tm._build_train_kwargs(tc, DATA)
    return tc, kw, get_cfg(overrides=kw)


def main():
    tc14, kw14, full14 = resolved(CFG_14)
    tc20, kw20, full20 = resolved(CFG_20)

    print("=" * 78)
    print("OASA scale=1.4 —— 开训前 args 落地核验")
    print("=" * 78)
    print(f"\n配置: {CFG_14}\n")

    fails = []
    print("--- ① 期望值断言 ---")
    for k, want in EXPECT.items():
        got = full14.get(k) if hasattr(full14, "get") else getattr(full14, k, None)
        got = getattr(full14, k, None)
        ok = (got == want) or (isinstance(want, float) and got is not None and abs(float(got) - want) < 1e-12)
        print(f"  [{'PASS' if ok else 'FAIL'}] {k:32s} = {got!r}"
              + ("" if ok else f"   (期望 {want!r})"))
        if not ok:
            fails.append(k)

    print("\n--- ② 只读诊断（不参与断言）---")
    for k in SHOW:
        print(f"       {k:32s} = {getattr(full14, k, None)!r}")
    print(f"       {'save_policy(best/last)':32s} = save={full14.save} period={full14.save_period}")
    print(f"       {'project/name':32s} = {full14.project}/{full14.name}")

    print("\n--- ③ scale=1.4 vs scale=2.0 的最终 args 逐项 diff ---")
    diff = []
    for k in sorted(set(vars(full14)) | set(vars(full20))):
        a, b = getattr(full14, k, None), getattr(full20, k, None)
        if a != b:
            diff.append((k, b, a))
    if not diff:
        print("  （无差异 —— 异常，至少 scale 与 name 应不同）")
        fails.append("resolved_args_identical")
    for k, v20, v14 in diff:
        print(f"  {k:32s} 2.0 -> {v20!r}   1.4 -> {v14!r}")
    only_expected = {k for k, _, _ in diff} <= {"object_scale_aug_scale", "name", "save_dir"}
    print(f"\n  差异项集合 = {{{', '.join(k for k, _, _ in diff)}}}")
    print(f"  [{'PASS' if only_expected else 'FAIL'}] 差异仅限 scale / name / save_dir(name 派生)")
    if not only_expected:
        fails.append("unexpected_resolved_arg_diff")

    print("\n" + "=" * 78)
    if fails:
        print(f"RESULT: FAIL ({len(fails)} 项) -> STOP — do not train")
        print(f"        {fails}")
        return 1
    print("RESULT: ALL CHECKS PASSED  ->  PROCEED")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
