"""_pre_resume_check.py — OASA scale=1.4 续训前检查（只读，CPU 即可）。

为什么必须先跑：ultralytics 的 resume 走
    self.args = get_cfg(ckpt_args)                     # 从 checkpoint 取
    for k in ("imgsz","batch","device","close_mosaic"): # 只有这 4 项允许被当前 kwargs 覆盖
即 **object_scale_aug_scale 不在可覆盖列表里** —— resume 用的是 checkpoint 里存的那份。
若 ckpt 里不是 1.4，续训会静默按 ckpt 的值跑，而配置文件写了什么都没用。

用法:
    python diagnostic/oasa_scale14/_pre_resume_check.py <last.pt> [expected_scale]
"""
import sys
from pathlib import Path

# ⚠ 必须先于 `import torch` 之后、**反序列化之前**把仓库根放到 sys.path[0]。
# 否则 torch.load 解析 checkpoint 里的 `ultralytics.nn.modules.conv.SilenceChannel`
# 会落到 pip 安装的 stock ultralytics（无该类）→ AttributeError: Can't get attribute。
# scripts/train.py:49-50 做的是同一件事。
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

EXPECT_OASA = {
    "object_scale_aug": True,
    "object_scale_aug_prob": 0.5,
    "object_scale_aug_scale": None,      # 由命令行给
    "object_scale_aug_small_area": 1024.0,
    "object_scale_aug_min_visible": 0.8,
}
EXPECT_OTHER = {
    "epochs": 300,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "lr0": 0.005,
    "seed": 42,
    "batch": 8,
    "imgsz": 1280,
    "channels": 5,
    "use_simotm": "RGBID",
    "ir_encoding": "clahe",
    "patience": 0,
}
# 可覆盖列表（resume 时唯一能被当前配置改掉的 4 项）
OVERRIDABLE = {"imgsz", "batch", "device", "close_mosaic"}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    ck = Path(sys.argv[1])
    expect_scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.4
    EXPECT_OASA["object_scale_aug_scale"] = expect_scale

    if not ck.exists():
        print(f"!! checkpoint 不存在: {ck}")
        return 1

    d = torch.load(str(ck), map_location="cpu", weights_only=False)
    # args 在 ckpt["train_args"]（ultralytics/nn/tasks.py:906,942 的 attempt_load_weights 也读这个键）。
    # 我最初误读成 ckpt["args"] —— 该键根本不存在，导致所有键报 <缺失> 的假失败。
    a = d.get("train_args") or {}
    if not a:
        ma = getattr(d.get("model"), "args", None)
        a = dict(ma) if isinstance(ma, dict) else {}
    a = dict(a)
    ep = d.get("epoch", None)
    bf = d.get("best_fitness", None)
    # strip_optimizer 后的「最终产物」：epoch=-1 且 optimizer/ema 被抹除，**不能用于 resume**
    stripped = (ep == -1) or (d.get("optimizer") is None and d.get("ema") is None)

    print("=" * 74)
    print(f"OASA 续训前检查: {ck}")
    print("=" * 74)
    print(f"  文件大小              : {ck.stat().st_size:,} bytes"
          f"   (>60MB 通常 = 未 strip，可用于 resume)")
    fails = []
    if stripped:
        print("  !! 这是 strip_optimizer 之后的『最终产物』：epoch=-1、optimizer/ema 已抹除")
        print("     => 只能用于推理/评测，**不能用于 resume**。请改用训练中途保存的 last.pt。")
        fails.append("checkpoint_is_stripped")
    else:
        print(f"  ckpt['epoch'] (0-idx) : {ep}   -> 续训从 0-idx {ep + 1} 开始，显示为 epoch {ep + 2}/300")
    print(f"  ckpt['best_fitness']  : {bf}")
    print(f"  optimizer 状态        : {'有' if d.get('optimizer') is not None else '缺失!!'}")
    print(f"  ema 状态              : {'有' if d.get('ema') is not None else '缺失!!'}")
    print(f"  train_args 键数       : {len(a)}{'' if a else '   <<< 缺失 -> resume 会用默认值，危险'}")
    if not a:
        fails.append("train_args_missing")
    print("\n--- OASA 键（**决定续训实际行为**）---")
    for k, want in EXPECT_OASA.items():
        got = a.get(k, "<缺失>")
        ok = (got == want) or (isinstance(want, float) and isinstance(got, (int, float))
                               and abs(float(got) - want) < 1e-9)
        print(f"  [{'PASS' if ok else 'FAIL'}] {k:32s} = {got!r}" + ("" if ok else f"   期望 {want!r}"))
        if not ok:
            fails.append(k)

    print("\n--- 其余关键键（同样来自 ckpt）---")
    for k, want in EXPECT_OTHER.items():
        got = a.get(k, "<缺失>")
        ok = (got == want) or (isinstance(want, float) and isinstance(got, (int, float))
                               and abs(float(got) - want) < 1e-9)
        print(f"  [{'PASS' if ok else 'FAIL'}] {k:32s} = {got!r}" + ("" if ok else f"   期望 {want!r}"))
        if not ok:
            fails.append(k)

    print("\n--- 必须看到 args.yaml 里 name / data ---")
    for k in ("name", "data", "project"):
        print(f"       {k:32s} = {a.get(k)!r}")

    print("\n--- 续训时可被当前配置覆盖的 4 项（其余不可覆盖）---")
    print(f"  {sorted(OVERRIDABLE)}")

    print("\n" + "=" * 74)
    if fails:
        print(f"RESULT: FAIL ({len(fails)}) -> 不要续训，先解决")
        print(f"        {fails}")
        return 1
    print("RESULT: PASS — checkpoint 里的 args 与预期一致，可以续训")
    print(f"        续训将从 epoch {(ep or 0) + 1} 继续到 {a.get('epochs')}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
