"""_e1_pre_resume_check.py — E1（RegionResponseGain）续训前硬检查。只读。

为什么 E1 比一般 resume 多一重风险：
  E1 的架构开关 `e1_enabled` / `e1_layers` 写在 **model yaml** 里，不是 train_args。
  resume 时 ultralytics 走 `setup_model()` → `cfg = weights.yaml`（**checkpoint 内嵌的 yaml**），
  完全不看 `--model_config`。所以：
    · 若 checkpoint 内嵌 yaml 缺 e1_layers → 重建出一个**没有 E1** 的模型
      → 参数量对不上 → `model.load()` 大面积缺键 → 但**不会报错**（strict=False）
      → 静默退化成「无 E1 的架构承接 E1 的权重」。这正是本项目反复踩的静默坑。
  ⇒ 本脚本必须直接读 `ckpt['model'].yaml` 复核，并**重建模型核对参数量**。

用法（云端）: python diagnostic/e1_region_gain/_e1_pre_resume_check.py <last.pt>
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXPECT_ARG = {
    "epochs": 300, "patience": 0, "batch": 8, "imgsz": 1280, "workers": 4,
    "lr0": 0.005, "seed": 42, "mosaic": 1.0, "close_mosaic": 10,
    "channels": 5, "use_simotm": "RGBID", "ir_encoding": "clahe",
}
EXPECT_NAME = "urban_multimodal_det_yolo11_rgbid_sepstem_clahe_e1"
EXPECT_E1_LAYERS = [13, 15, 17]
EXPECT_PARAMS_WITH_E1 = 20_077_332      # e1_enabled=true
EXPECT_PARAMS_NO_E1 = 20_061_972        # e1_enabled=false（退化后的参数量）

# resume 时唯一允许被当前 kwargs 覆盖的 4 项
OVERRIDABLE = {"imgsz", "batch", "device", "close_mosaic"}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    ck = Path(sys.argv[1])
    if not ck.exists():
        print(f"!! checkpoint 不存在: {ck}")
        return 1

    d = torch.load(str(ck), map_location="cpu", weights_only=False)
    a = dict(d.get("train_args") or {})
    ep, bf = d.get("epoch"), d.get("best_fitness")
    stripped = (ep == -1) or (d.get("optimizer") is None and d.get("ema") is None)
    fails = []

    print("=" * 80)
    print(f"E1 续训前检查: {ck}")
    print("=" * 80)
    print(f"  文件大小            : {ck.stat().st_size:,} bytes")
    if stripped:
        print("  !! strip_optimizer 之后的最终产物：epoch=-1、optimizer/ema 已抹除")
        print("     => **不能用于 resume**，请改用训练中途保存的 last.pt")
        fails.append("checkpoint_is_stripped")
    else:
        print(f"  ckpt['epoch'] (0-idx) : {ep}   -> 续训显示为 epoch {ep + 2}/300")
    print(f"  best_fitness        : {bf}")
    print(f"  optimizer / ema     : {'有' if d.get('optimizer') is not None else '缺失!!'} / "
          f"{'有' if d.get('ema') is not None else '缺失!!'}")
    print(f"  train_args 键数      : {len(a)}")

    # ---------- 1. train_args ----------
    print("\n--- ① train_args（**resume 时实际生效的那一份**）---")
    for k, want in EXPECT_ARG.items():
        got = a.get(k, "<缺失>")
        ok = (got == want) or (isinstance(want, float) and isinstance(got, (int, float))
                               and abs(float(got) - want) < 1e-9)
        print(f"  [{'PASS' if ok else 'FAIL'}] {k:16s} = {got!r}" + ("" if ok else f"   期望 {want!r}"))
        if not ok:
            fails.append(f"arg:{k}")
    nm = a.get("name")
    print(f"  [{'PASS' if nm == EXPECT_NAME else 'FAIL'}] {'name':16s} = {nm!r}")
    if nm != EXPECT_NAME:
        fails.append("arg:name")

    # ---------- 2. E1 架构（**本脚本的核心**）----------
    print("\n--- ② E1 架构开关（读 **checkpoint 内嵌的 model yaml**，不是 --model_config）---")
    # ⚠️ ultralytics 的**中途** checkpoint 存 "model": None、真权重在 "ema"；
    #    最终(strip 后)的产物反过来。与 attempt_load_one_weight 同样用 `model or ema`。
    m0 = d.get("model") or d.get("ema")
    print(f"  [诊断] type(ckpt['model'])      = {type(d.get('model'))}")
    print(f"  [诊断] type(ckpt['ema'])        = {type(d.get('ema'))}   (取用后者：model or ema)")
    print(f"  [诊断] type(ckpt['train_args']['model']) = {type(a.get('model'))}")
    print(f"  [诊断] ckpt['train_args']['model'] = {a.get('model')!r}")
    my = getattr(m0, "yaml", None)
    if isinstance(my, str):                       # 有些版本把 yaml 存成路径串
        try:
            from ultralytics.nn.tasks import yaml_model_load
            my = yaml_model_load(my)
            print(f"  [诊断] .yaml 是字符串路径，已 yaml_model_load 成 dict")
        except Exception as e:
            print(f"  [诊断] .yaml 字符串解析失败: {e}")
    if not isinstance(my, dict):                  # 回退：用训练时记录的 model 路径
        mp = a.get("model")
        if mp and Path(str(mp)).exists():
            from ultralytics.nn.tasks import yaml_model_load
            my = yaml_model_load(str(mp))
            print(f"  [诊断] 回退用 train_args['model'] = {mp} 作为架构来源")
    if not isinstance(my, dict):
        print("  [FAIL] 三个来源都拿不到可读的 model yaml ⇒ 无法确认架构")
        print("         （请把上面的 [诊断] 三行贴回，这是脚本的读取问题，不一定是 E1 的问题）")
        fails.append("no_embedded_yaml")
    else:
        e1e, e1l = my.get("e1_enabled"), my.get("e1_layers")
        print(f"  [{'PASS' if e1e is True else 'FAIL'}] e1_enabled = {e1e!r}")
        print(f"  [{'PASS' if list(e1l or []) == EXPECT_E1_LAYERS else 'FAIL'}] "
              f"e1_layers  = {e1l!r}   期望 {EXPECT_E1_LAYERS}")
        if e1e is not True:
            fails.append("e1_enabled")
        if list(e1l or []) != EXPECT_E1_LAYERS:
            fails.append("e1_layers")

        # ---------- 3. 按该 yaml 重建模型并核对参数量 ----------
        print("\n--- ③ 按 checkpoint 的 yaml 重建模型，核对参数量与 E1 模块数 ---")
        try:
            from ultralytics.nn.tasks import DetectionModel
            m = DetectionModel(my, nc=12, verbose=False)
            n = sum(p.numel() for p in m.parameters())
            # ⚠️ RegionResponseGain 是**嵌在 layer 里的 .e1 子模块**（见 models/yolo/detect/train.py:164
            #    的 `getattr(model.model[i], "e1", None)`），**不是** m.model 的独立顶层条目。
            #    首版按类名扫 m.model ⇒ 恒得 [] 的假失败。
            mods = [i for i, mm in enumerate(m.model)
                    if getattr(mm, "e1", None) is not None]
            n_e1_t = sum(1 for k in m.state_dict() if ".e1." in k)
            print(f"  含 .e1 子模块的 layer = {mods}   期望 {EXPECT_E1_LAYERS}")
            print(f"  state_dict 中 '.e1.' 张量数 = {n_e1_t}   期望 6"
                  f"（3 个模块 × (dw.weight + dw.bias)）")
            if n_e1_t != 6:
                fails.append("e1_tensor_count")
            print(f"  重建参数量 = {n:,}   （e1_enabled=true 应为 {EXPECT_PARAMS_WITH_E1:,}；"
                  f"false 则为 {EXPECT_PARAMS_NO_E1:,}）")
            print(f"  RegionResponseGain 实例 = {mods}   期望 {EXPECT_E1_LAYERS}")
            ok_p = (n == EXPECT_PARAMS_WITH_E1)
            print(f"  [{'PASS' if ok_p else 'FAIL'}] 参数量匹配 E1 架构")
            if not ok_p:
                fails.append("param_count")
                if n == EXPECT_PARAMS_NO_E1:
                    print("     ^^^ 这是 **无 E1** 的参数量 ⇒ checkpoint 未启用 E1，不要续训")
            ok_m = (mods == EXPECT_E1_LAYERS)
            print(f"  [{'PASS' if ok_m else 'FAIL'}] E1 模块位置正确")
            if not ok_m:
                fails.append("e1_module_positions")
        except Exception as e:
            print(f"  [FAIL] 重建失败: {type(e).__name__}: {e}")
            fails.append("rebuild_failed")

    print("\n--- ④ resume 时可被当前配置覆盖的 4 项（其余一律来自 ckpt）---")
    print(f"  {sorted(OVERRIDABLE)}")

    print("\n" + "=" * 80)
    if fails:
        print(f"RESULT: FAIL ({len(fails)}) -> STOP，不要续训")
        print(f"        {fails}")
        return 1
    print(f"RESULT: PASS — 可从 epoch {(ep or 0) + 1}（显示 {ep + 2}/300）续训")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
