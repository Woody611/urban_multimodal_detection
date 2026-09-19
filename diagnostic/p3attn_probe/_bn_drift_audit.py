"""diagnostic/p3attn_probe/_bn_drift_audit.py — 只读 BN 漂移审计。

用途：判断 P3 Identity Attention Probe 的官方成绩差异，是来自新增 attention，
还是来自「被冻结 backbone 的 BN running stats 仍会更新」（框架行为，本次不修改 trainer.py）。

做法（纯读取两个 checkpoint，不训练、不写任何正式产物）：
  1. 逐键比对 正式 best.pt 与 Probe best.pt 的**全部参数**
     —— 预期：**只有 model.16.attn.fc.{weight,bias} 不同**，其余参数逐位相同
        （这本身就是「冻结确实生效」的最强证据；若其余参数也变了，说明冻结失败）
  2. 逐层比对全部 BatchNorm 的 buffers：running_mean / running_var / num_batches_tracked
  3. 输出最大绝对变化、平均变化、最大相对变化，并列出明显漂移的层
  4. 给出裁决：BN 漂移是否可忽略

用法:
  python diagnostic/p3attn_probe/_bn_drift_audit.py \
      --reference runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt \
      --probe     runs/urban_multimodal_det_yolo11_rgbid_p3attn_probe/weights/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import attempt_load_one_weight  # noqa: E402

# 判定阈值（相对漂移）。1e-3 表示 BN 统计量变化超过千分之一即视为「明显」。
REL_DRIFT_FLAG = 1e-3
ABS_DRIFT_FLAG = 1e-3


def load_sd(path):
    m, _ = attempt_load_one_weight(str(path))
    return m.float().state_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="正式保底 best.pt（= Probe 的训练起点）")
    ap.add_argument("--probe", required=True, help="Probe 训练后的 best.pt")
    a = ap.parse_args()
    ref_p, prb_p = ROOT / a.reference, ROOT / a.probe
    for p in (ref_p, prb_p):
        if not p.exists():
            print(f"!! 不存在: {p}")
            return 2

    ref, prb = load_sd(ref_p), load_sd(prb_p)
    print("=" * 88)
    print("BN 漂移审计（只读）")
    print("=" * 88)
    print(f"  参考(训练起点) = {ref_p}")
    print(f"  Probe(训练后)  = {prb_p}")
    print(f"  键数: ref={len(ref)}  probe={len(prb)}")

    # ---------------- 1) 全部参数比对 ----------------
    print("\n" + "-" * 88)
    print("1) 全部参数（权重/偏置）逐键比对 —— 「冻结是否真的生效」")
    print("-" * 88)
    only_ref = sorted(set(ref) - set(prb))
    only_prb = sorted(set(prb) - set(ref))
    common = sorted(set(ref) & set(prb))
    changed, identical = [], 0
    for k in common:
        if ref[k].shape != prb[k].shape:
            changed.append((k, "shape 不一致"))
        elif not torch.equal(ref[k], prb[k]):
            changed.append((k, float((ref[k].float() - prb[k].float()).abs().max())))
        else:
            identical += 1
    print(f"  仅 ref 有   : {len(only_ref)}  {only_ref if only_ref else ''}")
    print(f"  仅 probe 有 : {len(only_prb)}  {only_prb if only_prb else ''}")
    print(f"  完全相同    : {identical} / {len(common)}")
    print(f"  发生变化的  : {len(changed)}")
    for k, v in changed[:20]:
        print(f"     {k}   max|Δ|={v}")
    non_attn_changed = [k for k, _ in changed if ".attn." not in k and k in ref and ref[k].dtype.is_floating_point]
    is_param = [k for k in non_attn_changed if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))]
    print()
    print(f"  => 除 attn 外发生变化的『参数』: {len(is_param)} 个"
          f"  {'(冻结生效 ✅)' if not is_param else '(冻结未完全生效 ⚠️)'}")
    for k in is_param[:20]:
        print(f"     {k}")

    # ---------------- 2) BN buffers ----------------
    print("\n" + "-" * 88)
    print("2) BatchNorm buffers 逐层比对（running_mean / running_var）")
    print("-" * 88)
    bn_keys = [k for k in common if k.endswith(("running_mean", "running_var"))]
    if not bn_keys:
        print("  !! 未找到任何 BN running 统计量")
        return 1
    n_layers = len(bn_keys) // 2
    print(f"  BN 张量数 = {len(bn_keys)}（约 {n_layers} 个 BN 层）")

    rows = []
    for k in bn_keys:
        d = (ref[k].float() - prb[k].float()).abs()
        denom = ref[k].float().abs().clamp_min(1e-12)
        rel = (d / denom)
        rows.append(dict(key=k, max_abs=float(d.max()), mean_abs=float(d.mean()),
                         max_rel=float(rel.max()), ref_scale=float(ref[k].float().abs().mean())))

    mean_rows = [r for r in rows if r["key"].endswith("running_mean")]
    var_rows = [r for r in rows if r["key"].endswith("running_var")]

    def summarize(name, rs):
        print(f"\n  【{name}】 n={len(rs)}")
        print(f"     最大绝对变化 = {max(r['max_abs'] for r in rs):.6e}")
        print(f"     平均绝对变化 = {sum(r['mean_abs'] for r in rs)/len(rs):.6e}")
        print(f"     最大相对变化 = {max(r['max_rel'] for r in rs):.6e}")

    summarize("running_mean", mean_rows)
    summarize("running_var", var_rows)

    # ---------------- 3) 明显漂移的层 ----------------
    # 主判据用【绝对】漂移：relative 指标在参考值接近 0 的通道上会失真
    # （实测自测：max|Δ|=1e-6 的层 max_rel 仍可达 2.9e-2）。故分两层列出。
    print("\n" + "-" * 88)
    print("3) 漂移层清单（主判据 = 绝对漂移；相对漂移仅作参考）")
    print("-" * 88)
    major = [r for r in rows if r["max_abs"] > ABS_DRIFT_FLAG]
    minor_rel = [r for r in rows if r["max_abs"] <= ABS_DRIFT_FLAG and r["max_rel"] > REL_DRIFT_FLAG]
    print(f"  【显著漂移】（max|Δ| > {ABS_DRIFT_FLAG}）: {len(major)} / {len(rows)} 个 BN 张量")
    for r in sorted(major, key=lambda z: -z["max_abs"])[:25]:
        print(f"     {r['key']:<58} max|Δ|={r['max_abs']:.4e}  max_rel={r['max_rel']:.4e}")
    print(f"\n  【相对大但绝对可忽略】（max|Δ| <= {ABS_DRIFT_FLAG} 且 max_rel > {REL_DRIFT_FLAG}）: "
          f"{len(minor_rel)} 个 —— 参考值接近 0 导致，不构成风险")
    for r in sorted(minor_rel, key=lambda z: -z["max_rel"])[:10]:
        print(f"     {r['key']:<58} max|Δ|={r['max_abs']:.4e}  max_rel={r['max_rel']:.4e}")

    # ---------------- 4) 裁决 ----------------
    g_mean = max(r["max_abs"] for r in mean_rows)
    g_var = max(r["max_abs"] for r in var_rows)
    g_rel = max(r["max_rel"] for r in rows)
    n_major = len(major)
    print("\n" + "=" * 88)
    print("裁决")
    print("=" * 88)
    print(f"  running_mean 最大绝对变化 = {g_mean:.6e}")
    print(f"  running_var  最大绝对变化 = {g_var:.6e}")
    print(f"  全局最大相对变化          = {g_rel:.6e}   （仅供参考，受近零通道放大）")
    print(f"  显著漂移(绝对)张量数      = {n_major} / {len(rows)}")
    print(f"  冻结层参数是否完全未动    = {not is_param}")
    print()
    if g_mean < 1e-3 and g_var < 1e-3:
        print("  => 【BN 漂移可忽略】（绝对判据）。可以主要将结果解释为 P3 attention 的影响。")
    else:
        print("  => 【BN 漂移明显】。必须标记：")
        print("     本实验不能作为纯 Attention 增益证据，")
        print("     结果可能包含 frozen-backbone BN drift 的影响。")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
