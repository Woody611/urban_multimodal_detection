"""_dryrun_local.py — 无 GPU / 无训练 / 无 forward 的端到端干跑，堵住"未验证面"。

背景：`finalize()` 与 A/B/C/D 判决逻辑在本机无法通过真实训练执行（无 GPU）；
上一轮上线时正是这类"接线"出了两次错（传错 JSON、resume 键不存在）。
本脚本把**除 forward 与 optimizer 之外的全部代码路径**跑一遍：

  D1  P3TrajProbe 构造（读 medium_cells.json；断言 G-86 == 86）
  D2  build()（读 val labels，选图数 == 281；**不做任何 forward**）
  D3  _nearest_centroid（合成向量；验证 control-only 质心 + 三角形状的准确率行为）
  D4  _decide（**合成轨迹**；验证 CASE A/B/C/D 四种情况都能被正确判出）
  D5  _render（decision.md 渲染不崩）
  D6  dump_epoch / finalize 的 CSV 写出（用合成记录，不碰真实 npz 语义）
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diagnostic/p3_trajectory_probe"))
from _probe import P3TrajProbe  # noqa: E402

OUT = Path(__file__).resolve().parent
FAILS = []


def ck(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}")
    if not ok:
        FAILS.append(n)


print("=" * 92); print("LOCAL DRY-RUN（无 GPU / 无训练 / 无 forward）"); print("=" * 92)

# ---------- D1 构造 ----------
tmp = Path(tempfile.mkdtemp(prefix="p3probe_dry_"))
p = P3TrajProbe(OUT / "medium_cells.json", tmp)
ck("D1 P3TrajProbe 构造 + G-86 == 86", len(p.g86) == 86, f"cells={len(p.cells)} g86={len(p.g86)}")
ck("D1b 角色分布符合 audit", (sum(r["role"] == "CONTROL" for r in p.cells) == 1070), "")

# ---------- D2 build()（只读 labels）----------
try:
    p.build()
    f = Path(p._ds.im_files[0]).name
    ck("D2 build() 选图数 == 281 且无 forward", len(p._sel) == 281, f"sel={len(p._sel)} 例={f}")
except Exception as e:
    ck("D2 build()", False, repr(e))

# ---------- D3 _nearest_centroid（合成向量）----------
rng = np.random.default_rng(0)
# 用 (class, role) 成对构造，保证 cls / role 长度严格一致（上一版是测试脚本自己的长度 bug）
_layout = []
for c in (0, 1, 2):
    _layout += [(c, "CONTROL")] * 150 + [(c, "G86")] * 20 + [(c, "MISSED_NON86")] * 10
cls = np.array([c for c, _ in _layout])
role = np.array([r for _, r in _layout], dtype=object)
assert len(cls) == len(role), (len(cls), len(role))
# 构造 3 类各一个方向；G86 的样本加噪声（可分性更差）
MU = rng.normal(size=(3, 16))
F = np.zeros((len(cls), 16), np.float32)
for i in range(len(cls)):
    F[i] = MU[cls[i]] * 1.0 + rng.normal(scale=0.35, size=16)
g86_idx = np.where(role == "G86")[0]
F[g86_idx] = MU[cls[g86_idx]] * 0.25 + rng.normal(scale=1.2, size=(len(g86_idx), 16))
m = P3TrajProbe._nearest_centroid(None, cls, role, F)
ck("D3 _nearest_centroid 跑通且 CONTROL > G86 > 随机",
   m["CONTROL"] > m["G86"] and "majority_prior" in m and m["G86"] < m["CONTROL"],
   f"G86={m['G86']:.3f} CONTROL={m['CONTROL']:.3f} prior={m['majority_prior']:.3f} "
   f"MISSED={m.get('MISSED_NON86', float('nan')):.3f}")
ck("D3b CONTROL 组准确率高（说明质心路线正确）", m["CONTROL"] > 0.9, f"{m['CONTROL']:.3f}")


# ---------- D4 _decide（合成轨迹）----------
def mk(eps, p3g, p3c, hg, hc, lg, align, mp, prior=0.29):
    p3 = [dict(epoch=e, feature="p3", G86=p3g[i], CONTROL=p3c[i],
               MISSED_NON86=p3c[i] - 0.05, majority_prior=prior) for i, e in enumerate(eps)]
    hh = [dict(epoch=e, feature="h", G86=hg[i], CONTROL=hc[i], MISSED_NON86=hc[i] - 0.05) for i, e in enumerate(eps)]
    ser = []
    for i, e in enumerate(eps):
        # _decide 现在按 role 读 G-86 行 ⇒ 合成 ser 必须带 role="G86"（真实 finalize 也会写该列）
        ser += [dict(epoch=e, role="G86", field="logit", median=lg[i], n=86, mean=lg[i], p25=0, p75=0),
                dict(epoch=e, role="G86", field="raw_align_pos_max", median=align[i], n=86, mean=align[i], p25=0, p75=0),
                dict(epoch=e, role="G86", field="mask_pos_membership_rate", median=mp[i], n=86, mean=mp[i], p25=0, p75=0)]
    return p3, hh, ser


E = [0, 1, 2, 5, 10, 15, 20, 25, 30]
cases = {}
# A: 起点即 ≈ 随机，且无下降
p3, hh, ser = mk(E, [0.28, 0.28, 0.29, 0.30, 0.30, 0.31, 0.30, 0.30, 0.29], [0.55] * 9,
                 [0.30] * 9, [0.55] * 9, [0.0] * 9, [0.02] * 9, [0.5] * 9)
cases["A"] = p._decide(E, p3, hh, ser)
# B: 早期明显 > 随机，随后显著下降，且 **P3 先退化** → cls → align → mask_pos（顺序稳定）
#    P3 于 ep5 首降(0.60→0.45)；cls 于 ep10；align 于 ep20；mask_pos 于 ep25
p3, hh, ser = mk(E, [0.60, 0.60, 0.58, 0.45, 0.40, 0.38, 0.35, 0.32, 0.30], [0.60] * 9,
                 [0.60, 0.60, 0.58, 0.45, 0.40, 0.38, 0.35, 0.32, 0.30], [0.60] * 9,
                 [3.0, 3.0, 3.0, 3.0, 1.0, 0.5, 0.0, -0.5, -1.0],
                 [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.08, 0.04, 0.02],
                 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.5])
cases["B"] = p._decide(E, p3, hh, ser)
# C: P3 全程 ≈ 随机，但 h 明显 > 随机**且稳定**（无 ≥0.10 的下降）
p3, hh, ser = mk(E, [0.28, 0.28, 0.29, 0.30, 0.29, 0.31, 0.30, 0.31, 0.30], [0.60] * 9,
                 [0.30, 0.35, 0.40, 0.50, 0.58, 0.60, 0.62, 0.61, 0.62], [0.62] * 9,
                 [-2.0] * 9, [0.05] * 9, [0.6] * 9)
cases["C"] = p._decide(E, p3, hh, ser)
# D: P3 下降，但 mask_pos 完全不动
p3, hh, ser = mk(E, [0.60, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.31, 0.29], [0.60] * 9,
                 [0.60] * 9, [0.60] * 9, [3.0] * 9, [0.20] * 9, [1.0] * 9)
cases["D"] = p._decide(E, p3, hh, ser)

for want in ("A", "B", "C", "D"):
    got = cases[want]["CASE"]
    ck(f"D4 CASE {want} 被正确判出", got == want,
       f"→ CASE={got}  p3_decline={cases[want]['p3_decline_total']}  "
       f"orders=({cases[want]['first_P3_degradation']},{cases[want]['first_cls_degradation']},"
       f"{cases[want]['first_alignment_degradation']},{cases[want]['first_mask_pos_loss']})")

# ---------- D5 _render ----------
try:
    md = p._render(cases["B"])
    ck("D5 _render 输出含 CASE 与时间序列表", "CASE = B" in md and "| epoch |" in md, f"{len(md)} chars")
except Exception as e:
    ck("D5 _render", False, repr(e))

# ---------- D6 dump_epoch 写出（合成记录）----------
try:
    fake = [dict(epoch=0, key=f"x#{i}", class_id=0, role="G86", p3_cell_x=1, p3_cell_y=1,
                 logit=-1.0, sigmoid=0.3, p3_norm=1.0, h_norm=2.0, ciou_pos_median=0.5,
                 ciou_pos_max=0.5, ciou_pool_max=0.5, raw_align_pos_max=0.01, raw_align_pool_max=0.01,
                 cls_pos_max=0.1, cls_pool_max=0.1, cls_center_max=0.1, n_pos=10, n_topk=10,
                 mask_pos_membership=1, topk_membership=1, n_in_gts=40, pos_mean_ciou=0.5,
                 raw_align_pos_mean=0.01) for i in range(86)]
    _roles = (["G86"] * 86 + ["CONTROL"] * 1070 + ["MISSED_NON86"] * 164)
    _cls = [0] * 1320
    p.vecs[0] = dict(vec_keys=[f"x#{i}" for i in range(1320)], vec_cls=_cls,
                     vec_role=_roles,
                     vec_p3=np.zeros((1320, 256), np.float32), vec_h=np.zeros((1320, 256), np.float32))
    p.dump_epoch(0, fake)
    ok = (tmp / "epoch_000.npz").exists()
    p.finalize()
    files = {f.name for f in tmp.iterdir()}
    need = {"epoch_000.npz", "epoch_summary.csv", "per_gt_trajectory.csv", "p3_metrics.csv",
            "h_metrics.csv", "tal_metrics.csv", "assignment_metrics.csv", "decision.json", "decision.md"}
    ck("D6 dump_epoch + finalize 全部产物写出", ok and need <= files,
       f"缺失={sorted(need - files)}")
except Exception as e:
    ck("D6 finalize", False, repr(e))

print()
print(f"DRYRUN_RESULT = {'PASS' if not FAILS else 'FAIL ' + str(FAILS)}")
print(f"（本干跑不含任何 forward / optimizer / 训练；产物在临时目录 {tmp}）")
