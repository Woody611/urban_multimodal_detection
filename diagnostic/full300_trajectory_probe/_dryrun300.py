"""_dryrun300.py — 无 GPU / 无训练 的干跑，堵住"未验证面"。

D1  Probe300 构造（读 p3 目录的 medium_cells.json；G-86 == 86）
D2  build()（读 val labels；选图数 == 281；无 forward）
D3  dump_epoch 会写 head_W / head_b（§9C/§10 的前提）
D4  ★ _analyze300 的四分解与 A/B/C/D **在四种人造机制下逐一判对**
      A 只动 h（表征漂移）/ B 只动 W（分类器漂移）/ C 两者都动 / D 都不动
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P3 = HERE.parent / "p3_trajectory_probe"
# ★ 必须把 ROOT 放最前：脚本目录占 sys.path[0]，否则 _probe300 → _probe → ultralytics
#   会解析到 site-packages（缺 yaml_load）—— 该陷阱本轮已第 3 次出现。
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(P3))
from _probe300 import Probe300  # noqa: E402

FAILS = []


def ck(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}")
    if not ok:
        FAILS.append(n)


print("=" * 92); print("FULL-300 LOCAL DRY-RUN（无 GPU / 无训练 / 无 forward）"); print("=" * 92)

# ---------- D1 / D2 ----------
tmp = Path(tempfile.mkdtemp(prefix="full300_dry_"))
p = Probe300(P3 / "medium_cells.json", tmp, observe_epochs=(0, 1, 2, 299))
ck("D1 Probe300 构造 + G-86 == 86", len(p.g86) == 86, f"cells={len(p.cells)} g86={len(p.g86)}")
try:
    p.build()
    ck("D2 build() 选图数 == 281（无 forward）", len(p._sel) == 281, f"sel={len(p._sel)}")
except Exception as e:
    ck("D2 build()", False, repr(e))

# ---------- D3 ----------
try:
    p.vecs[0] = dict(vec_keys=["a"] * 1320, vec_cls=[0] * 1320,
                     vec_role=["CONTROL"] * 1320, vec_p3=np.zeros((1320, 256), np.float32),
                     vec_h=np.zeros((1320, 256), np.float32))
    p._W_t = np.zeros((12, 256)); p._b_t = np.zeros(12)
    p.dump_epoch(0, [dict(epoch=0, key="a", class_id=0, role="CONTROL", p3_cell_x=0, p3_cell_y=0,
                          logit=0.0, sigmoid=0.5, p3_norm=0.0, h_norm=0.0, ciou_pos_median=0.0,
                          ciou_pos_max=0.0, ciou_pool_max=0.0, raw_align_pos_max=0.0,
                          raw_align_pool_max=0.0, cls_pos_max=0.0, cls_pool_max=0.0, cls_center_max=0.0,
                          n_pos=0, n_topk=0, mask_pos_membership=0, topk_membership=0, n_in_gts=0,
                          pos_mean_ciou=0.0, raw_align_pos_mean=0.0)])
    z = np.load(tmp / "epoch_000.npz", allow_pickle=False)
    ck("D3 dump_epoch 写出 head_W / head_b", "head_W" in z.files and "head_b" in z.files,
       f"keys={sorted(z.files)[:6]}…")
except Exception as e:
    ck("D3 dump_epoch", False, repr(e))


# ---------- D4 四分解判决（四种人造机制） ----------
def synth(case):
    d = Path(tempfile.mkdtemp(prefix=f"an_{case}_"))
    N, C = 1320, 3
    rng = np.random.default_rng(0)
    roles = ["G86"] * 86 + ["CONTROL"] * 1070 + ["MISSED_NON86"] * 164
    cls = np.array([i % C for i in range(N)])
    role = np.array(roles)
    # 基向量：W0[c] = e0（所有类相同，避免类间干扰）；h0[:,1] = 1 ⇒ e1·h0 = 1
    W0 = np.zeros((C, 256)); W0[:, 0] = 1.0
    h0 = rng.normal(scale=0.01, size=(N, 256)); h0[:, 1] = 1.0
    b0 = np.zeros(C)
    steps = [0, 1, 2, 299]
    for t in steps:
        frac = t / max(steps[-1], 1)
        ht = h0.copy(); Wt = W0.copy()
        if case in ("A", "C"):        # 只把 G-86 的 h 沿 −e0 漂移 ⇒ W0·ht 下降
            ht[role == "G86"] = h0[role == "G86"] - 3.0 * frac * np.eye(256)[0]
        if case in ("B", "C"):        # 只把 G-86 所涉类的 W 沿 −e1 漂移 ⇒ Wt·h0 下降
            for c in np.unique(cls[role == "G86"]):
                Wt[c] = W0[c] - 3.0 * frac * np.eye(256)[1]
        np.savez_compressed(d / f"epoch_{t:03d}.npz",
                            vec_keys=np.array([f"k{i}" for i in range(N)]), vec_cls=cls, vec_role=role,
                            vec_p3=np.zeros((N, 256), np.float32), vec_h=ht.astype(np.float32),
                            head_W=Wt, head_b=b0)
    return d


for case in ("A", "B", "C", "D"):
    d = synth(case)
    r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "_analyze300.py"), "--dir", str(d)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    got = "none"
    for ln in (r.stdout or "").splitlines():
        if "⇒ CASE" in ln:
            got = ln.split("CASE")[-1].strip()
    ck(f"D4 机制 {case} 被判出", got == case, f"→ CASE {got}  exit={r.returncode}")
    if r.returncode != 0:
        print("      STDERR:", (r.stderr or "")[-400:])

print()
print(f"DRYRUN300_RESULT = {'PASS' if not FAILS else 'FAIL ' + str(FAILS)}")
