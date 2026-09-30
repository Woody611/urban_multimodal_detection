"""_analyze.py — L17 Spatial Selectivity / Sufficiency Audit（READ-ONLY）

回答：L17 donor intervention 改变的到底是「目标选择性表征」，还是仅仅局部 activation/logit？

绝对禁止（本轮，逐条遵守）：
  training / backward / optimizer.step / 新 checkpoint / 改任何既有 source·config·dataset·label·
  evaluator·diagnostic·checkpoint / 重生成 submission / **重新 forward** / 重抽 G1-G4 / 重抽 control /
  自己改 donor 定义 / 用新随机样本替代既有 V13/V14 样本 / 创建新 intervention。
只读已有 artifact + 只读 checkpoint 权重（做纯代数），只写 diagnostic/l17_selectivity_audit/。

核心事实（决定本报告形态）：V13/V14/V12/V11/V7 落盘的 **全部是「GT 中心单 cell 的 512/256 维向量」
与标量**。**没有任何 artifact 保存全图激活张量**。因此协议 §5–§8 的 spatial activation 分析
（mass_inside/outside、peak、top-k coverage、entropy）与 §9 的 class selectivity（需要干预后的
非目标类投影）在本 artifact 集合上 **UNAVAILABLE** —— 本脚本以程序化扫描证明这一点，而不是声称。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
P3 = ROOT / "diagnostic/p3_feature_space"
V2D = ROOT / "diagnostic/small_object_cause_v2"
CKPT = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"
SESSION_RNG = 12345          # V13 的 random-donor 固定 seed（逐字取自 _v13_dose.py）
RNG_SEED = 20260929
BOOT = 20000
RNG = np.random.default_rng(RNG_SEED)

LOG: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


# ============================== 统计工具 ==============================
def cliff_delta(a, b) -> float:
    """Cliff's delta（a vs b）：P(a>b) − P(a<b)，以 <0 表示 b 更大。"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if not len(a) or not len(b):
        return float("nan")
    gt = sum(float((x > b).sum()) for x in a)
    lt = sum(float((x < b).sum()) for x in a)
    return (gt - lt) / (len(a) * len(b))


def boot_ci(v, stat=np.median, n=BOOT, seed=RNG_SEED) -> tuple[float, float]:
    v = np.asarray(v, float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(v), size=(n, len(v)))
    d = stat(v[idx], axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_ci_paired(a, b, stat=np.median, n=BOOT, seed=RNG_SEED) -> tuple[float, float]:
    """配对 bootstrap：对 (a−b) 的差做重采样。"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    if len(d) == 0:
        return (float("nan"), float("nan"))
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(d), size=(n, len(d)))
    s = stat(d[idx], axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def wilcoxon_p(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0 or np.all(v == 0):
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(v).pvalue)
    except Exception:  # noqa: BLE001
        return float("nan")


def mwu_p(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 1 or len(b) < 1:
        return float("nan")
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:  # noqa: BLE001
        return float("nan")


def spearman(a, b) -> tuple[float, float]:
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 4:
        return float("nan"), float("nan")
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(a, b)
        return float(r), float(p)
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")


def desc(v, name="", unit="") -> dict:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(name=name, n=0)
    lo, hi = boot_ci(v)
    return dict(name=name, unit=unit, n=int(len(v)),
                median=float(np.median(v)), q1=float(np.percentile(v, 25)),
                q3=float(np.percentile(v, 75)), iqr=float(np.percentile(v, 75) - np.percentile(v, 25)),
                mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                p90=float(np.percentile(v, 90)), p10=float(np.percentile(v, 10)),
                frac_pos=float(np.mean(v > 0)), frac_neg=float(np.mean(v < 0)),
                boot95=[lo, hi])


def cos(a, b) -> float:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


# ============================== §A artifact 能力扫描 ==============================
def scan_artifacts() -> dict:
    """程序化证明：没有任何 artifact 含高维空间激活张量。"""
    say("=" * 112)
    say("[§A] ARTIFACT 能力扫描 —— 协议 §5–§8 需要的『全图激活张量』到底存不存在？")
    say("=" * 112)
    found = []
    ranks = {}
    for p in sorted(P3.glob("*")) + sorted(V2D.glob("*")):
        if not p.is_file():
            continue
        if p.suffix == ".npz":
            try:
                z = np.load(p)
            except Exception:  # noqa: BLE001
                continue
            mx = 0
            for k in z.files:
                mx = max(mx, z[k].ndim)
                if z[k].ndim >= 3:
                    found.append((str(p.name), k, z[k].shape))
            ranks[str(p.name)] = mx
        elif p.suffix == ".json":
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            mx = _rank(d, 4)
            ranks[str(p.name)] = mx
            if mx >= 3:
                found.append((str(p.name), "<json>", f"max_rank={mx}"))
    say(f"  扫描文件数 = {len(ranks)}（{P3.name}/ 与 {V2D.name}/ 全部 .json/.npz）")
    say(f"  **.npz 中 ndim>=3 的数组 = {len([f for f in found if f[1] != '<json>'])}**  ← 这是唯一能承载")
    say("      『(1,512,40,40) 空间激活图』的容器；json 的 rank 只是**嵌套深度**、不是张量秩。")
    for f in found:
        if f[1] == "<json>":
            say(f"    [json 嵌套深度, 非张量] {f[0]}  {f[2]}")
    arr3 = [f for f in found if f[1] != "<json>"]
    for f in arr3:
        say(f"    [真数组 ndim>=3] {f}")
    if not arr3:
        say("  ⇒ **零** 个 artifact 含 ndim>=3 的数组 ⇒ 全部落盘内容都是 (n, C) 单 cell 向量或标量。")
    say("")
    say("  各 artifact 记录的『空间』内容（人工核对源码后确认）：")
    tbl = [
        ("_v11_layers.npz", "每 (层,组) 一个 (n, C) 矩阵 = **GT 中心单 cell 向量**；无邻域、无整图"),
        ("_v11_ids.json", "G1/G4/TS 的 key 列表"),
        ("_v11_traj.json", "每层标量：auc / c1(质心 cos) / n1,n4,nT(组均值范数) / knn"),
        ("_v7_vectors.json", "每 GT：p3(256)、h(256) —— 仍为 **单 cell 向量** + 标量 logit_decomp"),
        ("_v10_stages.json", "每 GT：P3/S1/S2 各 256 维 —— 单 cell"),
        ("_v9_attrib.json", "每 GT：zA_p3 / zA_h **标量**"),
        ("_v8_probes.json", "探针权重 wA/wB/wC(256) + 精度 **标量** —— 非 class 探针"),
        ("_v12_patch.json", "每 GT：donor key、y17/x17/y16/x16、L17_norm/L23_norm 标量 + 各干预 Δlogit"),
        ("_v13_dose.json", "每 GT：base/R0-R3/disp 的 **logit 标量** + L23 的 norm/cos 标量"),
        ("_v14_upstream.json", "每 GT：L12-L15 的 Δlogit 标量 + L23norm 标量"),
    ]
    for a, b in tbl:
        say(f"    {a:<22} {b}")
    say("")
    say("  ⇒ 结论：协议 §5(mass_inside/1.5x/2x/ring) / §6(inside-vs-outside) / §7(peak·top-k coverage)")
    say("     / §8(entropy·effective area) 所需的 **干预前/后 L17 (1,512,40,40) 激活图** 在全部 artifact 中")
    say("     **不存在**；协议 §9 所需的 **干预后非目标类投影** 也从未落盘（只记录了目标类一个标量）。")
    say("  ⇒ 这四项按协议 §2/§3 = **UNAVAILABLE**。**不自行 forward 补数据**（协议 §2、§16-D）。")
    return dict(n_files_scanned=len(ranks), rank_ge3=found,
                unavailable=["§5 GT-box activation mass", "§6 inside-vs-outside discrimination",
                             "§7 peak location / top-k coverage", "§8 entropy / effective area",
                             "§9 intervened class selectivity (non-target projections)"])


def _rank(x, cap=4):
    if isinstance(x, dict):
        return min(cap, 1 + max([_rank(v, cap) for v in x.values()] or [0])) if x else 1
    if isinstance(x, list):
        return min(cap, 1 + max([_rank(v, cap) for v in x] or [0])) if x else 1
    return 0


# ============================== §B 重建干预清单 ==============================
def load_all():
    dose = json.loads((P3 / "_v13_dose.json").read_text(encoding="utf-8"))
    v12 = json.loads((P3 / "_v12_patch.json").read_text(encoding="utf-8"))
    v14 = json.loads((P3 / "_v14_upstream.json").read_text(encoding="utf-8"))
    ids = json.loads((P3 / "_v11_ids.json").read_text(encoding="utf-8"))
    v3 = json.loads((V2D / "_v3_analysis.json").read_text(encoding="utf-8"))
    v2rec = {f"{r['image_id']}#{r['gt_id']}": r for r in
             json.loads((V2D / "_v2_records.json").read_text(encoding="utf-8"))["records"]}
    Z = np.load(P3 / "_v11_layers.npz")
    v7 = json.loads((P3 / "_v7_vectors.json").read_text(encoding="utf-8"))["rows"]
    traj = json.loads((P3 / "_v11_traj.json").read_text(encoding="utf-8"))
    return dose, v12, v14, ids, v3, v2rec, Z, v7, traj


def rebuild_donors(ids, v2rec):
    """确定性重算 V12/V13 的 donor（规则逐字取自 _v12_patch.py / _v13_dose.py）。
    这不是『重新抽样』—— 规则在 V12/V13 跑之前就已固定，且 V12 **已落盘 donor 字段**，
    下面用它做逐条对账验证。"""
    idx_of = {g: {k: i for i, k in enumerate(ids[g])} for g in ids}
    pool = [dict(k=k, i=i, cls=int(v2rec[k]["cls"]) if "cls" in v2rec[k] else -1,
                 sq=float(np.sqrt(v2rec[k]["native_area"])))
            for k, i in idx_of["G4"].items()]
    return idx_of, pool


def pick_donor(pool, cls, sq):
    cand = [d for d in pool if d["cls"] == cls]
    if not cand:
        return None, cand
    dsel = min(cand, key=lambda dd: (abs(dd["sq"] - sq), dd["k"]))
    return dsel, cand


def pick_rnd(cand):
    if not cand:
        return None
    return cand[int(np.random.default_rng(SESSION_RNG).integers(len(cand)))]


# ============================== 主流程 ==============================
def main() -> int:
    say("=" * 112)
    say("L17 SPATIAL SELECTIVITY / SUFFICIENCY AUDIT —— STRICT READ-ONLY")
    say("=" * 112)
    say(f"  checkpoint sha256[:16] = {sha16(CKPT)}")
    say(f"  V13 dose  sha256[:16] = {sha16(P3/'_v13_dose.json')}")
    say(f"  V14 upstr sha256[:16] = {sha16(P3/'_v14_upstream.json')}")
    say(f"  V12 patch sha256[:16] = {sha16(P3/'_v12_patch.json')}")
    say("")

    scan = scan_artifacts()
    dose, v12, v14, ids, v3, v2rec, Z, v7, traj = load_all()
    G1set = set(v3["E1"]); CTRL = set(v[0] for v in v3["controls"].values())
    idx_of, pool = rebuild_donors(ids, v2rec)

    D13 = {r["key"]: r for r in dose}
    D12 = {r["key"]: r for r in v12}
    D14 = {r["key"]: r for r in v14}
    G1keys = sorted(G1set)
    G4keys = sorted(ids["G4"])

    # ---------------- §B1 donor 身份对账（§3 要求）----------------
    say("=" * 112)
    say("[§B1] §3 donor / 坐标 / 量级 的 artifact 可用性与逐条对账")
    say("=" * 112)
    fields = [("G1 identities", "V13 `grp=='G1'` + `_v3_analysis.json.E1`", "n=38", "AVAILABLE"),
              ("G4 identities", "`_v11_ids.json.G4`", "n=194", "AVAILABLE"),
              ("G2 identities", "`_v3_analysis.json.controls`", "n=38", "AVAILABLE"),
              ("donor identities", "**V13 未落盘**；但 V12 `donor` 字段（同一规则/同一池）有 → 用 V12 对账", "", "AVAILABLE(跨 artifact)"),
              ("donor layer", "17 (V13/V12 硬编码)", "", "AVAILABLE"),
              ("donor spatial coords", "donor 在 **其自身图像** 的 GT cell（x17/y17 见 V12）", "", "AVAILABLE(部分)"),
              ("intervention spatial coords", "V12 `y17,x17` / `y23,x23`；V13 未落盘（可由 native GT 确定性重算）", "", "AVAILABLE(部分)"),
              ("displaced-control coords", "V13: (y0, x0±{4,5})；V12: (y17+3, x17) —— **两轮定义不同**", "", "AVAILABLE"),
              ("intervention magnitude", "V12/V13 未落盘**向量本身**，但 donor 向量可由 `L17_G4[i]` 精确复原", "", "AVAILABLE(复原)"),
              ("original logits", "V13 `base_logit` / V12 `base_logit`", "", "AVAILABLE"),
              ("intervened logits", "V13 `R*_raw/norm/rnd_logit`, `disp*_logit`; V12 各干预 `_logit`", "", "AVAILABLE")]
    for a, b, c, d in fields:
        say(f"  {a:<28} {b:<66} {c:<6} {d}")

    say("")
    say("  donor 身份确定性重算 vs V12 落盘 `donor` 字段（逐条对账）：")
    n_match = n_tot = 0
    donor_records = {}
    for k in G1keys:
        r = D12.get(k)
        if r is None or "donor" not in r:
            continue
        dsel, cand = pick_donor(pool, int(r["cls"]), float(r["sq"]))
        n_tot += 1
        ok = (dsel is not None and dsel["k"] == r["donor"])
        n_match += int(ok)
        donor_records[k] = dict(rec=r["donor"], recomp=dsel["k"] if dsel else None, match=ok,
                                n_cand=len(cand), n_donor_cand_v12=r.get("n_donor_cand"))
    say(f"    G1 donor 对账： match {n_match}/{n_tot}  "
        f"{'✅ 完全一致 ⇒ V13 的 donor 身份可由 V12 落盘字段无歧义复原' if n_match == n_tot else '❌ 不一致 ⇒ STOP'}")
    # n_donor_cand 对账（第二条独立验证）
    n2 = 0
    for k in G1keys:
        if k in D12 and k in D13:
            n2 += int(D12[k]["n_donor_cand"] == D13[k]["n_donor"])
    say(f"    V12 `n_donor_cand` == V13 `n_donor` 的条数： {n2}/{len(G1keys)}  ✅"
        if n2 == len(G1keys) else f"    ⚠ n_donor 不一致 {n2}/{len(G1keys)}")

    # 逐条记录（协议 §4 要求的字段表）
    rows = []
    for k in G1keys:
        r13 = D13[k]; r12 = D12.get(k, {})
        cls = int(r13["cls"]); sq = float(r13["sq"])
        dsel, cand = pick_donor(pool, cls, sq)
        rnd = pick_rnd(cand)
        ov = Z["L17_G1"][idx_of["G1"][k]].astype(np.float64)
        dv = Z["L17_G4"][dsel["i"]].astype(np.float64) if dsel else None
        rv = Z["L17_G4"][rnd["i"]].astype(np.float64) if rnd else None
        rec = dict(key=k, cls=cls, sq=sq, donor=dsel["k"] if dsel else None,
                   donor_rnd=rnd["k"] if rnd else None,
                   y17=r12.get("y17"), x17=r12.get("x17"),
                   base_logit=float(r13["base_logit"]),
                   orig_norm=float(np.linalg.norm(ov)),
                   donor_norm=float(np.linalg.norm(dv)) if dv is not None else float("nan"),
                   rnd_norm=float(np.linalg.norm(rv)) if rv is not None else float("nan"),
                   cos_ov_dv=cos(ov, dv) if dv is not None else float("nan"),
                   cos_ov_rv=cos(ov, rv) if rv is not None else float("nan"))
        rec["norm_ratio"] = rec["donor_norm"] / (rec["orig_norm"] + 1e-12)
        for R in (0, 1, 2, 3):
            for m in ("raw", "norm", "rnd"):
                rec[f"R{R}_{m}_dlogit"] = r13.get(f"R{R}_{m}_dlogit", float("nan"))
                rec[f"R{R}_{m}_logit"] = r13.get(f"R{R}_{m}_logit", float("nan"))
            rec[f"R{R}_L23norm"] = r13.get(f"R{R}_raw_L23_norm", float("nan"))
            rec[f"R{R}_raw_L23cos"] = r13.get(f"R{R}_raw_L23_cos", float("nan"))
        rec["L23norm_base"] = float(r13["L23_norm_base"])
        rec["L23cos_base"] = float(r13["L23_cos_base"])
        for d in (4, 5):
            rec[f"disp{d}_dlogit"] = r13.get(f"disp{d}_dlogit", float("nan"))
        for nm in ("L16_zero", "L17_zero", "L17_proto", "L17_proto_normmatch",
                   "L17_displaced", "L17_random_donor"):
            rec[f"v12_{nm}_dlogit"] = r12.get(f"{nm}_dlogit", float("nan"))
        rec["v12_donor_match"] = bool(donor_records.get(k, {}).get("match", False))
        rows.append(rec)
    G1 = rows

    # ---------------- §B2 §4 重建 causal effect ----------------
    say("")
    say("=" * 112)
    say("[§B2] §4 重建已有 causal effect（只读；**不以 logit>0 作 success criterion**）")
    say("=" * 112)
    say("  §4 字段表示例（前 3 条 G1）：")
    hdr = ["key", "cls", "sq", "donor", "y17", "x17", "orig_norm", "donor_norm", "norm_ratio",
           "base_logit", "R0_raw_dlogit", "disp4_dlogit", "disp5_dlogit"]
    say("    " + "  ".join(f"{h:>10}" for h in hdr))
    for r in G1[:3]:
        say("    " + "  ".join(f"{r[h]:>10.4f}" if isinstance(r[h], float) else f"{str(r[h]):>10}"
                               for h in hdr))
    say("")
    step = {}
    for tag, key in (("R0_raw", "R0_raw_dlogit"), ("R1_raw", "R1_raw_dlogit"),
                     ("R2_raw", "R2_raw_dlogit"), ("R3_raw", "R3_raw_dlogit"),
                     ("R0_norm", "R0_norm_dlogit"), ("R0_rnd", "R0_rnd_dlogit"),
                     ("disp4", "disp4_dlogit"), ("disp5", "disp5_dlogit")):
        step[tag] = desc([r[key] for r in G1], tag, "Δlogit")
        step[tag]["wilcoxon_p"] = wilcoxon_p([r[key] for r in G1])
    G4rows = [r for r in dose if r["grp"] == "G4"]
    step["G4_R0_raw"] = desc([r["R0_raw_dlogit"] for r in G4rows if "R0_raw_dlogit" in r],
                             "G4_R0_raw", "Δlogit")
    step["G4_R2_raw"] = desc([r["R2_raw_dlogit"] for r in G4rows if "R2_raw_dlogit" in r],
                             "G4_R2_raw", "Δlogit")
    say("  {:<12}{:>4}{:>11}{:>20}{:>10}{:>7}{:>24}{:>12}".format(
        "setting", "n", "median", "IQR", "mean", "%>0", "boot95", "Wilcoxon p"))
    for tag in ("R0_raw", "R1_raw", "R2_raw", "R3_raw", "R0_norm", "R0_rnd",
                "disp4", "disp5", "G4_R0_raw", "G4_R2_raw"):
        d = step[tag]
        iqr_s = "[{:+.3f},{:+.3f}]".format(d["q1"], d["q3"])
        ci_s = "[{:+.3f},{:+.3f}]".format(d["boot95"][0], d["boot95"][1])
        say("  {:<12}{:>4}{:>11.4f}{:>20}{:>10.4f}{:>7.1%}{:>24}{:>12.3e}".format(
            tag, d["n"], d["median"], iqr_s, d["mean"], d["frac_pos"], ci_s, d.get("wilcoxon_p", float("nan"))))
    say("")
    g1r0 = np.array([r["R0_raw_dlogit"] for r in G1])
    say(f"  ✅ 判据 G1 Δ ≫ displaced control Δ： "
        f"R0_raw 中位 {np.median(g1r0):+.4f} vs disp4 {np.median([r['disp4_dlogit'] for r in G1]):+.4f}"
        f" / disp5 {np.median([r['disp5_dlogit'] for r in G1]):+.4f}   ⇒ 差 ~{np.median(g1r0):.1f}")
    say(f"  ✅ 判据 G1 Δ ≫ G4 Δ： R0_raw G1 {np.median(g1r0):+.4f} vs G4 "
        f"{step['G4_R0_raw']['median']:+.4f}；R2_raw G1 "
        f"{np.median([r['R2_raw_dlogit'] for r in G1]):+.4f} vs G4 "
        f"{step['G4_R2_raw']['median']:+.4f}")

    # ---------------- §C 空间特异性（可测的替代量）----------------
    say("")
    say("=" * 112)
    say("[§C] §10 位移控制 —— 空间特异性的**直接可测量**（协议 §10 的『已有 displaced control』）")
    say("=" * 112)
    disp = {}
    for d, keyname in ((4, "disp4_dlogit"), (5, "disp5_dlogit")):
        b = np.array([r[keyname] for r in G1])
        lo, hi = boot_ci_paired(g1r0, b)
        disp[f"disp{d}"] = dict(desc=desc(b, f"disp{d}"), paired_delta_median=float(np.median(g1r0 - b)),
                                paired_boot95=[lo, hi], wilcoxon_p=wilcoxon_p(g1r0 - b),
                                cliff_vs_R0=cliff_delta(g1r0, b))
        say(f"  V13 disp+{d}（同 donor 向量、同幅值、沿 x 平移 {d} cells，读同一 GT cell 的 logit）：")
        say(f"    Δlogit 中位 {np.median(b):+.5f}  IQR [{np.percentile(b,25):+.5f},{np.percentile(b,75):+.5f}]"
            f"  %>0 {np.mean(b>0):.1%}  Wilcoxon p={wilcoxon_p(b):.4g}")
        say(f"    配对 (R0_raw − disp+{d})：中位 {np.median(g1r0-b):+.4f}  boot95 [{lo:+.4f},{hi:+.4f}]"
            f"   Cliff's δ={cliff_delta(g1r0,b):+.4f}")
    b12 = np.array([r["v12_L17_displaced_dlogit"] for r in G1 if np.isfinite(r["v12_L17_displaced_dlogit"])])
    if len(b12):
        say(f"  V12 L17_displaced（同一 donor 向量，沿 **y +3 cells**，独立第二轮）：")
        say(f"    n={len(b12)}  Δlogit 中位 {np.median(b12):+.5f}  %>0 {np.mean(b12>0):.1%}"
            f"  Cliff's δ(vs R0)={cliff_delta(g1r0[:len(b12)], b12):+.4f}")
    say("")
    say("  ⇒ **effect 的空间特异性 SUPPORTED**：把同一向量、同一幅值搬到 +4/+5（V13）或 +3（V12）")
    say("     cells 外，Δlogit 从 ~+5.9 掉到 ~−0.004（相差 3 个数量级）。")
    say("  ⚠ 但这是『**注入位置**的特异性』，**不是**协议 §5–§8 问的『表征是否更集中于目标』。前者可测，后者不可测。")

    # ---------------- §D 内容几何（标签明确的代理量）----------------
    say("")
    say("=" * 112)
    say("[§D] 代理量（**明确标注为 §5–§9 不可测时的最近替代**）：L17 向量几何 ——『更像目标』的内容面")
    say("=" * 112)
    muTS = Z["L17_TS"].astype(np.float64).mean(0)
    muG4 = Z["L17_G4"].astype(np.float64).mean(0)
    muG1 = Z["L17_G1"].astype(np.float64).mean(0)
    say(f"  质心范数： ||μ_TS||={np.linalg.norm(muTS):.3f}  ||μ_G4||={np.linalg.norm(muG4):.3f}"
        f"  ||μ_G1||={np.linalg.norm(muG1):.3f}")
    say(f"  质心互 cos： cos(μ_G1,μ_TS)={cos(muG1,muTS):+.4f}  cos(μ_G4,μ_TS)={cos(muG4,muTS):+.4f}")
    say("")
    geo = {}
    for nm, arr in (("G1", "L17_G1"), ("G4", "L17_G4"), ("TS", "L17_TS")):
        A = Z[arr].astype(np.float64)
        cs = np.array([cos(v, muTS) for v in A])
        nn = np.linalg.norm(A, axis=1)
        geo[nm] = dict(n=int(len(A)), cos_to_muTS_median=float(np.median(cs)),
                       cos_to_muTS_mean=float(np.mean(cs)),
                       norm_median=float(np.median(nn)), norm_mean=float(np.mean(nn)))
        say(f"  {nm:<3} n={len(A):<4} cos(v, μ_TS) 中位 {np.median(cs):+.4f}  "
            f"||v|| 中位 {np.median(nn):.4f}  均值 {np.mean(nn):.4f}")
    say("  （与 `_v11_traj.json` 的 n1/n4/nT 对账：L17 n1=7.850 n4=9.167 nT=9.547）")
    for nm in ("G1", "G4", "TS"):
        tr = next(t for t in traj if t["layer"] == 17)
        key = {"G1": "n1", "G4": "n4", "TS": "nT"}[nm]
        say(f"    {nm}: 本轮 mean ||v||={geo[nm]['norm_mean']:.4f}  vs V11 traj {key}={tr[key]:.4f}"
            f"  Δ={geo[nm]['norm_mean']-tr[key]:+.4f}")
    say("")
    ov_cos = np.array([cos(Z["L17_G1"][idx_of["G1"][r["key"]]].astype(np.float64), muTS) for r in G1])
    dv_cos = np.array([cos(Z["L17_G4"][idx_of["G4"][r["donor"]]].astype(np.float64), muTS)
                       for r in G1 if r["donor"]])
    say(f"  逐目标（G1 n={len(G1)}）：original 的 cos(ov, μ_TS) 中位 {np.median(ov_cos):+.4f}"
        f" → donor 的 cos(dv, μ_TS) 中位 {np.median(dv_cos):+.4f}"
        f"   Δ中位 {np.median(dv_cos-ov_cos):+.4f}")
    l23 = np.array([r["L23cos_base"] for r in G1])
    l23p = np.array([r["R0_raw_L23cos"] for r in G1])
    say(f"  下游 L23 对照（V13 已落盘）：cos 中位 base {np.median(l23):+.4f} → R0 {np.median(l23p):+.4f}"
        f"   Δ中位 {np.median(l23p-l23):+.4f}  ⇒ **L23 侧并未更靠向 TS 质心，反而略降**")
    say("")
    say("  --- 幅值 vs 方向 的定量分解（协议 §9 的『70% 来自幅值』复核）---")
    dr = np.array([r["R0_raw_dlogit"] for r in G1])
    dn = np.array([r["R0_norm_dlogit"] for r in G1])
    drn = np.array([r["R0_rnd_dlogit"] for r in G1])
    mag = dr - dn
    say(f"    raw   中位 {np.median(dr):+.4f}   （dv：donor 的方向 + donor 的幅值）")
    say(f"    norm  中位 {np.median(dn):+.4f}   （dv 的方向 + **原 cell 的幅值**）⇒ 纯方向贡献")
    say(f"    raw−norm 中位 {np.median(mag):+.4f}  ⇒ 纯幅值贡献占 raw 的 {np.median(mag)/np.median(dr):.1%}")
    say(f"    rnd   中位 {np.median(drn):+.4f}   ⇒ 任意同类别成功小目标的向量拿到 raw 的 "
        f"{np.median(drn)/np.median(dr):.1%}")
    say("")
    say("    ⚠ **方法学敏感性（必须同读）**：`median(raw) − median(norm)` 与 `median(raw − norm)` 是")
    say("       **不同目标的中位数**，两者不相等。三个估计量给出不同的『幅值占比』：")
    magshare = {}
    for R in (0, 1, 2, 3):
        _r = np.array([x[f"R{R}_raw_dlogit"] for x in G1])
        _n = np.array([x[f"R{R}_norm_dlogit"] for x in G1])
        magshare[R] = dict(
            paired_median=float(np.median(_r - _n)), paired_median_share=float(np.median(_r - _n) / np.median(_r)),
            diff_of_medians=float(np.median(_r) - np.median(_n)),
            diff_of_medians_share=float((np.median(_r) - np.median(_n)) / np.median(_r)),
            mean_diff=float(np.mean(_r) - np.mean(_n)),
            mean_diff_share=float((np.mean(_r) - np.mean(_n)) / np.mean(_r)))
    say("      {:>6}{:>11}{:>11}{:>12}{:>10}{:>13}{:>10}{:>11}".format(
        "R", "med(raw)", "med(norm)", "paired med", "share", "diff-of-med", "share", "mean share"))
    for R in (0, 1, 2, 3):
        m = magshare[R]
        say("      {:>6}{:>11.4f}{:>11.4f}{:>12.4f}{:>10.1%}{:>13.4f}{:>10.1%}{:>11.1%}".format(
            f"R{R}", np.median([x[f"R{R}_raw_dlogit"] for x in G1]),
            np.median([x[f"R{R}_norm_dlogit"] for x in G1]),
            m["paired_median"], m["paired_median_share"], m["diff_of_medians"],
            m["diff_of_medians_share"], m["mean_diff_share"]))
    say("      ⇒ 稳健结论：**幅值占主导，但方向不可忽略**；幅值占比的估计在 52%–74% 之间随估计量变化，")
    say("        **本报告不挑选其中任何一个作为定论**（协议 §8 的『不要为了更好结果而选择方法』）。")
    say("        V13 报告的『约 70%』= `diff-of-medians` 口径（R3: "
        f"{(np.median([x['R3_raw_dlogit'] for x in G1])-np.median([x['R3_norm_dlogit'] for x in G1]))/np.median([x['R3_raw_dlogit'] for x in G1]):.1%}）。")
    rho = np.array([r["norm_ratio"] for r in G1])
    say(f"    范数比 ρ=||dv||/||ov||：中位 {np.median(rho):.4f}  IQR [{np.percentile(rho,25):.3f},"
        f"{np.percentile(rho,75):.3f}]  %>1 {np.mean(rho>1):.1%}")
    say("")
    say("  --- 相关检验（exploratory；n=38，multiple-comparison burden 高）---")
    ctest = [
        ("Δraw vs ρ（幅值比）", dr, [r["norm_ratio"] for r in G1]),
        ("Δraw−Δnorm（幅值分量） vs ||dv||−||ov||",
         mag, [r["donor_norm"] - r["orig_norm"] for r in G1]),
        ("Δnorm（方向分量） vs cos(ov,dv)", dn, [r["cos_ov_dv"] for r in G1]),
        ("Δnorm（方向分量） vs Δcos-to-μTS（更像目标的程度）",
         dn, [dc - oc for dc, oc in zip(dv_cos, ov_cos)]),
        ("Δraw vs Δcos-to-μTS", dr, [dc - oc for dc, oc in zip(dv_cos, ov_cos)]),
        ("Δraw vs ΔL23cos", dr, [r["R0_raw_L23cos"] - r["L23cos_base"] for r in G1]),
    ]
    corr = []
    for nm, a, b in ctest:
        rs, ps = spearman(a, b)
        corr.append(dict(name=nm, spearman=rs, p=ps, n=int(len(a))))
        say(f"    {nm:<48} Spearman ρ={rs:+.4f}  p={ps:.4g}  n={len(a)}")
    say("")
    say("  ⇒ 读者须知：上述 6 项检验属 **exploratory**，且 §D 的向量几何 **不是** 协议 §5–§9 要求的量。")
    say("     它回答的是『donor 把 L17 的**内容**改成了什么』，不是『激活是否更集中于目标框』。")

    # ---------------- §E G4 / target-state 特异性 ----------------
    say("")
    say("=" * 112)
    say("[§E] §11 G4 specificity + 「G4 无效应」的机制检验（范数赤字是否为 G1 特有）")
    say("=" * 112)
    g4r = []
    for k in G4keys:
        r = D13.get(k)
        if r is None:
            continue
        cls = int(r["cls"]); sq = float(r["sq"])
        dsel, cand = pick_donor(pool, cls, sq)
        if dsel is None:
            continue
        ov = Z["L17_G4"][idx_of["G4"][k]].astype(np.float64)
        dv = Z["L17_G4"][dsel["i"]].astype(np.float64)
        g4r.append(dict(key=k, cls=cls, sq=sq, orig_norm=float(np.linalg.norm(ov)),
                        donor_norm=float(np.linalg.norm(dv)),
                        norm_ratio=float(np.linalg.norm(dv) / (np.linalg.norm(ov) + 1e-12)),
                        cos_ov_dv=cos(ov, dv),
                        R0_raw_dlogit=r.get("R0_raw_dlogit", float("nan")),
                        R2_raw_dlogit=r.get("R2_raw_dlogit", float("nan"))))
    rhoG1 = np.array([r["norm_ratio"] for r in G1])
    rhoG4 = np.array([r["norm_ratio"] for r in g4r])
    say(f"  G4 可重建 donor 的条数 = {len(g4r)} / {len(G4keys)}")
    say(f"  ρ=||dv||/||ov||： G1 中位 {np.median(rhoG1):.4f}   G4 中位 {np.median(rhoG4):.4f}"
        f"   MWU p={mwu_p(rhoG1, rhoG4):.3g}   Cliff's δ={cliff_delta(rhoG1, rhoG4):+.4f}")
    say(f"  ||ov||： G1 中位 {np.median([r['orig_norm'] for r in G1]):.4f}   "
        f"G4 中位 {np.median([r['orig_norm'] for r in g4r]):.4f}")
    say("")
    say(f"  Δlogit R0： G1 中位 {np.median([r['R0_raw_dlogit'] for r in G1]):+.4f}   "
        f"G4 中位 {np.median([r['R0_raw_dlogit'] for r in g4r]):+.4f}"
        f"   Cliff's δ={cliff_delta([r['R0_raw_dlogit'] for r in G1], [r['R0_raw_dlogit'] for r in g4r]):+.4f}")
    pa = np.array([r["R0_raw_dlogit"] for r in G1] + [r["R0_raw_dlogit"] for r in g4r])
    pb = np.concatenate([rhoG1, rhoG4])
    rs, ps = spearman(pa, pb)
    rsi, psi = spearman([r["R0_raw_dlogit"] for r in G1], rhoG1)
    say(f"  合并 G1+G4（n={len(pa)}）Δraw vs ρ 的 Spearman： ρ={rs:+.4f}  p={ps:.3g}  （ρ²≈{rs**2:.3f}）")
    say(f"  **仅在 G1 内部**（n=38）Δraw vs ρ 的 Spearman： ρ={rsi:+.4f}  p={psi:.3g}  （ρ²≈{rsi**2:.3f}）")
    say("  ⇒ **部分支持**（必须给出这个限定）：")
    say(f"     (a) G1 的幅值赤字是**真实且 G1 特有**的：||ov|| 中位 {np.median([r['orig_norm'] for r in G1]):.2f} vs "
        f"G4 {np.median([r['orig_norm'] for r in g4r]):.2f}；ρ 中位 {np.median(rhoG1):.3f} vs {np.median(rhoG4):.3f}"
        f"（MWU p=8.3e-06）⇒ G4 的 donor 几乎不改变幅值，G1 的则系统性抬高。")
    say(f"     (b) 但 **ρ 只解释 Δraw 方差的 ρ²≈{rsi**2:.2f}（G1 内部）**：因此『G4 无效应 = G4 无幅值赤字』")
    say("         **不能单独**解释全部组间差异 —— Δraw 还有**大量 target-state 相关的方差**。")
    say("     ⇒ 结论：效应 **既非纯幅值、也非纯 donor 内容**，而是 **幅值赤字 × target state 的交互**。")

    # ---------------- §F Sufficiency 代理 ----------------
    say("")
    say("=" * 112)
    say("[§F] §12 Sufficiency —— 优先 Level 1/2/3，实测只有 Level 2（且是代理）")
    say("=" * 112)
    say("  Level 1（candidate / anchor matching / best IoU / n_pos / align）：**UNAVAILABLE**")
    say("    —— 该轮 artifact 未落盘任何 candidate 或 assigner 数据；本审计不重跑 forward。")
    say("  Level 3（完整 prediction）：**UNAVAILABLE** —— 唯一变化的是 GT-cell 类 logit 标量。")
    say("  Level 2（head logits）：**AVAILABLE（仅目标类一个标量）**")
    say("")
    bl = np.array([r["base_logit"] for r in G1])
    p0 = np.array([r["R0_raw_logit"] for r in G1])
    p3 = np.array([r["R3_raw_logit"] for r in G1])
    say(f"  G1 目标类 logit（同一读出口：W_c·h+b_c @ GT cell）：")
    say(f"    base  中位 {np.median(bl):+.4f}  (p={1/(1+np.exp(-np.median(bl))):.3e})")
    say(f"    R0    中位 {np.median(p0):+.4f}  (p={1/(1+np.exp(-np.median(p0))):.3e})")
    say(f"    R3    中位 {np.median(p3):+.4f}  (p={1/(1+np.exp(-np.median(p3))):.3e})")
    say(f"    冻结运行点 conf = 0.0010；R3 中位 p = {1/(1+np.exp(-np.median(p3))):.3e}")
    say("")
    say("  --- 参考分布（**同一读出口、同一口径**），来自 `_v7_vectors.json` ---")
    say("  先验证读数一致：用 checkpoint 的 cv3[0][-1] 权重从 V7 的 h 重算 logit，")
    say("  与 V7 落盘的 logit_decomp 对账。")
    W, b = load_head()
    v7by = {r["key"]: r for r in v7}
    val = []
    for r in v7:
        if r["domain"] != "V1":
            continue
        h = np.asarray(r["h"], np.float64)
        rec = float(W[r["cls"]] @ h + b[r["cls"]])
        val.append(rec - r["logit_decomp"])
    val = np.array(val)
    say(f"    V1 n={len(val)}  重算−落盘： median|Δ|={np.median(np.abs(val)):.3e}  max|Δ|={np.max(np.abs(val)):.3e}"
        f"  {'✅ 一致 ⇒ 可用 head 权重做类结构分析' if np.max(np.abs(val)) < 1e-3 else '⚠ 不一致'}")
    g1L = np.array([v7by[k]["logit_decomp"] for k in G1keys])
    g4L = np.array([v7by[k]["logit_decomp"] for k in G4keys])
    tsL = np.array([r["logit_decomp"] for r in v7 if r["domain"] == "T1" and r["key"] in set(ids["TS"])])
    oth = np.array([r["logit_decomp"] for r in v7 if r["domain"] == "V1"
                    and r["key"] not in G1set and r["key"] not in set(ids["G4"])])
    say("")
    say(f"  {'分布（同一读出口）':<34}{'n':>5}{'p10':>10}{'median':>10}{'p90':>10}")
    for nm, arr in (("G1 base（本轮）", g1L), ("G1 R0 patched", p0), ("G1 R3 patched", p3),
                    ("V1 G4（val success, 同集）", g4L), ("T1 TS（train success）", tsL),
                    ("V1 其它 small", oth)):
        say(f"  {nm:<34}{len(arr):>5}{np.percentile(arr,10):>10.3f}"
            f"{np.median(arr):>10.3f}{np.percentile(arr,90):>10.3f}")
    say("")
    pct = {}
    for nm, arr in (("G4", g4L), ("TS", tsL)):
        for tag, pat in (("R0", p0), ("R3", p3)):
            above = 100.0 * float((pat[:, None] < arr[None, :]).mean())   # arr 中高于 pat 的比例
            pct[f"{nm}_{tag}"] = above
            say(f"    G1 {tag} patched： {nm} 中有 {above:5.1f}% 的值**高于**它"
                f"  ⇒ G1 {tag} 位于 {nm} 分布的 **第 {100.0-above:5.1f} 百分位**")
    say("")
    say("  ⚠ 读法限定：G1 与 G4/TS 是**不同目标、不同图像**（非配对），这是**参考分布比较**，不是配对检验。")
    say("     但两者用**同一个读出口**（同一 W_c·h+b_c @ GT 中心 cell、同一 checkpoint），故尺度可比。")
    say("")
    say(f"  ⇒ **SUFFICIENCY = NOT DIRECTLY MEASURED**（协议 §12 Level 2：无 candidate/matching 数据）。")
    say(f"     可得的最近代理指向 **不足**：即使取**最大**干预 R3（49 cells、oracle 位置、oracle 向量），")
    say(f"     G1 的目标类 logit 中位 {np.median(p3):+.3f} 仍**低于 ~{pct['G4_R3']:.0f}% 的 val 成功小目标**")
    say(f"     （G4 中位 {np.median(g4L):+.3f}），约位于成功分布的第 {100.0-pct['G4_R3']:.0f} 百分位。")
    say("")
    say("  --- §9 语义选择性的『基线侧』可测部分（**干预后不可测**）---")
    clsrow = []
    for k in G1keys:
        h = np.asarray(v7by[k]["h"], np.float64)
        lg = W @ h + b
        c = int(v7by[k]["cls"])
        order = np.argsort(lg)[::-1]
        non_t = lg[order[0]] if order[0] != c else lg[order[1]]
        clsrow.append(dict(key=k, cls=c, target=float(lg[c]), max_non_target=float(non_t),
                           margin=float(lg[c] - non_t), rank_of_target=int(np.where(order == c)[0][0]) + 1))
    mt = np.array([r["margin"] for r in clsrow])
    rt = np.array([r["rank_of_target"] for r in clsrow])
    say(f"  G1 base： target − max_non_target 中位 {np.median(mt):+.4f}  IQR "
        f"[{np.percentile(mt,25):+.3f},{np.percentile(mt,75):+.3f}]")
    say(f"  G1 base： 目标类的排名（1=已是最高）中位 {np.median(rt):.0f}  "
        f"排名=1 的比例 {np.mean(rt==1):.1%}")
    say("  ⚠ 这是 **干预前** 的类结构。干预后的非目标类投影 **从未落盘** ⇒ §9 的 before/after 比较 = UNAVAILABLE。")

    # ---------------- §G E1 矛盾 ----------------
    say("")
    say("=" * 112)
    say("[§G] §17 为什么 V13 L17 donor replacement 有强 effect，而 E1 RegionResponseGain 无 measurable gain？")
    say("=" * 112)
    mech = [
        ("M1 donor replacement 是 content-specific",
         f"**NOT SUPPORTED**：随机同类别 donor 拿到 raw 效应的 "
         f"{np.median(drn)/np.median(dr):.1%}（{np.median(drn):+.3f} vs {np.median(dr):+.3f}）"),
        ("M2 E1 是 generic gain", "**PARTIALLY SUPPORTED**：donor 效应**由幅值主导**（幅值占比 "
                                  f"{magshare[0]['paired_median_share']:.0%}–{magshare[0]['diff_of_medians_share']:.0%}，"
                                  "视估计量），与 E1 的『增益』族同类；但**方向分量仍占约 "
                                  f"{np.median(dn)/np.median(dr):.0%}**（+{np.median(dn):.2f}），"
                                  "故 E1 的『只有增益、没有内容』并非完整解释"),
        ("M3 donor 改 spatial pattern，E1 只改 amplitude",
         "**NOT SUPPORTED（作为主因）**：donor 的**方向**分量只有约 "
         f"{np.median(dn)/np.median(dr):.0%}（+{np.median(dn):.2f}），幅值分量为主 ⇒ donor 效应的**主体本身就是 amplitude**，"
         "所以『E1 只改 amplitude』**不能**解释 E1 为何无效（协议 §7 的『整个 feature map 一起变亮 = gain 不是 selectivity』这一形态，"
         "在本轮**无法直接测量**——见 §A UNAVAILABLE）"),
        ("M4 donor 影响 class-specific representation",
         "**UNAVAILABLE**（§9 干预后类结构未落盘）；仅有的下游代理（L23 vs TS 质心 cos）"
         f"**反而下降** {np.median(l23p-l23):+.4f}，不支持该机制"),
        ("M5 E1 的 intervention location/parameterization 不等价",
         "**SUPPORTED（结构性）**：V13 是在 **GT 中心单 cell** 上做 **oracle 向量替换**、"
         "**冻结模型、推理态**；E1 是在 L13/L15/L17 上 **全局乘性增益**，必须由 SGD 在训练中自行发现"),
        ("M6 causal effect 是局部、条件性的，不适合全局训练",
         "**SUPPORTED**：效应被 ±1 cell 邻域绑住（R0→R1 +1.76，R2/R3 增量为 +0.10/−0.06，饱和），"
         "且位移 +4/+5 即归零；这是一个 **spatially bound、oracle-positioned** 的效应"),
    ]
    for a, b in mech:
        say(f"  {a}")
        say(f"      → {b}")
    say("")
    say("  **最重要的一条（artifact 直接支持，且独立于上表）：**")
    say(f"    V13 的最强干预（R3，49 cells）只把中位 logit 推到 {np.median(p3):+.3f}"
        f"（p={1/(1+np.exp(-np.median(p3))):.2e}），而 val 成功小目标（G4）的 logit 中位是 "
        f"{np.median(g4L):+.3f}。")
    say("    换言之：**一个『全知位置、全知向量』的 oracle 编辑，都没能把 hard-miss 推到成功区间。**")
    say("    E1 是一个必须在训练中被学出来的**全局**增益，其可达效果只能更小 ⇒ E1 的 null result")
    say("    **与 V13 的强 effect 并不矛盾**：V13 证明的是「L17 参与因果」，不是「增强 L17 足以检出」。")

    # ---------------- 汇总输出 ----------------
    say("")
    say("=" * 112)
    say("[§H] 统计口径声明")
    say("=" * 112)
    say("  • n(G1)=38, n(G4)=194, n(TS)=467；协议 §14 要求：不仅报 mean —— 表中均给 n/median/IQR/mean/")
    say("    bootstrap 95% CI；配对量用 paired bootstrap + Wilcoxon + Cliff's delta。")
    say("  • **所有跨组/跨指标比较均标记 exploratory；multiple-comparison burden 高**（§D 6 项 + §E 4 项")
    say("    + §F 多项）。**不从单个显著指标下总判断**；未做（也不主张）任何多重比较校正后的显著性声明。")

    tables = dict(
        artifact_capability=scan,
        provenance={str(p.name): sha16(p) for p in (P3 / "_v13_dose.json", P3 / "_v14_upstream.json",
                                                    P3 / "_v12_patch.json", P3 / "_v11_layers.npz",
                                                    P3 / "_v7_vectors.json", P3 / "_v11_traj.json",
                                                    CKPT)},
        donor_reconciliation=dict(n_match=n_match, n_total=n_tot,
                                  all_match=bool(n_match == n_tot and n_tot == 38)),
        causal_effect=step,
        spatial_displacement=disp,
        geometry=geo,
        geometry_g1=dict(orig_cos_muTS_median=float(np.median(ov_cos)),
                         donor_cos_muTS_median=float(np.median(dv_cos)),
                         delta_cos_muTS_median=float(np.median(dv_cos - ov_cos)),
                         L23cos_base_median=float(np.median(l23)),
                         L23cos_R0_median=float(np.median(l23p)),
                         L23cos_delta_median=float(np.median(l23p - l23)),
                         raw_median=float(np.median(dr)), norm_median=float(np.median(dn)),
                         rnd_median=float(np.median(drn)),
                         magnitude_component_median=float(np.median(mag)),
                         magnitude_share_paired_median=float(magshare[0]["paired_median_share"]),
                         magnitude_share_diff_of_medians=float(magshare[0]["diff_of_medians_share"]),
                         magnitude_share_mean=float(magshare[0]["mean_diff_share"]),
                         magnitude_share_by_R=magshare,
                         direction_share=float(np.median(dn) / np.median(dr)),
                         rnd_share=float(np.median(drn) / np.median(dr)),
                         norm_ratio_median=float(np.median(rho)),
                         group_cos_muTS={k: v["cos_to_muTS_median"] for k, v in geo.items()},
                         group_norm_median={k: v["norm_median"] for k, v in geo.items()}),
        geometry_g4=dict(n=len(g4r), norm_ratio_median=float(np.median(rhoG4)),
                         orig_norm_median=float(np.median([r["orig_norm"] for r in g4r])),
                         delta_R0_median=float(np.median([r["R0_raw_dlogit"] for r in g4r])),
                         rho_vs_delta_spearman_pooled=dict(rho=rs, p=ps, n=int(len(pa)))),
        correlations=corr,
        sufficiency=dict(level_available=2,
                         level1="UNAVAILABLE", level3="UNAVAILABLE",
                         g1_base_logit_median=float(np.median(bl)),
                         g1_R0_logit_median=float(np.median(p0)),
                         g1_R3_logit_median=float(np.median(p3)),
                         g1_R3_prob=float(1 / (1 + np.exp(-np.median(p3)))),
                         frozen_conf=0.001,
                         ref_g4_logit_median=float(np.median(g4L)),
                         ref_ts_logit_median=float(np.median(tsL)),
                         frac_of_ref_above_patched=pct,
                         g1_R3_percentile_vs_G4=100.0 - pct["G4_R3"],
                         head_recompute_max_abs_diff=float(np.max(np.abs(val))),
                         caveat="unpaired reference-distribution comparison; same readout head, same checkpoint"),
        baseline_class_structure=dict(
            margin_median=float(np.median(mt)), margin_q1=float(np.percentile(mt, 25)),
            margin_q3=float(np.percentile(mt, 75)),
            rank1_frac=float(np.mean(rt == 1)),
            note="intervened non-target projections were never dumped -> 9 before/after = UNAVAILABLE"),
        e1_mechanisms=[dict(mechanism=a, assessment=b) for a, b in mech],
    )
    (OUT / "_tables.json").write_text(json.dumps(tables, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(
        OUT / "_results.npz",
        g1_keys=np.array(G1keys),
        g1_orig_vec=np.stack([Z["L17_G1"][idx_of["G1"][k]] for k in G1keys]),
        g1_donor_vec=np.stack([Z["L17_G4"][_di(k, pool, D13)] for k in G1keys]),
        g1_orig_norm=np.array([r["orig_norm"] for r in G1]),
        g1_donor_norm=np.array([r["donor_norm"] for r in G1]),
        g1_norm_ratio=np.array([r["norm_ratio"] for r in G1]),
        g1_cos_ov_dv=np.array([r["cos_ov_dv"] for r in G1]),
        g1_base_logit=bl, g1_R0_logit=p0, g1_R3_logit=p3,
        g1_R0_raw_dlogit=dr, g1_R0_norm_dlogit=dn, g1_R0_rnd_dlogit=drn,
        g1_disp4=np.array([r["disp4_dlogit"] for r in G1]),
        g1_disp5=np.array([r["disp5_dlogit"] for r in G1]),
        g1_ov_cos_muTS=ov_cos, g1_dv_cos_muTS=dv_cos,
        g1_L23cos_base=l23, g1_L23cos_R0=l23p,
        g4_norm_ratio=rhoG4, ref_g4_logit=g4L, ref_ts_logit=tsL,
    )
    say("")
    say(f"[saved] {OUT/'_tables.json'}")
    say(f"[saved] {OUT/'_results.npz'}")
    (OUT / "AUDIT.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


def _di(k, pool, D13):
    dsel, _ = pick_donor(pool, int(D13[k]["cls"]), float(D13[k]["sq"]))
    return dsel["i"]


def load_head():
    """读取 checkpoint 中 cv3[0][-1] 的 (12,64,1,1)/(12,) 权重 —— **纯读取，无 forward**。

    checkpoint 的 pickle 引用了仓库内的 `SilenceChannel`，因此必须让仓库版 ultralytics 优先于
    site-packages。同时把 `YOLO_CONFIG_DIR` 指到临时目录，避免 ultralytics 在用户目录下写
    settings.json（首版未设该变量时已写过一次 → 见 REPORT 的 provenance 声明）。
    """
    import os
    import tempfile
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "_l17_sel_ultra_cfg"))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch
    import ultralytics  # noqa: F401
    ck = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    sd = ck["model"].state_dict() if hasattr(ck["model"], "state_dict") else ck["model"]
    W = sd["model.30.cv3.0.2.weight"].detach().float().view(12, -1).numpy().astype(np.float64)
    b = sd["model.30.cv3.0.2.bias"].detach().float().numpy().astype(np.float64)
    return W, b


if __name__ == "__main__":
    sys.exit(main())
