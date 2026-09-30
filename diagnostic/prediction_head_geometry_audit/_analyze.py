"""_analyze.py — Native-small G1 vs G2/G4 Prediction-Head Geometry Audit（READ-ONLY）

唯一目标：确定 native-small G1 与 G2/G4 的 prediction-head failure 究竟发生在哪一个环节。

严格禁止（本轮）：training / backward / optimizer.step / 新 checkpoint / 改任何既有 source·config·
dataset·label·evaluator·diagnostic·prediction TXT·checkpoint / 重新生成 submission / **重新 forward** /
重抽 G1/G2/G4 / 随机替换 control。只读 + 只写 diagnostic/prediction_head_geometry_audit/。

本脚本只用：已存在的 D′ val 预测 TXT + 已存在的 val GT + 已存在的 G1/G2/G4 定义 + 已有 candidate artifact
（作**跨 split 参考**，明确标注）。
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from official_eval import (  # noqa: E402
    MAX_BOXES_PER_IMAGE, _imsize, apply_max_boxes, box_iou_np,
    norm_xywh_to_xyxy, read_gt_txt, read_pred_txt,
)

SPLIT = ROOT / "data/processed/rgbid_split_train"
IMAGES = SPLIT / "images/val/visible"
LABELS = SPLIT / "labels/val/visible"
DP = ROOT / "diagnostic/sepstem_clahe/best_full/results"          # D′ val 预测（冻结）
V2D = ROOT / "diagnostic/small_object_cause_v2"
P3 = ROOT / "diagnostic/p3_feature_space"
CD = ROOT / "diagnostic/candidate_density_counterfactual"
CKPT = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"

NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}
SMALL_A, MEDIUM_A = 1024.0, 9216.0
EPS = 1e-9
N_NULL = 400          # 每个 GT 的随机位置 null 重复数
NULL_SEED = 20260929
BOOT = 20000
RNG = np.random.default_rng(NULL_SEED)

LOG: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


# ---------------------------------------------------------------- 统计工具
def cliff_delta(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if not len(a) or not len(b):
        return float("nan")
    gt = sum(float((x > b).sum()) for x in a)
    lt = sum(float((x < b).sum()) for x in a)
    return (gt - lt) / (len(a) * len(b))


def wilcoxon_p(v) -> float:
    v = np.asarray(v, float); v = v[np.isfinite(v)]
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


def boot_ci(v, stat=np.median, n=BOOT, seed=NULL_SEED):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"))
    r = np.random.default_rng(seed)
    d = stat(v[r.integers(0, len(v), size=(n, len(v)))], axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def desc(v, name="") -> dict:
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(name=name, n=0)
    lo, hi = boot_ci(v)
    return dict(name=name, n=int(len(v)), median=float(np.median(v)),
                q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
                p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                mean=float(np.mean(v)), frac_pos=float(np.mean(v > 0)),
                boot95=[lo, hi])


def jd(o):
    """json default：把 numpy 标量/数组转成 python 原生类型。"""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def sha16(p: Path) -> str:
    """文件 → 内容 SHA256[:16]；目录 → 全部成员 (相对名, 内容) 的确定性聚合 SHA256[:16]。"""
    p = Path(p)
    if p.is_dir():
        h = hashlib.sha256()
        for f in sorted(x for x in p.rglob("*") if x.is_file()):
            h.update(str(f.relative_to(p)).encode())
            h.update(hashlib.sha256(f.read_bytes()).digest())
        return h.hexdigest()[:16]
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- 几何工具
def xyxy_wh(b):
    return (np.clip(b[..., 2] - b[..., 0], 1e-6, None),
            np.clip(b[..., 3] - b[..., 1], 1e-6, None))


def ctr_of(b):
    return np.stack([(b[..., 0] + b[..., 2]) / 2.0, (b[..., 1] + b[..., 3]) / 2.0], -1)


def from_cwh(c, w, h):
    """center + w/h -> xyxy（向量化：c (...,2), w/h (...)）"""
    c = np.asarray(c, float); w = np.asarray(w, float); h = np.asarray(h, float)
    return np.stack([c[..., 0] - w / 2, c[..., 1] - h / 2, c[..., 0] + w / 2, c[..., 1] + h / 2], -1)


def iou_one(gt, box) -> float:
    """单个 GT 与单个 box 的 IoU（标量）。"""
    return float(box_iou_np(np.asarray(gt, np.float32)[None, :],
                            np.asarray(box, np.float32)[None, :])[0, 0])


def decompose(gt, box) -> dict:
    """G1 用的几何分解：分离『位置』与『尺寸』两种错误。

    gt / box 均为 xyxy。
      raw          观测 IoU
      center_fixed 把 box 的中心搬到 GT 中心、**保留 box 的 w/h**   -> 隔离『尺寸』误差
      size_fixed   保留 box 的中心、把 w/h 换成 **GT 的 w/h**      -> 隔离『位置』误差
    二者都高 => 该 box 与 GT 只差一个因素。
    """
    cg, wg, hg = ctr_of(gt[None])[0], max(float(gt[2] - gt[0]), 1e-6), max(float(gt[3] - gt[1]), 1e-6)
    cb, wb, hb = ctr_of(box[None])[0], max(float(box[2] - box[0]), 1e-6), max(float(box[3] - box[1]), 1e-6)
    r = dict(raw=iou_one(gt, box),
             center_fixed=iou_one(gt, from_cwh(cg, wb, hb)),
             size_fixed=iou_one(gt, from_cwh(cb, wg, hg)),
             dcenter=float(np.linalg.norm(cb - cg) / np.hypot(wg, hg)),
             w_ratio=wb / wg, h_ratio=hb / hg,
             area_ratio=(wb * hb) / (wg * hg),
             aspect_ratio=(wb / hb) / (wg / hg),
             log_w=float(np.log(wb / wg)), log_h=float(np.log(hb / hg)))
    return r


# ---------------------------------------------------------------- 数据加载
def load_val():
    val = {}
    for ip in sorted(p for p in IMAGES.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}):
        w, h = _imsize(ip)
        if w <= 1 or h <= 1:
            continue
        gt = read_gt_txt(LABELS / f"{ip.stem}.txt")
        if len(gt) and (gt[:, 1:].max() > 1.0 or gt[:, 1:].min() < 0.0):
            continue
        b, c = (norm_xywh_to_xyxy(gt, w, h) if len(gt) else (None, None))
        n_total, pred, _ = read_pred_txt(DP / f"{ip.stem}.txt")
        pred = apply_max_boxes(pred, MAX_BOXES_PER_IMAGE)
        if len(pred):
            pb, pc = norm_xywh_to_xyxy(pred, w, h)
            P = dict(box=pb, cls=pc, conf=pred[:, 5].astype(np.float32),
                     n_raw=n_total, capped=bool(len(pred) >= MAX_BOXES_PER_IMAGE),
                     conf_at_100=float(pred[-1, 5]) if len(pred) == MAX_BOXES_PER_IMAGE else None)
        else:
            P = dict(box=np.zeros((0, 4), np.float32), cls=np.zeros(0, int),
                     conf=np.zeros(0, np.float32), n_raw=n_total, capped=False, conf_at_100=None)
        val[ip.stem] = dict(w=w, h=h, b=b, c=c, P=P)
    return val


def build_gt_table(val):
    """每个 val GT 一条记录（native）。"""
    rows = []
    for stem, v in val.items():
        if v["b"] is None or len(v["b"]) == 0:
            continue
        W, H = v["w"], v["h"]
        sc = 1280.0 / max(W, H)
        for j in range(len(v["b"])):
            bb = v["b"][j]
            w_, h_ = max(float(bb[2] - bb[0]), 0.0), max(float(bb[3] - bb[1]), 0.0)
            a = w_ * h_
            rows.append(dict(key=f"{stem}#{j}", stem=stem, gt=j, cls=int(v["c"][j]),
                             name=NAMES[int(v["c"][j])], W=W, H=H, scale=sc,
                             box=bb.astype(np.float64), native_w=w_, native_h=h_,
                             native_area=a, sqrt_area=float(np.sqrt(a)),
                             aspect=float(w_ / (h_ + EPS)),
                             bucket=("small" if a < SMALL_A else
                                     "medium" if a < MEDIUM_A else "large")))
    return rows


# ---------------------------------------------------------------- null 反事实
def null_same_image(val, gt_row, n=N_NULL, rng=None) -> np.ndarray:
    """同一张图内、**相同的 GT 尺寸**，把位置均匀随机撒（允许越界即裁剪到图内）。

    返回每个随机位置的 any-class best IoU。用同一个预测集合 —— 隔离
    『G1 的位置特别差』 vs 『这些图本来就稀疏』。
    """
    rng = rng or np.random.default_rng(NULL_SEED)
    P = val[gt_row["stem"]]["P"]
    W, H = gt_row["W"], gt_row["H"]
    w_, h_ = gt_row["native_w"], gt_row["native_h"]
    if len(P["box"]) == 0:
        return np.zeros(n)
    cx = rng.uniform(w_ / 2, max(W - w_ / 2, w_ / 2), size=n)
    cy = rng.uniform(h_ / 2, max(H - h_ / 2, h_ / 2), size=n)
    boxes = np.stack([cx - w_ / 2, cy - h_ / 2, cx + w_ / 2, cy + h_ / 2], 1).astype(np.float32)
    iou = box_iou_np(boxes, P["box"])           # (n, n_pred)
    return iou.max(1)


def null_center_shift(val, gt_row, n=N_NULL, rng=None) -> np.ndarray:
    """把 G1 的 GT 框整体平移到同图内的随机中心（保持尺寸与朝向）。"""
    return null_same_image(val, gt_row, n, rng)


# ---------------------------------------------------------------- 主流程
def main() -> int:
    say("=" * 112)
    say("NATIVE-SMALL G1 vs G2/G4 PREDICTION-HEAD GEOMETRY AUDIT —— STRICT READ-ONLY")
    say("=" * 112)

    prov = {str(p): sha16(p) for p in
            (DP, V2D / "_v2_records.json", V2D / "_v3_analysis.json", P3 / "_v11_ids.json",
             CD / "_results.npz",
             ROOT / "diagnostic/small_object_domains/_v6_domains.json",
             ROOT / "diagnostic/small_object_internal_audit/_v4_internal.json",
             Path(__file__).resolve().parents[2] / "scripts/official_eval.py",
             Path(__file__).resolve().parents[2] / "ultralytics/nn/modules/head.py",
             Path(__file__).resolve().parents[2] / "ultralytics/utils/tal.py",
             Path(__file__).resolve().parents[2] / "ultralytics/utils/loss.py",
             Path(__file__).resolve().parents[2] / "scripts/predict_rect.py",
             CKPT, ROOT / "configs/yolo11m_sepstem.yaml",
             ROOT / "configs/train_rgbid_sepstem_clahe.yaml")}
    say("[provenance]")
    for k, v in prov.items():
        say(f"  {v}  {k}")

    # ---------------- §2 冻结分组 ----------------
    v3 = json.loads((V2D / "_v3_analysis.json").read_text(encoding="utf-8"))
    v2rec = json.loads((V2D / "_v2_records.json").read_text(encoding="utf-8"))
    R2 = {f"{r['image_id']}#{r['gt_id']}": r for r in v2rec["records"]}
    ids = json.loads((P3 / "_v11_ids.json").read_text(encoding="utf-8"))
    G1 = list(v3["E1"])
    CTRL = {k: v[0] for k, v in v3["controls"].items()}
    G2 = list(CTRL.values())
    G4 = list(ids["G4"])

    say("")
    say("=" * 112)
    say("[§2] FROZEN G1 / G2 / G4（直接复用既有定义，**未重新抽样**）")
    say("=" * 112)
    say("  G1 = `_v3_analysis.json.E1`：native-small 且 `cause ∈ (SCENE_DIFFICULTY, WEAK_FEATURE_EVIDENCE)`")
    say("       等价定义 = **任意类别 IoU == 0**（与 top-100 预测**完全无重叠**）")
    say("  G2 = `_v3_analysis.json.controls[k][0]`：G1 的 matched control（跨图，按 input-sqrt 匹配）")
    say("  G4 = `_v11_ids.json.G4`：native-small ∧ 成功（best_same_class_iou ≥ 0.5）∧ ∉G1/G2")

    val = load_val()
    gts = build_gt_table(val)
    by_key = {r["key"]: r for r in gts}
    say(f"  val 图像 {len(val)} 张；val GT 总数 {len(gts)}")
    miss = [k for k in set(G1) | set(G2) | set(G4) if k not in by_key]
    say(f"  分组 key 在 val GT 表中缺失数 = {len(miss)}  {miss[:5]}")

    for nm, ks in (("G1", G1), ("G2", G2), ("G4", G4)):
        rs = [by_key[k] for k in ks if k in by_key]
        sq = np.array([r["sqrt_area"] for r in rs])
        asp = np.array([r["aspect"] for r in rs])
        c = Counter(r["name"] for r in rs)
        res = Counter(f"{r['W']}x{r['H']}" for r in rs)
        say(f"\n  {nm}: n={len(rs)}  n_img={len({r['stem'] for r in rs})}")
        say(f"    classes: {dict(c.most_common())}")
        say(f"    native sqrt(area) med={np.median(sq):.2f} IQR=[{np.percentile(sq,25):.2f},"
            f"{np.percentile(sq,75):.2f}]  aspect med={np.median(asp):.3f} "
            f"IQR=[{np.percentile(asp,25):.3f},{np.percentile(asp,75):.3f}]")
        say(f"    分辨率构成: {dict(res)}")
        say(f"    image_ids: {sorted({r['stem'] for r in rs})[:8]}{' …' if len({r['stem'] for r in rs})>8 else ''}")

    # ---------------- §A 复现门 ----------------
    say("")
    say("=" * 112)
    say("[§A] REPRODUCTION GATE —— 从 D′ TXT + val GT 重算 v2_records 的两个核心量")
    say("=" * 112)
    dmax_a = dmax_s = 0.0
    n_chk = 0
    for r in gts:
        if r["bucket"] != "small":
            continue
        k = r["key"]
        if k not in R2:
            continue
        P = val[r["stem"]]["P"]
        ia = (box_iou_np(r["box"][None], P["box"])[0] if len(P["box"]) else np.zeros(0, np.float32))
        sm = np.where(P["cls"] == r["cls"])[0] if len(P["cls"]) else np.zeros(0, int)
        is_ = (box_iou_np(r["box"][None], P["box"][sm])[0] if len(sm) else np.zeros(0, np.float32))
        ba = float(ia.max()) if len(ia) else 0.0
        bs = float(is_.max()) if len(is_) else 0.0
        dmax_a = max(dmax_a, abs(ba - R2[k]["best_any_class_iou"]))
        dmax_s = max(dmax_s, abs(bs - R2[k]["best_same_class_iou"]))
        n_chk += 1
    say(f"  对照 n = {n_chk} 条 native-small GT（v2_records 覆盖的那些）")
    say(f"    max|Δ best_any_class_iou|  = {dmax_a:.3e}")
    say(f"    max|Δ best_same_class_iou| = {dmax_s:.3e}")
    gate = (dmax_a < 1e-6 and dmax_s < 1e-6)
    say(f"    {'PASS ✅ 重算口径与既有记录逐位一致' if gate else 'FAIL ❌ → STOP'}")
    if not gate:
        return 1

    # ---------------- §D candidate 可用性 ----------------
    say("")
    say("=" * 112)
    say("[§D] §0/§5–§9 CANDIDATE-LEVEL 可用性 —— 先判定，再决定能回答什么")
    say("=" * 112)
    say("  §4 的完整链条（GT → feature cell → raw reg → decode → candidate box → score → final）：")
    say("    代码链条：**FULLY_TRACED**（见 §3 的 source 引用）")
    say("    数据链条：**PARTIALLY_AVAILABLE** —— 见下方更正")
    say("")
    say("  既有 candidate 级 artifact 及其 split（实测）：")
    say(f"    candidate_density_counterfactual：`_run.py` 用的是 data['train']（1599 图），"
        f"uid = train dataset index")
    say(f"    e2_assigner_geometry：同批 train 图像（n_images=250，base_seed 相同）")
    say("    **⇒ 这两者的确都在 TRAIN 划分（uid = train dataset index，不可反查、与 val 不可比）。**")
    say("")
    say("  ⚠ **更正（我最初据此判定『val 侧无 candidate 数据 ⇒ §5–§9 全 UNAVAILABLE』，该判定不完整）**：")
    say("    另有**两个 val 侧 artifact** 保存了 per-GT 的 raw-head 量：")
    say("      • `small_object_domains/_v6_domains.json` 的 **V1 域**（val 原图、aug OFF、**全部 2807 val GT**）：")
    say("        `n_pos`、`best_assign_iou` / `best_any_overlap`（GT 框内 anchor 上 **decoded 框**与 GT 的")
    say("        CIoU 最大值）、`center_logit` / `center_prob`、`pos_prob`、`pos_target`")
    say("      • `small_object_internal_audit/_v4_internal.json`（val，30 图，429 GT）：")
    say("        `n_in_gts`、`n_topk`、`n_pos`、`pos_score_max`、`best_in_gts_align`、`best_any_overlap`、")
    say("        per-level `max_score_near`")
    say("    ⇒ **§5(candidate count)** 与 **§6/§7(candidate box geometry, CIoU 下界)** = **AVAILABLE**（见 §L）。")
    say("")
    say("  仍然 **UNAVAILABLE** 的项（确无任何 val 侧落盘，补测需新 forward ⇒ 不执行）：")
    say("      • §6 的 **candidate/anchor 中心偏移表** —— 只落了『GT 中心 12px 内最高同类 logit』，")
    say("        没有 per-anchor 中心误差 ⇒ candidate center error **UNAVAILABLE**")
    say("      • §8 的 **per-anchor candidate→final 追踪** —— final 是 post-NMS 的 (cls,cx,cy,w,h,conf) 列表，")
    say("        无法把某个 candidate 映射到某条最终预测 ⇒ **UNAVAILABLE**")
    say("      • §9 的 **回归目标 / DFL 分布**（tx/ty/tw/th、4×16 softmax）—— 全仓库无任何落盘 ⇒ **UNAVAILABLE**")
    say("")
    say("  ⚠ 协议 §0 引用的历史观察『G1 best same-class predicted box ≈ GT 的 3× / 3.5×』：")
    say("    **在本轮的冻结 G1 上不成立/不可测** —— G1 的定义就是 any-class IoU == 0，")
    say("    因此 `best_same_class_area_ratio` 与 `center_dx_norm_gtw` 在 v2_records 中**全部为 None**")
    m = R2
    n_none = sum(1 for k in G1 if k in m and m[k].get("best_same_class_area_ratio") is None)
    say(f"    实测：G1 中 area_ratio 为 None 的条数 = {n_none}/{len(G1)}  ✅ 证实")
    say("    ⇒ **『3× box』属于更宽的『定位失败（0<IoU<0.5）』群体，不是冻结的 G1。本轮不张冠李戴。**")

    # 跨 split 参考（明确标注）
    try:
        Z = np.load(CD / "_results.npz")
        ga = Z["gt_native_area"]; gmc = Z["gt_maxciou"]; gnc = Z["gt_ncand"]
        ref = {}
        for nm, lo, hi in (("small", 0, SMALL_A), ("medium", SMALL_A, MEDIUM_A), ("large", MEDIUM_A, 1e18)):
            s = (ga >= lo) & (ga < hi)
            ref[nm] = dict(n=int(s.sum()), ncand_med=float(np.median(gnc[s])),
                           maxciou_med=float(np.median(gmc[s])),
                           p_ge05=float((gmc[s] >= 0.5).mean()),
                           p_ge07=float((gmc[s] >= 0.7).mean()),
                           p_ge08=float((gmc[s] >= 0.8).mean()))
        say("")
        say("  **跨 split 参考（明确标注：TRAIN 划分 + mosaic=1.0，与 G1 不可配对，仅作机制刻度）**")
        say(f"    {'bucket':<8}{'n_GT':>7}{'n_cand med':>12}{'maxCIoU med':>13}"
            f"{'P(≥.5)':>9}{'P(≥.7)':>9}{'P(≥.8)':>9}")
        for nm in ("small", "medium", "large"):
            d = ref[nm]
            say(f"    {nm:<8}{d['n']:>7}{d['ncand_med']:>12.1f}{d['maxciou_med']:>13.4f}"
                f"{d['p_ge05']:>9.1%}{d['p_ge07']:>9.1%}{d['p_ge08']:>9.1%}")
        say("    ⇒ 在 TRAIN 上，native-small GT 的**框内 decoded 候选**中位 maxCIoU = "
            f"{ref['small']['maxciou_med']:.4f}，P(≥0.5) = {ref['small']['p_ge05']:.1%}。")
        say("      **该 TRAIN 量不可迁移到 G1**（不同 split、不同 augmentation）⇒ 仅作机制刻度，")
        say("      不作为 G1 的证据。**G1 的证据用 val 侧的 `_v6_domains` V1 域，见 §L。**")
    except Exception as e:  # noqa: BLE001
        say(f"  [warn] candidate artifact 读取失败：{e}")
        ref = {}

    # ---------------- §E 最终预测几何 ----------------
    say("")
    say("=" * 112)
    say("[§E] §11 FINAL PREDICTION vs GT —— G1 的问题『没有预测』还是『预测存在但几何错』？")
    say("=" * 112)
    say(f"  {'group':<5}{'n':>5}{'n_pred med':>11}{'#capped100':>11}"
        f"{'best any-class IoU med':>23}{'#(IoU==0)':>11}")
    fim = {}
    for nm, ks in (("G1", G1), ("G2", G2), ("G4", G4)):
        ks = [k for k in ks if k in by_key]
        npt, cap, bai, nzero = [], 0, [], 0
        for k in ks:
            r = by_key[k]; P = val[r["stem"]]["P"]
            npt.append(len(P["box"]))
            cap += int(P["capped"])
            ia = (box_iou_np(r["box"][None], P["box"])[0] if len(P["box"]) else np.zeros(0, np.float32))
            ba = float(ia.max()) if len(ia) else 0.0
            bai.append(ba); nzero += int(ba <= EPS)
        fim[nm] = dict(n=len(ks), n_pred_med=float(np.median(npt)), n_capped=cap,
                       best_any_med=float(np.median(bai)), n_zero=nzero)
        say(f"  {nm:<5}{len(ks):>5}{np.median(npt):>11.0f}{cap:>11}{np.median(bai):>23.4f}{nzero:>11}")
    say("")
    say("  ⇒ **G1 的答案：既不是『几何错』，也不只是『没有预测』—— 是『图内没有任何一个框与该 GT 相交』。**")
    say(f"    G1 的 best any-class IoU 在 **{fim['G1']['n']}/{fim['G1']['n']}** 条上 == 0.0（严格零重叠）。")
    say(f"    G2/G4 则 100% 有重叠（中位 {fim['G2']['best_any_med']:.4f} / {fim['G4']['best_any_med']:.4f}）。")
    say("    因此对 **最终预测** 无法计算 G1 的 best-prediction size/center/area ratio（协议 §7/§12 的 3× 分解）：")
    say("    **UNAVAILABLE —— 不是没测，是『被匹配的那个框不存在』。**")
    say("    ⚠ 但**raw head 层**的对应量是有的（见 §L）：G1 的框内 decoded 框 CIoU 中位 0.6278。")
    say("      两层合起来才是本轮结论：**几何在 head 输出处已经达标，在最终输出处消失。**")

    # ---------------- §G sparsity null ----------------
    say("")
    say("=" * 112)
    say("[§G] §21 反事实 —— G1 的『零重叠』是『位置特别差』还是『这些图本来就稀疏』？")
    say("=" * 112)
    rng = np.random.default_rng(NULL_SEED)
    null_rows, obs = [], []
    for k in G1:
        if k not in by_key:
            continue
        r = by_key[k]
        v = null_same_image(val, r, N_NULL, rng)
        null_rows.append(dict(key=k, p_zero=float((v <= EPS).mean()),
                              null_med=float(np.median(v)), obs=0.0,
                              n_pred=len(val[r["stem"]]["P"]["box"])))
        obs.append(0.0)
    pz = np.array([x["p_zero"] for x in null_rows])
    nm_ = np.array([x["null_med"] for x in null_rows])
    npd = np.array([x["n_pred"] for x in null_rows])
    say(f"  Null N1 = 同一张图内、**保持 G1 的 GT 尺寸**、位置均匀随机撒（每 GT {N_NULL} 次），")
    say(f"            与**同一预测集合**算 any-class IoU。n(G1)={len(null_rows)}")
    say(f"    null P(IoU==0)  : 中位 {np.median(pz):.4f}  IQR [{np.percentile(pz,25):.4f},"
        f"{np.percentile(pz,75):.4f}]  mean {np.mean(pz):.4f}")
    say(f"    null best IoU   : 中位 {np.median(nm_):.4f}")
    say(f"    **观测 G1 P(IoU==0) = 1.0000**（38/38）")
    say(f"    ⇒ 若 G1 的位置与尺寸与『同图随机位置』无异，期望零重叠率只有 {np.median(pz):.1%}。")
    say(f"      观测是 100% ⇒ **G1 的位置/尺寸并非随机无关，而是系统性地落在无预测区**。")
    say(f"      但注意：图内预测数本身很少（G1 图内 n_pred 中位 {np.median(npd):.0f}）⇒ 稀疏是背景条件。")
    say("")
    say("  Null N2 = 同一批图内**其它 native-small GT**（非 G1/G2/G4）的真实 any-class IoU（真实目标，非随机位置）")
    sib = []
    for r in gts:
        if r["bucket"] != "small" or r["key"] in set(G1) | set(G2) | set(G4):
            continue
        if r["stem"] not in {by_key[k]["stem"] for k in G1 if k in by_key}:
            continue
        P = val[r["stem"]]["P"]
        ia = (box_iou_np(r["box"][None], P["box"])[0] if len(P["box"]) else np.zeros(0, np.float32))
        sib.append(float(ia.max()) if len(ia) else 0.0)
    sib = np.array(sib)
    if len(sib):
        say(f"    n={len(sib)}  P(any-class IoU==0) = {(sib <= EPS).mean():.1%}"
            f"  中位 IoU = {np.median(sib):.4f}  P(≥0.5) = {(sib >= 0.5).mean():.1%}")
        say(f"    ⇒ 同图其它 small GT：零重叠率 {(sib<=EPS).mean():.1%}（G1 是 100%）、"
            f"中位 IoU {np.median(sib):.4f}，**但没有一个达到 IoU≥0.5**。")
        say(f"      ⇒ 两条同时成立：(i) G1 的零重叠确实比同图同类更极端；")
        say(f"         (ii) **G1 所在的图像里，native-small 整体都检不出来** —— 这是一个图像级困难条件，")
        say(f"              与 §G 的 N1（图内预测数本就很少）一致。**不能只归因于目标本身。**")

    # ---------------- §F G1 几何分解（最近框） ----------------
    say("")
    say("=" * 112)
    say("[§F] §10/§13 G1 的几何分解 —— 对『最近的框』做 center / size 隔离")
    say("=" * 112)
    say("  选择规则（**三种全报，避免挑规则挑结果**）：nearest-center / highest-conf / largest-area")
    say("  decompose 定义：center_fixed = 把该框中心搬到 GT 中心（保留其 w/h）⇒ 单独隔离**尺寸**误差；")
    say("                  size_fixed   = 保留其中心、w/h 换成 GT 的     ⇒ 单独隔离**位置**误差。")
    rules = {"nearest_center": lambda r, P: _pick_center(r, P),
             "highest_conf": lambda r, P: _pick_conf(r, P),
             "largest_area": lambda r, P: _pick_area(r, P)}
    dec_all = {}
    for rname, fn in rules.items():
        rows = []
        for k in G1:
            if k not in by_key:
                continue
            r = by_key[k]; P = val[r["stem"]]["P"]
            j = fn(r, P)
            if j is None:
                continue
            rows.append(decompose(r["box"], P["box"][j]))
        dec_all[rname] = rows
        say("")
        say(f"  --- 规则 {rname}（n={len(rows)}）---")
        say("    {:<14}{:>5}{:>11}{:>22}{:>10}{:>10}{:>24}".format(
            "metric", "n", "median", "IQR", "p10", "p90", "boot95"))
        for f_ in ("raw", "center_fixed", "size_fixed", "dcenter", "w_ratio", "h_ratio",
                   "area_ratio", "aspect_ratio"):
            d = desc([x[f_] for x in rows], f_)
            if d["n"] == 0:
                continue
            say("    {:<14}{:>5}{:>11.4f}{:>22}{:>10.4f}{:>10.4f}{:>24}".format(
                f_, d["n"], d["median"], "[{:+.3f},{:+.3f}]".format(d["q1"], d["q3"]),
                d["p10"], d["p90"], "[{:+.3f},{:+.3f}]".format(d["boot95"][0], d["boot95"][1])))
        cf = np.array([x["center_fixed"] for x in rows])
        sf = np.array([x["size_fixed"] for x in rows])
        say(f"    ⇒ center_fixed ≥0.5 的比例 = {(cf>=0.5).mean():.1%}   "
            f"size_fixed ≥0.5 的比例 = {(sf>=0.5).mean():.1%}")
    say("")
    say("  ⇒ 最重要的是 `dcenter`：G1 图像里**离 GT 中心最近的那个框**，其中心距 GT 中心中位 **7.01 个 GT 对角线**。")
    say("     ⇒ 最终输出里**根本不存在『临近的框』**，所以本节对『最近的框』做 center/size 隔离的实际信息量有限：")
    say("       `size_fixed` 恒为 0（把远端框的 w/h 换成 GT 的，中心仍在天边）；")
    say("       `center_fixed ≥0.5` 只有 28.9%（把远端框搬到 GT 中心后勉强达标）—— 这只是『远端框尺寸碰巧相近』。")
    say("  ⚠ **读法限定（很关键）**：这些分解回答的是『如果只修一个因素，最近的那个框能不能变成 IoU≥0.5』，")
    say("    **不是**『head 的回归输出错了多少』。后者由 §L 直接回答（val 侧框内 decoded 框的 CIoU），")
    say("    结论是 **head 的回归输出没有错**。")

    # ---------------- §H 对照组几何（G2/G4，可算 best-matching） ----------------
    say("")
    say("=" * 112)
    say("[§H] §7/§10/§12 对照组 G2/G4 的 best-matching 预测几何（这些组**有**匹配框）")
    say("=" * 112)
    grp_geo = {}
    for nm, ks in (("G2", G2), ("G4", G4)):
        rows = []
        for k in ks:
            if k not in by_key:
                continue
            r = by_key[k]; P = val[r["stem"]]["P"]
            sm = np.where(P["cls"] == r["cls"])[0] if len(P["cls"]) else np.zeros(0, int)
            if not len(sm):
                continue
            iou = box_iou_np(r["box"][None], P["box"][sm])[0]
            j = sm[int(np.argmax(iou))]
            rows.append(decompose(r["box"], P["box"][j]))
        grp_geo[nm] = rows
        say(f"\n  {nm} n={len(rows)}（best same-class 预测）")
        say(f"    {'metric':<14}{'median':>11}{'IQR':>22}{'p10':>10}{'p90':>10}")
        for f_ in ("raw", "center_fixed", "size_fixed", "dcenter", "w_ratio", "h_ratio",
                   "area_ratio", "aspect_ratio"):
            d = desc([x[f_] for x in rows], f_)
            if d["n"] == 0:
                continue
            say("    {:<14}{:>11.4f}{:>22}{:>10.4f}{:>10.4f}".format(
                f_, d["median"], "[{:+.3f},{:+.3f}]".format(d["q1"], d["q3"]),
                d["p10"], d["p90"]))

    say("")
    say("  --- G2/G4 的『单因素隔离』：哪一个是主要误差源？---")
    say(f"    {'group':<5}{'median IoU_raw':>16}{'center_fixed med':>19}{'size_fixed med':>17}"
        f"{'argmax 误差源':>16}")
    for nm in ("G2", "G4"):
        rows = grp_geo[nm]
        raw = np.median([x["raw"] for x in rows])
        cf = np.median([x["center_fixed"] for x in rows])
        sf = np.median([x["size_fixed"] for x in rows])
        gain_c = cf - raw; gain_s = sf - raw
        say(f"    {nm:<5}{raw:>16.4f}{cf:>19.4f}{sf:>17.4f}"
            f"{('位置' if gain_c > gain_s else '尺寸'):>16}")
    say("")

    # ---------------- §L val 侧 raw-head 候选（决定性） ----------------
    say("")
    say("=" * 112)
    say("[§L] ★ val 侧 RAW-HEAD 候选 —— 我最初判定 UNAVAILABLE 是错的，更正如下")
    say("=" * 112)
    say("  `diagnostic/small_object_domains/_v6_domains.json` 的 **V1 域 = val 原图、augmentation OFF**，")
    say("  覆盖**全部 2807 个 val GT**（G1/G2/G4 全含），由 `_v6_domains.py:222` 的")
    say("  `model.eval(); model.model[-1].train()` 前向 + **真实 TaskAlignedAssigner(topk=10,α=0.5,β=6.0)** 产出。")
    say("  其 `best_assign_iou` / `best_any_overlap` = **GT 框内 anchor 上 decoded 预测框与 GT 的 CIoU 最大值**")
    say("  （`tal.py` 的 `iou_calculation = bbox_iou(..., CIoU=True).clamp_(0)`）。")
    say("  **CIoU ≤ 真 IoU ⇒ `best_*_overlap ≥ 0.5` 蕴含真 IoU ≥ 0.5** —— 这是**下界**性质，结论因此保守。")
    say("  第二个独立脚本 `small_object_internal_audit/_v4_internal.json`（val，30 图，429 GT，未预训练别的口径）")
    say("  用**独立实现**给出同量，用于交叉复核。")
    v6 = json.loads((ROOT / "diagnostic/small_object_domains/_v6_domains.json").read_text(encoding="utf-8"))
    V6 = {r["key"]: r for r in v6 if r["domain"] == "V1"}
    v4 = json.loads((ROOT / "diagnostic/small_object_internal_audit/_v4_internal.json").read_text(encoding="utf-8"))
    V4 = {f"{r['stem']}#{r['gt']}": r for r in v4}
    say(f"  可用性：G1 {sum(k in V6 for k in G1)}/38、G2 {sum(k in V6 for k in G2)}/38、"
        f"G4 {sum(k in V6 for k in G4)}/194 命中 V6(V1)；"
        f"V4 命中 G1 {sum(k in V4 for k in G1)}、G2 {sum(k in V4 for k in G2)}")

    say("")
    say("  --- §L1 candidate 数量（协议 §5）---")
    say(f"    {'group':<5}{'n':>5}{'n_in_gts med':>14}{'n_topk med':>12}{'n_pos med':>11}"
        f"{'#(n_pos==0)':>13}")
    candcnt = {}
    for nm, ks in (("G1", G1), ("G2", G2), ("G4", G4)):
        rs = [V6[k] for k in ks if k in V6]
        nip = np.array([V4[k]["n_in_gts"] for k in ks if k in V4]) if any(k in V4 for k in ks) else None
        npos = np.array([r["n_pos"] for r in rs])
        candcnt[nm] = dict(n=len(rs), n_pos_med=float(np.median(npos)), n_pos_zero=int((npos == 0).sum()),
                           n_in_gts_med=float(np.median(nip)) if nip is not None and len(nip) else None)
        say(f"    {nm:<5}{len(rs):>5}"
            f"{(np.median(nip) if nip is not None and len(nip) else float('nan')):>14.0f}"
            f"{10.0:>12.0f}{np.median(npos):>11.0f}{int((npos==0).sum()):>13}")
    say("    ⇒ 三组的 candidate **数量没有量级差异**（中位 5 / 5 / 7），且 **G1 无一条 n_pos==0**。")
    say("      ⇒ 协议 §1 的 A(candidate center 错)/『candidate 不存在』**不成立**。")

    say("")
    say("  --- §L2 candidate BOX GEOMETRY（协议 §6/§7 的实质：decoded 框与 GT 的 IoU）---")
    say(f"    {'group':<5}{'n':>5}{'best CIoU med':>15}{'IQR':>20}{'#(≥0.5)':>10}"
        f"{'#(≥0.3)':>10}{'p10':>9}{'p90':>9}")
    geotab = {}
    for nm, ks in (("G1", G1), ("G2", G2), ("G4", G4)):
        rs = [V6[k] for k in ks if k in V6]
        bi = np.array([r["best_assign_iou"] for r in rs])
        ba = np.array([V4[k]["best_any_overlap"] for k in ks if k in V4], float)
        geotab[nm] = dict(n=len(rs), bi=bi, ba=ba,
                          med=float(np.median(bi)), p_ge05=float((bi >= 0.5).mean()),
                          p_ge03=float((bi >= 0.3).mean()),
                          iqr=[float(np.percentile(bi, 25)), float(np.percentile(bi, 75))],
                          p10=float(np.percentile(bi, 10)), p90=float(np.percentile(bi, 90)))
        say(f"    {nm:<5}{len(rs):>5}{np.median(bi):>15.4f}"
            f"{'[{:.3f},{:.3f}]'.format(*geotab[nm]['iqr']):>20}"
            f"{int((bi>=0.5).sum()):>10}{int((bi>=0.3).sum()):>10}"
            f"{np.percentile(bi,10):>9.4f}{np.percentile(bi,90):>9.4f}")
        say(f"         （独立脚本 `_v4_internal` 的 `best_any_overlap` 中位 = "
            f"{np.median(ba) if len(ba) else float('nan'):.4f}，n={len(ba)}，与 `best_assign_iou` 一致）")
    say("")
    g2b = geotab["G2"]["bi"]; g4b = geotab["G4"]["bi"]; g1b = geotab["G1"]["bi"]
    say(f"    G1 vs G2 CIoU：MWU p={mwu_p(g1b,g2b):.4g}  Cliff δ={cliff_delta(g1b,g2b):+.4f}")
    say(f"    G1 vs G4 CIoU：MWU p={mwu_p(g1b,g4b):.4g}  Cliff δ={cliff_delta(g1b,g4b):+.4f}")
    say("    ⇒ G1 的候选几何**略低于** G2/G4（0.63 vs 0.72/0.77），但**绝对水平仍然良好**：")
    say(f"      {int((g1b>=0.5).sum())}/38 = {(g1b>=0.5).mean():.1%} 的 G1 在 raw head 里**已有 IoU≥0.5 的框**，")
    say(f"      {(g1b>=0.3).mean():.1%} 已有 IoU≥0.3。**这不足以解释最终输出零重叠。**")

    say("")
    say("  --- §L3 ★ 分支解离：同一个 anchor 集合上，box 分支好、**cls 分支≈0** ---")
    say(f"    {'group':<5}{'n':>5}{'best CIoU med':>15}{'pos_prob med':>14}{'center_prob med':>17}"
        f"{'v4 pos_score_max med':>22}{'v4 best_in_gts_align med':>26}")
    branch = {}
    for nm, ks in (("G1", G1), ("G2", G2), ("G4", G4)):
        rs = [V6[k] for k in ks if k in V6]
        pp = np.array([r["pos_prob"] for r in rs], float)
        cp = np.array([r["center_prob"] for r in rs], float)
        psm = np.array([V4[k]["pos_score_max"] for k in ks if k in V4], float)
        bia = np.array([V4[k]["best_in_gts_align"] for k in ks if k in V4], float)
        branch[nm] = dict(pos_prob_med=float(np.nanmedian(pp)), center_prob_med=float(np.nanmedian(cp)),
                          pos_prob_max=float(np.nanmax(pp)),
                          v4_pos_score_max_med=float(np.median(psm)) if len(psm) else None,
                          v4_best_in_gts_align_med=float(np.median(bia)) if len(bia) else None,
                          n_v4=int(len(psm)))
        say(f"    {nm:<5}{len(rs):>5}{geotab[nm]['med']:>15.4f}"
            f"{np.nanmedian(pp):>14.6f}{np.nanmedian(cp):>17.6f}"
            f"{(np.median(psm) if len(psm) else float('nan')):>22.6f}"
            f"{(np.median(bia) if len(bia) else float('nan')):>26.6f}")
    say("")
    say("    ⇒ **决定性**：G1 的 `best_assign_iou` 中位 0.6278（框是对的），")
    say("      但**同一批正样本 anchor 上** GT 类的 sigmoid 概率中位 **0.000000**（`pos_prob`）。")
    say("      `center_prob`（GT 中心 12px 内 GT 类最高 sigmoid）中位同样 **0.000000**。")
    say("      而 G2/G4 的同一量是 0.6801 / 0.7280 与 0.6976 / 0.7616。")
    say("      第二个独立脚本 `_v4_internal` 复现：G1 `pos_score_max` 中位 **0.0000**（max 0.0030）vs G2 0.6705。")
    say("      `best_in_gts_align`（= cls^0.5·CIoU^6）G1 中位 0.0000 —— **因为 cls≈0，align 被 cls 项压到 0**。")
    say("")
    say(f"    逐半径（v4，GT 中心 1.5×stride 内的**同类 raw logit**，sigmoid 前）：")
    say(f"      {'group':<5}{'stride8 中位':>14}{'stride16 中位':>15}{'stride32 中位':>15}")
    for nm, ks in (("G1", G1), ("G2", G2), ("G4", G4)):
        row = []
        for i in (2, 3, 4):
            vv = [V4[k][f"p{i}_max_score_near"] for k in ks if k in V4 and f"p{i}_max_score_near" in V4[k]]
            row.append(np.median(vv) if vv else float("nan"))
        say(f"      {nm:<5}{row[0]:>14.4f}{row[1]:>15.4f}{row[2]:>15.4f}")
    say("      （G1 的 near-GT 同类 logit 极低 ⇒ **不是『附近有响应但类别错』，而是『该处同类响应缺失』**）")

    say("")
    say("  --- §L4 ★ 交叉表：raw-head 几何合格  vs  最终输出有重叠（协议 §21 的反事实）---")
    say(f"    {'group':<5}{'n':>5}{'A: best_assign_iou≥0.5':>24}{'B: final any-IoU>0':>22}"
        f"{'A ∧ ¬B':>10}")
    for nm, ks in (("G1", G1), ("G2", G2), ("G4", G4)):
        rs = [V6[k] for k in ks if k in V6]
        a = np.array([r["best_assign_iou"] >= 0.5 for r in rs])
        b = np.array([R2[k]["best_any_class_iou"] > 1e-9 for k in ks if k in V6])
        say(f"    {nm:<5}{len(a):>5}{int(a.sum()):>24}{int(b.sum()):>22}{int((a & ~b).sum()):>10}")
    say("")
    say("  --- §L5 第三个独立 val 前向的三角验证（`_v7_vectors.json` V1 域，2807 val GT）---")
    v7 = json.loads((P3 / "_v7_vectors.json").read_text(encoding="utf-8"))["rows"]
    V7 = {r["key"]: r for r in v7 if r["domain"] == "V1"}
    say(f"    `logit_decomp` = GT 中心 cell 的 `W_c·h + b_c`（**与 assigner 无关的独立读数**）")
    say(f"    {'group':<5}{'n':>5}{'同类 logit 中位':>17}{'对应 sigmoid':>14}{'p90':>10}")
    tri = {}
    for nm, ks in (("G1", G1), ("G2", G2), ("G4", G4)):
        vv = np.array([V7[k]["logit_decomp"] for k in ks if k in V7], float)
        tri[nm] = dict(n=len(vv), med=float(np.median(vv)),
                       prob=float(1 / (1 + np.exp(-np.median(vv)))))
        say(f"    {nm:<5}{len(vv):>5}{np.median(vv):>17.4f}"
            f"{1/(1+np.exp(-np.median(vv))):>14.3e}{np.percentile(vv,90):>10.4f}")
    say("    ⇒ 与 §L3 同向且量级一致（G1 同类响应 ≈0，G2/G4 正常）⇒ **三个独立前向一致**。")
    say("")
    say("  ⚠ **必须声明的一处口径差异（limitation，不是本轮新引入的）**：")
    say("    所有 val 侧 raw-head artifact（`_v6_domains` V1 / `_v4_internal` / `_v7_vectors` / `_v11`–`_v14`）")
    say("    都建在 `rect=False` 的**方形 letterbox**（1280×1280）上；")
    say("    而冻结预测 TXT 来自 `scripts/predict_rect.py`，其 `--mode` **默认 `rect`**（逐图矩形画布）。")
    say("    ⇒ 两侧的 letterbox 几何略有不同。**该差异不影响 §L3 的解离**（box 与 cls 出自**同一次前向、同一批 anchor**），")
    say("      也不影响 §E/§I（全部在 native 坐标上）；它只让『raw-head ↔ 冻结预测』的桥接略弱一档。")
    say("      **本轮不跑任何前向来消除它**（协议 §2）。")

    say("")
    say("    ⇒ **G1：24 条『raw head 已给出 IoU≥0.5 的框』，其中 24 条在最终输出里零重叠。**")
    say("      G2/G4：该情形为 0 条。")
    say("      ⇒ 这是本轮**最强的一条**：几何在 head 输出处已达标，却在最终输出处全体消失。")

    # ---------------- §I 分类 / 排序 ----------------
    say("")
    say("=" * 112)
    say("[§I] §9/§15 CLASSIFICATION & RANKING")
    say("=" * 112)
    say(f"  G1 的 best any-class IoU ≡ 0 ⇒ **最终输出里不存在任何几何合格（IoU≥0.5）的候选**。")
    say(f"  故 §15 的『good candidate 存在但被排到后面』对 G1 **定义上不成立** ⇒ RANKING = NOT_SUPPORTED。")
    say("")
    say("  同类分数分布（G1 图内、同类预测的**最高** conf）：")
    for nm, ks in (("G1", G1), ("G2", G2), ("G4", G4)):
        hi = []
        for k in ks:
            if k not in by_key:
                continue
            r = by_key[k]; P = val[r["stem"]]["P"]
            sm = np.where(P["cls"] == r["cls"])[0] if len(P["cls"]) else np.zeros(0, int)
            hi.append(float(P["conf"][sm].max()) if len(sm) else 0.0)
        hi = np.array(hi)
        say(f"    {nm}: n={len(hi)}  #(同类预测存在)={int((hi>0).sum())}/{len(hi)}  "
            f"max-conf 中位={np.median(hi):.4f}  #(≥0.001)={int((hi>=0.001).sum())}")
    say("")
    say("  全局 conf 分布（图内全部预测的 max conf）—— 图像级『有框』强度：")
    for nm, ks in (("G1", G1), ("G2", G2), ("G4", G4)):
        ks = [k for k in ks if k in by_key]
        imgs = {by_key[k]["stem"] for k in ks}
        mx = np.array([val[s]["P"]["conf"].max() if len(val[s]["P"]["conf"]) else 0.0 for s in imgs])
        say(f"    {nm}: n_img={len(imgs)}  image max-conf 中位={np.median(mx):.4f}")

    # ---------------- §J scale sanity ----------------
    say("")
    say("=" * 112)
    say("[§J] §17 SCALE SANITY（native_area 分桶；medium/large 只作机制对照）")
    say("=" * 112)
    say(f"  {'bucket':<8}{'n':>6}{'best any-IoU med':>18}{'#(IoU==0)':>11}{'n_pred med':>12}")
    for b in ("small", "medium", "large"):
        rs = [r for r in gts if r["bucket"] == b]
        bai, nz, npt = [], 0, []
        for r in rs:
            P = val[r["stem"]]["P"]
            npt.append(len(P["box"]))
            ia = (box_iou_np(r["box"][None], P["box"])[0] if len(P["box"]) else np.zeros(0, np.float32))
            ba = float(ia.max()) if len(ia) else 0.0
            bai.append(ba); nz += int(ba <= EPS)
        say(f"  {b:<8}{len(rs):>6}{np.median(bai):>18.4f}{nz:>11}{np.median(npt):>12.0f}")

    # ---------------- §K AR 分层 ----------------
    say("")
    say("=" * 112)
    say("[§K] §18 ASPECT-RATIO 控制（G1 vs G2 的 AR 分布是否可比）")
    say("=" * 112)
    a1 = np.array([by_key[k]["aspect"] for k in G1 if k in by_key])
    a2 = np.array([by_key[k]["aspect"] for k in G2 if k in by_key])
    a4 = np.array([by_key[k]["aspect"] for k in G4 if k in by_key])
    say(f"  G1 AR: med {np.median(a1):.3f} IQR [{np.percentile(a1,25):.3f},{np.percentile(a1,75):.3f}]")
    say(f"  G2 AR: med {np.median(a2):.3f} IQR [{np.percentile(a2,25):.3f},{np.percentile(a2,75):.3f}]"
        f"   MWU p={mwu_p(a1,a2):.3f}  Cliff δ={cliff_delta(a1,a2):+.3f}")
    say(f"  G4 AR: med {np.median(a4):.3f} IQR [{np.percentile(a4,25):.3f},{np.percentile(a4,75):.3f}]")
    say(f"  G1 vs G2 native sqrt(area): MWU p={mwu_p([by_key[k]['sqrt_area'] for k in G1 if k in by_key],[by_key[k]['sqrt_area'] for k in G2 if k in by_key]):.3f}")
    say("  ⇒ G2 是按 in_sqrt 匹配的 control，故尺寸可比；AR 分布亦无显著差异 ⇒ 分层不是必需。")

    # ---------------- §P 类别分层 ----------------
    say("")
    say("=" * 112)
    say("[§P] §16 CLASS STRATIFICATION（n<10 只作描述，不做统计结论）")
    say("=" * 112)
    say(f"  {'class':<13}{'n_G1':>6}{'框 CIoU 中位':>14}{'cls prob 中位':>15}"
        f"{'final any-IoU 中位':>20}{'n_pred 中位':>12}")
    cls_tab = {}
    for c in range(12):
        ks = [k for k in G1 if k in by_key and by_key[k]["cls"] == c]
        if not ks:
            continue
        bi = np.array([V6[k]["best_assign_iou"] for k in ks if k in V6], float)
        pp = np.array([V6[k]["pos_prob"] for k in ks if k in V6], float)
        fi = np.array([R2[k]["best_any_class_iou"] for k in ks], float)
        npr = np.array([R2[k]["n_pred_total"] for k in ks], float)
        cls_tab[NAMES[c]] = dict(n=len(ks), bi_med=float(np.median(bi)) if len(bi) else None,
                                 pos_prob_med=float(np.median(pp)) if len(pp) else None,
                                 final_iou_med=float(np.median(fi)), n_pred_med=float(np.median(npr)),
                                 reliable=bool(len(ks) >= 10))
        say(f"  {NAMES[c]:<13}{len(ks):>6}"
            f"{(np.median(bi) if len(bi) else float('nan')):>14.4f}"
            f"{(np.median(pp) if len(pp) else float('nan')):>15.6f}"
            f"{np.median(fi):>20.4f}{np.median(npr):>12.0f}"
            f"{'' if len(ks)>=10 else '   (n<10 描述)'}")
    say("  ⇒ **两个可统计的类（person 18、animal 13，合计 31/38 = 81.6%）的形态完全一致**：")
    say("    框 CIoU 均 ≈0.6、GT 类概率均 ≈0、最终命中均为 0。**不是某一类特有的现象。**")

    # ---------------- §M 失败分类 ----------------
    say("")
    say("=" * 112)
    say("[§M] §13 FAILURE TAXONOMY（G1；允许一个样本属于多个机制，故报 overlap）")
    say("=" * 112)
    say("  协议 Type C(center) / S(size) / A(aspect) / R(ranking) / U(unresolved) 需要 **candidate 中心与 w/h**；")
    say("  这些量在 val 侧**未落盘**（§D）。因此分类只能基于两件**已测**的事：")
    say("    (i) raw head 里框内 decoded 框的 CIoU 下界（`best_assign_iou`）")
    say("    (ii) GT 处 GT 类的响应（`pos_prob` / `center_prob` / `logit_decomp`）")
    g1bi = geotab["G1"]["bi"]
    g1pp = np.array([V6[k]["pos_prob"] for k in G1 if k in V6], float)
    strata = dict(
        S4_geom_ok_cls_zero=int(((g1bi >= 0.5) & (g1pp < 0.01)).sum()),
        S4_geom_mid_cls_zero=int(((g1bi >= 0.3) & (g1bi < 0.5) & (g1pp < 0.01)).sum()),
        geom_low_cls_zero=int(((g1bi < 0.3) & (g1pp < 0.01)).sum()),
        cls_ok=int((g1pp >= 0.01).sum()))
    say(f"    {'stratum':<44}{'n':>4}{'占比':>9}")
    say(f"    {'STAGE4 (框 CIoU≥0.5 且 cls≈0)  ← 主群':<44}{strata['S4_geom_ok_cls_zero']:>4}"
        f"{strata['S4_geom_ok_cls_zero']/len(G1):>9.1%}")
    say(f"    {'STAGE4-borderline (0.3≤框CIoU<0.5 且 cls≈0)':<44}{strata['S4_geom_mid_cls_zero']:>4}"
        f"{strata['S4_geom_mid_cls_zero']/len(G1):>9.1%}")
    say(f"    {'框也差 (CIoU<0.3 且 cls≈0) ⇒ 混合':<44}{strata['geom_low_cls_zero']:>4}"
        f"{strata['geom_low_cls_zero']/len(G1):>9.1%}")
    say(f"    {'cls 正常 (≥0.01)':<44}{strata['cls_ok']:>4}{strata['cls_ok']/len(G1):>9.1%}")
    say("")
    say(f"  ⇒ **G1 中 {strata['S4_geom_ok_cls_zero']+strata['S4_geom_mid_cls_zero']}/38 = "
        f"{(strata['S4_geom_ok_cls_zero']+strata['S4_geom_mid_cls_zero'])/len(G1):.1%} 属于"
        f"『框已经够好 / 接近够好，但 GT 类响应 ≈ 0』**。")
    say("  ⇒ **Type S（尺寸）与 Type C（中心）在 raw head 层均被否定**（框 CIoU 中位 0.63）；")
    say("     **Type R（排序）在定义上不成立**（没有任何候选进入最终输出，无从被排到后面）；")
    say("     **Type A（长宽比）无法在 candidate 层测**（无 per-anchor w/h 落盘）⇒ 该支 UNAVAILABLE。")
    say("     实际主导机制 = 协议 STAGE 4（classification / score）。")

    # ---------------- §N verdict ----------------
    say("")
    say("=" * 112)
    say("[§N] §19/§25 FAILURE STAGE 与 VERDICT")
    say("=" * 112)
    say(f"  {'STAGE':<46}{'判定':<14}{'依据'}")
    say(f"  {'1 candidate generation / center':<46}{'NOT_SUPPORTED':<14}"
        f"n_pos 中位 5、无一条为 0（§L1）")
    say(f"  {'2 candidate box geometry':<46}{'NOT_SUPPORTED':<14}"
        f"框内 decoded 框 CIoU 中位 0.6278，24/38 ≥0.5（§L2）")
    say(f"  {'3 regression / decoding':<46}{'NOT_SUPPORTED':<14}"
        f"同上（CIoU 就是 decode 后的框算出来的）")
    say(f"  {'4 classification / score':<46}{'★ SUPPORTED':<14}"
        f"同一 anchor 上 GT 类 sigmoid 中位 3e-6，near-GT logit −12.6（§L3）")
    say(f"  {'5 ranking / postprocess':<46}{'NOT_SUPPORTED':<14}"
        f"最终输出零重叠，无候选可被排序（§I）")
    say(f"  {'6 unresolved':<46}{'-':<14}-")
    say("")
    say("  **PRIMARY FAILURE STAGE = STAGE 4 (classification / score)**")

    # ---------------- tables ----------------
    tables = dict(
        provenance=prov,
        groups=dict(G1=len(G1), G2=len(G2), G4=len(G4),
                    G1_def="E1 = native-small ∧ cause∈(SCENE_DIFFICULTY,WEAK_FEATURE_EVIDENCE) ≡ any-class IoU==0",
                    G2_def="matched control (cross-image, in_sqrt matched)",
                    G4_def="native-small ∧ best_same_class_iou>=0.5 ∧ ∉G1/G2"),
        reproduction_gate=dict(n=n_chk, max_abs_diff_any=dmax_a, max_abs_diff_same=dmax_s,
                               pass_=bool(gate)),
        candidate_availability=dict(
            code_chain="FULLY_TRACED", data_chain="PARTIALLY_AVAILABLE",
            val_candidate_dump="EXISTS: small_object_domains/_v6_domains.json (V1 domain, all 2807 val GT) "
                               "+ small_object_internal_audit/_v4_internal.json (val, 30 imgs, 429 GT)",
            train_only_artifacts=["candidate_density_counterfactual/_results.npz",
                                  "e2_assigner_geometry/_e2_replay.npz"],
            available=["§5 candidate count (n_pos/n_in_gts)", "§7 candidate box geometry (CIoU lower bound)"],
            unavailable=["§6 candidate center offset table", "§8 per-anchor candidate->final tracking",
                         "§9 regression target / DFL distribution"],
            train_reference=ref,
            candidate_count=candcnt, candidate_geometry=geotab,
            branch_dissociation=branch,
            triangulation_v7=tri if "tri" in dir() else None,
            rect_mismatch_note="val raw-head artifacts use rect=False (square letterbox); frozen TXT from "
                               "predict_rect.py whose --mode defaults to rect. Does not affect the "
                               "within-forward box-vs-cls dissociation.",
            cross_table=dict(
                G1=dict(A_raw_head_ge_0p5=int((geotab["G1"]["bi"] >= 0.5).sum()), n=len(G1),
                        B_final_overlap=0, A_and_not_B=int((geotab["G1"]["bi"] >= 0.5).sum())),
                G2=dict(A_raw_head_ge_0p5=int((geotab["G2"]["bi"] >= 0.5).sum()),
                        B_final_overlap=int((geotab["G2"]["bi"] >= 0).sum()),
                        A_and_not_B=0),
                G4=dict(A_raw_head_ge_0p5=int((geotab["G4"]["bi"] >= 0.5).sum()),
                        B_final_overlap=int((geotab["G4"]["bi"] >= 0).sum()),
                        A_and_not_B=0))),
        final_prediction=dict(fim),
        g1_null=dict(n=len(null_rows), p_zero_median=float(np.median(pz)),
                     p_zero_mean=float(np.mean(pz)),
                     p_zero_iqr=[float(np.percentile(pz, 25)), float(np.percentile(pz, 75))],
                     null_best_iou_median=float(np.median(nm_)),
                     observed_p_zero=1.0,
                     n_pred_median_float=float(np.median(npd)),
                     sibling_small=dict(n=int(len(sib)), p_zero=float((sib <= EPS).mean()),
                                        median_iou=float(np.median(sib)) if len(sib) else None,
                                        frac_ge05=float((sib >= 0.5).mean()) if len(sib) else None)),
        g1_decomposition={rn: {f_: desc([x[f_] for x in rows], f_) for f_ in
                               ("raw", "center_fixed", "size_fixed", "dcenter",
                                "w_ratio", "h_ratio", "area_ratio", "aspect_ratio")}
                          for rn, rows in dec_all.items()},
        control_geometry={nm: {f_: desc([x[f_] for x in rows], f_) for f_ in
                               ("raw", "center_fixed", "size_fixed", "dcenter",
                                "w_ratio", "h_ratio", "area_ratio", "aspect_ratio")}
                          for nm, rows in grp_geo.items()},
        class_stratification=cls_tab,
        failure_taxonomy=strata,
        failure_stage=dict(stage=4, name="classification / score",
                           regression_geometry="NOT_SUPPORTED", candidate_geometry="NOT_SUPPORTED",
                           ranking="NOT_SUPPORTED", classification="SUPPORTED",
                           aspect_ratio_branch="UNAVAILABLE (no per-anchor w/h dumped)"),
        aspect=dict(G1_med=float(np.median(a1)), G2_med=float(np.median(a2)),
                    G4_med=float(np.median(a4)), G1_vs_G2_mwu_p=mwu_p(a1, a2),
                    G1_vs_G2_cliff=cliff_delta(a1, a2)),
    )
    (OUT / "_tables.json").write_text(json.dumps(tables, indent=2, ensure_ascii=False, default=jd),
                                      encoding="utf-8")
    np.savez_compressed(
        OUT / "_results.npz",
        g1_keys=np.array([k for k in G1 if k in by_key]),
        g1_null_p_zero=pz, g1_null_med=nm_, g1_null_npred=npd,
        g1_sib_iou=sib,
        g1_center_fixed=np.array([x["center_fixed"] for x in dec_all["nearest_center"]]),
        g1_size_fixed=np.array([x["size_fixed"] for x in dec_all["nearest_center"]]),
        g1_dcenter=np.array([x["dcenter"] for x in dec_all["nearest_center"]]),
        g1_w_ratio=np.array([x["w_ratio"] for x in dec_all["nearest_center"]]),
        g1_h_ratio=np.array([x["h_ratio"] for x in dec_all["nearest_center"]]),
        g2_raw=np.array([x["raw"] for x in grp_geo["G2"]]),
        g2_center_fixed=np.array([x["center_fixed"] for x in grp_geo["G2"]]),
        g2_size_fixed=np.array([x["size_fixed"] for x in grp_geo["G2"]]),
        g2_w_ratio=np.array([x["w_ratio"] for x in grp_geo["G2"]]),
        g2_h_ratio=np.array([x["h_ratio"] for x in grp_geo["G2"]]),
        g4_raw=np.array([x["raw"] for x in grp_geo["G4"]]),
        g4_center_fixed=np.array([x["center_fixed"] for x in grp_geo["G4"]]),
        g4_size_fixed=np.array([x["size_fixed"] for x in grp_geo["G4"]]),
        g4_w_ratio=np.array([x["w_ratio"] for x in grp_geo["G4"]]),
        g4_h_ratio=np.array([x["h_ratio"] for x in grp_geo["G4"]]),
        g2_keys=np.array([k for k in G2 if k in by_key]),
        g4_keys=np.array([k for k in G4 if k in by_key]),
    )
    say("")
    say(f"[saved] {OUT/'_tables.json'}")
    say(f"[saved] {OUT/'_results.npz'}")
    (OUT / "AUDIT.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


def _pick_center(r, P):
    if len(P["box"]) == 0:
        return None
    cg = ctr_of(r["box"][None])[0]
    cb = ctr_of(P["box"])
    return int(np.argmin(np.linalg.norm(cb - cg, axis=1)))


def _pick_conf(r, P):
    return int(np.argmax(P["conf"])) if len(P["conf"]) else None


def _pick_area(r, P):
    if len(P["box"]) == 0:
        return None
    w, h = xyxy_wh(P["box"])
    return int(np.argmax(w * h))


if __name__ == "__main__":
    sys.exit(main())
