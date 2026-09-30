"""_analyze.py — Depth Reliability Small-Miss Attribution Audit（READ-ONLY）

只回答：native-small hard miss 是否系统性地处于 depth-invalid / depth-low-information 区域，
以及该关联是否能在 class / size / image-level 分层后保持。

严禁：训练 / backward / optimizer.step / 改任何既有源码·config·checkpoint·evaluator·dataset·labels·
augmentation·inference；**不重新 forward**（复用已有预测落盘）。

输入（全部只读）：
  G1 = diagnostic/small_object_cause_v2/_v3_analysis.json["E1"]            （38 个 small hard miss）
  G2 = diagnostic/small_object_cause_v2/_v3_analysis.json["controls"]      （38 个已有 matched control）
  G4 = _v2_records.json 中 native_area<1024 且 best_same_class_iou>=0.5 且非 G1/G2（small 成功）
  预测落盘 = diagnostic/sepstem_clahe/best_full/results  （= v2 records 的同一份 dump，D′）
  depth   = data/processed/rgbid_split_train/images/val/depth/
  labels  = data/processed/rgbid_split_train/labels/val/visible/

§4：depth 统计一律在 **native image coordinate** 上做（GT 归一化坐标 × 原图像素尺寸），
     不做任何 resize（HR 层 visible 与 depth 同为 1920×1080，1:1 对齐）。

§9 depth 编码（实读 base.py:344-347）：uint16 depth 会做 `clip(d/19999*255,0,255).astype(uint8)`，
     故 raw<~40 的非零像素在模型侧会塌成 0。本脚本同时报告 raw 与 encoded 两套口径。
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

AUDIT = Path(__file__).resolve().parent
SPLIT = ROOT / "data/processed/rgbid_split_train"
V_IMG, V_LAB, V_DEP = SPLIT / "images/val/visible", SPLIT / "labels/val/visible", SPLIT / "images/val/depth"
V_IR = SPLIT / "images/val/infrared"
DPRED = ROOT / "diagnostic/sepstem_clahe/best_full/results"
V3 = ROOT / "diagnostic/small_object_cause_v2/_v3_analysis.json"
V2 = ROOT / "diagnostic/small_object_cause_v2/_v2_records.json"

SENTINEL = 19999.0
SMALL_A, MEDIUM_A = 1024.0, 9216.0
NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}


def q(x, p):
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if x.size else float("nan")


def stats(x):
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    if not x.size:
        return dict(n=0)
    return dict(n=int(x.size), mean=float(x.mean()), std=float(x.std(ddof=1)) if x.size > 1 else 0.0,
                p10=q(x, 10), p25=q(x, 25), p50=q(x, 50), p75=q(x, 75), p90=q(x, 90))


def cliff_delta(a, b):
    """Cliff's delta（-1..1）。用秩统计避免 O(n*m) 双重循环。"""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    allv = np.concatenate([a, b]); r = allv.argsort().argsort().astype(np.float64) + 1
    n1, n2 = a.size, b.size
    r1 = r[:n1].sum()
    u = r1 - n1 * (n1 + 1) / 2.0                      # U for a
    return float(2.0 * u / (n1 * n2) - 1.0)


def mannwhitney_p(a, b):
    """双侧 Mann-Whitney U 的正态近似 p（仅作参考，不作唯一证据）。"""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    n1, n2 = a.size, b.size
    if n1 < 3 or n2 < 3:
        return float("nan")
    allv = np.concatenate([a, b]); r = allv.argsort().argsort().astype(np.float64) + 1
    u = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sd == 0:
        return float("nan")
    z = abs(u - mu) / sd
    return float(1.0 - math.erf(z / math.sqrt(2)))


def wilcoxon_p(d):
    """配对差值的符号秩检验（正态近似）。"""
    d = np.asarray(d, np.float64); d = d[np.isfinite(d) & (d != 0)]
    n = d.size
    if n < 6:
        return float("nan")
    r = np.abs(d).argsort().argsort().astype(np.float64) + 1
    wp, wn = r[d > 0].sum(), r[d < 0].sum()
    mu = n * (n + 1) / 4.0
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = abs(min(wp, wn) - mu) / sd
    # 双侧 p = 1 - erf(z/sqrt2)   （v1 写成 2*Φ(z) 会给出 >1 的非法值）
    return float(1.0 - math.erf(z / math.sqrt(2)))


def box_of(cx, cy, w, h, W, H, scale=1.0):
    """归一化 cxcywh → 以中心放大 scale 倍的 native 整数像素框（clipped，至少 1px）。"""
    hw, hh = w * W * scale / 2.0, h * H * scale / 2.0
    ccx, ccy = cx * W, cy * H
    x0, x1 = int(np.floor(ccx - hw)), int(np.ceil(ccx + hw))
    y0, y1 = int(np.floor(ccy - hh)), int(np.ceil(ccy + hh))
    x0, x1 = max(0, min(x0, W - 1)), max(1, min(x1, W))
    y0, y1 = max(0, min(y0, H - 1)), max(1, min(y1, H))
    return x0, y0, x1, y1


def patch_metrics(depth, grad, rgb, ir, cx, cy, w, h, W, H):
    out = {}
    regions = {}
    for nm, sc in (("r1", 1.0), ("r2", 1.5), ("r3", 2.0)):
        x0, y0, x1, y1 = box_of(cx, cy, w, h, W, H, sc)
        regions[nm] = (x0, y0, x1, y1)
        p = depth[y0:y1, x0:x1]
        out[f"{nm}_area"] = int(p.size)
        out[f"{nm}_zero_frac"] = float((p == 0).mean()) if p.size else float("nan")
    a1 = regions["r1"]; a3 = regions["r3"]
    p1 = depth[a1[1]:a1[3], a1[0]:a1[2]]
    p3 = depth[a3[1]:a3[3], a3[0]:a3[2]]
    z1, z3 = int((p1 == 0).sum()), int((p3 == 0).sum())
    ar1, ar3 = p1.size, p3.size
    out["ring_area"] = max(ar3 - ar1, 1)
    out["ring_zero_frac"] = (z3 - z1) / max(ar3 - ar1, 1)
    out["inner_minus_outer"] = out["r1_zero_frac"] - out["ring_zero_frac"]
    out["image_zero_frac"] = float((depth == 0).mean())
    out["relative_depth_deficit"] = out["r1_zero_frac"] - out["image_zero_frac"]

    p = p1.ravel().astype(np.float64)
    nz = p[p > 0]
    out["valid_ratio"] = float(nz.size / max(p.size, 1))
    if nz.size:
        out.update(raw_min=float(nz.min()), raw_max=float(nz.max()), raw_median=float(np.median(nz)),
                   raw_mean=float(nz.mean()), raw_std=float(nz.std(ddof=1)) if nz.size > 1 else 0.0,
                   raw_iqr=float(np.percentile(nz, 75) - np.percentile(nz, 25)),
                   raw_cv=float(nz.std() / nz.mean()) if nz.mean() > 0 else float("nan"))
        enc = np.clip(nz / SENTINEL * 255.0, 0, 255).astype(np.uint8)
        out["encoded_zero_of_raw_nonzero"] = float((enc == 0).mean())
    else:
        out.update(raw_min=np.nan, raw_max=np.nan, raw_median=np.nan, raw_mean=np.nan,
                   raw_std=np.nan, raw_iqr=np.nan, raw_cv=np.nan, encoded_zero_of_raw_nonzero=np.nan)
    g = grad[a1[1]:a1[3], a1[0]:a1[2]].astype(np.float64)
    gv = g[g > 0]
    out["grad_median"] = float(np.median(gv)) if gv.size else 0.0
    out["grad_mean"] = float(gv.mean()) if gv.size else 0.0
    for nm, arr in (("rgb", rgb), ("ir", ir)):
        if arr is None:
            out[f"{nm}_contrast"] = np.nan; out[f"{nm}_mean"] = np.nan; continue
        pp = arr[a1[1]:a1[3], a1[0]:a1[2]].astype(np.float64)
        out[f"{nm}_contrast"] = float(pp.std())
        out[f"{nm}_mean"] = float(pp.mean())
    return out


def main():
    print("=" * 108)
    print("DEPTH RELIABILITY SMALL-MISS ATTRIBUTION AUDIT（READ-ONLY，无 forward）")
    print("=" * 108)

    v3 = json.loads(V3.read_text(encoding="utf-8"))
    G1 = set(v3["E1"])
    G2 = {v[0] for v in v3["controls"].values()}
    v2 = {f"{r['image_id']}#{r['gt_id']}": r for r in json.loads(V2.read_text(encoding="utf-8"))["records"]}
    print(f"\n[G1/G2/G4 恢复]  G1={len(G1)}  G2={len(G2)}  v2 records={len(v2)}")

    from official_eval import (read_gt_txt, read_pred_txt, apply_max_boxes,
                              norm_xywh_to_xyxy, box_iou_np)  # noqa: E402
    from ultralytics.utils.patches import imread  # noqa: E402

    stems = sorted(p.stem for p in V_IMG.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    print(f"  val 图像 {len(stems)} 张")

    recs = {}
    n_done = 0
    for stem in stems:
        dp = V_DEP / f"{stem}.png"
        if not dp.exists():
            dp = V_DEP / f"{stem}.jpg"
        if not dp.exists():
            continue
        depth = imread(str(dp), -1)
        if depth is None:
            continue
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        H, W = depth.shape[:2]
        gt = read_gt_txt(V_LAB / f"{stem}.txt")
        if not len(gt):
            continue
        # depth 与 visible 必须同尺寸（否则 STOP B）
        vis_path = V_IMG / f"{stem}.png"
        if not vis_path.exists():
            vis_path = V_IMG / f"{stem}.jpg"
        vis = imread(str(vis_path), -1)
        if vis is None or vis.shape[:2] != (H, W):
            print(f"  [SKIP] {stem}: visible {None if vis is None else vis.shape[:2]} vs depth {(H, W)}")
            continue
        if vis.ndim == 2:
            vis = np.stack([vis] * 3, -1)
        irp = V_IR / f"{stem}.png"
        if not irp.exists():
            irp = V_IR / f"{stem}.jpg"
        ir = imread(str(irp), -1) if irp.exists() else None
        if ir is not None:
            if ir.ndim == 3:
                ir = ir[:, :, 0]
            if ir.shape[:2] != (H, W):
                ir = None
        else:
            ir = None
        pb = read_pred_txt(DPRED / f"{stem}.txt")[1]
        pb = apply_max_boxes(pb, 300)
        gb, gc = norm_xywh_to_xyxy(gt, W, H)
        pbb, pcc = (norm_xywh_to_xyxy(pb, W, H) if len(pb) else (np.zeros((0, 4)), np.zeros(0, int)))
        # |dx|+|dy|（对全图算一次）
        d32 = depth.astype(np.float32)
        grad = np.zeros_like(d32)
        grad[:, :-1] += np.abs(np.diff(d32, axis=1))
        grad[:-1, :] += np.abs(np.diff(d32, axis=0))
        for j in range(len(gt)):
            c = int(gt[j, 0]); cx, cy, w, h = (float(x) for x in gt[j, 1:5])
            same = np.flatnonzero(pcc == c) if len(pcc) else np.zeros(0, int)
            iou_s = float(box_iou_np(gb[j:j + 1], pbb[same])[0].max()) if len(same) else 0.0
            m = patch_metrics(d32, grad, vis, ir, cx, cy, w, h, W, H)
            m.update(stem=stem, gt_id=j, cls=c, name=NAMES.get(c, str(c)),
                     native_area=float(w * W * h * H), sqrt_area=float(np.sqrt(max(w * W * h * H, 0))),
                     aspect=float((w * W) / max(h * H, 1e-9)), iou=iou_s, img_w=W, img_h=H)
            recs[f"{stem}#{j}"] = m
        n_done += 1
        if n_done % 50 == 0:
            print(f"    ...{n_done} 图")
    print(f"  完成 {n_done} 图；GT 记录 {len(recs)}")

    # 校验：G1/G2 的 native_area 必须与 v2 记录一致
    bad = 0
    for k in list(G1) + list(G2):
        if k in recs and k in v2:
            if abs(recs[k]["native_area"] - v2[k]["native_area"]) > 1.0:
                bad += 1
    print(f"  [校验] G1/G2 native_area 与 v2 记录不一致数: {bad}  (0 = 坐标对应可靠，STOP B 通过)")

    # ---- 分组 ----
    def ok_success(k):
        return k in v2 and v2[k]["best_same_class_iou"] >= 0.5
    groups = defaultdict(list)
    for k, m in recs.items():
        a = m["native_area"]
        if k in G1:
            groups["small_hard_miss"].append(k)
        elif k in G2:
            groups["G2_matched_control"].append(k)
        elif a < SMALL_A and ok_success(k):
            groups["small_success"].append(k)
        elif SMALL_A <= a < MEDIUM_A and ok_success(k):
            groups["medium_success"].append(k)
        elif a >= MEDIUM_A and ok_success(k):
            groups["large_success"].append(k)
        elif a < SMALL_A:
            groups["small_other_miss"].append(k)
    for g, ks in sorted(groups.items()):
        print(f"    {g:<22} n={len(ks)}")

    # ---- §17 attribution matrix ----
    print("\n" + "=" * 108)
    print("§17 ATTRIBUTION MATRIX（depth 指标，native 坐标）")
    print("=" * 108)
    cols = [("r1_zero_frac", "GT zero"), ("r2_zero_frac", "1.5x zero"), ("r3_zero_frac", "2x zero"),
            ("ring_zero_frac", "ring zero"), ("valid_ratio", "valid ratio"),
            ("raw_std", "depth std"), ("raw_iqr", "depth IQR"), ("grad_median", "grad med"),
            ("relative_depth_deficit", "rel deficit"), ("encoded_zero_of_raw_nonzero", "enc-collapse")]
    order = ["small_hard_miss", "G2_matched_control", "small_success", "medium_success", "large_success"]
    hdr = f"  {'group':<22} {'n':>4} " + " ".join(f"{c[1]:>11}" for c in cols)
    print(hdr)
    for g in order:
        ks = [k for k in groups.get(g, []) if k in recs]
        if not ks:
            continue
        line = f"  {g:<22} {len(ks):>4} "
        for key, _ in cols:
            v = [recs[k][key] for k in ks]
            line += f"{q(v,50):>11.4f} "
        print(line)
    print("  （表内为 median）")

    # ---- §11/§12 配对比较：G1 vs G2（已有 matched control）----
    print("\n" + "=" * 108)
    print("§11/§12 PAIRED — G1（hard miss） vs G2（已有 matched control），逐对")
    print("=" * 108)
    pairs = [(k, v3["controls"][k][0]) for k in v3["E1"] if k in recs and v3["controls"][k][0] in recs]
    print(f"  可用配对 {len(pairs)} / {len(v3['E1'])}；匹配质量：class 一致 "
          f"{sum(1 for a,b in pairs if recs[a]['cls']==recs[b]['cls'])}/{len(pairs)}；"
          f"sqrt(area) 差中位 {q([abs(recs[a]['sqrt_area']-recs[b]['sqrt_area']) for a,b in pairs],50):.2f}px；"
          f"aspect 差中位 {q([abs(recs[a]['aspect']-recs[b]['aspect']) for a,b in pairs],50):.3f}")
    print(f"\n  {'metric':<30} {'G1 med':>9} {'G2 med':>9} {'配对Δ med':>11} {'Wilcoxon p':>11} "
          f"{'Cliff d':>9} {'MWU p':>9}")
    eff = {}
    for key, nm in cols:
        a = [recs[x][key] for x, _ in pairs]; b = [recs[y][key] for _, y in pairs]
        d = [recs[x][key] - recs[y][key] for x, y in pairs]
        eff[key] = dict(g1_med=q(a, 50), g2_med=q(b, 50), paired_delta_med=q(d, 50),
                        wilcoxon_p=wilcoxon_p(d), cliff=cliff_delta(a, b), mwu_p=mannwhitney_p(a, b))
        print(f"  {nm:<30} {q(a,50):>9.4f} {q(b,50):>9.4f} {q(d,50):>11.4f} "
              f"{wilcoxon_p(d):>11.3g} {cliff_delta(a,b):>9.3f} {mannwhitney_p(a,b):>9.3g}")

    # ---- §14 class 分层 ----
    print("\n" + "=" * 108)
    print("§14 CLASS 分层（G1 vs G2 的 r1_zero_frac 配对差）")
    print("=" * 108)
    print(f"  {'class':<12} {'n_pairs':>8} {'G1 med':>9} {'G2 med':>9} {'Δ med':>9}")
    per_cls = defaultdict(list)
    for x, y in pairs:
        per_cls[recs[x]["name"]].append((x, y))
    cls_tab = {}
    for nm, ps in sorted(per_cls.items(), key=lambda t: -len(t[1])):
        a = [recs[x]["r1_zero_frac"] for x, _ in ps]; b = [recs[y]["r1_zero_frac"] for _, y in ps]
        d = [recs[x]["r1_zero_frac"] - recs[y]["r1_zero_frac"] for x, y in ps]
        cls_tab[nm] = dict(n=len(ps), g1=q(a, 50), g2=q(b, 50), d=q(d, 50))
        flag = "" if len(ps) >= 8 else "  (n<8，仅描述统计)"
        print(f"  {nm:<12} {len(ps):>8} {q(a,50):>9.4f} {q(b,50):>9.4f} {q(d,50):>9.4f}{flag}")

    # ---- §15 size 分层 ----
    print("\n" + "=" * 108)
    print("§15 SIZE 分层（G1 内部按 native sqrt 四分位；与全体 small 成功组对照）")
    print("=" * 108)
    ks1 = [k for k in groups["small_hard_miss"] if k in recs]
    ss = [k for k in groups["small_success"] if k in recs]
    sq1 = np.array([recs[k]["sqrt_area"] for k in ks1])
    qs = np.percentile(sq1, [25, 50, 75])
    print(f"  {'stratum':<16} {'n_G1':>5} {'G1 zero med':>12} {'n_succ':>7} {'succ zero med':>14} {'Δ':>9}")
    size_tab = []
    for nm, msk in (("S1 (<=p25)", sq1 <= qs[0]), ("S2 (25-50%)", (sq1 > qs[0]) & (sq1 <= qs[1])),
                    ("S3 (50-75%)", (sq1 > qs[1]) & (sq1 <= qs[2])), ("S4 (>p75)", sq1 > qs[2])):
        g1s = [k for k, m in zip(ks1, msk) if m]
        lo, hi = sq1[msk].min() if msk.any() else 0, sq1[msk].max() if msk.any() else 0
        ssc = [k for k in ss if lo <= recs[k]["sqrt_area"] <= hi]
        a = [recs[k]["r1_zero_frac"] for k in g1s]; b = [recs[k]["r1_zero_frac"] for k in ssc]
        size_tab.append(dict(stratum=nm, n_g1=len(g1s), g1=q(a, 50), n_succ=len(ssc), succ=q(b, 50),
                             delta=q(a, 50) - q(b, 50) if a and b else float("nan")))
        print(f"  {nm:<16} {len(g1s):>5} {q(a,50):>12.4f} {len(ssc):>7} {q(b,50):>14.4f} "
              f"{(q(a,50)-q(b,50)):>9.4f}")

    # ---- §13 RGB/IR 分层 ----
    print("\n" + "=" * 108)
    print("§13 RGB / IR EVIDENCE 分层（图像域对比度，非模型特征；无 forward）")
    print("=" * 108)
    print(f"  {'分层':<26} {'n_G1':>5} {'n_succ':>7} {'G1 zero':>9} {'succ zero':>10} {'Δ':>9}")
    ir_tab = {}
    for tag, key in (("RGB contrast", "rgb_contrast"), ("IR contrast", "ir_contrast")):
        vals = [recs[k][key] for k in ks1 if np.isfinite(recs[k][key])]
        if not vals:
            print(f"  {tag}: 不可用"); continue
        med = np.median(vals)
        for side, nm in ((True, f"{tag} 低(<中位)"), (False, f"{tag} 高(>=中位)")):
            g1s = [k for k in ks1 if np.isfinite(recs[k][key]) and ((recs[k][key] < med) == side)]
            ssc = [k for k in ss if np.isfinite(recs[k][key]) and ((recs[k][key] < med) == side)]
            a = [recs[k]["r1_zero_frac"] for k in g1s]; b = [recs[k]["r1_zero_frac"] for k in ssc]
            ir_tab[nm] = dict(n_g1=len(g1s), n_succ=len(ssc), g1=q(a, 50), succ=q(b, 50),
                              delta=q(a, 50) - q(b, 50) if a and b else float("nan"))
            print(f"  {nm:<26} {len(g1s):>5} {len(ssc):>7} {q(a,50):>9.4f} {q(b,50):>10.4f} "
                  f"{(q(a,50)-q(b,50)):>9.4f}")

    # ---- §19 replication ----
    allsm = ks1 + ss
    z = [recs[k]["r1_zero_frac"] for k in allsm]
    print("\n" + "=" * 108)
    print("§19 REPLICATION（当前原始 depth / exact GT UID 重算；不引用旧数字）")
    print("=" * 108)
    print(f"  native-small GT 数 = {len(allsm)}；GT box depth zero_frac: 中位 {q(z,50):.4f} "
          f"p25 {q(z,25):.4f} p75 {q(z,75):.4f} p90 {q(z,90):.4f}")
    print(f"  完全 zero 的 small patch 比例 = {100*np.mean([x >= 1.0 for x in z]):.1f}%")
    print(f"  全图 depth zero_frac 中位 = {q([recs[k]['image_zero_frac'] for k in allsm],50):.4f}")

    json.dump(dict(effect_size=eff, class_strata=cls_tab, size_strata=size_tab, rgb_ir_strata=ir_tab,
                   groups={g: len(v) for g, v in groups.items()},
                   replication=dict(n=len(allsm), zero_median=q(z, 50), zero_p90=q(z, 90),
                                    fully_zero_frac=float(np.mean([x >= 1.0 for x in z])),
                                    image_zero_median=q([recs[k]["image_zero_frac"] for k in allsm], 50))),
              open(AUDIT / "_tables.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    np.savez_compressed(AUDIT / "_results.npz",
                        **{f"{k}__{c}": np.array([recs[k][c] for k in recs], dtype=np.float64)
                           for c in cols_key_all()})
    print(f"\n[saved] {AUDIT/'_tables.json'}  {AUDIT/'_results.npz'}")
    return 0


def cols_key_all():
    return ["r1_zero_frac", "r2_zero_frac", "r3_zero_frac", "ring_zero_frac", "valid_ratio",
            "image_zero_frac", "relative_depth_deficit", "raw_std", "raw_iqr", "raw_median",
            "grad_median", "encoded_zero_of_raw_nonzero", "rgb_contrast", "ir_contrast",
            "native_area", "sqrt_area", "aspect", "iou"]


if __name__ == "__main__":
    sys.exit(main())
