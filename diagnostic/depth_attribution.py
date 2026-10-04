"""depth_attribution.py — Phase F: Zero-GPU Depth Attribution Audit。

ZERO GPU / 不重新推理 / 不改数据。只读 D′/M4 frozen predictions + GT + **原始 depth 图**。

⚠ 分层（本审计的硬约束，见报告 §0）：
    HR  n=373  uint16 1ch 1920x1080  —— invalid 约定 = `<300mm`（base.py 既有，代码已核）⇒ valid = d>=300
    LR  n= 27  uint8  3ch 640x360    —— invalid 约定 **UNKNOWN**（base.py 对 uint8 不做 <300 处理）
                                        ⇒ **不解释 zero 为 invalid**，只报 nonzero_fraction（描述性）

用法: python -X utf8 diagnostic/depth_attribution.py --out diagnostic/batch2_eval/_depth_attrib.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "diagnostic"))
from official_eval import load_split  # noqa: E402
from gt_level_attribution import collect, size_of, SIZE_BINS, NAMES  # noqa: E402

IMAGES = ROOT / "data/processed/rgbid_split_train/images/val/visible"
LABELS = ROOT / "data/processed/rgbid_split_train/labels/val/visible"
DEPTH = ROOT / "data/processed/rgbid_split_train/images/val/depth"
DIRS = {"Dp": ROOT / "diagnostic/sepstem_clahe/best_full/results",
        "M4": ROOT / "diagnostic/batch2_eval/m4_best/results",
        "M1": ROOT / "diagnostic/batch1_eval/m1_best/results"}
INVALID_MM = 300.0  # base.py: im[im < 300] = 0.0


def stat(v, qs):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        d = {f'p{int(q)}': float('nan') for q in qs}
        d.update({'n': 0, 'med': float('nan'), 'mean': float('nan'), 'std': float('nan'), 'iqr': float('nan')})
        return d
    d = {f'p{int(q)}': float(np.percentile(v, q)) for q in qs}
    d['med'] = float(np.median(v)); d['mean'] = float(v.mean())
    d['std'] = float(v.std()); d['n'] = int(v.size)
    d['iqr'] = float(np.percentile(v, 75) - np.percentile(v, 25))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="diagnostic/batch2_eval/_depth_attrib.json")
    A = ap.parse_args()

    per_image, gstats = load_split(IMAGES, LABELS)
    S = {k: collect(per_image, d, 0.50) for k, d in DIRS.items()}
    keys = sorted(S["Dp"].keys())
    area, bbox_of = {}, {}
    for (i, s, w, h, gb, gc) in per_image:
        if gb is None:
            continue
        for c in np.unique(gc):
            for j, ii in enumerate(np.where(gc == c)[0]):
                b = gb[ii]
                area[(i, int(c), int(j))] = float((b[2]-b[0])*(b[3]-b[1]))
                bbox_of[(i, int(c), int(j))] = (b, w, h)

    # ---------- 逐图 depth statistics ----------
    byname = {p.stem: p for p in DEPTH.iterdir()}
    stem_by_id = {i: s for (i, s, *_x) in per_image}
    _cache = {}
    img_depth = {}
    chem_null = 0
    for (i, s, w, h, gb, gc) in per_image:
        p = byname.get(s)
        if p is None:
            img_depth[i] = None; chem_null += 1; continue
        d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if d is None:
            img_depth[i] = None; chem_null += 1; continue
        if d.ndim == 3:
            d = d[..., 0]
        hr = (d.dtype == np.uint16)
        rec = dict(stratum="HR" if hr else "LR", dtype=str(d.dtype), h=d.shape[0], w=d.shape[1])
        if hr:
            valid = d >= INVALID_MM
            rec["valid_fraction"] = float(valid.mean())
            rec["zero_fraction"] = float((d == 0).mean())
            vv = d[valid].astype(np.float32)
            rec["valid_convention"] = "d>=300 (base.py)"
            f = d.astype(np.float32)
            gx = np.abs(np.diff(f, axis=1)); gy = np.abs(np.diff(f, axis=0))
            rec["grad_mean"] = float(gx.mean()); rec["grad_p95"] = float(np.percentile(gx, 95))
            rec["grad_mag_mean"] = float((gx.mean() + gy.mean()) / 2)
        else:
            # ⚠ invalid 约定 UNKNOWN ⇒ 不把 zero 当 invalid；只报描述量
            vv = d[d > 0].astype(np.float32)
            rec["valid_fraction"] = float("nan")
            rec["nonzero_fraction"] = float((d > 0).mean())
            rec["zero_fraction"] = float((d == 0).mean())
            rec["valid_convention"] = "UNKNOWN (uint8; base.py 不做 <300 处理)"
            f = d.astype(np.float32)
            gx = np.abs(np.diff(f, axis=1))
            rec["grad_mean"] = float(gx.mean()); rec["grad_p95"] = float(np.percentile(gx, 95))
            rec["grad_mag_mean"] = float(gx.mean())
        rec.update({k: v for k, v in stat(vv, (1, 5, 25, 50, 75, 95, 99)).items() if k != 'n'})
        img_depth[i] = rec
    print(f"[depth] 每图统计完成，缺图/失败 = {chem_null}")
    hr_n = sum(1 for v in img_depth.values() if v and v['stratum'] == 'HR')
    print(f"[depth] HR={hr_n}  LR={len(per_image)-hr_n-chem_null}")

    # ---------- 逐 GT depth statistics ----------
    rows = []
    for k in keys:
        iid, c, j = k
        d, m4, m1 = S["Dp"][k], S["M4"][k], S["M1"][k]
        b, w, h = bbox_of[k]
        ph = f"{int(d['matched'])}{int(m4['matched'])}{int(m1['matched'])}"
        rec = dict(key=k, cls=c, area=area[k], size=size_of(area[k]), ph=ph,
                   dp=d['matched'], m4=m4['matched'], m1=m1['matched'],
                   dp_iou=d['matched_iou'], dp_conf=d['matched_conf'], m4_iou=m4['matched_iou'])
        di = img_depth.get(iid)
        if di is None:
            rows.append(rec); continue
        rec["stratum"] = di["stratum"]
        # GT-local
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, max(x2, x1 + 1)), min(h, max(y2, y1 + 1))
        if iid in _cache:
            dd = _cache[iid]
        else:
            _pp = byname.get(stem_by_id[iid])
            dd = cv2.imread(str(_pp), cv2.IMREAD_UNCHANGED) if _pp else None
            if dd is not None and dd.ndim == 3:
                dd = dd[..., 0]
            _cache[iid] = dd
        if dd is None:
            rows.append(rec); continue
        # 尺寸对齐（depth 与 visible 同尺寸，已验证；仍做兜底缩放到 (w,h)）
        if dd.shape[1] != w or dd.shape[0] != h:
            dd = cv2.resize(dd, (w, h), interpolation=cv2.INTER_NEAREST)
        patch = dd[y1:y2, x1:x2].astype(np.float32)
        if dd.dtype == np.uint16:
            vp = patch[patch >= INVALID_MM]
            rec["gt_valid_frac"] = float((patch >= INVALID_MM).mean())
            rec["depth_scale"] = "raw_uint16_mm"
        else:
            vp = patch[patch > 0]
            rec["gt_valid_frac"] = float("nan")
            rec["gt_nonzero_frac"] = float((patch > 0).mean())
            rec["depth_scale"] = "raw_uint8"
        s_ = stat(vp, (10, 50, 90))
        rec.update({"gt_med": s_['med'], "gt_mean": s_.get('mean', float('nan')),
                    "gt_std": s_.get('std', float('nan')), "gt_p10": s_['p10'], "gt_p90": s_['p90'], "gt_n": s_['n']})
        # ring = 1.5x box minus box，须完全落在图内且有效像素 >= 10
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rw, rh = (x2 - x1) * 0.75, (y2 - y1) * 0.75
        rx1, ry1 = int(round(cx - rw)), int(round(cy - rh))
        rx2, ry2 = int(round(cx + rw)), int(round(cy + rh))
        if rx1 < 0 or ry1 < 0 or rx2 > w or ry2 > h or (rx2-rx1) < 3 or (ry2-ry1) < 3:
            rec["depth_contrast"] = float("nan")
        else:
            ringmask = np.ones((ry2-ry1, rx2-rx1), bool)
            ix1, iy1 = max(x1, rx1)-rx1, max(y1, ry1)-ry1
            ix2, iy2 = min(x2, rx2)-rx1, min(y2, ry2)-ry1
            ringmask[iy1:iy2, ix1:ix2] = False
            ring = dd[ry1:ry2, rx1:rx2].astype(np.float32)[ringmask]
            if dd.dtype == np.uint16:
                ring = ring[ring >= INVALID_MM]
            else:
                ring = ring[ring > 0]
            if ring.size < 10 or not np.isfinite(rec["gt_med"]):
                rec["depth_contrast"] = float("nan")
            else:
                rec["depth_contrast"] = float(abs(rec["gt_med"] - np.median(ring)))
        rows.append(rec)

    tot = len(rows)
    c = Counter(r["ph"] for r in rows)
    print("\n" + "=" * 100)
    print("§0/§1 Provenance + 8-way phenotype（复用 Phase E，不重新定义）")
    print("=" * 100)
    print(f"  GT={tot}  " + "  ".join(f"{k}={c[k]}" for k in ("111","110","101","100","011","010","001","000")))
    assert tot == 2807 and sum(c.values()) == 2807

    # ---------- §5/§6/§13 Table A/B ----------
    def agg(sel, key, hr_only=True):
        v = [r.get(key) for r in sel if r.get("stratum") == "HR"] if hr_only else [r.get(key) for r in sel]
        v = [x for x in v if x is not None and np.isfinite(x)]
        return (len(v), float(np.median(v)) if v else float("nan"),
                float(np.percentile(v, 75) - np.percentile(v, 25)) if v else float("nan"))

    print("\n" + "=" * 100)
    print("Table A — phenotype × Depth quality（**HR 分层 only**（uint16，invalid 约定已核）；LR 见附注）")
    print("=" * 100)
    print(f"  {'phenotype':<12}{'n(HR)':>7}{'valid_frac':>12}{'depth_std':>12}{'contrast':>11}{'gt_med':>11}")
    tabA = {}
    for ph in ("111", "110", "101", "100", "010", "000", "011", "001"):
        sel = [r for r in rows if r["ph"] == ph]
        n_ac, vf, _ = agg(sel, "gt_valid_frac"); _, sd, _ = agg(sel, "gt_std")
        _, ct, _ = agg(sel, "depth_contrast"); _, gm, _ = agg(sel, "gt_med")
        tabA[ph] = dict(n_total=len(sel), n_hr=n_ac, valid_frac=vf, depth_std=sd, contrast=ct, gt_med=gm)
        print(f"  {ph:<12}{n_ac:>7}{vf:>12.4f}{sd:>12.1f}{ct:>11.1f}{gm:>11.1f}")

    print("\n" + "=" * 100)
    print("Table B — ★ 100 vs 010（最重要的对照）")
    print("=" * 100)
    A100 = [r for r in rows if r["ph"] == "100"]; A010 = [r for r in rows if r["ph"] == "010"]
    print(f"  {'metric':<20}{'100':>12}{'010':>12}{'delta':>12}")
    tabB = {}
    for key in ("gt_valid_frac", "gt_std", "depth_contrast", "gt_med", "area"):
        _, a, _ = agg(A100, key); _, b, _ = agg(A010, key)
        tabB[key] = dict(a=a, b=b, delta=a - b)
        print(f"  {key:<20}{a:>12.4f}{b:>12.4f}{a-b:>+12.4f}")
    # 配对 Mann-Whitney（scipy）
    try:
        from scipy.stats import mannwhitneyu
        for key in ("gt_valid_frac", "depth_contrast", "gt_std"):
            a = [r[key] for r in A100 if r.get("stratum") == "HR" and np.isfinite(r.get(key, np.nan))]
            b = [r[key] for r in A010 if r.get("stratum") == "HR" and np.isfinite(r.get(key, np.nan))]
            if len(a) > 3 and len(b) > 3:
                u, p = mannwhitneyu(a, b, alternative="two-sided")
                d = 2*u/(len(a)*len(b)) - 1
                print(f"    Mann-Whitney {key:<16} n={len(a)}/{len(b)}  Cliff's d={d:+.3f}  p={p:.4f}")
    except Exception as e:
        print(f"    (scipy 不可用: {e})")

    print("\n" + "=" * 100)
    print("Table C — size × (100/010) + Depth quality separation")
    print("=" * 100)
    print(f"  {'size':<9}{'100':>6}{'010':>6}{'net':>6}{'vf100':>9}{'vf010':>9}{'Δvf':>9}")
    for nm, _lo, _hi in SIZE_BINS:
        s100 = [r for r in A100 if r["size"] == nm]; s010 = [r for r in A010 if r["size"] == nm]
        _, v1, _ = agg(s100, "gt_valid_frac"); _, v0, _ = agg(s010, "gt_valid_frac")
        print(f"  {nm:<9}{len(s100):>6}{len(s010):>6}{len(s100)-len(s010):>+6}{v1:>9.4f}{v0:>9.4f}{v1-v0:>+9.4f}")

    print("\n" + "=" * 100)
    print("Table D — class × (D′_ONLY / M4_ONLY) + Depth quality")
    print("=" * 100)
    print(f"  {'class':<13}{'GT':>6}{'D′_ONLY':>9}{'M4_ONLY':>9}{'net':>7}{'vf(D′_ONLY)':>13}")
    dont = [r for r in rows if r["dp"] and not r["m4"]]; m4on = [r for r in rows if r["m4"] and not r["dp"]]
    for ci in range(12):
        sub = [r for r in rows if r["cls"] == ci]
        dd = [r for r in dont if r["cls"] == ci]
        if not sub:
            continue
        _, vf, _ = agg(dd, "gt_valid_frac")
        print(f"  {NAMES[ci]:<13}{len(sub):>6}{len(dd):>9}{len([r for r in m4on if r['cls']==ci]):>9}"
              f"{len(dd)-len([r for r in m4on if r['cls']==ci]):>+7}{vf:>13.4f}")

    # ---------- §9 D′_ONLY 的 IoU 分解 ----------
    print("\n" + "=" * 100)
    print("§9 D′_ONLY(131) 按 D′ matched IoU 分档 × Depth quality")
    print("=" * 100)
    edges = [(0.50, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.01)]
    print(f"  {'IoU band':<14}{'n':>6}{'vf p50':>10}{'contrast p50':>14}{'std p50':>10}")
    for lo, hi in edges:
        sel = [r for r in dont if r.get("dp_iou") is not None and lo <= r["dp_iou"] < hi]
        _, vf, _ = agg(sel, "gt_valid_frac"); _, ct, _ = agg(sel, "depth_contrast"); _, sd, _ = agg(sel, "gt_std")
        print(f"  [{lo:.2f},{hi:.2f}){'':<3}{len(sel):>6}{vf:>10.4f}{ct:>14.1f}{sd:>10.1f}")

    # ---------- §10 image-level Spearman ----------
    print("\n" + "=" * 100)
    print("Table E — image-level Spearman（HR 分层）")
    print("=" * 100)
    per_img = defaultdict(lambda: dict(dp=0, m4=0))
    for r in rows:
        if r["dp"] and not r["m4"]:
            per_img[r["key"][0]]["dp"] += 1
        if r["m4"] and not r["dp"]:
            per_img[r["key"][0]]["m4"] += 1
    xs, vfs, sds, grs = [], [], [], []
    for i, v in img_depth.items():
        if not v or v["stratum"] != "HR":
            continue
        xs.append(per_img[i]["dp"] - per_img[i]["m4"])
        vfs.append(v["valid_fraction"]); sds.append(v["std"] if 'std' in v else np.nan)
        grs.append(v.get("grad_mag_mean", np.nan))
    try:
        from scipy.stats import spearmanr
        print(f"  {'depth metric':<20}{'rho':>9}{'p':>10}{'n':>6}")
        for nm, arr in (("valid_fraction", vfs), ("raw_std", sds), ("grad_mag_mean", grs)):
            a = np.array(arr, float); b = np.array(xs, float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 10:
                rho, p = spearmanr(a[m], b[m])
                print(f"  {nm:<20}{rho:>+9.4f}{p:>10.4f}{int(m.sum()):>6}")
    except Exception as e:
        print(f"  (scipy 不可用: {e})")

    # ---------- LR stratum 附注 ----------
    print("\n" + "=" * 100)
    print("附注 — LR 分层（n=27 图，uint8 depth，invalid 约定 UNKNOWN）")
    print("=" * 100)
    ph_lr = Counter(r["ph"] for r in rows if r.get("stratum") == "LR")
    print(f"  8-way in LR: {dict(ph_lr)}  (总 {sum(ph_lr.values())} GT)")
    print(f"  ⚠ LR 的 zero **不解释为 invalid**（base.py 对 uint8 不做 <300 处理）⇒ 该分层不进 Table A/B 主表")

    Path(A.out).write_text(json.dumps(dict(tabA=tabA, tabB=tabB, n_rows=len(rows),
                                           lr_ph=dict(ph_lr)), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
