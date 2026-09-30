"""TEST_SET_DOMAIN_AUDIT — 只读的测试集 vs train/val 域审计。

严格只读：
  - 不训练、不推理、不加载任何 .pt 权重、不写任何提交、不修改任何模型/评测代码。
  - 只做 cv2.imread / PIL.open 读取 + 统计，产物仅为本目录下的 stats_light.json / stats.json。

关键背景（全量扫描才发现，单文件抽样会完全误判）：
  数据集是**双分辨率 / 双格式混合**：
    HR: 1920x1080 PNG —— visible/infrared = RGB 8bit；depth = I;16 (16-bit 单通道)
    LR: 640x360  JPEG  —— visible/infrared/depth 全部 RGB 8bit（IR/D 三通道恒等）
  因此所有统计必须按 (分辨率, 格式) 分组，混合均值无意义。
  depth 的 16-bit 解码复刻 ultralytics/data/loaders.py:666-669；LR 路径复刻 predict_rect.py:100
  「无 scaleup 封顶，小图放大 2×」。

性能：全部统计量走 cv2.calcHist / meanStdDev / spatialGradient 的 C++ 路径，
分位数由直方图 CDF 精确导出（8bit 精确，16bit 走 65536 桶精确）。
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")  # 静默 libpng/libjpeg 的逐文件告警

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

SETS = {
    "test": ROOT / "data/raw/test",
    "test_pre": ROOT / "data/raw/test初赛",
    "train_raw": ROOT / "data/raw/train",
    # split_train 是 train_raw 的 1600 张子集（同一批文件），不重复读盘；事后按 stem 派生。
    "split_val": ROOT / "data/processed/rgbid_split/images/val",
}
MODS = ["visible", "infrared", "depth"]
LABELS = {
    "train_raw": ROOT / "data/raw/train/labels",
    "split_train": ROOT / "data/processed/rgbid_split/labels/train/visible",
    "split_val": ROOT / "data/processed/rgbid_split/labels/val/visible",
}
Q = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]

D_INVALID_LT = 300.0     # loaders.py:668
D_SCALE = 19999.0        # loaders.py:669
NORM_LONG = 640          # 分辨率归一化代理的画布长边


# --------------------------------------------------------------------------- #
# 直方图 -> 精确分位数 / 矩 / 比例
# --------------------------------------------------------------------------- #
def hist_stats(hist: np.ndarray, label: str, bin_scale: float = 1.0) -> dict:
    """hist 必须覆盖全部取值域（8bit=256 桶，16bit=65536 桶）。bin_scale = 值/桶索引。"""
    n = float(hist.sum())
    if n <= 0:
        return {}
    p = hist / n
    cdf = np.cumsum(p)
    out = {}
    for q in Q:
        idx = int(np.searchsorted(cdf, q / 100.0))
        out[f"{label}_p{q}"] = round(idx * bin_scale, 6)
    nz = np.nonzero(p)[0]
    vals = nz * bin_scale
    mean = float((p[nz] * vals).sum())
    var = float((p[nz] * (vals - mean) ** 2).sum())
    out[f"{label}_mean"] = mean
    out[f"{label}_std"] = float(np.sqrt(max(var, 0.0)))
    out[f"{label}_min"] = float(nz[0] * bin_scale)
    out[f"{label}_max"] = float(nz[-1] * bin_scale)
    out[f"{label}_uniq"] = int(nz.size)
    pnz = p[nz]
    out[f"{label}_entropy"] = float(-(pnz * np.log2(pnz)).sum())
    out[f"{label}_median"] = out[f"{label}_p50"]
    out[f"{label}_iqr"] = out[f"{label}_p75"] - out[f"{label}_p25"]
    out[f"{label}_dr_p99_p1"] = out[f"{label}_p99"] - out[f"{label}_p1"]
    out[f"{label}_dr_p95_p5"] = out[f"{label}_p95"] - out[f"{label}_p5"]
    return out


def hist_fracs(hist: np.ndarray, label: str, lo_hi: list, ge: list, eq: list) -> dict:
    """比例全部由 CDF 精确导出。lo_hi=[(20,'lt'),...]，ge=[(250,'ge'),...]，eq=[255,...]。"""
    n = float(hist.sum())
    p = hist / n
    cdf = np.cumsum(p)
    out = {}
    for v, kind in lo_hi:
        out[f"{label}_frac_lt{v}"] = float(cdf[v - 1]) if v > 0 else 0.0
    for v, kind in ge:
        out[f"{label}_frac_ge{v}"] = float(1.0 - (cdf[v - 1] if v > 0 else 0.0))
    for v in eq:
        out[f"{label}_frac_eq{v}"] = float(p[v])
    return out


def h8(u8: np.ndarray) -> np.ndarray:
    return cv2.calcHist([u8], [0], None, [256], [0, 256]).ravel()


# --------------------------------------------------------------------------- #
# 结构 / 纹理代理（unlabeled proxy）
# --------------------------------------------------------------------------- #
def structure(u8: np.ndarray) -> dict:
    med = float(np.median(u8))
    lo, hi = int(max(0, 0.66 * med)), int(min(255, 1.33 * med))
    e = cv2.Canny(u8, lo, hi)
    lap = cv2.Laplacian(u8, cv2.CV_16S)
    lap_std = float(cv2.meanStdDev(lap)[1][0, 0])
    dx, dy = cv2.spatialGradient(u8)
    g = cv2.add(cv2.absdiff(dx, np.zeros_like(dx)), cv2.absdiff(dy, np.zeros_like(dy)))
    gff = g.astype(np.float32)
    tvd = float(cv2.absdiff(u8[:-1], u8[1:]).mean())
    tvr = float(cv2.absdiff(u8[:, :-1], u8[:, 1:]).mean())
    return {
        "edge_density": float((e > 0).sum() / e.size),
        "lap_var": lap_std ** 2,
        "grad_l1_mean": float(gff.mean()),
        "hf_frac_gt32": float((g > 32).mean()),
        "hf_frac_gt64": float((g > 64).mean()),
        "total_variation": (tvd + tvr) / 2.0,
    }


def structure_norm(u8: np.ndarray, long_side: int = NORM_LONG) -> dict:
    """分辨率归一化代理：缩放到固定长边后再测，使 HR/LR 可比。"""
    h, w = u8.shape[:2]
    s = long_side / max(h, w)
    if s < 1.0:
        r = cv2.resize(u8, (max(8, int(round(w * s))), max(8, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    else:
        r = u8
    med = float(np.median(r))
    e = cv2.Canny(r, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
    dx, dy = cv2.spatialGradient(r)
    g = cv2.add(cv2.absdiff(dx, np.zeros_like(dx)), cv2.absdiff(dy, np.zeros_like(dy)))
    return {"edge_density_norm": float((e > 0).sum() / e.size),
            "grad_l1_mean_norm": float(g.astype(np.float32).mean()),
            "lap_var_norm": float(cv2.meanStdDev(cv2.Laplacian(r, cv2.CV_16S))[1][0, 0] ** 2)}


def cc_proxy(u8: np.ndarray) -> dict:
    """连通域代理：salient = 偏离局部背景的像素。纯 proxy，无 GT 含义。原分辨率精确计算。"""
    resid = cv2.absdiff(u8, cv2.GaussianBlur(u8, (0, 0), 3.0))
    thr = max(8.0, float(resid.mean() + 2.0 * resid.std()))
    m = (resid > thr).astype(np.uint8)
    n_lab, _lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n_lab <= 1:
        return {"cc_n": 0, "cc_area_frac_mean": 0.0, "cc_area_frac_median": 0.0, "cc_max_area_frac": 0.0,
                "cc_diam_median_px": 0.0, "cc_diam_p10_px": 0.0, "cc_diam_p90_px": 0.0,
                "cc_frac_diam_lt16px": 0.0, "cc_frac_diam_lt32px": 0.0, "cc_frac_dim_lt32px": 0.0,
                "cc_salient_frac": 0.0, "cc_diam_median_relw": 0.0}
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    ww = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
    hh = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    tot = float(u8.size)
    d = np.sqrt(4.0 * areas / np.pi)
    return {
        "cc_n": int(n_lab - 1),
        "cc_area_frac_mean": float(areas.mean() / tot),
        "cc_area_frac_median": float(np.median(areas) / tot),
        "cc_max_area_frac": float(areas.max() / tot),
        "cc_diam_median_px": float(np.median(d)),
        "cc_diam_p10_px": float(np.percentile(d, 10)),
        "cc_diam_p90_px": float(np.percentile(d, 90)),
        "cc_frac_diam_lt16px": float((d < 16).mean()),
        "cc_frac_diam_lt32px": float((d < 32).mean()),
        "cc_frac_dim_lt32px": float(((ww < 32) & (hh < 32)).mean()),
        "cc_salient_frac": float(areas.sum() / tot),
        "cc_diam_median_relw": float(np.median(d)) / float(u8.shape[1]),
    }


# --------------------------------------------------------------------------- #
def letterbox_rect(w: int, h: int, imgsz: int = 1280, stride: int = 32):
    """复刻 predict_rect._preprocess_rect: 长边->imgsz(ceil, 无 scaleup 封顶) + stride 对齐画布。"""
    r = imgsz / max(h, w)
    rw, rh = min(int(np.ceil(w * r)), imgsz), min(int(np.ceil(h * r)), imgsz)
    cw = min(int(np.ceil(rw / stride)) * stride, imgsz)
    ch = min(int(np.ceil(rh / stride)) * stride, imgsz)
    pad = (cw * ch - rw * rh) / float(cw * ch) if cw * ch else 0.0
    return cw, ch, pad, r


def one(job):
    path, mod = job
    rec = {"stem": path.stem, "mod": mod}
    # 注意：cv2.imread 在 Windows 上对非 ASCII 路径（data/raw/test初赛）恒返回 None；
    # 必须走 imdecode(np.fromfile(...))。这是本审计脚本唯一的读图入口。
    try:
        raw = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:  # noqa: BLE001
        raw = None
    if raw is None:
        rec["error"] = "imread_none"
        return rec
    try:
        rec["filesize"] = path.stat().st_size
        rec["dtype"] = str(raw.dtype)
        rec["ndim"] = int(raw.ndim)
        h, w = raw.shape[:2]
        rec["w"], rec["h"] = int(w), int(h)
        rec["aspect"] = round(w / h, 4)
        cw, ch, pad, r = letterbox_rect(w, h)
        rec["lb_canvas"] = f"{cw}x{ch}"
        rec["lb_pad_frac"] = pad
        rec["lb_ratio"] = r
        rec["lb_effective_scale"] = r  # LR 为 2.0 = 放大，HR 为 0.667 = 缩小
        rec["nch"] = int(raw.shape[2]) if raw.ndim == 3 else 1
        rec["group"] = f"{w}x{h}_{rec['dtype']}"
        rec["is_hr"] = int(w >= 1920)

        # ---- 统一到 uint8 单通道（管线口径） ----
        if mod == "depth" and raw.dtype == np.uint16:
            rec["is16"] = 1
            h16 = cv2.calcHist([raw], [0], None, [65536], [0, 65536]).ravel()
            rec.update(hist_stats(h16, "d16"))
            tt = float(h16.sum())
            rec["d16_frac_lt300"] = float(h16[:300].sum() / tt)   # loaders.py 判为 invalid
            rec["d16_frac_zero"] = float(h16[0] / tt)
            rec["d16_frac_ge19999"] = float(h16[19999:].sum() / tt)  # 解码后被 clip 到 255
            rec["d16_frac_valid"] = float(h16[300:].sum() / tt)
            rec["d16_rawmax"] = int(np.nonzero(h16)[0][-1])
            rec["d16_rawmin"] = int(np.nonzero(h16)[0][0])
            f = raw.astype(np.float32)
            f[f < D_INVALID_LT] = 0.0
            u8 = np.clip(f / D_SCALE * 255.0, 0, 255).astype(np.uint8)
            rec["ch_identical"] = 1
            # valid-only 分布（排除 invalid 0）
            hv = h16[300:]
            if hv.sum() > 0:
                v = np.nonzero(hv)[0] + 300
                s = hv[hv > 0] / hv.sum()
                m = float((s * v).sum())
                rec["d16valid_mean"] = m
                rec["d16valid_std"] = float(np.sqrt(max(float((s * (v - m) ** 2).sum()), 0.0)))
                c = np.cumsum(s)
                for q in Q:
                    rec[f"d16valid_p{q}"] = float(v[int(np.searchsorted(c, q / 100.0))])
        elif raw.ndim == 3 and raw.shape[2] >= 3:
            ch_id = bool(np.array_equal(raw[..., 0], raw[..., 1]) and np.array_equal(raw[..., 0], raw[..., 2]))
            rec["ch_identical"] = int(ch_id)
            rec["is16"] = 0
            if ch_id:
                u8 = raw[..., 0]
            elif mod == "visible":
                u8 = cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2GRAY)
            else:
                u8 = raw[..., 0]
        else:
            rec["ch_identical"] = 1
            rec["is16"] = 0
            u8 = raw if raw.ndim == 2 else raw[..., 0]
        if u8.dtype != np.uint8:
            u8 = np.clip(u8.astype(np.float32) / max(1.0, float(u8.max())) * 255.0, 0, 255).astype(np.uint8)

        # ---- uint8 统计 ----
        hu = h8(u8)
        rec.update(hist_stats(hu, "u8"))
        rec.update(hist_fracs(hu, "u8", [(10, "lt"), (20, "lt"), (50, "lt")],
                              [(200, "ge"), (250, "ge"), (254, "ge")], [0, 255]))
        rec.update(structure(u8))
        rec.update(structure_norm(u8))
        rec.update(cc_proxy(u8))

        # ---- 可见光专属：色彩 / 曝光 ----
        if mod == "visible" and raw.ndim == 3 and raw.shape[2] == 3 and not rec["ch_identical"]:
            ch = raw[..., :3]
            mean, std = cv2.meanStdDev(ch)
            rec["ch_mean_b"], rec["ch_mean_g"], rec["ch_mean_r"] = [float(x) for x in mean.ravel()]
            rec["ch_std_b"], rec["ch_std_g"], rec["ch_std_r"] = [float(x) for x in std.ravel()]
            hsv = cv2.cvtColor(ch, cv2.COLOR_BGR2HSV)
            rec["sat_mean"] = float(hsv[..., 1].mean())
            rec["sat_lt30_frac"] = float((hsv[..., 1] < 30).mean())
            rec["rg_diff"] = float(cv2.absdiff(ch[..., 2], ch[..., 1]).mean())
            mx = cv2.max(cv2.max(ch[..., 0], ch[..., 1]), ch[..., 2])
            mn = cv2.min(cv2.min(ch[..., 0], ch[..., 1]), ch[..., 2])
            rec["chroma_mean"] = float(cv2.absdiff(mx, mn).mean())
            rec["frac_any_ch_ge255"] = float((mx >= 255).mean())
            rec["frac_any_ch_le2"] = float((mn <= 2).mean())
            rec["frac_all_ch_lt10"] = float((mx < 10).mean())
            lum = u8.astype(np.float32)
            bh, bw = lum.shape[0] // 4, lum.shape[1] // 4
            bl = np.array([lum[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw].mean()
                           for i in range(4) for j in range(4)])
            rec["block_lum_std"] = float(bl.std())
            rec["block_lum_min"] = float(bl.min())
            rec["block_lum_max"] = float(bl.max())
            rec["block_lum_range"] = float(bl.max() - bl.min())
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"stats_exc:{type(e).__name__}:{e}"
    return rec


# --------------------------------------------------------------------------- #
KEYS_EXCLUDE = {"w", "h", "aspect", "filesize", "lb_pad_frac", "lb_ratio", "lb_effective_scale",
                "is_hr", "is16", "ch_identical", "nch", "ndim"}


def summarize(recs):
    ok = [r for r in recs if "error" not in r]
    out = {"n": len(ok), "n_err": len(recs) - len(ok),
           "errors": [(r.get("stem"), r["error"]) for r in recs if "error" in r][:10]}
    if not ok:
        return out
    keys = sorted({k for r in ok for k, v in r.items()
                   if isinstance(v, float) and not isinstance(v, bool)})
    for k in keys:
        v = np.array([r[k] for r in ok if k in r], dtype=np.float64)
        if v.size == 0:
            continue
        out[k] = {"n": int(v.size), "mean": round(float(v.mean()), 6), "std": round(float(v.std()), 6),
                  "p5": round(float(np.percentile(v, 5)), 6), "p25": round(float(np.percentile(v, 25)), 6),
                  "median": round(float(np.median(v)), 6), "p75": round(float(np.percentile(v, 75)), 6),
                  "p95": round(float(np.percentile(v, 95)), 6),
                  "min": round(float(v.min()), 6), "max": round(float(v.max()), 6)}
    out["n_hr"] = int(sum(r.get("is_hr", 0) for r in ok))
    out["n_lr"] = int(len(ok) - sum(r.get("is_hr", 0) for r in ok))
    out["n_16bit"] = int(sum(r.get("is16", 0) for r in ok))
    out["groups"] = dict(Counter(r["group"] for r in ok))
    out["lb_canvas"] = dict(Counter(r["lb_canvas"] for r in ok))
    out["lb_pad_frac_uniq"] = sorted({round(r["lb_pad_frac"], 6) for r in ok})
    out["lb_ratio_uniq"] = sorted({round(r["lb_ratio"], 6) for r in ok})
    out["ch_identical"] = dict(Counter(str(r.get("ch_identical")) for r in ok))
    return out


def gt_stats(label_dir: Path):
    if not label_dir or not label_dir.exists():
        return None
    rows, per_img = [], []
    for f in sorted(label_dir.glob("*.txt")):
        txt = f.read_text().strip()
        if not txt:
            per_img.append(0)
            continue
        c = 0
        for line in txt.splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            rows.append((float(p[3]), float(p[4])))
            c += 1
        per_img.append(c)
    if not rows:
        return {"n_img": len(per_img), "n_box": 0}
    a = np.array(rows)
    ar_hr = a[:, 0] * a[:, 1] * 1920.0 * 1080.0   # 归一化 -> HR 像素面积
    ar_lr = a[:, 0] * a[:, 1] * 640.0 * 360.0     # 归一化 -> LR 像素面积
    return {
        "n_img": len(per_img), "n_box": int(len(rows)),
        "boxes_per_img_mean": round(float(np.mean(per_img)), 4),
        "boxes_per_img_median": float(np.median(per_img)),
        "empty_img_frac": round(float(np.mean([c == 0 for c in per_img])), 5),
        "norm_w_median": round(float(np.median(a[:, 0])), 6),
        "norm_h_median": round(float(np.median(a[:, 1])), 6),
        "rel_sqrt_area": {"median": round(float(np.median(np.sqrt(a[:, 0] * a[:, 1]))), 6),
                          "p10": round(float(np.percentile(np.sqrt(a[:, 0] * a[:, 1]), 10)), 6),
                          "p90": round(float(np.percentile(np.sqrt(a[:, 0] * a[:, 1]), 90)), 6)},
        "hr_px_sqrt_area": {k: round(float(v), 3) for k, v in zip(
            ["mean", "p10", "p25", "median", "p75", "p90"],
            [np.sqrt(ar_hr).mean(), *np.percentile(np.sqrt(ar_hr), [10, 25, 50, 75, 90])])},
        "hr_frac_coco_small": round(float((ar_hr < 32 * 32).mean()), 5),
        "hr_frac_coco_medium": round(float(((ar_hr >= 32 * 32) & (ar_hr < 96 * 96)).mean()), 5),
        "hr_frac_coco_large": round(float((ar_hr >= 96 * 96).mean()), 5),
        "lr_frac_coco_small": round(float((ar_lr < 32 * 32).mean()), 5),
        "hr_frac_side_lt32px": round(float(((a[:, 0] * 1920 < 32) | (a[:, 1] * 1080 < 32)).mean()), 5),
        "norm_w_gt1": round(float((a[:, 0] > 1).mean()), 6),
    }


def main():
    jobs, index = [], {}
    for sname, sdir in SETS.items():
        for mod in MODS:
            d = sdir / mod
            if not d.exists():
                print("MISSING", d, flush=True)
                continue
            fs = [p for p in sorted(d.iterdir())
                  if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")]
            index[(sname, mod)] = fs
            jobs += [(p, mod) for p in fs]
    print(f"total files to read: {len(jobs)}", flush=True)

    cache = OUT / "_records_cache.json"
    if cache.exists():
        results = json.loads(cache.read_text(encoding="utf-8"))
        print(f"loaded {len(results)} cached records", flush=True)
    else:
        with Pool(16) as pool:
            results = pool.map(one, jobs, chunksize=24)
        cache.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    by_parent = defaultdict(list)
    for r, (p, _m) in zip(results, jobs):
        by_parent[str(p.parent)].append(r)

    per = {}
    for (sname, mod), fs in index.items():
        recs = by_parent[str(fs[0].parent)]
        assert len(recs) == len(fs), (sname, mod, len(recs), len(fs))
        per.setdefault(sname, {})[mod] = {
            "stems": sorted({p.stem for p in fs}),
            "summary": summarize(recs),
            "by_group": {g: summarize([r for r in recs if r.get("group") == g])
                         for g in sorted({r.get("group") for r in recs if r.get("group")})},
            "records": recs,
        }

    pairing = {}
    for sname, mods in per.items():
        vis = set(mods.get("visible", {}).get("stems", []))
        ir = set(mods.get("infrared", {}).get("stems", []))
        dp = set(mods.get("depth", {}).get("stems", []))
        pairing[sname] = {
            "n_visible": len(vis), "n_infrared": len(ir), "n_depth": len(dp),
            "vis_and_ir": len(vis & ir), "vis_and_dp": len(vis & dp), "ir_and_dp": len(ir & dp),
            "all3": len(vis & ir & dp),
            "ir_not_in_vis": sorted(ir - vis)[:10], "dp_not_in_vis": sorted(dp - vis)[:10],
        }

    mismatch = {}
    for sname, mods in per.items():
        if "visible" not in mods:
            continue
        gmap = defaultdict(dict)
        for mod in MODS:
            for r in mods.get(mod, {}).get("records", []):
                gmap[r["stem"]][mod] = r.get("group")
        mm = [s for s, d in gmap.items() if len(set(v for v in d.values() if v)) > 1]
        mismatch[sname] = {"n_stem": len(gmap), "n_cross_format_mismatch": len(mm), "examples": mm[:10]}

    overlap = {}
    ts = set(per["train_raw"]["visible"]["stems"])
    ss = {p.stem for p in (ROOT / "data/processed/rgbid_split/labels/train/visible").glob("*.txt")}
    pv = set(per.get("split_val", {}).get("visible", {}).get("stems", []))
    for sname, mods in per.items():
        v = set(mods.get("visible", {}).get("stems", []))
        overlap[sname] = {"vs_train_raw": len(v & ts), "vs_split_train": len(v & ss), "vs_split_val": len(v & pv)}

    light = {"pairing": pairing, "group_mismatch": mismatch, "stem_overlap": overlap,
             "gt": {}, "summary": {}, "by_group": {}}
    for sname, ldir in LABELS.items():
        light["gt"][sname] = gt_stats(ldir)
    for sname, mods in per.items():
        light["summary"][sname] = {m: v["summary"] for m, v in mods.items()}
        light["by_group"][sname] = {m: v["by_group"] for m, v in mods.items()}
    (OUT / "stats_light.json").write_text(json.dumps(light, ensure_ascii=False, indent=1), encoding="utf-8")

    full = dict(light)
    full["records"] = {}
    for sname, mods in per.items():
        for m, v in mods.items():
            drop = {"w", "h"}
            full["records"][f"{sname}/{m}"] = [{k: x for k, x in r.items() if k not in ("path",)} for r in v["records"]]
    (OUT / "stats.json").write_text(json.dumps(full, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT / "stats_light.json", flush=True)


if __name__ == "__main__":
    main()
