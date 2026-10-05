# -*- coding: utf-8 -*-
"""P0 §3 — D' IR 数据流统计（CPU only，不训练、不碰 CUDA）。

复刻 base.py::load_and_preprocess_image 的 RGBID 分支里 **IR 的那一段**，分别统计：
  IR_raw         : cv2.imread(..., IMREAD_GRAYSCALE)（= 代码真正喂给 apply_ir_encoding 的数组）
  IR_after_CLAHE : apply_ir_encoding(IR_raw, "clahe")

对 train / val 两个 split 分别给出 min max mean std p01 p05 p25 p50 p75 p95 p99，
以及 per-image mean 的分布（看"图像间"差异，而不只是被单张图主导）。

实现要点（内存/时间有界）：
  - 抽样 SAMPLE_N 张/ split（等间隔取，确定性）；
  - 用 uint8 直方图（np.bincount）算分位数 → 精确到整数、内存 O(256)。
"""
import os
import sys
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ultralytics.data.base import apply_ir_encoding, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID  # noqa: E402
from ultralytics.utils.patches import imread  # noqa: E402  # 与 base.py 同一个 imread（3 维返回）

ROOT = "data/processed/rgbid_split_train/images"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ir_stats.json")
PCTS = (1, 5, 25, 50, 75, 95, 99)
SAMPLE_N = 200


def hist_of(files, ir_dir, fn):
    h = np.zeros(256, dtype=np.int64)
    means = []
    for f in files:
        ir = imread(os.path.join(ir_dir, f), cv2.IMREAD_GRAYSCALE)
        if ir is None:
            raise FileNotFoundError(f)
        out = fn(ir)
        assert out.dtype == np.uint8, (f, out.dtype, out.shape)
        h += np.bincount(np.asarray(out).ravel(), minlength=256)
        means.append(float(out.mean()))
    return h, np.array(means)


def describe(h, means):
    n = int(h.sum())
    x = np.arange(256, dtype=np.float64)
    mean = float((h * x).sum() / n)
    std = float(np.sqrt((h * (x - mean) ** 2).sum() / n))
    c = np.cumsum(h)
    def pct(p):
        return int(np.searchsorted(c, p / 100.0 * n, side="left"))
    d = {
        "n_pixels": n,
        "min": int(np.searchsorted(h, 1, side="left")),
        "max": int(255 - np.searchsorted(h[::-1], 1, side="left")),
        "mean": mean,
        "std": std,
        "pctl": {f"p{p:02d}": pct(p) for p in PCTS},
    }
    d["per_image_mean"] = {
        "n": int(means.size),
        "min": float(means.min()), "max": float(means.max()),
        "mean": float(means.mean()), "std": float(means.std()),
        "p05": float(np.percentile(means, 5)), "p50": float(np.percentile(means, 50)),
        "p95": float(np.percentile(means, 95)),
    }
    return d


def run_split(split):
    ir_dir = os.path.join(ROOT, split, "infrared")
    allf = sorted(os.listdir(ir_dir))
    idx = np.linspace(0, len(allf) - 1, min(SAMPLE_N, len(allf))).round().astype(int)
    files = [allf[i] for i in sorted(set(idx.tolist()))]

    ir0 = imread(os.path.join(ir_dir, files[0]), cv2.IMREAD_UNCHANGED)
    ir0g = imread(os.path.join(ir_dir, files[0]), cv2.IMREAD_GRAYSCALE)
    h_raw, m_raw = hist_of(files, ir_dir, lambda x: x)
    h_clh, m_clh = hist_of(files, ir_dir, lambda x: apply_ir_encoding(x, "clahe"))
    return {
        "n_images_total": len(allf),
        "n_images_sampled": len(files),
        "file_dtype_imread_unchanged": str(ir0.dtype),
        "file_shape_imread_unchanged": list(ir0.shape),
        "file_dtype_imread_grayscale": str(ir0g.dtype),
        "file_shape_imread_grayscale": list(ir0g.shape),
        "IR_raw": describe(h_raw, m_raw),
        "IR_after_CLAHE": describe(h_clh, m_clh),
    }


def main():
    res = {
        "tool": "ir_stats.py",
        "clahe": {"clipLimit": CLAHE_CLIP_LIMIT, "tileGridSize": list(CLAHE_TILE_GRID)},
        "source": ROOT,
        "sampling": f"linspace over sorted filenames, SAMPLE_N={SAMPLE_N}",
        "note": "IR_raw = cv2.imread(...,IMREAD_GRAYSCALE); CLAHE via ultralytics.data.base.apply_ir_encoding",
    }
    for split in ("train", "val"):
        res[split] = run_split(split)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

    for split in ("train", "val"):
        s = res[split]
        print(f"===== {split}: {s['n_images_sampled']}/{s['n_images_total']} sampled; "
              f"file(IMREAD_UNCHANGED) dtype={s['file_dtype_imread_unchanged']} shape={s['file_shape_imread_unchanged']}; "
              f"GRAYSCALE dtype={s['file_dtype_imread_grayscale']} shape={s['file_shape_imread_grayscale']}")
        for k in ("IR_raw", "IR_after_CLAHE"):
            d = s[k]; p = d["pctl"]
            print(f"  {k:15s} min={d['min']:3d} max={d['max']:3d} mean={d['mean']:7.3f} std={d['std']:7.3f} | "
                  f"p01={p['p01']:3d} p05={p['p05']:3d} p25={p['p25']:3d} p50={p['p50']:3d} "
                  f"p75={p['p75']:3d} p95={p['p95']:3d} p99={p['p99']:3d}")
            im = d["per_image_mean"]
            print(f"      per-image mean: {im['mean']:.2f} ± {im['std']:.2f}  "
                  f"p05={im['p05']:.1f} p50={im['p50']:.1f} p95={im['p95']:.1f} [{im['min']:.1f},{im['max']:.1f}]")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
