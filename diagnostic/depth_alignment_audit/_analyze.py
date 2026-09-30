"""_analyze.py — Depth↔RGB 空间对应基础设施审计（READ-ONLY）

回答：depth 文件与 visible/RGB、GT label 到底处于什么坐标系？训练 pipeline 在哪一步做了什么
resize/crop/registration？上一轮 STOP B 是「数据未配准」还是「审计坐标链路用错」？

禁止（本轮）：训练/backward/optimizer.step/checkpoint/改任何既有 source·config·data·label·
depth 原文件/augment·base·yaml·train.py·evaluator/重跑 forward/重生成 prediction/重定义 G1-G4。
只读 + 新建 diagnostic/depth_alignment_audit/。

成功标准（协议 §17）：COORDINATE_CHAIN = FULLY_EXPLAINED。
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw"
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
IDX = {"visible": ("visible", "visible"), "ir": ("infrared", "pairs_rgb_ir"),
       "depth": ("depth", "pairs_rgb_depth")}


def hdr(p: Path):
    """只读 header（PIL 懒加载，不解码像素）。"""
    try:
        with Image.open(p) as im:
            return dict(w=int(im.width), h=int(im.height), mode=im.mode, fmt=im.format,
                        n_frames=getattr(im, "n_frames", 1),
                        exif_orientation=(im.getexif().get(274) if im.getexif() else None))
    except Exception as e:  # noqa: BLE001
        return dict(err=f"{type(e).__name__}: {e}")


def main():
    print("=" * 108)
    print("DEPTH↔RGB SPATIAL ALIGNMENT INFRASTRUCTURE AUDIT（READ-ONLY）")
    print("=" * 108)

    # ================= §5 RAW FILE CORRESPONDENCE =================
    print("\n[§5] RAW FILE LEVEL 对应表（全部文件 header，只读）")
    out = {}
    for split, sub in (("train", "train"), ("val", None)):
        base = RAW / "train" if split == "train" else ROOT / "data/processed/rgbid_split_train/images/val"
        key = {}
        for mod, d in (("visible", "visible"), ("ir", "infrared"), ("depth", "depth")):
            files = [p for p in (base / d).iterdir() if p.suffix.lower() in IMG_EXT]
            key[mod] = {p.stem: p for p in files}
        stems = sorted(set(key["visible"]) | set(key["ir"]) | set(key["depth"]))
        print(f"\n  --- {split} (n_stem={len(stems)}) ---")
        miss = {m: 0 for m in key}
        combos = defaultdict(int)
        dt = defaultdict(Counter)
        for s in stems:
            hs = {}
            for m in ("visible", "ir", "depth"):
                p = key[m].get(s)
                if p is None:
                    miss[m] += 1
                    continue
                h = hdr(p)
                hs[m] = h
                dt[m][f"{h.get('fmt')}/{h.get('mode')}"] += 1
            if len(hs) == 3:
                combos[(hs["visible"]["w"], hs["visible"]["h"],
                        hs["ir"]["w"], hs["ir"]["h"],
                        hs["depth"]["w"], hs["depth"]["h"])] += 1
        print(f"    缺失 stem: " + ", ".join(f"{m}={c}" for m, c in miss.items()))
        for m in ("visible", "ir", "depth"):
            print(f"    {m:<8} 格式/模式: {dict(dt[m])}")
        print(f"    三方 (W,H) 组合（visible_w,visible_h,ir_w,ir_h,depth_w,depth_h）:")
        for k, c in sorted(combos.items(), key=lambda t: -t[1]):
            same = (k[0] == k[2] == k[4]) and (k[1] == k[3] == k[5])
            print(f"      {k}  n={c}  三方同尺寸={same}")
        out[split] = dict(n_stem=len(stems), missing=dict(miss),
                          formats={m: dict(dt[m]) for m in dt},
                          shape_combos={str(k): v for k, v in combos.items()})

        # EXIF orientation
        ex = Counter()
        for m in ("visible", "ir", "depth"):
            for s in list(key[m])[:200]:
                ex[(m, hdr(key[m][s]).get("exif_orientation"))] += 1
        print(f"    EXIF orientation(前200样本): {dict(ex)}")

    # ================= §15 depth 数值语义复核 =================
    print("\n[§15] DEPTH 数值语义（代码证据 + 只读直方图）")
    print("  base.py:344-347（RGBID 分支，实读）:")
    print("      im_depth[im_depth < 300] = 0.0                      # 无效深度(<300mm) 置 0")
    print("      im_depth = np.clip(im_depth / 19999.0 * 255.0, 0, 255).astype(np.uint8)")
    print("  ⇒ 有效阈值是 **300（mm）**，不是 0；19999 是饱和常数；0 只是『已被置零的无效值』之一")
    dep = sorted((RAW / "train/depth").glob("*.png"))[:20]
    rows = []
    for p in dep:
        a = np.array(Image.open(p))
        n = a.size
        rows.append(dict(stem=p.stem, dtype=str(a.dtype), shape=a.shape,
                         f0=float((a == 0).mean()),
                         f_lt300=float((a < 300).mean()),
                         f_ge300=float((a >= 300).mean()),
                         f_sat=float((a >= 19999).mean()),
                         vmin=int(a.min()), vmax=int(a.max())))
    print(f"\n  {'stem':<26}{'dtype':<9}{'shape':<16}{'==0':>8}{'<300':>8}{'>=300':>8}{'==19999':>9}"
          f"{'min':>7}{'max':>8}")
    for r in rows[:10]:
        print(f"  {r['stem']:<26}{r['dtype']:<9}{str(r['shape']):<16}{100*r['f0']:>7.2f}%"
              f"{100*r['f_lt300']:>7.2f}%{100*r['f_ge300']:>7.2f}%{100*r['f_sat']:>8.2f}%"
              f"{r['vmin']:>7}{r['vmax']:>8}")
    print(f"\n  20 张中位： ==0 {100*np.median([r['f0'] for r in rows]):.2f}%  "
          f"<300 {100*np.median([r['f_lt300'] for r in rows]):.2f}%  "
          f">=300 {100*np.median([r['f_ge300'] for r in rows]):.2f}%")
    print("  ⇒ 『==0』会**低估**无效比例：1..299 也是无效（被 pipeline 置零）。")

    # ================= §8 T2 平移搜索（raw 层，预注册 primary metric） =================
    print("\n[§7/§8] RAW 层对齐检验（预注册 primary metric = Sobel 幅值 Pearson r，8×8 下采样）")
    print("       secondary = Canny 边缘 Jaccard；二者均报告，**不以 secondary 改判 primary**")
    pairs = []
    for st in [p.stem for p in sorted((RAW / "train/depth").glob("*.png"))][:40]:
        pairs.append((RAW / "train/visible" / f"{st}.png", RAW / "train/depth" / f"{st}.png"))
    res = []
    for vp, dp in pairs:
        v = np.array(Image.open(vp).convert("L"), dtype=np.float32)
        d = np.array(Image.open(dp), dtype=np.float32)
        if d.ndim == 3:
            d = d[:, :, 0]
        if v.shape != d.shape:
            continue
        dv = np.zeros_like(d); dv[:, :-1] += np.abs(np.diff(d, axis=1)); dv[:-1, :] += np.abs(np.diff(d, axis=0))
        sv = cv2_sobel(v)
        A, B = ds8(sv), ds8(dv)
        r00 = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
        bc, bs = -9.0, None
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                c = float(np.corrcoef(A.ravel(), np.roll(np.roll(B, dy, 0), dx, 1).ravel())[0, 1])
                if c > bc:
                    bc, bs = c, (dy * 8, dx * 8)
        # Canny Jaccard
        ev = canny(v); ed = canny(d)
        jac = float((ev & ed).sum() / max((ev | ed).sum(), 1))
        res.append(dict(stem=vp.stem, r00=r00, rbest=bc, shift=bs, canny_jac=jac))
    r0 = [r["r00"] for r in res]; rb = [r["rbest"] for r in res]
    sh = np.array([r["shift"] for r in res])
    print(f"  n={len(res)}  primary(Sobel r) @(0,0) 中位 = {np.median(r0):.4f}")
    print(f"  primary 最佳平移后 中位 = {np.median(rb):.4f}  （提升 {np.median(rb)-np.median(r0):+.4f}）")
    print(f"  最佳位移 中位 (dy,dx) = ({int(np.median(sh[:,0]))},{int(np.median(sh[:,1]))})  "
          f"标准差 (dy,dx)=({sh[:,0].std():.1f},{sh[:,1].std():.1f})")
    print(f"  最佳位移是否高度分散（>16px 标准差 ⇒ 无法用单一 global translation 解决）: "
          f"{bool(sh[:,0].std()>16 or sh[:,1].std()>16)}")
    print(f"  secondary(Canny Jaccard) 中位 = {np.median([r['canny_jac'] for r in res]):.5f}")

    (AUDIT / "_tables.json").write_text(json.dumps(dict(raw_correspondence=out,
        alignment=dict(n=len(res), sobel_r00_median=float(np.median(r0)),
                       sobel_rbest_median=float(np.median(rb)),
                       best_shift_median=[int(np.median(sh[:, 0])), int(np.median(sh[:, 1]))],
                       best_shift_std=[float(sh[:, 0].std()), float(sh[:, 1].std())],
                       canny_jaccard_median=float(np.median([r["canny_jac"] for r in res]))),
        depth_semantics=dict(threshold_mm=300, saturation=19999),
        registration_metadata="NOT_FOUND"), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {AUDIT/'_tables.json'}")
    return 0


def ds8(a):
    H, W = a.shape[:2]
    return a[:H // 8 * 8, :W // 8 * 8].reshape(H // 8, 8, W // 8, 8).mean((1, 3))


def cv2_sobel(g):
    import cv2
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def canny(g):
    import cv2
    return cv2.Canny(np.clip(g, 0, 255).astype(np.uint8), 60, 160) > 0


if __name__ == "__main__":
    sys.exit(main())
