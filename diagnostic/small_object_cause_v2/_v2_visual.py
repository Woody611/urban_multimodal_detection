"""_v2_visual.py — V2 阶段 3（可计算 visual proxy）/ 10（可检测性）。

只读：从已有 RGB/IR/Depth 图按 GT 框裁局部区域，算可计算 proxy。
不改 preprocessing（只做无损裁剪与基础统计），不训练、不推理、不生成 prediction。

⚠ 明确限制：本脚本**不能**判断「人类能否识别」。它只给 CLEAR/AMBIGUOUS/LOW_VISIBILITY
的可计算 proxy，且 occlusion 用的是 GT 框重叠代理，不是像素级遮挡分割。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = Path(__file__).resolve().parent

from _v2_records import EPS0, IMAGES, LABELS, NAMES, SMALL_A, iter_split  # noqa: E402

MODS = ["visible", "infrared", "depth"]


def img_path(mod, stem):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
        p = IMAGES.parent.parent / "val" / mod / f"{stem}{ext}"
        if p.exists():
            return p
    d = IMAGES.parent / mod
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
        p = d / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def crop_stats(gray, x1, y1, x2, y2):
    """在 GT 框外扩 k 倍的局部区域内算 proxy。返回 dict。"""
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(gray.shape[1], int(np.ceil(x2))), min(gray.shape[0], int(np.ceil(y2)))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    r = gray[y1:y2, x1:x2].astype(np.float32)
    lap = cv2.Laplacian(r, cv2.CV_32F)
    gx = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    edge = cv2.Canny(gray[y1:y2, x1:x2].astype(np.uint8), 50, 150)
    return dict(lapvar=float(lap.var()), gradmean=float(mag.mean()),
                std=float(r.std()), mean=float(r.mean()),
                edgedens=float((edge > 0).mean()), npx=int(r.size))


def main():
    data = json.loads((OUT / "_v2_records.json").read_text(encoding="utf-8"))
    recs = data["records"]
    by_key = {(r["image_id"], r["gt_id"]): r for r in recs}

    # 载入 val GT 以计算遮挡代理
    gt_by = {}
    for stem, w, h, b, c in iter_split(IMAGES, LABELS):
        gt_by[stem] = dict(w=w, h=h, b=b, c=c)

    def occlusion_proxy(stem, j):
        """GT 框被**其它 GT 框**覆盖的面积比（代理，非像素级遮挡）。"""
        v = gt_by.get(stem)
        if v is None or v["b"] is None or len(v["b"]) < 2:
            return 0.0, 0
        others = [k for k in range(len(v["b"])) if k != j]
        if not others:
            return 0.0, 0
        box = v["b"][j]
        area = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
        if area <= 0:
            return 0.0, 0
        # 用「覆盖比例最大的单个其它框」作为主遮挡源，另给并集覆盖
        best = 0.0
        for k in others:
            ob = v["b"][k]
            ix1, iy1 = max(box[0], ob[0]), max(box[1], ob[1])
            ix2, iy2 = min(box[2], ob[2]), min(box[3], ob[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            best = max(best, inter / area)
        return float(best), len(others)

    target = [r for r in recs if r["best_any_class_iou"] <= EPS0]
    ctrl = [r for r in recs if r["best_same_class_iou"] >= 0.5]

    def measure(rs):
        acc = []
        for r in rs:
            stem, j = r["image_id"], r["gt_id"]
            v = gt_by.get(stem)
            if v is None or v["b"] is None:
                continue
            box = v["b"][j]
            x1, y1, x2, y2 = box
            w, h = max(x2 - x1, 1), max(y2 - y1, 1)
            k = 1.5  # 外扩 1.5 倍，含上下文
            X1, Y1 = x1 - w * k / 2, y1 - h * k / 2
            X2, Y2 = x2 + w * k / 2, y2 + h * k / 2
            row = dict(key=f"{stem}#{j}", name=r["name"], in_sqrt=r["sqrt_area"] * r["scale"])
            occ_best, n_nb = occlusion_proxy(stem, j)
            row["occ_proxy"] = occ_best
            row["n_other_gt"] = n_nb
            for mod in MODS:
                p = img_path(mod, stem)
                if p is None:
                    continue
                im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if im is None:
                    continue
                gray = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                st = crop_stats(gray, X1, Y1, X2, Y2)
                if st:
                    for kk, vv in st.items():
                        row[f"{mod}_{kk}"] = vv
            acc.append(row)
        return acc

    print("=" * 78)
    print("阶段 3/10 —— 可计算 visual proxy（missed vs detected，尺寸匹配对照）")
    print("=" * 78)
    T = measure(target)
    C = measure(ctrl)
    print(f"  missed(any-IoU==0) n={len(T)}   detected(same-IoU>=0.5) n={len(C)}")

    # 尺寸匹配：只保留 detected 中落在 missed 尺寸 P5-P95 区间内的
    lo, hi = np.percentile([r["in_sqrt"] for r in T], [5, 95])
    C2 = [r for r in C if lo <= r["in_sqrt"] <= hi]
    print(f"  尺寸匹配后对照 n={len(C2)}（区间 [{lo:.2f},{hi:.2f}]px）")

    keys = ["occ_proxy", "visible_lapvar", "visible_gradmean", "visible_std",
            "visible_edgedens", "infrared_std", "infrared_gradmean",
            "depth_std", "depth_gradmean"]
    print(f"\n  {'metric':<24}{'missed 中位':>13}{'detected 中位':>14}{'比值':>9}")
    for k in keys:
        a = [r[k] for r in T if k in r]
        b = [r[k] for r in C2 if k in r]
        if not a or not b:
            continue
        ma, mb = float(np.median(a)), float(np.median(b))
        print(f"  {k:<24}{ma:>13.4f}{mb:>14.4f}{(ma/(mb+1e-12)):>9.3f}")

    # 可检测性分级（只用可计算 proxy）
    def grade(r):
        v = r.get("visible_std", 0); g = r.get("visible_gradmean", 0)
        e = r.get("visible_edgedens", 0)
        if v > 40 and g > 40 and e > 0.10:
            return "CLEAR"
        if v > 22 and g > 20 and e > 0.04:
            return "AMBIGUOUS"
        return "LOW_VISIBILITY"

    from collections import Counter
    print(f"\n  可检测性分级  missed: {dict(Counter(grade(r) for r in T))}")
    print(f"                detected: {dict(Counter(grade(r) for r in C2))}")

    (OUT / "_v2_visual.json").write_text(json.dumps(dict(target=T, control=C2),
                                                    indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {OUT/'_v2_visual.json'}")
    return T, C2


if __name__ == "__main__":
    main()
