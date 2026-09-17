"""scripts/compare_rect_square.py — 阶段五：rect vs square 逐图一致性验证。

对 val 集按 8 类维度（小图/大图/高图/宽图/密集/稀疏/无效Depth/小目标）挑选代表性图片，
逐图对比 rect 与 square 两种 pipeline 的预测（框数、按 IoU 匹配），输出 per_image_comparison.csv，
并给出每张图是否「rect 框数 ≤ square 框数（对齐 val 的趋势）」的判定。

用法:
  python scripts/compare_rect_square.py \
      --rect_dir diagnostic/rect_pipeline_fix/validation_rect_predictions/results \
      --square_dir diagnostic/rect_pipeline_fix/validation_square_predictions/results \
      --val_images data/processed/depth_split_train/images/val/visible \
      --val_labels data/processed/depth_split_train/labels/val/visible \
      --val_depth data/processed/depth_split_train/images/val/depth \
      --out_csv diagnostic/rect_pipeline_fix/per_image_comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import cv2
import numpy as np

SMALL_OBJ_AREA = 0.003   # 归一化面积 < 0.3% 视为小目标
DENSE_N = 8              # GT 框数 >= 8 视为密集
SPARSE_N = 2             # GT 框数 <= 2 视为稀疏
INVALID_DEPTH_ZERO = 0.5  # depth 零值占比 > 50% 视为无效 Depth


def _read_preds(path: Path):
    """读预测 TXT，返回 [[cx,cy,w,h,conf,cid], ...]（原图归一化坐标）。"""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 6:
            continue
        cid, cx, cy, w, h, conf = int(p[0]), *map(float, p[1:6])
        out.append([cx, cy, w, h, conf, cid])
    return out


def _read_gt(path: Path):
    """读 GT label，返回 [[cx,cy,w,h,cid], ...]（归一化）。"""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cid, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
        out.append([cx, cy, w, h, cid])
    return out


def _iou(a, b):
    """两框 [cx,cy,w,h] 的 IoU。"""
    ax0, ay0 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax1, ay1 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx1, by1 = b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _match_iou(rect_preds, square_preds, thresh=0.5):
    """贪心按 IoU 匹配（跨两套预测），返回 (matched, rect_only, square_only)。"""
    matched = 0
    used = set()
    for r in rect_preds:
        best_j, best_i = -1, 0.0
        for j, s in enumerate(square_preds):
            if j in used:
                continue
            i = _iou(r, s)
            if i > best_i:
                best_i, best_j = i, j
        if best_j >= 0 and best_i >= thresh:
            used.add(best_j)
            matched += 1
    return matched, len(rect_preds) - matched, len(square_preds) - matched


def _categorize(stem, img_path, depth_path, gt):
    """返回该图命中的维度标签列表。"""
    tags = []
    im = cv2.imread(str(img_path))
    h, w = im.shape[:2]
    if max(h, w) < 800:
        tags.append("小图")
    else:
        tags.append("大图")
    ar = h / w
    if ar < 0.9:
        tags.append("宽图")
    elif ar > 1.1:
        tags.append("高图")
    n = len(gt)
    if n >= DENSE_N:
        tags.append("密集")
    if n <= SPARSE_N:
        tags.append("稀疏")
    has_small = any((g[2] * g[3]) < SMALL_OBJ_AREA for g in gt)
    if has_small:
        tags.append("小目标")
    d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if d is not None:
        zero_ratio = float((d == 0).mean())
        if zero_ratio > INVALID_DEPTH_ZERO:
            tags.append("无效Depth")
    return tags, h, w, n, has_small


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rect_dir", required=True)
    ap.add_argument("--square_dir", required=True)
    ap.add_argument("--val_images", required=True)
    ap.add_argument("--val_labels", required=True)
    ap.add_argument("--val_depth", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--min_per_tag", type=int, default=3,
                    help="每个维度至少选 min_per_tag 张（不足则全选）")
    args = ap.parse_args()

    rect_dir = Path(args.rect_dir)
    square_dir = Path(args.square_dir)
    val_images = Path(args.val_images)
    val_labels = Path(args.val_labels)
    val_depth = Path(args.val_depth)

    # 收集全量图片的维度标签
    rows = []
    for img_path in sorted(glob.glob(str(val_images / "*.jpg")) + glob.glob(str(val_images / "*.png"))):
        img_path = Path(img_path)
        stem = img_path.stem
        gt = _read_gt(val_labels / (stem + ".txt"))
        depth_path = val_depth / (stem + img_path.suffix)
        tags, h, w, n, has_small = _categorize(stem, img_path, depth_path, gt)
        rows.append({"stem": stem, "tags": tags, "h": h, "w": w, "n_gt": n})

    # 每个维度至少选 min_per_tag 张（尽量不重复选同一张）
    all_tags = ["小图", "大图", "宽图", "高图", "密集", "稀疏", "无效Depth", "小目标"]
    selected = {}  # stem -> row
    for tag in all_tags:
        candidates = [r for r in rows if tag in r["tags"]]
        if not candidates:
            print(f"[compare] 维度 '{tag}' 在 val 集无样本（跳过）")
            continue
        picked = 0
        for r in candidates:
            if picked >= args.min_per_tag:
                break
            if r["stem"] not in selected:
                selected[r["stem"]] = r
                picked += 1
        print(f"[compare] 维度 '{tag}': 命中 {len(candidates)} 张，选中 {picked} 张")

    # 逐图对比
    out_rows = []
    for stem, r in sorted(selected.items()):
        rect = _read_preds(rect_dir / (stem + ".txt"))
        square = _read_preds(square_dir / (stem + ".txt"))
        matched, rect_only, square_only = _match_iou(rect, square)
        out_rows.append({
            "stem": stem,
            "tags": ";".join(r["tags"]),
            "h": r["h"], "w": r["w"],
            "n_gt": r["n_gt"],
            "rect_n": len(rect),
            "square_n": len(square),
            "delta": len(rect) - len(square),
            "matched": matched,
            "rect_only": rect_only,
            "square_only": square_only,
        })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["stem", "tags", "h", "w", "n_gt", "rect_n", "square_n", "delta",
                  "matched", "rect_only", "square_only"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=fieldnames)
        wtr.writeheader()
        wtr.writerows(out_rows)

    # 汇总
    n = len(out_rows)
    rect_sum = sum(o["rect_n"] for o in out_rows)
    square_sum = sum(o["square_n"] for o in out_rows)
    print(f"\n[compare] 共对比 {n} 张图 → {out_csv}")
    print(f"[compare] rect 总框数={rect_sum}, square 总框数={square_sum}, 差={rect_sum - square_sum}")
    if n:
        avg = (rect_sum - square_sum) / n
        print(f"[compare] 平均每图 delta(rect-square)={avg:.2f}")


if __name__ == "__main__":
    main()
