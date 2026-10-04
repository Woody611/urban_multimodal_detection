"""_g86.py — 冻结 G-86 集合与稳定的 P3 cell 坐标（只读，零 forward）。

复用已有 audit 的**同一批 86 GT**（不重筛、不改阈值）：
  diagnostic/medium_93_47_attribution/per_gt_attribution.csv 第 33 列 certified_prefinal_loss == 1

cell 映射与 diagnostic/p3_feature_space/_v7_extract.py **完全一致**：
  cx, cy = 归一化中心 × 1280（rect=False / imgsz=1280 的 letterbox 画布）
  P3 网格 = 160×160 ⇒ cell = int(c / 8)
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from official_eval import _imsize, read_gt_txt, norm_xywh_to_xyxy  # noqa: E402

OUT = Path(__file__).resolve().parent
ATT = ROOT / "diagnostic/medium_93_47_attribution/per_gt_attribution.csv"
IMAGES = ROOT / "data/processed/rgbid_split_train/images/val/visible"
LABELS = ROOT / "data/processed/rgbid_split_train/labels/val/visible"

sel = [r for r in csv.DictReader(open(ATT, encoding="utf-8")) if int(float(r["certified_prefinal_loss"])) == 1]
assert len(sel) == 86, f"G-86 数量 = {len(sel)} != 86  -> BLOCKED"

rows = []
for i, r in enumerate(sel):
    stem = r["image_stem"]
    j = int(r["gt_id"])
    ip = IMAGES / f"{stem}.png"
    if not ip.exists():
        ip = ip.with_suffix(".jpg")
    w, h = _imsize(ip)
    gt = read_gt_txt(LABELS / f"{stem}.txt")
    assert len(gt), f"no labels for {stem}"
    gb, gc = norm_xywh_to_xyxy(gt, w, h)
    cls = int(gc[j])
    assert cls == int(float(r["gt_class"])), f"class mismatch {stem}#{j}"
    # _v7_extract.py 的口径：从 dataset label 的**归一化 xywh** 出发
    gtn = gt[j]  # cls, cx, cy, bw, bh （归一化）
    cxw, cyw, ww_, hh_ = [float(v) for v in gtn[1:5]]
    cx, cy = cxw * 1280.0, cyw * 1280.0
    rows.append(dict(
        idx=i, image_id=r["image_id"], image_stem=stem, label_id=j, key=f"{stem}#{j}",
        class_id=cls, class_name=r["gt_class_name"],
        native_w=w, native_h=h,
        bbox_xyxy_native=[round(float(v), 3) for v in gb[j]],
        x_center_canvas=round(cx, 4), y_center_canvas=round(cy, 4),
        p3_cell_x=int(min(max(cx / 8.0, 0), 159)), p3_cell_y=int(min(max(cy / 8.0, 0), 159)),
        in_collapse=int(r["in_collapse"]), in_wrong_class=int(r["in_wrong_class"]),
        head_n_pos=int(float(r["head_n_pos"])), head_ciou=round(float(r["head_best_assign_iou"]), 6),
        frozen_best_any_iou=round(float(r["best_any_iou"]), 6),
    ))

# 唯一性 / 稳定性校验
keys = [r["key"] for r in rows]
assert len(set(keys)) == 86, "G-86 key 不唯一"
stems = sorted({r["image_stem"] for r in rows})
meta = dict(
    n=86,
    source=str(ATT.relative_to(ROOT)),
    source_sha256=hashlib.sha256(ATT.read_bytes()).hexdigest(),
    definition="per_gt_attribution.csv 的 certified_prefinal_loss == 1（与上一轮 representation audit 同一批）",
    cell_mapping="_v7_extract.py 口径：canvas = 归一化中心 × 1280；P3 160×160 ⇒ cell = floor(c/8)",
    n_images=len(stems),
    image_stems=stems,
    rows=rows,
)
(OUT / "g86_frozen.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"G-86 = {len(rows)}  (断言 86 通过)")
print(f"  分布 {len(stems)} 张 val 图")
from collections import Counter
print(f"  类分布: {dict(Counter(r['class_name'] for r in rows).most_common())}")
print(f"  P3 cell 范围: x [{min(r['p3_cell_x'] for r in rows)}, {max(r['p3_cell_x'] for r in rows)}]  "
      f"y [{min(r['p3_cell_y'] for r in rows)}, {max(r['p3_cell_y'] for r in rows)}]")
print(f"  key 唯一 = {len(set(keys)) == 86}")
print(f"[saved] {OUT/'g86_frozen.json'}")
