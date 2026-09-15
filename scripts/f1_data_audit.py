"""scripts/f1_data_audit.py — F1 数据/模态对齐审计 + GT 几何统计（无需模型）。

产出:
  reports/f1_data_alignment_audit.json   # 10 项数据审计结论
  reports/f1_gt_class_stats.json         # 每类 GT 实例数/面积/小目标比例（后续拼 CSV）
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "train"
TEST = ROOT / "data" / "raw" / "test"
SPLIT = ROOT / "data" / "processed" / "depth_split_train"
REP = ROOT / "reports"
REP.mkdir(parents=True, exist_ok=True)

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
names = ["person", "boat", "animal", "seat", "sign", "bicycle",
         "car", "ball", "light", "garbage_can", "uav", "tricycle"]

def stems(d):
    return sorted(p.stem for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXT)

def dims(path):
    try:
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        return None

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

audit = {}

# ============ 1/2/8. 文件对应 + 尺寸一致 + 缺失重复 ============
raw_vis = stems(RAW / "visible")
raw_dep = stems(RAW / "depth")
raw_lbl = [p.stem for p in (RAW / "labels").iterdir() if p.suffix == ".txt"]
audit["raw_train"] = {
    "n_visible": len(raw_vis), "n_depth": len(raw_dep), "n_labels": len(raw_lbl),
    "visible_missing_depth": sorted(set(raw_vis) - set(raw_dep)),
    "depth_missing_visible": sorted(set(raw_dep) - set(raw_vis)),
    "missing_label": sorted(set(raw_vis) - set(raw_lbl)),
    "label_without_image": sorted(set(raw_lbl) - set(raw_vis)),
    "dup_visible": len(raw_vis) - len(set(raw_vis)),
    "dup_depth": len(raw_dep) - len(set(raw_dep)),
}

# 尺寸一致（raw train 全量，PIL 头读取）
dim_mismatch = []
for s in raw_vis:
    dv = dims(RAW / "visible" / f"{s}.png") or dims(RAW / "visible" / f"{s}.jpg")
    dd = dims(RAW / "depth" / f"{s}.png") or dims(RAW / "depth" / f"{s}.jpg")
    if dv is None or dd is None:
        dim_mismatch.append([s, "missing", dv, dd]); continue
    if dv != dd:
        dim_mismatch.append([s, dv, dd])
audit["raw_train_dim_mismatch"] = dim_mismatch
# 统计宽高分布（visible）
wh = {}
for s in raw_vis[:2000]:
    d = dims(RAW / "visible" / f"{s}.png") or dims(RAW / "visible" / f"{s}.jpg")
    if d:
        wh[d] = wh.get(d, 0) + 1
audit["visible_size_distribution"] = {f"{w}x{h}": c for (w, h), c in sorted(wh.items(), key=lambda x: -x[1])}

# ============ test 集 ============
test_vis = stems(TEST / "visible")
test_dep = stems(TEST / "depth")
audit["test"] = {
    "n_visible": len(test_vis), "n_depth": len(test_dep),
    "visible_missing_depth": sorted(set(test_vis) - set(test_dep)),
    "depth_missing_visible": sorted(set(test_dep) - set(test_vis)),
    "dup_visible": len(test_vis) - len(set(test_vis)),
    "dup_depth": len(test_dep) - len(set(test_dep)),
}

# ============ split 一致性 ============
for sp in ("train", "val"):
    v = stems(SPLIT / "images" / sp / "visible")
    d = stems(SPLIT / "images" / sp / "depth")
    l = [p.stem for p in (SPLIT / "labels" / sp / "visible").iterdir() if p.suffix == ".txt"]
    audit[f"split_{sp}"] = {
        "n_visible": len(v), "n_depth": len(d), "n_labels": len(l),
        "vis_depth_diff": sorted(set(v) ^ set(d)),
        "missing_label": sorted(set(v) - set(l)),
    }

# ============ 5/6/7. Depth 像素统计（采样）+ RGB 范围 ============
def depth_stats(sample_imgs):
    dtypes = {}
    invalid_ratio = []   # <300mm(16bit) 或 ==0(8bit) 视为无深度信号
    valid_vals = []
    raw_min, raw_max = [], []
    for p in sample_imgs:
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        if im.ndim == 3:
            im = im[..., 0]
        dt = str(im.dtype)
        dtypes[dt] = dtypes.get(dt, 0) + 1
        if im.dtype == np.uint16:
            invalid = (im < 300).mean()
            v = im[im >= 300]
            raw_min.append(int(im.min())); raw_max.append(int(im.max()))
        else:  # uint8 JPG：已归一化，0 为无信号
            invalid = (im == 0).mean()
            v = im[im > 0]
            raw_min.append(int(im.min())); raw_max.append(int(im.max()))
        invalid_ratio.append(float(invalid))
        if v.size:
            valid_vals.append(v.astype(np.float32))
    valid_vals = np.concatenate(valid_vals) if valid_vals else np.array([], dtype=np.float32)
    return {
        "dtype_counts": dtypes,
        "n_sampled": len(sample_imgs),
        "invalid_ratio_mean": float(np.mean(invalid_ratio)),
        "invalid_ratio_median": float(np.median(invalid_ratio)),
        "invalid_ratio_p90": float(np.percentile(invalid_ratio, 90)),
        "raw_min": min(raw_min), "raw_max": max(raw_max),
        "valid_percentiles_mm": {str(k): float(np.percentile(valid_vals, k))
                                 for k in (5, 25, 50, 75, 90, 95, 99)} if valid_vals.size else {},
    }

# 采样：val 全量 + train 前 300
val_depth = sorted((SPLIT / "images" / "val" / "depth").iterdir())
train_depth = sorted((SPLIT / "images" / "train" / "depth").iterdir())[:300]
audit["depth_stats_val"] = depth_stats(val_depth)
audit["depth_stats_train_sample"] = depth_stats(train_depth)

# RGB 像素范围（采样 100）
rgb_sample = sorted((RAW / "visible").iterdir())[:100]
rgb_ranges = []
for p in rgb_sample:
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        continue
    rgb_ranges.append([int(im.min()), int(im.max())])
audit["rgb_pixel_range_sample"] = {
    "n": len(rgb_ranges),
    "min": int(np.min([r[0] for r in rgb_ranges])),
    "max": int(np.max([r[1] for r in rgb_ranges])),
    "dtype": "uint8 BGR",
}

# ============ 9. train/val 重复 / 近重复 ============
tr = set(stems(SPLIT / "images" / "train" / "visible"))
va = set(stems(SPLIT / "images" / "val" / "visible"))
audit["split_overlap"] = {
    "train_val_stem_overlap": sorted(tr & va),
    "n_train": len(tr), "n_val": len(va),
}
# 近重复：按 stem 的公共前缀（去掉尾部序号）聚合，看是否有同前缀跨 split
def prefix(s):
    # 形如 000006_010_00000389 / shuming_1013_00000065 / 00000045
    parts = s.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else s
tr_pref = {prefix(s): s for s in tr}
va_pref = {prefix(s): s for s in va}
cross = sorted(set(tr_pref) & set(va_pref))
audit["near_dup_prefix_cross_split"] = cross
# 精确内容重复（MD5，val 全量 + train 采样）
def md5_map(imgs):
    m = {}
    for p in imgs:
        h = sha256(p)
        m.setdefault(h, []).append(p.name)
    return {h: v for h, v in m.items() if len(v) > 1}
audit["exact_dup_val"] = md5_map(sorted((SPLIT / "images" / "val" / "visible").iterdir())[:400])
audit["exact_dup_train_sample"] = md5_map(sorted((SPLIT / "images" / "train" / "visible").iterdir())[:400])

# ============ 10. 标签审计 + GT 每类面积/小目标 ============
lbl_dir = SPLIT / "labels" / "val" / "visible"  # 用 val（与评估口径一致）统计
nc = 12
instances = [0] * nc
oob = tiny = dup_box = empty = bad_cls = 0
areas = [[] for _ in range(nc)]   # 每类每实例面积（像素，用 1024 归一近似：area_norm * 1024^2）
tiny_thresh = 32 * 32             # < 32x32 px 视为小目标
for p in sorted(lbl_dir.iterdir()):
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        empty += 1
        continue
    seen = set()
    for l in lines:
        q = l.split()
        if len(q) < 5:
            bad_cls += 1; continue
        c = int(float(q[0]))
        cx, cy, bw, bh = map(float, q[1:5])
        if c < 0 or c >= nc:
            bad_cls += 1; continue
        if cx < 0 or cy < 0 or cx > 1 or cy > 1 or bw <= 0 or bh <= 0 or cx + bw/2 > 1 or cx - bw/2 < 0 or cy + bh/2 > 1 or cy - bh/2 < 0:
            oob += 1
        a = bw * bh * 1024 * 1024
        areas[c].append(a)
        instances[c] += 1
        key = (round(cx, 4), round(cy, 4), round(bw, 4), round(bh, 4), c)
        if key in seen:
            dup_box += 1
        seen.add(key)
audit["label_val_issues"] = {
    "n_files": len(list(lbl_dir.iterdir())),
    "empty_files": empty, "out_of_bounds_boxes": oob,
    "tiny_boxes(<32x32px)": tiny, "duplicate_boxes": dup_box,
    "bad_class_id": bad_cls,
    "instances_per_class": instances,
}

gt_stats = {}
for c in range(nc):
    ar = np.array(areas[c]) if areas[c] else np.array([0.0])
    small = (ar < tiny_thresh).sum() if ar.size else 0
    gt_stats[c] = {
        "class": names[c],
        "instances": int(instances[c]),
        "mean_area_px": float(ar.mean()),
        "median_area_px": float(np.median(ar)),
        "small_ratio": float(small / ar.size) if ar.size else 0.0,
    }

json.dump(audit, open(REP / "f1_data_alignment_audit.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
json.dump(gt_stats, open(REP / "f1_gt_class_stats.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# ---- 打印摘要 ----
print("=== 数据对齐审计摘要 ===")
print(f"raw train: visible={len(raw_vis)} depth={len(raw_dep)} labels={len(raw_lbl)}; "
      f"visible缺depth={len(audit['raw_train']['visible_missing_depth'])} depth缺visible={len(audit['raw_train']['depth_missing_visible'])} "
      f"缺label={len(audit['raw_train']['missing_label'])}")
print(f"raw train 尺寸不一致对数: {len(dim_mismatch)}")
print(f"visible 尺寸分布: {audit['visible_size_distribution']}")
print(f"test: visible={len(test_vis)} depth={len(test_dep)}")
print(f"split train/val stem 重叠: {len(audit['split_overlap']['train_val_stem_overlap'])}; 近重复跨split前缀: {len(cross)}")
print(f"Depth val: dtype={audit['depth_stats_val']['dtype_counts']} 无效比例mean={audit['depth_stats_val']['invalid_ratio_mean']:.3f} "
      f"有效值分位(mm)={audit['depth_stats_val']['valid_percentiles_mm']}")
print(f"Depth train样本: dtype={audit['depth_stats_train_sample']['dtype_counts']} 无效比例mean={audit['depth_stats_train_sample']['invalid_ratio_mean']:.3f}")
print(f"RGB 像素范围: {audit['rgb_pixel_range_sample']['min']}~{audit['rgb_pixel_range_sample']['max']}")
print(f"标签问题(val): empty={empty} OOB={oob} dup={dup_box} bad_cls={bad_cls}")
print("\n=== GT 每类统计 ===")
for c in range(nc):
    g = gt_stats[c]
    print(f"  {c:>2} {g['class']:<12} inst={g['instances']:>4}  mean_area={g['mean_area_px']:>9.0f}  "
          f"median_area={g['median_area_px']:>9.0f}  small_ratio={g['small_ratio']:.3f}")
print("\n已写出 reports/f1_data_alignment_audit.json 与 reports/f1_gt_class_stats.json")
