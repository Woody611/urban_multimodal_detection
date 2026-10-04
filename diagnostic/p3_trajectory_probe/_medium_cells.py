"""_medium_cells.py — 冻结「全部 medium val GT」的 cells + 角色标签（零 forward）。

角色标签**直接复用**上一轮 audit 的 per_gt_representation_audit.csv（含 in_86 / detected），
以保证与 audit **逐位一致**。

================================================================================
2026-10-02 坐标口径修复（详见 diagnostic/full300_trajectory_probe/EMA_PREFLIGHT_P1_P5.md §5）
================================================================================
旧版坐标：`cx, cy = 标签文件归一化中心 × 1280`。
  标签归一化中心是相对**原图**的，而 16:9 原图进 1280² 方形画布要 letterbox：
      canvas_y = cy_norm × H × gain + pad_y
  旧式漏掉 pad_y（1920×1080 / 640×360 ⇒ pad_y = 280）⇒ 采到**错误的网格 cell**
  （实测 |Δy| median 76.74 px = 9.6 个 stride-8 cell，max 274.82 px = 34.4 cell，1320 个 key 中只有 1 个恰好对）。

修复：改用 **dataset 的 `lab["bboxes"]`** —— ultralytics 已把 bbox 归一化到 **letterbox 后的画布**，
因此 `× 1280` 才是正确口径（与 `_v7_extract.py:128-129` 同源）。

⚠ 输出文件名改为 `medium_cells_canvas.json`，**不覆盖**被 5+ 个既有 diagnostic 目录按 SHA 引用的
   `medium_cells.json`（frozen legacy artifact）。
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = Path(__file__).resolve().parent
AUD = ROOT / "diagnostic/cls_representation_modality_audit/per_gt_representation_audit.csv"
V7 = ROOT / "diagnostic/p3_feature_space/_v7_vectors.json"
CANVAS_OUT = OUT / "medium_cells_canvas.json"     # ★ 新文件；legacy medium_cells.json 不动
IMGSZ = 1280
STRIDE8 = IMGSZ / 160.0                            # P3 特征图 160×160 ⇒ 8 px/cell

import csv  # noqa: E402

import numpy as np  # noqa: E402

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.utils import yaml_load  # noqa: E402

recs = list(csv.DictReader(open(AUD, encoding="utf-8")))
assert len(recs) == 1320, len(recs)


def build_val_ds():
    """与 `_v7_extract.py::build_ds(train_images=False, augment=False)` 逐项同参。"""
    data = yaml_load(str(ROOT / "data/processed/rgbid_split_train/dataset.yaml"))
    ov = dict(imgsz=IMGSZ, task="detect", rect=False, cache=False, single_cls=False, classes=None,
              fraction=1.0, channels=5, use_simotm="RGBID", object_scale_aug=False,
              mosaic=0.0, mixup=0.0, copy_paste=0.0,
              degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
              flipud=0.0, fliplr=0.5)
    ov["augment"] = False
    return build_yolo_dataset(get_cfg(overrides=ov),
                              str((Path(data["path"]) / data["val"]).resolve()), 8, data,
                              mode="val", use_simotm="RGBID",
                              pairs_rgb_ir=["visible", "infrared"],
                              pairs_rgb_depth=["visible", "depth"])


def canvas_cells_from_dataset():
    """key=f'{stem}#{j}' -> (cx_canvas, cy_canvas)，j 为 dataset bbox 下标（与 _v7_extract 同源）。"""
    ds = build_val_ds()
    out = {}
    for i in range(len(ds)):
        lab = ds[i]
        bb = lab.get("bboxes")
        if bb is None or not len(bb):
            continue
        stem = Path(ds.im_files[i]).stem
        arr = np.asarray(bb, dtype=np.float64)
        for j in range(len(arr)):
            out[f"{stem}#{j}"] = (float(arr[j][0]) * IMGSZ, float(arr[j][1]) * IMGSZ)
    return out, len(ds)


def main():
    cells, n_ds = canvas_cells_from_dataset()
    print(f"[coord] val dataset images = {n_ds}；canvas cells = {len(cells)}")

    out = []
    missing = []
    for r in recs:
        k = f"{r['image_stem']}#{r['gt_id']}"
        if k not in cells:
            missing.append(k)
            continue
        cx, cy = cells[k]
        in86 = int(float(r["in_86"]))
        det = int(float(r["detected"]))
        out.append(dict(key=k, image_stem=r["image_stem"], gt_id=int(r["gt_id"]),
                        class_id=int(float(r["gt_class"])),
                        p3_cell_x=int(min(max(cx / STRIDE8, 0), 159)),
                        p3_cell_y=int(min(max(cy / STRIDE8, 0), 159)),
                        x_center_canvas=round(cx, 4), y_center_canvas=round(cy, 4),
                        native_area=float(r["native_area"]), in_86=in86, frozen_detected=det,
                        frozen_best_same_iou=float(r["frozen_best_same_iou"]),
                        role=("G86" if in86 else ("CONTROL" if det else "MISSED_NON86"))))
    if missing:
        raise RuntimeError(f"cell 缺失 {len(missing)} 个，例：{missing[:5]}")

    # ---------- HARD REGRESSION vs 冻结参考 _v7_vectors.json (V1) ----------
    v1 = {r["key"]: r for r in json.loads(V7.read_text(encoding="utf-8"))["rows"]
          if r.get("domain") == "V1"}
    # 参考文件里的 center 是 round(x, 2)（_v7_extract.py:138）⇒ 逐位等价即 |Δ| <= 0.005
    TOL = 0.005 + 1e-9
    n_match = n_seen = 0
    max_dx = max_dy = 0.0
    for row in out:
        ref = v1.get(row["key"])
        if ref is None:
            continue
        n_seen += 1
        dx = abs(ref["center"][0] - row["x_center_canvas"])
        dy = abs(ref["center"][1] - row["y_center_canvas"])
        max_dx, max_dy = max(max_dx, dx), max(max_dy, dy)
        if dx <= TOL and dy <= TOL:
            n_match += 1
    print(f"[regression] 与 _v7_vectors.json V1 比对: {n_match}/{n_seen} 坐标逐位等价（tol=0.005，"
          f"参考为 2 位小数）  max|Δx|={max_dx:.6f}  max|Δy|={max_dy:.6f}")
    assert n_seen >= 1320 - 5, f"参考覆盖不足: {n_seen}"
    assert n_match == n_seen, f"坐标未逐位复现: {n_match}/{n_seen}"

    n86 = sum(r["role"] == "G86" for r in out)
    nC = sum(r["role"] == "CONTROL" for r in out)
    nM = sum(r["role"] == "MISSED_NON86" for r in out)
    meta = dict(n=len(out), n_g86=n86, n_control=nC, n_missed_non86=nM,
                role_source=str(AUD.relative_to(ROOT)),
                role_source_sha256=hashlib.sha256(AUD.read_bytes()).hexdigest(),
                metric_definition=("nearest-centroid top-1 class accuracy; centroids estimated from the "
                                   "CONTROL set only (role=='CONTROL'); identical definition to the prior audit"),
                coordinate_source="dataset_lab_bboxes_canvas_normalized",
                cell_mapping=("canvas = dataset lab['bboxes'] × 1280（letterbox 后坐标，含 pad）；"
                              "P3 160×160 ⇒ cell = floor(c/8)"),
                supersedes="medium_cells.json（其标签归一化 ×1280 口径漏 letterbox pad，已作废）",
                legacy_regression=dict(reference=str(V7.relative_to(ROOT)),
                                       v1_rows_seen=n_seen, exact_matches=n_match,
                                       max_abs_dx=max_dx, max_abs_dy=max_dy),
                n_images=len({r["image_stem"] for r in out}),
                rows=out)
    CANVAS_OUT.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"medium cells = {len(out)}   G86={n86}  CONTROL={nC}  MISSED_NON86={nM}")
    print(f"  涉及 {meta['n_images']} 张 val 图")
    print(f"  角色标签来源 SHA = {meta['role_source_sha256'][:16]}（与上一轮 audit 同一文件）")
    print(f"[saved] {CANVAS_OUT}")


if __name__ == "__main__":
    main()
