"""diagnostic/small_object_error_attribution/_attribution.py — Small-object 误差归因（只读）。

⚠️ READ-ONLY：不训练、不修改任何正式产物、不生成提交。

数据：正式 baseline 的官方预测 TXT（diagnostic/rgbid_inference_gain/RGBID_baseline/results，
      即产生 official 0.50928 的那一份）+ GT。**不重新预测**（除 §8 的 NMS 变体，另行运行）。

COCO 面积口径与既有 audit 一致：native 像素，small<1024 / medium 1024-9216 / large>9216
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "data/processed/rgbid_split"
BASE_RES = ROOT / "diagnostic/rgbid_inference_gain/RGBID_baseline/results"
IOUV = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
SMAX, MMAX = 1024, 9216
NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    bb = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def read_txt(p, n):
    if not p.exists():
        return np.zeros((0, n))
    rows = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.split()
        if len(s) >= n:
            rows.append([float(x) for x in s[:n]])
    return np.array(rows).reshape(-1, n) if rows else np.zeros((0, n))


def to_xyxy(a, w, h):
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def load(res_dir):
    """返回 per-image 记录列表。"""
    recs = []
    imgs = sorted((SPLIT / "images/val/visible").iterdir())
    for i, p in enumerate(imgs):
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        gt = read_txt(SPLIT / f"labels/val/visible/{p.stem}.txt", 5)
        if len(gt) and (gt[:, 1:].max() > 1.0 or gt[:, 1:].min() < 0.0):
            continue                                     # 与 official_map 一致：越界标签整图作废
        pr = read_txt(Path(res_dir) / f"{p.stem}.txt", 6)
        recs.append(dict(stem=p.stem, w=w, h=h,
                         gtb=to_xyxy(gt, w, h) if len(gt) else np.zeros((0, 4)),
                         gtc=gt[:, 0].astype(int) if len(gt) else np.zeros(0, int),
                         prb=to_xyxy(pr, w, h) if len(pr) else np.zeros((0, 4)),
                         prc=pr[:, 0].astype(int) if len(pr) else np.zeros(0, int),
                         prf=pr[:, 5] if len(pr) else np.zeros(0)))
    return recs


def official_tp(rec, thr):
    """官方 conf 顺序贪心匹配（逐类），返回 {img_idx: set(matched gt idx)} 与每个 pred 的 tp。"""
    out = {}
    for ri, r in enumerate(rec):
        for c in np.unique(np.concatenate([r["gtc"], r["prc"]])) if len(r["gtc"]) or len(r["prc"]) else []:
            gm = r["gtc"] == c
            pm = r["prc"] == c
            g, pb = r["gtb"][gm], r["prb"][pm]
            if len(pb) == 0:
                continue
            order = np.argsort(-r["prf"][pm], kind="stable")
            matched = np.zeros(len(g), bool)
            ii = iou_mat(pb, g)
            for k in order:
                cand = np.where(~matched)[0]
                if cand.size and ii[k, cand].max() >= thr:
                    matched[cand[int(np.argmax(ii[k, cand]))]] = True
            idxs = np.where(gm)[0]
            out[(ri, c)] = set(idxs[matched].tolist())
    return out


def main():
    rec = load(BASE_RES)
    print(f"载入 {len(rec)} 张有效 val 图")
    n_gt = sum(len(r["gtc"]) for r in rec)
    n_pr = sum(len(r["prc"]) for r in rec)
    print(f"GT={n_gt}  预测={n_pr}")

    # ---- 官方逐阈值 TP 集合 ----
    tp_sets = {t: official_tp(rec, t) for t in IOUV}

    # ---- 逐个 GT：best-match（同类 / 任意类）----
    rows = []
    for ri, r in enumerate(rec):
        g, gc = r["gtb"], r["gtc"]
        p, pc, pf = r["prb"], r["prc"], r["prf"]
        a = np.clip(g[:, 2] - g[:, 0], 0, None) * np.clip(g[:, 3] - g[:, 1], 0, None)
        for gi in range(len(g)):
            same = pc == gc[gi]
            i_any = iou_mat(g[gi:gi + 1], p)[0] if len(p) else np.zeros(0)
            i_cls = iou_mat(g[gi:gi + 1], p[same])[0] if same.any() else np.zeros(0)
            bi_any = float(i_any.max()) if len(i_any) else 0.0
            k_any = int(np.argmax(i_any)) if len(i_any) else -1
            bi_cls = float(i_cls.max()) if len(i_cls) else 0.0
            idx_cls = np.where(same)[0]
            k_cls = int(idx_cls[int(np.argmax(i_cls))]) if len(i_cls) else -1
            sz = 0 if a[gi] < SMAX else (1 if a[gi] < MMAX else 2)
            tps = {t: int(gi in tp_sets[t].get((ri, int(gc[gi])), set())) for t in IOUV}
            rows.append(dict(ri=ri, gi=gi, cls=int(gc[gi]), sz=sz, area=float(a[gi]),
                             sq=float(np.sqrt(a[gi])), w=float(g[gi, 2] - g[gi, 0]),
                             h=float(g[gi, 3] - g[gi, 1]),
                             bi_any=bi_any, bi_cls=bi_cls,
                             conf_any=float(pf[k_any]) if k_any >= 0 else np.nan,
                             conf_cls=float(pf[k_cls]) if k_cls >= 0 else np.nan,
                             pc_any=int(pc[k_any]) if k_any >= 0 else -1,
                             pc_cls=int(pc[k_cls]) if k_cls >= 0 else -1,
                             box_cls=p[k_cls].tolist() if k_cls >= 0 else None,
                             gt_box=g[gi].tolist(), tps=tps))
    json.dump(dict(n_gt=n_gt, n_pr=n_pr, n_img=len(rec), rows=rows),
              open(Path(__file__).resolve().parent / "_attr_rows.json", "w"), indent=1)
    print(f"逐 GT 记录 = {len(rows)}  ->  _attr_rows.json")
    return rows


if __name__ == "__main__":
    main()
