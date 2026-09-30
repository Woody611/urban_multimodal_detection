"""_v2_records.py — SMALL_OBJECT_CAUSE_ATTRIBUTION_V2，阶段 1/2/4/7/8/9。

只读：读 D′ 预测 + val/train GT + 图像头。只写 diagnostic/small_object_cause_v2/。
不训练、不推理、不生成新 prediction、不改 evaluator / 模型 / 配置 / checkpoint。

「完全无重叠」采用本次的严格定义：**任意类别** prediction 与该 GT 的 IoU == 0
（不是 <0.10，也不是「无同类」）。
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from official_eval import (  # noqa: E402
    MAX_BOXES_PER_IMAGE, _imsize, apply_max_boxes, box_iou_np,
    norm_xywh_to_xyxy, read_gt_txt, read_pred_txt,
)

SPLIT = ROOT / "data/processed/rgbid_split_train"
IMAGES = SPLIT / "images/val/visible"
LABELS = SPLIT / "labels/val/visible"
TR_IMAGES = SPLIT / "images/train/visible"
TR_LABELS = SPLIT / "labels/train/visible"
Dp = ROOT / "diagnostic/sepstem_clahe/best_full/results"
OUT = Path(__file__).resolve().parent

NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}
SMALL_A, MEDIUM_A = 1024.0, 9216.0
EPS0 = 1e-9
CONF_FLOOR = 0.001     # predict_rect 的 dump 下限（--conf 0.001）


def iter_split(images_dir, labels_dir, drop_corrupt=True):
    """与 official_eval.load_split 同口径，但额外返回 native 尺寸。"""
    for ip in sorted(p for p in Path(images_dir).iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}):
        w, h = _imsize(ip)
        if w <= 1 or h <= 1:
            continue
        gt = read_gt_txt(Path(labels_dir) / f"{ip.stem}.txt")
        if drop_corrupt and len(gt) and (gt[:, 1:].max() > 1.0 or gt[:, 1:].min() < 0.0):
            continue
        b, c = (norm_xywh_to_xyxy(gt, w, h) if len(gt) else (None, None))
        yield ip.stem, w, h, b, c


def main():
    # ---------- 加载 val ----------
    val = []
    for stem, w, h, b, c in iter_split(IMAGES, LABELS):
        val.append(dict(stem=stem, w=w, h=h, b=b, c=c))
    # ---------- 加载 D′ 预测 ----------
    preds = {}
    for v in val:
        _, p, _ = read_pred_txt(Dp / f"{v['stem']}.txt")
        p = apply_max_boxes(p, MAX_BOXES_PER_IMAGE)
        if len(p):
            pb, pc = norm_xywh_to_xyxy(p, v["w"], v["h"])
            preds[v["stem"]] = dict(box=pb, cls=pc, conf=p[:, 5].astype(np.float32))
        else:
            preds[v["stem"]] = dict(box=np.zeros((0, 4), np.float32),
                                    cls=np.zeros(0, int), conf=np.zeros(0, np.float32))

    # ================= 阶段 1：严格重定义 =================
    s1 = Counter()
    recs = []
    for v in val:
        if v["b"] is None or len(v["b"]) == 0:
            continue
        stem, W, H = v["stem"], v["w"], v["h"]
        P = preds[stem]
        sc = 1280.0 / max(W, H)                    # load_image 长边→1280
        area_n = np.clip(v["b"][:, 2] - v["b"][:, 0], 0, None) * np.clip(v["b"][:, 3] - v["b"][:, 1], 0, None)
        diag_n = np.sqrt(np.clip(v["b"][:, 2] - v["b"][:, 0], 0, None) ** 2
                         + np.clip(v["b"][:, 3] - v["b"][:, 1], 0, None) ** 2)
        ctr = np.stack([(v["b"][:, 0] + v["b"][:, 2]) / 2, (v["b"][:, 1] + v["b"][:, 3]) / 2], 1)

        for j in np.where(area_n < SMALL_A)[0]:
            s1["small_total"] += 1
            same = np.where(P["cls"] == v["c"][j])[0] if len(P["cls"]) else np.zeros(0, int)
            iou_s = box_iou_np(v["b"][j:j + 1], P["box"][same])[0] if len(same) else np.zeros(0, np.float32)
            iou_a = box_iou_np(v["b"][j:j + 1], P["box"])[0] if len(P["box"]) else np.zeros(0, np.float32)
            bi_s = float(iou_s.max()) if len(iou_s) else 0.0
            bi_a = float(iou_a.max()) if len(iou_a) else 0.0
            if bi_a > EPS0:
                s1["any_overlap_gt"] += 1
            else:
                s1["any_iou_zero_gt"] += 1
            if bi_s <= EPS0:
                s1["same_iou_zero"] += 1
            elif bi_s < 0.5:
                s1["same_iou_0to0.5"] += 1
            else:
                s1["same_iou_ge0.5"] += 1

            # ---- 阶段 2：完整 record ----
            def best(arr, idx):
                if not len(idx):
                    return None
                k = int(np.argmax(arr))
                return idx[k], float(arr[k])
            bs = best(iou_s, same)
            ba = best(iou_a, np.arange(len(P["box"])))
            r = dict(
                image_id=stem, gt_id=int(j), cls=int(v["c"][j]), name=NAMES[int(v["c"][j])],
                native_w=float(v["b"][j, 2] - v["b"][j, 0]), native_h=float(v["b"][j, 3] - v["b"][j, 1]),
                native_area=float(area_n[j]), native_diag=float(diag_n[j]),
                sqrt_area=float(np.sqrt(area_n[j])),
                input_w=float((v["b"][j, 2] - v["b"][j, 0]) * sc),
                input_h=float((v["b"][j, 3] - v["b"][j, 1]) * sc),
                img_w=W, img_h=H, scale=sc,
                best_same_class_iou=bi_s, best_any_class_iou=bi_a,
                n_pred_total=int(len(P["box"])), n_pred_same=int(len(same)),
            )
            if bs is not None and bi_s > EPS0:
                k = bs[0]
                pb = P["box"][k]
                r["best_same_class_conf"] = float(P["conf"][k])
                pw = float(max(pb[2] - pb[0], 0)); ph = float(max(pb[3] - pb[1], 0))
                r["best_same_class_area_ratio"] = float((pw * ph) / (area_n[j] + EPS0))
                r["center_dx_norm_gtw"] = float(abs((pb[0] + pb[2]) / 2 - ctr[j, 0]) / (r["native_w"] + EPS0))
                r["center_dy_norm_gth"] = float(abs((pb[1] + pb[3]) / 2 - ctr[j, 1]) / (r["native_h"] + EPS0))
            else:
                r.update(best_same_class_conf=0.0, best_same_class_area_ratio=None,
                         center_dx_norm_gtw=None, center_dy_norm_gth=None)
            if ba is not None and bi_a > EPS0:
                k = ba[0]
                pb = P["box"][k]
                r["best_any_class_conf"] = float(P["conf"][k])
                r["best_any_class_pred_cls"] = int(P["cls"][k])
                pw = float(max(pb[2] - pb[0], 0)); ph = float(max(pb[3] - pb[1], 0))
                r["best_any_class_area_ratio"] = float((pw * ph) / (area_n[j] + EPS0))
            else:
                r.update(best_any_class_conf=0.0, best_any_class_pred_cls=None,
                         best_any_class_area_ratio=None)

            # ---- 阶段 4：邻域 ----
            d = np.sqrt(((ctr - ctr[j]) ** 2).sum(1))
            thr = 3.0 * diag_n[j]
            nb = (d < thr)
            nb[j] = False
            nb_idx = np.where(nb)[0]
            same_nb = nb_idx[v["c"][nb_idx] == v["c"][j]]
            oth_nb = nb_idx[v["c"][nb_idx] != v["c"][j]]
            r["neighbor_gt_count"] = int(len(nb_idx))
            r["neighbor_same_class_count"] = int(len(same_nb))
            r["neighbor_other_class_count"] = int(len(oth_nb))
            r["neighbor_radius_px"] = float(thr)
            for tag, idxs in (("same", same_nb), ("other", oth_nb)):
                if len(idxs):
                    r[f"nearest_{tag}_class_distance"] = float(d[idxs].min())
                    r[f"nearest_{tag}_class_dist_norm"] = float(d[idxs].min() / (diag_n[j] + EPS0))
                else:
                    r[f"nearest_{tag}_class_distance"] = None
                    r[f"nearest_{tag}_class_dist_norm"] = None
            # 邻居是否被成功检测（同类 IoU≥0.5）
            det_ok = np.zeros(len(v["b"]), bool)
            for m in range(len(v["b"])):
                sm = np.where(P["cls"] == v["c"][m])[0] if len(P["cls"]) else np.zeros(0, int)
                if len(sm):
                    det_ok[m] = float(box_iou_np(v["b"][m:m + 1], P["box"][sm])[0].max()) >= 0.5
            r["near_same_class_detected"] = int((det_ok[same_nb]).sum()) if len(same_nb) else 0
            r["near_other_class_detected"] = int((det_ok[oth_nb]).sum()) if len(oth_nb) else 0
            r["self_detected"] = bool(det_ok[j])
            r["border_margin_norm"] = float(min(v["b"][j, 0], v["b"][j, 1],
                                                W - v["b"][j, 2], H - v["b"][j, 3]) / (diag_n[j] + EPS0))
            r["aspect"] = float(r["native_w"] / (r["native_h"] + EPS0))

            # ---- 阶段 7：弱候选（dump 下限 0.001）----
            for t in (0.001, 0.01, 0.05):
                r[f"n_pred_conf_ge_{t}"] = int((P["conf"] >= t).sum())
            r["weak_candidate_same_class"] = bool(
                len(same) and (P["conf"][same] >= CONF_FLOOR).any() and bi_s <= EPS0)
            recs.append(r)

    print("=" * 78)
    print("阶段 1 —— 严格重定义（any-class IoU == 0）")
    print("=" * 78)
    for k in ("small_total", "any_overlap_gt", "any_iou_zero_gt",
              "same_iou_zero", "same_iou_0to0.5", "same_iou_ge0.5"):
        print(f"  {k:22s} {s1[k]:5d}  ({s1[k]/max(1,s1['small_total']):6.2%})")

    zero = [r for r in recs if r["best_any_class_iou"] <= EPS0]
    print(f"\n  => 「完全无重叠」（任意类别 IoU==0）= {len(zero)}")
    print(f"  => 上一轮我报的「IoU<0.10」= {sum(1 for r in recs if r['best_any_class_iou'] < 0.10)}"
          f"   （阈值口径，非 ==0）")

    # ---- 阶段 4 归类 ----
    cat = Counter()
    for r in zero:
        if r["near_same_class_detected"] > 0:
            cat["INSTANCE_SEPARATION_CANDIDATE"] += 1
        elif r["near_other_class_detected"] > 0:
            cat["CLASS_CONFUSION_CANDIDATE"] += 1
        elif r["neighbor_gt_count"] > 0:
            cat["SCENE_DIFFICULTY_CANDIDATE"] += 1
        else:
            cat["ISOLATED_MISS"] += 1
    print("\n阶段 4 —— 邻域归类（仅对完全无重叠）")
    for k, v in cat.most_common():
        print(f"  {k:32s} {v:4d}  ({v/max(1,len(zero)):6.1%})")

    # ---- 阶段 8/9：class exposure + train/val 尺寸 ----
    tr_count, tr_small, tr_img = Counter(), Counter(), defaultdict(set)
    tr_sizes = []
    for stem, w, h, b, c in iter_split(TR_IMAGES, TR_LABELS):
        if b is None or len(b) == 0:
            continue
        a = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
        sc = 1280.0 / max(w, h)
        for j in range(len(b)):
            tr_count[int(c[j])] += 1
            if a[j] < SMALL_A:
                tr_small[int(c[j])] += 1
                tr_img[int(c[j])].add(stem)
                tr_sizes.append(float(np.sqrt(a[j]) * sc))
    val_sizes = [r["sqrt_area"] * r["scale"] for r in recs]

    def q(v, p):
        return float(np.percentile(v, p)) if len(v) else float("nan")
    stat = lambda v: dict(n=len(v), p5=q(v, 5), p10=q(v, 10), p25=q(v, 25),
                          median=q(v, 50), p75=q(v, 75), p90=q(v, 90), max=q(v, 100)) if v else {}

    print("\n阶段 9 —— train vs val small GT 输入空间 √area (px)")
    for nm, v in (("train", tr_sizes), ("val", val_sizes)):
        if v:
            print(f"  {nm:6s} n={len(v):5d}  P5={q(v,5):6.2f} P10={q(v,10):6.2f} P25={q(v,25):6.2f} "
                  f"中位={q(v,50):6.2f} P75={q(v,75):6.2f} P90={q(v,90):6.2f} max={q(v,100):6.2f}")

    print("\n阶段 8 —— class exposure（train）")
    print(f"  {'class':<14}{'train GT':>10}{'train small':>12}{'small%':>8}{'small imgs':>12}{'val small':>10}{'val miss率':>11}")
    for c in sorted(NAMES):
        vt = [r for r in recs if r["cls"] == c]
        vm = sum(1 for r in vt if r["best_any_class_iou"] <= EPS0)
        print(f"  {NAMES[c]:<14}{tr_count[c]:>10}{tr_small[c]:>12}"
              f"{tr_small[c]/max(1,tr_count[c]):>8.1%}{len(tr_img[c]):>12}{len(vt):>10}"
              f"{vm/max(1,len(vt)):>11.1%}")

    (OUT / "_v2_records.json").write_text(json.dumps(
        dict(stage1=dict(s1), records=recs, zero_ids=[(r["image_id"], r["gt_id"]) for r in zero],
             neighbor_cat=dict(cat),
             train_stats=dict(total=dict(tr_count), small=dict(tr_small),
                              small_imgs={NAMES[k]: len(v) for k, v in tr_img.items()},
                              size_stat=stat(tr_sizes)),
             val_size_stat=stat(val_sizes),
             conf_floor=CONF_FLOOR),
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {OUT/'_v2_records.json'}")
    return s1, recs, zero, cat


if __name__ == "__main__":
    main()
