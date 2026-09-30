"""_small_root_cause.py — Gate 0 Part A：small-object 错误类型只读根因审计。

只读：仅读取已有 val 预测 TXT / GT labels / 图像头；只写 diagnostic/ 下的 json。
不训练、不推理、不改任何 checkpoint / 配置 / evaluator / 预测文件。

口径：
  - small / medium / large 一律用 **native 像素面积**（COCO：small<32²=1024，medium<96²=9216）
  - IoU / 读取 / cap=100 全部**直接 import scripts/official_eval.py**，不重新实现
  - GT identity = (image_stem, gt_index_within_label_file) —— 三个模型共用同一 val 集，
    标签行序一致，因此该对应关系严格可用（不依赖 prediction index）
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
    MAX_BOXES_PER_IMAGE,
    apply_max_boxes,
    box_iou_np,
    load_split,
    norm_xywh_to_xyxy,
    read_pred_txt,
)

SPLIT = ROOT / "data/processed/rgbid_split_train"
IMAGES = SPLIT / "images/val/visible"
LABELS = SPLIT / "labels/val/visible"
OUT = Path(__file__).resolve().parent

MODELS = {
    "Dp":     ROOT / "diagnostic/sepstem_clahe/best_full/results",
    "OASA20": ROOT / "diagnostic/oasa_scale14/oasa20_best/results",
    "OASA14": ROOT / "diagnostic/oasa_scale14/oasa14_best/results",
}
NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}

SMALL_A, MEDIUM_A = 1024.0, 9216.0
LOW_CONF = 0.10          # 诊断阈值（非官方评分定义）
IOU_GRID = (0.10, 0.25, 0.50, 0.75)


def bucket(a):
    return "small" if a < SMALL_A else ("medium" if a < MEDIUM_A else "large")


def audit_model(results_dir, per_image):
    """返回每条 GT 的记录（跨全部尺度）。"""
    recs = []
    for img_id, stem, w, h, gt_b, gt_c in per_image:
        _, pred, _ = read_pred_txt(Path(results_dir) / f"{stem}.txt")
        pred = apply_max_boxes(pred, MAX_BOXES_PER_IMAGE)
        if len(pred):
            pbox, pcls = norm_xywh_to_xyxy(pred, w, h)
            pconf = pred[:, 5].astype(np.float32)
        else:
            pbox = np.zeros((0, 4), np.float32)
            pcls = np.zeros((0,), int)
            pconf = np.zeros((0,), np.float32)

        if gt_b is None or len(gt_b) == 0:
            continue
        gt_area = (np.clip(gt_b[:, 2] - gt_b[:, 0], 0, None)
                   * np.clip(gt_b[:, 3] - gt_b[:, 1], 0, None))

        for j in range(len(gt_b)):
            r = dict(stem=stem, gt_idx=j, cls=int(gt_c[j]), area=float(gt_area[j]),
                     bucket=bucket(float(gt_area[j])))
            same = np.where(pcls == gt_c[j])[0] if len(pcls) else np.zeros(0, int)
            iou_s = box_iou_np(gt_b[j:j + 1], pbox[same])[0] if len(same) else np.zeros(0, np.float32)
            if len(iou_s):
                k = int(np.argmax(iou_s))
                r["best_iou_same"] = float(iou_s[k])
                r["best_conf"] = float(pconf[same[k]])
                pb = pbox[same[k]]
                pw = float(max(pb[2] - pb[0], 0)); ph = float(max(pb[3] - pb[1], 0))
                gw = float(max(gt_b[j, 2] - gt_b[j, 0], 0)); gh = float(max(gt_b[j, 3] - gt_b[j, 1], 0))
                r["area_ratio"] = float((pw * ph) / (gw * gh + 1e-9))
                r["w_ratio"] = float(pw / (gw + 1e-9))
                r["h_ratio"] = float(ph / (gh + 1e-9))
                gcx, gcy = (gt_b[j, 0] + gt_b[j, 2]) / 2, (gt_b[j, 1] + gt_b[j, 3]) / 2
                pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                r["cdx"] = float(abs(pcx - gcx) / (gw + 1e-9))
                r["cdy"] = float(abs(pcy - gcy) / (gh + 1e-9))
            else:
                r.update(best_iou_same=0.0, best_conf=0.0, area_ratio=None,
                         w_ratio=None, h_ratio=None, cdx=None, cdy=None)
            r["n_pred_same"] = int(len(same))
            iou_a = box_iou_np(gt_b[j:j + 1], pbox)[0] if len(pbox) else np.zeros(0, np.float32)
            if len(iou_a):
                ka = int(np.argmax(iou_a))
                r["best_iou_any"] = float(iou_a[ka])
                r["best_any_cls"] = int(pcls[ka])
            else:
                r["best_iou_any"] = 0.0
                r["best_any_cls"] = None
            recs.append(r)
    return recs


def taxonomy(r):
    """互斥分类。返回 (主要类型, 子类)。"""
    iou_s, conf = r["best_iou_same"], r["best_conf"]
    if iou_s >= 0.5:
        return ("OK", "") if conf >= LOW_CONF else ("D", "good_loc_low_conf")
    if iou_s >= 0.1:
        ar, cdx, cdy = r["area_ratio"], r["cdx"], r["cdy"]
        sub = "L_other"
        if ar is not None:
            if ar > 2.0:
                sub = "L1_oversized"
            elif ar < 0.5:
                sub = "L2_undersized"
            elif (cdx or 0) > 0.5 or (cdy or 0) > 0.5:
                sub = "L3_center_offset"
            elif r["w_ratio"] and r["h_ratio"] and (max(r["w_ratio"] / (r["h_ratio"] + 1e-9),
                                                      r["h_ratio"] / (r["w_ratio"] + 1e-9)) > 2.0):
                sub = "L4_aspect_mismatch"
        return "L", sub
    if r["best_iou_any"] >= 0.5 and r["best_any_cls"] != r["cls"]:
        return "C", "misclassified"
    if r["n_pred_same"] > 0:
        return "H2", "same_class_exists_iou_lt_0.1"
    if r["best_iou_any"] >= 0.1:
        return "H4", "only_other_class_weak_overlap"
    return "H1", "no_candidate_at_all"


def hard_miss(r):
    return taxonomy(r)[0] in ("H1", "H2", "H4")


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else float("nan")


def pct(v, q):
    v = [x for x in v if x is not None]
    return float(np.percentile(v, q)) if v else float("nan")


def main():
    per_image, stats = load_split(IMAGES, LABELS)
    print("=" * 78)
    print("Gate 0 Part A — small-object 错误类型根因审计（只读）")
    print("=" * 78)
    print(f"val: images={stats['images']} corrupt={stats['corrupt']} GT={stats['gt']}")
    print(f"cap MAX_BOXES_PER_IMAGE={MAX_BOXES_PER_IMAGE}  LOW_CONF 阈值={LOW_CONF}（诊断用）")

    all_rec = {}
    for name, rd in MODELS.items():
        if not Path(rd).exists():
            print(f"!! {name} 预测目录不存在: {rd}")
            continue
        all_rec[name] = audit_model(rd, per_image)
        print(f"  {name}: {len(all_rec[name])} 条 GT 记录")

    res = {"split": dict(stats), "models": {}}

    # ---------- A2/A3/A5 ----------
    for name, recs in all_rec.items():
        small = [r for r in recs if r["bucket"] == "small"]
        m = {}
        m["n_gt"] = len(recs)
        m["n_small"] = len(small)
        m["iou_ge"] = {str(t): float(np.mean([r["best_iou_same"] >= t for r in small]))
                       for t in IOU_GRID}
        tax = Counter(taxonomy(r)[0] for r in small)
        m["taxonomy_small"] = {k: dict(count=v, frac=v / max(1, len(small)))
                               for k, v in sorted(tax.items())}
        sub = Counter(taxonomy(r)[1] for r in small if taxonomy(r)[0] in ("L", "H2", "C"))
        m["taxonomy_small_sub"] = dict(sub)
        for b in ("small", "medium", "large"):
            rb = [r for r in recs if r["bucket"] == b]
            m[f"scale_{b}"] = dict(
                n=len(rb),
                hard_miss_frac=float(np.mean([hard_miss(r) for r in rb])) if rb else float("nan"),
                best_iou_median=med([r["best_iou_same"] for r in rb]),
            )
        m["class_small"] = {}
        for c in sorted(NAMES):
            rc = [r for r in small if r["cls"] == c]
            if not rc:
                continue
            tx = Counter(taxonomy(r)[0] for r in rc)
            m["class_small"][NAMES[c]] = dict(
                gt=len(rc),
                hard_miss=tx.get("H1", 0) + tx.get("H2", 0) + tx.get("H4", 0),
                localization=tx.get("L", 0), classification=tx.get("C", 0),
                low_conf=tx.get("D", 0), ok=tx.get("OK", 0),
                best_iou_median=med([r["best_iou_same"] for r in rc]),
                best_iou_p25=pct([r["best_iou_same"] for r in rc], 25),
                best_iou_p75=pct([r["best_iou_same"] for r in rc], 75),
                area_ratio_median=med([r["area_ratio"] for r in rc]),
                center_offset_median=med([r["cdx"] for r in rc]),
                conf_median=med([r["best_conf"] for r in rc if r["best_conf"] > 0]),
            )
        res["models"][name] = m

    # ---------- A4: common hard miss ----------
    sets = {}
    keys = {}
    for name, recs in all_rec.items():
        sets[name] = {(r["stem"], r["gt_idx"]) for r in recs if r["bucket"] == "small" and hard_miss(r)}
        keys[name] = {(r["stem"], r["gt_idx"]) for r in recs if r["bucket"] == "small"}
    common_keys = set.intersection(*keys.values()) if keys else set()
    a4 = {"n_small_gt_common": len(common_keys)}
    for name in sets:
        hs = sets[name] & common_keys
        a4[f"{name}_hard_miss"] = len(hs)
        a4[f"{name}_hard_miss_frac"] = len(hs) / max(1, len(common_keys))
    if len(sets) == 3:
        n = [sets[k] & common_keys for k in ("Dp", "OASA20", "OASA14")]
        inter = n[0] & n[1] & n[2]
        union = n[0] | n[1] | n[2]
        a4["union"] = len(union)
        a4["intersection_3way"] = len(inter)
        a4["3way_common_miss_rate"] = len(inter) / max(1, len(union))
        a4["pairwise"] = {}
        for i, x in enumerate(("Dp", "OASA20", "OASA14")):
            for j, y in enumerate(("Dp", "OASA20", "OASA14")):
                if i < j:
                    a4["pairwise"][f"{x}&{y}"] = len((sets[x] & common_keys) & (sets[y] & common_keys))
    res["common_hard_miss"] = a4

    (OUT / "_small_root_cause.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[saved] {OUT / '_small_root_cause.json'}")

    # ---------- 打印 ----------
    for name, m in res["models"].items():
        print(f"\n===== {name} =====")
        print(f"  GT 总数 {m['n_gt']}  small {m['n_small']}")
        print("  small best same-class IoU ≥ " +
              "  ".join(f"{t}:{m['iou_ge'][str(t)]:.1%}" for t in IOU_GRID))
        print("  taxonomy:", {k: f"{v['count']} ({v['frac']:.1%})"
                              for k, v in m["taxonomy_small"].items()})
        print("  子类:", m["taxonomy_small_sub"])
        print("  尺度 hard-miss:", {b: f"{m['scale_'+b]['hard_miss_frac']:.1%}"
                                     for b in ("small", "medium", "large")})
    print("\n===== A4 common hard miss =====")
    for k, v in res["common_hard_miss"].items():
        print(f"  {k}: {v}")
    return res


if __name__ == "__main__":
    main()
