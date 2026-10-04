"""gt_level_attribution.py — D′ vs M1 的 GT-level attribution（ZERO GPU / 不重新推理）。

严格镜像 `scripts/official_eval.py::pr_curve_for_class` 的匹配语义：
  per-class **全局** confidence 降序、贪心 one-to-one、与"未被占用"的 GT 比 IoU，
  取 max-IoU 未占用 GT，IoU >= t 记 TP 并占用该 GT。**不发明新规则。**

用法:
  python -X utf8 diagnostic/gt_level_attribution.py \
      --dprime diagnostic/sepstem_clahe/best_full/results \
      --m1 diagnostic/batch1_eval/m1_best/results \
      --out diagnostic/batch1_eval/_gt_attribution.json [--modality-phenotype]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from official_eval import (  # noqa: E402
    EPS, IOU_THRESHOLDS, MAX_BOXES_PER_IMAGE, NC, apply_max_boxes, box_iou_np,
    evaluate, load_split, norm_xywh_to_xyxy, read_pred_txt,
)

IMAGES = ROOT / "data/processed/rgbid_split_train/images/val/visible"
LABELS = ROOT / "data/processed/rgbid_split_train/labels/val/visible"
NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}
SIZE_BINS = (("small", 0.0, 1024.0), ("medium", 1024.0, 9216.0), ("large", 9216.0, 1e18))
IOU_EDGES = [0.50, 0.60, 0.70, 0.80, 0.90, 1.0001]


def size_of(a):
    for nm, lo, hi in SIZE_BINS:
        if lo <= a < hi:
            return nm
    return "large"


# ------------------------------------------------------------------ 核心
def collect(per_image, results_dir, thr=0.50):
    """返回 {gt_key: 该 GT 在该模型下的状态}，完全按 official_eval 的贪心匹配。"""
    results_dir = Path(results_dir)
    # 1) 按类收集预测（与 evaluate 完全一致：整图 cap=100 → xyxy → 全局 conf 降序）
    preds_by_cls = {c: [] for c in range(NC)}
    for (img_id, stem, w, h, gb, gc) in per_image:
        n_total, pred, _ = read_pred_txt(results_dir / f"{stem}.txt")
        if len(pred) == 0:
            continue
        pred = apply_max_boxes(pred, MAX_BOXES_PER_IMAGE)
        if len(pred) == 0:
            continue
        boxes, cls = norm_xywh_to_xyxy(pred, w, h)
        conf = pred[:, 5]
        for i in range(len(boxes)):
            preds_by_cls[int(cls[i])].append((img_id, float(conf[i]), boxes[i]))

    # 2) GT by class
    gt_by_cls = {c: {} for c in range(NC)}
    for (img_id, stem, w, h, gb, gc) in per_image:
        if gb is None or len(gb) == 0:
            continue
        for c in np.unique(gc):
            m = gc == c
            gt_by_cls[int(c)][img_id] = gb[m]

    # 3) 贪心匹配（镜像 pr_curve_for_class），记录每个 GT 的命中信息
    state = {}
    for c in range(NC):
        gbi = gt_by_cls[c]
        for iid, arr in gbi.items():
            for j in range(len(arr)):
                state[(iid, c, j)] = dict(matched=False, matched_iou=None,
                                          matched_conf=None, best_iou=0.0, best_conf=0.0,
                                          n_cand=0)
        if not preds_by_cls[c] or not gbi:
            continue
        preds = sorted(preds_by_cls[c], key=lambda z: -z[1])
        used = {iid: np.zeros(len(v), bool) for iid, v in gbi.items()}
        for (img_id, cf, box) in preds:
            gt = gbi.get(img_id)
            if gt is None or len(gt) == 0:
                continue
            ious = box_iou_np(box[None, :], gt)[0]
            # best same-class candidate（未占用，不受匹配结果影响）
            for j in range(len(gt)):
                st = state[(img_id, c, j)]
                st["n_cand"] += 1
                if ious[j] > st["best_iou"]:
                    st["best_iou"] = float(ious[j])
                if cf > st["best_conf"]:
                    st["best_conf"] = float(cf)
            ious_used = ious.copy()
            ious_used[used[img_id]] = -1
            j = int(np.argmax(ious_used))
            if ious_used[j] >= thr:
                used[img_id][j] = True
                st = state[(img_id, c, j)]
                if not st["matched"]:
                    st["matched"] = True
                    st["matched_iou"] = float(ious_used[j])
                    st["matched_conf"] = float(cf)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dprime", default="diagnostic/sepstem_clahe/best_full/results")
    ap.add_argument("--m1", default="diagnostic/batch1_eval/m1_best/results")
    ap.add_argument("--out", default="diagnostic/batch1_eval/_gt_attribution.json")
    ap.add_argument("--modality-phenotype", action="store_true")
    A = ap.parse_args()
    Dp, M1 = (ROOT / A.dprime).resolve(), (ROOT / A.m1).resolve()

    # ================= Step 0 硬门 =================
    print("=" * 104)
    print("Step 0 — 数据锁定与硬门")
    print("=" * 104)
    per_image, stats = load_split(IMAGES, LABELS)
    stems_gt = {s for (_i, s, _w, _h, _b, _c) in per_image}
    sd = {p.stem for p in Dp.glob("*.txt")}
    sm = {p.stem for p in M1.glob("*.txt")}
    print(f"  GT        : images={stats['images']} corrupt={stats['corrupt']} 有效图={len(per_image)} GT={stats['gt']}")
    print(f"  D′ preds  : {len(sd)} TXT   ({Dp})")
    print(f"  M1 preds  : {len(sm)} TXT   ({M1})")
    # 注：preds 目录含 **corrupt 图** 的 TXT（评测器 drop_corrupt 跳过它们），
    # 因此正确谓词是「有效 GT 图 ⊆ 两侧 preds」+「两侧多出的 stem 相同」，
    # 而不是「preds 集合 == GT 集合」。
    extra_d, extra_m = sd - stems_gt, sm - stems_gt
    gates = [
        ("D′ 与 M1 的 image 集合完全一致", sd == sm),
        ("全部有效 GT 图都被 D′ 覆盖", not (stems_gt - sd)),
        ("全部有效 GT 图都被 M1 覆盖", not (stems_gt - sm)),
        ("两侧多出的 stem 相同（应为被 skip 的 corrupt 图）", extra_d == extra_m),
        (f"多出的是 corrupt 图（n={len(extra_d)}，GT 有效图 {len(stems_gt)}）", len(extra_d) <= 4),
    ]
    e_d = evaluate(per_image, Dp, max_boxes=MAX_BOXES_PER_IMAGE)
    e_m = evaluate(per_image, M1, max_boxes=MAX_BOXES_PER_IMAGE)
    gates.append(("D′ 复现 0.51528", abs(e_d["mAP50-95"] - 0.51528) < 5e-5))
    gates.append(("M1 复现（本机 m1_best）", abs(e_m["mAP50-95"] - 0.49951) < 5e-5))
    for nm, ok in gates:
        print(f"    {'✅' if ok else '❌'} {nm}")
    print(f"  D′ official mAP50-95 = {e_d['mAP50-95']:.5f}   M1 = {e_m['mAP50-95']:.5f}   Δ(D′−M1) = {e_d['mAP50-95']-e_m['mAP50-95']:+.5f}")
    # class id / 坐标约定：两侧走**同一个** load_split + norm_xywh_to_xyxy ⇒ 构造上一致
    print("  class id / 坐标约定：两侧均用同一个 load_split + norm_xywh_to_xyxy ⇒ 构造上一致")
    if not all(ok for _, ok in gates):
        print("\n❌ 硬门未过 —— STOP，不做 attribution")
        return 1

    SD = collect(per_image, Dp)
    SM = collect(per_image, M1)
    keys = sorted(SD.keys())
    assert set(keys) == set(SM.keys()), "GT key 集合不一致"
    print(f"  GT-level keys = {len(keys)}  (== GT 总数 {stats['gt']} ? {len(keys)==stats['gt']})")

    # ================= Step 1/2  outcome =================
    rows = []
    for (iid, c, j) in keys:
        d, m = SD[(iid, c, j)], SM[(iid, c, j)]
        if d["matched"] and m["matched"]:
            o = "BOTH_CORRECT"
        elif d["matched"] and not m["matched"]:
            o = "Dprime_ONLY"
        elif (not d["matched"]) and m["matched"]:
            o = "M1_ONLY"
        else:
            o = "BOTH_WRONG"
        rows.append(dict(img_id=int(iid), cls=int(c), gt_idx=int(j), outcome=o,
                         **{f"d_{k}": v for k, v in d.items()},
                         **{f"m_{k}": v for k, v in m.items()}))
    cnt = Counter(r["outcome"] for r in rows)
    tot = len(rows)
    print("\n" + "=" * 104)
    print("Step 2/3 — GT outcome（IoU=0.50，官方贪心匹配）")
    print("=" * 104)
    print(f"  {'outcome':<16}{'count':>7}{'% GT':>9}")
    for k in ("BOTH_CORRECT", "Dprime_ONLY", "M1_ONLY", "BOTH_WRONG"):
        print(f"  {k:<16}{cnt[k]:>7}{100*cnt[k]/tot:>8.2f}%")
    print(f"  {'TOTAL':<16}{tot:>7}{100.0:>8.2f}%")
    assert sum(cnt.values()) == tot
    net = cnt["Dprime_ONLY"] - cnt["M1_ONLY"]
    print(f"\n  D′_ONLY/total = {100*cnt['Dprime_ONLY']/tot:.2f}%   M1_ONLY/total = {100*cnt['M1_ONLY']/tot:.2f}%")
    print(f"  net rescue = {net:+d}  ({100*net/tot:+.2f}% of GT)")
    print(f"  consistency: net rescue {'>' if net>0 else '<'} 0 与 Δ(D′−M1)={e_d['mAP50-95']-e_m['mAP50-95']:+.5f} "
          f"{'同向 ✅' if (net>0) == ((e_d['mAP50-95']-e_m['mAP50-95'])>0) else '反向 ❌'}"
          f"   （⚠ rescue count 不是 mAP 分解，不可换算）")

    out = dict(meta=dict(dprime=str(Dp), m1=str(M1), n_gt=tot,
                         dprime_map=e_d["mAP50-95"], m1_map=e_m["mAP50-95"],
                         delta=e_d["mAP50-95"] - e_m["mAP50-95"]),
               outcome={k: cnt[k] for k in ("BOTH_CORRECT", "Dprime_ONLY", "M1_ONLY", "BOTH_WRONG")},
               net_rescue=net)

    # ================= Step 4  size =================
    # GT 面积：用 native 像素（与 A1/A2 的 size 桶一致）
    area = {}
    for (img_id, stem, w, h, gb, gc) in per_image:
        if gb is None:
            continue
        for c in np.unique(gc):
            idx = np.where(gc == c)[0]
            for jj, ii in enumerate(idx):
                b = gb[ii]
                area[(img_id, int(c), int(jj))] = float((b[2] - b[0]) * (b[3] - b[1]))
    for r in rows:
        r["area"] = area[(r["img_id"], r["cls"], r["gt_idx"])]
        r["size"] = size_of(r["area"])
    print("\n" + "=" * 104)
    print("Step 4 — Size attribution")
    print("=" * 104)
    print(f"  {'size':<9}{'GT':>6}{'D′_ONLY':>9}{'M1_ONLY':>9}{'BOTH_OK':>9}{'BOTH_WRONG':>12}{'rescue rate':>13}")
    sz_tab = {}
    for nm, _lo, _hi in SIZE_BINS:
        sub = [r for r in rows if r["size"] == nm]
        c2 = Counter(r["outcome"] for r in sub)
        rr = (c2["Dprime_ONLY"] - c2["M1_ONLY"]) / max(len(sub), 1)
        sz_tab[nm] = dict(gt=len(sub), d_only=c2["Dprime_ONLY"], m_only=c2["M1_ONLY"],
                          both_ok=c2["BOTH_CORRECT"], both_wrong=c2["BOTH_WRONG"], net=rr)
        print(f"  {nm:<9}{len(sub):>6}{c2['Dprime_ONLY']:>9}{c2['M1_ONLY']:>9}"
              f"{c2['BOTH_CORRECT']:>9}{c2['BOTH_WRONG']:>12}{100*rr:>12.2f}%")
    out["size"] = sz_tab

    # ================= Step 5  class =================
    print("\n" + "=" * 104)
    print("Step 5 — Class attribution（12 类全出）")
    print("=" * 104)
    print(f"  {'class':<13}{'GT':>6}{'D′_ONLY':>9}{'M1_ONLY':>9}{'net':>7}{'rescue rate':>13}")
    cls_tab = {}
    for c in range(NC):
        sub = [r for r in rows if r["cls"] == c]
        if not sub:
            cls_tab[NAMES[c]] = dict(gt=0, d_only=0, m_only=0, net=0.0); continue
        c2 = Counter(r["outcome"] for r in sub)
        rr = (c2["Dprime_ONLY"] - c2["M1_ONLY"]) / len(sub)
        cls_tab[NAMES[c]] = dict(gt=len(sub), d_only=c2["Dprime_ONLY"], m_only=c2["M1_ONLY"],
                                 net=rr, both_wrong=c2["BOTH_WRONG"])
        print(f"  {NAMES[c]:<13}{len(sub):>6}{c2['Dprime_ONLY']:>9}{c2['M1_ONLY']:>9}"
              f"{c2['Dprime_ONLY']-c2['M1_ONLY']:>+7}{100*rr:>12.2f}%")
    out["class"] = cls_tab

    # ================= Step 6  rescue type =================
    resc = [r for r in rows if r["outcome"] == "Dprime_ONLY"]
    R1 = [r for r in resc if r["m_n_cand"] > 0 and r["m_best_iou"] < 0.50]
    R2 = [r for r in resc if r["m_n_cand"] > 0 and r["m_best_iou"] >= 0.50]
    R3 = [r for r in resc if r["m_n_cand"] == 0]
    print("\n" + "=" * 104)
    print("Step 6 — D′ rescue 的类型（严格由预测内容判定，不猜）")
    print("=" * 104)
    print(f"  D′_ONLY 总数 = {len(resc)}")
    print(f"  R1 localization rescue      : M1 有同类候选但 maxIoU<0.50     → {len(R1):>4}  ({100*len(R1)/max(len(resc),1):.1f}%)")
    print(f"  R2 ranking/assignment rescue: M1 有同类候选且 maxIoU>=0.50    → {len(R2):>4}  ({100*len(R2)/max(len(resc),1):.1f}%)")
    print(f"  R3 no-candidate rescue      : M1 该图无同类候选               → {len(R3):>4}  ({100*len(R3)/max(len(resc),1):.1f}%)")
    assert len(R1) + len(R2) + len(R3) == len(resc)
    out["rescue_type"] = dict(R1_localization=len(R1), R2_ranking=len(R2), R3_no_candidate=len(R3))

    # ================= Step 7  rescue × size × IoU =================
    print("\n" + "=" * 104)
    print("Step 7 — D′_ONLY × size × D′ matched IoU")
    print("=" * 104)
    hdr = "  " + f"{'size':<9}" + "".join(f"{f'[{IOU_EDGES[i]:.2f},{IOU_EDGES[i+1]:.2f})':>14}" for i in range(5)) + f"{'total':>8}"
    print(hdr)
    mat = {}
    for nm, _lo, _hi in SIZE_BINS:
        sub = [r for r in resc if r["size"] == nm]
        cells = []
        for i in range(5):
            v = sum(1 for r in sub if IOU_EDGES[i] <= (r["d_matched_iou"] or 0) < IOU_EDGES[i + 1])
            cells.append(v)
        mat[nm] = cells
        print(f"  {nm:<9}" + "".join(f"{v:>14}" for v in cells) + f"{len(sub):>8}")
    out["rescue_size_iou"] = mat

    # ================= Step 8  phenotype 分布 =================
    print("\n" + "=" * 104)
    print("Step 8 — Multimodal rescue phenotype（四个 outcome 的分布对比）")
    print("=" * 104)

    def q(vals, p):
        return float(np.percentile(vals, p)) if len(vals) else float("nan")

    print(f"  {'outcome':<16}{'n':>6}{'area p50':>10}{'D′IoU p50':>11}{'D′conf p50':>12}{'M1IoU p50':>11}{'M1conf p50':>12}")
    phen = {}
    for k in ("BOTH_CORRECT", "Dprime_ONLY", "M1_ONLY", "BOTH_WRONG"):
        sub = [r for r in rows if r["outcome"] == k]
        a = [r["area"] for r in sub]
        di = [r["d_matched_iou"] for r in sub if r["d_matched_iou"] is not None]
        dc = [r["d_matched_conf"] for r in sub if r["d_matched_conf"] is not None]
        mi = [r["m_matched_iou"] for r in sub if r["m_matched_iou"] is not None]
        mc = [r["m_matched_conf"] for r in sub if r["m_matched_conf"] is not None]
        phen[k] = dict(n=len(sub), area_p50=q(a, 50), d_iou_p50=q(di, 50), d_conf_p50=q(dc, 50),
                       m_iou_p50=q(mi, 50), m_conf_p50=q(mc, 50))
        print(f"  {k:<16}{len(sub):>6}{q(a,50):>10.0f}{q(di,50):>11.3f}{q(dc,50):>12.3f}"
              f"{q(mi,50):>11.3f}{q(mc,50):>12.3f}")
    # D′_ONLY 内部的 M1 侧分布（未匹配，但候选仍在）
    mi_all = [r["m_best_iou"] for r in resc]
    mc_all = [r["m_best_conf"] for r in resc]
    print(f"\n  D′_ONLY 的 M1 侧 best same-class：IoU p50={q(mi_all,50):.3f}  conf p50={q(mc_all,50):.4f}  "
          f"（n_cand==0 的占 {len(R3)}/{len(resc)}）")
    out["phenotype"] = phen
    out["dprime_only_m1_best"] = dict(iou_p50=q(mi_all, 50), conf_p50=q(mc_all, 50))

    # ================= Step 9  可选 modality phenotype =================
    if A.modality_phenotype:
        print("\n" + "=" * 104)
        print("Step 9 — Modality phenotype（exploratory only；简单 CPU 图像统计，无 classifier）")
        print("=" * 104)
        try:
            import cv2
            base = ROOT / "data/processed/rgbid_split_train/images/val"
            need = defaultdict(set)
            for r in rows:
                if r["outcome"] in ("Dprime_ONLY", "BOTH_CORRECT"):
                    stem = next(s for (i, s, *_x) in per_image if i == r["img_id"])
                    need[stem].add(r["outcome"])
            gts = defaultdict(list)
            for (img_id, stem, w, h, gb, gc) in per_image:
                if gb is None:
                    continue
                for c in np.unique(gc):
                    for jj, ii in enumerate(np.where(gc == c)[0]):
                        gts[stem].append((gb[ii], int(c), int(jj)))
            stat = defaultdict(lambda: defaultdict(list))
            n_img = 0
            for stem in sorted(need):
                ip = base / "visible" / f"{stem}.jpg"
                if not ip.exists():
                    ip = base / "visible" / f"{stem}.png"
                vr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
                ir = cv2.imread(str(base / "infrared" / f"{stem}.jpg"), cv2.IMREAD_GRAYSCALE)
                dp = cv2.imread(str(base / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
                if vr is None:
                    continue
                n_img += 1
                H, W = vr.shape[:2]
                gv = cv2.cvtColor(vr, cv2.COLOR_BGR2GRAY)
                for r in rows:
                    if r["img_id"] != next(i for (i, s, *_x) in per_image if s == stem):
                        continue
                    if r["outcome"] not in ("Dprime_ONLY", "BOTH_CORRECT"):
                        continue
                    outd = r["outcome"]
                    b = None
                    for bb, cc, jidx in gts[stem]:
                        if cc == r["cls"] and jidx == r["gt_idx"]:
                            b = bb; break
                    if b is None:
                        continue
                    # ⚠ b 已经是**像素** xyxy（load_split 走 norm_xywh_to_xyxy），不要再乘 W/H
                    x1, y1, x2, y2 = [int(round(v)) for v in b]
                    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, max(x2, x1+1)), min(H, max(y2, y1+1))
                    patch = gv[y1:y2, x1:x2]
                    if patch.size < 4:
                        continue
                    stat[outd]["rgb_lum"].append(float(patch.mean()))
                    stat[outd]["rgb_std"].append(float(patch.std()))
                    stat[outd]["rgb_edge"].append(float(cv2.Laplacian(patch, cv2.CV_32F).var()))
                    if ir is not None:
                        p2 = ir[y1:y2, x1:x2].astype(np.float32)
                        stat[outd]["ir_std"].append(float(p2.std()))
                    if dp is not None:
                        p3 = dp[y1:y2, x1:x2].astype(np.float32)
                        stat[outd]["depth_valid_ratio"].append(float((p3 >= 300).mean()))
                        stat[outd]["depth_std"].append(float(p3.std()))
            print(f"  images used = {n_img}")
            print(f"  {'metric':<20}{'D′_ONLY p50':>14}{'BOTH_CORRECT p50':>20}")
            mp = {}
            for key in ("rgb_lum", "rgb_std", "rgb_edge", "ir_std", "depth_valid_ratio", "depth_std"):
                a = stat["Dprime_ONLY"][key]; b2 = stat["BOTH_CORRECT"][key]
                if not a or not b2:
                    continue
                mp[key] = dict(d_only=q(a, 50), both_ok=q(b2, 50), n_a=len(a), n_b=len(b2))
                print(f"  {key:<20}{q(a,50):>14.4f}{q(b2,50):>20.4f}   n={len(a)}/{len(b2)}")
                mp[key]["ci_a"] = [q(a, 25), q(a, 75)]
                mp[key]["ci_b"] = [q(b2, 25), q(b2, 75)]
            out["modality_phenotype"] = mp
            print("  ⚠ exploratory only（未做配对检验，n 见上）")
        except Exception as e:
            print(f"  ⚠ 跳过：{type(e).__name__}: {e}")

    # ================= Step 10 consistency =================
    print("\n" + "=" * 104)
    print("Step 10 — Consistency checks")
    print("=" * 104)
    d_only_sz = {k: sz_tab[k]["d_only"] for k in sz_tab}
    print(f"  Check A  D′_ONLY 的 size 分布 = {d_only_sz}")
    print(f"           small 占比 {100*d_only_sz['small']/max(len(resc),1):.1f}%  "
          f"medium+large {100*(d_only_sz['medium']+d_only_sz['large'])/max(len(resc),1):.1f}%")
    top_cls = sorted(((k, v["d_only"]) for k, v in cls_tab.items()), key=lambda z: -z[1])[:5]
    print(f"  Check B  D′_ONLY 最多的类 = {top_cls}")
    print(f"           与 A2 的显著类(light/sign/car/animal) 对比见上表")
    print(f"  Check C  net rescue = {net:+d}；Δ(D′−M1) = {e_d['mAP50-95']-e_m['mAP50-95']:+.5f}  同向")
    print("           ⚠ rescue count 不是 mAP decomposition，不可换算")

    Path(A.out).parent.mkdir(parents=True, exist_ok=True)
    Path(A.out).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n[report] {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
