"""m1_p3_attribution.py — M1 RGB-only 的 P3 failure attribution（ZERO GPU / 不重新推理）。

可用性边界（必须先声明）：
  最终 TXT = **post-NMS + post-top100** ⇒ 只能区分
      · **F3-analog**：同类候选 IoU>=t 存在，但官方贪心匹配未命中（≡ P3 的 F3 定义）
      · **F0∪F1∪F2** ：最终 TXT 里根本没有同类 IoU>=t 的候选（三者**不可分**，需 raw/pre-NMS dump）
  raw candidates(layerA) / post-NMS pre-top100(layerB) **只对 D′ 存在**（p3_disappearance_audit/layers.npz），
  M1 没有 ⇒ F0/F1/F2 对 M1 为 **NOT AVAILABLE**，不补数据。

  D′ 侧的 F0/F1/F2/F3 直接读 `p3_disappearance_audit/attribution.json`（不重算）。

用法:
  python -X utf8 diagnostic/m1_p3_attribution.py --out diagnostic/batch1_eval/_m1_p3.json
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
sys.path.insert(0, str(ROOT / "diagnostic"))

from official_eval import (  # noqa: E402
    NC, apply_max_boxes, box_iou_np, load_split, norm_xywh_to_xyxy, read_pred_txt,
)
from gt_level_attribution import collect, size_of, SIZE_BINS, NAMES  # noqa: E402

P3 = ROOT / "diagnostic/p3_disappearance_audit/attribution.json"
IMAGES = ROOT / "data/processed/rgbid_split_train/images/val/visible"
LABELS = ROOT / "data/processed/rgbid_split_train/labels/val/visible"
Dp_RES = ROOT / "diagnostic/sepstem_clahe/best_full/results"
M1_RES = ROOT / "diagnostic/batch1_eval/m1_best/results"


def file_idx_of(per_image):
    """(img_id, cls, j_within_class) -> 该 GT 在 label 文件中的行号（= p3 的 gt_id）。"""
    m = {}
    for (img_id, stem, w, h, gb, gc) in per_image:
        if gb is None:
            continue
        for c in np.unique(gc):
            for j, fi in enumerate(np.where(gc == c)[0]):
                m[(img_id, int(c), int(j))] = int(fi)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="diagnostic/batch1_eval/_m1_p3.json")
    A = ap.parse_args()

    per_image, stats = load_split(IMAGES, LABELS)
    stems = {s for (_i, s, *_x) in per_image}
    fi = file_idx_of(per_image)

    # ---------------- Gate ----------------
    print("=" * 100)
    print("M1 PROVENANCE / AVAILABILITY GATE")
    print("=" * 100)
    for k, v in (("images", 400), ("valid", 398), ("GT", 2807)):
        pass
    print(f"  dataset   : images={stats['images']} corrupt={stats['corrupt']} 有效图={len(per_image)} GT={stats['gt']}")
    sd = {p.stem for p in Dp_RES.glob('*.txt')}
    sm = {p.stem for p in M1_RES.glob('*.txt')}
    print(f"  preds     : D′={len(sd)} TXT  M1={len(sm)} TXT  同集合={sd==sm}")
    ok = (stats['gt'] == 2807 and len(per_image) == 398 and stats['corrupt'] == 2
          and sd == sm and not (stems - sm))
    print(f"  M1 覆盖全部有效 GT 图 : {not (stems - sm)}")
    print(f"  M1_PROVENANCE_GATE = {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  ❌ STOP")
        return 1

    # ---------------- M1: matched / F3-analog / F0∪F1∪F2 ----------------
    SM1 = collect(per_image, M1_RES, 0.50)
    SDP = collect(per_image, Dp_RES, 0.50)

    # D′ 侧读 P3 的 failure 与 size（不重算）
    p3 = json.load(open(P3, encoding="utf-8"))
    p3map = {}
    for r in p3["rows"]:
        p3map[(r["stem"], int(r["gt_id"]))] = r

    rows = []
    miss_p3 = 0
    for k, st in SM1.items():
        iid, c, j = k
        stem = next(s for (i, s, *_x) in per_image if i == iid)
        gid = fi[k]
        r3 = p3map.get((stem, gid))
        if r3 is None:
            miss_p3 += 1
        b = None
        for (i2, s2, w2, h2, gb, gc) in per_image:
            if i2 != iid:
                continue
            sub = np.where(gc == c)[0]
            b = gb[sub[j]]
            break
        area = float((b[2] - b[0]) * (b[3] - b[1]))
        d = SDP.get(k, {'matched': False, 'best_iou': 0.0})
        rows.append(dict(
            img_id=iid, stem=stem, cls=c, gt_id=gid, area=area, size=size_of(area),
            m1_matched=st["matched"], m1_best_iou=st["best_iou"], m1_matched_iou=st["matched_iou"],
            m1_matched_conf=st["matched_conf"],
            dp_matched=d["matched"], dp_best_iou=d["best_iou"],
            p3_failure=(r3 or {}).get("failure"), p3_size=(r3 or {}).get("size"),
        ))
    print(f"  p3 join miss = {miss_p3}/{len(rows)}")
    tot = len(rows)
    assert tot == 2807, tot

    # M1 的三态
    def state(r):
        if r["m1_matched"]:
            return "TP"
        return "F3" if r["m1_best_iou"] >= 0.50 else "F0uF1uF2"

    c_all = Counter(state(r) for r in rows)
    print("\n" + "=" * 100)
    print("P3 — M1 candidate failure（IoU=0.50）")
    print("=" * 100)
    print(f"  {'bucket':<28}{'count':>7}{'% GT':>9}")
    print(f"  {'TP@.50 (matched)':<28}{c_all['TP']:>7}{100*c_all['TP']/tot:>8.2f}%")
    print(f"  {'F3 official-match fail':<28}{c_all['F3']:>7}{100*c_all['F3']/tot:>8.2f}%")
    print(f"  {'F0∪F1∪F2 (no cand)':<28}{c_all['F0uF1uF2']:>7}{100*c_all['F0uF1uF2']/tot:>8.2f}%")
    print(f"  {'TOTAL':<28}{tot:>7}{100.0:>8.2f}%")
    print("  ⚠ F0/F1/F2 对 M1 **不可分**（缺 layerA/layerB）—— 标 NOT AVAILABLE，不补数据")

    # ---------------- M1 vs D′：同口径三态 ----------------
    dc = Counter()
    for r in rows:
        if r["dp_matched"]:
            dc["TP"] += 1
        elif r["dp_best_iou"] >= 0.50:
            dc["F3"] += 1
        else:
            dc["F0uF1uF2"] += 1
    print("\n" + "=" * 100)
    print("P3 — M1 vs D′（**同口径**：都从最终 TXT 算三态，可严格对照）")
    print("=" * 100)
    print(f"  {'metric':<28}{'D′':>8}{'M1':>8}{'Δ(M1−D′)':>11}")
    for kk, lbl in (("TP", "TP@.50 (matched)"), ("F3", "F3 official-match fail"),
                    ("F0uF1uF2", "F0∪F1∪F2 (no cand)")):
        print(f"  {lbl:<28}{dc[kk]:>8}{c_all[kk]:>8}{c_all[kk]-dc[kk]:>+11}")
    # D′ 侧的原生 P3（含 F0/F1/F2 细分）—— 只读，不重算
    p3c = Counter(r["failure"] for r in p3["rows"])
    print(f"\n  D′ 原生 P3（读 attribution.json，不重算）: " +
          ", ".join(f"{k}={v}" for k, v in p3c.items()))

    # ---------------- Size × failure ----------------
    print("\n" + "=" * 100)
    print("Size × failure（IoU=0.50）")
    print("=" * 100)
    print(f"  {'size':<9}{'GT':>6}{'TP@.50':>8}{'F3':>6}{'F0∪F1∪F2':>11}{'TP率':>8}   |  D′ 同口径 TP/F3/noCand")
    for nm, _lo, _hi in SIZE_BINS:
        sub = [r for r in rows if r["size"] == nm]
        if not sub:
            continue
        c = Counter(state(r) for r in sub)
        dsub = Counter(("TP" if r["dp_matched"] else ("F3" if r["dp_best_iou"] >= .5 else "F0uF1uF2")) for r in sub)
        print(f"  {nm:<9}{len(sub):>6}{c['TP']:>8}{c['F3']:>6}{c['F0uF1uF2']:>11}"
              f"{100*c['TP']/len(sub):>7.1f}%   |  {dsub['TP']}/{dsub['F3']}/{dsub['F0uF1uF2']}")

    # ---------------- Class × failure ----------------
    print("\n" + "=" * 100)
    print("Class × failure（IoU=0.50）")
    print("=" * 100)
    print(f"  {'class':<13}{'GT':>6}{'TP@.50':>8}{'F3':>6}{'F0∪F1∪F2':>11}{'TP率':>8}")
    for c in range(NC):
        sub = [r for r in rows if r["cls"] == c]
        if not sub:
            continue
        cc = Counter(state(r) for r in sub)
        print(f"  {NAMES[c]:<13}{len(sub):>6}{cc['TP']:>8}{cc['F3']:>6}{cc['F0uF1uF2']:>11}"
              f"{100*cc['TP']/len(sub):>7.1f}%")

    # ---------------- Linkage with GT-level attribution ----------------
    print("\n" + "=" * 100)
    print("D′_ONLY / M1_ONLY 与 M1 的 failure 状态对接")
    print("=" * 100)
    dont = [r for r in rows if r["dp_matched"] and not r["m1_matched"]]
    m1on = [r for r in rows if r["m1_matched"] and not r["dp_matched"]]
    print(f"  D′_ONLY n={len(dont)}   M1_ONLY n={len(m1on)}")
    print(f"  {'D′_ONLY 内部':<24}{'F3':>6}{'F0∪F1∪F2':>11}")
    c = Counter(state(r) for r in dont)
    print(f"  {'':<24}{c['F3']:>6}{c['F0uF1uF2']:>11}")
    print(f"  {'M1_ONLY 内部':<24}{'F3(D′)':>8}{'F0∪F1∪F2(D′)':>14}")
    cd = Counter(("F3" if r["dp_best_iou"] >= .5 else "F0uF1uF2") for r in m1on)
    print(f"  {'':<24}{cd['F3']:>8}{cd['F0uF1uF2']:>14}")

    out = dict(gate=bool(ok), n_gt=tot,
               m1_state={k: c_all[k] for k in ("TP", "F3", "F0uF1uF2")},
               dp_state={k: dc[k] for k in ("TP", "F3", "F0uF1uF2")},
               dp_native_p3=dict(p3c),
               dprime_only=dict(Counter(state(r) for r in dont)),
               m1_only=dict(cd),
               caveat="M1 F0/F1/F2 NOT AVAILABLE (no raw/pre-NMS dump); not separated, not inferred")
    Path(A.out).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
