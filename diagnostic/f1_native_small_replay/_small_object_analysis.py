"""_small_object_analysis.py — F1 vs D′ 的**小目标专项**分析（§35 small detection rate + §36 G1 follow-up）

只读预测 TXT + val label + 图像头；**无推理、无训练**。口径与 frozen evaluator 一致
（`official_eval` 的常量与函数直接 import；cap=100；native 面积分桶）。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from official_eval import (  # noqa: E402
    MAX_BOXES_PER_IMAGE, _imsize, apply_max_boxes, box_iou_np,
    norm_xywh_to_xyxy, read_gt_txt, read_pred_txt,
)

IMAGES = ROOT / "data/processed/rgbid_split_train/images/val/visible"
LABELS = ROOT / "data/processed/rgbid_split_train/labels/val/visible"
MODELS = {
    "D′": ROOT / "diagnostic/sepstem_clahe/best_full/results",
    "F1": ROOT / "diagnostic/f1_native_small_replay/infer_val/results",
}
V3 = ROOT / "diagnostic/small_object_cause_v2/_v3_analysis.json"
SMALL_A, MEDIUM_A = 1024.0, 9216.0
NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}
LOG: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def buck(a) -> str:
    return "small" if a < SMALL_A else ("medium" if a < MEDIUM_A else "large")


def main() -> int:
    say("=" * 108)
    say("F1 vs D′ —— SMALL-OBJECT SPECIAL ANALYSIS（§35 detection rate / §36 G1 follow-up）")
    say("=" * 108)

    gts = []
    for ip in sorted(p for p in IMAGES.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}):
        w, h = _imsize(ip)
        if w <= 1 or h <= 1:
            continue
        gt = read_gt_txt(LABELS / f"{ip.stem}.txt")
        if len(gt) == 0:
            continue
        if gt[:, 1:].max() > 1.0 or gt[:, 1:].min() < 0.0:
            continue
        gb, gc = norm_xywh_to_xyxy(gt, w, h)
        for j in range(len(gb)):
            a = max(float(gb[j, 2] - gb[j, 0]), 0) * max(float(gb[j, 3] - gb[j, 1]), 0)
            gts.append(dict(key=f"{ip.stem}#{j}", stem=ip.stem, cls=int(gc[j]),
                            box=gb[j].astype(np.float64), native_area=float(a), bucket=buck(a)))
    say(f"  val GT = {len(gts)}（与 official_eval 的 2807 同源口径）")

    per_model = {}
    for nm, d in MODELS.items():
        per = {}
        by_img = defaultdict(list)
        for g in gts:
            by_img[g["stem"]].append(g)
        for stem, gs in by_img.items():
            _, pr, _ = read_pred_txt(Path(d) / f"{stem}.txt")
            pr = apply_max_boxes(pr, MAX_BOXES_PER_IMAGE)
            if len(pr) == 0:
                pb, pc = np.zeros((0, 4), np.float32), np.zeros(0, int)
            else:
                ip = IMAGES / f"{stem}.png"
                if not ip.exists():
                    ip = ip.with_suffix(".jpg")
                w, h = _imsize(ip)
                pb, pc = norm_xywh_to_xyxy(pr, w, h)
            for g in gs:
                same = np.where(pc == g["cls"])[0] if len(pc) else np.zeros(0, int)
                iou_s = box_iou_np(g["box"][None], pb[same])[0] if len(same) else np.zeros(0, np.float32)
                iou_a = box_iou_np(g["box"][None], pb)[0] if len(pb) else np.zeros(0, np.float32)
                per[g["key"]] = dict(best_same=float(iou_s.max()) if len(iou_s) else 0.0,
                                     best_any=float(iou_a.max()) if len(iou_a) else 0.0,
                                     n_pred=int(len(pb)))
        per_model[nm] = per
    v3 = json.loads(V3.read_text(encoding="utf-8"))
    G1 = set(v3["E1"])

    # ---------------- §35 detection rate by native bucket ----------------
    say("")
    say("=" * 108)
    say("[§35] SMALL-OBJECT DETECTION RATE（official IoU≥0.5，同类匹配）")
    say("=" * 108)
    say(f"  {'bucket':<9}{'n GT':>7}{'model':<6}{'detected':>10}{'missed':>8}{'rate':>9}")
    tab = {}
    for b in ("small", "medium", "large"):
        ks = [g["key"] for g in gts if g["bucket"] == b]
        for nm in MODELS:
            det = sum(1 for k in ks if per_model[nm][k]["best_same"] >= 0.5)
            tab[(b, nm)] = dict(n=len(ks), detected=det, missed=len(ks) - det,
                                rate=det / max(len(ks), 1))
            say(f"  {b:<9}{len(ks):>7}{nm:<6}{det:>10}{len(ks)-det:>8}{det/max(len(ks),1):>9.4f}")
        a = tab[(b, "D′")]["rate"]; f = tab[(b, "F1")]["rate"]
        say(f"  {'':<9}{'Δ(F1−D′)':<13}{'':<10}{f-a:>+9.4f}")
    say("")
    ds, fs = tab[("small", "D′")], tab[("small", "F1")]
    say(f"  ⇒ native-small：D′ {ds['detected']}/{ds['n']} = {ds['rate']:.4f} → "
        f"F1 {fs['detected']}/{fs['n']} = {fs['rate']:.4f}　**Δ = {fs['rate']-ds['rate']:+.4f}"
        f"（{fs['detected']-ds['detected']:+d} 个目标）**")

    # ---------------- §36 G1 follow-up ----------------
    say("")
    say("=" * 108)
    say("[§36] G1 FOLLOW-UP（冻结 G1 = `_v3_analysis.json.E1`，38 条；定义未改）")
    say("=" * 108)
    g1 = [k for k in G1 if k in per_model["D′"]]
    say(f"  G1 命中 = {len(g1)}/38")
    say(f"  {'subset':<26}{'n':>5}{'model':<6}{'#IoU==0':>9}{'#IoU>0':>8}{'#IoU>=0.5':>11}{'median best_any':>17}")
    rows = {}
    subsets = [("G1（any-class IoU==0）", g1)]
    nonG1 = [g["key"] for g in gts if g["key"] not in G1 and g["bucket"] == "small"]
    subsets.append(("non-G1 native-small", nonG1))
    for label, ks in subsets:
        for nm in MODELS:
            v = np.array([per_model[nm][k]["best_any"] for k in ks], float)
            s_ = np.array([per_model[nm][k]["best_same"] for k in ks], float)
            rows[(label, nm)] = dict(n=len(ks), n_zero=int((v <= 1e-9).sum()),
                                     n_pos=int((v > 1e-9).sum()), n_det=int((s_ >= 0.5).sum()),
                                     median_any=float(np.median(v)) if len(v) else float("nan"))
            say(f"  {label:<26}{len(ks):>5}{nm:<6}{int((v<=1e-9).sum()):>9}{int((v>1e-9).sum()):>8}"
                f"{int((s_>=0.5).sum()):>11}{np.median(v) if len(v) else float('nan'):>17.4f}")
    say("")
    for label, ks in subsets:
        a, f = rows[(label, "D′")], rows[(label, "F1")]
        say(f"  {label}: D′ 零重叠 {a['n_zero']}/{a['n']} → F1 零重叠 {f['n_zero']}/{f['n']}"
            f"　检出 {a['n_det']} → {f['n_det']}（Δ={f['n_det']-a['n_det']:+d}）")

    # ---------------- 预测数量对照 ----------------
    say("")
    say("=" * 108)
    say("[附] 每图预测数 / 类分布变化")
    say("=" * 108)
    for nm in MODELS:
        np_ = np.array([per_model[nm][g["key"]]["n_pred"] for g in gts], float)
        say(f"  {nm:<4} n_pred 中位={np.median(np_):.0f}  mean={np_.mean():.2f}  "
            f"#(=100 cap)={int((np_==100).sum())}")

    (OUT / "_small_object.json").write_text(json.dumps(
        dict(detection_rate={f"{k[0]}|{k[1]}": v for k, v in tab.items()},
             g1_followup={f"{k[0]}|{k[1]}": v for k, v in rows.items()},
             n_gt=len(gts)), indent=2, ensure_ascii=False), encoding="utf-8")
    say("")
    say(f"[saved] {OUT/'_small_object.json'}")
    (OUT / "SMALL_OBJECT.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
