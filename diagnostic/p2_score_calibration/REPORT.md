# P2 — Offline Prediction Score Calibration (CPU-only, prediction-only)

**Date**: 2026-10-03
**Incumbent**: D′ = SepStem + IR-CLAHE
**Verdict**: `CASE C — P2 NEGATIVE`

---

## 0. Incumbent lock

| item | value |
|---|---|
| checkpoint | `runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt` |
| checkpoint sha256 | `1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda` |
| val predictions | `diagnostic/sepstem_clahe/best_full/results/` — 400 TXT |
| predictions bundle sha256 | `1a78255dbb9bd7b595a9777860816dbb8cc0c13d8bac55075952825e26c2b97c` |
| val split | `data/processed/rgbid_split/{images,labels}/val/visible` — 400 images |
| official evaluator | `scripts/official_eval.py` (frozen) |
| generation config | `predict_rect.py` mode=rect scaleup=True imgsz=1280 conf=0.001 iou=0.7 max_det=300 max_boxes=100 `use_simotm=RGBID` channels=5 `ir_encoding=clahe` |

Prediction file format: `class_id  cx  cy  w  h  conf` (6 columns, normalized).
**No full class logits/probabilities are stored** → §8 class-bias calibration is **skipped by rule**.
`conf` is a YOLO sigmoid probability, so §6B temperature is admissible — but see §4 below.

Nothing was retrained, re-inferred, or modified. No GPU. Only column 6 of the prediction
TXT was ever rewritten; columns 1–5 were copied token-for-token.

---

## 1. Baseline reproduction — EXACT (Q1)

Frozen evaluator (metric A, 100-box cap) on raw D′ predictions:

```
mAP50-95 = 0.515281   mAP50 = 0.778258   mAP75 = 0.510963
per-IoU  = [0.7783, 0.7635, 0.7284, 0.6958, 0.6468, 0.5110, 0.4115, 0.3150, 0.2074, 0.0953]
counts   = images 398 / gt 2807 / pred_total 4654 / pred_used 4482 / truncated 4 / empty 3
```

This is **bit-identical** to the frozen reference row
(`reports/OFFICIAL_EVAL_FREEZE_AND_HISTORICAL_RETEST.md:124`). Proceeded.

> Note: the prompt's reference (0.51528 / 0.7783 / 0.5110) is the **100-box-capped** figure.
> The uncapped historical number 0.51657 belongs to the superseded no-cap convention.

## 2. Prediction audit (§3)

| metric | value |
|---|---|
| images / raw boxes | 400 / 4715 (evaluator uses 398 / 4654; 2 corrupt images dropped by the frozen loader) |
| conf min / max / median | 0.001001 / 0.979883 / 0.452559 |
| conf p95 / p99 | 0.9589 / 0.9702 |
| boxes per image min/max/mean | 0 / 184 / 11.79 |
| images with > 100 boxes | **4** |
| full class scores present | **no** |

Class histogram (raw boxes): person 2068, animal 1041, car 465, light 322, bicycle 188,
sign 187, seat 160, garbage_can 110, boat 82, uav 50, ball 29, tricycle 13.

## 3. Calibration split (§5)

`calibration_split.json` — seed **20261003**, split by image stem, 400 → **320 calibration / 80 holdout**.
Generated once, never changed.

## 4. Why global calibration cannot move this metric

The frozen evaluator's AP is a **pure function of the per-class prediction order**:

1. predictions are matched in confidence-descending order (§六.2);
2. AP = 101-point interpolation of the resulting PR curve (§六.3), with `tail=zero`;
3. `min_conf = 0.0` — there is **no confidence floor**, so no box is ever excluded by its absolute score.

Any strictly monotone transform of the score preserves the order, hence the PR curve, hence
AP **exactly**. Power (`s^γ`, γ>0) and temperature (`σ(logit(s)/T)`, T>0) are both strictly
monotone, so both are rank-preserving and therefore predicted to give Δ ≡ 0 **by construction**.

Cross-class interaction exists in exactly one place: the **top-100 boxes/image cap** (§九(二)).
Only a *class-dependent* transform can reorder predictions of different classes and thereby
change which boxes survive truncation.

## 5. Results

Full table: `calibration_results.csv` (compact) and `classwise_full.csv` (all 36 class-wise configs).

| method | params | split | mAP50-95 | mAP50 | mAP75 | Δ mAP50-95 |
|---|---|---|---:|---:|---:|---:|
| raw | — | calibration | 0.435131 | 0.679103 | 0.447956 | — |
| power | γ ∈ {0.5,0.75,1.0,1.25,1.5,2.0} (6) | calibration | 0.435131 | 0.679103 | 0.447956 | **0.000000** (all 6) |
| temperature | T ∈ {0.5,0.75,1.0,1.25,1.5,2.0} (6) | calibration | 0.435131 | 0.679103 | 0.447956 | **0.000000** (all 6) |
| class-power (best) | cls=9 garbage_can, γ=1.5 | calibration | 0.435174 | 0.679175 | 0.448021 | **+0.000042** |
| class-power (worst) | cls=3 seat, γ=1.25 | calibration | 0.434808 | 0.677939 | 0.447986 | −0.000323 |
| class-bias | **SKIPPED** — no logits in predictions | calibration | — | — | — | — |
| raw | — | holdout | 0.601372 | 0.841107 | 0.621567 | — |
| strongest candidate (post-hoc) | frozen | holdout | 0.601372 | 0.841107 | 0.621567 | **0.000000** |
| raw | — | full val | 0.515281 | 0.778258 | 0.510963 | — |
| strongest candidate (post-hoc) | frozen | full val | 0.515311 | 0.778306 | 0.511008 | **+0.000030** |

All 36 class-wise configs lie in **[−0.000323, +0.000042]**.

**Integrity check** (every method, all 400 files): 0 count mismatches, 0 class mismatches,
0 coordinate mismatches. Columns 1–5 byte-identical; only column 6 differs.

## 6. Top-100 sensitivity audit (Q8, §13)

| transform | images with changed membership | boxes entering | boxes leaving |
|---|---:|---:|---:|
| raw (control) | 0 | 0 | 0 |
| pow_2.0 (rank-preserving) | **0** | 0 | 0 |
| cls_c9_g1.5 (best) | 1 | 3 | 3 |
| cls_c2_g0.75 | 1 | 14 | 14 |
| cls_c3_g1.25 | 1 | 4 | 4 |

Only **4 of 398** evaluated images exceed 100 raw boxes: `003262` (138), `003982` (184),
`003987` (145), `shuming_340_00000020` (105) — 3 in calibration, 1 in holdout.

**Every class-wise Δ traces to a single image, `003262`.** The three other truncated images
were unaffected. The truncation surface of this metric is one image wide.

## 7. G86 / low-score diagnostic (Q9, §14)

`g86_raw.json` — roles from `diagnostic/cls_representation_modality_audit/per_gt_representation_audit.csv`
(86 G86 / 1070 CONTROL / 164 MISSED_NON86, 1320 GT objects).

| role | n | same-class pred | median CIoU (same-class) | best **any-class** CIoU ≥ 0.5 | median GT-class sigmoid@stride8 |
|---|---:|---:|---:|---:|---:|
| G86 | 86 | 0 | — | **0 / 86** | 5.49e-08 (all 86 < 1e-3) |
| CONTROL | 1070 | 1070 | 0.797 | 1070 | 0.707 |
| MISSED_NON86 | 164 | 105 | 0.367 | 11 / 164 | 1.65e-05 |

For G86, the best **any-class** emitted box overlapping the GT has median CIoU **0.092**, and
**not one** reaches CIoU ≥ 0.5. The category the brief asks about —
"CIoU ≥ 0.50 **and** GT-class score < 1e-3" — is therefore **empty (0 objects)** in D′'s
emitted predictions: there is no well-localized box at a G86 location that a score change
could re-rank.

Paired test under `pow_1.5` over the same 1320 objects: **1286 scores changed, 0 rank changes,
0 top-100 membership changes**.

> Caveat: the earlier G1 analysis' "CIoU median 0.6278" refers to the raw stride-8 **anchor**
> grid before NMS/decoding, not the emitted prediction — a different object from the one
> measured here.

## 8. Decision

Per §10, a candidate may only enter holdout if its calibration-split Δ mAP50-95 > 0 **and** it
is not driven by a single class. Every class-wise candidate is *by construction* a single-class
change, and the largest positive Δ is **+4.2e-5** — roughly **200–350× below** the project's own
recorded checkpoint noise band (~0.008–0.015). No candidate qualified, so the holdout gate was
never formally entered; the holdout and full-val runs above are reported as **post-hoc
confirmation only** and were not used for selection.

```
CASE C — P2 NEGATIVE
```

Global calibration is exactly inert (Δ ≡ 0, provably and empirically). Class-wise calibration
moves only a single image's top-100 membership and yields at most +3.0e-5 on full val, three
orders of magnitude below the noise floor.

**The P2 prediction-calibration route is closed. No further parameter search. No online submission.**

## 9. Core questions (§19)

1. **Baseline reproduced bit-for-bit?** Yes — 0.515281 / 0.778258 / 0.510963, per-IoU row identical to the frozen reference.
2. **Which calibration improves official mAP50-95?** None. Global transforms: exactly 0. Class-wise: ≤ +4.2e-5.
3. **Calibration → holdout reproduced?** No (holdout Δ = 0.000000); the gate wasn't met, so holdout wasn't formally entered.
4. **Full val same direction?** Δ = +3.0e-5, sign positive but ~2 orders below noise — not a gain.
5. **Which IoU?** None; the per-IoU vector is unchanged for all global transforms.
6. **Which classes?** None systematically; the only non-zero entries are single-class top-100 reshuffles.
7. **Small / medium / large?** No stratum can differ: predictions and their within-class order are unchanged, so every stratum has Δ = 0.
8. **Top-100 membership churn?** Rank-preserving: 0 images. Class-wise: ≤ 1 image (≤14 boxes in/out).
9. **G86 re-ranked more sensibly?** Not applicable — G86 has no emitted box at CIoU ≥ 0.5 at all; 0 rank changes.
10. **Real ranking improvement or evaluator artifact?** Artifact. The only non-zero signal is top-100 truncation reshuffling on a single image.
11. **Close P2?** Yes — `CASE C / P2 NEGATIVE`, route closed, search not expanded.

## 10. Artifacts

```
diagnostic/p2_score_calibration/
  REPORT.md                    this report
  calibration_results.csv      §16 required table
  classwise_full.csv           all 36 class-wise configurations
  calibration_split.json       frozen 320/80 split (seed 20261003)
  prediction_audit.json        box/score/class audit + bundle hash
  incumbent_lock.json          checkpoint + prediction bundle hashes
  global_results.json          raw + 12 global transforms (calibration split)
  classwise_results.json       raw + 36 class-wise transforms (calibration split)
  holdout_results.json / .csv  holdout raw vs strongest candidate
  full_results.json / .csv     full-val raw vs strongest candidate
  g86_raw.json / g86_pow_1.5.json   G86 paired diagnostic
  dprime_raw_metricA.json      baseline reproduction
  p2_calibration.py            harness (imports the frozen evaluator, never edits it)
  p2_g86.py                    G86 paired analysis
```
