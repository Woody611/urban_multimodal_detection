# P3 — Pre-NMS → NMS → top-100 disappearance audit

**Date**: 2026-10-03 · **Checkpoint**: D′ = SepStem + IR-CLAHE · **Nature**: diagnostic-only

---

## P3 Audit Verdict

> **Small-GT failure is generated, not discarded.** Only 62.0% of small GT ever have a usable
> same-class box (IoU ≥ 0.5) *at any point*, and that number is already 62.0% **before NMS**.
> Post-processing removes 8 usable boxes in total (NMS 2, top-100 5, matching 1) out of 371 small GT.
> **F0 (raw generation) = 95 small GT (25.6%) — 19× the entire top-100 budget (5 GT) and 47× the
> NMS causal budget (2 GT).** → `CASE B — Diagnostic sufficient`.

---

## 0. Fixed object & reproducibility

| item | value |
|---|---|
| checkpoint | `runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt` |
| checkpoint sha256 | `1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda` |
| dataset split | `data/processed/rgbid_split/{images,labels}/val/visible` — 400 images (evaluator uses 398; 2 corrupt) |
| inference size / mode | imgsz **1280**, rect letterbox, `scaleup=True`, stride 32 |
| batch | 16 (`LoadImagesAndVideos`, `use_simotm=RGBID`, `channels=5`, `ir_encoding=clahe`) |
| code revision | git `5f1a8d5` (working tree dirty — see note) |
| NMS | `torchvision.ops.nms` via `ultralytics.utils.ops.non_max_suppression`, **class-aware** (`agnostic=False`, class offset `max_wh=7680`); conf 0.001 / iou 0.7 / `max_det` 300 / `multi_label=True` / `max_nms` 30000 |
| top-100 | `official_eval.apply_max_boxes` — `lexsort(cls, -conf)[:100]` |
| evaluator | `scripts/official_eval.py` sha256 `e82128abc082ed05f39dd2ef40f7d36680944911f1940a3f0a2f35eb38321783` |

**Reproducibility gate — PASSED.** Re-running the production rect inference and formatting Layer B
with the production writer reproduces the stored D′ predictions **byte-for-byte on 400/400 files**.
The official evaluator on the re-derived Layer B gives **mAP50-95 = 0.515281 / mAP50 = 0.778258 /
mAP75 = 0.510963** — the incumbent baseline exactly.

> Provenance note: the working tree modifies `scripts/predict_rect.py`, but the diff is **comments
> plus two unused `DEFAULT_*` constants** — no executable line in the inference path changes. The
> 400/400 bit-identical reproduction confirms this empirically.

**Layers recorded** (per image, all 400):
- **A** pre-NMS candidates — `xyxy, conf, cls` after the `cls > conf_thres` multi-label expansion,
  *no* extra confidence truncation beyond what the forward itself produces.
- **B** post-NMS — kept boxes, plus every suppressed candidate with its suppressor (index, IoU,
  same-class flag, confidences, areas), plus candidates cut by `i[:max_det]`.
- **C** top-100 — the evaluator's own cap.

> The YOLO11 head is anchor-free with **no separate objectness branch**; `conf = sigmoid(cls)`.
> The full 12-class score vector per candidate was computed in-loop but only the winning class was
> persisted — matching what the official evaluator can actually observe. Any consumer needing the
> full class vectors must re-run `_run_infer.py`.

---

## 1. Pipeline accounting

| Stage | Predictions/image | Small GT with a usable same-class box (IoU ≥ 0.5) | % |
|---|---:|---:|---:|
| Raw (pre-NMS) | **82.19** (median 54, max 680) | **230** | **62.0%** |
| After NMS | **11.79** (median 7, max 184) | 228 | 61.5% |
| After top-100 | **11.36** | 223 | 60.1% |
| Official matched | — | **222** | **59.8%** |

32877 raw candidates → 4715 after NMS (85.66% removed) → 4 images exceed 100 boxes.
The `max_det=300` cap **never fires** (largest image: 184 boxes).

## 2. Small GT disappearance (n = 371, official TP@0.5 = 222)

| Failure | Count | % of all small GT | % of the 149 non-TP |
|---|---:|---:|---:|
| **F0** raw candidate absent (IoU ≥ 0.10) | **95** | **25.61%** | 63.76% |
| **F1** NMS disappearance | 30 | 8.09% | 20.13% |
| **F2** top-100 disappearance | 11 | 2.96% | 7.38% |
| **F3** survives but official match fails | 13 | 3.50% | 8.72% |

Same funnel for reference — medium (n=1320, TP 80.6%): F0 155 / F1 79 / F2 1 / F3 21.
large (n=1116, TP 90.4%): F0 51 / F1 51 / F2 0 / F3 5.

## 3. NMS causal audit

The brief's warning applies in full: *"被 NMS 删除" ≠ NMS 造成的漏检*.

| | small | medium | large |
|---|---:|---:|---:|
| F1 GTs | 30 | 79 | 51 |
| median candidate IoU / GT | **0.371** | 0.803 | 0.885 |
| median suppressor IoU / GT | **0.329** | 0.777 | 0.985 |
| suppressor is same class | **30/30** | 79/79 | 51/51 |
| suppressor conf (median) | 0.594 | 0.886 | 0.939 |
| candidate conf (median) | 0.040 | 0.625 | 0.911 |
| suppressor / candidate area ratio | 2.60 | 0.97 | 0.99 |
| **candidate IoU ≥ 0.5** (a box that *could* have scored TP) | **4 / 30** | 19 / 79 | 15 / 51 |
| **… and would clear the top-100 cut** | **2 / 30** | 18 / 79 | 15 / 51 |

**On small GT the box NMS removes has median IoU 0.371 — it was never a usable detection.**
Only **2 of 371 small GT (0.54%)** lost a box that was simultaneously IoU ≥ 0.5 *and* confident
enough to have made the top-100 (`003653#2` conf 0.0013; `hehe_173_...0105#3` conf 0.434).
Both would still have had to win the greedy one-to-one assignment, so this is an **upper bound**.

Structurally: **30/30, 79/79, 51/51 suppressors are same-class** → with the `max_wh=7680` class
offset, **cross-class NMS suppression never occurs**. `max_det_cut` never fires.

## 4. Top-100 causal audit

| | value |
|---|---|
| F2 GTs | **11** (all small) |
| images involved | **2** — `003982` (8 GT), `003987` (3 GT) |
| median rank after NMS | 146 |
| median confidence | 0.0029 |
| **C1 — no top-100 cap** | mAP50-95 **0.516569** vs capped **0.515281** → **Δ = +0.001288** |
| small GT recoverable by removing the cap | 5 (223 → 228 usable) |

The uncapped figure reproduces the project's historical no-cap D′ number (0.51657) exactly.

## 5. 12-class breakdown (small GT only)

| class | n | TP@.5 | F0 | F1 | F2 | F3 | med best-IoU | med conf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| person | 195 | 56.4% | 32 | 35 | 9 | 9 | 0.654 | 0.095 |
| animal | 63 | 44.4% | 20 | 12 | 0 | 3 | 0.432 | 0.021 |
| sign | 30 | 30.0% | 9 | 10 | 0 | 2 | 0.713 | 0.627 |
| car | 18 | 16.7% | 5 | 10 | 0 | 0 | 0.683 | 0.054 |
| uav | 18 | 66.7% | 0 | 5 | 0 | 1 | 0.790 | 0.839 |
| bicycle | 12 | 8.3% | 6 | 2 | 0 | 3 | 0.173 | 0.024 |
| light | 11 | 0.0% | 7 | 4 | 0 | 0 | **0.000** | 0.011 |
| seat | 9 | 0.0% | 4 | 4 | 1 | 0 | 0.437 | 0.004 |
| ball | 6 | 16.7% | 2 | 0 | 1 | 2 | 0.773 | 0.184 |
| garbage_can | 6 | 0.0% | 1 | 2 | 0 | 3 | 0.614 | 0.211 |
| boat | 3 | 0.0% | 0 | 2 | 0 | 1 | 0.618 | 0.258 |

`light` is the extreme case: **median best pre-NMS same-class IoU = 0.000** — the model produces
no box at all at those locations, before any post-processing.

## 6. Reconciliation with the 2026-09-18 localization audit

That audit was run on a **different checkpoint** (the pre-SepStem baseline, official 0.50928) and at
the **post-NMS** prediction layer. The numbers are therefore **recomputed here, not compared**.

| historical figure | stage it was measured at | D′ recomputation |
|---|---|---|
| **20.8%** — best same-class box has IoU == 0 | post-NMS, old baseline | **18.3%** pre-NMS (best-IoU == 0), D′ |
| **6.5%** — no same-class prediction at all | post-NMS, old baseline | **4.3%** pre-NMS, D′ |
| **27.3%** — no usable same-class detection | post-NMS, old baseline | **25.6%** pre-NMS, D′ |
| (cross-check) small TP@0.50 = 58.0% | post-NMS, old baseline | **59.8%**, D′ |

**Answer to Q2: the 27.3% lives at the pre-NMS generation stage.** It is *identical* to the F0
bucket — 95/371 = 25.61% — and it is fully visible **before NMS runs**. The condition "best
same-class box has IoU == 0 (18.3%)" is also already true at pre-NMS, so it is **not** an artifact
of suppression or truncation.

The F0 shape signature survives too, at the raw stage: of the 70 F0 small GT that *do* have a
same-class box, the median area ratio is **1.90×** the GT and **61.4% exceed 1.5×** — i.e. the
"model emits a box for a neighbouring larger object" signature the 2026-09-18 report attributed to
`SMALL_SCALE_INFORMATION_LIMIT` is present in the **raw head output**. That further supports
localization/scale generation, not NMS.

## 7. Final CASE

```
CASE B — Diagnostic sufficient
```

**Next step (single recommendation):** do **not** open a post-processing optimization experiment —
the total causal budget is ~2 small GT (NMS) + Δ0.0013 mAP (top-100), both orders of magnitude below
the project's noise band (0.008–0.015). The remaining 25.6% F0 mass is a raw-generation problem.

Before any further GPU spend on the scale path, reconcile that evidence with the *already falsified*
scale interventions — OASA 1.4/2.0 (online 48.134 / 47.610 < D′ 48.712), the P2 probe
(0.50232 vs 0.50928), and the F1 native-small replay (online 48.204 vs 48.712). This audit alone
does **not** justify a new training run.

---

### Compliance

No training · no model modification · **no NMS / confidence / top-k sweep** · no TTA · no
multi-checkpoint comparison · no substitution of a different model · disappearance definitions were
fixed before analysis and not tuned afterwards. `torch 2.4.1+cpu`, no GPU used.

### Artifacts

```
diagnostic/p3_disappearance_audit/
  REPORT.md            this report
  _run_infer.py        instrumented production-path inference → layers.npz
  layers.npz           Layer A / B / suppression provenance / per-image accounting (400 imgs)
  meta_layers.json     provenance hashes + inference config + reproduction gate result
  _audit.py            per-GT F0–F3 attribution (+ evaluator-divergence assertions)
  attribution.json     2807 GT rows: S0/S1/S2/S3 values, suppressor stats, official match flags
  _tables.py           all tables in this report
  tables.json          machine-readable aggregate
```
